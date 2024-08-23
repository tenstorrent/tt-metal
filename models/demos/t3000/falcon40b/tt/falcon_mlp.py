# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from typing import List
from models.demos.t3000.falcon40b.tt.model_utils import falcon_prefill_matmul, determine_tensor_deallocation

from ttnn import ShardTensorToMesh, ReplicateTensorToMesh


class TtFalconMLP:
    def __init__(
        self,
        device_mesh,
        state_dict,
        base_url,
        layer_num,
        hidden_size: int,
        model_config,
        tt_cache_path,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.device_mesh = device_mesh
        self.hidden_size = hidden_size
        self.model_config = model_config

        layer_name = f"{base_url}.{layer_num}"

        dense_h_to_4h_str = f"{layer_name}.mlp.dense_h_to_4h.weight"
        dense_4h_to_h_str = f"{layer_name}.mlp.dense_4h_to_h.weight"

        w1_weight = self.state_dict.get(dense_h_to_4h_str) if self.state_dict else None
        w2_weight = self.state_dict.get(dense_4h_to_h_str) if self.state_dict else None

        self.dense_h_to_4h_weights = ttnn.as_tensor(
            w1_weight,
            dtype=self.model_config["DENSE_H_TO_4H_MM_WEIGHTS_DTYPE"],
            layout=ttnn.TILE_LAYOUT,
            device=self.device_mesh,
            memory_config=self.model_config["DENSE_H_TO_4H_MM_WEIGHTS_MEMCFG"],
            mesh_mapper=ShardTensorToMesh(self.device_mesh, dim=3),
            cache_file_name=tt_cache_path / dense_h_to_4h_str,
            preprocess=lambda x: torch.transpose(x.reshape(1, 1, *x.shape), -2, -1),
        )

        self.dense_4h_to_h_weights = ttnn.as_tensor(
            w2_weight,
            dtype=self.model_config["DENSE_4H_TO_H_MM_WEIGHTS_DTYPE"],
            layout=ttnn.TILE_LAYOUT,
            device=self.device_mesh,
            memory_config=self.model_config["DENSE_4H_TO_H_MM_WEIGHTS_MEMCFG"],
            mesh_mapper=ShardTensorToMesh(self.device_mesh, dim=2),
            cache_file_name=tt_cache_path / f"{dense_4h_to_h_str}_height_fractured",
            preprocess=lambda x: torch.transpose(x.reshape(1, 1, *x.shape), -2, -1),
        )

        self.output = None
        self._allocate_output_mlp_tensors()

    def set_model_config(self, model_config):
        self.model_config = model_config

        self._allocate_output_mlp_tensors()

    def _allocate_output_mlp_tensors(self):
        if self.model_config["LLM_MODE"] == "prefill":
            if self.output is not None:
                self.output.deallocate()

            seq_len = self.model_config["row_height"]

            # prepare output tensor on device
            out_shape = (1, 1, seq_len, self.dense_4h_to_h_weights.shape[-1])
            out_tensor = torch.zeros(out_shape).bfloat16()

            self.output = ttnn.as_tensor(
                out_tensor,
                self.model_config["DENSE_4H_TO_H_MM_OUTPUT_DTYPE"],
                layout=ttnn.TILE_LAYOUT,
                device=self.device_mesh,
                memory_config=self.model_config["DEFAULT_MEMCFG"],
                mesh_mapper=ReplicateTensorToMesh(self.device_mesh),
            )

    def __call__(
        self, x: List[ttnn.experimental.tensor.Tensor], llm_mode: str
    ) -> List[ttnn.experimental.tensor.Tensor]:
        if llm_mode == "prefill":
            return self.fwd_prefill(x)
        elif llm_mode == "decode":
            return self.fwd_decode(x)
        else:
            assert False

    def fwd_decode(self, x: List[ttnn.experimental.tensor.Tensor]) -> List[ttnn.experimental.tensor.Tensor]:
        hidden_states = ttnn.matmul(
            x,
            self.dense_h_to_4h_weights,
            program_config=self.model_config["DENSE_H_TO_4H_MM_PROGCFG"],
            memory_config=self.model_config["DENSE_H_TO_4H_MM_OUTPUT_MEMCFG"],
            dtype=self.model_config["DENSE_H_TO_4H_MM_OUTPUT_DTYPE"],
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
        )
        x.deallocate(True)

        hidden_states = ttnn.matmul(
            hidden_states,
            self.dense_4h_to_h_weights,
            program_config=self.model_config["DENSE_4H_TO_H_MM_PROGCFG"],
            memory_config=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_MEMCFG"],
            dtype=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_DTYPE"],
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
        )

        hidden_states = ttnn.sharded_to_interleaved(hidden_states, memory_config=self.model_config["DEFAULT_MEMCFG"])

        hidden_states = ttnn.get_device_tensors(
            hidden_states
        )  # Workaround for reduce_scatter only taking a vector of tensors and not device_mesh

        hidden_states = ttnn.get_device_tensors(
            ttnn.reduce_scatter(
                ttnn.aggregate_as_tensor(hidden_states),
                scatter_dim=3,
                math_op=ttnn.ReduceType.Sum,
                num_links=1,  # only unidirectional supported for now
                memory_config=self.model_config["DEFAULT_MEMCFG"],
            )
        )

        hidden_states = ttnn.aggregate_as_tensor(hidden_states)  # Workaround reverse

        hidden_states = ttnn.interleaved_to_sharded(
            hidden_states, self.model_config["MLP_REDUCE_SCATTER_OUTPUT_MEMCFG"]
        )

        # return TT Tensor
        return hidden_states

    def fwd_prefill(self, x: List[ttnn.experimental.tensor.Tensor]) -> List[ttnn.experimental.tensor.Tensor]:
        hidden_states = []
        should_deallocate_ln_tensors = determine_tensor_deallocation(
            self.model_config["layernorm_params"]["slice_size"], x.get_legacy_shape()[2]
        )

        mlp_num_slices = self.model_config["MLP_NUM_SLICES"]
        for slice_idx in range(mlp_num_slices):
            x_slice = ttnn.interleaved_to_sharded_partial(
                x,
                self.model_config["MLP_GRID_SIZE"],
                self.model_config["MLP_INPUT_SHARD_SPEC"],
                mlp_num_slices,
                slice_idx,
                self.model_config["MLP_INPUT_SHARD_LAYOUT"],
                ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR,
            )

            hidden_states_slice = falcon_prefill_matmul(
                x_slice,
                self.dense_h_to_4h_weights,
                self.model_config["COMPUTE_KERNEL_CONFIG"],
                output_mem_config=self.model_config["DENSE_H_TO_4H_MM_OPTIMIZED_OUTPUT_MEMCFG"],
                output_dtype=self.model_config["DENSE_H_TO_4H_MM_OUTPUT_DTYPE"],
                act=[ttnn.UnaryOpType.GELU, True],
                overwrite_subblock_w=1,  # Workaround for non deterministic output/hang; issue: 7066
                overwrite_subblock_h=1,
            )
            x_slice.deallocate(True)

            hidden_states_slice = falcon_prefill_matmul(
                hidden_states_slice,
                self.dense_4h_to_h_weights,
                self.model_config["COMPUTE_KERNEL_CONFIG"],
                output_mem_config=self.model_config["DENSE_4H_TO_H_MM_OPTIMIZED_OUTPUT_MEMCFG"],
                output_dtype=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_DTYPE"],
                overwrite_subblock_w=1,  # Workaround for non deterministic output/hang; issue: 7066
                overwrite_subblock_h=1,
            )

            ttnn.sharded_to_interleaved_partial(
                hidden_states_slice,
                self.output,
                mlp_num_slices,
                slice_idx,
                memory_config=self.model_config["DEFAULT_MEMCFG"],
            )
            hidden_states_slice.deallocate()

        # Deallocate input
        if should_deallocate_ln_tensors:
            x.deallocate(True)

        hidden_states = ttnn.get_device_tensors(
            self.output
        )  # Workaround for reduce_scatter only taking a vector of tensors and not device_mesh

        hidden_states = ttnn.get_device_tensors(
            ttnn.reduce_scatter(
                ttnn.aggregate_as_tensor(hidden_states),
                scatter_dim=3,
                math_op=ttnn.ReduceType.Sum,
                num_links=1,  # only one link supported for now
                memory_config=self.model_config["DEFAULT_MEMCFG"],
            )
        )

        hidden_states = ttnn.aggregate_as_tensor(hidden_states)  # Workaround reverse

        # return TT Tensor
        return hidden_states
