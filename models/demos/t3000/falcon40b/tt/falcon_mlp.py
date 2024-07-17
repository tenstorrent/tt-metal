# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from typing import List
from models.demos.t3000.falcon40b.tt.model_utils import falcon_prefill_matmul, determine_tensor_deallocation

from ttnn import ShardTensorToMesh


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
            mesh_mapper=ShardTensorToMesh(self.device_mesh, dim=3),
            cache_file_name=tt_cache_path / dense_4h_to_h_str,
            preprocess=lambda x: torch.transpose(x.reshape(1, 1, *x.shape), -2, -1),
        )

    def set_model_config(self, model_config):
        self.model_config = model_config

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

        hidden_states = ttnn.experimental.tensor.sharded_to_interleaved(
            hidden_states, output_mem_config=self.model_config["DEFAULT_MEMCFG"]
        )
        hidden_states = ttnn.all_gather(
            hidden_states,
            dim=3,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            memory_config=self.model_config["DEFAULT_MEMCFG"],
        )
        hidden_states = ttnn.experimental.tensor.interleaved_to_sharded(
            hidden_states, sharded_mem_config=self.model_config["MLP_ALL_GATHER_OUTPUT_MEMCFG"]
        )
        hidden_states = ttnn.matmul(
            hidden_states,
            self.dense_4h_to_h_weights,
            program_config=self.model_config["DENSE_4H_TO_H_MM_PROGCFG"],
            memory_config=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_MEMCFG"],
            dtype=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_DTYPE"],
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
        )
        # return TT Tensor
        return hidden_states

    def fwd_prefill(self, x: List[ttnn.experimental.tensor.Tensor]) -> List[ttnn.experimental.tensor.Tensor]:
        hidden_states = []
        should_deallocate_ln_tensors = determine_tensor_deallocation(
            self.model_config["layernorm_params"]["slice_size"], x.get_legacy_shape()[2]
        )
        hidden_states = falcon_prefill_matmul(
            x,
            self.dense_h_to_4h_weights,
            self.model_config["COMPUTE_KERNEL_CONFIG"],
            output_mem_config=self.model_config["DENSE_H_TO_4H_MM_OUTPUT_MEMCFG"],
            output_dtype=self.model_config["DENSE_H_TO_4H_MM_OUTPUT_DTYPE"],
            act=[ttnn.UnaryOpType.GELU, True],
            overwrite_subblock_w=1,  # Workaround for non deterministic output/hang; issue: 7066
            overwrite_subblock_h=1,
        )
        hidden_states = ttnn.all_gather(
            hidden_states,
            dim=3,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            memory_config=self.model_config["DEFAULT_MEMCFG"],
        )
        if should_deallocate_ln_tensors:
            x.deallocate(True)

        hidden_states = falcon_prefill_matmul(
            hidden_states,
            self.dense_4h_to_h_weights,
            self.model_config["COMPUTE_KERNEL_CONFIG"],
            output_mem_config=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_MEMCFG"],
            output_dtype=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_DTYPE"],
            overwrite_subblock_w=1,  # Workaround for non deterministic output/hang; issue: 7066
            overwrite_subblock_h=1,
        )
        # return TT Tensor
        return hidden_states
