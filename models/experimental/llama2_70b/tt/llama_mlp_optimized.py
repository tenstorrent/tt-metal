# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
from typing import List
import torch
from torch import nn
import ttnn.experimental as tt_lib
import ttnn
from models.utility_functions import torch2tt_tensor, tt2torch_tensor
from models.experimental.llama2_70b.tt.llama_common import (
    tt_all_gather_torch,
    get_weight_cache_path,
    get_weight_cache_path_ttnn,
)
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh, ConcatMeshToTensor, ListMeshToTensor


class TtLlamaMLP_optimized:
    def __init__(
        self,
        device_mesh,
        state_dict,
        base_url,
        layer_num,
        hidden_size: int,
        model_config,
        emulated=False,
        cache_path=None,
    ):
        self.state_dict = state_dict
        self.device_mesh = device_mesh
        self.num_devices = device_mesh.get_num_devices()
        self.model_config = model_config
        self.emulated = emulated

        self.hidden_size = hidden_size

        self.layer_name = f"{base_url}.{layer_num}"
        self.cache_path = cache_path

        self.load_weights()

    def set_model_config(self, model_config):
        self.model_config = model_config

    def load_weights(self):
        assert not hasattr(self, "w1_list"), "w1_list is already an attribute of this object"
        assert not hasattr(self, "w3_list"), "w3_list is already an attribute of this object"
        assert not hasattr(self, "w2_list"), "w2_list is already an attribute of this object"

        w1_str = f"{self.layer_name}.feed_forward.w1.weight"
        w2_str = f"{self.layer_name}.feed_forward.w2.weight"
        w3_str = f"{self.layer_name}.feed_forward.w3.weight"

        w1_dtype = ttnn.bfloat4_b
        w2_dtype = ttnn.bfloat8_b
        w3_dtype = ttnn.bfloat4_b

        # Do padding
        H = 8 * 1024
        PADDED_H4 = 32 * 1024
        H4 = 28 * 1024
        padded_w1 = torch.zeros(1, 1, H, PADDED_H4)
        padded_w2 = torch.zeros(1, 1, PADDED_H4, H)
        padded_w3 = torch.zeros(1, 1, H, PADDED_H4)
        padded_w1[:, :, :, :H4] = self.state_dict[w1_str].transpose(-2, -1)
        padded_w2[:, :, :H4, :] = self.state_dict[w2_str].transpose(-2, -1)
        padded_w3[:, :, :, :H4] = self.state_dict[w3_str].transpose(-2, -1)

        w1_ttnn = ttnn.as_tensor(
            padded_w1,
            dtype=w1_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device_mesh,
            memory_config=self.model_config["DRAM_MEMCFG"],
            mesh_mapper=ShardTensorToMesh(self.device_mesh, dim=3),
            # cache_file_name=self.cache_path / w1_str,
        )
        self.w1 = ttnn.to_device(w1_ttnn, self.device_mesh)
        w2_ttnn = ttnn.as_tensor(
            padded_w2,
            dtype=w2_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device_mesh,
            memory_config=self.model_config["DRAM_MEMCFG"],
            mesh_mapper=ShardTensorToMesh(self.device_mesh, dim=3),
            # cache_file_name=self.cache_path / w2_str,
        )
        self.w2 = ttnn.to_device(w2_ttnn, self.device_mesh)
        w3_ttnn = ttnn.as_tensor(
            padded_w3,
            dtype=w3_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device_mesh,
            memory_config=self.model_config["DRAM_MEMCFG"],
            mesh_mapper=ShardTensorToMesh(self.device_mesh, dim=3),
            # cache_file_name=self.cache_path / w3_str,
        )
        self.w3 = ttnn.to_device(w3_ttnn, self.device_mesh)

    def prepare_inputs(self, x, device_mesh):
        if self.model_config["LLM_MODE"] == "decode":
            x_multichip = ttnn.from_torch(
                x, layout=ttnn.TILE_LAYOUT, device=device_mesh, mesh_mapper=ReplicateTensorToMesh(device_mesh)
            )
            x_multichip = ttnn.to_memory_config(
                x_multichip,
                ttnn.create_sharded_memory_config(
                    shape=(32, 8192 // 32),
                    core_grid=ttnn.CoreGrid(y=4, x=8),
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                ),
            )
            return x_multichip
        # elif self.model_config["LLM_MODE"] == "prefill":
        #     x_multichip = []
        #     for i in range(self.num_devices):
        #         x_multichip.append(
        #             torch2tt_tensor(
        #                 x.clone(),
        #                 self.devices[i],
        #                 tt_dtype=self.model_config["LN_MLP_OUTPUT_DTYPE"],
        #             )
        #         )
        #     return x_multichip

    def __call__(self, x: List[tt_lib.tensor.Tensor]) -> List[tt_lib.tensor.Tensor]:
        # Decode should have input tensor of shape (seqlen=1, 1, batch, hidden_size)
        if self.model_config["LLM_MODE"] == "decode":
            return self.decode_forward(x)
        # Prefill should have input tensor of shape (1, batch, seqlen, hidden_size)
        elif self.model_config["LLM_MODE"] == "prefill":
            return self.prefill_forward(x)
        else:
            raise ValueError(f"Unknown llm_mode: {self.model_config['LLM_MODE']}")

    def prefill_forward(self, x: List[tt_lib.tensor.Tensor]) -> List[tt_lib.tensor.Tensor]:
        hidden_states = []
        w1_outs = []
        w3_outs = []

        seq_tiles = x[0].shape[2] // 32
        cores_y = 8 if seq_tiles % 8 == 0 else 4  # Pick largest possible coregrid for op
        self.model_config["PADDED_FF1_MM_PROGCFG"] = self.model_config["PADDED_FF1_MM_PROGCFG_LAMBDA"](
            seq_tiles, cores_y
        )
        self.model_config["PADDED_FF3_MM_PROGCFG"] = self.model_config["PADDED_FF3_MM_PROGCFG_LAMBDA"](
            seq_tiles, cores_y
        )
        self.model_config["PADDED_FF2_MM_PROGCFG"] = self.model_config["PADDED_FF2_MM_PROGCFG_LAMBDA"](
            seq_tiles, cores_y
        )
        block_sharded_memcfg = self.model_config["MLP_BLOCK_SHARDED_MEMCFG_LAMBDA"](x[0].shape[2], cores_y)
        for i in range(len(x)):
            # TODO: Use FP32 accumulate after the issue with primary.matmul with FP32 accumulate is fixed
            w1_outs.append(
                tt_lib.operations.primary.matmul(
                    x[i],
                    self.w1_list[i],
                    program_config=self.model_config["PADDED_FF1_MM_PROGCFG"],
                    output_mem_config=block_sharded_memcfg,
                    compute_kernel_config=self.model_config["COMPUTE_KERNEL_FP16_ACC_CONFIG_LOFI"],
                    # output_dtype=self.model_config["BFP8_DTYPE"],
                )
            )

        for i in range(len(x)):
            w3_outs.append(
                tt_lib.operations.primary.matmul(
                    x[i],
                    self.w3_list[i],
                    program_config=self.model_config["PADDED_FF3_MM_PROGCFG"],
                    output_mem_config=block_sharded_memcfg,
                    compute_kernel_config=self.model_config["COMPUTE_KERNEL_FP16_ACC_CONFIG_LOFI"],
                    # output_dtype=self.model_config["BFP8_DTYPE"],
                )
            )
            x[i].deallocate(True)

        for i in range(len(w1_outs)):
            hidden_states.append(ttnn.mul(w1_outs[i], w3_outs[i], dtype=ttnn.bfloat8_b))
            w1_outs[i].deallocate(True)
            w3_outs[i].deallocate(True)

        if self.emulated:
            hidden_states = tt_all_gather_torch(hidden_states, dim=-1)
        else:
            hidden_states = tt_lib.tensor.all_gather(
                hidden_states,
                dim=3,
                num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            )

        for i in range(len(hidden_states)):
            hidden_states[i] = tt_lib.operations.primary.matmul(
                hidden_states[i],
                self.w2_list[i],
                program_config=self.model_config["PADDED_FF2_MM_PROGCFG"],
                compute_kernel_config=self.model_config["COMPUTE_KERNEL_FP16_ACC_CONFIG"],
            )

        return hidden_states

    def decode_forward(self, x: List[tt_lib.tensor.Tensor]) -> List[tt_lib.tensor.Tensor]:
        hidden_states = []
        w1_outs = []
        w3_outs = []

        # ttnn.matmul , use_1d_systolic =
        w1_out = tt_lib.operations.primary.matmul_1d(
            x,
            self.w1,
            program_config=self.model_config["PADDED_FF1_MM_PROGCFG"],
            output_mem_config=self.model_config["WIDTH_SHARDED_MEMCFG"],
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG_LOFI"],
        )

        w3_out = tt_lib.operations.primary.matmul_1d(
            x,
            self.w3,
            program_config=self.model_config["PADDED_FF3_MM_PROGCFG"],
            output_mem_config=self.model_config["WIDTH_SHARDED_MEMCFG"],
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG_LOFI"],
        )
        x.deallocate(True)

        hidden_states = tt_lib.tensor.mul(
            w1_out,
            w3_out,
            output_mem_config=self.model_config["WIDTH_SHARDED_MEMCFG"],
            output_dtype=self.model_config["BFP8_DTYPE"],
        )
        w1_out.deallocate(True)
        w3_out.deallocate(True)

        hidden_states = tt_lib.tensor.sharded_to_interleaved(
            hidden_states, output_mem_config=self.model_config["L1_MEMCFG"]
        )

        # if self.emulated:
        #     hidden_states = tt_all_gather_torch(hidden_states, dim=-1)
        # else:
        #     hidden_states = tt_lib.tensor.all_gather(
        #         hidden_states,
        #         dim=3,
        #         num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
        #         output_mem_config=self.model_config["L1_MEMCFG"],
        #     )

        hidden_states = ttnn.all_gather(
            hidden_states,
            dim=3,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            memory_config=self.model_config["L1_MEMCFG"],
        )

        hidden_states = tt_lib.tensor.interleaved_to_sharded(
            hidden_states, sharded_mem_config=self.model_config["PADDED_MLP_ALL_GATHER_OUTPUT_MEMCFG"]
        )

        hidden_states = tt_lib.operations.primary.matmul_1d(
            hidden_states,
            self.w2,
            program_config=self.model_config["PADDED_FF2_MM_PROGCFG"],
            output_mem_config=self.model_config["WIDTH_SHARDED_MEMCFG"],
            # output_dtype=self.model_config["BFP8_DTYPE"],
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
        )
        return hidden_states
