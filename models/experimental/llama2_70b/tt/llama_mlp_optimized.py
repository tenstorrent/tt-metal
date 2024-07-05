# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
from typing import List
import torch
from torch import nn
from ttnn import experimental as tt_lib
import ttnn
from ttnn import ShardTensorToMesh


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
        read_cache=False,
    ):
        self.state_dict = state_dict
        self.device_mesh = device_mesh
        self.num_devices = device_mesh.get_num_devices()
        self.model_config = model_config
        self.emulated = emulated
        self.read_cache = read_cache

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

        padded_w1 = None
        padded_w2 = None
        padded_w3 = None
        if not self.read_cache:
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
            cache_file_name=self.cache_path / w1_str,
        )
        self.w1 = ttnn.to_device(w1_ttnn, self.device_mesh)
        w2_ttnn = ttnn.as_tensor(
            padded_w2,
            dtype=w2_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device_mesh,
            memory_config=self.model_config["DRAM_MEMCFG"],
            mesh_mapper=ShardTensorToMesh(self.device_mesh, dim=3),
            cache_file_name=self.cache_path / w2_str,
        )
        self.w2 = ttnn.to_device(w2_ttnn, self.device_mesh)
        w3_ttnn = ttnn.as_tensor(
            padded_w3,
            dtype=w3_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device_mesh,
            memory_config=self.model_config["DRAM_MEMCFG"],
            mesh_mapper=ShardTensorToMesh(self.device_mesh, dim=3),
            cache_file_name=self.cache_path / w3_str,
        )
        self.w3 = ttnn.to_device(w3_ttnn, self.device_mesh)

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
        # Prefill Reshape fix
        _, _, seq_len, _ = x.shape
        max_mm_seq_len = self.model_config["MAX_MM_SEQ_LEN"]
        batch_dim = 1 if seq_len < max_mm_seq_len else seq_len // max_mm_seq_len  # Find the division factor
        x = ttnn.reshape(x, (1, batch_dim, seq_len // batch_dim, self.hidden_size))

        w1_out = ttnn.matmul(
            x,
            self.w1,
            program_config=self.model_config["PADDED_FF1_MM_PROGCFG"],
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG_LOFI"],
            dtype=self.model_config["BFLOAT16_DTYPE"],
        )

        w3_out = ttnn.matmul(
            x,
            self.w3,
            program_config=self.model_config["PADDED_FF3_MM_PROGCFG"],
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG_LOFI"],
            dtype=self.model_config["BFLOAT16_DTYPE"],
        )
        x.deallocate(True)

        hidden_states = ttnn.mul(w1_out, w3_out, dtype=ttnn.bfloat8_b, memory_config=self.model_config["DRAM_MEMCFG"])
        w1_out.deallocate(True)
        w3_out.deallocate(True)

        hidden_states = ttnn.all_gather(
            hidden_states,
            dim=3,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
        )

        hidden_states = ttnn.matmul(
            hidden_states,
            self.w2,
            program_config=self.model_config["PADDED_FF2_MM_PROGCFG"],
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_FP16_ACC_CONFIG"],
            dtype=self.model_config["BFLOAT16_DTYPE"],
        )

        # Prefill Reshape fix (reverse)
        hidden_states = ttnn.reshape(hidden_states, (1, 1, seq_len, self.hidden_size // self.num_devices))

        return hidden_states

    def decode_forward(self, x: List[tt_lib.tensor.Tensor]) -> List[tt_lib.tensor.Tensor]:
        hidden_states = []

        w1_out = ttnn.matmul(
            x,
            self.w1,
            program_config=self.model_config["PADDED_FF1_MM_PROGCFG"],
            memory_config=self.model_config["WIDTH_SHARDED_MEMCFG"],
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG_LOFI"],
        )

        w3_out = ttnn.matmul(
            x,
            self.w3,
            program_config=self.model_config["PADDED_FF3_MM_PROGCFG"],
            memory_config=self.model_config["WIDTH_SHARDED_MEMCFG"],
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG_LOFI"],
        )
        x.deallocate(True)

        hidden_states = ttnn.mul(
            w1_out,
            w3_out,
            memory_config=self.model_config["WIDTH_SHARDED_MEMCFG"],
            dtype=self.model_config["BFP8_DTYPE"],
        )
        w1_out.deallocate(True)
        w3_out.deallocate(True)

        hidden_states = ttnn.all_gather(
            hidden_states,
            dim=3,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            memory_config=self.model_config["PADDED_MLP_ALL_GATHER_OUTPUT_MEMCFG"],
        )

        hidden_states = ttnn.matmul(
            hidden_states,
            self.w2,
            program_config=self.model_config["PADDED_FF2_MM_PROGCFG"],
            memory_config=self.model_config["WIDTH_SHARDED_MEMCFG"],
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
        )
        return hidden_states
