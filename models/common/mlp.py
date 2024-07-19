# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.common.modules import Mode, MultiModeModule


class MLP(MultiModeModule):
    def __init__(
        self,
        device,
        dim,
        state_dict,
        layer_num,
        model_config,
        weight_base_name="layers",
        weight_cache_path=None,
    ):
        is_device_mesh = device.__class__.__name__ == "DeviceMesh"
        self.state_dict = state_dict
        self.device = device
        self.num_devices = device.get_num_devices() if is_device_mesh else 1
        self.model_config = model_config

        self.dim = dim

        if layer_num is None:
            self.layer_name = ""
        elif weight_base_name is None:
            self.layer_name = f"{layer_num}"
        else:
            self.layer_name = f"{weight_base_name}.{layer_num}"
        self.weight_cache_path = weight_cache_path

        self.load_weights()

    @property
    def mode(self):
        if self.model_config:
            return {
                "decode": Mode.DECODE,
                "prefill": Mode.PREFILL,
            }[self.model_config["LLM_MODE"]]
        else:
            return self._mode

    @mode.setter
    def mode(self, value):
        if self.model_config:
            raise ValueError('Cannot set mode manually when model_config is used, set model_config["LLM_MODE"] instead')
        else:
            self._mode = value

    def set_model_config(self, model_config):
        self.model_config = model_config

    def load_weights(self):
        assert not hasattr(self, "w1_list"), "w1_list is already an attribute of this object"
        assert not hasattr(self, "w3_list"), "w3_list is already an attribute of this object"
        assert not hasattr(self, "w2_list"), "w2_list is already an attribute of this object"

        w1_str = f"{self.layer_name}.feed_forward.w1.weight"
        w2_str = f"{self.layer_name}.feed_forward.w2.weight"
        w3_str = f"{self.layer_name}.feed_forward.w3.weight"

        w2_dram_shard_str = f"{self.layer_name}.feed_forward.w2_dram_shard.weight"
        w3_dram_shard_str = f"{self.layer_name}.feed_forward.w3_dram_shard.weight"

        w1_dtype = ttnn.bfloat4_b
        w2_dtype = ttnn.bfloat8_b
        w3_dtype = ttnn.bfloat4_b

        padded_w1 = None
        padded_w2 = None
        padded_w3 = None
        if self.state_dict:
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

        # w1: 8k x 4k. width-sharded on 12 banks, 4224 over 12 banks.
        d0 = self.device.get_device(0)
        weight_grid = ttnn.experimental.tensor.CoreRangeSet(
            {
                ttnn.experimental.tensor.CoreRange(
                    ttnn.experimental.tensor.CoreCoord(0, 0),
                    ttnn.experimental.tensor.CoreCoord(d0.dram_grid_size().x - 1, d0.dram_grid_size().y - 1),
                )
            }
        )

        self.w1 = ttnn.as_tensor(
            padded_w1,
            dtype=w1_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=self.model_config["DRAM_MEMCFG"],
            mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=3),
            cache_file_name=self.weight_cache_path / w1_str if self.weight_cache_path else None,
        )

        w2_shard_shape = (32768, 96)  # Padded cols 1024/12
        w2_shard_spec = ttnn.ShardSpec(weight_grid, w2_shard_shape, ttnn.ShardOrientation.ROW_MAJOR, False)
        w2_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, w2_shard_spec)
        self.w2 = ttnn.as_tensor(
            padded_w2,
            dtype=w2_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=w2_memory_config,
            mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=3),
            cache_file_name=self.weight_cache_path / w2_dram_shard_str if self.weight_cache_path else None,
        )

        w3_shard_shape = (8192, 4224 // 12)  # padded cols to divide by 12
        w3_shard_spec = ttnn.ShardSpec(weight_grid, w3_shard_shape, ttnn.ShardOrientation.ROW_MAJOR, False)
        w3_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, w3_shard_spec)
        self.w3 = ttnn.as_tensor(
            padded_w3,
            dtype=w3_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=w3_mem_config,
            mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=3),
            cache_file_name=self.weight_cache_path / w3_dram_shard_str if self.weight_cache_path else None,
        )

    def prefill_forward(self, x):
        # Prefill Reshape fix
        _, _, seq_len, _ = x.shape
        max_mm_seq_len = self.model_config["MAX_MM_SEQ_LEN"]
        batch_dim = 1 if seq_len < max_mm_seq_len else seq_len // max_mm_seq_len  # Find the division factor
        x = ttnn.reshape(x, (1, batch_dim, seq_len // batch_dim, self.dim))

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
        hidden_states = ttnn.reshape(hidden_states, (1, 1, seq_len, self.dim // self.num_devices))

        return hidden_states

    def decode_forward(self, x):
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
            # memory_config=self.model_config["L1_MEMCFG"],
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
