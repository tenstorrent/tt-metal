# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
from typing import List
import torch
import ttnn
from ttnn import ShardTensorToMesh


class TtLlamaMLP_optimized:
    def __init__(
        self,
        mesh_device,
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
        self.mesh_device = mesh_device
        self.num_devices = mesh_device.get_num_devices()
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

        w1_dram_shard_str = f"{self.layer_name}.feed_forward.w1_dram_shard.weight"
        w2_dram_shard_str = f"{self.layer_name}.feed_forward.w2_dram_shard_reduce.weight"
        w3_dram_shard_str = f"{self.layer_name}.feed_forward.w3_dram_shard.weight"

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

        # w1: 8k x 4k. width-sharded on 12 banks, 4224 over 12 banks.
        device = self.mesh_device.get_device(0)
        weight_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1),
                )
            }
        )

        w3_shard_shape = (8192, 4224 // 12)  # padded cols to divide by 12
        w3_shard_spec = ttnn.ShardSpec(weight_grid, w3_shard_shape, ttnn.ShardOrientation.ROW_MAJOR, False)
        w3_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, w3_shard_spec)

        self.w1 = ttnn.as_tensor(
            padded_w1,
            dtype=w1_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=w3_mem_config,
            mesh_mapper=ShardTensorToMesh(self.mesh_device, dim=3),
            cache_file_name=self.cache_path / w1_dram_shard_str,
        )

        w2_shard_shape = (4096, 8448 // 12)  # expand dim is 32k / 8 chips padded
        w2_shard_spec = ttnn.ShardSpec(weight_grid, w2_shard_shape, ttnn.ShardOrientation.ROW_MAJOR, False)
        w2_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, w2_shard_spec)
        self.w2 = ttnn.as_tensor(
            padded_w2,
            dtype=w2_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=w2_memory_config,
            mesh_mapper=ShardTensorToMesh(self.mesh_device, dim=2),
            cache_file_name=self.cache_path / w2_dram_shard_str,
        )

        self.w3 = ttnn.as_tensor(
            padded_w3,
            dtype=w3_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=w3_mem_config,
            mesh_mapper=ShardTensorToMesh(self.mesh_device, dim=3),
            cache_file_name=self.cache_path / w3_dram_shard_str,
        )

    def __call__(self, x: List[ttnn.Tensor], mode="decode") -> List[ttnn.Tensor]:
        if mode == "decode":
            return self.decode_forward(x)
        elif mode == "prefill":
            return self.prefill_forward(x)
        else:
            raise ValueError(f"Unknown llm_mode: {mode}")

    def prefill_forward(self, x: List[ttnn.Tensor]) -> List[ttnn.Tensor]:
        # Prefill Reshape fix
        _, _, seq_len, _ = x.shape
        max_mm_seq_len = self.model_config["MAX_MM_SEQ_LEN"]
        if seq_len >= max_mm_seq_len:
            if seq_len % max_mm_seq_len != 0:
                raise ValueError(f"Sequence length {seq_len} is not divisible by {max_mm_seq_len}")
            batch_dim = 1 if seq_len < max_mm_seq_len else seq_len // max_mm_seq_len  # Find the division factor
            x = ttnn.reshape(x, (1, batch_dim, seq_len // batch_dim, self.hidden_size))
            pc1 = self.model_config["PREFILL_PADDED_FF1_MM_PROGCFG"]
            pc2 = self.model_config["PREFILL_PADDED_FF2_MM_PROGCFG"]
            pc3 = self.model_config["PREFILL_PADDED_FF3_MM_PROGCFG"]
        elif seq_len == 128:
            pc1 = self.model_config["PREFILL_PADDED_FF1_MM_PROGCFG_128"]
            pc2 = self.model_config["PREFILL_PADDED_FF2_MM_PROGCFG_128"]
            pc3 = self.model_config["PREFILL_PADDED_FF3_MM_PROGCFG_128"]
        else:
            # ttnn.linear does not allow None program_config if weights are sharded
            pc1 = self.model_config["PREFILL_PADDED_FF1_MM_PROGCFG"]
            pc2 = self.model_config["PREFILL_PADDED_FF2_MM_PROGCFG"]
            pc3 = self.model_config["PREFILL_PADDED_FF3_MM_PROGCFG"]

        w1_out = ttnn.linear(
            x,
            self.w1,
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG_LOFI"],
            core_grid=ttnn.CoreGrid(y=8, x=8) if not pc1 else None,
            dtype=ttnn.bfloat16,
            activation="silu" if not pc1 else None,
            program_config=pc1,
        )

        w3_out = ttnn.linear(
            x,
            self.w3,
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG_LOFI"],
            core_grid=ttnn.CoreGrid(y=8, x=8) if not pc3 else None,
            dtype=ttnn.bfloat16,
            program_config=pc3,
        )

        x.deallocate(True)

        hidden_states = ttnn.mul(w1_out, w3_out, dtype=ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w1_out.deallocate(True)
        w3_out.deallocate(True)

        hidden_states_mm = ttnn.linear(
            hidden_states,
            self.w2,
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
            core_grid=ttnn.CoreGrid(y=8, x=8) if not pc2 else None,
            dtype=ttnn.bfloat8_b,
            program_config=pc2,
        )
        hidden_states.deallocate(True)

        if seq_len >= max_mm_seq_len:
            # Prefill Reshape fix (reverse)
            hidden_states_mm = ttnn.reshape(hidden_states_mm, (1, 1, seq_len, self.hidden_size))

        hidden_states_reduced = ttnn.reduce_scatter(
            hidden_states_mm,
            scatter_dim=3,
            math_op=ttnn.ReduceType.Sum,
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        hidden_states_mm.deallocate(True)

        return hidden_states_reduced

    def decode_forward(self, x: List[ttnn.Tensor]) -> List[ttnn.Tensor]:
        hidden_states = []

        w1_out = ttnn.matmul(
            x,
            self.w1,
            program_config=self.model_config["PADDED_FF3_MM_PROGCFG"],
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG_LOFI"],
        )

        w3_out = ttnn.matmul(
            x,
            self.w3,
            program_config=self.model_config["PADDED_FF3_MM_PROGCFG"],
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG_LOFI"],
        )
        x.deallocate(True)

        hidden_states = ttnn.mul(
            w1_out,
            w3_out,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            input_tensor_a_activation=ttnn.UnaryOpType.SILU,
            dtype=ttnn.bfloat8_b,
        )

        w1_out.deallocate(True)
        w3_out.deallocate(True)

        hidden_states = ttnn.matmul(
            hidden_states,
            self.w2,
            program_config=self.model_config["PADDED_FF2_MM_PROGCFG"],
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
        )

        hidden_states_reduced = ttnn.reduce_scatter(
            hidden_states,
            scatter_dim=3,
            math_op=ttnn.ReduceType.Sum,
            num_links=1,
            memory_config=self.model_config["RESIDUAL_ADD_OUTPUT_MEMCFG"],
        )
        hidden_states.deallocate(True)
        return hidden_states_reduced
