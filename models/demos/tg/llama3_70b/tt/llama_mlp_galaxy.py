# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List
import ttnn
from models.demos.t3000.llama2_70b.tt.llama_common import ShardTensor2dMesh, ConcatMesh2DToTensor
from models.utility_functions import nearest_32
from models.demos.tg.llama3_70b.tt.llama_common import tt_all_reduce, tt_sharded_all_reduce
from models.demos.t3000.falcon40b.tt.model_utils import (
    matmul_2d_config_from_tensor_shapes as get_matmul_2d_config_from_tensor_shapes,
)


class TtLlamaMLP_galaxy:
    def __init__(
        self,
        mesh_device,
        cluster_shape,
        state_dict,
        base_url,
        layer_num,
        hidden_size: int,
        model_config,
        cache_path=None,
        read_cache=False,
    ):
        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.num_devices = mesh_device.get_num_devices()
        assert self.num_devices == 32, "Only 32 devices supported for TG"
        self.model_config = model_config
        self.read_cache = read_cache
        self.cluster_shape = cluster_shape

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

        # TODO: Reenable when DRAM-SHARDED PCC issues resolves
        # w1_cache_str = f"{self.layer_name}.feed_forward.w1_galaxy_dram_shard_unpadded.weight"
        # w2_cache_str = f"{self.layer_name}.feed_forward.w2_galaxy_dram_shard_unpadded.weight"
        # w3_cache_str = f"{self.layer_name}.feed_forward.w3_galaxy_dram_shard_unpadded.weight"
        w1_cache_str = f"{self.layer_name}.feed_forward.w1_galaxy_unpadded.weight"
        w2_cache_str = f"{self.layer_name}.feed_forward.w2_galaxy_unpadded.weight"
        w3_cache_str = f"{self.layer_name}.feed_forward.w3_galaxy_unpadded.weight"

        w1_dtype = ttnn.bfloat4_b
        w2_dtype = ttnn.bfloat8_b
        w3_dtype = ttnn.bfloat4_b

        w1 = None
        w2 = None
        w3 = None
        if not self.read_cache:
            w1 = self.state_dict[w1_str].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
            w2 = self.state_dict[w2_str].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
            w3 = self.state_dict[w3_str].transpose(-2, -1).unsqueeze(0).unsqueeze(0)

        self.w1 = ttnn.as_tensor(
            w1,
            dtype=w1_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            # memory_config=self.w1_mem_config,  # TODO: Reenable when DRAM-SHARDED PCC issues resolves
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ShardTensor2dMesh(self.mesh_device, dims=(2, 3), cluster_shape=self.cluster_shape),
            cache_file_name=self.cache_path / w1_cache_str,
        )

        self.w3 = ttnn.as_tensor(
            w3,
            dtype=w3_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            # memory_config=self.mlp_config["W1_MEM_CONFIG"](self.mesh_device, self.cluster_shape),  # TODO: Reenable when DRAM-SHARDED PCC issues resolves
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ShardTensor2dMesh(self.mesh_device, dims=(2, 3), cluster_shape=self.cluster_shape),
            cache_file_name=self.cache_path / w3_cache_str,
        )

        self.w2 = ttnn.as_tensor(
            w2,
            dtype=w2_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            # memory_config=self.mlp_config["W2_MEM_CONFIG"](self.mesh_device),  # TODO: Reenable when DRAM-SHARDED PCC issues resolves
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ShardTensor2dMesh(self.mesh_device, dims=(3, 2), cluster_shape=self.cluster_shape),
            cache_file_name=self.cache_path / w2_cache_str,
        )

    def __call__(self, x: List[ttnn.Tensor], mode="decode") -> List[ttnn.Tensor]:
        self.mlp_config = self.model_config["mlp"][mode]
        # Decode should have input tensor of shape (seqlen=1, 1, batch, hidden_size)
        if mode == "decode":
            return self.decode_forward(x)
        elif mode == "prefill":
            return self.prefill_forward(x)
        else:
            raise ValueError(f"Unknown llm_mode: {mode}")

    def decode_forward(self, x: List[ttnn.Tensor]) -> List[ttnn.Tensor]:
        w1_out = ttnn.matmul(
            x,
            self.w1,
            # program_config=self.mlp_config["FF1_DRAM_SHARDED_PROGCFG"],
            core_grid=ttnn.CoreGrid(y=1, x=8),
            compute_kernel_config=self.mlp_config["COMPUTE_KERNEL_LOFI"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )

        w3_out = ttnn.matmul(
            x,
            self.w3,
            # program_config=self.mlp_config["FF1_DRAM_SHARDED_PROGCFG"],  # TODO: Reenable when DRAM-SHARDED PCC issues resolves
            core_grid=ttnn.CoreGrid(y=1, x=8),
            compute_kernel_config=self.mlp_config["COMPUTE_KERNEL_LOFI"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        x.deallocate(True)

        w1_out = tt_sharded_all_reduce(
            w1_out,
            self.mesh_device,
            cluster_axis=1,
            num_links=2,
            memory_config=self.mlp_config["FF1_OUT_GATHERED_MEMCFG"],
        )
        w3_out = tt_sharded_all_reduce(
            w3_out,
            self.mesh_device,
            cluster_axis=1,
            num_links=2,
            memory_config=self.mlp_config["FF1_OUT_GATHERED_MEMCFG"],
        )

        w1_out = ttnn.to_memory_config(w1_out, self.mlp_config["FULL_GRID_MEMCFG"])
        w3_out = ttnn.to_memory_config(w3_out, self.mlp_config["FULL_GRID_MEMCFG"])

        hidden_states = ttnn.mul(
            w1_out,
            w3_out,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            input_tensor_a_activation=ttnn.UnaryOpType.SILU,
            dtype=ttnn.bfloat16,
        )
        w1_out.deallocate(True)
        w3_out.deallocate(True)

        hidden_states = ttnn.to_memory_config(hidden_states, self.mlp_config["FF2_ACT_MEMCFG"])
        hidden_states = ttnn.matmul(
            hidden_states,
            self.w2,
            # program_config=self.mlp_config["FF2_DRAM_SHARDED_PROGCFG"],  # TODO: Reenable when DRAM-SHARDED PCC issues resolves
            core_grid=ttnn.CoreGrid(y=1, x=8),
            compute_kernel_config=self.mlp_config["COMPUTE_KERNEL_LOFI"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )

        hidden_states = tt_sharded_all_reduce(
            hidden_states,
            self.mesh_device,
            cluster_axis=0,
            num_links=2,
            memory_config=self.mlp_config["FF2_OUT_GATHERED_MEMCFG"],
        )

        return hidden_states

    def prefill_forward(self, x: List[ttnn.Tensor]) -> List[ttnn.Tensor]:
        _, _, seq_len, _ = x.shape
        max_mm_seq_len = self.model_config["MAX_MM_SEQ_LEN"](seq_len)
        batch_dim = 1 if seq_len < max_mm_seq_len else seq_len // max_mm_seq_len  # Find the division factor
        x = ttnn.reshape(x, (1, batch_dim, seq_len // batch_dim, -1))

        w1_out = ttnn.matmul(
            x,
            self.w1,
            dtype=ttnn.bfloat16,
            program_config=self.mlp_config["FF1_PROGCFG"](seq_len),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
        )
        w3_out = ttnn.matmul(
            x,
            self.w3,
            dtype=ttnn.bfloat16,
            program_config=self.mlp_config["FF1_PROGCFG"](seq_len),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
        )

        w1_out = ttnn.reshape(w1_out, (1, 1, seq_len, -1))

        w1_out = tt_all_reduce(
            w1_out,
            self.mesh_device,
            cluster_axis=1,
            num_links=2,
        )

        w3_out = ttnn.reshape(w3_out, (1, 1, seq_len, -1))
        w3_out = tt_all_reduce(
            w3_out,
            self.mesh_device,
            cluster_axis=1,
            num_links=2,
        )

        hidden_states = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activation=ttnn.UnaryOpType.SILU,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        hidden_states = ttnn.reshape(hidden_states, (1, batch_dim, seq_len // batch_dim, -1))
        hidden_states = ttnn.matmul(
            hidden_states,
            self.w2,
            dtype=ttnn.bfloat16,
            program_config=self.mlp_config["FF2_PROGCFG"](seq_len),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
        )

        hidden_states = ttnn.reshape(hidden_states, (1, 1, seq_len, -1))

        hidden_states = tt_all_reduce(
            hidden_states,
            self.mesh_device,
            cluster_axis=0,
            num_links=2,
        )

        return hidden_states
