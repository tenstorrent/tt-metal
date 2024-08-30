# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
from typing import List
import torch
import ttnn
from ttnn import ReplicateTensorToMesh, ShardTensorToMesh

from models.demos.t3000.llama2_70b.tt.llama_attention_optimized import TtLlamaAttention_optimized
from models.demos.t3000.llama2_70b.tt.llama_mlp_optimized import TtLlamaMLP_optimized


class TtLlamaDecoder_optimized:
    def __init__(
        self,
        mesh_device,
        state_dict,
        base_url,
        layer_num,
        model_config,
        configuration,
        transformation_mats,
        cache_path=None,
        read_cache=False,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.num_devices = mesh_device.get_num_devices()
        self.model_config = model_config
        self.read_cache = read_cache

        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.n_local_heads = self.n_heads // self.num_devices
        self.padded_local_heads = 32
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = configuration.max_seq_len
        self.norm_eps = configuration.norm_eps
        self.rope_theta = configuration.rope_theta

        self.llama3 = configuration.vocab_size == 128256

        self.layer_name = f"{base_url}.{layer_num}"
        self.cache_path = cache_path

        self.attention = TtLlamaAttention_optimized(
            mesh_device,
            state_dict,
            base_url,
            layer_num,
            model_config,
            configuration,
            transformation_mats,
            cache_path=cache_path,
            read_cache=read_cache,
        )

        self.mlp = TtLlamaMLP_optimized(
            mesh_device,
            state_dict,
            base_url,
            layer_num,
            self.hidden_size,
            model_config,
            cache_path=cache_path,
            read_cache=read_cache,
        )

        self.load_weights()

    def set_model_config(self, model_config):
        self.model_config = model_config
        self.attention.set_model_config(model_config)
        self.mlp.set_model_config(model_config)

    def load_weights(self):
        """
        Loads weights that this layer is responsible for.
        Doesn't touch the weights of the submodules.
        """
        assert not hasattr(self, "attn_norm"), "attn_norm_list is already an attribute of this object"
        assert not hasattr(self, "ffn_norm"), "ffn_norm_list is already an attribute of this object"
        attn_norm_str = f"{self.layer_name}.attention_norm.weight"
        ffn_norm_str = f"{self.layer_name}.ffn_norm.weight"

        attn_norm_sharded_str = f"{self.layer_name}.attention_norm_sharded.weight"
        ffn_norm_sharded_str = f"{self.layer_name}.ffn_norm_sharded.weight"

        pt_attn_norm = None
        pt_ffn_norm = None
        if not self.read_cache:
            pt_attn_norm = self.state_dict[attn_norm_str].reshape([1, 1, -1, 32])
            pt_ffn_norm = self.state_dict[ffn_norm_str].reshape([1, 1, -1, 32])

        attn_norm_ttnn = ttnn.as_tensor(
            pt_attn_norm,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=self.model_config["DRAM_MEMCFG"],
            mesh_mapper=ReplicateTensorToMesh(self.mesh_device),
            cache_file_name=self.cache_path / attn_norm_str,
        )
        self.attn_norm = ttnn.to_device(attn_norm_ttnn, self.mesh_device)

        attn_norm_sharded_ttnn = ttnn.as_tensor(
            pt_attn_norm,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=self.model_config["DRAM_MEMCFG"],
            mesh_mapper=ShardTensorToMesh(self.mesh_device, dim=2),
            cache_file_name=self.cache_path / attn_norm_sharded_str,
        )
        self.attn_norm_sharded = ttnn.to_device(attn_norm_sharded_ttnn, self.mesh_device)

        ffn_norm_ttnn = ttnn.as_tensor(
            pt_ffn_norm,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=self.model_config["DRAM_MEMCFG"],
            mesh_mapper=ReplicateTensorToMesh(self.mesh_device),
            cache_file_name=self.cache_path / ffn_norm_str,
        )
        self.ffn_norm = ttnn.to_device(ffn_norm_ttnn, self.mesh_device)

        ffn_norm_sharded_ttnn = ttnn.as_tensor(
            pt_ffn_norm,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=self.model_config["DRAM_MEMCFG"],
            mesh_mapper=ShardTensorToMesh(self.mesh_device, dim=2),
            cache_file_name=self.cache_path / ffn_norm_sharded_str,
        )
        self.ffn_norm_sharded = ttnn.to_device(ffn_norm_sharded_ttnn, self.mesh_device)

    def __call__(
        self,
        xs: List[ttnn.Tensor],
        rot_mats: List[ttnn.Tensor],
        start_pos: int,
        attn_masks: List[ttnn.Tensor],
        user_id: int = 0,
    ) -> ttnn.Tensor:
        if self.model_config["LLM_MODE"] == "prefill":
            return self.prefill_forward(xs, rot_mats, start_pos, attn_masks, user_id)
        elif self.model_config["LLM_MODE"] == "decode":
            return self.decode_forward(xs, rot_mats, start_pos, attn_masks)
        else:
            raise ValueError(f"Unknown llm_mode: {self.model_config['LLM_MODE']}")

    def decode_forward(
        self,
        xs: List[ttnn.Tensor],
        rot_mats: List[ttnn.Tensor],
        start_pos: int,
        attn_masks: List[ttnn.Tensor],
    ) -> List[ttnn.Tensor]:
        ### xs (residual stream) is fractured on all chips
        xs_replicated = ttnn.all_gather(
            xs,
            dim=3,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            memory_config=self.model_config["DECODER_ALL_GATHER_OUTPUT_MEMCFG"],
        )

        # In-place RMSNorm
        attn_norm_replicated = ttnn.rms_norm(
            xs_replicated,
            epsilon=self.norm_eps,
            weight=self.attn_norm,
            program_config=self.model_config["LN_ATTN_PROGCFG"],
            memory_config=self.model_config["LN_ATTN_OUTPUT_MEMCFG"],
            compute_kernel_config=self.model_config["LN_COMPUTE_KERNEL_CONFIG"],
        )
        # attn_norm_replicated is sharded

        # attn_outs is fractured
        attn_outs = self.attention(attn_norm_replicated, rot_mats, start_pos, attn_masks)

        ### Fractured residual add
        # Add attn output to residiual first in place to save memory
        output = xs
        output = ttnn.add(
            output,
            attn_outs,
            memory_config=self.model_config["ATTN_ADD_OUTPUT_MEMCFG"],
        )
        attn_outs.deallocate(True)

        attn_resid_replicated = ttnn.all_gather(
            output,
            dim=3,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            memory_config=self.model_config["DECODER_ALL_GATHER_OUTPUT_MEMCFG"],
        )

        # In-place RMSNorm
        ffn_norm_replicated = ttnn.rms_norm(
            attn_resid_replicated,
            epsilon=self.norm_eps,
            weight=self.ffn_norm,
            program_config=self.model_config["LN_MLP_PROGCFG"],
            memory_config=self.model_config["LN_MLP_OUTPUT_MEMCFG"],
            compute_kernel_config=self.model_config["LN_COMPUTE_KERNEL_CONFIG"],
        )
        # ffn_norm_replicated is sharded

        ffn_out = self.mlp(ffn_norm_replicated)

        ### residual in place
        output = ttnn.add(
            output,
            ffn_out,
            memory_config=self.model_config["MLP_ADD_OUTPUT_MEMCFG"],
        )
        ffn_out.deallocate(True)

        return output

    def tt_distributed_rmsnorm(self, inp, epsilon, gamma):
        # Run distributed rmsnorm part 1
        tt_stats = ttnn.rms_norm_pre_all_gather(
            inp, compute_kernel_config=self.model_config["LN_COMPUTE_KERNEL_CONFIG"], dtype=ttnn.bfloat16
        )

        # AllGather stats
        tt_stats = ttnn.all_gather(
            tt_stats,
            dim=3,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            memory_config=self.model_config["DRAM_MEMCFG"],
        )

        # Run distributed rmsnorm part 2
        tt_out = ttnn.rms_norm_post_all_gather(
            inp,
            tt_stats,
            epsilon=epsilon,
            weight=gamma,
            compute_kernel_config=self.model_config["LN_COMPUTE_KERNEL_CONFIG"],
        )

        tt_stats.deallocate(True)

        return tt_out

    def prefill_forward(
        self,
        xs: List[ttnn.Tensor],
        rot_mats: List[ttnn.Tensor],
        start_pos: int,
        attn_masks: List[ttnn.Tensor],
        user_id: int = 0,
    ) -> List[ttnn.Tensor]:
        ### xs (residual stream) is fractured on all chips
        # TODO: Reenable when typcast supports multidevice
        # xs_replicated = []
        # for i in range(self.num_devices):
        #     xs_replicated.append(
        #         ttnn.experimental.tensor.typecast(ttnn.clone(xs[i]), dtype=ttnn.bfloat8_b)
        #     )

        attn_norm_interleaved = self.tt_distributed_rmsnorm(xs, self.norm_eps, self.attn_norm_sharded)
        attn_norm_interleaved = ttnn.all_gather(
            attn_norm_interleaved,
            dim=3,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            memory_config=self.model_config["DRAM_MEMCFG"],
        )

        # attn_outs is fractured
        attn_outs = self.attention(attn_norm_interleaved, rot_mats, start_pos, attn_masks, user_id)

        attn_norm_interleaved.deallocate(True)

        ### Fractured residual add
        residual = xs
        output = ttnn.add(residual, attn_outs)
        attn_outs.deallocate(True)

        ffn_norm_interleaved = self.tt_distributed_rmsnorm(output, self.norm_eps, self.ffn_norm_sharded)
        ffn_norm_interleaved = ttnn.all_gather(
            ffn_norm_interleaved,
            dim=3,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            memory_config=self.model_config["DRAM_MEMCFG"],
        )

        ffn_out = self.mlp(ffn_norm_interleaved)

        ### residual add
        output = ttnn.add(output, ffn_out)
        ffn_out.deallocate(True)
        return output
