# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List
import ttnn

from models.demos.tg.llama3_70b.tt.llama_attention_galaxy import TtLlamaAttention_galaxy
from models.demos.tg.llama3_70b.tt.llama_mlp_galaxy import TtLlamaMLP_galaxy
from models.demos.t3000.llama2_70b.tt.llama_common import (
    ShardTensor2dMesh,
)
from models.demos.tg.llama3_70b.tt.llama_common import tt_all_gather, tt_sharded_distributed_rmsnorm


class TtLlamaDecoder_galaxy:
    def __init__(
        self,
        mesh_device,
        cluster_shape,
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
        self.cluster_shape = cluster_shape

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

        self.attention = TtLlamaAttention_galaxy(
            mesh_device,
            cluster_shape,
            state_dict,
            base_url,
            layer_num,
            model_config,
            configuration,
            transformation_mats,
            cache_path=cache_path,
            read_cache=read_cache,
        )

        self.mlp = TtLlamaMLP_galaxy(
            mesh_device,
            cluster_shape,
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

        # attn_norm_cache_str = f"{self.layer_name}.attention_norm_galaxy.weight"
        # ffn_norm_cache_str = f"{self.layer_name}.ffn_norm_galaxy.weight"

        attn_norm_sharded_str = f"{self.layer_name}.attention_norm_sharded_galaxy.weight"
        ffn_norm_sharded_str = f"{self.layer_name}.ffn_norm_sharded_galaxy.weight"

        pt_attn_norm = None
        pt_ffn_norm = None
        if not self.read_cache:
            pt_attn_norm = self.state_dict[attn_norm_str].reshape([1, 1, -1, 32])
            pt_ffn_norm = self.state_dict[ffn_norm_str].reshape([1, 1, -1, 32])

        self.attn_norm_sharded = ttnn.as_tensor(
            pt_attn_norm,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ShardTensor2dMesh(self.mesh_device, (2, None), self.cluster_shape),
            cache_file_name=self.cache_path / attn_norm_sharded_str,
        )

        self.ffn_norm_sharded = ttnn.as_tensor(
            pt_ffn_norm,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ShardTensor2dMesh(self.mesh_device, (2, None), self.cluster_shape),
            cache_file_name=self.cache_path / ffn_norm_sharded_str,
        )

    def __call__(
        self,
        xs: List[ttnn.Tensor],
        rot_mats: List[ttnn.Tensor],
        start_pos: int,
        attn_masks: List[ttnn.Tensor],
        user_id: int = 0,
        mode="decode",
    ) -> ttnn.Tensor:
        self.decoder_config = self.model_config["decoder"][mode]
        if mode == "decode":
            return self.decode_forward(xs, rot_mats, start_pos, attn_masks)
        elif mode == "prefill":
            return self.prefill_forward(xs, rot_mats, attn_masks, user_id)
        else:
            raise ValueError(f"Unknown llm_mode: {mode}")

    def tt_distributed_rmsnorm(self, inp, epsilon, gamma):
        # Run distributed rmsnorm part 1
        tt_stats = ttnn.rms_norm_pre_all_gather(
            inp, compute_kernel_config=self.decoder_config["LN_COMPUTE_KERNEL_CONFIG"], dtype=ttnn.bfloat16
        )

        tt_stats = ttnn.reshape(
            tt_stats,
            ttnn.Shape((1, 1, inp.shape.with_tile_padding()[-2], 32), (1, 1, inp.shape.with_tile_padding()[-2], 32)),
        )  # TODO: Figure out why we need this

        tt_stats = tt_all_gather(
            tt_stats,
            mesh_device=self.mesh_device,
            dim=3,
            cluster_axis=1,
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Run distributed rmsnorm part 2
        tt_out = ttnn.rms_norm_post_all_gather(
            inp,
            tt_stats,
            epsilon=epsilon,
            weight=gamma,
            compute_kernel_config=self.decoder_config["LN_COMPUTE_KERNEL_CONFIG"],
        )

        tt_stats.deallocate(True)

        return tt_out

    def decode_forward(
        self,
        xs: List[ttnn.Tensor],
        rot_mats: List[ttnn.Tensor],
        start_pos: int,
        attn_masks: List[ttnn.Tensor],
    ) -> List[ttnn.Tensor]:
        attn_norm_out = tt_sharded_distributed_rmsnorm(
            self.mesh_device,
            xs,
            epsilon=self.norm_eps,
            gamma=self.attn_norm_sharded,
        )

        attn_norm_out = ttnn.to_memory_config(attn_norm_out, memory_config=self.decoder_config["ATTN_ACT_MEMCFG"])
        attn_outs = self.attention(attn_norm_out, rot_mats, start_pos, attn_masks, mode="decode")
        attn_outs = ttnn.to_memory_config(attn_outs, memory_config=self.decoder_config["MLP_ACT_MEMCFG"])

        output = xs
        output = ttnn.add(
            output,
            attn_outs,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        attn_outs.deallocate(True)

        ffn_norm_out = tt_sharded_distributed_rmsnorm(
            self.mesh_device,
            output,
            epsilon=self.norm_eps,
            gamma=self.ffn_norm_sharded,
        )

        ffn_norm_out = ttnn.to_memory_config(ffn_norm_out, memory_config=self.decoder_config["MLP_ACT_MEMCFG"])
        ffn_out = self.mlp(ffn_norm_out, mode="decode")
        ffn_norm_out.deallocate(True)
        ### residual add
        output = ttnn.add(
            output,
            ffn_out,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        ffn_out.deallocate(True)

        return output

    def prefill_forward(
        self,
        xs: List[ttnn.Tensor],
        rot_mats: List[ttnn.Tensor],
        attn_masks: List[ttnn.Tensor],
        user_id: int,
    ) -> List[ttnn.Tensor]:
        attn_outs = self.tt_distributed_rmsnorm(
            xs,
            epsilon=self.norm_eps,
            gamma=self.attn_norm_sharded,
        )

        attn_outs = self.attention(attn_outs, rot_mats, 0, attn_masks, user_id, mode="prefill")

        output = xs
        output = ttnn.add(
            output,
            attn_outs,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ffn_norm_out = self.tt_distributed_rmsnorm(
            output,
            epsilon=self.norm_eps,
            gamma=self.ffn_norm_sharded,
        )

        ffn_out = self.mlp(ffn_norm_out, mode="prefill")

        # residual add
        output = ttnn.add(
            output,
            ffn_out,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return output
