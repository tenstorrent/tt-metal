# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from .attention_wan import WanAttention
from ....layers.normalization import DistributedLayerNorm
from ....layers.linear import ColParallelLinear, Linear
from ....layers.feedforward import ParallelFeedForward

from ....layers.embeddings import WanPatchEmbed
from ....utils.mochi import get_rot_transformation_mat
from ....utils.substate import substate
from ....utils.padding import pad_vision_seq_parallel
from ....utils.tensor import bf16_tensor
from loguru import logger

from diffusers.models.transformers.transformer_wan import WanTimeTextImageEmbedding as TorchWanTimeTextImageEmbedding
from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed as TorchWanRotaryPosEmbed


class WanTransformerBlock:
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        cross_attention_norm: bool = True,
        eps: float = 1e-6,
        mesh_device=None,
        init=False,
        ccl_manager=None,
        parallel_config=None,
        is_fsdp=False,
    ):
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.cross_attention_norm = cross_attention_norm
        self.eps = eps

        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        fsdp_mesh_axis = self.parallel_config.sequence_parallel.mesh_axis if is_fsdp else None

        self.norm1 = DistributedLayerNorm(
            dim,
            norm_eps=eps,
            norm_elementwise_affine=False,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            init=init,
        )

        self.attn1 = WanAttention(
            dim=dim,
            num_heads=num_heads,
            eps=eps,
            mesh_device=mesh_device,
            init=init,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            is_fsdp=is_fsdp,
        )

        self.attn2 = WanAttention(
            dim=dim,
            num_heads=num_heads,
            eps=eps,
            mesh_device=mesh_device,
            init=init,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            is_fsdp=is_fsdp,
        )

        self.norm2 = (
            DistributedLayerNorm(
                dim,
                norm_eps=eps,
                norm_elementwise_affine=True,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                init=init,
            )
            if cross_attention_norm
            else None
        )

        self.ff = ParallelFeedForward(
            dim,
            inner_dim=ffn_dim,
            activation_fn="gelu",
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
            init=init,
            fsdp_mesh_axis=fsdp_mesh_axis,
        )

        self.norm3 = DistributedLayerNorm(
            dim,
            norm_eps=eps,
            norm_elementwise_affine=False,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            init=init,
        )

        self.scale_shift_table = None

        # NOTE: Used for DistributedLayerNorm workaround
        self.shard_spatial_on_tp = ColParallelLinear(
            in_features=dim,
            out_features=dim,
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            init=init,
        )

        self.ff_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # NOTE: This really should be FP32 acc but there's an L1 OOM issue in distributed LN,
        # which we're using to work around the large LN hang.
        self.layernorm_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        device_grid = self.mesh_device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(x=device_grid.x, y=device_grid.y)

    def to_cached_state_dict(self, path_prefix):
        cache_dict = {}

        # Cache linear layers
        norm1_cache = self.norm1.to_cached_state_dict(path_prefix + "norm1.")
        attn1_cache = self.attn1.to_cached_state_dict(path_prefix + "attn1.")
        attn2_cache = self.attn2.to_cached_state_dict(path_prefix + "attn2.")
        norm2_cache = self.norm2.to_cached_state_dict(path_prefix + "norm2.")
        ff_cache = self.ff.to_cached_state_dict(path_prefix + "ff.")
        norm3_cache = self.norm3.to_cached_state_dict(path_prefix + "norm3.")
        shard_spatial_on_tp_cache = self.shard_spatial_on_tp.to_cached_state_dict(path_prefix + "shard_spatial_on_tp.")
        ttnn.dump_tensor(path_prefix + "scale_shift_table", self.scale_shift_table)

        # Add prefixes for linear layers
        for key, value in norm1_cache.items():
            cache_dict[f"norm1.{key}"] = value
        for key, value in attn1_cache.items():
            cache_dict[f"attn1.{key}"] = value
        for key, value in attn2_cache.items():
            cache_dict[f"attn2.{key}"] = value
        for key, value in norm2_cache.items():
            cache_dict[f"norm2.{key}"] = value
        for key, value in ff_cache.items():
            cache_dict[f"ff.{key}"] = value
        for key, value in norm3_cache.items():
            cache_dict[f"norm3.{key}"] = value
        for key, value in shard_spatial_on_tp_cache.items():
            cache_dict[f"shard_spatial_on_tp.{key}"] = value
        cache_dict["scale_shift_table"] = path_prefix + "scale_shift_table"

        return cache_dict

    def from_cached_state_dict(self, cache_dict):
        self.norm1.from_cached_state_dict(substate(cache_dict, "norm1"))
        self.attn1.from_cached_state_dict(substate(cache_dict, "attn1"))
        self.attn2.from_cached_state_dict(substate(cache_dict, "attn2"))
        self.norm2.from_cached_state_dict(substate(cache_dict, "norm2"))
        self.ff.from_cached_state_dict(substate(cache_dict, "ff"))
        self.norm3.from_cached_state_dict(substate(cache_dict, "norm3"))
        self.shard_spatial_on_tp.from_cached_state_dict(substate(cache_dict, "shard_spatial_on_tp"))
        self.scale_shift_table = ttnn.load_tensor(cache_dict["scale_shift_table"], device=self.mesh_device)

    def load_state_dict(self, state_dict):
        self.norm1.load_state_dict(substate(state_dict, "norm1"))
        self.attn1.load_state_dict(substate(state_dict, "attn1"))
        self.attn2.load_state_dict(substate(state_dict, "attn2"))
        self.norm2.load_state_dict(substate(state_dict, "norm2"))

        def rename_ff_state(state):
            out_state = {
                f"{replacement}{k[len(prefix):]}": v
                for k, v in state.items()
                for prefix, replacement in [("net.0.proj", "ff1"), ("net.2", "ff2")]
                if prefix in k
            }
            return out_state

        self.ff.load_state_dict(rename_ff_state(substate(state_dict, "ffn")))
        self.norm3.load_state_dict(substate(state_dict, "norm3"))

        identity_tensor = torch.eye(self.dim)
        identity_state = {"weight": identity_tensor}
        self.shard_spatial_on_tp.load_state_dict(identity_state)

        self.scale_shift_table = bf16_tensor(state_dict["scale_shift_table"], device=self.mesh_device)

    def __call__(self, spatial_1BND, prompt_1BLP, temb_1BTD, N, rope_cos, rope_sin, trans_mat):
        """
        spatial_1BND: fractured N on SP, replicated D on TP
        prompt_1BLP: replicated on SP, replicated D on TP
        temb_1BTD: replicated on SP, replicated D on TP
        N: logical sequence length of the spatial input
        rope_cos_BANH: fractured N on SP, A (num_heads) on TP
        rope_sin_BANH: fractured N on SP, A (num_heads) on TP
        trans_mat: replicated on SP, replicated D on TP

        Outputs:
        spatial_1BND: fractured N on SP, replicated D on TP
        """

        assert temb_1BTD.shape[2] == 6, "wan2.2 14b expects 6 chunks in timestep embedding"

        shifted_temb_1BTD = self.scale_shift_table + temb_1BTD
        shift_msa_1B1D, scale_msa_1B1D, gate_msa_1B1D, c_shift_msa_1B1D, c_scale_msa_1B1D, c_gate_msa_1B1D = ttnn.chunk(
            shifted_temb_1BTD, 6, dim=2
        )

        # DistributedLayerNorm workaround
        if self.parallel_config.tensor_parallel.factor > 1:
            spatial_sharded_1BND = self.shard_spatial_on_tp(
                spatial_1BND, compute_kernel_config=self.ff_compute_kernel_config
            )
        else:
            spatial_sharded_1BND = spatial_1BND

        spatial_normed_1BND = self.norm1(
            spatial_sharded_1BND, compute_kernel_config=self.layernorm_compute_kernel_config
        )
        if self.parallel_config.tensor_parallel.factor > 1:
            spatial_normed_1BND = ttnn.experimental.all_gather_async(
                spatial_normed_1BND,
                persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                    spatial_normed_1BND.shape, 3, self.parallel_config.tensor_parallel.mesh_axis
                ),
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                    self.parallel_config.tensor_parallel.mesh_axis
                ),
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=self.parallel_config.tensor_parallel.mesh_axis,
            )
        spatial_normed_1BND = spatial_normed_1BND * (1.0 + scale_msa_1B1D) + shift_msa_1B1D

        # Self attention on spatial
        spatial_attn_1BND = self.attn1(
            spatial_1BND=spatial_normed_1BND,
            N=N,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            trans_mat=trans_mat,
        )

        # Residual
        spatial_1BND = spatial_1BND + spatial_attn_1BND * gate_msa_1B1D

        # Cross attention on prompt
        # DistributedLayerNorm workaround
        if self.parallel_config.tensor_parallel.factor > 1:
            spatial_sharded_1BND = self.shard_spatial_on_tp(
                spatial_1BND, compute_kernel_config=self.ff_compute_kernel_config
            )
        else:
            spatial_sharded_1BND = spatial_1BND
        spatial_normed_1BND = self.norm2(
            spatial_sharded_1BND, compute_kernel_config=self.layernorm_compute_kernel_config
        )
        if self.parallel_config.tensor_parallel.factor > 1:
            spatial_normed_1BND = ttnn.experimental.all_gather_async(
                spatial_normed_1BND,
                persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                    spatial_normed_1BND.shape, 3, self.parallel_config.tensor_parallel.mesh_axis
                ),
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                    self.parallel_config.tensor_parallel.mesh_axis
                ),
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=self.parallel_config.tensor_parallel.mesh_axis,
            )

        attn_output_1BND = self.attn2(
            spatial_1BND=spatial_normed_1BND,
            N=N,
            prompt_1BLP=prompt_1BLP,
        )
        spatial_1BND = spatial_1BND + attn_output_1BND

        # Feed Forward
        # DistributedLayerNorm workaround
        if self.parallel_config.tensor_parallel.factor > 1:
            spatial_sharded_1BND = self.shard_spatial_on_tp(
                spatial_1BND, compute_kernel_config=self.ff_compute_kernel_config
            )
        else:
            spatial_sharded_1BND = spatial_1BND
        spatial_normed_1BND = self.norm3(
            spatial_sharded_1BND, compute_kernel_config=self.layernorm_compute_kernel_config
        )
        if self.parallel_config.tensor_parallel.factor > 1:
            spatial_normed_1BND = ttnn.experimental.all_gather_async(
                spatial_normed_1BND,
                persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                    spatial_normed_1BND.shape, 3, self.parallel_config.tensor_parallel.mesh_axis
                ),
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                    self.parallel_config.tensor_parallel.mesh_axis
                ),
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=self.parallel_config.tensor_parallel.mesh_axis,
            )

        spatial_normed_1BND = spatial_normed_1BND * (1 + c_scale_msa_1B1D) + c_shift_msa_1B1D
        # NOTE: Cannot set core_grid for FF or you get L1 OOM. Needs to be fixed.
        spatial_ff_1BND = self.ff(
            spatial_normed_1BND, core_grid=None, compute_kernel_config=self.ff_compute_kernel_config
        )

        if self.parallel_config.tensor_parallel.factor > 1:
            # Gather spatial fractured on hidden dim
            spatial_ff_1BND = ttnn.experimental.all_gather_async(
                spatial_ff_1BND,
                persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                    spatial_ff_1BND.shape, 3, self.parallel_config.tensor_parallel.mesh_axis
                ),
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                    self.parallel_config.tensor_parallel.mesh_axis
                ),
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=self.parallel_config.tensor_parallel.mesh_axis,
            )

        spatial_1BND = spatial_1BND + spatial_ff_1BND * c_gate_msa_1B1D

        return spatial_1BND


class WanTransformer3DModel:
    def __init__(
        self,
        patch_size: tuple = (1, 2, 2),
        num_heads: int = 40,
        dim: int = 5120,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        rope_max_seq_len: int = 1024,
        mesh_device=None,
        init=False,
        ccl_manager=None,
        parallel_config=None,
        is_fsdp=True,
    ):
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        self.is_fsdp = is_fsdp
        self.fsdp_mesh_axis = self.parallel_config.sequence_parallel.mesh_axis if is_fsdp else None

        self.patch_size = patch_size
        self.dim = dim

        self.out_channels = out_channels

        # NOTE: Fallback
        self.rope = TorchWanRotaryPosEmbed(
            dim // num_heads,
            patch_size,
            rope_max_seq_len,
        )

        self.patch_embed = WanPatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=dim,
            mesh_device=mesh_device,
            init=init,
        )

        # NOTE: Torch fallback until we support WanCombinedTimestepCaptionEmbedding
        self.condition_embedder = TorchWanTimeTextImageEmbedding(
            dim=dim,
            time_freq_dim=freq_dim,
            time_proj_dim=dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=None,
            pos_embed_seq_len=None,
        )

        self.blocks = [
            WanTransformerBlock(
                dim=dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                cross_attention_norm=cross_attn_norm,
                eps=eps,
                mesh_device=mesh_device,
                init=init,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                is_fsdp=is_fsdp,
            )
            for i in range(num_layers)
        ]

        # NOTE: Used for DistributedLayerNorm workaround
        self.shard_spatial_on_tp = ColParallelLinear(
            in_features=dim,
            out_features=dim,
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            init=init,
        )

        self.norm_out = DistributedLayerNorm(
            dim,
            norm_eps=eps,
            norm_elementwise_affine=False,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            init=False,
        )

        self.proj_out = Linear(
            dim,
            patch_size[0] * patch_size[1] * patch_size[2] * self.out_channels,
            bias=True,
            mesh_device=mesh_device,
            init=init,
        )

        self.scale_shift_table = None

        self.hifi4_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        device_grid = self.mesh_device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(x=device_grid.x, y=device_grid.y)

    def to_cached_state_dict(self, path_prefix):
        cache_dict = {}

        # Cache patch embedding
        patch_embed_cache = self.patch_embed.to_cached_state_dict(path_prefix + "patch_embed.")
        for key, value in patch_embed_cache.items():
            cache_dict[f"patch_embed.{key}"] = value

        # Cache transformer blocks
        for i, block in enumerate(self.blocks):
            block_cache = block.to_cached_state_dict(path_prefix + f"blocks.{i}.")
            for key, value in block_cache.items():
                cache_dict[f"blocks.{i}.{key}"] = value

        # Cache fracture spatial input layer
        shard_spatial_on_tp_cache = self.shard_spatial_on_tp.to_cached_state_dict(path_prefix + "shard_spatial_on_tp.")
        for key, value in shard_spatial_on_tp_cache.items():
            cache_dict[f"shard_spatial_on_tp.{key}"] = value

        # Cache norm out layers
        norm_out_cache = self.norm_out.to_cached_state_dict(path_prefix + "norm_out.")
        proj_out_cache = self.proj_out.to_cached_state_dict(path_prefix + "proj_out.")

        for key, value in norm_out_cache.items():
            cache_dict[f"norm_out.{key}"] = value
        for key, value in proj_out_cache.items():
            cache_dict[f"proj_out.{key}"] = value

        ttnn.dump_tensor(path_prefix + "scale_shift_table", self.scale_shift_table)
        cache_dict["scale_shift_table"] = path_prefix + "scale_shift_table"

        # Torch fallbacks
        torch.save(self.rope.state_dict(), path_prefix + "rope.pt")
        torch.save(self.condition_embedder.state_dict(), path_prefix + "condition_embedder.pt")
        cache_dict["condition_embedder"] = path_prefix + "condition_embedder.pt"
        cache_dict["rope"] = path_prefix + "rope.pt"

        return cache_dict

    def from_cached_state_dict(self, cache_dict):
        self.patch_embed.from_cached_state_dict(substate(cache_dict, "patch_embed"))

        for i, block in enumerate(self.blocks):
            block.from_cached_state_dict(substate(cache_dict, f"blocks.{i}"))

        self.shard_spatial_on_tp.from_cached_state_dict(substate(cache_dict, "shard_spatial_on_tp"))
        self.norm_out.from_cached_state_dict(substate(cache_dict, "norm_out"))
        self.proj_out.from_cached_state_dict(substate(cache_dict, "proj_out"))
        self.scale_shift_table = ttnn.load_tensor(cache_dict["scale_shift_table"], device=self.mesh_device)

        # Torch fallbacks
        self.rope.load_state_dict(torch.load(cache_dict["rope"]))
        self.condition_embedder.load_state_dict(torch.load(cache_dict["condition_embedder"]))

    def load_state_dict(self, state_dict):
        self.rope.load_state_dict(substate(state_dict, "rope"))
        self.patch_embed.load_state_dict(substate(state_dict, "patch_embedding"))
        self.condition_embedder.load_state_dict(substate(state_dict, "condition_embedder"))
        for i, block in enumerate(self.blocks):
            block.load_state_dict(substate(state_dict, f"blocks.{i}"))
        self.norm_out.load_state_dict(substate(state_dict, "norm_out"))
        self.proj_out.load_state_dict(substate(state_dict, "proj_out"))

        identity_tensor = torch.eye(self.dim)
        identity_state = {"weight": identity_tensor}
        self.shard_spatial_on_tp.load_state_dict(identity_state)

        self.scale_shift_table = bf16_tensor(state_dict["scale_shift_table"], device=self.mesh_device)

    def prepare_rope_features(self, hidden_states):
        """
        Given video input, compute RoPE features.
        Return tensors on device.
        """
        rope_cos, rope_sin = self.rope(hidden_states)

        # Convert to TT tensors with proper sharding
        rope_cos_1HND = rope_cos.permute(0, 2, 1, 3)
        rope_sin_1HND = rope_sin.permute(0, 2, 1, 3)

        rope_cos_1HND = pad_vision_seq_parallel(
            rope_cos_1HND, chunk_size_lcm=256, num_devices=self.parallel_config.sequence_parallel.factor
        )
        rope_sin_1HND = pad_vision_seq_parallel(
            rope_sin_1HND, chunk_size_lcm=256, num_devices=self.parallel_config.sequence_parallel.factor
        )

        trans_mat = get_rot_transformation_mat()

        tt_rope_cos_1HND = bf16_tensor(
            rope_cos_1HND,
            device=self.mesh_device,
            mesh_axis=self.parallel_config.sequence_parallel.mesh_axis,
            shard_dim=-2,
        )
        tt_rope_sin_1HND = bf16_tensor(
            rope_sin_1HND,
            device=self.mesh_device,
            mesh_axis=self.parallel_config.sequence_parallel.mesh_axis,
            shard_dim=-2,
        )
        tt_trans_mat = bf16_tensor(trans_mat, device=self.mesh_device)

        logger.info(f"TT rope cos shape: {tt_rope_cos_1HND.shape}")
        logger.info(f"TT rope sin shape: {tt_rope_sin_1HND.shape}")
        logger.info(f"TT trans mat shape: {tt_trans_mat.shape}")

        return tt_rope_cos_1HND, tt_rope_sin_1HND, tt_trans_mat

    def prepare_conditioning(self, timestep, encoder_hidden_states):
        """
        Given torch inputs, execute the combined timestep and text embedding.
        Return tensors on device.
        """
        assert timestep.ndim == 1, "Wan2.2-T2V requires a 1D timestep tensor"
        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image=None, timestep_seq_len=None
        )
        assert encoder_hidden_states_image is None, "Wan2.2-T2V does not support image conditioning"

        timestep_proj = timestep_proj.unflatten(1, (6, -1))
        logger.info(f"temb shape: {temb.shape}")
        logger.info(f"encoder_hidden_states shape: {encoder_hidden_states.shape}")

        tt_temb_11BD = bf16_tensor(temb.unsqueeze(0).unsqueeze(0), device=self.mesh_device)
        tt_timestep_proj_1BTD = bf16_tensor(timestep_proj.unsqueeze(0), device=self.mesh_device)
        tt_prompt_1BLP = bf16_tensor(encoder_hidden_states.unsqueeze(0), device=self.mesh_device)

        logger.info(f"TT temb shape: {tt_temb_11BD.shape}")
        logger.info(f"TT timestep proj shape: {tt_timestep_proj_1BTD.shape}")
        logger.info(f"TT prompt shape: {tt_prompt_1BLP.shape}")
        return tt_temb_11BD, tt_timestep_proj_1BTD, tt_prompt_1BLP

    def preprocess_spatial_input(self, spatial):
        B, C, F, H, W = spatial.shape
        logger.info(f"Preprocessing spatial input with shape {spatial.shape}")
        assert B == 1, "Batch size must be 1"
        pF, pH, pW = self.patch_size
        patch_F, patch_H, patch_W = F // pF, H // pH, W // pW
        N = patch_F * patch_H * patch_W

        # Patchify video input
        spatial = spatial.reshape(B, C, patch_F, pF, patch_H, pH, patch_W, pW)
        spatial = spatial.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(1, B, N, pF * pH * pW * C)
        logger.info(f"spatial input after patchifying: {spatial.shape}")

        spatial = pad_vision_seq_parallel(
            spatial, chunk_size_lcm=256, num_devices=self.parallel_config.sequence_parallel.factor
        )
        logger.info(f"spatial input after padding: {spatial.shape}")

        spatial = bf16_tensor(
            spatial, device=self.mesh_device, mesh_axis=self.parallel_config.sequence_parallel.mesh_axis, shard_dim=-2
        )
        logger.info(f"TT spatial shape: {spatial.shape}")
        return spatial, N

    def postprocess_spatial_output(self, spatial_1BND, F, H, W, N):
        """
        This is the reverse of preprocess_spatial_input
        Input is of shape: 1 B (patch_F patch_H patch_W) (pF pH pW C)
        returns shape: B C F H W
        """
        assert len(spatial_1BND.shape) == 4
        assert spatial_1BND.shape[0] == 1
        B = spatial_1BND.shape[1]
        pF, pH, pW = self.patch_size
        patch_F, patch_H, patch_W = F // pF, H // pH, W // pW
        logger.info(f"Postprocessing spatial output with shape {spatial_1BND.shape}")

        # Gather sequence-parallel output
        spatial_1BND = ttnn.experimental.all_gather_async(
            spatial_1BND,
            persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                spatial_1BND.shape, 2, self.parallel_config.sequence_parallel.mesh_axis
            ),
            dim=2,
            multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                self.parallel_config.sequence_parallel.mesh_axis
            ),
            num_links=self.ccl_manager.num_links,
            topology=self.ccl_manager.topology,
            cluster_axis=self.parallel_config.sequence_parallel.mesh_axis,
        )
        logger.info(f"Spatial output after gathering: {spatial_1BND.shape}")
        spatial_BND = ttnn.to_torch(ttnn.get_device_tensors(spatial_1BND)[0]).squeeze(0)

        spatial_BND = spatial_BND[:, :N]  # Slice out sequence-parallel padding tokens
        logger.info(f"Spatial output after slicing: {spatial_BND.shape}")

        spatial_patches = spatial_BND.reshape(B, patch_F, patch_H, patch_W, pF, pH, pW, self.out_channels)
        logger.info(f"Spatial output after reshaping: {spatial_patches.shape}")

        spatial_BCFHW = spatial_patches.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, self.out_channels, F, H, W)
        logger.info(f"Spatial output after permuting: {spatial_BCFHW.shape}")
        return spatial_BCFHW

    def __call__(self, spatial, prompt, timestep):
        """
        Inputs are all torch tensors
        Output is torch tensor
        """

        B, C, F, H, W = spatial.shape
        pF, pH, pW = self.patch_size
        patch_F, patch_H, patch_W = F // pF, H // pH, W // pW
        N = patch_F * patch_H * patch_W

        rope_cos_1HND, rope_sin_1HND, trans_mat = self.prepare_rope_features(spatial)

        temb_11BD, timestep_proj_1BTD, prompt_1BLP = self.prepare_conditioning(timestep, prompt)

        spatial_1BNI, N = self.preprocess_spatial_input(spatial)

        spatial_1BND = self.patch_embed(spatial_1BNI)

        for idx, block in enumerate(self.blocks):
            spatial_1BND = block(
                spatial_1BND=spatial_1BND,
                prompt_1BLP=prompt_1BLP,
                temb_1BTD=timestep_proj_1BTD,
                N=N,
                rope_cos=rope_cos_1HND,
                rope_sin=rope_sin_1HND,
                trans_mat=trans_mat,
            )

        scale_shift_1BSD = self.scale_shift_table + temb_11BD
        shift_11BD, scale_11BD = ttnn.chunk(scale_shift_1BSD, 2, -2)

        ## SUPER HACKY WORKAROUND
        # Large tensor layernorm is hanging, issue #20789
        # The workaround is to use distributed layernorm by fracturing the input on the TP axis

        if self.parallel_config.tensor_parallel.factor > 1:
            spatial_fractured_1BND = self.shard_spatial_on_tp(
                spatial_1BND, core_grid=self.core_grid, compute_kernel_config=self.hifi4_compute_kernel_config
            )
        else:
            spatial_fractured_1BND = spatial_1BND
        spatial_norm_1BND = self.norm_out(spatial_fractured_1BND)

        if self.parallel_config.tensor_parallel.factor > 1:
            spatial_norm_1BND = ttnn.experimental.all_gather_async(
                spatial_norm_1BND,
                persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                    spatial_norm_1BND.shape, 3, self.parallel_config.tensor_parallel.mesh_axis
                ),
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                    self.parallel_config.tensor_parallel.mesh_axis
                ),
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=self.parallel_config.tensor_parallel.mesh_axis,
            )

        spatial_norm_1BND = spatial_norm_1BND * (1 + scale_11BD) + shift_11BD

        proj_out_1BNI = self.proj_out(
            spatial_norm_1BND, core_grid=self.core_grid, compute_kernel_config=self.hifi4_compute_kernel_config
        )

        spatial_out = self.postprocess_spatial_output(proj_out_1BNI, F, H, W, N)

        return spatial_out
