# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from .attention_mochi import MochiAttention
from ...layers.normalization import RMSNorm, DistributedLayerNorm
from ...layers.linear import ColParallelLinear, Linear
from ...layers.feedforward import ParallelFeedForward
from ...layers.embeddings import MochiPatchEmbed
from ...utils.mochi import stack_cos_sin, get_rot_transformation_mat
from ...utils.substate import substate
from ...utils.padding import pad_vision_seq_parallel
from ...utils.tensor import bf16_tensor, bf16_tensor_2dshard
from loguru import logger

from diffusers.models.embeddings import (
    MochiCombinedTimestepCaptionEmbedding as TorchMochiCombinedTimestepCaptionEmbedding,
)
from diffusers.models.transformers.transformer_mochi import MochiRoPE


class MochiTransformerBlock:
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        pooled_projection_dim: int,
        activation_fn: str = "swiglu",
        context_pre_only: bool = False,
        eps: float = 1e-6,
        mesh_device=None,
        init=False,
        ccl_manager=None,
        parallel_config=None,
        is_fsdp=False,
    ):
        self.context_pre_only = context_pre_only
        self.ff_inner_dim = (4 * dim * 2) // 3
        self.ff_context_inner_dim = (4 * pooled_projection_dim * 2) // 3
        self.dim = dim
        self.text_dim = pooled_projection_dim

        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        fsdp_mesh_axis = self.parallel_config.sequence_parallel.mesh_axis if is_fsdp else None

        rms_zero_kwargs = {
            "embedding_dim": dim,
            "norm_eps": eps,
            "norm_elementwise_affine": False,
            "bias": False,
            "mesh_device": mesh_device,
        }

        self.norm1_linear = ColParallelLinear(
            dim,
            4 * dim,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )
        self.norm1_norm = RMSNorm(**rms_zero_kwargs)

        if not context_pre_only:
            self.norm1_context_linear = ColParallelLinear(
                dim,
                4 * pooled_projection_dim,
                bias=True,
                mesh_device=mesh_device,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
                fsdp_mesh_axis=fsdp_mesh_axis,
                ccl_manager=ccl_manager,
            )
        else:
            self.norm1_context_linear = ColParallelLinear(
                dim,
                pooled_projection_dim,
                bias=True,
                mesh_device=mesh_device,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
                fsdp_mesh_axis=fsdp_mesh_axis,
                ccl_manager=ccl_manager,
            )
        self.norm1_context_norm = RMSNorm(**rms_zero_kwargs)

        self.attn1 = MochiAttention(
            query_dim=dim,
            added_kv_proj_dim=pooled_projection_dim,
            heads=num_attention_heads,
            head_dim=attention_head_dim,
            bias=False,
            added_proj_bias=False,
            out_dim=dim,
            out_context_dim=pooled_projection_dim,
            context_pre_only=context_pre_only,
            eps=1e-5,
            mesh_device=mesh_device,
            init=init,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            is_fsdp=is_fsdp,
        )

        self.norm2_norm = RMSNorm(**rms_zero_kwargs)
        self.norm2_context_norm = RMSNorm(**rms_zero_kwargs) if not self.context_pre_only else None

        self.norm3_norm = RMSNorm(**rms_zero_kwargs)
        self.norm3_context_norm = RMSNorm(**rms_zero_kwargs) if not self.context_pre_only else None

        self.ff = ParallelFeedForward(
            dim,
            inner_dim=self.ff_inner_dim,
            activation_fn=activation_fn,
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
            fsdp_mesh_axis=fsdp_mesh_axis,
        )

        self.ff_context = None
        if not context_pre_only:
            self.ff_context = ParallelFeedForward(
                pooled_projection_dim,
                inner_dim=self.ff_context_inner_dim,
                activation_fn=activation_fn,
                bias=False,
                mesh_device=mesh_device,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
                ccl_manager=ccl_manager,
                fsdp_mesh_axis=fsdp_mesh_axis,
            )

        self.norm4_norm = RMSNorm(**rms_zero_kwargs)
        self.norm4_context_norm = RMSNorm(**rms_zero_kwargs) if not self.context_pre_only else None

        self.temb_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self.ff_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # Found that HiFi4 RMSNorm hurts accuracy, so let RMSNorm use default
        # self.rms_compute_kernel_config = ttnn.init_device_compute_kernel_config(
        #     mesh_device.arch(),
        #     math_fidelity=ttnn.MathFidelity.HiFi4,
        #     math_approx_mode=False,
        #     fp32_dest_acc_en=True,
        #     packer_l1_acc=True,
        # )
        self.rms_compute_kernel_config = None

        device_grid = self.mesh_device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(x=device_grid.x, y=device_grid.y)

    def to_cached_state_dict(self, path_prefix):
        cache_dict = {}

        # Cache linear layers
        norm1_linear_cache = self.norm1_linear.to_cached_state_dict(path_prefix + "norm1_linear.")
        norm1_context_linear_cache = self.norm1_context_linear.to_cached_state_dict(
            path_prefix + "norm1_context_linear."
        )

        # Add prefixes for linear layers
        for key, value in norm1_linear_cache.items():
            cache_dict[f"norm1_linear.{key}"] = value
        for key, value in norm1_context_linear_cache.items():
            cache_dict[f"norm1_context_linear.{key}"] = value

        # Cache normalization layers
        norm1_norm_cache = self.norm1_norm.to_cached_state_dict(path_prefix + "norm1_norm.")
        norm1_context_norm_cache = self.norm1_context_norm.to_cached_state_dict(path_prefix + "norm1_context_norm.")
        norm2_norm_cache = self.norm2_norm.to_cached_state_dict(path_prefix + "norm2_norm.")
        norm3_norm_cache = self.norm3_norm.to_cached_state_dict(path_prefix + "norm3_norm.")
        norm4_norm_cache = self.norm4_norm.to_cached_state_dict(path_prefix + "norm4_norm.")

        # Add prefixes for norm layers
        for key, value in norm1_norm_cache.items():
            cache_dict[f"norm1_norm.{key}"] = value
        for key, value in norm1_context_norm_cache.items():
            cache_dict[f"norm1_context_norm.{key}"] = value
        for key, value in norm2_norm_cache.items():
            cache_dict[f"norm2_norm.{key}"] = value
        for key, value in norm3_norm_cache.items():
            cache_dict[f"norm3_norm.{key}"] = value
        for key, value in norm4_norm_cache.items():
            cache_dict[f"norm4_norm.{key}"] = value

        # Cache optional context norm layers
        if self.norm2_context_norm is not None:
            norm2_context_norm_cache = self.norm2_context_norm.to_cached_state_dict(path_prefix + "norm2_context_norm.")
            for key, value in norm2_context_norm_cache.items():
                cache_dict[f"norm2_context_norm.{key}"] = value

        if self.norm3_context_norm is not None:
            norm3_context_norm_cache = self.norm3_context_norm.to_cached_state_dict(path_prefix + "norm3_context_norm.")
            for key, value in norm3_context_norm_cache.items():
                cache_dict[f"norm3_context_norm.{key}"] = value

        if self.norm4_context_norm is not None:
            norm4_context_norm_cache = self.norm4_context_norm.to_cached_state_dict(path_prefix + "norm4_context_norm.")
            for key, value in norm4_context_norm_cache.items():
                cache_dict[f"norm4_context_norm.{key}"] = value

        # Cache attention layer
        attn1_cache = self.attn1.to_cached_state_dict(path_prefix + "attn1.")
        for key, value in attn1_cache.items():
            cache_dict[f"attn1.{key}"] = value

        # Cache feedforward layers
        ff_cache = self.ff.to_cached_state_dict(path_prefix + "ff.")
        for key, value in ff_cache.items():
            cache_dict[f"ff.{key}"] = value

        if self.ff_context is not None:
            ff_context_cache = self.ff_context.to_cached_state_dict(path_prefix + "ff_context.")
            for key, value in ff_context_cache.items():
                cache_dict[f"ff_context.{key}"] = value

        return cache_dict

    def from_cached_state_dict(self, cache_dict):
        self.norm1_linear.from_cached_state_dict(substate(cache_dict, "norm1_linear"))
        self.norm1_context_linear.from_cached_state_dict(substate(cache_dict, "norm1_context_linear"))

        self.norm1_norm.from_cached_state_dict(substate(cache_dict, "norm1_norm"))
        self.norm1_context_norm.from_cached_state_dict(substate(cache_dict, "norm1_context_norm"))
        self.norm2_norm.from_cached_state_dict(substate(cache_dict, "norm2_norm"))
        self.norm3_norm.from_cached_state_dict(substate(cache_dict, "norm3_norm"))
        self.norm4_norm.from_cached_state_dict(substate(cache_dict, "norm4_norm"))

        if self.norm2_context_norm is not None:
            self.norm2_context_norm.from_cached_state_dict(substate(cache_dict, "norm2_context_norm"))
        if self.norm3_context_norm is not None:
            self.norm3_context_norm.from_cached_state_dict(substate(cache_dict, "norm3_context_norm"))
        if self.norm4_context_norm is not None:
            self.norm4_context_norm.from_cached_state_dict(substate(cache_dict, "norm4_context_norm"))

        self.attn1.from_cached_state_dict(substate(cache_dict, "attn1"))
        self.ff.from_cached_state_dict(substate(cache_dict, "ff"))

        if self.ff_context is not None:
            self.ff_context.from_cached_state_dict(substate(cache_dict, "ff_context"))

    def load_state_dict(self, state_dict):
        self.norm1_linear.load_state_dict(substate(state_dict, "norm1.linear"))
        context_linear_key = "norm1_context.linear" if not self.context_pre_only else "norm1_context.linear_1"
        self.norm1_context_linear.load_state_dict(substate(state_dict, context_linear_key))
        self.attn1.load_state_dict(substate(state_dict, "attn1"))

        def rename_ff_state(state):
            out_state = {
                f"{replacement}{k[len(prefix):]}": v
                for k, v in state.items()
                for prefix, replacement in [("net.0.proj", "ff1"), ("net.2", "ff2")]
                if prefix in k
            }
            return out_state

        self.ff.load_state_dict(rename_ff_state(substate(state_dict, "ff")))
        if not self.context_pre_only:
            self.ff_context.load_state_dict(rename_ff_state(substate(state_dict, "ff_context")))

    def __call__(self, spatial_1BND, prompt_1BLP, temb_11BD, N, rope_cos, rope_sin, trans_mat):
        """
        spatial_1BND: fractured N on SP, replicated D on TP
        prompt_1BLP: replicated on SP, replicated D on TP
        temb_11BD: replicated on SP, replicated D on TP
        N: logical sequence length of the spatial input
        rope_cos_BANH: fractured N on SP, A (num_heads) on TP
        rope_sin_BANH: fractured N on SP, A (num_heads) on TP
        trans_mat: replicated on SP, replicated D on TP

        Outputs:
        spatial_1BND: fractured N on SP, fractured D on TP
        prompt_1BLP: replicated on SP, fractured D on TP
        """

        # NOTE: SILU is not very accurate, should try to move to host?
        silu_temb_11BD = ttnn.silu(temb_11BD)

        # Returns 4*dim fractured on TP
        mod_spatial_11BZ = self.norm1_linear(silu_temb_11BD, compute_kernel_config=self.temb_compute_kernel_config)
        if self.parallel_config.tensor_parallel.factor > 1:
            mod_spatial_11BZ = ttnn.experimental.all_gather_async(
                mod_spatial_11BZ,
                persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                    mod_spatial_11BZ.shape, 3, self.parallel_config.tensor_parallel.mesh_axis
                ),
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                    self.parallel_config.tensor_parallel.mesh_axis
                ),
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=self.parallel_config.tensor_parallel.mesh_axis,
            )

        scale_msa_11BD = mod_spatial_11BZ[:, :, :, : self.dim]
        gate_msa_11BD = mod_spatial_11BZ[:, :, :, self.dim : 2 * self.dim]
        scale_mlp_11BD = mod_spatial_11BZ[:, :, :, 2 * self.dim : 3 * self.dim]
        gate_mlp_11BD = mod_spatial_11BZ[:, :, :, 3 * self.dim :]

        # Norm spatial input (MochiRMSNormZero)
        spatial_normed_1BND = self.norm1_norm(spatial_1BND, compute_kernel_config=self.rms_compute_kernel_config) * (
            1.0 + scale_msa_11BD
        )

        # Returns 4*dim if not context_pre_only, dim if context_pre_only, fractured TP
        mod_prompt_11BZ = self.norm1_context_linear(
            silu_temb_11BD, compute_kernel_config=self.temb_compute_kernel_config
        )
        if self.parallel_config.tensor_parallel.factor > 1:
            mod_prompt_11BZ = ttnn.experimental.all_gather_async(
                mod_prompt_11BZ,
                persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                    mod_prompt_11BZ.shape, 3, self.parallel_config.tensor_parallel.mesh_axis
                ),
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                    self.parallel_config.tensor_parallel.mesh_axis
                ),
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=self.parallel_config.tensor_parallel.mesh_axis,
            )

        if not self.context_pre_only:
            prompt_scale_msa_11BD = mod_prompt_11BZ[:, :, :, : self.text_dim]
            prompt_gate_msa_11BD = mod_prompt_11BZ[:, :, :, self.text_dim : 2 * self.text_dim]
            prompt_scale_mlp_11BD = mod_prompt_11BZ[:, :, :, 2 * self.text_dim : 3 * self.text_dim]
            prompt_gate_mlp_11BD = mod_prompt_11BZ[:, :, :, 3 * self.text_dim :]
        else:
            prompt_scale_msa_11BD = mod_prompt_11BZ

        # Norm prompt input (MochiRMSNormZero)
        prompt_normed_1BLP = self.norm1_context_norm(
            prompt_1BLP, compute_kernel_config=self.rms_compute_kernel_config
        ) * (1.0 + prompt_scale_msa_11BD)

        spatial_attn_1BND, prompt_attn_1BLP = self.attn1(
            spatial_1BND=spatial_normed_1BND,
            prompt_1BLP=prompt_normed_1BLP,
            N=N,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            trans_mat=trans_mat,
        )

        # ModulatedRMSNorm
        spatial_attn_mod_1BND = self.norm2_norm(
            spatial_attn_1BND, compute_kernel_config=self.rms_compute_kernel_config
        ) * ttnn.tanh(gate_msa_11BD, fast_and_approximate_mode=False)

        # Residual
        spatial_1BND = spatial_1BND + spatial_attn_mod_1BND

        # Norm spatial input (MochiRMSNormZero) for FF
        spatial_normed_1BND = self.norm3_norm(spatial_1BND, compute_kernel_config=self.rms_compute_kernel_config) * (
            1.0 + scale_mlp_11BD
        )

        spatial_ff_1BND = self.ff(spatial_normed_1BND, compute_kernel_config=self.ff_compute_kernel_config)

        # Gather spatial FF output
        if self.parallel_config.tensor_parallel.factor > 1:
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

        spatial_ff_mod_1BND = self.norm4_norm(
            spatial_ff_1BND, compute_kernel_config=self.rms_compute_kernel_config
        ) * ttnn.tanh(gate_mlp_11BD, fast_and_approximate_mode=False)

        # Residual
        spatial_1BND = spatial_1BND + spatial_ff_mod_1BND

        if not self.context_pre_only:
            # Norm attention output (MochiRMSNormZero)
            prompt_attn_mod_1BLP = self.norm2_context_norm(
                prompt_attn_1BLP, compute_kernel_config=self.rms_compute_kernel_config
            ) * ttnn.tanh(prompt_gate_msa_11BD, fast_and_approximate_mode=False)

            # Residual
            prompt_1BLP = prompt_1BLP + prompt_attn_mod_1BLP

            # Norm prompt input (MochiRMSNormZero) for FF
            prompt_normed_1BLP = self.norm3_context_norm(
                prompt_1BLP, compute_kernel_config=self.rms_compute_kernel_config
            ) * (1.0 + prompt_scale_mlp_11BD)

            # TODO: Pass core_grid, compute_kernel_config for correctness check
            prompt_ff_1BLP = self.ff_context(prompt_normed_1BLP, compute_kernel_config=self.ff_compute_kernel_config)

            # Gather prompt FF output
            if self.parallel_config.tensor_parallel.factor > 1:
                prompt_ff_1BLP = ttnn.experimental.all_gather_async(
                    prompt_ff_1BLP,
                    persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                        prompt_ff_1BLP.shape, 3, self.parallel_config.tensor_parallel.mesh_axis
                    ),
                    dim=3,
                    multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                        self.parallel_config.tensor_parallel.mesh_axis
                    ),
                    num_links=self.ccl_manager.num_links,
                    topology=self.ccl_manager.topology,
                    cluster_axis=self.parallel_config.tensor_parallel.mesh_axis,
                )

            prompt_ff_mod_1BLP = self.norm4_context_norm(
                prompt_ff_1BLP, compute_kernel_config=self.rms_compute_kernel_config
            ) * ttnn.tanh(prompt_gate_mlp_11BD, fast_and_approximate_mode=False)

            # Residual
            prompt_1BLP = prompt_1BLP + prompt_ff_mod_1BLP

        return spatial_1BND, prompt_1BLP


class MochiTransformer3DModel:
    def __init__(
        self,
        patch_size: int = 2,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        num_layers: int = 48,
        pooled_projection_dim: int = 1536,
        in_channels: int = 12,
        out_channels=None,
        qk_norm: str = "rms_norm",
        text_embed_dim: int = 4096,
        time_embed_dim: int = 256,
        activation_fn: str = "swiglu",
        max_sequence_length: int = 256,
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

        self.patch_size = patch_size

        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim
        self.out_channels = out_channels or in_channels

        self.patch_embed = MochiPatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            mesh_device=mesh_device,
        )

        # NOTE: Torch fallback until we support MochiCombinedTimestepCaptionEmbedding
        self.time_embed = TorchMochiCombinedTimestepCaptionEmbedding(
            embedding_dim=inner_dim,
            pooled_projection_dim=pooled_projection_dim,
            text_embed_dim=text_embed_dim,
            time_embed_dim=time_embed_dim,
            num_attention_heads=8,
        )

        # NOTE: Fallback
        self.pos_frequencies = torch.nn.Parameter(torch.full((3, num_attention_heads, attention_head_dim // 2), 0.0))
        self.rope = MochiRoPE()

        self.transformer_blocks = [
            MochiTransformerBlock(
                dim=inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                pooled_projection_dim=pooled_projection_dim,
                activation_fn=activation_fn,
                context_pre_only=(i == num_layers - 1),
                mesh_device=mesh_device,
                init=init,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                is_fsdp=is_fsdp,
            )
            for i in range(num_layers)
        ]

        self.fracture_spatial_input = ColParallelLinear(
            in_features=inner_dim,
            out_features=inner_dim,
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
        )

        # self.norm_out_norm = LayerNorm(
        #     embedding_dim=inner_dim,
        #     norm_eps=1e-6,
        #     norm_elementwise_affine=False,
        #     mesh_device=mesh_device,
        #     init=False,
        #     # use_row_major_workaround=True, # Issue #20789, hang for large tensor layernorm
        # )

        self.norm_out_norm = DistributedLayerNorm(
            embedding_dim=inner_dim,
            norm_eps=1e-6,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

        self.norm_out_linear = Linear(
            in_features=inner_dim,
            out_features=inner_dim * 2,
            bias=True,
            mesh_device=mesh_device,
        )
        self.proj_out = Linear(
            in_features=inner_dim,
            out_features=patch_size * patch_size * self.out_channels,
            bias=True,
            mesh_device=mesh_device,
        )

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
        for i, block in enumerate(self.transformer_blocks):
            block_cache = block.to_cached_state_dict(path_prefix + f"transformer_blocks.{i}.")
            for key, value in block_cache.items():
                cache_dict[f"transformer_blocks.{i}.{key}"] = value

        # Cache fracture spatial input layer
        fracture_spatial_input_cache = self.fracture_spatial_input.to_cached_state_dict(
            path_prefix + "fracture_spatial_input."
        )
        for key, value in fracture_spatial_input_cache.items():
            cache_dict[f"fracture_spatial_input.{key}"] = value

        # Cache norm out layers
        norm_out_norm_cache = self.norm_out_norm.to_cached_state_dict(path_prefix + "norm_out_norm.")
        norm_out_linear_cache = self.norm_out_linear.to_cached_state_dict(path_prefix + "norm_out_linear.")
        proj_out_cache = self.proj_out.to_cached_state_dict(path_prefix + "proj_out.")

        for key, value in norm_out_norm_cache.items():
            cache_dict[f"norm_out_norm.{key}"] = value
        for key, value in norm_out_linear_cache.items():
            cache_dict[f"norm_out_linear.{key}"] = value
        for key, value in proj_out_cache.items():
            cache_dict[f"proj_out.{key}"] = value

        # Torch fallbacks
        torch.save(self.time_embed.state_dict(), path_prefix + "time_embed.pt")
        torch.save(self.pos_frequencies.data, path_prefix + "pos_frequencies.pt")
        torch.save(self.rope.state_dict(), path_prefix + "rope.pt")
        cache_dict["time_embed"] = path_prefix + "time_embed.pt"
        cache_dict["pos_frequencies"] = path_prefix + "pos_frequencies.pt"
        cache_dict["rope"] = path_prefix + "rope.pt"

        return cache_dict

    def from_cached_state_dict(self, cache_dict):
        self.patch_embed.from_cached_state_dict(substate(cache_dict, "patch_embed"))

        for i, block in enumerate(self.transformer_blocks):
            block.from_cached_state_dict(substate(cache_dict, f"transformer_blocks.{i}"))

        self.fracture_spatial_input.from_cached_state_dict(substate(cache_dict, "fracture_spatial_input"))
        self.norm_out_norm.from_cached_state_dict(substate(cache_dict, "norm_out_norm"))
        self.norm_out_linear.from_cached_state_dict(substate(cache_dict, "norm_out_linear"))
        self.proj_out.from_cached_state_dict(substate(cache_dict, "proj_out"))

        # Torch fallbacks
        self.time_embed.load_state_dict(torch.load(cache_dict["time_embed"]))
        self.pos_frequencies.data = torch.load(cache_dict["pos_frequencies"])
        self.rope.load_state_dict(torch.load(cache_dict["rope"]))

    def load_torch_state_dict(self, state_dict):
        self.patch_embed.load_torch_state_dict(substate(state_dict, "patch_embed"))
        self.time_embed.load_state_dict(substate(state_dict, "time_embed"))
        self.pos_frequencies.data = state_dict["pos_frequencies"]
        for i, block in enumerate(self.transformer_blocks):
            block.load_state_dict(substate(state_dict, f"transformer_blocks.{i}"))
        self.norm_out_norm.load_torch_state_dict(substate(state_dict, "norm_out.norm"))
        self.norm_out_linear.load_torch_state_dict(substate(state_dict, "norm_out.linear"))
        self.proj_out.load_torch_state_dict(substate(state_dict, "proj_out"))

        identity_tensor = torch.eye(self.inner_dim)
        identity_state = {"weight": identity_tensor}
        self.fracture_spatial_input.load_torch_state_dict(identity_state)

    def prepare_rope_features(self, T, H, W):
        pH, pW = H // self.patch_size, W // self.patch_size
        N = T * pH * pW
        rope_cos_NHD, rope_sin_NHD = self.rope(self.pos_frequencies, T, pH, pW, device="cpu", dtype=torch.float32)
        rope_cos_1HND, rope_sin_1HND = stack_cos_sin(
            rope_cos_NHD.unsqueeze(0).permute(0, 2, 1, 3), rope_sin_NHD.unsqueeze(0).permute(0, 2, 1, 3)
        )

        rope_cos_1HND = pad_vision_seq_parallel(
            rope_cos_1HND, num_devices=self.parallel_config.sequence_parallel.factor
        )
        rope_sin_1HND = pad_vision_seq_parallel(
            rope_sin_1HND, num_devices=self.parallel_config.sequence_parallel.factor
        )

        trans_mat = get_rot_transformation_mat()

        tt_rope_cos_1HND = bf16_tensor_2dshard(
            rope_cos_1HND,
            device=self.mesh_device,
            shard_mapping={
                self.parallel_config.sequence_parallel.mesh_axis: 2,
                self.parallel_config.tensor_parallel.mesh_axis: 1,
            },
        )
        tt_rope_sin_1HND = bf16_tensor_2dshard(
            rope_sin_1HND,
            device=self.mesh_device,
            shard_mapping={
                self.parallel_config.sequence_parallel.mesh_axis: 2,
                self.parallel_config.tensor_parallel.mesh_axis: 1,
            },
        )
        tt_trans_mat = bf16_tensor(trans_mat, device=self.mesh_device)

        logger.info(f"TT rope cos shape: {tt_rope_cos_1HND.shape}")
        logger.info(f"TT rope sin shape: {tt_rope_sin_1HND.shape}")
        logger.info(f"TT trans mat shape: {tt_trans_mat.shape}")

        return tt_rope_cos_1HND, tt_rope_sin_1HND, tt_trans_mat

    def prepare_timestep_text_features(self, timestep, text_embeds, encoder_attention_mask):
        """
        Given torch inputs, execute the combined timestep and text embedding.
        Return tensors on device.
        """
        temb, encoder_hidden_states = self.time_embed(
            timestep, text_embeds, encoder_attention_mask, hidden_dtype=torch.float32
        )

        valid_prompt_length = encoder_attention_mask.sum(dim=1).max().int().item()

        logger.info(f"temb shape: {temb.shape}")
        logger.info(f"encoder_hidden_states shape: {encoder_hidden_states.shape}")

        tt_temb_11BD = bf16_tensor(temb.unsqueeze(0).unsqueeze(0), device=self.mesh_device)
        tt_prompt_1BLP = bf16_tensor(encoder_hidden_states.unsqueeze(0), device=self.mesh_device)

        logger.info(f"valid prompt length: {valid_prompt_length}")
        prompt_shape = list(tt_prompt_1BLP.shape)
        prompt_shape[2] = valid_prompt_length
        tt_prompt_1BLP = ttnn.reshape(tt_prompt_1BLP, ttnn.Shape(prompt_shape), tt_prompt_1BLP.padded_shape)

        if valid_prompt_length < text_embeds.shape[-2]:
            logger.warning("Attention mask is not all ones. Truncating prompt")

        logger.info(f"TT temb shape: {tt_temb_11BD.shape}")
        logger.info(f"TT prompt shape: {tt_prompt_1BLP.shape}")
        return tt_temb_11BD, tt_prompt_1BLP

    def preprocess_spatial_input(self, spatial):
        B, C, T, H, W = spatial.shape
        logger.info(f"Preprocessing spatial input with shape {spatial.shape}")
        assert B == 1, "Batch size must be 1"
        pH, pW = H // self.patch_size, W // self.patch_size
        N = T * pH * pW
        spatial = spatial.reshape(B, C, T, pH, self.patch_size, pW, self.patch_size)
        spatial = spatial.permute(0, 2, 3, 5, 4, 6, 1).reshape(1, B, N, self.patch_size * self.patch_size * C)
        logger.info(f"spatial input after patchifying: {spatial.shape}")
        spatial = pad_vision_seq_parallel(spatial, num_devices=self.parallel_config.sequence_parallel.factor)
        logger.info(f"spatial input after padding: {spatial.shape}")
        spatial = bf16_tensor(
            spatial, device=self.mesh_device, mesh_axis=self.parallel_config.sequence_parallel.mesh_axis, shard_dim=-2
        )
        logger.info(f"TT spatial shape: {spatial.shape}")
        return spatial, N

    def postprocess_spatial_output(self, spatial_1BND, T, H, W, N):
        """
        This is the reverse of preprocess_spatial_input
        Input is of shape: 1 B (T pH pW) (p p C)
        returns shape: B C T (pH p) (pW p)
        """
        assert len(spatial_1BND.shape) == 4
        assert spatial_1BND.shape[0] == 1
        B = spatial_1BND.shape[1]
        pH, pW = H // self.patch_size, W // self.patch_size
        logger.info(f"Postprocessing spatial output with shape {spatial_1BND.shape}")

        assert (
            self.mesh_device.shape[self.parallel_config.sequence_parallel.mesh_axis] > 1
        ), "all_gather_async requires at least 1 device"

        # AllGather hanging for large seqlen
        # # Gather sequence-parallel output
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
        spatial_patches = spatial_BND.reshape(B, T, pH, pW, self.patch_size, self.patch_size, self.out_channels)
        logger.info(f"Spatial output after reshaping: {spatial_patches.shape}")
        spatial_BCTHW = spatial_patches.permute(0, 6, 1, 2, 4, 3, 5).reshape(B, self.out_channels, T, H, W)
        logger.info(f"Spatial output after permuting: {spatial_BCTHW.shape}")
        return spatial_BCTHW

    def __call__(self, spatial, prompt, timestep, prompt_attention_mask):
        """
        Inputs are all torch tensors
        Output is torch tensor
        """

        B, C, T, H, W = spatial.shape
        pH, pW = H // self.patch_size, W // self.patch_size
        N = T * pH * pW

        rope_cos_1HND, rope_sin_1HND, trans_mat = self.prepare_rope_features(T, H, W)

        temb_11BD, prompt_1BLP = self.prepare_timestep_text_features(timestep, prompt, prompt_attention_mask)

        spatial_1BNI, N = self.preprocess_spatial_input(spatial)
        spatial_1BND = self.patch_embed(spatial_1BNI)

        for idx, block in enumerate(self.transformer_blocks):
            spatial_1BND, prompt_1BLP = block(
                spatial_1BND, prompt_1BLP, temb_11BD, N, rope_cos_1HND, rope_sin_1HND, trans_mat
            )

        # Modulate the spatial output
        mod = self.norm_out_linear(ttnn.silu(temb_11BD), compute_kernel_config=self.hifi4_compute_kernel_config)
        scale, shift = ttnn.chunk(mod, 2, -1)

        ## SUPER HACKY WORKAROUND
        # Large tensor layernorm is hanging, issue #20789
        # The workaround is to use distributed layernorm by fracturing the input on the TP axis

        spatial_fractured_1BND = self.fracture_spatial_input(
            spatial_1BND, compute_kernel_config=self.hifi4_compute_kernel_config
        )
        spatial_norm_fractured_1BND = self.norm_out_norm(spatial_fractured_1BND)

        spatial_norm_1BND = ttnn.experimental.all_gather_async(
            spatial_norm_fractured_1BND,
            persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                spatial_norm_fractured_1BND.shape, 3, self.parallel_config.tensor_parallel.mesh_axis
            ),
            dim=3,
            multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                self.parallel_config.tensor_parallel.mesh_axis
            ),
            num_links=self.ccl_manager.num_links,
            topology=self.ccl_manager.topology,
            cluster_axis=self.parallel_config.tensor_parallel.mesh_axis,
        )

        spatial_norm_1BND = spatial_norm_1BND * (1 + scale) + shift

        proj_out_1BNI = self.proj_out(spatial_norm_1BND, compute_kernel_config=self.hifi4_compute_kernel_config)

        spatial_out = self.postprocess_spatial_output(proj_out_1BNI, T, H, W, N)

        return spatial_out
