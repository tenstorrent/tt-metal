# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ...layers.normalization import DistributedLayerNorm, LayerNorm
from ...layers.linear import ColParallelLinear, Linear
from ...layers.feedforward import ParallelFeedForward
from ...layers.embeddings import SD35CombinedTimestepTextProjEmbeddings, PatchEmbed
from ...utils.substate import substate
from .attention_sd35 import SD35JointAttention
from ...layers.module import Module


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/attention_processor.py
class SD35TransformerBlock(Module):
    def __init__(
        self,
        dim,
        num_heads,
        head_dim,
        context_pre_only,
        use_dual_attention=False,
        mesh_device=None,
        ccl_manager=None,
        parallel_config=None,
        padding_config=None,
    ):
        super().__init__()

        assert not use_dual_attention, "Expecting not dual attention"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.context_pre_only = context_pre_only
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        # TODO: Shuffle norm linear weights to match tensor parallelism
        self.norm1_linear = ColParallelLinear(
            dim,
            6 * dim,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
        )
        self.norm1_norm = DistributedLayerNorm(
            dim,
            norm_eps=1e-6,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

        # TODO: Shuffle norm linear weights to match tensor parallelism
        context_norm_dim = 6 * dim if not context_pre_only else 2 * dim
        self.norm1_context_linear = ColParallelLinear(
            dim,
            context_norm_dim,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
        )
        self.norm1_context_norm = DistributedLayerNorm(
            dim,
            norm_eps=1e-6,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

        self.attn = SD35JointAttention(
            query_dim=dim,
            head_dim=head_dim,
            heads=num_heads,
            out_dim=dim,
            bias=True,
            context_pre_only=context_pre_only,
            eps=1e-6,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            padding_config=padding_config,
        )

        self.norm2 = DistributedLayerNorm(
            dim,
            norm_eps=1e-6,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

        self.ff = ParallelFeedForward(
            dim=dim,
            dim_out=dim,
            activation_fn="gelu",
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )

        self.norm2_context = None
        self.ff_context = None

        if not context_pre_only:
            self.norm2_context = DistributedLayerNorm(
                dim,
                norm_eps=1e-6,
                norm_elementwise_affine=False,
                bias=False,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
            )
            self.ff_context = ParallelFeedForward(
                dim=dim,
                dim_out=dim,
                activation_fn="gelu",
                mesh_device=mesh_device,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
                ccl_manager=ccl_manager,
            )

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
        norm2_cache = self.norm2.to_cached_state_dict(path_prefix + "norm2.")

        # Add prefixes for norm layers
        for key, value in norm1_norm_cache.items():
            cache_dict[f"norm1_norm.{key}"] = value
        for key, value in norm1_context_norm_cache.items():
            cache_dict[f"norm1_context_norm.{key}"] = value
        for key, value in norm2_cache.items():
            cache_dict[f"norm2.{key}"] = value

        # Cache attention layer
        attn_cache = self.attn.to_cached_state_dict(path_prefix + "attn.")
        for key, value in attn_cache.items():
            cache_dict[f"attn.{key}"] = value

        # Cache feedforward layer
        ff_cache = self.ff.to_cached_state_dict(path_prefix + "ff.")
        for key, value in ff_cache.items():
            cache_dict[f"ff.{key}"] = value

        # Cache optional context layers
        if not self.context_pre_only:
            norm2_context_cache = self.norm2_context.to_cached_state_dict(path_prefix + "norm2_context.")
            ff_context_cache = self.ff_context.to_cached_state_dict(path_prefix + "ff_context.")

            for key, value in norm2_context_cache.items():
                cache_dict[f"norm2_context.{key}"] = value
            for key, value in ff_context_cache.items():
                cache_dict[f"ff_context.{key}"] = value

        return cache_dict

    def from_cached_state_dict(self, cache_dict):
        self.norm1_linear.from_cached_state_dict(substate(cache_dict, "norm1_linear"))
        self.norm1_context_linear.from_cached_state_dict(substate(cache_dict, "norm1_context_linear"))

        self.norm1_norm.from_cached_state_dict(substate(cache_dict, "norm1_norm"))
        self.norm1_context_norm.from_cached_state_dict(substate(cache_dict, "norm1_context_norm"))
        self.norm2.from_cached_state_dict(substate(cache_dict, "norm2"))

        self.attn.from_cached_state_dict(substate(cache_dict, "attn"))
        self.ff.from_cached_state_dict(substate(cache_dict, "ff"))

        if not self.context_pre_only:
            self.norm2_context.from_cached_state_dict(substate(cache_dict, "norm2_context"))
            self.ff_context.from_cached_state_dict(substate(cache_dict, "ff_context"))

    def load_torch_state_dict(self, state_dict):
        def _shuffle_ada_norm_linear(linear_state):
            # Rearrange QKV projections such column-fracturing shards the heads
            def _shuffle(x, in_dim):
                ndev = self.parallel_config.tensor_parallel.factor
                x = x.T
                cur_in_dim = x.shape[0]  # in_dim for weight, 1 for bias
                expansions = x.shape[-1] // in_dim
                x = x.reshape(-1, expansions, ndev, in_dim // ndev)
                x = x.permute(0, 2, 1, 3)
                x = x.reshape(cur_in_dim, -1)
                assert x.shape[1] == in_dim * expansions
                x = x.T
                return x

            in_dim = linear_state["weight"].shape[1]
            weight = _shuffle(linear_state["weight"], in_dim)
            out_state = {"weight": weight}
            if "bias" in linear_state:
                bias = _shuffle(linear_state["bias"].reshape(-1, 1), in_dim)
                bias = bias.squeeze()
                out_state["bias"] = bias
            return out_state

        def rename_ff_state(state):
            out_state = {
                f"{replacement}{k[len(prefix):]}": v
                for k, v in state.items()
                for prefix, replacement in [("net.0.proj", "ff1"), ("net.2", "ff2")]
                if prefix in k
            }
            return out_state

        self.norm1_linear.load_torch_state_dict(_shuffle_ada_norm_linear(substate(state_dict, "norm1.linear")))
        self.norm1_norm.load_torch_state_dict(substate(state_dict, "norm1.norm"))
        self.norm1_context_linear.load_torch_state_dict(
            _shuffle_ada_norm_linear(substate(state_dict, "norm1_context.linear"))
        )
        self.norm1_context_norm.load_torch_state_dict(substate(state_dict, "norm1_context.norm"))
        self.attn.load_state_dict(substate(state_dict, "attn"))
        self.norm2.load_torch_state_dict(substate(state_dict, "norm2"))
        self.ff.load_torch_state_dict(rename_ff_state(substate(state_dict, "ff")))
        if not self.context_pre_only:
            self.norm2_context.load_torch_state_dict(substate(state_dict, "norm2_context"))
            self.ff_context.load_torch_state_dict(rename_ff_state(substate(state_dict, "ff_context")))

    def forward(self, spatial_1BND, prompt_1BLD, time_embed_11BE, N, L):
        """
        spatial_1BND: fractured N on SP, fractured D on TP
        prompt_1BLD: replicated on SP, fractured D on TP
        time_embed_11BE: replicated
        """

        time_embed_11BE = ttnn.silu(time_embed_11BE, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        spatial_time_11BF = self.norm1_linear(time_embed_11BE)
        prompt_time_11BE = self.norm1_context_linear(time_embed_11BE)

        (
            spatial_shift_attn,
            spatial_scale_attn,
            spatial_gate_attn,
            spatial_shift_ff,
            spatial_scale_ff,
            spatial_gate_ff,
        ) = chunk_time(spatial_time_11BF, 6)

        spatial_normed_1BND = self.norm1_norm(
            spatial_1BND, dynamic_weight=(1 + spatial_scale_attn), dynamic_bias=spatial_shift_attn
        )

        if self.context_pre_only:
            prompt_scale_attn, prompt_shift_attn = chunk_time(prompt_time_11BE, 2)
            prompt_gate_attn = None
            prompt_shift_ff = None
            prompt_scale_ff = None
            prompt_gate_ff = None
        else:
            (
                prompt_shift_attn,
                prompt_scale_attn,
                prompt_gate_attn,
                prompt_shift_ff,
                prompt_scale_ff,
                prompt_gate_ff,
            ) = chunk_time(prompt_time_11BE, 6)

        prompt_normed_1BLD = self.norm1_context_norm(
            prompt_1BLD, dynamic_weight=(1 + prompt_scale_attn), dynamic_bias=prompt_shift_attn
        )

        if self.parallel_config.tensor_parallel.factor > 1:
            # Gather spatial, prompt before attention
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
                **self.ccl_manager.get_ag_hyperparams(spatial_normed_1BND.shape),
            )
            prompt_normed_1BLD = ttnn.experimental.all_gather_async(
                prompt_normed_1BLD,
                persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                    prompt_normed_1BLD.shape, 3, self.parallel_config.tensor_parallel.mesh_axis
                ),
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                    self.parallel_config.tensor_parallel.mesh_axis
                ),
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=self.parallel_config.tensor_parallel.mesh_axis,
                **self.ccl_manager.get_ag_hyperparams(prompt_normed_1BLD.shape),
            )

        spatial_attn_1BLD, prompt_attn_1BLD = self.attn(spatial_normed_1BND, prompt_normed_1BLD, N)
        spatial_attn_1BLD = spatial_attn_1BLD * spatial_gate_attn
        prompt_attn_1BLD = prompt_attn_1BLD * prompt_gate_attn if prompt_gate_attn is not None else None

        # residual
        spatial_1BND = spatial_1BND + spatial_attn_1BLD

        spatial_normed_1BND = self.norm2(
            spatial_1BND, dynamic_weight=(1 + spatial_scale_ff), dynamic_bias=spatial_shift_ff
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
                **self.ccl_manager.get_ag_hyperparams(spatial_normed_1BND.shape),
            )

        spatial_ff_1BND = self.ff(spatial_normed_1BND)
        spatial_ff_1BND = spatial_ff_1BND * spatial_gate_ff

        spatial_1BND += spatial_ff_1BND

        if self.context_pre_only:
            return spatial_1BND, None

        prompt_1BLD += prompt_attn_1BLD

        prompt_normed_1BLD = self.norm2_context(
            prompt_1BLD, dynamic_weight=(1 + prompt_scale_ff), dynamic_bias=prompt_shift_ff
        )

        if self.parallel_config.tensor_parallel.factor > 1:
            prompt_normed_1BLD = ttnn.experimental.all_gather_async(
                prompt_normed_1BLD,
                persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                    prompt_normed_1BLD.shape, 3, self.parallel_config.tensor_parallel.mesh_axis
                ),
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                    self.parallel_config.tensor_parallel.mesh_axis
                ),
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=self.parallel_config.tensor_parallel.mesh_axis,
                **self.ccl_manager.get_ag_hyperparams(prompt_normed_1BLD.shape),
            )

        prompt_ff_1BLD = self.ff_context(prompt_normed_1BLD)
        prompt_ff_1BLD = prompt_ff_1BLD * prompt_gate_ff

        prompt_1BLD += prompt_ff_1BLD

        return spatial_1BND, prompt_1BLD


def chunk_time(t: ttnn.Tensor, count: int) -> list[ttnn.Tensor]:
    size = t.shape[-1] // count
    return [t[:, :, :, i * size : (i + 1) * size] for i in range(count)]


class SD35Transformer2DModel(Module):
    def __init__(
        self,
        sample_size=128,
        patch_size=2,
        in_channels=16,
        num_layers=18,
        attention_head_dim=64,
        num_attention_heads=18,
        joint_attention_dim=4096,
        caption_projection_dim=1152,
        pooled_projection_dim=2048,
        out_channels=16,
        pos_embed_max_size=96,
        dual_attention_layers=(),
        mesh_device=None,
        ccl_manager=None,
        parallel_config=None,
        padding_config=None,
    ):
        super().__init__()

        self.sample_size = sample_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.joint_attention_dim = joint_attention_dim
        self.caption_projection_dim = caption_projection_dim
        self.pooled_projection_dim = pooled_projection_dim
        self.out_channels = out_channels
        self.pos_embed_max_size = pos_embed_max_size
        self.dual_attention_layers = dual_attention_layers
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self.out_channels = out_channels if out_channels is not None else in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        # Components
        self.pos_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,
            mesh_device=mesh_device,
            tp_mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            sp_mesh_axis=parallel_config.sequence_parallel.mesh_axis,
        )

        self.time_text_embed = SD35CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim,
            pooled_projection_dim=pooled_projection_dim,
            mesh_device=mesh_device,
        )

        self.context_embedder = ColParallelLinear(
            joint_attention_dim,
            caption_projection_dim,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
        )

        # Transformer blocks
        self.transformer_blocks = []
        for i in range(num_layers):
            block = SD35TransformerBlock(
                dim=self.inner_dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                context_pre_only=i == num_layers - 1,
                use_dual_attention=i in dual_attention_layers,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                padding_config=padding_config,
            )
            self.transformer_blocks.append(block)

        # Output normalization and projection
        self.norm_out_linear = Linear(self.inner_dim, 2 * self.inner_dim, mesh_device=mesh_device)
        self.norm_out_norm = LayerNorm(
            self.inner_dim, norm_elementwise_affine=False, norm_eps=1e-6, mesh_device=mesh_device
        )
        self.proj_out = Linear(self.inner_dim, patch_size * patch_size * self.out_channels, mesh_device=mesh_device)

        self.hifi_compute_kernel_config = ttnn.init_device_compute_kernel_config(
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

        # Cache embeddings
        pos_embed_cache = self.pos_embed.to_cached_state_dict(path_prefix + "pos_embed.")
        time_text_embed_cache = self.time_text_embed.to_cached_state_dict(path_prefix + "time_text_embed.")
        context_embedder_cache = self.context_embedder.to_cached_state_dict(path_prefix + "context_embedder.")

        for key, value in pos_embed_cache.items():
            cache_dict[f"pos_embed.{key}"] = value
        for key, value in time_text_embed_cache.items():
            cache_dict[f"time_text_embed.{key}"] = value
        for key, value in context_embedder_cache.items():
            cache_dict[f"context_embedder.{key}"] = value

        # Cache transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            block_cache = block.to_cached_state_dict(path_prefix + f"transformer_blocks.{i}.")
            for key, value in block_cache.items():
                cache_dict[f"transformer_blocks.{i}.{key}"] = value

        # Cache output layers
        norm_out_linear_cache = self.norm_out_linear.to_cached_state_dict(path_prefix + "norm_out_linear.")
        norm_out_norm_cache = self.norm_out_norm.to_cached_state_dict(path_prefix + "norm_out_norm.")
        proj_out_cache = self.proj_out.to_cached_state_dict(path_prefix + "proj_out.")

        for key, value in norm_out_linear_cache.items():
            cache_dict[f"norm_out_linear.{key}"] = value
        for key, value in norm_out_norm_cache.items():
            cache_dict[f"norm_out_norm.{key}"] = value
        for key, value in proj_out_cache.items():
            cache_dict[f"proj_out.{key}"] = value

        return cache_dict

    def from_cached_state_dict(self, cache_dict):
        self.pos_embed.from_cached_state_dict(substate(cache_dict, "pos_embed"))
        self.time_text_embed.from_cached_state_dict(substate(cache_dict, "time_text_embed"))
        self.context_embedder.from_cached_state_dict(substate(cache_dict, "context_embedder"))

        for i, block in enumerate(self.transformer_blocks):
            block.from_cached_state_dict(substate(cache_dict, f"transformer_blocks.{i}"))

        self.norm_out_linear.from_cached_state_dict(substate(cache_dict, "norm_out_linear"))
        self.norm_out_norm.from_cached_state_dict(substate(cache_dict, "norm_out_norm"))
        self.proj_out.from_cached_state_dict(substate(cache_dict, "proj_out"))

    def load_torch_state_dict(self, state_dict):
        self.pos_embed.load_torch_state_dict(substate(state_dict, "pos_embed"))
        self.time_text_embed.load_torch_state_dict(substate(state_dict, "time_text_embed"))
        self.context_embedder.load_torch_state_dict(substate(state_dict, "context_embedder"))

        for i, block in enumerate(self.transformer_blocks):
            block.load_torch_state_dict(substate(state_dict, f"transformer_blocks.{i}"))

        self.norm_out_linear.load_torch_state_dict(substate(state_dict, "norm_out.linear"))
        self.norm_out_norm.load_torch_state_dict(substate(state_dict, "norm_out.norm"))
        self.proj_out.load_torch_state_dict(substate(state_dict, "proj_out"))

    def forward(self, spatial, prompt_embed, pooled_projections, timestep, N, L):
        """
        Args:
            spatial: Input spatial tensor (latents) - fractured dim 2 along sp_axis
            prompt_embed: Text prompt embeddings - replicated
            pooled_projections: Pooled text projections - replicated
            timestep: Timestep tensor - replicated
        """
        spatial = self.pos_embed(spatial, already_unfolded=True)

        time_embed = self.time_text_embed(timestep, pooled_projections)
        prompt_embed = self.context_embedder(prompt_embed)

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            spatial, prompt_embed = block(spatial, prompt_embed, time_embed, N, L)
        # Final normalization and projection
        spatial_time = self.norm_out_linear(ttnn.silu(time_embed, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        scale, shift = chunk_time(spatial_time, 2)

        # Gather spatial such that it is fully replicated for final norm and projection
        if self.parallel_config.sequence_parallel.factor > 1:
            spatial = ttnn.experimental.all_gather_async(
                spatial,
                persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                    spatial.shape, 2, self.parallel_config.sequence_parallel.mesh_axis
                ),
                dim=2,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                    self.parallel_config.sequence_parallel.mesh_axis
                ),
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=self.parallel_config.sequence_parallel.mesh_axis,
                # chunks_per_sync=16,
                # num_workers_per_link=3,
                # num_buffers_per_channel=2,
            )
        if self.parallel_config.tensor_parallel.factor > 1:
            spatial = ttnn.experimental.all_gather_async(
                spatial,
                persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                    spatial.shape, 3, self.parallel_config.tensor_parallel.mesh_axis
                ),
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                    self.parallel_config.tensor_parallel.mesh_axis
                ),
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=self.parallel_config.tensor_parallel.mesh_axis,
                # chunks_per_sync=10,
                # num_workers_per_link=2,
                # num_buffers_per_channel=2,
            )

        spatial = self.norm_out_norm(spatial) * (1 + scale) + shift

        spatial_out = self.proj_out(spatial, compute_kernel_config=self.hifi_compute_kernel_config)

        # NOTE: While we should be able to gather on sequence after norm and proj,
        # it leads to terrible outputs for 2x2sp1tp0. Need to debug.
        # if self.parallel_config.sequence_parallel.factor > 1:
        #     spatial_out = ttnn.experimental.all_gather_async(
        #         spatial_out,
        #         persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
        #             spatial_out.shape, 2, self.parallel_config.sequence_parallel.mesh_axis
        #         ),
        #         dim=2,
        #         multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(),
        #         num_links=self.ccl_manager.num_links,
        #         topology=self.ccl_manager.topology,
        #         cluster_axis=self.parallel_config.sequence_parallel.mesh_axis,
        #         # chunks_per_sync=16,
        #         # num_workers_per_link=3,
        #         # num_buffers_per_channel=2,
        #     )

        return spatial_out

    def patchify(self, latents: torch.Tensor) -> torch.Tensor:
        # N, H, W, C -> 1, N, (H / P) * (W / P), P * P * C
        batch_size, height, width, channels = latents.shape
        patch = self.patch_size

        if height % patch != 0 or width % patch != 0:
            msg = f"height ({height}) and width ({width}) must be divisible by patch_size ({patch})"
            raise ValueError(msg)

        latents = latents.reshape([batch_size, height // patch, patch, width // patch, patch, channels])
        return latents.transpose(2, 3).flatten(3, 5).flatten(1, 2).unsqueeze(0)

    def unpatchify(self, spatial: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
        # 1, N, (H / P) * (W / P), P * P * C -> N, H, W, C
        one, batch_size, _, _ = spatial.shape
        assert one == 1
        patch = self.patch_size

        if height % patch != 0 or width % patch != 0:
            msg = f"height ({height}) and width ({width}) must be divisible by patch_size ({patch})"
            raise ValueError(msg)

        spatial = spatial.reshape([batch_size, height // patch, width // patch, patch, patch, -1])
        return spatial.transpose(2, 3).flatten(3, 4).flatten(1, 2)
