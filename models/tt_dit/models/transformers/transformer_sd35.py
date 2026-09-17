# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel as TorchSD3Transformer2DModel

import ttnn

from ...layers.embeddings import PatchEmbed, SD35CombinedTimestepTextProjEmbeddings
from ...layers.feedforward import ParallelFeedForward
from ...layers.linear import ColParallelLinear, Linear, prepare_chunked_linear_output
from ...layers.module import Module, ModuleList
from ...layers.normalization import DistributedLayerNorm, LayerNorm
from ...utils import cache
from ...utils.matmul import FusedMMRSConfig, register_fused_mmrs_configs
from ...utils.padding import PaddingConfig
from ...utils.substate import rename_substate
from .attention_sd35 import SD35JointAttention

if TYPE_CHECKING:
    from ...parallel.config import DiTParallelConfig
    from ...parallel.manager import CCLManager

_registered_mmrs_grids: set[tuple[int, int]] = set()


def _register_ff2_mmrs_config(mesh_device: ttnn.MeshDevice) -> None:
    """Register a fused MM+RS blocking for this device's actual compute grid.

    get_fused_mmrs_config's built-in fallbacks (swept table entries and the v2.3 rule engine) all
    assume a 12-wide Blackhole grid; this 4-chip QuietBox's grid is 11x10, which is both narrower
    than 12 and, being 11 (prime), can't split evenly the way 12 does. Constrain the matmul grid to
    11x8 (fits inside 11x10, leaving 2 rows for the reduce-scatter) with a simple blocking verified
    correct (PCC ~1.0 against a torch reference) rather than falling through to the unfitted 12x8
    default, which errors outright on this grid.
    """
    device_grid = mesh_device.compute_with_storage_grid_size()
    key = (device_grid.x, device_grid.y)
    if key in _registered_mmrs_grids:
        return
    if key != (11, 10):
        return
    register_fused_mmrs_configs(
        {
            device_grid: {
                # ff2 (main spatial FFN), batch flattened into the row dim: the fused op requires
                # batch size 1 (padded_shape[0]==1 and [1]==1), but this model batches CFG
                # cond+uncond as batch=2, so the caller flattens (1,2,4096,K) -> (1,1,8192,K)
                # before calling and reshapes back after (see forward()).
                # M=8192 (2*4096 spatial tokens), K=inner_dim/tp=9728/4=2432, N=dim_out=2432.
                (8192, 2432, 2432): FusedMMRSConfig(ttnn.CoreCoord(11, 8), 8, 4, 4, 2, 1, None, 1, mm_window_blocks=2),
            }
        }
    )
    _registered_mmrs_grids.add(key)


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
            activation_fn="gelu_tanh",
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )
        self._use_fused_ff_addcmul = parallel_config.tensor_parallel.factor > 1
        if self._use_fused_ff_addcmul:
            _register_ff2_mmrs_config(mesh_device)

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
                activation_fn="gelu_tanh",
                mesh_device=mesh_device,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
                ccl_manager=ccl_manager,
            )

        device_grid = self.mesh_device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(x=device_grid.x, y=device_grid.y)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "norm1.linear", "norm1_linear")
        rename_substate(state, "norm1_context.linear", "norm1_context_linear")
        rename_substate(state, "ff.net.0.proj", "ff.ff1")
        rename_substate(state, "ff.net.2", "ff.ff2")
        rename_substate(state, "ff_context.net.0.proj", "ff_context.ff1")
        rename_substate(state, "ff_context.net.2", "ff_context.ff2")

        prepare_chunked_linear_output(
            state,
            prefix="norm1_linear",
            device_count=self.parallel_config.tensor_parallel.factor,
            chunks=6,
        )
        prepare_chunked_linear_output(
            state,
            prefix="norm1_context_linear",
            device_count=self.parallel_config.tensor_parallel.factor,
            chunks=2 if self.context_pre_only else 6,
        )

    def forward(self, spatial_1BND, prompt_1BLD, time_embed_11BE, N):
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

        # Fused MM+RS for ff2's row-parallel matmul + its reduce-scatter (one op instead of two).
        # The gate-multiply + residual-add are done as separate, un-flattened ops afterward rather
        # than fused into the same op via addcmul_a/addcmul_b.
        #
        # Bug found by manual inspection (garbled/noise output on every 1x4 run, root-caused via
        # bisection): the previous version passed the real residual/gate as addcmul_a/addcmul_b
        # into forward_fused_addcmul directly. Confirmed by isolation test that the fused matmul+RS
        # itself (with this same batch-flatten) is correct -- feeding it neutral addcmul_a=0,
        # addcmul_b=1 and doing gate/residual separately produced a correct image; reintroducing the
        # real addcmul_a/addcmul_b (even after fixing the gate's flatten to avoid ttnn.repeat on a
        # degenerate broadcast dim, via a per-batch slice+concat) still produced noise. That points
        # at the fused kernel's own addcmul handling for this shape/mesh, not the Python-side
        # reshape/gate construction -- same class of issue as the fused-AGMM kernel bug already
        # flagged as blocked on this 1-row mesh. Keeping the fused matmul+RS (the expensive part)
        # and dropping only the addcmul fusion keeps most of the win: 7.03s vs 6.71-6.76s fully
        # fused (which was fast but wrong) and 7.16s with the whole fused path disabled.
        #
        # The fused kernel requires batch size 1 (asserts padded_shape[0]==1 and [1]==1), but this
        # model batches CFG cond+uncond as batch=2, so we flatten (1, B, N, .) -> (1, 1, B*N, .)
        # before the call and reshape back after. Only registered for the exact (M, K, N) this
        # model's spatial FFN hits on this device's compute grid (see _register_ff2_mmrs_config);
        # any other shape falls back to the plain path.
        b_dim, n_tok, k_dim = spatial_normed_1BND.shape[1], spatial_normed_1BND.shape[2], spatial_normed_1BND.shape[3]
        d_local = spatial_1BND.shape[3]
        if self._use_fused_ff_addcmul and b_dim * n_tok == 8192:
            x_flat = ttnn.reshape(spatial_normed_1BND, (1, 1, b_dim * n_tok, k_dim))
            residual_flat = ttnn.reshape(spatial_1BND, (1, 1, b_dim * n_tok, d_local))
            zero_flat = residual_flat * 0.0
            one_flat = zero_flat + 1.0
            ff2_flat = self.ff.forward_fused_addcmul(x_flat, addcmul_a=zero_flat, addcmul_b=one_flat, scalar=1.0)
            ff2_1BND = ttnn.reshape(ff2_flat, (1, b_dim, n_tok, d_local))
            spatial_1BND = spatial_1BND + ff2_1BND * spatial_gate_ff
        else:
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
        self.transformer_blocks = ModuleList()
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

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "norm_out.linear", "norm_out_linear")

    def forward(self, spatial, prompt_embed, pooled_projections, timestep, N):
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
            spatial, prompt_embed = block(spatial, prompt_embed, time_embed, N)
        # Final normalization and projection
        spatial_time = self.norm_out_linear(ttnn.silu(time_embed, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        scale, shift = chunk_time(spatial_time, 2)

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


class SD35Checkpoint:
    """An SD3.5 checkpoint: fetches weights and builds loaded transformers."""

    def __init__(self, name: str) -> None:
        self._name = name
        torch_transformer = TorchSD3Transformer2DModel.from_pretrained(
            name,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        torch_transformer.eval()
        assert isinstance(torch_transformer, TorchSD3Transformer2DModel)
        self._config = torch_transformer.config
        self._state_dict = torch_transformer.state_dict()

        self.num_channels_latents: int = self._config.in_channels
        self.joint_attention_dim: int = self._config.joint_attention_dim
        self.patch_size: int = self._config.patch_size

    def build(
        self,
        *,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
    ) -> SD35Transformer2DModel:
        """Construct an ``SD35Transformer2DModel`` for this checkpoint and load its weights."""
        device = ccl_manager.mesh_device
        c = self._config

        if c.num_attention_heads % parallel_config.tensor_parallel.factor != 0:
            padding_config = PaddingConfig.from_tensor_parallel_factor(
                c.num_attention_heads,
                c.attention_head_dim,
                parallel_config.tensor_parallel.factor,
            )
        else:
            padding_config = None

        model = SD35Transformer2DModel(
            sample_size=c.sample_size,
            patch_size=c.patch_size,
            in_channels=c.in_channels,
            num_layers=c.num_layers,
            attention_head_dim=c.attention_head_dim,
            num_attention_heads=c.num_attention_heads,
            joint_attention_dim=c.joint_attention_dim,
            caption_projection_dim=c.caption_projection_dim,
            pooled_projection_dim=c.pooled_projection_dim,
            out_channels=c.out_channels,
            pos_embed_max_size=c.pos_embed_max_size,
            dual_attention_layers=c.dual_attention_layers,
            mesh_device=device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            padding_config=padding_config,
        )
        cache.load_model(
            tt_model=model,
            get_torch_state_dict=lambda: self._state_dict,
            model_name=os.path.basename(self._name),
            subfolder="transformer",
            parallel_config=parallel_config,
            mesh_shape=tuple(device.shape),
            mesh_device=device,
        )
        return model
