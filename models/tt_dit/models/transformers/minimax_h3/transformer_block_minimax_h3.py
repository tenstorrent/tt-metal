# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import ttnn

from ....layers.feedforward import ParallelFeedForward
from ....layers.linear import ColParallelLinear
from ....layers.module import Module
from ....layers.normalization import DistributedRMSNorm
from ....parallel.config import DiTParallelConfig
from ....parallel.manager import CCLManager
from ....utils.substate import rename_substate
from .agmm_config import agmm_block_size
from .attention_minimax_h3 import MiniMaxH3Attention
from .mmrs_config import has_mmrs_config, register_mmrs_config

# Number of modalities the AdaLN table is indexed by: video (0), text (1), audio (2).
# Mirrors `MINIMAX_H3_MODALITY_NUM` in the reference; padding rows (-1) are clamped to 0.
MODALITY_NUM = 3

# Modulation parameters per block, in the reference's chunk order.
NUM_MODULATION_PARAMS = 6
(
    _SHIFT_MSA,
    _SCALE_MSA,
    _GATE_MSA,
    _SHIFT_MLP,
    _SCALE_MLP,
    _GATE_MLP,
) = range(NUM_MODULATION_PARAMS)


class MiniMaxH3TransformerBlock(Module):
    """One MiniMax-H3 block: pre-norm self-attention and SwiGLU feed-forward, each modulated by
    AdaLN parameters selected *per row* of the packed sequence.

    The per-row modulation is what sets this apart from Wan. A single projection turns the shared
    timestep embedding into a `(num_timesteps * MODALITY_NUM, hidden_size)` table for each of the six
    modulation parameters, and every row of the packed sequence indexes that table by
    `timestep_indices * MODALITY_NUM + token_tags`. One forward therefore serves rows at different
    noise levels and of different modalities. Wan, by contrast, broadcasts one modulation over the
    whole sequence, so it needs no gather at all.

    The projection is shared by `norm1` and `norm2` and by all three modalities, so -- unlike
    `AdaLayerNormZero` -- it cannot be folded into either norm and lives as a block-level module,
    named `adaln_proj` after the checkpoint.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        rotary_dim: int | None = None,
        ffn_dim: int,
        time_embed_dim: int,
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool = False,
        precomputed_adaln: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.ffn_dim = ffn_dim
        self.time_embed_dim = time_embed_dim

        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self.tp_mesh_axis = parallel_config.tensor_parallel.mesh_axis
        self.tp_factor = parallel_config.tensor_parallel.factor
        assert hidden_size % self.tp_factor == 0
        self.hidden_local = hidden_size // self.tp_factor

        fsdp_mesh_axis = parallel_config.sequence_parallel.mesh_axis if is_fsdp else None

        self.norm1 = DistributedRMSNorm(
            embedding_dim=hidden_size,
            norm_eps=norm_eps,
            norm_elementwise_affine=True,
            mesh_axis=self.tp_mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )
        self.attn = MiniMaxH3Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            qk_norm_eps=qk_norm_eps,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            is_fsdp=is_fsdp,
        )
        self.norm2 = DistributedRMSNorm(
            embedding_dim=hidden_size,
            norm_eps=norm_eps,
            norm_elementwise_affine=True,
            mesh_axis=self.tp_mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )
        # The reference's SwiGLU is `up, gate = proj(x).chunk(2); up * silu(gate)`, which is the
        # `gate_is_first=False` layout `ColParallelLinear` already assumes for activation_fn="swiglu".
        self.ff = ParallelFeedForward(
            hidden_size,
            inner_dim=ffn_dim,
            activation_fn="swiglu",
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=self.tp_mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )
        # With precomputed modulation the projection never exists on device: the caller passes the
        # six tables into `forward` instead, and this block's `adaln_proj.*` checkpoint keys are
        # dropped in `_prepare_torch_state`. See `adaln_cache_minimax_h3`.
        self.precomputed_adaln = precomputed_adaln
        self.adaln_proj = (
            None
            if precomputed_adaln
            else ColParallelLinear(
                time_embed_dim,
                NUM_MODULATION_PARAMS * hidden_size * MODALITY_NUM,
                bias=True,
                mesh_device=mesh_device,
                mesh_axis=self.tp_mesh_axis,
                fsdp_mesh_axis=fsdp_mesh_axis,
                ccl_manager=ccl_manager,
            )
        )

        self.mm_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.use_fused_agmm = ccl_manager.topology == ttnn.Topology.Ring and self.tp_factor > 1
        # ff1's block depends on per_core_M (hence M), so it's chosen per-forward; stash (K, N) here.
        # ff1 packs gate and up together for the fused SwiGLU, so its per-device N is 2 * ffn_dim / tp.
        self._ff1_kn = (hidden_size, 2 * ffn_dim // self.tp_factor)

    # ------------------------------------------------------------------ weights

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "ff.net.0.proj", "ff.ff1")
        rename_substate(state, "ff.net.2", "ff.ff2")

        # The reference lays the AdaLN projection's output dim out as
        # [modality][param][hidden], and `hidden` is what TP fractures. Reordering the output dim to
        # [device][modality][param][hidden_local] makes ColParallelLinear's contiguous column split
        # hand each device every (modality, param) pair restricted to its own hidden slice.
        def _reorder_for_tp(t: torch.Tensor) -> torch.Tensor:
            # t is [out, ...] in torch convention; out == 6 * hidden_size * MODALITY_NUM.
            trailing = t.shape[1:]
            t = t.reshape(MODALITY_NUM, NUM_MODULATION_PARAMS, self.tp_factor, self.hidden_local, *trailing)
            t = t.permute(2, 0, 1, 3, *range(4, t.ndim))
            return t.reshape(-1, *trailing)

        weight = state.pop("adaln_proj.linear.weight", None)
        bias = state.pop("adaln_proj.linear.bias", None)
        if self.precomputed_adaln:
            # Dropped: these are the 26 GB the precomputed table exists to keep off the
            # device. The pops above already strip them from the state the loader checks.
            return
        if weight is not None:
            state["adaln_proj.weight"] = _reorder_for_tp(weight)
        if bias is not None:
            state["adaln_proj.bias"] = _reorder_for_tp(bias.unsqueeze(-1)).squeeze(-1)

    # ------------------------------------------------------------------ modulation

    def _modulation_tables(self, temb: ttnn.Tensor) -> list[ttnn.Tensor]:
        """Project `temb` into the six per-(timestep, modality) modulation tables.

        `temb` is [1, 1, num_timesteps, time_embed_dim]. Returns six [num_timesteps * MODALITY_NUM,
        hidden_local] tables, row `t * MODALITY_NUM + modality`, matching the row order that
        `adaln_indices` addresses.

        The SiLU runs at `temb`'s own (float32) precision and only its result is cast
        down to the bfloat16 projection, as the reference is explicit about: every block reads the
        same `temb`, so rounding applied before the activation biases every block's modulation
        identically at every sampling step and accumulates over the denoising trajectory.
        """
        num_timesteps = temb.shape[2]
        # silu at temb's precision, then cast to the projection's dtype -- the reference's
        # `self.linear(silu(temb).to(self.linear.weight.dtype))`. Casting here also keeps the
        # resulting tables bfloat16, which `ttnn.embedding` requires of its weights.
        activated = ttnn.silu(temb)
        if activated.dtype != ttnn.bfloat16:
            activated = ttnn.typecast(activated, ttnn.bfloat16)

        # [1, 1, num_timesteps, MODALITY_NUM * 6 * hidden_local] per device
        projected = self.adaln_proj(activated, compute_kernel_config=self.mm_compute_kernel_config)

        # Fold the modality axis into rows: [num_timesteps, mod, param, h] -> [t * mod, param * h].
        # Done in row-major because the row count changes and num_timesteps is tiny (a handful of
        # rows), so the layout round-trip is negligible.
        rows = num_timesteps * MODALITY_NUM
        projected = ttnn.to_layout(projected, ttnn.ROW_MAJOR_LAYOUT)
        projected = ttnn.reshape(projected, (1, 1, rows, NUM_MODULATION_PARAMS * self.hidden_local))
        projected = ttnn.to_layout(projected, ttnn.TILE_LAYOUT)

        tables = []
        for p in range(NUM_MODULATION_PARAMS):
            table = ttnn.slice(projected, [0, 0, 0, p * self.hidden_local], [1, 1, rows, (p + 1) * self.hidden_local])
            # AdaLN applies the scales as (1 + scale). Adding the 1 to the table costs one elementwise
            # op on `num_timesteps * MODALITY_NUM` rows -- six, typically -- where doing it after the
            # gather would cost one over the whole packed sequence, per scale, per block.
            if p in (_SCALE_MSA, _SCALE_MLP):
                table = ttnn.add(table, 1.0)
            # ttnn.embedding wants a 2D [num_embeddings, embedding_dim] weight.
            table = ttnn.to_layout(table, ttnn.ROW_MAJOR_LAYOUT)
            table = ttnn.reshape(table, (rows, self.hidden_local))
            tables.append(ttnn.to_layout(table, ttnn.TILE_LAYOUT))
        return tables

    def _gather_rows(self, table: ttnn.Tensor, adaln_indices: ttnn.Tensor) -> ttnn.Tensor:
        """Select one table row per row of the local packed sequence -> [1, 1, S_local, hidden_local]."""
        out = ttnn.embedding(adaln_indices, table, layout=ttnn.TILE_LAYOUT)
        return ttnn.unsqueeze(out, 0)

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        spatial_1BND: ttnn.Tensor,
        N: int,
        temb: ttnn.Tensor | None,
        adaln_indices: ttnn.Tensor,
        rope_cos: ttnn.Tensor,
        rope_sin: ttnn.Tensor,
        modulation_tables: list[ttnn.Tensor] | None = None,
    ) -> ttnn.Tensor:
        """
        spatial_1BND: fractured N on SP, fractured hidden_size on TP
        temb: [1, 1, num_timesteps, time_embed_dim], replicated, float32
        adaln_indices: [1, 1, 1, N_local] integer row indices, fractured N on SP
        rope_cos/rope_sin: [1, 1, N_local, rotary_dim], fractured N on SP, replicated on TP
        N: logical (unfractured) packed sequence length

        Returns the block output, fractured N on SP and hidden_size on TP.
        """
        # Precomputed: the six tables come from the host-built cache, addressed by the same absolute
        # `adaln_indices`, so nothing downstream changes. Otherwise project `temb` on device.
        if modulation_tables is not None:
            if len(modulation_tables) != NUM_MODULATION_PARAMS:
                raise ValueError(f"expected {NUM_MODULATION_PARAMS} modulation tables, got {len(modulation_tables)}")
            tables = modulation_tables
        else:
            if self.precomputed_adaln:
                raise ValueError("block was built with precomputed_adaln but forward got no modulation_tables")
            tables = self._modulation_tables(temb)

        # ttnn.embedding takes [batch, seq] indices; uint32 is the dtype it expects.
        indices = ttnn.reshape(adaln_indices, (1, adaln_indices.shape[-1]))
        if indices.dtype != ttnn.uint32:
            indices = ttnn.typecast(indices, ttnn.uint32)

        def modulation(param: int) -> ttnn.Tensor:
            return self._gather_rows(tables[param], indices)

        # 1. Modulated self-attention. The (1 + scale) and shift are handed to the norm as a per-token
        # dynamic weight and bias, so the fused norm op applies the modulation itself rather than the
        # block following it with a separate multiply and add. The gated residual is one addcmul
        # (residual + gate * branch) rather than a multiply and an add.
        residual = spatial_1BND
        normed = self.norm1(
            spatial_1BND,
            dynamic_weight=modulation(_SCALE_MSA),
            dynamic_bias=modulation(_SHIFT_MSA),
        )
        # The gated residual is fused into to_out's matmul epilogue, so `attn` returns
        # `residual + attn_out * gate` directly rather than the block adding it afterwards.
        spatial_1BND = self.attn(
            normed,
            N=N,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            addcmul_residual=residual,
            addcmul_gate=modulation(_GATE_MSA),
        )

        # 2. Modulated feed-forward.
        residual = spatial_1BND
        normed = self.norm2(
            spatial_1BND,
            dynamic_weight=modulation(_SCALE_MLP),
            dynamic_bias=modulation(_SHIFT_MLP),
        )
        # ff1 gathers the TP-fractured input inside its matmul (all_gather_minimal_matmul_async) when
        # parallel_config is passed; ff2 is row-parallel and reduce-scatters back to TP-fractured.
        if not self.use_fused_agmm and self.tp_factor > 1:
            normed = self.ccl_manager.all_gather_persistent_buffer(normed, dim=3, mesh_axis=self.tp_mesh_axis)
        # ff2's reduce-scatter and the gated residual after it fuse into a single
        # minimal_matmul_strided_reduce_scatter_async, computing residual + ff2(...) * gate in one op.
        #
        # Only take it for a shape with a swept blocking. The generic fallback config runs the matmul
        # on 56 of the device's 120 cores at subblock 1x1, making the fused op a measured 45%
        # regression on this stage (1.75 -> 2.55 ms) -- and silently, since an unknown shape does not
        # warn. See mmrs_config for the sweep and the grid/bandwidth tradeoff behind it.
        #
        # Ring only: the op asserts `attributes.topology == Ring` ("MinimalMatmulStridedReduceScatter
        # Async only supports Ring topology"), so a line-cabled mesh has to take the unfused path
        # below. Gated here rather than left to fail, because the assert fires on the first denoise
        # step of the first request -- long after warmup reports the model loaded.
        # ff1's block depends on per_core_M (hence M = the packed sequence length), only known here.
        ff1_block_size = agmm_block_size(*self._ff1_kn, normed.padded_shape[-2])
        ff2_shape = (normed.shape[2], self.ffn_dim // self.tp_factor, self.hidden_size)
        if self.tp_factor > 1 and self.ccl_manager.topology == ttnn.Topology.Ring and has_mmrs_config(*ff2_shape):
            # M is only known here (it tracks the packed sequence length), so the blocking is
            # registered at the point of use rather than at construction. Idempotent and cheap.
            register_mmrs_config(*ff2_shape)
            return self.ff.forward_fused_addcmul(
                normed,
                residual,
                modulation(_GATE_MLP),
                compute_kernel_config=self.mm_compute_kernel_config,
                parallel_config=self.parallel_config if self.use_fused_agmm else None,
                default_block_size=ff1_block_size,
                force_transpose=False,  # M<N -> non-transposed via the op's M>N decision
            )
        ff_out = self.ff(
            normed,
            compute_kernel_config=self.mm_compute_kernel_config,
            parallel_config=self.parallel_config if self.use_fused_agmm else None,
            default_block_size=ff1_block_size,
            force_transpose=False,  # M<N -> non-transposed via the op's M>N decision
        )
        return ttnn.addcmul(residual, ff_out, modulation(_GATE_MLP))
