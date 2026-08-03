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
from .attention_minimax_h3 import MiniMaxH3Attention

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
        ffn_dim: int,
        time_embed_dim: int,
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool = False,
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
        self.adaln_proj = ColParallelLinear(
            time_embed_dim,
            NUM_MODULATION_PARAMS * hidden_size * MODALITY_NUM,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=self.tp_mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )

        self.mm_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

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

        The SiLU deliberately runs at `temb`'s own (float32) precision and only its result is cast
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
        temb: ttnn.Tensor,
        adaln_indices: ttnn.Tensor,
        rope_cos: ttnn.Tensor,
        rope_sin: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        spatial_1BND: fractured N on SP, fractured hidden_size on TP
        temb: [1, 1, num_timesteps, time_embed_dim], replicated, float32
        adaln_indices: [1, 1, 1, N_local] integer row indices, fractured N on SP
        rope_cos/rope_sin: [1, 1, N_local, rotary_dim], fractured N on SP, replicated on TP
        N: logical (unfractured) packed sequence length

        Returns the block output, fractured N on SP and hidden_size on TP.
        """
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
        attn_out = self.attn(normed, N=N, rope_cos=rope_cos, rope_sin=rope_sin)
        spatial_1BND = ttnn.addcmul(residual, attn_out, modulation(_GATE_MSA))

        # 2. Modulated feed-forward.
        residual = spatial_1BND
        normed = self.norm2(
            spatial_1BND,
            dynamic_weight=modulation(_SCALE_MLP),
            dynamic_bias=modulation(_SHIFT_MLP),
        )
        # ParallelFeedForward expects a replicated input; its RowParallelLinear reduce-scatters the
        # result back to TP-fractured. As in the attention module, bringup takes the unfused
        # all-gather path and folding it into the matmul is the performance follow-up.
        if self.tp_factor > 1:
            normed = self.ccl_manager.all_gather_persistent_buffer(normed, dim=3, mesh_axis=self.tp_mesh_axis)
        ff_out = self.ff(normed, compute_kernel_config=self.mm_compute_kernel_config)
        return ttnn.addcmul(residual, ff_out, modulation(_GATE_MLP))
