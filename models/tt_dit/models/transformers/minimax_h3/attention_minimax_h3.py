# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import torch

import ttnn
from models.common.utility_functions import is_blackhole

from ....layers.linear import ColParallelLinear
from ....layers.module import Module
from ....layers.normalization import DistributedRMSNorm
from ....parallel.config import DiTParallelConfig
from ....parallel.manager import CCLManager
from ....utils import sdpa_recipe
from ....utils.mochi import get_rot_transformation_mat
from ....utils.substate import pop_substate, rename_substate
from ....utils.tensor import bf16_tensor
from .agmm_config import agmm_block_size


def rope_channel_permutation(head_dim: int, rotary_dim: int) -> torch.Tensor:
    """Reorder a head's channels from MiniMax-H3's half-split RoPE layout to the interleaved one.

    The fused RoPE inside `dit_fused_distributed_rmsnorm` rotates by multiplying every 32-column tile
    by one 32x32 matrix, which pairs *adjacent* channels: `out[2i] = -in[2i+1]`, `out[2i+1] = in[2i]`.
    MiniMax-H3's reference instead pairs `i` with `i + rotary_dim/2`. The two are the same operation
    under this permutation of the rotary channels -- `out[2i] = in[i]`, `out[2i+1] = in[i + rot/2]` --
    with the `head_dim - rotary_dim` pass-through channels left where they are.

    Applied identically to the Q and K projection output channels, the QK-norm affine weight and the
    cos/sin tables. Attention sees Q and K only through `q . k`, which any *shared* permutation of the
    channel axis leaves unchanged, and V and `to_out` are untouched -- so the relayout is numerically
    neutral, not an approximation. Same trick as `transformer_ideogram4.rope_halfsplit_to_interleaved_perm`,
    extended to a rotary_dim narrower than head_dim.
    """
    half = rotary_dim // 2
    rotary = torch.stack([torch.arange(half), torch.arange(half) + half], dim=1).flatten()
    return torch.cat([rotary, torch.arange(rotary_dim, head_dim)])


def prepare_rope_tables(cos: torch.Tensor, sin: torch.Tensor, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn the reference's `[.., rotary_dim]` cos/sin into what the fused RoPE consumes.

    Permutes the rotary channels into the interleaved layout (see `rope_channel_permutation`) and pads
    out to `head_dim` with cos=1 / sin=0. Those pad channels are the ones MiniMax-H3 passes through
    unrotated: because the fused rotate only ever mixes channels *within* a 32-column tile, and the
    pass-through channels occupy whole tiles of their own, `sin=0` there makes the rotate an exact
    identity. That is what lets a partial-head-dim RoPE run on an op that has no notion of one.
    """
    rotary_dim = cos.shape[-1]
    perm = rope_channel_permutation(rotary_dim, rotary_dim)  # permute the rotary block only
    cos, sin = cos[..., perm], sin[..., perm]
    pad = head_dim - rotary_dim
    if pad:
        ones = torch.ones(*cos.shape[:-1], pad, dtype=cos.dtype)
        zeros = torch.zeros(*sin.shape[:-1], pad, dtype=sin.dtype)
        cos, sin = torch.cat([cos, ones], dim=-1), torch.cat([sin, zeros], dim=-1)
    return cos, sin


class MiniMaxH3Attention(Module):
    """Full self-attention over one packed sequence. MiniMax-H3 has no cross-attention.

    Two things differ from `WanAttention` and drive the shape of this module:

    * The attention inner dim (`num_heads * head_dim` = 7168) is *larger* than the residual stream
      (`hidden_size` = 5376), so `to_q/k/v` widen 5376 -> 7168 and `to_out` narrows 7168 -> 5376.
      Nothing here may assume `inner_dim == hidden_size`. Every projection is bias-free.
    * The query/key norms are RMSNorms over `head_dim` (128), not over the TP-sharded residual
      stream. They use `DistributedRMSNorm` in `per_head_norm` mode, which reduces over each head's
      head_dim *locally* -- no all-gather, since a head's channels all live on one device -- and in
      the same op splits the heads and applies RoPE. So one fused op replaces the norm, the head
      split and the whole rotary sequence. See `rope_channel_permutation` for how MiniMax-H3's
      partial, half-split rotary is made to fit an op that implements a full-width interleaved one.
    """

    # Named SDPA recipe of every SDPA call in this module on Blackhole. DiT models default to
    # FAST (legacy streaming numerics with the approximate exponential; user decision
    # 2026-09-25): at the models' shapes it is as accurate as the legacy HiFi2 / BF16-dest /
    # exact-exp setup within a few percent and at least as fast. Pass sdpa_precision to opt
    # up (e.g. BALANCED). See tests/ttnn/unit_tests/operations/sdpa/test_sdpa_dit_recipe_parity.py.
    sdpa_precision_default = ttnn.SDPAPrecision.FAST

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        rotary_dim: int | None = None,
        qk_norm_eps: float = 1e-5,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool = False,
        is_sequence_parallel: bool = True,
        sdpa_precision: ttnn.SDPAPrecision | None = None,
        sdpa_kv_dtype: ttnn.DataType | None = None,
    ) -> None:
        super().__init__()

        # Named SDPA recipe (sdpa_precision=None: sdpa_precision_default; see
        # models/tt_dit/utils/sdpa_recipe.py); legacy off Blackhole only. Validated before anything
        # touches the device.
        blackhole = is_blackhole()
        self.sdpa_precision = sdpa_recipe.resolve_precision(
            sdpa_precision, self.sdpa_precision_default, blackhole=blackhole, model="MiniMaxH3Attention"
        )
        self.sdpa_kv_dtype = sdpa_recipe.validate_recipe_args(
            self.sdpa_precision,
            sdpa_kv_dtype,
            head_dim=head_dim,
            model="MiniMaxH3Attention",
            is_blackhole=blackhole,
        )

        # is_sequence_parallel=False means the sequence is *replicated* on the SP axis rather than
        # fractured across it, so attention runs locally with plain SDPA and no ring all-gather. The
        # token refiner uses that: its text stream is short and every SP device holds all of it.
        self.is_sequence_parallel = is_sequence_parallel

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        # Channels [rotary_dim, head_dim) pass through the rotary embedding unrotated. None means
        # the whole head rotates, which is what the token refiner (no RoPE at all) leaves unused.
        self.rotary_dim = head_dim if rotary_dim is None else rotary_dim
        self.qk_norm_eps = qk_norm_eps

        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        tp_factor = parallel_config.tensor_parallel.factor
        assert num_heads % tp_factor == 0, f"{num_heads} heads must divide across TP={tp_factor}"
        self.n_local_heads = num_heads // tp_factor
        self.tp_mesh_axis = parallel_config.tensor_parallel.mesh_axis
        self.sp_mesh_axis = parallel_config.sequence_parallel.mesh_axis
        # Fractured sequence means attention has to gather K/V around the ring.
        self.use_ring = is_sequence_parallel and parallel_config.sequence_parallel.factor > 1

        fsdp_mesh_axis = self.sp_mesh_axis if is_fsdp else None

        # Fused QKV: one matmul, output split into three. The state dict is rearranged in
        # `_prepare_torch_state` so that column-parallel fracturing hands each device the same
        # 14 heads of q, k and v.
        self.to_qkv = ColParallelLinear(
            hidden_size,
            3 * self.inner_dim,
            chunks=3,
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=self.tp_mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )
        self.to_out = ColParallelLinear(
            self.inner_dim,
            hidden_size,
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=self.tp_mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )

        # QK-norm + head split + RoPE in one fused op. embedding_dim is the *inner* dim so the
        # per-device weight slice covers n_local_heads * head_dim; `per_head_norm=True` at the call
        # site makes the reduction per head and device-local.
        qk_norm_kwargs = dict(
            embedding_dim=self.inner_dim,
            norm_eps=qk_norm_eps,
            norm_elementwise_affine=True,
            mesh_axis=self.tp_mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )
        self.norm_q = DistributedRMSNorm(**qk_norm_kwargs)
        self.norm_k = DistributedRMSNorm(**qk_norm_kwargs)
        self.rope_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)

        # all_gather_minimal_matmul_async folds the TP all-gather into the matmul. Ring only: on a
        # line topology WanAttention measured the unfused path faster, so match that condition.
        self.use_fused_agmm = ccl_manager.topology == ttnn.Topology.Ring and tp_factor > 1

        # Ring SDPA reuses the joint-attention entry point with empty joint inputs, as WanAttention does.
        self.dummy_joint_input = bf16_tensor(torch.zeros((1, self.n_local_heads, 0, head_dim)), device=mesh_device)

        full_grid = mesh_device.compute_with_storage_grid_size()
        self.full_grid = full_grid
        self.sdpa_worker_grid = (full_grid.x - 1, full_grid.y)  # reserve last column for CCL
        self._sdpa_program_configs: dict[tuple[int, bool], ttnn.SDPAProgramConfig] = {}

        # The exp ring op walks head-SEGMENTS (a head's Q chunks split over segs_per_head rows) as
        # serial passes, ceil(n_local_heads * segs / rows) passes per row. Segmentation is what
        # balances 14 local heads over 10 rows: segs=1 gives 2 passes of 10-tile chunks with 6 rows
        # idle on the second pass, while segs=2 gives 3 passes of 5-tile chunks on every core --
        # 15 Q tile-rows per core instead of 20 on the bottleneck cores.
        self.exp_ring_max_passes = 3  # kMaxPasses in exp_ring_joint_sdpa_program_builder.cpp
        self.exp_ring_num_passes = math.ceil(self.n_local_heads / full_grid.y)
        # Exp ring runs only under a recipe (Blackhole): the op chooses its blocking.
        self.use_exp_ring_sdpa = (
            self.use_ring
            and self.sdpa_precision is not None
            and tp_factor == 4
            and parallel_config.sequence_parallel.factor == 32
            and self.exp_ring_num_passes <= self.exp_ring_max_passes
        )
        self._exp_sdpa_program_configs: dict[int, ttnn.SDPAProgramConfig | None] = {}

        # Legacy SDPA compute config (non-Blackhole only; recipes own their numerics).
        self.sdpa_compute_kernel_config = (
            None
            if self.sdpa_precision is not None
            else ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
            )
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
        rename_substate(state, "to_out.0", "to_out")

        def _interleave_heads(tensors: list[torch.Tensor]) -> torch.Tensor:
            """Reorder [out, in] weights so TP column-fracturing gives each device matching heads.

            Out dim is `num_heads * head_dim`. Reshaping it to [n_dev, n_local_heads, head_dim] and
            concatenating the tensors on the heads axis puts device `d`'s q, k and v heads
            contiguously inside shard `d`, which is also the order `chunks=3` splits them back out in.
            Device `d` therefore owns canonical heads `[d * n_local, (d + 1) * n_local)`, so simply
            all-gathering the attention output on TP rebuilds the canonical 7168-channel order that
            `to_out` expects.
            """
            n_dev = self.parallel_config.tensor_parallel.factor
            tensors = [t.T for t in tensors]  # -> [in, out]
            tensors = [t.reshape(t.shape[0], n_dev, self.n_local_heads, self.head_dim) for t in tensors]
            merged = torch.cat(tensors, dim=2)
            merged = merged.reshape(merged.shape[0], len(tensors) * self.inner_dim)
            return merged.T

        q_state = pop_substate(state, "to_q")
        k_state = pop_substate(state, "to_k")
        v_state = pop_substate(state, "to_v")

        # Relayout Q and K into the interleaved rotary channel order the fused RoPE consumes. Shared
        # between Q and K and absent from V, so Q.K is unchanged (see `rope_channel_permutation`).
        perm = rope_channel_permutation(self.head_dim, self.rotary_dim)

        def _permute_rotary(weight: torch.Tensor) -> torch.Tensor:
            # [out, in] with out == num_heads * head_dim; permute within each head.
            return weight.reshape(self.num_heads, self.head_dim, -1)[:, perm].reshape(weight.shape)

        state["to_qkv.weight"] = _interleave_heads(
            [_permute_rotary(q_state["weight"]), _permute_rotary(k_state["weight"]), v_state["weight"]]
        )

        # The reference's QK-norm affine is one head_dim vector shared by every head. The fused op
        # wants it spanning the whole inner dim, so permute it the same way and repeat per head.
        for name in ("norm_q", "norm_k"):
            sub = pop_substate(state, name)
            if "weight" in sub:
                state[f"{name}.weight"] = sub["weight"][perm].repeat(self.num_heads)

    # ------------------------------------------------------------------ helpers

    def _sdpa_program_config(self, seq_local: int, *, ring: bool) -> ttnn.SDPAProgramConfig:
        """Ring (worker grid) or dense (full grid) SDPA config for a per-device sequence length.

        Recipe (Blackhole): the grid only; SDPA chooses the chunks from the shape, recipe and L1 (the
        per-length chunks measured for legacy ring SDPA on 4x8 Galaxy -- (320, 384) at 5s, (256, 512)
        at 10s/15s -- are no longer used). Legacy (non-Blackhole): up to (256, 512) chunks.
        """
        key = (seq_local, ring)
        if key not in self._sdpa_program_configs:
            grid = (
                ttnn.CoreCoord(*self.sdpa_worker_grid) if ring else ttnn.CoreCoord(self.full_grid.x, self.full_grid.y)
            )
            if self.sdpa_precision is not None:
                self._sdpa_program_configs[key] = sdpa_recipe.recipe_config(grid)
            else:
                tile = ttnn.TILE_SIZE
                self._sdpa_program_configs[key] = ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=grid,
                    q_chunk_size=max(tile, min(256, (seq_local // tile) * tile)),
                    k_chunk_size=max(tile, min(512, (seq_local // tile) * tile)),
                    exp_approx_mode=False,  # NOTE: False is more correct
                )
        return self._sdpa_program_configs[key]

    def _attn_program_config(self, seq_local: int, *, ring: bool) -> ttnn.SDPAProgramConfig:
        """The ring/dense program config (kept as the call sites' entry point)."""
        return self._sdpa_program_config(seq_local, ring=ring)

    def _sdpa_kwargs(self) -> dict:
        """Recipe kwargs, or (non-Blackhole) the legacy compute config read at call time."""
        return sdpa_recipe.sdpa_kwargs(self.sdpa_precision, self.sdpa_compute_kernel_config)

    def _exp_sdpa_program_config(self, seq_local: int) -> ttnn.SDPAProgramConfig | None:
        """Exp ring SDPA config for a per-device sequence length, or None if it cannot use the op."""
        if not self.use_exp_ring_sdpa:
            return None
        if seq_local not in self._exp_sdpa_program_configs:
            self._exp_sdpa_program_configs[seq_local] = self._build_recipe_exp_sdpa_program_config(seq_local)
        return self._exp_sdpa_program_configs[seq_local]

    # Recipe exp ring blocking (docs/sdpa_precision.md): Q 128-320 in 32-row steps (odd tile counts
    # allowed -- recipes keep exp-ring state resident), K512 only, at most 3 passes.
    _RECIPE_EXP_Q_RANGE = (128, 320)
    _RECIPE_EXP_K_CHUNK = 512
    _RECIPE_EXP_MAX_PASSES = 3

    def _build_recipe_exp_sdpa_program_config(self, seq_local: int) -> ttnn.SDPAProgramConfig | None:
        """Whether a recipe exp-ring shape exists (else None: ring joint SDPA), as an op-selected config.

        The exp ring op gives Q chunk `x` to core column `x`, so a head's chunks (split over up to
        three head-segments) must fill a core row exactly. Search (cols, segs_per_head) for a
        tile-multiple q_chunk in 128..320 with at most 3 passes (ceil(n_local_heads * segs / rows));
        the op then chooses the Q chunk and grid width itself (same row-filling, pass and Q-range
        constraints, plus L1), and its host check rejects a shape that overflows L1. None (no recipe
        shape) falls back to ring joint SDPA.
        """
        tile = ttnn.TILE_SIZE
        rows = self.full_grid.y
        q_lo, q_hi = self._RECIPE_EXP_Q_RANGE
        max_passes = min(self.exp_ring_max_passes, self._RECIPE_EXP_MAX_PASSES)
        for cols in range(self.full_grid.x - 1, 1, -1):
            for segs in (1, 2, 3):
                chunks = cols * segs
                q_chunk = math.ceil(math.ceil(seq_local / chunks) / tile) * tile
                if (
                    math.ceil(seq_local / q_chunk) == chunks
                    and q_lo <= q_chunk <= q_hi
                    and math.ceil(self.n_local_heads * segs / rows) <= max_passes
                ):
                    return sdpa_recipe.recipe_config(self.full_grid)
        return None

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        spatial_1BND: ttnn.Tensor,
        N: int | None = None,
        rope_cos: ttnn.Tensor | None = None,
        rope_sin: ttnn.Tensor | None = None,
        addcmul_residual: ttnn.Tensor | None = None,
        addcmul_gate: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """
        spatial_1BND: fractured hidden_size on TP; fractured N on SP when `is_sequence_parallel`,
            otherwise replicated on SP.
        rope_cos/rope_sin: [1, 1, N_local, rotary_dim], fractured N on SP, replicated on TP. Both
            None skips the rotary embedding entirely, as the token refiner requires.
        N: logical (unfractured) sequence length. Only needed for ring attention.
        addcmul_residual/addcmul_gate: when both are given, the gated residual
            `addcmul_residual + to_out(...) * addcmul_gate` is folded into the to_out matmul's
            epilogue instead of running as separate ops. Both must be TP-fractured like the output.

        Returns the attention output with the same distribution as the input.
        """
        assert (addcmul_residual is None) == (addcmul_gate is None), "addcmul residual/gate come as a pair"
        assert (rope_cos is None) == (rope_sin is None), "rope_cos and rope_sin must be given together"
        # The fused RoPE consumes head_dim-wide tables, not the reference's rotary_dim-wide ones: the
        # pass-through channels must be present as cos=1 / sin=0. Passing the raw reference tables
        # here is silently wrong rather than a shape error, so check it. See `prepare_rope_tables`.
        if rope_cos is not None and rope_cos.shape[-1] != self.head_dim:
            msg = (
                f"rope tables must be head_dim ({self.head_dim}) wide, got {rope_cos.shape[-1]}; "
                "build them with prepare_rope_tables()"
            )
            raise ValueError(msg)

        tp_factor = self.parallel_config.tensor_parallel.factor
        assert not (self.use_ring and N is None), "ring attention needs the logical sequence length N"

        # Passing parallel_config puts ColParallelLinear on all_gather_minimal_matmul_async: the TP
        # all-gather of the K-fractured input folds into the matmul that consumes it, instead of
        # running as a separate op. Only on ring topologies -- on a line the unfused path is faster,
        # which is the same condition WanAttention uses.
        matmul_parallel_config = self.parallel_config if self.use_fused_agmm else None
        if not self.use_fused_agmm and tp_factor > 1:
            spatial_1BND = self.ccl_manager.all_gather_persistent_buffer(
                spatial_1BND, dim=3, mesh_axis=self.tp_mesh_axis
            )

        q_1BNF, k_1BNF, v_1BNF = self.to_qkv(
            spatial_1BND,
            compute_kernel_config=self.mm_compute_kernel_config,
            parallel_config=matmul_parallel_config,
            default_block_size=agmm_block_size(self.hidden_size, 3 * self.inner_dim // tp_factor),
        )

        def create_heads(inp: ttnn.Tensor) -> ttnn.Tensor:
            out, _, _ = ttnn.experimental.nlp_create_qkv_heads(
                inp,
                num_heads=self.n_local_heads,
                num_kv_heads=0,
                transpose_k_heads=False,
            )
            return out

        # One fused op per stream: per-head RMSNorm over head_dim, head split, and RoPE. It emits
        # head-split [B, n_local_heads, N, head_dim] directly, so Q and K need no create_heads.
        norm_kwargs = dict(
            num_heads_per_device=self.n_local_heads,
            per_head_norm=True,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            trans_mat=self.rope_trans_mat if rope_cos is not None else None,
        )
        q_BHNE = self.norm_q(q_1BNF, **norm_kwargs)
        k_BHNE = self.norm_k(k_1BNF, **norm_kwargs)
        v_BHNE = create_heads(v_1BNF)
        # LOW_PRECISION prepares Q/K/V after norm/RoPE and before the ring gathers K/V (the persistent
        # buffers below then take the prepared KV dtype). Every other recipe and legacy: unchanged.
        q_BHNE, k_BHNE, v_BHNE = sdpa_recipe.prepare_recipe_inputs(
            self.sdpa_precision, self.sdpa_kv_dtype, q_BHNE, k_BHNE, v_BHNE
        )

        # Sequence is fractured across SP, so attention must gather K/V around the ring.
        # The packed sequence is one attention document and logical_n masks the pad tail, so no mask.
        exp_program_config = self._exp_sdpa_program_config(q_BHNE.shape[2])
        if exp_program_config is not None:
            spatial_BHNE, _prompt, _lse = ttnn.transformer.exp_ring_joint_scaled_dot_product_attention(
                q_BHNE,
                k_BHNE,
                v_BHNE,
                self.dummy_joint_input,
                self.dummy_joint_input,
                self.dummy_joint_input,
                persistent_output_buffer_k=self.ccl_manager.get_ag_ping_pong_buffer(
                    k_BHNE.shape, 2, self.sp_mesh_axis, dtype=k_BHNE.dtype
                ),
                persistent_output_buffer_v=self.ccl_manager.get_ag_ping_pong_buffer(
                    v_BHNE.shape, 2, self.sp_mesh_axis, dtype=v_BHNE.dtype
                ),
                joint_strategy="rear",
                logical_n=N,
                program_config=exp_program_config,
                **self._sdpa_kwargs(),
                dim=2,
                multi_device_global_semaphore=self.ccl_manager.get_exp_ring_ping_pong_semaphore(self.sp_mesh_axis),
                num_links=self.ccl_manager.num_links,
                cluster_axis=self.sp_mesh_axis,
                mesh_device=self.mesh_device,
                topology=self.ccl_manager.topology,
                subdevice_id=self.ccl_manager.ccl_sub_device_id,
                num_workers_per_link=5,
                num_buffers_per_channel=32,
            )
        elif self.use_ring:
            spatial_BHNE, _prompt, _lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
                q_BHNE,
                k_BHNE,
                v_BHNE,
                self.dummy_joint_input,
                self.dummy_joint_input,
                self.dummy_joint_input,
                persistent_output_buffer_k=self.ccl_manager.get_ag_ping_pong_buffer(
                    k_BHNE.shape, 2, self.sp_mesh_axis, dtype=k_BHNE.dtype
                ),
                persistent_output_buffer_v=self.ccl_manager.get_ag_ping_pong_buffer(
                    v_BHNE.shape, 2, self.sp_mesh_axis, dtype=v_BHNE.dtype
                ),
                joint_strategy="rear",
                logical_n=N,
                program_config=self._attn_program_config(q_BHNE.shape[2], ring=True),
                **self._sdpa_kwargs(),
                dim=2,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(self.sp_mesh_axis),
                num_links=self.ccl_manager.num_links,
                cluster_axis=self.sp_mesh_axis,
                mesh_device=self.mesh_device,
                topology=self.ccl_manager.topology,
                subdevice_id=self.ccl_manager.ccl_sub_device_id,
                ccl_core_grid_offset=(self.sdpa_worker_grid[0], 0),
                use_column_major_ccl=True,
            )
        else:
            spatial_BHNE = ttnn.transformer.scaled_dot_product_attention(
                q_BHNE,
                k_BHNE,
                v_BHNE,
                is_causal=False,
                program_config=self._attn_program_config(q_BHNE.shape[2], ring=False),
                **self._sdpa_kwargs(),
            )

        spatial_1BND = ttnn.transformer.concatenate_heads(spatial_BHNE)
        spatial_1BND = ttnn.unsqueeze(spatial_1BND, 0)

        # Each device holds canonical heads [d * n_local, (d+1) * n_local), so gathering on TP
        # rebuilds the full inner_dim in canonical order for to_out -- fused into the matmul when
        # use_fused_agmm.
        if not self.use_fused_agmm and tp_factor > 1:
            spatial_1BND = self.ccl_manager.all_gather_persistent_buffer(
                spatial_1BND, dim=3, mesh_axis=self.tp_mesh_axis
            )

        # The gated residual rides along in the matmul epilogue on the fused path; on the unfused
        # path the op has no addcmul, so apply it afterwards.
        fuse_gate = addcmul_residual is not None and self.use_fused_agmm
        out = self.to_out(
            spatial_1BND,
            compute_kernel_config=self.mm_compute_kernel_config,
            parallel_config=matmul_parallel_config,
            default_block_size=agmm_block_size(self.inner_dim, self.hidden_size // tp_factor),
            addcmul_a=addcmul_residual if fuse_gate else None,
            addcmul_b=addcmul_gate if fuse_gate else None,
        )
        if addcmul_residual is not None and not fuse_gate:
            out = ttnn.addcmul(addcmul_residual, out, addcmul_gate)
        return out
