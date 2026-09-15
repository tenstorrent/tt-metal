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

    # Per-device sequence length -> measured-best ring SDPA (q_chunk_size, k_chunk_size).
    # See `_sdpa_program_config` for how these were obtained and why the optimum moves with length.
    # 4768 / 9216 / 13632 are 768P at 5s / 10s / 15s, packed and padded, divided by SP=8.
    measured_sdpa_chunk_sizes = {
        4768: (320, 384),
        9216: (256, 512),
        13632: (256, 512),
    }

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
    ) -> None:
        super().__init__()

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
        self.exp_ring_max_passes = 3  # kMaxPasses in exp_ring_joint_sdpa_program_factory.cpp
        self.exp_ring_num_passes = math.ceil(self.n_local_heads / full_grid.y)
        self.exp_ring_max_k_chunk = 512  # largest k worth trying; `_exp_sdpa_l1_bytes` picks down from here
        self.use_exp_ring_sdpa = (
            self.use_ring
            and is_blackhole()
            and tp_factor == 4
            and parallel_config.sequence_parallel.factor == 32
            and self.exp_ring_num_passes <= self.exp_ring_max_passes
        )
        self._exp_sdpa_program_configs: dict[int, ttnn.SDPAProgramConfig | None] = {}

        self.sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
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
        """Ring SDPA chunk sizes for a given per-device sequence length.

        Measured points come from the sweep in
        `tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_create_perf_table`
        (model configs `minimax_h3_{5s,10s,15s}_768p`) on 4x8 Blackhole Galaxy, TP=4 / SP=8. Anything
        else falls back to the generic rule below.

        The optimum depends on the *local* sequence length, which is why this is keyed on it rather
        than on the mesh shape the way `WanAttention.sdpa_chunk_size_map` is. At long sequences there
        is enough work that (256, 512) wins -- a larger k halves the ring's K-loop iterations. At 5s
        there is too little work to fill 110 cores that way: 14 heads x ceil(4768/256) = 266 work items
        over 110 cores rounds up to 3 per core and wastes ~19% of the slots, whereas q=320 gives 210
        items, 2 per core and ~4.5% waste. That is worth more than the larger k.

        L1 (1.57 MB) bounds the (q, k) product, and the bound is what makes k=384 interesting: k=1024
        never fits, (320, 512) reaches 1.65 MB and does not either, but (320, 384) does -- combining
        the good q with a k larger than 256. Neither chunk size has to be a power of two, only a
        multiple of TILE, and restricting the search to {256, 512} misses this point entirely.

        Padding is not the thing to tune. 4768 is 32 x 149 with 149 prime, so no sensible chunk size
        divides it, yet at q=320 the Q padding is only 0.67%. Core slot efficiency dominates, and the
        optimum in it is sharp rather than a plateau: at 5s q=288 measured 10.43 ms against q=320's
        7.81 ms. Slot efficiency is a good candidate generator but not a predictor -- q=416 at 10s has
        the best slot efficiency of any k=256 point there (97.6%) and measured the worst (28.39 ms).
        """
        key = (seq_local, ring)
        if key not in self._sdpa_program_configs:
            tile = ttnn.TILE_SIZE
            measured = self.measured_sdpa_chunk_sizes.get(seq_local)
            if measured is not None:
                q_chunk, k_chunk = measured
            else:
                q_chunk = max(tile, min(256, (seq_local // tile) * tile))
                k_chunk = max(tile, min(512, (seq_local // tile) * tile))
            grid = (
                ttnn.CoreCoord(*self.sdpa_worker_grid) if ring else ttnn.CoreCoord(self.full_grid.x, self.full_grid.y)
            )
            self._sdpa_program_configs[key] = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=grid,
                q_chunk_size=q_chunk,
                k_chunk_size=k_chunk,
                exp_approx_mode=False,  # NOTE: False is more correct
            )
        return self._sdpa_program_configs[key]

    # One accumulator entry and one Q chunk per pass, in tiles of `_EXP_L1_TILE_BYTES`. Mirrors the
    # CB table in exp_ring_joint_sdpa_program_factory.cpp; reproduces its measured 1,302,528 B at
    # (224, 512) exactly. Nothing in the op validates this, so an oversized shape would only surface
    # as a CB allocation failure at program build.
    _EXP_L1_TILE_BYTES = 2048  # bf16 and Float16_b tiles are both 2 KiB
    # CB space measured IN THE PIPELINE, not bare L1: the op's CBs must end below the lowest live
    # L1 buffer (global semaphores etc. occupy the top of L1), which a 15s run measured at
    # 1,504,000 with the CB region starting at 191,360. The factory checks the live value at build;
    # this constant only has to be a safe lower bound so the k search picks a buildable shape.
    _EXP_USABLE_L1_BYTES = 1_312_640
    # DEST tiles from `get_dest_reg_count`: 1024 * 16 / (32 * 32), halved because dst_full_sync_en is
    # off, not halved again because `sdpa_compute_kernel_config` has fp32_dest_acc_en off.
    _EXP_DST_TILES = 8
    # `determine_largest_subblock_size` in sdpa_subblock_utils.hpp, in its search order.
    _EXP_SUBBLOCKS = (
        (2, 4), (4, 2), (1, 8), (8, 1), (1, 7), (7, 1), (2, 3), (3, 2), (1, 6), (6, 1),
        (1, 5), (5, 1), (2, 2), (1, 4), (4, 1), (1, 3), (3, 1), (1, 2), (2, 1), (1, 1),
    )  # fmt: skip

    def _exp_streaming_compute_enabled(self, sq_t: int, sk_t: int) -> bool:
        """Whether the op picks its streaming compute path for a chunk shape.

        Mirrors `use_streaming_compute` in exp_ring_joint_sdpa_program_factory.cpp. The exp compute
        kernel `static_assert`s on it, so a shape the factory judges ineligible does not fall back --
        it fails to build the kernel. The binding term in practice is `sk_t % (dst / h) == 0`: it
        rejects k=320 at q=320 (h=1, so sk_t must be a multiple of 8, and 10 is not).
        """
        dst = self._EXP_DST_TILES
        for h, w in self._EXP_SUBBLOCKS:
            if h * w <= dst and sq_t % h == 0 and sk_t % w == 0:
                return h <= 2 and sk_t % (dst // h) == 0 and sq_t // h > 1
        return False

    def _exp_sdpa_l1_bytes(self, sq_t: int, sk_t: int, p: int, resident_q: bool = True) -> int:
        """L1 the exp op's circular buffers need for a (q, k, passes) shape, in tiles of head_dim.

        `p` is the candidate's pass count (head-segment scheduling makes it per-candidate:
        ceil(n_local_heads * segs / rows)). `resident_q=False` models the op's streamed-Q fallback:
        when the resident total does not fit, the factory sizes c_0 to a single chunk and the reader
        re-reads each pass's Q every ring iteration. The op selects the mode itself from this same
        arithmetic; the model only needs it to know which (q, k) shapes are buildable.
        """
        dh_t = self.head_dim // ttnn.TILE_SIZE
        tiles = (
            (p if resident_q else 1) * sq_t * dh_t  # c_0 Q: one chunk per pass, or one when streamed
            + 4 * sk_t * dh_t  # c_1/c_14 K and c_2/c_15 V, double buffered
            + 7  # c_3 mask, scalars, reciprocal scratch
            + 2 * p * sq_t  # c_6 / c_11 state FIFO running max and sum
            + p * sq_t * dh_t  # c_7 state FIFO partial output
            + sq_t  # c_17 stats out (c_10 is dead on this path and not allocated)
            + 16  # c_16 streaming output ping-pong
            + sq_t * sk_t  # c_24 qk intermediate
            + 2 * sq_t * dh_t  # c_25/c_26 output scratch halves
            + 4 * sq_t  # c_27-c_30 max/sum scratch halves
            + sq_t  # c_31 exp max diff
        )
        return tiles * self._EXP_L1_TILE_BYTES

    def _exp_sdpa_program_config(self, seq_local: int) -> ttnn.SDPAProgramConfig | None:
        """Exp ring SDPA config for a per-device sequence length, or None if it cannot use the op.

        The op gives Q chunk `x` to core column `x`, so a head's chunks must fill its row exactly:
        `ceil(seq_local / q_chunk)` has to equal the SDPA column count. That pins q_chunk to the
        window `[seq_local / cols, seq_local / (cols - 1))` for a given width, and only a TILE
        multiple will do, so a width is usable only if one lands there. `measured_sdpa_chunk_sizes`
        therefore does not apply on this path -- though at the H3 10s shape the window happens to
        give q=224, the value WanAttention measured anyway.

        Widest usable grid wins, so try `full_grid.x - 1` columns first and step down. A narrower
        grid frees no L1 whatsoever -- every CB is sized from (q_chunk, k_chunk, passes), never from
        the column count -- so it is never worth taking while a wider one fits. It is only worth
        taking when no wider width admits a q_chunk at all, and there the comparison is not against
        a full-grid exp config but against no exp config at all: at 5s, 11 columns admit nothing
        (96 -> 13 chunks, 128 -> 10) while 10 columns take q=128 with L1 to spare.

        k_chunk is the one genuinely free variable, and the only one that buys L1 headroom, so take
        the largest that both fits L1 and keeps the op on its streaming compute path -- the kernel
        static_asserts on the latter, so an ineligible k fails the build rather than falling back.
        A shape fits if either Q mode does: resident Q (all passes' chunks stay in L1, read once) or
        the op's streamed-Q fallback (one chunk resident, re-read per pass per ring iteration).
        That gives 512 at q=224 resident, and 384 at q=320 streamed -- where the k=256 that resident
        Q would force measured far slower (small k doubles the per-chunk flash overhead; see
        exp_more_heads_per_row.md §9).
        """
        if not self.use_exp_ring_sdpa:
            return None
        if seq_local not in self._exp_sdpa_program_configs:
            self._exp_sdpa_program_configs[seq_local] = self._build_exp_sdpa_program_config(seq_local)
        return self._exp_sdpa_program_configs[seq_local]

    def _build_exp_sdpa_program_config(self, seq_local: int) -> ttnn.SDPAProgramConfig | None:
        """Search (cols, segs_per_head, q_chunk, k_chunk) and take the lightest bottleneck load.

        The per-core matmul work per ring iteration is passes * q_chunk Q rows against the full
        K/V stream, so `passes * q_chunk` is the primary score: at 14 heads on 10 rows, segs=1
        gives 2 passes x 320 = 640 rows while segs=2 gives 3 passes x 160 = 480. Larger k_chunk
        is the tie-break (fewer per-chunk overheads), then wider grids.
        """
        tile = ttnn.TILE_SIZE
        rows = self.full_grid.y
        best = None
        for cols in range(self.full_grid.x - 1, 1, -1):
            for segs in (1, 2, 3):
                chunks = cols * segs
                q_chunk = math.ceil(math.ceil(seq_local / chunks) / tile) * tile
                if math.ceil(seq_local / q_chunk) != chunks:
                    continue  # this (cols, segs) admits no tile-multiple q_chunk
                passes = math.ceil(self.n_local_heads * segs / rows)
                if passes > self.exp_ring_max_passes:
                    continue
                for k_chunk in range(self.exp_ring_max_k_chunk, 0, -tile):
                    sq_t, sk_t = q_chunk // tile, k_chunk // tile
                    fits = (
                        self._exp_sdpa_l1_bytes(sq_t, sk_t, passes) <= self._EXP_USABLE_L1_BYTES
                        or self._exp_sdpa_l1_bytes(sq_t, sk_t, passes, resident_q=False) <= self._EXP_USABLE_L1_BYTES
                    )
                    if fits and self._exp_streaming_compute_enabled(sq_t, sk_t):
                        score = (passes * q_chunk, -k_chunk, -cols)
                        if best is None or score < best[0]:
                            best = (score, cols, q_chunk, k_chunk)
                        break
        if best is None:
            return None
        _, cols, q_chunk, k_chunk = best
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(cols + 1, self.full_grid.y),
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=False,  # NOTE: False is more correct
        )

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
                compute_kernel_config=self.sdpa_compute_kernel_config,
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
                program_config=self._sdpa_program_config(q_BHNE.shape[2], ring=True),
                compute_kernel_config=self.sdpa_compute_kernel_config,
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
                program_config=self._sdpa_program_config(q_BHNE.shape[2], ring=False),
                compute_kernel_config=self.sdpa_compute_kernel_config,
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
