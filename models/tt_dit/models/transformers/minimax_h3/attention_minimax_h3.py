# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os

import torch

import ttnn
from models.common.utility_functions import is_blackhole

from ....layers.linear import ColParallelLinear, RowParallelLinear, maybe_cast_activation
from ....layers.module import Module
from ....layers.normalization import DistributedRMSNorm
from ....parallel.config import DiTParallelConfig
from ....parallel.manager import CCLManager
from ....utils.mochi import get_rot_transformation_mat
from ....utils.substate import pop_substate, rename_substate
from ....utils.tensor import bf16_tensor
from . import lofi
from .agmm_config import agmm_kwargs, agmm_pins
from .length_classes import LENGTH_CLASSES, length_class
from .mmrs_config import has_row_parallel_mmrs_config, mmrs_blocking_override
from .sol_prepass import BLOCK as SOL_BLOCK
from .sol_prepass import SolGeometry, SolPrepass, sink_block_mask
from .sol_prepass_fused import SolPrepassFused

# Ring SDPA chunk pair of the dynamic-length path per length class, one program each. Measured per block
# on 4x8: (320, 384) is 5 % faster at 4736 rows/device, (256, 512) 5 % faster at 13664.
DYNAMIC_RING_SDPA_CHUNKS: dict[str, tuple[int, int]] = {"short": (320, 384), "long": (256, 512)}

_TO_OUT_ROW_PARALLEL = os.environ.get("MINIMAX_H3_TO_OUT_ROW_PARALLEL", "0") == "1"
_TO_OUT_BFP8_WEIGHT = os.environ.get("MINIMAX_H3_TO_OUT_BFP8_WEIGHT", "0") == "1"
_S1_LOGGED: set = set()


def _log_s1_once(dtype) -> None:
    if dtype not in _S1_LOGGED:
        _S1_LOGGED.add(dtype)
        print(f"[MINIMAX_H3_MK_S1] sparse ring output token-major, dtype {dtype}", flush=True)


# MINIMAX_H3_MK_S1_TOKEN_MAJOR_O (token-major output, default off): the sparse ring op writes its output token-major
# [1, 1, N, H * D] -- the concatenate_heads layout -- so the head concat goes; with "1" / "bf8" it also packs bfloat8_b
# when to_out takes a BFP8 activation, so the to_out cast goes too. "layout" = token-major only (bf16, bit-identical by
# construction).
_S1_TOKEN_MAJOR_O = os.environ.get("MINIMAX_H3_MK_S1_TOKEN_MAJOR_O", "0")
if _S1_TOKEN_MAJOR_O not in ("0", "1", "bf8", "layout"):
    raise ValueError(f"MINIMAX_H3_MK_S1_TOKEN_MAJOR_O must be 0, 1, bf8 or layout, got {_S1_TOKEN_MAJOR_O!r}")
# MINIMAX_H3_MK_E2A_HEAD_SPLIT (head-split output chunks, default off): the qkv all-gather-matmul writes V (chunk 2)
# already head-split [1, H, N, D] -- the nlp_create_qkv_heads layout -- so V's create_heads goes on every path (sparse,
# dense, refiner). Bit-identical (the same tiles at the addresses create_heads would move them to).
_E2A_HEAD_SPLIT = os.environ.get("MINIMAX_H3_MK_E2A_HEAD_SPLIT", "0")
if _E2A_HEAD_SPLIT not in ("0", "1"):
    raise ValueError(f"MINIMAX_H3_MK_E2A_HEAD_SPLIT must be 0 or 1, got {_E2A_HEAD_SPLIT!r}")
_E2A_LOGGED: list = []
if _TO_OUT_BFP8_WEIGHT and not _TO_OUT_ROW_PARALLEL:
    raise ValueError("MINIMAX_H3_TO_OUT_BFP8_WEIGHT=1 needs MINIMAX_H3_TO_OUT_ROW_PARALLEL=1")


def to_out_weight_tag() -> str:
    """Suffix for the transformer weight-cache subfolder: the row-parallel to_out shards its weight by rows (and 19b
    stores it in BFP8), so it must never load (or overwrite) today's cache. Empty with the switches off."""
    if not _TO_OUT_ROW_PARALLEL:
        return ""
    return "_toutrow_bf8" if _TO_OUT_BFP8_WEIGHT else "_toutrow"


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


def _binding_has_sol_mode() -> bool:
    """True if the installed ring_joint SDPA binding accepts the `sol_mode` keyword (probed from its TypeError text)."""
    raw = ttnn._ttnn.operations.transformer.ring_joint_scaled_dot_product_attention
    try:
        raw()
    except TypeError as e:
        return "sol_mode" in str(e)
    return False


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
    # 4736 / 9184 / 13664 are 768P at 5s / 10s / 15s with the perf gate's 39-token prompt, packed and
    # padded to SP * TILE, divided by SP=8 -- the values the pipeline logs as "rows/device". Padding
    # buckets prompt length: at 15 s any prompt of 1-250 tokens lands on 13664. The 5 s entry was
    # measured on Blackhole's 110 SDPA cores and not yet re-checked on Wormhole's 63.
    measured_sdpa_chunk_sizes = {
        4736: (320, 384),
        9184: (256, 512),
        13664: (256, 512),
    }
    measured_resident_k_chunks = {
        4736: 256,
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
        kv_gather_capacity: int | None = None,
        use_persistent_ccl_buffers: bool = True,
        sol_attn: bool = False,
        sol_tau: float = 1.0,
        sol_thresh_type: str = "diag",
        sol_band: int = 1,
        sol_prefix_mode: str = "separate",
        sol_dense_prefix: bool | None = None,
        sol_mode: str = "gather",
        sol_fused: bool = False,
        sol_mask_reuse: int = 1,
        dynamic_length: bool = False,
        norm_row_capacity: int | None = None,
    ) -> None:
        """`dynamic_length` freezes every length-keyed choice at construction (one program per op for every
        length); `norm_row_capacity` (per-device rows) sizes the fused QK norms' program on that path."""
        super().__init__()

        self.sol_attn = sol_attn
        self.sol_tau = sol_tau
        self.sol_thresh_type = sol_thresh_type
        self.sol_band = sol_band
        assert sol_prefix_mode in ("separate", "fused", "off"), sol_prefix_mode
        self.sol_prefix_mode = sol_prefix_mode
        if sol_dense_prefix is not None:
            self.sol_dense_prefix = sol_dense_prefix
        self.sol_prefix_broadcast = os.environ.get("SOL_PREFIX_BROADCAST", "1") != "0"
        self._sol_prefix_cache: dict = {}
        self.sol_mode = sol_mode
        self.sol_fused = sol_fused
        self.sol_mask_reuse = sol_mask_reuse
        self.sol_active = False
        self._sol_logical_n: int | None = None
        self.sol_sink_tokens = 0
        # Prefix spread (MINIMAX_H3_SOL_PREFIX_SPREAD, prefix_spread.py): the logical 64-token block of each packed
        # block when the pipeline permuted the rows to spread the dense prefix query blocks over SP; None = identity.
        # Set by the pipeline per request, with the row tables it permutes.
        self.sol_block_map: tuple[int, ...] | None = None
        self._sol_dense_cache: dict = {}  # _sol_dense_q_blocks, one entry
        self.sol_report_density = False
        self.sol_last_density: float | None = None
        self.sol_last_block_density: float | None = None
        self._sol_prepass: SolPrepass | None = None

        self.kv_gather_capacity = kv_gather_capacity
        self.use_persistent_ccl_buffers = use_persistent_ccl_buffers
        self.dynamic_length = dynamic_length

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
        self.to_out_row_parallel = _TO_OUT_ROW_PARALLEL and tp_factor > 1 and is_sequence_parallel
        if self.to_out_row_parallel:
            self.to_out = RowParallelLinear(
                self.inner_dim,
                hidden_size,
                bias=False,
                dtype=ttnn.bfloat8_b if _TO_OUT_BFP8_WEIGHT else ttnn.bfloat16,
                mesh_device=mesh_device,
                mesh_axis=self.tp_mesh_axis,
                fsdp_mesh_axis=fsdp_mesh_axis,
                ccl_manager=ccl_manager,
            )
            self.to_out.activation_dtype = None
            self.to_out.pin_output_bf16 = False
        else:
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
            **({"row_capacity": norm_row_capacity} if dynamic_length else {}),
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
        self._sdpa_program_configs: dict[tuple[int, bool, bool], ttnn.SDPAProgramConfig] = {}

        # Dynamic-length pins per length class, resolved once: no table scan and no orientation flip per call.
        self._qkv_pins = self._out_pins = self._ring_sdpa_configs = None
        if dynamic_length and self.use_fused_agmm:
            pin_kwargs = dict(full_grid=full_grid, num_links=ccl_manager.num_links)
            self._qkv_pins = agmm_pins(hidden_size, 3 * self.inner_dim // tp_factor, **pin_kwargs)
            self._out_pins = agmm_pins(self.inner_dim, hidden_size // tp_factor, **pin_kwargs)
        if dynamic_length and self.use_ring:
            self._ring_sdpa_configs = {
                cls.name: ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(*self.sdpa_worker_grid),
                    q_chunk_size=DYNAMIC_RING_SDPA_CHUNKS[cls.name][0],
                    k_chunk_size=DYNAMIC_RING_SDPA_CHUNKS[cls.name][1],
                    exp_approx_mode=False,
                )
                for cls in LENGTH_CLASSES
            }

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
            and ccl_manager.topology == ttnn.Topology.Ring
            and not dynamic_length  # the exp ring op keys its program on the length
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

        state["to_qkv.weight"] = lofi.round_weight(
            "qkv",
            _interleave_heads(
                [_permute_rotary(q_state["weight"]), _permute_rotary(k_state["weight"]), v_state["weight"]]
            ),
        )
        if "to_out.weight" in state:
            state["to_out.weight"] = lofi.round_weight("to_out", state["to_out.weight"])

        # The reference's QK-norm affine is one head_dim vector shared by every head. The fused op
        # wants it spanning the whole inner dim, so permute it the same way and repeat per head.
        for name in ("norm_q", "norm_k"):
            sub = pop_substate(state, name)
            if "weight" in sub:
                state[f"{name}.weight"] = sub["weight"][perm].repeat(self.num_heads)

    # ------------------------------------------------------------------ helpers

    def _sdpa_program_config(
        self,
        seq_local: int,
        *,
        ring: bool,
        windowed: bool = False,
        q_chunk: int | None = None,
        k_chunk: int | None = None,
    ) -> ttnn.SDPAProgramConfig:
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

        `windowed` caps k at 256 so the on-device mask CB fits in L1.
        """
        key = (seq_local, ring, windowed, q_chunk, k_chunk)
        if key not in self._sdpa_program_configs:
            tile = ttnn.TILE_SIZE
            measured = self.measured_sdpa_chunk_sizes.get(seq_local)
            q_override, k_override = q_chunk, k_chunk
            if measured is not None:
                q_chunk, k_chunk = measured
            else:
                q_chunk = max(tile, min(256, (seq_local // tile) * tile))
                k_chunk = max(tile, min(512, (seq_local // tile) * tile))
            if q_override is not None:
                q_chunk = q_override
                k_chunk = self.measured_resident_k_chunks.get(seq_local, k_chunk)
            if k_override is not None:
                k_chunk = k_override
            if windowed:
                k_chunk = min(k_chunk, 256)
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

        The state FIFO holds `p + 1` entries; this MUST match the factory's `state_fifo_entries`.
        """
        dh_t = self.head_dim // ttnn.TILE_SIZE
        fifo_entries = p + 1
        tiles = (
            (p if resident_q else 1) * sq_t * dh_t  # c_0 Q: one chunk per pass, or one when streamed
            + 4 * sk_t * dh_t  # c_1/c_14 K and c_2/c_15 V, double buffered
            + 7  # c_3 mask, scalars, reciprocal scratch
            + 2 * fifo_entries * sq_t  # c_6 / c_11 state FIFO running max and sum
            + fifo_entries * sq_t * dh_t  # c_7 state FIFO partial output
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
            exp_approx_mode=os.environ.get("SOL_EXP_APPROX", "0") == "1",
        )

    @property
    def sol_logical_n(self) -> int | None:
        return self._sol_logical_n

    @sol_logical_n.setter
    def sol_logical_n(self, value: int | None) -> None:
        self._sol_logical_n = value
        self._sol_prepass = None
        self._sol_prefix_cache = {}
        SolPrepassFused.new_generation()

    def _sol_tensors(self, q_BHNE: ttnn.Tensor, k_BHNE: ttnn.Tensor, v_BHNE: ttnn.Tensor) -> dict:
        """Run the routing pre-pass and return the ring op's `sol_*` keyword tensors."""
        self.sol_prepare(q_BHNE.shape[2])
        if isinstance(self._sol_prepass, SolPrepassFused):
            self._sol_prepass.pooled_v_bfp8 = bool(
                self._BF8_POOLED_V
                and self._SPARSE_RING
                and self._SPARSE_RING_DTYPE == "bf8"
                and self.sol_mode == "resident"
            )
        scale = self.head_dim**-0.5
        out = self._sol_prepass(q_BHNE, k_BHNE, v_BHNE, scale=scale, report_density=self.sol_report_density)
        return self._sol_kwargs(out)

    def sol_prepare(self, rows: int) -> None:
        """Host-side half of `_sol_tensors`: build the pre-pass if needed and prepare its constants for `rows` local Q
        rows. No device op is issued except a constant (re)upload, so a traced denoise calls this before replaying
        the captured blocks (whose `_sol_tensors` does not run on replay)."""
        assert self.sol_logical_n is not None, "set sol_logical_n (host int) before a sparse forward"
        sp_factor = self.parallel_config.sequence_parallel.factor
        cfg = self._sdpa_program_config(rows, ring=True, q_chunk=self._sol_q_chunk)
        geo = SolGeometry(
            shards=sp_factor,
            rows=rows,
            logical_n=self.sol_logical_n,
            q_chunk=cfg.q_chunk_size,
            k_chunk=cfg.k_chunk_size,
        )
        if self._sol_prepass is None:
            prepass_cls = SolPrepassFused if self.sol_fused else SolPrepass
            extra = {}
            if self.sol_fused and self._DYN_SPLIT and self._SPARSE_RING and self.sol_mode == "resident":
                extra["head_weights"] = True
            self._sol_prepass = prepass_cls(
                mesh_device=self.mesh_device,
                ccl_manager=self.ccl_manager,
                sp_axis=self.sp_mesh_axis,
                tau=self.sol_tau,
                thresh_type=self.sol_thresh_type,
                band=self.sol_band,
                dense_prefix_queries=self.sol_prefix_mode == "fused",
                mode=self._sol_prepass_mode,
                emit_bias=not (self._sol_compact_family or self.sol_mode == "resident"),
                row_max=False,
                **extra,
            )
        # The ring-joint reader's gather / compact paths attend block 0 of a shard exactly when a Q chunk keeps no block
        # there, but take the pooled bias from the mask, which then still counts block 0 as pooled (double weight).
        # Only the resident path corrects the bias, so the others rely on the pre-pass forcing block 0 into the mask.
        # The pre-pass can only force a full block, so block 0 of every shard must be full: a short or all-padding
        # block 0 (the last shards when logical_n leaves fewer than 64 tokens there) is never in the mask, and the
        # reader would then force it anyway.
        if self.sol_mode != "resident":
            assert (
                self._sol_prepass.force_first_chunk
            ), f"sol_mode={self.sol_mode!r} needs the pre-pass to force block 0 exact (force_first_chunk=True)"
            block0 = geo.block_lengths()[:, 0]
            short = [s for s in range(geo.shards) if int(block0[s]) < geo.block]
            assert not short, (
                f"sol_mode={self.sol_mode!r}: block 0 of shard(s) {short} has fewer than {geo.block} tokens "
                f"(lengths {[int(block0[s]) for s in short]}, rows {rows}, logical_n {self.sol_logical_n}); the "
                "reader would attend it exactly while the bias also pools it. Use sol_mode='resident' or a length "
                "that leaves at least one full block in every shard"
            )
        if self.sol_mask_reuse > 1 and not self.sol_fused:
            raise ValueError("sol_mask_reuse > 1 needs the fused pre-pass (sol_fused=True)")
        if self.sol_fused:
            self._sol_prepass.mask_reuse = int(self.sol_mask_reuse)
        scale = self.head_dim**-0.5
        if self.sol_block_map is not None and self.sol_prefix_mode == "separate":
            raise ValueError("the prefix spread (sol_block_map) needs sol_prefix_mode 'fused' or 'off', not 'separate'")
        self._sol_prepass.block_map = self.sol_block_map
        self._sol_prepass.prepare(geo, self.sol_sink_tokens, scale)

    def _sol_dense_q_blocks(self, rows: int) -> list[int]:
        """Prefix-spread step 2 (MINIMAX_H3_SOL_PREFIX_WEIGHTS=1, `fused` prefix mode, dynamic split on): the sparse
        op's dense Q blocks, global index shard * blocks per shard + local block, exactly the Q blocks Sol's `fused`
        rule makes dense (a sink token in the block, by the block's logical identity under `sol_block_map`). The op
        then splits each head's blocks over its cores by cost. Empty (the equal-count split) otherwise, or with more than 8 on
        a shard (the op's limit)."""
        if not (self._PREFIX_WEIGHTS and self.sol_prefix_mode == "fused" and self._DYN_SPLIT):
            return []
        key = (rows, repr(self.sol_sink_tokens), self.sol_logical_n, self.sol_block_map)
        cached = self._sol_dense_cache.get(key)
        if cached is not None:
            return cached
        sp = self.parallel_config.sequence_parallel.factor
        geo = SolGeometry(shards=sp, rows=rows, logical_n=self.sol_logical_n, q_chunk=SOL_BLOCK, k_chunk=SOL_BLOCK)
        nb = geo.nb
        sink = sink_block_mask(geo, self.sol_sink_tokens)[:, :nb].reshape(-1)  # logical blocks holding a sink token
        bmap = self.sol_block_map if self.sol_block_map is not None else range(sp * nb)
        dense = [g for g, lg in enumerate(bmap) if bool(sink[lg])]
        per_shard = [sum(1 for g in dense if g // nb == s) for s in range(sp)]
        if max(per_shard, default=0) > 8:
            from loguru import logger

            logger.info(f"prefix weights off: {max(per_shard)} dense Q blocks on one shard (the op takes at most 8)")
            dense = []
        self._sol_dense_cache = {key: dense}
        return dense

    def _sol_kwargs(self, out: dict) -> dict:
        if self.sol_report_density:
            self.sol_last_density = self._sol_prepass.last_density
            self.sol_last_block_density = getattr(self._sol_prepass, "last_block_density", None)
        kwargs = {
            "sol_block_mask": out["block_mask"],
            "sol_pooled_k": out["pooled_k"],
            "sol_pooled_v": out["pooled_v"],
        }
        if out.get("bias") is not None and not self._sol_compact_family and self.sol_mode != "resident":
            kwargs["sol_bias"] = out["bias"]
        if self.sol_mode == "resident" and out.get("row_max") is not None:
            kwargs["sol_row_max"] = out["row_max"]
        if self.sol_mode == "resident" and self._SPARSE_RING and out.get("head_weights") is not None:
            kwargs["sol_head_weights"] = out["head_weights"]
        if self._sol_compact_family or self.sol_mode == "resident":
            if not _binding_has_sol_mode():
                raise RuntimeError(
                    f"sol_mode={self.sol_mode!r} requested but the installed ring_joint SDPA binding has no sol_mode keyword"
                )
            kwargs["sol_mode"] = self.sol_mode
        return kwargs

    @property
    def sol_dense_prefix(self) -> bool:
        """Back-compat view of `sol_prefix_mode`: True is the fused mask rule, False switches the recompute off."""
        return self.sol_prefix_mode == "fused"

    @sol_dense_prefix.setter
    def sol_dense_prefix(self, value: bool) -> None:
        self.sol_prefix_mode = "fused" if value else "off"

    _SOL_PREFIX_EXTRA = 32
    _SOL_PREFIX_MARK = 4096.0

    def _sol_dense_segments(self, rows: int) -> list[tuple[int, int, int]]:
        """(chip, first row, end row) runs of local query rows that hold a sink token, in whole 64-token blocks.

        A prefix sink (an int) gives one run on the ring's first chip, rows [0, ceil(sink / 64) * 64): the original
        rule. A range sink (Ref2VA: text + audio, see `sol_prepass.sink_block_mask`) gives one run per chip it touches.
        """
        sp = self.parallel_config.sequence_parallel.factor
        geo = SolGeometry(shards=sp, rows=rows, logical_n=self.sol_logical_n, q_chunk=SOL_BLOCK, k_chunk=SOL_BLOCK)
        mask = sink_block_mask(geo, self.sol_sink_tokens)
        segments = []
        for chip in range(sp):
            b = 0
            while b < geo.nb:
                if not mask[chip, b]:
                    b += 1
                    continue
                e = b
                while e < geo.nb and mask[chip, e]:
                    e += 1
                r0, r1 = b * SOL_BLOCK, min(e * SOL_BLOCK, rows)
                assert r0 % ttnn.TILE_SIZE == 0 and r1 % ttnn.TILE_SIZE == 0, (r0, r1)
                segments.append((chip, r0, r1))
                b = e
        return segments

    def _sol_prefix_consts(self, rows: int) -> dict:
        """Per-request constants for `_sol_prefix_dense_rows`, rebuilt only when the geometry or the sink changes."""
        key = (rows, self.sol_sink_tokens, self.sol_logical_n)
        if self._sol_prefix_cache.get("key") == key:
            return self._sol_prefix_cache
        sp = self.parallel_config.sequence_parallel.factor
        segments = self._sol_dense_segments(rows)
        D, X = self.head_dim, self._SOL_PREFIX_EXTRA
        R = rows + X
        marker = torch.zeros(sp, 1, R, D + X)
        for s in range(sp):
            valid = max(0, min(rows, self.sol_logical_n - s * rows))
            marker[s, 0, valid:, D] = -1.0
            marker[s, 0, R - 1, D] = 0.0
        probe = torch.zeros(1, 1, R, D + X)
        probe[0, 0, R - 1, D] = 1.0
        qmark = torch.zeros(1, 1, 1, D + X)
        qmark[..., D] = self._SOL_PREFIX_MARK
        colfix = torch.ones(1, 1, 1, X)
        colfix[..., 0] = 0.0
        onehot = 1.0 - colfix
        dev, sp_axis = self.mesh_device, self.sp_mesh_axis
        flags = {}
        for chip in sorted({seg[0] for seg in segments}):
            flag = torch.zeros(sp, 1, 1, 1)
            flag[chip] = 1.0
            flags[chip] = bf16_tensor(flag, device=dev, mesh_axis=sp_axis, shard_dim=0)
        self._sol_prefix_cache = dict(
            key=key,
            marker=bf16_tensor(marker, device=dev, mesh_axis=sp_axis, shard_dim=0),
            probe=bf16_tensor(probe, device=dev),
            qmark=bf16_tensor(qmark, device=dev),
            colfix=bf16_tensor(colfix, device=dev),
            onehot=bf16_tensor(onehot, device=dev),
            segments=segments,
            flags=flags,
            cfg=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(self.full_grid.x, self.full_grid.y),
                q_chunk_size=128,
                k_chunk_size=512,
                exp_approx_mode=False,
            ),
        )
        return self._sol_prefix_cache

    def _sol_prefix_dense_rows(
        self, q_BHNE: ttnn.Tensor, k_BHNE: ttnn.Tensor, v_BHNE: ttnn.Tensor, out_BHNE: ttnn.Tensor
    ) -> ttnn.Tensor:
        """Exact attention for the prefix query rows, as a separate op after the sparse ring op.

        The prefix rows live on the ring's first chip. Every chip gets a copy of them (one all-gather), runs
        a plain dense SDPA of those rows against its OWN K/V shard, and the per-chip partial results are merged
        the way ring attention merges them -- which needs each chip's softmax normalizer Z_r. The plain SDPA
        does not output it, so the inputs carry a marker: 32 spare channels are appended to Q/K/V; one probe
        key (K = 0, V = unit vector in the spare channel) is appended to each shard, so the spare channel of
        the output is exactly 1 / (Z_r + 1); pad keys get a -1 in the spare channel of K against a +4096 in
        Q, which masks them (logit -362) without an attn_mask tensor. One all-reduce then sums the
        renormalized numerators and the Z_r, and the first chip's rows are overwritten with the quotient.
        Every chip attends only its own keys, so the dense work is spread over the ring instead of landing
        on the first chip's cores (the "fused" prefix mode).

        A range sink (Ref2VA text + audio) has rows on more than one chip: the same recompute runs once per
        (chip, rows) segment, with that chip as the sender and the only chip that overwrites.
        """
        rows = q_BHNE.shape[2]
        c = self._sol_prefix_consts(rows)
        for chip, r0, r1 in c["segments"]:
            out_BHNE = self._sol_dense_segment(q_BHNE, k_BHNE, v_BHNE, out_BHNE, c, chip, r0, r1)
        return out_BHNE

    def _sol_dense_segment(self, q_BHNE, k_BHNE, v_BHNE, out_BHNE, c: dict, chip: int, r0: int, r1: int) -> ttnn.Tensor:
        """Exact attention for query rows [r0, r1) of `chip` (see `_sol_prefix_dense_rows`)."""
        rows, H, D, X = q_BHNE.shape[2], self.n_local_heads, self.head_dim, self._SOL_PREFIX_EXTRA
        n_pre = r1 - r0
        ccl, sp_axis = self.ccl_manager, self.sp_mesh_axis

        q_pre = ttnn.slice(q_BHNE, [0, 0, r0, 0], [1, H, r1, D])
        if self.sol_prefix_broadcast and hasattr(ttnn.experimental, "broadcast_ring"):
            q_bc = ttnn.experimental.broadcast_ring(
                q_pre,
                sender_ring_index=chip,
                cluster_axis=sp_axis,
                num_links=ccl.num_links,
                topology=ccl.topology,
                subdevice_id=ccl.ccl_sub_device_id,
            )
            ttnn.deallocate(q_pre)
        else:
            q_all = ccl.all_gather(q_pre, dim=0, mesh_axis=sp_axis, use_hyperparams=False)
            ttnn.deallocate(q_pre)
            q_bc = ttnn.slice(q_all, [chip, 0, 0, 0], [chip + 1, H, n_pre, D])
            ttnn.deallocate(q_all)

        qx = ttnn.add(ttnn.pad(q_bc, [(0, 0), (0, 0), (0, 0), (0, X)], 0.0), c["qmark"])
        ttnn.deallocate(q_bc)
        kx = ttnn.add(ttnn.pad(k_BHNE, [(0, 0), (0, 0), (0, X), (0, X)], 0.0), c["marker"])
        vx = ttnn.add(ttnn.pad(v_BHNE, [(0, 0), (0, 0), (0, X), (0, X)], 0.0), c["probe"])

        ox = ttnn.transformer.scaled_dot_product_attention(
            qx,
            kx,
            vx,
            is_causal=False,
            scale=D**-0.5,
            program_config=c["cfg"],
            compute_kernel_config=self.sdpa_compute_kernel_config,
        )
        for t in (qx, kx, vx):
            ttnn.deallocate(t)
        o_part = ttnn.slice(ox, [0, 0, 0, 0], [1, H, n_pre, D])
        p = ttnn.slice(ox, [0, 0, 0, D], [1, H, n_pre, D + X])
        ttnn.deallocate(ox)

        a = ttnn.reciprocal(ttnn.add(p, c["colfix"]))
        ttnn.deallocate(p)
        a0 = ttnn.sum(ttnn.multiply(a, c["onehot"]), dim=3, keepdim=True)
        num = ttnn.multiply(o_part, a0)
        ttnn.deallocate(o_part)
        z = ttnn.subtract(a, 1.0)
        ttnn.deallocate(a)
        packed = ttnn.concat([num, z], dim=3)
        ttnn.deallocate(num)
        ttnn.deallocate(z)

        total = ccl.all_gather(
            ccl.reduce_scatter(packed, dim=2, mesh_axis=sp_axis), dim=2, mesh_axis=sp_axis, use_hyperparams=False
        )
        ttnn.deallocate(packed)
        n_sum = ttnn.slice(total, [0, 0, 0, 0], [1, H, n_pre, D])
        z_cols = ttnn.slice(total, [0, 0, 0, D], [1, H, n_pre, D + X])
        ttnn.deallocate(total)
        z_sum = ttnn.sum(ttnn.multiply(z_cols, c["onehot"]), dim=3, keepdim=True)
        ttnn.deallocate(z_cols)
        merged = ttnn.multiply(n_sum, ttnn.reciprocal(z_sum))
        ttnn.deallocate(n_sum)

        out_pre = ttnn.slice(out_BHNE, [0, 0, r0, 0], [1, H, r1, D])
        sel = ttnn.add(out_pre, ttnn.multiply(ttnn.subtract(merged, out_pre), c["flags"][chip]))
        ttnn.deallocate(merged)
        ttnn.deallocate(out_pre)
        pieces = [sel]
        if r0 > 0:
            pieces.insert(0, ttnn.slice(out_BHNE, [0, 0, 0, 0], [1, H, r0, D]))
        if r1 < rows:
            pieces.append(ttnn.slice(out_BHNE, [0, 0, r1, 0], [1, H, rows, D]))
        new_out = ttnn.concat(pieces, dim=2) if len(pieces) > 1 else sel
        for t in pieces:
            if t is not new_out:
                ttnn.deallocate(t)
        ttnn.deallocate(out_BHNE)
        return new_out

    _SOL_RESIDENT_Q_CHUNK = int(os.environ.get("SOL_RESIDENT_Q_CHUNK", "64"))
    _SOL_RESIDENT_PASSES = int(os.environ.get("SOL_RESIDENT_PASSES", "1"))
    _SOL_RESIDENT_K_CHUNK = int(os.environ.get("SOL_RESIDENT_K_CHUNK", "0"))

    @property
    def _sol_prepass_mode(self) -> str:
        return "compact" if self.sol_mode == "resident" else self.sol_mode

    @property
    def _sol_q_chunk(self) -> int | None:
        return self._SOL_RESIDENT_Q_CHUNK if self.sol_mode == "resident" else None

    @property
    def _sol_compact_family(self) -> bool:
        """compact and pack: block mask streamed through the chains, bias generated on device."""
        return self.sol_mode in ("compact", "pack")

    def sol_reset_mask_reuse(self) -> None:
        """Drop the block mask held for reuse, so the next sparse call routes afresh (start of every generation)."""
        if self._sol_prepass is not None and hasattr(self._sol_prepass, "reset_mask_reuse"):
            self._sol_prepass.reset_mask_reuse()

    @property
    def _sol_k_chunk(self) -> int | None:
        if self.sol_mode != "resident" or self._SOL_RESIDENT_K_CHUNK <= 0:
            return None
        return self._SOL_RESIDENT_K_CHUNK

    def _ring_attention(
        self, q_BHNE, k_BHNE, v_BHNE, logical_n, program_config, sol_kwargs: dict, **op_kwargs
    ) -> ttnn.Tensor:
        """One ring-joint SDPA call over the local Q rows (dense, or Sol-Attn with `sol_kwargs`)."""
        if sol_kwargs:
            # The pre-pass indexes the mask's K chunks by its geometry's k_chunk (`sol_prepare`), so the op must use
            # the same one (SOL_RESIDENT_K_CHUNK and the multi-pass per-pass configs can pick another).
            geo_k_chunk = self._sol_prepass.geo.k_chunk
            assert program_config.k_chunk_size == geo_k_chunk, (
                f"Sol-Attn: the op's k_chunk {program_config.k_chunk_size} differs from the pre-pass geometry's "
                f"{geo_k_chunk}; the block mask's K chunks would not line up"
            )
        spatial_BHNE, _prompt, _lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            input_tensor_q=q_BHNE,
            input_tensor_k=k_BHNE,
            input_tensor_v=v_BHNE,
            joint_tensor_q=self.dummy_joint_input,
            joint_tensor_k=self.dummy_joint_input,
            joint_tensor_v=self.dummy_joint_input,
            persistent_output_buffer_k=self.ccl_manager.get_ag_ping_pong_buffer(
                k_BHNE.shape, 2, self.sp_mesh_axis, dtype=k_BHNE.dtype, capacity=self.kv_gather_capacity
            ),
            persistent_output_buffer_v=self.ccl_manager.get_ag_ping_pong_buffer(
                v_BHNE.shape, 2, self.sp_mesh_axis, dtype=v_BHNE.dtype, capacity=self.kv_gather_capacity
            ),
            joint_strategy="rear",
            logical_n=logical_n,
            program_config=lofi.sdpa_program_config("dense_sdpa", program_config),
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
            **sol_kwargs,
            **op_kwargs,
        )
        return spatial_BHNE

    def _sol_resident_multipass(self, q_BHNE, k_BHNE, v_BHNE, logical_n, sol_kwargs: dict) -> ttnn.Tensor:
        """Resident mode in `_SOL_RESIDENT_PASSES` passes over contiguous 64-row-aligned ranges of the local Q rows.

        Pass p attends rows [r0, r1) with rows [c0, c1) of the block mask (one row per 64-row Q chunk), so the op sees a
        shorter local Q and 1/P of the resident chunks per core; K/V and the pooled tensors are the same for every pass.
        A local Q shorter than K is the op's cross-attention shape (`is_cross=True`: non-causal short Q against long
        K/V), which takes the logical length as a host int (`sol_logical_n`), not the tensor. The outputs are
        concatenated along the sequence.
        """
        assert self.sol_logical_n is not None, "multi-pass resident mode needs sol_logical_n (host int)"
        passes = self._SOL_RESIDENT_PASSES
        rows, H, D = q_BHNE.shape[2], q_BHNE.shape[1], q_BHNE.shape[3]
        qc = self._sol_q_chunk
        nqc = -(-rows // qc)
        mask = sol_kwargs["sol_block_mask"]
        outs = []
        for p in range(passes):
            c0, c1 = nqc * p // passes, nqc * (p + 1) // passes
            r0, r1 = c0 * qc, min(rows, c1 * qc)
            assert c1 > c0 and (r1 - r0) % ttnn.TILE_SIZE == 0, (rows, passes, p)
            q_p = ttnn.slice(q_BHNE, [0, 0, r0, 0], [1, H, r1, D])
            mask_p = ttnn.slice(mask, [0, 0, c0, 0], [1, mask.shape[1], c1, mask.shape[3]])
            kw = dict(sol_kwargs, sol_block_mask=mask_p)
            out_p = self._ring_attention(
                q_p,
                k_BHNE,
                v_BHNE,
                int(self.sol_logical_n),
                self._sdpa_program_config(r1 - r0, ring=True, q_chunk=qc, k_chunk=self._sol_k_chunk),
                kw,
                is_cross=True,
            )
            ttnn.deallocate(q_p)
            ttnn.deallocate(mask_p)
            outs.append(out_p)
        out = ttnn.concat(outs, dim=2)
        for t in outs:
            ttnn.deallocate(t)
        return out

    _SPARSE_RING = os.environ.get("MINIMAX_H3_SPARSE_RING", "0") == "1"
    _SPARSE_RING_DTYPE = os.environ.get("MINIMAX_H3_SPARSE_RING_DTYPE", "bf16")
    _BF8_NATIVE_QK = os.environ.get("MINIMAX_H3_BF8_NATIVE_QK", "0") == "1"
    _BF8_V_BEFORE_SPLIT = os.environ.get("MINIMAX_H3_BF8_V_BEFORE_SPLIT", "0") == "1"
    _BF8_POOLED_V = os.environ.get("MINIMAX_H3_BF8_POOLED_V", "0") == "1"
    # Prefix spread step 2: pass the `fused` dense Q blocks to the sparse op, whose dynamic
    # split then balances them by cost (sparse_dyn_split.hpp weighted_range). Off: the op's equal counts.
    _PREFIX_WEIGHTS = os.environ.get("MINIMAX_H3_SOL_PREFIX_WEIGHTS", "0") == "1"
    _DYN_SPLIT = os.environ.get("TT_SPARSE_RING_DYN_SPLIT", "") == "on"

    def _bf8_native_path(self, rows: int) -> bool:
        """Whether this call takes the 8-bit sparse ring path with native BFP8 q / k / v."""
        return (
            self._BF8_NATIVE_QK
            and self._SPARSE_RING
            and self._SPARSE_RING_DTYPE == "bf8"
            and self.use_ring
            and self.sol_attn
            and self.sol_active
            and self.sol_mode == "resident"
            and self.sol_prefix_mode != "separate"
            and self.sol_fused
            and self._exp_sdpa_program_config(rows) is None
        )

    def _s1_token_major_out(self):
        """MINIMAX_H3_MK_S1_TOKEN_MAJOR_O: None (off), or the output dtype for a token-major sparse-op output."""
        if _S1_TOKEN_MAJOR_O == "0" or self.sol_prefix_mode == "separate":
            return None  # the separate prefix pass rewrites rows of the head-major output
        bf8_consumer = (
            _S1_TOKEN_MAJOR_O in ("1", "bf8")
            and getattr(self.to_out, "activation_dtype", None) == ttnn.bfloat8_b
        )
        return ttnn.bfloat8_b if bf8_consumer else ttnn.bfloat16

    def _sparse_ring_attention(
        self, q_BHNE, k_BHNE, v_BHNE, sol_kwargs: dict, token_major_dtype=None
    ) -> ttnn.Tensor:
        """Resident-mode Sol-Attn through the sparse ring op; returns [1, H, N_local, D] like `_ring_attention`, or
        [1, 1, N_local, H * D] in `token_major_dtype` when that is given."""
        assert self.sol_logical_n is not None, "the sparse ring op needs sol_logical_n (host int)"
        if self._SPARSE_RING_DTYPE not in ("bf16", "bf8"):
            raise ValueError(f"MINIMAX_H3_SPARSE_RING_DTYPE must be bf16 or bf8, got {self._SPARSE_RING_DTYPE!r}")
        q, k, v = q_BHNE, k_BHNE, v_BHNE
        pk, pv = sol_kwargs["sol_pooled_k"], sol_kwargs["sol_pooled_v"]
        cast = []
        if self._SPARSE_RING_DTYPE == "bf8":
            ins = (q, k, v, pk, pv)
            outs8 = [t if t.dtype == ttnn.bfloat8_b else ttnn.typecast(t, ttnn.bfloat8_b) for t in ins]
            q, k, v, pk, pv = outs8
            cast = [o for o, i in zip(outs8, ins) if o is not i]
        program_config = self._sdpa_program_config(
            q.shape[2], ring=True, q_chunk=self._sol_q_chunk, k_chunk=self._sol_k_chunk
        )
        assert program_config.q_chunk_size == SOL_BLOCK, (program_config.q_chunk_size, SOL_BLOCK)
        program_config = lofi.sdpa_program_config("sparse_sdpa", program_config)
        out = ttnn.transformer.sparse_ring_sdpa(
            q,
            k,
            v,
            sol_kwargs["sol_block_mask"],
            pk,
            pv,
            persistent_output_buffer_k=self.ccl_manager.get_ag_ping_pong_buffer(
                k.shape, 2, self.sp_mesh_axis, dtype=k.dtype, capacity=self.kv_gather_capacity
            ),
            persistent_output_buffer_v=self.ccl_manager.get_ag_ping_pong_buffer(
                v.shape, 2, self.sp_mesh_axis, dtype=v.dtype, capacity=self.kv_gather_capacity
            ),
            logical_n=int(self.sol_logical_n),
            program_config=program_config,
            scale=self.head_dim**-0.5,
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
            block_size=SOL_BLOCK,
            **({"head_weights": sol_kwargs["sol_head_weights"]} if "sol_head_weights" in sol_kwargs else {}),
            **({"dense_q_blocks": dense} if (dense := self._sol_dense_q_blocks(q.shape[2])) else {}),
            **(
                {}
                if token_major_dtype is None
                else {"output_token_major": True, "output_dtype": token_major_dtype}
            ),
        )
        for t in cast:
            ttnn.deallocate(t)
        if token_major_dtype is not None:
            return out  # already the consumer's layout and dtype
        want = ttnn.bfloat16 if q_BHNE.dtype == ttnn.bfloat8_b else q_BHNE.dtype
        if out.dtype != want:
            out_bf16 = ttnn.typecast(out, want)
            ttnn.deallocate(out)
            out = out_bf16
        return out

    def row_parallel_gate_fusable(self, rows: int) -> bool:
        """Whether to_out will run the fused row-parallel MM + RS + addcmul for `rows` padded rows, i.e. whether a
        tile-row-mapped gate (MINIMAX_H3_MK_S4_GATE_TILEROW) can be used: only that op reads the gate through a map."""
        if not self.to_out_row_parallel:
            return False
        k = self.inner_dim // self.parallel_config.tensor_parallel.factor
        core_grid = self.mesh_device.compute_with_storage_grid_size()
        return has_row_parallel_mmrs_config(rows, k, self.hidden_size, core_grid)

    def _to_out_row_parallel(
        self, spatial_1BND, addcmul_residual, addcmul_gate, addcmul_gate_tile_row_map=None, addcmul_gate_col_offset=0
    ) -> ttnn.Tensor:
        """Local heads [1, 1, N, inner / tp] @ row shard [inner / tp, hidden] -> TP reduce-scatter on hidden.

        With the gated residual and a resolved blocking (mmrs_config) it is one fused op computing
        `residual + rs(partials) * gate`, like ff2; otherwise the unfused matmul + reduce-scatter (+ addcmul).
        The reduce-scatter output shares ff2's ping-pong buffer pair (same shape); to_out and ff2 alternate within a
        block, so a block's input, its to_out output and its ff2 output never share a buffer.
        """
        x = maybe_cast_activation(spatial_1BND, self.to_out.activation_dtype)
        if addcmul_residual is not None:
            m, k = x.padded_shape[-2], x.padded_shape[-1]
            core_grid = self.mesh_device.compute_with_storage_grid_size()
            if has_row_parallel_mmrs_config(m, k, self.hidden_size, core_grid):
                map_kwargs = (
                    {} if addcmul_gate_tile_row_map is None else {"addcmul_b_tile_row_map": addcmul_gate_tile_row_map}
                )
                if addcmul_gate_col_offset:
                    map_kwargs["addcmul_b_col_offset_tiles"] = addcmul_gate_col_offset
                return self.to_out.forward_fused_addcmul(
                    x,
                    addcmul_residual,
                    addcmul_gate,
                    compute_kernel_config=lofi.compute_config("to_out", self.mm_compute_kernel_config),
                    dtype=ttnn.bfloat16,
                    use_persistent_buffer=self.use_persistent_ccl_buffers,
                    mmrs_config=mmrs_blocking_override(m, k, self.hidden_size, core_grid),
                    **map_kwargs,
                )
        if addcmul_gate_tile_row_map is not None:
            # The gate is a tile-row table here, not a per-token tensor: only the fused op can read it.
            raise ValueError("a tile-row-mapped gate needs the fused row-parallel to_out (row_parallel_gate_fusable)")
        out = self.to_out(
            x,
            compute_kernel_config=lofi.compute_config("to_out", self.mm_compute_kernel_config),
            dtype=ttnn.bfloat16,
            use_persistent_buffer=self.use_persistent_ccl_buffers,
        )
        if addcmul_residual is not None:
            out = ttnn.addcmul(addcmul_residual, out, addcmul_gate)
        return out

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        spatial_1BND: ttnn.Tensor,
        logical_n: ttnn.Tensor | None = None,
        rope_cos: ttnn.Tensor | None = None,
        rope_sin: ttnn.Tensor | None = None,
        addcmul_residual: ttnn.Tensor | None = None,
        addcmul_gate: ttnn.Tensor | None = None,
        cu_window_seqlens: ttnn.Tensor | None = None,
        addcmul_gate_tile_row_map: ttnn.Tensor | None = None,
        addcmul_gate_col_offset: int = 0,
    ) -> ttnn.Tensor:
        """
        spatial_1BND: fractured hidden_size on TP; fractured N on SP when `is_sequence_parallel`,
            otherwise replicated on SP.
        rope_cos/rope_sin: [1, 1, N_local, rotary_dim], fractured N on SP, replicated on TP. Both
            None skips the rotary embedding entirely, as the token refiner requires.
        logical_n: logical (unfractured) sequence length as a [1, 1, 1, 1] uint32 device tensor;
            `None` on the non-ring path (token refiner).
        addcmul_residual/addcmul_gate: when both are given, the gated residual
            `addcmul_residual + to_out(...) * addcmul_gate` is folded into the to_out matmul's
            epilogue instead of running as separate ops. Both must be TP-fractured like the output.
        cu_window_seqlens: cumulative block-diagonal window boundaries `[0, ..., N]` (1-D int device
            tensor), plain-SDPA (non-ring) path only.
        addcmul_gate_tile_row_map: gate tile-row map (`MINIMAX_H3_MK_S4_GATE_TILEROW`): `addcmul_gate` is the small
            expanded gate table and this is the adaLN tile-row map; only the fused row-parallel to_out takes it
            (`row_parallel_gate_fusable`). `addcmul_gate_col_offset` (MINIMAX_H3_MK_S2L_ONE_GATHER): the gate is the
            column slice of a wider table starting at this tile column.

        Returns the attention output with the same distribution as the input.
        """
        assert (addcmul_residual is None) == (addcmul_gate is None), "addcmul residual/gate come as a pair"
        assert addcmul_gate_tile_row_map is None or addcmul_gate is not None, "a gate map needs the gate table"
        assert (
            cu_window_seqlens is None or not self.use_ring
        ), "cu_window_seqlens is only for the plain-SDPA (non-ring) path"
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
        assert not (self.use_ring and logical_n is None), "ring attention needs the logical sequence length logical_n"

        # Passing parallel_config puts ColParallelLinear on all_gather_minimal_matmul_async: the TP
        # all-gather of the K-fractured input folds into the matmul that consumes it, instead of
        # running as a separate op. Only on ring topologies -- on a line the unfused path is faster,
        # which is the same condition WanAttention uses.
        # Swept block sizes fit the fused op's CBs; the unfused matmul overruns L1 with them at long seq.
        matmul_parallel_config = self.parallel_config if self.use_fused_agmm else None
        if not self.use_fused_agmm and tp_factor > 1:
            spatial_1BND = self.ccl_manager.all_gather(
                spatial_1BND, dim=3, mesh_axis=self.tp_mesh_axis, use_hyperparams=False
            )

        # Head-split output chunks: V (chunk 2) head-split by the matmul writer itself, only where the fused AGMM
        # runs on a [1, 1, N, K] input (the op's own condition); everywhere else V is split below as before.
        e2a = (
            _E2A_HEAD_SPLIT == "1"
            and self.use_fused_agmm
            and len(spatial_1BND.shape) == 4
            and spatial_1BND.shape[0] == 1
            and spatial_1BND.shape[1] == 1
        )
        e2a_kwargs = {"head_split_chunks_mask": 0b100, "head_dim": self.head_dim} if e2a else {}
        q_1BNF, k_1BNF, v_1BNF = self.to_qkv(
            spatial_1BND,
            compute_kernel_config=lofi.compute_config("qkv", self.mm_compute_kernel_config),
            parallel_config=matmul_parallel_config,
            force_transpose=False,
            use_persistent_buffer=self.use_persistent_ccl_buffers,
            **self._agmm_kwargs(self._qkv_pins, self.hidden_size, 3 * self.inner_dim // tp_factor, spatial_1BND),
            **e2a_kwargs,
        )

        def create_heads(inp: ttnn.Tensor) -> ttnn.Tensor:
            out, _, _ = ttnn.experimental.nlp_create_qkv_heads(
                inp,
                num_heads=self.n_local_heads,
                num_kv_heads=0,
                transpose_k_heads=False,
                runtime_rows=self.dynamic_length,
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
        bf8_native = self._bf8_native_path(q_1BNF.shape[2])
        qk_dtype = {"dtype": ttnn.bfloat8_b} if bf8_native else {}
        q_BHNE = self.norm_q(q_1BNF, **norm_kwargs, **qk_dtype)
        k_BHNE = self.norm_k(k_1BNF, **norm_kwargs, **qk_dtype)
        if e2a:
            # Head-split output chunks: V is already [1, H, N, D]; only the BFP8 cast remains (per tile, so
            # cast-after-split == cast-then-split).
            v_BHNE = ttnn.typecast(v_1BNF, ttnn.bfloat8_b) if bf8_native else v_1BNF
            if not _E2A_LOGGED:
                _E2A_LOGGED.append(True)
                print("[MINIMAX_H3_MK_E2A] qkv matmul writes V head-split (no V create_heads)", flush=True)
        elif bf8_native and self._BF8_V_BEFORE_SPLIT:
            v8 = ttnn.typecast(v_1BNF, ttnn.bfloat8_b)
            v_BHNE = create_heads(v8)
            ttnn.deallocate(v8)
        else:
            v_BHNE = create_heads(v_1BNF)
            if bf8_native:
                v8 = ttnn.typecast(v_BHNE, ttnn.bfloat8_b)
                ttnn.deallocate(v_BHNE)
                v_BHNE = v8

        # Sequence is fractured across SP, so attention must gather K/V around the ring.
        # The packed sequence is one attention document and logical_n masks the pad tail, so no mask.
        exp_program_config = self._exp_sdpa_program_config(q_BHNE.shape[2])
        spatial_1BND_token_major = None  # set when the sparse op writes the concat layout
        if exp_program_config is not None:
            spatial_BHNE, _prompt, _lse = ttnn.transformer.exp_ring_joint_scaled_dot_product_attention(
                q_BHNE,
                k_BHNE,
                v_BHNE,
                self.dummy_joint_input,
                self.dummy_joint_input,
                self.dummy_joint_input,
                persistent_output_buffer_k=self.ccl_manager.get_ag_ping_pong_buffer(
                    k_BHNE.shape, 2, self.sp_mesh_axis, dtype=k_BHNE.dtype, capacity=self.kv_gather_capacity
                ),
                persistent_output_buffer_v=self.ccl_manager.get_ag_ping_pong_buffer(
                    v_BHNE.shape, 2, self.sp_mesh_axis, dtype=v_BHNE.dtype, capacity=self.kv_gather_capacity
                ),
                joint_strategy="rear",
                logical_n=logical_n,
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
            sol_kwargs = {}
            if self.sol_attn and self.sol_active:
                sol_kwargs = self._sol_tensors(q_BHNE, k_BHNE, v_BHNE)
            if sol_kwargs and self.sol_mode == "resident" and self._SPARSE_RING:
                token_major_dtype = self._s1_token_major_out()
                spatial_BHNE = self._sparse_ring_attention(
                    q_BHNE, k_BHNE, v_BHNE, sol_kwargs, token_major_dtype=token_major_dtype
                )
                if token_major_dtype is not None:
                    spatial_1BND_token_major = spatial_BHNE
                    _log_s1_once(token_major_dtype)
            elif sol_kwargs and self.sol_mode == "resident" and self._SOL_RESIDENT_PASSES > 1:
                spatial_BHNE = self._sol_resident_multipass(q_BHNE, k_BHNE, v_BHNE, logical_n, sol_kwargs)
            else:
                spatial_BHNE = self._ring_attention(
                    q_BHNE,
                    k_BHNE,
                    v_BHNE,
                    logical_n,
                    (
                        self._sdpa_program_config(
                            q_BHNE.shape[2], ring=True, q_chunk=self._sol_q_chunk, k_chunk=self._sol_k_chunk
                        )
                        if sol_kwargs
                        else self._ring_sdpa_program_config(q_BHNE.shape[2])
                    ),
                    sol_kwargs,
                    # The Sol paths keep the static-length program (the op's dynamic_local_n path has no Sol).
                    dynamic_local_n=self.dynamic_length and not sol_kwargs,
                )
            if sol_kwargs and self.sol_prefix_mode == "separate":
                spatial_BHNE = self._sol_prefix_dense_rows(q_BHNE, k_BHNE, v_BHNE, spatial_BHNE)
        else:
            spatial_BHNE = ttnn.transformer.scaled_dot_product_attention(
                q_BHNE,
                k_BHNE,
                v_BHNE,
                cu_window_seqlens=cu_window_seqlens,
                is_causal=False,
                program_config=self._sdpa_program_config(
                    q_BHNE.shape[2], ring=False, windowed=cu_window_seqlens is not None
                ),
                compute_kernel_config=self.sdpa_compute_kernel_config,
            )
        if spatial_1BND_token_major is not None:
            spatial_1BND = spatial_1BND_token_major  # the sparse op wrote the concatenate_heads layout itself
        else:
            spatial_1BND = ttnn.transformer.concatenate_heads(spatial_BHNE, runtime_rows=self.dynamic_length)
            spatial_1BND = ttnn.unsqueeze(spatial_1BND, 0)

        if self.to_out_row_parallel:
            return self._to_out_row_parallel(
                spatial_1BND,
                addcmul_residual,
                addcmul_gate,
                addcmul_gate_tile_row_map=addcmul_gate_tile_row_map,
                addcmul_gate_col_offset=addcmul_gate_col_offset,
            )
        if addcmul_gate_tile_row_map is not None:
            raise ValueError("a tile-row-mapped gate needs the fused row-parallel to_out (row_parallel_gate_fusable)")

        # Each device holds canonical heads [d * n_local, (d+1) * n_local), so gathering on TP
        # rebuilds the full inner_dim in canonical order for to_out -- fused into the matmul when
        # use_fused_agmm.
        if not self.use_fused_agmm and tp_factor > 1:
            spatial_1BND = self.ccl_manager.all_gather(
                spatial_1BND, dim=3, mesh_axis=self.tp_mesh_axis, use_hyperparams=False
            )

        # The gated residual rides along in the matmul epilogue on the fused path; on the unfused
        # path the op has no addcmul, so apply it afterwards.
        fuse_gate = addcmul_residual is not None and self.use_fused_agmm
        out = self.to_out(
            spatial_1BND,
            compute_kernel_config=lofi.compute_config("to_out", self.mm_compute_kernel_config),
            parallel_config=matmul_parallel_config,
            force_transpose=False,
            addcmul_a=addcmul_residual if fuse_gate else None,
            addcmul_b=addcmul_gate if fuse_gate else None,
            use_persistent_buffer=self.use_persistent_ccl_buffers,
            **self._agmm_kwargs(self._out_pins, self.inner_dim, self.hidden_size // tp_factor, spatial_1BND),
        )
        if addcmul_residual is not None and not fuse_gate:
            out = ttnn.addcmul(addcmul_residual, out, addcmul_gate)
        return out

    def _agmm_kwargs(self, pins: dict | None, k: int, n: int, x: ttnn.Tensor) -> dict:
        """The fused all-gather matmul's blocking kwargs at `x`'s per-device rows (class pin or swept table);
        none on the unfused path, whose matmul overruns L1 with the swept blockings at long sequence lengths."""
        if not self.use_fused_agmm:
            return {}
        return agmm_kwargs(pins, k, n, x.padded_shape[-2])

    def _ring_sdpa_program_config(self, seq_local: int) -> ttnn.SDPAProgramConfig:
        """The per-device length's class pin on the dynamic-length path, else the length-keyed table."""
        if self._ring_sdpa_configs is None:
            return self._sdpa_program_config(seq_local, ring=True)
        return self._ring_sdpa_configs[length_class(seq_local).name]
