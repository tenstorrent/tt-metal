# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``SdpaPrefillSplitK``: causal chunked-prefill attention of one Q chunk over a long paged prefix (the global layers'
``ttnn.transformer.chunked_scaled_dot_product_attention`` calls), with each (Q head, Q chunk)'s K range split over
several cores and merged by log-sum-exp.

Why: the stock op runs one (Q head, 128-row Q chunk) unit per core, each walking the whole prefix, and hands causal
units out in pairs. A served 4096-row chunk of the 31B (4 local Q heads) is 128 units = 64 pairs, so 64 of a
Blackhole chip's 120 cores each walk the prefix twice (probes/sdpa_qlen_bench.py, 2026-11-13: 50.9 ms after a 64K
prefix, the same as 3840 rows). Splitting every unit's K range into ``splits`` pieces gives ``128 * splits`` pieces
that the planner spreads over all cores (longest piece first), each a fraction of the walk.

One ``ttnn.generic_op`` program (``kernels/``): per core a list of pieces (head, Q chunk, K chunks [k_lo, k_hi),
output slot); the compute body per K chunk is the stock ``sdpa_inner_loop`` STANDARD body (compute_common.hpp) with
absolute tile coordinates for the causal diagonal. Outputs per slot ``j`` (``out`` / ``stats`` planes
``j * NQH + h``):

* ``out [1, splits * NQH, Sq, D]`` (q dtype, TILE): the softmax-normalised attention over the piece's keys;
* ``stats [1, splits * NQH, Sq, 64]`` fp32 TILE: column 0 = ``m`` (row max of the raw scores), column 32 = ``l``
  (``sum exp((s - m) * scale)``), so ``lse = scale * m + ln l``.

``merge`` combines the slots in fp32: ``M = max_j lse_j``, ``O = sum_j e^(lse_j - M) O_j / sum_j e^(lse_j - M)``.

Contract (``unsupported_reason`` refuses the rest; the caller keeps the stock op): q ``[1, NQH, Sq, D]`` bf16 TILE
interleaved, ``Sq`` a multiple of ``q_chunk_size``; ``k_cache / v_cache`` paged TILE interleaved (bf16 or bfloat8_b)
with the layer's own geometry (``num_kv_heads``, ``block_size``: the stock ``PagedCacheGeometryOverride`` view);
``page_table [1, max_blocks]`` int32 ROW_MAJOR; ``chunk_start`` a host int, a multiple of both chunk sizes and at
least ``splits`` K chunks so every piece holds keys; causal; ``q_chunk_size == k_chunk_size``.
"""

import functools
import heapq
import struct
from dataclasses import dataclass
from pathlib import Path

import torch

import ttnn
from models.demos.gemma4.custom_ops.grid import compute_grid, sdpa_decode_cb_budget_bytes

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 32
# The planner's cost of a piece's Q read + finalisation in K chunk walks: fitted to one BH chip (64K prefix, 4096
# rows: ~0.053 ms per K chunk + ~0.2 ms per piece).
PIECE_OVERHEAD_CHUNKS = 4
MAX_SPLITS = 8
# The split-K op's piece + merge overhead in units of one (head, q chunk) prefix walk, measured on one BH chip (4 Q
# heads, 4096 rows): 0.47 after a 16K prefix, 0.33 after 64K.
SPLITK_OVERHEAD_UNITS = 0.45
_TILE_BYTES = {"bfloat16": 2048, "bfloat8_b": 1088, "float32": 4096, "int32": 4096}
_DTYPE_NAME = {ttnn.bfloat16: "bfloat16", ttnn.bfloat8_b: "bfloat8_b", ttnn.float32: "float32"}


def _subblock(h, w, dst_size):
    """The stock ``determine_largest_subblock_size`` (sdpa_subblock_utils.hpp)."""
    order = [(2, 4), (4, 2), (1, 8), (8, 1), (1, 7), (7, 1), (2, 3), (3, 2), (1, 6), (6, 1)]
    order += [(1, 5), (5, 1), (2, 2), (1, 4), (4, 1), (1, 3), (3, 1), (1, 2), (2, 1), (1, 1)]
    for sh, sw in order:
        if sh * sw <= dst_size and h % sh == 0 and w % sw == 0:
            return sh, sw
    return 1, 1


def _granularity(count, cap):
    g = min(count, cap)
    while g > 1 and count % g:
        g -= 1
    return g


@dataclass
class Plan:
    splits: int
    cores: list  # per core: list of (head, q_chunk, k_lo, k_hi, slot)
    makespan: int  # planner cost units (K chunk walks)


@functools.lru_cache(maxsize=256)
def plan_pieces(num_heads, q_chunks, chunk_start_chunks, num_cores, splits=None, q_per_k=1, fixed_k=None):
    """Split every (head, q chunk)'s K range into ``splits`` pieces and assign them to ``num_cores`` cores longest
    first. Causal (``fixed_k=None``): q chunk ``qi`` attends K chunks ``[0, chunk_start_chunks + (qi + 1) * q_per_k)``
    (``q_per_k`` K chunks per Q chunk); otherwise every unit attends ``[0, fixed_k)``. ``splits=None`` picks the count
    (at most ``MAX_SPLITS`` and the shortest unit's K chunks) with the least makespan. Cached: a served engine sees
    few distinct (prefix, rows) shapes per chunk size, and the search is host time."""

    def n_of(qi):
        return fixed_k if fixed_k is not None else chunk_start_chunks + (qi + 1) * q_per_k

    def assign(s):
        pieces = []
        for h in range(num_heads):
            for qi in range(q_chunks):
                n = n_of(qi)
                bounds = [round(j * n / s) for j in range(s + 1)]
                for j in range(s):
                    pieces.append((bounds[j + 1] - bounds[j], h, qi, bounds[j], bounds[j + 1], j))
        pieces.sort(key=lambda p: -p[0])
        loads = [(0, c) for c in range(num_cores)]
        heapq.heapify(loads)
        cores = [[] for _ in range(num_cores)]
        for size, h, qi, lo, hi, j in pieces:
            load, c = heapq.heappop(loads)
            cores[c].append((h, qi, lo, hi, j))
            heapq.heappush(loads, (load + size + PIECE_OVERHEAD_CHUNKS, c))
        return Plan(s, cores, max(l for l, _ in loads))

    shortest = n_of(0)
    if splits is not None:
        assert 1 <= splits <= shortest, f"splits {splits} > the shortest unit's {shortest} K chunks"
        return assign(splits)
    best = None
    for s in range(1, min(MAX_SPLITS, shortest) + 1):
        p = assign(s)
        if best is None or p.makespan < best.makespan:
            best = p
    return best


def _cb_plan(q_dtype_name, kv, Sq_chunk_t, Sk_chunk_t, DHt, fp32_dest, pt_stick):
    """(name, tiles, dtype) per CB in index order, and the program's L1 bytes. qk_im single-buffered (the diagonal
    stamp re-reserves it in place); the statistics single-buffered (2 * Sq_chunk_t tiles: m, then l)."""
    im = "float32" if fp32_dest else "bfloat16"
    specs = [
        ("q_in", Sq_chunk_t * DHt * 2, q_dtype_name),
        ("k_in", Sk_chunk_t * DHt * 2, kv),
        ("v_in", Sk_chunk_t * DHt * 2, kv),
        ("mask_in", 2, "bfloat16"),
        ("identity_scale_in", 1, "bfloat16"),
        ("col_identity", 1, "bfloat16"),
        ("qk_im", Sq_chunk_t * Sk_chunk_t, im),
        ("out_im_A", Sq_chunk_t * DHt, "bfloat16"),
        ("out_im_B", Sq_chunk_t * DHt, "bfloat16"),
        ("max_A", Sq_chunk_t, "bfloat16"),
        ("max_B", Sq_chunk_t, "bfloat16"),
        ("sum_A", Sq_chunk_t, im),
        ("sum_B", Sq_chunk_t, im),
        ("exp_max_diff", Sq_chunk_t, "bfloat16"),
        ("out", Sq_chunk_t * DHt, "bfloat16"),
        ("stats", 2 * Sq_chunk_t, "float32"),
    ]
    return specs, sum(t * _TILE_BYTES[dt] for _, t, dt in specs) + pt_stick


class SdpaPrefillSplitK:
    @staticmethod
    def pays(num_heads, rows, q_chunk_size, cores, margin=0.9):
        """Whether this op beats the stock chunked SDPA for a call of ``num_heads`` Q heads and ``rows`` Q rows on
        ``cores`` cores, in units of one (head, q chunk) prefix walk. The stock causal schedule hands the units out in
        pairs when the q chunk count is even (``global_q_pair_distribute``), each unit on one core: a makespan of
        ``2 * ceil(pairs / cores)`` units (``ceil(units / cores)`` unpaired). This op spreads the same work over every
        core: ``units / cores`` plus ``SPLITK_OVERHEAD_UNITS``. It pays when that is under ``margin`` of the stock's.
        One BH Galaxy chip, 4 heads x 4096 rows: 2 -> 1.52 units predicted, 50.8 -> 35.5 ms measured at 64K; 8 heads
        on a 130-core chip (31B on P150x4): 2 -> 2.42, the stock op stays."""
        q_chunks = -(-int(rows) // int(q_chunk_size))
        units = int(num_heads) * q_chunks
        if q_chunks % 2 == 0:
            stock = 2 * -(-(units // 2) // int(cores))
        else:
            stock = -(-units // int(cores))
        return units / cores + SPLITK_OVERHEAD_UNITS < margin * stock

    @staticmethod
    def unsupported_reason(
        q,
        k_cache,
        v_cache,
        page_table,
        chunk_start,
        *,
        num_kv_heads,
        block_size,
        q_chunk_size,
        k_chunk_size,
        splits=None,
        causal=True,
        k_chunks=None,
    ):
        if q.dtype != ttnn.bfloat16 or q.layout != ttnn.TILE_LAYOUT or q.is_sharded():
            return "q must be bf16 TILE interleaved"
        if k_cache.dtype not in _DTYPE_NAME or v_cache.dtype != k_cache.dtype:
            return f"cache dtype {k_cache.dtype}"
        if q_chunk_size % TILE or k_chunk_size % TILE or (causal and q_chunk_size % k_chunk_size):
            return f"chunk sizes q {q_chunk_size} / k {k_chunk_size} (tile multiples; causal: k divides q)"
        b, nqh, sq, d = (int(v) for v in q.shape)
        if b != 1 or sq % q_chunk_size or d % TILE:
            return f"q shape {(b, nqh, sq, d)}"
        if nqh % int(num_kv_heads):
            return "q heads not a multiple of kv heads"
        if causal:
            if chunk_start % k_chunk_size or chunk_start % q_chunk_size:
                return f"chunk_start {chunk_start} not a multiple of the chunk sizes"
            if splits is not None and chunk_start // k_chunk_size + q_chunk_size // k_chunk_size < splits:
                return f"chunk_start {chunk_start}: fewer K chunks than {splits} splits"
        elif k_chunks is None or k_chunks < (splits or 1):
            return f"non-causal needs k_chunks >= splits (got {k_chunks})"
        if int(block_size) % TILE:
            return "block size"
        if (
            page_table.dtype != ttnn.int32
            or page_table.layout != ttnn.ROW_MAJOR_LAYOUT
            or int(page_table.shape[0]) != 1
        ):
            return "page table must be [1, max_blocks] int32 ROW_MAJOR"
        _, total = _cb_plan(
            "bfloat16",
            _DTYPE_NAME[k_cache.dtype],
            q_chunk_size // TILE,
            k_chunk_size // TILE,
            d // TILE,
            True,
            int(page_table.shape[1]) * 4,
        )
        budget = sdpa_decode_cb_budget_bytes(q.device().arch())
        if total > budget:
            return f"CBs {total} B > L1 budget {budget} B (page table {int(page_table.shape[1])} columns)"
        return None

    @staticmethod
    def golden(q, k, v, chunk_start, scale=1.0):
        """Torch reference: q [1, NQH, Sq, D], k / v [1, NKH, L, D] (L >= chunk_start + Sq), causal at chunk_start."""
        nqh, sq = q.shape[1], q.shape[2]
        nkh = k.shape[1]
        L = chunk_start + sq
        kk = k[:, :, :L].repeat_interleave(nqh // nkh, dim=1).float()
        vv = v[:, :, :L].repeat_interleave(nqh // nkh, dim=1).float()
        s = torch.einsum("bhqd,bhkd->bhqk", q.float(), kk) * scale
        pos = torch.arange(sq)[:, None] + chunk_start
        s = s.masked_fill(torch.arange(L)[None, :] > pos, float("-inf"))
        return torch.einsum("bhqk,bhkd->bhqd", torch.softmax(s, -1), vv)

    @staticmethod
    def merge(out, stats, splits, num_heads, scale=1.0, grid=None, with_stats=False):
        """Log-sum-exp merge of the ``splits`` slot planes of ``out`` / ``stats`` on device (``kernels/*_merge_*``):
        per output row tile ``w_j = exp((m_j - M) * scale) * l_j`` (``M = max_j m_j``), ``O = sum_j w_j O_j /
        sum_j w_j``, fp32 weights and accumulation. Returns ``[1, num_heads, Sq, D]`` in ``out``'s dtype; with
        ``with_stats`` also the merged statistics ``[1, num_heads, Sq, 64]`` fp32 (column 0 ``M``, column 32 ``L =
        sum_j w_j``: the input convention, so merged results merge again)."""
        device = out.device()
        S, H = int(splits), int(num_heads)
        _, _, sq, d = (int(v) for v in out.shape)
        Sqt, DHt = sq // TILE, d // TILE
        gx, gy = grid if grid is not None else compute_grid(device)
        T = H * Sqt
        ncores = min(gx * gy, T)
        base, extra = divmod(T, ncores)
        merged = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, H, sq, d]), out.dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        od = _DTYPE_NAME[out.dtype]
        specs = [
            ("stats_in", 4 * S, "float32"),
            ("o_in", 2 * DHt, od),
            ("max", 1, "float32"),
            ("w", S, "float32"),
            ("inv", 1, "float32"),
            ("wn", S, "float32"),
            ("acc", DHt, "float32"),
            ("out", 2 * DHt, od),
            ("sum", 1, "float32"),
            ("stats_out", 4, "float32"),
        ]
        cb = {name: i for i, (name, _, _) in enumerate(specs)}
        core_ranges = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(i % gx, i // gx), ttnn.CoreCoord(i % gx, i // gx)) for i in range(ncores)]
        )
        _cb_dtype = {"bfloat16": ttnn.bfloat16, "bfloat8_b": ttnn.bfloat8_b, "float32": ttnn.float32}
        cbs = [
            ttnn.CBDescriptor(
                total_size=t * _TILE_BYTES[dt],
                core_ranges=core_ranges,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=cb[name], data_format=_cb_dtype[dt], page_size=_TILE_BYTES[dt])
                ],
            )
            for name, t, dt in specs
        ]
        scale_bits = struct.unpack("<I", struct.pack("<f", float(scale)))[0]
        reader_ct = [S, H, Sqt, DHt, cb["stats_in"], cb["o_in"]]
        for t in (out, stats):
            reader_ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        merged_stats = (
            ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, H, sq, 64]), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
            )
            if with_stats
            else None
        )
        compute_ct = [S, DHt, scale_bits] + [cb[n] for n in ("stats_in", "o_in", "max", "w", "inv", "wn", "acc", "out")]
        compute_ct += [int(with_stats), cb["sum"], cb["stats_out"]]
        writer_ct = [H, Sqt, DHt, cb["out"], int(with_stats), cb["stats_out"]]
        writer_ct += list(ttnn.TensorAccessorArgs(merged).get_compile_time_args())
        writer_ct += list(ttnn.TensorAccessorArgs(merged_stats if with_stats else merged).get_compile_time_args())
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i in range(ncores):
            x, y = i % gx, i // gx
            start = i * base + min(i, extra)
            count = base + (1 if i < extra else 0)
            reader_rt[x][y] = [out.buffer_address(), stats.buffer_address(), start, count]
            writer_rt[x][y] = [
                merged.buffer_address(),
                start,
                count,
                merged_stats.buffer_address() if with_stats else 0,
            ]
            compute_rt[x][y] = [count]
        dst_size = 4  # fp32 dest
        defines = [
            ("STATS_GRANULARITY", "1"),
            ("SUB_EXP_GRANULARITY", "1"),
            ("DHT_GRANULARITY", str(_granularity(DHt, dst_size))),
            ("REDUCE_GRANULARITY", "1"),
            ("EXP_APPROX_MODE", "0"),
        ]
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "reader_merge_splitk.cpp"),
                core_ranges=core_ranges,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "writer_merge_splitk.cpp"),
                core_ranges=core_ranges,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "compute_merge_splitk.cpp"),
                core_ranges=core_ranges,
                compile_time_args=compute_ct,
                defines=defines,
                runtime_args=compute_rt,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
                ),
            ),
        ]
        io = [out, stats] + ([merged_stats] if with_stats else []) + [merged]
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return (merged, merged_stats) if with_stats else merged

    @classmethod
    def op(
        cls,
        q,
        k_cache,
        v_cache,
        page_table,
        chunk_start,
        *,
        num_kv_heads,
        block_size,
        scale=1.0,
        q_chunk_size=128,
        k_chunk_size=128,
        compute_kernel_config=None,
        exp_approx_mode=False,
        splits=None,
        merge=True,
        merge_stats=False,
        grid=None,
        causal=True,
        k_chunks=None,
    ):
        """Causal (``chunk_start`` = Q row 0's position) or, ``causal=False``, every Q row over K chunks ``[0,
        k_chunks)`` unmasked. Returns the merged ``[1, NQH, Sq, D]`` output (``merge=True``; with ``merge_stats``
        ``(out, stats)``, stats in the input convention) or ``(out, stats, plan)`` unmerged."""
        reason = cls.unsupported_reason(
            q,
            k_cache,
            v_cache,
            page_table,
            chunk_start,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            q_chunk_size=q_chunk_size,
            k_chunk_size=k_chunk_size,
            splits=splits,
            causal=causal,
            k_chunks=k_chunks,
        )
        assert reason is None, f"sdpa_prefill_splitk: {reason}"
        device = q.device()
        _, nqh, sq, d = (int(v) for v in q.shape)
        nkh = int(num_kv_heads)
        DHt = d // TILE
        Sq_chunk_t = q_chunk_size // TILE
        Sk_chunk_t = k_chunk_size // TILE
        gx, gy = grid if grid is not None else compute_grid(device)
        num_cores = gx * gy
        q_per_k = q_chunk_size // k_chunk_size if causal else 1
        plan = plan_pieces(
            nqh,
            sq // q_chunk_size,
            chunk_start // k_chunk_size if causal else 0,
            num_cores,
            splits,
            q_per_k=q_per_k,
            fixed_k=None if causal else int(k_chunks),
        )
        S = plan.splits
        cores = [(i, c) for i, c in enumerate(plan.cores) if c]

        if compute_kernel_config is None:
            fidelity, approx, fp32_dest = ttnn.MathFidelity.HiFi4, False, True
        else:
            fidelity = compute_kernel_config.math_fidelity
            approx = compute_kernel_config.math_approx_mode
            fp32_dest = compute_kernel_config.fp32_dest_acc_en
        dst_size = 4 if fp32_dest else 8
        qk_sh, qk_sw = _subblock(Sq_chunk_t, Sk_chunk_t, dst_size)
        out_sh, out_sw = _subblock(Sq_chunk_t, DHt, dst_size)
        assert causal or chunk_start == 0, "sdpa_prefill_splitk: an unmasked call takes chunk_start 0"
        im = "float32" if fp32_dest else "bfloat16"
        kv = _DTYPE_NAME[k_cache.dtype]

        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, S * nqh, sq, d]), q.dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        stats = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, S * nqh, sq, 64]), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )

        pt_stick = int(page_table.shape[1]) * 4
        cb_specs, total = _cb_plan("bfloat16", kv, Sq_chunk_t, Sk_chunk_t, DHt, fp32_dest, pt_stick)
        cb = {name: i for i, (name, _, _) in enumerate(cb_specs)}
        cb["page_table"] = len(cb_specs)
        budget = sdpa_decode_cb_budget_bytes(device.arch())
        assert total <= budget, f"sdpa_prefill_splitk: CBs {total} B > L1 budget {budget} B"

        core_ranges = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(i % gx, i // gx), ttnn.CoreCoord(i % gx, i // gx)) for i, _ in cores]
        )
        _cb_dtype = {"bfloat16": ttnn.bfloat16, "bfloat8_b": ttnn.bfloat8_b, "float32": ttnn.float32}
        cbs = [
            ttnn.CBDescriptor(
                total_size=t * _TILE_BYTES[dt],
                core_ranges=core_ranges,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=cb[name], data_format=_cb_dtype[dt], page_size=_TILE_BYTES[dt])
                ],
            )
            for name, t, dt in cb_specs
        ]
        cbs.append(
            ttnn.CBDescriptor(
                total_size=pt_stick,
                core_ranges=core_ranges,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=cb["page_table"], data_format=ttnn.int32, page_size=pt_stick)
                ],
            )
        )

        scale_bits = struct.unpack("<I", struct.pack("<f", float(scale)))[0]
        reader_ct = [
            nqh,
            nkh,
            sq // TILE,
            DHt,
            DHt,
            Sq_chunk_t,
            Sk_chunk_t,
            int(block_size) // TILE,
            pt_stick,
            num_cores,
            cb["q_in"],
            cb["k_in"],
            cb["v_in"],
            cb["page_table"],
        ]
        for t in (q, k_cache, v_cache, page_table):
            reader_ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        writer_ct = [
            nqh,
            S,
            sq // TILE,
            DHt,
            Sq_chunk_t,
            0x3F803F80,
            num_cores,
            cb["identity_scale_in"],
            cb["col_identity"],
            cb["mask_in"],
            cb["out"],
            cb["stats"],
        ]
        for t in (out, stats):
            writer_ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        compute_ct = [
            Sq_chunk_t,
            Sk_chunk_t,
            DHt,
            DHt,
            DHt,
            qk_sw,
            qk_sh,
            Sq_chunk_t // qk_sh,
            Sk_chunk_t // qk_sw,
            Sk_chunk_t,
            out_sw,
            out_sh,
            Sq_chunk_t // out_sh,
            DHt // out_sw,
            scale_bits,
        ]
        compute_ct += [
            cb[n]
            for n in (
                "q_in",
                "k_in",
                "v_in",
                "mask_in",
                "identity_scale_in",
                "col_identity",
                "qk_im",
                "out_im_A",
                "out_im_B",
                "max_A",
                "max_B",
                "sum_A",
                "sum_B",
                "exp_max_diff",
                "out",
                "stats",
            )
        ]
        defines = [
            ("STATS_GRANULARITY", str(_granularity(Sq_chunk_t, dst_size))),
            ("SUB_EXP_GRANULARITY", str(_granularity(Sk_chunk_t, dst_size))),
            ("DHT_GRANULARITY", str(_granularity(DHt, dst_size))),
            ("REDUCE_GRANULARITY", str(_granularity(Sq_chunk_t, dst_size // 2))),
            ("EXP_APPROX_MODE", str(int(bool(exp_approx_mode)))),
        ]

        # generic_op caches a program by its compile-time args and then requires the same runtime-arg count per
        # core, so every core's piece list is padded to ``slots`` (the busiest core's count rounded up to 4), which
        # rides in the compile-time args: a plan with more pieces per core is another program, one with as many is
        # a cache hit whatever the chunk start
        slots = -(-max(len(c) for _, c in cores) // 4) * 4
        reader_ct.append(slots)
        writer_ct.append(slots)
        compute_ct.append(slots)
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        # causal: Q row 0's tile; unmasked: past every K tile, so the diagonal stamp never applies
        q_base_tile = chunk_start // TILE if causal else int(k_chunks) * Sk_chunk_t
        for i, pieces in cores:
            x, y = i % gx, i // gx
            r = [
                q.buffer_address(),
                k_cache.buffer_address(),
                v_cache.buffer_address(),
                page_table.buffer_address(),
                len(pieces),
            ]
            w = [out.buffer_address(), stats.buffer_address(), len(pieces)]
            c = [len(pieces), q_base_tile]
            for h, qi, lo, hi, j in pieces:
                r += [h, qi, lo, hi]
                w += [h, qi, j]
                c += [qi, lo, hi]
            pad = slots - len(pieces)
            reader_rt[x][y] = r + [0] * (4 * pad)
            writer_rt[x][y] = w + [0] * (3 * pad)
            compute_rt[x][y] = c + [0] * (3 * pad)

        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "reader_prefill_splitk.cpp"),
                core_ranges=core_ranges,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "writer_prefill_splitk.cpp"),
                core_ranges=core_ranges,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "compute_prefill_splitk.cpp"),
                core_ranges=core_ranges,
                compile_time_args=compute_ct,
                defines=defines,
                runtime_args=compute_rt,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=fidelity, fp32_dest_acc_en=fp32_dest, math_approx_mode=approx
                ),
            ),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        ttnn.generic_op([q, k_cache, v_cache, page_table, stats, out], program)
        if not merge:
            return out, stats, plan
        merged = cls.merge(out, stats, S, nqh, scale=scale, grid=(gx, gy), with_stats=merge_stats)
        out.deallocate(True)
        stats.deallocate(True)
        return merged
