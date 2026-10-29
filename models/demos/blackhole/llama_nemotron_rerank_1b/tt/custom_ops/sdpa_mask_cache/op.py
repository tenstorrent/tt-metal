# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""ttnn's SDPA (``ttnn.transformer.scaled_dot_product_attention``) with a reader that reads a head-broadcast mask
from DRAM once per batch per core instead of once per Q head.

The mask this model passes is [B, 1, S, S]: the same for all 32 Q heads. Stock SDPA's reader reads each mask block
from DRAM again for every Q head (32 x 512 x 512 bfp4: 151 MB of mask reads vs 34 MB of K / V). The reader here
(``kernels/reader_mask_cache.cpp``, ttnn's ``reader_interleaved.cpp`` with the cache added) keeps
one batch's mask tiles (Sqt x Skt) in an extra L1 CB: a (Q chunk, K chunk) block is read from DRAM on its first use
for the batch and copied L1 -> mask CB for the other heads. A core runs several heads of one batch in a row (global Q
order: batch, head, Q chunk), so most blocks come from L1. Compute and writer are ttnn's kernels, unchanged.

This module is the host side: a Python port of ``SDPAOperation::SDPAProgramFactory::create_descriptor``
(ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp) and the helpers it calls, restricted to
the configuration this model runs (non-causal, scale 1, head-broadcast bfp4 mask with a block map, streaming compute,
concatenated-heads output, one K chunk). Branches that configuration cannot reach raise instead of being ported.
The C++ names are kept so each block can be compared with its source.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field

import ttnn
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.custom_ops._common import TILE_BYTES

_HERE = os.path.dirname(os.path.abspath(__file__))
READER_KERNEL = os.path.join(_HERE, "kernels", "reader_mask_cache.cpp")
_SDPA_KERNELS = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels"
WRITER_KERNEL = f"{_SDPA_KERNELS}/dataflow/writer_interleaved.cpp"
COMPUTE_KERNEL = f"{_SDPA_KERNELS}/compute/sdpa.cpp"

TILE = 32
INACTIVE = 0xFFFFFFFF  # sdpa_cb::CBIds::inactive
UINT32_MAX = 0xFFFFFFFF
INVALID, VALID = 0, 1  # hostdevcommon/common_values.hpp
# The mask cache is allocated only up to this size (it shares L1 with SDPA's other CBs and the model's resident
# residual); the reader's valid bits are one uint32, so at most 32 (Q chunk, K chunk) blocks.
MASK_CACHE_MAX_BYTES = 160 * 1024
# build_kv_chains: multicast only pays on chains of at least 4 cores.
MIN_MCAST_CHAIN_CORES = 4
# create_descriptor: K/V retention with more batches than this skips the chains.
MAX_RETAIN_CHAIN_BATCH = 8


# ---- helpers from the anonymous namespace of sdpa_program_factory.cpp / sdpa_subblock_utils.hpp ----


def determine_largest_subblock_size(block_height, block_width, dst_size, max_subblock_h=UINT32_MAX):
    subblocks = [
        (2, 4), (4, 2), (1, 8), (8, 1), (1, 7), (7, 1), (2, 3), (3, 2), (1, 6), (6, 1),
        (1, 5), (5, 1), (2, 2), (1, 4), (4, 1), (1, 3), (3, 1), (1, 2), (2, 1), (1, 1),
    ]  # fmt: skip
    for h, w in subblocks:
        if h * w > dst_size or h > max_subblock_h:
            continue
        if block_height % h or block_width % w:
            continue
        return h, w
    return 1, 1


def find_valid_granularity(tile_count, max_granularity):
    g = min(tile_count, max_granularity)
    while g > 1 and tile_count % g:
        g -= 1
    return g


def compute_dht_granularity(DHt, vDHt, dst_size):
    g = min(DHt, vDHt, dst_size)
    while g > 1 and (DHt % g or vDHt % g):
        g -= 1
    return g


def streaming_cb_out_tiles(out_out_subblock_h, out_out_subblock_w, dst_size, Sq_chunk_t, vDHt):
    # sdpa_streaming_qktv.hpp streaming_qktv_h
    if out_out_subblock_h == 1 and 2 * out_out_subblock_w <= dst_size and Sq_chunk_t >= 2:
        qktv_h = 2
    else:
        qktv_h = out_out_subblock_h
    return 2 * qktv_h * vDHt


def kv_chain_key_q_heads(nqh, nkh, nvh):
    q_per_k, q_per_v = nqh // nkh, nqh // nvh
    lo, hi = min(q_per_k, q_per_v), max(q_per_k, q_per_v)
    return lo if hi % lo == 0 else 1


def kv_retain_group(eligible, k_num_chunks, nqh, nkh, nvh):
    return kv_chain_key_q_heads(nqh, nkh, nvh) if eligible and k_num_chunks == 1 else 0


def global_q_range(i, first_extra_core, cores_doing_extra, base_chunks_per_core, extra_chunks_per_core, total):
    extras_before = min(i - first_extra_core if i > first_extra_core else 0, cores_doing_extra)
    start = i * base_chunks_per_core + extras_before * extra_chunks_per_core
    count = base_chunks_per_core
    if first_extra_core <= i < first_extra_core + cores_doing_extra:
        count += extra_chunks_per_core
    if start >= total:
        start, count = total, 0
    elif start + count > total:
        count = total - start
    return start, count


@dataclass
class CoreChainInfo:
    participates: bool = False
    is_injector: bool = False
    is_sink: bool = False
    batch: int = 0
    head: int = 0
    q_chunk_count: int = 0
    prev_physical: tuple = (0, 0)
    next_physical: tuple = (0, 0)
    next_core_q_chunks: int = 0
    mcast_num_dests: int = 0
    mcast_sender_wait: int = 0


@dataclass
class _HeadWork:
    batch: int
    head: int  # the segment key
    q_chunk_count: int
    global_start: int


@dataclass
class _CoreWork:
    physical: tuple
    global_q_start: int = 0
    global_q_count: int = 0
    head_work: list = field(default_factory=list)


def _rotate_injector_to_spread_dram(chain_order, segments, core_work, injector_phys_x):
    best_pos, best_dist = 0, 0
    for pos, idx in enumerate(chain_order):
        phys_x = core_work[segments[idx][0]].physical[0]
        min_dist = UINT32_MAX
        for ix in injector_phys_x:
            min_dist = min(min_dist, abs(phys_x - ix))
        if min_dist > best_dist:
            best_dist, best_pos = min_dist, pos
    if best_pos != 0:
        chain_order[0], chain_order[best_pos] = chain_order[best_pos], chain_order[0]


def build_kv_chains(device, grid_x, grid_y, B, NQH, key_q_heads, q_num_chunks, k_num_chunks, retain_kv, q_range):
    """build_kv_chains for non-causal calls: returns (core_chain_info per core, mcast_chains).

    A segment is a core's run of Q chunks of one chain key (batch, nq / key_q_heads). Keys whose Q chunks span
    several cores get a chain: the injector reads K / V from DRAM and forwards it along the chain (unicast), or
    multicasts it when every chain has >= 4 cores on one row with no other worker between them and equal Q counts."""
    num_cores = grid_x * grid_y
    segment_q_heads = key_q_heads  # non-causal: the segment spans the chain key
    num_keys = NQH // segment_q_heads
    key_q_chunks = segment_q_heads * q_num_chunks
    total_keys = B * num_keys
    core_work = []
    info = [CoreChainInfo() for _ in range(num_cores)]
    head_segments = [[] for _ in range(total_keys)]  # (core_idx, head_work_index)

    # First pass: each core's [g_start, g_start + g_count) split into (batch, key) segments.
    for i in range(num_cores):
        p = device.worker_core_from_logical_core(ttnn.CoreCoord(i % grid_x, i // grid_x))
        work = _CoreWork(physical=(p.x, p.y))
        core_work.append(work)
        g_start, g_count = q_range(i)
        work.global_q_start, work.global_q_count = g_start, g_count
        cursor, g_end = g_start, g_start + g_count
        while cursor < g_end:
            nb = cursor // (NQH * q_num_chunks)
            key = (cursor // key_q_chunks) % num_keys
            span = min(key_q_chunks - cursor % key_q_chunks, g_end - cursor)
            if span:
                work.head_work.append(_HeadWork(nb, key, span, cursor))
                head_id = nb * num_keys + key
                if head_id < len(head_segments):
                    head_segments[head_id].append((i, len(work.head_work) - 1))
            cursor += span

    def hw_of(seg):
        return core_work[seg[0]].head_work[seg[1]]

    # Second pass: a chain per key spanning several cores.
    injector_phys_x = []
    for head_id, segments in enumerate(head_segments):
        if len(segments) < 2:
            continue
        chain_start_idx = None
        for idx in range(len(segments) - 1):
            ci, hwi = segments[idx]
            if hwi >= len(core_work[ci].head_work) or info[ci].participates:
                continue
            if len(core_work[ci].head_work) == 1:
                chain_start_idx = idx
                break
        if chain_start_idx is None:
            for idx in range(len(segments) - 1):
                if not info[segments[idx][0]].participates:
                    chain_start_idx = idx
                    break
        if chain_start_idx is None:
            continue
        chain_order = []
        for step in range(len(segments)):
            idx = (chain_start_idx + step) % len(segments)
            ci, hwi = segments[idx]
            if ci >= len(core_work) or hwi >= len(core_work[ci].head_work):
                continue
            if info[ci].participates:
                break
            chain_order.append(idx)

        # Every chain core must start the key in the same slot of the two-chunk K / V CBs.
        def start_slot(idx):
            ci, hwi = segments[idx]
            if retain_kv:
                return hwi % 2
            work = core_work[ci]
            return ((work.head_work[hwi].global_start - work.global_q_start) * k_num_chunks) % 2

        chain_slot = start_slot(chain_order[0])
        chain_order = [idx for idx in chain_order if start_slot(idx) == chain_slot]
        if len(chain_order) < 2:
            continue

        ref_q = hw_of(segments[chain_order[0]]).q_chunk_count
        uniform_q = all(hw_of(segments[idx]).q_chunk_count == ref_q for idx in chain_order[1:])
        if uniform_q:
            _rotate_injector_to_spread_dram(chain_order, segments, core_work, injector_phys_x)
        else:
            chain_order.sort(key=lambda idx: -hw_of(segments[idx]).q_chunk_count)  # stable, descending
        injector_phys_x.append(core_work[segments[chain_order[0]][0]].physical[0])

        for pos, idx in enumerate(chain_order):
            ci = segments[idx][0]
            hw = hw_of(segments[idx])
            c = info[ci]
            c.participates = True
            c.batch, c.head, c.q_chunk_count = hw.batch, hw.head, hw.q_chunk_count
            if pos == 0:
                c.is_injector = True
            if pos == len(chain_order) - 1:
                c.is_sink = True
            if pos > 0:
                c.prev_physical = core_work[segments[chain_order[pos - 1]][0]].physical
            if pos + 1 < len(chain_order):
                nxt = segments[chain_order[pos + 1]]
                if nxt[0] < len(core_work) and nxt[1] < len(core_work[nxt[0]].head_work):
                    c.next_physical = core_work[nxt[0]].physical
                    c.next_core_q_chunks = hw_of(nxt).q_chunk_count

    # Third pass: multicast, all or nothing.
    candidates = []
    all_eligible = True
    for head_id, segments in enumerate(head_segments):
        if len(segments) < 2:
            continue
        chain_cores = [
            ci
            for ci, _ in segments
            if ci < len(info)
            and info[ci].participates
            and info[ci].batch == head_id // num_keys
            and info[ci].head == head_id % num_keys
        ]
        if len(chain_cores) < 2:
            continue
        ref_y = core_work[chain_cores[0]].physical[1]
        if any(core_work[ci].physical[1] != ref_y for ci in chain_cores[1:]):
            all_eligible = False
            break
        xs = [core_work[ci].physical[0] for ci in chain_cores]
        min_x, max_x = min(xs), max(xs)
        has_gap = False
        for ci, _ in segments:
            if ci >= len(core_work):
                continue
            px, py = core_work[ci].physical
            if py == ref_y and min_x <= px <= max_x and ci not in chain_cores:
                has_gap = True
                break
        if has_gap:
            all_eligible = False
            break
        ref_q_chunks = info[chain_cores[0]].q_chunk_count
        if any(info[ci].q_chunk_count != ref_q_chunks for ci in chain_cores[1:]):
            all_eligible = False
            break
        candidates.append((chain_cores, ref_q_chunks))

    if any(len(cores) < MIN_MCAST_CHAIN_CORES for cores, _ in candidates):
        all_eligible = False
    mcast_chains = 0
    if all_eligible and candidates:
        mcast_chains = len(candidates)
        for cores, ref_q_chunks in candidates:
            num_receivers = len(cores) - 1
            injector = next((ci for ci in cores if info[ci].is_injector), cores[0])
            xs = [core_work[ci].physical[0] for ci in cores]
            y = core_work[injector].physical[1]
            inj = info[injector]
            inj.prev_physical = (min(xs), y)  # multicast rectangle start
            inj.next_physical = (max(xs), y)  # multicast rectangle end
            inj.mcast_num_dests = num_receivers
            inj.mcast_sender_wait = num_receivers
            inj.next_core_q_chunks = ref_q_chunks
            for ci in cores:
                if ci == injector:
                    continue
                r = info[ci]
                r.prev_physical = core_work[injector].physical
                r.next_physical = (0, 0)
                r.next_core_q_chunks = 0
                r.is_sink = True
    return info, mcast_chains


# ---- the op ----


def _accessor(t):
    """TensorAccessorArgs(buffer).append_to: [args config, aligned page size]; nullptr (absent tensor) -> [0, 0]."""
    if t is None:
        return [0, 0]
    args = ttnn.TensorAccessorArgs(t).get_compile_time_args()
    if len(args) != 2:
        raise ValueError("only interleaved tensors are supported (sharded accessors take more args)")
    return list(args)


def _float_bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def sdpa_mask_cache(
    q,
    k,
    v,
    *,
    attn_mask,
    attn_mask_block_map,
    program_config,
    compute_kernel_config,
    is_causal=False,
    scale=1.0,
    output_concat_heads=True,
    memory_config=None,
):
    """``ttnn.transformer.scaled_dot_product_attention`` for this model's masked call, with the mask cache.

    q [B, NQH, S, D] bf16 TILE, k / v [B, NKH, S, D] bf16 TILE (interleaved, L1 or DRAM); attn_mask [B, 1, S, S]
    bfp4 TILE in DRAM (added unscaled: scale must be 1); attn_mask_block_map [B, 1, S / q_chunk, S / k_chunk] int32
    ROW_MAJOR in DRAM. Returns [B, 1, S, NQH * D] (concatenated heads), bf16 TILE in ``memory_config`` (default DRAM).
    Bit-identical to the stock op on the same inputs; raises on any configuration outside the one ported."""
    # ---- the configuration this port covers ----
    if is_causal:
        raise NotImplementedError("is_causal=True is not ported")
    if scale != 1.0:
        # ttnn's wrapper multiplies the mask by 1 / scale first; not ported.
        raise NotImplementedError("scale must be 1.0")
    if not output_concat_heads:
        raise NotImplementedError("output_concat_heads=False is not ported")
    if attn_mask is None or attn_mask_block_map is None:
        raise NotImplementedError("attn_mask and attn_mask_block_map are required")
    if program_config is None:
        raise NotImplementedError("program_config is required")
    if program_config.sub_core_grids is not None:
        raise NotImplementedError("sub_core_grids is not ported")
    if program_config.k_splits != 1 or program_config.segmented_accumulation:
        raise NotImplementedError("k_splits / segmented_accumulation are ring-joint only; expected 1 / False")
    if program_config.max_cores_per_head_batch != 16:
        raise NotImplementedError("max_cores_per_head_batch must be 16 (the default; regular SDPA ignores it)")
    ckc = compute_kernel_config
    if ckc.fp32_dest_acc_en:
        raise NotImplementedError("fp32_dest_acc_en=True selects the legacy compute kernel; not ported")
    if ckc.math_fidelity != ttnn.MathFidelity.HiFi4 or ckc.math_approx_mode or ckc.packer_l1_acc:
        raise NotImplementedError("compute config must be HiFi4, math_approx_mode False, packer_l1_acc False")
    for name, t in (("q", q), ("k", k), ("v", v)):
        if t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT or t.is_sharded():
            raise NotImplementedError(f"{name} must be bf16 TILE interleaved")
    if attn_mask.dtype != ttnn.bfloat4_b or attn_mask.layout != ttnn.TILE_LAYOUT:
        raise NotImplementedError("attn_mask must be bfp4 TILE")
    if attn_mask.memory_config().buffer_type != ttnn.BufferType.DRAM or attn_mask.is_sharded():
        raise ValueError("attn_mask must be DRAM interleaved")
    if attn_mask_block_map.dtype != ttnn.int32 or attn_mask_block_map.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("attn_mask_block_map must be int32 ROW_MAJOR")
    if attn_mask_block_map.memory_config().buffer_type != ttnn.BufferType.DRAM:
        raise ValueError("attn_mask_block_map must be in DRAM")
    memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
    if memory_config.is_sharded():
        raise NotImplementedError("sharded output is not ported")

    device = q.device()
    q_shape, k_shape, v_shape = list(q.shape), list(k.shape), list(v.shape)
    B, NQH, Sq, DH = q_shape
    NKH, NVH = k_shape[1], v_shape[1]  # resolve_effective_kv_geometry: no paged override
    Sk = k_shape[2]
    if k_shape[0] != B or v_shape[0] != B or v_shape[2] != Sk or k_shape[3] != DH or v_shape[3] != DH:
        raise ValueError(f"Q / K / V shapes disagree: {q_shape} {k_shape} {v_shape}")
    if NKH != NVH or NQH % NKH or NQH < NKH:
        raise ValueError(f"heads: Q {NQH}, K {NKH}, V {NVH}")
    if list(q.padded_shape) != q_shape or list(k.padded_shape) != k_shape or list(v.padded_shape) != v_shape:
        raise NotImplementedError("padded Q / K / V are not ported")
    mask_shape = list(attn_mask.shape)
    if mask_shape != [B, 1, Sq, Sk]:
        # [1, 1, S, S] (batch-broadcast) and per-head masks are not ported.
        raise NotImplementedError(f"attn_mask must be [{B}, 1, {Sq}, {Sk}], got {mask_shape}")

    q_chunk_size, k_chunk_size = int(program_config.q_chunk_size), int(program_config.k_chunk_size)
    if q_chunk_size % TILE or k_chunk_size % TILE:
        raise ValueError("chunk sizes must be multiples of 32")
    if B not in (8, 32) or Sq not in (128, 256, 512) or Sk != Sq or q_chunk_size != 128 or k_chunk_size != Sk:
        # The shapes this port was checked on (perf_tools/probe_sdpa_mask_cache.py). Others reach C++ branches
        # not ported here (several K chunks: no K/V retention, the unretained chain key; B 1: batch-broadcast mask).
        raise NotImplementedError(f"B {B}, S {Sq}x{Sk}, chunks q{q_chunk_size} k{k_chunk_size} not ported")
    map_shape = list(attn_mask_block_map.shape)
    if map_shape != [B, 1, -(-Sq // q_chunk_size), -(-Sk // k_chunk_size)]:
        raise ValueError(f"attn_mask_block_map shape {map_shape} does not match the mask and chunk sizes")

    # ---- create_descriptor ----
    # Q-head crossing (gqa_crossing_factor) is off with a mask: gqa_pack = 1.
    padded_Sq = -(-Sq // q_chunk_size) * q_chunk_size
    padded_Sk = -(-Sk // k_chunk_size) * k_chunk_size
    if padded_Sq != Sq or padded_Sk != Sk:
        raise NotImplementedError("chunk sizes must divide the sequence lengths (padded mask path not ported)")
    Sqt, Skt = padded_Sq // TILE, padded_Sk // TILE
    DHt = DH // TILE
    vDHt = DHt
    valid_Sqt, valid_Skt = -(-Sq // TILE), -(-Sk // TILE)
    use_padded_mask = False
    Sq_chunk_t, Sk_chunk_t = q_chunk_size // TILE, k_chunk_size // TILE
    q_num_chunks, k_num_chunks = padded_Sq // q_chunk_size, padded_Sk // k_chunk_size
    use_provided_mask = True
    broadcast_provided_mask_batch = mask_shape[0] == 1  # False here
    broadcast_provided_mask_heads = mask_shape[1] == 1  # True here
    compute_use_provided_mask = True
    generated_padding_mask = use_padded_mask
    use_mask_block_map = True
    use_k_range_ctrl = True

    fp32_dest_acc_en = bool(ckc.fp32_dest_acc_en)
    dst_full_sync_en = bool(ckc.dst_full_sync_en)

    grid = program_config.compute_with_storage_grid_size
    grid_x, grid_y = int(grid.x), int(grid.y)
    num_cores = grid_x * grid_y
    dev_grid = device.compute_with_storage_grid_size()
    if num_cores == 0 or num_cores > dev_grid.x * dev_grid.y:
        raise ValueError(f"grid {grid_x}x{grid_y} does not fit the device")
    exp_approx_mode = True if program_config.exp_approx_mode is None else bool(program_config.exp_approx_mode)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))])

    # Global Q scheduling: the flat B * NQH * q_num_chunks space split evenly; no pair distribution (not causal).
    total_q_chunks = B * NQH * q_num_chunks
    base_chunks_per_core = total_q_chunks // num_cores
    cores_doing_extra = total_q_chunks % num_cores
    extra_chunks_per_core = 1
    max_chunks_per_core = base_chunks_per_core + (extra_chunks_per_core if cores_doing_extra else 0)
    q_buffer_factor = 2 if max_chunks_per_core > 1 else 1

    dst_size = 4 if fp32_dest_acc_en else 8
    qk_in0_block_w = DHt
    qk_out_subblock_h, qk_out_subblock_w = determine_largest_subblock_size(Sq_chunk_t, Sk_chunk_t, dst_size)
    use_streaming_compute = not fp32_dest_acc_en

    # Lightweight masks are off with a provided mask; kv_retain: one K chunk keeps K / V across a key's Q chunks.
    lightweight_mask = False
    k_partial_col = 0
    kv_retain = kv_retain_group(use_streaming_compute, k_num_chunks, NQH, NKH, NVH)
    # kv_chain_mode_for: a plain K/V stream, not causal -> 1 (the non-causal lock-step chain).
    kv_chain_mode = 1
    # remainder_goes_to_last_cores is causal only: the extra chunks go to the first cores.
    first_extra_core = 0

    def q_range(i):
        return global_q_range(
            i, first_extra_core, cores_doing_extra, base_chunks_per_core, extra_chunks_per_core, total_q_chunks
        )

    kv_chain_semaphores = True  # not causal
    mask_tiles = Sq_chunk_t * Sk_chunk_t * 2  # double buffer
    kv_slots = 2  # kv_chain_slots: 3 only for causal chain mode 3
    q_tiles = Sq_chunk_t * DHt * q_buffer_factor
    k_tiles = Sk_chunk_t * DHt * kv_slots
    v_tiles = Sk_chunk_t * vDHt * kv_slots
    qk_tiles = Sq_chunk_t * Sk_chunk_t
    out_im_tiles = Sq_chunk_t * vDHt
    scale_tiles = 1
    statistics_tiles = Sq_chunk_t

    qk_in0_num_subblocks = Sq_chunk_t // qk_out_subblock_h
    qk_in1_num_subblocks = Sk_chunk_t // qk_out_subblock_w
    out_in0_block_w = Sk_chunk_t
    out_out_subblock_h, out_out_subblock_w = determine_largest_subblock_size(Sq_chunk_t, vDHt, dst_size, 2)
    out_in0_num_subblocks = Sq_chunk_t // out_out_subblock_h
    out_in1_num_subblocks = vDHt // out_out_subblock_w
    out0_t = streaming_cb_out_tiles(out_out_subblock_h, out_out_subblock_w, dst_size, Sq_chunk_t, vDHt)
    assert Sq_chunk_t % out_out_subblock_h == 0

    stats_granularity = find_valid_granularity(Sq_chunk_t, dst_size)
    sub_exp_granularity = find_valid_granularity(Sk_chunk_t, dst_size)
    dht_granularity = compute_dht_granularity(DHt, vDHt, dst_size)
    reduce_granularity = find_valid_granularity(Sq_chunk_t, dst_size // 2)

    bf16_one = 0x3F80
    packed_identity_scalar = (bf16_one << 16) | bf16_one
    scale_packed = _float_bits(scale)
    use_zigzag_balancing = False
    windowed_mode = 0  # WindowedMode::None

    reader_ct = [
        B, NQH, NKH, NVH, Skt, valid_Sqt, valid_Skt, DHt, vDHt, Sq_chunk_t, q_num_chunks, Sk_chunk_t, k_num_chunks,
        num_cores,
        0,  # use_causal_kernel
        int(use_provided_mask),
        int(broadcast_provided_mask_batch),
        int(broadcast_provided_mask_heads),
        int(generated_padding_mask),
        0,  # is_chunked
        0,  # block_size_t
        0,  # page_table_stick_size
        0,  # use_attention_sink
        0,  # use_mla
        0,  # mla_kv_overlap
        qk_out_subblock_h,
        0,  # sliding_window_size
        int(use_streaming_compute),
    ]  # fmt: skip
    sem_args_offset = len(reader_ct)
    sender_semaphore_id, receiver_semaphore_id, valid_semaphore_id = 0, 1, 2
    fwd_done_semaphore_id = 0  # causal chains only
    reader_ct += [
        sender_semaphore_id,
        receiver_semaphore_id,
        valid_semaphore_id,
        0,  # 31: mcast_enabled, set after the chains are built
        int(use_zigzag_balancing),  # 32
        windowed_mode,  # 33
        kv_chain_mode,  # 34
        int(use_mask_block_map),  # 35
        kv_slots,  # 36
        fwd_done_semaphore_id,  # 37
        kv_retain,  # 38
    ]
    reader_ct += _accessor(q) + _accessor(k) + _accessor(v) + _accessor(attn_mask)
    reader_ct += _accessor(None) * 3  # page_table, attention_sink, chunk_start_idx
    reader_ct += _accessor(None) * 2  # cu_window_seqlens, windowed_q_token_offset_tensor
    reader_ct += _accessor(attn_mask_block_map)

    writer_ct = [
        B, NQH, valid_Sqt, Sk, vDHt, Sq_chunk_t, q_num_chunks, Sk_chunk_t, k_num_chunks, packed_identity_scalar,
        num_cores,
        0,  # use_causal_kernel
        int(use_provided_mask),
        int(generated_padding_mask),
        0,  # is_chunked
        0,  # sliding_window_size
        int(lightweight_mask),  # 16
        int(use_streaming_compute),  # 17
        out_out_subblock_h,  # 18
        k_partial_col,  # 19
        int(use_zigzag_balancing),  # 20
        windowed_mode,  # 21
        int(output_concat_heads),  # 22
        sender_semaphore_id,  # 23
        receiver_semaphore_id,  # 24
        valid_semaphore_id,  # 25
        fwd_done_semaphore_id,  # 26
        kv_chain_mode,  # 27
        1,  # 28: gqa_pack
        1,  # 29: gqa_pack_sqt
    ]  # fmt: skip

    out_shape = [B, 1, Sq, NQH * DH]
    output = ttnn.allocate_tensor_on_device(ttnn.Shape(out_shape), q.dtype, ttnn.TILE_LAYOUT, device, memory_config)
    writer_ct += _accessor(output) + _accessor(None) * 2  # out, cu_window_seqlens, windowed_q_token_offset_tensor

    compute_ct = [
        Skt, DHt, vDHt, Sq_chunk_t, q_num_chunks, Sk_chunk_t, k_num_chunks, qk_in0_block_w, qk_out_subblock_w,
        qk_out_subblock_h, qk_in0_num_subblocks, qk_in1_num_subblocks, out_in0_block_w, out_out_subblock_w,
        out_out_subblock_h, out_in0_num_subblocks, out_in1_num_subblocks,
        0,  # use_causal_kernel
        int(compute_use_provided_mask),
        int(generated_padding_mask),
        0,  # is_chunked
        scale_packed,
        0,  # sliding_window_size
        0,  # use_attention_sink
        int(use_streaming_compute),
        valid_Skt,
        k_partial_col,
        int(use_zigzag_balancing),
        int(use_k_range_ctrl),
        kv_retain,
    ]  # fmt: skip

    defines = sorted(  # std::map order
        {
            "STATS_GRANULARITY": str(stats_granularity),
            "SUB_EXP_GRANULARITY": str(sub_exp_granularity),
            "DHT_GRANULARITY": str(dht_granularity),
            "REDUCE_GRANULARITY": str(reduce_granularity),
            "EXP_APPROX_MODE": str(int(exp_approx_mode)),
        }.items()
    )

    # ---- circular buffers, in create_descriptor's allocation order ----
    q_df, k_df, v_df, mask_df, out_df = q.dtype, k.dtype, v.dtype, attn_mask.dtype, q.dtype
    scalar_df = ttnn.bfloat16  # Q is not fp32
    im_df = stats_df = qk_im_df = sum_df = ttnn.bfloat16  # fp32_dest_acc_en False
    cbs = []

    def allocate_cb(page_size, num_pages, dtype):
        idx = len(cbs)
        cbs.append(
            ttnn.CBDescriptor(
                total_size=page_size * num_pages,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=page_size)],
            )
        )
        return idx

    def allocate_tile_cb(num_tiles, dtype):
        return allocate_cb(TILE_BYTES[dtype], num_tiles, dtype)

    cb = dict.fromkeys(
        [
            "q_in", "k_in", "v_in", "mask_in", "attention_sink", "identity_scale_in", "page_table", "col_identity",
            "chunk_start_idx_compute", "chunk_start_idx_writer", "recip_scratch", "out", "qk_im", "out_im_A",
            "out_im_B", "max_A", "max_B", "sum_A", "sum_B", "exp_max_diff", "cu_window_seqlens", "windowed_q_offset",
            "windowed_cu_reader", "windowed_k_range", "mask_block_map", "kv_fwd_ctrl", "mask_cache",
        ],
        INACTIVE,
    )  # fmt: skip
    cb["q_in"] = allocate_tile_cb(q_tiles, q_df)
    cb["k_in"] = allocate_tile_cb(k_tiles, k_df)
    cb["v_in"] = allocate_tile_cb(v_tiles, v_df)
    cb["kv_fwd_ctrl"] = cb["q_in"]  # kv_chain_mode < 2
    cb["mask_in"] = allocate_tile_cb(mask_tiles, mask_df)  # needs_mask_cb (provided mask), not lightweight
    # Not in ttnn: the head-broadcast mask cache, one batch's Sqt x Skt mask tiles.
    mask_cache_bytes = Sqt * Skt * TILE_BYTES[mask_df]
    if not (
        use_provided_mask
        and broadcast_provided_mask_heads
        and NQH > 1
        and q_num_chunks * k_num_chunks <= 32
        and mask_cache_bytes <= MASK_CACHE_MAX_BYTES
    ):
        raise NotImplementedError(
            f"mask cache does not apply ({q_num_chunks}x{k_num_chunks} blocks, {mask_cache_bytes} B of mask per batch)"
        )
    cb["mask_cache"] = allocate_tile_cb(Sqt * Skt, mask_df)
    # setup_windowed_cbs (not windowed): fallback ids, and the block map's {k_lo, k_hi} ctrl CB.
    for name in ("cu_window_seqlens", "windowed_q_offset", "windowed_cu_reader", "windowed_k_range"):
        cb[name] = cb["q_in"]
    cb["windowed_k_range"] = allocate_cb(16, 2, ttnn.int32)
    cb["identity_scale_in"] = allocate_tile_cb(scale_tiles, scalar_df)
    cb["col_identity"] = allocate_tile_cb(scale_tiles, scalar_df)
    block_map_stick_size = int(attn_mask_block_map.buffer_aligned_page_size())
    cb["mask_block_map"] = allocate_cb(block_map_stick_size, 1, ttnn.int32)
    cb["recip_scratch"] = allocate_tile_cb(1, im_df)  # streaming compute
    cb["qk_im"] = allocate_tile_cb(qk_tiles, qk_im_df)
    cb["out_im_A"] = allocate_tile_cb(out_im_tiles, im_df)
    cb["out_im_B"] = allocate_tile_cb(out_im_tiles, im_df)
    cb["max_A"] = allocate_tile_cb(statistics_tiles, stats_df)
    cb["max_B"] = allocate_tile_cb(statistics_tiles, stats_df)
    cb["sum_A"] = allocate_tile_cb(statistics_tiles, sum_df)
    cb["sum_B"] = allocate_tile_cb(statistics_tiles, sum_df)
    cb["exp_max_diff"] = allocate_tile_cb(statistics_tiles, stats_df)
    cb["out"] = allocate_tile_cb(out0_t, out_df)

    # sdpa_interleaved_cb_ids.hpp, plus mask_cache appended to the reader's list (not in ttnn)
    reader_ct += [
        cb[n]
        for n in (
            "q_in", "k_in", "v_in", "mask_in", "attention_sink", "page_table", "chunk_start_idx_compute",
            "chunk_start_idx_writer", "windowed_cu_reader", "windowed_k_range", "mask_block_map", "kv_fwd_ctrl",
            "mask_cache",
        )
    ]  # fmt: skip
    writer_ct += [
        cb[n]
        for n in (
            "mask_in", "identity_scale_in", "col_identity", "chunk_start_idx_writer", "out", "cu_window_seqlens",
            "windowed_q_offset", "kv_fwd_ctrl",
        )
    ]  # fmt: skip
    compute_ct += [
        cb[n]
        for n in (
            "q_in", "k_in", "v_in", "mask_in", "attention_sink", "identity_scale_in", "col_identity",
            "chunk_start_idx_compute", "recip_scratch", "out", "qk_im", "out_im_A", "out_im_B", "max_A", "max_B",
            "sum_A", "sum_B", "exp_max_diff", "windowed_k_range",
        )
    ]  # fmt: skip
    compute_ct += _accessor(output)

    semaphores = [
        ttnn.SemaphoreDescriptor(id=sender_semaphore_id, core_ranges=core_grid, initial_value=INVALID),
        ttnn.SemaphoreDescriptor(id=receiver_semaphore_id, core_ranges=core_grid, initial_value=INVALID),
        ttnn.SemaphoreDescriptor(id=valid_semaphore_id, core_ranges=core_grid, initial_value=VALID),
    ]

    # ---- K/V chains ----
    num_phases, read_offset, write_offset = 1, 0, 0
    core_chain_info = [CoreChainInfo() for _ in range(num_cores)]
    mcast_chains = 0
    chain_key_q_heads = 1
    kv_in_l1 = (
        k.memory_config().buffer_type == ttnn.BufferType.L1 and v.memory_config().buffer_type == ttnn.BufferType.L1
    )
    retain_skips_chains = kv_retain != 0 and B > MAX_RETAIN_CHAIN_BATCH
    if not retain_skips_chains:
        # Non-causal: with retention (one K chunk, asserted above) the chains share its key.
        # (Without retention the C++ takes kv_chain_key_q_heads_unretained; not ported.)
        assert kv_retain != 0
        key_q_heads = kv_retain
        chains, chains_mcast = build_kv_chains(
            device, grid_x, grid_y, B, NQH, key_q_heads, q_num_chunks, k_num_chunks, kv_retain != 0, q_range
        )
        drop_chains = kv_retain != 0 and kv_in_l1 and chains_mcast == 0
        if not drop_chains:
            core_chain_info, mcast_chains, chain_key_q_heads = chains, chains_mcast, key_q_heads
    reader_ct[sem_args_offset + 3] = 1 if mcast_chains > 0 else 0

    # ---- runtime args ----
    q_addr, k_addr, v_addr = q.buffer_address(), k.buffer_address(), v.buffer_address()
    mask_addr, map_addr, out_addr = (
        attn_mask.buffer_address(),
        attn_mask_block_map.buffer_address(),
        output.buffer_address(),
    )
    reader_rt, writer_rt, compute_rt = [], [], []
    for i in range(num_cores):
        core = ttnn.CoreCoord(i % grid_x, i // grid_x)
        g_start, g_count = q_range(i)
        c = core_chain_info[i]
        r = [q_addr, k_addr, v_addr, mask_addr, 0, 0, 0, num_phases, 0, read_offset]
        r += [int(c.participates), int(c.is_injector), int(c.is_sink), c.batch, c.head]
        r += [c.prev_physical[0], c.prev_physical[1], c.next_physical[0], c.next_physical[1]]
        r += [c.next_core_q_chunks, c.mcast_num_dests, c.mcast_sender_wait, chain_key_q_heads]
        r += [g_start, g_count]
        r += [0, 0, 0, 0]  # windowed tail: cu_window buffer, eles, q token offset, q offset buffer
        r += [map_addr, block_map_stick_size]
        reader_rt.append((core, r))
        writer_rt.append((core, [out_addr, num_phases, 0, 0, write_offset, 0, 0, g_start, g_count, 0, 0, 0, 0]))
        compute_rt.append((core, [num_phases, 0, 0, 0, g_start, g_count]))

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=READER_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_grid,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            defines=defines,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=WRITER_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_grid,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            defines=defines,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=COMPUTE_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_grid,
            compile_time_args=compute_ct,
            runtime_args=compute_rt,
            defines=defines,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ckc.math_fidelity,
                fp32_dest_acc_en=fp32_dest_acc_en,
                dst_full_sync_en=dst_full_sync_en,
                math_approx_mode=bool(ckc.math_approx_mode),
            ),
        ),
    ]
    program = ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=semaphores)
    ttnn.generic_op([q, k, v, attn_mask, attn_mask_block_map, output], program)
    return output
