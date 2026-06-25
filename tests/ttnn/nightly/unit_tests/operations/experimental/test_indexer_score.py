# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the DeepSeek-V3.2 DSA ``indexer_score`` op (GLM5 and DSv32 deployments).

    score[b, s, t] = sum_h relu(q[b, h, s, :] . k[b, t, :]) * w[b, h, s]

Causality from scalar ``chunk_start``: key ``t`` visible to query ``s`` iff
``t <= chunk_start + s``; future columns are -inf.

Two deployments, both Galaxy chunked prefill (50K history + 5K chunk =
55K all-gathered keys; the 5K-query chunk is sharded SP=8 -> 640 queries/device):

    GLM5   -  8 heads (per-device indexer)
    DSv32  - 16 heads (64-head indexer split across TP=4; the op sums only its
             heads and the -inf mask is head-independent, so -inf survives the
             cross-TP sum)

We test on single chip with an explicit ``chunk_start``, ``sp_rank`` selecting the ring
position. The multi-device QuietBox tests at the end instead derive ``chunk_start`` per
device from the real mesh coordinate (one hash-excluded program serving every SP rank).
This file is deliberately narrow (GLM5/DSv32 shapes only); broader knob/corner/validation
coverage will be migrated in separately.
"""

import os
import time
from unittest import mock

import pytest
import torch
from loguru import logger

import ttnn

pytestmark = pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="indexer_score is Blackhole-only")

GLX_DIM = 128  # indexer head dim
GLX_SQ = 640  # queries per device (5120 chunk / SP=8)
GLX_T = 56320  # all-gathered keys: 50K history + 5K chunk = 55K, tile-aligned
GLX_HISTORY = GLX_T - 8 * GLX_SQ  # 51200 keys visible to every query

# The two indexer deployments, by (id, head count).
GLX_CASES = [("glm5", 8), ("dsv32", 16)]
GLX_IDS = [c[0] for c in GLX_CASES]


def indexer_score_ref(q, k, w, chunk_start):
    """Per-head fp32 accumulation (a full [Hi,Sq,T] tensor is many GB at GLX sizes)."""
    b, hi, sq, _ = q.shape
    t = k.shape[2]
    q, k, w = q.float(), k.float(), w.float()
    score = torch.zeros(b, sq, t)
    for h in range(hi):
        score += torch.relu(q[:, h] @ k[:, 0].transpose(-2, -1)) * w[:, h]
    future = torch.arange(t).unsqueeze(0) > chunk_start + torch.arange(sq).unsqueeze(1)
    return score.masked_fill(future, float("-inf")).unsqueeze(1)


def make_inputs(heads, dim, sq, t, seed=42):
    """q [1,Hi,Sq,D], k [1,1,T,D], weights [1,Hi,Sq,1], all bf16.

    Weights are random so some gates are negative: -inf padding must stay distinguishable from
    low-but-valid (negative) scores by topk.
    """
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, heads, sq, dim, generator=g, dtype=torch.bfloat16)
    k = torch.randn(1, 1, t, dim, generator=g, dtype=torch.bfloat16)
    w = torch.randn(1, heads, sq, 1, generator=g, dtype=torch.bfloat16)
    return q, k, w


def to_device(t, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, device=device, layout=layout, dtype=dtype)


def run_indexer(
    q,
    k,
    w,
    chunk_start,
    device,
    program_config=None,
    q_dtype=ttnn.bfloat16,
    k_dtype=ttnn.bfloat16,
    compute_kernel_config=None,
):
    """Run the device op and return the row-major bf16 score as a torch tensor.

    q (srcB) and k (srcA) may be bfp8_b; weights stay bf16.
    """
    kwargs = {} if program_config is None else {"program_config": program_config}
    if compute_kernel_config is not None:
        kwargs["compute_kernel_config"] = compute_kernel_config
    out = ttnn.experimental.indexer_score(
        to_device(q, device, dtype=q_dtype),
        to_device(k, device, dtype=k_dtype),
        to_device(w, device),
        chunk_start_idx=chunk_start,
        **kwargs,
    )
    return ttnn.to_torch(out)


def assert_indexer_match(out, ref, sq, t, check_neg=False):
    """Check the -inf map is exact and the visible scores match the reference by PCC."""
    assert out.shape == (1, 1, sq, t)
    # -inf maps must agree exactly (<= bf16 lowest counts as masked)
    masked = ref == float("-inf")
    assert torch.equal(out <= torch.finfo(torch.bfloat16).min, masked)
    # visible values by PCC (0.999 floor for the bf16 device op)
    a, b = out[~masked].flatten().float(), ref[~masked].flatten().float()
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    assert pcc >= 0.999, f"PCC {pcc} < 0.999"
    if check_neg:
        # negative gates -> a zero-filled column can't masquerade as valid
        assert (ref[~masked] < 0).any()


def glx_config(heads):
    """GLX chunked-prefill knobs for the two deployments (GLM5 8h, DSv32 16h).

    - head_group_size=0 keeps all heads resident; head streaming re-reads q per output tile and is
      ~24x slower, so never used here.
    - QC=2 (q_chunk_size=64) reuses each resident K chunk across 2 q-rows, ~2x fewer redundant K
      reads (heads8 sp7 bfp8 0.73 -> 0.48 ms). Fits L1 for both 8 and 16 heads.
    - k_chunk: GLM5 (8h) uses KC=16 (k_chunk=512), the compute-ceiling optimum at 8 heads; DSv32
      (16h) is matmul-bound and stays KC=8 (k_chunk=256).
    """
    return ttnn.IndexerScoreProgramConfig(
        q_chunk_size=64,
        k_chunk_size=512 if heads <= 8 else 256,
        head_group_size=0,
    )


@pytest.mark.parametrize("case_id, heads", GLX_CASES, ids=GLX_IDS)
@pytest.mark.parametrize("q_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["q_bf16", "q_bfp8"])
@pytest.mark.parametrize("sp_rank", [0, 7], ids=["rank0", "rank7"])
def test_indexer_score_accuracy(device, sp_rank, q_dtype, case_id, heads):
    """GLX chunked prefill with the GLX knobs, boundary SP ranks, bfp8 k (the deployed dtype).

    bfp8 k (matmul srcA) halves k BW; PCC stays >= 0.999 (bfp8 quantization of well-conditioned k is
    below the bf16 sum's noise). Negative gates keep -inf padding distinguishable from low scores.
    """
    chunk_start = GLX_HISTORY + sp_rank * GLX_SQ
    q, k, w = make_inputs(heads, GLX_DIM, GLX_SQ, GLX_T)
    out = run_indexer(
        q, k, w, chunk_start, device, program_config=glx_config(heads), q_dtype=q_dtype, k_dtype=ttnn.bfloat8_b
    )
    ref = indexer_score_ref(q, k, w, chunk_start)
    assert_indexer_match(out, ref, GLX_SQ, GLX_T, check_neg=True)


# ---------------------------------------------------------------------------
# Mechanism coverage: small bf16 shapes that exercise every kernel path, checked the same way as the
# deployments (exact -inf map + PCC >= 0.999 + a negative gate present). Two axes -- config knobs on a
# fixed shape, and shape/geometry corners.
# ---------------------------------------------------------------------------
MINI = dict(heads=64, dim=128, sq=64, t=256)  # 64 heads, D=128, Sq=2 tiles, T=8 tiles


def _run_and_check(device, heads, dim, sq, t, chunk_start, q_chunk, k_chunk, head_group):
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=q_chunk, k_chunk_size=k_chunk, head_group_size=head_group)
    q, k, w = make_inputs(heads, dim, sq, t)
    out = run_indexer(q, k, w, chunk_start, device, program_config=cfg)
    ref = indexer_score_ref(q, k, w, chunk_start)
    assert_indexer_match(out, ref, sq, t, check_neg=True)


@pytest.mark.parametrize(
    "q_chunk, k_chunk, head_group, chunk_start",
    [
        (32, 32, 1, 128),  # default: 1 head resident, streamed per output tile
        (32, 32, 8, 128),  # head streaming, groups of 8
        (32, 32, 32, 128),  # head streaming, 2 groups
        (32, 32, 0, 128),  # all heads resident, single-tile chunks
        (64, 32, 16, 128),  # QC=2 multi-row group (HB=16 of 64 to fit L1)
        (32, 128, 0, 128),  # KC=4 chunked k, partial edge chunks
        (64, 128, 16, 128),  # QC=2 and KC=4 together
        (64, 128, 16, 160),  # chunk_start not a k_chunk multiple (diagonal mid-group)
    ],
    ids=["hb1", "hb8", "hb32", "hb_all", "qc2", "kc4", "qc2_kc4", "diag_unaligned"],
)
def test_indexer_score_knobs(device, q_chunk, k_chunk, head_group, chunk_start):
    """QC/KC/head_group sweep on a fixed 64-head shape: head streaming vs all-resident, single vs
    batched k chunks, multi-row q groups, and a diagonal that lands mid-chunk (chunk_start % KC != 0)."""
    _run_and_check(device, MINI["heads"], MINI["dim"], MINI["sq"], MINI["t"], chunk_start, q_chunk, k_chunk, head_group)


@pytest.mark.parametrize(
    "heads, dim, sq, t, chunk_start, q_chunk, k_chunk, head_group",
    [
        (64, 128, 128, 128, 0, 32, 32, 0),  # prefill square: no history, fully-causal triangle from tile 0
        (64, 128, 32, 32, 0, 32, 32, 0),  # single k-tile (Tt=1): in-tile diagonal only
        (64, 128, 32, 256, 0, 32, 32, 1),  # row-0 long tail: 1 valid k-tile, long -inf suffix
        (64, 128, 64, 256, 192, 32, 32, 0),  # fully-causal corner (chunk_start near T)
        (32, 64, 64, 256, 128, 32, 32, 0),  # narrow head dim (D=64, Dt=2)
        (16, 256, 64, 256, 128, 32, 32, 0),  # wide head dim (D=256, Dt=8)
        (64, 128, 64, 192, 128, 32, 128, 0),  # KC does not divide Tt (Tt=6, KC=4 -> 2-tile last unit)
        (64, 128, 64, 160, 96, 32, 128, 0),  # KC does not divide Tt (Tt=5, KC=4 -> 1-tile last unit)
        (64, 128, 64, 160, 96, 32, 64, 0),  # KC does not divide Tt (Tt=5, KC=2 -> 1-tile last unit)
        # head streaming (HB<Hi) AND KC not dividing Tt together: the reader must push a q block for
        # every padded column compute walks, else compute hangs on the partial last unit (regression
        # guard for the streaming/partial-KC deadlock).
        (64, 128, 64, 192, 128, 32, 128, 8),  # stream HB=8 + Tt=6, KC=4 -> 2-tile last unit
        (64, 128, 64, 160, 96, 32, 64, 8),  # stream HB=8 + Tt=5, KC=2 -> 1-tile last unit
        (16, 128, 128, 2048, 512, 64, 32, 0),  # multicore: QC=2 group split across cores (per-core strip writes)
    ],
    ids=[
        "prefill_square",
        "single_ktile",
        "row0_long_tail",
        "fully_causal",
        "dim64",
        "dim256",
        "kc_partial_tt6",
        "kc_partial_tt5_kc4",
        "kc_partial_tt5_kc2",
        "stream_kc_partial_tt6",
        "stream_kc_partial_tt5",
        "multicore_qc2",
    ],
)
def test_indexer_score_shapes(device, heads, dim, sq, t, chunk_start, q_chunk, k_chunk, head_group):
    """Shape/geometry coverage: prefill corners, single/partial k-tiles, narrow/wide head dims, KC not
    dividing Tt (partial last unit clipped by the writer), and a multicore QC=2 split where a single
    q-row-group is dealt across cores."""
    _run_and_check(device, heads, dim, sq, t, chunk_start, q_chunk, k_chunk, head_group)


# Indexed KV cache: k is a shared [B, 1, T, D] cache and cache_batch_idx selects the batch slot to score.
# The slot is excluded from the program hash, so switching slots reuses one cached program (no recompile);
# k may also be ND-sharded across DRAM banks; invalid inputs the hash does not pin are re-checked on a warm
# cache (validate_on_program_cache_hit).
IDX_CACHE = dict(heads=64, dim=128, sq=64, t=256, chunk_start=128)  # small all-resident shape, B slots


def _indexed_inputs(num_slots, seed=11):
    """q [1,Hi,Sq,D], a shared k cache [B,1,T,D], weights [1,Hi,Sq,1], all bf16. Slots differ so a
    wrong-slot read would change the scores (and fail the per-slot reference)."""
    c = IDX_CACHE
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, c["heads"], c["sq"], c["dim"], generator=g, dtype=torch.bfloat16)
    w = torch.randn(1, c["heads"], c["sq"], 1, generator=g, dtype=torch.bfloat16)
    k_cache = torch.randn(num_slots, 1, c["t"], c["dim"], generator=g, dtype=torch.bfloat16)
    return q, w, k_cache


def _check_slot(out, q, k_cache, w, b):
    """Compare a slot's device output against the reference computed on that [1,1,T,D] slice."""
    c = IDX_CACHE
    ref = indexer_score_ref(q, k_cache[b : b + 1], w, c["chunk_start"])
    assert_indexer_match(out, ref, c["sq"], c["t"], check_neg=True)


def _nd_sharded_dram_config(device, rows_per_shard):
    """ND-shard a [.., T, D] cache across DRAM banks: each [1, 1, rows_per_shard, D] block is one shard,
    round-robin over the banks."""
    dim = IDX_CACHE["dim"]
    num_banks = device.dram_grid_size().x
    cores = [ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in range(num_banks)]
    spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, rows_per_shard, dim],
        grid=ttnn.CoreRangeSet(cores),
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    return ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=spec)


def test_indexer_score_indexed_cache(device):
    """cache_batch_idx selects a slot of a shared [B,1,T,D] cache; every slot scores correctly AND
    switching slots on the same cache tensor does not recompile (the slot is excluded from the hash)."""
    c = IDX_CACHE
    B = 3
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=32, head_group_size=0)
    q, w, k_cache = _indexed_inputs(B)
    q_dev, w_dev = to_device(q, device), to_device(w, device)
    k_dev = to_device(k_cache, device)  # tiled [B,1,T,D], DRAM interleaved

    def run(b):
        return ttnn.to_torch(
            ttnn.experimental.indexer_score(
                q_dev, k_dev, w_dev, chunk_start_idx=c["chunk_start"], program_config=cfg, cache_batch_idx=b
            )
        )

    _check_slot(run(0), q, k_cache, w, 0)
    entries_after_first = device.num_program_cache_entries()  # one program now cached for this shape/cfg
    for b in range(1, B):
        _check_slot(run(b), q, k_cache, w, b)
    # Switching the indexed slot must NOT add a program (cache_batch_idx is a runtime arg, not in the hash).
    assert device.num_program_cache_entries() == entries_after_first, "switching cache_batch_idx recompiled"


def test_indexer_score_indexed_cache_nd_sharded_k(device):
    """The indexed k cache may be ND-sharded across DRAM banks (each [1,1,T,D] slot is one shard); the
    reader resolves it through a TensorAccessor, so every slot still scores correctly."""
    c = IDX_CACHE
    B = 2
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=32, head_group_size=0)
    q, w, k_cache = _indexed_inputs(B)
    q_dev, w_dev = to_device(q, device), to_device(w, device)
    k_mem = _nd_sharded_dram_config(device, rows_per_shard=c["t"])  # each [1,1,T,D] slot is one shard
    k_dev = ttnn.from_torch(k_cache, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, memory_config=k_mem)
    assert k_dev.memory_config().is_sharded()

    for b in range(B):
        out = ttnn.to_torch(
            ttnn.experimental.indexer_score(
                q_dev, k_dev, w_dev, chunk_start_idx=c["chunk_start"], program_config=cfg, cache_batch_idx=b
            )
        )
        _check_slot(out, q, k_cache, w, b)


def test_indexer_score_indexed_cache_rejects_oob(device, expect_error):
    """An out-of-range cache_batch_idx (>= B) is rejected -- including on a warm program cache, since the
    slot is re-validated on a cache hit (validate_on_program_cache_hit), not only at miss time."""
    c = IDX_CACHE
    B = 2
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=32, head_group_size=0)
    q, w, k_cache = _indexed_inputs(B)
    q_dev, w_dev = to_device(q, device), to_device(w, device)
    k_dev = to_device(k_cache, device)

    # Warm the program cache with a valid slot; an OOB slot then hits the SAME program (same hash) and must
    # still be rejected by the cache-hit validation.
    ttnn.experimental.indexer_score(
        q_dev, k_dev, w_dev, chunk_start_idx=c["chunk_start"], program_config=cfg, cache_batch_idx=0
    )
    with expect_error(RuntimeError, "cache_batch_idx"):
        ttnn.experimental.indexer_score(
            q_dev, k_dev, w_dev, chunk_start_idx=c["chunk_start"], program_config=cfg, cache_batch_idx=B
        )


def test_indexer_score_indexed_cache_requires_idx_for_multislot(device, expect_error):
    """A multi-slot [B,1,T,D] k cache with NO cache_batch_idx is ambiguous (which slot?) and must be
    rejected -- the non-indexed batch guard in validate_non_hashed. Pairs with the OOB-slot rejection to
    cover the indexed-cache batch invariants the program hash does not pin."""
    c = IDX_CACHE
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=32, head_group_size=0)
    q, w, k_cache = _indexed_inputs(2)  # B=2 slots, but no cache_batch_idx supplied below
    q_dev, w_dev, k_dev = to_device(q, device), to_device(w, device), to_device(k_cache, device)
    with expect_error(RuntimeError, "batch must be 1"):
        ttnn.experimental.indexer_score(q_dev, k_dev, w_dev, chunk_start_idx=c["chunk_start"], program_config=cfg)


def test_indexer_score_rejects_sharded_q(device, expect_error):
    """Only k may be sharded; q (and weights) must stay interleaved. A sharded q is refused by the input
    validation (validate_non_hashed)."""
    c = IDX_CACHE
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=32, head_group_size=0)
    q, w, k_cache = _indexed_inputs(1)
    q_mem = _nd_sharded_dram_config(device, rows_per_shard=32)  # shard q -> must be rejected (q stays interleaved)
    q_dev = ttnn.from_torch(q, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, memory_config=q_mem)
    w_dev, k_dev = to_device(w, device), to_device(k_cache, device)
    with expect_error(RuntimeError, "interleaved"):
        ttnn.experimental.indexer_score(q_dev, k_dev, w_dev, chunk_start_idx=c["chunk_start"], program_config=cfg)


def test_indexer_score_nd_sharded_k_multi_T(device):
    """ND-sharded k at two cache lengths with one FIXED shard shape. indexer_score scores all T keys, so T
    is always hashed -- both T values build their own program -- but each must still read the sharded banks
    correctly."""
    c = IDX_CACHE
    heads, dim, sq, chunk_start = c["heads"], c["dim"], c["sq"], c["chunk_start"]
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=32, head_group_size=0)
    k_mem = _nd_sharded_dram_config(device, rows_per_shard=128)  # fixed shard shape, independent of T
    device.clear_program_cache()
    for t in (256, 512):  # multiples of 128: same shard spec, different T (different shard count)
        q, k, w = make_inputs(heads, dim, sq, t)
        q_dev, w_dev = to_device(q, device), to_device(w, device)
        k_dev = ttnn.from_torch(k, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, memory_config=k_mem)
        out = ttnn.to_torch(
            ttnn.experimental.indexer_score(q_dev, k_dev, w_dev, chunk_start_idx=chunk_start, program_config=cfg)
        )
        assert_indexer_match(out, indexer_score_ref(q, k, w, chunk_start), sq, t, check_neg=True)
    assert device.num_program_cache_entries() == 2, "two distinct T must build two programs (T is hashed)"


# Runtime KV length / oversized persistent cache: k is allocated at its full T (the persistent buffer,
# which stays hashed -- it pins the grid/work-split) and kv_len selects the valid key prefix this dispatch.
# kv_len is excluded from the program hash, so a serving loop growing kv_len (<= T) reuses one program -- no
# recompile. Only output columns [0, kv_len) are written; the stale tail is sliced off.
KV_LEN = dict(heads=64, dim=128, sq=64, t=512, chunk_start=0)  # oversized T=512 buffer, Sq=2 tiles


def _kv_len_inputs(seed=23):
    """q [1,Hi,Sq,D], an oversized k buffer [1,1,T,D], weights [1,Hi,Sq,1], all bf16."""
    c = KV_LEN
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, c["heads"], c["sq"], c["dim"], generator=g, dtype=torch.bfloat16)
    w = torch.randn(1, c["heads"], c["sq"], 1, generator=g, dtype=torch.bfloat16)
    k = torch.randn(1, 1, c["t"], c["dim"], generator=g, dtype=torch.bfloat16)
    return q, w, k


def _check_kv_len(out, q, k, w, kv_len):
    """Only columns [0, kv_len) are valid; compare them to the reference on the first kv_len keys (the rest
    of the [1,1,Sq,T] output is the stale tail and is sliced off)."""
    c = KV_LEN
    assert out.shape == (1, 1, c["sq"], c["t"])
    ref = indexer_score_ref(q, k[:, :, :kv_len, :], w, c["chunk_start"])  # [1,1,Sq,kv_len]
    assert_indexer_match(out[:, :, :, :kv_len], ref, c["sq"], kv_len, check_neg=True)


@pytest.mark.parametrize(
    "q_chunk, k_chunk, head_group",
    [
        (32, 32, 0),  # all heads resident, single-tile k chunks (the full-strip path)
        (32, 32, 8),  # head streaming in groups of 8 (the per-column accumulate_row_streaming path)
        (32, 128, 0),  # KC=4 chunked k: kv_len can land mid-chunk and zero whole trailing chunks
        (64, 32, 16),  # QC=2 multi-row group, HB=16 of 64 resident
    ],
    ids=["resident", "stream", "chunked_k", "qc2"],
)
def test_indexer_score_runtime_kv_len(device, q_chunk, k_chunk, head_group):
    """One oversized k buffer (T=512) scored at several kv_len <= T: each writes only [0, kv_len) and
    matches the reference on the first kv_len keys, and growing kv_len on the SAME buffer does NOT recompile
    (kv_len is a runtime arg excluded from the hash). This is the persistent-cache serving loop, swept over
    the kernel paths (resident / head-streaming / chunked-k / multi-row group)."""
    c = KV_LEN
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=q_chunk, k_chunk_size=k_chunk, head_group_size=head_group)
    q, w, k = _kv_len_inputs()
    q_dev, w_dev, k_dev = to_device(q, device), to_device(w, device), to_device(k, device)

    def run(kv_len):
        return ttnn.to_torch(
            ttnn.experimental.indexer_score(
                q_dev, k_dev, w_dev, chunk_start_idx=c["chunk_start"], program_config=cfg, kv_len=kv_len
            )
        )

    _check_kv_len(run(64), q, k, w, 64)  # only the first 2 k-tiles valid; most work units are fully past kv_len
    entries_after_first = device.num_program_cache_entries()
    for kv_len in (128, 256, 512):  # grow the valid prefix within the same T=512 buffer
        _check_kv_len(run(kv_len), q, k, w, kv_len)
    assert device.num_program_cache_entries() == entries_after_first, "changing kv_len recompiled"


def test_indexer_score_rejects_bad_kv_len(device, expect_error):
    """kv_len must be tile-aligned, within (0, T], and leave room for the causal window
    (chunk_start + Sq <= kv_len). Each violation is rejected -- on a WARM cache too, since kv_len is excluded
    from the hash and re-validated on a program-cache hit (validate_on_program_cache_hit)."""
    c = KV_LEN
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=32, head_group_size=0)
    q, w, k = _kv_len_inputs()
    q_dev, w_dev, k_dev = to_device(q, device), to_device(w, device), to_device(k, device)

    # Warm the program cache with a valid kv_len; each bad one then hits the SAME program and must still fail.
    ttnn.experimental.indexer_score(
        q_dev, k_dev, w_dev, chunk_start_idx=c["chunk_start"], program_config=cfg, kv_len=128
    )
    for bad, why in [(c["t"] + 32, "above T"), (100, "not tile-aligned"), (32, "causal window > kv_len")]:
        with expect_error(RuntimeError, "kv_len"):
            ttnn.experimental.indexer_score(
                q_dev, k_dev, w_dev, chunk_start_idx=c["chunk_start"], program_config=cfg, kv_len=bad
            )


@pytest.mark.parametrize("case_id, heads", GLX_CASES, ids=GLX_IDS)
def test_indexer_score_determinism(device, case_id, heads):
    """Determinism on the production deployments (GLM5.1 8h, DSv32 16h, GLX chunked-prefill knobs at
    sp_rank 7, deployed bf16 q + bfp8 k). The op feeds a downstream top-k key selection, so any
    nondeterminism (reduction order, an unstable -inf boundary) would silently change which keys are kept.

    Follows the ring-joint-SDPA determinism rule: upload the device inputs once and reuse them for every
    iteration (this tests device-side determinism, not host re-generation), then require each output to be
    bit-identical to the first run.
    """
    num_iterations = 10
    chunk_start = GLX_HISTORY + 7 * GLX_SQ  # sp_rank 7: fullest causal case
    cfg = glx_config(heads)
    q, k, w = make_inputs(heads, GLX_DIM, GLX_SQ, GLX_T)
    # Upload once; the same device tensors are reused across all iterations.
    q_dev = to_device(q, device, dtype=ttnn.bfloat16)
    k_dev = to_device(k, device, dtype=ttnn.bfloat8_b)
    w_dev = to_device(w, device)

    reference = None
    for i in range(num_iterations):
        out = ttnn.to_torch(
            ttnn.experimental.indexer_score(q_dev, k_dev, w_dev, chunk_start_idx=chunk_start, program_config=cfg)
        )
        if reference is None:
            reference = out
        elif not torch.equal(reference, out):
            diff_mask = reference != out
            num_diffs = int(diff_mask.sum().item())
            both_finite = torch.isfinite(reference) & torch.isfinite(out)
            max_diff = (
                (reference[both_finite] - out[both_finite]).abs().max().item() if both_finite.any() else float("nan")
            )
            pytest.fail(
                f"indexer_score {case_id} output at iteration {i} differs from iteration 0: "
                f"{num_diffs} differing elements, max finite diff = {max_diff}"
            )
    logger.info(f"indexer_score {case_id} determinism verified: all {num_iterations} outputs identical")


# Blackhole post-commit / sanity coverage reuses test_indexer_score_accuracy above: the CI entry in
# tests/pipeline_reorg/ttnn-tests.yaml selects its sp_rank-7 GLM5.1/DSv32 cases via `-k "accuracy and
# rank7"`. No separate post-commit test (that would re-run the same cases under nightly); post-commit just
# runs a subset of the nightly accuracy parametrization.


# ---------------------------------------------------------------------------
# compute_kernel_config: math_fidelity is the one honored knob (default dtype-derived); the bf16-DEST
# half-sync modes are required, so fp32_dest_acc_en / dst_full_sync_en are rejected by validate.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.LoFi])
def test_indexer_score_compute_kernel_config(device, math_fidelity):
    """An explicit compute_kernel_config overrides math_fidelity; accuracy holds across fidelities."""
    heads, dim, sq, t, chunk_start = MINI["heads"], MINI["dim"], MINI["sq"], MINI["t"], 128
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=32, head_group_size=0)
    ckc = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=math_fidelity)
    q, k, w = make_inputs(heads, dim, sq, t)
    out = run_indexer(q, k, w, chunk_start, device, program_config=cfg, compute_kernel_config=ckc)
    ref = indexer_score_ref(q, k, w, chunk_start)
    assert_indexer_match(out, ref, sq, t, check_neg=True)


def test_indexer_score_rejects_fp32_dest_acc(device, expect_error):
    """fp32_dest_acc_en=True is not supported by the custom LLK -> validate must reject it."""
    heads, dim, sq, t = MINI["heads"], MINI["dim"], MINI["sq"], MINI["t"]
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=32, head_group_size=0)
    ckc = ttnn.init_device_compute_kernel_config(device.arch(), fp32_dest_acc_en=True)
    q, k, w = make_inputs(heads, dim, sq, t)
    with expect_error(RuntimeError, "fp32_dest_acc_en=false"):
        run_indexer(q, k, w, 128, device, program_config=cfg, compute_kernel_config=ckc)


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="perf test - run locally")
@pytest.mark.parametrize("case_id, heads", GLX_CASES, ids=GLX_IDS)
def test_indexer_score_perf(device, case_id, heads):
    """Wall-clock latency per op for the GLX shape at the fullest causal rank (sp7), bfp8 k.

    Inputs placed on device once (host transfer excluded), run program-cache-warm with a device sync
    around a fixed iteration count. Host-dispatched single-op latency (includes enqueue overhead, not
    pure device-kernel time -- use tracy for that). Logged ms is the signal; the assert is only a
    coarse hang / gross-regression guard (thresholds are board-dependent).
    """
    warmup_iters, measured_iters = 3, 20
    chunk_start = GLX_HISTORY + 7 * GLX_SQ
    cfg = glx_config(heads)
    q, k, w = make_inputs(heads, GLX_DIM, GLX_SQ, GLX_T)
    q_dev = to_device(q, device)
    k_dev = to_device(k, device, dtype=ttnn.bfloat8_b)
    w_dev = to_device(w, device)

    def run_once():
        return ttnn.experimental.indexer_score(q_dev, k_dev, w_dev, chunk_start_idx=chunk_start, program_config=cfg)

    for _ in range(warmup_iters):  # compile + program-cache warm
        run_once().deallocate()
    ttnn.synchronize_device(device)

    start = time.perf_counter()
    for _ in range(measured_iters):
        run_once().deallocate()
    ttnn.synchronize_device(device)
    ms_per_op = (time.perf_counter() - start) / measured_iters * 1e3

    logger.info(
        f"indexer_score {case_id} rank7 heads={heads} k=bfp8: "
        f"{ms_per_op:.3f} ms/op (mean of {measured_iters} iters)"
    )
    assert ms_per_op < 50.0, f"{case_id} rank7 k=bfp8: {ms_per_op:.3f} ms/op exceeds 50 ms guard (regression or hang)"


# ---------------------------------------------------------------------------
# sp_rank 7 math-utilization perf check (tracy device profiler; no accuracy check)
# math_util = matmul FLOPs / (cores x device cycles x matmul peak); duration from tracy, FLOPs from
# shape. Run locally with the profiler build (default):
#   pytest tests/ttnn/nightly/unit_tests/operations/experimental/test_indexer_score.py::test_indexer_score_sp7_math_util
# ---------------------------------------------------------------------------
SP7_CHUNK_START = GLX_HISTORY + 7 * GLX_SQ  # fullest causal case (99.5% valid)

# Blackhole matmul peak (tests/nightly/sdpa_perf_utils.py): 4096 mm FLOP/cycle/core at LoFi,
# halved per extra math-fidelity phase. The `fidelity` param below must match the factory's
# choice for the measured q/k dtypes (LoFi when both bfp8, else HiFi2).
_BH_CLOCK_GHZ = 1.35
_MM_FLOPS_PER_CYCLE_PER_CORE = {"LoFi": 4096, "HiFi2": 2048, "HiFi3": 1365, "HiFi4": 1024}


def sp7_valid_tiles():
    """Causal-valid output tiles V at sp_rank 7 = sum_s min(Tt, chunk_t + s + 1) over q-tile-rows."""
    chunk_t = SP7_CHUNK_START // 32
    tt_tiles = GLX_T // 32
    sqt = GLX_SQ // 32
    return sum(min(tt_tiles, chunk_t + s + 1) for s in range(sqt))


def indexer_mm_flops(valid_tiles, heads):
    """Matmul FLOPs the kernel performs: each valid 32x32 output tile is, per head, a 32x32xD
    matmul = (32*32) elements x (2*D) FLOPs; summed over heads and valid tiles."""
    return valid_tiles * heads * (32 * 32) * (2 * GLX_DIM)


# (heads, q_id, k_id, fidelity) cases for the sp7 profiler tests. fidelity must match the factory:
# q and k both bfp8 -> LoFi (1 phase), any bf16 input -> HiFi2. id is shared by perf_impl and math_util
# (which spawns it by id), so they must stay in lockstep. Both deployments run bfp8 k; the q_bfp8 rows
# add the LoFi path (bfp8 q halves the dominant q read and doubles matmul peak).
_SP7_PERF_CASES = [
    (8, "q_bf16", "k_bfp8", "HiFi2"),  # GLM5
    (16, "q_bf16", "k_bfp8", "HiFi2"),  # DSv32
    (8, "q_bfp8", "k_bfp8", "LoFi"),  # GLM5, bfp8 q -> LoFi
    (16, "q_bfp8", "k_bfp8", "LoFi"),  # DSv32, bfp8 q -> LoFi
]
_SP7_PERF_IDS = [f"heads{h}_{q}_{k}" for h, q, k, _ in _SP7_PERF_CASES]


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="perf test - run locally with tracy")
@pytest.mark.parametrize("heads, q_id, k_id", [(h, q, k) for h, q, k, _ in _SP7_PERF_CASES], ids=_SP7_PERF_IDS)
def test_indexer_score_sp7_perf_impl(device, heads, q_id, k_id):
    """Inner test profiled by tracy: a few indexer_score ops at GLX sp_rank 7. No accuracy check."""
    q_dtype = ttnn.bfloat16 if q_id == "q_bf16" else ttnn.bfloat8_b
    k_dtype = ttnn.bfloat16 if k_id == "k_bf16" else ttnn.bfloat8_b
    q, k, w = make_inputs(heads, GLX_DIM, GLX_SQ, GLX_T)
    q_dev = to_device(q, device, dtype=q_dtype)
    k_dev = to_device(k, device, dtype=k_dtype)
    w_dev = to_device(w, device)
    cfg = glx_config(heads)
    for _ in range(5):  # tracy logs each op's device duration; the outer test takes the min
        ttnn.experimental.indexer_score(
            q_dev, k_dev, w_dev, chunk_start_idx=SP7_CHUNK_START, program_config=cfg
        ).deallocate()
    ttnn.synchronize_device(device)


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="perf test - run locally with tracy")
@pytest.mark.parametrize("heads, q_id, k_id, fidelity", _SP7_PERF_CASES, ids=_SP7_PERF_IDS)
def test_indexer_score_sp7_math_util(heads, q_id, k_id, fidelity):
    """sp_rank 7 matmul math utilization from a tracy device-kernel-duration measurement.

    Spawns the inner perf_impl test under the device profiler, reads the minimum
    DEVICE KERNEL DURATION, and reports math_util = mm_flops / (cores * cycles * peak).
    No accuracy check.
    """
    from tracy.process_model_log import run_device_profiler
    from tests.nightly.sdpa_perf_utils import post_process_ops_log

    subdir = "ttnn_indexer_score_sp7"
    perf_id = f"heads{heads}_{q_id}_{k_id}"
    command = (
        "pytest tests/ttnn/nightly/unit_tests/operations/experimental/test_indexer_score.py::"
        f"test_indexer_score_sp7_perf_impl[{perf_id}]"
    )
    with mock.patch.dict(os.environ, {"CI": "false"}):
        run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])
    r = post_process_ops_log(
        subdir,
        float_columns=["CORE COUNT", "DEVICE KERNEL DURATION [ns]"],
        columns=["ATTRIBUTES"],
        sum_vals=False,
        has_signposts=False,
    )
    assert len(r["DEVICE KERNEL DURATION [ns]"]) > 0, "profiler returned no indexer_score ops"

    core_count = int(r["CORE COUNT"][0])
    duration_ns = float(r["DEVICE KERNEL DURATION [ns]"].min())

    valid_tiles = sp7_valid_tiles()
    mm_flops = indexer_mm_flops(valid_tiles, heads)
    peak = _MM_FLOPS_PER_CYCLE_PER_CORE[fidelity]
    cycles = duration_ns * _BH_CLOCK_GHZ
    theoretical_flops = core_count * cycles * peak
    math_util = (mm_flops / theoretical_flops) * 100 if theoretical_flops > 0 else 0.0

    logger.info(
        f"indexer_score sp7 heads={heads} {q_id} {k_id} ({fidelity}): device={duration_ns / 1e6:.3f} ms, "
        f"cores={core_count}, V={valid_tiles} tiles, mm={mm_flops / 1e9:.1f} GFLOP, "
        f"peak={peak} FLOP/cyc/core @ {_BH_CLOCK_GHZ} GHz -> math_util={math_util:.1f}%"
    )


# ---------------------------------------------------------------------------
# Device-perf op-test band check (CI-gated by INDEXER_SCORE_PERF_CHECKS=1; runs in the "ops perf tests"
# job of perf-device-models). Mirrors SDPA's test_sdpa_perf_check: measure sp_rank-7 HiFi2 math
# utilization (the deployed bf16 q + bfp8 k path) via tracy and assert it stays within a symmetric +/-
# band, catching both regressions and unexpected speedups. Expected values were measured on a Blackhole
# dev board; the band is wider than SDPA's 1% while the op's cross-board perf is still being characterized.
# ---------------------------------------------------------------------------
INDEXER_PERF_MARGIN = 0.02  # symmetric +/- 2%

_INDEXER_PERF_CHECK_CONFIGS = [
    # (case_id, heads, expected_util) at HiFi2 (bf16 q, bfp8 k), sp_rank 7
    ("glm5", 8, 70.1),
    ("dsv32", 16, 76.1),
]


@pytest.mark.skipif(
    os.environ.get("INDEXER_SCORE_PERF_CHECKS") != "1",
    reason="Set INDEXER_SCORE_PERF_CHECKS=1 to run (CI: ops perf tests job)",
)
@pytest.mark.parametrize(
    "case_id, heads, expected_util",
    _INDEXER_PERF_CHECK_CONFIGS,
    ids=[f"{case_id}_heads{heads}" for case_id, heads, _ in _INDEXER_PERF_CHECK_CONFIGS],
)
def test_indexer_score_perf_check(case_id, heads, expected_util):
    """GLM5.1 / DSv32 sp_rank-7 HiFi2 math utilization via tracy, asserted within +/- INDEXER_PERF_MARGIN."""
    from tracy.process_model_log import run_device_profiler
    from tests.nightly.sdpa_perf_utils import post_process_ops_log

    subdir = "ttnn_indexer_score_perf_check"
    perf_id = f"heads{heads}_q_bf16_k_bfp8"  # HiFi2 (bf16 q, bfp8 k)
    command = (
        "pytest tests/ttnn/nightly/unit_tests/operations/experimental/test_indexer_score.py::"
        f"test_indexer_score_sp7_perf_impl[{perf_id}]"
    )
    with mock.patch.dict(os.environ, {"CI": "false"}):
        run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])
    r = post_process_ops_log(
        subdir,
        float_columns=["CORE COUNT", "DEVICE KERNEL DURATION [ns]"],
        columns=["ATTRIBUTES"],
        sum_vals=False,
        has_signposts=False,
    )
    assert len(r["DEVICE KERNEL DURATION [ns]"]) > 0, "profiler returned no indexer_score ops"

    core_count = int(r["CORE COUNT"][0])
    duration_ns = float(r["DEVICE KERNEL DURATION [ns]"].min())
    mm_flops = indexer_mm_flops(sp7_valid_tiles(), heads)
    peak = _MM_FLOPS_PER_CYCLE_PER_CORE["HiFi2"]
    cycles = duration_ns * _BH_CLOCK_GHZ
    utilization = (mm_flops / (core_count * cycles * peak)) * 100 if core_count > 0 else 0.0

    lower = expected_util * (1 - INDEXER_PERF_MARGIN)
    upper = expected_util * (1 + INDEXER_PERF_MARGIN)
    logger.info(
        f"indexer_score perf check {case_id} heads={heads} (HiFi2): duration={duration_ns / 1e6:.3f} ms, "
        f"math_util={utilization:.2f}% (expected {expected_util:.2f}%, band [{lower:.2f}, {upper:.2f}])"
    )
    assert lower <= utilization <= upper, (
        f"Math utilization {utilization:.2f}% outside band [{lower:.2f}, {upper:.2f}] "
        f"(expected {expected_util:.2f}%, margin +/- {INDEXER_PERF_MARGIN * 100:.1f}%)"
    )


# ---------------------------------------------------------------------------
# Generalized-multicast regime coverage (accuracy): exercises the banded-product scheduler paths the
# small knobs/shapes tests miss -- chiefly G > grid.y (groups phase-stacked onto rows, the case that
# would deadlock the k-mcast column if rows took uneven group counts), G prime > grid.y (rows_used==1,
# k-mcast off but still correct), uneven k-band split across columns (U not a multiple of grid.x),
# partial last band, and phase-stacking under head streaming. Same exact -inf + PCC + negative-gate
# check as the deployments. q_chunk/k_chunk are in ELEMENTS (tiles*32); head_group 0 = all resident.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "heads, dim, sq, t, chunk_start, q_chunk, k_chunk, head_group",
    [
        # G>gy, uniform groups/row (G=12 -> rows_used=6, 2 groups/row), U==grid.x (k-mcast on)
        (8, 128, 384, 704, 128, 32, 64, 0),
        # G>gy with U NOT a multiple of grid.x=11 (G=20 -> rows_used=10, 2 groups/row; U=50, uneven cols)
        (8, 128, 640, 1600, 256, 32, 32, 0),
        # G prime > gy (Sqt=11, QC=1 -> G=11 -> rows_used=1, k-mcast off): correctness without mcast
        (8, 128, 352, 512, 128, 32, 32, 0),
        # G>gy + KC does not divide Tt (partial last band) + U<grid.x (G=12, U=7)
        (8, 128, 384, 608, 64, 32, 96, 0),
        # G>gy + head streaming (HB=8<16): q-mcast + k-mcast on, phase-stacked groups
        (16, 128, 384, 704, 128, 32, 64, 8),
        # G>gy big (G=40 -> rows_used=10, 4 groups/row) + uneven U
        (8, 128, 1280, 1600, 256, 32, 32, 0),
    ],
    ids=[
        "Ggy_uniform",
        "Ggy_uneven_U",
        "Ggy_prime",
        "Ggy_partial_kc",
        "Ggy_stream",
        "Ggy_big",
    ],
)
def test_indexer_score_genmcast_regimes(device, heads, dim, sq, t, chunk_start, q_chunk, k_chunk, head_group):
    """Banded-product scheduler regimes beyond the original knobs/shapes coverage: G>grid.y phase
    stacking, prime G, uneven k-band columns, partial bands, and streaming -- all checked for exact
    causality + PCC like the deployments."""
    _run_and_check(device, heads, dim, sq, t, chunk_start, q_chunk, k_chunk, head_group)


def test_indexer_score_streaming_qmcast_uneven_bands(device):
    """Streaming (HB<Hi) q-mcast with an UNEVEN k-band split across a grid row -- the situation a naive
    per-output-tile q-mcast would deadlock on: cores in the same row own different band counts, so they
    would issue different numbers of q reads / mcast rendezvous and the row's sender would wait forever
    on receivers that already finished.

    The dummy-pad scheduler fixes this by padding every core's streaming band loop to max_bands (the
    row's widest column) with phantom q-only bands, so the q-mcast handshake count is uniform per row.
    This test pins that the op stays correct (no hang, exact -inf + PCC) on that shape. U=23 is prime,
    so min(U, grid.x) never divides it -> the band split is uneven (max_bands > min band count) on any
    Blackhole grid width.
    """
    heads, dim, sq, t = 16, 128, 128, 736  # Hi=16, Sqt=4, Tt=23
    chunk_start, q_chunk, k_chunk, head_group = 128, 32, 32, 8  # QC=1, KC=1, HB=8 (< Hi -> streaming)

    grid = device.compute_with_storage_grid_size()
    QC, KC = q_chunk // 32, k_chunk // 32
    G, U = (sq // 32) // QC, ((t // 32) + KC - 1) // KC
    cols_used = min(U, grid.x)
    max_bands = (U + cols_used - 1) // cols_used
    min_bands = U // cols_used
    # Precondition: this really is the deadlock-prone shape (streaming, >1 column, uneven bands).
    assert head_group < heads, "must be streaming (HB < Hi)"
    assert cols_used > 1, f"need >1 column for q-mcast: cols_used={cols_used}"
    assert max_bands > min_bands, f"need an uneven band split: U={U}, cols_used={cols_used}"

    _run_and_check(device, heads, dim, sq, t, chunk_start, q_chunk, k_chunk, head_group)


# ==============================================================================================
# Multi-device (QuietBox, 4 BH) tests: PER-DEVICE chunk_start derived from the mesh coordinate.
# ==============================================================================================
# Use the `mesh_device` fixture (vs `device` above); they auto-skip on a single chip (the fixture skips
# when requested devices exceed the machine's), so they no-op on the single-chip nightly and run on a
# QuietBox. A SINGLE mesh dispatch where each device is a different SP rank: chunk_start = chunk_start_idx
# + r*Sq (r = linearized index along cluster_axis), one hash-excluded program for all. Two layouts:
#   - 1D SP=4 (flat mesh): history 25600 + chunk 4*640 -> T 28160; cluster_axis unset (linear order).
#   - 2D SP=2 x TP=2:      history 25600 + chunk 2*640 -> T 26880; cluster_axis = SP axis, heads split.
# Functional only (exact -inf map + PCC >= 0.999 per SP rank); reuse indexer_score_ref / assert_indexer_match.

QB_DIM = 128  # indexer head dim
QB_SQ = 640  # queries per SP rank (preserved from the SP=8 deployment)
QB_HISTORY = 25600  # 25k history, tile-aligned (800 tiles)

# GLM5 (8 heads) and DSv32 (16 heads), as in the single-device deployment cases above.
QB_CASES = [("glm5", 8), ("dsv32", 16)]
QB_IDS = [c[0] for c in QB_CASES]


def _global_inputs(heads, chunk, t, seed):
    """Global GLX tensors (bf16; deployed dtypes applied at shard time): q/w over `chunk` queries, k over
    `t` all-gathered keys. Weights are random so some gates are negative (-inf must stay distinguishable
    from low-but-valid scores)."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, heads, chunk, QB_DIM, generator=g, dtype=torch.bfloat16)
    k = torch.randn(1, 1, t, QB_DIM, generator=g, dtype=torch.bfloat16)
    w = torch.randn(1, heads, chunk, 1, generator=g, dtype=torch.bfloat16)
    return q, k, w


def _to_mesh(mesh_device, t, dtype, mapper):
    """from_torch with the shared tiled-layout boilerplate."""
    return ttnn.from_torch(t, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=dtype, mesh_mapper=mapper)


def _per_sp_ref(q_g, k_g, w_g, sp_count, history):
    """Reference: each SP rank's full-head score (chunk_start = history + sp*Sq), concatenated along seq."""
    refs = []
    for sp in range(sp_count):
        sl = slice(sp * QB_SQ, (sp + 1) * QB_SQ)
        refs.append(indexer_score_ref(q_g[:, :, sl, :], k_g, w_g[:, :, sl, :], history + sp * QB_SQ))
    return torch.cat(refs, dim=2)


# ---- 1D mesh: SP=4 (flat QuietBox), cluster_axis unset (linear device order) ------------------
QB_SP = 4  # devices (QuietBox); SP ring positions 0..3
QB_CHUNK = QB_SP * QB_SQ  # 2560 chunk queries (2.5k), sharded SP=4 -> 640/device
QB_T = QB_HISTORY + QB_CHUNK  # 28160 all-gathered keys (880 tiles)


def _shard_1d(mesh_device, heads, seed):
    """SP=4 inputs: q/w sharded along seq (each device its own 640 rows), k replicated. Deployed dtypes."""
    q_g, k_g, w_g = _global_inputs(heads, QB_CHUNK, QB_T, seed)
    shard = ttnn.ShardTensorToMesh(mesh_device, dim=2)
    q_dev = _to_mesh(mesh_device, q_g, ttnn.bfloat16, shard)
    w_dev = _to_mesh(mesh_device, w_g, ttnn.bfloat16, shard)
    k_dev = _to_mesh(mesh_device, k_g, ttnn.bfloat8_b, ttnn.ReplicateTensorToMesh(mesh_device))
    return q_g, k_g, w_g, q_dev, k_dev, w_dev


@pytest.mark.parametrize("mesh_device", [QB_SP], indirect=True)
@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_indexer_score_qb_per_device_chunk_start(mesh_device, case_id, heads):
    """One mesh dispatch over 4 BH devices, each deriving its own chunk_start from its coordinate.
    Validate each device's output against its own chunk_start reference."""
    q_g, k_g, w_g, q_dev, k_dev, w_dev = _shard_1d(mesh_device, heads, seed=42)

    # chunk_start_idx OMITTED -> the op deduces base = T - sp_ring*Sq = QB_HISTORY (sp_ring = 4 devices,
    # cluster_axis unset), then device r gets base + r*Sq. No chunk_start passed at all.
    out = ttnn.experimental.indexer_score(q_dev, k_dev, w_dev, program_config=glx_config(heads))
    out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2))

    ref = _per_sp_ref(q_g, k_g, w_g, QB_SP, QB_HISTORY)
    assert_indexer_match(out_t, ref, QB_CHUNK, QB_T, check_neg=True)


@pytest.mark.parametrize("mesh_device", [QB_SP], indirect=True)
@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_indexer_score_qb_one_compile_all_chunk_starts(mesh_device, case_id, heads):
    """chunk_start is excluded from the program hash: running several different bases must add exactly
    ONE program-cache entry (the first compile), proving no per-value recompile."""
    _, _, _, q_dev, k_dev, w_dev = _shard_1d(mesh_device, heads, seed=7)

    # Three distinct chunk-start bases (all within the causal window). Only the first should compile.
    bases = [QB_HISTORY, QB_HISTORY - QB_SQ, QB_HISTORY - 2 * QB_SQ]

    entries_before = mesh_device.num_program_cache_entries()
    for base in bases:
        ttnn.experimental.indexer_score(
            q_dev, k_dev, w_dev, chunk_start_idx=base, program_config=glx_config(heads)
        ).deallocate()

    added = mesh_device.num_program_cache_entries() - entries_before
    assert added == 1, f"expected 1 program-cache entry across 3 distinct chunk_start bases, got {added}"


# ---- 2D mesh: SP=2 (one axis) x TP=2 (the other, head-split) ----------------------------------
# Exercises cluster_axis on a real 2D mesh, which a 1xN mesh can't: chunk_start must vary along the
# SP axis only, while the two TP devices sharing an SP position get the SAME chunk_start.
QB2_SP = 2  # sequence-parallel ranks (chunk_start varies along this axis)
QB2_TP = 2  # tensor-parallel ranks (heads split across this axis)
QB2_CHUNK = QB2_SP * QB_SQ  # 1280 chunk queries, sharded SP=2 -> 640/device
QB2_T = QB_HISTORY + QB2_CHUNK  # 26880 keys (840 tiles)
QB2_SP_AXIS = 0  # mesh rows = SP ring (passed as cluster_axis)
QB2_TP_AXIS = 1  # mesh cols = TP head split


def _axis_dims(sp_dim, tp_dim):
    """A 2-tuple indexed by mesh axis: SP axis -> sp_dim, TP axis -> tp_dim. Used for both the q/w shard
    mapper and the output composer (sharding and concat use the same dims here)."""
    dims = [None, None]
    dims[QB2_SP_AXIS], dims[QB2_TP_AXIS] = sp_dim, tp_dim
    return tuple(dims)


@pytest.mark.parametrize("mesh_device", [(QB2_SP, QB2_TP)], ids=["2x2"], indirect=True)
@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_indexer_score_qb_sp2_tp2(mesh_device, case_id, heads):
    """One mesh dispatch over a 2x2 mesh: chunk_start derived per-device from the coordinate along
    cluster_axis (SP), constant across the TP axis; heads split across TP. Each TP device computes a
    partial head-sum, which the test sums back (the TP all-reduce) and validates per SP rank against
    its own full-head, own-chunk_start reference."""
    q_g, k_g, w_g = _global_inputs(heads, QB2_CHUNK, QB2_T, seed=42)

    # q/w: seq (dim 2) sharded along the SP axis, heads (dim 1) along the TP axis. k replicated.
    mesh_shape = tuple(mesh_device.shape)
    qw_dims = _axis_dims(sp_dim=2, tp_dim=1)
    shard_qw = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=qw_dims)
    q_dev = _to_mesh(mesh_device, q_g, ttnn.bfloat16, shard_qw)
    w_dev = _to_mesh(mesh_device, w_g, ttnn.bfloat16, shard_qw)
    k_dev = _to_mesh(mesh_device, k_g, ttnn.bfloat8_b, ttnn.ReplicateTensorToMesh(mesh_device))

    # chunk_start_idx OMITTED: the op deduces base = T - sp_ring*Sq, where sp_ring is the mesh extent
    # along cluster_axis (2, the SP axis) -- NOT the total device count (4). Device (sp, tp) then gets
    # chunk_start = base + sp*Sq -- identical for both TP devices at SP position sp.
    out = ttnn.experimental.indexer_score(
        q_dev,
        k_dev,
        w_dev,
        cluster_axis=QB2_SP_AXIS,
        program_config=glx_config(heads // QB2_TP),  # per-device head count
    )
    # Concat SP shards along seq (dim 2) and the TP head-partials along the size-1 head dim (dim 1), then
    # SUM the partials (the TP all-reduce) -> full [1,1,1280,T] score.
    out_t = ttnn.to_torch(
        out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=qw_dims)
    )
    out_t = out_t.float().sum(dim=1, keepdim=True)

    ref = _per_sp_ref(q_g, k_g, w_g, QB2_SP, QB_HISTORY)
    assert_indexer_match(out_t, ref, QB2_CHUNK, QB2_T, check_neg=True)
