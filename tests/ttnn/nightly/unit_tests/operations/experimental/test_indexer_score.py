# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the two lightning-indexer scorers that share one device op:

  indexer_score_dsa - DeepSeek-V3.2 DSA / GLM-5:
      score[b, 0, s, t] = sum_h relu(q[b,h,s,:] . k[b,t,:]) * w[b,h,s]
      ReLU + learned per-head gates, ALL heads summed into one row [B,1,Sq,T].

  indexer_score_msa - MiniMax M3 MSA:
      score[b, g, s, t] = sum_{h in group g} (q[b,h,s,:] . k[b,t,:]) * scale
      Raw dot, no learned gates (just a 1/sqrt(d) ``scale``), Hi heads partitioned into
      ``num_groups`` GQA groups summed within each group (one plane per group), optionally
      block-max-pooled (``block_size>0``) to [B,G,Sq,T/block_size] for the block top-k.

Both share the factory + kernels (the flavour is compile-time args). Causality: key ``t``
visible to query ``s`` iff ``t <= chunk_start + s``; future columns/blocks are -inf.

Deployments are Galaxy chunked prefill (50K history + 5K chunk = 55K keys; the chunk is SP=8
-> 640 q/device): GLM5 (8h), DSv32 (16h, 64-head DSA split across TP=4), M3 (MSA per-GQA-group,
block-max-pooled). Most cases run on one chip with explicit ``chunk_start`` (``sp_rank`` =
ring position); the QuietBox tests derive ``chunk_start`` per device from the mesh coordinate.

Run all (perf/tracy self-skip unless INDEXER_SCORE_PERF_CHECKS=1):
    scripts/run_safe_pytest.sh --run-all <this file>
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


def indexer_score_dsa_ref(q, k, w, chunk_start):
    """DeepSeek-V3.2 / GLM-5 DSA reference: sum_h relu(q.kT) * w over ALL Hi heads into one plane
    -> [b, 1, sq, t]. Per-head fp32 accumulation (a full [Hi,Sq,T] tensor is many GB at GLX sizes).
    """
    b, hi, sq, _ = q.shape
    t = k.shape[2]
    q, k, w = q.float(), k.float(), w.float()
    score = torch.zeros(b, sq, t)
    for h in range(hi):
        score += torch.relu(q[:, h] @ k[:, 0].transpose(-2, -1)) * w[:, h]
    future = torch.arange(t).unsqueeze(0) > chunk_start + torch.arange(sq).unsqueeze(1)
    return score.masked_fill(future, float("-inf")).unsqueeze(1)


def msa_block_max_pool(scores, block_size, chunk_start):
    """MSA block-max-pool + forced-local (the M3 selection step, MSA-only): max over each block_size-key
    block of the causal-masked scores [b,g,sq,t] -> [b,g,sq,t//block_size]. A fully-future block is -inf.
    Then forced-local (sparse_local_block=1): each query's own block (= (chunk_start + s) // block_size) is
    set to +inf. Used both inside indexer_score_msa_ref (block_size>0) and to pool the op's own unpooled
    output in the exact-vs-unpooled cross-checks."""
    b, g, sq, t = scores.shape
    nb = t // block_size
    pooled = scores.reshape(b, g, sq, nb, block_size).amax(dim=-1)
    local = (chunk_start + torch.arange(sq)) // block_size  # each query's own block column [sq]
    pooled[:, :, torch.arange(sq), local] = float("inf")
    return pooled


def indexer_score_msa_ref(q, k, w, chunk_start, num_groups=1, block_size=0):
    """MiniMax-M3 MSA reference (mirrors the indexer_score_msa op): raw dot (NO relu), gated by w (the
    constant 1/sqrt(d) scale), partitioned into num_groups contiguous GQA groups of Hi/num_groups and summed
    WITHIN each group only -> [b, num_groups, sq, t]. num_groups==1 sums all heads into one plane.

    block_size>0 block-max-pools the planes into the M3 block selection -> [b, num_groups, sq, t//block_size].
    """
    b, hi, sq, _ = q.shape
    t = k.shape[2]
    hog = hi // num_groups
    q, k, w = q.float(), k.float(), w.float()
    future = torch.arange(t).unsqueeze(0) > chunk_start + torch.arange(sq).unsqueeze(1)
    planes = []
    for g in range(num_groups):
        score = torch.zeros(b, sq, t)
        for h in range(g * hog, (g + 1) * hog):
            score += (q[:, h] @ k[:, 0].transpose(-2, -1)) * w[:, h]
        planes.append(score.masked_fill(future, float("-inf")))
    scores = torch.stack(planes, dim=1)  # [b, num_groups, sq, t]
    return msa_block_max_pool(scores, block_size, chunk_start) if block_size else scores


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


def _extra_kwargs(program_config, compute_kernel_config):
    kwargs = {} if program_config is None else {"program_config": program_config}
    if compute_kernel_config is not None:
        kwargs["compute_kernel_config"] = compute_kernel_config
    return kwargs


def run_dsa(
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
    """Run indexer_score_dsa (relu + learned gates + head-sum) and return the bf16 score as torch.

    q (srcB) and k (srcA) may be bfp8_b; weights stay bf16. (The block-cyclic K remap derives sp from the
    mesh and so is exercised on a real SP mesh -- see the multidevice tests -- not simulated single-chip.)
    """
    out = ttnn.experimental.indexer_score_dsa(
        to_device(q, device, dtype=q_dtype),
        to_device(k, device, dtype=k_dtype),
        to_device(w, device),
        chunk_start_idx=chunk_start,
        **_extra_kwargs(program_config, compute_kernel_config),
    )
    return ttnn.to_torch(out)


def run_msa(
    q,
    k,
    chunk_start,
    device,
    scale=1.0,
    num_groups=1,
    block_size=0,
    program_config=None,
    q_dtype=ttnn.bfloat16,
    k_dtype=ttnn.bfloat16,
    compute_kernel_config=None,
    cache_batch_idx=None,
    kv_len=None,
):
    """Run indexer_score_msa (raw dot, constant `scale` gate, per-group planes) and return torch.

    No weights tensor: M3 has no learned gates, only `scale` (run as a constant gate in-op).
    cache_batch_idx / kv_len are the same runtime, hash-excluded persistent-cache knobs as DSA.
    """
    persistent = {}
    if cache_batch_idx is not None:
        persistent["cache_batch_idx"] = cache_batch_idx
    if kv_len is not None:
        persistent["kv_len"] = kv_len
    out = ttnn.experimental.indexer_score_msa(
        to_device(q, device, dtype=q_dtype),
        to_device(k, device, dtype=k_dtype),
        chunk_start_idx=chunk_start,
        scale=scale,
        num_groups=num_groups,
        block_size=block_size,
        **persistent,
        **_extra_kwargs(program_config, compute_kernel_config),
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


def assert_grouped_match(out, ref, num_groups, sq, t):
    """Per-group [1,G,Sq,T] check: exact -inf map + PCC>=0.999 (cross-group leakage fails the PCC)."""
    assert out.shape == (1, num_groups, sq, t)
    masked = ref == float("-inf")
    assert torch.equal(out <= torch.finfo(torch.bfloat16).min, masked)
    a, b = out[~masked].flatten().float(), ref[~masked].flatten().float()
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    assert pcc >= 0.999, f"PCC {pcc} < 0.999"


def assert_pooled_match(out, ref, num_groups, sq, nblocks, pcc_floor=0.999):
    """Pooled [1,G,Sq,nblocks] check: exact -inf map + exact +inf (forced-local) map + PCC on the rest.
    block-max amplifies the bf16 per-token error, so the PCC floor is relaxed for the large-T shape."""
    assert out.shape == (1, num_groups, sq, nblocks), f"{out.shape} != {(1, num_groups, sq, nblocks)}"
    masked = ref == float("-inf")
    assert torch.equal(out <= torch.finfo(torch.bfloat16).min, masked)
    forced = ref == float("inf")  # forced-local block (sparse_local_block=1)
    assert torch.equal(out == float("inf"), forced), "forced-local +inf block mismatch"
    keep = ~masked & ~forced
    a, b = out[keep].flatten().float(), ref[keep].flatten().float()
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    assert pcc >= pcc_floor, f"PCC {pcc} < {pcc_floor}"


def glx_config(heads):
    """GLX chunked-prefill knobs for the two deployments (GLM5 8h, DSv32 16h).

    head_group_size=0 keeps all heads resident (streaming is ~24x slower). QC=2 reuses each K chunk
    across 2 q-rows (~2x fewer K reads). k_chunk: GLM5 KC=16 (compute optimum), DSv32 KC=8 (matmul-bound).
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
    out = run_dsa(
        q, k, w, chunk_start, device, program_config=glx_config(heads), q_dtype=q_dtype, k_dtype=ttnn.bfloat8_b
    )
    ref = indexer_score_dsa_ref(q, k, w, chunk_start)
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
    out = run_dsa(q, k, w, chunk_start, device, program_config=cfg)
    ref = indexer_score_dsa_ref(q, k, w, chunk_start)
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
    dividing Tt (partial last unit), and a multicore QC=2 split."""
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
    ref = indexer_score_dsa_ref(q, k_cache[b : b + 1], w, c["chunk_start"])
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
            ttnn.experimental.indexer_score_dsa(
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
            ttnn.experimental.indexer_score_dsa(
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
    ttnn.experimental.indexer_score_dsa(
        q_dev, k_dev, w_dev, chunk_start_idx=c["chunk_start"], program_config=cfg, cache_batch_idx=0
    )
    with expect_error(RuntimeError, "cache_batch_idx"):
        ttnn.experimental.indexer_score_dsa(
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
        ttnn.experimental.indexer_score_dsa(q_dev, k_dev, w_dev, chunk_start_idx=c["chunk_start"], program_config=cfg)


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
        ttnn.experimental.indexer_score_dsa(q_dev, k_dev, w_dev, chunk_start_idx=c["chunk_start"], program_config=cfg)


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
            ttnn.experimental.indexer_score_dsa(q_dev, k_dev, w_dev, chunk_start_idx=chunk_start, program_config=cfg)
        )
        assert_indexer_match(out, indexer_score_dsa_ref(q, k, w, chunk_start), sq, t, check_neg=True)
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
    ref = indexer_score_dsa_ref(q, k[:, :, :kv_len, :], w, c["chunk_start"])  # [1,1,Sq,kv_len]
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
    """One oversized k buffer (T=512) scored at several kv_len <= T: each writes only [0, kv_len), matches
    the reference there, and growing kv_len does NOT recompile (kv_len is hash-excluded). Swept over the
    kernel paths (resident / streaming / chunked-k / multi-row)."""
    c = KV_LEN
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=q_chunk, k_chunk_size=k_chunk, head_group_size=head_group)
    q, w, k = _kv_len_inputs()
    q_dev, w_dev, k_dev = to_device(q, device), to_device(w, device), to_device(k, device)

    def run(kv_len):
        return ttnn.to_torch(
            ttnn.experimental.indexer_score_dsa(
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
    ttnn.experimental.indexer_score_dsa(
        q_dev, k_dev, w_dev, chunk_start_idx=c["chunk_start"], program_config=cfg, kv_len=128
    )
    for bad, why in [(c["t"] + 32, "above T"), (100, "not tile-aligned"), (32, "causal window > kv_len")]:
        with expect_error(RuntimeError, "kv_len"):
            ttnn.experimental.indexer_score_dsa(
                q_dev, k_dev, w_dev, chunk_start_idx=c["chunk_start"], program_config=cfg, kv_len=bad
            )


@pytest.mark.parametrize("case_id, heads", GLX_CASES, ids=GLX_IDS)
def test_indexer_score_determinism(device, case_id, heads):
    """Determinism on the deployments (sp_rank 7, bf16 q + bfp8 k). The op feeds a downstream top-k, so any
    nondeterminism would silently change which keys are kept. Inputs uploaded once and reused (device-side
    determinism); every output must be bit-identical to the first run."""
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
            ttnn.experimental.indexer_score_dsa(q_dev, k_dev, w_dev, chunk_start_idx=chunk_start, program_config=cfg)
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


# Post-commit reuses test_indexer_score_accuracy: the CI entry in tests/pipeline_reorg/ttnn-tests.yaml
# selects its sp_rank-7 cases via `-k "accuracy and rank7"`.


# ---------------------------------------------------------------------------
# MiniMax M3 MSA path (indexer_score_msa): raw dot (no ReLU), no per-head gates, just a 1/sqrt(d) scale.
# At the group-aligned deployment each device owns one index head (Hi=1), so num_groups=1 and [1,1,Sq,T]
# is that group's score row for the downstream block-max top-k.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "q_chunk, k_chunk, head_group",
    [(32, 32, 0), (32, 128, 0), (32, 32, 8)],
    ids=["fallback_kc1", "fullstrip_kc4", "stream_hb8"],
)
def test_indexer_score_msa_compute_paths(device, q_chunk, k_chunk, head_group):
    """MSA raw dot (no relu) over every compute path: per-column fallback (KC=1), head-major full-strip
    (KC=4 all-resident), and head streaming (HB=8). MINI shape, num_groups=1, scale gate."""
    heads, dim, sq, t, chunk_start = MINI["heads"], MINI["dim"], MINI["sq"], MINI["t"], 128
    scale = dim**-0.5
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=q_chunk, k_chunk_size=k_chunk, head_group_size=head_group)
    q, k, _ = make_inputs(heads, dim, sq, t)
    out = run_msa(q, k, chunk_start, device, scale=scale, program_config=cfg)
    w_scale = torch.full((1, heads, sq, 1), scale, dtype=torch.bfloat16)  # MSA's constant gate = scale
    ref = indexer_score_msa_ref(q, k, w_scale, chunk_start)
    assert_indexer_match(out, ref, sq, t)


def test_indexer_score_dsa_msa_differ(device):
    """DSA and MSA must differ: with the SAME constant gate the only difference is DSA's relu, so on
    negative dot products the visible scores must differ (guards against the frontends sharing a path)."""
    heads, dim, sq, t, chunk_start = MINI["heads"], MINI["dim"], MINI["sq"], MINI["t"], 128
    scale = 1.0
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=128, head_group_size=0)
    q, k, _ = make_inputs(heads, dim, sq, t)
    w_const = torch.full((1, heads, sq, 1), scale, dtype=torch.bfloat16)  # match MSA's gate so only relu differs
    out_dsa = run_dsa(q, k, w_const, chunk_start, device, program_config=cfg)
    out_msa = run_msa(q, k, chunk_start, device, scale=scale, program_config=cfg)
    visible = out_dsa > torch.finfo(torch.bfloat16).min  # exclude the -inf causal mask
    assert not torch.equal(out_dsa[visible], out_msa[visible]), "DSA == MSA (relu had no effect / shared path)"


@pytest.mark.parametrize("sp_rank", [0, 7], ids=["rank0", "rank7"])
@pytest.mark.parametrize("q_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["q_bf16", "q_bfp8"])
@pytest.mark.parametrize("k_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["k_bf16", "k_bfp8"])
def test_indexer_score_m3_per_group(device, k_dtype, q_dtype, sp_rank):
    """MiniMax M3 MSA indexer, per GQA group as deployed at TP=4 (one index head per device, Hi=1): raw
    dot scaled by 1/sqrt(d), no ReLU, no gates. GLX chunked-prefill geometry; output [1,1,Sq,T] is the
    group's score row. bfp8 k is the deployed dtype."""
    heads, dim = 1, GLX_DIM  # one GQA group's single index head, head_dim 128
    sq, t = GLX_SQ, GLX_T
    chunk_start = GLX_HISTORY + sp_rank * GLX_SQ
    scale = dim**-0.5
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=512, head_group_size=0)
    q, k, _ = make_inputs(heads, dim, sq, t)
    out = run_msa(q, k, chunk_start, device, scale=scale, program_config=cfg, q_dtype=q_dtype, k_dtype=k_dtype)
    w_scale = torch.full((1, heads, sq, 1), scale, dtype=torch.bfloat16)
    ref = indexer_score_msa_ref(q, k, w_scale, chunk_start)
    assert_indexer_match(out, ref, sq, t)


# ---------------------------------------------------------------------------
# indexer_score_msa num_groups > 1: per-GQA-group output [B, G, Sq, T], multiple groups resident on ONE
# chip (the TP < 4 fallback; the head-reduction is partitioned into G accumulators in-kernel, one plane
# per group, NO cross-group sum). Requires all heads resident + full-strip (KC>=2).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "heads, num_groups",
    [(4, 4), (8, 4), (8, 2), (64, 4)],
    ids=["g4_hog1", "g4_hog2", "g2_hog4", "g4_hog16"],
)
def test_indexer_score_multigroup(device, heads, num_groups):
    """num_groups>1 emits one plane per group, each summing only its Hi/G heads (no cross-group sum). Spans
    hog=1/2/4 and the 64-head MINI geometry; each plane checked against the per-group reference (a plane
    summing the wrong heads fails the PCC)."""
    dim, sq, t, chunk_start = 128, 64, 256, 128
    scale = dim**-0.5
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=64, head_group_size=0)  # all resident, KC=2
    q, k, _ = make_inputs(heads, dim, sq, t)
    out = run_msa(q, k, chunk_start, device, scale=scale, num_groups=num_groups, program_config=cfg)
    w_scale = torch.full((1, heads, sq, 1), scale, dtype=torch.bfloat16)  # MSA's constant gate = scale
    ref = indexer_score_msa_ref(q, k, w_scale, chunk_start, num_groups)
    assert_grouped_match(out, ref, num_groups, sq, t)


def test_indexer_score_multigroup_m3(device):
    """MiniMax M3 with multiple GQA groups on one chip (TP<4 fallback): 4 groups, one index head each,
    raw dot, scale gate = 1/sqrt(d). Output [1,4,Sq,T] is the 4 per-group score rows for the downstream
    block-max top-k.
    """
    heads, num_groups, dim = 4, 4, 128
    sq, t, chunk_start = 128, 256, 128
    scale = dim**-0.5
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=64, head_group_size=0)
    q, k, _ = make_inputs(heads, dim, sq, t)
    out = run_msa(q, k, chunk_start, device, scale=scale, num_groups=num_groups, program_config=cfg)
    w_scale = torch.full((1, heads, sq, 1), scale, dtype=torch.bfloat16)
    ref = indexer_score_msa_ref(q, k, w_scale, chunk_start, num_groups)
    assert_grouped_match(out, ref, num_groups, sq, t)


def test_indexer_score_multigroup_equals_single(device):
    """num_groups=Hi (one head per group) must equal running each head as its own single-group MSA op:
    plane g of the grouped output == indexer_score_msa(q[:, g:g+1]). Direct cross-check that the in-kernel
    per-group split matches the validated single-plane path head-for-head (same scale gate for both).
    """
    heads, dim, sq, t, chunk_start = 4, 128, 64, 256, 128
    scale = 1.0  # same constant gate for grouped and single so the comparison is exact
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=64, head_group_size=0)
    q, k, _ = make_inputs(heads, dim, sq, t)
    grouped = run_msa(q, k, chunk_start, device, scale=scale, num_groups=heads, program_config=cfg)
    for g in range(heads):
        single = run_msa(q[:, g : g + 1], k, chunk_start, device, scale=scale, program_config=cfg)
        assert torch.equal(grouped[:, g : g + 1], single), f"group {g} plane != single-head op"


@pytest.mark.parametrize(
    "k_chunk, head_group, match",
    [(32, 0, "k_chunk_size"), (64, 4, "all heads resident")],
    ids=["kc1_rejected", "streaming_rejected"],
)
def test_indexer_score_multigroup_rejects(device, expect_error, k_chunk, head_group, match):
    """num_groups>1 requires all heads resident + the full-strip path; reject KC<2 and head streaming."""
    heads, dim, sq, t = 8, 128, 64, 256
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=k_chunk, head_group_size=head_group)
    q, k, _ = make_inputs(heads, dim, sq, t)
    with expect_error(RuntimeError, match):
        run_msa(q, k, 128, device, num_groups=2, program_config=cfg)


def test_indexer_score_multigroup_rejects_indivisible(device, expect_error):
    """num_groups must divide Hi -- e.g. 8 heads / 3 groups is rejected (uneven group sizes)."""
    heads, dim, sq, t = 8, 128, 64, 256
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=64, head_group_size=0)
    q, k, _ = make_inputs(heads, dim, sq, t)
    with expect_error(RuntimeError, "and divide Hi 8"):
        run_msa(q, k, 128, device, num_groups=3, program_config=cfg)


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
    out = run_dsa(q, k, w, chunk_start, device, program_config=cfg, compute_kernel_config=ckc)
    ref = indexer_score_dsa_ref(q, k, w, chunk_start)
    assert_indexer_match(out, ref, sq, t, check_neg=True)


def test_indexer_score_rejects_fp32_dest_acc(device, expect_error):
    """fp32_dest_acc_en=True is not supported by the custom LLK -> validate must reject it."""
    heads, dim, sq, t = MINI["heads"], MINI["dim"], MINI["sq"], MINI["t"]
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=32, head_group_size=0)
    ckc = ttnn.init_device_compute_kernel_config(device.arch(), fp32_dest_acc_en=True)
    q, k, w = make_inputs(heads, dim, sq, t)
    with expect_error(RuntimeError, "fp32_dest_acc_en=false"):
        run_dsa(q, k, w, 128, device, program_config=cfg, compute_kernel_config=ckc)


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="perf test - run locally")
@pytest.mark.parametrize("case_id, heads", GLX_CASES, ids=GLX_IDS)
def test_indexer_score_perf(device, case_id, heads):
    """Wall-clock latency per op for the GLX shape at the fullest causal rank (sp7), bfp8 k. Host-dispatched
    single-op latency (includes enqueue overhead; use tracy for pure device time). Logged ms is the signal;
    the assert is a coarse hang/regression guard (board-dependent)."""
    warmup_iters, measured_iters = 3, 20
    chunk_start = GLX_HISTORY + 7 * GLX_SQ
    cfg = glx_config(heads)
    q, k, w = make_inputs(heads, GLX_DIM, GLX_SQ, GLX_T)
    q_dev = to_device(q, device)
    k_dev = to_device(k, device, dtype=ttnn.bfloat8_b)
    w_dev = to_device(w, device)

    def run_once():
        return ttnn.experimental.indexer_score_dsa(q_dev, k_dev, w_dev, chunk_start_idx=chunk_start, program_config=cfg)

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
# sp_rank 7 perf helpers (tracy device profiler; no accuracy check). math_util = matmul FLOPs /
# (cores x device cycles x matmul peak); duration from tracy, FLOPs from shape. Consumed by the DSA
# band checks in test_indexer_score_math_util (run with INDEXER_SCORE_PERF_CHECKS=1).
# ---------------------------------------------------------------------------
SP7_CHUNK_START = GLX_HISTORY + 7 * GLX_SQ  # fullest causal case (99.5% valid)

# Blackhole matmul peak (tests/nightly/sdpa_perf_utils.py): 4096 mm FLOP/cycle/core at LoFi, halved per
# extra math-fidelity phase. The band checks measure the deployed HiFi2 path (bf16 q, bfp8 k).
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


# perf_impl inner targets profiled by tracy, at the deployed HiFi2 dtypes (bf16 q + bfp8 k). One node per
# model deployment; test_indexer_score_math_util spawns them by id, so the ids must stay in lockstep.
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="perf test - run locally with tracy")
@pytest.mark.parametrize("case_id, heads", [("glm5", 8), ("dsv32", 16)], ids=["glm5", "dsv32"])
def test_indexer_score_sp7_perf_impl(device, case_id, heads):
    """Inner test profiled by tracy: a few indexer_score_dsa ops at GLX sp_rank 7 (bf16 q, bfp8 k). No
    accuracy check."""
    q, k, w = make_inputs(heads, GLX_DIM, GLX_SQ, GLX_T)
    q_dev = to_device(q, device, dtype=ttnn.bfloat16)
    k_dev = to_device(k, device, dtype=ttnn.bfloat8_b)
    w_dev = to_device(w, device)
    cfg = glx_config(heads)
    for _ in range(5):  # tracy logs each op's device duration; the outer test takes the min
        ttnn.experimental.indexer_score_dsa(
            q_dev, k_dev, w_dev, chunk_start_idx=SP7_CHUNK_START, program_config=cfg
        ).deallocate()
    ttnn.synchronize_device(device)


# Short-query-chunk perf: the SAME two deployments (GLM5, DSv32) on the SAME keys (T=56320), but resharded
# TP=1 / SP=32 instead of the deployed TP=4 / SP=8. Two things change together:
#   - SP=32 splits the 5120-query prefill chunk 32 ways -> 160 q/device (vs GLX_SQ=640 at SP=8): a SHORT chunk.
#   - TP=1 puts every index head on the one device (no tensor-parallel head split), so the per-device head
#     count is 4x the deployed TP=4 count: GLM5 8h -> 32h (glm5_tp1), DSv32 16h -> 64h (dsv32_tp1).
# So these are exactly GLM5/DSv32 at TP=1/SP=32 -- many heads AND a short sequence. At QC=1 the 160-query
# chunk is 5 q-groups -> only 5 of the 10 grid rows, which is what the block-split scheduler fills
# (num_blocks=2 -> 110 cores). chunk_start at the end of the keys = fullest causal. Resharding preserves the
# per-device heads x queries product (4x heads x 1/4 queries), so glm5_tp1 has essentially the same matmul
# FLOPs and wall-clock as the deployed glm5 8h/640 once the grid is filled.
SHORT_SQ = 160  # 5 q-tile-rows: 5120-query prefill chunk / SP=32
SHORT_CHUNK_START = GLX_T - SHORT_SQ  # 56160: queries at the end of the all-gathered keys (fullest causal)


def short_valid_tiles():
    """Causal-valid output tiles V for the 160-query chunk at SHORT_CHUNK_START: sum_s min(Tt, chunk_t+s+1)."""
    chunk_t = SHORT_CHUNK_START // 32
    tt_tiles = GLX_T // 32
    sqt = SHORT_SQ // 32
    return sum(min(tt_tiles, chunk_t + s + 1) for s in range(sqt))


def short_config(heads):
    """QC=1 (q_chunk=32): the 160-query chunk makes 5 q-groups, half the 10-row grid -> block-split fills it.
    head_group_size=0 keeps all heads resident; k_chunk is the largest that fits L1 at this head count -- the
    4x head count from TP=1 leaves no room for the deployed KC, so KC=8 (k_chunk=256) at <=32 heads (glm5_tp1)
    and the smaller KC=4 (k_chunk=128) at 64 heads (dsv32_tp1). KC does not change the core count here
    (band_count stays >> cols, so num_blocks=2 -> 110 cores either way)."""
    return ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=256 if heads <= 32 else 128, head_group_size=0)


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="perf test - run locally with tracy")
@pytest.mark.parametrize("case_id, heads", [("dsv32_tp1", 64), ("glm5_tp1", 32)], ids=["dsv32_tp1", "glm5_tp1"])
def test_indexer_score_short_seq_perf_impl(device, case_id, heads):
    """Inner test profiled by tracy: GLM5/DSv32 resharded TP=1/SP=32 -- a SHORT 160-query chunk (QC=1) that
    under-fills the grid, so the block-split scheduler replicates each q-group across num_blocks=2 row-blocks
    (110 cores). bf16 q + bfp8 k. No accuracy check."""
    q, k, w = make_inputs(heads, GLX_DIM, SHORT_SQ, GLX_T)
    q_dev = to_device(q, device, dtype=ttnn.bfloat16)
    k_dev = to_device(k, device, dtype=ttnn.bfloat8_b)
    w_dev = to_device(w, device)
    cfg = short_config(heads)
    for _ in range(5):  # tracy logs each op's device duration; the outer test takes the min
        ttnn.experimental.indexer_score_dsa(
            q_dev, k_dev, w_dev, chunk_start_idx=SHORT_CHUNK_START, program_config=cfg
        ).deallocate()
    ttnn.synchronize_device(device)


INDEXER_PERF_MARGIN = 0.02  # symmetric +/- 2% band on the expected math util (catches regressions AND speedups)


# ---------------------------------------------------------------------------
# Generalized-multicast regime coverage (accuracy): scheduler paths the knobs/shapes tests miss -- G >
# grid.y (groups phase-stacked onto rows), prime G (rows_used==1, k-mcast off), uneven k-band split,
# partial last band, and phase-stacking under streaming. Same exact -inf + PCC check as the deployments.
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
    """Streaming (HB<Hi) q-mcast with an UNEVEN k-band split across a grid row -- a naive per-output-tile
    q-mcast would deadlock (cores in a row issue different q-read counts). The phantom-band pad to
    max_bands keeps the q-mcast count uniform; this pins no-hang + exact -inf + PCC. U=23 is prime, so the
    band split is uneven on any Blackhole grid width."""
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


# ---------------------------------------------------------------------------
# Block-split grid fill (accuracy): short sequences (group_count < grid.y) leave grid rows idle, so each
# q-group is replicated across num_blocks row-blocks with its band range split across them (a band-chunk
# per block). This is the path with BOTH group_rows>1 (per-block k-mcast is a contiguous vertical rect)
# AND num_blocks>1 (two-or-more mcast rectangles per column, disjoint output columns, no reduce) -- the
# genmcast prime case only exercises group_rows==1 (k-mcast off). Same exact -inf + PCC + neg-gate check.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "heads, dim, sq, t, chunk_start, q_chunk, k_chunk, head_group",
    [
        (8, 128, 160, 2816, 1024, 32, 64, 0),  # G=5 -> group_rows=5, num_blocks=2 (target 64h/160 shape)
        (8, 128, 64, 3584, 1536, 32, 64, 0),  # G=2 -> group_rows=2, num_blocks=5 (deep split, 2 rows/block)
        (8, 128, 96, 2176, 768, 32, 64, 0),  # G=3 -> group_rows=3, num_blocks=3 (9 rows used)
        (8, 128, 160, 2880, 1024, 32, 128, 0),  # G=5, num_blocks=2 + KC not dividing Tt (partial last band)
    ],
    ids=["fill_5x2", "fill_2x5", "fill_3x3", "fill_5x2_partial"],
)
def test_indexer_score_block_split_fill(device, heads, dim, sq, t, chunk_start, q_chunk, k_chunk, head_group):
    """Under-filled short sequences split each q-group's bands across num_blocks row-blocks to use the idle
    rows. Asserts the shape really hits the block-split-with-k-mcast regime (group_rows>1 AND num_blocks>1),
    then checks exact causality + PCC. The scheduler/mcast change is head-independent, so heads=8 (L1-safe)
    exercises it fully."""
    grid = device.compute_with_storage_grid_size()
    QC, KC = q_chunk // 32, k_chunk // 32
    group_count = (sq // 32) // QC
    band_count = ((t // 32) + KC - 1) // KC
    group_rows = max(d for d in range(1, min(group_count, grid.y) + 1) if group_count % d == 0)
    cols_used = min(band_count, grid.x)
    num_blocks = max(1, min(grid.y // group_rows, band_count // cols_used))
    # Precondition: this is the target regime -- per-block k-mcast (>1 row/block) AND >1 band-row-block.
    assert group_rows > 1, f"need a multi-row block for k-mcast: group_rows={group_rows}"
    assert num_blocks > 1, f"need block replication: num_blocks={num_blocks}"

    _run_and_check(device, heads, dim, sq, t, chunk_start, q_chunk, k_chunk, head_group)


# ==============================================================================================
# Multi-device (QuietBox, 4 BH) tests: PER-DEVICE chunk_start derived from the mesh coordinate.
# ==============================================================================================
# Use the `mesh_device` fixture (auto-skips on a single chip). A SINGLE mesh dispatch where each device is
# a different SP rank: chunk_start = chunk_start_idx + r*Sq (r = linearized index along cluster_axis), one
# hash-excluded program for all. Two layouts:
#   - 1D SP=4 (flat mesh): history 25600 + chunk 4*640 -> T 28160; cluster_axis unset (linear order).
#   - 2D SP=2 x TP=2:      history 25600 + chunk 2*640 -> T 26880; cluster_axis = SP axis, heads split.
# Functional only (exact -inf map + PCC >= 0.999 per SP rank).

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
        refs.append(indexer_score_dsa_ref(q_g[:, :, sl, :], k_g, w_g[:, :, sl, :], history + sp * QB_SQ))
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
    out = ttnn.experimental.indexer_score_dsa(q_dev, k_dev, w_dev, program_config=glx_config(heads))
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
        ttnn.experimental.indexer_score_dsa(
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
    out = ttnn.experimental.indexer_score_dsa(
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


# ==============================================================================================
# Multichip MSA (MiniMax M3): the same per-device chunk_start mechanism as the DSA QB tests, through the
# indexer_score_msa frontend (raw dot, constant 1/sqrt(d) scale, no weights). num_groups=1 head-sums into
# one [1,1,Sq,T] plane; the 2x2 case splits heads across TP and sums the partials back. Functional only.
M3_QB_HEADS = 4  # MiniMax M3 sparse_num_index_heads
M3_QB_SCALE = QB_DIM**-0.5  # 1/sqrt(d), the M3 indexer scale (folded into the constant gate)


def _msa_per_sp_ref(q_g, k_g, sp_count, history):
    """MSA reference: each SP rank's raw-dot, scale-gated, head-summed score (chunk_start = history + sp*Sq),
    concatenated along seq. No learned gates -> a constant M3_QB_SCALE gate, raw dot (no relu)."""
    w = torch.full((1, q_g.shape[1], QB_SQ, 1), M3_QB_SCALE, dtype=torch.bfloat16)
    refs = []
    for sp in range(sp_count):
        sl = slice(sp * QB_SQ, (sp + 1) * QB_SQ)
        refs.append(indexer_score_msa_ref(q_g[:, :, sl, :], k_g, w, history + sp * QB_SQ))
    return torch.cat(refs, dim=2)


@pytest.mark.parametrize("mesh_device", [QB_SP], indirect=True)
def test_indexer_score_qb_msa_per_device_chunk_start(mesh_device):
    """MSA over 4 BH devices (SP=4), each deriving its own chunk_start from its coordinate (cluster_axis
    unset -> linear order). Raw dot + constant scale, num_groups=1 -> one head-summed [1,1,Sq,T] plane."""
    q_g, k_g, _ = _global_inputs(M3_QB_HEADS, QB_CHUNK, QB_T, seed=42)
    shard = ttnn.ShardTensorToMesh(mesh_device, dim=2)
    q_dev = _to_mesh(mesh_device, q_g, ttnn.bfloat16, shard)
    k_dev = _to_mesh(mesh_device, k_g, ttnn.bfloat8_b, ttnn.ReplicateTensorToMesh(mesh_device))

    # chunk_start_idx OMITTED -> the op deduces base = T - sp_ring*Sq = QB_HISTORY, then device r gets
    # base + r*Sq (r = linearized index, cluster_axis unset). The constant gate is synthesized per-device.
    out = ttnn.experimental.indexer_score_msa(
        q_dev, k_dev, scale=M3_QB_SCALE, num_groups=1, program_config=glx_config(M3_QB_HEADS)
    )
    out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2))

    ref = _msa_per_sp_ref(q_g, k_g, QB_SP, QB_HISTORY)
    assert_indexer_match(out_t, ref, QB_CHUNK, QB_T, check_neg=True)


@pytest.mark.parametrize("mesh_device", [(QB2_SP, QB2_TP)], ids=["2x2"], indirect=True)
def test_indexer_score_qb_msa_sp2_tp2(mesh_device):
    """MSA over a 2x2 mesh: chunk_start derived per-device along cluster_axis (SP), constant across TP; the
    M3 index heads split across TP. Each device head-sums its half (num_groups=1); the test sums the TP
    partials (the all-reduce) and validates per SP rank against its own raw-dot, own-chunk_start reference."""
    q_g, k_g, _ = _global_inputs(M3_QB_HEADS, QB2_CHUNK, QB2_T, seed=42)

    # q: seq (dim 2) sharded along the SP axis, heads (dim 1) along the TP axis. k replicated.
    mesh_shape = tuple(mesh_device.shape)
    q_dims = _axis_dims(sp_dim=2, tp_dim=1)
    shard_q = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=q_dims)
    q_dev = _to_mesh(mesh_device, q_g, ttnn.bfloat16, shard_q)
    k_dev = _to_mesh(mesh_device, k_g, ttnn.bfloat8_b, ttnn.ReplicateTensorToMesh(mesh_device))

    # chunk_start_idx OMITTED: base deduced as T - sp_ring*Sq, sp_ring = mesh extent along cluster_axis (2,
    # the SP axis). Device (sp, tp) gets chunk_start = base + sp*Sq -- identical for both TP devices at sp.
    out = ttnn.experimental.indexer_score_msa(
        q_dev,
        k_dev,
        cluster_axis=QB2_SP_AXIS,
        scale=M3_QB_SCALE,
        num_groups=1,
        program_config=glx_config(M3_QB_HEADS // QB2_TP),  # per-device head count
    )
    out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=q_dims))
    out_t = out_t.float().sum(dim=1, keepdim=True)  # TP all-reduce: sum the head-partials

    ref = _msa_per_sp_ref(q_g, k_g, QB2_SP, QB_HISTORY)
    assert_indexer_match(out_t, ref, QB2_CHUNK, QB2_T, check_neg=True)


# MiniMax M3 MSA math-utilization perf check (tracy; no accuracy check). ONE device of an SP=8 x TP=4 mesh:
# TP=4 = one GQA group/device (num_groups=1, Hi=1, no cross-TP all-reduce); SP=8 shards the chunk (640
# q/device); KV all-gathered (50176 prefix + 5120 chunk = 55296 keys). Same FLOP model as the DSA util
# test (the scoring matmul is identical). Measured at the deployed fused config: block_size=128 with QC=2
# and the grid-aligned q/K multicast (see M3_T). With one index head the matmul is a small slice (DMA +
# block-pool dominate), so util is far below the 8/16-head numbers; the TP<4 fallback (num_groups>1) is higher.
# ---------------------------------------------------------------------------
M3_DIM = 128  # sparse_index_dim (== GLX_DIM)
M3_SQ = 640  # queries/device: 5120 prefill chunk / SP=8
M3_HISTORY = 50176  # ~50K cached prefix, tile- and k_chunk-aligned
M3_T = 56320  # 55296 real keys PADDED to 1760*32 = 11*160*32, so Tt=1760 divides the 11-wide BH grid -> the
# dense deal grid-aligns, enabling the q/K multicast. The 1024 padded keys are causally masked (future), so
# m3_valid_tiles is unchanged.
M3_CHUNK_START = M3_HISTORY + 7 * M3_SQ  # sp_rank 7 = fullest-causal device


def m3_valid_tiles():
    """Causal-valid output tiles V at the fullest (sp_rank 7) M3 device: sum_s min(Tt, chunk_t + s + 1)
    over this device's q-tile-rows (one index head, so no head factor)."""
    chunk_t = M3_CHUNK_START // 32
    tt_tiles = M3_T // 32
    sqt = M3_SQ // 32
    return sum(min(tt_tiles, chunk_t + s + 1) for s in range(sqt))


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="perf test - run locally with tracy")
def test_indexer_score_msa_m3_perf_impl(device):
    """Inner test profiled by tracy: a few indexer_score_msa ops for one SP=8 x TP=4 M3 device (1 group,
    640 q, 55296 kv, raw dot, scale=1/sqrt(d), bf16 q + bfp8 k). No accuracy check."""
    q, k, _ = make_inputs(1, M3_DIM, M3_SQ, M3_T)
    q_dev = to_device(q, device, dtype=ttnn.bfloat16)
    k_dev = to_device(k, device, dtype=ttnn.bfloat8_b)
    # Deployed config: block_size=128 fuses the per-128-key block max (the writer emits the tiny block-score
    # tensor, not the full ~70 MB score that made the unfused path memory-bound). QC=2 + grid-aligned q/K
    # multicast removes K-read redundancy; KC=32 (blocks_per_unit=8) is the largest pool unit that fits L1.
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=1024, head_group_size=0)
    for _ in range(5):  # tracy logs each op's device duration; the outer test takes the min
        ttnn.experimental.indexer_score_msa(
            q_dev,
            k_dev,
            chunk_start_idx=M3_CHUNK_START,
            scale=M3_DIM**-0.5,
            num_groups=1,
            block_size=128,
            program_config=cfg,
        ).deallocate()
    ttnn.synchronize_device(device)


# ---------------------------------------------------------------------------
# The math-utilization band checks (CI-gated by INDEXER_SCORE_PERF_CHECKS=1), all at the deployed HiFi2 dtypes
# (bf16 q + bfp8 k): the deployed TP=4/SP=8 shapes GLM5, DSv32 and MiniMax-M3 at sp_rank 7, plus the resharded
# TP=1/SP=32 grid-fill shapes glm5_tp1 and dsv32_tp1 (block-split, fullest causal). Each spawns its perf_impl
# under tracy, reads the min DEVICE KERNEL DURATION, computes math_util = matmul FLOPs / (cores x device
# cycles x matmul peak), and asserts it within +/- INDEXER_PERF_MARGIN of the value measured on a Blackhole
# dev board. mm_flops is a thunk so the shape-derived FLOP count is evaluated at run time.
# (M3 is a single index head, so its matmul is a small slice -- the block-pool dominates -- hence the much
# lower expected util than the multi-head DSA cases.)
# ---------------------------------------------------------------------------
_MATH_UTIL_CASES = [
    # (case_id, perf_impl_node_id, profiler_subdir, mm_flops_thunk, expected_util) -- HiFi2 (bf16 q, bfp8 k)
    (
        "glm5",
        "test_indexer_score_sp7_perf_impl[glm5]",
        "ttnn_indexer_score_sp7",
        lambda: indexer_mm_flops(sp7_valid_tiles(), 8),
        70.1,
    ),
    (
        "dsv32",
        "test_indexer_score_sp7_perf_impl[dsv32]",
        "ttnn_indexer_score_sp7",
        lambda: indexer_mm_flops(sp7_valid_tiles(), 16),
        76.1,
    ),
    (
        "minimax_m3",
        "test_indexer_score_msa_m3_perf_impl",
        "ttnn_indexer_score_msa_m3",
        lambda: m3_valid_tiles() * (32 * 32) * (2 * M3_DIM),
        43.55,
    ),
    # Block-split grid fill: GLM5/DSv32 resharded TP=1/SP=32 -- a short 160-query chunk (QC=1, 5 q-groups) the
    # scheduler spreads across num_blocks=2 row-blocks (110 cores); without the fill these would use only 55
    # cores at ~half the util. These guard the feature's headline shapes. glm5_tp1 (32h) preserves the deployed
    # glm5 8h/640 per-device work, so it matches that wall-clock once the grid is full; its util tracks DSv32
    # (shared KC=8, matmul-bound). dsv32_tp1 (64h) carries twice the heads.
    (
        "dsv32_tp1",
        "test_indexer_score_short_seq_perf_impl[dsv32_tp1]",
        "ttnn_indexer_score_short_seq",
        lambda: indexer_mm_flops(short_valid_tiles(), 64),
        77.31,
    ),
    (
        "glm5_tp1",
        "test_indexer_score_short_seq_perf_impl[glm5_tp1]",
        "ttnn_indexer_score_short_seq",
        lambda: indexer_mm_flops(short_valid_tiles(), 32),
        75.54,
    ),
]


@pytest.mark.skipif(
    os.environ.get("INDEXER_SCORE_PERF_CHECKS") != "1",
    reason="Set INDEXER_SCORE_PERF_CHECKS=1 to run (CI: ops perf tests job)",
)
@pytest.mark.parametrize(
    "case_id, perf_id, subdir, mm_flops_thunk, expected_util",
    _MATH_UTIL_CASES,
    ids=[c[0] for c in _MATH_UTIL_CASES],
)
def test_indexer_score_math_util(case_id, perf_id, subdir, mm_flops_thunk, expected_util):
    """Per-deployment HiFi2 (bf16 q, bfp8 k) matmul math utilization via tracy, asserted within +/-
    INDEXER_PERF_MARGIN: GLM5 / DSv32 / MiniMax-M3 at the deployed TP=4/SP=8, plus glm5_tp1 / dsv32_tp1 at the
    resharded TP=1/SP=32 grid-fill shapes. Spawns the case's perf_impl under the profiler and compares the
    achieved math_util to the expected value (measured on a BH dev board)."""
    from tracy.process_model_log import run_device_profiler
    from tests.nightly.sdpa_perf_utils import post_process_ops_log

    command = "pytest tests/ttnn/nightly/unit_tests/operations/experimental/test_indexer_score.py::" + perf_id
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
    peak = _MM_FLOPS_PER_CYCLE_PER_CORE["HiFi2"]
    cycles = duration_ns * _BH_CLOCK_GHZ
    utilization = (mm_flops_thunk() / (core_count * cycles * peak)) * 100 if core_count > 0 else 0.0

    lower = expected_util * (1 - INDEXER_PERF_MARGIN)
    upper = expected_util * (1 + INDEXER_PERF_MARGIN)
    logger.info(
        f"indexer_score math util {case_id} (HiFi2): duration={duration_ns / 1e6:.3f} ms, cores={core_count}, "
        f"math_util={utilization:.2f}% (expected {expected_util:.2f}%, band [{lower:.2f}, {upper:.2f}])"
    )
    assert lower <= utilization <= upper, (
        f"{case_id} math utilization {utilization:.2f}% outside band [{lower:.2f}, {upper:.2f}] "
        f"(expected {expected_util:.2f}%, margin +/- {INDEXER_PERF_MARGIN * 100:.1f}%)"
    )


# ---------------------------------------------------------------------------
# block-max-pool (block_size > 0): MiniMax M3 MSA block scoring. The op fuses the per-block max, so the
# output is [B, G, Sq, T/block_size]; the downstream topk picks per-group top-k BLOCKS. block_size==0 is
# byte-identical to before. Pooled-path constraints: T % bs == 0, k_chunk % bs == 0, k_chunk | T, and
# k_chunk/bs in {8,16,24,32} (16 B output-slice alignment). bs=128, k_chunk=1024 -> blocks_per_unit=8.
# ---------------------------------------------------------------------------
BLOCK_POOL_BS = 128  # MiniMax M3 sparse_block_size


@pytest.mark.parametrize("num_groups", [1, 4], ids=["g1", "g4"])
@pytest.mark.parametrize("k_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["k_bf16", "k_bfp8"])
def test_indexer_score_block_pool(device, k_dtype, num_groups):
    """block_size=128 block-max-pool on a small MSA shape. Small chunk_start -> some blocks fully future
    (-inf) and one straddles the causal boundary. Checked per group against indexer_score_msa_ref. 0.995 PCC
    floor: block-max amplifies the bf16 raw-dot error; the pool itself is pinned exact by the tests below."""
    heads, dim, sq, t = 4, GLX_DIM, 128, 2048
    chunk_start = 512  # leaves fully-future blocks for early queries + a straddling block
    scale = dim**-0.5
    q, k, _ = make_inputs(heads, dim, sq, t)
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=1024, head_group_size=0)
    out = run_msa(
        q,
        k,
        chunk_start,
        device,
        scale=scale,
        num_groups=num_groups,
        block_size=BLOCK_POOL_BS,
        k_dtype=k_dtype,
        program_config=cfg,
    )
    w_scale = torch.full((1, heads, sq, 1), scale, dtype=torch.bfloat16)
    ref = indexer_score_msa_ref(q, k, w_scale, chunk_start, num_groups, block_size=BLOCK_POOL_BS)
    assert_pooled_match(out, ref, num_groups, sq, t // BLOCK_POOL_BS, pcc_floor=0.995)


@pytest.mark.parametrize("k_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["k_bf16", "k_bfp8"])
@pytest.mark.parametrize("sp_rank", [0, 7], ids=["rank0", "rank7"])
def test_indexer_score_block_pool_m3(device, k_dtype, sp_rank):
    """MSA block scoring at GLX geometry: 4 GQA groups on one chip (TP<4 fallback), raw dot, scale gate,
    block_size=128 -> per-group block scores [1,4,640,440]."""
    heads, dim, sq, t = 4, GLX_DIM, GLX_SQ, GLX_T
    chunk_start = GLX_HISTORY + sp_rank * GLX_SQ
    scale = 1.0 / (dim**0.5)
    q, k, _ = make_inputs(heads, dim, sq, t)
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=1024, head_group_size=0)
    out = run_msa(
        q,
        k,
        chunk_start,
        device,
        scale=scale,
        num_groups=heads,
        block_size=BLOCK_POOL_BS,
        k_dtype=k_dtype,
        program_config=cfg,
    )
    w_scale = torch.full((1, heads, sq, 1), scale, dtype=torch.bfloat16)
    ref = indexer_score_msa_ref(q, k, w_scale, chunk_start, heads, block_size=BLOCK_POOL_BS)
    # 0.995 floor: block-max amplifies the bf16 raw-dot error; the exact pool logic is pinned by the -inf
    # map here and by test_indexer_score_block_pool_exact_vs_unpooled.
    assert_pooled_match(out, ref, heads, sq, t // BLOCK_POOL_BS, pcc_floor=0.995)


def test_indexer_score_block_pool_exact_vs_unpooled(device):
    """Pool exactness, free of matmul precision: block-max-pooling the op's OWN unpooled bf16 scores must
    equal the op's pooled output (both share the same matmul accumulator, so only the in-kernel reduce-MAX
    differs). Isolates the fused pool from the bf16 q.kT error that relaxes the fp32-reference comparison."""
    heads, dim, sq, t = 4, GLX_DIM, 128, 2048
    chunk_start = 512
    q, k, _ = make_inputs(heads, dim, sq, t)
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=1024, head_group_size=0)
    unpooled = run_msa(q, k, chunk_start, device, num_groups=heads, program_config=cfg)
    pooled = run_msa(q, k, chunk_start, device, num_groups=heads, block_size=BLOCK_POOL_BS, program_config=cfg)
    # torch max over the op's own [1,G,Sq,T] scores, with the same forced-local (+inf) the pooled path applies.
    ref = msa_block_max_pool(unpooled.float(), BLOCK_POOL_BS, chunk_start)
    masked = ref == float("-inf")
    assert torch.equal(pooled <= torch.finfo(torch.bfloat16).min, masked)
    # bf16 max is exact selection of identical values -> the visible block maxes must match bit-for-bit
    # (forced-local blocks are +inf on both sides: inf == inf under torch.equal).
    assert torch.equal(pooled[~masked].float(), ref[~masked])


# ---------------------------------------------------------------------------
# MSA persistent KV cache: cache_batch_idx (indexed slot) and kv_len (valid prefix) are the same runtime,
# hash-excluded pass-throughs the DSA frontend exposes; the device op + all 3 kernels are mode-agnostic for
# them. The g1 (Hi=1) shape drives MSA's fused-streaming K read -- the path that must apply the indexed-slot
# page offset (a wrong-slot read silently returns slot 0) and the runtime kv_len mask.
# ---------------------------------------------------------------------------
MSA_PERSIST = dict(dim=128, sq=64, t=256, chunk_start=128)


def _msa_scale_w(heads, sq, scale):
    """MSA's constant gate materialized for the reference (the op synthesizes it in-kernel)."""
    return torch.full((1, heads, sq, 1), scale, dtype=torch.bfloat16)


@pytest.mark.parametrize("num_groups", [1, 4], ids=["g1_fused", "g4"])
def test_indexer_score_msa_indexed_cache(device, num_groups):
    """MSA cache_batch_idx selects a slot of a shared [B,1,T,D] cache; every slot scores correctly AND
    switching slots does not recompile (the slot is hash-excluded). g1 (Hi=1) drives the fused-streaming K
    read -- the path that must add the indexed-slot page offset, so a wrong-slot read fails the per-slot
    reference. g4 exercises the same on the non-fused per-group-plane path."""
    c = MSA_PERSIST
    B, heads, scale = 3, num_groups, c["dim"] ** -0.5  # Hi == num_groups: g1 -> fused single head, g4 -> 4 planes
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=64, head_group_size=0)
    g = torch.Generator().manual_seed(17)
    q = torch.randn(1, heads, c["sq"], c["dim"], generator=g, dtype=torch.bfloat16)
    k_cache = torch.randn(B, 1, c["t"], c["dim"], generator=g, dtype=torch.bfloat16)  # slots differ
    q_dev, k_dev = to_device(q, device), to_device(k_cache, device)
    w_scale = _msa_scale_w(heads, c["sq"], scale)

    def run(b):
        return ttnn.to_torch(
            ttnn.experimental.indexer_score_msa(
                q_dev,
                k_dev,
                num_groups=num_groups,
                chunk_start_idx=c["chunk_start"],
                scale=scale,
                program_config=cfg,
                cache_batch_idx=b,
            )
        )

    def check(out, b):
        ref = indexer_score_msa_ref(q, k_cache[b : b + 1], w_scale, c["chunk_start"], num_groups)
        assert_grouped_match(out, ref, num_groups, c["sq"], c["t"])

    check(run(0), 0)
    entries = device.num_program_cache_entries()
    for b in range(1, B):
        check(run(b), b)
    assert device.num_program_cache_entries() == entries, "switching cache_batch_idx recompiled"


@pytest.mark.parametrize("num_groups", [1, 4], ids=["g1_fused", "g4"])
def test_indexer_score_msa_runtime_kv_len(device, num_groups):
    """MSA over an oversized T=512 buffer scored at several kv_len<=T: only [0,kv_len) is valid, matches the
    reference there, and growing kv_len does NOT recompile (kv_len is hash-excluded). g1 drives the
    fused-streaming K read (kv_len masking on the fused path); g4 the per-group-plane path."""
    heads, scale, dim, sq, t, chunk_start = num_groups, num_groups, 128, 64, 512, 0
    scale = dim**-0.5
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=64, head_group_size=0)
    g = torch.Generator().manual_seed(29)
    q = torch.randn(1, heads, sq, dim, generator=g, dtype=torch.bfloat16)
    k = torch.randn(1, 1, t, dim, generator=g, dtype=torch.bfloat16)
    q_dev, k_dev = to_device(q, device), to_device(k, device)
    w_scale = _msa_scale_w(heads, sq, scale)

    def run(kv_len):
        return ttnn.to_torch(
            ttnn.experimental.indexer_score_msa(
                q_dev,
                k_dev,
                num_groups=num_groups,
                chunk_start_idx=chunk_start,
                scale=scale,
                program_config=cfg,
                kv_len=kv_len,
            )
        )

    def check(out, kv_len):
        ref = indexer_score_msa_ref(q, k[:, :, :kv_len, :], w_scale, chunk_start, num_groups)
        assert_grouped_match(out[:, :, :, :kv_len], ref, num_groups, sq, kv_len)

    check(run(64), 64)  # most work units fully past kv_len
    entries = device.num_program_cache_entries()
    for kv_len in (128, 256, 512):
        check(run(kv_len), kv_len)
    assert device.num_program_cache_entries() == entries, "changing kv_len recompiled"


def test_indexer_score_msa_block_pool_kv_len(device):
    """block_size pooling + a block-aligned runtime kv_len: only blocks within the valid prefix are written
    and match the pooled reference there. Pins the pooled-path kv_len composition (the writer emits whole
    blocks; kv_len is guarded to a block boundary)."""
    heads, num_groups, dim, sq, t = 4, 4, GLX_DIM, 128, 2048
    chunk_start, scale = 512, GLX_DIM**-0.5
    kv_len = 1024  # block-aligned (8 * 128), < T=2048; causal window 512+128=640 <= kv_len
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=1024, head_group_size=0)
    q, k, _ = make_inputs(heads, dim, sq, t)
    out = run_msa(
        q,
        k,
        chunk_start,
        device,
        scale=scale,
        num_groups=num_groups,
        block_size=BLOCK_POOL_BS,
        program_config=cfg,
        kv_len=kv_len,
    )
    w_scale = _msa_scale_w(heads, sq, scale)
    ref = indexer_score_msa_ref(q, k[:, :, :kv_len, :], w_scale, chunk_start, num_groups, block_size=BLOCK_POOL_BS)
    nb = kv_len // BLOCK_POOL_BS
    assert_pooled_match(out[:, :, :, :nb], ref, num_groups, sq, nb, pcc_floor=0.995)


def test_indexer_score_msa_rejects_kv_len_not_block_aligned(device, expect_error):
    """With block_size>0 a runtime kv_len must be a multiple of block_size (whole blocks are written); a
    tile-aligned-but-not-block-aligned kv_len is rejected -- on a WARM cache too, since kv_len is re-validated
    on a program-cache hit."""
    heads, num_groups, dim, sq, t = 4, 4, GLX_DIM, 128, 2048
    chunk_start, scale = 512, GLX_DIM**-0.5
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=1024, head_group_size=0)
    q, k, _ = make_inputs(heads, dim, sq, t)
    q_dev, k_dev = to_device(q, device), to_device(k, device)

    # Warm the program cache with a block-aligned kv_len; the mis-aligned one then hits the SAME program.
    ttnn.experimental.indexer_score_msa(
        q_dev,
        k_dev,
        num_groups=num_groups,
        chunk_start_idx=chunk_start,
        scale=scale,
        block_size=BLOCK_POOL_BS,
        program_config=cfg,
        kv_len=1024,
    )
    with expect_error(RuntimeError, "multiple of block_size"):
        ttnn.experimental.indexer_score_msa(
            q_dev,
            k_dev,
            num_groups=num_groups,
            chunk_start_idx=chunk_start,
            scale=scale,
            block_size=BLOCK_POOL_BS,
            program_config=cfg,
            kv_len=1024 + 32,  # tile-aligned, not a multiple of 128
        )


def test_indexer_score_msa_indexed_cache_nd_sharded_k(device):
    """MSA indexed cache that is ND-sharded across DRAM banks (each [1,1,T,D] slot is one shard): the fused
    single-head reader (Hi=1) resolves the sharded banks through a TensorAccessor AND applies the indexed-slot
    page offset, so every slot still scores correctly. Mirrors the DSA ND-sharded indexed-cache test for MSA's
    fused-streaming K read (a dropped/wrong offset would silently return slot 0)."""
    c = MSA_PERSIST
    B, heads, scale = 2, 1, c["dim"] ** -0.5  # Hi=1 -> fused single-head read
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=64, head_group_size=0)
    g = torch.Generator().manual_seed(19)
    q = torch.randn(1, heads, c["sq"], c["dim"], generator=g, dtype=torch.bfloat16)
    k_cache = torch.randn(B, 1, c["t"], c["dim"], generator=g, dtype=torch.bfloat16)  # slots differ
    q_dev = to_device(q, device)
    k_mem = _nd_sharded_dram_config(device, rows_per_shard=c["t"])  # each [1,1,T,D] slot is one shard
    k_dev = ttnn.from_torch(k_cache, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, memory_config=k_mem)
    assert k_dev.memory_config().is_sharded()
    w_scale = _msa_scale_w(heads, c["sq"], scale)

    for b in range(B):
        out = ttnn.to_torch(
            ttnn.experimental.indexer_score_msa(
                q_dev,
                k_dev,
                num_groups=1,
                chunk_start_idx=c["chunk_start"],
                scale=scale,
                program_config=cfg,
                cache_batch_idx=b,
            )
        )
        ref = indexer_score_msa_ref(q, k_cache[b : b + 1], w_scale, c["chunk_start"], 1)
        assert_grouped_match(out, ref, 1, c["sq"], c["t"])


@pytest.mark.parametrize(
    "q_chunk, k_chunk, head_group",
    [
        (32, 32, 0),  # all heads resident, single-tile k chunks (per-column fallback path)
        (32, 128, 0),  # KC=4 chunked k: kv_len can land mid-chunk and zero whole trailing chunks
        (32, 32, 8),  # head streaming in groups of 8 (per-column accumulate_row_streaming path)
    ],
    ids=["fallback_kc1", "chunked_k_kc4", "stream_hb8"],
)
def test_indexer_score_msa_runtime_kv_len_compute_paths(device, q_chunk, k_chunk, head_group):
    """MSA runtime kv_len swept over the raw-dot compute paths (per-column fallback / chunked-k / head
    streaming) -- brings MSA to the 4-path parity of the DSA kv_len sweep. num_groups=1 (streaming and
    fallback are single-plane paths). Growing kv_len must NOT recompile (kv_len is hash-excluded)."""
    heads, dim, sq, t, chunk_start = 64, 128, 64, 512, 0  # oversized T=512
    scale = dim**-0.5
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=q_chunk, k_chunk_size=k_chunk, head_group_size=head_group)
    g = torch.Generator().manual_seed(31)
    q = torch.randn(1, heads, sq, dim, generator=g, dtype=torch.bfloat16)
    k = torch.randn(1, 1, t, dim, generator=g, dtype=torch.bfloat16)
    q_dev, k_dev = to_device(q, device), to_device(k, device)
    w_scale = _msa_scale_w(heads, sq, scale)

    def run(kv_len):
        return ttnn.to_torch(
            ttnn.experimental.indexer_score_msa(
                q_dev,
                k_dev,
                num_groups=1,
                chunk_start_idx=chunk_start,
                scale=scale,
                program_config=cfg,
                kv_len=kv_len,
            )
        )

    def check(out, kv_len):
        ref = indexer_score_msa_ref(q, k[:, :, :kv_len, :], w_scale, chunk_start, 1)
        assert_grouped_match(out[:, :, :, :kv_len], ref, 1, sq, kv_len)

    check(run(64), 64)  # most work units fully past kv_len
    entries = device.num_program_cache_entries()
    for kv_len in (128, 256, 512):
        check(run(kv_len), kv_len)
    assert device.num_program_cache_entries() == entries, "changing kv_len recompiled"


def test_indexer_score_msa_indexed_cache_rejects(device, expect_error):
    """MSA mirrors the DSA indexed-cache rejections: an out-of-range cache_batch_idx (>= B) is rejected on a
    WARM cache (re-validated on hit), and a multi-slot cache with NO cache_batch_idx is ambiguous and rejected.
    Confirms the MSA frontend forwards cache_batch_idx into the shared validation."""
    c = MSA_PERSIST
    B, heads, scale = 2, 1, c["dim"] ** -0.5
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=64, head_group_size=0)
    g = torch.Generator().manual_seed(21)
    q = torch.randn(1, heads, c["sq"], c["dim"], generator=g, dtype=torch.bfloat16)
    k_cache = torch.randn(B, 1, c["t"], c["dim"], generator=g, dtype=torch.bfloat16)
    q_dev, k_dev = to_device(q, device), to_device(k_cache, device)

    # Warm the program cache with a valid slot; the OOB slot then hits the SAME program and must still fail.
    ttnn.experimental.indexer_score_msa(
        q_dev, k_dev, num_groups=1, chunk_start_idx=c["chunk_start"], scale=scale, program_config=cfg, cache_batch_idx=0
    )
    with expect_error(RuntimeError, "cache_batch_idx"):
        ttnn.experimental.indexer_score_msa(
            q_dev,
            k_dev,
            num_groups=1,
            chunk_start_idx=c["chunk_start"],
            scale=scale,
            program_config=cfg,
            cache_batch_idx=B,
        )
    with expect_error(RuntimeError, "batch must be 1"):
        ttnn.experimental.indexer_score_msa(
            q_dev, k_dev, num_groups=1, chunk_start_idx=c["chunk_start"], scale=scale, program_config=cfg
        )


def test_indexer_score_msa_rejects_bad_kv_len(device, expect_error):
    """MSA mirrors the DSA base kv_len rejections (block_size=0): kv_len must be tile-aligned, within (0, T],
    and leave room for the causal window (chunk_start + Sq <= kv_len). Each violation fails on a WARM cache
    too (kv_len is hash-excluded and re-validated on a program-cache hit)."""
    heads, dim, sq, t, chunk_start = 1, 128, 64, 512, 0
    scale = dim**-0.5
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=64, head_group_size=0)
    g = torch.Generator().manual_seed(33)
    q = torch.randn(1, heads, sq, dim, generator=g, dtype=torch.bfloat16)
    k = torch.randn(1, 1, t, dim, generator=g, dtype=torch.bfloat16)
    q_dev, k_dev = to_device(q, device), to_device(k, device)

    # Warm with a valid kv_len; each bad one then hits the SAME program and must still fail on the hit.
    ttnn.experimental.indexer_score_msa(
        q_dev, k_dev, num_groups=1, chunk_start_idx=chunk_start, scale=scale, program_config=cfg, kv_len=128
    )
    for bad in (t + 32, 100, 32):  # above T / not tile-aligned / causal window (>= chunk_start+Sq=64) violated
        with expect_error(RuntimeError, "kv_len"):
            ttnn.experimental.indexer_score_msa(
                q_dev, k_dev, num_groups=1, chunk_start_idx=chunk_start, scale=scale, program_config=cfg, kv_len=bad
            )


def test_indexer_score_msa_indexed_cache_block_pool(device):
    """MSA indexed cache combined with block-max-pool: cache_batch_idx selects a slot of a shared cache and
    the pooled writer emits per-block scores for that slot. Exercises the slot page offset on the block-pool
    path (the kv_len+pool test uses a single-slot cache), matching M3's paged block-selection deployment."""
    heads, num_groups, dim, sq, t = 4, 4, GLX_DIM, 128, 2048
    B, chunk_start, scale = 2, 512, GLX_DIM**-0.5
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=1024, head_group_size=0)
    g = torch.Generator().manual_seed(23)
    q = torch.randn(1, heads, sq, dim, generator=g, dtype=torch.bfloat16)
    k_cache = torch.randn(B, 1, t, dim, generator=g, dtype=torch.bfloat16)  # slots differ
    q_dev, k_dev = to_device(q, device), to_device(k_cache, device)
    w_scale = _msa_scale_w(heads, sq, scale)

    def run(b):
        return ttnn.to_torch(
            ttnn.experimental.indexer_score_msa(
                q_dev,
                k_dev,
                num_groups=num_groups,
                chunk_start_idx=chunk_start,
                scale=scale,
                block_size=BLOCK_POOL_BS,
                program_config=cfg,
                cache_batch_idx=b,
            )
        )

    run(0)
    entries = device.num_program_cache_entries()  # one pooled program cached now
    for b in range(B):
        ref = indexer_score_msa_ref(q, k_cache[b : b + 1], w_scale, chunk_start, num_groups, block_size=BLOCK_POOL_BS)
        assert_pooled_match(run(b), ref, num_groups, sq, t // BLOCK_POOL_BS, pcc_floor=0.995)
    assert device.num_program_cache_entries() == entries, "switching cache_batch_idx recompiled"


@pytest.mark.parametrize(
    "block_size, k_chunk_size, blocks_per_unit",
    # blocks_per_unit = KC / block_tiles = k_chunk_size / block_size. Keep KC=32 (same as the passing
    # tests above, so L1 fits) and shrink block_size to push blocks_per_unit past 8.
    [
        (64, 1024, 16),  # block_tiles=2, KC=32 -> 16
        (32, 1024, 32),  # block_tiles=1, KC=32 -> 32 (max allowed)
    ],
    ids=["bpu16", "bpu32"],
)
@pytest.mark.parametrize("num_groups", [1, 4], ids=["g1", "g4"])
def test_indexer_score_block_pool_large_blocks_per_unit(device, num_groups, block_size, k_chunk_size, blocks_per_unit):
    """blocks_per_unit > 8 routes the in-kernel pool to the library reduce (compute_kernel_lib::reduce)
    instead of the batched custom reduce (the <=8 fast path). Exact-vs-unpooled pins the reduce bit-for-bit,
    free of bf16 matmul noise, confirming the fallback lands each block max in col 0 as the writer expects."""
    heads, dim, sq, t = 4, GLX_DIM, 128, 2048
    chunk_start = 512
    q, k, _ = make_inputs(heads, dim, sq, t)
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=k_chunk_size, head_group_size=0)
    unpooled = run_msa(q, k, chunk_start, device, num_groups=num_groups, program_config=cfg)
    pooled = run_msa(q, k, chunk_start, device, num_groups=num_groups, block_size=block_size, program_config=cfg)
    ref = msa_block_max_pool(unpooled.float(), block_size, chunk_start)
    masked = ref == float("-inf")
    assert torch.equal(pooled <= torch.finfo(torch.bfloat16).min, masked)
    assert torch.equal(pooled[~masked].float(), ref[~masked])


@pytest.mark.parametrize(
    "block_size, k_chunk_size, t, match",
    [
        # bs=48 is not a multiple of 32 -> rejected before the divisibility checks
        (48, 1024, 2048, "block_size 48 must be a multiple of 32"),
        # bs=128, KC=16 -> blocks_per_unit=4, not a multiple of 8 (16 B output-slice alignment)
        (128, 512, 2048, "to be a multiple of 8 so each unit"),
        # Tt=80 not divisible by KC=32 -> a partial last work unit
        (128, 1024, 2560, "to divide T 2560"),
    ],
    ids=["bs_not_tile_multiple", "slice_unaligned", "partial_unit"],
)
def test_indexer_score_block_pool_validation(device, expect_error, block_size, k_chunk_size, t, match):
    """The pooled-path constraints are rejected loudly (asserting the SPECIFIC FATAL fires) rather than
    silently producing a misaligned write -- `match` pins each case to its own guard."""
    heads, dim, sq = 4, GLX_DIM, 128
    q, k, _ = make_inputs(heads, dim, sq, t)
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=k_chunk_size, head_group_size=0)
    with expect_error(RuntimeError, match):
        run_msa(q, k, 512, device, num_groups=heads, block_size=block_size, program_config=cfg)


# ---------------------------------------------------------------------------
# Block-cyclic (per-SP-shard slab) K layout. Chunked prefill packs each SP shard's per-chunk slab
# (chunk_local = global_chunk/sp keys) back-to-back, so the SP-gathered [B,1,T,D] k is in a PERMUTED physical
# order and the reader reads it back in natural token order (invP per tile) -- scores (and the block-max-pool
# over token-contiguous blocks) come out correct. Interface matches ttnn.transformer.sparse_sdpa: the caller
# names the MESH AXIS the cache was striped over (block_cyclic_sp_axis, sp DERIVED from the mesh) and passes
# the per-shard chunk length (block_cyclic_chunk_local). The permutation is only non-trivial for sp > 1, so it
# is exercised on a REAL SP mesh (below), NOT simulated on one chip; the pure invP tile-math is unit-tested
# device-free.
# ---------------------------------------------------------------------------
def _slab_positions(ring_size, chunk, t):
    """P[r] = natural token position physically stored at slab row r. Physical row r (shard c = r//sll, local
    row lr = r%sll) holds natural token (lr//cl)*chunk + c*cl + (lr%cl), where sll = T/ring_size (per-shard
    local length), cl = chunk/ring_size (per-shard slab width). This is the layout the in-kernel invP inverts.
    Canonical (production) version: models/demos/deepseek_v3_d_p/tt/mla/utils.py::blockcyclic_positions; kept
    local here so this op unit test has no model dependency."""
    sll, cl = t // ring_size, chunk // ring_size
    c = torch.arange(ring_size).repeat_interleave(sll)
    lr = torch.arange(sll).repeat(ring_size)
    return (lr // cl) * chunk + c * cl + (lr % cl)


def _slab_inv_positions(ring_size, chunk, t):
    """invP[n] = physical slab row holding natural token n (inverse of _slab_positions). Row-granularity form
    of the in-kernel per-tile SLAB_KTILE remap: slab = n//chunk, rem = n%chunk, c = rem//cl, off = rem%cl ->
    r = c*sll + slab*cl + off (sll = t/ring_size, cl = chunk/ring_size)."""
    sll, cl = t // ring_size, chunk // ring_size
    n = torch.arange(t)
    slab = n // chunk
    rem = n - slab * chunk
    c = rem // cl
    off = rem - c * cl
    return c * sll + slab * cl + off


def _to_slab(k_nat, ring_size, chunk):
    """Permute a natural-order [1,1,T,D] k into its slab physical layout: k_slab[r] = k_nat[P[r]]."""
    t = k_nat.shape[2]
    P = _slab_positions(ring_size, chunk, t)
    k_slab = k_nat.clone()
    k_slab[0, 0] = k_nat[0, 0][P]
    return k_slab


@pytest.mark.parametrize("ring_size", [2, 4, 8], ids=["sp2", "sp4", "sp8"])
@pytest.mark.parametrize("chunk_local,t", [(32, 2048), (64, 4096), (128, 3072)], ids=["cl32", "cl64", "cl128"])
def test_indexer_score_slab_invp_math(ring_size, chunk_local, t):
    """Pure-function (device-free) check of the block-cyclic remap math the reader kernel bakes in: invP must
    be the exact inverse of the block-cyclic layout P over the whole cache, for every legal (sp, chunk, T).
    Locks the SLAB_KTILE arithmetic independent of any device; the end-to-end remap is validated on a real
    SP mesh below. (Requires T a whole number of global chunks and chunk divisible by sp.)"""
    chunk = ring_size * chunk_local  # global chunk = sp * per-shard chunk
    if t % chunk or (t // ring_size) % chunk_local:
        pytest.skip("layout must tile the cache evenly (whole global chunks, >=1 slab per shard)")
    P = _slab_positions(ring_size, chunk, t)
    invP = _slab_inv_positions(ring_size, chunk, t)
    ident = torch.arange(t)
    assert torch.equal(P[invP], ident), "P[invP(n)] != n -- invP is not the layout's inverse"
    assert torch.equal(invP[P], ident), "invP[P(r)] != r -- not a bijection"


# ---- Real-mesh block-cyclic remap: sp DERIVED from block_cyclic_sp_axis on an (sp, 1) mesh ----
# The remap is the identity at sp=1, so the natural->physical PERMUTATION arithmetic only runs on a real SP
# mesh. K is stored block-cyclic and REPLICATED; q/w are SP-sharded on seq (per-chip chunk_local rows). The
# reader remaps each logical K-tile so the score matches the contiguous-K, natural-order per-SP reference.
# Reuses the QB constants (QB_HISTORY % (QB_SP*QB_SQ) == 0 -> T is a whole number of global chunks, and
# T/QB_SP spans many slabs so the permutation is non-trivial).
@pytest.mark.parametrize("mesh_device", [(QB_SP, 1)], ids=["sp4"], indirect=True)
@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_indexer_score_qb_block_cyclic(mesh_device, case_id, heads):
    """DSA over a REAL SP=4 mesh with a block-cyclic K cache. sp is read from block_cyclic_sp_axis=0 (the mesh
    rows); block_cyclic_chunk_local = QB_SQ (per-chip seq). chunk_start OMITTED -> the slab path deduces base =
    T - global_chunk = QB_HISTORY, device r gets base + r*QB_SQ. Score must match the natural-order reference."""
    q_g, k_nat, w_g = _global_inputs(heads, QB_CHUNK, QB_T, seed=42)
    k_bc = _to_slab(k_nat, QB_SP, QB_CHUNK)  # global chunk = QB_SP * QB_SQ = QB_CHUNK
    mesh_shape = tuple(mesh_device.shape)
    shard = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=(2, None))  # seq on SP axis, repl. axis 1
    q_dev = _to_mesh(mesh_device, q_g, ttnn.bfloat16, shard)
    w_dev = _to_mesh(mesh_device, w_g, ttnn.bfloat16, shard)
    k_dev = _to_mesh(mesh_device, k_bc, ttnn.bfloat8_b, ttnn.ReplicateTensorToMesh(mesh_device))

    out = ttnn.experimental.indexer_score_dsa(
        q_dev,
        k_dev,
        w_dev,
        cluster_axis=0,  # SP axis (same axis the cache was striped over)
        block_cyclic_sp_axis=0,
        block_cyclic_chunk_local=QB_SQ,
        program_config=glx_config(heads),
    )
    out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=(2, 1)))
    ref = _per_sp_ref(q_g, k_nat, w_g, QB_SP, QB_HISTORY)
    assert_indexer_match(out_t, ref, QB_CHUNK, QB_T, check_neg=True)


# ---- Q sequence sharded across BOTH mesh axes via cluster_axis=None (block_cyclic_chunk_local == tp*q_isl) ----
# On a 2x2 mesh with K block-cyclic over ONE axis (sp=2, tp=2), Q's sequence is split across all 4 devices.
# cluster_axis=None ranks device (a,b) by its row-major position in the device list (a*B+b), so the flat
# linearization IS a row-major nested 2D seq shard: device r's q-row 0 sits at base + r*Sq. For a slab-aligned
# base every device stays inside its slab (no straddle), so the flat shard + block-cyclic remap reassembles to
# a plain contiguous natural-order scoring of the whole chunk. A NAMED cluster_axis is rejected (its rank would
# miss the second axis's offset).
@pytest.mark.parametrize("mesh_device", [(2, 2)], ids=["2x2"], indirect=True)
@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_indexer_score_qb_both_axes_seq(mesh_device, case_id, heads, expect_error):
    """Q seq sharded across both axes (all 4 devices) with cluster_axis=None: chunk_local == tp*q_isl (tp=2),
    which the guard allows only for cluster_axis=None. K block-cyclic over sp=2, aligned base -> no straddle,
    so the reassembled score equals the contiguous natural-order reference. Also asserts a NAMED cluster_axis
    (which would mis-rank) is still rejected."""
    sp, chunk_global, t = 2, 256, 512  # cl=128, Sq per device = 256/4 = 64 (2 tiles); 2 chunks -> >1 slab/shard
    q_g, k_nat, w_g = _global_inputs(heads, chunk_global, t, seed=42)
    k_bc = _to_slab(k_nat, sp, chunk_global)
    shard = ttnn.ShardTensorToMesh(mesh_device, dim=2)  # flat row-major over all 4 devices == None's device order
    q_dev = _to_mesh(mesh_device, q_g, ttnn.bfloat16, shard)
    w_dev = _to_mesh(mesh_device, w_g, ttnn.bfloat16, shard)
    k_dev = _to_mesh(mesh_device, k_bc, ttnn.bfloat8_b, ttnn.ReplicateTensorToMesh(mesh_device))
    kw = dict(block_cyclic_sp_axis=0, block_cyclic_chunk_local=chunk_global // sp, program_config=glx_config(heads))

    # cluster_axis=None -> Q linearized row-major over all 4 devices (both axes). chunk_start OMITTED ->
    # base = T - global_chunk = 256 (slab-aligned). Reassembled == contiguous natural scoring at base.
    out = ttnn.experimental.indexer_score_dsa(q_dev, k_dev, w_dev, cluster_axis=None, **kw)
    out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2))
    ref = indexer_score_dsa_ref(q_g, k_nat, w_g, t - chunk_global)
    assert_indexer_match(out_t, ref, chunk_global, t, check_neg=True)

    # A NAMED cluster_axis with tp*q_isl is rejected (rank would miss the second axis's seq offset).
    with expect_error(RuntimeError, "NAMED cluster_axis"):
        ttnn.experimental.indexer_score_dsa(q_dev, k_dev, w_dev, cluster_axis=0, **kw)


# ---- Mid-slab-boundary STRADDLE (drives the need for the straddle geometry, #48500 slice #2) ----
# A NON-slab-aligned chunk_start makes each device's queries cross a cache-slab boundary, so the causal
# diagonal must JUMP by (chunk_global - cl) at q-row (cl - offset). Scaled-down stand-in for the maxedge
# iter2 case in test_mla (iters_isl=[2560,2592,5120] -> chunk_start 5152, offset 32): here sp=2,
# chunk_global=1280, cl=640, chunk_start=704 (= cumulative 384+320) -> offset 64 (2 tiles), so q-tiles 18-19
# straddle. The CURRENT op has no straddle (linear diagonal) -> this FAILS until the straddle is carved in.
ST_CHUNK = 1280  # global chunk (tokens); per-shard cl = ST_CHUNK/sp
ST_CS = 704  # mid-slab chunk_start (704 % 640 = 64 offset); the 384+320 cumulative of the maxedge stand-in
ST_T = 3840  # cache length: whole chunks (3*1280) covering the fullest device's straddled window
ST_MSA_T = 5120  # block-pool straddle cache: 4*1280 whole chunks AND a whole number of k_chunk=1024 (bs=128)


def _straddle_ref(q_g, k_nat, w_g, sp, chunk_global, chunk_start, t_len):
    """Per-SP-rank reference over natural-order K, mirroring the update_padded_kv_cache WRITER rotation
    (== rotated_chip_positions): each token has a FIXED, ISL-independent block-cyclic home. Device r's
    local query row s sits at pos(r, s) = (lr // cl)*chunk_global + r*cl + (lr % cl) with lr = update_idxt(r)
    + s. This is NOT the linear chunk_start + r*Sq: a mid-slab chunk_start rotates which chip owns which
    block (boundary_chip), so a token's placement must not depend on the cumulative ISL. Only the boundary
    chip is mid-slab (its rows straddle a slab boundary); the others are block-aligned."""
    heads, gq = q_g.shape[1], q_g.shape[2]
    sq = gq // sp  # per-device query rows (== cl in the SP-only block-cyclic layout)
    cl = chunk_global // sp
    boundary_slab = chunk_start // chunk_global
    boundary_chip = (chunk_start // cl) % sp
    offset = chunk_start % cl
    refs = []
    for r in range(sp):
        update_idxt = (
            (boundary_slab + 1) * cl
            if r < boundary_chip
            else (boundary_slab * cl + offset if r == boundary_chip else boundary_slab * cl)
        )
        sl = slice(r * sq, (r + 1) * sq)
        qh, kh, wh = q_g[:, :, sl, :].float(), k_nat[:, 0].float(), w_g[:, :, sl, :].float()
        score = torch.zeros(1, sq, t_len)
        for h in range(heads):
            score += torch.relu(qh[:, h] @ kh.transpose(-2, -1)) * wh[:, h]
        lr = update_idxt + torch.arange(sq)
        pos = (lr // cl) * chunk_global + r * cl + (lr % cl)  # block-cyclic home (writer rotation)
        future = torch.arange(t_len).unsqueeze(0) > pos.unsqueeze(1)
        refs.append(score.masked_fill(future, float("-inf")).unsqueeze(1))
    return torch.cat(refs, dim=2)


def _straddle_msa_pooled_ref(q_g, k_nat, sp, chunk_global, chunk_start, t_len, scale, block_size):
    """Per-SP-rank straddled MSA block-pool reference: raw dot (NO relu), constant `scale` gate, head-summed,
    causal-masked over the STRADDLED natural position, then block-max-pooled with the forced-local +inf stamp
    on each query's OWN (straddled) block. Mirrors _straddle_ref but for the block-pool path the writer's
    forced-local straddle jump lives on -- concatenated per-SP-rank along seq. cl / chunk_global / block_size
    are block-aligned, so a slab boundary is also a block boundary and the jump moves whole blocks."""
    heads = q_g.shape[1]
    sq = q_g.shape[2] // sp
    cl = chunk_global // sp
    nb = t_len // block_size
    boundary_slab = chunk_start // chunk_global
    boundary_chip = (chunk_start // cl) % sp
    offset = chunk_start % cl
    refs = []
    for r in range(sp):
        # Writer rotation (== _straddle_ref / update_padded_kv_cache): each token's block-cyclic home is
        # ISL-independent, so device r starts at its true logical block (update_idxt), NOT chunk_start + r*Sq.
        update_idxt = (
            (boundary_slab + 1) * cl
            if r < boundary_chip
            else (boundary_slab * cl + offset if r == boundary_chip else boundary_slab * cl)
        )
        sl = slice(r * sq, (r + 1) * sq)
        qh, kh = q_g[:, :, sl, :].float(), k_nat[:, 0].float()
        score = torch.zeros(1, sq, t_len)
        for h in range(heads):
            score += (qh[:, h] @ kh.transpose(-2, -1)) * scale  # MSA: raw dot, no relu
        lr = update_idxt + torch.arange(sq)
        pos = (lr // cl) * chunk_global + r * cl + (lr % cl)  # block-cyclic home (writer rotation)
        future = torch.arange(t_len).unsqueeze(0) > pos.unsqueeze(1)
        pooled = score.masked_fill(future, float("-inf")).reshape(1, sq, nb, block_size).amax(dim=-1)
        pooled[:, torch.arange(sq), pos // block_size] = float("inf")  # forced-local: own block
        refs.append(pooled.unsqueeze(1))
    return torch.cat(refs, dim=2)


@pytest.mark.parametrize("mesh_device", [(2, 1)], ids=["sp2"], indirect=True)
@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_indexer_score_qb_straddle(mesh_device, case_id, heads):
    """Mid-slab-boundary straddle + block-cyclic chip rotation on a REAL SP mesh: block-cyclic K + a
    non-slab-aligned chunk_start (704) makes each device's queries straddle a slab boundary AND (when the
    start block index isn't a multiple of sp) rotates which chip owns which block. The causal diagonal must
    use the writer's rotation (rotated_chip_positions), not the linear chunk_start + r*Sq (sp2 here: cl=640,
    boundary_chip=1, offset=64). Geometry is sp-generic (sp derived from the mesh); also verified locally at
    sp=8 (cl=160, boundary_chip=4 mid-ring) to guard against overfitting to sp=2 -- kept at sp=2 in CI to
    avoid an 8-chip reservation. Scaled stand-in for the maxedge rotated-prefill case (test_mla).

    Also a PROGRAM-CACHE regression: chunk_start (and the derived per-device chunk_start_tiles / straddle)
    are hash-EXCLUDED runtime args, re-patched by override_runtime_arguments on a hit. Two chunk_starts with
    a DIFFERENT boundary_chip run through ONE cached program -- ST_CS (mid-slab: rotation + straddle) then 0
    (aligned: linear) -- so the second call must be a cache hit (no recompile) yet still correct, exercising
    the re-patch and not just create_at."""
    sp = mesh_device.shape[0]
    q_g, k_nat, w_g = _global_inputs(heads, ST_CHUNK, ST_T, seed=42)
    k_bc = _to_slab(k_nat, sp, ST_CHUNK)
    mesh_shape = tuple(mesh_device.shape)
    shard = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=(2, None))
    q_dev = _to_mesh(mesh_device, q_g, ttnn.bfloat16, shard)
    w_dev = _to_mesh(mesh_device, w_g, ttnn.bfloat16, shard)
    k_dev = _to_mesh(mesh_device, k_bc, ttnn.bfloat8_b, ttnn.ReplicateTensorToMesh(mesh_device))
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=128, head_group_size=0)

    def run(chunk_start):
        out = ttnn.experimental.indexer_score_dsa(
            q_dev,
            k_dev,
            w_dev,
            chunk_start_idx=chunk_start,
            cluster_axis=0,
            block_cyclic_sp_axis=0,
            block_cyclic_chunk_local=ST_CHUNK // sp,
            program_config=cfg,
        )
        out_t = ttnn.to_torch(
            out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=(2, 1))
        )
        ref = _straddle_ref(q_g, k_nat, w_g, sp, ST_CHUNK, chunk_start, ST_T)
        assert_indexer_match(out_t, ref, ST_CHUNK, ST_T, check_neg=True)

    mesh_device.enable_program_cache()
    run(ST_CS)  # miss: boundary_chip=1 (sp2), offset 64 -> rotation + straddle
    entries = mesh_device.num_program_cache_entries()
    run(0)  # hit: boundary_chip=0, offset 0 -> linear; must re-patch the geometry, not recompile
    assert (
        mesh_device.num_program_cache_entries() == entries
    ), "switching boundary_chip recompiled -- chunk_start / straddle must be hash-excluded runtime args"


@pytest.mark.parametrize("mesh_device", [(QB_SP, 1)], ids=["sp4"], indirect=True)
def test_indexer_score_qb_msa_block_cyclic(mesh_device):
    """MSA (raw dot, constant scale, num_groups=1) over a REAL SP=4 block-cyclic K cache: sp read from
    block_cyclic_sp_axis=0, the reader remaps each logical K-tile so the head-summed per-SP score matches the
    natural-order reference. The reader remap is block_size-independent (it presents natural-order K; pooling
    then runs over that), so block-max-pool-over-block-cyclic is covered by composition of this remap test and
    the single-chip contiguous pooled tests -- no separate pooled mesh case needed."""
    q_g, k_nat, _ = _global_inputs(M3_QB_HEADS, QB_CHUNK, QB_T, seed=42)
    k_bc = _to_slab(k_nat, QB_SP, QB_CHUNK)
    mesh_shape = tuple(mesh_device.shape)
    shard = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=(2, None))
    q_dev = _to_mesh(mesh_device, q_g, ttnn.bfloat16, shard)
    k_dev = _to_mesh(mesh_device, k_bc, ttnn.bfloat8_b, ttnn.ReplicateTensorToMesh(mesh_device))

    out = ttnn.experimental.indexer_score_msa(
        q_dev,
        k_dev,
        cluster_axis=0,
        scale=M3_QB_SCALE,
        num_groups=1,
        block_cyclic_sp_axis=0,
        block_cyclic_chunk_local=QB_SQ,
        program_config=glx_config(M3_QB_HEADS),
    )
    out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=(2, 1)))
    ref = _msa_per_sp_ref(q_g, k_nat, QB_SP, QB_HISTORY)
    assert_indexer_match(out_t, ref, QB_CHUNK, QB_T, check_neg=True)


@pytest.mark.parametrize("mesh_device", [(2, 1)], ids=["sp2"], indirect=True)
def test_indexer_score_qb_msa_block_cyclic_straddle(mesh_device):
    """Mid-slab-boundary straddle on the MSA BLOCK-POOL path (real sp=2 mesh): block-cyclic K + a non-slab-
    aligned chunk_start (704, offset 64) makes each device's queries straddle a slab boundary. The DSA straddle
    test above covers the compute-side causal diagonal (full-strip write); THIS pins the writer's block-pool
    forced-local +inf stamp, which must jump to the query's OWN block ACROSS the boundary -- a path the DSA
    (write_strip) test never exercises. Fails on the pre-straddle op (linear diagonal -> +inf stamped on the
    wrong block)."""
    sp = 2
    bs = BLOCK_POOL_BS  # 128; cl (640) / chunk_global (1280) / bs all block-aligned -> jump moves whole blocks
    q_g, k_nat, _ = _global_inputs(M3_QB_HEADS, ST_CHUNK, ST_MSA_T, seed=42)
    k_bc = _to_slab(k_nat, sp, ST_CHUNK)
    mesh_shape = tuple(mesh_device.shape)
    shard = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=(2, None))
    q_dev = _to_mesh(mesh_device, q_g, ttnn.bfloat16, shard)
    k_dev = _to_mesh(mesh_device, k_bc, ttnn.bfloat8_b, ttnn.ReplicateTensorToMesh(mesh_device))

    out = ttnn.experimental.indexer_score_msa(
        q_dev,
        k_dev,
        chunk_start_idx=ST_CS,  # explicit, mid-slab -> offset 64 -> straddle
        cluster_axis=0,
        scale=M3_QB_SCALE,
        num_groups=1,
        block_size=bs,
        block_cyclic_sp_axis=0,
        block_cyclic_chunk_local=ST_CHUNK // sp,
        program_config=ttnn.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=1024, head_group_size=0),
    )
    out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=(2, 1)))
    ref = _straddle_msa_pooled_ref(q_g, k_nat, sp, ST_CHUNK, ST_CS, ST_MSA_T, M3_QB_SCALE, bs)
    assert_pooled_match(out_t, ref, 1, ST_CHUNK, ST_MSA_T // bs, pcc_floor=0.995)


def test_indexer_score_rejects_partial_block_cyclic_args(device, expect_error):
    """block_cyclic_sp_axis and block_cyclic_chunk_local define the cache's layout and must be passed TOGETHER
    -- passing only one is a caller error (the per-shard width is undefined). Host-side guard, single chip."""
    heads, dim, sq, t = 64, GLX_DIM, 64, 512
    q, k, w = make_inputs(heads, dim, sq, t)
    q_dev, k_dev, w_dev = to_device(q, device), to_device(k, device), to_device(w, device)
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=128, head_group_size=0)
    for kwargs in [{"block_cyclic_sp_axis": 0}, {"block_cyclic_chunk_local": 256}]:
        with expect_error(RuntimeError, "both be set or both unset"):
            ttnn.experimental.indexer_score_dsa(q_dev, k_dev, w_dev, chunk_start_idx=0, program_config=cfg, **kwargs)
