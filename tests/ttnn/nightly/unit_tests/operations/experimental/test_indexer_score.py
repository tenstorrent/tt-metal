# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the two lightning-indexer scorers that share one device op:

  indexer_score_dsa - DeepSeek-V3.2 DSA / GLM-5:
      score[b, 0, s, t] = sum_h relu(q[b,h,s,:] . k[b,t,:]) * w[b,h,s]
      ReLU + learned per-head gates, ALL heads summed into one selection row [B,1,Sq,T].

  indexer_score_msa - MiniMax M3 MSA:
      score[b, g, s, t] = sum_{h in group g} (q[b,h,s,:] . k[b,t,:]) * scale
      Raw dot (no relu), no learned gates (just a 1/sqrt(d) ``scale``), the Hi heads
      partitioned into ``num_groups`` GQA groups summed within each group (one output
      plane per group, no cross-group sum), optionally block-max-pooled (``block_size>0``)
      to [B,G,Sq,T/block_size] for the downstream per-group block top-k.

Both share the program factory + kernels (the flavour is compile-time args). Causality
from scalar ``chunk_start``: key ``t`` visible to query ``s`` iff ``t <= chunk_start + s``;
future columns/blocks are -inf.

Deployments, all Galaxy chunked prefill (50K history + 5K chunk = 55K all-gathered
keys; the 5K-query chunk is sharded SP=8 -> 640 queries/device):

    GLM5   -  8 heads (per-device DSA indexer)
    DSv32  - 16 heads (64-head DSA indexer split across TP=4; the op sums only its
             heads and the -inf mask is head-independent, so -inf survives the
             cross-TP sum)
    M3     -  MSA per-GQA-group (Hi=1/device at TP=4, or num_groups>1 on one chip),
             raw dot, no gates, block-max-pooled for the downstream block top-k

SP enters only via ``chunk_start``, so this is single-chip with ``sp_rank``
selecting the ring position.

Running:
  - Inner develop loop (fast, ~10 small-shape cases, no GLX-scale / no tracy perf):
        scripts/run_safe_pytest.sh -m dev_loop <this file>
    Covers a representative of each path: DSA ref (all-resident), MSA raw-dot ref, DSA!=MSA,
    per-group (ref + exact split), block-pool (ref + bit-exact), and a couple validation guards.
  - Pre-push gate (everything, incl. GLX-scale accuracy/determinism; perf/tracy still self-skip
    unless run locally / INDEXER_SCORE_PERF_CHECKS=1):
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


def indexer_score_ref(q, k, w, chunk_start, apply_relu=True):
    """Per-head fp32 accumulation (a full [Hi,Sq,T] tensor is many GB at GLX sizes).

    apply_relu=True  -> sum_h relu(q.kT) * w   (DeepSeek-V3.2 / GLM-5 lightning indexer)
    apply_relu=False -> sum_h (q.kT)    * w    (MiniMax M3 MSA: raw dot, scale folded into w)
    """
    b, hi, sq, _ = q.shape
    t = k.shape[2]
    q, k, w = q.float(), k.float(), w.float()
    score = torch.zeros(b, sq, t)
    for h in range(hi):
        qk = q[:, h] @ k[:, 0].transpose(-2, -1)
        score += (torch.relu(qk) if apply_relu else qk) * w[:, h]
    future = torch.arange(t).unsqueeze(0) > chunk_start + torch.arange(sq).unsqueeze(1)
    return score.masked_fill(future, float("-inf")).unsqueeze(1)


def indexer_score_grouped_ref(q, k, w, chunk_start, num_groups, apply_relu=True):
    """Per-GQA-group reference: partition the Hi heads into num_groups contiguous groups of Hi/num_groups
    and sum act(q.kT)*w WITHIN each group only -> [b, num_groups, sq, t]. num_groups==1 == indexer_score_ref.
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
            qk = q[:, h] @ k[:, 0].transpose(-2, -1)
            score += (torch.relu(qk) if apply_relu else qk) * w[:, h]
        planes.append(score.masked_fill(future, float("-inf")))
    return torch.stack(planes, dim=1)  # [b, num_groups, sq, t]


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

    q (srcB) and k (srcA) may be bfp8_b; weights stay bf16.
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
):
    """Run indexer_score_msa (raw dot, constant `scale` gate, per-group planes) and return torch.

    No weights tensor: M3 has no learned gates, only `scale` (run as a constant gate in-op).
    """
    out = ttnn.experimental.indexer_score_msa(
        to_device(q, device, dtype=q_dtype),
        to_device(k, device, dtype=k_dtype),
        chunk_start_idx=chunk_start,
        scale=scale,
        num_groups=num_groups,
        block_size=block_size,
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
    """Per-group [1,G,Sq,T] check: exact -inf map + PCC>=0.999 on visible scores (same bar as the single
    plane), so any cross-group leakage (a plane summing another group's heads) fails the PCC."""
    assert out.shape == (1, num_groups, sq, t)
    masked = ref == float("-inf")
    assert torch.equal(out <= torch.finfo(torch.bfloat16).min, masked)
    a, b = out[~masked].flatten().float(), ref[~masked].flatten().float()
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    assert pcc >= 0.999, f"PCC {pcc} < 0.999"


def block_max_pool_ref(scores, block_size, chunk_start):
    """Reference block-max-pool: max over each block_size-key block of the (already causal-masked) per-group
    scores [b, g, sq, t] -> [b, g, sq, t//block_size] (MiniMax M3 MSA block scores). A fully-future block is
    all -inf -> -inf; a causal-straddling block keeps only its visible tokens (the mask is applied before
    the max, as in the M3 reference). Then forced-local (M3 sparse_local_block=1): each query's own block is
    always selected, so its block score is set to +inf -- query s (global key position chunk_start + s) owns
    block (chunk_start + s) // block_size."""
    b, g, sq, t = scores.shape
    nb = t // block_size
    pooled = scores.reshape(b, g, sq, nb, block_size).amax(dim=-1)
    local = (chunk_start + torch.arange(sq)) // block_size  # each query's own block column [sq]
    pooled[:, :, torch.arange(sq), local] = float("inf")
    return pooled


def assert_pooled_match(out, ref, num_groups, sq, nblocks, pcc_floor=0.999):
    """Pooled [1,G,Sq,nblocks] check: exact -inf map (a fully-masked block stays -inf) + exact +inf map
    (each query's forced-local block) + PCC on the remaining visible block maxes. The -inf/+inf maps
    (checked first) pin the block->column mapping, causal masking, and forced-local exactly; block-max
    amplifies the bf16 per-token error (the max is biased toward the most positively-rounded token), so the
    visible-value PCC floor is relaxed for the large-T raw-dot deployment shape."""
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
    out = run_dsa(
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
    out = run_dsa(q, k, w, chunk_start, device, program_config=cfg)
    ref = indexer_score_ref(q, k, w, chunk_start)
    assert_indexer_match(out, ref, sq, t, check_neg=True)


@pytest.mark.parametrize(
    "q_chunk, k_chunk, head_group, chunk_start",
    [
        (32, 32, 1, 128),  # default: 1 head resident, streamed per output tile
        (32, 32, 8, 128),  # head streaming, groups of 8
        (32, 32, 32, 128),  # head streaming, 2 groups
        pytest.param(
            32, 32, 0, 128, marks=pytest.mark.dev_loop
        ),  # all heads resident, single-tile chunks (dev-loop DSA)
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


# Blackhole post-commit / sanity coverage reuses test_indexer_score_accuracy above: the CI entry in
# tests/pipeline_reorg/ttnn-tests.yaml selects its sp_rank-7 GLM5.1/DSv32 cases via `-k "accuracy and
# rank7"`. No separate post-commit test (that would re-run the same cases under nightly); post-commit just
# runs a subset of the nightly accuracy parametrization.


# ---------------------------------------------------------------------------
# MiniMax M3 MSA path (indexer_score_msa): raw dot product (no ReLU), no per-head gates, just a 1/sqrt(d)
# scale. At the group-aligned deployment (TP = num GQA groups) each device owns one index head (Hi=1), so
# num_groups=1 and the [1,1,Sq,T] output is that group's score row, fed to the downstream block-max top-k.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "q_chunk, k_chunk, head_group",
    [(32, 32, 0), pytest.param(32, 128, 0, marks=pytest.mark.dev_loop), (32, 32, 8)],
    ids=["fallback_kc1", "fullstrip_kc4", "stream_hb8"],
)
def test_indexer_score_msa_compute_paths(device, q_chunk, k_chunk, head_group):
    """MSA raw dot (no relu) over every compute path: the per-column fallback (KC=1), the head-major
    full-strip path (KC=4 all-resident), and head streaming (HB=8) -- so the no-relu packer config is
    exercised on each. 64-head MINI shape, num_groups=1, scale gate; checked like the deployments."""
    heads, dim, sq, t, chunk_start = MINI["heads"], MINI["dim"], MINI["sq"], MINI["t"], 128
    scale = dim**-0.5
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=q_chunk, k_chunk_size=k_chunk, head_group_size=head_group)
    q, k, _ = make_inputs(heads, dim, sq, t)
    out = run_msa(q, k, chunk_start, device, scale=scale, program_config=cfg)
    w_scale = torch.full((1, heads, sq, 1), scale, dtype=torch.bfloat16)  # MSA's constant gate = scale
    ref = indexer_score_ref(q, k, w_scale, chunk_start, apply_relu=False)
    assert_indexer_match(out, ref, sq, t)


@pytest.mark.dev_loop
def test_indexer_score_dsa_msa_differ(device):
    """DSA and MSA must compute genuinely different math: with the SAME constant gate, the only difference
    is DSA's relu. On inputs with negative dot products the visible scores must differ -- a guard against
    a frontend silently sharing the other's relu/no-relu compile-time path."""
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
@pytest.mark.parametrize("k_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["k_bf16", "k_bfp8"])
def test_indexer_score_m3_per_group(device, k_dtype, sp_rank):
    """MiniMax M3 MSA indexer, per GQA group as deployed at TP=4 (one index head per device, Hi=1).

    Raw dot product q.k scaled by 1/sqrt(d), no ReLU, no per-head gates (MSA's constant scale gate).
    GLX-style chunked prefill geometry (D=128, 640 q/device, 56320 keys, SP=8 ring position via
    chunk_start). Output [1,1,Sq,T] is the group's score row for the downstream block-max top-k. bfp8 k
    is the bandwidth-friendly deployed dtype.
    """
    heads, dim = 1, GLX_DIM  # one GQA group's single index head, head_dim 128
    sq, t = GLX_SQ, GLX_T
    chunk_start = GLX_HISTORY + sp_rank * GLX_SQ
    scale = dim**-0.5
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=512, head_group_size=0)
    q, k, _ = make_inputs(heads, dim, sq, t)
    out = run_msa(q, k, chunk_start, device, scale=scale, program_config=cfg, k_dtype=k_dtype)
    w_scale = torch.full((1, heads, sq, 1), scale, dtype=torch.bfloat16)
    ref = indexer_score_ref(q, k, w_scale, chunk_start, apply_relu=False)
    assert_indexer_match(out, ref, sq, t)


# ---------------------------------------------------------------------------
# indexer_score_msa num_groups > 1: per-GQA-group output [B, G, Sq, T], multiple groups resident on ONE
# chip (the TP < 4 fallback; the head-reduction is partitioned into G separate accumulators in-kernel,
# one output plane per group, NO cross-group summation). Requires all heads resident + full-strip (KC>=2).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "heads, num_groups",
    [pytest.param(4, 4, marks=pytest.mark.dev_loop), (8, 4), (8, 2), (64, 4)],
    ids=["g4_hog1", "g4_hog2", "g2_hog4", "g4_hog16"],
)
def test_indexer_score_multigroup(device, heads, num_groups):
    """num_groups>1 emits one plane per group, each summing only its Hi/G heads (no cross-group sum).

    Spans hog=1 (MiniMax M3: one index head per group), hog=2/4 (several index heads per group), and the
    64-head MINI geometry. Each plane is checked against the per-group reference (exact -inf map + PCC over
    distinct random heads), so a plane that summed the wrong heads would fail the PCC.
    """
    dim, sq, t, chunk_start = 128, 64, 256, 128
    scale = dim**-0.5
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=64, head_group_size=0)  # all resident, KC=2
    q, k, _ = make_inputs(heads, dim, sq, t)
    out = run_msa(q, k, chunk_start, device, scale=scale, num_groups=num_groups, program_config=cfg)
    w_scale = torch.full((1, heads, sq, 1), scale, dtype=torch.bfloat16)  # MSA's constant gate = scale
    ref = indexer_score_grouped_ref(q, k, w_scale, chunk_start, num_groups, apply_relu=False)
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
    ref = indexer_score_grouped_ref(q, k, w_scale, chunk_start, num_groups, apply_relu=False)
    assert_grouped_match(out, ref, num_groups, sq, t)


@pytest.mark.dev_loop
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
def test_indexer_score_multigroup_rejects(device, k_chunk, head_group, match):
    """num_groups>1 requires all heads resident + the full-strip path; reject KC<2 and head streaming."""
    heads, dim, sq, t = 8, 128, 64, 256
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=k_chunk, head_group_size=head_group)
    q, k, _ = make_inputs(heads, dim, sq, t)
    with pytest.raises(RuntimeError, match=match):
        run_msa(q, k, 128, device, num_groups=2, program_config=cfg)


@pytest.mark.dev_loop
def test_indexer_score_multigroup_rejects_indivisible(device):
    """num_groups must divide Hi -- e.g. 8 heads / 3 groups is rejected (uneven group sizes)."""
    heads, dim, sq, t = 8, 128, 64, 256
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=64, head_group_size=0)
    q, k, _ = make_inputs(heads, dim, sq, t)
    with pytest.raises(RuntimeError, match="and divide Hi 8"):
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
    ref = indexer_score_ref(q, k, w, chunk_start)
    assert_indexer_match(out, ref, sq, t, check_neg=True)


@pytest.mark.dev_loop
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
        ttnn.experimental.indexer_score_dsa(
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
# block-max-pool (block_size > 0): MiniMax M3 MSA block scoring. The op fuses the per-128-key block max
# into the score op, so the output is [B, G, Sq, T/block_size] instead of [B, G, Sq, T]; the downstream
# topk then picks per-group top-k BLOCKS. block_size==0 (every other test here) is byte-identical to before.
#
# Validation for the pooled path (Blackhole 16 B DRAM-write alignment + the full-strip pool):
#   T % block_size == 0, k_chunk_size % block_size == 0, k_chunk_size | T (no partial unit),
#   and k_chunk_size/block_size in {8,16,24,32} (each unit's row-major output slice is 16 B-aligned).
# block_size=128 (the M3 block) with k_chunk_size=1024 gives blocks_per_unit=8 -> satisfies all of these.
# ---------------------------------------------------------------------------
BLOCK_POOL_BS = 128  # MiniMax M3 sparse_block_size


@pytest.mark.parametrize("num_groups", [1, pytest.param(4, marks=pytest.mark.dev_loop)], ids=["g1", "g4"])
@pytest.mark.parametrize("k_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["k_bf16", "k_bfp8"])
def test_indexer_score_block_pool(device, k_dtype, num_groups):
    """block_size=128 block-max-pool on a small synthetic shape (MSA). chunk_start is small so some blocks
    are fully future (-inf) and one straddles the causal boundary -- the block max must keep only visible
    tokens and a fully-masked block must stay -inf. Checked per group against block_max_pool_ref.

    0.995 PCC floor (as in test_indexer_score_block_pool_m3): block-max amplifies the bf16 raw-dot
    matmul error (no ReLU clamp). The fused pool itself is pinned bit-exact by the -inf map here and by
    test_indexer_score_block_pool_exact_vs_unpooled, so this floor only bounds the matmul precision."""
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
    ref = block_max_pool_ref(
        indexer_score_grouped_ref(q, k, w_scale, chunk_start, num_groups, apply_relu=False), BLOCK_POOL_BS, chunk_start
    )
    assert_pooled_match(out, ref, num_groups, sq, t // BLOCK_POOL_BS, pcc_floor=0.995)


@pytest.mark.parametrize("k_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["k_bf16", "k_bfp8"])
@pytest.mark.parametrize("sp_rank", [0, 7], ids=["rank0", "rank7"])
def test_indexer_score_block_pool_m3(device, k_dtype, sp_rank):
    """MiniMax M3 MSA block scoring at GLX chunked-prefill geometry: 4 GQA groups on one chip (the TP<4
    fallback), raw dot, scale gate = 1/sqrt(d), block_size=128 -> per-group block scores [1,4,640,440].
    k_chunk_size=1024 (KC=32 | Tt=1760, blocks_per_unit=8)."""
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
    ref = block_max_pool_ref(
        indexer_score_grouped_ref(q, k, w_scale, chunk_start, heads, apply_relu=False), BLOCK_POOL_BS, chunk_start
    )
    # 0.995 floor: block-max amplifies the bf16 per-token matmul error over the full 56320-key raw-dot
    # reduction (no ReLU clamp). The exact pool logic is pinned by the -inf map here and, free of matmul
    # error, by test_indexer_score_block_pool_exact_vs_unpooled below.
    assert_pooled_match(out, ref, heads, sq, t // BLOCK_POOL_BS, pcc_floor=0.995)


@pytest.mark.dev_loop
def test_indexer_score_block_pool_exact_vs_unpooled(device):
    """Pool exactness, free of matmul precision: block-max-pooling the op's OWN unpooled bf16 scores must
    equal the op's pooled output. Both MSA runs share the identical bf16 matmul accumulator, so the only
    difference is the in-kernel reduce-MAX (+ col-0 extract + block-column write) -- which must reproduce
    a plain torch max over the same bf16 values exactly. Isolates the fused pool from the bf16 q.kT error
    that relaxes the fp32-reference comparison above."""
    heads, dim, sq, t = 4, GLX_DIM, 128, 2048
    chunk_start = 512
    q, k, _ = make_inputs(heads, dim, sq, t)
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=1024, head_group_size=0)
    unpooled = run_msa(q, k, chunk_start, device, num_groups=heads, program_config=cfg)
    pooled = run_msa(q, k, chunk_start, device, num_groups=heads, block_size=BLOCK_POOL_BS, program_config=cfg)
    # torch max over the op's own [1,G,Sq,T] scores, with the same forced-local (+inf) the pooled path applies.
    ref = block_max_pool_ref(unpooled.float(), BLOCK_POOL_BS, chunk_start)
    masked = ref == float("-inf")
    assert torch.equal(pooled <= torch.finfo(torch.bfloat16).min, masked)
    # bf16 max is exact selection of identical values -> the visible block maxes must match bit-for-bit
    # (forced-local blocks are +inf on both sides: inf == inf under torch.equal).
    assert torch.equal(pooled[~masked].float(), ref[~masked])


@pytest.mark.parametrize(
    "block_size, k_chunk_size, t, match",
    [
        # bs=48 is not a multiple of 32 -> rejected before the divisibility checks
        pytest.param(48, 1024, 2048, "block_size 48 must be a multiple of 32", marks=pytest.mark.dev_loop),
        # bs=128, KC=16 -> blocks_per_unit=4, not a multiple of 8 (16 B output-slice alignment)
        (128, 512, 2048, "to be a multiple of 8 so each unit"),
        # Tt=80 not divisible by KC=32 -> a partial last work unit
        (128, 1024, 2560, "to divide T 2560"),
    ],
    ids=["bs_not_tile_multiple", "slice_unaligned", "partial_unit"],
)
def test_indexer_score_block_pool_validation(device, block_size, k_chunk_size, t, match):
    """The pooled-path constraints are rejected loudly (asserting the SPECIFIC FATAL fires) rather than
    silently producing a misaligned write -- `match` pins each case to its own guard."""
    heads, dim, sq = 4, GLX_DIM, 128
    q, k, _ = make_inputs(heads, dim, sq, t)
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=k_chunk_size, head_group_size=0)
    with pytest.raises(RuntimeError, match=match):
        run_msa(q, k, 512, device, num_groups=heads, block_size=block_size, program_config=cfg)
