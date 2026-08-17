# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Intrinsic accuracy of the MSA indexer kernel ``ttnn.experimental.indexer_score_msa``.

GIVEN CORRECT INPUTS, how close is the device kernel's per-block scores to the fp32 reference?
This deliberately isolates the *kernel's* numerical accuracy from the block-cyclic cache-read
handling (the ``msa_sp_attention`` gather + the ops' in-kernel ``block_cyclic_*`` remap in
``tt/attention/msa.py``) that arranges ``index_k`` in the chunked path. Here ``index_k`` is built
directly and handed to the op, so any divergence is the kernel's own (bf16 dot accumulation,
bf16-scale gate fold, block-max-pool precision, causal-mask edges, k_chunk tiling).

We compare the FULL pre-selection score field (the indexer's output *before* top-k):
the op returns ``block_scores [1, num_groups, Sq, nblk]`` bf16 row-major, where each entry is the
max over a ``block_size``-key block of ``scale * (index_q @ index_k^T)`` with future keys masked
to ``-inf`` and the query's own (current) block forced to ``+inf``. The fp32 reference reproduces
exactly that (a verbatim port of ``reference/model.py:msa_block_selection`` up to the top-k step).

We report BOTH PCC and elementwise atol/rtol + max-abs-error + the worst ``(query, block)`` cell,
masking out the non-finite sentinels (``-inf`` future, ``+inf`` forced-local) which both sides set
identically and which would otherwise dominate any norm.

DEVICE-GUARDED: this opens the mesh device, so it is skipped unless ``RUN_INDEXER_ACCURACY=1`` is
set (so it never auto-runs in a suite while the galaxy is occupied). Runnable later by the team:

    RUN_INDEXER_ACCURACY=1 pytest models/demos/minimax_m3/tests/unit/test_indexer_score_msa_accuracy.py
"""

import os

import pytest
import torch
from loguru import logger

import ttnn

from ..test_factory import parametrize_mesh_with_fabric

# Real M3 MSA dims (configs/MiniMax-M3/config.json sparse_attention_config).
INDEX_DIM = 128  # sparse_index_dim
BLOCK = 128  # sparse_block_size
K_CHUNK = 1024  # program_config.k_chunk_size (T must be a multiple of this)
Q_CHUNK = 64  # program_config.q_chunk_size (Sq must be a multiple of this)
NUM_GROUPS = 1  # local KV heads at TP=4 (one shared index-k head, one index-q group)

# Accuracy gates. Justified in the file's analysis comment below and the report:
#   - bf16 dot over INDEX_DIM=128 + bf16 gate-scale fold (HiFi2, bf16 DEST acc, NO fp32 dest) gives
#     a per-score relative error a few x 2^-8. block-max-pool is exact (a selection, not an arith op),
#     so it does not add error; it can only *change which* element's error surfaces.
#   - We pick a rtol that comfortably passes the kernel-as-built and a tight-but-real atol scaled to
#     the score magnitude. These are starting gates; tighten once fp32-dest acc lands (see report).
PCC_MIN = 0.999
RTOL = 0.05
ATOL_SCALE = 0.02  # atol = ATOL_SCALE * (typical |score|); set per-case from the reference spread


def _msa_block_scores_ref(index_q, index_k, scale, block_size, chunk_start):
    """fp32 reference for the indexer's pre-selection block scores.

    Verbatim math of ``reference/model.py:msa_block_selection`` up to (and including) the block
    max-pool + forced-local stamp, but WITHOUT top-k (that happens in a separate op, not in
    ``indexer_score_msa``). Returns ``[G, Sq, nblk]`` fp32.

    ``chunk_start`` is the global position of query row 0 (so query row s attends keys [0, chunk_start+s]).
    This mirrors the kernel's per-device ``chunk_start_tiles`` causal offset.
    """
    G, S, T = index_q.shape[1], index_q.shape[2], index_k.shape[2]
    nblk = (T + block_size - 1) // block_size
    tpad = nblk * block_size

    scores = scale * (index_q.float() @ index_k.float().transpose(-1, -2))  # [1, G, S, T]
    kpos = torch.arange(T)
    qpos = torch.arange(S) + chunk_start  # global query positions
    scores = scores.masked_fill(kpos[None, None, None, :] > qpos[None, None, :, None], float("-inf"))
    if tpad > T:
        scores = torch.cat([scores, scores.new_full((1, G, S, tpad - T), float("-inf"))], dim=-1)

    bs = scores.view(1, G, S, nblk, block_size).max(-1).values  # block max-pool [1, G, S, nblk]
    # forced-local current block (+inf), matching the kernel's writer stamp (sparse_local_block).
    local = (qpos // block_size).clamp(max=nblk - 1)
    bs[0, :, torch.arange(S), local] = float("inf")
    return bs[0]  # [G, S, nblk]


def _compute_pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    return 1.0 if denom == 0 else (a @ b).item() / denom


@pytest.mark.skipif(
    os.getenv("RUN_INDEXER_ACCURACY") != "1",
    reason="opens the mesh device; set RUN_INDEXER_ACCURACY=1 to run (kept device-free by default)",
)
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@pytest.mark.parametrize(
    "Sq, T",
    [
        (640, 5120),  # M3 SP shard (5120/8) x a 5-chunk context; nblk=40
        (640, 10240),  # longer context, 10 k_chunks; stresses k-chunk tiling
        (128, 5120),  # short query, tiny grid
    ],
    ids=["Sq640_T5120", "Sq640_T10240", "Sq128_T5120"],
)
def test_indexer_score_msa_accuracy(mesh_device, device_params, Sq, T, reset_seeds):
    """atol/rtol + PCC of indexer_score_msa block scores vs the fp32 reference, at real M3 dims."""
    assert T % K_CHUNK == 0, f"T={T} must be a multiple of k_chunk_size={K_CHUNK}"
    assert Sq % Q_CHUNK == 0, f"Sq={Sq} must be a multiple of q_chunk_size={Q_CHUNK}"
    assert T >= Sq, "single-shot prefill: full context >= query window"

    G = NUM_GROUPS
    scale = INDEX_DIM**-0.5
    chunk_start = T - Sq  # query window is the tail of the gathered context (single-device, rank 0)
    nblk = T // BLOCK

    torch.manual_seed(0)
    # ~unit-RMS index vectors (post-norm/post-RoPE index_q/index_k are RMSNorm'd, so O(1) entries).
    iq = torch.randn(1, G, Sq, INDEX_DIM, dtype=torch.bfloat16)
    ik = torch.randn(1, 1, T, INDEX_DIM, dtype=torch.bfloat16)

    # fp32 reference from the SAME bf16 inputs the device sees (so we measure the kernel's error, not
    # the input-quantization error, which is common to both).
    ref = _msa_block_scores_ref(iq, ik, scale, BLOCK, chunk_start)  # [G, Sq, nblk] fp32

    def to_dev(t):
        kwargs = {}
        if isinstance(mesh_device, ttnn.MeshDevice):
            kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(mesh_device)
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, **kwargs)

    iq_t, ik_t = to_dev(iq), to_dev(ik)
    bs = ttnn.experimental.indexer_score_msa(
        iq_t,
        ik_t,
        num_groups=G,
        chunk_start_idx=chunk_start,
        scale=scale,
        block_size=BLOCK,
        program_config=ttnn.IndexerScoreProgramConfig(q_chunk_size=Q_CHUNK, k_chunk_size=K_CHUNK, head_group_size=0),
        seq_shard_axes=[],
    )
    dev = ttnn.to_torch(ttnn.get_device_tensors(bs)[0]).float()[0]  # [G, Sq, nblk]
    assert dev.shape == ref.shape, f"shape {tuple(dev.shape)} != {tuple(ref.shape)}"

    # ---- sentinel handling -----------------------------------------------------------------------
    # Both sides set future blocks to -inf and the forced-local block to +inf. Verify the sentinels
    # AGREE (a structural / causality check) then exclude them from the numeric error (they would
    # make atol/rtol meaningless). bf16 has exact +inf/-inf so this is a clean equality.
    ref_finite = torch.isfinite(ref)
    dev_finite = torch.isfinite(dev)
    sentinel_mismatch = int((ref_finite != dev_finite).sum().item())
    # Where one side is finite and the other isn't, OR both inf but different sign -> structural bug.
    both_inf = (~ref_finite) & (~dev_finite)
    sign_mismatch = int((both_inf & (torch.sign(ref) != torch.sign(dev))).sum().item())
    logger.info(
        f"[Sq={Sq} T={T}] sentinel finite-mask mismatches={sentinel_mismatch} "
        f"inf-sign mismatches={sign_mismatch} (both should be 0)"
    )

    # ---- numeric accuracy on the finite (real-score) cells ---------------------------------------
    mask = ref_finite & dev_finite
    n = int(mask.sum().item())
    assert n > 0, "no finite cells to compare"
    r = ref[mask]
    d = dev[mask]

    abs_err = (d - r).abs()
    rel_err = abs_err / r.abs().clamp(min=1e-6)
    max_abs = abs_err.max().item()
    max_rel = rel_err.max().item()
    typ = r.abs().median().item()  # typical score magnitude (sets a sensible atol)
    pcc = _compute_pcc(d[None], r[None])

    # locate the worst-abs-error (query, block) for triage.
    flat = abs_err.argmax().item()
    idx = torch.nonzero(mask, as_tuple=False)[flat].tolist()  # [g, q, blk]
    logger.info(
        f"[Sq={Sq} T={T}] finite cells={n}  PCC={pcc:.6f}  max|abs|={max_abs:.4e}  "
        f"max rel={max_rel:.4e}  median|score|={typ:.4e}  worst@(g,q,blk)={idx} "
        f"ref={r[flat].item():.5e} dev={d[flat].item():.5e}"
    )

    atol = ATOL_SCALE * max(typ, 1e-3)
    # Structural checks are hard requirements (causality must be bit-exact).
    assert sentinel_mismatch == 0, f"causal/forced-local sentinel structure diverged ({sentinel_mismatch} cells)"
    assert sign_mismatch == 0, f"{sign_mismatch} cells have opposite-sign infinities"
    # Numeric gates.
    assert pcc >= PCC_MIN, f"PCC {pcc:.6f} < {PCC_MIN}"
    assert torch.allclose(d, r, rtol=RTOL, atol=atol), (
        f"block scores exceed atol={atol:.4e}/rtol={RTOL}: max|abs|={max_abs:.4e} max rel={max_rel:.4e} "
        f"worst@(g,q,blk)={idx} ref={r[flat].item():.5e} dev={d[flat].item():.5e}"
    )
