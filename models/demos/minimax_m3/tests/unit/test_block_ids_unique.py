# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Do the MSA block-ids ever contain a DUPLICATE block per (group, query) row?

Motivation. ``sparse_sdpa_msa`` gathers the selected blocks by *linearly walking* the block-id
prefix (``for chunk in [0, n_active): block_id = idx_ptr[chunk]``) with NO dedup. The fp32 host
reference and the ``M3_CPU_SDPA`` ablation instead apply the selection through a boolean scatter
(``bm.scatter_(1, b, True)`` in ``tt/attention/msa.py::_cpu_fp32_sdpa``), which de-duplicates. So
if the device ever emits the same ``block_id`` twice in a row's active prefix, the device kernel
double-counts that block's keys in QK/softmax/PV while the reference counts it once — a small,
depth-compounding, precision-independent discrepancy that would be misattributed to bf16.

Code-reading established (verified in the kernels, not the comments):
  * ``indexer_score_msa`` force-locals the current block by OVERWRITING its pooled score with +inf
    (``writer_indexer_score.cpp``: ``POOL_POS_INF_BF16 = 0x7F80``), not by appending — so force-local
    alone cannot duplicate.
  * ``topk_large_indices`` marks ``-inf``-scored lanes with the sentinel ``0xFFFFFFFF``
    (``topk_large_indices/compute.cpp``).

The one link NOT closed by reading is whether the TopK-XL LLK can emit a duplicate index on TIED
scores. This test settles that by observation:
  1. ``test_block_ids_unique_real`` — real ``indexer_score_msa`` -> ``topk_large_indices`` at M3 dims;
     asserts per row: sentinels form a contiguous SUFFIX (the reader's binary search assumes this),
     the valid prefix has NO duplicate, the forced-local block appears exactly once, and the valid
     count matches the causal expectation.
  2. ``test_topk_large_indices_ties_unique`` — feeds hand-built all-tied block scores straight into
     ``topk_large_indices`` to probe the LLK tie behavior in isolation (does it repeat an index?).

DEVICE-GUARDED: opens the mesh device, so it is skipped unless ``RUN_BLOCK_IDS_CHECK=1`` is set (so
it never auto-runs in a suite while the galaxy is occupied). Runnable by the team:

    RUN_BLOCK_IDS_CHECK=1 pytest models/demos/minimax_m3/tests/unit/test_block_ids_unique.py
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
TOPK_BLOCKS = 16  # sparse_topk_blocks (also the k passed to topk_large_indices)
NUM_GROUPS = 1  # local KV heads at TP=4 (one shared index-k head, one index-q group)

SENT = 0xFFFFFFFF  # topk_large_indices sentinel (matches reader/_cpu_fp32_sdpa)


def _read_block_ids(bids_tensor):
    """Device block-ids tensor [1, G, S, TOPK] uint32 -> torch int64 [G, S, TOPK] with sentinels
    normalized to 0xFFFFFFFF (guards against a signed-int32 to_torch wrapping 0xFFFFFFFF -> -1)."""
    t = ttnn.to_torch(ttnn.get_device_tensors(bids_tensor)[0])[0].to(torch.int64)  # [G, S, TOPK]
    return t & 0xFFFFFFFF


def _check_rows(bids, nblk, chunk_start):
    """Assert per (group, query) row invariants over block-ids [G, S, TOPK].

    Returns a small stats dict for logging. Raises AssertionError on the first violated invariant.
    """
    G, S, topk = bids.shape
    dup_rows = 0
    suffix_violations = 0
    local_missing = 0
    total_valid = 0
    worst = None  # (g, s, row) for the first duplicate found, for triage

    for g in range(G):
        for s in range(S):
            row = bids[g, s]  # [TOPK] int64
            is_sent = row == SENT

            # (1) sentinels must be a contiguous SUFFIX (reader binary-search invariant).
            n_valid = int((~is_sent).sum().item())
            if not bool(is_sent[n_valid:].all()) or bool(is_sent[:n_valid].any()):
                suffix_violations += 1
                raise AssertionError(
                    f"[g={g} s={s}] sentinels are not a contiguous suffix: row={row.tolist()}"
                )

            valid = row[:n_valid]
            total_valid += n_valid

            # (2) valid block-ids must be in range.
            assert n_valid > 0, f"[g={g} s={s}] row has zero valid blocks: {row.tolist()}"
            assert int(valid.min().item()) >= 0 and int(valid.max().item()) < nblk, (
                f"[g={g} s={s}] valid block-id out of range [0,{nblk}): {valid.tolist()}"
            )

            # (3) NO duplicate in the valid prefix — the core check.
            uniq = int(torch.unique(valid).numel())
            if uniq != n_valid:
                dup_rows += 1
                if worst is None:
                    worst = (g, s, row.tolist())

            # (4) forced-local (current) block present exactly once.
            qpos = chunk_start + s
            local_block = min(qpos // BLOCK, nblk - 1)
            local_count = int((valid == local_block).sum().item())
            if local_count != 1:
                local_missing += 1

            # (5) causal expectation: finite blocks = those with block_start <= qpos, capped at topk.
            expected_valid = min(qpos // BLOCK + 1, topk)
            assert n_valid == expected_valid, (
                f"[g={g} s={s} qpos={qpos}] valid count {n_valid} != expected {expected_valid}; "
                f"row={row.tolist()}"
            )

    if dup_rows and worst is not None:
        g, s, row = worst
        raise AssertionError(
            f"DUPLICATE block-id found in {dup_rows}/{G * S} rows. First: g={g} s={s} row={row}"
        )
    assert local_missing == 0, f"forced-local block not present-exactly-once in {local_missing} rows"
    return {"rows": G * S, "dup_rows": dup_rows, "avg_valid": total_valid / (G * S)}


@pytest.mark.skipif(
    os.getenv("RUN_BLOCK_IDS_CHECK") != "1",
    reason="opens the mesh device; set RUN_BLOCK_IDS_CHECK=1 to run (kept device-free by default)",
)
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@pytest.mark.parametrize(
    "Sq, T",
    [
        (640, 5120),  # M3 SP shard (5120/8) x a 5-chunk context; nblk=40 > topk=16 -> real selection
        (128, 5120),  # short query window, same context
        (640, 10240),  # longer context, 10 k_chunks; nblk=80
    ],
    ids=["Sq640_T5120", "Sq128_T5120", "Sq640_T10240"],
)
def test_block_ids_unique_real(mesh_device, device_params, Sq, T, reset_seeds):
    """Real indexer_score_msa -> topk_large_indices at M3 dims: no duplicate block-id per row,
    sentinels a contiguous suffix, forced-local present once, causal valid-count exact."""
    assert T % K_CHUNK == 0, f"T={T} must be a multiple of k_chunk_size={K_CHUNK}"
    assert Sq % Q_CHUNK == 0, f"Sq={Sq} must be a multiple of q_chunk_size={Q_CHUNK}"
    assert T >= Sq, "single-shot prefill: full context >= query window"

    G = NUM_GROUPS
    scale = INDEX_DIM**-0.5
    chunk_start = T - Sq  # query window is the tail of the context (single-device, rank 0)
    nblk = T // BLOCK

    torch.manual_seed(0)
    # ~unit-RMS index vectors (post-norm/post-RoPE index_q/index_k are RMSNorm'd, so O(1) entries).
    iq = torch.randn(1, G, Sq, INDEX_DIM, dtype=torch.bfloat16)
    ik = torch.randn(1, 1, T, INDEX_DIM, dtype=torch.bfloat16)

    def to_dev(t):
        kwargs = {}
        if isinstance(mesh_device, ttnn.MeshDevice):
            kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(mesh_device)
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, **kwargs)

    block_scores = ttnn.experimental.indexer_score_msa(
        to_dev(iq),
        to_dev(ik),
        num_groups=G,
        chunk_start_idx=chunk_start,
        scale=scale,
        block_size=BLOCK,
        program_config=ttnn.IndexerScoreProgramConfig(q_chunk_size=Q_CHUNK, k_chunk_size=K_CHUNK, head_group_size=0),
        cluster_axis=None,
    )
    block_ids = ttnn.experimental.topk_large_indices(block_scores, k=TOPK_BLOCKS)  # [1, G, Sq, TOPK] uint32
    bids = _read_block_ids(block_ids)  # [G, Sq, TOPK] int64
    assert bids.shape == (G, Sq, TOPK_BLOCKS), f"unexpected block_ids shape {tuple(bids.shape)}"

    stats = _check_rows(bids, nblk, chunk_start)
    logger.info(
        f"[Sq={Sq} T={T}] block-ids OK: rows={stats['rows']} dup_rows={stats['dup_rows']} "
        f"avg_valid_per_row={stats['avg_valid']:.2f} (nblk={nblk}, topk={TOPK_BLOCKS})"
    )


@pytest.mark.skipif(
    os.getenv("RUN_BLOCK_IDS_CHECK") != "1",
    reason="opens the mesh device; set RUN_BLOCK_IDS_CHECK=1 to run (kept device-free by default)",
)
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@pytest.mark.parametrize("nblk", [16, 32, 40], ids=["nblk16", "nblk32", "nblk40"])
def test_topk_large_indices_ties_unique(mesh_device, device_params, nblk, reset_seeds):
    """Probe the TopK-XL LLK tie behavior in isolation: feed ALL-EQUAL block scores and assert the
    returned indices are still distinct (never a repeated block-id). This is the one link the kernel
    force-local overwrite + sentinel-marking code could not close by reading.

    A row of `nblk` identical finite scores is a maximal-tie: any top-k over it must still return k
    DISTINCT positions. If the LLK ever repeats an index here, the reader would double-count that
    block for every such query."""
    S = 128  # a full tile-row of queries, all with the same tied scores
    scores = torch.full((1, 1, S, nblk), 0.5, dtype=torch.bfloat16)  # maximal tie: every block equal

    kwargs = {}
    if isinstance(mesh_device, ttnn.MeshDevice):
        kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(mesh_device)
    scores_t = ttnn.from_torch(scores, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, **kwargs)

    block_ids = ttnn.experimental.topk_large_indices(scores_t, k=TOPK_BLOCKS)  # [1, 1, S, TOPK]
    bids = _read_block_ids(block_ids)  # [1, S, TOPK]

    dup_rows = 0
    worst = None
    for s in range(S):
        row = bids[0, s]
        # No sentinels here (all scores finite), so every lane must be a valid, DISTINCT block-id.
        assert int((row == SENT).sum().item()) == 0, f"[s={s}] unexpected sentinel in all-finite row: {row.tolist()}"
        assert int(row.min().item()) >= 0 and int(row.max().item()) < nblk, f"[s={s}] out-of-range id: {row.tolist()}"
        if int(torch.unique(row).numel()) != TOPK_BLOCKS:
            dup_rows += 1
            if worst is None:
                worst = (s, row.tolist())

    if dup_rows:
        s, row = worst
        raise AssertionError(
            f"TopK-XL repeated an index on tied scores in {dup_rows}/{S} rows (nblk={nblk}). "
            f"First: s={s} row={row}  -> reader would DOUBLE-COUNT this block."
        )
    logger.info(f"[nblk={nblk}] tie probe OK: all {S} rows returned {TOPK_BLOCKS} distinct block-ids")
