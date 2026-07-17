# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
The REAL M3 MSA (sparse-attention) prefill pipeline at the DEPLOYED TP=4 x SP=8 per-device shape.

The merged ops are designed for M3's TP=4 deployment: 4 GQA groups / 4 index-q heads spread one per
device, all scoring against ONE shared index-k head (op requires k_shape[1]==1), num_groups=1. The 16
attention q heads in a device's group share that group's block selection. So one device's slice is:
  16 attention q heads (64/4) + 1 KV head (4/4) + 1 index-q head + 1 shared index-k head, num_groups=1.

Real chunked-prefill sizes (IS sends a 5120-token chunk + the prepared KV cache): under SP=8 the chunk
is 5120/8 = 640 query rows/device, attending to a ~5120-key context (the chunk; the cache grows it
across turns, but ~5k is the realistic per-step unit). 5120/128 = 40 blocks, top-16 -> real sparsity.

Pipeline (merged ops):
  indexer_score_msa(iq[1,1,S,128], ik[1,1,T,128], num_groups=1, block_size=128)
    -> block-max-pooled scores [1,1,S,T/128]  (op already force-locals the current block)
  host: top-16 blocks -> block-ids [1,1,S,16] int32
  sparse_sdpa_msa(q[1,16,S,128] RM, k/v[1,1,T,128] TILE, block-ids uint32 RM, block_size=128) -> [1,16,S,128]
  vs sparse_attention_ref_msa (the op's own block-level golden), same block-ids.
"""

import os
import sys

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

# Locate the shared sparse-SDPA test util relative to the repo root (5 levels up from this
# file), so the import works regardless of pytest's invocation cwd.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 5)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "tests/ttnn/unit_tests/operations/sdpa"))
from sparse_sdpa_msa_test_utils import sparse_attention_ref_msa  # noqa: E402

# Per-device TP=4 slice of M3: 64 q / 4 kv -> 16 q / 1 kv; 1 index-q head + 1 shared index-k; num_groups=1.
H_DEV, NKV_DEV, NIDX_DEV, NGROUPS = 16, 1, 1, 1
HEAD_DIM, BLOCK, TOPK_BLK = 128, 128, 16


@pytest.mark.parametrize("S,T", [(640, 5120)], ids=["chunk640_ctx5120"])  # 5120 chunk/SP=8; 40 blocks
def test_msa_prefill(device, S, T):
    torch.manual_seed(0)
    chunk_start = T - S  # the chunk is the tail of the ~5k context; query s sees keys [0 : chunk_start+s]

    q = torch.randn(1, H_DEV, S, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    k = torch.randn(1, NKV_DEV, T, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    v = torch.randn(1, NKV_DEV, T, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    iq = torch.randn(1, NIDX_DEV, S, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    ik = torch.randn(1, 1, T, HEAD_DIM, dtype=torch.bfloat16) * 0.1  # ONE shared index-k head (op: k_shape[1]==1)

    def dev_tile(t, dt=ttnn.bfloat16):
        return ttnn.from_torch(t.to(torch.bfloat16), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)

    def dev_rm(t, dt):
        return ttnn.from_torch(t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # --- indexer block scores (device); op already force-locals the current block (no sink, per upstream) ---
    block_scores = ttnn.experimental.indexer_score_msa(
        dev_tile(iq),
        dev_tile(ik),
        chunk_start_idx=chunk_start,
        scale=HEAD_DIM**-0.5,
        num_groups=NGROUPS,
        block_size=BLOCK,
        program_config=ttnn.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=1024, head_group_size=0),
    )
    bs = ttnn.to_torch(block_scores).float()  # [1, NGROUPS, S, nblk]
    block_ids = bs.topk(TOPK_BLK, dim=-1).indices.to(torch.int32)  # [1, NGROUPS, S, 16]

    # --- device op (mirror run_op_msa_native) + the op's golden, SAME block-ids ---
    out = ttnn.transformer.sparse_sdpa_msa(
        dev_rm(q.to(torch.float32), ttnn.bfloat16),
        dev_tile(k),
        dev_tile(v),
        dev_rm(block_ids, ttnn.uint32),
        scale=HEAD_DIM**-0.5,
        block_size=BLOCK,
    )
    out_t = ttnn.to_torch(out)[:, :H_DEV].float()
    ref = sparse_attention_ref_msa(q.float(), k.float(), v.float(), block_ids, HEAD_DIM**-0.5)

    passing, pcc = comp_pcc(ref, out_t, 0.99)
    logger.info(f"MSA prefill (TP=4 x SP=8 per-device slice, S={S} T={T}) vs op golden: pcc={pcc}")
    assert passing, f"MSA pipeline PCC fail: {pcc}"
