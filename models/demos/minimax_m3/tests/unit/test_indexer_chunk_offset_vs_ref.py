# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""indexer_score_msa per-device `chunk_offset` equivalence.

The SP=8 MSA path keeps the query sharded (each chip's S/sp rows start at a different global position
`cached_len + sp_rank*S_local`), so the indexer's causal diagonal must be per-device. The op gained an
optional `chunk_offset` tensor (uint32, ONE tile/device, value at [0,0] = chunk-start IN TILES) that the
reader streams into cb_offset and the compute/writer kernels use as a RUNTIME mask diagonal instead of
the compile-time `chunk_start_idx`.

This proves the mechanism on a single device: binding `chunk_offset = X/32` (with chunk_start_idx=0)
must produce the SAME block scores as the compile-time scalar `chunk_start_idx = X`. (The SP-sharded
per-device variant just feeds each chip a different tile value.)
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

HEAD_DIM, BLOCK = 128, 128


@pytest.mark.parametrize("S,T,chunk_start", [(640, 5120, 4480)], ids=["chunk640_ctx5120_start4480"])
def test_indexer_chunk_offset_equiv(device, S, T, chunk_start):
    assert chunk_start % ttnn.TILE_SIZE == 0
    torch.manual_seed(0)
    iq = torch.randn(1, 1, S, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    ik = torch.randn(1, 1, T, HEAD_DIM, dtype=torch.bfloat16) * 0.1

    def dev_tile(t):
        return ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=1024, head_group_size=0)
    common = dict(scale=HEAD_DIM**-0.5, num_groups=1, block_size=BLOCK, program_config=cfg)

    # A) compile-time scalar chunk_start_idx = X.
    scores_scalar = ttnn.to_torch(
        ttnn.experimental.indexer_score_msa(dev_tile(iq), dev_tile(ik), chunk_start_idx=chunk_start, **common)
    ).float()

    # B) runtime per-device offset: chunk_start_idx=0 + chunk_offset tile holding X/32 (tiles).
    off = torch.zeros(1, 1, ttnn.TILE_SIZE, ttnn.TILE_SIZE, dtype=torch.int32)
    off[0, 0, 0, 0] = chunk_start // ttnn.TILE_SIZE
    tt_off = ttnn.from_torch(off, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
    scores_runtime = ttnn.to_torch(
        ttnn.experimental.indexer_score_msa(
            dev_tile(iq), dev_tile(ik), chunk_start_idx=0, chunk_offset=tt_off, **common
        )
    ).float()

    passing, pcc = comp_pcc(scores_scalar, scores_runtime, 0.9999)
    logger.info(f"indexer chunk_offset (runtime X/32={chunk_start // ttnn.TILE_SIZE}) vs scalar X={chunk_start}: pcc={pcc}")
    assert passing, f"chunk_offset != scalar chunk_start: pcc={pcc}"
