# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC of TtParallelEmbedding vs torch F.embedding, for BOTH sharding modes:

  * 1D (shard_vocab_on_sp=False): emb_dim sharded on TP, vocab replicated across SP. No CCL in the
    lookup; one TP all-gather rebuilds full hidden.
  * 2D (shard_vocab_on_sp=True):  vocab ALSO sharded on SP (Megatron vocab-parallel) -> per-row masked
    lookup + SP reduce-scatter(seq) + TP all-gather. ~0.5 GiB/device less at the cost of 2 SP CCL ops.

Both must reproduce the exact torch gather (embedding is a copy — no compute — so PCC ~1.0 in bf16).
The 2D case is the real check: the per-row vocab mask + cross-SP reduce must reassemble every token's
row correctly (each token is resolved by exactly one SP row).

Fast (~seconds, galaxy SP=8 x TP=4). Run:
  pytest models/demos/minimax_m3/tests/unit/test_parallel_embedding_vs_ref.py -s
"""

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.minimax_m3.config import MeshConfig
from models.demos.minimax_m3.tt.ccl import CCLManager
from models.demos.minimax_m3.tt.parallel_embedding import TtParallelEmbedding
from models.demos.minimax_m3.utils.general_utils import get_default_num_links

from ..test_factory import parametrize_mesh_with_fabric

VOCAB, EMB_DIM = 2048, 256  # VOCAB % sp(8) == 0 and % tp(4) == 0; EMB_DIM % tp == 0


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
@pytest.mark.parametrize("shard_vocab", [False, True], ids=["1d_hidden", "2d_vocab"])
def test_parallel_embedding(mesh_device, device_params, shard_vocab, reset_seeds):
    rows, cols = tuple(mesh_device.shape)
    assert (rows, cols) == (8, 4)
    sp, tp, sp_axis = rows, cols, 0
    s_local = 64  # multiple of TILE_SIZE (32)
    S = sp * s_local

    torch.manual_seed(0)
    table = torch.randn(VOCAB, EMB_DIM, dtype=torch.bfloat16)
    tokens = torch.randint(0, VOCAB, (S,), dtype=torch.int32)

    mesh_config = MeshConfig((rows, cols), tp=tp)
    ccl = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)

    emb = TtParallelEmbedding(
        mesh_device,
        VOCAB,
        EMB_DIM,
        mesh_config,
        ccl,
        torch_weight=table,
        dtype=ttnn.bfloat16,
        shard_vocab_on_sp=shard_vocab,
    )

    # tokens: SP-seq-sharded [1, 1, s_local] per device (row r owns [r*s_local:(r+1)*s_local]), TP-replicated.
    tt_tokens = ttnn.from_torch(
        tokens.reshape(sp, 1, s_local),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(sp_axis, None)),
    )

    out = emb.forward(tt_tokens)  # per device [1, 1, s_local, EMB_DIM], SP-seq-sharded, TP-replicated

    # Reassemble: full hidden is TP-replicated, so col 0 suffices; concat the SP rows' seq shards.
    dts = ttnn.get_device_tensors(out)
    full = torch.cat([ttnn.to_torch(dts[r * cols]).float() for r in range(rows)], dim=2)  # [1,1,S,EMB_DIM]

    ref = F.embedding(tokens.long(), table.float()).reshape(1, 1, S, EMB_DIM)

    assert tuple(full.shape) == (1, 1, S, EMB_DIM), f"bad output shape {tuple(full.shape)}"
    ok, pcc = comp_pcc(ref, full, 0.999)
    logger.info(f"parallel embedding ({'2D vocab' if shard_vocab else '1D hidden'}) PCC={pcc}")
    assert ok, f"PCC too low: {pcc}"
