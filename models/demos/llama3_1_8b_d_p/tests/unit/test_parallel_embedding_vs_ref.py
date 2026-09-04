# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: the sharded token embedding vs ``torch.nn.functional.embedding``, for BOTH sharding modes.

  * **1D** (bring-up default) — shard ``emb_dim`` across TP, replicate vocab. Lookup is local, then one
    TP all-gather rebuilds the full hidden.
  * **2D** vocab-parallel — also shard ``vocab`` across SP, so each device holds
    ``[vocab/sp, emb/tp]``. Needs an SP all-gather of the tokens, a sentinel-clamped local lookup and
    an SP reduce-scatter. Both must produce the identical result; if they diverge, the sentinel
    clamping or the vocab offset is wrong and only shows up as a few wrong token embeddings.

Llama's vocab is 128256, which is divisible by SP=8 (16032 per row), so the 2D path is available
without padding the table.
"""

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.llama3_1_8b_d_p.tt.parallel_embedding import TtParallelEmbedding

from ..test_factory import concat_sp, llama_config, make_ccl, make_mesh_config, parametrize_mesh_with_fabric

PCC = 0.99


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
@pytest.mark.parametrize("shard_vocab_on_sp", [False, True], ids=["1d", "2d"])
@pytest.mark.parametrize("seq_len", [2048], ids=["s2048"])
def test_parallel_embedding_vs_ref(mesh_device, device_params, shard_vocab_on_sp, seq_len, reset_seeds):
    cfg = llama_config()
    mesh_config = make_mesh_config(mesh_device)
    rows, cols = tuple(mesh_device.shape)
    sp = mesh_config.sp
    assert cfg.vocab_size % sp == 0, f"vocab {cfg.vocab_size} must divide by sp {sp} for the 2D path"

    # A small random table at the real vocab size would be 128256 x 4096 = 2 GiB in fp32 on host; use
    # the real vocab but a reduced emb_dim multiple of tp so the mapper math is identical and the host
    # cost is not.
    emb_dim = cfg.hidden_size
    weight = torch.randn(cfg.vocab_size, emb_dim, dtype=torch.bfloat16) * 0.02
    tokens = torch.randint(0, cfg.vocab_size, (1, 1, seq_len), dtype=torch.int32)
    reference = F.embedding(tokens.long().reshape(1, seq_len), weight.float()).reshape(1, 1, seq_len, emb_dim)

    emb = TtParallelEmbedding(
        mesh_device=mesh_device,
        vocab_size=cfg.vocab_size,
        emb_dim=emb_dim,
        mesh_config=mesh_config,
        ccl_manager=make_ccl(mesh_device),
        torch_weight=weight,
        cache_file_name=None,
        dtype=ttnn.bfloat16,
        shard_vocab_on_sp=shard_vocab_on_sp,
    )

    # Tokens are SP-seq-sharded, replicated across TP — the contract prefill_forward uses.
    dims = [None, None]
    dims[mesh_config.sp_axis] = 2
    tokens_tt = ttnn.from_torch(
        tokens,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=tuple(dims)),
    )

    out_tt = emb(tokens_tt)
    out = concat_sp(mesh_device, out_tt, mesh_config).reshape(1, 1, seq_len, emb_dim)

    passing, pcc = comp_pcc(reference, out, PCC)
    logger.info(f"parallel_embedding {'2d' if shard_vocab_on_sp else '1d'} s={seq_len}: {pcc}")
    assert passing, f"PCC fail: {pcc}"
