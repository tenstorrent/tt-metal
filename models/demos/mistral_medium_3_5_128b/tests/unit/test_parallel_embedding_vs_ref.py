# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Token embedding vs ``torch.nn.functional.embedding`` for both sharding modes: 1D (hidden over TP,
vocab replicated — the package default) and 2D (vocab also over SP). A lookup is exact, so the check is
bit-equality, with vocab-shard boundary ids and the table's first/last rows in the prompt.

2D is exercised at the chunked shape (5120 tokens). At the one-shot 10240-token shape its closing SP
reduce-scatter returned zeros / wrong rows for per-shard rows 1088..1099 in the run-1 bring-up (and
once hung the mesh), so that case is an opt-in diagnostic: ``MISTRAL_EMBED_2D_DIAG=1``.
"""

import os

import pytest
import torch

from models.demos.mistral_medium_3_5_128b.tt.embedding import Embedding
from models.demos.mistral_medium_3_5_128b.tt.model import Model

from .common import CFG, assert_pcc, randn, residual_to_torch


def _tokens(seq_len, sp):
    g = torch.Generator().manual_seed(seq_len)
    ids = torch.randint(0, CFG.vocab_size, (seq_len,), generator=g)
    shard = CFG.vocab_size // sp
    edges = [0, CFG.vocab_size - 1] + [shard * r + d for r in range(1, sp) for d in (-1, 0)]
    ids[: len(edges)] = torch.tensor(edges)
    ids[-len(edges) :] = torch.tensor(edges)
    return ids


def _check(galaxy_mesh, mesh_config, ccl_manager, shard_vocab_on_sp, seq_len):
    table = randn(CFG.vocab_size, CFG.hidden_size, seed=101)
    ids = _tokens(seq_len, mesh_config.sp)
    ref = torch.nn.functional.embedding(ids, table)[None, None]
    emb = Embedding(galaxy_mesh, mesh_config, ccl_manager, table, shard_vocab_on_sp=shard_vocab_on_sp)
    out = residual_to_torch(emb(Model.token_tensor(ids, galaxy_mesh, mesh_config)), galaxy_mesh, mesh_config)
    assert_pcc(f"embedding[{'2d' if shard_vocab_on_sp else '1d'}, {seq_len}]", ref, out)
    bad = (out != ref.float()).any(-1).flatten().nonzero().flatten()
    assert bad.numel() == 0, f"{bad.numel()} embedding rows differ, first positions {bad[:16].tolist()}"


@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "shard_vocab_on_sp, seq_len",
    [(False, 10240), (False, 5120), (True, 5120)],
    ids=["1d_one_shot", "1d_chunk", "2d_chunk"],
)
def test_embedding_vs_ref(galaxy_mesh, mesh_config, ccl_manager, shard_vocab_on_sp, seq_len):
    _check(galaxy_mesh, mesh_config, ccl_manager, shard_vocab_on_sp, seq_len)


@pytest.mark.timeout(900)
@pytest.mark.skipif(os.environ.get("MISTRAL_EMBED_2D_DIAG") != "1", reason="opt-in: 2D embedding at 10240 tokens")
def test_embedding_2d_one_shot_diagnostic(galaxy_mesh, mesh_config, ccl_manager):
    _check(galaxy_mesh, mesh_config, ccl_manager, True, 10240)
