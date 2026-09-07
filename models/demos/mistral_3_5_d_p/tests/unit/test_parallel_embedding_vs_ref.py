# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Embedding lookup vs ``torch.nn.functional.embedding``, for BOTH sharding modes.
Pattern: ``minimax_m3/tests/unit/test_parallel_embedding_vs_ref.py``.

  * **1D** (``shard_vocab_on_sp=False``): emb_dim sharded on TP, vocab replicated across SP. No CCL
    in the lookup; one TP all-gather rebuilds full hidden.
  * **2D** (``shard_vocab_on_sp=True``, the default): vocab ALSO sharded on SP (Megatron
    vocab-parallel) -> per-row sentinel lookup + SP reduce-scatter(seq) + TP all-gather.

An embedding is a copy, not a compute, so both modes must reproduce the torch gather essentially
exactly — the bar here is 0.999, not the spec's 0.99, and the row-wise checks below are exact. The
2D case is the harder one: the per-row vocab shift and the cross-SP reduce must reassemble every
token's row, with each token resolved by exactly one SP row. A sentinel/clamp mistake shows up as
tokens near a shard boundary coming back zero, which is why the token set deliberately includes the
first and last id of every shard.

**The chunk lengths here are the point.** The 2D path is correct up to 4096 tokens per chunk and
corrupts rows above it (see ``tt/parallel_embedding.py`` for the measurements), which is why 1D is
the default and why every case runs at the spec's real ``chunk_size`` of 5120 as well as at short
lengths. An earlier version of this file tested 128 and 256 tokens only, passed, and let the bug
through to the P1 KV-PCC run — where it showed up as layer 0's K at 0.995 with nothing raised.
:func:`test_2d_embedding_is_broken_above_4096_tokens` pins the limitation so it cannot be
rediscovered the same way.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral_3_5_d_p.spec import SPEC
from models.demos.mistral_3_5_d_p.tt.parallel_embedding import TtParallelEmbedding

from ..test_factory import build_mesh_and_ccl, parametrize_mesh

# Reduced from the real [131072, 12288] so the host can hold the reference table; VOCAB stays
# divisible by sp (4) and tp (8), and EMB_DIM by tp, which is what the sharding needs.
VOCAB, EMB_DIM = 4096, 512


def boundary_tokens(vocab, sp, count):
    """A token set that includes each SP vocab shard's first and last id, then random filler.

    The 2D lookup shifts each row's ids by ``r*vocab_local - 1`` and clamps onto zero sentinels, so
    the ids most likely to expose an off-by-one are exactly the shard boundaries.
    """
    vocab_local = vocab // sp
    edges = []
    for r in range(sp):
        edges += [r * vocab_local, (r + 1) * vocab_local - 1]
    filler = torch.randint(0, vocab, (max(count - len(edges), 0),), dtype=torch.int32)
    return torch.cat([torch.tensor(edges, dtype=torch.int32), filler])[:count]


@parametrize_mesh()
@pytest.mark.parametrize("shard_vocab", [False, True], ids=["1d_hidden", "2d_vocab"])
@pytest.mark.parametrize("s_total", [256, 4096], ids=["s256", "s4096"])
def test_parallel_embedding_vs_ref(mesh_device, device_params, shard_vocab, s_total, reset_seeds):
    """Both modes vs torch, at a short chunk and at 4096 — the longest chunk 2D still handles."""
    rows, cols = tuple(mesh_device.shape)
    sp = rows
    s_local = s_total // sp
    assert s_local % ttnn.TILE_SIZE == 0

    table = torch.randn(VOCAB, EMB_DIM, dtype=torch.bfloat16)
    tokens = boundary_tokens(VOCAB, sp, s_total)

    mesh_config, ccl = build_mesh_and_ccl(mesh_device)
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

    # tokens: SP-seq-sharded [1, 1, s_local] per device (row r owns [r*s_local, (r+1)*s_local)),
    # replicated across TP — the layout the runtime's H2D delivery produces.
    tt_tokens = ttnn.from_torch(
        tokens.reshape(sp, 1, s_local),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(SPEC.sp_axis, None)),
    )

    out = emb.forward(tt_tokens)
    assert tuple(out.shape)[-1] == EMB_DIM, f"expected full hidden after the TP gather, got {tuple(out.shape)}"

    # Full hidden is TP-replicated, so column 0 suffices; concat the SP rows' seq shards.
    shards = ttnn.get_device_tensors(out)
    full = torch.cat([ttnn.to_torch(shards[r * cols]).float() for r in range(rows)], dim=2)

    ref = F.embedding(tokens.long(), table.float()).reshape(1, 1, s_total, EMB_DIM)
    assert tuple(full.shape) == (1, 1, s_total, EMB_DIM), f"bad output shape {tuple(full.shape)}"

    ok, pcc = comp_pcc(ref, full, 0.999)
    logger.info(f"parallel embedding ({'2D vocab' if shard_vocab else '1D hidden'}) pcc={pcc}")
    assert ok, f"embedding PCC too low: {pcc}"


def _lookup(mesh_device, tokens, table, *, shard_vocab):
    """Run TtParallelEmbedding over SP-seq-sharded tokens and return the reassembled [S, EMB] rows."""
    rows, cols = tuple(mesh_device.shape)
    s_local = tokens.numel() // rows
    mesh_config, ccl = build_mesh_and_ccl(mesh_device)
    emb = TtParallelEmbedding(
        mesh_device, VOCAB, EMB_DIM, mesh_config, ccl, torch_weight=table, shard_vocab_on_sp=shard_vocab
    )
    tt_tokens = ttnn.from_torch(
        tokens.reshape(rows, 1, s_local),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(SPEC.sp_axis, None)),
    )
    shards = ttnn.get_device_tensors(emb.forward(tt_tokens))
    return torch.cat([ttnn.to_torch(shards[r * cols]).float().reshape(s_local, EMB_DIM) for r in range(rows)], dim=0)


def _bad_rows(got, ref, *, tolerance=0.05):
    """Row indices whose lookup is wrong. A zero row means no SP row resolved the token; a wrong-value
    row means it was resolved at the wrong index, or summed twice."""
    return (got - ref).abs().max(dim=-1).values.gt(tolerance).nonzero().flatten()


@parametrize_mesh()
@pytest.mark.parametrize("s_total", [128, 4096, 5120], ids=["s128", "s4096", "s5120"])
def test_1d_embedding_resolves_every_token_exactly(mesh_device, device_params, s_total, reset_seeds):
    """The DEFAULT (1D) path, row by row, including at the spec's real chunk_size of 5120.

    Row-wise rather than by PCC: PCC is forgiving of a handful of wrong rows in a large tensor, and a
    handful of wrong rows is exactly what the 2D path produces. 5120 is here because the spec's chunk
    is 5120 — the length at which the 2D path silently corrupts.
    """
    table = torch.randn(VOCAB, EMB_DIM, dtype=torch.bfloat16)
    tokens = boundary_tokens(VOCAB, mesh_device.shape[SPEC.sp_axis], s_total)
    got = _lookup(mesh_device, tokens, table, shard_vocab=False)
    ref = F.embedding(tokens.long(), table.float())
    bad = _bad_rows(got, ref)
    logger.info(f"1D embedding at {s_total} tokens: {len(bad)} bad rows, pcc={comp_pcc(ref, got, 0.999)[1]}")
    assert len(bad) == 0, f"1D embedding got {len(bad)} wrong rows at {s_total} tokens (first: {bad[:8].tolist()})"


@parametrize_mesh()
def test_2d_embedding_is_broken_above_4096_tokens(mesh_device, device_params, reset_seeds):
    """Pin the 2D path's limitation: exact at 4096 tokens, corrupt at 5120.

    A documented-defect test, not a wish. It exists so the limitation cannot be rediscovered the way
    it was found the first time — as an unexplained 0.995 on layer 0's KV in the P1 run — and so that
    whoever fixes the underlying op finds out here, by this test starting to fail.
    """
    table = torch.randn(VOCAB, EMB_DIM, dtype=torch.bfloat16)
    sp = mesh_device.shape[SPEC.sp_axis]

    tokens_ok = boundary_tokens(VOCAB, sp, 4096)
    ref_ok = F.embedding(tokens_ok.long(), table.float())
    bad_ok = _bad_rows(_lookup(mesh_device, tokens_ok, table, shard_vocab=True), ref_ok)
    assert len(bad_ok) == 0, f"2D embedding should be exact at 4096 tokens, got {len(bad_ok)} bad rows"

    tokens_bad = boundary_tokens(VOCAB, sp, 5120)
    got = _lookup(mesh_device, tokens_bad, table, shard_vocab=True)
    ref = F.embedding(tokens_bad.long(), table.float())
    bad = _bad_rows(got, ref)
    zeros = got.abs().max(dim=-1).values.lt(1e-6).sum().item()
    logger.info(
        f"2D embedding at 5120 tokens: {len(bad)} bad rows ({zeros} of them zero), "
        f"first at {bad[:8].tolist()} — the documented defect"
    )
    assert len(bad) > 0, (
        "the 2D embedding is now correct at 5120 tokens. If the underlying op was fixed, flip "
        "DEFAULT_SHARD_VOCAB_ON_SP back to True in tt/parallel_embedding.py, update its docstring, "
        "and delete this test."
    )
