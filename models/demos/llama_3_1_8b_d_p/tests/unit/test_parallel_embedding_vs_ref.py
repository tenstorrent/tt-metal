# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sharded token embedding vs `torch.nn.functional.embedding`, in BOTH sharding modes.

Target mesh (8, 4), random weights. Structure follows
`minimax_m3/tests/unit/test_parallel_embedding_vs_ref.py`.

Both modes are tested because the model picks one at construction and a bring-up that only exercises
the default has not tested what it ships:

* **1D** — shard `emb_dim` across TP, replicate the vocab. Lookup is local, then one TP all-gather.
* **2D (default)** — also shard `vocab` across SP. Each device holds `[vocab/sp, emb/tp]`; the
  forward needs an SP all-gather of the tokens, a sentinel-clamped per-row lookup, and an SP
  reduce-scatter. More CCL, less memory.

Llama's `[128256, 4096]` table needs no padding in either mode: 128256 / 8 = 16032 and
128256 / 4 = 32064, both exact.

The 2D path's sentinel trick is what makes a token belonging to another SP row resolve to zero
instead of to a wrong row, so `test_2d_resolves_every_token_exactly_once` checks it directly rather
than trusting the aggregate PCC — a handful of mis-resolved tokens out of thousands barely moves PCC.
"""

import pytest
import torch

import ttnn
from models.demos.llama_3_1_8b_d_p.reference.model import REF_DTYPE
from models.demos.llama_3_1_8b_d_p.tt.parallel_embedding import TtParallelEmbedding

from ..test_factory import assert_pcc, parametrize_target_mesh

SEQ = 1024


def _tokens(vocab_size, seq_len, seed=0):
    torch.manual_seed(seed)
    return torch.randint(0, vocab_size, (1, 1, seq_len), dtype=torch.int32)


@parametrize_target_mesh()
@pytest.mark.parametrize("shard_vocab_on_sp", [False, True], ids=["1d_emb_on_tp", "2d_vocab_on_sp"])
@pytest.mark.parametrize("sharded_residual", [True, False], ids=["sharded_residual", "replicated_residual"])
def test_parallel_embedding_vs_ref(
    mesh_device, device_params, config, mesh_config, ccl_manager, topology_name,
    shard_vocab_on_sp, sharded_residual, monkeypatch,
):
    """Embedding lookup vs torch, for both sharding modes and both residual layouts."""
    monkeypatch.setenv("LLAMA31_8B_SHARDED_RESIDUAL", "1" if sharded_residual else "0")

    torch.manual_seed(0)
    weight = torch.randn(config.vocab_size, config.hidden_size, dtype=REF_DTYPE) * 0.02
    tokens = _tokens(config.vocab_size, SEQ)
    golden = torch.nn.functional.embedding(tokens.long().reshape(1, SEQ), weight).reshape(
        1, 1, SEQ, config.hidden_size
    )

    emb = TtParallelEmbedding(
        mesh_device=mesh_device,
        vocab_size=config.vocab_size,
        emb_dim=config.hidden_size,
        mesh_config=mesh_config,
        ccl_manager=ccl_manager,
        torch_weight=weight,
        shard_vocab_on_sp=shard_vocab_on_sp,
    )

    # Tokens are SP-seq-sharded and replicated across TP.
    dims = [None, None]
    dims[mesh_config.sp_axis] = 2
    tt_tokens = ttnn.from_torch(
        tokens,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
    )
    out = emb.forward(tt_tokens)
    ttnn.synchronize_device(mesh_device)

    if sharded_residual:
        # emb/tp per TP column: concat sequence over SP rows and features over TP cols.
        got = ttnn.to_torch(
            out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 3), mesh_shape=tuple(mesh_device.shape))
        )
    else:
        got = ttnn.to_torch(
            out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 0), mesh_shape=tuple(mesh_device.shape))
        )[:1]

    mode = "2d" if shard_vocab_on_sp else "1d"
    residual = "sharded" if sharded_residual else "replicated"
    assert_pcc(f"embedding[{mode},{residual}]", golden, got.to(REF_DTYPE), topology_name)


@parametrize_target_mesh()
def test_2d_resolves_every_token_exactly_once(
    mesh_device, device_params, config, mesh_config, ccl_manager, topology_name, monkeypatch
):
    """Under 2D vocab sharding, each token must be resolved by exactly ONE SP row.

    The vocab shards partition the table, so a token is real on one row and out of range on the
    other seven, where the sentinel must clamp it to a zero row. If the sentinel were wrong the
    reduce-scatter would sum a wrong row into the answer — and a few bad tokens out of a thousand
    barely dent PCC, so this uses a weight that makes each token's embedding UNIQUE and identifiable.
    """
    monkeypatch.setenv("LLAMA31_8B_SHARDED_RESIDUAL", "0")
    # Row v of the table is the constant v: then embedding(t) must be exactly t everywhere.
    weight = torch.arange(config.vocab_size, dtype=torch.float32)[:, None].repeat(1, config.hidden_size)
    weight = (weight * 1e-3).to(REF_DTYPE)  # keep values inside bf16's exact-integer range
    tokens = _tokens(config.vocab_size, SEQ, seed=7)

    emb = TtParallelEmbedding(
        mesh_device=mesh_device,
        vocab_size=config.vocab_size,
        emb_dim=config.hidden_size,
        mesh_config=mesh_config,
        ccl_manager=ccl_manager,
        torch_weight=weight,
        shard_vocab_on_sp=True,
    )
    dims = [None, None]
    dims[mesh_config.sp_axis] = 2
    tt_tokens = ttnn.from_torch(
        tokens, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
    )
    out = emb.forward(tt_tokens)
    ttnn.synchronize_device(mesh_device)
    got = ttnn.to_torch(
        out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 0), mesh_shape=tuple(mesh_device.shape))
    )[:1].float()

    expected = (tokens.reshape(SEQ).float() * 1e-3)[None, None, :, None].expand(1, 1, SEQ, config.hidden_size)
    # A doubly-resolved token reads ~2x its value; an unresolved one reads 0. Both are gross errors
    # in this weight, so a tight relative tolerance catches them per token.
    rel = (got - expected).abs() / (expected.abs() + 1e-3)
    worst_token = rel.max(dim=-1).values.reshape(SEQ)
    bad = (worst_token > 0.05).nonzero().flatten()
    assert bad.numel() == 0, f"{bad.numel()} tokens mis-resolved under 2D vocab sharding, first: {bad[:8].tolist()}"
