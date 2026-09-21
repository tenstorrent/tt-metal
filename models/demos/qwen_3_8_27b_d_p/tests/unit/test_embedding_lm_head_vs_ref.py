# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Embedding and LM head vs torch references, at the model's REAL vocab (248320 x 5120).

Full width on purpose: these two tensors are the ones a reduced-config model test has to shrink,
so if they are not measured at their real size here they are not measured at all. The tokens are
SP-sharded, so each row embeds only its own block and the lookup is local.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.qwen_3_8_27b_d_p.tt.lm_head import LMHead, padded_vocab
from models.demos.qwen_3_8_27b_d_p.tt.parallel_embedding import ParallelEmbedding

from ..test_factory import mesh_setup, parametrize_mesh, unit_test_config
from .helpers import WEIGHT_DTYPE, assert_tp_replicated, check_pcc, from_sp_sharded, randn

S_LOCAL = 128


@parametrize_mesh()
def test_parallel_embedding_vs_ref(mesh, submesh_shape, device_params):
    """Lookup vs ``torch.nn.functional.embedding``, with the hidden dim sharded across TP and the
    result all-gathered back to full width."""
    cfg = unit_test_config()
    mesh_config, ccl = mesh_setup(mesh)
    total = S_LOCAL * mesh_config.sp

    weight = randn(cfg.vocab_size, cfg.hidden_size, seed=71, scale=0.02)
    tokens = torch.randint(0, cfg.vocab_size, (1, total), generator=torch.Generator().manual_seed(72))
    expected = torch.nn.functional.embedding(tokens, weight).reshape(1, 1, total, cfg.hidden_size)

    emb = ParallelEmbedding(
        mesh,
        cfg.vocab_size,
        cfg.hidden_size,
        {"weight": weight},
        mesh_config=mesh_config,
        ccl_manager=ccl,
    )
    tt_tokens = ttnn.from_torch(
        tokens.reshape(1, 1, 1, total),
        device=mesh,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=mesh_config.sequence_parallel(mesh, seq_dim=3),
    )
    out = emb(tt_tokens)
    assert_tp_replicated(out, mesh_config, "embedding output")
    got = from_sp_sharded(out, mesh_config)
    check_pcc("parallel_embedding", expected, got, shape=(1, 1, total, cfg.hidden_size))


@parametrize_mesh()
def test_sp_rows_embed_their_own_tokens(mesh, submesh_shape, device_params):
    """Row ``r`` must embed chunk tokens ``[r*s_local, (r+1)*s_local)``. If the token shard and the
    RoPE shard disagree, every attention layer rotates the wrong tokens and only the composed
    output looks plausible."""
    cfg = unit_test_config(vocab_size=4096, hidden_size=512)
    mesh_config, ccl = mesh_setup(mesh)
    s_local, total = 32, 32 * mesh_config.sp
    weight = randn(cfg.vocab_size, cfg.hidden_size, seed=73, scale=0.02)
    tokens = torch.arange(total).reshape(1, total) % cfg.vocab_size

    emb = ParallelEmbedding(
        mesh, cfg.vocab_size, cfg.hidden_size, {"weight": weight}, mesh_config=mesh_config, ccl_manager=ccl
    )
    tt_tokens = ttnn.from_torch(
        tokens.reshape(1, 1, 1, total),
        device=mesh,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=mesh_config.sequence_parallel(mesh, seq_dim=3),
    )
    out = emb(tt_tokens)
    shards = ttnn.get_device_tensors(out)
    for r in range(mesh_config.sp):
        got = ttnn.to_torch(shards[r * mesh_config.tp]).reshape(1, 1, s_local, cfg.hidden_size)
        expected = torch.nn.functional.embedding(tokens[:, r * s_local : (r + 1) * s_local], weight).reshape(
            1, 1, s_local, cfg.hidden_size
        )
        check_pcc(f"embedding[row{r}]", expected, got)


@parametrize_mesh()
def test_lm_head_vs_ref(mesh, submesh_shape, device_params):
    """Vocab-column-parallel projection vs a torch linear, composed back across the TP columns.

    The padded vocab is checked too: the pad must be added BEFORE the shard, or the per-device
    logit ranges stop lining up with the global vocab offsets.
    """
    cfg = unit_test_config()
    mesh_config, _ccl = mesh_setup(mesh)
    tokens = 64

    weight = randn(cfg.vocab_size, cfg.hidden_size, seed=74, scale=0.02)
    x = randn(1, 1, tokens, cfg.hidden_size, seed=75, scale=0.5)
    expected = torch.nn.functional.linear(x.float(), weight.float())

    head = LMHead(
        mesh,
        cfg.vocab_size,
        cfg.hidden_size,
        {"weight": weight},
        mesh_config=mesh_config,
        weight_dtype=WEIGHT_DTYPE,
    )
    assert head.padded_vocab_size == padded_vocab(cfg.vocab_size, mesh_config.tp)
    assert head.padded_vocab_size % mesh_config.tp == 0

    tt_x = ttnn.from_torch(
        x.float(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_config.replicate(mesh),
    )
    out = head(tt_x)
    shards = ttnn.get_device_tensors(out)
    full = torch.cat([ttnn.to_torch(shards[c]) for c in range(mesh_config.tp)], dim=-1)
    got = full[..., : cfg.vocab_size]
    check_pcc("lm_head", expected, got, shape=(1, 1, tokens, cfg.vocab_size))
