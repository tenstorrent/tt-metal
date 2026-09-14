# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The two whole-model components the decoder never needed: parallel embedding and the LM head.

The embedding is tested in **both** sharding modes and then against each other. The 2D
vocab-parallel path is the one with real failure modes — the sentinel padding, the per-row vocab
shift, and the SP reduce-scatter that both sums across vocab shards and re-scatters the sequence — so
"1D and 2D agree bit-for-bit" is the check that matters, not either one alone.

The LM head is checked at the real vocab (128256, which happens to be tile-aligned per TP column)
and at a deliberately mis-aligned vocab, so the padding branch is exercised rather than merely
present.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.llama_3_1_8b.tests.common import (
    assert_pcc,
    cfg_full,
    galaxy_mesh,
    make_ccl,
    pcc,
    spec_mesh_config,
    sp_shard_activation,
    to_torch_sp_concat,
)
from models.demos.llama_3_1_8b.tt.lm_head import LMHead, padded_vocab
from models.demos.llama_3_1_8b.tt.parallel_embedding import ParallelEmbedding

SEQ = 1024


def _token_tensor(ids, mesh_device, mc):
    """Token ids as the SP-sharded uint32 tensor the model's embedding consumes."""
    sp = mc.sp
    s_local = len(ids) // sp
    tok = torch.tensor(ids, dtype=torch.int32).reshape(sp, 1, s_local)
    return ttnn.from_torch(
        tok,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(mc.sp_axis, None)),
    )


@galaxy_mesh()
@pytest.mark.parametrize("shard_vocab", [False, True], ids=["emb1d", "emb2d"])
def test_parallel_embedding_vs_ref(mesh_device, device_params, shard_vocab, topology_name):
    cfg = cfg_full()
    mc = spec_mesh_config(mesh_device)
    ccl = make_ccl(mesh_device)
    torch.manual_seed(0)
    # A small table: the point is the sharding and the lookup, and a 1 GiB table costs minutes of
    # host->device transfer per parametrization for no extra coverage. vocab stays divisible by sp.
    vocab = 4096
    table = torch.randn(vocab, cfg.hidden_size) * 0.02
    ids = torch.randint(0, vocab, (SEQ,)).tolist()

    emb = ParallelEmbedding(
        mesh_device, vocab, cfg.hidden_size, mc, ccl, torch_weight=table, shard_vocab_on_sp=shard_vocab
    )
    out = emb(_token_tensor(ids, mesh_device, mc))
    got = to_torch_sp_concat(out, mesh_device, mc)

    ref = torch.nn.functional.embedding(torch.tensor(ids), table).reshape(1, 1, SEQ, cfg.hidden_size)
    assert_pcc(f"embedding_{'2d' if shard_vocab else '1d'}[{topology_name}]", pcc(ref, got))


@galaxy_mesh()
def test_embedding_1d_and_2d_agree(mesh_device, device_params):
    """The two layouts must produce the same tensor — they are interchangeable by env knob."""
    cfg = cfg_full()
    mc = spec_mesh_config(mesh_device)
    ccl = make_ccl(mesh_device)
    torch.manual_seed(0)
    vocab = 4096
    table = torch.randn(vocab, cfg.hidden_size) * 0.02
    ids = torch.randint(0, vocab, (SEQ,)).tolist()

    outs = []
    for shard in (False, True):
        emb = ParallelEmbedding(
            mesh_device, vocab, cfg.hidden_size, mc, ccl, torch_weight=table, shard_vocab_on_sp=shard
        )
        outs.append(to_torch_sp_concat(emb(_token_tensor(ids, mesh_device, mc)), mesh_device, mc))
    p = pcc(outs[0], outs[1])
    logger.info(f"embedding 1D vs 2D: PCC {p:.6f}")
    assert p > 0.99999, "the two embedding sharding modes disagree"


@galaxy_mesh()
@pytest.mark.parametrize("vocab", [128256, 1000], ids=["real_vocab", "unaligned_vocab"])
def test_lm_head_vs_ref(mesh_device, device_params, vocab, topology_name):
    """Column-parallel LM head, with the vocab padding branch exercised by the unaligned case."""
    cfg = cfg_full()
    mc = spec_mesh_config(mesh_device)
    torch.manual_seed(0)

    class _Cfg:
        hidden_size = cfg.hidden_size
        vocab_size = vocab

    w = torch.randn(vocab, cfg.hidden_size) * 0.02
    head = LMHead(mesh_device, _Cfg, mc, state_dict={"weight": w})
    assert head.padded_vocab == padded_vocab(vocab, mc.tp)
    assert head.padded_vocab % (ttnn.TILE_SIZE * mc.tp) == 0

    x = torch.randn(1, 1, SEQ, cfg.hidden_size) * 0.5
    out = head(sp_shard_activation(x, mesh_device, mc))

    dims = [None, None]
    dims[mc.tp_axis] = -1
    dims[mc.sp_axis] = 2
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(dims), mesh_shape=mesh_device.shape)
    got = ttnn.to_torch(out, mesh_composer=composer).float()[..., :vocab]

    ref = (x.to(torch.float16) @ w.to(torch.float16).transpose(0, 1)).float()
    assert_pcc(f"lm_head[{topology_name}] vocab={vocab}", pcc(ref, got))


@galaxy_mesh()
def test_final_norm_vs_ref(mesh_device, device_params, topology_name):
    """The model's tail norm — the same RMSNorm class, applied to the final instance."""
    from models.demos.llama_3_1_8b.reference.model import RMSNorm as RefRMSNorm
    from models.demos.llama_3_1_8b.tt.rms_norm import RMSNorm
    from models.demos.llama_3_1_8b.tests.common import assert_tp_replicas_agree

    cfg = cfg_full()
    mc = spec_mesh_config(mesh_device)
    torch.manual_seed(0)
    x = torch.randn(1, 1, SEQ, cfg.hidden_size) * 0.5
    weight = torch.randn(cfg.hidden_size) * 0.1

    norm = RMSNorm(mesh_device, cfg.hidden_size, cfg.rms_norm_eps, state_dict={"weight": weight})
    got = assert_tp_replicas_agree(
        norm(sp_shard_activation(x, mesh_device, mc)), mesh_device, mc, name="final_norm"
    )
    ref = RefRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
    ref.weight.data = weight.to(torch.float16)
    assert_pcc(f"final_norm[{topology_name}]", pcc(ref(x.to(torch.float16)).float(), got))
