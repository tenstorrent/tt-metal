# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Numerics for the missing-op workarounds vs reference_cpu (status.md step 2).

IndexerCPU runs its functional path (use_fp8_path=False) — same path the
single-chip port matched. Non-interleaved RoPE stays a host pre-step (F1).
topk equality is per-row index-set overlap, not PCC: ties may reorder.
"""

import pytest
import torch

import ttnn
from models.demos.deepseek_v32.reference_cpu.model import IndexerCPU, ModelArgs
from models.demos.deepseek_v32.reference_cpu.utils import apply_rotary_emb, precompute_freqs_cis
from models.demos.deepseek_v32.reference_cpu.weights import init_random
from models.demos.deepseek_v32.tests.mesh_utils import parametrize_mesh_device
from models.demos.deepseek_v32.tt import ops
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = pytest.mark.dev  # fast op-numerics tests — inner loop

LOGITS_PCC = 0.99


def _dev(t, mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        t, device=mesh_device, layout=layout, dtype=dtype, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )


@parametrize_mesh_device()
@pytest.mark.parametrize("seq", [128], ids=["s128"])
def test_indexer_logits_numerics(mesh_device, seq):
    args = ModelArgs(max_batch_size=1)
    torch.manual_seed(42)
    idx_cpu = IndexerCPU(args, use_fp8_path=False).eval()
    init_random(idx_cpu)

    x = torch.randn(1, seq, args.dim, dtype=torch.bfloat16)
    qr = torch.randn(1, seq, args.q_lora_rank, dtype=torch.bfloat16)
    freqs_cis = precompute_freqs_cis(args)[:seq]
    mask = torch.full((seq, seq), float("-inf")).triu_(1)
    with torch.no_grad():
        _, ref_score = idx_cpu(x, qr, 0, freqs_cis, mask)  # [1, S, S], mask included

        # Device-op inputs (host stems mirror IndexerCPU; on-device stems are step 3).
        q = idx_cpu.wq_b(qr).view(1, seq, args.index_n_heads, args.index_head_dim)
        q_pe, q_nope = torch.split(q, [64, args.index_head_dim - 64], dim=-1)
        q = torch.cat([apply_rotary_emb(q_pe, freqs_cis, interleaved=False), q_nope], dim=-1)
        k = idx_cpu.k_norm(idx_cpu.wk(x))
        k_pe, k_nope = torch.split(k, [64, args.index_head_dim - 64], dim=-1)
        k = torch.cat([apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, interleaved=False).squeeze(2), k_nope], dim=-1)
        w = idx_cpu.weights_proj(x.float()) * args.index_n_heads**-0.5 * idx_cpu.softmax_scale

    # seq (128) < full-model k_chunk (256): cap KC to Skv/32 (op requires KC <= Skv/32).
    logits = ops.indexer_logits(
        _dev(q.permute(0, 2, 1, 3), mesh_device),
        _dev(k.unsqueeze(1), mesh_device),
        _dev(w.unsqueeze(1), mesh_device),
        program_config=ops.indexer_program_config(seq),
    )
    got = ops._to_host(logits)[0, 0] + mask
    assert_with_pcc(ref_score[0].float(), got.float(), LOGITS_PCC)


@parametrize_mesh_device()
@pytest.mark.parametrize("skv,k", [(512, 64), (4096, 2048)], ids=["k64", "k2048"])
def test_topk_indices_match(mesh_device, skv, k):
    torch.manual_seed(0)
    logits = torch.randn(1, 1, 128, skv, dtype=torch.bfloat16)
    # topk_large_indices is ROW_MAJOR bf16 in (chains off indexer_score's row-major out).
    got = ops._to_host(ops.topk_indices(_dev(logits, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT), k)).long()[0, 0]
    want = torch.topk(logits.float(), k, dim=-1).indices[0, 0]
    overlap = torch.tensor([len(set(g.tolist()) & set(w.tolist())) for g, w in zip(got, want)]).float() / k
    # bf16 ties at the k-th score swap boundary entries — a band that grows with k
    # (~0.5% on random logits). Allow max(2 indices, 1%) per row.
    assert overlap.min() >= 1 - max(2 / k, 0.01), f"min row overlap {overlap.min():.4f}"


@parametrize_mesh_device()
@pytest.mark.parametrize("start_pos", [0, 256], ids=["single_shot", "chunked"])
def test_sparse_mla_numerics(mesh_device, start_pos):
    torch.manual_seed(1)
    # h=128 → 32 heads/chip after the TP=4 split; sparse_sdpa needs H/tp a multiple of 32.
    h, sq, skv, k = 128, 128, 512, 64
    q = torch.randn(1, h, sq, 576, dtype=torch.bfloat16)
    kvpe = torch.randn(1, 1, skv, 576, dtype=torch.bfloat16)
    # All indices in-bounds and causal (<= start_pos + row), no sentinels — so the op
    # (softmax over every gathered slot) equals the dense reference below. The op itself
    # does no causal math; masking would arrive as the 0xFFFFFFFF sentinel from topk.
    idx = (torch.rand(1, 1, sq, k) * (start_pos + torch.arange(sq).view(sq, 1) + 1)).to(torch.int32)
    scale = 576**-0.5

    sel = kvpe[0, 0][idx.long()[0, 0]]
    ref = torch.einsum(
        "hsk,skc->hsc",
        (torch.einsum("hsd,skd->hsk", q[0].float(), sel.float()) * scale).softmax(-1),
        sel[..., :512].float(),
    )
    q_sharded = ttnn.from_torch(
        q,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        # SP-shard sequence (dim2, mesh axis 0) + TP-shard heads (dim1, axis 1): the
        # production sparse_mla layout. indices stay replicated (the op partitions them).
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 1)),
    )
    kvpe_dev = ttnn.from_torch(
        kvpe,  # full-T latent [1, 1, skv, 576], replicated on device (ROW_MAJOR bf16)
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    out = ops.sparse_mla(
        q_sharded,
        kvpe_dev,
        _dev(idx, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32),
        scale,
        start_pos=start_pos,
    )
    out_t = ttnn.to_torch(
        out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape)
    )[:1]
    assert_with_pcc(ref, out_t[0].float(), 0.99)  # bf16 online-softmax op (matches sparse_sdpa's own PCC bar)
