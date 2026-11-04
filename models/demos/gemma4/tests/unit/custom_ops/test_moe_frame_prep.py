# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MoeFramePrep (custom op) vs the torch golden on a single chip: the decode MoE kernel's x / ids / scores for one
row-chip's slice of the lane frame (real rows, a partly padded slice, an all-padding slice), interleaved and
height-sharded score outputs, and the program-cache hit on fresh buffers."""

import pytest
import torch

import ttnn
from models.demos.gemma4.custom_ops.moe_frame_prep import MoeFramePrep
from models.demos.gemma4.custom_ops.router_topk import RouterTopk

SINGLE_CHIP = [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")]
H, E, K, B = 2816, 128, 8, 32


def _scores_sharded_config(tokens_per_device=B, k=K):
    """The dispatch's expert-scores config (models/common/moe/tt_moe_decode_config.py): height-sharded L1,
    [1, k] shards over a (tokens / 8) x 8 core grid."""
    ny = min(8, tokens_per_device)
    nx = (tokens_per_device + ny - 1) // ny
    spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nx - 1, ny - 1))}),
        [1, k],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)


def _inputs(R, seed):
    g = torch.Generator().manual_seed(seed)
    hidden = (torch.randn(1, 1, R, H, generator=g) * 2).to(torch.bfloat16)
    packed = torch.zeros(1, 1, R, 2 * K)
    for r in range(R):
        ids = torch.randperm(E, generator=g)[:K]
        w = torch.rand(K, generator=g) + 0.1
        packed[0, 0, r, :K] = (w / w.sum()).to(torch.bfloat16).float()
        packed[0, 0, r, K:] = ids.float()
    return hidden, packed.to(torch.bfloat16)


def _rowbase(mesh_device, value):
    return ttnn.from_torch(
        torch.full((1, 1, 1, 16), value, dtype=torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
    )


@pytest.mark.parametrize("R, rowbase", [(128, 0), (128, 32), (128, 96), (128, 128), (64, 32), (32, 0)])
@pytest.mark.parametrize("scores_mem", ["interleaved", "sharded"])
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_moe_frame_prep_matches_golden(mesh_device, device_params, R, rowbase, scores_mem):
    hidden, packed = _inputs(R, seed=R + rowbase)
    x_ref, ids_ref, sc_ref = MoeFramePrep.golden(hidden.float(), packed.float(), rowbase, K, B)
    t_hidden = ttnn.from_torch(hidden, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    t_packed = ttnn.from_torch(packed, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
    t_base = _rowbase(mesh_device, rowbase)
    s_mem = _scores_sharded_config() if scores_mem == "sharded" else None
    x, ids, scores = MoeFramePrep.op(t_hidden, t_packed, t_base, K, scores_memory_config=s_mem, B=B)
    assert x.layout == ttnn.ROW_MAJOR_LAYOUT and x.memory_config().buffer_type == ttnn.BufferType.L1
    assert ids.dtype == ttnn.uint16 and list(ids.shape) == [B, 1, 1, K]
    assert list(scores.shape) == [B, 1, 1, K] and (scores.is_sharded() == (scores_mem == "sharded"))
    x_t = ttnn.to_torch(x).float()
    ids_t = ttnn.to_torch(ids).to(torch.int64)
    sc_t = ttnn.to_torch(scores).float()
    print(
        f"[R{R} base{rowbase} {scores_mem}] max|x diff| {(x_t - x_ref).abs().max():.4f} ids equal {torch.equal(ids_t, ids_ref)} "
        f"max|score diff| {(sc_t - sc_ref).abs().max():.5f}"
    )
    assert torch.equal(x_t, x_ref), "x rows differ from the untilized frame rows"
    assert torch.equal(ids_t, ids_ref), "expert ids differ"
    assert torch.equal(sc_t, sc_ref), "scores differ"
    # program-cache hit on fresh buffers (new addresses as runtime args)
    t_hidden2 = ttnn.from_torch(hidden, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    t_packed2 = ttnn.from_torch(packed, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
    x2, ids2, sc2 = MoeFramePrep.op(t_hidden2, t_packed2, t_base, K, scores_memory_config=s_mem, B=B)
    assert torch.equal(ttnn.to_torch(x2).float(), x_t)
    assert torch.equal(ttnn.to_torch(ids2).to(torch.int64), ids_t)
    assert torch.equal(ttnn.to_torch(sc2).float(), sc_t)


@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_moe_frame_prep_from_router_packed(mesh_device, device_params):
    """End to end on one chip: router_topk.op_packed rows -> frame prep: the ids / scores the kernel would see equal
    the packed golden's (ids exact, weights to bf16)."""
    g = torch.Generator().manual_seed(11)
    logits = (torch.randn(1, 1, B, E, generator=g) * 3).to(torch.bfloat16)
    pes = (torch.rand(E, generator=g) + 0.5).to(torch.bfloat16)
    golden = RouterTopk.golden_packed(logits, pes, K)
    t_logits = ttnn.from_torch(logits, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    t_pes = ttnn.from_torch(
        pes.reshape(1, 1, 1, E), device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16
    )
    packed = RouterTopk.op_packed(t_logits, t_pes, _rowbase(mesh_device, 0), K)
    hidden = (torch.randn(1, 1, B, H, generator=g)).to(torch.bfloat16)
    t_hidden = ttnn.from_torch(hidden, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    x, ids, scores = MoeFramePrep.op(t_hidden, packed, _rowbase(mesh_device, 0), K, B=B)
    ids_t = ttnn.to_torch(ids).to(torch.int64).reshape(B, K)
    sc_t = ttnn.to_torch(scores).float().reshape(B, K)
    ids_ref = golden[0, 0, :, K:].to(torch.int64)
    sc_ref = golden[0, 0, :, :K]
    # the kernel breaks exact bf16 ties differently from torch.topk: count a mismatch only off a tie
    lg = logits.float()[0, 0]
    bad = 0
    for u in range(B):
        if not torch.equal(ids_t[u].sort().values, ids_ref[u].sort().values):
            kth = torch.topk(lg[u], K).values[-1]
            diff = set(ids_t[u].tolist()) ^ set(ids_ref[u].tolist())
            bad += int(not all(lg[u, e] == kth for e in diff))
    assert bad == 0, "selected experts differ from the golden beyond exact ties"
    sel = ids_t == ids_ref
    assert (sc_t[sel] - sc_ref[sel]).abs().max() <= 1e-2 * sc_ref.abs().max() + 1e-3
    assert torch.equal(ttnn.to_torch(x).float(), hidden.float().reshape(B, 1, 1, H))
