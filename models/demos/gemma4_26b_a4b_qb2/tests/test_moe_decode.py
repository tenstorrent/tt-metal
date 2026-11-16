# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""moe_decode (tt/moe_decode) batch-1 unit tests vs torch, per chip of the QB2 1x4 mesh (WS4 proof).

Each chip holds its tensor-parallel slice of every expert's intermediate dimension; the routed row comes in the
RouterTopk.op_packed format. Shapes: Gemma 4 26B (H 2816, I 704 -> 192 per chip, k 8, GeGLU) and gpt-oss 20B
(H 2880, I 2880 -> 720 -> 768 per chip, E 32, k 4, SwiGLU); weights BFP8 and BFP4. The reference uses the weights
round-tripped through the same block-float dtype, so the check isolates the kernels from quantization. Also checks
router_topk_b1 against the HF router tail. The read rate is measured by perf/tools/moe_decode_bench.py (this
fixture opens the mesh without a trace region).
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gemma4_26b_a4b_qb2.tt.moe_decode import (
    MoeDecodeB1,
    down_layout,
    gate_up_bias_layout,
    gate_up_local_layout,
    router_topk_b1,
    upload_per_chip,
    upload_rows_per_chip,
)

TILE_BYTES = {ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}
# I: per chip (tensor parallel over 4), real columns then zero padding to a multiple of 32 per pair layout
SHAPES = {
    "gemma": dict(H=2816, I=176, I_pad=192, E=128, k=8, activation="geglu", bias=False),
    "gpt_oss": dict(H=2880, I=720, I_pad=768, E=32, k=4, activation="swiglu", bias=True),
}


def quantized(mesh, t, dtype):
    """t round-tripped through dtype on device (block-float quantization of the weights)."""
    dev = ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )
    return ttnn.to_torch(ttnn.get_device_tensors(dev)[0]).float()


def act(g, u, activation, limit=7.0, alpha=1.702):
    if activation == "geglu":
        return torch.nn.functional.gelu(g) * u
    g = g.clamp(max=limit)
    return (u.clamp(-limit, limit) + 1) * g * torch.sigmoid(alpha * g)


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b], ids=["bfp8", "bfp4"])
@pytest.mark.parametrize("shape", list(SHAPES), ids=list(SHAPES))
def test_moe_decode_b1(qb2_mesh, shape, dtype):
    mesh = qb2_mesh
    cfg = SHAPES[shape]
    H, real, inter, E, k = cfg["H"], cfg["I"], cfg["I_pad"], cfg["E"], cfg["k"]
    banks = mesh.dram_grid_size().x
    torch.manual_seed(0)
    pad = inter - real

    def padded(t, dim):
        return torch.nn.functional.pad(t, (0, pad) if dim == -1 else (0, 0, 0, pad))

    gates = [padded(torch.randn(E, H, real) / H**0.5, -1).bfloat16() for _ in range(4)]
    ups = [padded(torch.randn(E, H, real) / H**0.5, -1).bfloat16() for _ in range(4)]
    downs = [padded(torch.randn(E, real, H) / real**0.5, -2).bfloat16() for _ in range(4)]
    w_gu = upload_per_chip(mesh, [gate_up_local_layout(g, u, banks, 3) for g, u in zip(gates, ups)], dtype)
    if cfg["bias"]:
        b_g = [padded(torch.randn(E, real) * 0.1, -1).bfloat16() for _ in range(4)]
        b_u = [padded(torch.randn(E, real) * 0.1, -1).bfloat16() for _ in range(4)]
        b_d = [torch.randn(E, H).bfloat16() * 0.1] + [torch.zeros(E, H, dtype=torch.bfloat16)] * 3  # chip 0 only
        tt_bgu = upload_rows_per_chip(mesh, [gate_up_bias_layout(g, u) for g, u in zip(b_g, b_u)])
        down_bias_row = real  # the first zero-padding row of I carries the down bias
        w_down = upload_per_chip(
            mesh, [down_layout(d, banks, bias=b, bias_row=down_bias_row) for d, b in zip(downs, b_d)], dtype
        )
    else:
        b_g = b_u = [torch.zeros(E, inter)] * 4
        b_d = [torch.zeros(E, H)] * 4
        tt_bgu, down_bias_row = None, None
        w_down = upload_per_chip(mesh, [down_layout(d, banks) for d in downs], dtype)
    x = torch.randn(1, 1, 1, H)
    ids = torch.randperm(E)[:k]
    w = torch.rand(k) + 0.1
    row = torch.zeros(1, 1, 32, 2 * k)
    row[0, 0, 0, :k], row[0, 0, 0, k:] = w / w.sum(), ids.float()
    rep = ttnn.ReplicateTensorToMesh(mesh)
    tt_x = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    tt_route = ttnn.from_torch(row, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
    op = MoeDecodeB1(mesh, hidden=H, inter_chip=inter, k=k, weight_dtype=dtype, activation=cfg["activation"])
    out = op(tt_x, tt_route, w_gu, w_down, tt_bgu, down_bias_row)
    xb, wb = x.bfloat16().float().reshape(-1), row[0, 0, 0, :k].bfloat16().float()
    for chip, dev in enumerate(ttnn.get_device_tensors(out)):
        got = ttnn.to_torch(dev).float().reshape(-1)[:H]
        ref = torch.zeros(H)
        for s in range(k):
            e = int(ids[s])
            g, u, d = (quantized(mesh, t[chip][e][None, None].float(), dtype)[0, 0] for t in (gates, ups, downs))
            gb, ub, db = (t[chip][e].float() for t in (b_g, b_u, b_d))
            ref += wb[s] * (act(xb @ g + gb, xb @ u + ub, cfg["activation"]) @ d + db)
        ok, pcc = comp_pcc(ref, got, 0.995)
        print(f"MOE_DECODE {shape} {dtype} chip={chip} pcc={pcc:.5f}", flush=True)
        assert ok, f"{shape} {dtype} chip {chip}: PCC {pcc}"


def test_moe_decode_split_flat(qb2_mesh):
    """WS5 merged MoE: k = 8 routed + 3 always-routed slots (a shared MLP as virtual experts), flat x, the down
    program's two sums (slots 0..7 / 8..10) written flat into one partial buffer (fused_decode/boundary.py)."""
    from models.demos.gemma4_26b_a4b_qb2.tt.fused_decode.boundary import flat_from_tile_rows, flat_to_tile_rows

    mesh = qb2_mesh
    H, inter, E, k, split = 2816, 192, 131, 11, 8
    dtype = ttnn.bfloat8_b
    banks = mesh.dram_grid_size().x
    torch.manual_seed(0)
    gates = [(torch.randn(E, H, inter) / H**0.5).bfloat16() for _ in range(4)]
    ups = [(torch.randn(E, H, inter) / H**0.5).bfloat16() for _ in range(4)]
    downs = [(torch.randn(E, inter, H) / inter**0.5).bfloat16() for _ in range(4)]
    w_gu = upload_per_chip(mesh, [gate_up_local_layout(g, u, banks, 3) for g, u in zip(gates, ups)], dtype)
    w_down = upload_per_chip(mesh, [down_layout(d, banks) for d in downs], dtype)
    T = -(-H // 1024)
    rep = ttnn.ReplicateTensorToMesh(mesh)
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    def flat_config(pages):
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(core, (32, 32 * pages), ttnn.ShardOrientation.ROW_MAJOR),
        )

    x = torch.randn(H).bfloat16()
    tt_x = ttnn.from_torch(
        flat_to_tile_rows(x.float(), T)[None, None].bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=flat_config(T),
        mesh_mapper=rep,
    )
    part = ttnn.from_torch(
        torch.zeros(1, 1, 32, 32 * 2 * T),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=flat_config(2 * T),
        mesh_mapper=rep,
    )
    ids = torch.cat([torch.randperm(128)[:split], torch.arange(128, 131)])
    w = torch.cat([torch.softmax(torch.randn(split), 0), torch.ones(k - split)])
    row = torch.zeros(1, 1, 32, 2 * k)
    row[0, 0, 0, :k], row[0, 0, 0, k:] = w, ids.float()
    tt_route = ttnn.from_torch(row, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
    op = MoeDecodeB1(mesh, hidden=H, inter_chip=inter, k=k, weight_dtype=dtype, split=split)
    op.local_readers = 3
    op(tt_x, tt_route, w_gu, w_down, x_flat=True, flat_out=(part, T * 2048, 0))
    xb, wb = x.float(), row[0, 0, 0, :k].bfloat16().float()
    for chip, dev in enumerate(ttnn.get_device_tensors(part)):
        flat = flat_from_tile_rows(ttnn.to_torch(dev).float()[0, 0], 2 * T * 1024)
        got_shared, got_routed = flat[:H], flat[T * 1024 : T * 1024 + H]
        refs = [torch.zeros(H), torch.zeros(H)]
        for s in range(k):
            e = int(ids[s])
            g, u, d = (quantized(mesh, t[chip][e][None, None].float(), dtype)[0, 0] for t in (gates, ups, downs))
            refs[s >= split] += wb[s] * (act(xb @ g, xb @ u, "geglu") @ d)
        for name, ref, got in (("routed", refs[0], got_routed), ("shared", refs[1], got_shared)):
            ok, pcc = comp_pcc(ref, got, 0.995)
            print(f"MOE_DECODE split {name} chip={chip} pcc={pcc:.5f}", flush=True)
            assert ok, f"split {name} chip {chip}: PCC {pcc}"


def test_router_topk_b1(qb2_mesh):
    mesh = qb2_mesh
    rep = ttnn.ReplicateTensorToMesh(mesh)
    E, k = 128, 8
    for trial in range(10):
        torch.manual_seed(trial)
        logits = torch.randn(1, 1, 1, E) * (3.0 if trial % 2 else 0.5)
        if trial == 3:
            logits[..., 5] = logits[..., 9]
        pes = torch.rand(E) + 0.5
        tl = ttnn.from_torch(
            logits,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        tp = ttnn.from_torch(
            pes.reshape(1, 1, 1, E), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep
        )
        row = ttnn.to_torch(ttnn.get_device_tensors(router_topk_b1(tl, tp, k))[0]).float()[0, 0, 0]
        probs = torch.softmax(logits[0, 0, 0].double(), -1)
        v, i = torch.topk(probs, k)
        expect = dict(zip(i.tolist(), (v / v.sum() * pes.bfloat16().double()[i]).tolist()))
        got = dict(zip(row[k:].long().tolist(), row[:k].tolist()))
        assert set(got) == set(expect), (trial, got, expect)
        for e, val in got.items():
            assert abs(val - expect[e]) <= 0.008 * expect[e], (trial, e, val, expect[e])
