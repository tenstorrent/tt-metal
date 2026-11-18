# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# Perf harness: decode B=32 boundary A / B on a QB2 1x4 ring (4x p300c).
# Runs (1) the composed chain built from the llama31_8b_qb2 decoder's own decode ops and configs and
# (2) the lego TPBoundary fused op, each captured in a metal trace and replayed REPLAYS times.
# Run with TT_METAL_DEVICE_PROFILER=1 (tracy capture segfaults on TPBoundary) and read the per-replay device time
# (slowest chip, first kernel start to last kernel end) with
#   python tp_boundary_trace_span.py generated/profiler/.logs/profile_log_device.csv
# p300c gives fast dispatch an 11-wide worker grid; decode_config assumes 12, so mm_grid is narrowed below.

import pytest
import torch

import ttnn
from models.common.tt_ccl import TT_CCL
from models.demos.llama31_8b_qb2.tt.decoder import PROJECTION_GEOMETRY, LlamaDecoder
from models.demos.llama31_8b_qb2.tt.precision import load_precision_config
from dataclasses import replace

from ttnn.lego.tp_boundary import TPBoundary, decode_config

TP, HIDDEN, INTER, B = 4, 4096, 14336, 32
EPS = 1e-5
REPLAYS = 10


@pytest.fixture
def qb2_mesh():
    cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = 8192
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING, router_config=cfg)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=16384, trace_region_size=64 << 20)
    try:
        yield mesh
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def shards(mesh, ts, dtype, dim=0, mem=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        torch.cat(ts, dim=dim).contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=mem,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim),
    )


def make_decoder(mesh, local):
    """A LlamaDecoder carrying only what the decode boundary ops read (same code paths as from_state_dict)."""
    d = LlamaDecoder.__new__(LlamaDecoder)
    p = load_precision_config()
    d.mesh_device = mesh
    d.eps = EPS
    d.grid = mesh.compute_with_storage_grid_size()
    d.core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(d.grid.x - 1, d.grid.y - 1))})
    d.ccl = TT_CCL(mesh)
    d.decode_workspace = None
    d.hidden = 1024
    d.residual_memcfg = d._width_memcfg(4096, 8)
    d.local_residual_memcfg = d._width_memcfg(d.hidden, 8)
    d.norm_program = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=d.grid, subblock_w=4, block_h=1, block_w=16, inplace=False
    )
    d.decode_weights, d.decode_programs, d.decode_computes, d.decode_inputs = {}, {}, {}, {}
    d.decode_logical_widths = {}
    banks = mesh.dram_grid_size().x
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    for role, ts in local.items():
        opts = PROJECTION_GEOMETRY[role]
        kdim, ndim = ts[0].shape
        alignment = 32 * banks * opts["readers"]
        padded_n = ((ndim + alignment - 1) // alignment) * alignment
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram_grid, [kdim, padded_n // banks], ttnn.ShardOrientation.ROW_MAJOR),
        )
        d.decode_weights[role] = shards(
            mesh,
            [torch.nn.functional.pad(t, (0, padded_n - ndim)) for t in ts],
            getattr(ttnn, p["weight_groups"][role]),
            mem=mem,
        )
        d.decode_logical_widths[role] = ndim
        d.decode_inputs[role] = d._width_memcfg(kdim, opts["cores"])
        d.decode_programs[role] = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=opts["block"],
            per_core_M=1,
            per_core_N=(padded_n + 32 * opts["cores"] - 1) // (32 * opts["cores"]),
            num_workers_per_dram_bank=opts["readers"],
        )
    d._configure_precision(p)
    d.prepare_decode(B)
    return d


def chain_a(d, residual, attention):
    """_decode_finish up to the SwiGLU product (o_proj -> RS -> add -> AG -> norm -> gate_up -> silu*mul)."""
    projected = d._output_projection(attention, "o")
    residual = ttnn.add(
        residual, ttnn.to_memory_config(projected, d.local_residual_memcfg), memory_config=d.local_residual_memcfg
    )
    normalized = d._norm_input(residual, decode=True, site="mlp")
    normalized = ttnn.to_memory_config(normalized, d.decode_inputs["gate_up"])
    packed = ttnn.to_memory_config(d._decode_linear(normalized, "gate_up"), ttnn.L1_MEMORY_CONFIG)
    gw = d.decode_logical_widths["gate_up"] // 2
    gate, up = packed[:, :, :, :gw], packed[:, :, :, gw:]
    product = ttnn.mul(
        gate,
        up,
        input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        memory_config=gate.memory_config(),
        dtype=ttnn.bfloat16,
    )
    return residual, product


def chain_b(d, residual, product):
    """down -> RS -> add (end of _decode_finish), then the next layer's norm input and qkv projection."""
    down = d._output_projection(product, "down")
    residual = ttnn.add(
        residual, ttnn.to_memory_config(down, d.local_residual_memcfg), memory_config=d.local_residual_memcfg
    )
    n = d._norm_input(residual, decode=True, site="attn")
    return residual, d._decode_linear(n, "qkv")


def torch_ref(boundary, x, inp, W1, W2, n_out):
    h = torch.cat(x, -1) + sum(inp[j] @ W1[j] for j in range(TP))
    n = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + EPS)
    outs = []
    for j in range(TP):
        y = n @ W2[j]
        if boundary == "A":
            g, u = y[:, :n_out], y[:, n_out:]
            y = torch.nn.functional.silu(g) * u
        outs.append(y)
    return h, torch.cat(outs, -1)


def run_traced(mesh, fn, label):
    fn()  # compile
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    for _ in range(REPLAYS):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        ttnn.ReadDeviceProfiler(mesh)
    ttnn.release_trace(mesh, tid)
    print(f"TPB_TRACE {label} trace_id={tid}")
    return out


@pytest.mark.parametrize("boundary", ["A", "B"])
def test_tp_boundary_decode_perf(qb2_mesh, boundary):
    torch.manual_seed(0)
    mesh = qb2_mesh
    k_in = HIDDEN if boundary == "A" else INTER
    n_out = INTER // TP if boundary == "A" else (HIDDEN + 2 * 1024) // TP
    x = torch.randn(B, HIDDEN)
    inp = torch.randn(B, k_in)
    w1 = [torch.randn(k_in // TP, HIDDEN) / (k_in**0.5) for _ in range(TP)]
    if boundary == "A":
        gate = [torch.randn(HIDDEN, n_out) / HIDDEN**0.5 for _ in range(TP)]
        up = [torch.randn(HIDDEN, n_out) / HIDDEN**0.5 for _ in range(TP)]
        w2_concat = [torch.cat([g, u], 1) for g, u in zip(gate, up)]  # decode layout: gate | up
        # fused-op (prefill) layout: gate/up interleaved tile by tile
        w2_paired = [
            torch.stack([g.reshape(HIDDEN, -1, 32), u.reshape(HIDDEN, -1, 32)], 2).reshape(HIDDEN, -1)
            for g, u in zip(gate, up)
        ]
    else:
        w2_concat = w2_paired = [torch.randn(HIDDEN, n_out) / HIDDEN**0.5 for _ in range(TP)]
    role1, role2 = ("o", "gate_up") if boundary == "A" else ("down", "qkv")
    d = make_decoder(mesh, {role1: w1, role2: w2_concat})

    # --- composed chain ---
    xs = [x[:, j * 1024 : (j + 1) * 1024] for j in range(TP)]
    residual = shards(mesh, [t[None, None] for t in xs], ttnn.bfloat16, dim=-1, mem=d.local_residual_memcfg)
    inp_chain = shards(mesh, [inp[None, None]], ttnn.bfloat16, dim=-1, mem=ttnn.L1_MEMORY_CONFIG)
    fn = (lambda: chain_a(d, residual, inp_chain)) if boundary == "A" else (lambda: chain_b(d, residual, inp_chain))
    h_c, out_c = run_traced(mesh, fn, f"chain_{boundary}")

    # --- lego fused op ---
    x_t = shards(mesh, [x[None, None]], ttnn.bfloat16, dim=-1)
    inp_t = shards(mesh, [inp[None, None]], ttnn.bfloat16, dim=-1, mem=ttnn.L1_MEMORY_CONFIG)
    w1_t = shards(mesh, w1, ttnn.bfloat8_b)
    w2_t = shards(mesh, w2_paired, ttnn.bfloat4_b if boundary == "A" else ttnn.bfloat8_b)
    gx = mesh.compute_with_storage_grid_size().x  # p300c: 11 (decode_config assumes 12)
    op = TPBoundary(mesh, EPS, boundary, B, n_out, cfg=replace(decode_config(boundary), mm_grid=(gx, 1)))
    h_f, out_f = run_traced(mesh, lambda: op(x_t, inp_t, w1_t, w2_t), f"fused_{boundary}")

    # precision against fp32 torch on the dequantized weights the chain holds
    def dev(t):
        return [ttnn.to_torch(s).float().reshape(ttnn.to_torch(s).shape[-2:]) for s in ttnn.get_device_tensors(t)]

    W1 = [t[:, :HIDDEN] for t in dev(d.decode_weights[role1])]
    W2 = [t[:, : (2 * n_out if boundary == "A" else n_out)] for t in dev(d.decode_weights[role2])]
    h_ref, out_ref = torch_ref(
        boundary, xs, [inp[:, j * (k_in // TP) : (j + 1) * (k_in // TP)] for j in range(TP)], W1, W2, n_out
    )
    for name, h_t, o_t in (("chain", h_c, out_c), ("fused", h_f, out_f)):
        got_h = torch.cat(dev(h_t), -1)[:B]
        got_o = torch.cat([t[:B, :n_out] for t in dev(o_t)], -1)
        ph = torch.corrcoef(torch.stack([h_ref.flatten(), got_h.flatten()]))[0, 1].item()
        po = torch.corrcoef(torch.stack([out_ref.flatten(), got_o.flatten()]))[0, 1].item()
        print(f"TPB_PCC {name}_{boundary} resid={ph:.6f} out={po:.6f}")
