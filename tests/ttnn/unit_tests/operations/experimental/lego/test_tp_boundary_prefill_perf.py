# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# Perf harness: prefill boundary A / B at S rows on a QB2 1x4 ring (4x p300c).
# (1) the composed chain from the llama31_8b_qb2 decoder's own prefill ops and configs, (2) the lego TPBoundary
# fused op; each captured in a metal trace and replayed REPLAYS times. Run with TT_METAL_DEVICE_PROFILER=1 and read
# the times with tp_boundary_trace_span.py. prefill_config assumes a 12-wide grid; mm_grid and rs_workers are
# narrowed below for p300c's 11 columns.

from dataclasses import replace

import pytest
import torch

import ttnn
from models.common.tt_ccl import TT_CCL
from models.demos.llama31_8b_qb2.tt.decoder import LlamaDecoder
from models.demos.llama31_8b_qb2.tt.precision import load_precision_config

from ttnn.lego.tp_boundary import TPBoundary, prefill_config

from .test_tp_boundary_decode_perf import qb2_mesh, shards, run_traced  # noqa: F401

TP, HIDDEN, INTER = 4, 4096, 14336
EPS = 1e-5


def make_decoder(mesh, weights):
    d = LlamaDecoder.__new__(LlamaDecoder)
    p = load_precision_config()
    d.mesh_device = mesh
    d.eps = EPS
    d.grid = mesh.compute_with_storage_grid_size()
    d.ccl = TT_CCL(mesh)
    d.decode_workspace = None
    d.decode_weights = {}
    d.prefill_program = ttnn.MinimalMatmulConfig(
        M_block_size=4,
        K_block_size=8,
        N_block_size=16,
        subblock_h=2,
        subblock_w=4,
        compute_with_storage_grid_size=d.grid,
    )
    d.prefill_mlp_program = ttnn.MinimalMatmulConfig(
        M_block_size=4,
        K_block_size=8,
        N_block_size=24,
        subblock_h=2,
        subblock_w=4,
        compute_with_storage_grid_size=ttnn.CoreCoord(11, 8),
    )
    d.prefill_weights = {
        role: shards(mesh, ts, getattr(ttnn, p["weight_groups"][role])) for role, ts in weights.items()
    }
    d._configure_precision(p)
    return d


# decoder._finish's gate_up config for rows <= 256
small_mlp_program = ttnn.MinimalMatmulConfig(
    M_block_size=1,
    K_block_size=8,
    N_block_size=8,
    subblock_h=1,
    subblock_w=4,
    compute_with_storage_grid_size=ttnn.CoreCoord(11, 8),
)


def chain_a(d, x, attention):
    """_finish up to the SwiGLU product: o_proj -> RS -> add -> AG -> norm -> gate_up (fused SwiGLU)."""
    projected = d._rs(d._linear(attention, "o"), decode=False, site="o")
    residual = ttnn.add(x, projected, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    normalized = d._norm_input(residual, decode=False, site="mlp")
    normalized = ttnn.typecast(normalized, d.activation_dtype) if normalized.dtype != d.activation_dtype else normalized
    product = ttnn.experimental.minimal_matmul(
        normalized,
        d.prefill_weights["gate_up"],
        dtype=d.activation_output_dtype,
        fuse_swiglu=True,
        config=small_mlp_program if x.padded_shape[2] <= 256 else d.prefill_mlp_program,
        compute_kernel_config=d.prefill_computes["gate_up"],
        memory_config=ttnn.L1_MEMORY_CONFIG if x.padded_shape[2] <= 256 else ttnn.DRAM_MEMORY_CONFIG,
    )
    return residual, product


def chain_b(d, x, product):
    """down -> RS -> add, then the next layer's AG -> norm -> qkv."""
    down = d._rs(d._linear(product, "down"), decode=False, site="down")
    residual = ttnn.add(x, down, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    n = d._norm_input(residual, decode=False, site="attn")
    return residual, d._linear(n, "qkv")


@pytest.mark.parametrize("S", [128, 1024])
@pytest.mark.parametrize("boundary", ["A", "B"])
def test_tp_boundary_prefill_perf(qb2_mesh, boundary, S):
    torch.manual_seed(0)
    mesh = qb2_mesh
    k_in = HIDDEN if boundary == "A" else INTER
    n_out = INTER // TP if boundary == "A" else (HIDDEN + 2 * 1024) // TP
    n_w2 = 2 * n_out if boundary == "A" else n_out
    x = torch.randn(S, HIDDEN)
    inp = torch.randn(S, k_in)
    w1 = [torch.randn(k_in // TP, HIDDEN) / (k_in**0.5) for _ in range(TP)]
    w2 = [torch.randn(HIDDEN, n_w2) / HIDDEN**0.5 for _ in range(TP)]  # A: gate/up interleaved tile by tile
    role1, role2 = ("o", "gate_up") if boundary == "A" else ("down", "qkv")
    d = make_decoder(mesh, {role1: w1, role2: w2})

    xs = [x[:, j * 1024 : (j + 1) * 1024] for j in range(TP)]
    x_chain = shards(mesh, [t[None, None] for t in xs], ttnn.bfloat16, dim=-1)
    inp_t = shards(mesh, [inp[None, None]], ttnn.bfloat16, dim=-1, mem=ttnn.L1_MEMORY_CONFIG)
    fn = (lambda: chain_a(d, x_chain, inp_t)) if boundary == "A" else (lambda: chain_b(d, x_chain, inp_t))
    h_c, out_c = run_traced(mesh, fn, f"chain_{boundary}")

    x_t = shards(mesh, [x[None, None]], ttnn.bfloat16, dim=-1)
    gx = mesh.compute_with_storage_grid_size().x  # p300c fast dispatch: 11 (prefill_config assumes 12)
    cfg = prefill_config(boundary, S)
    cfg = replace(
        cfg,
        mm_grid=(gx, cfg.mm_grid[1]),
        mm1_grid=(gx, cfg.mm1_grid[1]) if cfg.mm1_grid else None,
        rs_workers=4 if gx < 12 else cfg.rs_workers,
    )
    op = TPBoundary(mesh, EPS, boundary, S, n_out, cfg=cfg)
    h_f, out_f = run_traced(
        mesh, lambda: op(x_t, inp_t, d.prefill_weights[role1], d.prefill_weights[role2]), f"fused_{boundary}"
    )

    def dev(t):
        return [ttnn.to_torch(s).float().reshape(ttnn.to_torch(s).shape[-2:]) for s in ttnn.get_device_tensors(t)]

    W1, W2 = dev(d.prefill_weights[role1]), dev(d.prefill_weights[role2])
    h_ref = x + sum(inp[:, j * (k_in // TP) : (j + 1) * (k_in // TP)] @ W1[j] for j in range(TP))
    n = h_ref * torch.rsqrt(h_ref.pow(2).mean(-1, keepdim=True) + EPS)
    outs = []
    for j in range(TP):
        y = n @ W2[j]
        if boundary == "A":
            y = y.reshape(S, -1, 2, 32)
            y = (torch.nn.functional.silu(y[:, :, 0]) * y[:, :, 1]).reshape(S, -1)
        outs.append(y)
    out_ref = torch.cat(outs, -1)
    for name, h_t, o_t in (("chain", h_c, out_c), ("fused", h_f, out_f)):
        got_h = torch.cat(dev(h_t), -1)
        got_o = torch.cat(dev(o_t), -1)
        ph = torch.corrcoef(torch.stack([h_ref.flatten(), got_h.flatten()]))[0, 1].item()
        po = torch.corrcoef(torch.stack([out_ref.flatten(), got_o.flatten()]))[0, 1].item()
        print(f"TPB_PCC {name}_{boundary}{S} resid={ph:.6f} out={po:.6f}")
