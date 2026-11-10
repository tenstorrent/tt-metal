# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.lego.tp_boundary.TPBoundary on a 1x4 ring against fp32 torch math on the same (quantized) weights.

Boundary A: o_proj -> reduce-scatter + residual -> all-gather -> RMSNorm -> gate_up with SwiGLU
Boundary B: down   -> reduce-scatter + residual -> all-gather -> RMSNorm -> qkv
Shapes are Llama-3.1-8B at TP=4 (hidden 4096, intermediate 14336). S=32 runs the decode configuration.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from ttnn.lego.tp_boundary import TPBoundary

TP, HIDDEN, INTER = 4, 4096, 14336
EPS = 1e-5


def router_config(max_payload_size):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


def shards(mesh, ts, dtype, dim=0, mem=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        torch.cat(ts, dim=dim).contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=mem,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim),
    )


def device_values(t):
    """Per-device tensors back on host as fp32 (block-float weights come back dequantized)."""
    return [ttnn.to_torch(d).float().reshape(ttnn.to_torch(d).shape[-2:]) for d in ttnn.get_device_tensors(t)]


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "fabric_router_config": router_config(8192),
            "l1_small_size": 16384,
        }
    ],
    indirect=True,
    ids=["fabric_ring"],
)
@pytest.mark.parametrize("S", [32, 128, 1024], ids=["decode", "S128", "S1024"])
@pytest.mark.parametrize("boundary", ["A", "B"])
def test_tp_boundary(mesh_device, boundary, S):
    torch.manual_seed(0)
    mesh = mesh_device
    k_in = HIDDEN if boundary == "A" else INTER  # o_proj / down input width (all devices)
    n_out = INTER // TP if boundary == "A" else (HIDDEN + 2 * 1024) // TP  # per device: SwiGLU product / qkv
    n_w2 = 2 * n_out if boundary == "A" else n_out

    x = torch.randn(S, HIDDEN)
    inp = torch.randn(S, k_in)
    w1 = [torch.randn(k_in // TP, HIDDEN) / (k_in**0.5) for _ in range(TP)]
    w2 = [torch.randn(HIDDEN, n_w2) / (HIDDEN**0.5) for _ in range(TP)]

    x_t = shards(mesh, [x[None, None]], ttnn.bfloat16, dim=-1)
    inp_t = shards(mesh, [inp[None, None]], ttnn.bfloat16, dim=-1, mem=ttnn.L1_MEMORY_CONFIG)
    w1_t = shards(mesh, w1, ttnn.bfloat8_b)
    w2_t = shards(mesh, w2, ttnn.bfloat4_b if boundary == "A" else ttnn.bfloat8_b)

    op = TPBoundary(mesh, EPS, boundary, S, n_out)
    for _ in range(2):  # the second call reuses the cached program and must give the same result
        h_t, out_t = op(x_t, inp_t, w1_t, w2_t)

    # fp32 reference on the values the devices hold (bf16 activations, block-float weights)
    xs, ins = device_values(x_t), device_values(inp_t)
    W1, W2 = device_values(w1_t), device_values(w2_t)
    h = torch.cat(xs, -1) + sum(ins[d] @ W1[d] for d in range(TP))
    n = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + EPS)
    outs = []
    for d in range(TP):
        y = n @ W2[d]
        if boundary == "A":  # gate/up interleaved tile by tile
            y = y.reshape(S, -1, 2, 32)
            y = torch.nn.functional.silu(y[:, :, 0]) * y[:, :, 1]
            y = y.reshape(S, -1)
        outs.append(y)
    ref_out = torch.cat(outs, -1)

    got_h = torch.cat(device_values(h_t), -1)
    got_out = torch.cat(device_values(out_t), -1)
    passing, pcc = comp_pcc(h, got_h, 0.9999)
    assert passing, f"residual PCC {pcc}"
    passing, pcc = comp_pcc(ref_out, got_out, 0.999)
    assert passing, f"output PCC {pcc}"
