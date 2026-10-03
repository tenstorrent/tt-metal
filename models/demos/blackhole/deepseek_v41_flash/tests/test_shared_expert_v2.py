# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DSV41SharedExpertV2 vs DSV41SharedExpert: PCC (layer 2 weights) and per-call time in traces ((t(3)-t(1))/2)."""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference.ref_layer import pcc
from models.demos.blackhole.deepseek_v41_flash.tests.test_shared_variants import chain_ms
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import load_moe_layer
from models.demos.blackhole.deepseek_v41_flash.tt.shared_expert import DSV41SharedExpert
from models.demos.blackhole.deepseek_v41_flash.tt.shared_expert_v2 import DSV41SharedExpertV2

D = 5120
VARIANTS = [
    ("dram", {}),
    ("split1d", {}),
    ("fused1d", {}),
]
if os.environ.get("V2_SWEEP"):
    VARIANTS = eval(os.environ["V2_SWEEP"])
TS = [int(t) for t in os.environ.get("V2_TS", "4,8,16,32").split(",")]


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 200_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_shared_expert_v2(mesh_device):
    md = mesh_device
    w = load_moe_layer(2, experts=[0])
    sid = next(iter(w["shared_w0"]))
    w0, w1, w2 = w["shared_w0"][sid], w["shared_w1"][sid], w["shared_w2"][sid]
    cur = DSV41SharedExpert(md, w0, w1, w2)
    rep = ttnn.ReplicateTensorToMesh(md)
    dev0 = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
    # realistic input: unit-variance rows; also a x4 amplified one so the clamps are active
    for name, kw in VARIANTS:
        try:
            v2 = DSV41SharedExpertV2(md, w0, w1, w2, mode=name, **kw)
        except Exception as e:
            print(f"SE2 {name} {kw} BUILD FAILED {str(e)[:200]}", flush=True)
            continue
        for T in TS:
            line = f"SE2 {name} {kw} T={T}"
            try:
                ps = []
                for scale in (1.0, 4.0):
                    hh = (torch.randn(1, 1, T, D) * scale).to(torch.bfloat16)
                    h = ttnn.from_torch(
                        hh,
                        device=md,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=rep,
                    )
                    ps.append(pcc(dev0(v2.forward(h)), dev0(cur.forward(h))))
                line += f" pcc {ps[0]:.6f}/{ps[1]:.6f}"
                if not os.environ.get("V2_NOTIME"):
                    line += f" v2 {chain_ms(md, lambda: v2.forward(h)) * 1e3:.1f}us cur {chain_ms(md, lambda: cur.forward(h)) * 1e3:.1f}us"
            except Exception as e:
                line += f" FAILED {str(e)[:300]}"
            print(line, flush=True)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 200_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_shared_expert_v2_breakdown(mesh_device):
    """Per-op cost of the fused1d variant inside a trace."""
    md = mesh_device
    w = load_moe_layer(2, experts=[0])
    sid = next(iter(w["shared_w0"]))
    w0, w1, w2 = w["shared_w0"][sid], w["shared_w1"][sid], w["shared_w2"][sid]
    kw = eval(os.environ.get("V2_BKW", "dict(pn1=4, bw1=8, pn2=4, bw2=4)"))
    v2 = DSV41SharedExpertV2(md, w0, w1, w2, mode=os.environ.get("V2_BMODE", "fused1d"), **kw)
    rep = ttnn.ReplicateTensorToMesh(md)
    T = 4
    h = ttnn.from_torch(
        torch.randn(1, 1, T, D).to(torch.bfloat16),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    lin = lambda x, w, pc, dt: ttnn.linear(x, w, dtype=dt, compute_kernel_config=v2.ckc, program_config=pc)
    gu = lin(h, v2.w01, v2.pc1, ttnn.float32)
    g, u = gu[:, :, :, : v2.inter], gu[:, :, :, v2.inter :]
    act = ttnn.multiply(
        g, u, input_tensor_a_activations=v2.act_a, input_tensor_b_activations=v2.act_b, dtype=ttnn.float32
    )
    ms = lambda f: chain_ms(md, f) * 1e3
    print(f"SE2B mm1 {ms(lambda: lin(h, v2.w01, v2.pc1, ttnn.float32)):.1f}us", flush=True)
    print(f"SE2B mm1 bf16 out {ms(lambda: lin(h, v2.w01, v2.pc1, ttnn.bfloat16)):.1f}us", flush=True)
    print(f"SE2B slice x2 {ms(lambda: (gu[:, :, :, : v2.inter], gu[:, :, :, v2.inter:])):.1f}us", flush=True)
    print(
        f"SE2B mul {ms(lambda: ttnn.multiply(g, u, input_tensor_a_activations=v2.act_a, input_tensor_b_activations=v2.act_b, dtype=ttnn.float32)):.1f}us",
        flush=True,
    )
    print(f"SE2B mm2 {ms(lambda: lin(act, v2.w2, v2.pc2, ttnn.float32)):.1f}us", flush=True)
    print(f"SE2B tiny op {ms(lambda: ttnn.add(h, h)):.1f}us", flush=True)
    print(f"SE2B full {ms(lambda: v2.forward(h)):.1f}us", flush=True)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 200_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_shared_expert_v2_breakdown_dram(mesh_device):
    md = mesh_device
    w = load_moe_layer(2, experts=[0])
    sid = next(iter(w["shared_w0"]))
    w0, w1, w2 = w["shared_w0"][sid], w["shared_w1"][sid], w["shared_w2"][sid]
    v2 = DSV41SharedExpertV2(md, w0, w1, w2, mode="dram", **eval(os.environ.get("V2_BKW", "{}")))
    rep = ttnn.ReplicateTensorToMesh(md)
    T = 4
    h = ttnn.from_torch(
        torch.randn(1, 1, T, D).to(torch.bfloat16),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    ms = lambda f: chain_ms(md, f) * 1e3
    lin = lambda x, w, pc, mem, dt: ttnn.linear(
        x, w, dtype=dt, compute_kernel_config=v2.ckc, program_config=pc, memory_config=mem
    )
    hs = ttnn.to_memory_config(h, v2.m_in)
    g = lin(hs, v2.w0, v2.pc1, v2.m_mid, v2.mid_dtype)
    u = lin(hs, v2.w1, v2.pc1, v2.m_mid, v2.mid_dtype)
    act = ttnn.multiply(
        g,
        u,
        input_tensor_a_activations=v2.act_a,
        input_tensor_b_activations=v2.act_b,
        dtype=v2.mid_dtype,
        memory_config=v2.m_mid,
    )
    o = lin(act, v2.w2, v2.pc2, v2.m_out, ttnn.float32)
    print(f"SE2B to_L1 shard {ms(lambda: ttnn.to_memory_config(h, v2.m_in)):.1f}us", flush=True)
    print(f"SE2B gate mm {ms(lambda: lin(hs, v2.w0, v2.pc1, v2.m_mid, v2.mid_dtype)):.1f}us", flush=True)
    print(
        f"SE2B mul {ms(lambda: ttnn.multiply(g, u, input_tensor_a_activations=v2.act_a, input_tensor_b_activations=v2.act_b, dtype=v2.mid_dtype, memory_config=v2.m_mid)):.1f}us",
        flush=True,
    )
    print(f"SE2B mm2 {ms(lambda: lin(act, v2.w2, v2.pc2, v2.m_out, ttnn.float32)):.1f}us", flush=True)
    print(f"SE2B out to DRAM {ms(lambda: ttnn.to_memory_config(o, ttnn.DRAM_MEMORY_CONFIG)):.1f}us", flush=True)
