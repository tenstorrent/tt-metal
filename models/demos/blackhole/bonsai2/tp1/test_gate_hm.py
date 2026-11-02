# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_1C_GDN_GATE equivalence: ops/gate_hm (gated norm + silu(z) gate, one op) against today's chain (rms_norm on
the head-major o, the zero-copy flatten, silu(z), multiply), both against float64 from the same inputs, for the
fp32 (dense) and bf16 (ternary out-proj) outputs. The op must be at least as close to float64, with identical shape.

    pytest models/demos/blackhole/bonsai2/tp1/test_gate_hm.py -s
"""
import importlib.util
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt.gdn.tp import _silu_mul, head_major_flat_view

_spec = importlib.util.spec_from_file_location(
    "gate_hm_op", os.path.join(os.path.dirname(os.path.abspath(__file__)), "ops", "gate_hm", "op.py")
)
_op = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_op)

NV, D = 48, 128


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("out_dtype", [ttnn.float32, ttnn.bfloat16])
def test_gate_hm(mesh_device, out_dtype):
    torch.manual_seed(0)
    L1 = ttnn.L1_MEMORY_CONFIG
    o_t = torch.randn(NV, 1, D) * 0.5
    z_t = (torch.randn(1, 1, NV * D) * 2).to(torch.bfloat16)
    w_t = (torch.rand(D) + 0.5).to(torch.bfloat16)
    o = ttnn.from_torch(o_t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=L1)
    z = ttnn.from_torch(z_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=L1)
    w = ttnn.from_torch(w_t.reshape(1, 1, D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)

    # today's chain (tp.py GDN_HMO path)
    out_n = ttnn.rms_norm(o, weight=w, epsilon=1e-6, memory_config=L1)
    kw = {"dtype": ttnn.bfloat16} if out_dtype == ttnn.bfloat16 else {}
    chain = ttnn.to_torch(_silu_mul(head_major_flat_view(out_n), z, L1, **kw)).double()

    out = _op.alloc_output(mesh_device, NV * D, dtype=out_dtype)
    fused = ttnn.to_torch(_op.gate_hm(o, z, w, out)).double()

    oo = o_t.double().reshape(NV, D)
    ref = (oo * torch.rsqrt(oo.pow(2).mean(-1, keepdim=True) + 1e-6) * w_t.double()).reshape(1, 1, NV * D)
    ref = ref * torch.nn.functional.silu(z_t.double())
    # Stage diagnostics: each intermediate against its own float64 value, magnitude per channel tile (t = 0..3).
    oo64 = o_t.double().reshape(NV, D)
    rinv64 = torch.rsqrt(oo64.pow(2).mean(-1, keepdim=True) + 1e-6)
    stages = [oo64, oo64 * w_t.double(), oo64 * w_t.double() * rinv64]
    for st_i, want in enumerate(stages):
        got = ttnn.to_torch(_op.gate_hm(o, z, w, _op.alloc_output(mesh_device, NV * D, dtype=out_dtype), stage=st_i))
        got = got.double().reshape(NV, D)
        per_t = [(got[:, 32 * t:32 * t + 32].abs().mean() / want[:, 32 * t:32 * t + 32].abs().mean()).item() for t in range(4)]
        err = ((got - want).abs().max() / want.abs().max()).item()
        print(f"  stage {st_i}: max rel err {err:.3e}  magnitude per channel tile {[round(x, 4) for x in per_t]}")
    # v2 stages 4 / 5: sum(o^2) and rinv, which land in columns 0..3 of every packed row t (channels 32t..32t+3).
    for st_i, want in ((4, oo64.pow(2).sum(-1, keepdim=True)), (5, rinv64)):
        got = ttnn.to_torch(_op.gate_hm(o, z, w, _op.alloc_output(mesh_device, NV * D, dtype=out_dtype), stage=st_i))
        got = got.double().reshape(NV, 4, 32)[:, :, :4]  # [head, packed row t, column c < 4]
        rel = ((got - want.reshape(NV, 1, 1)).abs() / want.abs().reshape(NV, 1, 1)).amax(dim=(0, 2))
        print(f"  stage {st_i}: max rel err per packed row {[f'{x:.2e}' for x in rel.tolist()]}  "
              f"head 0 got {got[0, :, 0].tolist()} want {want[0].item():.5g}")
    assert chain.shape == fused.shape, (chain.shape, fused.shape)
    ec = ((chain - ref).abs().max() / ref.abs().max()).item()
    ef = ((fused - ref).abs().max() / ref.abs().max()).item()
    mag = (fused.abs().mean() / ref.abs().mean()).item()
    print(f"{out_dtype}: chain vs float64 {ec:.3e}   fused vs float64 {ef:.3e}   fused/ref magnitude {mag:.6f}   "
          f"bit-exact with chain {100 * (chain == fused).float().mean().item():.2f}%")
    assert torch.isfinite(fused).all() and abs(mag - 1) < 1e-2
    assert ef <= ec * 1.05
