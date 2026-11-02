# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_1C_GDN_FCONV equivalence: the fused head-major conv op (ops/conv_hm) against today's CONV_HM + GDN_HMI
path (qkv -> [1,80,128] reshape, state copies, mul + 3 mac + silu, fp32 typecast, permute, slices), both scored
against float64 from the same bf16 inputs. Expect: the shifted state bit-exact (it is pure data movement), q/k/v
at least as close to float64 as today's path (fp32 accumulation vs the chain's per-tap bf16 rounding).

    pytest models/demos/blackhole/bonsai2/tp1/test_conv_hm_fused.py -s
"""
import importlib.util
import os

import pytest
import torch

import ttnn

_spec = importlib.util.spec_from_file_location(
    "conv_hm_op", os.path.join(os.path.dirname(os.path.abspath(__file__)), "ops", "conv_hm", "op.py")
)
_op = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_op)

NK, NV, D, K = 16, 48, 128, 4
R = 2 * NK + NV


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_conv_hm_fused(mesh_device):
    torch.manual_seed(0)
    bf = torch.bfloat16

    def dev(t, dt=ttnn.bfloat16):
        return ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh_device)

    qkv_t = torch.randn(1, 1, R * D).to(bf)
    st_t = [torch.randn(1, R, D).to(bf) for _ in range(K)]
    tap_t = [(torch.randn(1, R, D) * 0.3).to(bf) for _ in range(K)]
    qkv = dev(qkv_t)
    taps = [dev(t) for t in tap_t]

    # today's path
    st0 = [dev(t) for t in st_t]
    qkv_hm = ttnn.reshape(qkv, (1, R, D))
    for j in range(K - 1):
        ttnn.copy(st0[j + 1], st0[j])
    ttnn.copy(qkv_hm, st0[K - 1])
    acc = ttnn.multiply(st0[0], taps[0])
    for j in range(1, K):
        acc = ttnn.mac(st0[j], taps[j], acc)
    conv_h = ttnn.permute(ttnn.typecast(ttnn.silu(acc), ttnn.float32), (1, 0, 2))  # [R, 1, D]
    ref_dev = [ttnn.slice(conv_h, (a, 0, 0), (b, 1, D)) for a, b in ((0, NK), (NK, 2 * NK), (2 * NK, R))]

    # fused
    st1 = [dev(t) for t in st_t]
    q, k, v = _op.alloc_outputs(mesh_device, NK, NV)
    _op.conv_hm(qkv, st1, taps, q, k, v)

    # float64 reference
    win = [t.double() for t in st_t[1:]] + [qkv_t.double().reshape(1, R, D)]
    conv64 = torch.nn.functional.silu(sum(win[j] * tap_t[j].double() for j in range(K))).reshape(R, 1, D)
    ref64 = [conv64[:NK], conv64[NK : 2 * NK], conv64[2 * NK :]]

    for j in range(K):
        a, b = ttnn.to_torch(st0[j]), ttnn.to_torch(st1[j])
        assert a.shape == b.shape and bool((a == b).all()), f"state row {j} differs"
    print("state shift: all K rows bit-exact")
    for name, r_dev, f_dev, r64 in zip("qkv", ref_dev, (q, k, v), ref64):
        a, b = ttnn.to_torch(r_dev).double(), ttnn.to_torch(f_dev).double()
        assert a.shape == b.shape, (name, a.shape, b.shape)
        ea, eb = (a - r64).abs().max().item(), (b - r64).abs().max().item()
        print(f"{name}: shape {tuple(b.shape)}  today vs float64 {ea:.3e}  fused vs float64 {eb:.3e}  "
              f"bit-exact with today {100 * (a == b).float().mean().item():.2f}%")
        assert eb <= ea * 1.05
