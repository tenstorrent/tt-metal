# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Single-device tests of the ops the Wan2.2 UMT5 text encoder runs, at the LoudBox (2, 4) per-device shapes.

UMT5-XXL: d_model 4096, 64 heads of d_kv 64, sequence 512. Tensor parallel 4 leaves 16 heads per device; 8 is the
Galaxy per-device count.
"""

import os
import statistics

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program

SEQ = 512
D_KV = 64
HEADS = [16, 8]
WIDTHS = [1024, 4096]


def _hifi4(device, l1_acc=False):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=l1_acc,
    )


def _to_dev(device, t, dtype):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _qk(device, heads):
    torch.manual_seed(0)
    q = torch.randn(1, heads, SEQ, D_KV, dtype=torch.bfloat16)
    k = torch.randn(1, heads, D_KV, SEQ, dtype=torch.bfloat16)
    return q, k, _to_dev(device, q, ttnn.bfloat16), _to_dev(device, k, ttnn.bfloat16)


def _av(device, heads):
    torch.manual_seed(1)
    a = torch.softmax(torch.randn(1, heads, SEQ, SEQ), dim=-1).to(torch.bfloat16)
    v = torch.randn(1, heads, SEQ, D_KV, dtype=torch.bfloat16)
    return a, v, _to_dev(device, a, ttnn.bfloat16), _to_dev(device, v, ttnn.bfloat16)


def _scores(device, heads, dtype):
    torch.manual_seed(2)
    s = (torch.randn(1, heads, SEQ, SEQ) * 3).to(torch.float32 if dtype == ttnn.float32 else torch.bfloat16)
    return s, _to_dev(device, s, dtype)


def _rms_inputs(device, width):
    torch.manual_seed(3)
    x = torch.randn(1, 1, SEQ, width, dtype=torch.bfloat16)
    w = torch.rand(1, 1, 1, width, dtype=torch.bfloat16) + 0.5
    return x, w, _to_dev(device, x, ttnn.bfloat16), ttnn.from_torch(
        w.reshape(1, 1, width // 32, 32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )


def _rms_ref(x, w, eps):
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()


def _rms(device, tt_x, tt_w, eps, cfg):
    stats = ttnn.rms_norm_pre_all_gather(tt_x, compute_kernel_config=cfg, dtype=ttnn.bfloat16)
    return ttnn.rms_norm_post_all_gather(tt_x, stats, epsilon=eps, weight=tt_w, compute_kernel_config=cfg)


@pytest.mark.parametrize("heads", HEADS)
@pytest.mark.parametrize("l1_acc", [False, True], ids=["noacc", "l1acc"])
def test_umt5_matmul_qk(device, heads, l1_acc):
    q, k, tq, tk = _qk(device, heads)
    out = ttnn.matmul(tq, tk, compute_kernel_config=_hifi4(device, l1_acc))
    assert_with_pcc(torch.matmul(q.float(), k.float()), ttnn.to_torch(out).float(), 0.999)


@pytest.mark.parametrize("heads", HEADS)
def test_umt5_matmul_qk_default(device, heads):
    q, k, tq, tk = _qk(device, heads)
    out = ttnn.matmul(tq, tk)
    assert_with_pcc(torch.matmul(q.float(), k.float()), ttnn.to_torch(out).float(), 0.998)


@pytest.mark.parametrize("heads", HEADS)
def test_umt5_matmul_av(device, heads):
    a, v, ta, tv = _av(device, heads)
    out = ttnn.matmul(ta, tv)
    assert_with_pcc(torch.matmul(a.float(), v.float()), ttnn.to_torch(out).float(), 0.998)


@pytest.mark.parametrize("heads", HEADS)
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16], ids=["fp32", "bf16"])
def test_umt5_softmax(device, heads, dtype):
    s, ts = _scores(device, heads, dtype)
    out = ttnn.softmax(ts, dim=-1, compute_kernel_config=_hifi4(device))
    assert_with_pcc(torch.softmax(s.float(), dim=-1), ttnn.to_torch(out).float(), 0.9995)


@pytest.mark.parametrize("heads", HEADS)
def test_umt5_softmax_numeric_stable(device, heads):
    s, ts = _scores(device, heads, ttnn.float32)
    out = ttnn.softmax(ts, dim=-1, numeric_stable=True, compute_kernel_config=_hifi4(device))
    assert_with_pcc(torch.softmax(s.float(), dim=-1), ttnn.to_torch(out).float(), 0.9995)


@pytest.mark.parametrize("width", WIDTHS)
def test_umt5_rmsnorm_distributed(device, width):
    x, w, tx, tw = _rms_inputs(device, width)
    out = _rms(device, tx, tw, 1e-6, _hifi4(device))
    assert_with_pcc(_rms_ref(x, w, 1e-6), ttnn.to_torch(out).float(), 0.998)


def _measure(device, tag, call):
    for _ in range(2):
        call()
    ttnn.synchronize_device(device)
    ns = []
    for _ in range(int(os.environ.get("UMT5_N", "10"))):
        _, rec = profile_realtime_program(device, call, collect_all=True)
        rec = [r for r in rec if r["duration_ns"] > 0]
        assert rec, "no program record"
        ns.append(sum(r["duration_ns"] for r in rec))
    for i, v in enumerate(ns):
        print(f"UMT5-SAMPLE {tag} i={i} duration_ns={v:.0f}")
    print(f"UMT5-MEDIAN {tag} duration_ns={statistics.median(ns):.0f}")


@pytest.mark.parametrize("heads", HEADS)
def test_umt5_matmul_qk_perf(device, heads):
    _, _, tq, tk = _qk(device, heads)
    _measure(device, f"mm_qk_h{heads}", lambda: ttnn.matmul(tq, tk))


@pytest.mark.parametrize("heads", HEADS)
def test_umt5_matmul_av_perf(device, heads):
    _, _, ta, tv = _av(device, heads)
    _measure(device, f"mm_av_h{heads}", lambda: ttnn.matmul(ta, tv))


@pytest.mark.parametrize("heads", HEADS)
def test_umt5_softmax_perf(device, heads):
    _, ts = _scores(device, heads, ttnn.float32)
    cfg = _hifi4(device)
    _measure(device, f"softmax_h{heads}", lambda: ttnn.softmax(ts, dim=-1, compute_kernel_config=cfg))


@pytest.mark.parametrize("width", WIDTHS)
def test_umt5_rmsnorm_distributed_perf(device, width):
    _, _, tx, tw = _rms_inputs(device, width)
    cfg = _hifi4(device)
    _measure(device, f"rms_pre_w{width}", lambda: ttnn.rms_norm_pre_all_gather(tx, compute_kernel_config=cfg, dtype=ttnn.bfloat16))
    stats = ttnn.rms_norm_pre_all_gather(tx, compute_kernel_config=cfg, dtype=ttnn.bfloat16)
    _measure(
        device,
        f"rms_post_w{width}",
        lambda: ttnn.rms_norm_post_all_gather(tx, stats, epsilon=1e-6, weight=tw, compute_kernel_config=cfg),
    )
