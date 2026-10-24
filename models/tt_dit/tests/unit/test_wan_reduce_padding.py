# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Raw padded output tiles of the reduce users Wan2.2 runs. With PADDUMP_DIR set, every case writes its raw padded
tensor to <dir>/<name>.npy so two builds can be compared bit for bit, padding included."""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest
import torch

import ttnn

SEQ = 512


def _cfg(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def _raw(t):
    p = ttnn.reshape(t, t.padded_shape, t.padded_shape, skip_padding_fill=True)
    return ttnn.to_torch(p).to(torch.float32).numpy()


def _record(name, t):
    raw = _raw(t)
    shape = tuple(t.shape)
    mask = np.ones(raw.shape, dtype=bool)
    mask[tuple(slice(0, s) for s in shape)] = False
    pad = raw[mask]
    nz = int(np.count_nonzero(pad))
    d = hashlib.sha1(raw.tobytes()).hexdigest()[:12]
    print(f"PADCASE {name} logical={shape} padded={tuple(t.padded_shape)} pad_elems={pad.size} pad_nonzero={nz} digest={d}")
    out = os.environ.get("PADDUMP_DIR")
    if out:
        os.makedirs(out, exist_ok=True)
        np.save(os.path.join(out, name + ".npy"), raw)


def _dev(device, t, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


@pytest.mark.parametrize("width", [1024, 4096])
@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("op", ["mean", "sum", "max", "var"])
def test_reduce_padding(device, width, dim, op):
    torch.manual_seed(3)
    x = torch.randn(1, 1, SEQ, width).to(torch.bfloat16)
    tx = _dev(device, x)
    f = {"mean": ttnn.mean, "sum": ttnn.sum, "max": ttnn.max, "var": ttnn.var}[op]
    kw = {} if op == "max" else {"compute_kernel_config": _cfg(device)}
    out = f(tx, dim=dim, keepdim=True, **kw)
    _record(f"red_{op}_d{-dim}_w{width}", out)


@pytest.mark.parametrize("width", [1024, 4096])
def test_rms_stats_padding(device, width):
    torch.manual_seed(4)
    x = torch.randn(1, 1, SEQ, width).to(torch.bfloat16)
    w = torch.randn(1, 1, 1, width).to(torch.bfloat16)
    tx, tw = _dev(device, x), _dev(device, w.expand(1, 1, 32, width).contiguous())
    stats = ttnn.rms_norm_pre_all_gather(tx, compute_kernel_config=_cfg(device), dtype=ttnn.bfloat16)
    _record(f"rmsstats_w{width}", stats)
    out = ttnn.rms_norm_post_all_gather(tx, stats, epsilon=1e-6, weight=tw, compute_kernel_config=_cfg(device))
    _record(f"rmspost_w{width}", out)
    ref = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
    got = ttnn.to_torch(out).float()
    assert torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), dim=0) > 0.999


@pytest.mark.parametrize("heads", [16, 8])
def test_softmax_padding(device, heads):
    torch.manual_seed(5)
    s = (torch.randn(1, heads, SEQ, SEQ) * 3).to(torch.bfloat16)
    out = ttnn.softmax(_dev(device, s), dim=-1, compute_kernel_config=_cfg(device))
    _record(f"softmax_h{heads}", out)


@pytest.mark.parametrize("width", [384, 192, 96])
@pytest.mark.parametrize("kind", ["rms", "ln"])
def test_norm_padding(device, width, kind):
    torch.manual_seed(6)
    x = torch.randn(1, 1, 1024, width).to(torch.bfloat16)
    w = torch.randn(1, 1, 1, width).to(torch.bfloat16)
    tx, tw = _dev(device, x), _dev(device, w.expand(1, 1, 32, width).contiguous())
    f = ttnn.rms_norm if kind == "rms" else ttnn.layer_norm
    out = f(tx, epsilon=1e-6, weight=tw, compute_kernel_config=_cfg(device))
    _record(f"{kind}_w{width}", out)
