# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Single-device Wan2.2 VAE kernels (conv3d, RMSNorm, tilize, untilize) at LoudBox (2,4) 480p per-device shapes."""

from __future__ import annotations

import hashlib
import os
import statistics

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program
from tests.ttnn.utils_for_testing import check_with_pcc

ALIGN = 32

# (Cin, Cout, kernel, T, H, W, (C_in_block, C_out_block, T_out_block, H_out_block, W_out_block))
CONVS = {
    "s0_in": (32, 384, (3, 3, 3), 9, 30, 26, (32, 384, 1, 2, 16)),
    "s0_res": (384, 384, (3, 3, 3), 9, 30, 26, (96, 96, 7, 16, 2)),
    "s0_tconv": (384, 768, (3, 1, 1), 9, 30, 26, (192, 768, 1, 16, 2)),
    "s1_up": (384, 192, (1, 3, 3), 14, 60, 52, (384, 96, 1, 4, 8)),
    "s1_chg": (192, 384, (3, 3, 3), 16, 60, 52, (96, 128, 5, 4, 8)),
    "s1_res": (384, 384, (3, 3, 3), 16, 60, 52, (96, 128, 5, 8, 4)),
    "s1_tconv": (384, 768, (3, 1, 1), 16, 60, 52, (384, 384, 3, 8, 4)),
    "s2_up": (384, 192, (1, 3, 3), 28, 120, 104, (384, 96, 1, 4, 8)),
    "s2_res": (192, 192, (3, 3, 3), 30, 120, 104, (96, 96, 6, 8, 4)),
    "s3_up": (192, 96, (1, 3, 3), 28, 240, 208, (192, 96, 1, 4, 16)),
    "s3_res": (96, 96, (3, 3, 3), 30, 240, 208, (96, 96, 7, 2, 16)),
    "out": (96, 3, (3, 3, 3), 30, 240, 208, (96, 32, 1, 16, 8)),
}
CORRECT = ["s0_in", "s0_res", "s0_tconv", "s1_up", "s1_chg", "s1_res", "s1_tconv", "s2_up", "s2_res", "s3_up", "s3_res", "out"]
PERF = ["s0_res", "s1_res", "s2_res", "s3_res", "s1_up", "s0_tconv"]

# (C, rows) with rows = T * H * W for a 7 frame chunk at each stage
NORMS = {
    "s0": (384, 7 * 30 * 26),
    "s1": (384, 7 * 60 * 52),
    "s2": (192, 7 * 120 * 104),
    "s3": (96, 7 * 240 * 208),
}

# (T, H, W, C) of the BTHWC activation converted between layouts
LAYOUTS = {
    "s0": (7, 30, 26, 384),
    "s1": (7, 60, 52, 384),
    "s2": (7, 120, 104, 192),
    "s3": (7, 240, 208, 96),
}


def _samples(mesh_device, call, tag):
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("Real-time profiler must be active")
    for _ in range(2):
        call()
    ttnn.synchronize_device(mesh_device)
    ns = []
    for _ in range(int(os.environ.get("VAE_N", "10"))):
        _, rec = profile_realtime_program(mesh_device, call, collect_all=True)
        rec = [r for r in rec if r["duration_ns"] > 0]
        assert rec, "no program record"
        ns.append(max(r["duration_ns"] for r in rec))
    for i, v in enumerate(ns):
        print(f"VAE-SAMPLE {tag} i={i} duration_ns={v:.0f}")
    print(f"VAE-MEDIAN {tag} duration_ns={statistics.median(ns):.0f}")


def _digest(t):
    return hashlib.sha1(t.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()[:12]


def _conv(mesh_device, key, T, H, W, golden):
    cin, cout, k, _, _, _, blk = CONVS[key]
    torch.manual_seed(0)
    pad = (0, k[1] // 2, k[2] // 2)
    grid = mesh_device.compute_with_storage_grid_size()
    cfg = ttnn.Conv3dConfig(
        weights_dtype=ttnn.bfloat16,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        C_in_block=blk[0],
        C_out_block=blk[1],
        T_out_block=blk[2],
        H_out_block=blk[3],
        W_out_block=blk[4],
        compute_with_storage_grid_size=(grid.x, grid.y),
    )
    x = torch.randn(1, T, H, W, cin, dtype=torch.bfloat16)
    w = (torch.randn(cout, cin, *k) * (cin * k[0] * k[1] * k[2]) ** -0.5).to(torch.bfloat16)
    b = torch.randn(1, cout, dtype=torch.bfloat16)
    x_pad = x if cin % ALIGN == 0 else torch.nn.functional.pad(x, (0, ALIGN - cin % ALIGN))
    tt_x = ttnn.from_torch(x_pad, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_w = ttnn.experimental.prepare_conv3d_weights(
        weight_tensor=ttnn.from_torch(w, dtype=ttnn.bfloat16, pad_value=0),
        groups=1,
        C_in_block=cfg.C_in_block,
        alignment=ALIGN,
        device=mesh_device,
    )
    tt_b = ttnn.from_torch(b, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, pad_value=0)
    kc = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    def call():
        return ttnn.experimental.conv3d(
            input_tensor=tt_x,
            weight_tensor=tt_w,
            device=mesh_device,
            bias_tensor=tt_b,
            dtype=ttnn.bfloat16,
            output_channels=cout,
            kernel_size=k,
            stride=(1, 1, 1),
            groups=1,
            padding=pad,
            dilation=(1, 1, 1),
            padding_mode="zeros",
            config=cfg,
            compute_kernel_config=kc,
        )

    return call, x, w, b, pad, cout


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("key", CORRECT)
def test_wan_vae_conv3d(mesh_device, key):
    cin, cout, k, T, H, W, blk = CONVS[key]
    T, H, W = min(T, 5), min(H, 16), min(W, 16)
    call, x, w, b, pad, cout = _conv(mesh_device, key, T, H, W, True)
    out = ttnn.to_torch(call())
    t_out, h_out, w_out = T - k[0] + 1, H + 2 * pad[1] - k[1] + 1, W + 2 * pad[2] - k[2] + 1
    out = out.reshape(1, t_out, h_out, w_out, cout).permute(0, 4, 1, 2, 3).float()
    ref = torch.nn.functional.conv3d(
        x.permute(0, 4, 1, 2, 3).float(), w.float(), b.reshape(-1).float(), padding=pad
    )
    ok, msg = check_with_pcc(ref, out, pcc=0.999)
    logger.info(f"VAE-CONV {key} {msg}")
    assert ok, msg


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("key", PERF)
def test_wan_vae_conv3d_perf(mesh_device, key):
    cin, cout, k, T, H, W, blk = CONVS[key]
    call, *_ = _conv(mesh_device, key, T, H, W, False)
    print(f"VAE-DIGEST conv_{key} {_digest(ttnn.to_torch(call()).to(torch.bfloat16))}")
    _samples(mesh_device, call, f"conv_{key}")


def _norm(mesh_device, name, layout):
    c, rows = NORMS[name]
    torch.manual_seed(0)
    x = torch.randn(1, 1, rows, c, dtype=torch.bfloat16)
    g = torch.rand(1, c, dtype=torch.bfloat16) + 0.5
    tt_x = ttnn.from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, layout=layout)
    tt_g = ttnn.from_torch(g, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    kc = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    def call():
        return ttnn.experimental.dit_rms_norm_unary_fused(
            tt_x, epsilon=1e-12, weight=tt_g, bias=None, compute_kernel_config=kc, activation="silu"
        )

    xf = x.float()
    ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-12) * g.float()
    return call, torch.nn.functional.silu(ref)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
@pytest.mark.parametrize("name", list(NORMS))
def test_wan_vae_rmsnorm(mesh_device, name, layout):
    call, ref = _norm(mesh_device, name, layout)
    out = ttnn.to_torch(call()).float()
    ok, msg = check_with_pcc(ref, out, pcc=0.9995)
    logger.info(f"VAE-NORM {name} {msg}")
    assert ok, msg


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
@pytest.mark.parametrize("name", list(NORMS))
def test_wan_vae_rmsnorm_perf(mesh_device, name, layout):
    call, _ = _norm(mesh_device, name, layout)
    tag = f"norm_{name}_{'tile' if layout == ttnn.TILE_LAYOUT else 'rm'}"
    print(f"VAE-DIGEST {tag} {_digest(ttnn.to_torch(call()).to(torch.bfloat16))}")
    _samples(mesh_device, call, tag)


def _layout(mesh_device, name, to_tile):
    t, h, w, c = LAYOUTS[name]
    torch.manual_seed(0)
    x = torch.randn(1, t, h, w, c, dtype=torch.bfloat16)
    src = ttnn.TILE_LAYOUT if not to_tile else ttnn.ROW_MAJOR_LAYOUT
    dst = ttnn.TILE_LAYOUT if to_tile else ttnn.ROW_MAJOR_LAYOUT
    tt_x = ttnn.from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, layout=src)

    def call():
        return ttnn.to_layout(tt_x, dst)

    return call, x


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("to_tile", [True, False], ids=["tilize", "untilize"])
@pytest.mark.parametrize("name", list(LAYOUTS))
def test_wan_vae_layout(mesh_device, name, to_tile):
    call, x = _layout(mesh_device, name, to_tile)
    out = ttnn.to_torch(call())
    assert out.shape == x.shape
    assert torch.equal(out, x)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("to_tile", [True, False], ids=["tilize", "untilize"])
@pytest.mark.parametrize("name", list(LAYOUTS))
def test_wan_vae_layout_perf(mesh_device, name, to_tile):
    call, _ = _layout(mesh_device, name, to_tile)
    _samples(mesh_device, call, f"layout_{name}_{'tilize' if to_tile else 'untilize'}")
