# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Layout tax of the late vocoder bands: per-op time on the same 1.9 MB fp32 tensor laid out as ``(T, C)`` rows of
8 channels (band 6 today, 32-byte DRAM pages) versus ``k`` time steps packed into one row ``(T/k, k*C)``.

Times the ops that dominate bands 5-6 (elementwise add/multiply, tilize + snake_beta + untilize, the T halo
exchange, a T concat) plus the conv3d cost of the packed dense formulation of a 1-D conv and of the 2x anti-alias
upsampler, on the 4x8 mesh at factor 8 (32 chips, one op stream each). Wall per op after a warm call, 20 reps
between mesh syncs, so small ops read as the dispatch floor.

    pytest models/tt_dit/tests/models/minimax_h3/tools/layout_tax_probe.py -s
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.layers.audio_ops import _t_neighbor_pad, depthwise_tap_filter
from models.tt_dit.parallel.config import ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.conv3d import get_conv3d_config
from models.tt_dit.utils.test import line_params_8k

MESH_4X8 = [
    pytest.param(
        (4, 8),
        {**line_params_8k, "require_exact_physical_num_devices": True, "l1_small_size": 65536},
        id="mesh4x8_8k",
    )
]
T_BAND6, C_BAND6 = 60000, 8  # per-chip rows in band 6 at 600 latents, factor 8
PACK = [1, 4, 8, 16, 32]
REPS = 20


def _time(mesh_device, fn, reps=REPS):
    out = fn()
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    ttnn.synchronize_device(mesh_device)
    return (time.perf_counter() - t0) / reps * 1e6


def _conv3d_us(mesh_device, x_BTC, c_out, kernel, compute):
    B, T, C = x_BTC.shape
    cfg = get_conv3d_config(
        C, c_out, (kernel, 1, 1), ttnn.float32, grid_size=mesh_device.compute_with_storage_grid_size()
    )
    w = ttnn.from_torch(
        torch.randn(kernel * C, c_out) * 0.01,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        device=mesh_device,
    )
    x5 = ttnn.reshape(x_BTC, (B, T, 1, 1, C))

    def run():
        return ttnn.experimental.conv3d(
            input_tensor=x5,
            weight_tensor=w,
            bias_tensor=None,
            config=cfg,
            output_channels=c_out,
            kernel_size=(kernel, 1, 1),
            stride=(1, 1, 1),
            padding=(kernel // 2, 0, 0),
            dilation=(1, 1, 1),
            padding_mode="zeros",
            dtype=ttnn.float32,
            compute_kernel_config=compute,
        )

    return _time(mesh_device, run)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_4X8, indirect=["mesh_device", "device_params"])
def test_layout_tax(mesh_device):
    pc = ParallelFactor(factor=8, mesh_axis=1)
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    compute = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    rows = []
    for k in PACK:
        T, C = T_BAND6 // k, C_BAND6 * k
        x = ttnn.from_torch(torch.randn(1, T, C), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
        a = ttnn.from_torch(
            torch.rand(1, 1, C) + 0.5, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device
        )
        b = ttnn.from_torch(
            torch.rand(1, 1, C) + 0.5, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device
        )
        r = {"k": k, "shape": (T, C), "page_B": C * 4}
        r["add"] = _time(mesh_device, lambda: ttnn.add(x, x))
        r["mul_ch"] = _time(mesh_device, lambda: ttnn.multiply(x, a))
        r["tilize"] = _time(mesh_device, lambda: ttnn.to_layout(x, ttnn.TILE_LAYOUT))
        xt = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        at, bt = ttnn.to_layout(a, ttnn.TILE_LAYOUT), ttnn.to_layout(b, ttnn.TILE_LAYOUT)
        r["snake"] = _time(mesh_device, lambda: ttnn.snake_beta(xt, at, bt))
        r["untilize"] = _time(mesh_device, lambda: ttnn.to_layout(xt, ttnn.ROW_MAJOR_LAYOUT))
        r["halo6"] = _time(
            mesh_device,
            lambda: _t_neighbor_pad(
                x, pad_left=6, pad_right=6, parallel_config=pc, ccl_manager=ccl, padding_mode="replicate"
            ),
        )
        r["concatT"] = _time(mesh_device, lambda: ttnn.concat([x, x], dim=1))
        if k == 1:
            # today's exact depthwise 12-tap filter on the unpacked layout (band 6: C=8)
            cache = {}
            r["dw12"] = _time(
                mesh_device,
                lambda: depthwise_tap_filter(
                    x, [0.05] * 12, 1, mesh_device=mesh_device, dtype=ttnn.float32, cache=cache
                ),
            )
            # the 1-D conv as it runs today: channels padded to 32
            x32 = ttnn.from_torch(
                torch.randn(1, T, 32), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device
            )
            r["conv_k3"] = _conv3d_us(mesh_device, x32, 32, 3, compute)
            r["conv_k11"] = _conv3d_us(mesh_device, x32, 32, 11, compute)
        else:
            # packed dense formulations: a k-packed 1-D conv (K=3 -> 2 taps, K=11 -> ceil(10/k)+1 taps) and the 2x
            # upsampler (out 2*k*C); the extra FLOPs are the price of the wide layout
            r["conv_k3"] = _conv3d_us(mesh_device, x, C, 2, compute)
            r["conv_k11"] = _conv3d_us(mesh_device, x, C, -(-10 // k) + 1, compute)
            r["up2x"] = _conv3d_us(mesh_device, x, 2 * C, 2, compute)
        rows.append(r)
        logger.info(f"LAYOUT_TAX {r}")
    keys = ["add", "mul_ch", "tilize", "snake", "untilize", "halo6", "concatT", "dw12", "conv_k3", "conv_k11", "up2x"]
    logger.info("=== layout tax, us per op (4x8, factor 8, per-chip band-6 tensor 60000x8 fp32 = 1.9 MB) ===")
    logger.info(f"{'k':>3s} {'shape':>14s} {'page':>5s} " + " ".join(f"{k:>9s}" for k in keys))
    for r in rows:
        logger.info(
            f"{r['k']:3d} {str(r['shape']):>14s} {r['page_B']:5d} "
            + " ".join(f"{r[k]:9.0f}" if k in r else f"{'-':>9s}" for k in keys)
        )
