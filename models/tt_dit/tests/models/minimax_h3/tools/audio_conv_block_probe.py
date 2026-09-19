# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""One-chip sweep of conv3d's T_out / C_out blocks for the H3 vocoder's operand-split convs, at the per-device shapes.

Only the two blocks that do not touch arithmetic are swept (the per-output K order is fixed by C_in_block, which stays),
so every candidate must be bit-identical to today's table entry; the probe asserts that and prints the timing table.
    pytest models/tt_dit/tests/models/minimax_h3/tools/audio_conv_block_probe.py -s
"""

import time

import pytest
import torch
from loguru import logger

import ttnn

from .....layers.audio_ops import Conv1dViaConv3d
from .....models.audio_vae.minimax_h3.blockings_minimax_h3_audio import register_h3_audio_blockings
from .....utils.conv3d import aligned_channels

SINGLE_DEVICE = [pytest.param((1, 1), {"l1_small_size": 65536}, id="single_device")]
# (C_in, C_out, k, dilation, T per device): the AMP convs of bands 0-4 (T-shard 8, 600 latents), batch 2.
SHAPES = [
    (512, 512, 3, 1, 375),
    (512, 512, 11, 1, 375),
    (256, 256, 7, 1, 1875),
    (256, 256, 11, 1, 1875),
    (256, 256, 3, 3, 1875),
    (128, 128, 7, 1, 3750),
    (64, 64, 7, 1, 7500),
    (32, 32, 7, 1, 15000),
    (32, 32, 11, 5, 15000),
]
T_OUT_CANDIDATES = (8, 16, 32)
C_OUT_CANDIDATES = (32, 64, 128, 256, 512)


def _timed(mesh_device, fn, n=10):
    out = fn()
    ttnn.synchronize_device(mesh_device)
    best = float("inf")
    for _ in range(n):
        ttnn.synchronize_device(mesh_device)
        mark = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(mesh_device)
        best = min(best, time.perf_counter() - mark)
    return out, best * 1e3


def _config_like(base, *, t_out, c_out, grid):
    return ttnn.Conv3dConfig(
        weights_dtype=base.weights_dtype,
        output_layout=base.output_layout,
        T_out_block=t_out,
        W_out_block=base.W_out_block,
        H_out_block=base.H_out_block,
        C_out_block=c_out,
        C_in_block=base.C_in_block,
        compute_with_storage_grid_size=grid,
        operand_split=base.operand_split,
    )


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_audio_conv_block_sweep(mesh_device):
    register_h3_audio_blockings()
    grid = mesh_device.compute_with_storage_grid_size()
    torch.manual_seed(0)
    summary = []
    for c_in, c_out, k, dil, t_local in SHAPES:
        conv = Conv1dViaConv3d(
            c_in, c_out, kernel_size=k, dilation=dil, mesh_device=mesh_device, dtype=ttnn.float32, split_mode="kernel"
        )
        conv.load_torch_state_dict(
            {"weight": torch.randn(c_out, c_in, k) * (1.0 / (c_in * k) ** 0.5), "bias": torch.randn(c_out) * 0.1}
        )
        x = ttnn.from_torch(torch.randn(2, t_local, c_in), device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32)
        base = conv.conv_config
        today = (base.C_in_block, base.C_out_block, base.T_out_block)
        ref, t_today = _timed(mesh_device, lambda: conv(x))
        ref_t = ttnn.to_torch(ref)
        rows = [(today, t_today, True)]
        padded_out = aligned_channels(max(32, c_out))
        for t_out in T_OUT_CANDIDATES:
            for co in C_OUT_CANDIDATES:
                if co > padded_out or padded_out % co or (co, t_out) == (base.C_out_block, base.T_out_block):
                    continue
                conv.conv_config = _config_like(base, t_out=t_out, c_out=co, grid=grid)
                try:
                    out, ms = _timed(mesh_device, lambda: conv(x))
                except Exception as exc:  # noqa: BLE001
                    logger.info(f"  ({c_in},{c_out},k{k},d{dil}) T{t_out} Cout{co}: FAILED {type(exc).__name__}: {str(exc)[:100]}")
                    continue
                equal = bool(torch.equal(ttnn.to_torch(out), ref_t))
                rows.append(((base.C_in_block, co, t_out), ms, equal))
                logger.info(f"  ({c_in},{c_out},k{k},d{dil}) Cin{base.C_in_block} Cout{co} T{t_out}: {ms:.3f} ms  bit-identical {equal}")
        conv.conv_config = base
        # Split-cost decomposition at today's blocking: off = one bf16-operand pass, weight = two passes (host-split W,
        # no activation split), kernel = three passes + the in-kernel SFPU activation split. The SFPU share is roughly
        # (kernel - weight) - (weight - off); it bounds what a producer-side hi/lo (no per-tap split) could save.
        split_ms = {}
        for mode in ("off", "weight", "kernel"):
            conv.split_mode = mode
            try:
                _, split_ms[mode] = _timed(mesh_device, lambda: conv(x))
            except Exception as exc:  # noqa: BLE001
                logger.info(f"  split_mode={mode}: FAILED {type(exc).__name__}: {str(exc)[:100]}")
        conv.split_mode = "kernel"
        if len(split_ms) == 3:
            sfpu = (split_ms["kernel"] - split_ms["weight"]) - (split_ms["weight"] - split_ms["off"])
            logger.info(
                f"SPLITCOST ({c_in},{c_out},k{k},d{dil}) T{t_local}: off {split_ms['off']:.3f} ms, weight {split_ms['weight']:.3f} ms, "
                f"kernel {split_ms['kernel']:.3f} ms -> per extra pass ~{split_ms['weight'] - split_ms['off']:.3f} ms, "
                f"SFPU split ~{sfpu:.3f} ms ({100 * sfpu / max(split_ms['kernel'], 1e-9):.0f} % of the kernel-split conv)"
            )
        rows.sort(key=lambda r: r[1])
        best = next((r for r in rows if r[2]), rows[0])
        summary.append((c_in, c_out, k, dil, t_local, today, t_today, best))
        logger.info(
            f"CONVBLK ({c_in},{c_out},k{k},d{dil}) T{t_local}: today Cin{today[0]} Cout{today[1]} T{today[2]} {t_today:.3f} ms; "
            f"best Cin{best[0][0]} Cout{best[0][1]} T{best[0][2]} {best[1]:.3f} ms ({t_today / max(best[1], 1e-9):.2f}x) identical {best[2]}"
        )
        ttnn.deallocate(x)
    logger.info("=== conv block sweep summary (one chip, operand split on, min of 10) ===")
    for c_in, c_out, k, dil, t_local, today, t_today, best in summary:
        logger.info(
            f"  ({c_in:>3},{c_out:>3},k{k:>2},d{dil}) T{t_local:>5}: today {today} {t_today:7.3f} ms -> best {best[0]} {best[1]:7.3f} ms "
            f"({t_today / max(best[1], 1e-9):.2f}x)"
        )
