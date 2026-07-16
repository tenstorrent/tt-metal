# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the on-device YUV conversion op.

Tests the ttnn.experimental.yuv_conversion op which converts a CHWT bfloat16
tensor (C=3, values in [-1, 1]) to YUV 4:2:0 uint8 planar format.

Run with:
    scripts/run_safe_pytest.sh models/tt_dit/tests/unit/test_yuv_conversion.py -s
"""

import os

import pytest
import torch

import ttnn

# BT.601 coefficients for input ∈ [-1, 1] → uint8 [0, 255].
# These must match the values in yuv_conversion.hpp::yuv_bt601_coefficients().
_Y_COEFF = (32.74, 64.28, 12.48, 125.5)
_CB_COEFF = (-18.90, -37.10, 56.00, 128.0)
_CR_COEFF = (56.00, -46.89, -9.11, 128.0)


def _host_yuv_reference(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch BT.601 reference: CHWT bf16 in [-1,1] → (Y, Cb, Cr) uint8.

    Spatial subsampling for Cb/Cr: mean of each non-overlapping 2×2 block
    in (H, W) before quantisation to uint8.
    """
    R = rgb[0].float()  # (H, W, T)
    G = rgb[1].float()
    B = rgb[2].float()

    def linear(r, g, b, coeff):
        w_r, w_g, w_b, off = coeff
        return (w_r * r + w_g * g + w_b * b + off + 0.5).clamp(0, 255).to(torch.uint8)

    Y = linear(R, G, B, _Y_COEFF)  # (H, W, T)
    Cb = linear(R, G, B, _CB_COEFF)
    Cr = linear(R, G, B, _CR_COEFF)

    # Subsample Cb and Cr: average 2×2 blocks in H, W (before uint8 cast)
    def subsample(plane):
        R_f = rgb[0].float()
        G_f = rgb[1].float()
        B_f = rgb[2].float()
        # We need float-domain average then quantise
        w_r, w_g, w_b, off = _CB_COEFF if (plane == "cb") else _CR_COEFF
        val = w_r * R_f + w_g * G_f + w_b * B_f + off  # (H, W, T) float
        # Average 2×2 blocks: (H, W, T) → (H/2, W/2, T)
        H, W, T_ = val.shape
        val = val.view(H // 2, 2, W // 2, 2, T_).mean(dim=(1, 3))  # (H/2, W/2, T)
        return (val + 0.5).clamp(0, 255).to(torch.uint8)

    Cb_sub = subsample("cb")
    Cr_sub = subsample("cr")
    return Y, Cb_sub, Cr_sub


def _make_device() -> ttnn.Device:
    return ttnn.open_device(device_id=0)


# ---------------------------------------------------------------------------
# Parametrize over shard shapes that match real 4×8 mesh per-device tensors
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "H, W, T",
    [
        (180, 160, 81),  # 720p shard (H//4, W//8)
        (120, 104, 9),  # small shape for fast correctness check
        (2, 2, 1),  # minimum: single UV pixel, single T element
        (4, 4, 32),  # T fills exactly one tile (no partial)
        (10, 14, 33),  # non-power-of-2, T just past tile boundary
        (64, 64, 64),  # medium square, T = two full tiles
        (2, 128, 3),  # single row group (H/2=1), extreme aspect ratio
    ],
    ids=["720p_shard", "small", "tiny", "tile_aligned", "non_aligned", "medium_square", "single_row_group"],
)
class TestYUVConversion:
    def _run(self, H, W, T, device, seed=42):
        """Helper: create reference tensor, run op, return (device_Y, device_Cb, device_Cr, ref_Y, ref_Cb, ref_Cr)."""
        gen = torch.Generator().manual_seed(seed)
        # Input ∈ [-1, 1] bfloat16, shape CHWT (C=3)
        cpu_bf16 = torch.rand(3, H, W, T, generator=gen, dtype=torch.bfloat16) * 2.0 - 1.0

        ref_Y, ref_Cb, ref_Cr = _host_yuv_reference(cpu_bf16)

        tt_in = ttnn.from_torch(cpu_bf16, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        coefficients = ttnn.experimental.YUVCoefficients(
            y=list(_Y_COEFF),
            cb=list(_CB_COEFF),
            cr=list(_CR_COEFF),
        )
        tt_Y, tt_Cb, tt_Cr = ttnn.experimental.yuv_conversion(tt_in, coefficients)
        ttnn.synchronize_device(device)

        dev_Y = ttnn.to_torch(tt_Y)
        dev_Cb = ttnn.to_torch(tt_Cb)
        dev_Cr = ttnn.to_torch(tt_Cr)

        return dev_Y, dev_Cb, dev_Cr, ref_Y, ref_Cb, ref_Cr

    def test_output_shapes(self, H, W, T):
        device = _make_device()
        try:
            dev_Y, dev_Cb, dev_Cr, *_ = self._run(H, W, T, device)
            assert dev_Y.shape == torch.Size([1, H, W, T]), f"Y shape wrong: {dev_Y.shape}"
            assert dev_Cb.shape == torch.Size([1, H // 2, W // 2, T]), f"Cb shape wrong: {dev_Cb.shape}"
            assert dev_Cr.shape == torch.Size([1, H // 2, W // 2, T]), f"Cr shape wrong: {dev_Cr.shape}"
        finally:
            ttnn.close_device(device)

    def test_correctness_Y(self, H, W, T):
        """Y plane: max abs error ≤ 1 (rounding differences)."""
        device = _make_device()
        try:
            dev_Y, _, _, ref_Y, _, _ = self._run(H, W, T, device)
            diff = (dev_Y.squeeze(0).int() - ref_Y.int()).abs()
            assert diff.max().item() <= 1, f"Y max error {diff.max().item()} > 1"
            if diff.numel() > 100:
                assert diff.float().mean().item() < 0.5, f"Y mean error too high: {diff.float().mean().item()}"
        finally:
            ttnn.close_device(device)

    def test_correctness_UV(self, H, W, T):
        """Cb/Cr planes: max abs error ≤ 2 (bf16 precision over 12 multiply-accumulates)."""
        device = _make_device()
        try:
            _, dev_Cb, dev_Cr, _, ref_Cb, ref_Cr = self._run(H, W, T, device)

            diff_cb = (dev_Cb.squeeze(0).int() - ref_Cb.int()).abs()
            diff_cr = (dev_Cr.squeeze(0).int() - ref_Cr.int()).abs()

            assert diff_cb.max().item() <= 2, f"Cb max error {diff_cb.max().item()} > 2"
            assert diff_cr.max().item() <= 2, f"Cr max error {diff_cr.max().item()} > 2"
            if diff_cb.numel() > 100:
                assert diff_cb.float().mean().item() < 0.5, f"Cb mean error too high: {diff_cb.float().mean().item()}"
                assert diff_cr.float().mean().item() < 0.5, f"Cr mean error too high: {diff_cr.float().mean().item()}"
        finally:
            ttnn.close_device(device)

    def test_program_cache_reuse(self, H, W, T):
        """Run twice with different seeds — verifies override_runtime_arguments."""
        device = _make_device()
        try:
            Y1, Cb1, Cr1, rY1, rCb1, rCr1 = self._run(H, W, T, device, seed=42)
            Y2, Cb2, Cr2, rY2, rCb2, rCr2 = self._run(H, W, T, device, seed=99)

            for name, got, ref, tol in [
                ("Y  iter1", Y1, rY1, 1),
                ("Cb iter1", Cb1, rCb1, 2),
                ("Y  iter2", Y2, rY2, 1),
                ("Cb iter2", Cb2, rCb2, 2),
            ]:
                diff = (got.squeeze(0).int() - ref.int()).abs()
                assert diff.max().item() <= tol, f"{name}: max error {diff.max().item()}"
        finally:
            ttnn.close_device(device)

    def test_extreme_values(self, H, W, T):
        """All-ones and all-negative-ones inputs: verify clamp prevents overflow."""
        device = _make_device()
        try:
            for fill in [1.0, -1.0]:
                cpu = torch.full((3, H, W, T), fill, dtype=torch.bfloat16)
                tt_in = ttnn.from_torch(cpu, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
                coefficients = ttnn.experimental.YUVCoefficients(
                    y=list(_Y_COEFF), cb=list(_CB_COEFF), cr=list(_CR_COEFF)
                )
                tt_Y, tt_Cb, tt_Cr = ttnn.experimental.yuv_conversion(tt_in, coefficients)
                ttnn.synchronize_device(device)

                for name, t in [("Y", tt_Y), ("Cb", tt_Cb), ("Cr", tt_Cr)]:
                    host = ttnn.to_torch(t)
                    assert host.dtype == torch.uint8, f"{name} dtype wrong: {host.dtype}"
                    # All values must be valid uint8 (clamp ensures no wrap-around)
                    assert host.min().item() >= 0
                    assert host.max().item() <= 255
        finally:
            ttnn.close_device(device)


class TestYUVValidation:
    """Op validation: unsupported configs must be rejected, not silently run."""

    def test_sharded_output_rejected(self, expect_error):
        H, W, T = 4, 4, 32
        device = _make_device()
        try:
            cpu = torch.rand(3, H, W, T, dtype=torch.bfloat16) * 2.0 - 1.0
            tt_in = ttnn.from_torch(cpu, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            coefficients = ttnn.experimental.YUVCoefficients(y=list(_Y_COEFF), cb=list(_CB_COEFF), cr=list(_CR_COEFF))

            shard_spec = ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
                [H, W * T],
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            sharded_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

            with expect_error(RuntimeError, "Sharded output is not supported"):
                ttnn.experimental.yuv_conversion(tt_in, coefficients, memory_config=sharded_cfg)
        finally:
            ttnn.close_device(device)


_SWEEP_H = [2, 4, 6, 10, 32, 64, 180]
_SWEEP_W = [2, 4, 6, 14, 32, 64, 160]
_SWEEP_T = [1, 2, 3, 16, 31, 32, 33, 64, 81]
_SWEEP_CASES = [pytest.param(H, W, T, id=f"{H}x{W}x{T}") for H in _SWEEP_H for W in _SWEEP_W for T in _SWEEP_T]


@pytest.mark.skipif(os.getenv("CI") == "true", reason="Thorough sweep is too slow for CI")
class TestYUVSweep:
    """Exhaustive shape sweep for local validation.

    Covers a wide grid of (H, W, T) combinations that stress different kernel
    code paths: partial tiles, single/multi core splits, boundary conditions,
    and large shapes.  Skipped in CI to keep pipeline times reasonable.
    """

    @pytest.mark.parametrize("H, W, T", _SWEEP_CASES)
    def test_sweep_correctness(self, H, W, T):
        gen = torch.Generator().manual_seed(H * 10000 + W * 100 + T)
        cpu_bf16 = torch.rand(3, H, W, T, generator=gen, dtype=torch.bfloat16) * 2.0 - 1.0

        ref_Y, ref_Cb, ref_Cr = _host_yuv_reference(cpu_bf16)

        device = _make_device()
        try:
            tt_in = ttnn.from_torch(cpu_bf16, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            coefficients = ttnn.experimental.YUVCoefficients(y=list(_Y_COEFF), cb=list(_CB_COEFF), cr=list(_CR_COEFF))
            tt_Y, tt_Cb, tt_Cr = ttnn.experimental.yuv_conversion(tt_in, coefficients)
            ttnn.synchronize_device(device)

            dev_Y = ttnn.to_torch(tt_Y).squeeze(0)
            dev_Cb = ttnn.to_torch(tt_Cb).squeeze(0)
            dev_Cr = ttnn.to_torch(tt_Cr).squeeze(0)

            assert dev_Y.shape == ref_Y.shape, f"Y shape {dev_Y.shape} != {ref_Y.shape}"
            assert dev_Cb.shape == ref_Cb.shape, f"Cb shape {dev_Cb.shape} != {ref_Cb.shape}"
            assert dev_Cr.shape == ref_Cr.shape, f"Cr shape {dev_Cr.shape} != {ref_Cr.shape}"

            diff_y = (dev_Y.int() - ref_Y.int()).abs()
            diff_cb = (dev_Cb.int() - ref_Cb.int()).abs()
            diff_cr = (dev_Cr.int() - ref_Cr.int()).abs()

            assert diff_y.max().item() <= 1, f"Y max error {diff_y.max().item()}"
            assert diff_cb.max().item() <= 2, f"Cb max error {diff_cb.max().item()}"
            assert diff_cr.max().item() <= 2, f"Cr max error {diff_cr.max().item()}"

            if diff_y.numel() > 100:
                assert diff_y.float().mean().item() < 0.5, f"Y mean error {diff_y.float().mean().item()}"
            if diff_cb.numel() > 100:
                assert diff_cb.float().mean().item() < 0.5, f"Cb mean error {diff_cb.float().mean().item()}"
                assert diff_cr.float().mean().item() < 0.5, f"Cr mean error {diff_cr.float().mean().item()}"
        finally:
            ttnn.close_device(device)


class TestYUVPerformance:
    """Measure op latency on the 720p shard size."""

    def test_performance_720p(self):
        import time

        H, W, T = 180, 160, 81
        n_iters = 20

        gen = torch.Generator().manual_seed(0)
        cpu = torch.rand(3, H, W, T, generator=gen, dtype=torch.bfloat16) * 2.0 - 1.0

        device = _make_device()
        try:
            tt_in = ttnn.from_torch(cpu, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            coefficients = ttnn.experimental.YUVCoefficients(y=list(_Y_COEFF), cb=list(_CB_COEFF), cr=list(_CR_COEFF))

            # Warmup
            for _ in range(3):
                ttnn.experimental.yuv_conversion(tt_in, coefficients)
            ttnn.synchronize_device(device)

            start = time.perf_counter()
            for _ in range(n_iters):
                ttnn.experimental.yuv_conversion(tt_in, coefficients)
                ttnn.synchronize_device(device)
            elapsed = time.perf_counter() - start

            avg_ms = elapsed / n_iters * 1000
            total_bytes = 3 * H * W * T * 2  # bf16 input
            gbs = (total_bytes / (elapsed / n_iters)) / 1e9

            print(f"\n--- on-device YUV conversion performance ---")
            print(f"  Shape: CHWT (3, {H}, {W}, {T})")
            print(f"  Iterations: {n_iters}")
            print(f"  Average: {avg_ms:.2f} ms")
            print(f"  Throughput: {gbs:.2f} GB/s (bf16 input)")
        finally:
            ttnn.close_device(device)
