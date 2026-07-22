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

# BT.601 coefficients for input ∈ [-1, 1] → limited-range uint8.
# These must match yuv_conversion.hpp::yuv_coefficients(BT601, MinusOneToOne, Limited).
_Y_COEFF = (32.74, 64.28, 12.48, 125.5)
_CB_COEFF = (-18.90, -37.10, 56.00, 128.0)
_CR_COEFF = (56.00, -46.89, -9.11, 128.0)

# (Kr, Kb) for the standard colorspaces (Kg = 1 - Kr - Kb).
_KR_KB = {"BT601": (0.299, 0.114), "BT709": (0.2126, 0.0722), "BT2020": (0.2627, 0.0593)}


def _derive_coeffs(color_space: str, input_range: str, output_range: str):
    """Python mirror of the C++ yuv_coefficients() derivation (independent oracle)."""
    kr, kb = _KR_KB[color_space]
    kg = 1.0 - kr - kb
    lo, hi = (-1.0, 1.0) if input_range == "MinusOneToOne" else (0.0, 1.0)
    s = 1.0 / (hi - lo)
    yscale, yoff, cscale = (219.0, 16.0, 224.0) if output_range == "Limited" else (255.0, 0.0, 255.0)
    off_y = yoff - yscale * s * lo
    cbk = cscale * s / (2.0 * (1.0 - kb))
    crk = cscale * s / (2.0 * (1.0 - kr))
    y = (yscale * s * kr, yscale * s * kg, yscale * s * kb, off_y)
    cb = (-cbk * kr, -cbk * kg, cbk * (1.0 - kb), 128.0)
    cr = (crk * (1.0 - kr), -crk * kg, -crk * kb, 128.0)
    return y, cb, cr


def _yuv_reference(rgb: torch.Tensor, y_coeff, cb_coeff, cr_coeff):
    """Pure-PyTorch reference: CHWT bf16 RGB → (Y, Cb, Cr) uint8 for given coeffs.

    Spatial subsampling for Cb/Cr: mean of each non-overlapping 2×2 block
    in (H, W) before quantisation to uint8.
    """
    R = rgb[0].float()  # (H, W, T)
    G = rgb[1].float()
    B = rgb[2].float()
    H, W, T_ = R.shape

    def linear(coeff):
        w_r, w_g, w_b, off = coeff
        return w_r * R + w_g * G + w_b * B + off  # (H, W, T) float

    Y = (linear(y_coeff) + 0.5).clamp(0, 255).to(torch.uint8)

    # Cb/Cr are 4:2:0 subsampled: average each non-overlapping 2×2 block in
    # (H, W) in the float domain, then quantise once.  (RGB→chroma is affine, so
    # averaging RGB then converting equals converting then averaging.)
    def subsample(coeff):
        val = linear(coeff).view(H // 2, 2, W // 2, 2, T_).mean(dim=(1, 3))  # (H/2, W/2, T)
        return (val + 0.5).clamp(0, 255).to(torch.uint8)

    return Y, subsample(cb_coeff), subsample(cr_coeff)


def _host_yuv_reference(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BT.601 limited-range reference for input ∈ [-1, 1]."""
    return _yuv_reference(rgb, _Y_COEFF, _CB_COEFF, _CR_COEFF)


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
        tt_Y, tt_Cb, tt_Cr = ttnn.experimental.yuv_conversion(tt_in, coefficients=coefficients)
        ttnn.synchronize_device(device)

        dev_Y = ttnn.to_torch(tt_Y)
        dev_Cb = ttnn.to_torch(tt_Cb)
        dev_Cr = ttnn.to_torch(tt_Cr)

        return dev_Y, dev_Cb, dev_Cr, ref_Y, ref_Cb, ref_Cr

    def test_output_shapes(self, device, H, W, T):
        dev_Y, dev_Cb, dev_Cr, *_ = self._run(H, W, T, device)
        assert dev_Y.shape == torch.Size([1, H, W, T]), f"Y shape wrong: {dev_Y.shape}"
        assert dev_Cb.shape == torch.Size([1, H // 2, W // 2, T]), f"Cb shape wrong: {dev_Cb.shape}"
        assert dev_Cr.shape == torch.Size([1, H // 2, W // 2, T]), f"Cr shape wrong: {dev_Cr.shape}"

    def test_correctness_Y(self, device, H, W, T):
        """Y plane: max abs error ≤ 1 (rounding differences)."""
        dev_Y, _, _, ref_Y, _, _ = self._run(H, W, T, device)
        diff = (dev_Y.squeeze(0).int() - ref_Y.int()).abs()
        assert diff.max().item() <= 1, f"Y max error {diff.max().item()} > 1"
        if diff.numel() > 100:
            assert diff.float().mean().item() < 0.5, f"Y mean error too high: {diff.float().mean().item()}"

    def test_correctness_UV(self, device, H, W, T):
        """Cb/Cr planes: max abs error ≤ 2 (bf16 precision over 12 multiply-accumulates)."""
        _, dev_Cb, dev_Cr, _, ref_Cb, ref_Cr = self._run(H, W, T, device)

        diff_cb = (dev_Cb.squeeze(0).int() - ref_Cb.int()).abs()
        diff_cr = (dev_Cr.squeeze(0).int() - ref_Cr.int()).abs()

        assert diff_cb.max().item() <= 2, f"Cb max error {diff_cb.max().item()} > 2"
        assert diff_cr.max().item() <= 2, f"Cr max error {diff_cr.max().item()} > 2"
        if diff_cb.numel() > 100:
            assert diff_cb.float().mean().item() < 0.5, f"Cb mean error too high: {diff_cb.float().mean().item()}"
            assert diff_cr.float().mean().item() < 0.5, f"Cr mean error too high: {diff_cr.float().mean().item()}"

    def test_program_cache_reuse(self, device, H, W, T):
        """Second (cache-hit) run with a different input at a DIFFERENT address,
        to verify override_runtime_arguments re-points the kernels to the new
        buffers.

        The first input and a dummy are kept allocated across the second
        invocation so the allocator cannot hand back the first input's address;
        otherwise a stale-address bug in override_runtime_arguments would go
        undetected (the second input would coincidentally reuse the same slot).
        """
        coefficients = ttnn.experimental.YUVCoefficients(y=list(_Y_COEFF), cb=list(_CB_COEFF), cr=list(_CR_COEFF))

        def run(seed):
            gen = torch.Generator().manual_seed(seed)
            cpu = torch.rand(3, H, W, T, generator=gen, dtype=torch.bfloat16) * 2.0 - 1.0
            tt_in = ttnn.from_torch(cpu, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            tt_Y, tt_Cb, tt_Cr = ttnn.experimental.yuv_conversion(tt_in, coefficients=coefficients)
            ttnn.synchronize_device(device)
            outs = (ttnn.to_torch(tt_Y), ttnn.to_torch(tt_Cb), ttnn.to_torch(tt_Cr))
            return cpu, tt_in, outs

        cpu1, in1, out1 = run(42)  # keep in1 alive across the second run
        # Occupy space so the second input can't be handed the first's address.
        dummy = ttnn.from_torch(
            torch.zeros(3, H, W, T, dtype=torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        cpu2, in2, out2 = run(99)

        assert in2.buffer_address() != in1.buffer_address(), (
            "second input reused the first input's DRAM address; this run would not "
            "exercise override_runtime_arguments"
        )

        for name, (Y, Cb, Cr), cpu in [("iter1", out1, cpu1), ("iter2", out2, cpu2)]:
            ref_Y, ref_Cb, ref_Cr = _host_yuv_reference(cpu)
            assert (Y.squeeze(0).int() - ref_Y.int()).abs().max().item() <= 1, f"{name} Y mismatch"
            assert (Cb.squeeze(0).int() - ref_Cb.int()).abs().max().item() <= 2, f"{name} Cb mismatch"
            assert (Cr.squeeze(0).int() - ref_Cr.int()).abs().max().item() <= 2, f"{name} Cr mismatch"

        del dummy, in1, in2

    def test_extreme_values(self, device, H, W, T):
        """All-ones and all-negative-ones inputs: verify clamp prevents overflow."""
        for fill in [1.0, -1.0]:
            cpu = torch.full((3, H, W, T), fill, dtype=torch.bfloat16)
            tt_in = ttnn.from_torch(cpu, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            coefficients = ttnn.experimental.YUVCoefficients(y=list(_Y_COEFF), cb=list(_CB_COEFF), cr=list(_CR_COEFF))
            tt_Y, tt_Cb, tt_Cr = ttnn.experimental.yuv_conversion(tt_in, coefficients=coefficients)
            ttnn.synchronize_device(device)

            for name, t in [("Y", tt_Y), ("Cb", tt_Cb), ("Cr", tt_Cr)]:
                host = ttnn.to_torch(t)
                assert host.dtype == torch.uint8, f"{name} dtype wrong: {host.dtype}"
                # All values must be valid uint8 (clamp ensures no wrap-around)
                assert host.min().item() >= 0
                assert host.max().item() <= 255


class TestYUVValidation:
    """Op validation: unsupported configs must be rejected, not silently run."""

    def test_sharded_output_rejected(self, device, expect_error):
        H, W, T = 4, 4, 32
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
            ttnn.experimental.yuv_conversion(tt_in, coefficients=coefficients, memory_config=sharded_cfg)


class TestYUVColorSpaceAPI:
    """The colorspace/range front-end derives the coefficients internally."""

    def test_bt601_limited_matches_reference(self, device):
        # Default selectors (BT601, [-1,1], Limited) must reproduce the BT601
        # limited-range coefficients the host reference uses.
        H, W, T = 64, 64, 64
        gen = torch.Generator().manual_seed(7)
        cpu = torch.rand(3, H, W, T, generator=gen, dtype=torch.bfloat16) * 2.0 - 1.0
        ref_Y, ref_Cb, ref_Cr = _host_yuv_reference(cpu)

        tt_in = ttnn.from_torch(cpu, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        y, cb, cr = ttnn.experimental.yuv_conversion(
            tt_in,
            color_space=ttnn.experimental.YUVColorSpace.BT601,
            input_range=ttnn.experimental.RGBRange.MinusOneToOne,
            output_range=ttnn.experimental.YUVRange.Limited,
        )
        ttnn.synchronize_device(device)
        assert (ttnn.to_torch(y).squeeze(0).int() - ref_Y.int()).abs().max().item() <= 1, "Y != BT601 reference"
        assert (ttnn.to_torch(cb).squeeze(0).int() - ref_Cb.int()).abs().max().item() <= 2, "Cb != BT601 reference"
        assert (ttnn.to_torch(cr).squeeze(0).int() - ref_Cr.int()).abs().max().item() <= 2, "Cr != BT601 reference"

    def test_colorspaces_run_and_differ(self, device):
        # Each colorspace produces valid uint8; BT601 and BT709 luma differ.
        H, W, T = 64, 64, 64
        gen = torch.Generator().manual_seed(11)
        cpu = torch.rand(3, H, W, T, generator=gen, dtype=torch.bfloat16) * 2.0 - 1.0
        tt_in = ttnn.from_torch(cpu, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        luma = {}
        for cs in (
            ttnn.experimental.YUVColorSpace.BT601,
            ttnn.experimental.YUVColorSpace.BT709,
            ttnn.experimental.YUVColorSpace.BT2020,
        ):
            y, _, _ = ttnn.experimental.yuv_conversion(tt_in, color_space=cs)
            ttnn.synchronize_device(device)
            hy = ttnn.to_torch(y)
            assert hy.dtype == torch.uint8 and hy.min().item() >= 0 and hy.max().item() <= 255
            luma[cs] = hy

        assert not torch.equal(
            luma[ttnn.experimental.YUVColorSpace.BT601], luma[ttnn.experimental.YUVColorSpace.BT709]
        ), "BT601 and BT709 luma should differ"

    @pytest.mark.parametrize("input_range", ["MinusOneToOne", "ZeroToOne"])
    def test_input_range(self, device, input_range):
        # Both input normalizations must convert correctly against an
        # independently-derived reference for that range.
        #
        # ZeroToOne is ~1 LSB less precise on Y: its derived weights are ~2x
        # larger than MinusOneToOne's (e.g. wG 128.5 vs 64.3) and so pack to a
        # coarser bf16 grid (step 1.0 near 128 vs 0.5 near 64). Y tolerance is
        # relaxed to 2 for that case accordingly.
        H, W, T = 64, 64, 64
        gen = torch.Generator().manual_seed(13)
        rgb01 = torch.rand(3, H, W, T, generator=gen, dtype=torch.bfloat16)  # [0, 1]
        rgb = rgb01 if input_range == "ZeroToOne" else (rgb01.float() * 2.0 - 1.0).to(torch.bfloat16)

        y_c, cb_c, cr_c = _derive_coeffs("BT601", input_range, "Limited")
        ref_Y, ref_Cb, ref_Cr = _yuv_reference(rgb, y_c, cb_c, cr_c)

        tt_in = ttnn.from_torch(rgb, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        y, cb, cr = ttnn.experimental.yuv_conversion(
            tt_in,
            color_space=ttnn.experimental.YUVColorSpace.BT601,
            input_range=getattr(ttnn.experimental.RGBRange, input_range),
            output_range=ttnn.experimental.YUVRange.Limited,
        )
        ttnn.synchronize_device(device)
        y_tol = 1 if input_range == "MinusOneToOne" else 2
        assert (ttnn.to_torch(y).squeeze(0).int() - ref_Y.int()).abs().max().item() <= y_tol, f"{input_range} Y"
        assert (ttnn.to_torch(cb).squeeze(0).int() - ref_Cb.int()).abs().max().item() <= 2, f"{input_range} Cb"
        assert (ttnn.to_torch(cr).squeeze(0).int() - ref_Cr.int()).abs().max().item() <= 2, f"{input_range} Cr"


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
    def test_sweep_correctness(self, device, H, W, T):
        gen = torch.Generator().manual_seed(H * 10000 + W * 100 + T)
        cpu_bf16 = torch.rand(3, H, W, T, generator=gen, dtype=torch.bfloat16) * 2.0 - 1.0

        ref_Y, ref_Cb, ref_Cr = _host_yuv_reference(cpu_bf16)

        tt_in = ttnn.from_torch(cpu_bf16, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        coefficients = ttnn.experimental.YUVCoefficients(y=list(_Y_COEFF), cb=list(_CB_COEFF), cr=list(_CR_COEFF))
        tt_Y, tt_Cb, tt_Cr = ttnn.experimental.yuv_conversion(tt_in, coefficients=coefficients)
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


class TestYUVPerformance:
    """Measure op latency on the 720p shard size."""

    def test_performance_720p(self, device):
        import time

        H, W, T = 180, 160, 81
        n_iters = 20

        gen = torch.Generator().manual_seed(0)
        cpu = torch.rand(3, H, W, T, generator=gen, dtype=torch.bfloat16) * 2.0 - 1.0

        tt_in = ttnn.from_torch(cpu, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        coefficients = ttnn.experimental.YUVCoefficients(y=list(_Y_COEFF), cb=list(_CB_COEFF), cr=list(_CR_COEFF))

        # Warmup
        for _ in range(3):
            ttnn.experimental.yuv_conversion(tt_in, coefficients=coefficients)
        ttnn.synchronize_device(device)

        start = time.perf_counter()
        for _ in range(n_iters):
            ttnn.experimental.yuv_conversion(tt_in, coefficients=coefficients)
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
