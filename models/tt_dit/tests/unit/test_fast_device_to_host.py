# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Tests for fast_device_to_host with Wan 2.2 VAE output shapes.

Parametrized resolutions:
  720p: BCTHW = (1, 3, 81, 720, 1280)
  480p: BCTHW = (1, 3, 81, 480, 832)

Sharding: H (dim 3) on TP axis (mesh axis 0), W (dim 4) on SP axis (mesh axis 1).

Run with:
    pytest models/tt_dit/tests/unit/test_fast_device_to_host.py -k "bh_4x32" --timeout=300
"""

import math
import time

import numpy as np
import pytest
import torch

import ttnn

from ...parallel.manager import CCLManager
from ...utils.tensor import (
    _reassemble_2d,
    _yuv_planar_d2h,
    fast_device_to_host,
    fast_device_to_host_yuv,
    float_to_uint8,
    typed_tensor_2dshard,
)
from ...utils.test import line_params, ring_params

# Wan 2.2 VAE output — BCTHW (H and W are parametrized per-test)
B, C, T = 1, 3, 81
TP_AXIS = 0  # mesh axis for height
SP_AXIS = 1  # mesh axis for width
H_DIM = 3  # BCTHW dimension for height
W_DIM = 4  # BCTHW dimension for width


def _make_reference_tensor(H: int, W: int) -> torch.Tensor:
    """Create a deterministic reference tensor with the Wan VAE output shape."""
    gen = torch.Generator().manual_seed(42)
    return torch.randn(B, C, T, H, W, generator=gen, dtype=torch.bfloat16)


def _shard_to_device(ref: torch.Tensor, mesh_device: ttnn.MeshDevice) -> ttnn.Tensor:
    """Shard a BCTHW tensor: H on TP axis (0), W on SP axis (1)."""
    return typed_tensor_2dshard(
        ref,
        mesh_device,
        shard_mapping={TP_AXIS: H_DIM, SP_AXIS: W_DIM},
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
    )


def _make_concat_dims() -> list[int | None]:
    dims: list[int | None] = [None, None]
    dims[TP_AXIS] = H_DIM
    dims[SP_AXIS] = W_DIM
    return dims


def _make_ccl_manager(mesh_device: ttnn.MeshDevice, num_links: int, topology: ttnn.Topology) -> CCLManager | None:
    return CCLManager(mesh_device, num_links=num_links, topology=topology)


# ---------------------------------------------------------------------------
# YUV conversion helpers (BT.601, [-1, 1] bf16 -> uint8)
# ---------------------------------------------------------------------------

# These must match yuv_conversion.hpp::yuv_bt601_coefficients() and the values
# in test_yuv_conversion.py.
_YUV_Y_COEFF = (32.74, 64.28, 12.48, 125.5)
_YUV_CB_COEFF = (-18.90, -37.10, 56.00, 128.0)
_YUV_CR_COEFF = (56.00, -46.89, -9.11, 128.0)


def _host_yuv_reference(rgb_chwt: torch.Tensor):
    """Pure-PyTorch BT.601 reference: CHWT bf16 in [-1,1] -> (Y, Cb, Cr) uint8.

    Y is full-res (H, W, T); Cb/Cr are 4:2:0 subsampled to (H/2, W/2, T) via
    the mean of each non-overlapping 2x2 block in (H, W) before quantise.
    """
    R = rgb_chwt[0].float()
    G = rgb_chwt[1].float()
    B = rgb_chwt[2].float()

    def _quantise(coeff):
        w_r, w_g, w_b, off = coeff
        return (w_r * R + w_g * G + w_b * B + off + 0.5).clamp(0, 255).to(torch.uint8)

    Y = _quantise(_YUV_Y_COEFF)

    def _subsample(coeff):
        w_r, w_g, w_b, off = coeff
        val = w_r * R + w_g * G + w_b * B + off  # (H, W, T) float
        H_, W_, T_ = val.shape
        val = val.view(H_ // 2, 2, W_ // 2, 2, T_).mean(dim=(1, 3))  # (H/2, W/2, T)
        return (val + 0.5).clamp(0, 255).to(torch.uint8)

    return Y, _subsample(_YUV_CB_COEFF), _subsample(_YUV_CR_COEFF)


# ---------------------------------------------------------------------------
# Test parametrization
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mesh_device, num_links, device_params, topology",
    [[(4, 32), 2, ring_params, ttnn.Topology.Ring], [(4, 8), 2, line_params, ttnn.Topology.Linear]],
    ids=["bh_4x32", "bh_4x8"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "height, width",
    [(720, 1280), (480, 832)],
    ids=["720p", "480p"],
)
class TestFastDeviceToHost:
    def test_correctness(self, mesh_device, num_links, device_params, topology, height, width):
        """Round-trip: shard to device -> fast_device_to_host -> compare to reference."""
        ref = _make_reference_tensor(height, width)
        tt_tensor = _shard_to_device(ref, mesh_device)
        ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)
        concat_dims = _make_concat_dims()

        result = fast_device_to_host(tt_tensor, mesh_device, concat_dims, ccl_manager=ccl_manager)

        rank = int(ttnn.distributed_context_get_rank()) if ttnn.using_distributed_env() else 0
        assert result is not None, f"Rank {rank} got None from fast_device_to_host"
        assert result.shape == ref.shape, f"Shape mismatch: {result.shape} vs {ref.shape}"
        torch.testing.assert_close(result, ref, rtol=0, atol=0)

    def test_performance(self, mesh_device, num_links, device_params, topology, height, width):
        """Measure average D2H time over 10 iterations."""
        n_iters = 10

        ref = _make_reference_tensor(height, width)
        tt_tensor = _shard_to_device(ref, mesh_device)
        ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)
        concat_dims = _make_concat_dims()

        # Warmup
        fast_device_to_host(tt_tensor, mesh_device, concat_dims, ccl_manager=ccl_manager)

        # Sync + barrier before measurement
        ttnn.synchronize_device(mesh_device)
        if ttnn.using_distributed_env():
            ttnn.distributed_context_barrier()

        start = time.perf_counter()
        for _ in range(n_iters):
            video = fast_device_to_host(tt_tensor, mesh_device, concat_dims, ccl_manager=ccl_manager)
            video = video.permute(0, 2, 3, 4, 1).float()
        # Sync + barrier before measurement
        ttnn.synchronize_device(mesh_device)
        if ttnn.using_distributed_env():
            ttnn.distributed_context_barrier()
        end = time.perf_counter()

        avg_s = (end - start) / n_iters
        tensor_bytes = B * C * T * height * width * 2
        throughput_gbs = (tensor_bytes / avg_s) / 1e9

        rank = int(ttnn.distributed_context_get_rank()) if ttnn.using_distributed_env() else 0
        if rank == 0:
            print(f"\n--- fast_device_to_host + permute + float performance (root=0) ---")
            print(f"  Mesh shape:    {tuple(mesh_device.shape)}")
            print(f"  Output shape:  (1, {T}, {height}, {width}, {C}) float32 (BTHWC)")
            print(f"  Output size:   {tensor_bytes / 1e6:.1f} MB")
            print(f"  Iterations:    {n_iters}")
            print(f"  Average time:  {avg_s * 1000:.1f} ms")
            print(f"  Throughput:    {throughput_gbs:.2f} GB/s")

    def test_uint8_accuracy(self, mesh_device, num_links, device_params, topology, height, width):
        """Compare on-device uint8 conversion against host float32 path.

        Runs twice with different input tensors to verify program cache reuse
        (override_runtime_arguments must correctly update buffer addresses).
        """
        import torch

        concat_dims = _make_concat_dims()
        ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)

        for iteration, seed in enumerate([42, 123]):
            gen = torch.Generator().manual_seed(seed)
            ref = torch.rand(B, C, T, height, width, generator=gen, dtype=torch.bfloat16) * 2.0 - 1.0
            tt_tensor = _shard_to_device(ref, mesh_device)

            # --- Host reference path: bf16 → float32 → ×255 → uint8 ---
            host_uint8 = ((ref.float() + 1.0) * 0.5 * 255.0).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 4, 1)

            # --- Device path: pre_transfer_fn handles bf16 → uint8 on device ---
            device_uint8 = fast_device_to_host(
                tt_tensor,
                mesh_device,
                concat_dims,
                ccl_manager=ccl_manager,
                pre_transfer_fn=float_to_uint8,
                permute=(0, 2, 3, 4, 1),
            )

            diff = (device_uint8.int() - host_uint8.int()).abs()
            max_err = diff.max().item()
            mean_err = diff.float().mean().item()
            pct_exact = (diff == 0).float().mean().item() * 100

            label = "fresh" if iteration == 0 else "cached"
            print(f"\n--- uint8 accuracy, iter {iteration} ({label}) ---")
            print(f"  Max error:     {max_err}")
            print(f"  Mean error:    {mean_err:.4f}")
            print(f"  Exact match:   {pct_exact:.1f}%")
            assert max_err <= 1, f"Max uint8 error {max_err} > 1 on iter {iteration} ({label})"

    def test_uint8_performance(self, mesh_device, num_links, device_params, topology, height, width):
        """Measure on-device uint8 conversion + D2H + permute."""
        import torch

        n_iters = 10
        gen = torch.Generator().manual_seed(42)
        ref = torch.rand(B, C, T, height, width, generator=gen, dtype=torch.bfloat16)
        tt_tensor = _shard_to_device(ref, mesh_device)
        concat_dims = _make_concat_dims()
        ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)

        def _convert_and_transfer():
            return fast_device_to_host(
                tt_tensor,
                mesh_device,
                concat_dims,
                ccl_manager=ccl_manager,
                pre_transfer_fn=float_to_uint8,
                permute=(0, 2, 3, 4, 1),
            )

        # Warmup
        _convert_and_transfer()

        ttnn.synchronize_device(mesh_device)
        start = time.perf_counter()
        for _ in range(n_iters):
            _convert_and_transfer()
        ttnn.synchronize_device(mesh_device)
        end = time.perf_counter()

        avg_s = (end - start) / n_iters
        tensor_bytes = B * C * T * height * width
        throughput_gbs = (tensor_bytes / avg_s) / 1e9

        rank = int(ttnn.distributed_context_get_rank()) if ttnn.using_distributed_env() else 0
        if rank == 0:
            print(f"\n--- on-device uint8 + D2H + permute performance (root=0) ---")
            print(f"  Mesh shape:    {tuple(mesh_device.shape)}")
            print(f"  Output shape:  (1, {T}, {height}, {width}, {C}) uint8 (BTHWC)")
            print(f"  Output size:   {tensor_bytes / 1e6:.1f} MB")
            print(f"  Iterations:    {n_iters}")
            print(f"  Average time:  {avg_s * 1000:.1f} ms")
            print(f"  Throughput:    {throughput_gbs:.2f} GB/s")

    def test_yuv_correctness(self, mesh_device, num_links, device_params, topology, height, width):
        """fast_device_to_host_yuv: BCTHW bf16 [-1,1] -> on-device YUV 4:2:0 -> planar uint8.

        Compares the full pipeline (on-device permute + YUV kernel + batched D2H +
        planar concat) against a PyTorch BT.601 reference assembled into the same
        ``[Y | Cb | Cr]`` planar layout that ``fast_device_to_host_yuv`` emits.
        """
        Hu, Wu = height // 2, width // 2
        hw = height * width
        uv = Hu * Wu

        gen = torch.Generator().manual_seed(42)
        # BCTHW bf16 in [-1, 1] — same value range the YUV kernel and the Wan VAE produce.
        ref_bcthw = torch.rand(B, C, T, height, width, generator=gen, dtype=torch.bfloat16) * 2.0 - 1.0

        # Host reference: convert BCTHW (1, 3, T, H, W) -> CHWT (3, H, W, T) and run
        # the same BT.601 conversion the kernel does.
        ref_chwt = ref_bcthw.squeeze(0).permute(0, 2, 3, 1)  # (3, H, W, T)
        ref_Y, ref_Cb, ref_Cr = _host_yuv_reference(ref_chwt)
        # ref_Y: (H, W, T); ref_Cb/Cr: (H/2, W/2, T).

        # Build the expected planar (T, hw + 2*uv) buffer in the same layout as
        # the function's output: per-frame [Y | Cb | Cr], T outermost.
        expected = np.empty((T, hw + 2 * uv), dtype=np.uint8)
        expected[:, :hw] = ref_Y.permute(2, 0, 1).reshape(T, hw).numpy()
        expected[:, hw : hw + uv] = ref_Cb.permute(2, 0, 1).reshape(T, uv).numpy()
        expected[:, hw + uv :] = ref_Cr.permute(2, 0, 1).reshape(T, uv).numpy()

        tt_tensor = _shard_to_device(ref_bcthw, mesh_device)
        ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)
        actual = fast_device_to_host_yuv(tt_tensor, mesh_device, ccl_manager=ccl_manager, debug=True)

        rank = int(ttnn.distributed_context_get_rank()) if ttnn.using_distributed_env() else 0
        if rank != 0:
            return

        assert actual is not None, f"Rank {rank} got None from fast_device_to_host_yuv"
        assert actual.shape == expected.shape, f"shape: {actual.shape} vs {expected.shape}"

        diff = np.abs(actual.astype(np.int32) - expected.astype(np.int32))
        y_diff = diff[:, :hw]
        cb_diff = diff[:, hw : hw + uv]
        cr_diff = diff[:, hw + uv :]

        print(f"\n--- fast_device_to_host_yuv correctness ({height}x{width}) ---")
        print(f"  Y : max err {int(y_diff.max())}, mean err {y_diff.mean():.4f}")
        print(f"  Cb: max err {int(cb_diff.max())}, mean err {cb_diff.mean():.4f}")
        print(f"  Cr: max err {int(cr_diff.max())}, mean err {cr_diff.mean():.4f}")

        assert int(y_diff.max()) <= 1, f"Y max err {int(y_diff.max())} > 1"
        assert int(cb_diff.max()) <= 2, f"Cb max err {int(cb_diff.max())} > 2"
        assert int(cr_diff.max()) <= 2, f"Cr max err {int(cr_diff.max())} > 2"

    def test_yuv_performance(self, mesh_device, num_links, device_params, topology, height, width):
        """Time fast_device_to_host_yuv: on-device YUV + batched D2H + planar concat."""
        n_iters = 10
        Hu, Wu = height // 2, width // 2

        gen = torch.Generator().manual_seed(42)
        ref_bcthw = torch.rand(B, C, T, height, width, generator=gen, dtype=torch.bfloat16) * 2.0 - 1.0
        tt_tensor = _shard_to_device(ref_bcthw, mesh_device)
        ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)

        # Warmup (also forces program cache population).
        fast_device_to_host_yuv(tt_tensor, mesh_device, ccl_manager=ccl_manager)
        ttnn.synchronize_device(mesh_device)

        start = time.perf_counter()
        for _ in range(n_iters):
            fast_device_to_host_yuv(tt_tensor, mesh_device, ccl_manager=ccl_manager)
        ttnn.synchronize_device(mesh_device)
        end = time.perf_counter()

        avg_s = (end - start) / n_iters
        output_bytes = T * (height * width + 2 * Hu * Wu)
        throughput_gbs = (output_bytes / avg_s) / 1e9

        rank = int(ttnn.distributed_context_get_rank()) if ttnn.using_distributed_env() else 0
        if rank == 0:
            print(f"\n--- fast_device_to_host_yuv performance ({height}x{width}, root=0) ---")
            print(f"  Mesh shape:    {tuple(mesh_device.shape)}")
            print(f"  Output shape:  (T={T}, plane_bytes={output_bytes // T}) uint8 yuv420p")
            print(f"  Output size:   {output_bytes / 1e6:.1f} MB")
            print(f"  Iterations:    {n_iters}")
            print(f"  Average time:  {avg_s * 1000:.1f} ms")
            print(f"  Throughput:    {throughput_gbs:.2f} GB/s")


# ---------------------------------------------------------------------------
# TestYUVConversionMesh — on-device YUV + D2H + planar concat on a 4x8 mesh
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mesh_device, num_links, device_params, topology",
    [[(4, 8), 2, line_params, ttnn.Topology.Linear]],
    ids=["bh_4x8"],
    indirect=["mesh_device", "device_params"],
)
class TestYUVConversionMesh:
    """End-to-end on-device YUV conversion + D2H + planar concat on 4x8, 720p.

    Uses ``ttnn.experimental.yuv_conversion`` (see test_yuv_conversion.py for
    the single-device kernel test).  Input is CHWT bf16 in [-1, 1]; outputs
    are 3 uint8 tensors (Y full-res, Cb/Cr 4:2:0).  Sharding: H on TP axis 0,
    W on SP axis 1 — per-shard CHWT is (3, 180, 160, 81).

    Run with:
        pytest models/tt_dit/tests/unit/test_fast_device_to_host.py::TestYUVConversionMesh -s
    """

    @staticmethod
    def _shard_input(host_chwt: torch.Tensor, mesh_device: ttnn.MeshDevice) -> ttnn.Tensor:
        """Shard a CHWT (3, H, W, T) bf16 tensor: H on TP axis 0, W on SP axis 1."""
        return typed_tensor_2dshard(
            host_chwt,
            mesh_device,
            shard_mapping={TP_AXIS: 1, SP_AXIS: 2},  # mesh axis -> CHWT dim
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
        )

    @staticmethod
    def _coefficients():
        return ttnn.experimental.YUVCoefficients(y=list(_YUV_Y_COEFF), cb=list(_YUV_CB_COEFF), cr=list(_YUV_CR_COEFF))

    def test_correctness(self, mesh_device, num_links, device_params, topology):
        """CHWT bf16 -> on-device YUV -> per-component D2H -> compare to host."""
        H, W, T_ = 720, 1280, 81
        Hu, Wu = H // 2, W // 2

        gen = torch.Generator().manual_seed(42)
        cpu_chwt = torch.rand(3, H, W, T_, generator=gen, dtype=torch.bfloat16) * 2.0 - 1.0
        ref_Y, ref_Cb, ref_Cr = _host_yuv_reference(cpu_chwt)

        tt_in = self._shard_input(cpu_chwt, mesh_device)
        ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)

        tt_Y, tt_Cb, tt_Cr = ttnn.experimental.yuv_conversion(tt_in, self._coefficients())

        # Y is (1, H, W, T); Cb/Cr are (1, H/2, W/2, T).  Mesh axis 0 -> tensor
        # dim 1 (H), mesh axis 1 -> tensor dim 2 (W).
        concat_dims = [1, 2]
        dev_Y = fast_device_to_host(tt_Y, mesh_device, concat_dims, ccl_manager=ccl_manager)
        dev_Cb = fast_device_to_host(tt_Cb, mesh_device, concat_dims, ccl_manager=ccl_manager)
        dev_Cr = fast_device_to_host(tt_Cr, mesh_device, concat_dims, ccl_manager=ccl_manager)

        rank = int(ttnn.distributed_context_get_rank()) if ttnn.using_distributed_env() else 0
        if rank != 0:
            return  # Non-root ranks: nothing to verify

        assert dev_Y is not None and dev_Cb is not None and dev_Cr is not None
        assert dev_Y.shape == torch.Size([1, H, W, T_]), f"Y shape {dev_Y.shape}"
        assert dev_Cb.shape == torch.Size([1, Hu, Wu, T_]), f"Cb shape {dev_Cb.shape}"
        assert dev_Cr.shape == torch.Size([1, Hu, Wu, T_]), f"Cr shape {dev_Cr.shape}"

        # dev_* are (1, H, W, T); ref_* are (H, W, T).  Squeeze B and compare directly.
        dev_Y_hwt = dev_Y.squeeze(0)  # (H, W, T)
        dev_Cb_hwt = dev_Cb.squeeze(0)  # (H/2, W/2, T)
        dev_Cr_hwt = dev_Cr.squeeze(0)

        # Spot-check a few positions before the full diff so a failure points
        # at the data flow rather than the comparison expression.
        for h, w, t in [(0, 0, 0), (10, 20, 5), (H - 1, W - 1, T_ - 1), (H // 2, W // 3, T_ // 4)]:
            print(f"  Y[{h:>3},{w:>4},{t:>2}]  dev={int(dev_Y_hwt[h, w, t])}  ref={int(ref_Y[h, w, t])}")

        diff_Y = (dev_Y_hwt.int() - ref_Y.int()).abs()
        diff_Cb = (dev_Cb_hwt.int() - ref_Cb.int()).abs()
        diff_Cr = (dev_Cr_hwt.int() - ref_Cr.int()).abs()

        max_Y, max_Cb, max_Cr = diff_Y.max().item(), diff_Cb.max().item(), diff_Cr.max().item()

        print(f"\n--- YUV correctness 4x8 720p ---")
        print(f"  Y : max err {max_Y}, mean err {diff_Y.float().mean().item():.4f}")
        print(f"  Cb: max err {max_Cb}, mean err {diff_Cb.float().mean().item():.4f}")
        print(f"  Cr: max err {max_Cr}, mean err {diff_Cr.float().mean().item():.4f}")

        assert max_Y <= 1, f"Y max err {max_Y} > 1"
        assert max_Cb <= 2, f"Cb max err {max_Cb} > 2"
        assert max_Cr <= 2, f"Cr max err {max_Cr} > 2"

    def test_performance(self, mesh_device, num_links, device_params, topology):
        """Time on-device YUV + batched D2H + planar yuv420p concat."""
        H, W, T_ = 720, 1280, 81
        Hu, Wu = H // 2, W // 2
        n_iters = 10

        gen = torch.Generator().manual_seed(42)
        cpu_chwt = torch.rand(3, H, W, T_, generator=gen, dtype=torch.bfloat16) * 2.0 - 1.0

        tt_in = self._shard_input(cpu_chwt, mesh_device)
        coefficients = self._coefficients()

        def _convert_and_planar():
            tt_Y, tt_Cb, tt_Cr = ttnn.experimental.yuv_conversion(tt_in, coefficients)
            return _yuv_planar_d2h(tt_Y, tt_Cb, tt_Cr, mesh_device, H, W, T_)

        # Warmup (also forces JIT/program-cache population).
        _convert_and_planar()
        ttnn.synchronize_device(mesh_device)

        start = time.perf_counter()
        for _ in range(n_iters):
            planar = _convert_and_planar()
        ttnn.synchronize_device(mesh_device)
        end = time.perf_counter()

        avg_s = (end - start) / n_iters
        output_bytes = T_ * (H * W + 2 * Hu * Wu)
        throughput_gbs = output_bytes / avg_s / 1e9

        rank = int(ttnn.distributed_context_get_rank()) if ttnn.using_distributed_env() else 0
        if rank == 0:
            print(f"\n--- on-device YUV + D2H + planar concat (4x8, 720p, root=0) ---")
            print(f"  Mesh shape:    {tuple(mesh_device.shape)}")
            print(f"  Output shape:  (T={T_}, plane_bytes={output_bytes // T_}) uint8 yuv420p")
            print(f"  Output size:   {output_bytes / 1e6:.1f} MB")
            print(f"  Iterations:    {n_iters}")
            print(f"  Average time:  {avg_s * 1000:.1f} ms")
            print(f"  Throughput:    {throughput_gbs:.2f} GB/s")


# ---------------------------------------------------------------------------
# TestPermuteUint8Mesh — isolate ttnn.permute correctness on multi-device uint8
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mesh_device, num_links, device_params, topology",
    [[(4, 8), 2, line_params, ttnn.Topology.Linear]],
    ids=["bh_4x8"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "H, W, T_",
    [
        (8, 16, 4),  # tiny: per-shard (2, 2, 4) — easy to hand-verify
        (180, 160, 81),  # per-shard (45, 20, 81)
        (720, 1280, 81),  # actual YUV Y output (per-shard 180, 160, 81)
    ],
    ids=["tiny", "med", "yuv_y_720p"],
)
class TestPermuteUint8Mesh:
    """Isolate ``ttnn.permute`` correctness on a row_major uint8 multi-device
    sharded tensor — the exact pattern used after ``yuv_conversion``.

    Two tests:
      - ``test_no_permute_roundtrip`` is a baseline: shard + D2H without permute.
        If this fails, the bug isn't permute at all.
      - ``test_permute_bhwt_to_bcthw`` runs ``ttnn.permute(t, (0, 3, 1, 2))``
        and compares against ``host.permute(0, 3, 1, 2)``.

    Run with:
        pytest models/tt_dit/tests/unit/test_fast_device_to_host.py::TestPermuteUint8Mesh -s
    """

    def test_no_permute_roundtrip(self, mesh_device, num_links, device_params, topology, H, W, T_):
        """Baseline: shard a uint8 BHWT tensor, D2H, expect bit-exact match."""
        if H % 4 != 0 or W % 8 != 0:
            pytest.skip(f"H={H} or W={W} not divisible by mesh shape (4, 8)")

        gen = torch.Generator().manual_seed(42)
        host = torch.randint(0, 256, (1, H, W, T_), generator=gen, dtype=torch.uint8)

        tt_in = typed_tensor_2dshard(
            host,
            mesh_device,
            shard_mapping={TP_AXIS: 1, SP_AXIS: 2},  # H on axis 0, W on axis 1
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint8,
        )

        ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)
        dev_out = fast_device_to_host(tt_in, mesh_device, [1, 2], ccl_manager=ccl_manager)

        rank = int(ttnn.distributed_context_get_rank()) if ttnn.using_distributed_env() else 0
        if rank != 0:
            return

        assert dev_out.shape == host.shape, f"shape: {dev_out.shape} vs {host.shape}"
        diff = (dev_out.int() - host.int()).abs()
        max_err = diff.max().item()
        nz = (dev_out != 0).sum().item()
        print(f"\n  no_permute  H={H} W={W} T={T_}: max err {max_err}, nonzero {nz}/{dev_out.numel()}")
        assert max_err == 0, f"baseline roundtrip failed: max err {max_err}"

    def test_permute_bhwt_to_bcthw(self, mesh_device, num_links, device_params, topology, H, W, T_):
        """ttnn.permute(0, 3, 1, 2) on a sharded row_major uint8 tensor.

        Same call site as TestYUVConversionMesh, but on a deterministic random
        tensor instead of a kernel output — so a failure here is unambiguous.
        """
        if H % 4 != 0 or W % 8 != 0:
            pytest.skip(f"H={H} or W={W} not divisible by mesh shape (4, 8)")

        gen = torch.Generator().manual_seed(42)
        host = torch.randint(0, 256, (1, H, W, T_), generator=gen, dtype=torch.uint8)
        host_permuted = host.permute(0, 3, 1, 2).contiguous()  # (1, T, H, W)

        tt_in = typed_tensor_2dshard(
            host,
            mesh_device,
            shard_mapping={TP_AXIS: 1, SP_AXIS: 2},
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint8,
        )

        # On-device permute under test.
        tt_out = ttnn.permute(tt_in, (0, 3, 1, 2))

        # After permute: input dim 1 (H, sharded on axis 0) moved to output dim 2;
        # input dim 2 (W, sharded on axis 1) moved to output dim 3.
        ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)
        dev_out = fast_device_to_host(tt_out, mesh_device, [2, 3], ccl_manager=ccl_manager)

        rank = int(ttnn.distributed_context_get_rank()) if ttnn.using_distributed_env() else 0
        if rank != 0:
            return

        print(f"\n  permute  H={H} W={W} T={T_} perm=(0,3,1,2):")
        print(f"    expected shape: {tuple(host_permuted.shape)}, got: {tuple(dev_out.shape)}")

        # Diagnostic: a few specific positions, plus nonzero count.
        for b, t, h, w in [(0, 0, 0, 0), (0, T_ - 1, H - 1, W - 1), (0, T_ // 2, H // 2, W // 2)]:
            print(f"    out[{b},{t},{h},{w}]  dev={int(dev_out[b, t, h, w])}  " f"ref={int(host_permuted[b, t, h, w])}")
        nz = (dev_out != 0).sum().item()
        print(f"    nonzero: {nz}/{dev_out.numel()}")

        assert dev_out.shape == host_permuted.shape, f"shape mismatch: {dev_out.shape} vs {host_permuted.shape}"
        diff = (dev_out.int() - host_permuted.int()).abs()
        max_err = diff.max().item()
        print(f"    max err: {max_err}")
        assert max_err == 0, f"ttnn.permute broken on uint8 multi-device: max err {max_err}"


# ---------------------------------------------------------------------------
# TestD2HLayoutPerformance
# ---------------------------------------------------------------------------

# Dimension-ordering configs: perm applied to (C=0, T=1, H=2, W=3) dims,
# followed by the resulting positions of H and W in the permuted tensor.
_DIM_ORDER_CONFIGS: dict[str, tuple[tuple[int, ...], int, int]] = {
    #        perm from (C,T,H,W)   h_pos  w_pos
    "CTHW": ((0, 1, 2, 3), 2, 3),
    "THWC": ((1, 2, 3, 0), 1, 2),
    "CHWT": ((0, 2, 3, 1), 1, 2),
}

# (label, C, H, W); T=81 frames is fixed
_D2H_SHAPES = [
    ("720p_3c", 3, 720, 1280),  # full RGB 720p
    ("480p_3c", 3, 480, 832),  # full RGB 480p
    ("720p_1c", 1, 720, 1280),  # Y channel 720p
    ("480p_1c", 1, 480, 832),  # Y channel 480p
    ("360p_1c", 1, 360, 640),  # UV pooled 720p (H//2, W//2)
    ("240p_1c", 1, 240, 416),  # UV pooled 480p
]

_D2H_T = 81  # frames
_D2H_TP_SIZE = 4  # mesh axis-0 devices (height sharding)
_D2H_SP_SIZE = 8  # mesh axis-1 devices (width sharding)
_DTYPE_BYTES = 1  # bytes per uint8 element (on-device after RGB/YUV conversion)


def _d2h_shard_shape(
    C: int,
    H: int,
    W: int,
    perm: tuple[int, ...],
    h_dim: int,
    w_dim: int,
) -> tuple[int, ...]:
    """Per-device logical shard shape after permuting (C,T,H,W) and splitting H/W."""
    base = [C, _D2H_T, H, W]
    shape = [base[i] for i in perm]
    shape[h_dim] //= _D2H_TP_SIZE
    shape[w_dim] //= _D2H_SP_SIZE
    return tuple(shape)


def _d2h_alloc_bytes(shard_shape: tuple[int, ...], layout: ttnn.Layout) -> int:
    """Estimated bytes allocated per device, including padding.

    TILE: last two dims rounded up to multiples of 32 (one 32×32 uint8 tile = 1 024 B).
    ROW_MAJOR: no tile padding; reports exact logical size.
    """
    s = list(shard_shape)
    if layout == ttnn.TILE_LAYOUT:
        s[-2] = math.ceil(s[-2] / 32) * 32
        s[-1] = math.ceil(s[-1] / 32) * 32
    return math.prod(s) * _DTYPE_BYTES


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [[(4, 8), line_params]],
    ids=["bh_4x8"],
    indirect=["mesh_device", "device_params"],
)
class TestD2HLayoutPerformance:
    """Sweep dim-ordering × layout × shape internally and report a single perf table.

    For each (shape, layout, dim_order):
      - Shards a (C, T, H, W) uint8 tensor onto a 4×8 mesh
        (H split over axis-0, W split over axis-1).
        uint8 reflects the on-device dtype after RGB/YUV color conversion.
      - Warms up, then times 10 iterations of:
            out = tt_tensor.cpu(blocking=False)
            ttnn.synchronize_device(mesh_device)
      - Records per-device allocated size (with tile padding where applicable),
        average transfer time, and throughput.

    All results are emitted in a single summary table at the end.

    Run with:
        pytest models/tt_dit/tests/unit/test_fast_device_to_host.py \\
            -k "TestD2HLayoutPerformance" --timeout=600 -s
    """

    def test_transfer_speed(self, mesh_device: ttnn.MeshDevice, device_params):
        n_iters = 10
        layouts = (("row_major", ttnn.ROW_MAJOR_LAYOUT), ("tile", ttnn.TILE_LAYOUT))
        results = []

        for shape_name, channels, height, width in _D2H_SHAPES:
            for layout_name, layout in layouts:
                for dim_order_name, (perm, h_dim, w_dim) in _DIM_ORDER_CONFIGS.items():
                    gen = torch.Generator().manual_seed(42)
                    host = torch.randint(0, 256, (channels, _D2H_T, height, width), generator=gen, dtype=torch.uint8)
                    host = host.permute(perm).contiguous()

                    tt_tensor = typed_tensor_2dshard(
                        host,
                        mesh_device,
                        shard_mapping={TP_AXIS: h_dim, SP_AXIS: w_dim},
                        layout=layout,
                        dtype=ttnn.uint8,
                    )

                    shard = _d2h_shard_shape(channels, height, width, perm, h_dim, w_dim)
                    logical_per_dev = math.prod(shard) * _DTYPE_BYTES
                    alloc_per_dev = _d2h_alloc_bytes(shard, layout)
                    logical_total = channels * _D2H_T * height * width * _DTYPE_BYTES
                    padding_overhead = (alloc_per_dev / logical_per_dev - 1.0) * 100.0

                    # Warmup. `out` must be held until after synchronize: the dispatch
                    # descriptor stores a raw pointer into the host buffer, and dropping
                    # the returned tensor before the async read completes is a UAF.
                    out = tt_tensor.cpu(blocking=False)
                    ttnn.synchronize_device(mesh_device)

                    start = time.perf_counter()
                    for _ in range(n_iters):
                        out = tt_tensor.cpu(blocking=False)
                        ttnn.synchronize_device(mesh_device)
                    elapsed_s = time.perf_counter() - start
                    del out

                    avg_ms = elapsed_s / n_iters * 1_000
                    throughput_gbs = (logical_total / (elapsed_s / n_iters)) / 1e9

                    results.append(
                        {
                            "shape": shape_name,
                            "layout": layout_name,
                            "dim_order": dim_order_name,
                            "shard": str(shard),
                            "alloc_mb": alloc_per_dev / 1e6,
                            "pad_pct": padding_overhead,
                            "total_mb": logical_total / 1e6,
                            "avg_ms": avg_ms,
                            "gbs": throughput_gbs,
                        }
                    )

                    del tt_tensor

        rank = int(ttnn.distributed_context_get_rank()) if ttnn.using_distributed_env() else 0
        if rank != 0:
            return

        shard_w = max(len(r["shard"]) for r in results)
        cols = (
            ("shape", 8, "<", "s"),
            ("layout", 10, "<", "s"),
            ("dim_order", 10, "<", "s"),
            ("shard", shard_w, "<", "s"),
            ("alloc/dev (MB)", 15, ">", ".2f"),
            ("pad %", 7, ">", ".1f"),
            ("total (MB)", 11, ">", ".2f"),
            ("avg (ms)", 9, ">", ".1f"),
            ("GB/s", 7, ">", ".2f"),
        )
        keys = ("shape", "layout", "dim_order", "shard", "alloc_mb", "pad_pct", "total_mb", "avg_ms", "gbs")

        header = " ".join(f"{name:{align}{width}}" for name, width, align, _ in cols)
        print(f"\n--- D2H layout sweep: mesh {tuple(mesh_device.shape)} ---")
        print(header)
        print("-" * len(header))
        for r in results:
            cells = []
            for (_, width, align, fmt), key in zip(cols, keys):
                cells.append(f"{r[key]:{align}{width}{fmt}}")
            print(" ".join(cells))


# ---------------------------------------------------------------------------
# Host-only rearrange benchmark
# ---------------------------------------------------------------------------


def test_host_rearrange_speed():
    """Compare host-side RGB interleaved concat across several backends.

    720p, C=3, uint8. Output: (T, H, W, 3) — ffmpeg AV_PIX_FMT_RGB24 layout.

    Variants × dim_orders (CTHW, CHWT):
      - naive:          per-shard numpy strided scatter into out.
      - reassemble:     production `_reassemble_2d` (single-thread torch
                        permute().contiguous() + assign).
      - threaded:       numpy scatter parallelized across a ThreadPoolExecutor.
      - torch_threaded: same as `threaded` but using torch.Tensor.copy_() on a
                        torch.from_numpy view (different SIMD/iter backend).
      - numba (if installed): @njit(parallel=True) kernels with explicit C=3
                              unroll, separate CTHW/CHWT loop nesting.

    Mesh: 4 (TP/H) x 8 (SP/W). Per-shard (3, T, 180, 160) or (3, 180, 160, T).

    Run with:
        pytest models/tt_dit/tests/unit/test_fast_device_to_host.py::test_host_rearrange_speed -s
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    C, T, H, W = 3, 81, 720, 1280
    TP, SP = 4, 8
    h_per, w_per = H // TP, W // SP

    n_iters = 10
    n_workers = min(8, os.cpu_count() or 8)
    n_shards = TP * SP
    mesh_coords = [(r, c) for r in range(TP) for c in range(SP)]

    def src_view_torch(shard, dim_order):
        """Zero-copy (T, h_per, w_per, C) torch view of the shard."""
        # CTHW: (C, T, h, w) -> permute(1,2,3,0) -> (T, h, w, C)
        # CHWT: (C, h, w, T) -> permute(3,1,2,0) -> (T, h, w, C)
        if dim_order == "CTHW":
            return shard.permute(1, 2, 3, 0)
        return shard.permute(3, 1, 2, 0)

    def src_view_numpy(shard, dim_order):
        return src_view_torch(shard, dim_order).numpy()

    # --- Variant 1: naive numpy per-shard strided scatter -------------------

    def rgb_concat_naive(shards, dim_order):
        out = np.empty((T, H, W, C), dtype=np.uint8)
        for (r, c), shard in zip(mesh_coords, shards):
            out[:, r * h_per : (r + 1) * h_per, c * w_per : (c + 1) * w_per, :] = src_view_numpy(shard, dim_order)
        return out

    # --- Variant 2: production `_reassemble_2d` -----------------------------

    def rgb_concat_reassemble(shards, dim_order):
        if dim_order == "CTHW":
            permute = (1, 2, 3, 0)
            concat_dims = [2, 3]
        else:  # CHWT
            permute = (3, 1, 2, 0)
            concat_dims = [1, 2]
        shard_shape = list(shards[0].shape)
        out = _reassemble_2d(mesh_coords, shards, shard_shape, (TP, SP), concat_dims, permute=permute)
        return out.numpy()  # zero-copy: out is contiguous

    # --- Variant 3: threaded numpy scatter ----------------------------------

    pool = ThreadPoolExecutor(max_workers=n_workers)

    def rgb_concat_threaded(shards, dim_order):
        out = np.empty((T, H, W, C), dtype=np.uint8)
        srcs = [src_view_numpy(s, dim_order) for s in shards]

        def write_one(out, src, r, c):
            out[:, r * h_per : (r + 1) * h_per, c * w_per : (c + 1) * w_per, :] = src

        futures = [pool.submit(write_one, out, src, r, c) for (r, c), src in zip(mesh_coords, srcs)]
        for f in futures:
            f.result()
        return out

    # --- Variant 4: torch.copy_() + threaded --------------------------------

    def rgb_concat_torch_threaded(shards, dim_order):
        out = np.empty((T, H, W, C), dtype=np.uint8)
        out_t = torch.from_numpy(out)
        srcs = [src_view_torch(s, dim_order) for s in shards]

        def write_one(out_t, src, r, c):
            out_t[:, r * h_per : (r + 1) * h_per, c * w_per : (c + 1) * w_per, :].copy_(src)

        futures = [pool.submit(write_one, out_t, src, r, c) for (r, c), src in zip(mesh_coords, srcs)]
        for f in futures:
            f.result()
        return out

    # --- Variant 5: numba parallel JIT --------------------------------------

    try:
        import numba

        # CTHW source: (C, T, h, w) contiguous. Inner w loop has stride 1 in src
        # for each fixed C, and the C unroll generates 3 sequential streams the
        # hardware prefetcher handles well. Output is (T, H, W, C) contiguous.
        @numba.njit(parallel=True, cache=True, boundscheck=False)
        def _scatter_rgb_cthw(src, out, r_off, c_off, h_per, w_per, T_):
            for t in numba.prange(T_):
                for h in range(h_per):
                    h_g = r_off + h
                    for w in range(w_per):
                        w_g = c_off + w
                        out[t, h_g, w_g, 0] = src[0, t, h, w]
                        out[t, h_g, w_g, 1] = src[1, t, h, w]
                        out[t, h_g, w_g, 2] = src[2, t, h, w]

        # CHWT source: (C, h, w, T) contiguous. Inner t loop has stride 1 in src
        # for each fixed (C, h, w); dest writes hop by H*W*C across t. Three
        # source streams again, but with stride T=81 between (h, w) pixels.
        @numba.njit(parallel=True, cache=True, boundscheck=False)
        def _scatter_rgb_chwt(src, out, r_off, c_off, h_per, w_per, T_):
            for h in numba.prange(h_per):
                h_g = r_off + h
                for w in range(w_per):
                    w_g = c_off + w
                    for t in range(T_):
                        out[t, h_g, w_g, 0] = src[0, h, w, t]
                        out[t, h_g, w_g, 1] = src[1, h, w, t]
                        out[t, h_g, w_g, 2] = src[2, h, w, t]

        # Force JIT compilation up front (first call is slow).
        _dummy_src = np.zeros((C, 1, 1, 1), dtype=np.uint8)
        _dummy_out = np.zeros((1, 1, 1, C), dtype=np.uint8)
        _scatter_rgb_cthw(_dummy_src, _dummy_out, 0, 0, 1, 1, 1)
        _scatter_rgb_chwt(_dummy_src, _dummy_out, 0, 0, 1, 1, 1)

        def rgb_concat_numba(shards, dim_order):
            out = np.empty((T, H, W, C), dtype=np.uint8)
            kernel = _scatter_rgb_cthw if dim_order == "CTHW" else _scatter_rgb_chwt
            for (r, c), shard in zip(mesh_coords, shards):
                # shard contiguous: (C, T, h, w) for CTHW; (C, h, w, T) for CHWT
                kernel(shard.numpy(), out, r * h_per, c * w_per, h_per, w_per, T)
            return out

        HAS_NUMBA = True
    except ImportError:
        HAS_NUMBA = False
        rgb_concat_numba = None  # type: ignore[assignment]

    variants = [
        ("naive", rgb_concat_naive),
        ("reassemble", rgb_concat_reassemble),
        ("threaded", rgb_concat_threaded),
        ("torch_threaded", rgb_concat_torch_threaded),
    ]
    if HAS_NUMBA:
        variants.append(("numba", rgb_concat_numba))

    configs = [
        ("CTHW", (C, T, h_per, w_per)),
        ("CHWT", (C, h_per, w_per, T)),
    ]

    try:
        # --- Correctness check: every variant matches `naive` for both dim_orders ---
        for name, shape in configs:
            gen = torch.Generator().manual_seed(0)
            check_shards = [torch.randint(0, 256, shape, generator=gen, dtype=torch.uint8) for _ in range(n_shards)]
            ref = rgb_concat_naive(check_shards, name)
            for vname, fn in variants[1:]:
                got = fn(check_shards, name)
                assert np.array_equal(ref, got), f"variant '{vname}' disagrees with 'naive' for {name}"

        # --- Benchmark ---
        rows = []
        for name, shape in configs:
            shards = [torch.empty(shape, dtype=torch.uint8) for _ in range(n_shards)]
            for variant_name, fn in variants:
                # Warmup
                out = fn(shards, name)

                start = time.perf_counter()
                for _ in range(n_iters):
                    out = fn(shards, name)
                elapsed = time.perf_counter() - start

                avg_ms = elapsed / n_iters * 1_000
                gbs = (out.nbytes / (elapsed / n_iters)) / 1e9
                rows.append((variant_name, name, avg_ms, gbs, str(out.shape)))

        print()
        print(f"  threads = {n_workers}, numba = {HAS_NUMBA}")
        header = f"  {'variant':<16} {'dim_order':<10} {'avg (ms)':>10} {'GB/s':>8} {'output shape':>22}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for variant_name, name, avg_ms, gbs, shape in rows:
            print(f"  {variant_name:<16} {name:<10} {avg_ms:>10.2f} {gbs:>8.2f} {shape:>22}")
    finally:
        pool.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Host-only YUV 4:2:0 planar concat benchmark
# ---------------------------------------------------------------------------


def test_yuv_planar_concat_speed():
    """Compare host-side YUV 4:2:0 planar concat across several backends.

    Variants × two on-device dim_orders (CTHW, CHWT):
      - naive:          per-shard numpy strided scatter into planar view.
      - vectorized:     torch.stack + permute + contiguous, single big write per plane.
      - threaded:       numpy scatter parallelized across a ThreadPoolExecutor.
      - torch_threaded: same shape as `threaded` but using torch.Tensor.copy_()
                        on torch.as_strided views — different SIMD/iter backend.
      - numba (if installed): @njit(parallel=True) kernels, separate CTHW/CHWT
                              specializations with explicit loop nesting.

    Output shape: (T, H*W + 2*(H/2 * W/2)) uint8 — standard 4:2:0 planar
    [Y | U | V] layout with one row per frame.

    Mesh: 4 (TP/H) x 8 (SP/W). Y per-shard (180, 160); U/V per-shard (90, 80).

    Run with:
        pytest models/tt_dit/tests/unit/test_fast_device_to_host.py::test_yuv_planar_concat_speed -s
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    H, W, T = 720, 1280, 81
    TP, SP = 4, 8
    Hu, Wu = H // 2, W // 2

    h_per_y, w_per_y = H // TP, W // SP
    h_per_uv, w_per_uv = Hu // TP, Wu // SP

    hw = H * W
    uv = Hu * Wu
    row_stride = hw + 2 * uv  # bytes per frame (uint8)

    n_iters = 10
    n_workers = min(8, os.cpu_count() or 8)
    n_shards = TP * SP
    mesh_coords = [(r, c) for r in range(TP) for c in range(SP)]

    def to_thw_view(shard: torch.Tensor, dim_order: str) -> np.ndarray:
        """Zero-copy (T, h, w) numpy view of a per-device shard (C=1)."""
        v = shard.squeeze(0).numpy()  # CTHW -> (T, h, w); CHWT -> (h, w, T)
        if dim_order == "CHWT":
            v = v.transpose(2, 0, 1)  # strided view, no copy
        return v

    def make_views(out):
        y_view = np.lib.stride_tricks.as_strided(out, shape=(T, H, W), strides=(row_stride, W, 1), writeable=True)
        u_view = np.lib.stride_tricks.as_strided(
            out[:, hw:], shape=(T, Hu, Wu), strides=(row_stride, Wu, 1), writeable=True
        )
        v_view = np.lib.stride_tricks.as_strided(
            out[:, hw + uv :], shape=(T, Hu, Wu), strides=(row_stride, Wu, 1), writeable=True
        )
        return y_view, u_view, v_view

    # --- Variant 1: naive per-shard strided scatter -------------------------

    def planar_concat_naive(y_shards, u_shards, v_shards, dim_order):
        out = np.empty((T, row_stride), dtype=np.uint8)
        y_view, u_view, v_view = make_views(out)

        def scatter(shards, view, h_per, w_per):
            for (r, c), shard in zip(mesh_coords, shards):
                view[:, r * h_per : (r + 1) * h_per, c * w_per : (c + 1) * w_per] = to_thw_view(shard, dim_order)

        scatter(y_shards, y_view, h_per_y, w_per_y)
        scatter(u_shards, u_view, h_per_uv, w_per_uv)
        scatter(v_shards, v_view, h_per_uv, w_per_uv)
        return out

    # --- Variant 2: torch.stack + permute + contiguous, one big write -------

    def planar_concat_vectorized(y_shards, u_shards, v_shards, dim_order):
        out = np.empty((T, row_stride), dtype=np.uint8)

        def assemble(shards, h_per, w_per):
            """Stack all shards into one (T, TP*h_per, SP*w_per) contiguous tensor."""
            stacked = torch.stack(shards).squeeze(1)  # (32, ...)
            if dim_order == "CTHW":
                # (32, T, h, w) -> (TP, SP, T, h, w) -> (T, TP, h, SP, w)
                t = stacked.reshape(TP, SP, T, h_per, w_per).permute(2, 0, 3, 1, 4)
            else:  # CHWT
                # (32, h, w, T) -> (TP, SP, h, w, T) -> (T, TP, h, SP, w)
                t = stacked.reshape(TP, SP, h_per, w_per, T).permute(4, 0, 2, 1, 3)
            return t.contiguous().reshape(T, TP * h_per, SP * w_per)

        y_full = assemble(y_shards, h_per_y, w_per_y)  # (T, H, W)
        u_full = assemble(u_shards, h_per_uv, w_per_uv)  # (T, Hu, Wu)
        v_full = assemble(v_shards, h_per_uv, w_per_uv)  # (T, Hu, Wu)

        # `*_full` is contiguous, so .reshape(T, plane) is a zero-copy view.
        # The assignment is one strided-dest write per plane (numpy collapses
        # the matching inner two strides into a per-row memcpy).
        out[:, :hw] = y_full.reshape(T, hw).numpy()
        out[:, hw : hw + uv] = u_full.reshape(T, uv).numpy()
        out[:, hw + uv :] = v_full.reshape(T, uv).numpy()
        return out

    # --- Variant 3: threaded scatter ----------------------------------------

    pool = ThreadPoolExecutor(max_workers=n_workers)

    def planar_concat_threaded(y_shards, u_shards, v_shards, dim_order):
        out = np.empty((T, row_stride), dtype=np.uint8)
        y_view, u_view, v_view = make_views(out)

        # Convert torch shards to numpy views on the main thread (zero-copy).
        # Doing this off-thread would risk torch/Python contention on the GIL.
        y_np = [to_thw_view(s, dim_order) for s in y_shards]
        u_np = [to_thw_view(s, dim_order) for s in u_shards]
        v_np = [to_thw_view(s, dim_order) for s in v_shards]

        def write_one(view, src, r, c, h_per, w_per):
            view[:, r * h_per : (r + 1) * h_per, c * w_per : (c + 1) * w_per] = src

        futures = []
        for (r, c), src in zip(mesh_coords, y_np):
            futures.append(pool.submit(write_one, y_view, src, r, c, h_per_y, w_per_y))
        for (r, c), src in zip(mesh_coords, u_np):
            futures.append(pool.submit(write_one, u_view, src, r, c, h_per_uv, w_per_uv))
        for (r, c), src in zip(mesh_coords, v_np):
            futures.append(pool.submit(write_one, v_view, src, r, c, h_per_uv, w_per_uv))
        for f in futures:
            f.result()
        return out

    # --- Variant 4: torch native copy + threaded ----------------------------

    def planar_concat_torch_threaded(y_shards, u_shards, v_shards, dim_order):
        out = np.empty((T, row_stride), dtype=np.uint8)
        out_t = torch.from_numpy(out)

        # Strided 3D views via torch.as_strided (no copy, shares storage with `out`).
        y_view = out_t.as_strided((T, H, W), (row_stride, W, 1), 0)
        u_view = out_t.as_strided((T, Hu, Wu), (row_stride, Wu, 1), hw)
        v_view = out_t.as_strided((T, Hu, Wu), (row_stride, Wu, 1), hw + uv)

        def src_view(shard):
            v = shard.squeeze(0)  # CTHW: (T, h, w); CHWT: (h, w, T)
            if dim_order == "CHWT":
                v = v.permute(2, 0, 1)
            return v

        # Pre-build all source views on the main thread to avoid contention.
        y_src = [src_view(s) for s in y_shards]
        u_src = [src_view(s) for s in u_shards]
        v_src = [src_view(s) for s in v_shards]

        def write_one(view, src, r, c, h_per, w_per):
            view[:, r * h_per : (r + 1) * h_per, c * w_per : (c + 1) * w_per].copy_(src)

        futures = []
        for (r, c), src in zip(mesh_coords, y_src):
            futures.append(pool.submit(write_one, y_view, src, r, c, h_per_y, w_per_y))
        for (r, c), src in zip(mesh_coords, u_src):
            futures.append(pool.submit(write_one, u_view, src, r, c, h_per_uv, w_per_uv))
        for (r, c), src in zip(mesh_coords, v_src):
            futures.append(pool.submit(write_one, v_view, src, r, c, h_per_uv, w_per_uv))
        for f in futures:
            f.result()
        return out

    # --- Variant 5: numba parallel JIT --------------------------------------

    try:
        import numba

        # CTHW source: (T, h_per, w_per) contiguous. Inner w loop is stride-1
        # in both source and dest -> LLVM emits memcpy-equivalent SIMD.
        @numba.njit(parallel=True, cache=True, boundscheck=False)
        def _scatter_cthw_numba(src, out_flat, plane_offset, row_stride_, W_, r_off, c_off, h_per, w_per, T_):
            for t in numba.prange(T_):
                row_base = t * row_stride_ + plane_offset + r_off * W_ + c_off
                for h in range(h_per):
                    base_dst = row_base + h * W_
                    for w in range(w_per):
                        out_flat[base_dst + w] = src[t, h, w]

        # CHWT source: (h_per, w_per, T) contiguous. Inner T loop is stride-1
        # in source and stride row_stride in dest -> sequential strided stores.
        # Compiled loop avoids numpy's per-element dispatch overhead.
        @numba.njit(parallel=True, cache=True, boundscheck=False)
        def _scatter_chwt_numba(src, out_flat, plane_offset, row_stride_, W_, r_off, c_off, h_per, w_per, T_):
            for h in numba.prange(h_per):
                row_off = plane_offset + (r_off + h) * W_ + c_off
                for w in range(w_per):
                    dst_off = row_off + w
                    for t in range(T_):
                        out_flat[t * row_stride_ + dst_off] = src[h, w, t]

        # Force JIT compilation up front (first call is slow).
        _dummy_src_cthw = np.zeros((1, 1, 1), dtype=np.uint8)
        _dummy_src_chwt = np.zeros((1, 1, 1), dtype=np.uint8)
        _dummy_out = np.zeros(row_stride, dtype=np.uint8)
        _scatter_cthw_numba(_dummy_src_cthw, _dummy_out, 0, row_stride, W, 0, 0, 1, 1, 1)
        _scatter_chwt_numba(_dummy_src_chwt, _dummy_out, 0, row_stride, W, 0, 0, 1, 1, 1)

        def planar_concat_numba(y_shards, u_shards, v_shards, dim_order):
            out = np.empty((T, row_stride), dtype=np.uint8)
            out_flat = out.reshape(-1)

            kernel = _scatter_cthw_numba if dim_order == "CTHW" else _scatter_chwt_numba

            def scatter(shards, plane_offset, plane_W, h_per, w_per):
                for (r, c), shard in zip(mesh_coords, shards):
                    src = shard.squeeze(0).numpy()  # CTHW: (T, h, w); CHWT: (h, w, T)
                    kernel(src, out_flat, plane_offset, row_stride, plane_W, r * h_per, c * w_per, h_per, w_per, T)

            scatter(y_shards, 0, W, h_per_y, w_per_y)
            scatter(u_shards, hw, Wu, h_per_uv, w_per_uv)
            scatter(v_shards, hw + uv, Wu, h_per_uv, w_per_uv)
            return out

        HAS_NUMBA = True
    except ImportError:
        HAS_NUMBA = False
        planar_concat_numba = None  # type: ignore[assignment]

    variants = [
        ("naive", planar_concat_naive),
        ("vectorized", planar_concat_vectorized),
        ("threaded", planar_concat_threaded),
        ("torch_threaded", planar_concat_torch_threaded),
    ]
    if HAS_NUMBA:
        variants.append(("numba", planar_concat_numba))

    configs = [
        # name, Y shard shape, UV shard shape
        ("CTHW", (1, T, h_per_y, w_per_y), (1, T, h_per_uv, w_per_uv)),
        ("CHWT", (1, h_per_y, w_per_y, T), (1, h_per_uv, w_per_uv, T)),
    ]

    try:
        # --- Correctness check: every variant matches `naive` for both dim_orders ---
        for name, y_shape, uv_shape in configs:
            gen = torch.Generator().manual_seed(0)
            cy = [torch.randint(0, 256, y_shape, generator=gen, dtype=torch.uint8) for _ in range(n_shards)]
            cu = [torch.randint(0, 256, uv_shape, generator=gen, dtype=torch.uint8) for _ in range(n_shards)]
            cv = [torch.randint(0, 256, uv_shape, generator=gen, dtype=torch.uint8) for _ in range(n_shards)]
            ref = planar_concat_naive(cy, cu, cv, name)
            for vname, fn in variants[1:]:
                got = fn(cy, cu, cv, name)
                assert np.array_equal(ref, got), f"variant '{vname}' disagrees with 'naive' for {name}"

        # --- Benchmark ---
        rows = []
        for name, y_shape, uv_shape in configs:
            y_shards = [torch.empty(y_shape, dtype=torch.uint8) for _ in range(n_shards)]
            u_shards = [torch.empty(uv_shape, dtype=torch.uint8) for _ in range(n_shards)]
            v_shards = [torch.empty(uv_shape, dtype=torch.uint8) for _ in range(n_shards)]

            for variant_name, fn in variants:
                # Warmup
                out = fn(y_shards, u_shards, v_shards, name)

                start = time.perf_counter()
                for _ in range(n_iters):
                    out = fn(y_shards, u_shards, v_shards, name)
                elapsed = time.perf_counter() - start

                avg_ms = elapsed / n_iters * 1_000
                gbs = (out.nbytes / (elapsed / n_iters)) / 1e9
                rows.append((variant_name, name, avg_ms, gbs, str(out.shape)))

        print()
        print(f"  threads = {n_workers},  numba = {HAS_NUMBA}")
        header = f"  {'variant':<16} {'dim_order':<10} {'avg (ms)':>10} {'GB/s':>8} {'output shape':>20}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for variant_name, name, avg_ms, gbs, shape in rows:
            print(f"  {variant_name:<16} {name:<10} {avg_ms:>10.2f} {gbs:>8.2f} {shape:>20}")
    finally:
        pool.shutdown(wait=True)


# ---------------------------------------------------------------------------
# On-device permute overhead benchmark
# ---------------------------------------------------------------------------


def test_device_permute_overhead():
    """Benchmark on-device ttnn.permute across all dim-ordering pairs.

    Orderings tested: CTHW, THWC, CHWT — all 6 directed pairs.
    Shapes emulate per-shard sizes on a 4×8 mesh (H//4, W//8):
      720p 3-channel (3,81,180,160), 720p 1-channel (1,81,180,160),
      360p 1-channel (1,81,90,80). Dtype: bfloat16.

    100 timed iterations (sync per iteration) after a 10-iteration warmup.

    Run with:
        pytest models/tt_dit/tests/unit/test_fast_device_to_host.py::test_device_permute_overhead -s
    """
    n_warmup = 10
    n_iters = 100
    T = 81

    _ORDERINGS = {
        "CTHW": list("CTHW"),
        "THWC": list("THWC"),
        "CHWT": list("CHWT"),
    }

    # Per-shard sizes emulating a 4x8 mesh (H // 4, W // 8).
    shapes_cfg = [
        ("720p_3c", 3, 720 // 4, 1280 // 8),
        ("720p_1c", 1, 720 // 4, 1280 // 8),
        ("360p_1c", 1, 360 // 4, 640 // 8),
    ]

    device = ttnn.open_device(device_id=0)
    try:
        rows = []

        for shape_name, C, H, W in shapes_cfg:
            dim_vals = {"C": C, "T": T, "H": H, "W": W}

            for src_name, src_dims in _ORDERINGS.items():
                src_shape = tuple(dim_vals[d] for d in src_dims)
                tensor_bytes = math.prod(src_shape) * 2  # bfloat16

                host = torch.zeros(src_shape, dtype=torch.bfloat16)
                tt_in = ttnn.from_torch(host, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)

                for dst_name, dst_dims in _ORDERINGS.items():
                    if dst_name == src_name:
                        continue

                    perm = tuple(src_dims.index(d) for d in dst_dims)

                    for _ in range(n_warmup):
                        out = ttnn.permute(tt_in, perm)
                    ttnn.synchronize_device(device)

                    start = time.perf_counter()
                    for _ in range(n_iters):
                        out = ttnn.permute(tt_in, perm)
                        ttnn.synchronize_device(device)
                    elapsed_s = time.perf_counter() - start

                    avg_ms = elapsed_s / n_iters * 1_000
                    throughput_gbs = (tensor_bytes / (elapsed_s / n_iters)) / 1e9

                    rows.append(
                        {
                            "shape": shape_name,
                            "src": src_name,
                            "dst": dst_name,
                            "src_shape": str(src_shape),
                            "perm": str(perm),
                            "avg_ms": avg_ms,
                            "gbs": throughput_gbs,
                        }
                    )

                del tt_in
    finally:
        ttnn.close_device(device)

    perm_w = max(len("perm"), max(len(r["perm"]) for r in rows))
    shape_w = max(len("src shape"), max(len(r["src_shape"]) for r in rows))
    cols = (
        ("shape", 10, "<", "s"),
        ("src", 6, "<", "s"),
        ("dst", 6, "<", "s"),
        ("perm", perm_w, "<", "s"),
        ("src shape", shape_w, "<", "s"),
        ("avg (ms)", 10, ">", ".2f"),
        ("GB/s", 8, ">", ".2f"),
    )
    keys = ("shape", "src", "dst", "perm", "src_shape", "avg_ms", "gbs")

    header = " ".join(f"{name:{align}{width}}" for name, width, align, _ in cols)
    print("\n--- on-device ttnn.permute latency (single device, bfloat16, row-major) ---")
    print(header)
    print("-" * len(header))
    for r in rows:
        cells = [f"{r[key]:{align}{width}{fmt}}" for (_, width, align, fmt), key in zip(cols, keys)]
        print(" ".join(cells))
