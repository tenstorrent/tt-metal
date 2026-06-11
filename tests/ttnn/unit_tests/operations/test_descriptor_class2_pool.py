# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Regression test for the descriptor-framework slow-path rebuild perf bug (issue #46506), pool family.

Pool2D (ttnn.max_pool2d / ttnn.avg_pool2d) is CB-bound-stable: every per-dispatch tensor address
rides on a sharded CBDescriptor.buffer binding that the framework patches automatically on a cache
hit (input via raw_in_cb, output via out_cb, optional index output via out_idx_cb). Its kernel
runtime args are all shape/sharding-derived (and hash-covered), and the auxiliary reader_indices /
scalar_config buffer addresses live in COMPILE-TIME args of stable, workload-owned buffers. So
Pool2D declares an empty get_dynamic_runtime_args() to opt into the descriptor fast-path (no
create_workload_descriptor() rebuild on a program-cache hit).

This module sets TT_METAL_FORBID_DESCRIPTOR_REBUILD_ON_CACHE_HIT=1 BEFORE importing ttnn, so that
ANY op falling to the slow-path rebuild raises (TT_FATAL) instead of silently rebuilding. Each
config is run 3x with the program cache enabled:
  - 1st call: cache miss -> resolve_bindings() validates each declared Buffer* binding.
  - 2nd/3rd call: cache hit -> the guard fires if the op rebuilds its descriptor.
Numerical correctness is checked against a torch reference via PCC.

Must run with the env var set; the module sets it before importing ttnn.
"""

import os

# Must be set before ttnn first dispatches (the adapter reads it via getenv on first slow-path hit).
os.environ["TT_METAL_FORBID_DESCRIPTOR_REBUILD_ON_CACHE_HIT"] = "1"

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)


def _run_thrice(dev, fn):
    """Run fn 3x with the same config so calls 2 and 3 cache-hit. Raises if the op rebuilds its
    descriptor on a cache hit (FORBID_DESCRIPTOR_REBUILD guard) or misplaces a Buffer* binding
    (resolve_bindings validation on the first/miss call)."""
    out = None
    for _ in range(3):
        out = fn()
        ttnn.synchronize_device(dev)
    return out


# Small spatial dims, batch 16 (per the test contract). Auto-sharding picks HEIGHT_SHARDED for
# these shapes; the input is supplied as the flat (1, 1, N*H*W, C) row-major layout pool expects.
BATCH_SIZE = 16
IN_H = 16
IN_W = 16
CHANNELS = 32
KERNEL = (3, 3)
STRIDE = (2, 2)
PADDING = (1, 1)
DILATION = (1, 1)


def _make_input(device):
    torch.manual_seed(0)
    torch_nhwc = torch.randn(BATCH_SIZE, IN_H, IN_W, CHANNELS, dtype=torch.bfloat16)
    flat = torch_nhwc.reshape(1, 1, BATCH_SIZE * IN_H * IN_W, CHANNELS)
    ttnn_input = ttnn.from_torch(flat, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    # NCHW reference for torch ops.
    torch_nchw = torch_nhwc.permute(0, 3, 1, 2).float()
    return ttnn_input, torch_nchw


def _out_hw():
    out_h = (IN_H + 2 * PADDING[0] - DILATION[0] * (KERNEL[0] - 1) - 1) // STRIDE[0] + 1
    out_w = (IN_W + 2 * PADDING[1] - DILATION[1] * (KERNEL[1] - 1) - 1) // STRIDE[1] + 1
    return out_h, out_w


def _ttnn_to_nchw(ttnn_out, out_h, out_w):
    # Pool output is flat (1, 1, N*out_h*out_w, C) row-major -> (N, C, out_h, out_w).
    t = ttnn.to_torch(ttnn_out).float().reshape(BATCH_SIZE, out_h, out_w, CHANNELS)
    return t.permute(0, 3, 1, 2)


def test_max_pool2d_no_rebuild(device):
    ttnn_input, torch_nchw = _make_input(device)
    out = _run_thrice(
        device,
        lambda: ttnn.max_pool2d(
            input_tensor=ttnn_input,
            batch_size=BATCH_SIZE,
            input_h=IN_H,
            input_w=IN_W,
            channels=CHANNELS,
            kernel_size=KERNEL,
            stride=STRIDE,
            padding=PADDING,
            dilation=DILATION,
        ),
    )
    out_h, out_w = _out_hw()
    ttnn_nchw = _ttnn_to_nchw(out, out_h, out_w)
    ref = torch.nn.functional.max_pool2d(
        torch_nchw, kernel_size=KERNEL, stride=STRIDE, padding=PADDING, dilation=DILATION
    )
    passing, pcc = check_with_pcc(ref, ttnn_nchw, 0.999)
    assert passing, f"max_pool2d PCC mismatch: {pcc}"


def test_max_pool2d_return_indices_no_rebuild(device):
    # return_indices exercises the second (out_idx) CB buffer binding and the {start_row, start_col}
    # runtime args -- still all shape-derived, so the op must stay on the fast-path.
    ttnn_input, _ = _make_input(device)

    def run():
        out, idx = ttnn.max_pool2d(
            input_tensor=ttnn_input,
            batch_size=BATCH_SIZE,
            input_h=IN_H,
            input_w=IN_W,
            channels=CHANNELS,
            kernel_size=KERNEL,
            stride=STRIDE,
            padding=PADDING,
            dilation=DILATION,
            return_indices=True,
        )
        return out

    _run_thrice(device, run)


def test_avg_pool2d_no_rebuild(device):
    # avg_pool2d with count_include_pad=False + non-zero padding materializes the per-stick scalar
    # config tensor (workload-owned, baked into compile-time args) -- exercises that path stays
    # fast on cache hit.
    ttnn_input, torch_nchw = _make_input(device)
    out = _run_thrice(
        device,
        lambda: ttnn.avg_pool2d(
            input_tensor=ttnn_input,
            batch_size=BATCH_SIZE,
            input_h=IN_H,
            input_w=IN_W,
            channels=CHANNELS,
            kernel_size=KERNEL,
            stride=STRIDE,
            padding=[PADDING[0], PADDING[0], PADDING[1], PADDING[1]],
            count_include_pad=False,
        ),
    )
    out_h, out_w = _out_hw()
    ttnn_nchw = _ttnn_to_nchw(out, out_h, out_w)
    ref = torch.nn.functional.avg_pool2d(
        torch_nchw, kernel_size=KERNEL, stride=STRIDE, padding=PADDING, count_include_pad=False
    )
    passing, pcc = check_with_pcc(ref, ttnn_nchw, 0.999)
    assert passing, f"avg_pool2d PCC mismatch: {pcc}"
