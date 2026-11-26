# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Quasar max_pool2d return_indices: INT32 indices on Quasar, UINT16/UINT32 elsewhere."""

import pytest
import torch

import ttnn


def _is_quasar():
    return "quasar" in ttnn.get_arch_name()


def _expected_index_dtype(h, w):
    if _is_quasar():
        return ttnn.int32
    return ttnn.uint16 if h * w <= 65535 else ttnn.uint32


def _max_pool2d_with_indices(device, x_nhwc, k, s, p):
    n, h, w, c = x_nhwc.shape
    x = ttnn.from_torch(
        x_nhwc.reshape(1, 1, n * h * w, c), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    return ttnn.experimental.quasar.max_pool2d(
        input_tensor=x,
        batch_size=n,
        input_h=h,
        input_w=w,
        channels=c,
        kernel_size=[k, k],
        stride=[s, s],
        padding=[p, p],
        dilation=[1, 1],
        ceil_mode=False,
        applied_shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        deallocate_input=False,
        reallocate_halo_output=True,
        return_indices=True,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "n, h, w, c, k, s, p",
    [
        pytest.param(1, 8, 8, 64, 3, 2, 1, id="2-channel-blocks"),
        pytest.param(2, 12, 10, 40, 3, 2, 1, id="partial-channel-block-batch2"),
        pytest.param(1, 6, 6, 32, 5, 1, 2, id="5x5-full-tile"),
        pytest.param(1, 257, 257, 1, 3, 2, 1, id="indices-above-65535"),
    ],
)
def test_max_pool2d_indices(device, n, h, w, c, k, s, p):
    torch.manual_seed(0)
    x_nhwc = torch.randn((n, h, w, c), dtype=torch.bfloat16)
    out, idx = _max_pool2d_with_indices(device, x_nhwc, k, s, p)
    assert idx.dtype == _expected_index_dtype(h, w)

    x_nchw = x_nhwc.permute(0, 3, 1, 2).float()
    ref = torch.nn.functional.max_pool2d(x_nchw, k, s, p)
    oh, ow = ref.shape[2:]
    n_out = n * oh * ow
    ref_v = ref.permute(0, 2, 3, 1).reshape(n_out, c)
    got = ttnn.to_torch(out).float().reshape(-1, out.shape[-1])[:n_out, :c]
    assert torch.equal(got, ref_v), "values differ from torch (max of bf16 is exact)"

    # bf16 ties are common: check that each index points at a maximum inside its window
    # rather than matching torch's tie-break.
    got_idx = ttnn.to_torch(idx).to(torch.int64).reshape(-1, idx.shape[-1])[:n_out, :c]
    assert ((got_idx >= 0) & (got_idx < h * w)).all(), "index outside the input"
    out_pos = torch.arange(n_out) % (oh * ow)
    top = (out_pos // ow * s - p).unsqueeze(1)
    left = (out_pos % ow * s - p).unsqueeze(1)
    row, col = got_idx // w, got_idx % w
    assert ((row >= top) & (row < top + k) & (col >= left) & (col < left + k)).all(), "index outside its window"
    batch = (torch.arange(n_out) // (oh * ow)).unsqueeze(1)
    picked = x_nchw.permute(0, 2, 3, 1).reshape(n, h * w, c)[batch, got_idx, torch.arange(c)]
    assert torch.equal(picked, ref_v), "indices do not point at the window maxima"
    if h * w > 65535:
        assert got_idx.max() > 65535, "expected winning indices beyond the uint16 range"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_max_pool2d_indices_large_window_rejected(device, expect_error):
    if not _is_quasar():
        pytest.skip("INT32 indices are Quasar-only; other archs support large index-returning windows")
    # C <= 16 reduces 16 rows at a time, so a 5x5 window (25 rows) needs several reductions.
    x_nhwc = torch.randn((1, 6, 6, 1), dtype=torch.bfloat16)
    with expect_error(RuntimeError, "does not yet support windows of 5x5"):
        _max_pool2d_with_indices(device, x_nhwc, 5, 1, 2)
