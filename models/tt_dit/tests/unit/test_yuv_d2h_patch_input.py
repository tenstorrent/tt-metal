# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-side coverage for the patchified-input route of ``fast_device_to_host_yuv``.

The device ops are stubbed: these tests check the shape math and that the patchified tensor goes to
``rgb_to_yuv`` as is, with no reshape or permute. Bit-exactness on device is
``test_rgb_to_yuv_patch_input.py``.
"""

import pytest

import ttnn

from ...utils import yuv_d2h


class _ShapeOnly:
    """Stands in for a ttnn tensor / mesh device."""

    def __init__(self, shape):
        self.shape = shape


@pytest.fixture
def stub_device(monkeypatch):
    calls = {}

    def rgb_to_yuv(tensor, **kwargs):
        calls["rgb_to_yuv"] = (tensor, kwargs)
        return "Y", "Cb", "Cr"

    def planar_d2h(tt_Y, tt_Cb, tt_Cr, mesh_device, H, W, T, **kwargs):
        calls["planar"] = (H, W, T, kwargs["out_H"], kwargs["out_W"])
        return "planar"

    def no_device_op(*args, **kwargs):
        raise AssertionError("the patchified route must not reshape or permute")

    monkeypatch.setattr(ttnn, "using_distributed_env", lambda: False)
    monkeypatch.setattr(ttnn.experimental, "rgb_to_yuv", rgb_to_yuv)
    monkeypatch.setattr(ttnn, "permute", no_device_op)
    monkeypatch.setattr(ttnn, "reshape", no_device_op)
    monkeypatch.setattr(yuv_d2h, "_yuv_planar_d2h", planar_d2h)
    return calls


def test_patchified_input_goes_straight_to_rgb_to_yuv(stub_device):
    # Per-shard (1, T, H/p, W/p, 3*p*p) on a 2x4 mesh, p=4: H = 4*4*2, W = 6*4*4.
    tensor = _ShapeOnly((1, 8, 4, 6, 48))
    out = yuv_d2h.fast_device_to_host_yuv(
        tensor, _ShapeOnly((2, 4)), coefficients=object(), logical_h=30, logical_w=90, input_patch_size=4
    )

    assert out == "planar"
    passed, kwargs = stub_device["rgb_to_yuv"]
    assert passed is tensor
    assert kwargs["input_patch_size"] == 4
    assert stub_device["planar"] == (32, 96, 8, 30, 90)


def test_patchified_input_checks_channels(stub_device, expect_error):
    with expect_error(AssertionError, "channels"):
        yuv_d2h.fast_device_to_host_yuv(
            _ShapeOnly((1, 8, 4, 6, 12)), _ShapeOnly((2, 4)), coefficients=object(), input_patch_size=4
        )


def test_patchified_input_is_single_host_only(stub_device, monkeypatch, expect_error):
    monkeypatch.setattr(ttnn, "using_distributed_env", lambda: True)
    with expect_error(ValueError, "single-host"):
        yuv_d2h.fast_device_to_host_yuv(
            _ShapeOnly((1, 8, 4, 6, 48)), _ShapeOnly((2, 4)), coefficients=object(), input_patch_size=4
        )
