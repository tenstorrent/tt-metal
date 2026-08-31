# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-side coverage for the YUV 4:2:0 planar assembly.

The C++/AVX2 scatter is checked byte-for-byte against a NumPy port of the torch
fallback in ``yuv_d2h._yuv_planar_d2h``, so the two assembly paths cannot drift.
Cases cover both source layouts, every tile-tail branch in the CHWT transpose,
and logical crops (including one that drops a whole shard column).
"""

import numpy as np
import pytest

from ...utils.planar_concat import HAS_CPP_PLANAR_CONCAT, planar_concat_cpp
from ...utils.yuv_d2h import fast_device_to_host_yuv

_SENTINEL = 0xAA


def _reference_planar_concat(y_shards, u_shards, v_shards, dim_order, mesh_shape, out_H, out_W, T, fill=_SENTINEL):
    """NumPy port of the torch scatter fallback: the byte layout both paths must produce."""
    _, SP = mesh_shape
    out_Hu, out_Wu = out_H // 2, out_W // 2
    out_hw, out_uv = out_H * out_W, out_Hu * out_Wu
    out = np.full((T, out_hw + 2 * out_uv), fill, dtype=np.uint8)

    planes = [
        (out[:, :out_hw].reshape(T, out_H, out_W), y_shards, out_H, out_W),
        (out[:, out_hw : out_hw + out_uv].reshape(T, out_Hu, out_Wu), u_shards, out_Hu, out_Wu),
        (out[:, out_hw + out_uv :].reshape(T, out_Hu, out_Wu), v_shards, out_Hu, out_Wu),
    ]
    for view, shards, bound_h, bound_w in planes:
        for idx, shard in enumerate(shards):
            r, c = idx // SP, idx % SP
            s = shard[0]
            if dim_order == "CHWT":
                s = np.transpose(s, (2, 0, 1))  # (h, w, T) -> (T, h, w)
            h_per, w_per = s.shape[1], s.shape[2]
            r0, c0 = r * h_per, c * w_per
            vh, vw = min(h_per, bound_h - r0), min(w_per, bound_w - c0)
            if vh <= 0 or vw <= 0:
                continue  # shard lies entirely in the padded tail
            view[:, r0 : r0 + vh, c0 : c0 + vw] = s[:, :vh, :vw]
    return out


def _make_shards(rng, n, h_per, w_per, T, dim_order):
    shape = (1, h_per, w_per, T) if dim_order == "CHWT" else (1, T, h_per, w_per)
    return [np.ascontiguousarray(rng.integers(0, 256, size=shape, dtype=np.uint8)) for _ in range(n)]


# (dim_order, TP, SP, h_per, w_per, T, crop_h, crop_w)
_CASES = [
    pytest.param("CHWT", 2, 2, 32, 32, 32, None, None, id="chwt_clean_tiles"),
    pytest.param("CHWT", 2, 2, 32, 48, 32, None, None, id="chwt_wtail16"),
    pytest.param("CHWT", 2, 2, 32, 40, 32, None, None, id="chwt_wtail_generic"),
    pytest.param("CHWT", 2, 2, 32, 32, 40, None, None, id="chwt_ttail"),
    pytest.param("CHWT", 2, 2, 32, 48, 40, None, None, id="chwt_wtail16_ttail"),
    pytest.param("CHWT", 2, 2, 32, 40, 40, None, None, id="chwt_wtail_ttail"),
    pytest.param("CHWT", 2, 2, 32, 32, 8, None, None, id="chwt_t_under_tile"),
    pytest.param("CHWT", 4, 8, 16, 32, 33, None, None, id="chwt_4x8_mesh"),
    pytest.param("CHWT", 2, 2, 32, 32, 32, 50, 54, id="chwt_crop"),
    pytest.param("CHWT", 2, 4, 32, 32, 32, 40, 34, id="chwt_crop_drops_shard_col"),
    pytest.param("CTHW", 2, 2, 32, 32, 32, None, None, id="cthw_clean_tiles"),
    pytest.param("CTHW", 2, 2, 24, 40, 17, None, None, id="cthw_ragged"),
    pytest.param("CTHW", 2, 2, 32, 32, 16, 50, 54, id="cthw_crop"),
]


@pytest.mark.skipif(not HAS_CPP_PLANAR_CONCAT, reason="planar concat extension not built (models/tt_dit/utils/cpp)")
@pytest.mark.parametrize("dim_order, TP, SP, h_per, w_per, T, crop_h, crop_w", _CASES)
def test_planar_concat_matches_reference(dim_order, TP, SP, h_per, w_per, T, crop_h, crop_w):
    rng = np.random.default_rng(0xC0FFEE)
    n = TP * SP
    y = _make_shards(rng, n, h_per, w_per, T, dim_order)
    u = _make_shards(rng, n, h_per // 2, w_per // 2, T, dim_order)
    v = _make_shards(rng, n, h_per // 2, w_per // 2, T, dim_order)

    out_H = h_per * TP if crop_h is None else crop_h
    out_W = w_per * SP if crop_w is None else crop_w
    expected = _reference_planar_concat(y, u, v, dim_order, (TP, SP), out_H, out_W, T)

    got = np.full(expected.shape, _SENTINEL, dtype=np.uint8)
    planar_concat_cpp(y, u, v, dim_order, (TP, SP), out=got, out_H=out_H, out_W=out_W)

    assert np.array_equal(got, expected), f"{int((got != expected).sum())} bytes differ from the scatter reference"


@pytest.mark.skipif(not HAS_CPP_PLANAR_CONCAT, reason="planar concat extension not built (models/tt_dit/utils/cpp)")
@pytest.mark.parametrize("crop_h, crop_w", [(None, None), (50, 54)], ids=["full", "cropped"])
def test_planar_concat_writes_every_output_byte(crop_h, crop_w):
    """Two different fills must produce identical output; any byte that differs was never written.

    Comparing against the reference alone cannot catch this — both leave the same gaps.
    """
    rng = np.random.default_rng(11)
    TP, SP, h_per, w_per, T = 2, 2, 34, 34, 9
    y = _make_shards(rng, TP * SP, h_per, w_per, T, "CHWT")
    u = _make_shards(rng, TP * SP, h_per // 2, w_per // 2, T, "CHWT")
    v = _make_shards(rng, TP * SP, h_per // 2, w_per // 2, T, "CHWT")

    out_H = h_per * TP if crop_h is None else crop_h
    out_W = w_per * SP if crop_w is None else crop_w
    row = out_H * out_W + 2 * (out_H // 2) * (out_W // 2)

    outs = []
    for fill in (0xAA, 0x55):
        buf = np.full((T, row), fill, dtype=np.uint8)
        planar_concat_cpp(y, u, v, "CHWT", (TP, SP), out=buf, out_H=out_H, out_W=out_W)
        outs.append(buf)

    assert np.array_equal(outs[0], outs[1]), f"{int((outs[0] != outs[1]).sum())} output bytes were never written"


class _ShapeOnly:
    """Stands in for a ttnn tensor / mesh device: the crop check runs before any device call."""

    def __init__(self, shape):
        self.shape = shape


@pytest.mark.parametrize(
    "logical_h, logical_w",
    [(63, 64), (64, 63), (66, 64), (64, 66), (0, 64), (-2, 64), (64, 0)],
    ids=["odd_h", "odd_w", "h_over_H", "w_over_W", "zero_h", "negative_h", "zero_w"],
)
def test_fast_device_to_host_yuv_rejects_invalid_crop(logical_h, logical_w, expect_error):
    """An oversized crop leaves the tail of the buffer uninitialized and an odd one desyncs
    the chroma planes from the luma plane; both must fail before any device work."""
    with expect_error(ValueError, "logical_"):
        fast_device_to_host_yuv(
            _ShapeOnly((1, 3, 8, 32, 32)),
            _ShapeOnly((2, 2)),
            coefficients=object(),  # skip the ttnn BT.601 default
            logical_h=logical_h,
            logical_w=logical_w,
        )
