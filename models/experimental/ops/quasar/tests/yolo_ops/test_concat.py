# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.concat``  (YOLOv8 backbone / neck / detect head).

Model call sites (many). The fully-determinable ones live in the detect head and
dist2bbox; the neck/backbone channel-concats depend on per-block conv widths that
are not recoverable from the read-only source without tracing every conv, so one
representative channel-concat is included and marked uncertain inline.

Detect-head concats (identical for yolov8l & yolov8s):
  * cv2/cv3 join, dim=3   ttnn_yolov8l.py:734 / ttnn_yolov8s.py:555
        x[i] = ttnn.concat((a, b), dim=3)   # a=[.,.,hw^2,64], b=[.,.,hw^2,80] -> 144
        (64 = reg_max*4, 80 = nc; no = 144 at ttnn_yolov8l.py:729)
  * per-scale flatten join, dim=1   ttnn_yolov8l.py:744 / ttnn_yolov8s.py:565
        x_cat = ttnn.concat(xi, 1)          # [1,6400,144]+[1,1600,144]+[1,400,144]
  * final output join, dim=1   ttnn_yolov8l.py:757 / ttnn_yolov8s.py:578
        ttnn.concat((dbox, sigmoid(cls)), dim=1)   # [1,4,A] + [1,80,A] -> [1,84,A]

dist2bbox concat, dim=1   tt_yolov8l_utils.py:100 / tt_yolov8s_utils.py:98
        ttnn.concat([c_xy, wh], 1)          # [1,2,A] + [1,2,A] -> [1,4,A]

Neck channel-concats, dim=-1   ttnn_yolov8l.py:494/499 (C2f) and sharded_concat*
(ttnn_yolov8l.py:1034/1074/1087/1102, SPPF :542): channel widths model-specific.

A = total detect anchors: 8400 @ 640 (80/40/20), 33600 @ 1280 (160/80/40).
Reference: torch.cat; concat is value-preserving so we require exact equality (assert_lossless).
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U

_NC = 80
_NO = 144  # nc + reg_max*4 = 80 + 64


# ---------------------------------------------------------------------------
# Spatial detect cv2/cv3 concat (dim=3): [1,1,hw*hw,64] + [1,1,hw*hw,80] -> 144.
# hw is the feature-map side -> int `hw` param so conftest classifies (hw<=40 fits).
# yolov8l/yolov8s @ 640 -> hw in {80,40,20}; yolov8l @ 1280 -> hw in {160,80,40}.
# ---------------------------------------------------------------------------
@U.with_default_mesh()
@pytest.mark.parametrize("hw", [80, 40, 20, 160], ids=["hw80", "hw40", "hw20", "hw160-l1280"])
def test_concat_detect_cv2cv3(ttnn_mesh_device, reset_seeds, hw):
    """cv2/cv3 join per scale, dim=3 (ttnn_yolov8l.py:734)."""
    mesh = ttnn_mesh_device
    hh = hw * hw

    a_torch = U.torch_rand((1, 1, hh, 64))  # cv2 (box) -> reg_max*4
    b_torch = U.torch_rand((1, 1, hh, _NC))  # cv3 (cls) -> nc
    a = U.to_tt(a_torch, mesh)
    b = U.to_tt(b_torch, mesh)

    out = ttnn.concat([a, b], dim=3)

    ref = torch.cat([a_torch.float(), b_torch.float()], dim=3)
    U.assert_lossless(ref, out, mesh_device=mesh)


# ---------------------------------------------------------------------------
# Channel-wise / anchor-wise concats (non-spatial). `out_shape` tuple param so the
# conftest classifies by element count.  (id, dim, in_shapes, out_shape)
# ---------------------------------------------------------------------------
_CHANNEL_SITES = [
    # per-scale flatten join, dim=1 (ttnn_yolov8l.py:744) @ 640
    ("detect_xi-640", 1, ((1, 6400, _NO), (1, 1600, _NO), (1, 400, _NO)), (1, 8400, _NO)),
    # per-scale flatten join @ 1280 (yolov8l only)
    ("detect_xi-1280", 1, ((1, 25600, _NO), (1, 6400, _NO), (1, 1600, _NO)), (1, 33600, _NO)),
    # dist2bbox c_xy/wh join, dim=1 (tt_yolov8l_utils.py:100) @ 640
    ("dist2bbox-640", 1, ((1, 2, 8400), (1, 2, 8400)), (1, 4, 8400)),
    ("dist2bbox-1280", 1, ((1, 2, 33600), (1, 2, 33600)), (1, 4, 33600)),
    # final detect output join dbox+sigmoid(cls), dim=1 (ttnn_yolov8l.py:757) @ 640
    ("detect_out-640", 1, ((1, 4, 8400), (1, _NC, 8400)), (1, 84, 8400)),
    ("detect_out-1280", 1, ((1, 4, 33600), (1, _NC, 33600)), (1, 84, 33600)),
    # YOLOv8l model.15 input: upsampled 512-channel tensor + 256-channel skip -> 768.
    ("neck_channel-640", -1, ((1, 1, 6400, 512), (1, 1, 6400, 256)), (1, 1, 6400, 768)),
]


@U.with_default_mesh()
@pytest.mark.parametrize(
    "name, dim, in_shapes, out_shape",
    [pytest.param(*s, id=s[0]) for s in _CHANNEL_SITES],
)
def test_concat_channel(ttnn_mesh_device, reset_seeds, name, dim, in_shapes, out_shape):
    mesh = ttnn_mesh_device

    parts_torch = [U.torch_rand(s) for s in in_shapes]
    parts_tt = [U.to_tt(p, mesh) for p in parts_torch]

    out = ttnn.concat(parts_tt, dim=dim)

    ref = torch.cat([p.float() for p in parts_torch], dim=dim)
    assert tuple(ref.shape) == tuple(out_shape), f"{name}: torch.cat -> {tuple(ref.shape)} != {out_shape}"
    U.assert_lossless(ref, out, mesh_device=mesh)


# ---------------------------------------------------------------------------
# Model-faithful (class C): detect-head concats on L1-interleaved TILE inputs.
# All detect-head concats run on _DETECT_MEM_CONFIG = L1_MEMORY_CONFIG for res <= 640
# (ttnn_yolov8l.py:772-773): detect_xi join (:744), dist2bbox c_xy/wh (tt_yolov8l_utils.py:100),
# final detect_out (:757). The cv2/cv3 join (:734) consumes detect-cv2 outputs that
# are sharded_to_interleaved into L1 (ttnn_yolov8l.py:261-265), so it is L1-fed even
# at 1280. Same shapes as the DRAM detect cases, input memory_config DRAM -> L1.
# (id, dim, in_shapes, out_shape)
# ---------------------------------------------------------------------------
_DETECT_L1_SITES = [
    # cv2/cv3 spatial join, dim=3 (ttnn_yolov8l.py:734); L1-fed via :261-265 even @ 1280
    ("cv2cv3-640", 3, ((1, 1, 6400, 64), (1, 1, 6400, _NC)), (1, 1, 6400, _NO)),  # hw=80
    ("cv2cv3-1280", 3, ((1, 1, 25600, 64), (1, 1, 25600, _NC)), (1, 1, 25600, _NO)),  # hw=160
    # per-scale flatten join, dim=1 (ttnn_yolov8l.py:744) @ 640
    ("detect_xi-640", 1, ((1, 6400, _NO), (1, 1600, _NO), (1, 400, _NO)), (1, 8400, _NO)),
    # dist2bbox c_xy/wh join, dim=1 (tt_yolov8l_utils.py:100) @ 640
    ("dist2bbox-640", 1, ((1, 2, 8400), (1, 2, 8400)), (1, 4, 8400)),
    # final detect output join dbox+sigmoid(cls), dim=1 (ttnn_yolov8l.py:757) @ 640
    ("detect_out-640", 1, ((1, 4, 8400), (1, _NC, 8400)), (1, 84, 8400)),
]


@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize(
    "name, dim, in_shapes, out_shape",
    [pytest.param(*s, id=s[0]) for s in _DETECT_L1_SITES],
)
def test_concat_detect_l1(ttnn_mesh_device, reset_seeds, name, dim, in_shapes, out_shape):
    """detect-head concats on L1 buffers (ttnn_yolov8l.py:734/744/757, tt_yolov8l_utils.py:100; mem cfg :772-773)."""
    mesh = ttnn_mesh_device

    parts_torch = [U.torch_rand(s) for s in in_shapes]
    parts_tt = [U.to_tile_l1(p, mesh) for p in parts_torch]

    out = ttnn.concat(parts_tt, dim=dim)

    ref = torch.cat([p.float() for p in parts_torch], dim=dim)
    assert tuple(ref.shape) == tuple(out_shape), f"{name}: torch.cat -> {tuple(ref.shape)} != {out_shape}"
    U.assert_lossless(ref, out, mesh_device=mesh)


# ---------------------------------------------------------------------------
# Model-faithful (structural): neck channel-concat on HEIGHT_SHARDED L1 operands.
# sharded_concat (ttnn_yolov8l.py:80-93, def at :47-64 for the SPPF variant), called
# at ttnn_yolov8l.py:1034/1074/1087/1102, shards each operand HEIGHT-wise over the
# _SPPF_CORE_GRID (64 cores WH / 80 cores BH, ttnn_yolov8l.py:777-781) in L1 with
# shard shape (shard_height, C), then concats along the channel dim (dim=3). We
# rebuild that sharded input state via U.height_sharded_memcfg and concat dim=-1.
# ---------------------------------------------------------------------------
# Neck feature-map [1,1,hw*hw,C]; exact channel widths are model-config specific
# (see the neck_channel note above) so 320/320 mirrors the detect-input widths as a
# stand-in shape only. num_cores=64 = SPPF grid best-estimate (64 WH / 80 BH), fits
# both and divides 6400 evenly (6400/64=100).
@pytest.mark.blackhole_scale
@U.with_default_mesh()
def test_concat_neck_sharded(ttnn_mesh_device, reset_seeds):
    """neck channel-concat on HEIGHT_SHARDED L1 operands (sharded_concat, ttnn_yolov8l.py:1034; def :47-64)."""
    mesh = ttnn_mesh_device
    hw = 80  # neck concat feeding model.15 runs at the 80x80 (P3) level
    # The real neck concat is UNEQUAL width: upsample(512) + backbone skip(256) -> 768
    # (model.15.cv1 in_ch = 768; branch configs.json c2f model.15 cv1 = [1,1,0,256,768]).
    Ca, Cb = 512, 256
    rows = hw * hw
    num_cores = 64  # _SPPF_CORE_GRID best-estimate (64 WH / 80 BH); 6400 % 64 == 0

    a_torch = U.torch_rand((1, 1, rows, Ca))
    b_torch = U.torch_rand((1, 1, rows, Cb))
    # ROW_MAJOR HEIGHT-sharded (neck feature maps are row-major, so the shard height need
    # not be tile-aligned); each operand keeps its own channel width and the concat is on
    # the non-sharded channel dim -> 512 + 256 = 768.
    a = U.to_tt(
        a_torch,
        mesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=U.height_sharded_memcfg(mesh, num_cores, (1, 1, rows, Ca)),
    )
    b = U.to_tt(
        b_torch,
        mesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=U.height_sharded_memcfg(mesh, num_cores, (1, 1, rows, Cb)),
    )

    out = ttnn.concat([a, b], dim=-1)

    ref = torch.cat([a_torch.float(), b_torch.float()], dim=-1)
    assert tuple(ref.shape) == (1, 1, rows, Ca + Cb), f"neck concat -> {tuple(ref.shape)}"
    U.assert_lossless(ref, out, mesh_device=mesh)
