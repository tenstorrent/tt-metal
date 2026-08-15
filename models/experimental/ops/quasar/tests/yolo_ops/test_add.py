# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.add``  (YOLOv8 dist2bbox math + bottleneck residual).

Model call sites:
  * dist2bbox (tt_yolov8l_utils.py:95,97 / tt_yolov8s_utils.py:93,95)
        x2y2 = anchor_points + rb     # [1,2,A] + [1,2,A]  (ttnn.add via `+`)
        c_xy = x1y1 + x2y2            # [1,2,A] + [1,2,A]
    ``lt, rb = ttnn.split(distance, 2, 1)`` -> each [1,2,A]; ``anchor_points`` is
    make_anchors' ``a`` = [1,2,A] (tt_yolov8l_utils.py:80,84).
  * bottleneck residual add (ttnn_yolov8l.py:330 / ttnn_yolov8s.py:330)
        return ttnn.add(x, cv2) if self.shortcut else cv2   # spatial [1,1,hw^2,C]

A = total detect anchors: 8400 @ 640, 33600 @ 1280 (yolov8l only). Detect head is
identical for yolov8l/yolov8s. Model runs these in bfloat8_b; we use the default
bfloat16 to exercise the op cleanly. Reference: torch a + b.  PCC 0.999.
"""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U


# dist2bbox adds operate on [1, 2, A] operands. shape tuple param -> element-count
# classification (both are small enough to fit the emulator at 640).
_ANCHOR_SITES = [
    pytest.param((1, 2, 8400), id="dist2bbox-640"),  # yolov8l & yolov8s
    pytest.param((1, 2, 33600), id="dist2bbox-1280"),  # yolov8l only
]


@U.with_default_mesh()
@pytest.mark.parametrize("shape", _ANCHOR_SITES)
def test_add_dist2bbox(ttnn_mesh_device, reset_seeds, shape):
    """anchor+rb / x1y1+x2y2 (tt_yolov8l_utils.py:95,97)."""
    mesh = ttnn_mesh_device

    a_torch = U.torch_rand(shape)
    b_torch = U.torch_rand(shape)
    a = U.to_tt(a_torch, mesh)
    b = U.to_tt(b_torch, mesh)

    out = ttnn.add(a, b)

    ref = a_torch.float() + b_torch.float()
    U.assert_pcc(ref, out, pcc=0.999, mesh_device=mesh)


# ---------------------------------------------------------------------------
# Model-faithful (class C): dist2bbox adds on L1-interleaved TILE inputs. dist2bbox
# runs inside the detect head; anchor_points/rb/x1y1/x2y2 live on
# _DETECT_MEM_CONFIG = L1_MEMORY_CONFIG for res <= 640 (ttnn_yolov8l.py:772-773).
# Same shapes as test_add_dist2bbox, DRAM -> L1 input only.
# ---------------------------------------------------------------------------
@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize("shape", _ANCHOR_SITES)
def test_add_dist2bbox_l1(ttnn_mesh_device, reset_seeds, shape):
    """anchor+rb / x1y1+x2y2 on L1 detect buffers (tt_yolov8l_utils.py:95,97; mem cfg ttnn_yolov8l.py:772-773)."""
    mesh = ttnn_mesh_device

    a_torch = U.torch_rand(shape)
    b_torch = U.torch_rand(shape)
    a = U.to_tile_l1(a_torch, mesh)
    b = U.to_tile_l1(b_torch, mesh)

    out = ttnn.add(a, b)

    ref = a_torch.float() + b_torch.float()
    U.assert_pcc(ref, out, pcc=0.999, mesh_device=mesh)


# ---------------------------------------------------------------------------
# Model-faithful (structural): bottleneck residual add on HEIGHT_SHARDED L1 operands.
# ttnn_yolov8l.py:330  return ttnn.add(x, cv2) if self.shortcut else cv2
# Both operands are regular-conv outputs produced with memory_config=None
# (ttnn_yolov8l.py:255) => the conv leaves them HEIGHT_SHARDED in L1; the add runs
# on that sharded state (no interleaving in between). We rebuild that state directly
# via U.height_sharded_memcfg at a neck feature-map shape [1,1,hw*hw,C].
# ---------------------------------------------------------------------------
# hw is the feature-map side; C=128 representative (per-block width is model-config
# specific, see test_add_residual note). num_cores mirrors the neck HEIGHT shard
# (SPPF grid is 64 on WH / 80 on BH, ttnn_yolov8l.py:777-781); 64 fits both and
# divides hw*hw evenly at these sizes (6400/64=100, 1600/64=25, 400/64 -> use 80).
@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize("hw", [80, 40], ids=["hw80", "hw40"])
def test_add_residual_sharded(ttnn_mesh_device, reset_seeds, hw):
    """bottleneck shortcut add on HEIGHT_SHARDED L1 conv outputs (ttnn_yolov8l.py:330, conv mem cfg :255)."""
    mesh = ttnn_mesh_device
    C = 128  # representative bottleneck channel width (see test_add_residual note)
    shape = (1, 1, hw * hw, C)

    # Conv outputs are TILE, so the HEIGHT shard height must be tile-aligned (a plain
    # 64-core grid gives shard height 100/25 -> "shard shape must be tile sized").
    memcfg = U.height_sharded_tile_memcfg(mesh, shape)

    a_torch = U.torch_rand(shape)
    b_torch = U.torch_rand(shape)
    a = U.to_tt(a_torch, mesh, memory_config=memcfg)  # TILE (default) — model's conv-output layout
    b = U.to_tt(b_torch, mesh, memory_config=memcfg)

    out = ttnn.add(a, b)

    ref = a_torch.float() + b_torch.float()
    U.assert_pcc(ref, out, pcc=0.999, mesh_device=mesh)


# Bottleneck residual: ttnn.add(x, cv2), spatial [1,1,hw*hw,C]. hw is the feature-map
# side -> int `hw` param (hw<=40 fits emulator). C is a bottleneck channel width; the
# exact per-block value is model-config-specific (NOTE: 128 is a representative
# stand-in), but the op is same-shape elementwise regardless of C.
@U.with_default_mesh()
@pytest.mark.parametrize("hw", [80, 40, 20], ids=["hw80", "hw40", "hw20"])
def test_add_residual(ttnn_mesh_device, reset_seeds, hw):
    """bottleneck shortcut add (ttnn_yolov8l.py:330)."""
    mesh = ttnn_mesh_device
    shape = (1, 1, hw * hw, 128)  # C=128 representative (see note)

    a_torch = U.torch_rand(shape)
    b_torch = U.torch_rand(shape)
    a = U.to_tt(a_torch, mesh)
    b = U.to_tt(b_torch, mesh)

    out = ttnn.add(a, b)

    ref = a_torch.float() + b_torch.float()
    U.assert_pcc(ref, out, pcc=0.999, mesh_device=mesh)
