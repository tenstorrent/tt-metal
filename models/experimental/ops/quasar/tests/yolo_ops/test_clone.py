# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.clone`` (YOLOv8, 640x640, batch 1).

The neck saves bf16 copies of the C2f outputs (twelve/fifteen/eighteen) for the
skip connections that feed the detect head:

    twelve = ttnn.clone(c2f_12, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)

``ttnn.clone`` is a value-preserving copy (here a bf16->bf16 copy into L1), so the
cloned tensor must equal the input; PCC 0.999.

Model call sites (branch ``origin/sdawle/yolov8_bh``):
  * models/demos/yolov8l/tt/ttnn_yolov8l.py:1044 (twelve), :1082 (fifteen), :1096 (eighteen)
  * models/demos/yolov8s/tt/ttnn_yolov8s.py:776 (twelve), :803 (fifteen), :817 (eighteen)

The tensors are flattened neck feature maps [1, 1, H*W, C]. H==W is the spatial
size (parametrized by ``hw``: 20/40 fit the emulator, 80 runs on real Blackhole).
Neck channel widths are the standard YOLOv8l/s neck sizes; clone is
channel-independent so the exact value is not load-bearing.
"""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U

# (hw, C, id) — neck feature-map resolution/channels for yolov8l (l_*) and yolov8s (s_*).
_FMAPS = [
    (20, 512, "l_p5_20x512"),
    (40, 512, "l_p4_40x512"),
    (80, 256, "l_p3_80x256"),
    (20, 256, "s_p5_20x256"),
    (40, 256, "s_p4_40x256"),
    (80, 128, "s_p3_80x128"),
]


@U.with_default_mesh()
@pytest.mark.parametrize("hw, c", [pytest.param(hw, c, id=i) for (hw, c, i) in _FMAPS])
def test_clone(ttnn_mesh_device, reset_seeds, hw, c):
    mesh = ttnn_mesh_device

    x_torch = U.torch_rand((1, 1, hw * hw, c))
    x = U.to_tt(x_torch, mesh)  # TILE, bf16, DRAM interleaved

    out = ttnn.clone(x, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)

    U.assert_lossless(x_torch, out, mesh_device=mesh)


# =============================================================================
# Model-faithful variant: reproduce the model's REAL layout / buffer state.
# =============================================================================


@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize("hw, c", [pytest.param(hw, c, id=i + "_rm_l1") for (hw, c, i) in _FMAPS])
def test_clone_rm_l1(ttnn_mesh_device, reset_seeds, hw, c):
    """Model-faithful ``ttnn.clone`` — ROW_MAJOR input in L1 (the model's real state).

    In the neck, ``c2f_12`` is brought to ROW_MAJOR in L1 (``sharded_to_interleaved``
    into ``ttnn.L1_MEMORY_CONFIG`` then ``to_row_major_if_needed``) immediately before
    it is cloned for the skip connection:

        c2f_12 = ttnn.sharded_to_interleaved(c2f_12, ttnn.L1_MEMORY_CONFIG)
        c2f_12 = to_row_major_if_needed(c2f_12)
        twelve = ttnn.clone(c2f_12, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)

    So the clone's input is ROW_MAJOR_LAYOUT in L1 (not the TILE/DRAM state the
    existing ``test_clone`` uses). Clone is a value-preserving bf16->bf16 L1 copy;
    the round-trip must equal the input at PCC 0.999.

    Model call sites (branch ``origin/sdawle/yolov8_bh``):
      * models/demos/yolov8l/tt/ttnn_yolov8l.py:1044 (twelve)
        (same pattern at :1082 fifteen, :1096 eighteen; yolov8s.py:776 / :803 / :817)
    """
    mesh = ttnn_mesh_device

    x_torch = U.torch_rand((1, 1, hw * hw, c))
    # ROW_MAJOR, bf16, in L1 — the model's exact clone-input buffer state.
    x = U.to_tt(x_torch, mesh, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

    out = ttnn.clone(x, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)

    U.assert_lossless(x_torch, out, mesh_device=mesh)
