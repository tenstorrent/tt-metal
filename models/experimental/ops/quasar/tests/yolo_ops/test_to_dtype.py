# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.to_dtype`` (YOLOv8, 640x640, batch 1).

The detect head casts decoded boxes to bfloat8_b before the stride multiply:

    # dbox = ttnn.to_dtype(dbox, dtype=ttnn.bfloat8_b)

Note: in the shipped source this exact ``ttnn.to_dtype`` line is commented out — the
model's LIVE bf16->bfloat8_b cast is done via ``ttnn.to_memory_config(..., dtype=bfloat8_b)``
(anchors/strides, ttnn_yolov8l.py:751,754) and ``ttnn.multiply(..., dtype=bfloat8_b)``,
which are covered by ``test_to_memory_config_dtype_cast`` (test_to_memory_config.py) and
``test_multiply``. This file keeps a direct check of the ``ttnn.to_dtype`` op itself for
the model's declared dtype-conversion of ``dbox``.

``ttnn.to_dtype`` is a **host-side** op — it operates on a host tensor (a device tensor
trips ``host_storage != nullptr``), so this test does NOT exercise the device; it just
verifies the host dtype conversion. The cast is value-preserving up to precision, so the
reference is ``x`` compared at PCC 0.99 (bfloat8_b block-float target).

Model call sites (branch ``origin/sdawle/yolov8_bh``):
  * models/demos/yolov8l/tt/ttnn_yolov8l.py:753  (# dbox = ttnn.to_dtype(dbox, dtype=ttnn.bfloat8_b))
  * models/demos/yolov8s/tt/ttnn_yolov8s.py:574  (same, commented)
  * live equivalents: ttnn_yolov8l.py:751,754 to_memory_config(..., dtype=bfloat8_b);
    :1044 ttnn.clone(..., dtype=bfloat16)

``dbox`` is (1, 4, 8400) at 640x640 (4 box coords over 8400 anchors); a neck
feature-map case is also covered.
"""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U

# (id, shape) — bf16 -> bf8 dtype cast targets.
_CASES = [
    ("dbox_1x4x8400", (1, 4, 8400)),  # ttnn_yolov8l.py:753 / ttnn_yolov8s.py:574 (dbox)
    ("fmap_1x1x400x512", (1, 1, 400, 512)),  # neck fmap (P5, C=512); clone/memcfg dtype casts
]


@U.with_default_mesh()
@pytest.mark.parametrize("name, shape", [pytest.param(*c, id=c[0]) for c in _CASES])
def test_to_dtype_bf16_to_bf8(ttnn_mesh_device, reset_seeds, name, shape):
    mesh = ttnn_mesh_device

    x_torch = U.torch_rand(shape)
    # ttnn.to_dtype operates on a HOST tensor — a device tensor trips "host_storage != nullptr".
    x = ttnn.from_torch(
        x_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,  # bfloat8_b requires TILE
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh),
    )

    out = ttnn.to_dtype(x, ttnn.bfloat8_b)
    assert out.dtype == ttnn.bfloat8_b, f"expected bfloat8_b, got {out.dtype}"

    U.assert_pcc(x_torch, out, pcc=0.99, mesh_device=mesh)  # 0.99: lower-precision target
