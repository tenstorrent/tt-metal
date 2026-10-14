# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Runtime helpers for the Panoptic-DeepLab graph-capture-derived per-op suite
(``tests/panoptic_deeplab_ops/``).

``graph_case.py`` is model-agnostic except for this module, which supplies the four
things it needs: the ``ttnn_mesh_device`` parametrization helper, torch<->ttnn
conversion, random tensor data, and the PCC assertion.

As in ``tests/qwen3_vl_ops/op_utils.py``, the conversion and PCC helpers are re-exported
from the llama suite rather than duplicated. ``with_default_mesh`` is the one thing that
differs: the convolution / pooling / upsample ops this model runs allocate their halo and
sliding-window config buffers from the L1_SMALL region, which a mesh opened with the
default ``l1_small_size`` of 0 does not have. The capture's run opened its device with
``PDL_L1_SMALL_SIZE``, so every case here does the same.

The ``ttnn_mesh_device`` fixture itself comes from ``tests/conftest.py`` one level up
(it forwards every key of a dict param but ``mesh_shape`` to ``open_mesh_device``), and
``reset_seeds`` from the repo-root ``conftest.py``.
"""

from __future__ import annotations

import pytest
import torch  # noqa: F401

import ttnn  # noqa: F401

from models.experimental.llama32_1b_quasar.tests.ops.op_utils import (  # noqa: F401
    DEFAULT_MESH,
    assert_pcc,
    assert_shape_dtype,
    comp_pcc,
    from_tt,
    to_tt,
    torch_rand,
)
from models.experimental.ops.quasar.panoptic_deeplab.tt.common import PDL_L1_SMALL_SIZE


def with_default_mesh(*shapes):
    """Parametrize the ``ttnn_mesh_device`` fixture (indirect) with PDL's L1_SMALL size."""
    params = [{"mesh_shape": shape, "l1_small_size": PDL_L1_SMALL_SIZE} for shape in (shapes or [DEFAULT_MESH])]
    return pytest.mark.parametrize("ttnn_mesh_device", params, indirect=True)


# Panoptic-DeepLab (Detectron2 R-52, Cityscapes), the model this suite's capture was taken
# from. Present so a generated case's shapes can be read against the config that produced
# them; nothing here is used to build a case (every number in a case comes from the capture).
INPUT_HW = (512, 1024)  # Cityscapes train size; the stem sees 512x1024x3
NUM_CLASSES = 19
BACKBONE_CHANNELS = {"res2": 256, "res3": 512, "res4": 1024, "res5": 2048}
DECODER_CHANNELS = 256
INS_EMBED_HEAD_CHANNELS = 32
