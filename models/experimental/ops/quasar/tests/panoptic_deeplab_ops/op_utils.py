# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Runtime helpers for the Panoptic-DeepLab graph-capture-derived per-op suite
(``tests/panoptic_deeplab_ops/``).

``graph_case.py`` is model-agnostic except for this module, which supplies the things
it needs: the ``ttnn_mesh_device`` parametrization helper, torch<->ttnn conversion,
random tensor data, the PCC and bit-exact assertions, and the compute configs the
capture cannot record (``default_kwargs``).

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

import os

import pytest
import torch  # noqa: F401

import ttnn

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
from models.experimental.ops.quasar.tests.yolo_ops.op_utils import assert_lossless  # noqa: F401


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


# The capture records a compute kernel config only as an object address, so the generator
# drops it (``DROP_KWARGS``) and the op would run at its default fidelity. Every conv in the
# model is built from the same defaults — ``panoptic_deeplab/tt/model_configs.py``
# ``default_overrides`` (LoFi, fp32 dest accumulation, no packer L1 accumulation), turned into
# a compute config by ``models/tt_cnn/tt/builder.py:to_compute_config`` — so restore those.
# Not restored: the per-layer ``packer_l1_acc=True`` the model sets on its deep 1x1 bottleneck
# and ASPP convs (``register_layer_override``), since a case does not say which layer it was.
def _conv_compute_config(mesh_device):
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


_DEFAULT_KWARGS = {"ttnn.conv2d": {"compute_config": _conv_compute_config}}


def default_kwargs(op_name, mesh_device):
    """Keyword arguments the model passed to ``op_name`` that the capture could not record."""
    return {name: build(mesh_device) for name, build in _DEFAULT_KWARGS.get(op_name, {}).items()}


# Ops whose mainline implementation fails on Quasar with "DataMovementKernel is not supported on
# Quasar" (legacy ProgramDescriptor ops) and whose Metal 2.0 copy under ``ttnn.experimental.quasar``
# takes the same arguments — measured on craq-sim 8x4 with this suite. The same routing as
# ``tests/yolo_ops/op_utils.py`` (#60455). Only ops that fail on mainline are routed, so the suite
# keeps reporting which mainline ops work on Quasar; everything else runs the mainline op.
# ``TTNN_GRAPH_OPS_QUASAR_FORKS=1|0`` forces either choice (the forks also run on WH/BH).
_QUASAR_FORKED_OPS = ("ttnn.conv2d", "ttnn.add", "ttnn.multiply", "ttnn.max_pool2d", "ttnn.avg_pool2d", "ttnn.move")


def quasar_forks_enabled() -> bool:
    v = os.environ.get("TTNN_GRAPH_OPS_QUASAR_FORKS", "auto").lower()
    if v in ("1", "true", "yes"):
        return True
    if v in ("0", "false", "no"):
        return False
    return "quasar" in ttnn.get_arch_name()


def quasar_routed(op_name, op):
    """``ttnn.experimental.quasar``'s copy of ``op`` when it is routed (see above), else ``op``."""
    if op_name not in _QUASAR_FORKED_OPS or not quasar_forks_enabled():
        return op
    return getattr(ttnn.experimental.quasar, op_name.removeprefix("ttnn."))
