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
from the llama suite rather than duplicated. Two things differ. ``assert_pcc`` also checks
the output's magnitude (see its docstring). ``with_default_mesh`` opens the mesh with
L1_SMALL: the convolution / pooling / upsample ops this model runs allocate their halo and
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

import ttnn

from models.experimental.llama32_1b_quasar.tests.ops.op_utils import (  # noqa: F401
    DEFAULT_MESH,
    assert_pcc as _assert_pcc_only,
    assert_shape_dtype,
    comp_pcc,
    from_tt,
    to_tt,
    torch_rand,
)
from models.experimental.ops.quasar.panoptic_deeplab.tt.common import PDL_L1_SMALL_SIZE
from models.experimental.ops.quasar.tests.yolo_ops.op_utils import assert_lossless  # noqa: F401


def assert_pcc(torch_ref: torch.Tensor, tt_out: ttnn.Tensor = None, *, pcc=0.99, mesh_device=None, host=None):
    """The llama suite's ``assert_pcc``, plus a check that the output has the reference's magnitude.

    PCC is blind to scale: ``ttnn.multiply(x, 4)`` returning ``x`` unchanged scores 1.0.
    So the L2 norms of the compared elements must also agree, to within ``2 * (1 - pcc)``
    (at least 2%). An op that is correct up to its precision floor stays well inside that,
    since quantization noise only shifts the norm by about ``1 / pcc``. A wrong scalar or a
    dropped scale does not.
    """
    got = host if host is not None else from_tt(tt_out, mesh_device)
    if host is None:
        # Keep the strict length check the shared helper applies to a full readback.
        assert got.numel() >= torch_ref.numel(), f"output has {got.numel()} elements, expected >= {torch_ref.numel()}"
    _assert_pcc_only(torch_ref, pcc=pcc, host=got)

    n = min(got.numel(), torch_ref.numel())
    ref_flat, got_flat = torch_ref.float().reshape(-1)[:n], got.float().reshape(-1)[:n]
    keep = torch.isfinite(ref_flat)
    ref_norm, got_norm = ref_flat[keep].norm().item(), got_flat[keep].norm().item()
    tol = max(0.02, 2 * (1 - pcc))
    if ref_norm == 0:
        assert got_norm == 0, f"reference is all zeros but the output has L2 norm {got_norm:.4g}"
    else:
        ratio = got_norm / ref_norm
        assert abs(ratio - 1) <= tol, (
            f"output L2 norm is {ratio:.4f}x the reference's ({got_norm:.4g} vs {ref_norm:.4g}), "
            f"outside 1 +/- {tol:.3f}: the values correlate but have the wrong magnitude"
        )


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
