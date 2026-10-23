# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Runtime helpers for the Qwen3-VL graph-capture-derived per-op suite (``tests/qwen3_vl_ops/``).

``graph_case.py`` is model-agnostic except for this module, which supplies the four
things it needs: the ``ttnn_mesh_device`` parametrization helper, torch<->ttnn
conversion, random tensor data, and the PCC assertion.

Following the sibling ``tests/yolo_ops/op_utils.py``, these are re-exported from the
llama suite rather than duplicated — they are arch- and model-agnostic (mesh
composition, PCC), and keeping one copy means a fidelity fix reaches every suite.

The ``ttnn_mesh_device`` fixture itself comes from ``tests/conftest.py`` one level up,
and ``reset_seeds`` from the repo-root ``conftest.py``.
"""

from __future__ import annotations

import pytest  # noqa: F401  (re-exported convenience for generated op files)
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
    with_default_mesh,
)

# Qwen3-VL-4B-Instruct, the model this suite's capture was taken from. Present so a
# generated case's shapes can be read against the config that produced them; nothing
# here is used to build a case (every number in a case comes from the capture).
HF_MODEL = "Qwen/Qwen3-VL-4B-Instruct"

DIM = 2560  # text hidden_size
N_LAYERS = 36  # text decoder layers
N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE = 9728  # text MLP hidden
VOCAB = 151936

VISION_DIM = 1024  # vision hidden_size
VISION_DEPTH = 24  # vision blocks
VISION_HEADS = 16
VISION_INTERMEDIATE = 4096
VISION_OUT_DIM = 2560  # patch-merger output width (== text DIM)
DEEPSTACK_INDEXES = (5, 11, 17)  # vision blocks that feed the deepstack mergers
