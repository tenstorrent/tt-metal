# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Runtime helpers for the GPT-OSS graph-capture-derived per-op suite (``tests/gpt_oss_ops/``).

``graph_case.py`` is model-agnostic except for this module, which supplies the four things it
needs: the ``ttnn_mesh_device`` parametrization helper, torch<->ttnn conversion, random tensor
data, and the PCC assertion.

As in the sibling ``tests/qwen3_vl_ops`` and ``tests/yolo_ops`` suites, these are re-exported from
the llama op suite rather than duplicated -- they are arch- and model-agnostic (mesh composition,
PCC), so one copy means a fidelity fix reaches every suite.

The ``ttnn_mesh_device`` fixture itself comes from ``tests/conftest.py`` one level up, and
``reset_seeds`` from the repo-root ``conftest.py``.
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

# gpt-oss-20b, the model this suite's capture was taken from (configs/gpt-oss-20b/config.json).
# Present so a generated case's shapes can be read against the config that produced them; nothing
# here is used to build a case -- every number in a case comes from the capture.
HF_MODEL = "gpt-oss-20b"

DIM = 2880  # hidden_size
N_LAYERS = 24
N_HEADS = 64
N_KV_HEADS = 8
HEAD_DIM = 64
INTERMEDIATE = 2880  # per-expert MLP width
VOCAB = 201088

NUM_EXPERTS = 32  # num_local_experts
EXPERTS_PER_TOK = 4  # top-k routing
