# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_matmul.py (and test_matmul_small_grid.py) — do NOT
# regenerate this file from a capture.
#
# The captured gpt-oss calls are ``ttnn.matmul``. On Quasar there are two matmuls: the base op, which
# has no MX support, and ``ttnn.experimental.quasar.matmul``, which d2d7db4 made MX-ready (interm
# formats, the 1056 B MX tile stride in the readers, mm_uninit). These tests run the same cases
# through the Quasar copy so both matmuls have coverage:
#
#   * test_matmul / test_matmul_small_grid          base ``ttnn.matmul``; under
#                                                   TTNN_GRAPH_OPS_BF8_AS_MXFP8R/P the cases that
#                                                   build an MX tensor skip (graph_case._NO_MX_OPS);
#   * test_matmul_quasar / test_matmul_small_grid_quasar
#                                                   ``ttnn.experimental.quasar.matmul`` on every case,
#                                                   MX or bfloat16.
# ---------------------------------------------------------------------------
"""``ttnn.experimental.quasar.matmul`` on the captured gpt-oss ``ttnn.matmul`` calls."""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.gpt_oss_ops import graph_case as G
from models.experimental.ops.quasar.tests.gpt_oss_ops import test_matmul as T
from models.experimental.ops.quasar.tests.gpt_oss_ops import test_matmul_small_grid as TS

_OP = ttnn.experimental.quasar.matmul


@G.with_default_mesh()
@pytest.mark.parametrize("case", T.CASES, ids=[c["id"] for c in T.CASES])
def test_matmul_quasar(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)


@G.with_default_mesh()
@pytest.mark.parametrize("case", TS.CASES, ids=[c["id"] for c in TS.CASES])
def test_matmul_small_grid_quasar(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
