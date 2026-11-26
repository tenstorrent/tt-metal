# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_scaled_dot_product_attention.py — do NOT
# regenerate this file from a capture.
#
# Both captured cases set SDPAProgramConfig.compute_with_storage_grid_size = [8, 4]
# (32 cores), because the capture ran on an 8x4 device. On a device with a smaller
# compute grid (for example the 2x1 grid of the 2-core Quasar emulator) the op
# FATALs with "Provided grid must not contain more cores than the device".
#
# The gpt-oss model does not use a fixed grid for SDPA: its program configs take
# compute_with_storage_grid_size from get_compute_grid_size()
# (models/experimental/ops/quasar/gpt_oss/tt/attention/config.py), which is the
# device grid limited to 8x8. On a 2x1 device the model therefore passes a 2x1 grid.
#
# Besides the grid, the shapes are reduced so the variants fit the 2-node emulator and its
# ``-m emulator`` subset (128-row cap on the query, which the captured 64-head x 128-token
# query exceeds at 8192 rows). Each variant keeps the captured call (causal, bfloat16 TILE,
# interleaved DRAM, attention sink, chunk sizes 32/32, the case's sliding window) with
# 2 query heads x 64 tokens (128 rows) against one KV head (GQA 2:1) and a sliding window of
# 32 tokens (case 00) so the window actually cuts the 64-token sequence. The sequence spans
# two 32-token q and k chunks. The captured 32-core grid is the generated test's.
#
# graph_case.run_case skips a variant when the device compute grid is smaller than the
# variant's grid rectangle, so the file is safe to run on any device.
# ---------------------------------------------------------------------------
"""Small-grid (2-core) variants of ``scaled_dot_product_attention`` (prefill)."""

import copy

import pytest

import ttnn
from models.experimental.ops.quasar.tests.gpt_oss_ops import graph_case as G
from models.experimental.ops.quasar.tests.gpt_oss_ops.test_scaled_dot_product_attention import CASES as _CAPTURED

_OP = ttnn.experimental.quasar.transformer.scaled_dot_product_attention


_Q_HEADS = 2
_KV_HEADS = 1
_SEQ = 64
_HEAD_DIM = 64
_WINDOW = 32  # replaces the captured 128 (case 00), which is longer than the reduced sequence


def _case_with_grid(base, grid_xy):
    case = copy.deepcopy(base)
    grid_x, grid_y = grid_xy
    case[
        "id"
    ] = f"{base['id'].replace('64x128x64', f'{_Q_HEADS}x{_SEQ}x{_HEAD_DIM}')}-{grid_x * grid_y}core_{grid_x}x{grid_y}"
    case["kwargs"]["program_config"]["fields"]["compute_with_storage_grid_size"] = [grid_x, grid_y]
    q, k, v = case["args"]
    q["shape"] = [1, _Q_HEADS, _SEQ, _HEAD_DIM]
    k["shape"] = [1, _KV_HEADS, _SEQ, _HEAD_DIM]
    v["shape"] = [1, _KV_HEADS, _SEQ, _HEAD_DIM]
    case["kwargs"]["attention_sink"]["shape"] = [1, _Q_HEADS, 1, 1]
    if case["kwargs"]["sliding_window_size"]["v"] is not None:
        case["kwargs"]["sliding_window_size"]["v"] = _WINDOW
    case["outs"][0]["shape"] = [1, _Q_HEADS, _SEQ, _HEAD_DIM]
    return case


CASES = [_case_with_grid(c, (2, 1)) for c in _CAPTURED]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_scaled_dot_product_attention_small_grid(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
