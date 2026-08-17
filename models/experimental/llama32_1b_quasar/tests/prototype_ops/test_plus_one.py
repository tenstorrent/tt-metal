# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.plus_one``.

Model call site (models/llama32_1b/model.py):
  * L991  increment_positions — ttnn.plus_one(current_pos, skip_negative_entries=True)
  * L992  increment_positions — ttnn.plus_one(rot_mat_idxs)

``plus_one`` increments an int position tensor in place (no return value). It is
used to advance ``current_pos`` / ``rot_mat_idxs`` by one decode step. With
``skip_negative_entries=True`` padded/negative slots are left untouched (padding
slots in a batched decode carry -1). Positions are one int per decode user, so
the tensor is [batch] (ROW_MAJOR, int32). Reference is ``x + 1`` (with negatives
preserved when skip_negative_entries is set).
"""

import pytest
import torch

import ttnn
from models.experimental.llama32_1b_quasar.tests.prototype_ops import op_utils as U


@U.with_default_mesh()
@pytest.mark.parametrize("batch", [pytest.param(b, id=f"b{b}") for b in U.DECODE_BATCHES])
def test_plus_one(ttnn_mesh_device, reset_seeds, batch):
    mesh = ttnn_mesh_device

    pos_torch = torch.arange(batch, dtype=torch.int32)
    pos = U.to_tt(pos_torch, mesh, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

    # In-place increment (model.py:992, rot_mat_idxs path — no skip).
    ttnn.plus_one(pos)

    got = U.from_tt(pos, mesh).flatten()[:batch].long()
    ref = (pos_torch + 1).long()
    assert torch.equal(got, ref), f"plus_one mismatch:\n  got: {got[:8]}\n  ref: {ref[:8]}"


@U.with_default_mesh()
@pytest.mark.parametrize("batch", [pytest.param(b, id=f"b{b}") for b in U.DECODE_BATCHES])
def test_plus_one_skip_negative(ttnn_mesh_device, reset_seeds, batch):
    """skip_negative_entries=True: negative (padding) slots must stay unchanged (model.py:991)."""
    mesh = ttnn_mesh_device

    pos_torch = torch.arange(batch, dtype=torch.int32)
    if batch > 1:
        pos_torch[-1] = -1  # simulate a padded decode slot
    pos = U.to_tt(pos_torch, mesh, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

    ttnn.plus_one(pos, skip_negative_entries=True)

    got = U.from_tt(pos, mesh).flatten()[:batch].long()
    ref = pos_torch.clone().long()
    ref[ref >= 0] += 1  # non-negative entries incremented, negatives untouched
    assert torch.equal(got, ref), f"plus_one(skip_negative) mismatch:\n  got: {got[:8]}\n  ref: {ref[:8]}"
