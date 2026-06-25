# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

from models.experimental.diffusion_gemma.memory_budget import estimate_canvas_kv_scratch_bytes


def test_qb2_canvas_kv_scratch_estimate_matches_gemma4_tp4_shapes():
    est = estimate_canvas_kv_scratch_bytes(tp=4, batch_size=1, bytes_per_elem=2)

    assert est.sliding_bytes == int(12.5 * 2**20)
    assert est.full_attention_bytes == int(2.5 * 2**20)
    assert est.total_bytes == 15 * 2**20


def test_canvas_kv_scratch_scales_with_batch():
    batch1 = estimate_canvas_kv_scratch_bytes(tp=4, batch_size=1)
    batch4 = estimate_canvas_kv_scratch_bytes(tp=4, batch_size=4)

    assert batch4.total_bytes == 4 * batch1.total_bytes


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tp": 0},
        {"batch_size": 0},
        {"bytes_per_elem": 0},
    ],
)
def test_canvas_kv_scratch_rejects_invalid_dimensions(kwargs):
    with pytest.raises(ValueError):
        estimate_canvas_kv_scratch_bytes(**kwargs)
