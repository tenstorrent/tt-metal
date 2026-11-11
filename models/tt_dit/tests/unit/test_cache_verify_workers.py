# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact-selectable CPU controls for bounded host verification."""
from types import SimpleNamespace

import pytest
import torch

from models.tt_dit.utils import cache


def model(tensor):
    parameter = SimpleNamespace(data=tensor, on_host=True, device=None)
    return SimpleNamespace(named_children=lambda: (), named_parameters=lambda: [("weight", parameter)])


@pytest.mark.parametrize("workers", [1, 2, 4, 99])
def test_native_worker_admission_and_default(tmp_path, workers):
    tensor = cache.ttnn.from_torch(torch.ones(35, 67), dtype=cache.ttnn.bfloat8_b, layout=cache.ttnn.TILE_LAYOUT)
    cache.ttnn.dump_tensor(tmp_path / "weight.tensorbin", tensor)
    stats = {}
    assert cache.verify_saved_model(model(tensor), tmp_path, workers=workers, stats=stats)
    assert stats["workers"] == min(workers, 4)
    assert stats["parallel_pairs"] == (0 if workers == 1 else 1)
    assert stats["peak_workspace_bytes"] <= 512 << 20
