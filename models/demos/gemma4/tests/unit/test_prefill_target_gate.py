# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pin ``is_t3k_dense_target``, the gate on the tuned prefill matmul path.

That path (tuned 2D/1D program configs, HiFi4, L1 in0 hoisting, the tall-M
reshape) was measured on dense Gemma4 12B / 31B on a full Wormhole T3K only.
Every other (SKU, variant) must keep the bare ``ttnn.linear`` it runs today,
because nothing here has been measured on it -- Blackhole and the single-card
Wormhole SKUs are where the rest of the Gemma4 CI matrix lives.

Device-free: the gate reads a mesh device's device count and compute grid, both
of which a stub supplies.
"""

from __future__ import annotations

import json
import os
import types
from unittest.mock import patch

import pytest

from models.demos.gemma4.tt.dram_sharded import is_t3k_dense_target

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs")

# The only two cells that may take the tuned path.
_TUNED = {("gemma-4-12B-it", "T3K"), ("gemma-4-31B-it", "T3K")}

_VARIANTS = [
    "gemma-4-12B-it",
    "gemma-4-31B-it",
    "gemma-4-26B-A4B-it",  # MoE
    "gemma-4-E2B-it",  # per-layer input embeddings
    "gemma-4-E4B-it",  # per-layer input embeddings
]

# (name, num_devices, grid_x, grid_y)
_MESHES = [
    ("T3K", 8, 8, 8),
    ("T3K-x2-harvested", 8, 8, 7),  # legal 8-device WH mesh, illegal 8x8 grids
    ("N150", 1, 8, 7),
    ("N300", 2, 8, 7),
    ("WH-1x4", 4, 8, 8),
]


class _StubMesh:
    def __init__(self, num_devices, grid_x, grid_y):
        self._n = num_devices
        self._grid = types.SimpleNamespace(x=grid_x, y=grid_y)

    def get_num_devices(self):
        return self._n

    def compute_with_storage_grid_size(self):
        return self._grid


def _config(variant):
    with open(os.path.join(_CONFIG_DIR, variant, "config.json")) as handle:
        raw = json.load(handle)
    return types.SimpleNamespace(**raw.get("text_config", raw))


@pytest.mark.parametrize("variant", _VARIANTS)
@pytest.mark.parametrize("mesh", _MESHES, ids=[m[0] for m in _MESHES])
def test_gate_fires_only_for_dense_12b_31b_on_wormhole_t3k(variant, mesh):
    mesh_name, num_devices, grid_x, grid_y = mesh
    with patch("models.demos.gemma4.tt.dram_sharded.is_blackhole", return_value=False):
        actual = is_t3k_dense_target(_StubMesh(num_devices, grid_x, grid_y), _config(variant))
    assert actual is ((variant, mesh_name) in _TUNED)


@pytest.mark.parametrize("variant", _VARIANTS)
def test_gate_is_off_on_every_blackhole_sku(variant):
    """Blackhole keeps main's prefill path: none of this was measured there."""
    with patch("models.demos.gemma4.tt.dram_sharded.is_blackhole", return_value=True):
        for _name, num_devices, grid_x, grid_y in _MESHES:
            assert is_t3k_dense_target(_StubMesh(num_devices, grid_x, grid_y), _config(variant)) is False


def test_gate_survives_a_mesh_that_cannot_report_its_grid():
    """A stub / uninitialised mesh must decline rather than raise."""

    class _Broken:
        def get_num_devices(self):
            raise RuntimeError("device not open")

    with patch("models.demos.gemma4.tt.dram_sharded.is_blackhole", return_value=False):
        assert is_t3k_dense_target(_Broken(), _config("gemma-4-12B-it")) is False
        assert is_t3k_dense_target(object(), _config("gemma-4-12B-it")) is False


def test_the_two_config_predicates_are_what_separate_the_variants():
    """Guard the discriminator itself: if a future variant changes shape here,
    the gate silently widens, so assert the properties it keys on."""
    for variant in _VARIANTS:
        config = _config(variant)
        is_moe = bool(getattr(config, "enable_moe_block", False))
        has_pli = bool(getattr(config, "hidden_size_per_layer_input", 0))
        expected_dense_target = variant in ("gemma-4-12B-it", "gemma-4-31B-it")
        assert (not is_moe and not has_pli) is expected_dense_target, variant
