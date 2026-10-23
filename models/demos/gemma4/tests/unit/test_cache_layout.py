# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The warm-cache identity pins every knob that names a tensorbin file. Pure Python."""

from types import SimpleNamespace

from models.demos.gemma4.tt.common import gemma4_cache_layout


def _galaxy_config(ep):
    return SimpleNamespace(tp=8, ep=ep, tp_axis=0, weight_fracture=True)


def test_expert_partition_keys_the_cache_layout():
    galaxy = SimpleNamespace(shape=(8, 4))
    ep4 = gemma4_cache_layout(galaxy, _galaxy_config(ep=4))
    ep1 = gemma4_cache_layout(galaxy, _galaxy_config(ep=1))  # GEMMA4_MOE_EP=0: experts whole per chip
    assert ep4["expert_partition"] == "ep4tp8_tp_axis0"
    assert ep1["expert_partition"] == "ep1tp8_tp_axis0"
    assert ep4 != ep1
    assert ep4["weight_layout"] == "fractured-tp_axis0"


def test_single_device_layout_has_the_trivial_partition():
    layout = gemma4_cache_layout(SimpleNamespace(), None)
    assert layout["expert_partition"] == "ep1tp1_tp_axis1"
    assert layout["weight_layout"] == "tp"
