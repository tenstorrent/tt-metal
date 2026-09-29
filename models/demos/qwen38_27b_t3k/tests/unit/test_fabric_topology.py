# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A ring topology on a linear fabric must be refused, not left to hang.

Measured on a T3K: `all_reduce_async` with Topology.Ring completes in 0.06 s on
FABRIC_1D_RING and never completes on FABRIC_1D, while the same op with Topology.Linear
runs fine on FABRIC_1D. Only the mismatch deadlocks, and it does so as a device timeout
minutes into serving rather than as an error at startup.
"""

import pytest

import ttnn
from models.demos.qwen38_27b_t3k.tt.decoder_tp import _RING_FABRICS, validate_fabric_topology

_LINEAR_FABRICS = ["FABRIC_1D", "FABRIC_1D_NEIGHBOR_EXCHANGE", "FABRIC_2D"]


@pytest.mark.parametrize("name", _LINEAR_FABRICS)
def test_ring_topology_is_refused_on_a_fabric_without_the_wrap_link(monkeypatch, expect_error, name):
    fabric = getattr(ttnn.FabricConfig, name)
    monkeypatch.setattr(ttnn, "get_fabric_config", lambda: fabric)
    with expect_error(ValueError, "Ring collectives need a ring fabric"):
        validate_fabric_topology(ttnn.Topology.Ring)


@pytest.mark.parametrize("name", sorted(f.name for f in _RING_FABRICS))
def test_ring_topology_is_accepted_on_a_ring_fabric(monkeypatch, name):
    monkeypatch.setattr(ttnn, "get_fabric_config", lambda: getattr(ttnn.FabricConfig, name))
    validate_fabric_topology(ttnn.Topology.Ring)


@pytest.mark.parametrize("name", _LINEAR_FABRICS + ["FABRIC_1D_RING", "DISABLED"])
def test_linear_topology_is_accepted_on_any_fabric(monkeypatch, name):
    # A linear collective needs no wrap-around hop, so no fabric excludes it.
    monkeypatch.setattr(ttnn, "get_fabric_config", lambda: getattr(ttnn.FabricConfig, name))
    validate_fabric_topology(ttnn.Topology.Linear)


def test_an_unopened_fabric_is_not_judged(monkeypatch):
    # Construction can precede the caller opening a fabric; only a configured mismatch is a bug.
    monkeypatch.setattr(ttnn, "get_fabric_config", lambda: ttnn.FabricConfig.DISABLED)
    validate_fabric_topology(ttnn.Topology.Ring)


def test_the_ring_fabric_set_is_not_empty_and_excludes_plain_1d():
    # A typo in the names would silently disable the whole guard.
    assert _RING_FABRICS
    assert ttnn.FabricConfig.FABRIC_1D_RING in _RING_FABRICS
    assert ttnn.FabricConfig.FABRIC_1D not in _RING_FABRICS
