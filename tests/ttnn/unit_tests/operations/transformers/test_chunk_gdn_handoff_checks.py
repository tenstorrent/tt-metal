# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The fused chunk_gdn program with the hand-off protocol's run-time checks compiled in
(ChunkGdnFusedProgramConfig(handoff_checks=True): sequence-valued VALID flags and a per-slot data canary, see
device/chunk_gdn_handoff_protocol.md "Runtime checks"). The checks report through the watcher's ASSERT, so this file
is meant for the watcher legs (TT_METAL_WATCHER=2); without the watcher it still exercises the sequence flags (a
wrong flag hangs the equality wait) and the canary traffic. Bit-exactness against the two-phase program is the
functional gate; a program-cache delta of 1 pins that the fused program ran."""

import pytest
import torch

from models.common.utility_functions import is_blackhole
from tests.ttnn.unit_tests.operations.transformers.test_chunk_gdn_fused import (
    _const_tiles,
    _fused,
    _fused_vs_phased,
    _hw_only,
    _make_inputs,
    _run_op,
    _skip_unless_geometry_fits,
)

pytestmark = pytest.mark.skipif(not is_blackhole(), reason="chunk_gated_delta_rule is Blackhole-only")


@pytest.mark.parametrize("handoff_depth", [2, 3], ids=["d2", "d3"])
@pytest.mark.parametrize(
    "hk, hv, nc, nv, np_producers",
    [
        pytest.param(4, 12, 8, 2, 7, id="bh12-nv2np7-nc8"),  # the 27B TP-4 shape, two receivers per head
        pytest.param(4, 12, 7, 2, 3, id="bh12-nv2np3-nc7"),  # NC == 2*NP+1: odd/even slot alternation
        pytest.param(1, 4, 3, 4, 2, id="bh4-nv4np2-nc3"),  # four v_beta slices, first slot wraparound
        pytest.param(4, 12, 64, None, None, id="bh12-auto-nc64", marks=_hw_only),  # production chunk count
    ],
)
def test_fused_handoff_checks_bit_exact(device, hk, hv, nc, nv, np_producers, handoff_depth):
    if nv is not None:
        _skip_unless_geometry_fits(device, hv, nv, np_producers, nc, placement=1)
    (o_ph, fs_ph), (o_fu, fs_fu), delta, _ = _fused_vs_phased(
        device, hk, hv, nc, nv, np_producers, 20261005, handoff_depth=handoff_depth, handoff_checks=True
    )
    assert delta == 1, f"expected exactly one fused program, got {delta} new cache entries"
    assert torch.equal(o_ph, o_fu) and torch.equal(fs_ph, fs_fu), "fused (handoff_checks) != phased"


def test_handoff_checks_config_field():
    import ttnn

    f = ttnn.ChunkGdnFusedProgramConfig(handoff_checks=True)
    assert f.handoff_checks is True and "handoff_checks=True" in repr(f)
    assert ttnn.ChunkGdnFusedProgramConfig().handoff_checks is False


@pytest.mark.parametrize("handoff_depth", [2, 3], ids=["d2", "d3"])
@_hw_only
def test_fused_handoff_checks_launch_stress(device, handoff_depth):
    """200 back-to-back launches with the checks on, each bit-exact against the phased reference: a timing race in
    the sequence flags or the canary shows as a mismatch, a tripped assert, or a C9 timeout under the watcher."""
    _skip_unless_geometry_fits(device, 12, 2, 7, 8, placement=1)
    (o_ph, fs_ph), (o_fu, fs_fu), delta, (tensors, const_tiles, s0) = _fused_vs_phased(
        device, 4, 12, 8, 2, 7, 20261005, handoff_depth=handoff_depth, handoff_checks=True
    )
    assert delta == 1 and torch.equal(o_ph, o_fu) and torch.equal(fs_ph, fs_fu)
    cfg = _fused(2, 7, handoff_depth=handoff_depth, handoff_checks=True)
    for i in range(2, 201):
        o_i, fs_i = _run_op(device, tensors, const_tiles, s0, cfg)
        assert torch.equal(o_ph, o_i) and torch.equal(fs_ph, fs_i), f"launch {i} differs from the phased reference"


def test_handoff_fault_config_field():
    import ttnn

    f = ttnn.ChunkGdnFusedProgramConfig(handoff_checks=True, handoff_fault=3)
    assert f.handoff_fault == 3 and "handoff_fault=3" in repr(f)
    assert ttnn.ChunkGdnFusedProgramConfig().handoff_fault == 0


@pytest.mark.parametrize(
    "fused_kwargs, message",
    [
        pytest.param(dict(handoff_fault=1), "handoff_fault requires handoff_checks", id="fault-without-checks"),
        pytest.param(dict(handoff_checks=True, handoff_fault=6), "handoff_fault must be in", id="fault-out-of-range"),
    ],
)
def test_handoff_fault_refused_on_host(device, fused_kwargs, message):
    """A fault without the checks, or an unknown fault id, is refused before any program is built."""
    _, tensors, s0 = _make_inputs(device, 1, 8 * 32, 4, 12, True, seed=20261005)
    with pytest.raises(RuntimeError, match=message):
        _run_op(device, tensors, _const_tiles(device), s0, _fused(2, 7, **fused_kwargs))
