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
    _fused_vs_phased,
    _hw_only,
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
