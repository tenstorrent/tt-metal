# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Adaptive draft groups (MIMO_SPEC_SLOTS=8 MIMO_SPEC_GROUPS=adaptive) in the vLLM adapter: a pass's group width
follows its request count (reference.spec_decode.pass_width). On a pass that relays chain rows, whose relay step
reads the commit of the group each row held in the pass it chains onto, a wider width waits for a pass the host
builds, and a narrower one keeps every group's owner and ring. Host only; skipped unless the process runs with both
variables set."""

from types import SimpleNamespace

import pytest

from models.experimental.mimo_v2_6_flash.reference.spec_decode import (
    ADAPTIVE,
    ADAPTIVE_WIDE_ROWS,
    DRAFT_GROUPS,
    pass_width,
)
from models.experimental.mimo_v2_6_flash.tt.generator_vllm import MiMoV2ForCausalLM

pytestmark = pytest.mark.skipif(not ADAPTIVE, reason="run with MIMO_SPEC_SLOTS=8 MIMO_SPEC_GROUPS=adaptive")


def adapter(width, owners):
    """A one-lane adapter at group ``width`` whose groups belong to ``owners`` (lane slots), with an MTP drafter that
    records the widths its page table is pointed at."""
    m = MiMoV2ForCausalLM.__new__(MiMoV2ForCausalLM)
    m.set_widths = []
    m.models = [SimpleNamespace(mtp=SimpleNamespace(set_width=m.set_widths.append))]
    m.lane_width = [width]
    m.group_owners = [list(owners)]
    m.mtp_ring = [["ring"] * DRAFT_GROUPS]
    m.dflash_ring = [[None] * DRAFT_GROUPS]
    m.chain_stats = {"width_changes": 0, "width_holds": 0}
    return m


def test_three_to_two_requests_widens_on_a_host_built_pass():
    # Three requests in groups of 2; the one in group 1 finished.
    m = adapter(2, [0, None, 2, None])
    assert pass_width(2) == 4
    assert m._set_width(0, 2) == 4
    assert m.group_owners[0] == [None] * DRAFT_GROUPS and m.mtp_ring[0] == [None] * DRAFT_GROUPS
    assert m.set_widths == [4] and m.chain_stats["width_changes"] == 1


def test_a_relay_pass_does_not_widen():
    # The same two requests chained onto the pass before: slot 2's relay reads group 2 of that pass's page, which
    # groups of 4 slots do not have.
    m = adapter(2, [0, None, 2, None])
    assert m._set_width(0, 2, hold=True) == 2
    assert m.group_owners[0] == [0, None, 2, None] and m.mtp_ring[0] == ["ring"] * DRAFT_GROUPS
    assert m.set_widths == [] and m.chain_stats == {"width_changes": 0, "width_holds": 1}
    # The next pass the host builds widens.
    assert m._set_width(0, 2) == 4 and m.set_widths == [4]


def test_a_relay_pass_at_the_right_width_changes_nothing():
    m = adapter(4, [0, 1, None, None])
    assert m._set_width(0, 2, hold=True) == 4
    assert m.chain_stats == {"width_changes": 0, "width_holds": 0} and m.group_owners[0][:2] == [0, 1]


@pytest.mark.skipif(ADAPTIVE_WIDE_ROWS != 2, reason="8 slots: a third request narrows the groups to 2 slots")
def test_a_relay_pass_narrows_keeping_the_groups():
    # Two requests in groups 0 and 1 of 4 slots chain on while a third joins: groups of 2 slots, the relayed rows
    # keep groups 0 and 1 (and their rings), the third takes a free group.
    m = adapter(4, [0, 1, None, None])
    assert m._set_width(0, 3, hold=True) == 2
    assert m.group_owners[0] == [0, 1, None, None] and m.mtp_ring[0] == ["ring"] * DRAFT_GROUPS
    assert m.set_widths == [2] and m.chain_stats == {"width_changes": 1, "width_holds": 0}
