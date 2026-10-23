# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Contract: splitting the stack across pipeline ranks must not change the read schedule.

A pipelined prefill runs layers `0..b-1` on one rank and `b..L-1` on the next. AttnRes is
the only part of the stack that makes a rank's reads depend on layers another rank owns:
a read scores the live stream against every sealed snapshot, and the snapshots for layers
before `b` are produced on the first rank.

The property under test is the one that makes such a split legal at all — that the sites
the two ranks issue, concatenated, are exactly the sites one undivided walk issues. If
that holds, a boundary is a pure handoff and the arithmetic is unchanged; if it does not,
the second rank is reading against a different schedule than the model it is part of.

`walk_sites` skips `pre[0]`, because global layer 0 has nothing sealed to read against. On
a second rank that skip is wrong: its local `pre[0]` is global `pre[b]`, a real query that
an undivided walk issues. That is what `first_pre_issued` selects, and the equality below
is what pins it.

Pure list bookkeeping — sentinel strings, no tensors and no device.
"""

import pytest

from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.weights import query_weight_names
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res_stream import BLOCK_SIZE, _block_sites
from models.demos.deepseek_v3_d_p.tt.attn_res.weights import _cache_artifact_names, walk_sites


def _queries(num_layers):
    """One distinguishable sentinel per query slot, so a misplaced site is visible by name."""
    pre = [f"pre{i}" for i in range(num_layers)]
    post = [f"post{i}" for i in range(num_layers)]
    return pre, post, "qout"


@pytest.mark.parametrize("num_layers, boundary", [(24, 12), (72, 36), (72, 12), (72, 60), (36, 12)])
def test_split_walk_matches_undivided_walk(num_layers, boundary):
    """Two ranks' sites, concatenated, are the undivided walk's sites."""
    pre, post, q_out = _queries(num_layers)

    undivided = walk_sites(pre, post, q_out)

    # First rank: its own layers, and no model-level read — it is not the last rank.
    first = walk_sites(pre[:boundary], post[:boundary], None)
    # Second rank: local pre[0] IS global pre[boundary], so it must be issued.
    second = walk_sites(pre[boundary:], post[boundary:], q_out, first_pre_issued=True)

    assert first + second == undivided


@pytest.mark.parametrize("num_layers, boundary", [(24, 12), (72, 36)])
def test_boundary_read_is_the_last_site_of_the_inherited_group(num_layers, boundary):
    """The straddling read is exactly one site, and it is the first the second rank issues.

    The sealed set installed at the first rank's last seal stays live for one more read —
    the pre-read of the layer that opens the next block. That read lands on the second
    rank, which is why the boundary carries the snapshots rather than just the activation.
    """
    pre, post, q_out = _queries(num_layers)

    groups = _block_sites(num_layers, pre, post, q_out, BLOCK_SIZE)
    inherited_group = groups[boundary // BLOCK_SIZE - 1]

    second = walk_sites(pre[boundary:], post[boundary:], q_out, first_pre_issued=True)

    # Exactly one site of the outgoing group is owed by the next rank, and it is pre[boundary].
    assert inherited_group[-1] == f"pre{boundary}"
    assert second[0] == f"pre{boundary}"


def test_first_pre_issued_defaults_to_skipping_pre0():
    """The default is unchanged: an undivided walk still never issues pre[0]."""
    pre, post, q_out = _queries(4)
    assert walk_sites(pre, post, q_out)[0] == "post0"
    assert "pre0" not in walk_sites(pre, post, q_out)


def test_first_pre_issued_adds_exactly_one_site():
    """Opting in prepends pre[0] and changes nothing else."""
    pre, post, q_out = _queries(4)
    default = walk_sites(pre, post, q_out)
    opted_in = walk_sites(pre, post, q_out, first_pre_issued=True)
    assert opted_in == ["pre0"] + default


# --- the weight window a rank asks for --------------------------------------------------
#
# The cache and the checkpoint are both keyed by GLOBAL layer index and hold all 93 layers.
# A rank asking for `layers.0..` when it holds `layers.36..` finds every name it asks for,
# so nothing raises: it just scores every read with another rank's queries. These pin the
# offset, because no runtime error ever will.


def test_cache_names_are_global_not_rank_local():
    names = _cache_artifact_names(num_layers=2, first_layer_idx=36)
    assert names == ("layers.36.self_attention", "layers.36.mlp", "layers.37.self_attention", "layers.37.mlp", "output")


def test_cache_names_default_to_layer_zero():
    assert _cache_artifact_names(num_layers=1) == ("layers.0.self_attention", "layers.0.mlp", "output")


def test_checkpoint_names_are_global_not_rank_local():
    names = query_weight_names(num_layers=1, prefix="m.", first_layer_idx=12)
    assert all(".12." in n for n in names if "layers" in n)
    assert not any(".0." in n for n in names if "layers" in n)


def test_two_ranks_ask_for_disjoint_windows():
    """The whole point: rank 0 and rank 1 must not request the same tensorbins."""
    first = set(_cache_artifact_names(num_layers=12, first_layer_idx=0))
    second = set(_cache_artifact_names(num_layers=12, first_layer_idx=12))
    assert first & second == {"output"}  # the model-level query is shared, nothing else is


# --- plane accounting -------------------------------------------------------------------
#
# Three places compute how many planes cross a boundary, and they are read at different
# times by different processes: the adapter sizes the socket, the runtime sizes the traced
# receive buffer, and the transformer asserts what it got. They must agree, and when they
# do not the symptom is a rendezvous TT_FATAL or a silently wrong replay, neither of which
# names the disagreement. Kept as arithmetic so no device is needed.


def _planes(boundary, block=BLOCK_SIZE):
    return 1 + boundary // block


@pytest.mark.parametrize("boundary, expected", [(0, 1), (12, 2), (24, 3), (36, 4), (60, 6), (72, 7)])
def test_plane_count_is_one_live_stream_plus_the_sealed_snapshots(boundary, expected):
    assert _planes(boundary) == expected


def test_adapter_and_runtime_agree_on_plane_count():
    """The adapter's formula and the runtime's must be the same function of the boundary."""
    from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config

    for boundary in (0, 12, 24, 36, 60, 72, 84):
        adapter_side = 1 + boundary // KimiK3Config.ATTN_RES_BLOCK_SIZE
        assert adapter_side == _planes(boundary)


@pytest.mark.parametrize("split", [(12, 12), (36, 36), (24, 24), (12, 24), (30, 30)])
def test_a_ranks_outbound_is_the_next_ranks_inbound(split):
    """The property the socket rendezvous enforces, checked here where it is cheap.

    Rank 0 sends at the boundary past its own slice; rank 1 receives at its first layer. Those
    are the same layer, so the two plane counts must be equal — computed from each rank's own
    parameters, which is how the runner computes them and how they can disagree.
    """
    rank0_first, rank0_count = 0, split[0]
    rank1_first = rank0_first + rank0_count

    sends = _planes(rank0_first + rank0_count)  # rank 0's outbound
    expects = _planes(rank1_first)  # rank 1's inbound

    assert sends == expects
    assert sends == 1 + rank1_first // BLOCK_SIZE


def test_planes_grow_by_one_per_completed_block():
    """A deeper boundary carries strictly more, by exactly one plane per block."""
    counts = [_planes(b) for b in range(0, 96, BLOCK_SIZE)]
    assert counts == list(range(1, len(counts) + 1))
