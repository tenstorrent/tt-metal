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
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res_stream import (
    BLOCK_SIZE,
    _block_sites,
    rank_site_groups,
    sealed_before,
)
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
    return 1 + sealed_before(boundary, block)


@pytest.mark.parametrize("boundary, expected", [(0, 1), (12, 2), (24, 3), (36, 4), (60, 6), (72, 7)])
def test_plane_count_is_one_live_stream_plus_the_sealed_snapshots(boundary, expected):
    assert _planes(boundary) == expected


def test_adapter_and_runtime_agree_on_plane_count():
    """The adapter's formula and the runtime's must be the same function of the boundary."""
    for boundary in range(KimiK3Config.NUM_LAYERS + 1):
        assert KimiK3Config.attn_res_planes(boundary) == _planes(boundary)


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
    # Seals at 0, 12, 24 precede a boundary at 30: ceil, not floor.
    assert sends == 1 + -(-rank1_first // BLOCK_SIZE)


def test_planes_grow_by_one_per_completed_block():
    """A deeper boundary carries strictly more, by exactly one plane per block."""
    counts = [_planes(b) for b in range(0, 96, BLOCK_SIZE)]
    assert counts == list(range(1, len(counts) + 1))


@pytest.mark.parametrize("boundary, expected", [(1, 2), (11, 2), (13, 3), (85, 9), (92, 9), (93, 9)])
def test_plane_count_off_a_block_edge_counts_the_seal_at_the_block_start(boundary, expected):
    """Seals fire at 0, 12, ...: a boundary inside block k has already passed seal 12k."""
    assert _planes(boundary) == expected


# --- unaligned entry: a window may start and end anywhere -------------------------------
#
# The single-layer suite runs any layer alone, so a rank's first layer need not open a block.
# The walk must then (a) seal on GLOBAL multiples of the block size, and (b) read every site
# of the inherited block against the inherited set. `rank_site_groups` is the bookkeeping;
# these check it against the undivided walk's grouping for every window shape that matters.

NUM = KimiK3Config.NUM_LAYERS


def _global_groups():
    """The undivided walk's groups as global site names, with the sealed depth each reads."""
    pre, post, q_out = _queries(NUM)
    groups = _block_sites(NUM, pre, post, q_out, BLOCK_SIZE)
    return {site: depth + 1 for depth, group in enumerate(groups) for site in group}, groups


def _named(first, groups):
    def name(site):
        kind, local = site
        return "qout" if kind == "out" else f"{kind}{first + local}"

    return [[name(site) for site in group] for group in groups]


def _windows():
    for first in range(NUM):
        for count in sorted({1, 2, BLOCK_SIZE, NUM - first}):
            if first + count <= NUM:
                yield first, count


@pytest.mark.parametrize("first, count", list(_windows()))
def test_rank_groups_refine_the_undivided_walk(first, count):
    depth_of, global_groups = _global_groups()
    head, groups = rank_site_groups(first, count, BLOCK_SIZE)
    named = _named(first, ([head] if head is not None else []) + groups)
    is_last = first + count == NUM
    if not is_last:
        # A non-last rank keeps q_out at the end of its final group, never read (`handoff`
        # ends the walk first); it is there so aligned ranks batch exactly as before.
        assert named[-1][-1] == "qout"
        named[-1] = named[-1][:-1]
        named = [group for group in named if group]

    # Same sites, same order, as the undivided walk restricted to this window. The sentinels are
    # global names, so slicing them gives the window's global sites directly.
    pre, post, _ = _queries(NUM)
    expected = walk_sites(pre[first : first + count], post[first : first + count], None, first_pre_issued=first > 0)
    flat = [site for group in named for site in group]
    assert flat == expected + (["qout"] if is_last else [])

    # Every group is a contiguous run of ONE global group, so it reads that group's sealed depth.
    for group in named:
        depths = {depth_of[site] for site in group}
        assert len(depths) == 1, f"group {group} spans sealed depths {depths}"
        whole = global_groups[depths.pop() - 1]
        start = whole.index(group[0])
        assert whole[start : start + len(group)] == group, f"group {group} is not a contiguous run of {whole}"

    # The head reads against exactly what was inherited; each later group against one more seal.
    if head is not None:
        assert depth_of[named[0][0]] == sealed_before(first)
    seals = [layer for layer in range(first, first + count) if layer % BLOCK_SIZE == 0]
    assert len(groups) == len(seals)


@pytest.mark.parametrize("first", [0, 12, 36, 84])
@pytest.mark.parametrize("count", [1, 2, 12])
def test_rank_groups_on_a_block_edge_match_the_original_grouping(first, count):
    """Aligned windows batch exactly as before the walk learned global positions: no numerics move."""
    count = min(count, NUM - first)
    pre, post, q_out = _queries(count)
    head, groups = rank_site_groups(first, count, BLOCK_SIZE)

    def local_name(site):
        kind, local = site
        return "qout" if kind == "out" else f"{kind}{local}"

    actual = [[local_name(site) for site in group] for group in groups]
    assert actual == _block_sites(count, pre, post, q_out, BLOCK_SIZE)
    assert (head is None) == (first == 0)
    if head is not None:
        assert [local_name(site) for site in head] == ["pre0"]


@pytest.mark.parametrize("first", [1, 5, 11, 13, 27, 47, 49, 85, 87, 92])
def test_single_layer_window_off_a_block_edge(first):
    """One layer mid-block: both of its reads are against the inherited set, and nothing seals."""
    head, groups = rank_site_groups(first, 1, BLOCK_SIZE)
    assert head == [("pre", 0), ("post", 0), ("out", None)]
    assert groups == []


def test_rank_groups_reject_empty_windows(expect_error):
    with expect_error(ValueError, "num_layers >= 1"):
        rank_site_groups(0, 0, BLOCK_SIZE)


# --- the walk itself, driven through a stub op -------------------------------------------
#
# The bookkeeping above is necessary but not the walk: `TtAttnResWalk.open_layer` decides when to
# seal and which batch a read draws from. A stub op records, for every read, the query it was
# given and how many planes were sealed when its batch was built; the walk runs on stand-in tensors
# that only carry a plane count. Every window must read exactly the undivided walk's sites against
# exactly the undivided walk's depths.


class _Planes:
    def __init__(self, planes):
        self.shape = (1, planes, 1, 1)


class _StubOp:
    def __init__(self):
        self.reads = []

    def inter_block(self, block_residual, queries):
        return ("batch", block_residual.shape[1], tuple(queries)), None, None

    def merge(self, partial, shift, mass, running_sum, q, site=0, pending=None):
        _, depth, queries = partial
        assert queries[site] == q, f"read {q} took site {site} of a batch built for {queries[site]}"
        self.reads.append((q, depth))
        return (_Planes(1), _Planes(1)) if pending is not None else _Planes(1)


class _StubTtnn:
    @staticmethod
    def concat(tensors, dim):
        return _Planes(sum(t.shape[1] for t in tensors))

    @staticmethod
    def add(a, b):
        return _Planes(1)

    @staticmethod
    def deallocate(tensor):
        pass


def _walk_reads(first, count, monkeypatch):
    from models.demos.deepseek_v3_d_p.tt.attn_res import attn_res_stream
    from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res_stream import TtAttnResWalk

    monkeypatch.setattr(attn_res_stream, "ttnn", _StubTtnn)
    pre, post, q_out = _queries(NUM)
    op = _StubOp()
    inherited = _Planes(sealed_before(first)) if first else None
    walk = TtAttnResWalk(
        op,
        _Planes(1),
        pre[first : first + count],
        post[first : first + count],
        q_out,
        count,
        inherited_block_residual=inherited,
        first_layer_idx=first,
    )
    for local in range(count):
        _, _ = walk.open_layer(local)
        walk.write(_Planes(1))
        walk.read()
        walk.write(_Planes(1))
    if first + count == NUM:
        walk.finish()
    else:
        walk.handoff()
    return op.reads


@pytest.mark.parametrize("first, count", list(_windows()))
def test_walk_reads_every_site_against_the_undivided_depth(first, count, monkeypatch):
    depth_of, _ = _global_groups()
    pre, post, q_out = _queries(NUM)
    expected = walk_sites(pre[first : first + count], post[first : first + count], None, first_pre_issued=first > 0)
    if first + count == NUM:
        expected = expected + [q_out]
    reads = _walk_reads(first, count, monkeypatch)
    assert [q for q, _ in reads] == expected
    assert all(depth == depth_of[q] for q, depth in reads), [(q, d, depth_of[q]) for q, d in reads if d != depth_of[q]]


def test_walk_rejects_a_wrong_inherited_depth(monkeypatch, expect_error):
    from models.demos.deepseek_v3_d_p.tt.attn_res import attn_res_stream
    from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res_stream import TtAttnResWalk

    monkeypatch.setattr(attn_res_stream, "ttnn", _StubTtnn)
    pre, post, q_out = _queries(NUM)
    with expect_error(ValueError, "inherits 2 sealed snapshots, got 1"):
        TtAttnResWalk(
            _StubOp(),
            _Planes(1),
            pre[13:],
            post[13:],
            q_out,
            NUM - 13,
            inherited_block_residual=_Planes(1),
            first_layer_idx=13,
        )


def test_walk_rejects_layers_out_of_order(monkeypatch, expect_error):
    from models.demos.deepseek_v3_d_p.tt.attn_res import attn_res_stream
    from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res_stream import TtAttnResWalk

    monkeypatch.setattr(attn_res_stream, "ttnn", _StubTtnn)
    pre, post, q_out = _queries(NUM)
    walk = TtAttnResWalk(_StubOp(), _Planes(1), pre[:2], post[:2], q_out, 2)
    with expect_error(AssertionError, "out of order"):
        walk.open_layer(1)


def test_walk_requires_a_position_with_an_inherited_set(monkeypatch, expect_error):
    """An inherited set alone cannot say whether the walk enters on a block edge or mid-block."""
    from models.demos.deepseek_v3_d_p.tt.attn_res import attn_res_stream
    from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res_stream import TtAttnResWalk

    monkeypatch.setattr(attn_res_stream, "ttnn", _StubTtnn)
    pre, post, q_out = _queries(NUM)
    with expect_error(ValueError, "needs its first_layer_idx"):
        TtAttnResWalk(_StubOp(), _Planes(1), pre[12:], post[12:], q_out, NUM - 12, inherited_block_residual=_Planes(1))
