# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1 — the cross-core combine's two shipped knobs, pinned on the HOST.

DO NOT DELETE.  These are the two decisions Refinement 1 turned into derived rules
instead of constants, and both are invisible to a numerical test: the op produces the
same values whichever way they go, so only an assertion on the built descriptor can
catch a later phase quietly flattening them back.

  1. THE COMBINE TREE'S ARITY.  `f0` was the constant 4; it is now the largest divisor
     of GROUP_SIZE in the measured [COMBINE_TREE_F0_MIN, _MAX] band that the tree's two
     existing gates admit.  The expectations below are the MEASURED optima from the
     sweep recorded at the constants' definition -- if a later change moves one, that is
     a perf decision that has to be re-measured, not a test to relax.
  2. WHICH NoC CARRIES THE COMBINE.  NOC_0 (with a reader/writer swap) when x is a
     resident shard, NOC_1 when the reader still streams x from DRAM.  The interleaved
     row is the load-bearing one: NOC_0 there measured 0.667x.

Nothing here dispatches; every assertion is on the host-built ProgramDescriptor.
"""

from __future__ import annotations

import pytest
import torch

import ttnn

from ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor import (
    COMBINE_TREE_F0_MAX,
    COMBINE_TREE_F0_MIN,
    _PC_NONE,
    _combine_tree_arity,
    _residual_depth,
    create_program_descriptor,
)

_ML = ttnn.TensorMemoryLayout


# (group_size, expected (f0, f1) or None) — every entry is a MEASURED point.
_ARITY = [
    (4, None),  # f1 would be 1: a level that gathers one member is a hop, not a fold
    (8, None),  # measured 0.866x with the tree; the deleted-fold gate keeps it flat
    (9, None),  # odd, and no admissible divisor
    (28, None),  # EVERY arity measured a ~5% loss against the flat root here
    (30, (6, 5)),  # 10 is the fastest arity but deletes only 17 fold tiles -> gated out
    (32, (8, 4)),  # 4736 ns vs 4979 at the seed's f0=4 and 5253 flat
    (40, (10, 4)),  # 4789 vs 5064 at f0=4
    (64, (8, 8)),  # 5047 vs 5498 at f0=4 and 6695 flat
]


@pytest.mark.parametrize("group_size, expected", _ARITY, ids=[f"G{g}" for g, _ in _ARITY])
def test_combine_tree_arity_is_the_measured_rule(group_size, expected):
    assert _combine_tree_arity(group_size, 1) == expected


@pytest.mark.parametrize("group_size", list(range(2, 121)))
def test_combine_tree_arity_invariants_hold_at_every_group_size(group_size):
    """The two things the KERNELS static_assert, asserted here for every reachable group."""
    tree = _combine_tree_arity(group_size, 1)
    if tree is None:
        return
    f0, f1 = tree
    assert COMBINE_TREE_F0_MIN <= f0 <= COMBINE_TREE_F0_MAX, "f0 escaped the measured band"
    assert f1 >= 2, "a slot-tree level that gathers one member is a hop, not a fold"
    assert f0 * f1 >= group_size, "the slot tree must cover GROUP_SIZE"


def test_residual_depth_default_is_byte_identical():
    """CB_R_DEPTH is parked at its trivial default: the residual ring follows cb_x_depth."""
    for depth_x in (1, 2, 3):
        assert _residual_depth(depth_x) == depth_x


def _config():
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    return cfg


def _descriptor(device, shape, memory_layout, shard):
    from eval.sharding import shard_config

    dtype = ttnn.bfloat16
    if memory_layout == _ML.INTERLEAVED:
        mc = ttnn.DRAM_MEMORY_CONFIG
    else:
        mc = shard_config(shard[0], shard[1], memory_layout, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
    x = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mc,
    )
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dtype, ttnn.TILE_LAYOUT, device, mc)
    g = ttnn.from_torch(
        torch.zeros(1, 1, 1, shape[-1], dtype=torch.bfloat16), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
    )
    return create_program_descriptor(
        x, out, weight=g, epsilon=1e-12, compute_kernel_config=_config(), program_config=_PC_NONE
    )


def _noc_of(kernel):
    """(is_swapped, noc) for a kernel descriptor's config."""
    cfg = kernel.config
    if isinstance(cfg, ttnn.DataMovementConfigDescriptor):
        # `.value`, not the member: the nanobind NOC enum's NOC_0 / NOC_1 are aliases of
        # RISCV_0_default / RISCV_1_default and do not compare equal to their own alias.
        return True, cfg.noc.value
    return False, None


#: (shape, memory_layout, shard, combine-on-NOC_0?)
_NOC_CASES = [
    # x is a RESIDENT shard: the reader has no activation stream, the combine takes NOC_0.
    ((1, 1, 32, 7168), _ML.WIDTH_SHARDED, ([32, 256], (7, 4)), True),
    ((1, 1, 8192, 1024), _ML.BLOCK_SHARDED, ([1024, 128], (8, 8)), True),
    # x STREAMS from DRAM through the accessor: NOC_0 belongs to the reader.  Measured
    # 9044 -> 13550 ns (0.667x) when this one was allowed to swap.
    ((1, 1, 32, 7168), _ML.INTERLEAVED, None, False),
    # No combine at all -- the plain row split must stay the seed's reader/writer pair.
    ((1, 1, 8192, 1024), _ML.INTERLEAVED, None, False),
]


@pytest.mark.parametrize(
    "shape, memory_layout, shard, swapped",
    _NOC_CASES,
    ids=["width_shard_28c", "block_shard_64c", "interleaved_width_split", "interleaved_row_split"],
)
def test_combine_noc_is_gated_on_a_resident_x(device, shape, memory_layout, shard, swapped):
    d = _descriptor(device, shape, memory_layout, shard)
    reader_swapped, reader_noc = _noc_of(d.kernels[0])
    writer_swapped, writer_noc = _noc_of(d.kernels[1])
    assert reader_swapped == swapped and writer_swapped == swapped, (
        "the combine's NoC choice must be a SWAP of both kernels or of neither -- a writer on "
        "NOC_0 with the reader left there drives one engine from two RISCs, and HANGS"
    )
    if swapped:
        assert writer_noc == ttnn.NOC.NOC_0.value and reader_noc == ttnn.NOC.NOC_1.value


# ---------------------------------------------------------------------------
# Refinement 2 (Lamp L-FIN) — the finalize site and the two transport knobs
# ---------------------------------------------------------------------------
#
# Same reason as everything above: all three are INVISIBLE to a numerical test.  The op
# produces bit-identical output whichever way each goes (measured: pcc and rel-RMS agree
# to every printed digit across the whole sweep), so only an assertion on the built
# descriptor can catch a later phase flattening one back.
#
#   COMBINE_FIN_SPREAD          WHERE the rsqrt runs.  Parked at ROOT after measuring the
#                               spread at 0.953-1.004x -- see the note at
#                               `_combine_fin_spread`.  Kept as a LIVE knob, and this test
#                               asserts it is still live (both settings build) as well as
#                               that its default is the byte-identical one.
#   COMBINE_MCAST_FIRE_AND_FORGET   the stat multicast's pre-handshake, elided when the
#                               combine runs exactly ONE round.
#   COMBINE_MCAST_FACES         faces the IDENTITY-path multicast carries (3 = faces 0..2,
#                               one transaction covering both column-carrying faces).

from ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor import (  # noqa: E402
    COMBINE_FIN_SPREAD,
    COMBINE_MCAST_FACES,
    COMBINE_MCAST_FIRE_AND_FORGET,
    GATHER_FACES,
    _combine_fin_spread,
    _combine_gather_faces_ct,
    _combine_mcast_pre_handshake,
)

_WRITER_CT_FACES = 15  # the PACKED face-count word in rms_norm_ttnn_writer.cpp
_COMPUTE_CT_FIN_SPREAD = 23


def test_fin_spread_default_is_the_root_and_is_byte_identical():
    """Lamp L-FIN's knob is parked at ROOT, which allocates nothing and changes no arg."""
    assert COMBINE_FIN_SPREAD is False
    assert _combine_fin_spread(True) is False
    assert _combine_fin_spread(False) is False


def test_fin_spread_is_still_a_live_knob():
    """Parked, not deleted: flipping the module constant must actually move the CT arg."""
    import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as PD

    saved = PD.COMBINE_FIN_SPREAD
    try:
        PD.COMBINE_FIN_SPREAD = True
        assert PD._combine_fin_spread(True) is True
        assert PD._combine_fin_spread(False) is False, "the spread is meaningless off the combine path"
    finally:
        PD.COMBINE_FIN_SPREAD = saved


@pytest.mark.parametrize(
    "combine, single_round, expected",
    [
        (True, True, False),  # one round: the landing CB is provably still at its boot state
        (True, False, True),  # round n+1's send would race round n's drain
        (False, True, True),  # no combine, no multicast
    ],
)
def test_mcast_pre_handshake_is_gated_on_a_single_round(combine, single_round, expected):
    assert COMBINE_MCAST_FIRE_AND_FORGET is True
    assert _combine_mcast_pre_handshake(combine, single_round) is expected


@pytest.mark.parametrize(
    "combine, compact, expected_mcast_faces",
    [
        (True, False, COMBINE_MCAST_FACES),  # identity path: the trim applies
        (True, True, 0),  # compact: the un-permute matmul needs every column finite
        (False, False, 0),  # no combine: the word must be the seed's literal GATHER_FACES
    ],
)
def test_packed_face_word_keeps_the_gather_byte_and_gates_the_mcast_byte(combine, compact, expected_mcast_faces):
    word = _combine_gather_faces_ct(combine, compact)
    assert word & 0xFF == GATHER_FACES, "the gather's own face count may never move"
    assert (word >> 8) & 0xFF == expected_mcast_faces
    if not combine:
        assert word == GATHER_FACES, "a non-combine build must emit the seed's literal word"


#: (shape, memory_layout, shard, compact?, single_round?)
_TRANSPORT_CASES = [
    ((1, 1, 32, 7168), _ML.WIDTH_SHARDED, ([32, 256], (7, 4)), False, True),
    ((1, 1, 32, 1024), _ML.WIDTH_SHARDED, ([32, 128], (8, 1)), False, True),
    ((1, 1, 1024, 512), _ML.WIDTH_SHARDED, ([1024, 128], (4, 1)), True, None),
    ((1, 1, 8192, 1024), _ML.BLOCK_SHARDED, ([1024, 128], (8, 8)), True, False),
    ((1, 1, 8192, 1024), _ML.INTERLEAVED, None, None, None),  # no combine
]


@pytest.mark.parametrize(
    "shape, memory_layout, shard, compact, single_round",
    _TRANSPORT_CASES,
    ids=["width_28c", "width_8c", "width_compact_4c", "block_64c", "no_combine"],
)
def test_transport_words_reach_the_writer(device, shape, memory_layout, shard, compact, single_round):
    """End-to-end: the two gates land in the writer's CT list where the kernel reads them."""
    d = _descriptor(device, shape, memory_layout, shard)
    writer_ct = list(d.kernels[1].compile_time_args)
    word = writer_ct[_WRITER_CT_FACES]
    assert word & 0xFF == GATHER_FACES
    if compact is None:  # no combine at all
        assert word == GATHER_FACES
        return
    assert (word >> 8) & 0xFF == (0 if compact else COMBINE_MCAST_FACES)
    if single_round is not None:
        # McastArgs<18, ...>: [active, data_ready, consumer_ready, num_active, flags, span]
        flags = writer_ct[18 + 4]
        assert bool(flags & 0x1) is (
            not single_round
        ), "the pre-handshake bit must be OFF exactly on a single-round combine"


def test_compute_carries_the_finalize_site(device):
    d = _descriptor(device, (1, 1, 32, 7168), _ML.WIDTH_SHARDED, ([32, 256], (7, 4)))
    assert list(d.kernels[2].compile_time_args)[_COMPUTE_CT_FIN_SPREAD] == 0
