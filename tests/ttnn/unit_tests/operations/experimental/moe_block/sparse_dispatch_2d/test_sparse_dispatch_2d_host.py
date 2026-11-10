# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""sparse_dispatch_2d host math, no device: each port's input rings and output streams (C++ `wiring`) and the
group / pack defaults.

    pytest tests/ttnn/unit_tests/operations/experimental/moe_block/sparse_dispatch_2d/test_sparse_dispatch_2d_host.py
"""

import itertools

import ttnn

mb = ttnn.experimental.moe_block

MESHES = [(1, 2), (1, 4), (2, 2), (2, 4), (4, 2), (3, 5), (4, 4), (8, 4), (4, 8)]
WRAPS = list(itertools.product((False, True), repeat=2))
OPP = {"X+": "X-", "X-": "X+", "Y+": "Y-", "Y-": "Y+"}


def _cases():
    for (R, C), (tx, ty), L in itertools.product(MESHES, WRAPS, (1, 2)):
        yield R, C, tx, ty, L


def test_streams_land_in_inputs():
    """Every stream a port sends ends in an input ring of the port it names at the neighbour, and every input ring is
    fed by exactly one stream: the END protocol needs each ring's sender to exist and be unique."""
    for R, C, tx, ty, L in _cases():
        plan = mb.mesh_plan_uniform(R, C, tx, ty, L)
        w = mb.sparse_dispatch_2d_wiring(R, C, tx, ty, L)
        fed = {}
        for chip, ports in w.items():
            assert set(ports) == set(plan[chip]), (R, C, chip)
            for d, (ring_mask, streams) in ports.items():
                assert len(streams) == 3
                assert any(on for on, _, _ in streams), (R, C, chip, d)
                peer = tuple(plan[chip][d]["peer"])
                for s, (on, target, ring) in enumerate(streams):
                    if not on:
                        continue
                    assert target in w[peer], (R, C, chip, d, s)
                    peer_mask = w[peer][target][0]
                    assert (peer_mask >> ring) & 1, (R, C, chip, d, s, target, ring)
                    if on == 2:  # (terminal: END(GMAX) only, into ring 0 of the link's own port there)
                        assert (s, target, ring) == (0, OPP[d], 0) and peer_mask & 8, (R, C, chip, d)
                    key = (peer, target, ring)
                    assert key not in fed, (R, C, key)
                    fed[key] = (chip, d)
        for chip, ports in w.items():
            for d, (ring_mask, _) in ports.items():
                for k in range(3):
                    assert bool((ring_mask >> k) & 1) == ((chip, d, k) in fed), (R, C, tx, ty, chip, d, k)


def test_defaults():
    # (on its own: 8 rows a group - in the MoE block its L1 plan picks the group; pack 2 when two rows fit a packet)
    assert mb.sparse_dispatch_2d_defaults(7168, 7232) == (8, 1)
    assert mb.sparse_dispatch_2d_defaults(12288, 12352) == (8, 1)
    assert mb.sparse_dispatch_2d_defaults(4096, 8192) == (8, 2)
