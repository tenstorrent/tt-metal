# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""routed_experts_combine host math (C++, bound in ttnn.experimental.moe_block), no device: the credit policy and what
combine_auto picks.

    pytest tests/ttnn/unit_tests/operations/experimental/moe_block/routed_experts_combine/test_routed_experts_combine_host.py
"""

import collections
import itertools

import ttnn

mb = ttnn.experimental.moe_block

# (N, S, W): the matrix models' exchange shapes (GLM-5.3, K3 latent, DSv4.1 H) on LoudBox / Galaxy meshes + edges
SHAPES = [
    (8, 640, 4096),
    (32, 160, 4096),
    (32, 160, 3584),
    (8, 2560, 3584),
    (32, 640, 3584),
    (4, 1280, 2048),
    (16, 320, 5120),
    (2, 64, 1024),
]
ARENAS = [64 << 10, 400 << 10, 900 << 10, 1200 << 10, 1400 << 10]


def test_credit_policy():
    """Credits go back every 8 consumed rows; idle snapshots at once below 16 chips, batched (8 rows or 8100 cycles old)
    from 16 chips (the Galaxy)."""
    for N in (2, 4, 8, 15):
        assert mb.combine_credit_policy(N) == (8, 1, 0)
    for N in (16, 32):
        assert mb.combine_credit_policy(N) == (8, 8, 8100)


def test_auto_invariants():
    """Self-contained (no reference): what combine_auto picks fits the arena and holds a slot per source chip."""
    for (N, S, W), arena in itertools.product(SHAPES, ARENAS):
        for cells, ports in ((19, 8), (30, 8)):
            got = mb.combine_auto(N, S, W, cells, ports, arena_bytes=arena)
            if not got["fits"]:
                assert got["why"]
                continue
            lay = got["layout"]
            assert lay["fits"] and lay["PSL"] >= N - 1 and lay["PSL"] < 4096
            assert max(lay["acc_used"], lay["port_used"]) <= arena
            assert 1 <= got["n_acc"] <= 16
    counts = collections.Counter(len(mb.combine_state_layout(N, A)) for N in (8, 32) for A in (1, 16))
    assert counts == {10: 4}
