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


def _ref_credit_policy(N, S, A, PSL, tu, K_):
    """sparse_combine_rf_prepare's inline credit policy (rows_by_chip's K_ the most rows of a (home, token))."""
    cred_idle, cred_age = (8, 8100) if N >= 16 else (1, 0)
    if tu["credidle"]:
        cred_idle, cred_age = tu["credidle"], tu["credage"]
    cred_thr, policy = tu["credthr"], "default"
    MAXT = (S + A - 1) // A
    if tu["credauto"] and MAXT * K_ <= PSL:
        cred_thr, (cred_idle, cred_age), policy = 64, (8, 4050), "end"
    return cred_thr, cred_idle, cred_age, policy


def test_credit_policy():
    for N, S, A, PSL, credidle, credage, credauto, credthr, K_ in itertools.product(
        (4, 8, 16, 32), (160, 640), (5, 16), (7, 64, 248), (0, 4), (0, 3000), (False, True), (8, 16), (0, 8)
    ):
        tu = mb.CombineTuning(credidle=credidle, credage=credage, credauto=credauto, credthr=credthr)
        exp = _ref_credit_policy(
            N, S, A, PSL, dict(credidle=credidle, credage=credage, credauto=credauto, credthr=credthr), K_
        )
        assert mb.combine_credit_policy(N, S, A, PSL, tu, K_) == exp


def test_auto_invariants():
    """Self-contained (no reference): what combine_auto picks fits the arena and holds a slot per source chip."""
    for (N, S, W), arena in itertools.product(SHAPES, ARENAS):
        for cells, ports in ((19, 8), (30, 8)):
            got = mb.combine_auto(N, S, W, cells, ports, arena_bytes=arena, piece_max=max(7168, 2 * W))
            if not got["fits"]:
                assert got["why"]
                continue
            lay = got["layout"]
            assert lay["fits"] and lay["PSL"] >= N - 1 and lay["PSL"] < 4096
            assert max(lay["acc_used"], lay["port_used"]) <= arena
            assert 1 <= got["n_acc"] <= 16
    counts = collections.Counter(len(mb.combine_state_layout(N, A)) for N in (8, 32) for A in (1, 16))
    assert counts == {10: 4}
