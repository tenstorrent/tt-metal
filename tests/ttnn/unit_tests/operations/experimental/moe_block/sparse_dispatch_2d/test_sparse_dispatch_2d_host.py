# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""sparse_dispatch_2d host math, no device: each port's input rings and output streams (C++ `wiring`) and the
group / pack defaults.

    pytest tests/ttnn/unit_tests/operations/experimental/moe_block/sparse_dispatch_2d/test_sparse_dispatch_2d_host.py

The invariant tests run anywhere. The parity test compares against the Python reference
(sparse_all_gather_2d._wiring of mstaletovic/moe-trim, loaded from $MOE_BLOCK_REF_TREE) and skips without it.
"""

import importlib
import itertools
import os
import sys
import types
from pathlib import Path

import pytest

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
    # (tt_k3_token_split_moe.dispatch: 8 rows a group while 2 W <= 7168, else 2; pack 2 when two rows fit a packet)
    assert mb.sparse_dispatch_2d_defaults(3584, 7168, 7232) == (8, 1)
    assert mb.sparse_dispatch_2d_defaults(6144, 12288, 12352) == (2, 1)
    assert mb.sparse_dispatch_2d_defaults(2048, 4096, 8192) == (8, 2)


@pytest.fixture(scope="module")
def ref_sag():
    """The reference module sparse_all_gather_2d (loaded as ttnn.bringup.combine_rf.* without that tree's package
    __init__s; it imports this tree's ttnn)."""
    tree = os.environ.get("MOE_BLOCK_REF_TREE")
    if not tree:
        pytest.skip("MOE_BLOCK_REF_TREE not set (the Python reference tree)")
    bringup = Path(tree) / "ttnn" / "ttnn" / "bringup"
    if not (bringup / "combine_rf" / "sparse_all_gather_2d.py").exists():
        pytest.skip(f"reference tree not found: {tree}")
    saved = {k: sys.modules.get(k) for k in ("ttnn.bringup", "ttnn.bringup.combine_rf")}
    had_attr = hasattr(ttnn, "bringup")
    pkg = types.ModuleType("ttnn.bringup")
    pkg.__path__ = [str(bringup)]
    sub = types.ModuleType("ttnn.bringup.combine_rf")
    sub.__path__ = [str(bringup / "combine_rf")]
    pkg.combine_rf = sub
    sys.modules["ttnn.bringup"], sys.modules["ttnn.bringup.combine_rf"] = pkg, sub
    if not had_attr:
        ttnn.bringup = pkg
    yield importlib.import_module("ttnn.bringup.combine_rf.sparse_all_gather_2d")
    for k in list(sys.modules):
        if k.startswith("ttnn.bringup"):
            del sys.modules[k]
    for k, v in saved.items():
        if v is not None:
            sys.modules[k] = v
    if not had_attr:
        del ttnn.bringup


class _FakeMesh:
    def __init__(self, R, C):
        self.shape = (R, C)

    def get_fabric_node_id(self, coord):
        return coord


def test_wiring_matches_reference(ref_sag, monkeypatch):
    monkeypatch.setattr(ttnn, "get_neighbor_eth_directions", lambda mesh, a, b: [0], raising=False)
    monkeypatch.setattr(ttnn, "get_forwarding_link_indices", lambda a, b: [0, 1, 2, 3], raising=False)
    for R, C, tx, ty, L in _cases():
        chips, nb, _ = ref_sag._mesh_plan(_FakeMesh(R, C), L, (tx, ty))
        got = mb.sparse_dispatch_2d_wiring(R, C, tx, ty, L)
        for coord, ports in chips.items():
            for d in ports:
                exp_mask, exp_streams = ref_sag._wiring(chips, nb, coord, d)
                exp = (exp_mask, [(on, t, k) for on, t, k in exp_streams])
                assert got[coord][d][0] == exp[0] and list(got[coord][d][1]) == exp[1], (R, C, tx, ty, L, coord, d)
