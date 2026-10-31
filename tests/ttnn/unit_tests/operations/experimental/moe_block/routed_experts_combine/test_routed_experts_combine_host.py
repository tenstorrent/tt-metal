# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""routed_experts_combine host math (C++, bound in ttnn.experimental.moe_block) against the Python reference of the
trimmed tree (mstaletovic/moe-trim): row pieces, the state layout, the L1 sizing (combine_l1_layout / combine_auto),
the credit policy, the exchange's arena reserve and the planner's host model (plan_reference). No device.

    pytest tests/ttnn/unit_tests/operations/experimental/moe_block/routed_experts_combine/test_routed_experts_combine_host.py

The reference modules load from $MOE_BLOCK_REF_TREE (default /data/mstaletovic/tt-metal-trimb); the parity tests skip
without it, the self-contained checks always run.
"""

import collections
import importlib
import itertools
import os
import random
import sys
import types
from pathlib import Path

import pytest

import ttnn

mb = ttnn.experimental.moe_block

REF_TREE = Path(os.environ.get("MOE_BLOCK_REF_TREE", "/data/mstaletovic/tt-metal-trimb"))

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


@pytest.fixture(scope="module")
def ref():
    """The reference modules: sparse_combine_l1, sparse_combine_rf, sparse_all_gather_2d, combine_plan."""
    bringup = REF_TREE / "ttnn" / "ttnn" / "bringup"
    if not (bringup / "combine_rf" / "sparse_combine_rf.py").exists():
        pytest.skip(f"reference tree not found: {REF_TREE}")
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
    names = ("sparse_combine_l1", "sparse_combine_rf", "sparse_all_gather_2d", "combine_plan")
    mods = types.SimpleNamespace(**{n: importlib.import_module(f"ttnn.bringup.combine_rf.{n}") for n in names})
    yield mods
    for k in list(sys.modules):
        if k.startswith("ttnn.bringup"):
            del sys.modules[k]
    for k, v in saved.items():
        if v is not None:
            sys.modules[k] = v
    if not had_attr:
        del ttnn.bringup


def mode_of(name):
    return {None: None, "ports apart": mb.CombineMode.PortsApart, "acc_on_ports": mb.CombineMode.AccOnPorts}[name]


def test_pieces_and_state(ref):
    rf = ref.sparse_combine_rf
    for W, pm in itertools.product((512, 1024, 2048, 3584, 4096, 5120, 7168), (None, 7168, 10240, 4096)):
        assert mb.combine_pieces(W, pm) == rf._pieces(W, pm), (W, pm)
    for N, A in itertools.product((1, 2, 4, 8, 16, 32), range(1, 17)):
        assert mb.combine_state_layout(N, A) == rf._state(N, A), (N, A)
    for S, A, NT in itertools.product((32, 160, 640, 2560), range(1, 17), (1, 2, 4, 5)):
        assert mb.combine_acc_cb_bytes(S, A, NT) == rf._acc_cbs(S, A, NT)


def test_l1_layout(ref):
    rf = ref.sparse_combine_rf
    for (N, S, W), arena, A, fused, qp, keep in itertools.product(
        SHAPES, ARENAS, (1, 5, 11, 16), (False, True), (8, 64), (2, 8)
    ):
        piece = max(7168, 2 * W)
        kw = dict(arena_bytes=arena, acc_on_ports=fused, inj_slots=qp, depth=8, piece_max=piece, local_keep=keep)
        exp = rf.combine_l1_layout(N, S, W, A, **kw)
        got = mb.combine_l1_layout(N, S, W, A, **kw)
        assert got == exp, (N, S, W, arena, A, fused, qp, keep)
    exp = rf.combine_l1_layout(8, 640, 4096, 8, arena_bytes=1 << 20)  # (two pieces a row: piece_max 7168 < 8192 B)
    assert mb.combine_l1_layout(8, 640, 4096, 8, arena_bytes=1 << 20) == exp


def test_combine_auto(ref):
    rf = ref.sparse_combine_rf
    for (N, S, W), arena, (cells, ports), a_max, mode in itertools.product(
        SHAPES, ARENAS, ((19, 8), (19, 14), (30, 8), (6, 6), (4, 8)), (16, 5, 11), (None, "ports apart", "acc_on_ports")
    ):
        piece = max(7168, 2 * W)
        exp = rf.combine_auto(
            N, S, W, cells, ports, arena_bytes=arena, depth=8, piece_max=piece, a_max=a_max, mode=mode
        )
        got = mb.combine_auto(
            N, S, W, cells, ports, arena_bytes=arena, depth=8, piece_max=piece, a_max=a_max, mode=mode_of(mode)
        )
        assert got == exp, (N, S, W, arena, cells, ports, a_max, mode)


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


def _ref_arena(free, flat_cbs, stat, man_b, in_program):
    """k3 exchange_arena_reserve + the arena's rows (fused_arena_reserve, WORDS_ROWS 192)."""
    fused = lambda b: (b + 63) // 64 * 64 + 192 * 64 + 2048  # noqa: E731
    reserve = fused(max(flat_cbs, stat)) + man_b + 1024 if in_program else fused(max(flat_cbs, man_b + stat)) + 1024
    return reserve, ((free - reserve) // 64 // 32 * 32 - 192) * 64


def test_arena_sizing(ref):
    sag = ref.sparse_all_gather_2d
    for N, S, W, grp in itertools.product((8, 32), (160, 640, 2560), (3584, 4096, 7168), (2, 8)):
        assert mb.dispatch_static_cb_bytes(N, S, W, group=grp) == sag.static_cb_bytes(N, S, W, group=grp)
    for free, flat_cbs, stat, man_b, inp in itertools.product(
        (1_200_000, 1_400_000), (60_000, 200_000), (80_000, 150_000), (0, 40_960), (False, True)
    ):
        reserve, arena = _ref_arena(free, flat_cbs, stat, man_b, inp)
        assert mb.combine_exchange_arena_reserve(flat_cbs, stat, man_b, inp) == reserve
        assert mb.combine_exchange_arena_bytes(free, reserve) == arena
        assert mb.combine_fused_arena_reserve(flat_cbs) == ref.sparse_combine_rf.fused_arena_reserve(flat_cbs)


def _routing(N, S, K, E, seed, skew):
    """[N, S, K] distinct expert ids per token (uniform or Zipf-ish), and bf16 weight bits."""
    rng = random.Random(seed)
    weights = [1.0 / (e + 1) ** skew for e in range(E)]
    ids, wb = [], []
    for _ in range(N * S):
        row = set()
        while len(row) < K:
            row.add(rng.choices(range(E), weights)[0] if skew else rng.randrange(E))
        ids += sorted(row)
        wb += [rng.randrange(0x3C00, 0x3F80) for _ in range(K)]
    return ids, wb


@pytest.mark.parametrize("token_major", (False, True))
def test_plan_reference(ref, monkeypatch, token_major):
    import torch

    monkeypatch.setenv("TT_FABRIC_TIE_PARITY", "1")  # (the reference's default: ties by the source's parity, as main)
    cp = ref.combine_plan
    for (R, C), wrap, S, K, A, skew in itertools.product(
        ((1, 4), (2, 4), (4, 4), (8, 4)), ((False, False), (True, True)), (32, 64), (4, 8), (5, 16), (0, 1.2)
    ):
        N = R * C
        EPC = 4
        E = N * EPC
        wrap = (wrap[0] and C > 2, wrap[1] and R > 2)
        ids, wb = _routing(N, S, K, E, seed=R * 100 + C + S + K + A, skew=skew)
        luts = mb.port_luts_uniform(R, C, wrap[0], wrap[1], 2)
        rows_cap = (N * S * K // N + EPC * 32 + 31) // 32 * 32  # (tight: overflow rows are dropped alike)
        tid, twb = torch.tensor(ids).reshape(N, S, K), torch.tensor(wb).reshape(N, S, K)
        for me in range(N):
            ch = (me // C, me % C)
            lut, P = luts["lut"][ch], luts["num_ports"][ch]
            exp = cp.plan_reference(
                tid, twb, me, R, C, S, K, EPC, E, A, 2, P, lut, rows_cap, wrap=wrap, token_major=token_major
            )
            got = mb.combine_plan_reference(
                ids, wb, me, R, C, S, K, EPC, E, A, 2, P, list(lut), rows_cap,
                wrap_x=wrap[0], wrap_y=wrap[1], token_major=token_major,
            )  # fmt: skip
            for k in ("counts", "regions", "token_index", "table", "plan"):
                assert got[k] == exp[k], (R, C, wrap, S, K, A, skew, me, k)


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
