# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The cross-chip top-k merge (fused/merges/topk_merge.py) on the mesh the model opens.

Mock Galaxy (HY4_MOCK=1): every case compiles its programs (the merge, its launch-floor twin, the scan's cut) and
checks CB / L1 / kernel sizes; nothing runs. Board, smallest first, each new kernel build alone with -x and a
short operation timeout:

    TT_METAL_OPERATION_TIMEOUT_SECONDS=25 pytest <this file> -x -k "test_topk and pair"
    TT_METAL_OPERATION_TIMEOUT_SECONDS=25 pytest <this file> -x -k "test_topk and (row0 or col0)"
    TT_METAL_OPERATION_TIMEOUT_SECONDS=25 pytest <this file> -x -k "test_topk and galaxy"
    TT_METAL_OPERATION_TIMEOUT_SECONDS=25 pytest <this file> -x -k "test_topk_hub and pair-8"
    TT_METAL_OPERATION_TIMEOUT_SECONDS=25 pytest <this file> -x -k "hub"
    pytest <this file> -x -k test_latency

A case runs the merge on lists with many exact ties, short contexts and chips with nothing visible (tests/cases.py)
and compares, for every chip and user: ``keep`` with the sorted golden (exact), the cut page it wrote to DRAM, the
cut key every chip derived; then runs the scan's cut program on the merge's cut page and compares the selected
(chip, stripe row) ids with the golden's (exact, as sets and in the chip's rank order). Result lines: HY4_MK6.

Wait graph of the program (per chip: 8 user cores U0..U7 with one kernel on NCRISC, one sender core S with the
collective's sender on BRISC; nothing else runs):

    U_u  DRAM read of its survivor (barrier)          DRAM, always answers
    U_u  hand_over round 0: NoC write to S + barrier  S's L1, always answers
    S    fabric open handshake, per direction         this chip's routers (always running)
    S    ready bits of the 16 ops                     the hand-overs of U_0..U_7 (round 0: unconditional; round 1:
                                                      after U_u's round 0 landed). Bounded (READY timeout).
    S    a free slot in a router's worker channel     routers drain into L1 writes: a landing never pushes back
    U_u  round 0 arrival counter >= members           S's own-slot count (after its sends) and one fused
                                                      write + increment per other member's S, sent as soon as
                                                      that member's U_u handed over, which waits for nothing
                                                      from another chip. Bounded (LANDING timeout).
    U_u  round 1 arrival counter >= members           the same, for hand-overs made after round 0 landed on each
                                                      member. No member's round 0 send waits for a round 1 event.
    U_0  mailbox counter >= users                     one increment per U_u of this chip after its merge. Bounded.
    S    fabric close                                 this chip's routers
A member that never runs the program, or members that disagree on the site's round (state left by a run that
stopped: TopkMerge.reset), cannot be waited out: the bounded waits then end the program and the flags say so.

Wait graph of the hub form (per chip: hub core H and user cores U0..U7 with one kernel each on NCRISC, sender S):

    U_u  DRAM read of its survivor                    DRAM
    U_u  NoC write of its vector to S's staging       S's L1, always answers; then it counts on H
    H    users' count >= users (round r)              round 0: every U_u, unconditionally. Round r > 0: every U_u
                                                      after H's record of round r - 1. Bounded.
    S    the round's ready bit (one op per round)     H, after the users' count. Bounded.
    H    arrival counter >= members                   S's own-slot count and one fused write + increment per other
                                                      member's S, sent when that chip's H raised its ready bit,
                                                      which needs only that chip's users (and, for r > 0, that
                                                      chip's landing of round r - 1). Bounded.
    U_u  record count >= 1                            H, after it decided the round for all users. Bounded.
A user core writes its next vector into S's staging only after H's record, and H writes records only after the
landing completed, which S's own-slot count ends: S has by then read the staging buffer for the last time.
No wait is on a later event of the same or another chip: user -> hub -> sender -> hubs -> users, round by round.
"""

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.collective import collective as cl
from models.experimental.hy4_preview.fused.common import tracing
from models.experimental.hy4_preview.fused.indexer_scan import indexer_scan as scan
from models.experimental.hy4_preview.fused.merges import topk_merge as tm
from models.experimental.hy4_preview.fused.merges.tests import cases
from models.experimental.hy4_preview.tt import prefill_bridge as pb

GALAXY_MESH = (8, 4)
RESULTS = Path(os.environ.get("HY4_MK6_RESULTS", "generated/hy4_preview/logs/mk6_topk_latency.json"))
CLOCK_HZ = 1.343e9  # AICLK as the board reports it at open; the kernels' stamps count its cycles
MOCK = cl.is_mock()

galaxy = pytest.mark.parametrize(
    "device_params, mesh_device",
    [(dict(pb.device_params(), trace_region_size=32 << 20), GALAXY_MESH)],
    indirect=True,
    ids=["8x4-torus"],
)

MEMBERS = {
    "pair": [(0, 0), (0, 1)],
    "pairv": [(0, 0), (1, 0)],
    "row0": [(0, c) for c in range(4)],
    "col0": [(r, 0) for r in range(8)],
    "galaxy": None,
}


def build(mesh_device, name, timing=False, with_cut=True, hub_bits=None):
    """The per-user form, or with ``hub_bits`` the hub form with those rounds."""
    if hub_bits is None:
        merge = tm.TopkMerge(mesh_device, MEMBERS[name], timing=timing)
    else:
        merge = tm.TopkMergeHub(mesh_device, MEMBERS[name], bits=hub_bits, timing=timing)
    cut = None
    if with_cut:
        cut = scan.Cut(merge.scan_view())
        ttnn.deallocate(merge.cuts)
        merge.cuts = cut.cuts
    return merge, cut


def check(name, merge, cut, words, tag):
    """Every member's result against the golden; with ``cut`` also the ids. Returns the golden."""
    gold, got = tm.golden(words), merge.read()
    chips, users = len(merge.coords), merge.users
    assert not bool((got["flags"] & tm.FLAG_TIMEOUT).any()), f"{name} {tag}: a wait ran out: {got['flags'].tolist()}"
    wrong = (got["keep"] != gold["keep"]).nonzero().tolist()
    assert not wrong, (
        f"{name} {tag}: keep differs at (chip, user) {wrong[:8]}: device "
        f"{[int(got['keep'][c, u]) for c, u in wrong[:8]]} golden {[int(gold['keep'][c, u]) for c, u in wrong[:8]]}"
    )
    assert torch.equal(got["cut_word"], (1 << 31) | gold["keep"])
    assert torch.equal(got["cuts"][:, :users], (1 << 31) | gold["keep"]), f"{name} {tag}: the cut page in DRAM differs"
    kept_all = (got["flags"] & tm.FLAG_ALL) != 0
    assert torch.equal(
        got["key"], torch.where(kept_all, 0, gold["key"][None].expand(chips, users))
    ), f"{name} {tag}: cut key"
    merge.site.check_idle()
    if cut is not None:
        cut()
        ttnn.synchronize_device(merge.mesh_device)
        want = cases.selected_ids(words, gold["keep"])
        have = [[] for _ in range(users)]
        for slot, coord in enumerate(merge.coords):
            counts, rows = cut.read(merge.device_index[coord])
            assert torch.equal(counts, gold["keep"][slot]), f"{name} {tag}: the cut's counts on chip {coord}"
            for u in range(users):
                ids = rows[u, : int(counts[u])]
                assert torch.equal(
                    ids, tm.stamp_row(words[slot, u, : int(counts[u])] & 0xFFFF)
                ), f"{name} {tag}: chip {coord} user {u}: the cut's rows are not the chip's first {int(counts[u])}"
                have[u] += [(slot, int(r)) for r in ids]
        differences = sum(len(set(w) ^ set(h)) for w, h in zip(want, have))
        assert differences == 0, f"{name} {tag}: {differences} selected ids differ from the golden"
    return gold


def run_case(mesh_device, name, seeds=(0, 1, 2), hub_bits=None):
    merge, cut = build(mesh_device, name, hub_bits=hub_bits)
    chips = len(merge.coords)
    moved = merge.bytes_moved()
    name = name if hub_bits is None else f"{name} hub {'+'.join(map(str, hub_bits))}"
    logger.info(
        f"HY4_MK6 topk {name}: {chips} chips, routes {merge.site.routes} ({moved['routes']}), {merge.site.parts} part(s) of "
        f"{merge.part_bytes} B, per chip and merge: {moved['injected']} B injected in {moved['packets']} packets, "
        f"{moved['received']} B received; L1 per core {merge.site.l1_bytes() + 4096} B"
    )
    if MOCK:
        merge.compile()
        merge.compile(nop=True)
        cut.compile()
        logger.info(f"HY4_MK6 topk {name}: compiled on the mock mesh")
        return
    kinds = list(cases.KINDS)
    for seed in seeds:
        rotated = kinds[seed % len(kinds) :] + kinds[: seed % len(kinds)]
        words = cases.make(chips, merge.users, seed, rotated)
        merge.set_survivors(words)
        merge()
        gold = check(name, merge, cut, words, f"seed {seed}")
        got = merge.read()
        recount = int(got["recounts"].sum() if "recounts" in got else ((got["flags"] & tm.FLAG_RECOUNT) != 0).sum())
        logger.info(
            f"HY4_MK6 topk {name} seed {seed}: round 1 counted again on {recount} of {chips * merge.users} cores; "
            f"keep exact on {chips} chips x {merge.users} users "
            f"({[k for k in rotated[: merge.users]]}); selected per user {gold['keep'].sum(0).tolist()}, "
            f"ids exact (0 differences)"
        )


@galaxy
@pytest.mark.parametrize("name", list(MEMBERS))
def test_topk(mesh_device, name):
    run_case(mesh_device, name)


HUB_CASES = [
    ("pair", (8, 8)),
    ("pair", (6, 5, 5)),
    ("row0", (8, 8)),
    ("col0", (8, 8)),
    ("galaxy", (8, 8)),
    ("galaxy", (6, 5, 5)),
]


@galaxy
@pytest.mark.parametrize("name, bits", HUB_CASES, ids=[f"{n}-{'+'.join(map(str, b))}" for n, b in HUB_CASES])
def test_topk_hub(mesh_device, name, bits):
    """The hub form (one fabric payload per chip and round): the same checks, exact keep and ids."""
    run_case(mesh_device, name, hub_bits=bits)


def hub_phase_table(got):
    """us per phase of the hub form (min / median / max over chips; user phases over chips and users)."""
    ticks = lambda x: torch.sort((x & 0xFFFFFFFF).double().reshape(-1) * 1e6 / CLOCK_HZ).values
    h, s = got["hub_stamps"], got["stamps"]
    rows = {
        "user: survivor read": s[..., 1] - s[..., 0],
        "user: round 0 vector": s[..., 2] - s[..., 1],
        "user: start to result": s[..., 5] - s[..., 0],
        "hub: start to round 0 ready (users delivered)": h[:, 1] - h[:, 0],
        "hub: round 0 on the fabric": h[:, 2] - h[:, 1],
        "hub: round 0 decisions, 8 users": h[:, 3] - h[:, 2],
        "hub: records out, users count, round 1 ready": h[:, 4] - h[:, 3],
        "hub: round 1 on the fabric": h[:, 5] - h[:, 4],
        "hub: round 1 decisions": h[:, 6] - h[:, 5],
        "hub: start to end": h[:, 7] - h[:, 0],
    }
    out = {}
    for label, value in rows.items():
        us = ticks(value)
        out[label] = (float(us[0]), float(us[len(us) // 2]), float(us[-1]))
    return out


@galaxy
@pytest.mark.parametrize("bits", [(8, 8), (6, 5, 5)], ids=["8+8", "6+5+5"])
def test_hub_timing_build(mesh_device, bits):
    """The hub form built with timing stamps, on the whole mesh: where its time goes."""
    merge, _ = build(mesh_device, "galaxy", timing=True, with_cut=False, hub_bits=bits)
    if MOCK:
        merge.compile()
        return
    for seed, kinds in ((3, None), (4, ["dense"])):
        words = cases.make(len(merge.coords), merge.users, seed, kinds)
        merge.set_survivors(words)
        merge()
        check("galaxy hub", merge, None, words, f"timing seed {seed}")
        got = merge.read()
        logger.info(
            f"HY4_MK6 topk galaxy hub {'+'.join(map(str, bits))} phases ({'mixed' if kinds is None else kinds[0]} lists; "
            f"us at {CLOCK_HZ / 1e6:.0f} MHz, min / median / max; round 1 counted again on {int(got['recounts'].sum())} "
            f"of {got['recounts'].numel()} user cores): "
            + "; ".join(f"{k}: {a:.1f} / {b:.1f} / {c:.1f}" for k, (a, b, c) in hub_phase_table(got).items())
        )


def phase_table(merge, got):
    """us per phase from the kernels' stamps (min / median / max over chips and users)."""
    s = got["stamps"].reshape(-1, 9)  # start, loaded, counts0, handed0, landed0, counts1, handed1, landed1, end
    names = [
        ("survivor read from DRAM", 1, 0),
        ("round 0 counts", 2, 1),
        ("round 0 hand-over", 3, 2),
        ("round 0 on the fabric (handed to landed)", 4, 3),
        ("round 0 decision + round 1 counts", 5, 4),
        ("round 1 hand-over", 6, 5),
        ("round 1 on the fabric", 7, 6),
        ("decision + cut page", 8, 7),
        ("kernel start to end", 8, 0),
    ]
    out = {}
    for label, later, earlier in names:
        us = torch.sort(((s[:, later] - s[:, earlier]) & 0xFFFFFFFF).double() * 1e6 / CLOCK_HZ).values
        out[label] = (float(us[0]), float(us[len(us) // 2]), float(us[-1]))
    return out


@galaxy
@pytest.mark.parametrize("name", ["pair", "galaxy"])
def test_timing_build(mesh_device, name):
    """The kernels built with timing stamps: exact as the plain build, and where a user core's time goes."""
    merge, cut = build(mesh_device, name, timing=True, with_cut=False)
    if MOCK:
        merge.compile()
        return
    for seed, kinds in ((3, None), (4, ["dense"]), (5, ["ties"])):
        words = cases.make(len(merge.coords), merge.users, seed, kinds)
        merge.set_survivors(words)
        merge()
        check(name, merge, None, words, f"timing seed {seed}")
        table = phase_table(merge, merge.read())
        logger.info(
            f"HY4_MK6 topk {name} phases ({'mixed' if kinds is None else kinds[0]} lists; us at {CLOCK_HZ / 1e6:.0f} MHz, "
            "min / median / max over chips and users): "
            + "; ".join(f"{k}: {a:.1f} / {b:.1f} / {c:.1f}" for k, (a, b, c) in table.items())
        )


@galaxy
def test_latency(mesh_device):
    """Traced latency on 8x4, 8 users: per form (per-user landing cores; hub with two and with three rounds) the
    merge alone, its launch floor (the same program shape with kernels that return at once), and the merge
    followed by the scan's cut program; 20 replays back to back, one synchronize, best of three, with 1 and with
    16 calls per trace. After the replays the result is checked again."""
    if MOCK:
        pytest.skip("latency needs the board")
    table = {"ttnn_form_ms_per_user": 6.581, "forms": {}}
    for label, hub_bits in (("hub 8+8", (8, 8)), ("hub 6+5+5", (6, 5, 5)), ("per-user cores 8+8", None)):
        merge, cut = build(mesh_device, "galaxy", hub_bits=hub_bits)
        entry = table["forms"][label] = {
            "bytes": merge.bytes_moved(),
            "l1_bytes_per_core": merge.site.l1_bytes() + 4096,
            "us": {},
        }
        for kinds in (None, ["dense"]):
            lists = "mixed" if kinds is None else kinds[0]
            words = cases.make(len(merge.coords), merge.users, 7, kinds)
            merge.set_survivors(words)
            merge()
            check("galaxy", merge, cut, words, f"latency {label} {lists} first run")
            forms = [("merge", merge)]
            if kinds is None:
                forms += [("launch floor", lambda: merge(nop=True)), ("merge + cut", lambda: (merge(), cut()))]
            for form, fn in forms:
                us = {
                    calls: 1e3 * tracing.traced_ms(mesh_device, fn, 20, calls=calls)["replay_ms"] for calls in (1, 16)
                }
                entry["us"][f"{form}, {lists} lists"] = us
                logger.info(
                    f"HY4_MK6 topk latency 8x4, 8 users, {label}, {lists} lists, {form}: {us[1]:.1f} us per call as a "
                    f"trace of its own, {us[16]:.1f} us per call with 16 per trace (ttnn form: 6581 us per user, PERF.md)"
                )
            merge()
            check("galaxy", merge, cut, words, f"latency {label} {lists} after the replays")
        ttnn.deallocate(merge.state)
        merge.site.release()
    RESULTS.write_text(json.dumps(table, indent=1))
