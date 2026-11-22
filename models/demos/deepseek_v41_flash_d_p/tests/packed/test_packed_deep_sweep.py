# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.T DEEP sweep (OWNER 2026-11-21; <= 20 min, run when the packed path changes): SLOT ISOLATION AT CHIP BOUNDARIES
over many compositions (tests/packed/isolation_common.py: the contract, the two worlds, what is compared; every
segment against the same request alone through the serving entry, bit-exact; DECODER_SKIP=decode).

Part 1, ORDERS: six requests, each a history chunk and a step chunk of one kind:
    P1 steady [2048, 4096)   P2 replay [1920, 4096) write_start 2048   P3 1 chip [256, 456)
    P4 split part [1024, 2048)   P5 1 token [512, 513)   P6 acks only [1024, 1064) (cap 1024)
  their history chunks packed together (27 chips), then their step chunk (24 chips), in all 24 orders of P1-P4
  (P5 / P6 at rotating places), every order in another assignment of a 12-slot pool (slots reused by other requests
  of other kinds: leftovers of every kind under every kind). 48 packed chunks against 12 alone ones.
Part 2, FUZZ (seed fixed, reproducible): 14 packed chunks of up to 6 segments of an arrival stream of requests (fresh,
  continued, split at a chip boundary and continued in the next chunk, replayed, 1 chip, 1 token, acks only, partial
  last chunks), slots from a pool of 8 reused as requests end, segment order shuffled; a long request L whose first
  five chunks are whole chunks (8192 rows, 32 chips) and which then joins the packed chunks at history 40960
  (long-history gathers, the long candidate pool). Bystanders: every chunk, one idle slot holding a request's data
  (whole slabs before / after the chunk), and a canary slot nothing uses (before the first chunk / after the last).

Metrics (gated): packed_deep_output_mismatches, packed_deep_served_mismatches, packed_deep_ack_mismatches,
packed_deep_bystander_mismatches, packed_deep_fallback_mismatches (== 0), packed_deep_output_max_abs_diff (== 0).
Recorded: packed_deep_segments, packed_deep_one_ring_layers, packed_deep_long_pool_layers, packed_deep_hide_regions,
packed_deep_program_builds_after_warmup, packed_deep_sweep_s.

    BRINGUP_CHUNK=8192 DS41_PACKED=on ... (the gate's switches) scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/packed/test_packed_deep_sweep.py
"""

from __future__ import annotations

import importlib
import itertools
import random
import time

import pytest

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec
from models.demos.deepseek_v41_flash_d_p.tests.packed import isolation_common as _ic
from models.demos.deepseek_v41_flash_d_p.tests.packed import test_packed_fast_tier as _ft

_ic = importlib.reload(_ic)  # (the resident runner keeps helper modules of an earlier request: take this file's)
_ft = importlib.reload(_ft)

S = spec()
pytestmark = device_timeout(S)
BATCH_SHARED = True  # (the gate's accuracy load: every layer, with the determinism check, OV-61)
MAX_SEQ = _ft.MAX_SEQ
BATCH_MAX_SEQ = 114688  # (the load covers the slots' capacity: test_packed_fast_tier MAX_SEQ)
CHIP = 256
POOL_ORDERS = list(range(6, 18))  # part 1's packed slots (6..17)
POOL_FUZZ = [0, 1, 2, 3, 4, 5, 18, 19]  # part 2's packed slots (the warm-up's junk slots 0-5 among them)
CANARY = 20
FIRST_ALONE = 21
SEED = 20261121
FUZZ_CHUNKS = 14
BYSTANDERS = 6  # idle-slot windows checked (whole slabs before / after), besides the canary

# part 1: (history chunk segment, step chunk segment) a request
KINDS = {
    "P1": ((0, 2048, None), (2048, 4096, None)),
    "P2": ((0, 2048, None), (1920, 4096, 2048)),  # (a resumed request: [0, 2048) written, the window replayed)
    "P3": ((0, 256, None), (256, 456, None)),
    "P4": ((0, 1024, None), (1024, 2048, None)),
    "P5": ((0, 512, None), (512, 513, None)),
    "P6": ((0, 1024, None), (1024, 1064, None)),
}


def _chips(s: int, e: int) -> int:
    return -(-max(1, e - s) // CHIP)


def orders_plan():
    """Part 1 -> (requests {name: (text offset, slot)}, chunks). Request names carry the order index (a request
    lives two chunks: its history and its step); every order's six requests take the pool from a rotating base."""
    reqs, chunks = {}, []
    for k, kind in enumerate(KINDS):  # (the references: each kind alone once, in a slot of its own)
        reqs[kind] = (1000 * k + 37 * (k + 1), -1)
    base = ["P1", "P2", "P3", "P4"]
    for j, perm in enumerate(itertools.permutations(base)):
        order = list(perm)
        order.insert(j % 5, "P5")
        order.insert((j * 2 + 1) % 6, "P6")
        names = {}
        for k, kind in enumerate(KINDS):
            n = f"{kind}o{j}"
            slot = POOL_ORDERS[(j * 5 + k) % len(POOL_ORDERS)]
            reqs[n] = (1000 * k + 37 * (k + 1), slot, kind)  # (a kind's text and boundaries: its reference's)
            names[kind] = n
        hist = [(names[kd], *KINDS[kd][0]) for kd in order]
        step = [(names[kd], *KINDS[kd][1]) for kd in order]
        assert sum(_chips(s, e) for _, s, e, _ in hist) <= 32 and sum(_chips(s, e) for _, s, e, _ in step) <= 32
        chunks += [hist, step]
    return reqs, chunks


def fuzz_plan(text_len: int):
    """Part 2 (seed SEED) -> (requests, chunks, bystanders)."""
    rnd = random.Random(SEED)
    reqs, chunks, bystanders = {}, [], []
    # L: five whole chunks, then packed at history 40960
    reqs["L"] = (0, POOL_FUZZ[0])
    for c in range(5):
        chunks.append([("L", c * 8192, (c + 1) * 8192, None)])
    free = list(POOL_FUZZ[1:])
    active = {}  # name -> dict(pos, split_rest, slot, last_end)
    l_pos = 40960
    n_req = 0
    idle_since = {}  # slot -> chunk index after which it sits idle with data
    for ci in range(FUZZ_CHUNKS):
        segs, freed = [], []
        chips = 0
        if l_pos + 2048 <= MAX_SEQ and ci % 2 == 0:
            ws, s0 = (l_pos, l_pos - 128) if ci % 4 == 2 else (None, l_pos)  # (resumed: the window replayed)
            segs.append(("L", s0, l_pos + 2048, ws))
            chips += 8
            l_pos += 2048
        # continue active requests (a split remainder first)
        for name in list(active):
            if len(segs) >= 6:
                break
            a = active[name]
            if rnd.random() < 0.25 and not a["rest"]:
                continue  # (parked this chunk: a bystander with data)
            p = a["pos"]
            rest = bool(a["rest"])
            if rest:
                n = a["rest"]
                a["rest"] = 0
            else:
                n = rnd.choice([2048, 1024, 256, 200, 777, 1, 40, 1500])
            kind_end = p + n
            if n == 40:
                p = p // 64 * 64  # (acks only: cap = start)
                kind_end = p + 40
            if chips + _chips(p, kind_end) > 32 or kind_end + 1 > MAX_SEQ:
                continue
            ws = None
            if (
                n >= 1024 and not rest and p >= 128 and rnd.random() < 0.3
            ):  # (resumed: the window replayed, [0, p) written)
                ws, p = p, p - 128
            if chips + _chips(p, kind_end) > 32:
                continue
            segs.append((name, p, kind_end, ws))
            chips += _chips(p, kind_end)
            if n in (1, 40, 777, 1500) or rnd.random() < 0.15:  # (a partial / last chunk: the request ends)
                del active[name]
                freed.append(a["slot"])  # (free from the next chunk on: one slot a segment a chunk)
                idle_since.pop(a["slot"], None)
            else:
                a["pos"] = -(-kind_end // 64) * 64 if kind_end % 64 else kind_end
                if a["pos"] != kind_end:  # (a continuing request's chunk ends on a kv block)
                    segs[-1] = (name, p, a["pos"], ws)
        # new arrivals
        while len(segs) < 6 and free and chips < 32:
            n = rnd.choice([2048, 1024, 512, 256, 2560])
            split = rnd.random() < 0.3 and n >= 1024
            first = n // 2 // CHIP * CHIP if split else n
            if chips + _chips(0, first) > 32:
                break
            slot = free.pop(rnd.randrange(len(free)))
            name = f"F{n_req}"
            n_req += 1
            reqs[name] = (rnd.randrange(0, text_len - MAX_SEQ // 4) // 32 * 32, slot)
            segs.append((name, 0, first, None))
            chips += _chips(0, first)
            active[name] = dict(pos=first, rest=n - first if split else 0, slot=slot)
            idle_since.pop(slot, None)
        rnd.shuffle(segs)
        chunks.append(segs)
        free += freed
        used = {reqs[x[0]][1] for x in segs}
        for name, a in active.items():
            if a["slot"] not in used:
                idle_since.setdefault(a["slot"], len(chunks) - 1)
    # bystanders: a slot idle with data from the chunk after its last use to the chunk it is used again (or the end)
    last_use = {}
    for k, c in enumerate(chunks):
        for x in c:
            s = reqs[x[0]][1]
            if s in last_use and k - last_use[s] >= 2:
                bystanders.append((s, last_use[s], k))
            last_use[s] = k
    # (a whole-slab read is ~0.7 GB a slot: at most BYSTANDERS idle windows, spread over the sequence)
    step = max(1, -(-len(bystanders) // BYSTANDERS))
    bystanders = bystanders[::step][:BYSTANDERS] + [(CANARY, -1, len(chunks))]
    return reqs, chunks, bystanders


@mesh_parametrize(timeout_s=3600)
def test_packed_deep_sweep(mesh_device):
    if mesh_device is None:
        pytest.skip("device test")
    t0 = time.time()
    from models.demos.common.bringup.reference.golden import Golden

    text_len = int(Golden.for_rung(S, "s56320").tokens().numel())
    o_reqs, o_chunks = orders_plan()
    f_reqs, f_chunks, f_by = fuzz_plan(text_len)
    n_alone = len(KINDS) + len(f_reqs)  # (part 1: one reference a kind; part 2: every request)
    slots = FIRST_ALONE + n_alone
    print(
        f"DEEP: orders {len(o_chunks)} chunks / {len(o_reqs)} requests; fuzz {len(f_chunks)} chunks / {len(f_reqs)} requests"
    )
    for k, c in enumerate(f_chunks):
        print(f"  fuzz chunk {k}: {c}")

    def fuzz(bench, text, v):
        R = {n: _ic.Req(n, off, slot) for n, (off, slot) in f_reqs.items()}
        for r, s in zip(R.values(), range(FIRST_ALONE + len(KINDS), slots)):
            r.alone_slot = s
        w = _ic.Worlds(bench, text, R, f_chunks, f_by, label="packed_deep fuzz")
        w.packed(w.alone(), v)

    # part 1 through run_tier (its alone slots FIRST_ALONE ..), part 2 as its extra world
    reqs = dict(o_reqs)
    v, paths, hide, fallback, bench, _ = _ft.run_tier(
        mesh_device, "packed_deep", reqs, o_chunks, [], MAX_SEQ, slots=slots, extra=fuzz, alone_from=FIRST_ALONE
    )
    wall = time.time() - t0
    metrics.record("packed_deep_sweep_s", round(wall, 1))
    assert v.out_mm == 0 and v.served_mm == 0 and v.ack_mm == 0 and v.bystander_mm == 0, "slot isolation broken"
    assert v.out_diff == 0, v.out_diff
    assert fallback == 0, f"{fallback} packed chunks did not run the one call"
