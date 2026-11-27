# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side model of PfStream's credit protocol (kernels/pfs/pfs_stream.cpp + pfs_member_rd.cpp), asserted before any
device run (coordinator 10:59Z). Random asynchronous schedules of every actor:

- member m (reader): credit block c as soon as c < popped + P and c < v + P (slots v .. c free), polling; land block v
  when its data AND its lane's VALID (>= v / 2 + 1) arrived; member 0 publishes the addresses before its first credit;
- member m (compute): pop the oldest landed block;
- streamer (st, L): its list i -> global q (even / odd K-blocks of the stream's N-blocks); waits TOKEN (lane chain),
  then per block the parity word READY[i mod 2] >= M ((g - 1) NI / 2 + i / 2 + 1), then multicasts: per member the
  data and then the VALID set travel in FIFO order on its lane (NoC L), arriving at random later steps;
- launches g = 1 .. G: an actor starts launch g + 1 only after finishing launch g (members: all blocks popped).

Invariants checked at every streamer READY pass and multicast:
  I1 every member credited exactly (g, i) — the count is not met by other blocks' credits;
  I2 the parity word holds no credit of a later same-parity block or of a later launch;
  I3 the PARK slot q mod P is free at every member (it popped q - P);
  I4 member 0 published the addresses before any streamer read them;
  I5 progress: every schedule finishes (no deadlock).
"""

from __future__ import annotations

import random


def _lists(nst: int, nh: int, nbk: int) -> dict:
    """(st, L) -> [global q] in the streamer's order (pfs_stream.cpp: i -> h = i / (NBK / 2), j = 2 (i mod NBK / 2) + L)"""
    out = {}
    for st in range(nst):
        for lane in range(2):
            out[(st, lane)] = [
                (st * nh + i // (nbk // 2)) * nbk + 2 * (i % (nbk // 2)) + lane for i in range(nh * nbk // 2)
            ]
    return out


def simulate(nst: int, nh: int, nbk: int, park: int, members: int = 4, launches: int = 3, seed: int = 0,
             max_steps: int = 2_000_000, parity_words: bool = True) -> dict:  # fmt: skip
    """parity_words=False = v2b's single READY counter per streamer (must fail I1 / I2: the checker's sanity)"""
    assert nbk % 2 == 0
    rng = random.Random(seed)
    lists = _lists(nst, nh, nbk)
    owner = {q: (key, i) for key, qs in lists.items() for i, q in enumerate(qs)}
    nq = nst * nh * nbk
    ni = nh * nbk // 2
    assert ni % 2 == 0
    # member state
    mg = [1] * members  # launch
    mv = [0] * members  # next block to land
    mc = [0] * members  # next block to credit
    mp = [0] * members  # blocks popped by the compute
    published = [False]  # member 0's addresses (per launch it republishes; the first matters)
    pub_launch = [0]
    inbox = [[[], []] for _ in range(members)]  # per member, per lane: FIFO of ("data", g, q) / ("valid", g, value)
    landed = [set() for _ in range(members)]  # (g, q) data landed
    valid = [[0, 0] for _ in range(members)]  # per-launch semaphores (re-initialised at the member's launch start)
    # streamer state
    sg = {k: 1 for k in lists}
    si = {k: 0 for k in lists}
    token = {k: 0 for k in lists}
    ready = {k: [[], []] for k in lists}  # parity words: lists of (g, i, m)
    done_streamers = {k: 0 for k in lists}
    steps = 0
    while True:
        if all(g > launches for g in mg) and all(sg[k] > launches for k in lists):
            return dict(steps=steps, ok=True)
        steps += 1
        if steps > max_steps:
            raise AssertionError(
                f"I5 no progress (deadlock?) seed {seed}: mg {mg} mv {mv} mc {mc} mp {mp} sg {sg} si {si}"
            )
        actors = []
        for m in range(members):
            if mg[m] <= launches:
                actors += [("mr", m), ("mc", m), ("ml", m)]
        for k in lists:
            if sg[k] <= launches:
                actors.append(("s", k))
        kind, a = rng.choice(actors)
        if kind == "mr":  # member reader: one poll iteration (credit ahead, else land)
            m = a
            if m == 0 and pub_launch[0] < mg[m]:
                published[0] = True
                pub_launch[0] = mg[m]
            c, v, p = mc[m], mv[m], mp[m]
            if c < nq and c - v < park and c < p + park:
                key, i = owner[c]
                ready[key][(i & 1) if parity_words else 0].append((mg[m], i, m))
                mc[m] += 1
            elif c > v:
                lane = v & 1
                if (mg[m], v) in landed[m] and valid[m][lane] >= v // 2 + 1:
                    mv[m] += 1
        elif kind == "ml":  # deliver the head of one lane FIFO to member m
            m = a
            lane = rng.randrange(2)
            if inbox[m][lane]:
                what, g, x = inbox[m][lane].pop(0)
                if what == "data":
                    landed[m].add((g, x))
                elif g == mg[m]:  # a VALID SET of the member's current launch: per lane FIFO, so it only grows
                    assert x >= valid[m][lane], f"VALID went back on member {m} lane {lane}: {valid[m][lane]} -> {x}"
                    valid[m][lane] = x
                else:
                    raise AssertionError(f"VALID of launch {g} reached member {m} in launch {mg[m]} (seed {seed})")
        elif kind == "mc":  # member compute: pop the oldest landed block; launch boundary
            m = a
            if mp[m] < mv[m]:
                mp[m] += 1
            if mp[m] == nq:
                mg[m] += 1
                mv[m] = mc[m] = mp[m] = 0
                valid[m] = [0, 0]
        else:  # streamer
            key = a
            st, lane = key
            g, i = sg[key], si[key]
            if i == 0 and st > 0 and token[key] < g:
                continue
            if i < ni:
                if parity_words:
                    need = members * ((g - 1) * (ni // 2) + i // 2 + 1)
                    word = ready[key][i & 1]
                else:
                    need = members * ((g - 1) * ni + i + 1)
                    word = ready[key][0]
                if len(word) < need:
                    continue
                cur = [(gg, ii) for gg, ii, _ in word]
                # I2: nothing for a later same-parity block / later launch
                bad = [
                    (gg, ii)
                    for gg, ii in cur
                    if gg > g or (gg == g and ii > i and (ii - i) % (2 if parity_words else 1) == 0)
                ]
                assert not bad, f"I2 parity word of {key} holds {bad[:4]} while waiting ({g}, {i}) seed {seed}"
                # I1: every member credited exactly (g, i)
                got = sorted(mm for gg, ii, mm in word if (gg, ii) == (g, i))
                assert got == list(range(members)), f"I1 {key} ({g}, {i}) credited by {got} seed {seed}"
                # I4: addresses published
                assert published[0], f"I4 {key} read the addresses before member 0 published (seed {seed})"
                q = lists[key][i]
                # I3: the slot is free at every member (in launch g, popped >= q - park + 1)
                for m in range(members):
                    if mg[m] == g:
                        assert (
                            mp[m] >= q - park + 1
                        ), f"I3 slot of q {q} busy at member {m} (popped {mp[m]}) seed {seed}"
                    else:
                        assert mg[m] > g or False, f"I3 member {m} in launch {mg[m]} < {g} seed {seed}"
                for m in range(members):
                    inbox[m][lane].append(("data", g, q))
                    inbox[m][lane].append(("valid", g, q // 2 + 1))
                si[key] += 1
            else:  # the end of the launch: write barrier (FIFOs drained to every member), TOKEN to (st + 1, L)
                if any(e[1] == g for m in range(members) for e in inbox[m][lane]):
                    continue
                if st + 1 < nst:
                    token[(st + 1, lane)] += 1
                sg[key] += 1
                si[key] = 0
                done_streamers[key] += 1


_CHECKED: dict = {}  # geometry -> total steps: the seeds are fixed, so a passed geometry passes again


def check(nst: int, nh: int, nbk: int, park: int, seeds: int = 12, members: int = 4, launches: int = 3) -> int:
    """run `seeds` random schedules of the geometry (members / launches reduced: the invariants are per member);
    raises AssertionError on any violation; returns the total steps. Memoized per process: every stream of a 45-layer
    GroupStack (16 chunk instances x 5 streams per layer) re-simulated the same few geometries (0.4-3 s each)."""
    key = (nst, nh, nbk, park, seeds, members, launches)
    if key not in _CHECKED:
        total = 0
        for s in range(seeds):
            total += simulate(nst, nh, nbk, park, members=members, launches=launches, seed=s)["steps"]
        _CHECKED[key] = total
    return _CHECKED[key]


if __name__ == "__main__":
    import sys

    for geo in [(2, 2, 4, 4), (2, 2, 4, 3), (2, 2, 4, 2), (3, 4, 16, 4), (12, 4, 16, 4)]:
        print(geo, "steps", check(*geo, seeds=8 if geo[0] < 12 else 2), flush=True)
    # the v2b single-counter protocol must FAIL the model (the checker's sanity)
    caught = 0
    for s in range(8):
        try:
            simulate(2, 2, 4, 4, seed=s, parity_words=False)
        except AssertionError as e:
            caught += 1
            last = str(e)[:160]
    print("v2b single counter: violations in", caught, "of 8 seeds;", last if caught else "")
    sys.exit(0 if caught else 1)


# ---------------------------------------------------------------------------------------------- PMK_MCAST ring-2 model
# moe2's lever 3 (kernels/pmk/ffn_reader.hpp PMK_MCAST): one streamer + (G - 1) receivers share xffn's 2-slot ring.
# Receiver, per chunk n (1-based): cb_reserve_back (slot free: its compute popped chunk n - 2) -> +1 READY (one word on
# the streamer) -> wait VALID >= n -> push. Streamer: cb_reserve_back own slot -> DRAM read -> wait READY >= (G - 1) n
# -> flush -> multicast data + VALID SET n -> push. Stage 7a modelled the same protocol in its tree (simulate_ring2 /
# check_ring2 @ 368d9d4e01e, J1-J5); that file is not on this host, so this is an independent implementation of the
# same invariants with the same name / signature -- 7a's version replaces it on merge.
#   J1 a receiver's slot n mod 2 is free (chunk n - 2 popped) when the streamer's multicast of chunk n lands;
#   J2 READY >= (G - 1) n implies EVERY receiver credited chunk n (one word suffices: a receiver credits n + 1 only
#      after it pushed n, which needs VALID >= n, which the streamer sets after the READY wait of n);
#   J3 VALID is monotonic and a receiver never pushes chunk n before its data landed;
#   J4 every core of the rectangle consumes the same chunk sequence (ring positions aligned);
#   J5 progress: every schedule finishes (no deadlock), over several launches (words reset per launch: program
#      semaphores / the BRISC zeroing of the READY word).
def simulate_ring2(g: int, chunks_per_unit: int, units: int = 3, launches: int = 2, seed: int = 0,
                   max_steps: int = 500_000, slots: int = 2) -> dict:  # fmt: skip
    """slots (perf5, PMK_RING_SLOTS): the rings hold `slots` chunks on both sides (J1: slot n mod S free = chunk n - S popped)"""
    assert g >= 2 and slots >= 2 and chunks_per_unit % slots == 0, (g, chunks_per_unit, slots)
    rng = random.Random(seed)
    n_chunks = units * chunks_per_unit
    steps = 0
    for launch in range(launches):
        ready = 0  # the streamer's READY word (zeroed per launch)
        valid = 0  # the receivers' VALID semaphore (set by multicast; program semaphore: 0 per launch)
        s_issued = 0  # streamer: chunks read into its own ring (<= popped + 2)
        s_sent = 0  # streamer: chunks multicast (VALID = s_sent)
        s_popped = 0  # streamer's compute
        r_credit = [0] * (g - 1)  # receiver i: chunks credited
        r_pushed = [0] * (g - 1)  # receiver i: chunks pushed (consumed by its compute later)
        r_popped = [0] * (g - 1)
        landed = [0] * (g - 1)  # receiver i: chunks whose multicast data arrived (in flight = s_sent - landed)
        while s_popped < n_chunks or any(p < n_chunks for p in r_popped):
            steps += 1
            if steps > max_steps:
                raise AssertionError(
                    f"J5 deadlock-ish: no progress after {max_steps} steps (G={g}, cpu={chunks_per_unit})"
                )
            actor = rng.randrange(2 + 2 * (g - 1))
            if actor == 0:  # streamer
                if s_issued < n_chunks and s_issued < s_popped + slots:
                    s_issued += 1  # cb_reserve_back own slot + DRAM read
                elif s_sent < s_issued and ready >= (g - 1) * (s_sent + 1):
                    n = s_sent + 1
                    for i in range(g - 1):  # J2: every receiver credited chunk n
                        assert r_credit[i] >= n, ("J2", i, r_credit[i], n)
                        # J1: the receiver's slot n mod S is free: it popped chunk n - S (credit implies reserve)
                        assert r_popped[i] >= n - slots, ("J1", i, r_popped[i], n)
                    s_sent = n  # multicast + VALID SET n (data before VALID: same NoC, same source)
                    valid = max(valid, n)  # J3 monotonic
                elif s_popped < s_sent or s_popped < s_issued:
                    pass  # waiting: the streamer's own compute pops below
            elif actor == 1:  # streamer's compute pops a pushed chunk (pushed = issued... the push follows the mcast)
                if s_popped < s_sent:
                    s_popped += 1
            else:
                i, kind = (actor - 2) // 2, (actor - 2) % 2
                if kind == 0:  # receiver's BRISC
                    n = r_credit[i] + 1
                    if r_credit[i] == r_pushed[i] and n <= n_chunks and r_popped[i] >= n - slots:
                        r_credit[i] = n  # cb_reserve_back ok -> +1 READY
                        ready += 1
                    elif r_credit[i] > r_pushed[i]:
                        n = r_pushed[i] + 1
                        if landed[i] < s_sent and rng.random() < 0.7:
                            landed[i] += 1  # the multicast data arrives, in order, with its VALID set behind it
                        # the receiver's view of VALID: the SET of chunk n travels behind chunk n's data on the same
                        # NoC from the same source, so VALID >= n is visible only once the data landed (J3)
                        assert landed[i] <= valid, ("J3", i, landed[i], valid)
                        if landed[i] >= n:
                            r_pushed[i] = n
                else:  # receiver's compute pops
                    if r_popped[i] < r_pushed[i]:
                        r_popped[i] += 1
        # J4: every core consumed the same sequence
        assert s_popped == n_chunks and all(p == n_chunks for p in r_popped), ("J4", s_popped, r_popped)
        assert ready == (g - 1) * n_chunks, ("J2-end", ready)
    return dict(g=g, chunks_per_unit=chunks_per_unit, launches=launches, steps=steps)


def check_ring2(g: int, chunks_per_unit: int, seeds: int = 10, launches: int = 2, slots: int = 2) -> dict:
    """called at PmkMoe2 construction when mcast > 1 (as PfStreamMM calls check): raises on any invariant"""
    out = None
    for seed in range(seeds):
        for units in (1, 3):
            out = simulate_ring2(g, chunks_per_unit, units=units, launches=launches, seed=seed, slots=slots)
    return out
