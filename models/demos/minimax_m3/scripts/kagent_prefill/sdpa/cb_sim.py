#!/usr/bin/env python3
"""CB-protocol simulator for the packed sparse_sdpa_msa kernels: runs reader / writer / compute of one core as
coroutines over circular buffers with the factory's capacities and reports deadlock or unbalanced CBs.
The kernels' CB operations are transcribed from the .cpp sources (same order)."""
import sys

sys.path.insert(0, __import__("os").path.dirname(__file__))
import random

import torch
from packed_model import SENT, reader_groups


def make_caps(G, Skt=4, DHt=4, vDHt=4):
    Sqt = G // 2
    caps = dict(
        q_rm=16 * G,
        q_in=Sqt * DHt,
        k=2 * Skt * DHt,
        v=2 * Skt * vDHt,
        scale=1,
        qk=Skt,
        corr=1,
        out_im=Sqt * vDHt,
        out_rm=Sqt * vDHt,
        idx=1,
        ctrl=2,
        col=1,
        recip=1,
        kreq=2,
        kack=2,
        neginf=1,
        halfmask=2,
        vmask=2 * G,
    )
    for r in range(Sqt):
        for n, c in (("max", 1), ("sum", 1), ("out", vDHt)):
            for sd in (0, 1):
                caps[f"{n}{sd}_{r}"] = c
    return caps


def reader(groups, G, Skt=4, DHt=4):
    yield ("reserve", "idx", 1)
    for grp in groups:
        yield ("reserve", "vmask", G)
        yield ("push", "vmask", G)
        yield ("reserve", "q_rm", 16 * G)
        yield ("push", "q_rm", 16 * G)
        yield ("reserve", "ctrl", 1)
        yield ("push", "ctrl", 1)
        for e in range(len(grp["uid"])):
            yield ("reserve", "k", Skt * DHt)
            yield ("reserve", "v", Skt * DHt)
            yield ("reserve", "kreq", 1)
            yield ("push", "kreq", 1, (e == len(grp["uid"]) - 1, grp["g"]))
            yield ("wait", "kack", 1)
            yield ("pop", "kack", 1)
            yield ("push", "k", Skt * DHt)
            yield ("push", "v", Skt * DHt)


def writer(total, vDHt=4):
    for cb, n in (("scale", 1), ("col", 1), ("neginf", 1), ("halfmask", 2)):
        yield ("reserve", cb, n)
        yield ("push", cb, n)
    rem = total
    while rem > 0:
        last = False
        while not last:
            yield ("wait", "kreq", 1)
            msg = yield ("peek", "kreq")
            last, g = msg
            yield ("pop", "kreq", 1)
            yield ("reserve", "kack", 1)
            yield ("push", "kack", 1)
        for r in range((g + 1) // 2):
            yield ("wait", "out_rm", vDHt)
            yield ("pop", "out_rm", vDHt)
        rem -= g


def compute(groups, G, total, Skt=4, DHt=4, vDHt=4):
    Sqt = G // 2
    yield ("wait", "scale", 1)
    yield ("wait", "neginf", 1)
    yield ("wait", "halfmask", 2)
    done = 0
    side = [0] * Sqt
    gi = 0
    while done < total:
        grp = groups[gi]
        gi += 1
        for b in range(Sqt):  # tilize
            yield ("wait", "q_rm", 32)
            yield ("reserve", "q_in", DHt)
            yield ("push", "q_in", DHt)
            yield ("pop", "q_rm", 32)
        yield ("wait", "ctrl", 1)
        g = grp["g"]
        n_rows = (g + 1) // 2
        yield ("wait", "vmask", G)
        yield ("wait", "q_in", Sqt * DHt)
        started = 0
        for e in range(len(grp["uid"])):
            sel = grp["umask"][e] & ((1 << g) - 1)
            yield ("wait", "k", Skt * DHt)
            yield ("wait", "v", Skt * vDHt)
            for r in range(n_rows):
                if (sel >> (2 * r)) & 3 == 0:
                    continue
                first = not (started >> r) & 1
                st = 0 if first else side[r]
                nx = 0 if first else 1 - st
                yield ("reserve", "qk", Skt)
                yield ("reserve", f"sum{nx}_{r}", 1)
                yield ("reserve", f"out{nx}_{r}", vDHt)
                yield ("push", "qk", Skt)  # hold-wr-ptr push
                yield ("reserve", f"max{nx}_{r}", 1)
                if not first:
                    yield ("wait", f"max{st}_{r}", 1)
                yield ("wait", "qk", Skt)
                yield ("push", f"max{nx}_{r}", 1)
                yield ("wait", f"max{nx}_{r}", 1)  # sub_exp
                yield ("wait", "qk", Skt)
                if not first:
                    yield ("reserve", "corr", 1)
                    yield ("wait", f"max{st}_{r}", 1)
                    yield ("wait", f"max{nx}_{r}", 1)
                    yield ("push", "corr", 1)
                    yield ("wait", f"out{st}_{r}", vDHt)
                    yield ("wait", f"sum{st}_{r}", 1)
                    yield ("wait", "corr", 1)
                    yield ("pop", "corr", 1)
                    yield ("pop", f"out{st}_{r}", vDHt)
                    yield ("pop", f"max{st}_{r}", 1)
                    yield ("pop", f"sum{st}_{r}", 1)
                yield ("push", f"sum{nx}_{r}", 1)
                yield ("push", f"out{nx}_{r}", vDHt)
                yield ("pop", "qk", Skt)
                side[r] = nx
                started |= 1 << r
            yield ("pop", "k", Skt * DHt)
            yield ("pop", "v", Skt * vDHt)
        for r in range(n_rows):
            st = side[r]
            if not (started >> r) & 1:
                yield ("reserve", "out_im", vDHt)
                yield ("push", "out_im", vDHt)
                continue
            yield ("wait", "col", 1)
            yield ("wait", f"sum{st}_{r}", 1)
            yield ("reserve", "recip", 1)
            yield ("push", "recip", 1)
            yield ("pop", f"sum{st}_{r}", 1)
            yield ("wait", f"out{st}_{r}", vDHt)
            yield ("wait", "recip", 1)
            yield ("reserve", "out_im", vDHt)
            yield ("push", "out_im", vDHt)
            yield ("pop", "recip", 1)
            yield ("pop", f"out{st}_{r}", vDHt)
            yield ("pop", f"max{st}_{r}", 1)
        yield ("pop", "ctrl", 1)
        yield ("pop", "vmask", G)
        yield ("pop", "q_in", Sqt * DHt)
        for r in range(n_rows):
            yield ("wait", "out_im", vDHt)
            yield ("reserve", "out_rm", vDHt)
            yield ("push", "out_rm", vDHt)
            yield ("pop", "out_im", vDHt)
        done += g


def simulate(groups, G, total):
    caps = make_caps(G)
    pushed = {c: 0 for c in caps}
    popped = {c: 0 for c in caps}
    reserved = {c: 0 for c in caps}
    msgs = {"kreq": []}
    ks = {"reader": reader(groups, G), "writer": writer(total), "compute": compute(groups, G, total)}
    pend = {k: next(g) for k, g in ks.items()}
    alive = set(ks)

    def ready(op):
        kind, cb = op[0], op[1]
        if kind == "reserve":
            return caps[cb] - (pushed[cb] - popped[cb]) >= op[2]
        if kind == "wait":
            return pushed[cb] - popped[cb] >= op[2]
        return True

    steps = 0
    while alive:
        progress = False
        for k in list(alive):
            while k in alive and ready(pend[k]):
                op = pend[k]
                kind, cb = op[0], op[1]
                send = None
                if kind == "push":
                    pushed[cb] += op[2]
                    assert pushed[cb] - popped[cb] <= caps[cb], f"{k}: push overflow {cb}"
                    if len(op) > 3:
                        msgs[cb].append(op[3])
                elif kind == "pop":
                    popped[cb] += op[2]
                    assert popped[cb] <= pushed[cb], f"{k}: pop underflow {cb}"
                    if cb in msgs:
                        msgs[cb].pop(0)
                elif kind == "peek":
                    send = msgs[cb][0]
                try:
                    pend[k] = ks[k].send(send) if send is not None else next(ks[k])
                except StopIteration:
                    alive.discard(k)
                progress = True
                steps += 1
        if not progress:
            return f"DEADLOCK: " + ", ".join(f"{k} blocked on {pend[k]}" for k in alive)
    bad = [c for c in caps if pushed[c] != popped[c] and c not in ("scale", "col", "neginf", "halfmask", "idx")]
    return f"ok ({steps} ops)" if not bad else f"UNBALANCED {bad}"


if __name__ == "__main__":
    cap = torch.load("/mnt/data/kernel-agent/dev/prefill/runs/c5120-v2/dump/cuts_ev/msa_block_ids.pt")
    rows = cap[30]["block_ids"][0][:1280].tolist()
    rows = [[(i if 0 <= i < 0xFFFFFFF0 else SENT) for i in r] for r in rows]
    worst = "ok"
    for G in (2, 4, 6, 8):
        res = {}
        for core in range(130):
            st = core * 9 + min(core, 110)
            cnt = 9 + (1 if core < 110 else 0)
            groups = list(reader_groups(rows, st, cnt, 1280, G, 51200))
            r = simulate(groups, G, cnt)
            res[r.split(" ")[0]] = res.get(r.split(" ")[0], 0) + 1
            if not r.startswith("ok"):
                print(G, core, r)
                worst = r
        print(f"G={G}: {res}")
    rnd = random.Random(5)
    for trial in range(300):
        G = rnd.choice([2, 4, 6, 8])
        S = rnd.randint(1, 40)
        rows = []
        for t in range(S):
            n = rnd.randint(1, 16)
            r = [rnd.randrange(20) for _ in range(n)]
            rows.append(r + [SENT] * (16 - n))
        groups = list(reader_groups(rows, 0, S, S, G, 0, causal=False))
        r = simulate(groups, G, S)
        if not r.startswith("ok"):
            print("random", trial, G, S, r)
            worst = r
    print("ALL OK" if worst == "ok" else "FAIL")
