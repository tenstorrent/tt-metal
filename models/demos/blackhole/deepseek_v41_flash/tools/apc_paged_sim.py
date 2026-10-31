# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU simulation of prefix-cache hit policies for APC_PAGED_DESIGN.md (no device, no model).

Policies, per request (prompt tokens = a numpy int array, chunk C = 512):
  none      every prompt computed from 0
  host_pow2 today's host tier: snapshots only at chunk boundaries that are 512 * 2^k (host budget unlimited), hit = deepest stored boundary s <= min(common prefix, e - 128)
  dev512    device-resident block cache per mesh row: pages + key slots per 512-block, an exact halo slot (the tail state) at EVERY chunk end the row computed;
            hit = deepest boundary s (multiple of 512) <= e - 128 with the halo present and every block of [0, s) present on the row; halo slots are an LRU of H per row, blocks an LRU of
            P chunk-blocks per row (pages are plentiful: the pool is sized for max_model_len x users); the request goes to the row with the deepest hit among the rows that have a free user
            (each row is busy with probability p_busy, at least one row is free); ``xrow`` = a hit on any row is usable (device-to-device copy)
TTFT model fitted to the apc_suite measurements (8192 cold 5.8 s, hit at 4096 3.64 s, multi-turn turn 10 6.78 s): f + chunks * c (+ r_host for a host restore).
"""
import argparse
import hashlib
from collections import OrderedDict

import numpy as np

C = 512
ZR = 2560  # DSV41_PF_REPLAY: layers 21-39 only run on the last 2560 tokens of a prompt (OPTIMIZATIONS.md 8.6)
F, CH_A, CH_AB, R_DEV = (
    0.12,
    0.30,
    0.52,
    0.02,
)  # s: fixed per request; per 512-chunk replay (A = layers 0-20 only, AB = all 40 layers: chunk start in the replay zone); device restore (page table + key/halo copies, estimate). Fitted to the apc_suite colds: 1024: 1.17, 1638: 2.22, 8192: 5.8, 32k: 20.05, 64k: 39.2, 100k: 59.6 s, hit at 4096 of 8192: 3.64 s


def r_host(s):
    """host snapshot restore seconds (server logs of apc_suite: 512 0.05, 4096 0.10, 8192 0.14, 16384 0.27, 32768 0.48, 65536 0.95)"""
    return 0.04 + 1.45e-5 * s


def seg(rng_seed, n):
    return np.random.default_rng(rng_seed).integers(1, 100000, size=n, dtype=np.int64)


def chain(tokens, top):
    """{s: key} for every multiple s of C up to top"""
    h, out = hashlib.sha256(), {}
    for s in range(C, top + 1, C):
        h.update(tokens[s - C : s].tobytes())
        out[s] = h.copy().hexdigest()
    return out


class Host:
    def __init__(self, tip=False, div=False):
        self.div, self.seen = div, set()  # host_div: tip + a snapshot at the divergence point from earlier prompts
        self.tip = tip  # host_tip: ALSO a snapshot at the last full chunk boundary of every prompt (the cheap interim of the design note)
        self.snaps = {}  # key -> full (layers 21-39 halos valid)

    def hit(self, t, e):
        top = (e - 128) // C * C
        ch = chain(t, top)
        p0 = max(0, (e - ZR) // C * C)
        for s in range(top, 0, -C):
            if ch[s] in self.snaps and (s <= p0 or self.snaps[ch[s]]):
                return s
        return 0

    def after(self, t, e):
        ch = chain(t, e // C * C)
        sdiv = max(
            [s for s, k in ch.items() if k in self.seen] or [0]
        )  # deepest boundary an earlier prompt shared (the divergence point)
        self.seen.update(ch.values())
        for s, k in ch.items():
            k2 = s // C
            if k2 & (k2 - 1) == 0 or (self.tip and s == e // C * C) or (self.div and s == sdiv):
                self.snaps[k] = self.snaps.get(k, False) or (s - C >= max(0, (e - ZR) // C * C))


class Row:
    def __init__(self, H, P, thin=1):
        self.H, self.P, self.thin = H, P, thin
        self.blocks = OrderedDict()  # key -> None (a chunk-block: pages + key slot), LRU
        self.halos = OrderedDict()  # key -> None, LRU

    def hit(self, t, e):
        top = (e - 128) // C * C
        ch = chain(t, top)
        best = 0
        p0 = max(0, (e - ZR) // C * C)  # first chunk start of the new prompt that runs layers 21-39 (bounded replay)
        for s in range(C, top + 1, C):
            if ch[s] not in self.blocks:
                break  # blocks of [0, s) must all be present
            if ch[s] in self.halos and (
                s <= p0 or self.halos[ch[s]]
            ):  # a halo without the layers 21-39 part (False) is only usable at s <= p0
                best = s
        return best, ch

    def touch(self, ch, s):
        for k in range(C, s + 1, C):
            self.blocks.move_to_end(ch[k])
        if s and ch[s] in self.halos:
            self.halos.move_to_end(ch[s])

    def insert(self, t, e, s0):
        ch = chain(t, e // C * C)
        for s, k in ch.items():
            if k in self.blocks:
                self.blocks.move_to_end(k)
            else:
                self.blocks[k] = None
            if (s > s0 or s0 == 0) and (
                s == e // C * C or (s // C) % self.thin == 0
            ):  # chunk ends the request computed (an exact halo exists there); thin: interior halos only on every thin-th chunk end, the tip always
                hasb = s - C >= max(
                    0, (e - ZR) // C * C
                )  # the chunk ran layers 21-39 (its start is in the replay zone): the halo is complete
                if k in self.halos:
                    self.halos.move_to_end(k)
                    self.halos[k] = self.halos[k] or hasb
                else:
                    self.halos[k] = hasb
        while len(self.halos) > self.H:
            self.halos.popitem(last=False)
        while len(self.blocks) > self.P:
            self.blocks.popitem(last=False)


def run(reqs, policy, H=128, P=4096, p_busy=0.0, xrow=False, seed=0, thin=1):
    rng = np.random.default_rng(seed)
    host, rows = Host(tip=policy in ("host_tip", "host_div"), div=(policy == "host_div")), [
        Row(H, P, thin) for _ in range(4)
    ]
    tot_new = tot_tok = 0
    ttft = []
    hits = 0
    for t in reqs:
        e = len(t)
        s = 0
        if policy in ("host_pow2", "host_tip", "host_div"):
            s = host.hit(t, e)
            r = r_host(s) if s else 0
        elif policy == "dev512":
            free = [i for i in range(4) if rng.random() >= p_busy] or [int(rng.integers(4))]
            cand = list(range(4)) if xrow else free
            best = (0, None, None)
            for i in cand:
                b, ch = rows[i].hit(t, e)
                if b > best[0]:
                    best = (b, i, ch)
            s = best[0]
            if best[1] is not None and best[1] not in free and xrow:
                tgt = free[int(rng.integers(len(free)))]
            else:
                tgt = best[1] if best[1] is not None else free[int(rng.integers(len(free)))]
            if s:
                rows[best[1]].touch(best[2], s)
            r = R_DEV if s else 0
            rows[tgt].insert(t, e, s)
        else:
            r = 0
        if policy in ("host_pow2", "host_tip", "host_div"):
            host.after(t, e)
        new = e - s
        p0 = max(0, (e - ZR) // C * C)
        chunks = sum(CH_AB if s + k * C >= p0 else CH_A for k in range(-(-new // C)))
        tot_new += new
        tot_tok += e
        hits += s > 0
        ttft.append(F + chunks + r)
    return dict(
        recompute_frac=round(tot_new / tot_tok, 3),
        hit_req=round(hits / len(reqs), 2),
        ttft_mean=round(float(np.mean(ttft)), 2),
        ttft_p90=round(float(np.percentile(ttft, 90)), 2),
    )


def agents(n_conv, turns, sys_len=6000, grow=3900, cap=30000, first=(1000, 3000), seed=1):
    """n_conv agent conversations over a shared system prompt; every turn the prompt = previous prompt + ~500 generated tokens + ~grow new tokens; a conversation restarts after cap tokens;
    requests are issued in a random order, one per conversation per round."""
    rng = np.random.default_rng(seed)
    sysp = seg(1, sys_len)
    conv = [None] * n_conv
    cid = 10
    out = []
    for rnd in range(turns):
        for c in rng.permutation(n_conv):
            if conv[c] is None or len(conv[c]) + 500 + grow > cap:
                cid += 1
                conv[c] = np.concatenate([sysp, seg(cid, int(rng.integers(*first)))])
            else:
                cid += 1
                conv[c] = np.concatenate(
                    [conv[c], seg(cid, 500), seg(cid + 100000, max(200, int(rng.normal(grow, grow * 0.2))))]
                )
            out.append(conv[c].copy())
    return out


def shared_sys(n, sys_len, tail=(1000, 3000), seed=2):
    rng = np.random.default_rng(seed)
    sysp = seg(1, sys_len)
    return [np.concatenate([sysp, seg(100 + i, int(rng.integers(*tail)))]) for i in range(n)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, nargs="+", default=[128])
    ap.add_argument("--P", type=int, default=4096)
    ap.add_argument("--thin", type=int, nargs="+", default=[1])
    a = ap.parse_args()
    cases = [
        ("agents x8  sys6k <=30k", agents(8, 14)),
        ("agents x16 sys6k <=30k", agents(16, 8)),
        ("agents x32 sys6k <=30k", agents(32, 5)),
        ("agents x8  8k-16k +1.6k", agents(8, 20, sys_len=1500, grow=1600, cap=16400, first=(5000, 6500))),
        ("sys 6k + 2k tail x64", shared_sys(64, 6000)),
        ("sys 30k + 2k tail x32", shared_sys(32, 30000)),
    ]
    print(
        "mean TTFT (s) [recompute fraction of prompt tokens]; dev512: pb = P(home row busy), x = hit on any row usable via copy"
    )
    print(
        f"{'case':24s} {'none':>12s} {'host_pow2':>12s} {'host_tip':>12s} {'host_div':>12s} | "
        + " ".join(f"H{h}/t{t} pb0 pb.5 x".rjust(24) for h in a.H for t in a.thin)
    )
    for name, reqs in cases:
        cells = []
        for pol, kw in (("none", {}), ("host_pow2", {}), ("host_tip", {}), ("host_div", {})):
            r = run(reqs, pol, **kw)
            cells.append(f"{r['ttft_mean']:5.2f} [{r['recompute_frac']:.2f}]")
        dev = []
        for h in a.H:
            for t in a.thin:
                rs = [
                    run(reqs, "dev512", H=h, P=a.P, thin=t, **kw)
                    for kw in (dict(p_busy=0.0), dict(p_busy=0.5), dict(p_busy=0.5, xrow=True))
                ]
                dev.append(" ".join(f"{r['ttft_mean']:4.2f}[{r['recompute_frac']:.2f}]" for r in rs))
        print(f"{name:24s} {cells[0]:>12s} {cells[1]:>12s} {cells[2]:>12s} {cells[3]:>12s} | " + " | ".join(dev))
