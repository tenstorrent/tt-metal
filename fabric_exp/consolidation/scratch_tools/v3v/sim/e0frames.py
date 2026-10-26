#!/usr/bin/env python3
"""e0frames.py: pass-level model of a one-hop stream where E0 (sender) binds and pays per completion frame.
E0: one loop iteration per packet sent (base cost E0B), plus F when it finds new completions (one credit return per
iteration that sees any). It needs a free far slot (16) and the wire (one packet per WIRE cycles).
E1: progress pass Cp (+jitter), idle pass Ci; batches as v3t: >4 waiting on progress, >=4 on an idle pass, a partial
batch after H idle passes in a row; a batch waits until its newest slot's write has left (WR cycles after issue);
a frame costs E1 CC. Frames reach E0 after DRET; packets reach E1 DFWD after E0 sends them.
Prints cycles/packet and E1 frames per packet."""
import random, sys

def run(Cp, H, Ci=18, E0B=122, F=35, WIRE=118, DFWD=600, DRET=600, WR=70, CC=45, jit=12, npk=20000, seed=1, S=16):
    rnd = random.Random(seed)
    send_t = []; frames_at = []  # (time frame reaches E0, number of slots credited)
    # E0 state
    t0 = 0.0; credited_seen = 0; last_wire = -1e9; fi = 0; credits_arrived = 0
    # E1 state
    t1 = 0.0; retired = 0; credited = 0; idle = 0; issue_t = {}; nframes = 0
    arrive = []
    k = 0
    while credited < npk - 8:
        # advance whichever core is behind
        if t0 <= t1 and k < npk:
            # E0 iteration: take frames that arrived by t0
            new = False
            while fi < len(frames_at) and frames_at[fi][0] <= t0:
                credits_arrived += frames_at[fi][1]; fi += 1; new = True
            cost = E0B + (F if new else 0) + rnd.uniform(-jit, jit)
            if k - credits_arrived < S:
                ts = max(t0, last_wire + WIRE)
                send_t.append(ts); arrive.append(ts + DFWD); last_wire = ts; k += 1
                t0 = ts + cost
            else:
                t0 += 20 + (F if new else 0)   # idle E0 iteration waiting for credits
            continue
        # E1 pass
        progress = False
        if retired < len(arrive) and arrive[retired] <= t1:
            issue_t[retired] = t1; retired += 1; t1 += Cp + rnd.uniform(0, jit); progress = True
        else:
            t1 += Ci
            if retired >= len(arrive) and k >= npk:
                pass
        unc = retired - credited
        if progress:
            idle = 0; check = unc > 4
        else:
            if unc != 0 and unc < 4: idle += 1
            check = unc >= 4 or (unc != 0 and idle >= H)
        if not check: continue
        batch = min(unc, 4)
        newest = credited + batch - 1
        if issue_t[newest] + WR > t1: continue
        idle = 0; t1 += CC; nframes += 1
        frames_at.append((t1 + DRET, batch)); credited += batch
    return (send_t[-1] - send_t[1000]) / (len(send_t) - 1001), nframes / credited

if __name__ == "__main__":
    for H in (4, 8):
        print(f"hold {H} idle passes")
        for Cp in (95, 90, 87, 84, 81, 78, 72):
            r = [run(Cp, H, seed=s) for s in (1, 2, 3)]
            print(f"   E1 per packet {Cp}: {sum(x[0] for x in r)/3:7.2f} cycles/packet, {sum(x[1] for x in r)/3:.3f} frames/packet")
