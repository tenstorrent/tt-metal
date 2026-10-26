#!/usr/bin/env python3
"""credsim.py: a pass-level model of E1's completion batching on a 16-slot receiver ring, fed by a sender with S slots
(closed loop: the sender may send packet k once the credit for packet k-S is back). Compares batching policies:
  T  v3t: progress -> batch of 4 once >4 wait; idle -> 4 at once if >=4 wait, else all after 4 idle passes (2D);
          a batch may straddle a record word
  V  v3v: as T, but every batch is cut at the end of the oldest slot's record word (4-slot groups)
  P  v3v2: as V, but after a cut the next idle pass credits the rest
  E  cut only on progress passes; idle passes as T
Reports per policy: packets/kcycle, mean credit delay (retire -> credit frame), cut batches, batch-size histogram."""
import random, sys, statistics

def run(policy, L, S=16, N=16, Cp=75, Ci=30, Cc=50, W=85, Dfwd=600, Dret=600, npk=20000, jitter=20, seed=1, idle_thr=4):
    rnd = random.Random(seed)
    t = 0.0
    sent_t = {}          # packet -> time sender sent it
    arrive = {}          # packet -> arrival at E1
    credit_back = {}     # packet -> time its credit reaches the sender
    next_send = 0; last_send_t = -1e9
    retired = 0          # packets retired by E1 (next slot index = retired % N)
    retire_t = {}
    issue_t = {}
    credited = 0         # packets credited
    idle_polls = 0
    cuts = 0; hist = {}
    delays = []
    sender_free = 0.0
    def sender_progress(now):
        nonlocal next_send, last_send_t
        # the sender sends as soon as it has a credit and the link is free (period L with jitter)
        while next_send < npk:
            k = next_send
            need = k - S
            ready = 0.0 if need < 0 else credit_back.get(need)
            if ready is None:
                return
            ts = max(last_send_t + L + rnd.uniform(-jitter, jitter), ready)
            if ts > now + 10000:
                return
            sent_t[k] = ts; arrive[k] = ts + Dfwd; last_send_t = ts; next_send += 1
    while credited < npk - 8:
        sender_progress(t)
        progress = False
        if retired < next_send and arrive.get(retired, 1e18) <= t and retired - credited < N:
            # process one packet
            issue_t[retired] = t; retire_t[retired] = t + Cp; retired += 1; t += Cp + rnd.uniform(0, jitter)
            progress = True
        else:
            t += Ci + rnd.uniform(0, jitter / 2)
        unc = retired - credited
        if progress:
            idle_polls = 0
            check = unc > 4
        else:
            idle_polls += 1 if unc != 0 and unc < 4 else 0
            check = unc >= 4 or (unc != 0 and idle_polls >= idle_thr)
        if not check:
            continue
        base = credited % N
        batch = min(unc, 4)
        cut_now = False
        if policy in ("V", "P") or (policy == "E" and progress) or (policy == "G" and unc > 4):
            b2 = min(batch, 4 - (base & 3))
            if b2 < batch: cuts += 1; cut_now = True
            batch = b2
        newest = credited + batch - 1
        if issue_t[newest] + W > t:      # newest slot's write still outgoing: no credit this pass
            continue
        idle_polls = idle_thr - 1 if (policy == "P" and cut_now) else 0
        t += Cc
        for k in range(credited, credited + batch):
            credit_back[k] = t + Dret
            delays.append(t - retire_t[k])
        hist[batch] = hist.get(batch, 0) + 1
        credited += batch
    n = credited
    thr = n / (t / 1000.0)
    return thr, statistics.mean(delays), cuts, dict(sorted(hist.items()))

if __name__ == "__main__":
    for L in (125.7, 115.0):
        for Dl in (500, 700, 800):
            print(f"L={L} Dfwd=Dret={Dl}")
            for pol in ("T", "V", "P", "E"):
                thr, d, cuts, h = run(pol, L, Dfwd=Dl, Dret=Dl)
                print(f"   {pol}: {1000/thr:7.2f} cycles/pkt  mean credit delay {d:6.1f}  cuts {cuts:5d}  batches {h}")
