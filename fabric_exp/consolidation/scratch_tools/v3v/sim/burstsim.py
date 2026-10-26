#!/usr/bin/env python3
"""bsim.py: closed-loop model of a 1-hop stream through E0 and E1 with the worker's credit window, for bursty
senders (N packets, then G idle cycles). Pass-level E0 and E1, explicit E1 completion policy.

Policy fields: hold (idle passes before a partial batch), cut (batch ends at the record word's end), wait (a full
batch whose newest write has not left waits for it, bounded), T (batch target), ring (E1 slots)."""
import heapq, sys, statistics, random

DEF = dict(c=53.0, o=55.0, d_w=40.0, s0=60.0, p0=25.0, c_cw=30.0, d_n=40.0, d_f=560.0, s1=80.0, w_leave=20.0,
           p1=22.0, c_f=45.0, d_r=560.0, W=17, ring=16, jit=6.0)

POL = {
    'v3tk': dict(hold=4, cut=False, wait=False, T=4),
    'v3v6': dict(hold=8, cut=True, wait=True, T=4),
    'v3x':  dict(hold=4, cut=True, wait=True, T=4),
}


def run(pol, N, G, npk=6000, prm=None, seed=1, trace=False):
    p = dict(DEF); p.update(prm or {})
    rnd = random.Random(seed)
    J = lambda: rnd.uniform(0, p['jit'])
    T, hold, cut, wait = pol['T'], pol['hold'], pol['cut'], pol['wait']
    R = p['ring']
    # worker
    w_send = []              # time worker made packet k available (slot written)
    credit_seen = []         # (time, cumulative credits) visible to worker, appended in time order
    # E0 -> E1
    e1_arr = []              # arrival time of packet k at E1 ring
    frames = []              # (arrival time at E0, n)
    stats = dict(frames=0, partial=0, cut=0, waits=0, blocked=0.0, credit_delay=[])
    retire_t = {}
    # --- worker: compute send times lazily as credits become known; simplest: co-simulate by time stepping all actors
    t_w = 0.0; k_w = 0; lap_pos = 0
    e0_t = 0.0; e0_next = 0; far_free = R; e0_credited = 0; fi = 0
    e1_t = 0.0; e1_next = 0; unc = 0; credited = 0; idle_polls = 0; issue = {}
    worker_credits = 0  # cumulative credits worker can see at time t (built from credit_seen)
    cs_i = 0
    def worker_free(t):
        nonlocal cs_i, worker_credits
        while cs_i < len(credit_seen) and credit_seen[cs_i][0] <= t:
            worker_credits = credit_seen[cs_i][1]; cs_i += 1
        return p['W'] - (k_w - worker_credits)
    done = False
    while credited < npk - 16:
        # advance the actor with the smallest local time
        tmin = min(t_w if k_w < npk else 1e18, e0_t, e1_t)
        if k_w < npk and t_w == tmin:
            if worker_free(t_w) > 0:
                w_send.append(t_w + p['d_w']); k_w += 1; lap_pos += 1
                t_w += p['c'] + J()
                if lap_pos == N:
                    lap_pos = 0; t_w += p['o'] + G
            else:
                # wait until next credit visible
                nxt = credit_seen[cs_i][0] if cs_i < len(credit_seen) else None
                if nxt is None or nxt <= t_w:
                    # credits not yet produced: step to the earliest of the other actors
                    nt = min(e0_t, e1_t)
                    stats['blocked'] += max(nt, t_w + 1) - t_w
                    t_w = max(nt, t_w + 1)
                else:
                    stats['blocked'] += nxt - t_w
                    t_w = nxt
            continue
        if e0_t == tmin:
            t = e0_t
            did = False
            if e0_next < len(w_send) and w_send[e0_next] <= t and far_free > 0:
                t += p['s0'] + J(); far_free -= 1
                e1_arr.append(t + p['d_f']); e0_next += 1; did = True
            n = 0
            while fi < len(frames) and frames[fi][0] <= t:
                n += frames[fi][1]; fi += 1
            if n:
                far_free += n; e0_credited += n
                t += p['c_cw']
                credit_seen.append((t + p['d_n'], e0_credited)); did = True
            if not did:
                t += p['p0']
            e0_t = t
            continue
        # E1 pass
        t = e1_t
        progress = False
        if e1_next < len(e1_arr) and e1_arr[e1_next] <= t:
            t += p['s1'] + J()
            issue[e1_next] = t; retire_t[e1_next] = t
            e1_next += 1; unc += 1; progress = True
        else:
            t += p['p1']
        # completions
        if progress:
            idle_polls = 0
            check = unc > T
        else:
            if unc >= T:
                check = True
            elif unc != 0:
                idle_polls += 1
                check = idle_polls >= hold
            else:
                check = False
        if check:
            batch = min(unc, T)
            base = credited % R
            if cut and T == 4 and R % 4 == 0:
                b2 = min(batch, 4 - (base & 3))
                if b2 < batch: stats['cut'] += 1
                batch = b2
            newest = credited + batch - 1
            leave = issue[newest] + p['w_leave']
            ok = True
            if leave > t:
                if wait and batch == T and leave - t <= 256:
                    stats['waits'] += 1; t = leave
                else:
                    ok = False
            if ok:
                idle_polls = 0
                t += p['c_f']
                frames.append((t + p['d_r'], batch)); stats['frames'] += 1
                if batch < T: stats['partial'] += 1
                for k in range(credited, credited + batch):
                    stats['credit_delay'].append(t - retire_t[k])
                credited += batch; unc -= batch
        e1_t = t
    # measure: worker time per packet over the middle of the run
    a, b = npk // 4, 3 * npk // 4
    rate = (w_send[b] - w_send[a]) / (b - a)
    return rate, stats


if __name__ == '__main__':
    cases = [(1, 0), (4, 500), (8, 500), (32, 500), (8, 2000), (32, 2000)]
    for N, G in cases:
        row = []
        for name in ('v3tk', 'v3v6', 'v3x'):
            r, s = run(POL[name], N, G)
            row.append('%s %6.1f (fr/pk %.2f part %d cut %d)' % (name, r, s['frames'] / 6000, s['partial'], s['cut']))
        print('N%-2d G%-4d ' % (N, G) + ' | '.join(row))
