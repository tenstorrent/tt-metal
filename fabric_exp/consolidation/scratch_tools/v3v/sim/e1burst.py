#!/usr/bin/env python3
"""e1only.py: E1's completion batching alone, for periodic bursts of N packets arriving `a` cycles apart (optionally a
gap of `gb` cycles before packet j of each burst), bursts every P cycles. Per policy: frames per burst, and the credit
time of the burst's last packet after its arrival (the burst-tail credit latency), averaged over steady state."""
import sys, statistics

def run(pol, N=8, a=100.0, P=1040.0, s1=62.0, p1=22.0, c_f=45.0, w_leave=20.0, ring=16, j=None, gb=0.0, nb=60):
    T, hold, cut, wait = pol['T'], pol['hold'], pol['cut'], pol['wait']
    arr = []
    for b in range(nb):
        t = b * P
        for k in range(N):
            if j is not None and k == j:
                t += gb
            arr.append(t); t += a
    t = 0.0; nxt = 0; unc = 0; cred = 0; idle = 0; issue = {}; credit_t = {}; frames = []
    while cred < len(arr) - N:
        progress = False
        if nxt < len(arr) and arr[nxt] <= t:
            t += s1; issue[nxt] = t; nxt += 1; unc += 1; progress = True
        else:
            # jump idle passes in bulk when nothing pending and nothing to check
            t += p1
        if progress:
            idle = 0; check = unc > T
        else:
            if unc >= T: check = True
            elif unc: idle += 1; check = idle >= hold
            else: check = False
        if not check: continue
        batch = min(unc, T); base = cred % ring
        if (cut is True or (cut == 'progress' and progress)) and T == 4 and ring % 4 == 0:
            batch = min(batch, 4 - (base & 3))
        newest = cred + batch - 1
        leave = issue[newest] + w_leave
        if leave > t:
            if wait and batch == T: t = leave
            else: continue
        idle = 0; t += c_f
        for k in range(cred, cred + batch): credit_t[k] = t
        frames.append((t, batch)); cred += batch; unc -= batch
    tails = [credit_t[b * N + N - 1] - arr[b * N + N - 1] for b in range(nb // 3, nb - 2)]
    nfr = sum(1 for (tt, bb) in frames if nb // 3 * P <= tt < (nb - 2) * P) / (nb - 2 - nb // 3)
    allc = [credit_t[k] - arr[k] for k in range(nb // 3 * N, (nb - 2) * N)]
    return statistics.mean(tails), nfr, statistics.mean(allc)

POL = {'v3tk': dict(hold=4, cut=False, wait=False, T=4), 'v3v6': dict(hold=8, cut=True, wait=True, T=4),
       'v3x': dict(hold=4, cut=True, wait=True, T=4), 'v3x_nocut': dict(hold=4, cut=False, wait=True, T=4),
       'v3x2': dict(hold=4, cut='progress', wait=True, T=4)}
if __name__ == '__main__':
    for j, gb in ((None, 0), (5, 150), (5, 300), (1, 150), (1, 300), (2, 200), (6, 200)):
        print('gap before packet %s: %d' % (j, gb))
        for a in (60, 80, 100, 120):
            row = []
            for name, pol in POL.items():
                tl, nf, al = run(pol, a=a, j=j, gb=gb)
                row.append('%s tail %5.0f fr %.2f avg %5.0f' % (name, tl, nf, al))
            print('  a=%3d ' % a + ' | '.join(row))


def summary():
    """Mean over the scenario grid used in V3E1_NOTES §12.21."""
    out = {}
    grids = (('16B N8', 8, (1040, 1092), 62, 20, (60, 70, 80, 90, 100, 110, 120),
              ((None, 0), (1, 150), (1, 300), (5, 150), (5, 300), (2, 200), (6, 200), (3, 120), (4, 100))),
             ('4352 N32', 32, (4400,), 88, 80, (110, 130, 150), ((None, 0), (9, 200), (17, 300), (1, 200))))
    for label, N, Ps, s1, wl, As, gaps in grids:
        print('==', label)
        for name, pol in POL.items():
            tl = nf = al = 0; n = 0
            for a in As:
                for j, gb in gaps:
                    for P in Ps:
                        t1, f1, a1 = run(pol, N=N, a=a, P=P, s1=s1, w_leave=wl, j=j, gb=gb, nb=40)
                        tl += t1; nf += f1; al += a1; n += 1
            print('  %-10s burst-tail credit %4.0f  frames/burst %.2f  mean credit latency %4.0f' % (name, tl / n, nf / n, al / n))
