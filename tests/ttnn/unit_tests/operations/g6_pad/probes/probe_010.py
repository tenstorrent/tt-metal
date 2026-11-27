from ttnn.operations.g6_pad.g6_pad_program_descriptor import band_runs, _FillEnumeration, _balanced

es = 2
rb = 16 * es
fb = 16 * rb


def face_bytes_of_rows(a, b):
    s = set()
    for r in range(a, b):
        for c in range(32):
            f = (r // 16) * 2 + (c // 16)
            off = f * fb + (r % 16) * rb + (c % 16) * es
            s.update(range(off, off + es))
    return s


for a in range(1, 32):
    for b in range(a + 1, 33):
        runs = band_runs(a, b, es)
        got = set()
        for off, ln in runs:
            got.update(range(off, off + ln))
        assert got == face_bytes_of_rows(a, b), (a, b, runs)
        assert len(runs) <= 3
print("band runs ok; examples", band_runs(8, 16, 2), band_runs(2, 32, 2), band_runs(4, 20, 2), band_runs(16, 32, 2))


def brute(lead_in, lead_out, ht_in, wt_in, ht_out, wt_out):
    out = []
    for o0 in range(lead_out[0]):
        for o1 in range(lead_out[1]):
            img = o0 < lead_in[0] and o1 < lead_in[1]
            for ht in range(ht_out):
                for wt in range(wt_out):
                    if img and ht < ht_in and wt < wt_in:
                        continue
                    out.append((o0 * lead_out[1] + o1, ht, wt))
    return out


import itertools

n = 0
for li0, li1, lo0x, lo1x, hi, wi, hox, wox in itertools.product(
    [1, 2], [1, 3], [0, 2], [0, 1], [1, 3], [1, 2], [0, 1], [0, 2]
):
    lead_in = (li0, li1)
    lead_out = (li0 + lo0x, li1 + lo1x)
    ho = hi + hox
    wo = wi + wox
    ref = brute(lead_in, lead_out, hi, wi, ho, wo)
    e = _FillEnumeration(lead_in, lead_out, hi, wi, ho, wo)
    for f, c in enumerate(ref):
        assert e.coords(f) == c, (lead_in, lead_out, hi, wi, ho, wo, f, e.coords(f), c)
    total = len(ref)
    for N in (1, 3, 7):
        for k in range(N):
            start, cnt = _balanced(total, N, N - 1 - k)
            if cnt == 0:
                continue
            o, ht, wt = e.coords(start)
            o0, o1 = divmod(o, lead_out[1])
            seq = []
            for f in range(cnt):
                seq.append((o0 * lead_out[1] + o1, ht, wt))
                wt += 1
                while wt == wo and f + 1 < cnt:
                    ht += 1
                    if ht == ho:
                        ht = 0
                        o1 += 1
                        if o1 == lead_out[1]:
                            o1 = 0
                            o0 += 1
                    wt = wi if (o0 < lead_in[0] and o1 < lead_in[1] and ht < hi) else 0
            assert seq == ref[start : start + cnt], (lead_in, lead_out, hi, wi, ho, wo, N, k)
    n += 1
print("fill enumeration ok over", n, "geometries")
