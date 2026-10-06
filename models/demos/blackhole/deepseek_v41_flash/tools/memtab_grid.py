import re

rows = []
for b in (4, 8, 16, 32, 64, 128):
    for g in ("G1", "G2a", "G2b"):
        f = f"/mnt/tt-data/ssinghal/dsv4-logs/grid_b{b}_{g}.log"
        try:
            L = open(f, errors="ignore").read().split("\n")
        except:
            continue
        cap = None
        seg = None
        mb = None
        d = {}
        for ln in L:
            m = re.search(r"=== session scenario (\S+) ===", ln)
            if m:
                seg = m.group(1)
                d.setdefault(seg, 0)
                continue
            m = re.search(r"MEMLOG (.*?)\s+allocated\s+([\d.]+) MiB/bank\s+free\s+([\d.]+)", ln)
            if m:
                a = float(m.group(2))
                fr = float(m.group(3))
                if m.group(1).strip() == "model built":
                    mb = a
                    cap = a + fr
                if seg:
                    d[seg] = max(d[seg], a)
        for s, v in d.items():
            rows.append((b, s, mb, cap, v))
for r in rows:
    print(
        "b%d %-12s modelbuilt %7.1f cap %7.1f peak %7.1f headroom %7.1f"
        % (r[0], r[1], r[2] or 0, r[3] or 0, r[4], (r[3] or 0) - r[4])
    )
