#!/usr/bin/env python3
"""table.py: §3.1/§3.2 tables from fwdcount outputs: per image, loop size and the first forward copy's executed path,
register writes, loads, counter stores, for lead / (a) / (b). Executed path = instructions outside the not-ready wait
loops; for the stateless forward its polls run once, so path = all its command instructions."""
import re, subprocess, sys
S = "/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/ns"
reps = [l.split(None, 1) for l in open(f"{S}/reps2.txt") if l.strip()]
builds = [("lead", "v4rxg3_O3", "v4rxg3"), ("(a)", "v4rxnsa_O3", "v4rxnsa"), ("(b)", "v4rxnsb_O3", "v4rxnsb")]
res = {}
for tag, b, v in builds:
    out = subprocess.run([sys.executable, f"{S}/fwdcount.py", b, v] + [r[0] for r in reps], capture_output=True, text=True).stdout
    open(f"{S}/fwd_final_{tag.strip('()')}.txt", "w").write(out)
    cur = None
    for l in out.splitlines():
        m = re.match(r"(\S+): loop (\d+), forward copies (\d+)", l)
        if m:
            cur = m.group(1); res[(tag, cur)] = {"loop": int(m.group(2)), "copies": int(m.group(3)), "fwd": None}
            continue
        m = re.match(r"\s+path\s+(\d+) cmd\s+(\d+) \(reg_w\s+(\d+) st\s+(\d+) ready_r\s+(\d+) ld\s+(\d+) br\s+(\d+) alu\s+(\d+); wait\s+(\d+)\)", l)
        if m and res[(tag, cur)]["fwd"] is None:
            path, cmd, regw, st, rr, ld, br, alu, wait = map(int, m.groups())
            stateless = rr <= 2 and wait <= 6 and regw >= 12
            res[(tag, cur)]["fwd"] = dict(path=cmd if stateless else path, regw=regw, ld=ld, st=st,
                                          kind="stateless" if stateless else ("full" if regw <= 7 else "partial"))
print("| E1 image | shape | lead: forward (instr / reg writes / loads / counter st) | (a) | (b) | loop lead / (a) / (b) |")
print("| --- | --- | --- | --- | --- | --- |")
for cfg, shape in reps:
    cells = []
    for tag, _, _ in builds:
        f = res[(tag, cfg)]["fwd"]
        cells.append(f"{f['kind']} {f['path']} / {f['regw']} / {f['ld']} / {f['st']}" if f else "-")
    loops = " / ".join(str(res[(t, cfg)]["loop"]) for t, _, _ in builds)
    print(f"| {cfg.split('__')[0]} {cfg.split('__')[1][:6]} | {shape.strip()} (x{res[('lead', cfg)]['copies']}) | " + " | ".join(cells) + f" | {loops} |")
