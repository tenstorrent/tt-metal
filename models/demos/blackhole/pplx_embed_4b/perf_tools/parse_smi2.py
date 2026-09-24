import datetime
import glob
import json
import os
import re
import sys

dev = int(sys.argv[1])
logf = sys.argv[2]
files = sorted(glob.glob("/tmp/smi_samples/*.json"))
rows = []
lim = None
for f in files:
    raw = open(f).read()
    i = raw.find("{")
    if i < 0:
        continue
    try:
        d = json.loads(raw[i:])
    except Exception:
        continue
    di = d.get("device_info") or []
    if len(di) <= dev:
        continue
    x = di[dev]
    t = x.get("telemetry", {})
    sm = x.get("smbus_telem", {})
    if lim is None and sm:
        hx = lambda v: int(str(v), 16) if isinstance(v, str) and v.startswith("0x") else v
        lim = {
            k: hx(sm.get(k))
            for k in ("BOARD_POWER_LIMIT", "AICLK_LIMIT_MAX", "AICLK_ARB_MAX", "AICLK_ARB_MIN", "TDP", "TDC")
        }
    rows.append(
        (
            float(os.path.basename(f)[:-5]),
            t.get("aiclk"),
            t.get("power"),
            t.get("board_power"),
            t.get("current"),
            t.get("asic_temperature"),
        )
    )
print("samples:", len(rows), "limits:", lim)
t0 = rows[0][0] if rows else 0
its = []
for m in re.finditer(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}).*Iteration (\d+): ([0-9.]+)ms",
    open(logf, "rb").read().decode("utf-8", "ignore"),
):
    its.append(
        (
            datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").timestamp() - t0,
            int(m.group(2)),
            float(m.group(3)),
        )
    )
first_it = its[0][0] if its else 0
last_it = its[-1][0] if its else 0
print(
    f"{'t':>8s} {'aiclk MHz':>10s} {'power W':>8s} {'board W':>8s} {'current A':>9s} {'temp C':>7s}   iterations in window"
)
for j, (ts, a, p, bp, c, tmp) in enumerate(rows):
    if (ts - t0) < first_it - 6 or (ts - t0) > last_it + 3:
        continue
    nxt = rows[j + 1][0] if j + 1 < len(rows) else ts + 2
    win = [f"it{n}={v:.0f}" for (tt, n, v) in its if ts <= tt < nxt]
    print(f"t+{ts-t0:6.1f} {a!s:>10s} {p!s:>8s} {bp!s:>8s} {c!s:>9s} {tmp!s:>7s}   {' '.join(win)}")
