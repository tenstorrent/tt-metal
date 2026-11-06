#!/usr/bin/env python3
"""rf_matched.py: matched-count multi-VC rows for the report data refresh, generated inside the device job from the same
job's pass-a contended rates (so every flow of a combination finishes together and one run gives the link's combined
rate; consolidation/mk_mvc_matched.py and mvc3_matched.py did this between jobs).

  rf_matched.py vc02 RESULTS PREFIX ROUTER...   VC0 + VC2 on one 2D E/W link (pcs3 rows): from <PREFIX>_<ROUTER>_vc02ew<S>.log
                                               (VC0's rate under contention) and _vc02bew<S>.log (VC2's) writes
                                               RESULTS/<PREFIX>_<ROUTER>_vc02m<S>.yaml: VC0 4M packets, VC2 4M x c0 / c2
  rf_matched.py mvc3 RESULTS PREFIX SET...      the VC1 split rows (mvc3_yamls.py's flows): from <PREFIX>_<SET>_{base,lag1,
                                               lag0}.log writes RESULTS/<PREFIX>_<SET>_match.yaml (mvc3_matched.py's
                                               rule: N_i = 1.5e9 / r_i, r_i each flow's rate under contention)
Sizes 256, 1024, 4352 B."""
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
FE = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(FE, "consolidation"))
sys.path.insert(0, os.path.join(FE, "report", "scripts"))
SIZES = (256, 1024, 4352)
PAT = re.compile(r"\((VC\d)\)(?: on device .*?:| with) (\d+) packets in (\d+) cycles")
N0 = 4_000_000

VC02 = """# VC0 + VC2 with matched counts ({router}, {size} B; fabric_exp/report/refresh/rf_matched.py, from this job's pass a):
# VC0 {n0} packets, VC2 {n2}: VC0 / VC2 cycles per packet under contention {c0:.1f} / {c2:.1f}, so both senders finish
# together and the summary CSV is the link's combined rate.
allocation_policies:
  receiver:
    policy: RoundRobin
    max_configs_per_core: 4
Tests:
- name: PcsVC02M{size}
  benchmark_mode: true
  sync: false  # each sender times its own run
  fabric_setup:
    topology: Mesh
    use_vc2: true
  parametrization_params:
    ntype:
    - unicast_write
    size:
    - {size}
  defaults:
    ftype: unicast
    num_packets: {n0}
  senders:
  - device:
    - 0
    - 0
    vc_id: 0
    patterns:
    - num_packets: {n0}
      destination:
        device:
        - 0
        - 1
  - device:
    - 0
    - 0
    vc_id: 2
    patterns:
    - num_packets: {n2}
      destination:
        device:
        - 0
        - 1
"""


def rate(path, vc):
    for line in open(path, errors="replace"):
        m = PAT.search(re.sub(r"\x1b\[[0-9;]*m", "", line))
        if m and m.group(1) == vc:
            return int(m.group(3)) / int(m.group(2))
    raise SystemExit(f"no {vc} line in {path}")


def vc02(res, prefix, routers):
    for r in routers:
        for s in SIZES:
            c0 = rate(os.path.join(res, f"{prefix}_{r}_vc02ew{s}.log"), "VC0")
            c2 = rate(os.path.join(res, f"{prefix}_{r}_vc02bew{s}.log"), "VC2")
            n2 = int(round(N0 * c0 / c2, -3))
            p = os.path.join(res, f"{prefix}_{r}_vc02m{s}.yaml")
            open(p, "w").write(VC02.format(router=r, size=s, n0=N0, n2=n2, c0=c0, c2=c2))
            print(f"{os.path.basename(p)}: VC0 {N0}, VC2 {n2} (c0 {c0:.1f}, c2 {c2:.1f})")


def mvc3(res, prefix, sets):
    import build_multivc3 as b
    import mvc3_yamls as y
    b.RES = res
    T = 1.5e9
    for s in sets:
        three = s.startswith("A")
        rows = {}
        for g in ("base", "lag1", "lag0"):
            r = b.parse(f"{prefix}_{s}_{g}")
            if r is None:
                sys.exit(f"no log {prefix}_{s}_{g}")
            rows.update(r[0])
        flows = [0, 1, 2] if three else [0, 1]
        tests, info = [], []
        for size in SIZES:
            cnt = {}
            for f in flows:
                fl = rows[(s, f"T012m{f}" if three else f"P01m{f}", size)]
                cnt[f] = int(round(T / (fl[f][1] / fl[f][0]) / 1e3) * 1e3)
            t = y.test_yaml(f"MVC3_{s}_M{''.join(map(str, flows))}_s{size}", flows, cnt, three)
            tests.append(t.replace(f"size: {y.SIZES}", f"size: [{size}]"))
            info.append(f"{size} B: " + ", ".join(f"F{f} {cnt[f]} (r {T / cnt[f]:.1f})" for f in flows))
        p = os.path.join(res, f"{prefix}_{s}_match.yaml")
        open(p, "w").write(f"# fabric_exp/report/refresh/rf_matched.py from this job's pass a (set {s}): N_i = {T:.1e} / r_i.\n"
                           "# " + "; ".join(info) + "\nallocation_policies:\n  receiver:\n    policy: RoundRobin\n"
                           "    max_configs_per_core: 4\nTests:\n" + "\n\n".join(tests) + "\n")
        print(os.path.basename(p), "|", "; ".join(info))


if __name__ == "__main__":
    mode, res, prefix, *names = sys.argv[1:]
    {"vc02": vc02, "mvc3": mvc3}[mode](res, prefix, names)
