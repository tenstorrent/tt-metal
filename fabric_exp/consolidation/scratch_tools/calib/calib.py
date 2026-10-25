import csv, os, re, statistics, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loopstat import stat
X = "fabric_exp"
V = {"s74c": ["dq74c", "dq74cr2", "dq74cr3"], "s77t": ["dq77t"], "s77s": ["dq77s", "dq77sv", "dq77s3", "dq77s4"],
     "s77sn": ["dq77sn"], "s77u": ["dq77u"], "s77": ["dq77", "dq77r2"], "s78w": ["dq78w"], "s78l": ["dq78l"],
     "s78": ["dq78", "dq78r2"], "s78n": ["dq78n"], "s78n0": ["dq78n0"], "s77si": ["dq77si"], "s77si0": ["dq77si0"],
     "s79c": ["dq79c", "dq79cr2"], "s79m": ["dq79m", "dq79mr2", "dq79m3"], "s79": ["dq79", "dq79r2"], "s80": ["dq80", "dq80r2"]}
ROWS = ["EW_PureWrite16", "EW_PureWrite1024", "EW_MixSize4", "NS_PureWrite16", "FWD_PureWrite16"]
def row(lbls, name):
    xs = []
    for l in lbls:
        f = f"{X}/results/{l}_13129088_mix.csv"
        if os.path.exists(f):
            for r in csv.DictReader(open(f)):
                if r["test_name"] == name: xs.append(1.35e9 / float(r["avg_packets_per_s"]))
    return statistics.mean(xs) if xs else float("nan")
H = "10733501891889981293"
print(f"{'var':7s} {'loop':>5s} {'ldsp':>5s} {'stsp':>5s} " + " ".join(f"{r[:14]:>14s}" for r in ROWS))
for v, l in V.items():
    e = f"{X}/variants/{v}/elf/{H}/active_erisc.elf"
    if not os.path.exists(e): continue
    m = re.search(r"(\d+) insns, (\d+) sp loads, (\d+) sp stores", stat(e))
    print(f"{v:7s} {m[1]:>5s} {m[2]:>5s} {m[3]:>5s} " + " ".join(f"{row(l, r):14.1f}" for r in ROWS))
