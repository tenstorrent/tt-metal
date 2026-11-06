#!/usr/bin/env python3
"""bt_report.py: the bubble-threshold sweep's tables (variants/v4txa11t/README.md).

  bt_report.py glx JOB_RESULTS_DIR     Galaxy (frs_glx_frgap.sbatch configs a11t2..a11t6): labels PASS_a11tN_SUITE
  bt_report.py p150 RESULTS_DIR        P150 (frg_bt_p150_runs.sh): labels frgbt<PASS>_TN_{ring,t24,t42}
  bt_report.py p150u RESULTS_DIR       P150 rule-cost arms (frg_bt_p150_runs.sh TAG=frgbu): A2, A16, C2, T8; % vs A2
  bt_report.py p150v RESULTS_DIR       P150 fraction-vs-cap arms (TAG=frgbv): A2 A16 C2 C8 G2 G32 G48; % vs A2
  bt_report.py bs JOB_RESULTS_DIR      Galaxy slot depth x threshold (configs bs<ARM><T>, RING_JAM_FIX.md §9.3):
                                       labels PASS_bs<ARM><T>_SUITE; % against bsA2 (today's slots at T 2)

Per row (suite, test, size, ntype): cycles/packet (1.35e9 / the summary CSV's avg_packets_per_s) per arm as
"pass a / pass b", each arm's mean against T2's mean (%), "*" where the two passes of an arm differ by more than 1%.
Then the validation results and the slots each run logged (TT_FABRIC_EXP_LOG_SLOTS=1)."""
import collections
import csv
import glob
import gzip
import os
import re
import sys

ARMS = ["T2", "T3", "T4", "T5", "T6"]
U_ARMS = ["A2", "A16", "C2", "T8"]
V_ARMS = ["A2", "A16", "C2", "C8", "G2", "G32", "G48"]
BS_ARMS = ["N2", "N8", "A2", "B2", "C2", "D2", "E2", "F2", "C6", "D6", "E6", "F6", "A8", "A12", "A16", "C8", "G2", "G32",
           "G48"]
# The 1D ring receiver slots of each bs arm (N: the defaults, 32 on the 1D ring): T / R and R - T per config.
BS_RECEIVER = {"A": 32, "B": 16, "C": 16, "D": 16, "E": 16, "F": 32, "G": 64, "N": 32}


def rates(path):
    out = {}
    for r in csv.DictReader(open(path)):
        out[(r["test_name"], int(r["packet_size"]), r.get("ntype", ""))] = 1.35e9 / float(r["avg_packets_per_s"])
    return out


def read_log(log):
    """A log's text, or its .gz copy's (copy_results.sh gzips logs over 1 MB); "" when neither exists."""
    if os.path.exists(log):
        return open(log, errors="replace").read()
    if os.path.exists(log + ".gz"):
        return gzip.open(log + ".gz", "rt", errors="replace").read()
    return ""


def slots(log):
    s = collections.Counter(re.findall(r"VC0 sender slots \{([0-9, ]+)\} receiver slots \{([0-9, ]+)\}",
                                       read_log(log)))
    return "; ".join(f"{{{a}}}/{{{b}}} x{n}" for (a, b), n in s.most_common(3))


def result_line(log):
    t = re.sub(r"\x1b\[[0-9;]*m", "", read_log(log))
    if not t:
        return "no log"
    m = re.findall(r"Tests: [0-9]+ executed, [0-9]+ failed", t)
    return m[-1] if m else "NO RESULT"


def main():
    mode, d = sys.argv[1], sys.argv[2]
    arm_order, base_arm = {"bs": (BS_ARMS, "A2"), "p150u": (U_ARMS, "A2"), "p150v": (V_ARMS, "A2")}.get(mode, (ARMS, "T2"))
    tag, arm_re = {"p150u": ("frgbu", r"(A2|A16|C2|T8)"), "p150v": ("frgbv", r"(A2|A16|C2|C8|G2|G32|G48)")}.get(
        mode, ("frgbt", r"(T[2-6])"))
    runs = collections.defaultdict(dict)  # (suite, arm) -> {pass: rates}
    logs, vals = {}, []
    if mode == "bs":
        for f in sorted(glob.glob(f"{d}/*_bs*_*.log") + glob.glob(f"{d}/*_bs*_*.log.gz")):
            f = f[:-3] if f.endswith(".gz") else f
            b = os.path.basename(f)[:-4]
            p, c, s = b.split("_", 2)
            arm = c[2:]
            if s.startswith("frg_val") or s == "check":
                vals.append((arm, s, result_line(f)))
                continue
            csvp = f[:-4] + ".csv"
            if os.path.exists(csvp):
                runs[(s, arm)][p] = rates(csvp)
                logs[(s, arm, p)] = slots(f)
    elif mode == "glx":
        for f in sorted(glob.glob(f"{d}/*_a11t[2-6]_*.log") + glob.glob(f"{d}/*_a11t[2-6]_*.log.gz")):
            f = f[:-3] if f.endswith(".gz") else f
            b = os.path.basename(f)[:-4]
            p, c, s = b.split("_", 2)
            arm = "T" + c[-1]
            if s == "frg_val":
                vals.append((arm, s, result_line(f)))
                continue
            if s == "check":
                vals.append((arm, s, result_line(f)))
            csvp = f[:-4] + ".csv"
            if os.path.exists(csvp) and s != "check":
                runs[(s, arm)][p] = rates(csvp)
                logs[(s, arm, p)] = slots(f)
    else:
        for f in sorted(glob.glob(f"{d}/{tag}*_*_*.log") + glob.glob(f"{d}/{tag}*_*_*.log.gz")):
            f = f[:-3] if f.endswith(".gz") else f
            b = os.path.basename(f)[:-4]
            m = re.match(tag + r"(\w)_" + arm_re + r"_(ring35|ring|t24|t42)$", b)
            if not m:
                continue
            p, arm, s = m.groups()
            if p == "v":
                vals.append((arm, s, result_line(f)))
                continue
            csvp = f[:-4] + ".csv"
            if os.path.exists(csvp):
                runs[(s, arm)][p] = rates(csvp)
                logs[(s, arm, p)] = slots(f)
    suites = sorted({s for s, _ in runs})
    for s in suites:
        arms = [a for a in arm_order if (s, a) in runs]
        if mode == "bs" and (s, base_arm) not in runs and (s, "N2") in runs:
            base_arm = "N2"  # a suite run without arm A (e.g. the torus rows: N2 vs N8)
        keys = sorted({k for a in arms for pr in runs[(s, a)].values() for k in pr})
        print(f"\n## {s}")
        print("| test | size | ntype | " + " | ".join(arms) + " |")
        print("| --- | ---: | --- | " + " | ".join("---:" for _ in arms) + " |")
        for k in keys:
            base = [runs[(s, base_arm)][p][k] for p in sorted(runs.get((s, base_arm), {})) if k in runs[(s, base_arm)][p]]
            bm = sum(base) / len(base) if base else None
            cells = []
            for a in arms:
                v = [runs[(s, a)][p][k] for p in sorted(runs[(s, a)]) if k in runs[(s, a)][p]]
                if not v:
                    cells.append("-")
                    continue
                m = sum(v) / len(v)
                flag = "*" if len(v) > 1 and (max(v) - min(v)) / min(v) > 0.01 else ""
                rel = f" ({100 * (m - bm) / bm:+.1f}%)" if bm and a != base_arm else ""
                cells.append(" / ".join(f"{x:.1f}" for x in v) + flag + rel)
            nt = k[2].replace("NOC_UNICAST_", "").replace("unicast_", "").lower()
            print(f"| {k[0]} | {k[1]} | {nt} | " + " | ".join(cells) + " |")
        for a in arms:
            for p in sorted(runs[(s, a)]):
                print(f"   slots {p} {a} {s}: {logs.get((s, a, p), '')}")
    if mode in ("bs", "p150v"):
        seen = [a for a in arm_order if any(a == x for _, x in runs)]
        print("\n## T / R and R - T (1D ring receiver R; T = the injection threshold)")
        for a in seen:
            r, t = BS_RECEIVER[a[0]], int(a[1:])
            print(f"   {a}: R {r}, T {t}: T/R {t / r:.3f}, R - T {r - t}")
    if vals:
        print("\n## validation")
        for arm, s, r in sorted(vals, key=lambda v: (arm_order.index(v[0]) if v[0] in arm_order else 99, v[1])):
            print(f"   {arm} {s}: {r}")


if __name__ == "__main__":
    main()
