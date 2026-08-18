# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Mechanical submission check for the ModernBERT bounty.

Every claim that ships - in the demo README, the shipped source, and the
submission texts - is checked against the artifact it came from. Run it instead
of trusting a summary.

    python models/demos/modernbert/tests/check_submission.py [--docs DIR]

--docs points at the directory holding validation.log, the two perf CSVs and the
submission markdown. Checks whose inputs are absent are reported SKIP, never
silently passed.

Exit code 0 only when nothing is FAIL.
"""

import argparse
import collections
import csv
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
MODEL = REPO / "models" / "demos" / "modernbert"

results = []


def record(status, name, detail=""):
    results.append((status, name, detail))


def sh(*cmd, cwd=REPO):
    out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return out.stdout.strip()


# --------------------------------------------------------------------------
# 1. Repository state
# --------------------------------------------------------------------------
def check_repo():
    dirty = sh("git", "status", "--porcelain")
    record("FAIL" if dirty else "PASS", "working tree clean", dirty.replace("\n", "; ")[:120])

    branch = sh("git", "rev-parse", "--abbrev-ref", "HEAD")
    unpushed = sh("git", "rev-list", "--count", f"origin/{branch}..{branch}")
    record("FAIL" if unpushed not in ("0", "") else "PASS", "no unpushed commits", f"{unpushed} ahead")
    return sh("git", "rev-parse", "HEAD")


# --------------------------------------------------------------------------
# 2. Artifacts describe the current commit
# --------------------------------------------------------------------------
def check_artifact_commit(docs, head):
    log = docs / "modernbert_validation.log"
    if not log.exists():
        record("SKIP", "validation log names HEAD", "log absent")
        return
    m = re.search(r"commit:\s*([0-9a-f]{7,40})", log.read_text())
    if not m:
        record("FAIL", "validation log names HEAD", "no commit line")
        return
    ok = head.startswith(m.group(1)) or m.group(1).startswith(head[:8])
    record("PASS" if ok else "FAIL", "validation log names HEAD", f"log={m.group(1)[:12]} head={head[:12]}")

    for name in ("modernbert_pr_comment.md", "modernbert_performance_report.md", "modernbert_validation_summary.md"):
        f = docs / name
        if not f.exists():
            record("SKIP", f"{name} names HEAD", "absent")
            continue
        # only backticked tokens containing a letter are git hashes; bare digit
        # runs are PCC values and must not be matched
        hashes = {h for h in re.findall(r"`([0-9a-f]{8,40})`", f.read_text()) if re.search(r"[a-f]", h)}
        stale = [h for h in hashes if not head.startswith(h)]
        record("PASS" if not stale else "FAIL", f"{name} names HEAD", f"stale: {sorted(stale)[:3]}")


# --------------------------------------------------------------------------
# 3. Shipped docs carry no superseded absolutes or session narrative
# --------------------------------------------------------------------------
NARRATIVE = [
    r"earlier revision",
    r"\bwas tried\b",
    r"both tried",
    r"turned out",
    r"used to be",
    r"re-checked after",
    r"first reported",
    r"\bwe (tried|measured|found|chose|kept)\b",
]
# whole-model millisecond figures from baselines that no longer exist
SUPERSEDED = [r"30\.39", r"32\.17", r"43\.64", r"47\.33", r"56\.62", r"8\.83 ms", r"21\.62 →", r"391\.7"]


def check_language(docs):
    targets = sorted(MODEL.rglob("*.py")) + [MODEL / "README.md"]
    targets = [t for t in targets if "__pycache__" not in str(t) and t.name != Path(__file__).name]
    for pat_list, label in ((NARRATIVE, "no session narrative"), (SUPERSEDED, "no superseded figures")):
        hits = []
        for f in targets:
            text = f.read_text(errors="ignore")
            for pat in pat_list:
                for m in re.finditer(pat, text, re.I):
                    line = text[: m.start()].count("\n") + 1
                    hits.append(f"{f.relative_to(REPO)}:{line}")
        record("PASS" if not hits else "FAIL", f"shipped files: {label}", "; ".join(hits[:4]))


# --------------------------------------------------------------------------
# 4. README accuracy table matches the validation log
# --------------------------------------------------------------------------
def check_pcc_against_log(docs):
    log = docs / "modernbert_validation.log"
    if not log.exists():
        record("SKIP", "README PCCs appear in the log", "log absent")
        return
    logged = set(re.findall(r"PCC=([0-9]\.[0-9]{6,10})", log.read_text()))
    logged_r = {round(float(v), 8) for v in logged}
    readme = (MODEL / "README.md").read_text()
    table = readme.split("## Accuracy")[1].split("### Negative controls")[0] if "## Accuracy" in readme else ""
    claimed = {round(float(v), 8) for v in re.findall(r"\b(0\.99[0-9]{4,8})\b", table)}
    missing = sorted(c for c in claimed if not any(abs(c - l) < 5e-8 for l in logged_r))
    record(
        "PASS" if not missing else "FAIL",
        "README PCCs appear in the log",
        f"{len(claimed)} claimed, unmatched: {missing[:4]}",
    )


# --------------------------------------------------------------------------
# 5. The CI gate value is inside the measured device-perf report
# --------------------------------------------------------------------------
def check_submission_pccs(docs):
    """Every PCC quoted in a submission text must appear in the validation log."""
    log = docs / "modernbert_validation.log"
    if not log.exists():
        record("SKIP", "submission PCCs appear in the log", "log absent")
        return
    logged = {round(float(v), 8) for v in re.findall(r"PCC=([0-9]\.[0-9]{4,10})", log.read_text())}
    for name in ("modernbert_pr_description.md", "modernbert_pr_comment.md", "modernbert_performance_report.md"):
        f = docs / name
        if not f.exists():
            record("SKIP", f"{name} PCCs appear in the log", "absent")
            continue
        claimed = {round(float(v), 8) for v in re.findall(r"\b(0\.9[0-9]{3,8})\b", f.read_text())}

        # a claim may be a correctly-rounded form of a logged value
        def matches(c, text):
            dp = len(text.split(".")[1])
            return any(round(l, dp) == c for l in logged)

        claimed_txt = set(re.findall(r"\b(0\.9[0-9]{3,8})\b", f.read_text()))
        missing = sorted(t for t in claimed_txt if not matches(float(t), t))
        record(
            "PASS" if not missing else "FAIL",
            f"{name} PCCs appear in the log",
            f"{len(claimed_txt)} claimed, unmatched: {missing[:4]}",
        )


def check_gate(docs):
    test = (MODEL / "tests" / "test_modernbert_device_perf.py").read_text()
    m = re.search(r"\[8,\s*256,\s*([0-9.]+),", test)
    if not m:
        record("FAIL", "gate value parsed", "no parametrize row")
        return
    gate = float(m.group(1))
    rep = docs / "modernbert_device_perf_report.csv"
    if not rep.exists():
        record("SKIP", "gate brackets the measurement", "report absent")
        return
    rows = list(csv.reader(rep.open()))
    hdr = [c.strip() for c in rows[0]]
    val = [c.strip() for c in rows[1]]
    got = float(val[hdr.index("AVG DEVICE KERNEL SAMPLES/S")])
    lo = float(val[hdr.index("Lower Threshold AVG DEVICE KERNEL SAMPLES/S")])
    hi = float(val[hdr.index("Upper Threshold AVG DEVICE KERNEL SAMPLES/S")])
    record("PASS" if lo <= got <= hi else "FAIL", "gate brackets the measurement", f"{got} in [{lo}, {hi}]")
    record(
        "PASS" if abs(lo - gate * 0.97) < 0.01 else "FAIL",
        "report thresholds derive from the gate",
        f"gate={gate} lower={lo}",
    )

    # the submission texts must quote the measurement from the attached report,
    # not a value from an earlier profiling run
    for name in ("modernbert_pr_comment.md", "modernbert_performance_report.md"):
        f = docs / name
        if not f.exists():
            record("SKIP", f"{name} quotes the measured samples/s", "absent")
            continue
        quoted = {float(v) for v in re.findall(r"\*\*([0-9]{3}\.[0-9]{1,2}) AVG DEVICE KERNEL", f.read_text())}
        bad = {q for q in quoted if abs(q - got) > 0.05}
        record(
            "PASS" if quoted and not bad else ("SKIP" if not quoted else "FAIL"),
            f"{name} quotes the measured samples/s",
            f"report={got:.2f} quoted={quoted or 'none found'}",
        )

    # the gate must not be contradicted anywhere it is quoted
    for name in ("README.md",):
        txt = (MODEL / name).read_text()
        others = {float(v) for v in re.findall(r"\b(3[0-9]{2}\.[0-9])\s*\(CI gate\)", txt)}
        bad = {o for o in others if abs(o - gate) > 0.05}
        record("PASS" if not bad else "FAIL", f"{name} quotes the shipped gate", f"gate={gate} found={bad or 'ok'}")


# --------------------------------------------------------------------------
# 6. Op-breakdown tables agree with the perf sheet they cite
# --------------------------------------------------------------------------
GROUP = {
    "MatmulDeviceOperation": "Matmul",
    "SDPAOperation": "SDPA",
    "RotaryEmbeddingHfDeviceOperation": "RotaryEmbeddingHf",
    "LayerNormDeviceOperation": "LayerNorm",
    "ShardedToInterleavedDeviceOperation": "reshards",
    "InterleavedToShardedDeviceOperation": "reshards",
    "NlpCreateHeadsDeviceOperation": "heads",
    "NLPConcatHeadsDeviceOperation": "heads",
    "BinaryNgDeviceOperation": "binary",
    "EmbeddingsDeviceOperation": "embedding",
}


def check_op_counts(docs):
    csvf = docs / "modernbert_perf_b8s256.csv"
    if not csvf.exists():
        record("SKIP", "op counts match the perf sheet", "csv absent")
        return
    counts = {}
    for r in csv.DictReader(csvf.open()):
        try:
            d = float(r["DEVICE KERNEL DURATION [ns]"] or 0)
        except ValueError:
            continue
        if d <= 0:
            continue
        g = GROUP.get((r.get("OP CODE") or "").strip())
        if g:
            counts[g] = counts.get(g, 0) + 1
    expected = {"Matmul": 110, "SDPA": 22, "RotaryEmbeddingHf": 44, "LayerNorm": 45, "reshards": 134, "heads": 44}
    bad = {k: (counts.get(k), v) for k, v in expected.items() if counts.get(k) != v}
    record("PASS" if not bad else "FAIL", "op counts match the perf sheet", f"{bad or 'all match'}")


# --------------------------------------------------------------------------
# 7. Formatting the CI enforces
# --------------------------------------------------------------------------
def check_derived_claims(docs):
    """Claims computed from the perf sheet or the log, not quoted from them."""
    csvf, log = docs / "modernbert_perf_b8s256.csv", docs / "modernbert_validation.log"
    texts = [(n, docs / n) for n in ("modernbert_pr_comment.md", "modernbert_performance_report.md")]

    if csvf.exists():
        rows = list(csv.DictReader(csvf.open()))

        def dur(r):
            try:
                return float(r["DEVICE KERNEL DURATION [ns]"] or 0)
            except ValueError:
                return 0.0

        sd = collections.defaultdict(list)
        for r in rows:
            if "SDPA" in (r.get("OP CODE") or ""):
                sd[bool((r.get("INPUT_3_X_PAD[LOGICAL]") or "").strip())].append(dur(r))
        if sd[False] and sd[True]:
            full = sum(sd[False]) / len(sd[False]) / 1000
            slide = sum(sd[True]) / len(sd[True]) / 1000
            ratio = round(100 * (slide / full - 1))
            for name, f in texts:
                if not f.exists():
                    continue
                claimed = {int(v) for v in re.findall(r"sliding ones are (\d+)% more expensive", f.read_text())}
                claimed |= {int(v) for v in re.findall(r"sliding layers are (\d+)% more expensive", f.read_text())}
                bad = {c for c in claimed if c != ratio}
                record(
                    "PASS" if not bad else "FAIL",
                    f"{name} SDPA split ratio",
                    f"computed {ratio}% claimed {claimed or 'none'}",
                )

        util = collections.defaultdict(list)
        for r in rows:
            if "Matmul" not in (r.get("OP CODE") or ""):
                continue
            k = r["INPUT_0_X_PAD[LOGICAL]"].split("[")[0] + "->" + r["INPUT_1_X_PAD[LOGICAL]"].split("[")[0]
            try:
                util[k].append(float(r.get("PM FPU UTIL (%)") or 0))
            except ValueError:
                pass
            computed = {round(sum(v) / len(v), 1) for v in util.values()}
        for name, f in texts:
            if not f.exists():
                continue
            claimed = {
                float(v)
                for v in re.findall(r"\(768→2304\) ([0-9.]+)%|\(768→768\) ([0-9.]+)%", f.read_text())
                for v in v
                if v
            }
            bad = {c for c in claimed if not any(abs(c - x) < 0.15 for x in computed)}
            record("PASS" if not bad else "FAIL", f"{name} FPU utilisation", f"unmatched {bad or 'none'}")

    if log.exists():
        raw = re.findall(r"\[(NC[^\]]*)\]", log.read_text())
        controls = len({re.sub(r"-ablate\d+-\d+$", "", m) for m in raw})
        controls += sum(1 for k in ("[padding-control]", "[rope convention]") if k in log.read_text())
        words = {
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
            "thirteen": 13,
            "fourteen": 14,
            "fifteen": 15,
            "sixteen": 16,
        }
        for name, f in texts:
            if not f.exists():
                continue
            txt = f.read_text()
            if "Negative controls" not in txt:
                record("SKIP", f"{name} negative-control count", "document makes no such claim")
                continue
            m = re.search(r"\*\*Negative controls\.\*\* ([A-Za-z]+)[.,]", txt)
            if not m:
                record("FAIL", f"{name} negative-control count", "claim present but count unparseable")
                continue
            claimed = words.get(m.group(1).lower())
            record(
                "PASS" if claimed == controls else "FAIL",
                f"{name} negative-control count",
                f"log has {controls}, text says {m.group(1)}",
            )


def check_format():
    for tool, args in (("black", ["--line-length", "120", "--fast", "--check"]), ("isort", ["--check-only"])):
        out = subprocess.run(
            [sys.executable, "-m", tool, *args, str(MODEL)],
            capture_output=True,
            text=True,
        )
        record("PASS" if out.returncode == 0 else "FAIL", f"{tool} clean", out.stderr.strip().split("\n")[-1][:90])


# --------------------------------------------------------------------------
# 8. Submission prose is wrapped and free of placeholders
# --------------------------------------------------------------------------
def check_prose(docs):
    for name in ("modernbert_pr_description.md", "modernbert_pr_comment.md"):
        f = docs / name
        if not f.exists():
            record("SKIP", f"{name} wrapped", "absent")
            continue
        long = [i + 1 for i, l in enumerate(f.read_text().split("\n")) if len(l) > 100 and "|" not in l]
        record("PASS" if not long else "FAIL", f"{name} wrapped", f"lines {long[:5]}")
        holes = re.findall(r"\bTODO\b|\bTBD\b|FIXME|<OWNER_ID>|<TIER>|XXX", f.read_text())
        record("PASS" if not holes else "FAIL", f"{name} has no placeholders", f"{set(holes)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default=str(Path.home() / "Desktop"))
    args = ap.parse_args()
    docs = Path(args.docs)

    head = check_repo()
    check_artifact_commit(docs, head)
    check_language(docs)
    check_pcc_against_log(docs)
    check_submission_pccs(docs)
    check_gate(docs)
    check_op_counts(docs)
    check_derived_claims(docs)
    check_format()
    check_prose(docs)

    width = max(len(n) for _, n, _ in results)
    for status, name, detail in results:
        mark = {"PASS": "  ok  ", "FAIL": " FAIL ", "SKIP": " skip "}[status]
        print(f"[{mark}] {name:<{width}}  {detail}")
    fails = sum(1 for s, _, _ in results if s == "FAIL")
    skips = sum(1 for s, _, _ in results if s == "SKIP")
    print(f"\n{len(results) - fails - skips} passed, {fails} failed, {skips} skipped")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
