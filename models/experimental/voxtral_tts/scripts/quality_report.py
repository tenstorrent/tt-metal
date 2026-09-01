# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Every check on this branch, in one command, emitting a JSON that two runs can be diffed on.

    python .../scripts/quality_report.py --tier fast  --tag mychange     # ~3 min,  no audio
    python .../scripts/quality_report.py --tier full  --tag mychange     # ~20 min, + all gates
    python .../scripts/quality_report.py --tier audio --tag mychange     # ~50 min, + WER + MOS
    python .../scripts/quality_report.py --compare baseline mychange     # the paired diff

WHY THIS EXISTS. The checks were accumulating as one-off probes, so "did this change hurt quality"
depended on remembering which of a dozen scripts to run and what its numbers used to be. Every one
of them is now here, and every one writes its number into `generated/quality_<tag>.json`.

THE ONE RULE THIS FILE ENFORCES: **nothing on this branch is judged against a recorded number.**
STATUS.md §6.15 and §6.52 are both cases where a level from another session was compared against a
fresh one and read as a regression that did not exist — the codes gate's "10/288 vs 86/288" cost a
session's worth of doubt for exactly this reason. So `--compare` takes TWO TAGS and diffs them, and
the intended workflow is: run the tier on the base commit, make the change, run it again, compare.

A METRIC THAT FAILS TO PARSE IS RECORDED AS `null` AND FLAGGED. It is never silently dropped — a
gate whose output format drifts must break loudly, not quietly report success.

MOS needs an isolated venv (`tests/probes/mos_setup.sh`): DistillMOS pulls torchaudio, which §2
records as breaking `transformers` in the main venv. If /tmp/mosvenv is absent, MOS is skipped and
said so.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
GEN = os.path.join(HERE, "generated")
GATES = os.path.join(HERE, "tests", "gates.py")
MOSVENV = "/tmp/mosvenv/bin/python"

# Tolerances are the branch's own MEASURED noise floors, not guesses -- STATUS.md 6.15, 6.52, 6.63.
TOL = {
    "decode_mean_pp": 0.10, "decode_p90_pp": 0.15, "decode_min_pcc": 0.0002,
    # every PCC needs a tolerance; flow_velocity_pcc had none and so defaulted to ZERO, flagging a
    # 3.58e-06 move as a regression. Same defect codes_real_n had (6.62).
    "prefill_pcc_last": 0.0002, "codec_pcc_t24": 0.0002, "flow_velocity_pcc": 0.0002,
    "wiring_pcc": 0.0002, "flow_codes_74": 0,
    # codes_real_n and codes_real_pct are the SAME measurement in two units, and they disagreed:
    # 34 -> 37 read WORSE at zero tolerance while 3.9% -> 4.3% read "same" at 0.5. Give the count
    # the tolerance its own percentage implies (0.5% of 864 measured codes ~ 4).
    "codes_real_pct": 0.5, "codes_real_n": 4,
    "wer_longform": 0, "mos_longform": 0.05,
    # 0.01, not the 0.05 mos_longform carries. That figure covers the audio tier's three seeds and
    # its case-order dependency (6.21); the per-language set runs one process in a fixed order and
    # was MEASURED fully deterministic -- two independent generations produced 120 byte-identical
    # wavs and MOS agreeing to four decimals. So this is insurance against a library update
    # rounding differently, not a noise floor.
    **{f"mos_lang_{l}": 0.01 for l in ("ar", "de", "en", "es", "fr", "hi", "it", "nl", "pt")},
    "mos_lang_min": 0.01,
    # A run that loses MOS entirely must not read as unchanged: 1 -> 0 is a regression to report.
    "mos_available": 0,
    # 0.5 ms is not a guess: three identical audio-tier runs on unchanged HEAD spread 0.390 ms
    # (§6.63). ms_per_frame is the PRIMARY timing gate -- a block A/B is a screen only, because it
    # measures device time with dispatch overlapped and the real loop drains at 10 host crossings
    # per frame. §6.62 is the worked example: -2.124 ms on the blocks, 0 on the frame.
    "ms_per_frame": 0.5, "clicks_total": 0, "clipped_max_pct": 0.0,
}


# Every metric a tier MUST produce. Checked by presence, not just for None -- a regex that
# matches nothing leaves the key ABSENT, which an "is None" test sails straight past. That is how
# wer_longform went missing from a report that otherwise looked complete.
EXPECTED = {
    "fast": ["pytest_passed", "pytest_failed", "flow_velocity_pcc", "flow_semantic_exact",
             "flow_codes_74", "codes_real_pct", "codes_real_n", "codes_synth_n"],
    "full": ["wiring_pcc", "prefill_pcc_last", "prefill_n_cases", "codec_pcc_t24",
             "decode_mean_pp", "decode_p90_pp", "decode_min_pcc"],
    "audio": ["audio_runs", "wer_longform", "wer_longform_words", "n_utterances", "clicks_total",
              "mos_available",
              "clipped_max_pct", "terminated", "ms_per_frame", "rtf"],
}
MOS_KEYS = ["mos_mean", "mos_longform", "mos_min"]
# Per-language MOS. Pooled MOS cannot see one language turning robotic while staying intelligible,
# and WER cannot see naturalness at all, so this is the axis both other gates miss.
MOS_LANGS = ("ar", "de", "en", "es", "fr", "hi", "it", "nl", "pt")
MOS_LANG_KEYS = [f"mos_lang_{l}" for l in MOS_LANGS] + ["mos_lang_min"]


# Reported but NOT gated -- STATUS.md 6.62. codes_synth_n is non-monotonic in precision (6.59) and
# mos_mean/mos_min are dominated by short prompts, which 6.7 treats as seed noise. Tail risk is
# measured by tests/probes/tail_probe.py instead, which counts failures over many seeds.
REPORT_ONLY = ("codes_synth_n", "mos_mean", "mos_min", "mos_lang_spread")


def sh(cmd, timeout=3600, python=None):
    env = dict(os.environ)
    env.setdefault("TT_METAL_HOME", REPO)
    env["PYTHONPATH"] = f"{REPO}/ttnn:{REPO}/tools:{REPO}"
    exe = python or sys.executable
    try:
        r = subprocess.run([exe] + cmd, cwd=REPO, env=env, capture_output=True, text=True,
                           timeout=timeout)
        return r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return "__TIMEOUT__"


def grab(out, pattern, cast=float):
    m = re.search(pattern, out)
    return cast(m.group(1)) if m else None


def gate_json(gate, timeout):
    """Run one gate in its own process and return its metrics dict.

    JSON, not a regex over the printed tables: the tables are for humans and get reworded, and a
    metric that silently stops matching its pattern is exactly how a gate reports success while
    measuring nothing. A missing key still lands as None here and the REQUIRED check below turns
    that into a loud exit 2.
    """
    out = sh([GATES, "--gate", gate, "--json"], timeout=timeout)
    for line in out.splitlines():
        if line.startswith("GATE_JSON:"):
            try:
                return json.loads(line[len("GATE_JSON:"):])
            except json.JSONDecodeError:
                return {}
    return {}


def run_fast(res, log):
    log("pytest")
    # `not slow` keeps the fast tier at its documented ~3.5 min. The slow marks are the on-device
    # PCC tests added when the gates became pytest -- prefill x15, decode x15, the teacher-forced
    # whole model -- and the full tier runs those as its own gates anyway, so including them here
    # would pay for them twice.
    o = sh(["-m", "pytest", os.path.join(HERE, "tests"), "-q", "-m", "not slow"], timeout=900)
    res["pytest_passed"] = grab(o, r"(\d+) passed", int)
    res["pytest_failed"] = grab(o, r"(\d+) failed", int) or 0

    log("gate flow")
    m = gate_json("flow", timeout=900)
    res["flow_velocity_pcc"] = m.get("flow_velocity_pcc")
    res["flow_semantic_exact"] = m.get("flow_semantic_exact")
    res["flow_codes_74"] = m.get("flow_codes_74")

    log("gate codes")
    m = gate_json("codes", timeout=1800)
    res["codes_real_pct"] = m.get("codes_real_pct")
    res["codes_real_n"] = m.get("codes_real_n")
    res["codes_synth_n"] = m.get("codes_synth_n")


def run_full(res, log):
    log("gate wiring")
    res["wiring_pcc"] = gate_json("wiring", timeout=900).get("wiring_pcc")
    log("gate prefill26 (15 prompts)")
    m = gate_json("prefill26", timeout=2400)
    res["prefill_pcc_last"] = m.get("prefill_pcc_last")   # min across cases, as always recorded
    res["prefill_n_cases"] = m.get("prefill_n_cases")

    log("gate codec")
    res["codec_pcc_t24"] = gate_json("codec", timeout=1200).get("codec_pcc_t24")

    log("gate decode (15 prompts x 22 frames)")
    m = gate_json("decode", timeout=2400)
    res["decode_mean_pp"] = m.get("decode_mean_pp")
    res["decode_p90_pp"] = m.get("decode_p90_pp")
    res["decode_min_pcc"] = m.get("decode_min_pcc")


def run_audio(res, log, tag, seeds):
    """Generate, then score WER + artifacts + MOS. One process per seed -- §6.21: a case's frame
    count depends on what ran before it in the same process, so arms need identical history."""
    gen = os.path.join(HERE, "scripts", "generate_quality_set.py")
    tags = []
    for s in seeds:
        t = f"{tag}s{s}"
        log(f"generate 15 cases, seed {s}")
        sh([gen, "--tag", t, "--seed", str(s)], timeout=3600)
        tags.append(t)
    jsons = [os.path.join(GEN, f"results{t}.json") for t in tags]
    jsons = [j for j in jsons if os.path.exists(j)]
    res["audio_runs"] = len(jsons)
    if not jsons:
        return

    log("WER (scipy scorer)")
    o = sh([os.path.join(HERE, "scripts", "score_quality_set_scipy.py")] + jsons, timeout=3600)
    # Anchored on the distinctive "c<n>=" column rather than on leading whitespace: the tag is
    # right-aligned in a 10-char field, so a tag of 11+ chars starts at column 0 and an "^\s+"
    # pattern silently matches nothing. That is exactly how this metric went missing without the
    # report noticing -- see the EXPECTED check below, which is the real fix.
    lf = re.findall(r"^\s*\S+\s+(\d+)\s+(\d+)\s+c\d+=", o, re.M)
    if lf:
        res["wer_longform"] = sum(int(a) for a, _ in lf)
        res["wer_longform_words"] = sum(int(b) for _, b in lf)

    # ---- artifacts, over every utterance of every seed ----
    import statistics
    rows = [x for j in jsons for x in json.load(open(j))]
    res["n_utterances"] = len(rows)
    res["clicks_total"] = sum(r["click_count"] for r in rows)
    res["clipped_max_pct"] = max(r["clipped_%"] for r in rows)
    res["terminated"] = sum(1 for r in rows if r["terminated"])
    # 6.52: case 0 is the FIRST utterance of each process and pays one-time program-cache
    # compilation -- 3.3 s over 5.4 s of audio, RTF 1.346. Including it inflated a reported RTF
    # from 0.507 to 0.694 and stopped it reconciling with anything.
    lfr = [r for r in rows if len(r["text"].split()) >= 20 and r["case"] != 0]
    if lfr:
        msf = [r["gen_ms_per_frame"] for r in lfr if "gen_ms_per_frame" in r]
        res["ms_per_frame"] = round(statistics.mean(msf), 3) if msf else None
        res["rtf"] = round(statistics.mean(r["rtf"] for r in lfr), 4)

    # ---- MOS, isolated venv ----
    res["mos_available"] = 1 if os.path.exists(MOSVENV) else 0
    if os.path.exists(MOSVENV):
        log("MOS (DistillMOS, isolated venv)")
        o = sh([os.path.join(HERE, "tests", "probes", "mos_batch.py")] +
               [f"{tag}s{s}" for s in seeds], timeout=3600, python=MOSVENV)
        res["mos_mean"] = grab(o, r"MOS_MEAN ([\d.]+)")
        res["mos_longform"] = grab(o, r"MOS_LONGFORM ([\d.]+)")
        res["mos_min"] = grab(o, r"MOS_MIN ([\d.]+)")

        # Per language, on the sentences of ~20 words and up -- the medium and long WER bands, so
        # the clips are the same text the WER cells score and the two axes compare directly.
        log(f"per-language clips + MOS ({len(MOS_LANGS)} languages)")
        sh([os.path.join(HERE, "scripts", "generate_language_set.py"), "--tag", tag], timeout=7200)
        o = sh([os.path.join(HERE, "tests", "probes", "mos_perlang.py"), tag], timeout=3600,
               python=MOSVENV)
        for l in MOS_LANGS:
            res[f"mos_lang_{l}"] = grab(o, rf"MOS_LANG_{l} ([\d.]+)")
        res["mos_lang_min"] = grab(o, r"MOS_LANG_MIN ([\d.]+)")
        res["mos_lang_spread"] = grab(o, r"MOS_LANG_SPREAD ([\d.]+)")
    else:
        res["mos_mean"] = None
        res["_mos_note"] = f"{MOSVENV} absent -- run tests/probes/mos_setup.sh"


def compare(a, b):
    A = json.load(open(os.path.join(GEN, f"quality_{a}.json")))
    B = json.load(open(os.path.join(GEN, f"quality_{b}.json")))
    keys = [k for k in dict.fromkeys(list(A) + list(B)) if not k.startswith("_")]
    print(f"\n  {'metric':<24} {a[:14]:>14} {b[:14]:>14} {'delta':>11}   verdict")
    worse = same = 0
    for k in keys:
        x, y = A.get(k), B.get(k)
        if k in REPORT_ONLY:
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                print(f"  {k:<24} {x:>14.6g} {y:>14.6g} {y-x:>+11.4g}   (reported, not gated)")
            continue
        if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, (int, float)) \
                or not isinstance(y, (int, float)):
            if x != y:
                print(f"  {k:<24} {str(x):>14} {str(y):>14} {'':>11}   CHANGED")
            continue
        d = y - x
        tol = TOL.get(k, 0)
        if abs(d) <= tol:
            v = "same (within tol)"
            same += 1
        else:
            # for these, LOWER is better; everything else higher is better
            lower_better = k in ("decode_mean_pp", "decode_p90_pp", "flow_codes_74",
                                 "codes_real_pct", "codes_real_n", "codes_synth_n",
                                 "wer_longform", "ms_per_frame", "rtf", "clicks_total",
                                 "clipped_max_pct", "pytest_failed")
            good = (d < 0) if lower_better else (d > 0)
            v = "BETTER" if good else "*** WORSE ***"
            worse += 0 if good else 1
        print(f"  {k:<24} {x:>14.6g} {y:>14.6g} {d:>+11.4g}   {v}")
    print(f"\n  {worse} metric(s) worse beyond tolerance, {same} within tolerance.")
    print("  Tolerances are the branch's measured noise floors (§6.15 decode spread 0.45 pp,")
    print("  §6.52 timing floor 0.070 ms, §6.7 short-bucket WER is seed noise).")
    return 1 if worse else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=("fast", "full", "audio"), default="fast")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--compare", nargs=2, metavar=("TAG_A", "TAG_B"))
    a = ap.parse_args()
    if a.compare:
        raise SystemExit(compare(*a.compare))

    t0 = time.time()
    res = {"_tag": a.tag, "_tier": a.tier}
    res["_commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                                    capture_output=True, text=True).stdout.strip()
    res["_dirty"] = bool(subprocess.run(["git", "status", "--porcelain", HERE], cwd=REPO,
                                        capture_output=True, text=True).stdout.strip())

    def log(m):
        print(f"  [{time.time()-t0:6.1f}s] {m}", flush=True)

    run_fast(res, log)
    if a.tier in ("full", "audio"):
        run_full(res, log)
    if a.tier == "audio":
        # Never lose the gate results to a crash in the audio stage -- the JSON is written either
        # way, with the failure recorded in it.
        try:
            run_audio(res, log, a.tag, [int(s) for s in a.seeds.split(",")])
        except Exception as e:
            res["_audio_error"] = f"{type(e).__name__}: {e}"
            log(f"AUDIO STAGE FAILED: {res['_audio_error']}")

    res["_seconds"] = round(time.time() - t0, 1)
    want = list(EXPECTED["fast"])
    if a.tier in ("full", "audio"):
        want += EXPECTED["full"]
    if a.tier == "audio":
        want += EXPECTED["audio"]
        if os.path.exists(MOSVENV):
            want += MOS_KEYS + MOS_LANG_KEYS
    missing = [k for k in want if res.get(k) is None]
    os.makedirs(GEN, exist_ok=True)
    dst = os.path.join(GEN, f"quality_{a.tag}.json")
    json.dump(res, open(dst, "w"), indent=2, sort_keys=True)

    print(f"\n  wrote {dst}  ({res['_seconds']}s, commit {res['_commit']}"
          f"{' DIRTY' if res['_dirty'] else ''})")
    for k in sorted(res):
        if not k.startswith("_"):
            print(f"    {k:<24} {res[k]}")
    if missing:
        print(f"\n  *** {len(missing)} METRIC(S) FAILED TO PARSE: {', '.join(missing)}")
        print("  A gate's output format probably drifted. Fix the regex in TOL/run_* rather than")
        print("  accepting a report with holes in it.")
        raise SystemExit(2)
    if res.get("pytest_failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
