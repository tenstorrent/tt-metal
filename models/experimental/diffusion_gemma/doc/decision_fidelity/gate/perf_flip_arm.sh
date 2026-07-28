#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Absolute-quality gate for the two measured perf levers that are still default OFF:
# DG_MOE_CONCAT (-29.9%) and DG_NORM_FULLCANVAS (-17.9%), ~1.9x together.
#
# WHY THIS SHAPE. Both levers change the committed tokens, so the TT-vs-TT bit-identity comparison
# that has gated them until now cannot answer the question: it compares a candidate against a
# baseline our own #48291 record calls degenerate, so a lever can only pass by changing nothing.
#
# WHICH METRIC. `run_upfront_gpqa.sh` now runs **gpqa_diamond_cot_zeroshot** with
# `exact_match,flexible-extract` — the same task and filter the A100 reference used, so arm scores
# are directly comparable to its **70.71% / 70.20%** bar (two reps, thinking, 262k), whose 0.5 pp
# spread is the resolution.
#
# It used to run `r1_gpqa_diamond` / `exact_match,none`, and that produced a meaningless gate: the
# strict filter extracts a bare \boxed{} value, the r1 prompt never asks for one, and on the 07-28
# full run only **4 of 198** responses contained a boxed answer, scoring 6.57%. A metric with four
# positives cannot separate two arms. The CoT task fixes both halves — its prompt instructs "put your
# final answer (only the letter A, B, C, or D) within \boxed{}" and its flexible-extract filter reads
# the boxed value first, then explicit answer markers, and only accepts A-D.
#
# The collapse rate is reported alongside as a within-arm diagnostic -- guard fires and empty
# responses, per-request binaries at n=198 -- because they are what explains a low score (the 07-28
# r1 run fired the guard on 146 of 198, clustered at blocks 0-3, with 26 empty). Not the bar.
#
# THE BASELINE IS THE A100 CUDA RESULT, not a TT arm: 70.71% and 70.20% flexible-extract over two
# reference reps in thinking mode. Each TT arm is scored directly against that. Do NOT spend 3.7 h of
# device time re-measuring a TT baseline -- the `base` arm below exists only for someone who
# explicitly wants the shipped-default TT number for its own sake, and it is not the bar.
#
# ARMS, one variable each vs the shipped defaults, so any delta is attributable:
#   both    + DG_MOE_CONCAT=1 DG_NORM_FULLCANVAS=1 -- what would actually ship; run this first
#   concat  + DG_MOE_CONCAT=1
#   norm    + DG_NORM_FULLCANVAS=1
#   base    shipped defaults -- OPTIONAL
#
# The `both` arm is not redundant. The levers touch different things (MoE layout vs norm shape) but
# both perturb the same bf16 reductions feeding the same argmax, so their fidelity effects are not
# guaranteed to be independent — this module has already been burned assuming that
# (doc/decision_fidelity/device_gumbel_restored.md: DG_DENOISE_SLIDING_WINDOW looked clean on one
# arm and regressed on question 2 of the second).
#
# CONCAT NEEDS DRAM. The concat weights cost 7.8 GiB, so its arms run at TRACE_REGION_SIZE=4 GiB
# (measured floor 3.04 GiB at reveal_pmax 4096; see doc/optimize_perf/flag_triage_20260728.md).
# The base and norm arms use the same value so the reservation is not a second variable.
#
# BLAST RADIUS, stated because it changes what a pass means: DG_MOE_CONCAT also folds the COMMIT
# MoE (tt/commit_batched.py calls the same _denoise_moe_forward seam and batched commit is the
# default), and commit hidden states build the committed-prefix KV, so it compounds across blocks.
# A pass on this gate is a pass on two components, not one.
#
# Usage:
#   perf_flip_arm.sh                    # both, concat, norm -- sequential, ~4 h each
#   ARMS="base concat" perf_flip_arm.sh # subset
#
# Each arm is a full 198-sample run through the real vLLM server via run_upfront_gpqa.sh, so the
# result is directly comparable to the reference numbers above and to whatever the current baseline
# run produced.

set -uo pipefail

R=${TT_METAL_ROOT:-/home/zni/tt-metal}
TTSMI=${TT_SMI_BIN:-/home/zni/ttis-verify/.workflow_venvs/.venv_tt_smi/bin/tt-smi}
OUT_ROOT=${OUT_ROOT:-/home/zni/dg_runs/perf_flip_gate}
TRACE=${TRACE_REGION_SIZE:-4294967296}
ARMS=${ARMS:-"both concat norm"}

# A case rather than an associative array: under `set -u`, an empty value in a
# `declare -A` literal is reported as an unbound variable on this bash, and `base` must be empty.
arm_env() {
    case "$1" in
        base) echo "" ;;
        concat) echo "DG_MOE_CONCAT=1" ;;
        norm) echo "DG_NORM_FULLCANVAS=1" ;;
        both) echo "DG_MOE_CONCAT=1 DG_NORM_FULLCANVAS=1" ;;
        *) return 1 ;;
    esac
}

mkdir -p "$OUT_ROOT"
echo "DG_PERF_FLIP_GATE_BEGIN $(date -u +%FT%TZ) arms='${ARMS}' trace=${TRACE}"

for arm in $ARMS; do
    if ! env_for_arm=$(arm_env "$arm"); then
        echo "ERROR: unknown arm '${arm}'; valid: base concat norm both" >&2
        exit 2
    fi
    out="$OUT_ROOT/$arm"
    if find "$out" -name "results_*.json" 2>/dev/null | grep -q .; then
        echo "=== $arm SKIP (already has a results file)"
        continue
    fi
    echo "=== $arm  env='${env_for_arm}'  -> $out  [$(date -u +%T)]"
    # shellcheck disable=SC2086
    env ${env_for_arm} \
        TT_SMI_BIN="$TTSMI" \
        TRACE_REGION_SIZE="$TRACE" \
        RESET_BEFORE=1 RESET_AFTER=1 \
        THINKING_MODE=1 \
        OUTPUT_ROOT="$out" \
        bash "$R/models/experimental/diffusion_gemma/doc/optimize_perf/run_upfront_gpqa.sh" full \
        > "$OUT_ROOT/${arm}.out" 2>&1
    echo "    exit=$? [$(date -u +%T)]"
done

echo
echo "=== scores ==="
python3 - "$OUT_ROOT" $ARMS <<'PYEOF'
import json, pathlib, re, sys

root = pathlib.Path(sys.argv[1])
arms = sys.argv[2:]

# THE BAR: the A100 CUDA reference, flexible-extract, thinking, two reps. Not a TT arm.
CUDA_REF = (0.7071, 0.7020)
CUDA_SPREAD = abs(CUDA_REF[0] - CUDA_REF[1]) * 100.0  # 0.5 pp -- the reference's own repeatability


def measure(arm):
    d = root / arm
    res = sorted(d.rglob("full/*/results_*.json"))
    smp = sorted(d.rglob("full/*/samples_*.jsonl"))
    if not res or not smp:
        return None
    out = {"guard": 0}
    blob = json.loads(res[-1].read_text())
    for task, m in blob.get("results", {}).items():
        for k, v in m.items():
            # Take whatever exact_match filter this task emits rather than assuming one it lacks.
            if k.startswith("exact_match,"):
                out["exact"], out["filter"] = v, k.split(",", 1)[1]
        out["n"] = blob.get("n-samples", {}).get(task, {}).get("effective")
    texts = [(json.loads(l).get("resps") or [[""]])[0][0] for l in open(smp[-1])]
    out["empty"] = sum(1 for t in texts if not t.strip())
    out["boxed"] = sum(1 for t in texts if re.search(r"\\boxed", t))
    log = d / "server.log"
    if log.exists():
        out["guard"] = log.read_text(errors="replace").count("ending request at block")
    return out


hdr = "  %-8s %-5s %-20s %-12s %-12s %s"
print(hdr % ("arm", "n", "exact_match", "guard", "empty", "boxed"))
got = {}
for arm in arms:
    m = measure(arm)
    if m is None:
        print("  %-8s NO RESULT" % arm)
        continue
    got[arm] = m
    print(hdr % (arm, m.get("n"), "%.4f (%s)" % (m.get("exact", float("nan")), m.get("filter", "?")),
                 m["guard"], m["empty"], m["boxed"]))

bar = sum(CUDA_REF) / len(CUDA_REF)
print()
print("  BAR: A100 CUDA reference %.2f%% / %.2f%% flexible-extract (thinking, 2 reps)," % (
    CUDA_REF[0] * 100, CUDA_REF[1] * 100))
print("       mean %.2f%%, own repeatability %.1f pp. Each TT arm is scored against this," % (
    bar * 100, CUDA_SPREAD))
print("       NOT against a TT baseline.")
print()
for arm in arms:
    if arm not in got:
        continue
    m = got[arm]
    exact = m.get("exact")
    if exact is None:
        print("    %-8s no exact_match in the results file" % arm)
        continue
    gap = (exact - bar) * 100.0
    # A TT arm at or above the reference is unambiguous. Below it, the question is whether THIS
    # lever is responsible, which needs the other arms -- so report the gap and say so rather than
    # pronouncing on a single number.
    if gap >= -CUDA_SPREAD:
        verdict = "at the reference bar"
    elif gap >= -5.0:
        verdict = "within 5 pp of the bar"
    else:
        verdict = "%0.1f pp below the bar" % -gap
    print("    %-8s %.2f%%  (%+.2f pp vs bar)  %s   [guard %d/%s, empty %d, boxed %d]" % (
        arm, exact * 100, gap, verdict, m["guard"], m.get("n"), m["empty"], m["boxed"]))
print()
print("  A gap below the bar is not by itself attributable to the lever under test: the shipped")
print("  defaults are themselves below it (#48291). Run the single-flag arms to attribute.")
PYEOF
echo "DG_PERF_FLIP_GATE_END $(date -u +%FT%TZ)"
