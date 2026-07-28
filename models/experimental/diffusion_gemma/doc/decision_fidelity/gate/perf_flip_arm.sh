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
# What decides them is ABSOLUTE correctness against the CUDA reference — flexible-extract exact
# match on the same 198 GPQA-Diamond questions the reference answered at 70.71% and 70.20% over two
# repetitions. That 0.5 pp spread is the resolution: a difference under ~1-1.5 pp is not a result.
#
# THREE ARMS, one variable each vs the shipped defaults, so any delta is attributable:
#   base    shipped defaults (SDPA grid already flipped; both levers off)
#   concat  + DG_MOE_CONCAT=1
#   norm    + DG_NORM_FULLCANVAS=1
# and optionally
#   both    + both, which is what would actually ship
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
#   perf_flip_arm.sh                    # all four arms, sequential, ~4 h each
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
ARMS=${ARMS:-"base concat norm both"}

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
        OUTPUT_ROOT="$out" \
        bash "$R/models/experimental/diffusion_gemma/doc/optimize_perf/run_upfront_gpqa.sh" full \
        > "$OUT_ROOT/${arm}.out" 2>&1
    echo "    exit=$? [$(date -u +%T)]"
done

echo
echo "=== scores ==="
python3 - "$OUT_ROOT" $ARMS <<'PYEOF'
import json, pathlib, sys

root = pathlib.Path(sys.argv[1])
arms = sys.argv[2:]
scores = {}
for arm in arms:
    files = sorted((root / arm).rglob("results_*.json"))
    if not files:
        print(f"  {arm:8s} NO RESULT")
        continue
    d = json.loads(files[-1].read_text())
    for task, m in d.get("results", {}).items():
        acc = m.get("exact_match,flexible-extract")
        se = m.get("exact_match_stderr,flexible-extract")
        n = d.get("n-samples", {}).get(task, {}).get("effective")
        if acc is None:
            continue
        scores[arm] = acc
        print(f"  {arm:8s} {task:22s} flexible={acc:.4f} stderr={se:.4f} n={n}")

if "base" in scores:
    print()
    print("  vs base (the resolution of this gate is ~1-1.5 pp; the CUDA reference's own")
    print("  run-to-run spread is 0.5 pp over two repetitions):")
    for arm in arms:
        if arm == "base" or arm not in scores:
            continue
        delta = (scores[arm] - scores["base"]) * 100.0
        verdict = "within noise" if abs(delta) < 1.5 else ("REGRESSION" if delta < 0 else "improvement")
        print(f"    {arm:8s} {delta:+.2f} pp   {verdict}")
print()
print("  CUDA reference bar: 70.71% / 70.20% flexible-extract (2 reps, thinking, 262k).")
PYEOF
echo "DG_PERF_FLIP_GATE_END $(date -u +%FT%TZ)"
