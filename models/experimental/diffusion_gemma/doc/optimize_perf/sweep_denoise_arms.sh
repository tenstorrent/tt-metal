#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Generic interleaved A/B harness for denoise-path levers, over demo/serving_smoke.py.
#
# Generalizes sweep_denoise_qchunk.sh: instead of one hard-coded env var it takes arms of the
# form NAME:ENV1=V1[,ENV2=V2...] so several levers (and their combinations) can be swept in one
# interleaved pass. Everything the qchunk sweep established about method is kept:
#
#   * repetitions are INTERLEAVED (arm1,arm2,...,arm1,arm2,...) so device drift is spread across
#     arms rather than absorbed by whichever ran first — the qchunk sweep found rep 1 slower than
#     rep 2 for every arm, which is exactly the drift this defends against;
#   * steady state = mean(per_block_latency_s[1:]); block 0 carries program compilation and
#     measuring it once produced a bogus "+50% regression";
#   * committed_sha256 is reported per arm so a decision change is never mistaken for a speedup.
#     It is reported, NOT asserted: a lever that changes reduction order legitimately changes the
#     trajectory, and bit-identity to a baseline our own #48291 record calls degenerate is not by
#     itself evidence of quality. Treat a sha change as "needs the absolute quality gate", not
#     "failed".
#
# Usage:
#   sweep_denoise_arms.sh "base:" "hifi4:DG_SPARSE_MOE_HIFI4=1" "nomoe:DG_SKIP=moe"
#   UPFRONT=1 REVEAL_PMAX=4096 STEPS=48 sweep_denoise_arms.sh ...
#
# An empty env list (``base:``) runs the shipped defaults.

set -uo pipefail

TT_METAL_ROOT="${TT_METAL_ROOT:-/home/zni/tt-metal}"
MODEL_VENV="${MODEL_VENV:-/home/zni/venvs/tt-diffusion-gemma}"
DG_CKPT="${DG_CKPT:-/home/zni/dg_models/diffusiongemma-26B-A4B-it}"
MESH="${MESH:-P150x4}"
NUM_BLOCKS="${NUM_BLOCKS:-3}"
STEPS="${STEPS:-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
NUM_LAYERS="${NUM_LAYERS:-}"
REPS="${REPS:-3}"
UPFRONT="${UPFRONT:-0}"
REVEAL_PMAX="${REVEAL_PMAX:-}"
GUMBEL_MODE="${GUMBEL_MODE:-}"
TRACE_REGION_SIZE="${TRACE_REGION_SIZE:-4294967296}" # 4 GiB. Measured: the 48 up-front traces fit at reveal_pmax=4096 (3.04 GiB used, 3 GiB fails) AND in the same 4 GiB at reveal_pmax=16384 (48/48 captured, MeshTraceId 0..47, run local100_trace4g 2026-07-29), so trace memory does NOT scale with reveal_pmax -- the "Scale this WITH reveal_pmax" note this line used to carry was wrong. Over-reserving is not free: the region is DRAM nothing else can allocate, and 6->4 GiB returns exactly 2.0 GiB to the pool, which is what keeps the per-request leak from OOMing the engine at max_model_len 16384. Undersizing fails loudly at capture. See doc/optimize_perf/bisect_trace_region.sh
# The shipped degeneracy guard ends a request at the first collapsed block. That is right for
# serving and wrong for a latency A/B: arms would emit different block counts and the steady-state
# mean would be taken over different work. Off by default here; the arms are compared on
# committed_sha256 for decisions, not on whether the guard fired.
DEGENERACY_POLICY="${DEGENERACY_POLICY:-off}"
# The stable-and-confident early halt fires at a data-dependent step (measured 9/2/2 for a 48-step
# budget), and a lever that changes the numerics also moves where it fires — so arms would be
# compared over different amounts of denoise work, with the fixed commit cost dominating what is
# left. A negative threshold disables the halt so every arm runs exactly --max-denoising-steps.
# Set ENTROPY_STOP=0.005 to measure the shipped end-to-end behaviour instead.
ENTROPY_STOP="${ENTROPY_STOP:--1}"
OUT_DIR="${OUT_DIR:-/tmp/dg_arm_sweep}"

if [ "$#" -lt 1 ]; then
    echo "usage: $0 NAME:[ENV=V[,ENV=V...]] [NAME:...]" >&2
    exit 2
fi
ARMS=("$@")

# --upfront needs a materialized Gumbel source (argmax is None and cannot be refreshed between
# trace replays). ``device`` is the only materialized mode: the ``host`` mode this used to pick was
# DELETED on 2026-07-28 after being measured NOT to be the TT language-drift cause (same drifting
# prompts as ``device``, repairs 0, costs 1.40x per request; the real cause was the canvas attending
# prefill pad keys, fixed in d0936d4da4f). ``device`` is seeded, so a fixed --seed still keeps the
# run reproducible across arms -- but absolute ms/step is ~1.94x lower than the host-Gumbel numbers
# recorded in doc/optimize_perf/winter_borrow_20260727.md and committed_sha256 values from earlier
# host-mode sweeps will not reproduce.
if [ -z "${GUMBEL_MODE}" ]; then
    if [ "${UPFRONT}" != "0" ]; then GUMBEL_MODE=device; else GUMBEL_MODE=argmax; fi
fi

mkdir -p "$OUT_DIR"
PY="${MODEL_VENV}/bin/python"

EXTRA=()
[ -n "${NUM_LAYERS}" ] && EXTRA+=(--num-layers "${NUM_LAYERS}")
if [ "${UPFRONT}" != "0" ]; then
    EXTRA+=(--upfront)
    [ -n "${REVEAL_PMAX}" ] && EXTRA+=(--reveal-pmax "${REVEAL_PMAX}")
fi

echo "DG_ARM_SWEEP_BEGIN arms=${#ARMS[@]} blocks=${NUM_BLOCKS} steps=${STEPS} reps=${REPS} upfront=${UPFRONT} gumbel=${GUMBEL_MODE} out=${OUT_DIR}"

for rep in $(seq 1 "${REPS}"); do
    for arm in "${ARMS[@]}"; do
        name="${arm%%:*}"
        envspec="${arm#*:}"
        tag="${name}_rep${rep}"
        log="${OUT_DIR}/${tag}.log"
        echo "--- rep=${rep} arm=${name} env='${envspec}' -> ${log}"

        envargs=(
            TT_METAL_HOME="${TT_METAL_ROOT}"
            PYTHONPATH="${TT_METAL_ROOT}"
            MESH_DEVICE="${MESH}"
            DG_TRACE_REGION_SIZE="${TRACE_REGION_SIZE}"
            DG_DEGENERACY_POLICY="${DEGENERACY_POLICY}"
        )
        if [ -n "${envspec}" ]; then
            IFS=',' read -ra kvs <<<"${envspec}"
            for kv in "${kvs[@]}"; do
                [ -n "${kv}" ] && envargs+=("${kv}")
            done
        fi

        env "${envargs[@]}" \
            "${PY}" -u -m models.experimental.diffusion_gemma.demo.serving_smoke \
            --checkpoint "${DG_CKPT}" \
            --mesh "${MESH}" \
            --max-seq-len "${MAX_SEQ_LEN}" \
            --num-blocks "${NUM_BLOCKS}" \
            --max-denoising-steps "${STEPS}" \
            --gumbel-mode "${GUMBEL_MODE}" \
            --seed 0 \
            --entropy-stop-threshold "${ENTROPY_STOP}" \
            --disable-eos-stop \
            --local-files-only \
            "${EXTRA[@]}" \
            --metrics-json "${OUT_DIR}/${tag}.json" \
            >"${log}" 2>&1
        rc=$?
        sha=$("${PY}" -c "import json,sys;print(json.load(open(sys.argv[1]))['committed_sha256'])" \
            "${OUT_DIR}/${tag}.json" 2>/dev/null)
        echo "rep=${rep} arm=${name} rc=${rc} committed_sha256=${sha:-MISSING}"
    done
done

echo
echo "=== verdict ==="
REPS="${REPS}" "${PY}" - "$OUT_DIR" "${ARMS[@]}" <<'PYEOF'
import json, os, statistics, sys, pathlib

out = pathlib.Path(sys.argv[1])
arms = [a.split(":", 1)[0] for a in sys.argv[2:]]
reps = int(os.environ.get("REPS", "1"))

runs, missing = {}, []
for name in arms:
    for rep in range(1, reps + 1):
        tag = f"{name}_rep{rep}"
        log, meta = out / f"{tag}.log", out / f"{tag}.json"
        text = log.read_text(errors="replace") if log.exists() else ""
        if "DG_VLLM_SERVING_SMOKE_SUCCESS" not in text or not meta.exists():
            missing.append(tag)
            continue
        try:
            runs.setdefault(name, []).append(json.loads(meta.read_text()))
        except Exception:
            missing.append(tag)
for tag in missing:
    print(f"  !! run did not complete: {tag}")

shas = {n: {b["committed_sha256"] for b in blobs} for n, blobs in runs.items()}
all_shas = set().union(*shas.values()) if shas else set()
print(f"committed_sha256 distinct values across all runs: {len(all_shas)}")
for name in arms:
    if name in shas:
        print(f"  {name:>16}: {sorted(s[:16] for s in shas[name])}")

print()
print(f"{'arm':>16} {'n':>3} {'steady_mean_s':>14} {'per-rep steady means':<34} {'block0_s':>10}")
steady = {}
for name in arms:
    if name not in runs:
        continue
    per_rep, block0 = [], []
    for b in runs[name]:
        lat = b.get("per_block_latency_s") or []
        block0.append(lat[0] if lat else float("nan"))
        tail = lat[1:]
        if tail:
            per_rep.append(statistics.mean(tail))
    if not per_rep:
        print(f"{name:>16} {0:>3}  NO STEADY-STATE SAMPLE (need >=2 blocks)")
        continue
    steady[name] = statistics.mean(per_rep)
    # Report the denoise-step count per arm: a steady mean is only comparable when every arm ran
    # the same amount of work, and the early halt is exactly what breaks that.
    steps = sorted({tuple(b.get("denoise_steps_per_block") or []) for b in runs[name]})
    print(
        f"{name:>16} {len(per_rep):>3} {steady[name]:>14.3f} "
        f"{str([round(x, 3) for x in per_rep]):<34} {statistics.mean(block0):>10.3f}  steps={steps}"
    )

if steady:
    base = steady.get(arms[0])
    print()
    if base:
        for name in arms:
            if name in steady:
                print(f"  {name:>16} vs {arms[0]}: {(steady[name] / base - 1.0) * 100.0:+.1f}%")
    print(f"  fastest: {min(steady, key=steady.get)}")
print("DG_ARM_SWEEP_RESULT " + ("ALL_BIT_IDENTICAL" if len(all_shas) == 1 else "DIVERGED"))
PYEOF
