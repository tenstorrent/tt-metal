#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Blackhole Galaxy MoE benchmark driver. Runs from $TT_METAL_HOME on a Galaxy node.
#
#   run_bench.sh prefill [RESULTS_DIR]   # TtMoe prefill layer, 8x4 TorusXY, one row per model + traced CI gates
#   run_bench.sh decode  [RESULTS_DIR]   # moe_compute on one chip, model x decode-batch sweep (+ FP32 DEST rows)
#
# Knobs (see moe_models.py): MOE_BENCH_MODELS, MOE_BENCH_CUSTOM, MOE_BENCH_DECODE_USERS, MOE_BENCH_DECODE_REPEATS.
# Results: per-session profiler CSVs, logs, a tt-smi snapshot, and roofline.{md,json} from roofline.py.
set -uo pipefail
MODE=${1:?usage: run_bench.sh prefill|decode [RESULTS_DIR]}
cd "${TT_METAL_HOME:?}"
D=models/demos/deepseek_v3_d_p/tests/moe_bench
RES=${2:-${MOE_BENCH_RESULTS:-generated/moe_bench}/${MODE}-${SLURM_JOB_ID:-local}-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$RES"
echo "[moe_bench] results -> $RES"

export TT_METAL_TDP_LIMIT_WATTS=${TT_METAL_TDP_LIMIT_WATTS:-160}
export DS_MOE_CACHE_DIR=${DS_MOE_CACHE_DIR:-/tmp/${USER:-bench}_moe_bench_cache}

git rev-parse HEAD >"$RES/commit" 2>/dev/null
hostname >"$RES/host"
# Clocks, DDR speed and power limits for the roofline's peaks.
tt-smi -s >"$RES/tt_smi_snapshot.json" 2>"$RES/tt_smi.err" || true

# run_profiled NAME PYTEST_ARGS...: one pytest session under the device profiler, artifacts in $RES/prof_NAME.
run_profiled() {
    local name=$1
    shift
    echo "[moe_bench] === $name: pytest $*"
    python3 -m tracy -r -v -o "$RES/prof_$name" -m pytest --timeout 0 -p no:cacheprovider "$@" >"$RES/$name.log" 2>&1
    local rc=$?
    grep -E '^(PASSED|FAILED|ERROR|SKIPPED|XFAIL)|^=+ .*(passed|failed|error)' "$RES/$name.log" | tail -60
    echo "[moe_bench] $name rc=$rc csv: $(find "$RES/prof_$name" -name 'ops_perf_results*.csv' | head -1)"
}

case "$MODE" in
prefill)
    run_profiled prefill "$D/test_prefill_moe_bench.py"
    # Wall-clock cross-check: the traced CI gates (DeepSeek V3, Kimi K2.7 / K3). They print the measured time
    # even when it falls outside their band. Not profiled: tracy would perturb the traced replay.
    export KIMI_MOE_PERF_IGNORE_POWER=1
    for t in models/demos/deepseek_v3_d_p/tests/perf/test_kimi_moe_perf.py \
        "models/demos/deepseek_v3_d_p/tests/perf/test_moe_perf.py::test_deepseek_v3_moe_perf_galaxy"; do
        n=gate_$(basename "${t%%::*}" .py)
        echo "[moe_bench] === $n"
        MESH_DEVICE=TG python3 -m pytest --timeout 0 -p no:cacheprovider "$t" >"$RES/$n.log" 2>&1
        grep -E 'expected [0-9,]+ ns|ns \(|band|PASSED|FAILED|SKIPPED|^=+ ' "$RES/$n.log" | tail -20
    done
    ;;
decode)
    run_profiled decode "$D/test_decode_moe_compute_bench.py"
    # Accuracy-friendly setting: FP32 DEST accumulate at the u=32 decode batch.
    MOE_BENCH_DECODE_FP32_DEST=1 MOE_BENCH_DECODE_USERS=32 run_profiled decode_fp32dest "$D/test_decode_moe_compute_bench.py"
    ;;
*)
    echo "unknown mode $MODE" >&2
    exit 2
    ;;
esac

python3 "$D/roofline.py" "$RES" && cat "$RES/roofline.md"
echo "[moe_bench] done: $RES"
