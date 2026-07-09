#!/usr/bin/env bash
# Profile ONLY LTX stage-2 denoise under Tracy, keeping the device-profiler DRAM buffer
# from wrapping (which silently drops ops). Mirrors tmp/run_ltx.sh's env exactly, then swaps
# the bare pytest for a `python -m tracy -p -r -v` capture with a large --op-support-count.
#
# Why this isolates stage-2: LTX_TRACED=1 captures stage-1 and stage-2 as separate device
# traces, so tmp/parse_tracy.py segments by METAL TRACE ID (no source markers). We shrink
# stage-1 to a single step (LTX_S1_SIGMAS=1.0,0.0) so it barely dents the op budget; the fast
# stage-2 is already ONE step (LTX_FAST sets LTX_S2_SIGMAS=0.909375,0.0), so this profiles the
# ENTIRE shipped stage-2, not a fraction. Sigma values change only the scalar timestep, never a
# tensor shape, so per-op device times are identical to a real multi-step run.
#
# NO device job is launched by this script's author — the supervisor runs it through the broker.
set -uo pipefail
WT=/home/smarton/tt-metal/.claude/worktrees/ltx-perf-clean
cd "$WT" || exit 3

# --- run_ltx.sh base env (kept in sync) ---
export TT_METAL_HOME="$WT"
export PYTHONPATH="$WT/ttnn:$WT:$WT/tools"
export GEMMA_PATH="${GEMMA_PATH:-/home/kevinmi/.cache/huggingface/hub/models--google--gemma-3-12b-it-qat-q4_0-unquantized/snapshots/68f7ee4fbd59087436ada77ed2d62f373fdd4482/}"
export LTX_CHECKPOINT="${LTX_CHECKPOINT:-/home/kevinmi/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/76730e634e70a28f4e8d51f5e29c08e40e2d8e74/ltx-2.3-22b-distilled-1.1.safetensors}"
export TT_DIT_CACHE_DIR="${TT_DIT_CACHE_DIR:-/home/smarton/.cache/tt-dit}"
export TT_METAL_CACHE="${TT_METAL_CACHE:-/home/smarton/.cache/tt-metal-cache-ltxperf650}"
export TT_METAL_OPERATION_TIMEOUT_SECONDS="${TT_METAL_OPERATION_TIMEOUT_SECONDS:-300}"
export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES="${TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES:-0}"
export LTX_TIME_STAGES=1
export LTX_TRACED="${LTX_TRACED:-1}"          # required: separate s1/s2 traces + gen#1 steady-state replay
export NUM_FRAMES="${NUM_FRAMES:-145}"
export NO_PROMPT="${NO_PROMPT:-1}"
export SEED="${SEED:-10}"

# --- profiling overrides ---
export LTX_FAST="${LTX_FAST:-1}"              # real shipped operating point: all_bf8_lofi quant + fast sigmas
export LTX_S1_SIGMAS="1.0,0.0"                # minimal stage-1 (1 step) — wins over LTX_FAST's setdefault
export LTX_S2_SIGMAS="0.909375,0.0"           # the shipped fast 1-step stage-2 (explicit so it's unambiguous)
export RUN_VBENCH=0                            # perf-only: no quality gate (shapes/times unchanged by it)
export RUN_CLIP=0
# Device-profiler DRAM holds this many programs/RISC before it wraps and drops ops. Default is
# 1000; a whole minimal traced run (2 gens x [s1+s2+vae+audio]) needs more. 30000 is generous;
# parse_tracy.py flags any drop so you can raise it. Pre-warm the prompt embed cache once so the
# Gemma encoder is skipped and does not eat budget on gen#0.
OP_SUPPORT_COUNT="${OP_SUPPORT_COUNT:-30000}"

echo "=== LTX STAGE-2 PROFILE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "buildid: $(git rev-parse --short HEAD)"
echo "env: LTX_FAST=$LTX_FAST QUANT=${LTX_QUANT:-setdefault} S1=$LTX_S1_SIGMAS S2=$LTX_S2_SIGMAS traced=$LTX_TRACED op_support=$OP_SUPPORT_COUNT frames=$NUM_FRAMES"

./python_env/bin/python -m tracy -p -r -v --op-support-count "$OP_SUPPORT_COUNT" \
  -m pytest -q -s \
  models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_pipeline_distilled \
  -k bh_4x8sp1tp0_ring -p no:cacheprovider --timeout=0
rc=$?
echo "=== TRACY RC=$rc ==="

# Newest report the tracy '-r' post-process just wrote.
CSV=$(ls -t "$WT"/generated/profiler/reports/*/ops_perf_results_*.csv 2>/dev/null | head -1)
echo "CSV: ${CSV:-<none found>}"
if [ -n "${CSV:-}" ]; then
  ./python_env/bin/python tmp/parse_tracy.py "$CSV"
fi
exit $rc
