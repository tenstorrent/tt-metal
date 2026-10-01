#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Run the Wan2.2 T2V and I2V unit + e2e test suites on a single host.
#
# Mirrors the CI legs in tests/pipeline_reorg/models_unit_tests.yaml and
# tests/pipeline_reorg/models_e2e_tests.yaml ("TT-DiT Wan2.2-{T2V,I2V}-A14B
# {unit,e2e} tests"), with one deliberate difference: the `not 720p` /
# `resolution_720p`-only filters are dropped, so both 480p and 720p run.
#
# Every leg runs even if an earlier one fails; the exit code is non-zero if any
# leg failed, and a per-leg summary is printed at the end.
#
# Usage:
#   ./run_wan2_2_tests.sh                      # everything, arch auto-detected
#   ./run_wan2_2_tests.sh --suite unit
#   ./run_wan2_2_tests.sh --model i2v
#   ./run_wan2_2_tests.sh --list               # collect only, run nothing
#   ./run_wan2_2_tests.sh -k resolution_480p   # extra -k, ANDed into every leg
#   ./run_wan2_2_tests.sh -- -x --pdb          # trailing args go to pytest

set -uo pipefail

SUITE=all
MODEL=all
ARCH=auto
EXTRA_K=""
LIST_ONLY=0
DRY_RUN=0
FAIL_FAST=0
DEDUPE=1
LOG_DIR=""
PYTEST_EXTRA=()

usage() {
    sed -n '6,22p' "$0" | sed 's/^# \?//'
    cat <<'EOF'

Options:
  --suite {unit,e2e,all}   Which suite(s) to run (default: all)
  --model {t2v,i2v,all}    Which model(s) to run (default: all)
  --arch {bh,wh,auto}      Pick the galaxy-specific parametrizations (default: auto)
  -k EXPR                  Extra pytest -k expression, ANDed into every leg
  --list                   pytest --collect-only -q for each leg
  --dry-run                Print the commands without running them
  --fail-fast              Stop after the first failing leg
  --no-dedupe              Also run the e2e pipeline legs when --suite all
                           (by default they are skipped: the unit legs already
                           run test_pipeline_wan{,_i2v}.py across every mesh
                           config and both resolutions)
  --log-dir DIR            Where to write per-leg logs
                           (default: $TT_METAL_HOME/generated/wan2_2_tests/<ts>)
  -h, --help               This message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --suite)     SUITE="$2"; shift 2 ;;
        --model)     MODEL="$2"; shift 2 ;;
        --arch)      ARCH="$2"; shift 2 ;;
        -k)          EXTRA_K="$2"; shift 2 ;;
        --list)      LIST_ONLY=1; shift ;;
        --dry-run)   DRY_RUN=1; shift ;;
        --fail-fast) FAIL_FAST=1; shift ;;
        --no-dedupe) DEDUPE=0; shift ;;
        --log-dir)   LOG_DIR="$2"; shift 2 ;;
        -h|--help)   usage; exit 0 ;;
        --)          shift; PYTEST_EXTRA=("$@"); break ;;
        *)           echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$SUITE" in unit|e2e|all) ;; *) echo "--suite must be unit|e2e|all" >&2; exit 2 ;; esac
case "$MODEL" in t2v|i2v|all) ;; *) echo "--model must be t2v|i2v|all" >&2; exit 2 ;; esac
case "$ARCH"  in bh|wh|auto)  ;; *) echo "--arch must be bh|wh|auto" >&2; exit 2 ;; esac

# ---------------------------------------------------------------------------
# Repo / arch discovery
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${TT_METAL_HOME:=$(cd "${SCRIPT_DIR}/../../../../.." && pwd)}"
export TT_METAL_HOME
export PYTHONPATH="${PYTHONPATH:-${TT_METAL_HOME}}"

if [[ "$ARCH" == auto ]]; then
    if [[ -n "${ARCH_NAME:-}" ]]; then
        case "$ARCH_NAME" in *blackhole*) ARCH=bh ;; *wormhole*) ARCH=wh ;; esac
    fi
fi
if [[ "$ARCH" == auto ]]; then
    smi="$(tt-smi -ls 2>/dev/null)"
    if   grep -qi blackhole <<<"$smi"; then ARCH=bh
    elif grep -qi wormhole  <<<"$smi"; then ARCH=wh
    else echo "could not detect arch, pass --arch bh|wh" >&2; exit 2
    fi
fi

# Galaxy-specific parametrization ids. The 4x8 mesh cases carry
# require_exact_physical_num_devices, so on a non-galaxy host they self-skip.
if [[ "$ARCH" == bh ]]; then
    GLX_ID="bh_glx"                             # umt5 / embeddings: 4x8, 2 links
    PERF_ID="ring_bh_4x8_sp1tp0"                # test_performance_wan.py
    PIPE_T2V_ID="4x8sp1tp0nl2_ring_is_fsdp0"    # test_pipeline_wan.py
    PIPE_I2V_ID="4x8sp1tp0nl2_linear_is_fsdp0"  # test_pipeline_wan_i2v.py (no 4x8 ring case)
    I2V_UNIT_K=""                               # BH has the DRAM for is_fsdp0
else
    GLX_ID="wh_glx"                             # umt5 / embeddings: 4x8, 4 links
    PERF_ID="wh_4x8_sp1tp0"
    PIPE_T2V_ID="4x8sp1tp0nl4_ring_is_fsdp1"
    PIPE_I2V_ID="4x8sp1tp0nl4_ring_is_fsdp1"
    I2V_UNIT_K="is_fsdp1"                       # wh_galaxy OOMs on is_fsdp0 (CI hack)
fi

TS="$(date +%Y%m%d-%H%M%S)"
: "${LOG_DIR:=${TT_METAL_HOME}/generated/wan2_2_tests/${TS}}"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Environment (from the CI legs)
# ---------------------------------------------------------------------------
unset TT_METAL_WATCHER   # blocked by #50886
unset HF_HUB_OFFLINE     # umt5 fetches the Wan2.2 tokenizer from the hub
export NO_PROMPT=1
export TT_DIT_CACHE_DIR="${TT_DIT_CACHE_DIR:-/tmp/TT_DIT_CACHE}"
if [[ -z "${VBENCH_CACHE_DIR:-}" ]]; then
    for d in /mnt/MLPerf/vbench /mnt/models/vbench; do
        [[ -d "$d" ]] && { export VBENCH_CACHE_DIR="$d" TORCH_HOME="$d/torch_hub"; break; }
    done
fi
[[ -n "${VBENCH_CACHE_DIR:-}" ]] || echo "WARNING: no vbench cache found; quality checks may download" >&2

WAN_DIR="models/tt_dit/tests/models/wan2_2"

# ---------------------------------------------------------------------------
# Leg runner
# ---------------------------------------------------------------------------
declare -a RESULTS=()
FAILED=0

# run <name> <timeout> <target> [k_expr] [extra pytest args...]
run() {
    local name="$1" timeout="$2" target="$3" kexpr="${4:-}"
    shift 4 2>/dev/null || shift $#

    if [[ -n "$EXTRA_K" ]]; then
        if [[ -n "$kexpr" ]]; then kexpr="($kexpr) and ($EXTRA_K)"; else kexpr="$EXTRA_K"; fi
    fi

    local -a cmd=(pytest --timeout "$timeout" "$target")
    [[ -n "$kexpr" ]] && cmd+=(-k "$kexpr")
    cmd+=("$@")
    (( LIST_ONLY )) && cmd+=(--collect-only -q)
    cmd+=("${PYTEST_EXTRA[@]+"${PYTEST_EXTRA[@]}"}")

    local log="${LOG_DIR}/${name}.log"
    echo
    echo "=============================================================================="
    echo "[$name]  ${cmd[*]}"
    echo "  log: $log"
    echo "=============================================================================="

    if (( DRY_RUN )); then
        RESULTS+=("DRYRUN  $name")
        return 0
    fi

    local start=$SECONDS
    ( cd "$TT_METAL_HOME" && "${cmd[@]}" ) 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    local dur=$(( SECONDS - start ))

    # The pipeline tests save generated video into the cwd (wan_t2v_*.mp4,
    # wan_i2v_*.mp4, wan_output_video_*.mp4). Keep them with the logs instead of
    # leaving them in the repo root.
    shopt -s nullglob
    for vid in "$TT_METAL_HOME"/wan_*.mp4; do mv -f "$vid" "$LOG_DIR/"; done
    shopt -u nullglob

    if (( rc == 0 )); then
        RESULTS+=("$(printf 'PASS    %-34s %5ds' "$name" "$dur")")
    else
        RESULTS+=("$(printf 'FAIL(%d) %-34s %5ds  %s' "$rc" "$name" "$dur" "$log")")
        FAILED=1
        if (( FAIL_FAST )); then
            summary
            exit 1
        fi
    fi
}

skip_leg() {
    RESULTS+=("$(printf 'SKIP    %-34s        %s' "$1" "$2")")
    echo
    echo "[skip] $1 -- $2"
}

summary() {
    echo
    echo "=============================================================================="
    echo "Wan2.2 test summary   (suite=$SUITE model=$MODEL arch=$ARCH)"
    echo "logs: $LOG_DIR"
    echo "=============================================================================="
    printf '%s\n' "${RESULTS[@]+"${RESULTS[@]}"}"
}

want() { [[ "$1" == all || "$1" == "$2" ]]; }

# The unit legs run test_pipeline_wan{,_i2v}.py over every mesh config at both
# resolutions, which is a superset of the CI e2e pipeline legs. Skip those in
# --suite all unless --no-dedupe.
pipeline_covered_by_unit() { (( DEDUPE )) && [[ "$SUITE" == all ]]; }

# ---------------------------------------------------------------------------
# Unit legs
# ---------------------------------------------------------------------------
if want "$SUITE" unit; then
    if want "$MODEL" t2v; then
        run t2v-unit-rope         1000  "$WAN_DIR/test_rope.py"           ""
        run t2v-unit-attention    1000  "$WAN_DIR/test_attention_wan.py"  ""
        # CI: -k "not f32 and not 720p and not f81"; 720p kept, f81 ids are
        # commented out upstream so that clause was dead. f32 stays excluded
        # (float32 variants are covered by the bf16 pass and are very slow).
        run t2v-unit-vae         10000  "$WAN_DIR/test_vae_wan2_1.py"     "not f32"  --maxfail 10
        run t2v-unit-transformer 10000  "$WAN_DIR/test_transformer_wan.py" ""        --maxfail 10
        run t2v-unit-pipeline    10000  "$WAN_DIR/test_pipeline_wan.py"    ""        --maxfail 10
        run t2v-unit-umt5         1000  "models/tt_dit/tests/encoders/umt5/test_umt5.py" "$GLX_ID"
        run t2v-unit-embeddings   1000  "models/tt_dit/tests/unit/test_embeddings.py::test_wan_time_text_image_embedding" "$GLX_ID"
    fi
    if want "$MODEL" i2v; then
        run i2v-unit-pipeline    10000  "$WAN_DIR/test_pipeline_wan_i2v.py" "$I2V_UNIT_K" --maxfail 10
    fi
fi

# ---------------------------------------------------------------------------
# e2e legs
# ---------------------------------------------------------------------------
if want "$SUITE" e2e; then
    if want "$MODEL" t2v; then
        run t2v-e2e-perf-480p     2700  "$WAN_DIR/test_performance_wan.py" "$PERF_ID and resolution_480p and t2v"
        run t2v-e2e-perf-720p     1200  "$WAN_DIR/test_performance_wan.py" "$PERF_ID and resolution_720p and t2v"
        if pipeline_covered_by_unit; then
            skip_leg t2v-e2e-pipeline "covered by t2v-unit-pipeline (--no-dedupe to run)"
        else
            run t2v-e2e-pipeline  3000  "$WAN_DIR/test_pipeline_wan.py" "$PIPE_T2V_ID"
        fi
    fi
    if want "$MODEL" i2v; then
        run i2v-e2e-perf-480p     2700  "$WAN_DIR/test_performance_wan.py" "$PERF_ID and resolution_480p and i2v"
        run i2v-e2e-perf-720p     1200  "$WAN_DIR/test_performance_wan.py" "$PERF_ID and resolution_720p and i2v"
        if pipeline_covered_by_unit; then
            skip_leg i2v-e2e-pipeline "covered by i2v-unit-pipeline (--no-dedupe to run)"
        else
            run i2v-e2e-pipeline  3000  "$WAN_DIR/test_pipeline_wan_i2v.py" "$PIPE_I2V_ID"
        fi
    fi
fi

summary
exit $FAILED
