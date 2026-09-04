#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Run the FLUX.2 unit + e2e test suites on a single host.
#
# Unlike Wan2.2, FLUX.2 has NO CI legs in tests/pipeline_reorg/*.yaml, so the
# legs below are derived from the test files themselves rather than mirrored
# from a pipeline. Two consequences worth knowing:
#
#  * The flux2 device_params do NOT set require_exact_physical_num_devices, so a
#    (2,2) or (1,8) parametrization does NOT self-skip on a 32-chip galaxy -- it
#    opens a submesh and really runs. Legs therefore pin the mesh id matching the
#    host instead of relying on self-skip. Use --mesh/-k to widen.
#  * black-forest-labs/FLUX.2-dev is a manually gated HF repo (CI is blocked on
#    it, see #54499) and is not staged under /mnt/models. You need an HF token
#    on its allowlist; see PREREQUISITES below.
#
# All resolutions run by default (1024 -> 8192 for perf); cap with --max-res.
#
# Usage:
#   ./run_flux2_tests.sh                       # everything, arch auto-detected
#   ./run_flux2_tests.sh --suite unit
#   ./run_flux2_tests.sh --max-res 2048        # skip the 4096/8192 perf legs
#   ./run_flux2_tests.sh --list                # collect only, run nothing
#   ./run_flux2_tests.sh -- -x                 # trailing args go to pytest

set -uo pipefail

SUITE=all
ARCH=auto
MESH=auto
EXTRA_K=""
MAX_RES=8192
WITH_PROFILE=0
LIST_ONLY=0
DRY_RUN=0
FAIL_FAST=0
LOG_DIR=""
PYTEST_EXTRA=()

usage() {
    sed -n '6,28p' "$0" | sed 's/^# \?//'
    cat <<'EOF'

Options:
  --suite {unit,e2e,all}   Which suite(s) to run (default: all)
  --arch {bh,wh,auto}      Pick the arch-specific parametrizations (default: auto)
  --mesh ID                Override the mesh-id filter for every leg (advanced;
                           the per-test ids differ, so this is usually wrong --
                           prefer -k)
  -k EXPR                  Extra pytest -k expression, ANDed into every leg
  --max-res N              Skip perf/profile resolutions above N
                           (N in 1024 2048 4096 8192; default 8192)
  --with-profile           Also run test_transformer_profile (72 heavy cases,
                           a profiling sweep, not a correctness test)
  --list                   pytest --collect-only -q for each leg
  --dry-run                Print the commands without running them
  --fail-fast              Stop after the first failing leg
  --log-dir DIR            Where to write per-leg logs and generated images
                           (default: $TT_METAL_HOME/generated/flux2_tests/<ts>)
  -h, --help               This message

PREREQUISITES
  * Activate the venv:  source $TT_METAL_HOME/python_env/bin/activate
  * FLUX.2-dev is gated. Request access at
    https://huggingface.co/black-forest-labs/FLUX.2-dev then either
    `huggingface-cli login` or export HF_TOKEN=<token>. Without it every leg
    fails with a 403 GatedRepoError.
  * First run downloads the full FLUX.2-dev checkpoint (tens of GB); it is not
    pre-staged under /mnt/models (only FLUX.1-dev/schnell are).
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --suite)        SUITE="$2"; shift 2 ;;
        --arch)         ARCH="$2"; shift 2 ;;
        --mesh)         MESH="$2"; shift 2 ;;
        -k)             EXTRA_K="$2"; shift 2 ;;
        --max-res)      MAX_RES="$2"; shift 2 ;;
        --with-profile) WITH_PROFILE=1; shift ;;
        --list)         LIST_ONLY=1; shift ;;
        --dry-run)      DRY_RUN=1; shift ;;
        --fail-fast)    FAIL_FAST=1; shift ;;
        --log-dir)      LOG_DIR="$2"; shift 2 ;;
        -h|--help)      usage; exit 0 ;;
        --)             shift; PYTEST_EXTRA=("$@"); break ;;
        *)              echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$SUITE"   in unit|e2e|all) ;; *) echo "--suite must be unit|e2e|all" >&2; exit 2 ;; esac
case "$ARCH"    in bh|wh|auto)   ;; *) echo "--arch must be bh|wh|auto" >&2; exit 2 ;; esac
case "$MAX_RES" in 1024|2048|4096|8192) ;; *) echo "--max-res must be 1024|2048|4096|8192" >&2; exit 2 ;; esac

# ---------------------------------------------------------------------------
# Repo / arch discovery
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${TT_METAL_HOME:=$(cd "${SCRIPT_DIR}/../../../../.." && pwd)}"
export TT_METAL_HOME
export PYTHONPATH="${PYTHONPATH:-${TT_METAL_HOME}}"

if [[ "$ARCH" == auto && -n "${ARCH_NAME:-}" ]]; then
    case "$ARCH_NAME" in *blackhole*) ARCH=bh ;; *wormhole*) ARCH=wh ;; esac
fi
if [[ "$ARCH" == auto ]]; then
    smi="$(tt-smi -ls 2>/dev/null)"
    if   grep -qi blackhole <<<"$smi"; then ARCH=bh
    elif grep -qi wormhole  <<<"$smi"; then ARCH=wh
    else echo "could not detect arch, pass --arch bh|wh" >&2; exit 2
    fi
fi

# Per-test mesh parametrization ids. The ids differ between files, hence one
# variable per test rather than a single global mesh filter.
VAE_ID="4x8"            # test_vae_flux2.py         also has 1x1, 1x8
ENC_ID="4x8"            # test_prompt_encoder.py    only 4x8 exists
if [[ "$ARCH" == bh ]]; then
    XF_ID="bh_4x8_ring"     # test_transformer_flux2.py::test_transformer
    PIPE_ID="bh_4x8"        # test_pipeline_flux2.py
    PERF_IDS="bh_glx_linear or bh_glx_ring_sp0tp1_nofsdp or bh_glx_ring_sp1tp0 or bh_glx_ring_sp0tp1_fsdp"
    PROFILE_IDS="bh_4x8_ring_nofsdp or bh_4x8_ring_sp1tp0_nofsdp or bh_4x8_ring_fsdp"
else
    XF_ID="wh_2x4_linear"
    PIPE_ID="wh_4x8"
    PERF_IDS=""             # test_performance_flux2.py parametrizes BH meshes only
    PROFILE_IDS=""          # test_transformer_profile is BH-only too
fi
if [[ "$MESH" != auto ]]; then
    VAE_ID="$MESH"; ENC_ID="$MESH"; XF_ID="$MESH"; PIPE_ID="$MESH"
    PERF_IDS="$MESH"; PROFILE_IDS="$MESH"
fi

# Resolutions, capped by --max-res.
ALL_RES=(1024 2048 4096 8192)
RES=()
for r in "${ALL_RES[@]}"; do (( r <= MAX_RES )) && RES+=("$r"); done

TS="$(date +%Y%m%d-%H%M%S)"
: "${LOG_DIR:=${TT_METAL_HOME}/generated/flux2_tests/${TS}}"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Environment (same conventions as the Flux.1 CI legs)
# ---------------------------------------------------------------------------
unset TT_METAL_WATCHER   # blocked by #50886
unset HF_HUB_OFFLINE     # FLUX.2-dev is not pre-staged; the hub is the source
export NO_PROMPT=1       # test_pipeline_flux2 blocks on input() without this
export TT_DIT_CACHE_DIR="${TT_DIT_CACHE_DIR:-/tmp/TT_DIT_CACHE}"
if [[ -z "${HF_TOKEN:-}" && ! -f "${HF_HOME:-$HOME/.cache/huggingface}/token" ]]; then
    echo "WARNING: no HF credentials found (HF_TOKEN unset, no cached token)." >&2
    echo "         black-forest-labs/FLUX.2-dev is gated; expect 403 GatedRepoError." >&2
fi

FLUX2_DIR="models/tt_dit/tests/models/flux2"

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

    # test_pipeline_flux2 saves flux2_*.png into the cwd; keep them with the logs.
    shopt -s nullglob
    for png in "$TT_METAL_HOME"/flux2_*.png; do mv -f "$png" "$LOG_DIR/"; done
    shopt -u nullglob

    if (( rc == 0 )); then
        RESULTS+=("$(printf 'PASS    %-34s %5ds' "$name" "$dur")")
    else
        RESULTS+=("$(printf 'FAIL(%d) %-34s %5ds  %s' "$rc" "$name" "$dur" "$log")")
        FAILED=1
        if (( FAIL_FAST )); then summary; exit 1; fi
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
    echo "FLUX.2 test summary   (suite=$SUITE arch=$ARCH max-res=$MAX_RES)"
    echo "logs/images: $LOG_DIR"
    echo "=============================================================================="
    printf '%s\n' "${RESULTS[@]+"${RESULTS[@]}"}"
}

want() { [[ "$1" == all || "$1" == "$2" ]]; }

# ---------------------------------------------------------------------------
# Unit legs -- PCC/correctness against the torch reference
# ---------------------------------------------------------------------------
if want "$SUITE" unit; then
    # VAE decoder, pruned to 1 resnet per up_block by the test itself.
    run flux2-unit-vae          3000 "$FLUX2_DIR/test_vae_flux2.py"         "$VAE_ID"
    # Mistral3 prompt encoder + the prompt-embedding upsampler.
    run flux2-unit-encoder      3000 "$FLUX2_DIR/test_prompt_encoder.py"    "$ENC_ID"
    # Both all_blocks and single_blocks variants at 1024x1024.
    run flux2-unit-transformer 10000 "$FLUX2_DIR/test_transformer_flux2.py::test_transformer" "$XF_ID"
    # 12-step 1024x1024 generation; writes flux2_*.png next to the logs.
    run flux2-unit-pipeline    10000 "$FLUX2_DIR/test_pipeline_flux2.py"    "$PIPE_ID"
fi

# ---------------------------------------------------------------------------
# e2e legs -- end-to-end throughput, one leg per resolution
# ---------------------------------------------------------------------------
if want "$SUITE" e2e; then
    if [[ -z "$PERF_IDS" ]]; then
        skip_leg flux2-e2e-perf "test_performance_flux2.py parametrizes Blackhole meshes only"
    else
        for r in "${RES[@]}"; do
            run "flux2-e2e-perf-${r}" 10000 "$FLUX2_DIR/test_performance_flux2.py" \
                "($PERF_IDS) and ${r}x${r}"
        done
    fi

    if (( WITH_PROFILE )); then
        if [[ -z "$PROFILE_IDS" ]]; then
            skip_leg flux2-e2e-transformer-profile "test_transformer_profile is Blackhole-only"
        else
            for r in "${RES[@]}"; do
                (( r > 4096 )) && continue   # profile sweep tops out at 4096
                run "flux2-e2e-xf-profile-${r}" 10000 \
                    "$FLUX2_DIR/test_transformer_flux2.py::test_transformer_profile" \
                    "($PROFILE_IDS) and ${r}"
            done
        fi
    else
        skip_leg flux2-e2e-transformer-profile "profiling sweep, opt in with --with-profile"
    fi
fi

summary
exit $FAILED
