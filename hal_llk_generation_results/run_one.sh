#!/usr/bin/env bash
# =============================================================================
# Run ONE HAL LLK generation run with the run-8 configuration, end to end:
#
#   1. harness    the agent writes the LLKs on top of HAL (scripts/harness/run_hal_gen.sh), then the harness
#                 checks: original test, HAL test after tt-smi -r, original right after HAL (dirty state),
#                 code size, metrics -> <run dir>/logs/SUMMARY.txt
#   2. postcheck  tt-smi -r -> HAL, no reset -> original, no reset -> HAL, tt-smi -r -> HAL (each -k add)
#                 + static TT/TTI counts of the harness build (if ELFRAY is set)
#   3. perf       LLK perf test (perf_eltwise_binary, -k Elwadd), original and HAL alternating PERF_REPS times,
#                 each after tt-smi -r, LLK asserts off -> cycle ratio HAL / original
#   4. static     compile-only builds of the original and HAL test (-k add) with LLK asserts on and off
#                 -> code size (.text), TTI, TT
#
# Usage (inside a container / machine that sees exactly ONE Blackhole board):
#   ./run_one.sh <name>                       # all four steps, in the background; prints the log path
#   CHECK_ONLY=1 ./run_one.sh <name>          # only the checks below, starts nothing
#   FOREGROUND=1 ./run_one.sh <name>          # stay attached
#   RUN_DIR=<run dir> STEPS="postcheck perf static" ./run_one.sh <name>   # later steps on an existing run
#
# Settings (environment variables):
#   VENV        python venv with pytest, tt-smi and pandas (sourced if given; otherwise pytest must be in PATH)
#   CLAUDE_DIR  folder with the claude binary, put first in PATH (the runs used Claude Code 2.1.287)
#   ELFRAY      path to the elfray binary (optional; without it the TT/TTI counts are skipped)
#   WORK        where runs and results go (default /proj_sw/user_dev/$USER/hal-repro, else $HOME/hal-repro)
#   REPO        tt-metal clone holding the three commits below (default: the git checkout this folder is in)
#   PROMPT_SET  8 = prompt of runs 8a-8e (default), 7 = prompt of runs 7a-7e
#   STEPS       default "harness postcheck perf static"
#   PERF_REPS   default 3
#   DRY_RUN=1   harness without the agent (setup + original test only), for trying the setup
#   STRICT_VERSIONS=1  stop if the claude version differs from the runs (default: warning only)
#
# Duration: the agent has the limits of the runs (4 h, 500 turns, $60); in runs 8a-8e it worked 84-223 min and
# cost $22.80-48.44 (RESULTS.md section 5). Harness setup and checks ~30 min, postcheck ~15, perf ~10, static ~15 min.
# =============================================================================
set -uo pipefail

NAME=${1:-}
[[ $NAME =~ ^[A-Za-z0-9_-]+$ ]] || { echo "usage: $0 <name>   (letters, digits, _ or -; e.g. repro1)"; exit 2; }

PKG=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HARNESS=$PKG/scripts/harness
if [[ -z "${WORK:-}" ]]; then
  if [[ -d /proj_sw/user_dev/$USER ]]; then WORK=/proj_sw/user_dev/$USER/hal-repro; else WORK=$HOME/hal-repro; fi
fi
REPO=${REPO:-$(git -C "$PKG" rev-parse --show-toplevel 2>/dev/null || true)}
PROMPT_SET=${PROMPT_SET:-8}
STEPS=${STEPS:-harness postcheck perf static}
PERF_REPS=${PERF_REPS:-3}
CHECK_ONLY=${CHECK_ONLY:-0}
FOREGROUND=${FOREGROUND:-0}
DRY_RUN=${DRY_RUN:-0}
STRICT_VERSIONS=${STRICT_VERSIONS:-0}
ELFRAY=${ELFRAY:-}
RUN_DIR=${RUN_DIR:-}
RUNS_ROOT=$WORK/hal-runs
RESULTS_ROOT=$WORK/hal-results
STATE=$WORK/run_one/$NAME          # this script's own log and progress file

# Configuration of runs 8a-8e (runs/run8*/harness_logs/run_meta.txt)
BASE_COMMIT=c0e38f1ff70b7ee70c526fb44cbbf16736a341ed
HAL_COMMIT=b9e522d39ae534f0281499dae11b425574d89d0d
DOCS_COMMIT=d8c6fa7c22fb89e109178778a537e5860fae0ac6
DOCS_INVALID="address_counters_hal_interface.html replay_hal_interface.html gpr_ops_hal_interface.html atomic_hal_interface.html"
MODEL=claude-opus-5-5
CLAUDE_VERSION_OF_RUNS=2.1.287
case "$PROMPT_SET" in
  8) PROMPT_FILE=$HARNESS/prompt-run6b-cfgcost.md; PROMPT_MD5=3dbea124246f22cae0a3ce1d444d19b7 ;;
  7) PROMPT_FILE=$HARNESS/prompt-run6b.md;         PROMPT_MD5=411819d28e8e8e5766d81caad6b95b15 ;;
  *) echo "PROMPT_SET must be 8 or 7"; exit 2 ;;
esac
# Branches on GitHub that hold the commits above (fetched only if a commit is missing in REPO)
FETCH_REMOTE=${FETCH_REMOTE:-origin}
FETCH_REFS=${FETCH_REFS:-ctr-sbratic/hal-c0e38f1f-movd2a ctr-sbratic/hal-docs-pr59009}

has_step() { [[ " $STEPS " == *" $1 "* ]]; }
now() { date -u +%FT%TZ; }
md5of() { md5sum < "$1" | cut -d' ' -f1; }
die() { echo "STOP: $*" >&2; exit 2; }
warn() { echo "WARNING: $*"; WARNINGS+=("$*"); }
WARNINGS=()

# ------------------------------------------------------------------ checks ---
# Run in the foreground before anything starts, also when the rest goes to the background.
checks() {
  echo "== checks ($(now))"
  [[ -n "${VENV:-}" ]] && { source "$VENV/bin/activate" || die "cannot activate VENV=$VENV"; }
  [[ -n "${CLAUDE_DIR:-}" ]] && export PATH=$CLAUDE_DIR:$PATH
  export PATH=$PATH:$HOME/.local/bin DISABLE_AUTOUPDATER=1
  for c in git python3 pytest tt-smi claude md5sum setsid timeout; do
    command -v $c > /dev/null || die "'$c' is not in PATH (VENV / CLAUDE_DIR?)"
  done
  python3 -c "import pandas" 2> /dev/null || { has_step perf && die "pandas missing in the venv (needed for the perf summary)"; }

  # one board: the harness and the checks run tt-smi -r, which resets every board this machine sees
  local nb; nb=$(ls /dev/tenstorrent 2> /dev/null | wc -l)
  [[ $nb == 1 ]] || die "this machine sees $nb Tenstorrent devices; run inside a container / reservation with exactly one board"
  local busy; busy=$(ps -u "$USER" -o args= | grep -E "[p]ytest|[c]laude -p|[r]un_hal_gen.sh|[t]t-smi" | wc -l)
  [[ $busy == 0 ]] || die "$busy pytest / claude -p / harness / tt-smi processes of $USER are already running here"

  local v; v=$(claude --version 2> /dev/null | awk '{print $1}')
  if [[ "$v" != "$CLAUDE_VERSION_OF_RUNS" ]]; then
    [[ $STRICT_VERSIONS == 1 ]] && die "claude $v, the runs used $CLAUDE_VERSION_OF_RUNS"
    warn "claude $v, the runs used $CLAUDE_VERSION_OF_RUNS (STRICT_VERSIONS=1 makes this a stop)"
  fi
  if has_step harness && [[ $DRY_RUN != 1 ]]; then
    timeout 30 claude auth status < /dev/null 2> /dev/null | grep -q '"loggedIn": true' || die "claude is not logged in (claude auth status)"
  fi

  # harness files: same as in the package (MD5SUMS.txt) and the prompt of the runs
  (cd "$HARNESS" && awk '{print $1 "  " $2}' MD5SUMS.txt | md5sum -c --quiet) || die "files in $HARNESS differ from MD5SUMS.txt"
  [[ $(md5of "$PROMPT_FILE") == "$PROMPT_MD5" ]] || die "$PROMPT_FILE is not the prompt of the runs"

  # the three commits
  [[ -n "$REPO" ]] && git -C "$REPO" rev-parse --git-dir > /dev/null 2>&1 || die "REPO is not set and $PKG is not inside a git checkout"
  local missing=()
  for c in $BASE_COMMIT $HAL_COMMIT $DOCS_COMMIT; do git -C "$REPO" cat-file -e "$c^{commit}" 2> /dev/null || missing+=($c); done
  if (( ${#missing[@]} )) && [[ -n "$FETCH_REFS" ]]; then
    echo "fetching $FETCH_REFS from $FETCH_REMOTE (missing: ${missing[*]})"
    local specs=() r; for r in $FETCH_REFS; do specs+=("+refs/heads/$r:refs/remotes/$FETCH_REMOTE/$r"); done
    git -C "$REPO" fetch -q "$FETCH_REMOTE" "${specs[@]}" || die "git fetch failed"
    missing=()
    for c in $BASE_COMMIT $HAL_COMMIT $DOCS_COMMIT; do git -C "$REPO" cat-file -e "$c^{commit}" 2> /dev/null || missing+=($c); done
  fi
  (( ${#missing[@]} == 0 )) || die "commits missing in $REPO: ${missing[*]}"

  # The agent must not be able to read this folder (generated code of earlier runs). The harness denies the agent
  # REPO, RESULTS_ROOT and the folders next to RUNS_ROOT (see run_hal_gen.sh step 5), so this folder must be
  # inside one of those.
  local pk rp rr top; pk=$(realpath "$PKG"); rp=$(realpath "$REPO"); rr=$(realpath -m "$RUNS_ROOT")
  if [[ $rr == /proj_sw/user_dev/$USER/* ]]; then top=/proj_sw/user_dev/$USER; else top=$(dirname "$rr"); fi
  if [[ $pk == "$rr"/* ]]; then die "$PKG is inside RUNS_ROOT ($RUNS_ROOT); the agent could read it"; fi
  [[ $pk == "$rp"/* || $pk == "$top"/* || $pk == /localdev/$USER/* ]] \
    || die "$PKG is outside REPO and outside $top, so the agent could read it; clone it under REPO or $top"
  if [[ $pk != "$rp"/* && -n "${VIRTUAL_ENV:-}" ]]; then   # the harness does not deny the folder that holds the venv
    local first base
    for base in "$top" /localdev/$USER; do
      [[ $pk == "$base"/* ]] || continue
      first=${pk#"$base"/}; first=$base/${first%%/*}
      [[ $(realpath "$VIRTUAL_ENV")/ == "$first"/* ]] && die "$PKG shares the folder $first with the venv; the agent could read it"
    done
  fi

  if [[ -n "$RUN_DIR" ]]; then
    [[ -f $RUN_DIR/logs/SUMMARY.txt ]] || die "RUN_DIR=$RUN_DIR has no logs/SUMMARY.txt"
    has_step harness && die "RUN_DIR is set: leave 'harness' out of STEPS"
  fi
  [[ -e $STATE ]] && die "$STATE exists (name '$NAME' was used before); pick another name"
  # The harness does not deny the agent $HOME (only ~/.claude and ~/hal_run_*); a tt-metal checkout there holds the
  # original LLKs and the HAL source.
  local h; for h in $(find "$HOME" -maxdepth 4 -type d -path '*/tt_metal/tt-llk' 2> /dev/null); do
    [[ $(realpath "$h") == "$rp"/* ]] || warn "tt-metal checkout in \$HOME the agent could read: ${h%/tt_metal/tt-llk}"
  done
  [[ -z "$ELFRAY" || -x "$ELFRAY" ]] || die "ELFRAY=$ELFRAY is not an executable"
  [[ -n "$ELFRAY" ]] || warn "ELFRAY not set: no TT/TTI counts (code size and cycles are still measured)"

  echo "package:   $PKG"
  echo "repo:      $REPO"
  echo "work:      $WORK  (runs: $RUNS_ROOT)"
  echo "board:     $(ls /dev/tenstorrent | tr '\n' ' ')"
  echo "claude:    $(command -v claude) $v"
  echo "prompt:    $(basename "$PROMPT_FILE") (runs ${PROMPT_SET}a-${PROMPT_SET}e)"
  echo "steps:     $STEPS$([[ $DRY_RUN == 1 ]] && echo ' (DRY_RUN: no agent)')"
  echo "checks OK"
}

# -------------------------------------------------------------- the steps ---
step_harness() {
  local log=$RUNS_ROOT/harness_$(date +%Y%m%d_%H%M%S)_B_$NAME.log
  mkdir -p "$RUNS_ROOT" "$RESULTS_ROOT"
  echo "harness start $(now), log $log" | tee -a "$P"
  (cd "$HARNESS" && BG=0 DRY_RUN=$DRY_RUN MODE=B TESTS="test_eltwise_binary:add" HAL_FILTER=0 PROMPT_FILE=$PROMPT_FILE \
    BASE_COMMIT=$BASE_COMMIT EXPECT_PASSED="test_eltwise_binary:add=1660" \
    HAL_COMMIT=$HAL_COMMIT DOCS_COMMIT=$DOCS_COMMIT DOCS_INVALID="$DOCS_INVALID" \
    MODEL=$MODEL TIMEOUT=4h MAX_TURNS=500 MAX_BUDGET=60 ELFRAY=$ELFRAY \
    REPO=$REPO RUNS_ROOT=$RUNS_ROOT RESULTS_ROOT=$RESULTS_ROOT bash ./run_hal_gen.sh > "$log" 2>&1 < /dev/null)
  local rc=$?
  RUN_DIR=$(grep -oP '^\[harness [0-9:]+\] Run: \K\S+' "$log" | head -1); RUN_DIR=${RUN_DIR%/}
  echo "harness exit $rc $(now), run dir ${RUN_DIR:-NONE}" | tee -a "$P"
  [[ $DRY_RUN == 1 ]] && { echo "DRY_RUN: no agent code, later steps skipped" | tee -a "$P"; return 1; }
  [[ -n "$RUN_DIR" && -f $RUN_DIR/logs/SUMMARY.txt ]] || { echo "no SUMMARY.txt: later steps skipped, see $log" | tee -a "$P"; return 1; }
  local cec; cec=$(grep -oP '^claude_exit_code=\K.*' "$RUN_DIR/logs/run_meta.txt")
  [[ $cec == 0 ]] || { echo "claude_exit_code=$cec (agent did not end normally): later steps skipped" | tee -a "$P"; return 1; }
  return 0
}

pytest_k() {  # <tests dir> <python test> <-k expr> <log>
  (cd "$1" && timeout 60m pytest "python_tests/$2" -k "$3" -q > "$4" 2>&1); echo "exit $?, $(tail -1 "$4")"
}

step_postcheck() {   # = scripts/postcheck/postcheck_7x.sh
  local T=$RUN_DIR/ws/tt_metal/tt-llk/tests L=$RUN_DIR/postcheck_$NAME
  mkdir -p "$L"; export TMPDIR=$RUN_DIR/tmp
  echo "postcheck start $(now)" | tee -a "$P"
  {
    echo "== 1. tt-smi -r, HAL test";     tt-smi -r > "$L/reset1.txt" 2>&1; echo "check1_hal_after_reset.txt: $(pytest_k "$T" test_eltwise_binary_hal.py add "$L/check1_hal_after_reset.txt")"
    echo "== 2. no reset, original test"; echo "check2_original_noreset.txt: $(pytest_k "$T" test_eltwise_binary.py add "$L/check2_original_noreset.txt")"
    echo "== 3. no reset, HAL test";      echo "check3_hal_noreset.txt: $(pytest_k "$T" test_eltwise_binary_hal.py add "$L/check3_hal_noreset.txt")"
    echo "== 4. tt-smi -r, HAL test";     tt-smi -r > "$L/reset4.txt" 2>&1; echo "check4_hal_after_reset.txt: $(pytest_k "$T" test_eltwise_binary_hal.py add "$L/check4_hal_after_reset.txt")"
  } | tee "$L/checks_$NAME.txt"
  if [[ -n "$ELFRAY" ]]; then
    ELFRAY=$ELFRAY python3 "$PKG/scripts/static_counts/tt_tti_counts.py" "$RUN_DIR" --objdump "$T/sfpi/compiler/bin/riscv-tt-elf-objdump" \
      --md "$L/tt_tti_counts.md" --json "$L/tt_tti_counts.json" > "$L/tt_tti_counts_stdout.txt" 2>&1 || echo "tt_tti_counts failed" | tee -a "$P"
  fi
  echo "postcheck end $(now), checks with exit 0: $(grep -c ': exit 0,' "$L/checks_$NAME.txt") of 4" | tee -a "$P"
}

# Hard-link copy of the run's workspace for perf and static builds. Every file that is replaced is removed first
# (hard links are shared with the run dir and must never be written in place).
make_copy_ws() {
  CW=$WORK/copy_ws/$NAME; CT=$CW/tt_metal/tt-llk/tests
  [[ -d $CW ]] || { mkdir -p "$(dirname "$CW")" && cp -al "$RUN_DIR/ws" "$CW"; } || return 1
  rm -rf "$CT/.pytest_cache" "$CW/tt_metal/tt-llk/perf_data"
  rm -f "$CT"/*.log   # test_run*.log, test_errors*.log: the framework writes these in place every session
}

step_perf() {   # = scripts/perf/setup_ws.sh + driver_perf.sh + run_perf.sh + perf_summary_8x.py
  make_copy_ws || { echo "perf: workspace copy failed" | tee -a "$P"; return 1; }
  local O=$WORK/perf/$NAME args=() h
  rm -f "$CT/sources/eltwise_binary_fpu_perf_hal.cpp" "$CT/python_tests/perf_eltwise_binary_hal.py"
  for h in $(ls "$CT/hal_llk"); do args+=(-e "s#^\#include \"$h\"#\#include \"../hal_llk/$h\"#"); done
  sed "${args[@]}" "$CT/sources/eltwise_binary_fpu_perf.cpp" > "$CT/sources/eltwise_binary_fpu_perf_hal.cpp"
  sed 's#sources/eltwise_binary_fpu_perf.cpp#sources/eltwise_binary_fpu_perf_hal.cpp#g' \
    "$CT/python_tests/perf_eltwise_binary.py" > "$CT/python_tests/perf_eltwise_binary_hal.py"
  local nhal norig
  nhal=$(grep -c '^#include "../hal_llk/' "$CT/sources/eltwise_binary_fpu_perf_hal.cpp" || true)
  norig=$(grep '^#include "llk_' "$CT/sources/eltwise_binary_fpu_perf_hal.cpp" | grep -vc '"llk_defs.h"' || true)
  mkdir -p "$O"
  echo "perf kernel: includes to hal_llk $nhal, original llk includes left $norig (runs 7a-8e: 7 and 0)" | tee "$O/SOURCE.txt"
  [[ $norig == 0 && $nhal -gt 0 ]] || { echo "perf: HAL perf kernel check failed" | tee -a "$P"; return 1; }
  echo "perf start $(now), $PERF_REPS x (original, HAL), asserts off" | tee -a "$P"
  local i lab py
  for i in $(seq 1 "$PERF_REPS"); do
    for lab in "p${NAME}_orig_r$i:perf_eltwise_binary.py" "${NAME}_r$i:perf_eltwise_binary_hal.py"; do
      py=${lab#*:}; lab=${lab%%:*}; mkdir -p "$O/$lab"
      tt-smi -r > "$O/$lab/reset.txt" 2>&1
      (cd "$CT" && TT_LLK_DISABLE_ASSERTS=1 PERF_RUN_TAG=$lab PERF_KEEP_RUNS=0 TMPDIR=$WORK/tmp/$lab \
         timeout 90m pytest "python_tests/$py" -k Elwadd -q > "$O/$lab/pytest.txt" 2>&1)
      echo "$lab: exit $?, $(tail -1 "$O/$lab/pytest.txt")" | tee -a "$O/progress.txt"
      mkdir -p "$O/$lab/perf_data"; cp -r "$CW/tt_metal/tt-llk/perf_data/runs/$lab/." "$O/$lab/perf_data/" 2> /dev/null
      rm -rf "$WORK/tmp/$lab"
    done
  done
  python3 "$PKG/scripts/perf/perf_summary_8x.py" "$O" "$O/perf_summary.json" "$NAME" 2>&1 | tee "$O/perf_summary.txt"
  echo "perf end $(now): $(tail -1 "$O/perf_summary.txt")" | tee -a "$P"
}

step_static() {   # = scripts/static_counts/build_counts.sh (no card access)
  make_copy_ws || { echo "static: workspace copy failed" | tee -a "$P"; return 1; }
  local O=$WORK/static_counts/$NAME TOOLS=$CT/sfpi/compiler/bin mode D rc
  echo "static start $(now)" | tee -a "$P"
  for mode in assert_on assert_off; do
    D=$O/$mode; rm -rf "$D"; mkdir -p "$D/codesize_tmp"
    if [[ $mode == assert_off ]]; then export TT_LLK_DISABLE_ASSERTS=1; else unset TT_LLK_DISABLE_ASSERTS; fi
    (cd "$CT" && TMPDIR=$D/codesize_tmp nice -n 19 timeout 2h pytest python_tests/test_eltwise_binary.py \
       python_tests/test_eltwise_binary_hal.py -k add --compile-producer -n 8 -q -p no:cacheprovider > "$D/compile.txt" 2>&1)
    rc=$?
    echo "$mode: compile exit $rc, $(grep -E '(passed|failed|error)' "$D/compile.txt" | tail -1)" | tee -a "$O/status.txt"
    if [[ -n "$ELFRAY" ]]; then
      ELFRAY=$ELFRAY nice -n 19 python3 "$PKG/scripts/static_counts/tt_tti_counts.py" "$D" --objdump "$TOOLS/riscv-tt-elf-objdump" \
        --md "$D/tt_tti_counts.md" --json "$D/tt_tti_counts.json" > "$D/tt_tti_counts_stdout.txt" 2>&1 \
        || echo "$mode: tt_tti_counts failed" | tee -a "$O/status.txt"
    fi
    nice -n 19 python3 "$HARNESS/codesize.py" --artefacts "$D/codesize_tmp/tt-llk-build" --tools "$TOOLS" \
      --orig eltwise_binary_test.cpp --hal eltwise_binary_hal_test.cpp --json "$D/codesize.json" --md "$D/codesize.md" \
      > /dev/null 2>&1 || echo "$mode: codesize failed" | tee -a "$O/status.txt"
  done
  unset TT_LLK_DISABLE_ASSERTS
  echo "static end $(now)" | tee -a "$P"
}

# ------------------------------------------------------------------- main ---
if [[ "${__RUN_ONE_CHILD:-}" != 1 ]]; then
  checks || exit 2
  (( ${#WARNINGS[@]} )) && printf '  warning: %s\n' "${WARNINGS[@]}"
  [[ $CHECK_ONLY == 1 ]] && { echo "CHECK_ONLY=1: nothing started"; exit 0; }
  mkdir -p "$STATE"
  printf '%s\n' "${WARNINGS[@]+"${WARNINGS[@]}"}" > "$STATE/warnings.txt"
  if [[ $FOREGROUND != 1 ]]; then
    __RUN_ONE_CHILD=1 setsid nohup bash "$PKG/$(basename "${BASH_SOURCE[0]}")" "$@" > "$STATE/run_one.log" 2>&1 < /dev/null &
    echo "started in the background (pid $!)"
    echo "progress: tail -f $STATE/progress.txt"
    echo "full log: $STATE/run_one.log"
    echo "stop:     touch $STATE/STOP   (no further step starts; the running step finishes)"
    exit 0
  fi
fi

# environment again in the child (functions above use it)
[[ -n "${VENV:-}" ]] && source "$VENV/bin/activate"
[[ -n "${CLAUDE_DIR:-}" ]] && export PATH=$CLAUDE_DIR:$PATH
export PATH=$PATH:$HOME/.local/bin DISABLE_AUTOUPDATER=1
P=$STATE/progress.txt
echo "run_one $NAME start $(now), host $(hostname), board $(ls /dev/tenstorrent | tr '\n' ' '), run_one.sh md5 $(md5of "${BASH_SOURCE[0]}")" | tee -a "$P"
stopped() { [[ -e $STATE/STOP ]] && { echo "STOP file: $1 not started $(now)" | tee -a "$P"; return 0; }; return 1; }

if has_step harness; then
  step_harness || { echo "run_one end $(now)" | tee -a "$P"; exit 1; }
fi
echo "run dir: $RUN_DIR" | tee -a "$P"
has_step postcheck && ! stopped postcheck && step_postcheck
has_step perf      && ! stopped perf      && step_perf
has_step static    && ! stopped static    && step_static

{
  echo
  echo "==== run_one $NAME: where the results are ===="
  echo "harness summary:  $RUN_DIR/logs/SUMMARY.txt"
  has_step postcheck && echo "postcheck:        $RUN_DIR/postcheck_$NAME/checks_$NAME.txt"
  has_step perf      && echo "cycles:           $WORK/perf/$NAME/perf_summary.json (l1.ratio_median = HAL / original, L1->L1 per tile)"
  has_step static    && echo "static counts:    $WORK/static_counts/$NAME/{assert_on,assert_off}/{codesize,tt_tti_counts}.md"
  echo "Compare with SUMMARY_RUN8.md (asserts-off numbers) and RESULTS.md in $PKG."
} | tee -a "$P"
echo "run_one end $(now)" | tee -a "$P"
