#!/usr/bin/env bash
# =============================================================================
# HAL LLK generation harness  (v4: modes A and B, background operation, sweep over multiple tests)
#
#   Mode A (translation):  Claude sees the original LLKs + the HAL documentation.
#   Mode B (from scratch): Claude sees only the LLK function signatures (spec), the test and
#                          the HAL documentation; the original LLKs are forbidden to it.
#
# At the end the results are archived automatically to ~/hal-results/<run_id>.tar.gz
# (+ unpacked SUMMARY, report and metadata in ~/hal-results/<run_id>/).
#
# Usage:
#   MODE=B DOCS_DIR=~/hal-docs-snapshot ./run_hal_gen.sh   # HAL from the PR + Confluence snapshot
#   MODE=B CONFLUENCE=1 ./run_hal_gen.sh    # HAL from the PR + Confluence via MCP
#   MODE=B ./run_hal_gen.sh                 # runs in the background, survives closing VS Code
#   MODE=A ./run_hal_gen.sh
#   BG=0 MODE=B ./run_hal_gen.sh            # in the foreground (for debugging)
#   MODE=B BASE_COMMIT=<hash> MODEL=sonnet TIMEOUT=6h ./run_hal_gen.sh
#   MODE=B TESTS="test_eltwise_binary test_pack:k_expr" ./run_hal_gen.sh   # sweep (mode B only)
#   DRY_RUN=1 MODE=B TESTS="..." ./run_hal_gen.sh   # preparation + original tests only, without Claude
#   BASE_COMMIT="" EXPECT_PASSED="" MODE=B HAL_COMMIT=<hash> ./run_hal_gen.sh   # without the frozen test set (1660)
#
# Watching: ./watch_run.sh
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------- parameters ---
MODE=${MODE:-A}
# TRIVIAL = test validity check: the original LLKs with empty bodies must fail (without Claude)
# RAW / RAWNODOCS = the same task without HAL, with raw reference material (with / without ISA documentation)
case "$MODE" in A|B|RAW|RAWNODOCS|TRIVIAL) ;; *) echo "MODE must be A, B, RAW, RAWNODOCS or TRIVIAL"; exit 1;; esac
is_gen_mode() { [[ "$MODE" == B || "$MODE" == RAW || "$MODE" == RAWNODOCS ]]; }   # the agent writes from scratch (spec)
is_raw_mode() { [[ "$MODE" == RAW || "$MODE" == RAWNODOCS ]]; }
HAL_COMMIT=${HAL_COMMIT:-pr59009}      # source of the HAL implementation (git fetch origin pull/59009/head:pr59009)
# Frozen test set (decision of 28.10.): BASE = c0e38f1f (runs 3-4, eltwise add = 1660 passed), kept in the repo
# by the local branch ctr-sbratic/frozen-tests-1660. BASE_COMMIT="" = old behaviour: HAL_COMMIT if its
# LLKs do not use HAL, otherwise merge-base(MAIN_REF, HAL_COMMIT).
BASE_COMMIT=${BASE_COMMIT-c0e38f1ff70b7ee70c526fb44cbbf16736a341ed}
# "test[:k]=N ...": the original test must give exactly N passed, otherwise the run stops after the baseline. "" = no check.
EXPECT_PASSED=${EXPECT_PASSED-test_eltwise_binary:add=1660}
# "Rules as in run 5" variant (28.10.): the protection from 21.10. can be switched off, a ready-made prompt can be supplied.
HAL_FILTER=${HAL_FILTER:-1}            # 1 = compiler output filter + deny the build folder (since 21.10.); 0 = as in run 5
PROMPT_FILE=${PROMPT_FILE:-}           # ready-made prompt (e.g. prompt_used.md of run 5); empty = prompt-base.md + api-*.md
SIBLING_DENY=${SIBLING_DENY:-1}        # 1 = the agent does not read other folders next to RUNS_ROOT nor in /localdev/$USER (except the venv)
MAIN_REF=${MAIN_REF:-origin/main}
DOCS_COMMIT=${DOCS_COMMIT:-6f965e1e}   # where the HTML documentation (docs/hal/) is taken from
DOCS_DIR=${DOCS_DIR:-}                 # folder with a documentation snapshot (e.g. a Confluence export); takes precedence
DOCS_INVALID=${DOCS_INVALID:-address_counters_hal_interface.html replay_hal_interface.html}  # get a warning
DOCS_REMOVE=${DOCS_REMOVE:-}           # removed completely from docs/hal/
CONFLUENCE=${CONFLUENCE:-0}            # 1 = Claude reads Confluence via MCP (the server must be configured in Claude Code)
MCP_SERVER=${MCP_SERVER:-atlassian}    # name of the MCP server in Claude Code (claude mcp list)
CONFLUENCE_PAGE=${CONFLUENCE_PAGE:-https://tenstorrent.atlassian.net/wiki/spaces/LLK/pages/2708341004/Blackhole+HAL}
REPO=${REPO:-$HOME/tt-metal}
RUNS_ROOT=${RUNS_ROOT:-/localdev/$USER/hal-runs}
MODEL=${MODEL:-opus}
SKIP_BASELINE=${SKIP_BASELINE:-0}
OP_FILTER=${OP_FILTER:-add}
# List of tests: "python_test_name[:k_expr]" separated by spaces. Without ':' there is no -k filter.
TESTS=${TESTS:-test_eltwise_binary:$OP_FILTER}
read -r -a TEST_ENTRIES <<< "$TESTS"
NTESTS=${#TEST_ENTRIES[@]}
(( NTESTS > 0 )) || { echo "TESTS is empty"; exit 1; }
[[ "$MODE" == A && "$NTESTS" -gt 1 ]] && { echo "Mode A supports only one test (prompt-A.md)"; exit 1; }
if [[ "$MODE" != A ]]; then TIMEOUT=${TIMEOUT:-$(( NTESTS == 1 ? 4 : 2 + 2 * NTESTS ))h}; MAX_TURNS=${MAX_TURNS:-$(( NTESTS == 1 ? 500 : 300 * NTESTS ))};
else                        TIMEOUT=${TIMEOUT:-2h}; MAX_TURNS=${MAX_TURNS:-300}; fi
TEST_TIMEOUT=${TEST_TIMEOUT:-60m}      # timeout for one run of one test (baseline / final check)
TRIV_TIMEOUT=${TRIV_TIMEOUT:-4h}       # mode TRIVIAL: empty bodies take ~2 s per variant, so it needs longer
DRY_RUN=${DRY_RUN:-0}                  # 1 = preparation + original tests, without Claude
DIRTY_CHECK=${DIRTY_CHECK:-1}          # 1 = after each final test run the original without a reset (dirty state)
MAX_BUDGET=${MAX_BUDGET:-}             # spending limit in USD for one run (--max-budget-usd); empty = no limit
ELFRAY=${ELFRAY:-}                     # path to the elfray binary (empty = no TT/TTI counting)
# Reference material for the RAW arm (TEMPORARY, until the team decides): paths in the repo, copied into raw_ref/
RAW_REFMAT=${RAW_REFMAT:-tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_ops.h tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_instr_params.h tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_defs.h tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_structs.h tt_metal/hw/inc/internal/tt-1xx/blackhole/cfg_defines.h}
RAW_ISA_DIR=${RAW_ISA_DIR:-}           # local folder with the ISA documentation (e.g. tt-isa-documentation/BlackholeA0); required for MODE=RAW
FULL_REPO=${FULL_REPO:-1}              # 1 = the whole tt-metal (needed: tests/sfpi-info.sh points outside tt-llk)
RESULTS_ROOT=${RESULTS_ROOT:-$HOME/hal-results}   # permanent result archives (home survives the release)
CODESIZE_SAMPLE=${CODESIZE_SAMPLE:-200} # how many variants to measure for code size (0 = all)
BG=${BG:-1}

HARNESS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ------------------------------------------- running in the background (BG=1) ---
if [[ "$BG" == 1 && -z "${__HAL_CHILD:-}" ]]; then
  mkdir -p "$RUNS_ROOT"
  LOG="$RUNS_ROOT/harness_$(date +%Y%m%d_%H%M%S)_$MODE.log"
  __HAL_CHILD=1 setsid nohup "$0" "$@" > "$LOG" 2>&1 < /dev/null &
  echo "Started in the background (PID $!), mode $MODE."
  echo "Script log:   tail -f $LOG"
  echo "Watching:     $HARNESS_DIR/watch_run.sh"
  echo "Stopping:     pkill -f 'claude -p'; pkill -f run_hal_gen.sh"
  exit 0
fi

LLK=tt_metal/tt-llk
BH=$LLK/tt_llk_blackhole
HAL_SRC=$BH/llk_lib/hal
HAL_DOCS=$LLK/docs/hal
SPEC_DIR=$LLK/llk_spec/tt_llk_blackhole
OUT_DIR=$LLK/tests/hal_llk
PY_DIR=$LLK/tests/python_tests
SRC_DIR=$LLK/tests/sources

log() { echo -e "\n[harness $(date +%H:%M:%S)] $*"; }
die() { echo -e "\n[harness] ERROR: $*" >&2; exit 1; }

case "$MODE" in
  A)         NEED=(prompt-A.md hal-settings-A.json) ;;
  B)         NEED=(prompt-base.md api-hal.md hal-settings-B.json); API_FILE=api-hal.md ;;
  RAW)       NEED=(prompt-base.md api-raw.md hal-settings-RAW.json); API_FILE=api-raw.md ;;
  RAWNODOCS) NEED=(prompt-base.md api-rawnodocs.md hal-settings-RAWNODOCS.json); API_FILE=api-rawnodocs.md ;;
  TRIVIAL)   NEED=() ;;
esac
for f in ${NEED[@]+"${NEED[@]}"}; do [[ -f "$HARNESS_DIR/$f" ]] || die "Missing $HARNESS_DIR/$f"; done
[[ "$MODE" == RAW && ! -d "$RAW_ISA_DIR" ]] && die "MODE=RAW requires RAW_ISA_DIR (folder with the ISA documentation); without it use MODE=RAWNODOCS"
[[ "$MODE" != A && ! -f "$HARNESS_DIR/make_spec.py" ]] && die "Missing $HARNESS_DIR/make_spec.py"
for f in metrics.py codesize.py codestats.py elfray_tt_tti.py; do [[ -f "$HARNESS_DIR/$f" ]] || die "Missing $HARNESS_DIR/$f"; done
command -v claude >/dev/null || die "claude is not in PATH"
command -v pytest >/dev/null || die "pytest is not in PATH (activate the venv)"

# ------------------------------------------------------------ repo checks ---
git -C "$REPO" rev-parse --verify -q "$HAL_COMMIT^{commit}" >/dev/null \
  || die "HAL_COMMIT '$HAL_COMMIT' does not exist in $REPO (git fetch the HAL docs branch?)"
HAL_SHA=$(git -C "$REPO" rev-parse "$HAL_COMMIT")

llk_hal_users() {  # LLK files at commit $1 that include hal/
  git -C "$REPO" grep -lE '#include *"[^"]*hal/' "$1" -- \
    "$BH/llk_lib/*.h" "$BH/common/inc/*.h" ":(exclude)$HAL_SRC" 2>/dev/null | sed "s#^$1:##" || true
}
if [[ -z "$BASE_COMMIT" ]]; then
  if [[ -z "$(llk_hal_users "$HAL_SHA")" ]]; then
    BASE_COMMIT=$HAL_SHA
    log "BASE_COMMIT = HAL_COMMIT (${HAL_SHA:0:8}): the LLKs on it do not use HAL"
  else
    git -C "$REPO" rev-parse --verify -q "$MAIN_REF^{commit}" >/dev/null \
      || die "$MAIN_REF does not exist in $REPO (git fetch origin main?)"
    BASE_COMMIT=$(git -C "$REPO" merge-base "$MAIN_REF" "$HAL_SHA") \
      || die "No merge-base between $MAIN_REF and $HAL_COMMIT"
    log "BASE_COMMIT = merge-base($MAIN_REF, $HAL_COMMIT) = ${BASE_COMMIT:0:8}"
  fi
fi
git -C "$REPO" rev-parse --verify -q "$BASE_COMMIT^{commit}" >/dev/null || die "BASE_COMMIT '$BASE_COMMIT' does not exist"
BASE_SHA=$(git -C "$REPO" rev-parse "$BASE_COMMIT")

# For each test: python test -> the only C++ source it uses -> names of the HAL copies.
T_NAME=(); T_K=(); T_PY=(); T_HALPY=(); T_CPP=(); T_HALCPP=()
for e in "${TEST_ENTRIES[@]}"; do
  name=${e%%:*}; k=""; [[ "$e" == *:* ]] && k=${e#*:}
  py=$PY_DIR/$name.py
  git -C "$REPO" cat-file -e "$BASE_SHA:$py" 2>/dev/null || die "$py does not exist at the BASE commit"
  srcs=$(git -C "$REPO" show "$BASE_SHA:$py" | grep -oE 'sources/[A-Za-z0-9_]+\.cpp' | sort -u)
  [[ $(echo "$srcs" | grep -c .) == 1 ]] || die "$name: expected exactly one sources/*.cpp, found: $(echo $srcs)"
  stem=$(basename "$srcs" .cpp)
  if [[ "$stem" == *_test ]]; then hal_stem=${stem%_test}_hal_test; else hal_stem=${stem}_hal; fi
  git -C "$REPO" cat-file -e "$BASE_SHA:$SRC_DIR/$stem.cpp" 2>/dev/null || die "$SRC_DIR/$stem.cpp does not exist at the BASE commit"
  for c in ${T_CPP[@]+"${T_CPP[@]}"}; do [[ "$c" == "$SRC_DIR/$stem.cpp" ]] && die "Two tests use the same source $stem.cpp"; done
  T_NAME+=("$name"); T_K+=("$k"); T_PY+=("$py"); T_HALPY+=("$PY_DIR/${name}_hal.py")
  T_CPP+=("$SRC_DIR/$stem.cpp"); T_HALCPP+=("$SRC_DIR/$hal_stem.cpp")
done
# Runs one python test from tests/ (cwd): run_test <path to .py> <k_expr or ""> <output log> [additional pytest args...]
run_test() {
  local args=("python_tests/$(basename "$1")")
  [[ -n "$2" ]] && args+=(-k "$2")
  local out=$3; shift 3
  timeout "$TEST_TIMEOUT" pytest "${args[@]}" "$@" -q > "$out" 2>&1
}
# Frozen number of tests: expected_of <test[:k]> -> N from EXPECT_PASSED (empty if not given)
expected_of() {
  local kv
  for kv in $EXPECT_PASSED; do [[ "${kv%=*}" == "$1" ]] && { echo "${kv##*=}"; return 0; }; done
  return 0
}
# check_expected <test index> <log of the original test>: stops if the "passed" count is not the frozen number
check_expected() {
  local entry=${T_NAME[$1]}${T_K[$1]:+:${T_K[$1]}} exp got
  exp=$(expected_of "$entry")
  [[ -z "$exp" ]] && return 0
  got=$(grep -oE '[0-9]+ passed' "$2" | tail -1 | grep -oE '[0-9]+' || echo 0)
  [[ "$got" == "$exp" ]] || die "The number of tests changed: $entry expected $exp passed (EXPECT_PASSED), got $got. See $2"
  echo "  $entry: $got passed, as frozen (EXPECT_PASSED)"
}
# The command as the agent sees it (for the prompt)
test_cmd() { if [[ -n "$2" ]]; then echo "pytest python_tests/$(basename "$1") -k '$2'"; else echo "pytest python_tests/$(basename "$1")"; fi; }

HAL_USERS=$(llk_hal_users "$BASE_SHA")
[[ -n "$HAL_USERS" ]] && { echo "$HAL_USERS"; die "LLK files at the BASE commit already use HAL. Choose an earlier BASE."; }

RUN_ID=$(date +%Y%m%d_%H%M%S)_${MODE}_${BASE_SHA:0:8}_${MODEL}
RUN_DIR=$RUNS_ROOT/$RUN_ID
WS=$RUN_DIR/ws
LOGS=$RUN_DIR/logs
mkdir -p "$WS" "$LOGS" "$RUN_DIR/tmp"
# Everything the test framework compiles (kernel ELFs) goes into this run, not into the shared /tmp,
# so that the code size metrics measure exactly this run.
export TMPDIR="$RUN_DIR/tmp"
log "Run: $RUN_DIR (mode $MODE)"

# ------------------------------------------------- 1. pre-HAL code, without .git ---
log "1/9 Export of BASE commit ${BASE_SHA:0:8}"
if [[ "$FULL_REPO" == 1 ]]; then
  git -C "$REPO" archive "$BASE_SHA" | tar -x -C "$WS"
else
  git -C "$REPO" archive "$BASE_SHA" "$LLK" | tar -x -C "$WS"
fi

# ------------------------------------------ 2. HAL code + documentation ---
log "2/9 HAL from ${HAL_SHA:0:8}, documentation"
rm -rf "$WS/$HAL_SRC" "$WS/$HAL_DOCS"
git -C "$REPO" archive "$HAL_SHA" "$HAL_SRC" | tar -x -C "$WS"
mkdir -p "$WS/$HAL_DOCS"
DOCS_SHA=""
if [[ -n "$DOCS_COMMIT" ]]; then
  git -C "$REPO" rev-parse --verify -q "$DOCS_COMMIT^{commit}" >/dev/null || die "DOCS_COMMIT '$DOCS_COMMIT' does not exist"
  DOCS_SHA=$(git -C "$REPO" rev-parse "$DOCS_COMMIT")
  if git -C "$REPO" cat-file -e "$DOCS_SHA:$HAL_DOCS" 2>/dev/null; then
    git -C "$REPO" archive "$DOCS_SHA" "$HAL_DOCS" | tar -x -C "$WS"
  else
    echo "WARNING: $HAL_DOCS does not exist at DOCS_COMMIT ${DOCS_SHA:0:8}"
  fi
fi
python3 - "$WS/$HAL_DOCS" "${DOCS_DIR:-}" "$DOCS_INVALID" "$DOCS_REMOVE" "${DOCS_SHA:0:8}" <<'PY'
import sys, os, shutil, re
docs, snap, invalid, remove, docs_sha = sys.argv[1:]
invalid, remove = invalid.split(), remove.split()
src = {f: f"git {docs_sha} (docs/hal/)" for f in os.listdir(docs)}
if snap:
    if not os.path.isdir(snap):
        sys.exit(f"DOCS_DIR '{snap}' does not exist")
    for f in sorted(os.listdir(snap)):
        p = os.path.join(snap, f)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(docs, f)); src[f] = f"snapshot {os.path.basename(os.path.normpath(snap))}/"
for f in remove:
    if os.path.exists(os.path.join(docs, f)):
        os.remove(os.path.join(docs, f)); src.pop(f, None)
BANNER = ("WARNING (added by the experiment harness): the maintainer has stated that this page is NOT "
          "valid for the current HAL implementation. Treat everything on it as possibly outdated; prefer "
          "the code's behaviour as observed in tests, and report any conflict you find.")
for f in invalid:
    p = os.path.join(docs, f)
    if not os.path.exists(p):
        continue
    t = open(p, errors="replace").read()
    if f.endswith((".html", ".htm")):
        b = f'<div style="border:3px solid red;padding:8px;margin:8px"><b>{BANNER}</b></div>'
        t = re.sub(r"(<body[^>]*>)", r"\1" + b, t, count=1) if re.search(r"<body", t) else b + t
    else:
        t = f"> **{BANNER}**\n\n" + t
    open(p, "w").write(t)
lines = ["# HAL documentation available in this experiment", "",
         "Generated by the harness. Every document in this folder, where it comes from and its status.",
         "If two documents disagree, say so in your report and name both sources.", "",
         "| File | Source | Status |", "|---|---|---|"]
for f in sorted(src):
    st = "NOT VALID (outdated, see warning on the page)" if f in invalid else "valid"
    lines.append(f"| {f} | {src[f]} | {st} |")
if remove:
    lines += ["", "Removed because they are known to be outdated: " + ", ".join(remove)]
open(os.path.join(docs, "README_DOCS.md"), "w").write("\n".join(lines) + "\n")
print(f"[harness] documentation: {len(src)} files, marked invalid: {len([f for f in invalid if f in src])}, removed: {len(remove)}")
PY
if is_raw_mode; then
  # RAW arm: without HAL (neither code nor documentation); raw reference material in raw_ref/ instead
  rm -rf "$WS/$HAL_SRC" "$WS/$HAL_DOCS"
  REF="$WS/$LLK/raw_ref"; mkdir -p "$REF"
  {
    echo "# Raw reference material available in this experiment"; echo
    echo "Generated by the harness. Copies of these files; include the originals by name in your code."; echo
    echo "| File here | Copied from | What it is |"; echo "|---|---|---|"
  } > "$REF/README_REF.md"
  for f in $RAW_REFMAT; do
    [[ -f "$WS/$f" ]] || die "RAW_REFMAT: $f does not exist at the BASE commit"
    cp "$WS/$f" "$REF/$(basename "$f")"
    echo "| $(basename "$f") | $f | header |" >> "$REF/README_REF.md"
  done
  if [[ "$MODE" == RAW ]]; then
    cp -a "$RAW_ISA_DIR" "$REF/isa"
    echo "| isa/ | $(basename "$RAW_ISA_DIR") (ISA documentation) | Tensix ISA documentation |" >> "$REF/README_REF.md"
  fi
  echo "[harness] raw_ref: $(ls "$REF" | wc -l) items"
fi

# --------------------------------------------- 3. spec for mode B ---
if is_gen_mode; then
  log "3/9 Spec (signatures only) for mode $MODE"
  python3 "$HARNESS_DIR/make_spec.py" "$WS/$BH/llk_lib" "$WS/$SPEC_DIR/llk_lib"
  python3 "$HARNESS_DIR/make_spec.py" "$WS/$BH/common/inc" "$WS/$SPEC_DIR/common/inc" \
    --exclude ckernel_ops.h ckernel_template.h
else
  log "3/9 (mode $MODE: spec not needed)"
fi

# --------------------------------------- 4. python test for the HAL version ---
log "4/9 Python tests for the HAL versions ($NTESTS)"
for i in "${!T_NAME[@]}"; do
  o=$(basename "${T_CPP[$i]}"); h=$(basename "${T_HALCPP[$i]}")
  sed "s#sources/$o#sources/$h#g" "$WS/${T_PY[$i]}" > "$WS/${T_HALPY[$i]}"
  grep -q "sources/$h" "$WS/${T_HALPY[$i]}" || die "sed did not find 'sources/$o' in ${T_PY[$i]}"
  grep -q "sources/$o" "$WS/${T_HALPY[$i]}" && die "${T_HALPY[$i]} still mentions sources/$o"
  echo "  ${T_NAME[$i]}: $o -> $h"
done

# ----------------------------------------------- 5. permissions for this run ---
# '/path' in a --settings file is resolved from the folder of that file, so it must be in $WS.
# In addition, reading other runs and the repo is denied (the whole HAL / the originals exist there).
log "5/9 Permissions"
[[ "$MODE" == TRIVIAL ]] && echo "  (mode TRIVIAL: Claude is not started, permissions are not needed)"
[[ "$MODE" == TRIVIAL ]] || MODE="$MODE" CONFLUENCE="$CONFLUENCE" MCP_SERVER="$MCP_SERVER" HAL_FILTER="$HAL_FILTER" SIBLING_DENY="$SIBLING_DENY" VENV_DIR="${VIRTUAL_ENV:-}" python3 - "$HARNESS_DIR/hal-settings-$MODE.json" "$WS/hal-settings.json" "$RUNS_ROOT" "$RUN_DIR" "$REPO" "$RESULTS_ROOT" "${T_CPP[@]}" <<'PY'
import json, sys, os, glob
src, dst, runs_root, run_dir, repo, results_root = sys.argv[1:7]
orig_sources = sys.argv[7:]
cfg = json.load(open(src))
deny = cfg["permissions"]["deny"]
for f in orig_sources:  # the original test sources are not modified
    rule = f"Edit(/{f})"
    if rule not in deny:
        deny.append(rule)
def abs_rule(p):  # absolute path in settings syntax: //absolute/path/**
    return f"Read(/{os.path.realpath(p)}/**)"
for d in glob.glob(os.path.join(runs_root, "*")):
    if os.path.isdir(d) and os.path.realpath(d) != os.path.realpath(run_dir):
        deny.append(abs_rule(d))
deny.append(abs_rule(repo))
deny.append(abs_rule(results_root))
if os.environ.get("MODE") == "B" and os.environ.get("HAL_FILTER") == "1":  # build folder of the tests (may contain compiler output)
    deny.append(abs_rule(os.path.join(os.environ["TMPDIR"], "tt-llk-build")))
if os.environ.get("SIBLING_DENY") == "1":  # the user's other folders: worktrees with the originals, dry runs, tools, harness
    venv = os.path.realpath(os.environ["VENV_DIR"]) + "/" if os.environ.get("VENV_DIR") else ""
    for parent in sorted({os.path.dirname(os.path.realpath(runs_root)), os.path.join("/localdev", os.environ.get("USER", ""))}):
        for d in sorted(glob.glob(os.path.join(parent, "*"))):
            rd = os.path.realpath(d)
            if not os.path.isdir(rd) or rd == os.path.realpath(runs_root) or (venv and venv.startswith(rd + "/")):
                continue
            if abs_rule(d) not in deny:
                deny.append(abs_rule(d))
if os.environ.get("CONFLUENCE") == "1":
    cfg["permissions"].setdefault("allow", []).append("mcp__" + os.environ.get("MCP_SERVER", "atlassian"))
    cfg.pop("disableClaudeAiConnectors", None)   # Confluence can go through the claude.ai connector
json.dump(cfg, open(dst, "w"), indent=2)
print(f"[harness] {len(deny)} deny rules")
PY
# The same HAL/git denial as in the project settings of the folder Claude is started from ($WS).
if [[ "$MODE" != TRIVIAL ]]; then
  mkdir -p "$WS/.claude"
  cp "$HARNESS_DIR/ws-claude-settings.json" "$WS/.claude/settings.json"
fi
# Mode B: the HAL implementation is not shown to the agent even through compiler messages. The test framework does all
# compilation through run_shell_command (helpers/utils.py); its exception (compiler stderr)
# goes through hal_output_filter.py before pytest prints it. Before the baseline commit, so that it does not
# look like a change by the agent. HAL_FILTER=0: no filter (as in run 5).
if [[ "$MODE" == B && "$HAL_FILTER" == 1 ]]; then
  HELP="$WS/$LLK/tests/python_tests/helpers"
  grep -q "^def run_shell_command" "$HELP/utils.py" 2>/dev/null || die "no run_shell_command in $HELP/utils.py"
  cp "$HARNESS_DIR/hal_output_filter.py" "$HELP/_hal_output_filter.py"
  NWORDS=$(python3 "$HARNESS_DIR/hal_output_filter.py" --build-words "$HELP/_hal_visible_words.txt" \
    "$WS/$HAL_DOCS" "$WS/$LLK/llk_spec" "$WS/$LLK/tests/python_tests" "$WS/$LLK/tests/helpers" "$WS/$LLK/tests/sources")
  cat >> "$HELP/utils.py" <<'EOF'


# --- HAL experiment harness: hide HAL implementation details in compiler output (mode B) ---
from ._hal_output_filter import hide_hal as _hal_hide

_hal_orig_run_shell_command = run_shell_command


def run_shell_command(*args, **kwargs):
    try:
        return _hal_orig_run_shell_command(*args, **kwargs)
    except RuntimeError as exc:
        message = _hal_hide(str(exc))
    raise RuntimeError(message)
EOF
  echo "  compiler output filter enabled ($NWORDS visible identifiers from the documentation, spec and tests)"
fi

# ---------------------------------------------------- 6. baseline git for diff ---
git -C "$WS" init -q
git -C "$WS" add -A
git -C "$WS" -c user.name=harness -c user.email=harness@local commit -qm baseline

cat > "$LOGS/run_meta.txt" <<EOF
mode=$MODE
base_commit=$BASE_SHA
hal_commit=$HAL_SHA
docs_commit=${DOCS_SHA:-}
docs_dir=${DOCS_DIR:-}
docs_invalid=$DOCS_INVALID
docs_remove=$DOCS_REMOVE
confluence=$CONFLUENCE
confluence_page=$CONFLUENCE_PAGE
main_ref=$MAIN_REF
model=$MODEL
timeout=$TIMEOUT
max_turns=$MAX_TURNS
max_budget_usd=${MAX_BUDGET:-}
op_filter=$OP_FILTER
tests=$TESTS
expect_passed=$EXPECT_PASSED
hal_filter=$HAL_FILTER
prompt_file=$PROMPT_FILE
sibling_deny=$SIBLING_DENY
dry_run=$DRY_RUN
raw_refmat=$(is_raw_mode && echo "$RAW_REFMAT")
raw_isa_dir=$([[ "$MODE" == RAW ]] && echo "$RAW_ISA_DIR")
claude_version=$(claude --version 2>/dev/null || echo unknown)
host=$(hostname)
date=$(date -Iseconds)
EOF

# ------------------------------------- 7. the original test must pass ---
cd "$WS/$LLK/tests"
log "6/9 setup_testing_env.sh"
./setup_testing_env.sh > "$LOGS/setup_env.txt" 2>&1 || die "setup_testing_env.sh failed, see $LOGS/setup_env.txt"
pip install -q -r requirements.txt > "$LOGS/pip.txt" 2>&1 || echo "WARNING: pip install did not succeed, see $LOGS/pip.txt"
if [[ "$MODE" == B && "$HAL_FILTER" == 1 ]]; then   # the filter must work: a synthetic error from a "HAL header" must be hidden
  (cd python_tests && python3 - <<'PY'
from helpers.utils import run_shell_command as r
try:
    r("echo '../x/llk_lib/hal/a.h:1:2: error: probe-secret' 1>&2; false")
except RuntimeError as e:
    err = str(e).split("Command's stderr:", 1)[-1]
    assert "probe-secret" not in err and "[HAL implementation detail hidden]" in err, str(e)
else:
    raise SystemExit("run_shell_command did not raise an error")
PY
  ) > "$LOGS/hal_filter_check.txt" 2>&1 || die "compiler output filter does not work, see $LOGS/hal_filter_check.txt"
  echo "  compiler output filter: check passed"
fi
echo "sfpi_compiler=$("$WS/$LLK/tests/sfpi/compiler/bin/riscv-tt-elf-g++" --version 2>/dev/null | head -1 || echo unknown)" >> "$LOGS/run_meta.txt"
echo "pricing=price from Claude Code (total_cost_usd), price list as of $(date +%F)" >> "$LOGS/run_meta.txt"

# ------------------------------------ mode TRIVIAL: test validity check ---
# For each test: original (after a reset) -> the same test with empty LLK bodies right away, without a reset
# -> again after a reset. Expected: 0 passes in both trivial runs.
if [[ "$MODE" == TRIVIAL ]]; then
  LIB="$WS/$BH/llk_lib"; ORIG_LIB="$RUN_DIR/llk_lib_orig"; TRIV_LIB="$RUN_DIR/llk_lib_trivial"
  cp -a "$LIB" "$ORIG_LIB"; cp -a "$LIB" "$TRIV_LIB"
  python3 "$HARNESS_DIR/make_spec.py" "$ORIG_LIB" "$TRIV_LIB" --trivial
  [[ -d "$ORIG_LIB/experimental" ]] && python3 "$HARNESS_DIR/make_spec.py" "$ORIG_LIB/experimental" "$TRIV_LIB/experimental" --trivial
  use_lib() { rm -rf "$LIB"; cp -a "$1" "$LIB"; }
  passed_of() { grep -oE '[0-9]+ passed' "$1" | tail -1 | grep -oE '[0-9]+' || echo 0; }
  ran_of() {   # sum of passed+failed+errors from the final pytest line ("=== ... in N.NNs ===")
    local line t=0 x
    line=$(grep -E '=+ .* in [0-9.]+s' "$1" | tail -1)
    for x in $(grep -oE '[0-9]+ (passed|failed|errors?)' <<< "$line" | grep -oE '^[0-9]+'); do t=$((t + x)); done
    echo "$t"
  }
  interrupted() { grep -q 'KeyboardInterrupt\|Timeout' "$1"; }
  TRIV_BAD=0; TRIV_PARTIAL=0
  for i in "${!T_NAME[@]}"; do
    n=${T_NAME[$i]}
    log "TRIVIAL $n: original (after a reset)"
    use_lib "$ORIG_LIB"; tt-smi -r > /dev/null 2>&1 || true
    run_test "${T_PY[$i]}" "${T_K[$i]}" "$LOGS/baseline_$n.txt" || die "Original test $n does not pass, see $LOGS/baseline_$n.txt"
    echo "  original: $(tail -1 "$LOGS/baseline_$n.txt")"
    check_expected "$i" "$LOGS/baseline_$n.txt"
    use_lib "$TRIV_LIB"
    log "TRIVIAL $n: empty bodies, right after the original (without a reset)"
    TEST_TIMEOUT=$TRIV_TIMEOUT run_test "${T_PY[$i]}" "${T_K[$i]}" "$LOGS/trivial_noreset_$n.txt" --junitxml="$LOGS/trivial_noreset_$n.xml" || true
    echo "  without reset: $(tail -1 "$LOGS/trivial_noreset_$n.txt")"
    log "TRIVIAL $n: empty bodies, after tt-smi -r"
    tt-smi -r > /dev/null 2>&1 || true
    TEST_TIMEOUT=$TRIV_TIMEOUT run_test "${T_PY[$i]}" "${T_K[$i]}" "$LOGS/trivial_reset_$n.txt" --junitxml="$LOGS/trivial_reset_$n.xml" || true
    echo "  after reset: $(tail -1 "$LOGS/trivial_reset_$n.txt")"
    for v in noreset reset; do
      python3 - "$LOGS/trivial_${v}_$n.xml" > "$LOGS/trivial_${v}_${n}_passed.txt" <<'PY' || true
import sys, xml.etree.ElementTree as ET
try:
    root = ET.parse(sys.argv[1]).getroot()
except Exception as e:
    sys.exit(f"no junit xml: {e}")
for tc in root.iter("testcase"):
    name = tc.get("name")
    if not name or name == "None":      # interruption trace (e.g. timeout), not a real variant
        continue
    if not any(ch.tag in ("failure", "error", "skipped") for ch in tc):
        print(name)
PY
      np=$(wc -l < "$LOGS/trivial_${v}_${n}_passed.txt" 2>/dev/null || echo 0); (( np == 0 )) || TRIV_BAD=1
      interrupted "$LOGS/trivial_${v}_$n.txt" && TRIV_PARTIAL=1
    done
  done
  use_lib "$ORIG_LIB"
  {
    echo; echo "==================== TRIVIAL AGENT ===================="
    echo "Run:     $RUN_DIR"
    echo "BASE:    ${BASE_SHA:0:8}"
    echo "Expected: 0 passes with empty LLK bodies (otherwise the test does not check anything on those variants)"
    for i in "${!T_NAME[@]}"; do
      n=${T_NAME[$i]}
      printf '  %-30s original: %s\n' "$n" "$(tail -1 "$LOGS/baseline_$n.txt")"
      for v in noreset reset; do
        lbl=$([[ $v == noreset ]] && echo "without reset" || echo "after reset  ")
        printf '  %-30s %s: %s passes out of %s run (original: %s)%s\n' "" "$lbl" \
          "$(wc -l < "$LOGS/trivial_${v}_${n}_passed.txt" 2>/dev/null || echo ?)" "$(ran_of "$LOGS/trivial_${v}_$n.txt")" \
          "$(passed_of "$LOGS/baseline_$n.txt")" "$(interrupted "$LOGS/trivial_${v}_$n.txt" && echo ' — INTERRUPTED, not all variants were run')"
      done
    done
    if [[ $TRIV_BAD != 0 ]]; then echo "Result: PROBLEM: some variants pass with empty bodies, see logs/trivial_*_passed.txt"
    elif [[ $TRIV_PARTIAL != 0 ]]; then echo "Result: VALID for the variants that were run, but the run was INTERRUPTED (TRIV_TIMEOUT=$TRIV_TIMEOUT) — not all variants were checked"
    else echo "Result: VALID (no variant passes with empty bodies, all were run)"; fi
    echo "=========================================================="
  } | tee "$LOGS/SUMMARY.txt"
  mkdir -p "$RESULTS_ROOT/$RUN_ID"
  tar -czf "$RESULTS_ROOT/$RUN_ID.tar.gz" -C "$RUN_DIR" logs || echo "WARNING: archiving failed"
  cp "$LOGS"/SUMMARY.txt "$LOGS"/run_meta.txt "$LOGS"/trivial_*_passed.txt "$RESULTS_ROOT/$RUN_ID/" 2>/dev/null || true
  exit 0
fi

if [[ "$SKIP_BASELINE" != 1 ]]; then
  log "7/9 Original tests ($NTESTS)"
  for i in "${!T_NAME[@]}"; do
    out="$LOGS/baseline_${T_NAME[$i]}.txt"
    tt-smi -r > /dev/null 2>&1 || true
    run_test "${T_PY[$i]}" "${T_K[$i]}" "$out" \
      || die "Original test ${T_NAME[$i]} does not pass. See $out (hung card? tt-smi -r)"
    echo "  ${T_NAME[$i]}: $(tail -1 "$out")"
    check_expected "$i" "$out"
  done
elif [[ -n "$EXPECT_PASSED" ]]; then
  echo "WARNING: SKIP_BASELINE=1, the frozen number of tests (EXPECT_PASSED) was not checked"
fi

# ------------------------------------------------------- 8. generation ---
cd "$WS"
TABLE="| Original test source | Python test | Test source you write | Python test for your source | Command (from tt_metal/tt-llk/tests) |"$'\n'"|---|---|---|---|---|"
for i in "${!T_NAME[@]}"; do
  TABLE+=$'\n'"| ${T_CPP[$i]#$LLK/} | ${T_PY[$i]#$LLK/} | ${T_HALCPP[$i]#$LLK/} | ${T_HALPY[$i]#$LLK/} | \`$(test_cmd "${T_HALPY[$i]}" "${T_K[$i]}")\` |"
done
if [[ -n "$PROMPT_FILE" ]]; then
  [[ -f "$PROMPT_FILE" ]] || die "PROMPT_FILE '$PROMPT_FILE' does not exist"
  PROMPT=$(cat "$PROMPT_FILE")
elif is_gen_mode; then
  # The same prompt for all arms: prompt-base.md + a small file with the API-dependent part
  PROMPT=$(TABLE="$TABLE" python3 - "$HARNESS_DIR/prompt-base.md" "$HARNESS_DIR/$API_FILE" <<'PY'
import os, re, sys
base, api = sys.argv[1], sys.argv[2]
d, key = {}, None
for ln in open(api).read().split("\n"):
    m = re.fullmatch(r"@@(\w+)", ln)
    if m:
        key = m.group(1); d[key] = []; continue
    if key:
        d[key].append(ln)
t = open(base).read().replace("{{TEST_TABLE}}", os.environ["TABLE"])
for k, v in d.items():
    t = t.replace("{{" + k + "}}", "\n".join(v).rstrip("\n") if k == "API_CANNOT_SEE" else "\n".join(v).strip("\n"))
left = sorted(set(re.findall(r"\{\{\w+\}\}", t)))
if left:
    sys.exit("unfilled placeholders in the prompt: " + ", ".join(left))
print(t, end="")
PY
) || die "assembling the prompt failed"
else
  PROMPT=$(cat "$HARNESS_DIR/prompt-$MODE.md")
fi
if [[ "$CONFLUENCE" == 1 ]]; then
  claude mcp list 2>/dev/null | grep -q "^$MCP_SERVER" \
    || echo "WARNING: MCP server '$MCP_SERVER' is not in 'claude mcp list'; Claude will not have Confluence"
  PROMPT+=$'\n\n'"$(sed "s#{{PAGE}}#$CONFLUENCE_PAGE#g" "$HARNESS_DIR/prompt-confluence.md")"
fi
printf '%s\n' "$PROMPT" > "$LOGS/prompt_used.md"
if [[ "$DRY_RUN" == 1 ]]; then
  log "DRY_RUN=1: preparation and original tests done, Claude is not started"
  echo "Workspace: $WS"; echo "Prompt:    $LOGS/prompt_used.md"
  for i in "${!T_NAME[@]}"; do echo "  ${T_NAME[$i]}: $(tail -1 "$LOGS/baseline_${T_NAME[$i]}.txt" 2>/dev/null)"; done
  exit 0
fi
log "8/9 Claude generates (mode $MODE, model $MODEL, timeout $TIMEOUT, tests $NTESTS)"
START=$(date +%s)
set +e
timeout "$TIMEOUT" claude -p "$PROMPT" \
  --model "$MODEL" \
  --settings "$WS/hal-settings.json" \
  --permission-mode acceptEdits \
  --max-turns "$MAX_TURNS" \
  ${MAX_BUDGET:+--max-budget-usd "$MAX_BUDGET"} \
  --output-format stream-json --verbose \
  > "$LOGS/transcript.jsonl" 2> "$LOGS/claude_stderr.txt"
CLAUDE_RC=$?
set -e
END=$(date +%s)
{ echo "claude_exit_code=$CLAUDE_RC"; echo "claude_seconds=$((END-START))"; } >> "$LOGS/run_meta.txt"
[[ $CLAUDE_RC == 124 ]] && echo "WARNING: Claude interrupted because of the timeout ($TIMEOUT)."

# Claude's final report
python3 - "$LOGS/transcript.jsonl" > "$LOGS/final_report.md" <<'PY' || true
import json, sys
for l in open(sys.argv[1]):
    try: e = json.loads(l)
    except Exception: continue
    if e.get("type") == "result": print(e.get("result", ""))
PY

# ----------------------------------------- 9. independent check and report ---
log "9/9 Independent HAL tests (each after tt-smi -r) and checks"
cd "$WS/$LLK/tests"
TEST_RC=0
T_RC=()
: > "$LOGS/reset_before_final.txt"
for i in "${!T_NAME[@]}"; do
  tt-smi -r >> "$LOGS/reset_before_final.txt" 2>&1 || true
  set +e
  run_test "${T_HALPY[$i]}" "${T_K[$i]}" "$LOGS/hal_test_${T_NAME[$i]}.txt"
  rc=$?
  set -e
  T_RC+=("$rc"); (( rc == 0 )) || TEST_RC=1
  echo "hal_test_exit_code_${T_NAME[$i]}=$rc" >> "$LOGS/run_meta.txt"
  echo "  ${T_NAME[$i]}: exit $rc, $(tail -1 "$LOGS/hal_test_${T_NAME[$i]}.txt")"
  if [[ "$DIRTY_CHECK" == 1 ]]; then
    # Dirty state: the original test right after the agent's code, without a reset; it must pass
    set +e
    run_test "${T_PY[$i]}" "${T_K[$i]}" "$LOGS/dirty_${T_NAME[$i]}.txt"
    drc=$?
    set -e
    echo "dirty_exit_code_${T_NAME[$i]}=$drc" >> "$LOGS/run_meta.txt"
    echo "    original right after (without reset): exit $drc, $(tail -1 "$LOGS/dirty_${T_NAME[$i]}.txt")"
  fi
done
echo "hal_test_exit_code=$TEST_RC" >> "$LOGS/run_meta.txt"
# combined log (compatibility with old tools)
for i in "${!T_NAME[@]}"; do echo "== ${T_NAME[$i]}"; cat "$LOGS/hal_test_${T_NAME[$i]}.txt"; done > "$LOGS/hal_test.txt"

cd "$WS"
if is_raw_mode; then
{
  echo "# RAW arm: use of HAL (forbidden, ideally empty)"
  grep -rnE '#include *"[^"]*hal/|\bhal::' "$OUT_DIR" "${T_HALCPP[@]}" 2>/dev/null || echo "(none)"
  echo; echo "# Number of lines with raw instructions (informational)"
  grep -rcE 'TTI_|TT_OP_' "$OUT_DIR" 2>/dev/null || true
} > "$LOGS/raw_check.txt"
else
{
  echo "# Raw TT instructions in the generated code (ideally empty, except NOT_IN_HAL parts)"
  grep -rnE 'TTI_|TT_OP_|cfg_reg_rmw' "$OUT_DIR" "${T_HALCPP[@]}" 2>/dev/null || echo "(none)"
  echo; echo "# NOT_IN_HAL marks"
  grep -rn 'NOT_IN_HAL' "$OUT_DIR" 2>/dev/null || echo "(none)"
} > "$LOGS/raw_check.txt"
fi

# Attempts to access forbidden paths (from the transcript)
python3 - "$LOGS/transcript.jsonl" "$MODE" "$RUN_DIR" "$RUNS_ROOT" "$REPO" "$RESULTS_ROOT" > "$LOGS/forbidden_access.txt" <<'PY' || true
import json, sys, re, os
path, mode, run_dir, runs_root, repo, results_root = sys.argv[1:]
run_dir, runs_root, repo, results_root = (os.path.realpath(x) for x in (run_dir, runs_root, repo, results_root))
rel = [r"llk_lib/hal\b"]
if mode in ("B", "RAW", "RAWNODOCS"):
    rel.append(r"(?<!llk_spec/)tt_llk_(blackhole|wormhole_b0|quasar)/")
    rel.append(r"tt_metal/hw/(ckernels|inc)/")
if mode in ("RAW", "RAWNODOCS"):
    rel.append(r"docs/hal\b")
rel_pat = re.compile("|".join(rel))
abs_path = re.compile(r"(/[\w.\-/~]+)")
def bad(txt):
    if rel_pat.search(txt):
        return True
    for p in abs_path.findall(txt):
        if "hal_run_" in p or "hal-restore" in p or p.startswith(repo + "/") or p == repo or p.startswith(results_root):
            return True
        if p.startswith(runs_root + "/") and not p.startswith(run_dir):
            return True
    return False
hits = 0
for l in open(path):
    try: e = json.loads(l)
    except Exception: continue
    if e.get("type") != "assistant": continue
    for c in e["message"].get("content", []):
        if c.get("type") != "tool_use" or c.get("name") not in ("Read", "Grep", "Glob", "Bash"):
            continue
        txt = json.dumps(c.get("input", {}))
        if bad(txt):
            hits += 1
            print(f"{c['name']}: {txt[:300]}")
print(f"\n# total attempts: {hits}")
PY

python3 - "$LOGS/transcript.jsonl" > "$LOGS/mcp_calls.txt" <<'PY' || true
import json, sys
n = 0
for l in open(sys.argv[1]):
    try: e = json.loads(l)
    except Exception: continue
    if e.get("type") != "assistant": continue
    for c in e["message"].get("content", []):
        if c.get("type") == "tool_use" and c.get("name", "").startswith("mcp__"):
            n += 1
            print(f"{c['name']}: {json.dumps(c.get('input', {}), ensure_ascii=False)[:400]}")
print(f"\n# total MCP calls: {n}")
PY

# Mode B: traces of probing the HAL implementation outside the documentation (only numbers in probe_check.txt;
# probe_check_details.txt may contain HAL names that the documentation does not give).
if [[ "$MODE" == B ]]; then
  python3 "$HARNESS_DIR/probe_check.py" "$LOGS/transcript.jsonl" "$WS/$HAL_DOCS" "$LOGS/probe_check_details.txt" \
    "$OUT_DIR" "${T_HALCPP[@]}" > "$LOGS/probe_check.txt" 2>&1 || echo "WARNING: probe_check.py failed"
fi

python3 "$HARNESS_DIR/codestats.py" "$OUT_DIR" "${T_HALCPP[@]}" --json "$LOGS/codestats.json" --md "$LOGS/codestats.md" > /dev/null 2>&1 \
  || echo "WARNING: codestats.py failed"

git -C "$WS" add -A
git -C "$WS" diff --cached --stat HEAD > "$LOGS/diff_stat.txt"
git -C "$WS" diff --cached HEAD > "$LOGS/diff.patch"

# The test framework deletes the build folder at the start of every pytest session, so the originals and HAL versions
# are compiled together in one session without the card (--compile-producer), into a separate TMPDIR.
log "Code size: compiling the originals and HAL versions in one session (without the card)"
CS_TMP="$RUN_DIR/codesize_tmp"; mkdir -p "$CS_TMP"
CS_ARGS=()
for i in "${!T_NAME[@]}"; do CS_ARGS+=("python_tests/$(basename "${T_PY[$i]}")" "python_tests/$(basename "${T_HALPY[$i]}")"); done
# A common -k filter only if all tests have the same one; otherwise all variants are compiled.
CS_K=(-k "${T_K[0]}")
for k in "${T_K[@]}"; do [[ "$k" == "${T_K[0]}" ]] || CS_K=(); done
[[ -z "${T_K[0]}" ]] && CS_K=()
(cd "$WS/$LLK/tests" && TMPDIR="$CS_TMP" timeout 2h pytest "${CS_ARGS[@]}" ${CS_K[@]+"${CS_K[@]}"} --compile-producer -n 32 -q \
  > "$LOGS/codesize_compile.txt" 2>&1) || echo "WARNING: compiling for code size failed (see logs/codesize_compile.txt)"
CS_ART="$CS_TMP/tt-llk-build"
TOOLS="$WS/$LLK/tests/sfpi/compiler/bin"
for i in "${!T_NAME[@]}"; do
  n=${T_NAME[$i]}; o=$(basename "${T_CPP[$i]}"); h=$(basename "${T_HALCPP[$i]}")
  python3 "$HARNESS_DIR/codesize.py" --artefacts "$CS_ART" --tools "$TOOLS" --orig "$o" --hal "$h" \
    --sample "$CODESIZE_SAMPLE" --json "$LOGS/codesize_$n.json" --md "$LOGS/codesize_$n.md" --csv "$LOGS/codesize_$n.csv" > /dev/null 2>&1 \
    || echo "WARNING: codesize.py failed for $n"
  if [[ -n "$ELFRAY" && -x "$ELFRAY" ]]; then
    python3 "$HARNESS_DIR/elfray_tt_tti.py" --elfray "$ELFRAY" --objdump "$TOOLS/riscv-tt-elf-objdump" --artefacts "$CS_ART" \
      --orig "$o" --hal "$h" --label "$RUN_ID / $n" --md "$LOGS/elfray_tt_tti_$n.md" --json "$LOGS/elfray_tt_tti_$n.json" > /dev/null 2>&1 \
      || echo "WARNING: elfray_tt_tti.py failed for $n"
  fi
done
python3 "$HARNESS_DIR/metrics.py" "$RUN_DIR" --json "$LOGS/metrics.json" --md "$LOGS/metrics.md" > /dev/null 2>&1 \
  || echo "WARNING: metrics.py failed"
ARCHIVE="$RESULTS_ROOT/$RUN_ID.tar.gz"

RAW_HITS=$(grep -cE 'TTI_|TT_OP_|cfg_reg_rmw' "$LOGS/raw_check.txt" || true)
FORB=$(tail -1 "$LOGS/forbidden_access.txt" 2>/dev/null || echo "?")
cat <<EOF | tee "$LOGS/SUMMARY.txt"

==================== SUMMARY ====================
Run:              $RUN_DIR
Mode:             $MODE
BASE / HAL:       ${BASE_SHA:0:8} / ${HAL_SHA:0:8}
Model:            $MODEL
Claude duration:  $(( (END-START)/60 )) min (exit $CLAUDE_RC)
Tests:            $NTESTS ($TESTS)
HAL tests:        $([[ $TEST_RC == 0 ]] && echo ALL PASSED || echo NOT ALL PASSED) (each after tt-smi -r)
$(for i in "${!T_NAME[@]}"; do printf '  %-32s %s | original: %s\n' "${T_NAME[$i]}" "$(tail -1 "$LOGS/hal_test_${T_NAME[$i]}.txt" 2>/dev/null)" "$(tail -1 "$LOGS/baseline_${T_NAME[$i]}.txt" 2>/dev/null)"; [[ -f "$LOGS/dirty_${T_NAME[$i]}.txt" ]] && printf '  %-32s dirty state (original right after, without reset): %s\n' "" "$(tail -1 "$LOGS/dirty_${T_NAME[$i]}.txt")"; done)
Code (LOC / hex / shifts / raw / hal::): $(python3 -c "
import json;t=json.load(open('$LOGS/codestats.json'))['total']
print(t['loc'], '/', t['hex_literals'], '/', t['numeric_shifts'], '/', t['raw_instructions'], '/', t['hal_uses'])" 2>/dev/null || echo "?")
First full pass:   $(python3 -c "
import json;p=json.load(open('$LOGS/metrics.json'))['process']
print(', '.join(t + ' iter ' + str(v['iteration']) + ' (' + str(v['elapsed_min']) + ' min)' for t, v in sorted(p['first_full_pass'].items())) or 'none')" 2>/dev/null || echo "?")
Raw instr.:       $RAW_HITS lines (logs/raw_check.txt)
Forbidden access: $FORB (logs/forbidden_access.txt)
$([[ "$MODE" == B ]] && echo "HAL probing:      $(tail -1 "$LOGS/probe_check.txt" 2>/dev/null | sed 's/^# //') (logs/probe_check.txt)")
Tokens/cost:      $(python3 -c "import json;m=json.load(open('$LOGS/metrics.json'));t=m['tokens'];print(f\"in {m['tokens_total_input']:,} (cache {m['cache_hit_ratio']}), out {t['output_tokens']:,}, \${m['cost_usd']}\")" 2>/dev/null || echo "?")
Pytest runs:      $(python3 -c "import json;m=json.load(open('$LOGS/metrics.json'));print(m['pytest_runs_total'], m['pytest_runs_by_kind'])" 2>/dev/null || echo "?")
Confluence (MCP): $(tail -1 "$LOGS/mcp_calls.txt" 2>/dev/null | sed 's/# //') (logs/mcp_calls.txt)
Code size (text, original -> HAL, mean per variant):
$(for i in "${!T_NAME[@]}"; do printf '  %-32s %s\n' "${T_NAME[$i]}" "$(python3 -c "
import json;m=json.load(open('$LOGS/codesize_${T_NAME[$i]}.json'))['per_component']
print(', '.join(c + ' ' + str(m[c]['orig']['text']['mean']) + '->' + str(m[c]['hal']['text']['mean']) + 'B' for c in ('unpack','math','pack')))" 2>/dev/null || echo "?")"; done)
Metrics:          $LOGS/metrics.md, $LOGS/codesize_<test>.md, $LOGS/elfray_tt_tti_<test>.md
Report:           $LOGS/final_report.md
Code:             $WS/$OUT_DIR
Archive (permanent): $ARCHIVE
================================================
EOF
# ------------------------------------------- permanent archive in home ---
mkdir -p "$RESULTS_ROOT/$RUN_ID"
ARCH_PATHS=("ws/$OUT_DIR")
for i in "${!T_NAME[@]}"; do ARCH_PATHS+=("ws/${T_HALCPP[$i]}" "ws/${T_HALPY[$i]}"); done
tar -czf "$ARCHIVE" -C "$RUN_DIR" logs \
  $(cd "$RUN_DIR" && ls -d "${ARCH_PATHS[@]}" 2>/dev/null) \
  || echo "WARNING: archiving failed"
# the most important files, unpacked, for a quick look
cp "$LOGS/final_report.md" "$LOGS/run_meta.txt" "$LOGS/raw_check.txt" "$LOGS/diff_stat.txt" "$LOGS/SUMMARY.txt" "$LOGS/metrics.md" "$LOGS/metrics.json" "$LOGS/mcp_calls.txt" "$LOGS/prompt_used.md" \
   "$LOGS"/codesize_*.md "$LOGS"/codesize_*.json "$LOGS"/elfray_tt_tti_* "$LOGS"/hal_test_*.txt "$LOGS"/baseline_*.txt \
   "$LOGS"/dirty_*.txt "$LOGS"/codestats.* "$LOGS"/probe_check.txt \
   "$RESULTS_ROOT/$RUN_ID/" 2>/dev/null || true

echo "Results saved: $ARCHIVE"
