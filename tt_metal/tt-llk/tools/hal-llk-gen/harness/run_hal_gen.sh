#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# HAL LLK generation harness: one generation run of a Claude Code agent, plus independent checks.
#
#   MODE=B        The agent writes the LLK functions used by the selected tests from scratch on top of
#                 HAL. It sees the tests, a signature-only spec of the LLK library and the HAL
#                 documentation; the original LLKs and the HAL source are read-denied.
#   MODE=TRIVIAL  Test validity check, no agent: the original LLKs with empty function bodies must
#                 fail every selected variant.
#
# Normally started by ../hal-llk-gen (which sets every parameter from a profile). Steps:
#   1 export BASE_COMMIT (tests + original LLKs) into <run>/ws        5 deny list for the agent
#   2 HAL source from HAL_COMMIT, HAL docs from DOCS_COMMIT            6 setup_testing_env.sh
#   3 spec (signatures only) of llk_lib/ and common/inc/               7 original tests (baseline)
#   4 python test copies that run the agent's sources                  8 agent (claude -p)
#   9 independent checks: HAL test after tt-smi -r, original right after it (dirty state), raw-instruction,
#     forbidden-access, MCP and probing checks, code statistics, code size, metrics -> logs/SUMMARY.txt
#
# Required: REPO RUNS_ROOT RESULTS_ROOT BASE_COMMIT HAL_COMMIT DOCS_COMMIT (or DOCS_DIR) PROMPT_FILE (mode B)
# Optional: see "parameters" below.
set -euo pipefail

# ---------------------------------------------------------------- parameters ---
MODE=${MODE:-B}
case "$MODE" in B|TRIVIAL) ;; *) echo "MODE must be B or TRIVIAL"; exit 1;; esac
REPO=${REPO:?REPO (tt-metal git checkout that holds the commits) is required}
RUNS_ROOT=${RUNS_ROOT:?RUNS_ROOT is required}
RESULTS_ROOT=${RESULTS_ROOT:?RESULTS_ROOT is required}
BASE_COMMIT=${BASE_COMMIT:?BASE_COMMIT (tests + original LLKs) is required}
HAL_COMMIT=${HAL_COMMIT:?HAL_COMMIT (HAL implementation) is required}
DOCS_COMMIT=${DOCS_COMMIT-}            # commit whose tt_metal/tt-llk/docs/hal/ is given to the agent
DOCS_DIR=${DOCS_DIR:-}                 # folder with extra/replacement doc pages; copied over DOCS_COMMIT's pages
DOCS_INVALID=${DOCS_INVALID-}          # doc pages that get a "not valid for the current HAL" banner
DOCS_REMOVE=${DOCS_REMOVE:-}           # doc pages removed from the agent's copy
PROMPT_FILE=${PROMPT_FILE:-}           # agent prompt (mode B)
# "test[:k] ..." python tests (tests/python_tests/<test>.py) with an optional -k expression
TESTS=${TESTS:-test_eltwise_binary:add}
# "test[:k]=N ...": the original test must give exactly N passed, otherwise the run stops after the baseline
EXPECT_PASSED=${EXPECT_PASSED-}
MODEL=${MODEL:-claude-opus-5-5}
RUN_TAG=${RUN_TAG:-}                   # appended to the run id (the run name of ../hal-llk-gen)
SIBLING_DENY=${SIBLING_DENY:-1}        # 1 = also deny the user's other folders (see step 5)
SIBLING_ROOTS=${SIBLING_ROOTS:-/proj_sw/user_dev/$USER /localdev/$USER}
EXTRA_DENY=${EXTRA_DENY:-}             # more absolute paths the agent must not read
SKIP_BASELINE=${SKIP_BASELINE:-0}
DRY_RUN=${DRY_RUN:-0}                  # 1 = setup + original tests only, no agent
DIRTY_CHECK=${DIRTY_CHECK:-1}          # 1 = original test right after each final HAL test, without reset
TEST_TIMEOUT=${TEST_TIMEOUT:-60m}      # one pytest run of one test (baseline / final check)
TRIV_TIMEOUT=${TRIV_TIMEOUT:-4h}       # mode TRIVIAL: empty bodies take ~2 s per variant
MAX_BUDGET=${MAX_BUDGET:-}             # USD limit for the agent (--max-budget-usd); empty = none
ELFRAY=${ELFRAY:-}                     # elfray binary (optional; TT/TTI estimate of the harness build)
CODESIZE_SAMPLE=${CODESIZE_SAMPLE:-200} # variants measured for code size (0 = all)
MAIN_REF=${MAIN_REF:-origin/main}

read -r -a TEST_ENTRIES <<< "$TESTS"
NTESTS=${#TEST_ENTRIES[@]}
(( NTESTS > 0 )) || { echo "TESTS is empty"; exit 1; }
TIMEOUT=${TIMEOUT:-$(( NTESTS == 1 ? 4 : 2 + 2 * NTESTS ))h}
MAX_TURNS=${MAX_TURNS:-$(( NTESTS == 1 ? 500 : 300 * NTESTS ))}

HARNESS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
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

for f in make_spec.py metrics.py codesize.py codestats.py elfray_tt_tti.py probe_check.py settings/hal-settings-B.json; do
  [[ -f "$HARNESS_DIR/$f" ]] || die "Missing $HARNESS_DIR/$f"
done
if [[ "$MODE" == B && "$DRY_RUN" != 1 ]]; then command -v claude >/dev/null || die "claude is not in PATH"; fi
if [[ "$MODE" == B ]]; then [[ -f "$PROMPT_FILE" ]] || die "PROMPT_FILE '$PROMPT_FILE' does not exist"; fi
command -v pytest >/dev/null || die "pytest is not in PATH (activate the venv)"
[[ -n "$DOCS_COMMIT" || -n "$DOCS_DIR" || "$MODE" == TRIVIAL ]] || die "DOCS_COMMIT or DOCS_DIR is required"

# ------------------------------------------------------------ repo checks ---
git -C "$REPO" rev-parse --verify -q "$HAL_COMMIT^{commit}" >/dev/null || die "HAL_COMMIT '$HAL_COMMIT' does not exist in $REPO"
HAL_SHA=$(git -C "$REPO" rev-parse "$HAL_COMMIT")
git -C "$REPO" rev-parse --verify -q "$BASE_COMMIT^{commit}" >/dev/null || die "BASE_COMMIT '$BASE_COMMIT' does not exist in $REPO"
BASE_SHA=$(git -C "$REPO" rev-parse "$BASE_COMMIT")

llk_hal_users() {  # LLK files at commit $1 that include hal/
  git -C "$REPO" grep -lE '#include *"[^"]*hal/' "$1" -- \
    "$BH/llk_lib/*.h" "$BH/common/inc/*.h" ":(exclude)$HAL_SRC" 2>/dev/null | sed "s#^$1:##" || true
}
HAL_USERS=$(llk_hal_users "$BASE_SHA")
[[ -n "$HAL_USERS" ]] && { echo "$HAL_USERS"; die "LLK files at BASE_COMMIT already use HAL. Choose an earlier BASE_COMMIT."; }

# For each test: python test -> the only C++ source it uses -> names of the HAL copies.
T_NAME=(); T_K=(); T_PY=(); T_HALPY=(); T_CPP=(); T_HALCPP=()
for e in "${TEST_ENTRIES[@]}"; do
  name=${e%%:*}; k=""; [[ "$e" == *:* ]] && k=${e#*:}
  py=$PY_DIR/$name.py
  git -C "$REPO" cat-file -e "$BASE_SHA:$py" 2>/dev/null || die "$py does not exist at BASE_COMMIT"
  srcs=$(git -C "$REPO" show "$BASE_SHA:$py" | grep -oE 'sources/[A-Za-z0-9_]+\.cpp' | sort -u)
  [[ $(echo "$srcs" | grep -c .) == 1 ]] || die "$name: expected exactly one sources/*.cpp, found: $(echo $srcs)"
  stem=$(basename "$srcs" .cpp)
  if [[ "$stem" == *_test ]]; then hal_stem=${stem%_test}_hal_test; else hal_stem=${stem}_hal; fi
  git -C "$REPO" cat-file -e "$BASE_SHA:$SRC_DIR/$stem.cpp" 2>/dev/null || die "$SRC_DIR/$stem.cpp does not exist at BASE_COMMIT"
  for c in ${T_CPP[@]+"${T_CPP[@]}"}; do [[ "$c" == "$SRC_DIR/$stem.cpp" ]] && die "Two tests use the same source $stem.cpp"; done
  T_NAME+=("$name"); T_K+=("$k"); T_PY+=("$py"); T_HALPY+=("$PY_DIR/${name}_hal.py")
  T_CPP+=("$SRC_DIR/$stem.cpp"); T_HALCPP+=("$SRC_DIR/$hal_stem.cpp")
done
# run_test <path to .py> <k_expr or ""> <output log> [more pytest args...]   (cwd = tests/)
run_test() {
  local args=("python_tests/$(basename "$1")")
  [[ -n "$2" ]] && args+=(-k "$2")
  local out=$3; shift 3
  timeout "$TEST_TIMEOUT" pytest "${args[@]}" "$@" -q > "$out" 2>&1
}
expected_of() {  # <test[:k]> -> N from EXPECT_PASSED (empty if not given)
  local kv
  for kv in $EXPECT_PASSED; do [[ "${kv%=*}" == "$1" ]] && { echo "${kv##*=}"; return 0; }; done
  return 0
}
check_expected() {  # <test index> <log of the original test>: stop if "passed" is not the frozen number
  local entry=${T_NAME[$1]}${T_K[$1]:+:${T_K[$1]}} exp got
  exp=$(expected_of "$entry")
  [[ -z "$exp" ]] && return 0
  got=$(grep -oE '[0-9]+ passed' "$2" | tail -1 | grep -oE '[0-9]+' || echo 0)
  [[ "$got" == "$exp" ]] || die "The number of tests changed: $entry expected $exp passed (EXPECT_PASSED), got $got. See $2"
  echo "  $entry: $got passed, as expected (EXPECT_PASSED)"
}

RUN_ID=$(date +%Y%m%d_%H%M%S)_${MODE}_${BASE_SHA:0:8}_${MODEL}${RUN_TAG:+_$RUN_TAG}
RUN_DIR=$RUNS_ROOT/$RUN_ID
WS=$RUN_DIR/ws
LOGS=$RUN_DIR/logs
[[ -e "$RUN_DIR" ]] && die "$RUN_DIR already exists"
mkdir -p "$WS" "$LOGS" "$RUN_DIR/tmp"
# Everything the test framework compiles goes into this run (not the shared /tmp), so that the code size
# metrics measure exactly this run.
export TMPDIR="$RUN_DIR/tmp"
log "Run: $RUN_DIR (mode $MODE)"

# ------------------------------------------------- 1. tests + original LLKs ---
log "1/9 Export of BASE commit ${BASE_SHA:0:8}"
git -C "$REPO" archive "$BASE_SHA" | tar -x -C "$WS"   # whole tree: tests/sfpi-info.sh points outside tt-llk
rm -rf "$WS/$LLK/tools/hal-llk-gen"                       # this tool, if BASE_COMMIT contains it

# ------------------------------------------ 2. HAL code + documentation ---
log "2/9 HAL from ${HAL_SHA:0:8}, documentation"
rm -rf "$WS/$HAL_SRC" "$WS/$HAL_DOCS"
git -C "$REPO" archive "$HAL_SHA" "$HAL_SRC" | tar -x -C "$WS"
mkdir -p "$WS/$HAL_DOCS"
DOCS_SHA=""
if [[ -n "$DOCS_COMMIT" ]]; then
  git -C "$REPO" rev-parse --verify -q "$DOCS_COMMIT^{commit}" >/dev/null || die "DOCS_COMMIT '$DOCS_COMMIT' does not exist"
  DOCS_SHA=$(git -C "$REPO" rev-parse "$DOCS_COMMIT")
  git -C "$REPO" cat-file -e "$DOCS_SHA:$HAL_DOCS" 2>/dev/null || die "$HAL_DOCS does not exist at DOCS_COMMIT ${DOCS_SHA:0:8}"
  git -C "$REPO" archive "$DOCS_SHA" "$HAL_DOCS" | tar -x -C "$WS"
fi
python3 - "$WS/$HAL_DOCS" "${DOCS_DIR:-}" "$DOCS_INVALID" "$DOCS_REMOVE" "${DOCS_SHA:0:8}" <<'PY'
import os
import re
import shutil
import sys

docs, snap, invalid, remove, docs_sha = sys.argv[1:]
invalid, remove = invalid.split(), remove.split()
src = {f: f"git {docs_sha} (docs/hal/)" for f in os.listdir(docs)}
if snap:
    if not os.path.isdir(snap):
        sys.exit(f"DOCS_DIR '{snap}' does not exist")
    for f in sorted(os.listdir(snap)):
        p = os.path.join(snap, f)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(docs, f))
            src[f] = f"snapshot {os.path.basename(os.path.normpath(snap))}/"
for f in remove:
    if os.path.exists(os.path.join(docs, f)):
        os.remove(os.path.join(docs, f))
        src.pop(f, None)
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

# --------------------------------------------------- 3. spec for mode B ---
if [[ "$MODE" == B ]]; then
  log "3/9 Spec (signatures only)"
  python3 "$HARNESS_DIR/make_spec.py" "$WS/$BH/llk_lib" "$WS/$SPEC_DIR/llk_lib"
  python3 "$HARNESS_DIR/make_spec.py" "$WS/$BH/common/inc" "$WS/$SPEC_DIR/common/inc" \
    --exclude ckernel_ops.h ckernel_template.h
else
  log "3/9 (mode $MODE: no spec)"
fi

# --------------------------------------- 4. python tests for the HAL sources ---
log "4/9 Python tests for the HAL versions ($NTESTS)"
for i in "${!T_NAME[@]}"; do
  o=$(basename "${T_CPP[$i]}"); h=$(basename "${T_HALCPP[$i]}")
  sed "s#sources/$o#sources/$h#g" "$WS/${T_PY[$i]}" > "$WS/${T_HALPY[$i]}"
  grep -q "sources/$h" "$WS/${T_HALPY[$i]}" || die "sed did not find 'sources/$o' in ${T_PY[$i]}"
  grep -q "sources/$o" "$WS/${T_HALPY[$i]}" && die "${T_HALPY[$i]} still mentions sources/$o"
  echo "  ${T_NAME[$i]}: $o -> $h"
done

# ----------------------------------------------- 5. permissions for this run ---
# '/path' in a --settings file is resolved from the folder of that file, so the file is written into $WS.
# Denied on top of settings/hal-settings-B.json: editing the original test sources, reading the other runs,
# REPO (full HAL source and original LLKs), RESULTS_ROOT, EXTRA_DENY and, with SIBLING_DENY=1, every other
# folder in SIBLING_ROOTS (worktrees, tools, results), except the folder that holds RUNS_ROOT (descended
# into) and the folder that holds the active venv. Each path is written as given, as its real path and
# under every symlink name in SIBLING_ROOTS that points to it.
log "5/9 Permissions"
if [[ "$MODE" == TRIVIAL ]]; then
  echo "  (mode TRIVIAL: no agent, no permissions)"
else
  SIBLING_DENY="$SIBLING_DENY" SIBLING_ROOTS="$SIBLING_ROOTS" EXTRA_DENY="$EXTRA_DENY" VENV_DIR="${VIRTUAL_ENV:-}" \
  python3 - "$HARNESS_DIR/settings/hal-settings-B.json" "$WS/hal-settings.json" "$WS/.claude/settings.json" \
    "$RUNS_ROOT" "$RUN_DIR" "$REPO" "$RESULTS_ROOT" "${T_CPP[@]}" <<'PY'
import glob
import json
import os
import sys

src, dst, ws_dst, runs_root, run_dir, repo, results_root = sys.argv[1:8]
orig_sources = sys.argv[8:]
cfg = json.load(open(src))
deny = cfg["permissions"]["deny"]
roots = [r for r in os.environ.get("SIBLING_ROOTS", "").split() if os.path.isdir(r)]


def add(rule, rules=deny):
    if rule not in rules:
        rules.append(rule)


for f in orig_sources:  # the original test sources are not modified
    add(f"Edit(/{f})")
aliases = [(os.path.realpath(e), e) for r in roots for e in glob.glob(os.path.join(r, "*"))
           if os.path.islink(e) and os.path.isdir(e)]


def path_forms(p):
    rp = os.path.realpath(p)
    out = [os.path.abspath(p), rp]
    for target, link in aliases:
        if rp == target or rp.startswith(target + "/"):
            out.append(link + rp[len(target):])
    return list(dict.fromkeys(out))


def add_deny(p):  # absolute path in settings syntax: //absolute/path/**
    for f in path_forms(p):
        add(f"Read(/{f}/**)")


for d in glob.glob(os.path.join(runs_root, "*")):
    if os.path.isdir(d) and os.path.realpath(d) != os.path.realpath(run_dir):
        add_deny(d)
add_deny(repo)
add_deny(results_root)
for p in os.environ.get("EXTRA_DENY", "").split():
    add_deny(p)
if os.environ.get("SIBLING_DENY") == "1":
    venv = os.path.realpath(os.environ["VENV_DIR"]) + "/" if os.environ.get("VENV_DIR") else ""
    runs_real = os.path.realpath(runs_root)

    def deny_children(parent):
        for d in sorted(glob.glob(os.path.join(parent, "*"))):
            rd = os.path.realpath(d)
            if not os.path.isdir(rd) or rd == runs_real or (venv and venv.startswith(rd + "/")):
                continue
            if runs_real.startswith(rd + "/"):  # holds RUNS_ROOT: deny what is next to RUNS_ROOT
                deny_children(d)
                continue
            add_deny(d)

    tops = [r for r in roots if runs_real.startswith(os.path.realpath(r) + "/")] or [os.path.dirname(runs_real)]
    for parent in sorted(set(tops) | set(roots)):
        deny_children(parent)
# HAL source anywhere under the sibling roots (also inside folders that are not denied, e.g. the venv's)
ws_rules = []
for r in sorted(set(roots + [os.path.dirname(os.path.realpath(runs_root))])):
    for f in path_forms(r):
        add(f"Read(/{f}/**/llk_lib/hal/**)")
        add(f"Read(/{f}/**/llk_lib/hal/**)", ws_rules)
for rule in ["Bash(git show *)", "Bash(git diff *)", "Bash(git grep *)", "Bash(git log -p*)"]:
    add(rule, ws_rules)
json.dump(cfg, open(dst, "w"), indent=2)
os.makedirs(os.path.dirname(ws_dst), exist_ok=True)
json.dump({"permissions": {"deny": ws_rules}}, open(ws_dst, "w"), indent=2)
print(f"[harness] {len(deny)} deny rules")
PY
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
model=$MODEL
timeout=$TIMEOUT
max_turns=$MAX_TURNS
max_budget_usd=${MAX_BUDGET:-}
tests=$TESTS
expect_passed=$EXPECT_PASSED
prompt_file=$PROMPT_FILE
prompt_md5=$([[ -f "$PROMPT_FILE" ]] && md5sum < "$PROMPT_FILE" | cut -d' ' -f1)
sibling_deny=$SIBLING_DENY
run_tag=$RUN_TAG
dry_run=$DRY_RUN
claude_version=$(claude --version 2>/dev/null || echo unknown)
host=$(hostname)
date=$(date -Iseconds)
EOF

# ------------------------------------------------------------- 7. baseline ---
cd "$WS/$LLK/tests"
log "6/9 setup_testing_env.sh"
./setup_testing_env.sh > "$LOGS/setup_env.txt" 2>&1 || die "setup_testing_env.sh failed, see $LOGS/setup_env.txt"
pip install -q -r requirements.txt > "$LOGS/pip.txt" 2>&1 || echo "WARNING: pip install did not succeed, see $LOGS/pip.txt"
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
  ran_of() {   # passed + failed + errors from the final pytest line
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
import sys
import xml.etree.ElementTree as ET

try:
    root = ET.parse(sys.argv[1]).getroot()
except Exception as e:
    sys.exit(f"no junit xml: {e}")
for tc in root.iter("testcase"):
    name = tc.get("name")
    if not name or name == "None":  # interruption trace (e.g. timeout), not a real variant
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
          "$(passed_of "$LOGS/baseline_$n.txt")" "$(interrupted "$LOGS/trivial_${v}_$n.txt" && echo ' - INTERRUPTED, not all variants were run')"
      done
    done
    if [[ $TRIV_BAD != 0 ]]; then echo "Result: PROBLEM: some variants pass with empty bodies, see logs/trivial_*_passed.txt"
    elif [[ $TRIV_PARTIAL != 0 ]]; then echo "Result: VALID for the variants that were run, but the run was INTERRUPTED (TRIV_TIMEOUT=$TRIV_TIMEOUT)"
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
  echo "WARNING: SKIP_BASELINE=1, the expected number of tests (EXPECT_PASSED) was not checked"
fi

# ------------------------------------------------------- 8. generation ---
cd "$WS"
PROMPT=$(cat "$PROMPT_FILE")
printf '%s\n' "$PROMPT" > "$LOGS/prompt_used.md"
if [[ "$DRY_RUN" == 1 ]]; then
  log "DRY_RUN=1: setup and original tests done, the agent is not started"
  echo "Workspace: $WS"; echo "Prompt:    $LOGS/prompt_used.md"
  for i in "${!T_NAME[@]}"; do echo "  ${T_NAME[$i]}: $(tail -1 "$LOGS/baseline_${T_NAME[$i]}.txt" 2>/dev/null)"; done
  exit 0
fi
log "8/9 Agent generates (model $MODEL, timeout $TIMEOUT, max turns $MAX_TURNS, budget ${MAX_BUDGET:-none})"
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
[[ $CLAUDE_RC == 124 ]] && echo "WARNING: the agent was stopped by the timeout ($TIMEOUT)."

# The agent's final report = the last "result" message of the session
python3 - "$LOGS/transcript.jsonl" > "$LOGS/final_report.md" <<'PY' || true
import json
import sys

for line in open(sys.argv[1]):
    try:
        e = json.loads(line)
    except Exception:
        continue
    if e.get("type") == "result":
        print(e.get("result", ""))
PY

# ----------------------------------------- 9. independent checks and report ---
log "9/9 Independent HAL tests (each after tt-smi -r) and checks"
cd "$WS/$LLK/tests"
TEST_RC=0
: > "$LOGS/reset_before_final.txt"
for i in "${!T_NAME[@]}"; do
  tt-smi -r >> "$LOGS/reset_before_final.txt" 2>&1 || true
  set +e
  run_test "${T_HALPY[$i]}" "${T_K[$i]}" "$LOGS/hal_test_${T_NAME[$i]}.txt"
  rc=$?
  set -e
  (( rc == 0 )) || TEST_RC=1
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
for i in "${!T_NAME[@]}"; do echo "== ${T_NAME[$i]}"; cat "$LOGS/hal_test_${T_NAME[$i]}.txt"; done > "$LOGS/hal_test.txt"

cd "$WS"
{
  echo "# Raw TT instructions in the generated code (ideally empty, except NOT_IN_HAL parts)"
  grep -rnE 'TTI_|TT_OP_|cfg_reg_rmw' "$OUT_DIR" "${T_HALCPP[@]}" 2>/dev/null || echo "(none)"
  echo; echo "# NOT_IN_HAL marks"
  grep -rn 'NOT_IN_HAL' "$OUT_DIR" 2>/dev/null || echo "(none)"
} > "$LOGS/raw_check.txt"

# Attempts to access forbidden paths (tool calls in the transcript; denied or not)
python3 - "$LOGS/transcript.jsonl" "$RUN_DIR" "$RUNS_ROOT" "$REPO" "$RESULTS_ROOT" > "$LOGS/forbidden_access.txt" <<'PY' || true
import json
import os
import re
import sys

path, run_dir, runs_root, repo, results_root = sys.argv[1:]
run_dir, runs_root, repo, results_root = (os.path.realpath(x) for x in (run_dir, runs_root, repo, results_root))
rel_pat = re.compile(r"llk_lib/hal\b|(?<!llk_spec/)tt_llk_(blackhole|wormhole_b0|quasar)/|tt_metal/hw/(ckernels|inc)/")
abs_path = re.compile(r"(/[\w.\-/~]+)")


def bad(txt):
    if rel_pat.search(txt):
        return True
    for p in abs_path.findall(txt):
        if p.startswith(repo + "/") or p == repo or p.startswith(results_root):
            return True
        if p.startswith(runs_root + "/") and not p.startswith(run_dir):
            return True
    return False


hits = 0
for line in open(path):
    try:
        e = json.loads(line)
    except Exception:
        continue
    if e.get("type") != "assistant":
        continue
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
import json
import sys

n = 0
for line in open(sys.argv[1]):
    try:
        e = json.loads(line)
    except Exception:
        continue
    if e.get("type") != "assistant":
        continue
    for c in e["message"].get("content", []):
        if c.get("type") == "tool_use" and c.get("name", "").startswith("mcp__"):
            n += 1
            print(f"{c['name']}: {json.dumps(c.get('input', {}), ensure_ascii=False)[:400]}")
print(f"\n# total MCP calls: {n}")
PY

# Traces of probing the HAL implementation outside the documentation (numbers only in probe_check.txt;
# probe_check_details.txt may contain HAL names that the documentation does not give).
python3 "$HARNESS_DIR/probe_check.py" "$LOGS/transcript.jsonl" "$WS/$HAL_DOCS" "$LOGS/probe_check_details.txt" \
  "$OUT_DIR" "${T_HALCPP[@]}" > "$LOGS/probe_check.txt" 2>&1 || echo "WARNING: probe_check.py failed"

python3 "$HARNESS_DIR/codestats.py" "$OUT_DIR" "${T_HALCPP[@]}" --json "$LOGS/codestats.json" --md "$LOGS/codestats.md" > /dev/null 2>&1 \
  || echo "WARNING: codestats.py failed"

git -C "$WS" add -A
git -C "$WS" diff --cached --stat HEAD > "$LOGS/diff_stat.txt"
git -C "$WS" diff --cached HEAD > "$LOGS/diff.patch"

# The test framework deletes its build folder at the start of every pytest session, so the originals and the
# HAL versions are compiled together in one session without the card (--compile-producer), in a separate TMPDIR.
log "Code size: compiling the originals and HAL versions in one session (no card)"
CS_TMP="$RUN_DIR/codesize_tmp"; mkdir -p "$CS_TMP"
CS_ARGS=()
for i in "${!T_NAME[@]}"; do CS_ARGS+=("python_tests/$(basename "${T_PY[$i]}")" "python_tests/$(basename "${T_HALPY[$i]}")"); done
CS_K=(-k "${T_K[0]}")   # a common -k filter only if all tests have the same one
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
HAL probing:      $(tail -1 "$LOGS/probe_check.txt" 2>/dev/null | sed 's/^# //') (logs/probe_check.txt)
Tokens/cost:      $(python3 -c "import json;m=json.load(open('$LOGS/metrics.json'));t=m['tokens'];print(f\"in {m['tokens_total_input']:,} (cache {m['cache_hit_ratio']}), out {t['output_tokens']:,}, \${m['cost_usd']}\")" 2>/dev/null || echo "?")
Pytest runs:      $(python3 -c "import json;m=json.load(open('$LOGS/metrics.json'));print(m['pytest_runs_total'], m['pytest_runs_by_kind'])" 2>/dev/null || echo "?")
MCP calls:        $(tail -1 "$LOGS/mcp_calls.txt" 2>/dev/null | sed 's/# //') (logs/mcp_calls.txt)
Code size (text, original -> HAL, mean per variant):
$(for i in "${!T_NAME[@]}"; do printf '  %-32s %s\n' "${T_NAME[$i]}" "$(python3 -c "
import json;m=json.load(open('$LOGS/codesize_${T_NAME[$i]}.json'))['per_component']
print(', '.join(c + ' ' + str(m[c]['orig']['text']['mean']) + '->' + str(m[c]['hal']['text']['mean']) + 'B' for c in ('unpack','math','pack')))" 2>/dev/null || echo "?")"; done)
Metrics:          $LOGS/metrics.md, $LOGS/codesize_<test>.md, $LOGS/elfray_tt_tti_<test>.md
Report:           $LOGS/final_report.md
Code:             $WS/$OUT_DIR
Archive:          $ARCHIVE
================================================
EOF
# ------------------------------------------------------- result archive ---
mkdir -p "$RESULTS_ROOT/$RUN_ID"
ARCH_PATHS=("ws/$OUT_DIR")
for i in "${!T_NAME[@]}"; do ARCH_PATHS+=("ws/${T_HALCPP[$i]}" "ws/${T_HALPY[$i]}"); done
tar -czf "$ARCHIVE" -C "$RUN_DIR" logs \
  $(cd "$RUN_DIR" && ls -d "${ARCH_PATHS[@]}" 2>/dev/null) \
  || echo "WARNING: archiving failed"
cp "$LOGS/final_report.md" "$LOGS/run_meta.txt" "$LOGS/raw_check.txt" "$LOGS/diff_stat.txt" "$LOGS/SUMMARY.txt" "$LOGS/metrics.md" "$LOGS/metrics.json" "$LOGS/mcp_calls.txt" "$LOGS/prompt_used.md" \
   "$LOGS"/codesize_*.md "$LOGS"/codesize_*.json "$LOGS"/elfray_tt_tti_* "$LOGS"/hal_test_*.txt "$LOGS"/baseline_*.txt \
   "$LOGS"/dirty_*.txt "$LOGS"/codestats.* "$LOGS"/probe_check.txt \
   "$RESULTS_ROOT/$RUN_ID/" 2>/dev/null || true

echo "Results saved: $ARCHIVE"
