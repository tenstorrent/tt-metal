#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# orchestrator_steps.sh — the executable steps of the Quasar codegen orchestrator.
#
# Every pipeline step the orchestrator (agents/quasar/orchestrator.md) runs is a
# function here. The playbook just sources this file and calls one function per
# step, passing per-run values as arguments — so no bash is hand-assembled in
# the prompt and no state is trusted to survive between Bash calls.
#
# Usage: run with cwd = $WORKTREE_DIR/tt_metal/tt-llk, then
#     source codegen/scripts/quasar/orchestrator_steps.sh
#     execute_step_writer_failed "unknown type 'vFloat'" 2

# Physical scripts dir (…/codegen/scripts), resolved from this file's location
# so python helpers are found regardless of cwd. `codegen` is a symlink in the
# worktree; following it to the source copy is fine — same code.
_ORCH_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- env-free state/run-json helpers ---------------------------------------
# Worktree root from cwd (cwd == <wt>/tt_metal/tt-llk). Subshell: no cwd change.
_wt()  { ( cd ../.. && pwd ); }
# LOG_DIR is the one bootstrap key kept in the worktree file.
_LOG() { python "$_ORCH_SCRIPTS/state.py" --worktree-dir "$(_wt)" get LOG_DIR; }
# Run-state accessors — `_L` is set once at the top of each function below.
sg()   { python "$_ORCH_SCRIPTS/state.py" --log-dir "$_L" get "$1"; }
ss()   { python "$_ORCH_SCRIPTS/state.py" --log-dir "$_L" set "$@"; }
rj()   { local sub="$1"; shift; python "$_ORCH_SCRIPTS/run_json_writer.py" "$sub" --log-dir "$_L" "$@"; }
# refresh_cost.sh recovers everything itself; hand it LOG_DIR to skip a lookup.
refresh_cost() { LOG_DIR="${_L:-$(_LOG)}" bash "$_ORCH_SCRIPTS/refresh_cost.sh"; }

# ===========================================================================
# Any step — emit a mid-step progress message (does not change the step).
# Arg: <message>.
# ===========================================================================
execute_step_message() {
    local _L; _L="$(_LOG)"
    rj message --message "$1"
}

# ===========================================================================
# Input — validate the router's handoff. Arg: absolute worktree dir.
# ===========================================================================
execute_step_validate_input() {
    local wt="$1"
    [ -n "$wt" ] && [ -d "$wt" ] || { echo "REJECT: WORKTREE_DIR missing or not a directory: '$wt'"; return 1; }
    cd "$wt/tt_metal/tt-llk" || { echo "REJECT: cannot cd into $wt/tt_metal/tt-llk"; return 1; }

    local S="$_ORCH_SCRIPTS" kn ta sm wb ldb ok=1
    kn="$(python "$S/state.py" --worktree-dir "$wt" get KERNEL_NAME)"
    ta="$(python "$S/state.py" --worktree-dir "$wt" get TARGET_ARCH)"
    sm="$(python "$S/state.py" --worktree-dir "$wt" get SFPI_MODE)"
    wb="$(python "$S/state.py" --worktree-dir "$wt" get WORKTREE_BRANCH)"
    ldb="$(python "$S/state.py" --worktree-dir "$wt" get LOG_DIR_BASE)"

    [ -n "$kn" ] || { echo "REJECT: KERNEL_NAME is empty"; ok=0; }
    [ "$ta" = "quasar" ] || { echo "REJECT: TARGET_ARCH must be 'quasar' (got '$ta')"; ok=0; }
    { [ "$sm" = "true" ] || [ "$sm" = "false" ]; } || { echo "REJECT: SFPI_MODE must be exactly true/false (got '$sm')"; ok=0; }
    [ -n "$wb" ] || { echo "REJECT: WORKTREE_BRANCH is empty"; ok=0; }
    [ "$ldb" = "/proj_sw/user_dev/llk_code_gen" ] || { echo "REJECT: LOG_DIR_BASE must be /proj_sw/user_dev/llk_code_gen (got '$ldb')"; ok=0; }
    [ "$ok" = 1 ] || return 1
    echo "OK: KERNEL_NAME=$kn TARGET_ARCH=$ta SFPI_MODE=$sm WORKTREE_BRANCH=$wb LOG_DIR_BASE=$ldb"
}

# ===========================================================================
# Step 0 — validate environment prerequisites (settings.validate()).
# ===========================================================================
execute_step_validate_env() {
    ( cd "$_ORCH_SCRIPTS/../.." \
        && PYTHONPATH=.. python -c "from codegen.config.settings import settings; issues = settings.validate(); [print(f'ISSUE: {i}') for i in issues]; exit(1) if issues else print('Environment OK')" )
}

# ===========================================================================
# Step 0 — compute run identity + timing and seed both state files.
# ===========================================================================
execute_step_setup_run() {
    local S="$_ORCH_SCRIPTS" wt
    wt="$(_wt)"

    local START_TIME KERNEL_NAME TARGET_ARCH WORKTREE_BRANCH SFPI_MODE LOG_DIR_BASE
    local RUN_ID LOG_DIR GIT_COMMIT CODEGEN_VERSION PROMPT BATCH_ID MODEL RUN_TYPE
    START_TIME="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    KERNEL_NAME="$(python "$S/state.py" --worktree-dir "$wt" get KERNEL_NAME)"
    TARGET_ARCH="$(python "$S/state.py" --worktree-dir "$wt" get TARGET_ARCH)"
    WORKTREE_BRANCH="$(python "$S/state.py" --worktree-dir "$wt" get WORKTREE_BRANCH)"
    SFPI_MODE="$(python "$S/state.py" --worktree-dir "$wt" get SFPI_MODE)"
    LOG_DIR_BASE="$(python "$S/state.py" --worktree-dir "$wt" get LOG_DIR_BASE)"
    RUN_ID="$(date +%Y-%m-%d)_${KERNEL_NAME}_${TARGET_ARCH}_$(head -c 4 /dev/urandom | xxd -p)"
    LOG_DIR="$LOG_DIR_BASE/quasar/$RUN_ID"
    GIT_COMMIT="$(git -C "$wt" rev-parse HEAD 2>/dev/null || echo unknown)"
    CODEGEN_VERSION="$(cat "$S/../agents/quasar/VERSION" 2>/dev/null | tr -d '[:space:]' || echo "")"
    PROMPT="Generate ${KERNEL_NAME} for ${TARGET_ARCH}"   # the original user prompt, verbatim
    BATCH_ID="${CODEGEN_BATCH_ID:-}"                       # empty string if not a batch run
    MODEL="${CODEGEN_MODEL:-$(python "$S/session_cost.py" --print-model 2>/dev/null)}"
    MODEL="${MODEL:-sonnet}"
    RUN_TYPE="$([ -n "$BATCH_ID" ] && echo ci || echo manual)"
    mkdir -p "$LOG_DIR/instructions"

    # LOG_DIR is the bootstrap key — write it to the worktree file so every
    # later step (and refresh_cost.sh) can recover it with no env vars.
    python "$S/state.py" --worktree-dir "$wt" set LOG_DIR "$LOG_DIR"

    # Everything else lives in the run-state file ($LOG_DIR/state.json).
    local _L="$LOG_DIR"
    ss WORKTREE_DIR    "$wt"
    ss KERNEL_NAME     "$KERNEL_NAME"
    ss TARGET_ARCH     "$TARGET_ARCH"
    ss WORKTREE_BRANCH "$WORKTREE_BRANCH"
    ss SFPI_MODE       "$SFPI_MODE" --json
    ss RUN_ID          "$RUN_ID"
    ss START_TIME      "$START_TIME"
    ss GIT_COMMIT      "$GIT_COMMIT"
    ss CODEGEN_VERSION "$CODEGEN_VERSION"
    ss PROMPT          "$PROMPT"
    ss BATCH_ID        "$BATCH_ID"
    ss MODEL           "$MODEL"
    ss RUN_TYPE        "$RUN_TYPE"
    echo "LOG_DIR=$LOG_DIR RUN_ID=$RUN_ID"
}

# ===========================================================================
# Step 1 — print candidate kernel files across arches for the agent to grep.
# ===========================================================================
execute_step_discover_kernels() {
    local arch f
    # SFPU kernels in the tt-llk library (path - arch)
    for arch in blackhole wormhole_b0 quasar; do
        for f in tt_llk_$arch/common/inc/sfpu/ckernel_sfpu_*.h; do
            [ -e "$f" ] && echo "$f - $arch"
        done
    done
    # hw SFPU ops (path - arch)
    for arch in blackhole wormhole_b0 quasar; do
        for f in ../hw/ckernels/$arch/metal/llk_api/llk_sfpu/ckernel_sfpu_*.h; do
            [ -e "$f" ] && echo "$f - $arch"
        done
    done
    # Math/Pack/Unpack kernels (path - arch)
    for arch in blackhole wormhole_b0 quasar; do
        for f in tt_llk_$arch/llk_lib/llk_math_*.h tt_llk_$arch/llk_lib/llk_pack*.h tt_llk_$arch/llk_lib/llk_unpack*.h; do
            [ -e "$f" ] && echo "$f - $arch"
        done
    done
}

# ===========================================================================
# Step 1 — record the chosen kernel identity + derived generated-file path.
# Args: <kernel_type> <ref_arch> <kernel_path>
# ===========================================================================
execute_step_set_kernel_identity() {
    local _L; _L="$(_LOG)"
    local kernel_type="$1" ref_arch="$2" kernel_path="$3" kn gen
    kn="$(sg KERNEL_NAME)"
    ss KERNEL_TYPE "$kernel_type"
    ss REF_ARCH    "$ref_arch"
    ss KERNEL_PATH "$kernel_path"
    if [ "$kernel_type" = "sfpu" ]; then
        gen="tt_metal/hw/ckernels/quasar/metal/llk_api/llk_sfpu/ckernel_sfpu_${kn}.h"
    else
        gen="tt_metal/tt-llk/tt_llk_quasar/${kernel_path}"
    fi
    ss GENERATED_KERNEL "$gen"
    echo "KERNEL_TYPE=$kernel_type REF_ARCH=$ref_arch GENERATED_KERNEL=$gen"
}

# ===========================================================================
# Step 2 — write the initial run.json, capture the session, seed all counters,
#          and snapshot the agent playbooks.
# ===========================================================================
execute_step_write_initial_run_json() {
    local _L; _L="$(_LOG)"
    local S="$_ORCH_SCRIPTS" wt; wt="$(_wt)"

    local PIPELINE_STEPS_JSON='[
  {"id":"analyzer","name":"Analyze","desc":"Research arch + analyze reference, produce solution approach"},
  {"id":"writer","name":"Write","desc":"Scaffold + fill kernel, compile-check"},
  {"id":"tester","name":"Test","desc":"Write/extend tests, run, internal 5-attempt fix loop"},
  {"id":"refiner","name":"Refine","desc":"Rewrite analysis after writer/tester failure (max 2 refinements)"},
  {"id":"optimizer","name":"Optimize","desc":"Replay-buffer optimization (success only)"},
  {"id":"format","name":"Format","desc":"Run pre-commit formatters on generated files"}
]'

    local RUN_ID START_TIME GIT_COMMIT CODEGEN_VERSION PROMPT BATCH_ID MODEL RUN_TYPE
    local KERNEL_NAME KERNEL_TYPE TARGET_ARCH REF_ARCH KERNEL_PATH GENERATED_KERNEL
    RUN_ID="$(sg RUN_ID)"
    START_TIME="$(sg START_TIME)"
    GIT_COMMIT="$(sg GIT_COMMIT)"
    CODEGEN_VERSION="$(sg CODEGEN_VERSION)"
    PROMPT="$(sg PROMPT)"
    BATCH_ID="$(sg BATCH_ID)"
    MODEL="$(sg MODEL)"
    RUN_TYPE="$(sg RUN_TYPE)"
    KERNEL_NAME="$(sg KERNEL_NAME)"
    KERNEL_TYPE="$(sg KERNEL_TYPE)"
    TARGET_ARCH="$(sg TARGET_ARCH)"
    REF_ARCH="$(sg REF_ARCH)"
    KERNEL_PATH="$(sg KERNEL_PATH)"
    GENERATED_KERNEL="$(sg GENERATED_KERNEL)"

    python "$S/run_json_writer.py" init \
        --log-dir "$_L" \
        --run-id "$RUN_ID" \
        --kernel "$KERNEL_NAME" \
        --kernel-type "$KERNEL_TYPE" \
        --arch "$TARGET_ARCH" \
        --reference-arch "$REF_ARCH" \
        --reference-file "tt_llk_${REF_ARCH}/${KERNEL_PATH}" \
        --generated-file "$GENERATED_KERNEL" \
        --start-time "$START_TIME" \
        --first-step "analyzer" \
        --first-message "Analyzing ${REF_ARCH} reference and producing solution approach for ${KERNEL_NAME}" \
        --prompt "$PROMPT" \
        --batch-id "$BATCH_ID" \
        --model "$MODEL" \
        --run-type "$RUN_TYPE" \
        --git-commit "$GIT_COMMIT" \
        --version "$CODEGEN_VERSION" \
        --phases-total 1 \
        --pipeline-steps "$PIPELINE_STEPS_JSON"

    local _SESSION_PAIR SESSION_ID PROJECT_CWD
    _SESSION_PAIR="$(python "$S/session_cost.py" --print-session 2>/dev/null || echo "")"
    SESSION_ID="$(echo "$_SESSION_PAIR" | awk '{print $1}')"
    PROJECT_CWD="$(echo "$_SESSION_PAIR" | cut -d' ' -f2-)"
    if [ -n "$SESSION_ID" ]; then
        ss SESSION_ID  "$SESSION_ID"
        ss PROJECT_CWD "$PROJECT_CWD"
    fi

    # Counters + soft-outcome fields, all seeded to their zero values.
    ss CYCLE                1        --json   # current writer-tester cycle (1..3)
    ss MAX_CYCLES           3        --json   # hard cap
    ss REFINEMENT_COUNT     0        --json   # how many times the refiner ran
    ss COMPILATION_ATTEMPTS 0        --json   # every compile across writer + tester loop
    ss DEBUG_CYCLES         0        --json   # refinement iterations (== REFINEMENT_COUNT at end)
    ss PHASES_TOTAL         1        --json   # cycles attempted; bumped to 2/3 as refinements happen
    ss PHASES_COMPLETED     0        --json   # 1 if a cycle passed, else 0
    ss TESTS_TOTAL          0        --json   # total test variants run in the successful cycle
    ss TESTS_PASSED         0        --json
    ss LINES_GENERATED      0        --json
    ss TESTS_GENERATED      false    --json   # true if the tester created new test files
    ss OPTIMIZED            false    --json   # true if optimizer applied a change
    ss OPTIMIZATION_TYPE    none              # replay | sfpi | none
    ss FORMATS_TESTED_JSON   '[]'    --json
    ss FORMATS_EXCLUDED_JSON '{}'    --json
    ss TOKENS_JSON '{"input":0,"output":0,"cache_read":0,"cache_creation":0,"total":0,"cost_usd":0}' --json
    ss OBSTACLE             ""

    # Agent playbook snapshot (frozen copy of what actually ran).
    local SRC="$wt/tt_metal/tt-llk/codegen/agents/quasar"
    cp "$SRC/llk-analyzer.md"         "$_L/instructions/"
    cp "$SRC/llk-kernel-writer.md"    "$_L/instructions/"
    cp "$SRC/llk-tester.md"           "$_L/instructions/"
    cp "$SRC/llk-analysis-refiner.md" "$_L/instructions/"
    cp "$SRC/llk-optimizer.md"        "$_L/instructions/"

    refresh_cost   # capture the initial spend in run.json
}

# ===========================================================================
# Step 4 — verify the analyzer produced a complete analysis doc. Prints
# OK / MISSING / INCOMPLETE and returns non-zero if the doc is unusable.
# ===========================================================================
execute_step_verify_analysis() {
    local _L; _L="$(_LOG)"
    local kn f h missing=""; kn="$(sg KERNEL_NAME)"
    f="codegen/artifacts/${kn}_analysis.md"
    [ -f "$f" ] || { echo "MISSING: $f does not exist"; return 1; }
    for h in "Problem Statement" "Target Pattern Survey" "Available Instructions" \
             "Semantic.*Instruction Mapping" "Solution Approach" "Format Applicability" \
             "Complexity & Phases"; do
        grep -qiE "^#{1,4}.*${h}" "$f" || missing="${missing}; ${h}"
    done
    [ -z "$missing" ] || { echo "INCOMPLETE: $f missing sections${missing}"; return 1; }
    echo "OK: $f has all required sections"
}

# ===========================================================================
# Step 4 — analyzer failed. Arg: <first meaningful error line>.
# ===========================================================================
execute_step_analyzer_failed() {
    local _L; _L="$(_LOG)"
    local err="$1" kn; kn="$(sg KERNEL_NAME)"
    ss ANALYZER_ERROR_LINE "$err"
    rj failure --step "analyzer" --agent "analyzer" --type "agent_error" \
        --message "$err" --resolved "false"
    rj finalize --status "failed" --final-result "compile_error" \
        --final-message "Analyzer failed to produce ${kn}_analysis.md"
}

# ===========================================================================
# Step 4 — analyzer passed; advance to writer cycle 1.
# ===========================================================================
execute_step_analyzer_passed() {
    local _L; _L="$(_LOG)"
    local kn; kn="$(sg KERNEL_NAME)"
    rj advance --new-step "writer" --new-message "Cycle 1 — writing kernel from analysis" \
        --prev-result "success" \
        --prev-message "Analysis complete — codegen/artifacts/${kn}_analysis.md" \
        --agent "writer"
    rj phase-start --phase 1 --name "cycle 1 (fresh analysis)"
    refresh_cost   # capture analyzer spend in run.json
}

# ===========================================================================
# Step 5a — reset per-cycle counters before spawning the writer.
# ===========================================================================
execute_step_writer_setup() {
    local _L; _L="$(_LOG)"
    ss PHASE_COMPILES           0    --json
    ss PHASE_TEST_DETAILS        ""
    ss PHASE_COMPILE_ERRORS_JSON '[]' --json
    # Safe per-cycle defaults for the two values the tester subagent produces —
    # so a tester that returns without writing them can't crash phase-end
    # (--debug-cycles "" is an argparse int error) or the compile-count math.
    ss PHASE_DEBUGS             0    --json
    ss TESTER_COMPILE_COUNT     0    --json
}

# ===========================================================================
# Step 5a — writer reported FAILED. Args: <first_compile_error_line> <compiles_this_attempt(1|2)>
# ===========================================================================
execute_step_writer_failed() {
    local _L; _L="$(_LOG)"
    local err="$1" n="$2" cycle ca pc pcej
    cycle="$(sg CYCLE)"
    ss FIRST_COMPILE_ERROR_LINE "$err"
    ss COMPILES_THIS_ATTEMPT     "$n" --json
    ss PREV_RESULT               "compile_error"

    ca="$(sg COMPILATION_ATTEMPTS)"
    pc="$(sg PHASE_COMPILES)"
    pcej="$(sg PHASE_COMPILE_ERRORS_JSON)"
    ca=$((ca + n))
    pc=$((pc + n))
    pcej="$(python -c "
import json, sys
errors = json.loads(sys.argv[1])
errors.append(sys.argv[2])
print(json.dumps(errors))
" "$pcej" "$err")"
    ss COMPILATION_ATTEMPTS      "$ca"   --json
    ss PHASE_COMPILES            "$pc"   --json
    ss PHASE_COMPILE_ERRORS_JSON "$pcej" --json

    rj failure --step "writer_cycle_${cycle}" --agent "writer" --type "compile_error" \
        --message "$err" --resolved "false"
    rj phase-end --phase "$cycle" --test-result "failed" \
        --compilation-attempts "$pc" --debug-cycles 0 \
        --test-details "writer compile failed: $err" \
        --compile-errors-json "$pcej"
    refresh_cost

    # Tell the orchestrator which branch to take (so it need not track CYCLE).
    local mc; mc="$(sg MAX_CYCLES)"
    if [ "$cycle" -ge "$mc" ]; then
        echo "AT_CAP=yes (cycle ${cycle}/${mc}) — next: execute_step_mark_status failed compile_error, then Step 8"
    else
        echo "AT_CAP=no (cycle ${cycle}/${mc}) — next: Step 5c refiner"
    fi
}

# ===========================================================================
# Set the terminal STATUS/FINAL_RESULT pair. Args: <status> <final_result>
# (e.g. failed compile_error  |  compiled test_failure)
# ===========================================================================
execute_step_mark_status() {
    local _L; _L="$(_LOG)"
    ss STATUS       "$1"
    ss FINAL_RESULT "$2"
}

# ===========================================================================
# Step 5b — advance to the tester.
# ===========================================================================
execute_step_tester_advance() {
    local _L; _L="$(_LOG)"
    local cycle; cycle="$(sg CYCLE)"
    rj advance --new-step "tester" \
        --new-message "Cycle ${cycle} — writing/running tests (internal 5-attempt loop)" \
        --prev-result "success" --prev-message "Cycle ${cycle} writer compiled" --agent "tester"
    rj phase-test --phase "$cycle" --state "running"
}

# ===========================================================================
# Step 5b — tester reported PASS.
# ===========================================================================
execute_step_tester_passed() {
    local _L; _L="$(_LOG)"
    local cycle ca pc tcc pd tt tp pcej
    cycle="$(sg CYCLE)"
    ca="$(sg COMPILATION_ATTEMPTS)"; pc="$(sg PHASE_COMPILES)"; tcc="$(sg TESTER_COMPILE_COUNT)"
    ca=$((ca + tcc)); pc=$((pc + tcc))
    ss COMPILATION_ATTEMPTS "$ca" --json
    ss PHASE_COMPILES       "$pc" --json

    pd="$(sg PHASE_DEBUGS)"; tt="$(sg TESTS_TOTAL)"; tp="$(sg TESTS_PASSED)"; pcej="$(sg PHASE_COMPILE_ERRORS_JSON)"
    rj phase-end --phase "$cycle" --test-result "passed" \
        --compilation-attempts "$pc" --debug-cycles "$pd" \
        --test-details "${tp}/${tt} variants passed" --compile-errors-json "$pcej"

    if [ "$tp" = "$tt" ] && [ "$tt" -gt 0 ] 2>/dev/null; then
        ss STATUS "success"; ss FINAL_RESULT "success"
    else
        ss STATUS "compiled"; ss FINAL_RESULT "test_failure"
    fi
    ss PHASES_COMPLETED 1 --json   # a cycle passed
    refresh_cost
}

# ===========================================================================
# Step 5b — tester reported STUCK. Arg: <last failure signature>.
# ===========================================================================
execute_step_tester_stuck() {
    local _L; _L="$(_LOG)"
    local last="$1" cycle ca pc tcc pd pcej
    cycle="$(sg CYCLE)"
    ss TESTER_LAST_FAILURE_LINE "$last"
    ss PREV_RESULT              "test_failure"
    ca="$(sg COMPILATION_ATTEMPTS)"; pc="$(sg PHASE_COMPILES)"; tcc="$(sg TESTER_COMPILE_COUNT)"
    ca=$((ca + tcc)); pc=$((pc + tcc))
    ss COMPILATION_ATTEMPTS "$ca" --json
    ss PHASE_COMPILES       "$pc" --json
    pd="$(sg PHASE_DEBUGS)"; pcej="$(sg PHASE_COMPILE_ERRORS_JSON)"
    rj failure --step "tester_cycle_${cycle}" --agent "tester" --type "test_failure" \
        --message "$last" --resolved "false"
    rj phase-end --phase "$cycle" --test-result "failed" \
        --compilation-attempts "$pc" --debug-cycles "$pd" \
        --test-details "tester STUCK after 5 attempts: $last" --compile-errors-json "$pcej"
    refresh_cost

    # Tell the orchestrator which branch to take (so it need not track CYCLE).
    local mc; mc="$(sg MAX_CYCLES)"
    if [ "$cycle" -ge "$mc" ]; then
        echo "AT_CAP=yes (cycle ${cycle}/${mc}) — next: execute_step_mark_status compiled test_failure, then Step 8"
    else
        echo "AT_CAP=no (cycle ${cycle}/${mc}) — next: Step 5c refiner"
    fi
}

# ===========================================================================
# Step 5b — tester reported ENV_ERROR (infra broken; kernel innocent). Arg: <diagnosis>.
# Sets the terminal state itself — the refiner is NOT invoked.
# ===========================================================================
execute_step_tester_env_error() {
    local _L; _L="$(_LOG)"
    local diag="$1" cycle ca pc tcc pd pcej
    cycle="$(sg CYCLE)"
    ss TESTER_ENV_DIAGNOSIS "$diag"
    ca="$(sg COMPILATION_ATTEMPTS)"; pc="$(sg PHASE_COMPILES)"; tcc="$(sg TESTER_COMPILE_COUNT)"
    ca=$((ca + tcc)); pc=$((pc + tcc))
    ss COMPILATION_ATTEMPTS "$ca" --json
    ss PHASE_COMPILES       "$pc" --json
    pd="$(sg PHASE_DEBUGS)"; pcej="$(sg PHASE_COMPILE_ERRORS_JSON)"
    rj failure --step "tester_cycle_${cycle}" --agent "tester" --type "infra_error" \
        --message "$diag" --resolved "false"
    rj phase-end --phase "$cycle" --test-result "failed" \
        --compilation-attempts "$pc" --debug-cycles "$pd" \
        --test-details "ENV_ERROR: $diag" --compile-errors-json "$pcej"
    ss OBSTACLE     "$diag"
    ss STATUS       "compiled"
    ss FINAL_RESULT "test_failure"
    refresh_cost
}

# ===========================================================================
# Step 5c — advance to the refiner. Arg: <failure summary of the failed cycle>.
# ===========================================================================
execute_step_refiner_advance() {
    local _L; _L="$(_LOG)"
    local summary="$1" cycle prev
    cycle="$(sg CYCLE)"; prev="$(sg PREV_RESULT)"
    rj advance --new-step "refiner" \
        --new-message "Cycle ${cycle} failed — refining analysis (v${cycle})" \
        --prev-result "$prev" --prev-message "Cycle ${cycle} failed: ${summary}" --agent "refiner"
}

# ===========================================================================
# Step 5c — bump refinement counters after the refiner ran.
# ===========================================================================
execute_step_refiner_bump() {
    local _L; _L="$(_LOG)"
    local cycle rc dc pt
    cycle="$(sg CYCLE)"; rc="$(sg REFINEMENT_COUNT)"
    rc=$((rc + 1)); dc=$rc; pt=$((cycle + 1))
    ss REFINEMENT_COUNT "$rc" --json
    ss DEBUG_CYCLES     "$dc" --json
    ss PHASES_TOTAL     "$pt" --json
    rj metric --patch-json "{\"phases_total\": ${pt}, \"debug_cycles\": ${dc}}"
}

# ===========================================================================
# Step 5c — refiner reported ESCALATE. Arg: <reason>.
# ===========================================================================
execute_step_refiner_escalate() {
    local _L; _L="$(_LOG)"
    local reason="$1" cycle prev
    cycle="$(sg CYCLE)"; prev="$(sg PREV_RESULT)"
    ss REFINER_REASON "$reason"
    rj failure --step "refiner_v${cycle}" --agent "refiner" --type "agent_error" \
        --message "refiner escalated: ${reason}" --resolved "false"
    if [ "$prev" = "compile_error" ]; then
        ss STATUS "failed"
    else
        ss STATUS "compiled"
    fi
    ss FINAL_RESULT "$prev"
}

# ===========================================================================
# Step 5c — refiner reported REFINED; bump the cycle and re-enter the writer.
# ===========================================================================
execute_step_refiner_refined() {
    local _L; _L="$(_LOG)"
    local cycle
    cycle="$(sg CYCLE)"; cycle=$((cycle + 1))
    ss CYCLE "$cycle" --json
    rj advance --new-step "writer" \
        --new-message "Cycle ${cycle} — writing kernel from refined analysis" \
        --prev-result "success" \
        --prev-message "Refinement v$((cycle - 1)) complete — analysis rewritten in place" \
        --agent "writer"
    rj phase-start --phase "$cycle" --name "cycle ${cycle} (after refinement v$((cycle - 1)))"
}

# ===========================================================================
# Step 6a — snapshot the pre-optimization kernel for comparison/rollback.
# ===========================================================================
execute_step_optimizer_snapshot() {
    local _L; _L="$(_LOG)"
    local wt gen; wt="$(_wt)"; gen="$(sg GENERATED_KERNEL)"
    cp "$wt/$gen" "$_L/pre_opt_$(basename "$gen")"
}

# ===========================================================================
# Step 6a — advance to the optimizer (message depends on SFPI_MODE).
# ===========================================================================
execute_step_optimizer_advance() {
    local _L; _L="$(_LOG)"
    local sfpi kn cycle msg
    sfpi="$(sg SFPI_MODE)"; kn="$(sg KERNEL_NAME)"; cycle="$(sg CYCLE)"
    if [ "$sfpi" = "true" ]; then
        msg="Reimplementing ${kn} in SFPI and comparing instruction count vs the TTI baseline"
    else
        msg="Applying replay-buffer optimization to ${kn}"
    fi
    rj advance --new-step "optimizer" --new-message "$msg" \
        --prev-result "success" --prev-message "Cycle ${cycle} passed — entering optimization" \
        --agent "optimizer"
}

# ===========================================================================
# Refresh live cost into run.json (call at any boundary that lacks its own).
# ===========================================================================
execute_step_refresh_cost() {
    local _L; _L="$(_LOG)"
    refresh_cost
}

# ===========================================================================
# Step 7 — advance to the format/prettify step.
# ===========================================================================
execute_step_format_advance() {
    local _L; _L="$(_LOG)"
    local opt; opt="$(sg OPTIMIZED)"
    rj advance --new-step "format" \
        --new-message "Running pre-commit formatters on generated files" \
        --prev-result "success" --prev-message "Optimization complete (optimized=${opt})" \
        --agent "format"
}

# ===========================================================================
# Step 8a — record final line count of the generated kernel.
# ===========================================================================
execute_step_gather_metrics() {
    local _L; _L="$(_LOG)"
    local wt gen n; wt="$(_wt)"; gen="$(sg GENERATED_KERNEL)"
    n="$(wc -l < "$wt/$gen")"
    ss LINES_GENERATED "$n" --json
    echo "LINES_GENERATED=$n"
}

# ===========================================================================
# Step 8b — finalize run.json and append the runs.jsonl entry.
# ===========================================================================
execute_step_finalize_run() {
    local _L; _L="$(_LOG)"
    local S="$_ORCH_SCRIPTS"
    export END_TIME GIT_COMMIT CYCLE MAX_CYCLES REFINEMENT_COUNT PHASES_TOTAL PHASES_COMPLETED
    export COMPILATION_ATTEMPTS DEBUG_CYCLES TESTS_TOTAL TESTS_PASSED LINES_GENERATED TESTS_GENERATED
    export OPTIMIZED OPTIMIZATION_TYPE FORMATS_TESTED_JSON FORMATS_EXCLUDED_JSON TOKENS_JSON OBSTACLE
    export STATUS FINAL_RESULT KERNEL_NAME TARGET_ARCH LOG_DIR
    END_TIME="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    LOG_DIR="$_L"
    GIT_COMMIT="$(sg GIT_COMMIT)"
    CYCLE="$(sg CYCLE)"; MAX_CYCLES="$(sg MAX_CYCLES)"; REFINEMENT_COUNT="$(sg REFINEMENT_COUNT)"
    PHASES_TOTAL="$(sg PHASES_TOTAL)"; PHASES_COMPLETED="$(sg PHASES_COMPLETED)"
    COMPILATION_ATTEMPTS="$(sg COMPILATION_ATTEMPTS)"; DEBUG_CYCLES="$(sg DEBUG_CYCLES)"
    TESTS_TOTAL="$(sg TESTS_TOTAL)"; TESTS_PASSED="$(sg TESTS_PASSED)"; LINES_GENERATED="$(sg LINES_GENERATED)"
    TESTS_GENERATED="$(sg TESTS_GENERATED)"; OPTIMIZED="$(sg OPTIMIZED)"; OPTIMIZATION_TYPE="$(sg OPTIMIZATION_TYPE)"
    FORMATS_TESTED_JSON="$(sg FORMATS_TESTED_JSON)"; FORMATS_EXCLUDED_JSON="$(sg FORMATS_EXCLUDED_JSON)"
    TOKENS_JSON="$(sg TOKENS_JSON)"; OBSTACLE="$(sg OBSTACLE)"
    STATUS="$(sg STATUS)"; FINAL_RESULT="$(sg FINAL_RESULT)"
    KERNEL_NAME="$(sg KERNEL_NAME)"; TARGET_ARCH="$(sg TARGET_ARCH)"

    python "$S/run_json_writer.py" finalize \
        --log-dir "$_L" \
        --end-time "$END_TIME" \
        --status "$STATUS" \
        --final-result "$FINAL_RESULT" \
        --final-message "Run complete — ${KERNEL_NAME} on ${TARGET_ARCH} (${CYCLE}/${MAX_CYCLES} cycles)" \
        --patch-json "$(python - <<'PY'
import json, os
patch = {
    "phases_total": int(os.environ["PHASES_TOTAL"]),
    "phases_completed": int(os.environ["PHASES_COMPLETED"]),
    "compilation_attempts": int(os.environ["COMPILATION_ATTEMPTS"]),
    "debug_cycles": int(os.environ["DEBUG_CYCLES"]),
    "tests_total": int(os.environ["TESTS_TOTAL"]),
    "tests_passed": int(os.environ["TESTS_PASSED"]),
    "lines_generated": int(os.environ["LINES_GENERATED"]),
    "tests_generated": os.environ["TESTS_GENERATED"].lower() == "true",
    "prettified": False,
    "formatted": True,
    "optimized": os.environ.get("OPTIMIZED", "false").lower() == "true",
    "optimization_type": os.environ.get("OPTIMIZATION_TYPE", "none"),
    "formats_tested": json.loads(os.environ.get("FORMATS_TESTED_JSON", "[]")),
    "formats_excluded": json.loads(os.environ.get("FORMATS_EXCLUDED_JSON", "{}")),
    "obstacle": os.environ.get("OBSTACLE") or None,
    # Derived from steps_completed in run.json (always current, written
    # atomically by run_json_writer.py) rather than tracked as a separate value.
    "agents": json.loads(open(os.environ["LOG_DIR"] + "/run.json").read()).get("steps_completed", []),
    "tokens": json.loads(os.environ.get("TOKENS_JSON", "{\"input\":0,\"output\":0,\"cache_read\":0,\"cache_creation\":0,\"total\":0}")),
    "refinement_count": int(os.environ.get("REFINEMENT_COUNT", "0")),
    "cycles_attempted": int(os.environ.get("CYCLE", "1")),
    "cycles_cap": int(os.environ.get("MAX_CYCLES", "3")),
    # Base commit the worktree branch was cut from (origin/main at Step 0). The
    # generated.patch archived below applies cleanly on top of this commit.
    "base_commit": os.environ.get("GIT_COMMIT", "unknown"),
    "artifact_patch": "generated.patch",
}
print(json.dumps(patch))
PY
)"

    refresh_cost   # final authoritative refresh — overwrites the TOKENS_JSON

    # $LOG_DIR/run.json is the authoritative per-run record — derive the
    # runs.jsonl entry from it so the two artifacts stay in sync.
    python -c "import json; d=json.load(open('$_L/run.json')); print(json.dumps(d))" \
        >> /proj_sw/user_dev/llk_code_gen/quasar/runs.jsonl
}

# ===========================================================================
# Step 8 — copy simulator logs into LOG_DIR (before worktree cleanup).
# ===========================================================================
execute_step_copy_sim_logs() {
    local _L; _L="$(_LOG)"
    local ta; ta="$(sg TARGET_ARCH)"
    cp tests/python_tests/${ta}/emu_*_.log     "$_L/" 2>/dev/null || true
    cp tests/python_tests/${ta}/tt-exalens.log "$_L/" 2>/dev/null || true
}

# ===========================================================================
# Step 8 — capture every run-touched file as a single git diff.
# ===========================================================================
execute_step_write_generated_patch() {
    local _L; _L="$(_LOG)"
    local ta; ta="$(sg TARGET_ARCH)"
    local PATHSPEC="tt_llk_${ta} tests codegen/artifacts :(top)tt_metal/hw/ckernels/quasar/metal/llk_api/llk_sfpu"
    git add -AN -- $PATHSPEC 2>/dev/null || true
    git diff --binary HEAD -- $PATHSPEC > "$_L/generated.patch"
    git reset -q -- $PATHSPEC 2>/dev/null || true
}

# ===========================================================================
# Step 8 — extract subagent transcripts into LOG_DIR (non-fatal).
# ===========================================================================
execute_step_extract_transcripts() {
    local _L; _L="$(_LOG)"
    local S="$_ORCH_SCRIPTS" sid pcwd
    sid="$(sg SESSION_ID)"; pcwd="$(sg PROJECT_CWD)"
    python "$S/extract_run_transcripts.py" --log-dir "$_L" \
        ${sid:+--session-id "$sid" --project-cwd "$pcwd"} \
        || echo "extract_run_transcripts: skipped (non-fatal)"
}

# ===========================================================================
# Step 8 — build the report from run.json + agent logs + transcripts, write it
# to codegen/artifacts/<kernel>_report.md, and print it.
# ===========================================================================
execute_step_write_report() {
    local _L; _L="$(_LOG)"
    local S="$_ORCH_SCRIPTS" kn; kn="$(sg KERNEL_NAME)"
    python "$S/quasar/build_report.py" --log-dir "$_L" --out "codegen/artifacts/${kn}_report.md"
}

# ===========================================================================
# Step 8 — copy the final report into LOG_DIR.
# ===========================================================================
execute_step_copy_report() {
    local _L; _L="$(_LOG)"
    local kn; kn="$(sg KERNEL_NAME)"
    cp "codegen/artifacts/${kn}_report.md" "$_L/"
}
