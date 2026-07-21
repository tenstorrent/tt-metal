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

# --- out-of-space (ENOSPC) guard -------------------------------------------
# True when a captured error string carries the out-of-space signature.
_is_enospc() { printf '%s' "$1" | grep -qiE 'no space left on device|errno 28|enospc'; }
# High-priority abort banner for the orchestrator, on stderr so it stands out.
_no_space_banner() {
    printf '%s\n' \
      "################################################################" \
      "## NO SPACE LEFT ON DEVICE — HIGH PRIORITY, STOP NOW" \
      "## Spawn no agents and run no further steps. Report this run" \
      "## FAILED with reason: no space on device. Then run exactly:" \
      "##   execute_step_report_no_space \"<current step>\"" \
      "################################################################" >&2
}
# Run a disk-writing command; on an out-of-space failure print the abort banner
# and return 28, otherwise pass its output and exit code through unchanged.
_disk_guard() {
    local out rc
    out="$("$@" 2>&1)"; rc=$?
    [ -n "$out" ] && printf '%s\n' "$out"
    if [ "$rc" -ne 0 ] && _is_enospc "$out"; then _no_space_banner; return 28; fi
    return "$rc"
}

# --- env-free state/run-json helpers ---------------------------------------
# Worktree root from cwd (cwd == <wt>/tt_metal/tt-llk). Subshell: no cwd change.
_wt()  { ( cd ../.. && pwd ); }
# LOG_DIR is the one bootstrap key kept in the worktree file.
_LOG() { python "$_ORCH_SCRIPTS/state.py" --worktree-dir "$(_wt)" get LOG_DIR; }
# Run-state accessors — `_L` is set once at the top of each function below.
# ss/rj write to disk, so they run through _disk_guard; sg only reads.
sg()   { python "$_ORCH_SCRIPTS/state.py" --log-dir "$_L" get "$1"; }
ss()   { _disk_guard python "$_ORCH_SCRIPTS/state.py" --log-dir "$_L" set "$@"; }
rj()   { local sub="$1"; shift; _disk_guard python "$_ORCH_SCRIPTS/run_json_writer.py" "$sub" --log-dir "$_L" "$@"; }
# refresh_cost.sh recovers everything itself; hand it LOG_DIR to skip a lookup.
refresh_cost() { LOG_DIR="${_L:-$(_LOG)}" bash "$_ORCH_SCRIPTS/refresh_cost.sh"; }

# Pipeline stages shown on the dashboard. "setup" is entered before the worktree
# exists (execute_step_begin_setup) so a crash during worktree/venv/SFPI setup is
# visible; the remaining stages run inside the orchestrator. Shared by
# begin_setup and write_initial_run_json so both agree on the plan.
_PIPELINE_STEPS_JSON='[
  {"id":"setup","name":"Setup","desc":"Create worktree + build test venv/SFPI"},
  {"id":"analyzer","name":"Analyze","desc":"Research arch + analyze reference, produce solution approach"},
  {"id":"writer","name":"Write","desc":"Scaffold + fill kernel, compile-check"},
  {"id":"tester","name":"Test","desc":"Write/extend tests, run, internal 5-attempt fix loop"},
  {"id":"refiner","name":"Refine","desc":"Rewrite analysis after writer/tester failure (max 2 refinements)"},
  {"id":"optimizer","name":"Optimize","desc":"Replay-buffer optimization (success only)"},
  {"id":"format","name":"Format","desc":"Run pre-commit formatters on generated files"}
]'

# ===========================================================================
# Any step — emit a mid-step progress message (does not change the step).
# Arg: <message>.
# ===========================================================================
execute_step_message() {
    local _L; _L="$(_LOG)"
    rj message --message "$1"
}

# ===========================================================================
# Any step — out-of-space terminal handler. Call this the moment a step prints
# the NO SPACE banner. It retries the run.json failed-finalize every 30s for
# up to 10 minutes until the write lands (once space frees), appends the
# runs.jsonl entry, then returns. After it returns, report the run failed and
# run no further steps. Arg: <step where space ran out>.
# ===========================================================================
execute_step_report_no_space() {
    local _L; _L="$(_LOG)"
    local where="${1:-unknown}" deadline=$(( SECONDS + 600 )) attempt=0 rc
    while :; do
        attempt=$(( attempt + 1 ))
        python "$_ORCH_SCRIPTS/run_json_writer.py" finalize \
            --log-dir "$_L" \
            --status failed \
            --final-result compile_error \
            --final-message "Run aborted at ${where} — no space left on device" \
            --patch-json '{"obstacle":"no space left on device"}' >/dev/null 2>&1
        rc=$?
        if [ "$rc" -eq 0 ]; then
            python -c "import json; d=json.load(open('$_L/run.json')); print(json.dumps(d))" \
                >> /proj_sw/user_dev/llk_code_gen/quasar/runs.jsonl 2>/dev/null || true
            echo "NO_SPACE_REPORTED: run.json finalized failed (no space on device) after ${attempt} attempt(s)"
            return 0
        fi
        if [ "$SECONDS" -ge "$deadline" ]; then
            _no_space_banner
            echo "NO_SPACE_UNREPORTABLE: run.json still unwritable after 10 min / ${attempt} attempts — device full" >&2
            return 28
        fi
        echo "NO_SPACE_RETRY: finalize failed (attempt ${attempt}) — device full, retry in 30s" >&2
        sleep 30
    done
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
# Step 0 (pre-worktree) — put the run in motion BEFORE worktree setup so a crash
# during setup (git fetch hang, worktree add, venv/SFPI build) is visible on the
# dashboard. Computes the run identity, seeds run.json at status=running /
# step=setup, and echoes RUN_ID / LOG_DIR / START_TIME for the router to thread
# into worktree state so execute_step_setup_run reuses them.
# Args: <kernel> <arch> <log_dir_base>
# ===========================================================================
execute_step_begin_setup() {
    local kernel="$1" arch="$2" log_dir_base="$3"
    local START_TIME RUN_ID LOG_DIR BATCH_ID MODEL RUN_TYPE CODEGEN_VERSION
    START_TIME="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    RUN_ID="$(date +%Y-%m-%d)_${kernel}_${arch}_$(head -c 4 /dev/urandom | xxd -p)"
    LOG_DIR="$log_dir_base/quasar/$RUN_ID"
    BATCH_ID="${CODEGEN_BATCH_ID:-}"
    MODEL="${CODEGEN_MODEL:-$(python "$_ORCH_SCRIPTS/session_cost.py" --print-model 2>/dev/null)}"; MODEL="${MODEL:-sonnet}"
    RUN_TYPE="$([ -n "$BATCH_ID" ] && echo ci || echo manual)"
    CODEGEN_VERSION="$(cat "$_ORCH_SCRIPTS/../agents/quasar/VERSION" 2>/dev/null | tr -d '[:space:]' || echo "")"
    _disk_guard mkdir -p "$LOG_DIR/instructions" || return $?

    local _L="$LOG_DIR"
    # Identity keys begin_setup owns; execute_step_setup_run reuses these post-worktree.
    ss RUN_ID      "$RUN_ID"
    ss LOG_DIR     "$LOG_DIR"
    ss START_TIME  "$START_TIME"
    ss KERNEL_NAME "$kernel"
    ss TARGET_ARCH "$arch"

    rj init \
        --run-id "$RUN_ID" \
        --kernel "$kernel" \
        --arch "$arch" \
        --start-time "$START_TIME" \
        --first-step "setup" \
        --first-message "Creating worktree + building test venv/SFPI for ${kernel}" \
        --prompt "Generate ${kernel} for ${arch}" \
        --batch-id "$BATCH_ID" \
        --model "$MODEL" \
        --run-type "$RUN_TYPE" \
        --version "$CODEGEN_VERSION" \
        --phases-total 1 \
        --pipeline-steps "$_PIPELINE_STEPS_JSON" || return $?
    echo "LOG_DIR=$LOG_DIR RUN_ID=$RUN_ID START_TIME=$START_TIME"
}

# ===========================================================================
# Step 0 (post-worktree) — record that worktree setup finished, before the
# orchestrator's own setup steps run. Keeps the run on the setup step; the
# writer of the initial run.json advances setup → analyzer. Arg: <log_dir>.
# ===========================================================================
execute_step_setup_ready() {
    local _L="$1"
    rj message --message "Worktree + test env ready — entering orchestrator"
}

# ===========================================================================
# Step 0 — compute run identity + timing and seed both state files.
# ===========================================================================
execute_step_setup_run() {
    local S="$_ORCH_SCRIPTS" wt
    wt="$(_wt)"

    local START_TIME KERNEL_NAME TARGET_ARCH WORKTREE_BRANCH SFPI_MODE SKIP_TESTER HIDE_EXISTING_KERNEL LOG_DIR_BASE
    local RUN_ID LOG_DIR GIT_COMMIT CODEGEN_VERSION PROMPT BATCH_ID MODEL RUN_TYPE
    START_TIME="$(python "$S/state.py" --worktree-dir "$wt" get START_TIME)"; START_TIME="${START_TIME:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"
    KERNEL_NAME="$(python "$S/state.py" --worktree-dir "$wt" get KERNEL_NAME)"
    TARGET_ARCH="$(python "$S/state.py" --worktree-dir "$wt" get TARGET_ARCH)"
    WORKTREE_BRANCH="$(python "$S/state.py" --worktree-dir "$wt" get WORKTREE_BRANCH)"
    SFPI_MODE="$(python "$S/state.py" --worktree-dir "$wt" get SFPI_MODE)"
    SKIP_TESTER="$(python "$S/state.py" --worktree-dir "$wt" get SKIP_TESTER)"; SKIP_TESTER="${SKIP_TESTER:-false}"
    HIDE_EXISTING_KERNEL="$(python "$S/state.py" --worktree-dir "$wt" get HIDE_EXISTING_KERNEL)"; HIDE_EXISTING_KERNEL="${HIDE_EXISTING_KERNEL:-false}"
    LOG_DIR_BASE="$(python "$S/state.py" --worktree-dir "$wt" get LOG_DIR_BASE)"
    # Reuse begin_setup's identity if the router threaded it into worktree state;
    # otherwise compute fresh (flows that skip begin_setup).
    RUN_ID="$(python "$S/state.py" --worktree-dir "$wt" get RUN_ID)"
    LOG_DIR="$(python "$S/state.py" --worktree-dir "$wt" get LOG_DIR)"
    if [ -z "$RUN_ID" ] || [ -z "$LOG_DIR" ]; then
        RUN_ID="$(date +%Y-%m-%d)_${KERNEL_NAME}_${TARGET_ARCH}_$(head -c 4 /dev/urandom | xxd -p)"
        LOG_DIR="$LOG_DIR_BASE/quasar/$RUN_ID"
    fi
    GIT_COMMIT="$(git -C "$wt" rev-parse HEAD 2>/dev/null || echo unknown)"
    CODEGEN_VERSION="$(cat "$S/../agents/quasar/VERSION" 2>/dev/null | tr -d '[:space:]' || echo "")"
    PROMPT="Generate ${KERNEL_NAME} for ${TARGET_ARCH}"   # the original user prompt, verbatim
    BATCH_ID="${CODEGEN_BATCH_ID:-}"                       # empty string if not a batch run
    MODEL="${CODEGEN_MODEL:-$(python "$S/session_cost.py" --print-model 2>/dev/null)}"
    MODEL="${MODEL:-sonnet}"
    RUN_TYPE="$([ -n "$BATCH_ID" ] && echo ci || echo manual)"
    _disk_guard mkdir -p "$LOG_DIR/instructions" || return $?

    # LOG_DIR is the bootstrap key — write it to the worktree file so every
    # later step (and refresh_cost.sh) can recover it with no env vars.
    _disk_guard python "$S/state.py" --worktree-dir "$wt" set LOG_DIR "$LOG_DIR" || return $?

    # Everything else lives in the run-state file ($LOG_DIR/state.json).
    local _L="$LOG_DIR"
    ss WORKTREE_DIR    "$wt"
    ss KERNEL_NAME     "$KERNEL_NAME"
    ss TARGET_ARCH     "$TARGET_ARCH"
    ss WORKTREE_BRANCH "$WORKTREE_BRANCH"
    ss SFPI_MODE       "$SFPI_MODE" --json
    ss SKIP_TESTER     "$SKIP_TESTER" --json
    ss HIDE_EXISTING_KERNEL "$HIDE_EXISTING_KERNEL" --json
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
# Args: <kernel_type> <ref_arch> <kernel_path> [gen_path]
# gen_path: explicit repo-root-relative quasar dest. Required for non-SFPU
#           (Quasar uses semantic names, not the reference's letter-based ones,
#           so the dest cannot be derived mechanically). Empty for SFPU.
# ===========================================================================
execute_step_set_kernel_identity() {
    local _L; _L="$(_LOG)"
    local kernel_type="$1" ref_arch="$2" kernel_path="$3" gen_path="$4" kn gen
    kn="$(sg KERNEL_NAME)"
    ss KERNEL_TYPE "$kernel_type"
    ss REF_ARCH    "$ref_arch"
    ss KERNEL_PATH "$kernel_path"
    if [ -n "$gen_path" ]; then
        gen="$gen_path"                       # caller supplied the quasar dest (non-SFPU semantic name)
    elif [ "$kernel_type" = "sfpu" ]; then
        gen="tt_metal/hw/ckernels/quasar/metal/llk_api/llk_sfpu/ckernel_sfpu_${kn}.h"
    else
        gen="tt_metal/tt-llk/tt_llk_quasar/${kernel_path#tt_llk_*/}"   # fallback; letter name not auto-renamed
    fi
    ss GENERATED_KERNEL "$gen"
    echo "KERNEL_TYPE=$kernel_type REF_ARCH=$ref_arch GENERATED_KERNEL=$gen"
}

# ===========================================================================
# Step 1b — hide the existing implementation so the pipeline regenerates blind.
# When HIDE_EXISTING_KERNEL=true, git-remove AND commit the target op's files on
# the worktree branch, so the working tree AND HEAD carry no trace of them: the
# analyzer/writer follow their normal git-read policy and simply find no prior
# implementation (`git show HEAD:<path>` returns nothing). No-op unless the flag
# is set. base_commit (GIT_COMMIT, captured at setup BEFORE this) is unchanged,
# so the final generated.patch is still computed against origin/main.
# Run AFTER execute_step_set_kernel_identity and BEFORE the analyzer.
# ===========================================================================
execute_step_hide_existing_kernel() {
    local _L; _L="$(_LOG)"
    local hide; hide="$(sg HIDE_EXISTING_KERNEL 2>/dev/null || echo false)"
    if [ "$hide" != "true" ]; then echo "hide_existing_kernel: not requested — skipping"; return 0; fi
    local wt kn kt gen; wt="$(_wt)"; kn="$(sg KERNEL_NAME)"; kt="$(sg KERNEL_TYPE)"; gen="$(sg GENERATED_KERNEL)"
    local -a files=(); [ -n "$gen" ] && files+=("$gen")
    if [ "$kt" = "sfpu" ]; then
        files+=("tt_metal/tt-llk/tt_llk_quasar/common/inc/sfpu/ckernel_sfpu_${kn}.h")
        # Per-op metal LLK-API entry point (llk_math_eltwise_{unary,binary,ternary}_sfpu_{op}.h) —
        # arity isn't tracked in state, so glob it; anchored on ${kn} so it can't match the
        # shared *_init.h / *_macros.h files.
        files+=("tt_metal/hw/ckernels/quasar/metal/llk_api/llk_sfpu/llk_math_eltwise_*_sfpu_${kn}.h")
    fi
    local removed=0 f
    for f in "${files[@]}"; do
        if git -C "$wt" ls-files --error-unmatch -- "$f" >/dev/null 2>&1; then
            git -C "$wt" rm -q -f -- "$f" && { removed=$((removed + 1)); echo "  hid: $f"; }
        fi
    done
    if [ "$removed" -gt 0 ]; then
        git -C "$wt" -c user.name=llk_code_gen -c user.email=llk_code_gen@tenstorrent.com \
            commit -q -m "codegen: hide existing ${kn} implementation for blind regeneration" || true
        echo "hide_existing_kernel: removed ${removed} file(s), committed on worktree branch"
    else
        echo "hide_existing_kernel: no tracked ${kn} files to hide"
    fi
}

# ===========================================================================
# Step 2 — write the initial run.json, capture the session, seed all counters,
#          and snapshot the agent playbooks.
# ===========================================================================
execute_step_write_initial_run_json() {
    local _L; _L="$(_LOG)"
    local S="$_ORCH_SCRIPTS" wt; wt="$(_wt)"

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

    if [ -f "$_L/run.json" ]; then
        # begin_setup already created run.json at step=setup. Patch in the kernel
        # identity now known, then advance setup → analyzer (setup stays in the
        # history as a completed stage).
        local patch
        patch="$(python - "$KERNEL_TYPE" "$REF_ARCH" "$KERNEL_PATH" "$GENERATED_KERNEL" "$GIT_COMMIT" <<'PY'
import json, sys
kt, ra, rf, gf, gc = sys.argv[1:6]
print(json.dumps({"kernel_type": kt, "reference_arch": ra, "reference_file": rf,
                  "generated_file": gf, "git_commit": gc}))
PY
)"
        rj metric --patch-json "$patch" || return $?
        rj advance --new-step "analyzer" --prev-result success \
            --prev-message "Worktree + test env ready" \
            --new-message "Analyzing ${REF_ARCH} reference and producing solution approach for ${KERNEL_NAME}" \
            --agent analyzer || return $?
    else
        # No begin_setup (flow skipped it) — create run.json fresh at analyzer.
        _disk_guard python "$S/run_json_writer.py" init \
            --log-dir "$_L" \
            --run-id "$RUN_ID" \
            --kernel "$KERNEL_NAME" \
            --kernel-type "$KERNEL_TYPE" \
            --arch "$TARGET_ARCH" \
            --reference-arch "$REF_ARCH" \
            --reference-file "${KERNEL_PATH}" \
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
            --pipeline-steps "$_PIPELINE_STEPS_JSON" || return $?
    fi

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
    ss PRETTIFIED           false    --json   # true if the prettifier ran (success path only)
    ss FORMATTED            false    --json   # true if pre-commit formatting ran
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
    # Set terminal state like every other terminal handler; Step 8's finalize_run
    # performs the single authoritative rj finalize. Do NOT finalize here (that
    # would double-finalize, and Step 8 would then read STATUS/FINAL_RESULT unset).
    ss STATUS       "failed"
    ss FINAL_RESULT "compile_error"
    ss OBSTACLE     "Analyzer failed to produce ${kn}_analysis.md: ${err}"
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
    ss PHASE_TEST_DETAILS        ""           # per-phase test-details string; each phase-end step sets + reads it
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
    local err="$1" n="$2" cycle ca pc pcej cta
    cycle="$(sg CYCLE)"
    ss FIRST_COMPILE_ERROR_LINE "$err"
    ss PREV_RESULT               "compile_error"
    ss COMPILES_THIS_ATTEMPT     "$n" --json   # compiles this attempt; folded into the totals below

    cta="$(sg COMPILES_THIS_ATTEMPT)"
    ca="$(sg COMPILATION_ATTEMPTS)"
    pc="$(sg PHASE_COMPILES)"
    pcej="$(sg PHASE_COMPILE_ERRORS_JSON)"
    ca=$((ca + cta))
    pc=$((pc + cta))
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
    ss PHASE_TEST_DETAILS "writer compile failed: $err"
    rj phase-end --phase "$cycle" --test-result "failed" \
        --compilation-attempts "$pc" --debug-cycles 0 \
        --test-details "$(sg PHASE_TEST_DETAILS)" \
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
    ss PHASE_TEST_DETAILS "${tp}/${tt} variants passed"
    rj phase-end --phase "$cycle" --test-result "passed" \
        --compilation-attempts "$pc" --debug-cycles "$pd" \
        --test-details "$(sg PHASE_TEST_DETAILS)" --compile-errors-json "$pcej"

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
    ss PHASE_TEST_DETAILS "tester STUCK after 5 attempts: $last"
    rj phase-end --phase "$cycle" --test-result "failed" \
        --compilation-attempts "$pc" --debug-cycles "$pd" \
        --test-details "$(sg PHASE_TEST_DETAILS)" --compile-errors-json "$pcej"
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
    ss PHASE_TEST_DETAILS "ENV_ERROR: $diag"
    rj phase-end --phase "$cycle" --test-result "failed" \
        --compilation-attempts "$pc" --debug-cycles "$pd" \
        --test-details "$(sg PHASE_TEST_DETAILS)" --compile-errors-json "$pcej"
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

# Resolve the file holding the generated ALGORITHM, not the thin metal wrapper.
# For SFPU the algorithm lives in the tt-llk lib impl (the metal file is only a
# forwarding wrapper); return it when present in the worktree, else fall back to
# GENERATED_KERNEL. Non-SFPU: GENERATED_KERNEL already IS the algorithm. Prints a
# worktree-root-relative path. Requires $_L set by the caller (dynamic scope).
_algo_file() {
    local wt kn kt gen algo
    wt="$(_wt)"; kn="$(sg KERNEL_NAME)"; kt="$(sg KERNEL_TYPE)"; gen="$(sg GENERATED_KERNEL)"
    if [ "$kt" = "sfpu" ]; then
        algo="tt_metal/tt-llk/tt_llk_quasar/common/inc/sfpu/ckernel_sfpu_${kn}.h"
        [ -f "$wt/$algo" ] && { printf '%s' "$algo"; return; }
    fi
    printf '%s' "$gen"
}

# ===========================================================================
# Step 6a — snapshot the pre-optimization algorithm file for comparison/rollback.
# ===========================================================================
execute_step_optimizer_snapshot() {
    local _L; _L="$(_LOG)"
    local wt algo; wt="$(_wt)"; algo="$(_algo_file)"
    [ -f "$wt/$algo" ] && cp "$wt/$algo" "$_L/pre_opt_$(basename "$algo")"
}

# ===========================================================================
# Step 8a — snapshot the final generated kernel into LOG_DIR as the bare
# ckernel_sfpu_{op}.h. This is the optimized/final version the dashboard code
# section renders, paired with the pre_opt_ snapshot. Skip silently when no
# kernel was produced (e.g. analyzer failed). Record the filename for finalize.
# ===========================================================================
execute_step_snapshot_generated_kernel() {
    local _L; _L="$(_LOG)"
    local wt gen algo base; wt="$(_wt)"; gen="$(sg GENERATED_KERNEL)"; algo="$(_algo_file)"
    [ -f "$wt/$algo" ] || { echo "no generated kernel to snapshot"; return 0; }
    base="$(basename "$algo")"
    cp "$wt/$algo" "$_L/$base"                       # the whole final (post-opt) algorithm file
    ss OPTIMIZED_KERNEL_FILE "$base"
    # Keep the thin metal wrapper as a secondary artifact when it is a distinct file.
    [ "$algo" != "$gen" ] && [ -f "$wt/$gen" ] && cp "$wt/$gen" "$_L/wrapper_$(basename "$gen")"
    # Record the algorithm file (not the wrapper) as the run's generated file.
    rj metric --patch-json "{\"generated_file\": \"$algo\"}" 2>/dev/null || true
    echo "GENERATED_KERNEL_FILE=$base"
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
    local wt algo n; wt="$(_wt)"; algo="$(_algo_file)"
    n="$(wc -l < "$wt/$algo" 2>/dev/null || echo 0)"   # 0 when no kernel was generated (e.g. analyzer failed)
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
    export PRETTIFIED FORMATTED OPTIMIZED_KERNEL_FILE
    export STATUS FINAL_RESULT KERNEL_NAME TARGET_ARCH LOG_DIR WORKTREE_BRANCH
    END_TIME="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    LOG_DIR="$_L"
    GIT_COMMIT="$(sg GIT_COMMIT)"
    WORKTREE_BRANCH="$(sg WORKTREE_BRANCH)"
    CYCLE="$(sg CYCLE)"; MAX_CYCLES="$(sg MAX_CYCLES)"; REFINEMENT_COUNT="$(sg REFINEMENT_COUNT)"
    PHASES_TOTAL="$(sg PHASES_TOTAL)"; PHASES_COMPLETED="$(sg PHASES_COMPLETED)"
    COMPILATION_ATTEMPTS="$(sg COMPILATION_ATTEMPTS)"; DEBUG_CYCLES="$(sg DEBUG_CYCLES)"
    TESTS_TOTAL="$(sg TESTS_TOTAL)"; TESTS_PASSED="$(sg TESTS_PASSED)"; LINES_GENERATED="$(sg LINES_GENERATED)"
    TESTS_GENERATED="$(sg TESTS_GENERATED)"; OPTIMIZED="$(sg OPTIMIZED)"; OPTIMIZATION_TYPE="$(sg OPTIMIZATION_TYPE)"
    FORMATS_TESTED_JSON="$(sg FORMATS_TESTED_JSON)"; FORMATS_EXCLUDED_JSON="$(sg FORMATS_EXCLUDED_JSON)"
    TOKENS_JSON="$(sg TOKENS_JSON)"; OBSTACLE="$(sg OBSTACLE)"
    PRETTIFIED="$(sg PRETTIFIED)"; FORMATTED="$(sg FORMATTED)"
    OPTIMIZED_KERNEL_FILE="$(sg OPTIMIZED_KERNEL_FILE)"
    STATUS="$(sg STATUS)"; FINAL_RESULT="$(sg FINAL_RESULT)"
    KERNEL_NAME="$(sg KERNEL_NAME)"; TARGET_ARCH="$(sg TARGET_ARCH)"

    _disk_guard python "$S/run_json_writer.py" finalize \
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
    "prettified": os.environ.get("PRETTIFIED", "false").lower() == "true",
    "formatted": os.environ.get("FORMATTED", "false").lower() == "true",
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
    "git_branch": os.environ.get("WORKTREE_BRANCH", ""),
    "artifact_patch": "generated.patch",
    # Final generated kernel (bare ckernel_sfpu_{op}.h) snapshotted alongside
    # pre_opt_*; null when no kernel was produced (e.g. analyzer failed).
    "artifact_optimized_kernel": os.environ.get("OPTIMIZED_KERNEL_FILE") or None,
}
print(json.dumps(patch))
PY
)" || return $?

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
    local ta base; ta="$(sg TARGET_ARCH)"
    # Diff against the recorded base_commit (origin/main, captured at setup before
    # any blind-regeneration hide commit), NOT HEAD — so the patch is the net change
    # vs origin/main and applies cleanly there, regardless of an intermediate
    # HIDE_EXISTING_KERNEL deletion commit. Falls back to HEAD if unknown.
    base="$(sg GIT_COMMIT)"; { [ -n "$base" ] && [ "$base" != "unknown" ]; } || base="HEAD"
    local PATHSPEC="tt_llk_${ta} tests codegen/artifacts :(top)tt_metal/hw/ckernels/quasar/metal/llk_api :(top)tt_metal/hw/inc/api"
    git add -AN -- $PATHSPEC 2>/dev/null || true
    git diff --binary "$base" -- $PATHSPEC > "$_L/generated.patch"
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
    # codegen/artifacts/ is gitignored in the worktree, so generated.patch never
    # carries the analysis doc — copy it here or it's lost when the worktree is
    # removed. Tolerant: absent on an analyzer-failed run.
    cp "codegen/artifacts/${kn}_analysis.md" "$_L/" 2>/dev/null || true
}
