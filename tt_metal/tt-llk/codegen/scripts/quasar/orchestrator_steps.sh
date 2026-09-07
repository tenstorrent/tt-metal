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
  {"id":"optimizer","name":"Optimize","desc":"Replay-buffer / SFPI optimization; perf loop vs the original kernel when a baseline exists (success only)"},
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

    local S="$_ORCH_SCRIPTS" kn ta sm qsb wb ldb ok=1
    kn="$(python "$S/state.py" --worktree-dir "$wt" get KERNEL_NAME)"
    ta="$(python "$S/state.py" --worktree-dir "$wt" get TARGET_ARCH)"
    sm="$(python "$S/state.py" --worktree-dir "$wt" get SFPI_MODE)"
    qsb="$(python "$S/state.py" --worktree-dir "$wt" get QSR_SIM_BACKEND)"
    wb="$(python "$S/state.py" --worktree-dir "$wt" get WORKTREE_BRANCH)"
    ldb="$(python "$S/state.py" --worktree-dir "$wt" get LOG_DIR_BASE)"

    [ -n "$kn" ] || { echo "REJECT: KERNEL_NAME is empty"; ok=0; }
    [ "$ta" = "quasar" ] || { echo "REJECT: TARGET_ARCH must be 'quasar' (got '$ta')"; ok=0; }
    { [ "$sm" = "true" ] || [ "$sm" = "false" ]; } || { echo "REJECT: SFPI_MODE must be exactly true/false (got '$sm')"; ok=0; }
    { [ "$qsb" = "emu" ] || [ "$qsb" = "vcs" ]; } || { echo "REJECT: QSR_SIM_BACKEND must be emu/vcs (got '$qsb')"; ok=0; }
    [ -n "$wb" ] || { echo "REJECT: WORKTREE_BRANCH is empty"; ok=0; }
    [ "$ldb" = "/proj_sw/user_dev/llk_code_gen" ] || { echo "REJECT: LOG_DIR_BASE must be /proj_sw/user_dev/llk_code_gen (got '$ldb')"; ok=0; }
    [ "$ok" = 1 ] || return 1
    echo "OK: KERNEL_NAME=$kn TARGET_ARCH=$ta SFPI_MODE=$sm QSR_SIM_BACKEND=$qsb WORKTREE_BRANCH=$wb LOG_DIR_BASE=$ldb"
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
    local START_TIME RUN_ID LOG_DIR BATCH_ID MODEL RUN_TYPE CODEGEN_VERSION PROMPT QSR_BACKEND QUEUE_ITEM_ID
    QSR_BACKEND="${QSR_SIM_BACKEND:-emu}"
    [ "$QSR_BACKEND" = emulator ] && QSR_BACKEND=emu
    { [ "$QSR_BACKEND" = emu ] || [ "$QSR_BACKEND" = vcs ]; } || {
        echo "REJECT: QSR_SIM_BACKEND must be emu or vcs (got '$QSR_BACKEND')" >&2
        return 1
    }
    START_TIME="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    RUN_ID="$(date +%Y-%m-%d)_${kernel}_${arch}_$(head -c 4 /dev/urandom | xxd -p)"
    LOG_DIR="$log_dir_base/quasar/$RUN_ID"
    BATCH_ID="${CODEGEN_BATCH_ID:-}"
    MODEL="${CODEGEN_MODEL:-$(python "$_ORCH_SCRIPTS/session_cost.py" --print-model 2>/dev/null)}"; MODEL="${MODEL:-sonnet}"
    RUN_TYPE="$([ -n "$BATCH_ID" ] && echo ci || echo manual)"
    CODEGEN_VERSION="$(cat "$_ORCH_SCRIPTS/../agents/quasar/VERSION" 2>/dev/null | tr -d '[:space:]' || echo "")"
    PROMPT="${CODEGEN_USER_PROMPT:-Generate ${kernel} for ${arch}}"
    QUEUE_ITEM_ID="${CODEGEN_QUEUE_ITEM_ID:-}"
    _disk_guard mkdir -p "$LOG_DIR/instructions" || return $?

    local _L="$LOG_DIR"
    # Identity keys begin_setup owns; execute_step_setup_run reuses these post-worktree.
    ss RUN_ID      "$RUN_ID"
    ss LOG_DIR     "$LOG_DIR"
    ss START_TIME  "$START_TIME"
    ss KERNEL_NAME "$kernel"
    ss TARGET_ARCH "$arch"
    ss QSR_SIM_BACKEND "$QSR_BACKEND"
    ss QUEUE_ITEM_ID "$QUEUE_ITEM_ID"

    rj init \
        --run-id "$RUN_ID" \
        --kernel "$kernel" \
        --arch "$arch" \
        --start-time "$START_TIME" \
        --first-step "setup" \
        --first-message "Creating worktree + building test venv/SFPI for ${kernel}" \
        --prompt "$PROMPT" \
        --batch-id "$BATCH_ID" \
        --model "$MODEL" \
        --run-type "$RUN_TYPE" \
        --version "$CODEGEN_VERSION" \
        --phases-total 1 \
        --pipeline-steps "$_PIPELINE_STEPS_JSON" || return $?
    local bootstrap_patch
    bootstrap_patch="$(python - "$QSR_BACKEND" "$QUEUE_ITEM_ID" <<'PY'
import json, sys
backend, queue_item_id = sys.argv[1:3]
patch = {"quasar_backend": backend}
if queue_item_id:
    patch["queue_item_id"] = queue_item_id
print(json.dumps(patch))
PY
)"
    rj metric --patch-json "$bootstrap_patch" || return $?
    echo "LOG_DIR=$LOG_DIR RUN_ID=$RUN_ID START_TIME=$START_TIME QSR_SIM_BACKEND=$QSR_BACKEND"
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

    local START_TIME KERNEL_NAME TARGET_ARCH WORKTREE_BRANCH SFPI_MODE QSR_BACKEND QUEUE_ITEM_ID LOCK_TESTS REMOVE_TESTS HIDE_EXISTING_KERNEL LOG_DIR_BASE
    local RUN_ID LOG_DIR GIT_COMMIT CODEGEN_VERSION PROMPT BATCH_ID MODEL RUN_TYPE
    START_TIME="$(python "$S/state.py" --worktree-dir "$wt" get START_TIME)"; START_TIME="${START_TIME:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"
    KERNEL_NAME="$(python "$S/state.py" --worktree-dir "$wt" get KERNEL_NAME)"
    TARGET_ARCH="$(python "$S/state.py" --worktree-dir "$wt" get TARGET_ARCH)"
    WORKTREE_BRANCH="$(python "$S/state.py" --worktree-dir "$wt" get WORKTREE_BRANCH)"
    SFPI_MODE="$(python "$S/state.py" --worktree-dir "$wt" get SFPI_MODE)"
    QSR_BACKEND="$(python "$S/state.py" --worktree-dir "$wt" get QSR_SIM_BACKEND)"
    QSR_BACKEND="${QSR_BACKEND:-${QSR_SIM_BACKEND:-emu}}"
    [ "$QSR_BACKEND" = emulator ] && QSR_BACKEND=emu
    { [ "$QSR_BACKEND" = emu ] || [ "$QSR_BACKEND" = vcs ]; } || {
        echo "REJECT: QSR_SIM_BACKEND must be emu or vcs (got '$QSR_BACKEND')" >&2
        return 1
    }
    QUEUE_ITEM_ID="${CODEGEN_QUEUE_ITEM_ID:-}"
    LOCK_TESTS="$(python "$S/state.py" --worktree-dir "$wt" get LOCK_TESTS)"; LOCK_TESTS="${LOCK_TESTS:-false}"
    REMOVE_TESTS="$(python "$S/state.py" --worktree-dir "$wt" get REMOVE_TESTS)"; REMOVE_TESTS="${REMOVE_TESTS:-false}"
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
    PROMPT="${CODEGEN_USER_PROMPT:-Generate ${KERNEL_NAME} for ${TARGET_ARCH}}"
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
    ss QSR_SIM_BACKEND "$QSR_BACKEND"
    ss QUEUE_ITEM_ID   "$QUEUE_ITEM_ID"
    ss LOCK_TESTS     "$LOCK_TESTS" --json
    ss REMOVE_TESTS   "$REMOVE_TESTS" --json
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
#
# Hiding spans all three layers an op can occupy — the metal LLK-API dest, the
# tt-llk lib implementation wherever it lives under the arch tree (sfpu/,
# experimental/, or any future subfolder), and the compute-level API entry
# point. Every candidate is a git pathspec expanded through `git ls-files`, so
# globs match and untracked/absent paths drop out silently. Keying only on
# GENERATED_KERNEL is what previously let an op implemented under
# common/inc/experimental/ survive a "blind" run untouched.
#
# GENERATED_KERNEL is the only hidden path that comes back: the writer generates
# it fresh, exactly as it does when nothing was hidden. Every other hidden path —
# the tt-llk lib implementation and the compute-level API entry point — stays
# hidden for the whole run and gets no new version written.
# Run AFTER execute_step_set_kernel_identity and BEFORE the analyzer.
# ===========================================================================
execute_step_hide_existing_kernel() {
    local _L; _L="$(_LOG)"
    local hide; hide="$(sg HIDE_EXISTING_KERNEL 2>/dev/null || echo false)"
    if [ "$hide" != "true" ]; then echo "hide_existing_kernel: not requested — skipping"; return 0; fi

    # Resolve the session id before touching the repo: git-guard is armed by a marker file keyed
    # on it, so without an id the op can be hidden but its recovery cannot be blocked. Fail while
    # the worktree is still pristine, rather than after the removals are committed.
    local _sid
    _sid="$(sg SESSION_ID 2>/dev/null || echo "")"
    [ -z "$_sid" ] && _sid="${CLAUDE_CODE_SESSION_ID:-}"
    [ -z "$_sid" ] && _sid="$(python "$_ORCH_SCRIPTS/session_cost.py" --print-session 2>/dev/null | awk '{print $1}')"
    if [ -z "$_sid" ]; then
        ss GUARD_ARMED false --json
        echo "  ERROR: SESSION_ID unresolved — git-guard cannot be armed, so a blind run cannot" >&2
        echo "  be enforced. Nothing was hidden. Set CLAUDE_CODE_SESSION_ID or clear" >&2
        echo "  HIDE_EXISTING_KERNEL and rerun." >&2
        return 1
    fi

    local wt kn kt gen; wt="$(_wt)"; kn="$(sg KERNEL_NAME)"; kt="$(sg KERNEL_TYPE)"; gen="$(sg GENERATED_KERNEL)"
    local -a patterns=(); [ -n "$gen" ] && patterns+=("$gen")
    if [ "$kt" = "sfpu" ]; then
        # Anywhere under the arch's tt-llk tree — common/inc/sfpu/, common/inc/experimental/,
        # and any future subfolder. Anchored on ckernel_sfpu_${kn}.h so it can never match the
        # shared *_init.h / *_macros.h files.
        patterns+=("tt_metal/tt-llk/tt_llk_quasar/**/ckernel_sfpu_${kn}.h")
        # Per-op metal LLK-API entry point (llk_math_eltwise_{unary,binary,ternary}_sfpu_{op}.h) —
        # arity isn't tracked in state, so glob it; anchored on ${kn} so it can't match the
        # shared *_init.h / *_macros.h files.
        patterns+=("tt_metal/hw/ckernels/quasar/metal/llk_api/llk_sfpu/llk_math_eltwise_*_sfpu_${kn}.h")
    fi
    # Compute-level API entry point (eltwise_unary/${kn}.h, eltwise_binary/${kn}.h, …). Arity
    # isn't tracked in state, so glob the category folder; the basename is exact, so an op
    # folded into a shared header (activations.h, comp.h) yields no match and nothing is hidden.
    patterns+=("tt_metal/hw/inc/api/compute/*/${kn}.h")
    # Drive removals off `git ls-files` so glob pathspecs expand; a file removed by an earlier
    # pattern leaves the index and cannot be matched twice.
    local removed=0 pat f; local -a hidden=()
    for pat in "${patterns[@]}"; do
        while IFS= read -r f; do
            [ -n "$f" ] || continue
            git -C "$wt" rm -q -f -- "$f" && {
                removed=$((removed + 1)); echo "  hid: $f"
                # GENERATED_KERNEL is regenerated by the writer, so it IS this run's output and
                # must stay in generated.patch. Record only the paths that stay hidden.
                [ "$f" = "$gen" ] || hidden+=("$f")
            }
        done < <(git -C "$wt" ls-files -- "$pat")
    done
    # Step 8 turns these into exclude pathspecs so generated.patch carries no deletion of a
    # file this run deliberately ignored. Space-separated: every path is repo-relative and
    # space-free.
    ss HIDDEN_FILES "${hidden[*]}"

    # Arm git-guard for the rest of this session: from here on it denies every git-history read,
    # so the implementation just hidden cannot be recovered from the object store the worktree
    # shares with the parent repo. A marker file rather than an env var, because
    # HIDE_EXISTING_KERNEL is decided after Claude has started.
    _disk_guard touch "${TMPDIR:-/tmp}/codegen-blind-run-${_sid}" || return $?
    ss GUARD_ARMED true --json
    ss SESSION_ID "$_sid"
    echo "  git-guard armed (blind mode) for session $_sid"
    # A test that #includes a header we just hid would fail to compile, and under LOCK_TESTS
    # nobody may fix it. Repoint such includes at the generated kernel, or drop the line when
    # the file already includes it. Only ckernel_sfpu_<op>.h includes are touched, only inside
    # this arch's tests (other arches spell the same include but resolve their own file), and
    # the path is matched exactly: "sfpu/x.h" is also a suffix of "llk_sfpu/x.h".
    local -a includers=() repointed=()
    local gen_inc=""
    if [ "${#hidden[@]}" -gt 0 ]; then
        local rt arch_ doomed="" b b_re inc gen_base gen_re
        local inc_re='^[[:space:]]*#[[:space:]]*include[[:space:]]*"'
        rt="$(sg REMOVE_TESTS 2>/dev/null || echo false)"; arch_="$(sg TARGET_ARCH)"
        [ "$rt" = "true" ] && doomed="$(git -C "$wt" ls-files -- \
            "tt_metal/tt-llk/tests/python_tests/${arch_}/test_*${kn}*_${arch_}.py" \
            "tt_metal/tt-llk/tests/sources/${arch_}/*${kn}*_test.cpp" 2>/dev/null)"
        local -a scope=("tt_metal/tt-llk/tests/sources/${arch_}" "tt_metal/tt-llk/tests/python_tests/${arch_}" \
                        "tt_metal/tt-llk/tests/helpers/include/*_${arch_}.h")
        # How a test spells the generated kernel's include.
        gen_base="$(basename "$gen")"
        case "$gen" in */metal/llk_api/*) gen_inc="${gen#*/metal/llk_api/}" ;; *) gen_inc="$gen_base" ;; esac
        gen_re="$(printf '%s' "$gen_inc" | sed 's/[.]/\\./g')"
        for f in "${hidden[@]}"; do
            # The hidden header as a test includes it (last two path components).
            b="$(printf '%s\n' "$f" | awk -F/ '{print $(NF-1)"/"$NF}')"
            b_re="$(printf '%s' "$b" | sed 's/[.]/\\./g')"
            while IFS= read -r inc; do
                [ -n "$inc" ] || continue
                case "$doomed" in *"$inc"*) continue ;; esac
                if [ -n "$gen" ] && [ "$(basename "$f")" = "$gen_base" ]; then
                    local note
                    if grep -qE "${inc_re}${gen_re}\"" "$wt/$inc"; then
                        sed -i -E "\\|${inc_re}${b_re}\"|d" "$wt/$inc"
                        note="$inc: dropped include of $b (already includes $gen_inc)"
                    else
                        sed -i -E "s|(${inc_re})${b_re}\"|\\1${gen_inc}\"|" "$wt/$inc"
                        note="$inc: $b -> $gen_inc"
                    fi
                    if grep -qE "^[[:space:]]*#[[:space:]]*include[[:space:]]*[\"<]${b_re}[\">]" "$wt/$inc"; then
                        includers+=("$inc includes $b (could not be repointed)")   # e.g. <angle> include
                    else
                        repointed+=("$note")
                        git -C "$wt" add -- "$inc"
                    fi
                else
                    includers+=("$inc includes $b")
                fi
            done < <(git -C "$wt" grep -l -E -e "^[[:space:]]*#[[:space:]]*include[[:space:]]*[\"<]${b_re}[\">]" \
                        -- "${scope[@]}" 2>/dev/null)
        done
    fi
    local joined=""; for inc in "${repointed[@]}"; do joined="${joined}${joined:+ ; }${inc}"; done
    ss REPOINTED_INCLUDES "$joined"
    if [ "${#repointed[@]}" -gt 0 ]; then
        echo "  REPOINTED: test include(s) now target the regenerated kernel (${gen_inc}); part of the locked test from here on:"
        printf '             %s\n' "${repointed[@]}"
    fi
    if [ "${#includers[@]}" -gt 0 ]; then
        echo "  WARNING: a hidden header is still included by a test source on the branch."
        echo "           This run's compile WILL fail and no agent can fix it:"
        printf '             %s\n' "${includers[@]}"
        echo "           ACTION: stop the listed file(s) including the hidden header, or clear"
        echo "           HIDE_EXISTING_KERNEL. REMOVE_TESTS only deletes the op's dedicated tests."
    fi
    if [ "$removed" -gt 0 ]; then
        _infra_commit "codegen: hide existing ${kn} implementation for blind regeneration" \
            && echo "hide_existing_kernel: removed ${removed} file(s), repointed ${#repointed[@]} test include(s), committed on worktree branch" \
            || echo "hide_existing_kernel: removed ${removed} file(s), repointed ${#repointed[@]} test include(s) — COMMIT FAILED (see ERROR above); the files are gone from the working tree but not from HEAD"
    else
        echo "hide_existing_kernel: no tracked ${kn} files to hide"
    fi
}

# Commit what is staged, as the codegen user. --no-verify because the repo's pre-commit
# hook would reformat staged test files and abort these bookkeeping commits.
_infra_commit() {
    local out
    if out="$(git -C "$wt" -c user.name=llk_code_gen -c user.email=llk_code_gen@tenstorrent.com \
                commit -q --no-verify -m "$1" 2>&1)"; then
        return 0
    fi
    echo "  ERROR: git commit failed on the worktree branch:" >&2
    printf '%s\n' "$out" | sed 's/^/         /' >&2
    return 1
}

# ===========================================================================
# Step 1c — remove the op's existing tests so the tester authors them fresh. When
# REMOVE_TESTS=true, git-remove AND commit the op's arch-specific dedicated test
# files (Python test + C++ source) on the worktree branch. An op with no dedicated
# file lives in a shared unified SFPU test: a whole-file rm would destroy every
# other op's cases, so instead locate the shared files that register the op, record
# them in SHARED_TEST_FILES, and print them — the orchestrator excises only this op's
# slice (Step 2c) and commits it via execute_step_commit_test_excision. No-op unless
# the flag is set. base_commit (GIT_COMMIT, captured at setup BEFORE this) is
# unchanged, so the final generated.patch still diffs against origin/main.
# Run AFTER execute_step_set_kernel_identity and BEFORE the analyzer.
# ===========================================================================
execute_step_remove_existing_tests() {
    local _L; _L="$(_LOG)"
    local remove; remove="$(sg REMOVE_TESTS 2>/dev/null || echo false)"
    if [ "$remove" != "true" ]; then echo "remove_existing_tests: not requested — skipping"; return 0; fi
    local wt kn arch; wt="$(_wt)"; kn="$(sg KERNEL_NAME)"; arch="$(sg TARGET_ARCH)"
    ss SHARED_TEST_FILES ""
    # Dedicated files are arch-specific and op-anchored; patterns tolerate a prefix
    # (e.g. test_sfpu_where_quasar.py) but never match a shared unified test. Keep the
    # wildcards quoted so git expands them as repo-root-relative pathspecs — the shell's
    # cwd is already inside tt_metal/tt-llk, so shell globbing would double the prefix
    # and match nothing. git ls-files returns only tracked matches.
    local -a patterns=(
        "tt_metal/tt-llk/tests/python_tests/${arch}/test_*${kn}*_${arch}.py"
        "tt_metal/tt-llk/tests/sources/${arch}/*${kn}*_test.cpp"
    )
    local removed=0 pat f
    for pat in "${patterns[@]}"; do
        while IFS= read -r f; do
            [ -n "$f" ] || continue
            git -C "$wt" rm -q -f -- "$f" && { removed=$((removed + 1)); echo "  removed: $f"; }
        done < <(git -C "$wt" ls-files -- "$pat")
    done
    if [ "$removed" -gt 0 ]; then
        _infra_commit "codegen: remove existing ${kn} tests for regeneration" \
            && echo "remove_existing_tests: removed ${removed} dedicated file(s), committed on worktree branch" \
            || echo "remove_existing_tests: removed ${removed} dedicated file(s) — COMMIT FAILED (see ERROR above)"
        return 0
    fi
    # No dedicated file — the op registers into a shared unified test. Locate the shared
    # .py / dispatcher header / .cpp that carry the op (a registration token next to the
    # op name, case-insensitive) so the orchestrator excises only this op's slice.
    local -a shared=()
    while IFS= read -r f; do [ -n "$f" ] && shared+=("$f"); done < <(
        git -C "$wt" grep -il -E "(MathOperation|OpConfig|SfpuType|BinaryOp|prepare_|ckernel_sfpu_)[A-Za-z0-9_.:]*${kn}" \
            -- "tt_metal/tt-llk/tests/python_tests/${arch}" \
               "tt_metal/tt-llk/tests/sources/${arch}" \
               "tt_metal/tt-llk/tests/helpers/include/sfpu_operations_${arch}.h" 2>/dev/null
    )
    ss SHARED_TEST_FILES "${shared[*]}"
    if [ "${#shared[@]}" -gt 0 ]; then
        echo "remove_existing_tests: no dedicated ${kn} test files — op lives in a shared test."
        echo "  SHARED_TEST_FILES — orchestrator excises only the ${kn} slice, then runs execute_step_commit_test_excision:"
        printf '    %s\n' "${shared[@]}"
    else
        echo "remove_existing_tests: no dedicated ${kn} test files and no shared registration located — orchestrator must find and excise the op's slice"
    fi
}

# ===========================================================================
# Step 1c (shared path) — commit the orchestrator's excision of the op's slice
# from the shared unified test. Stage the files recorded in SHARED_TEST_FILES and
# commit them on the worktree branch. No-op when nothing was recorded or staged.
# Run AFTER the orchestrator edits the shared files and BEFORE the analyzer.
# ===========================================================================
execute_step_commit_test_excision() {
    local _L; _L="$(_LOG)"
    local wt kn files; wt="$(_wt)"; kn="$(sg KERNEL_NAME)"; files="$(sg SHARED_TEST_FILES)"
    [ -n "$files" ] || { echo "commit_test_excision: no shared files recorded — nothing to commit"; return 0; }
    git -C "$wt" add -- $files
    if git -C "$wt" diff --cached --quiet; then
        echo "commit_test_excision: no staged changes — nothing to commit"
        return 0
    fi
    _infra_commit "codegen: remove existing ${kn} cases from shared test for regeneration" \
        && echo "commit_test_excision: committed ${kn} excision on worktree branch" \
        || echo "commit_test_excision: ${kn} excision staged — COMMIT FAILED (see ERROR above)"
}

# ===========================================================================
# Perf comparison against the original kernel (Steps 2a, 6 and 8).
#
# Runs only when HIDE_EXISTING_KERNEL=true and LOCK_TESTS=true (same test for both
# kernels) and a perf module collects the op. Step 2a measures the original before
# the hide. The optimizer measures candidates with execute_step_perf_measure and
# keeps or reverts them; the orchestrator records the verdict with
# execute_step_perf_finalize. Each measurement is one run_test.sh run; its CSV is
# moved to $LOG_DIR and compared with perf_eval.py. Callers read one printed line
# and never touch a CSV.
# ===========================================================================

# Prints why the perf gate is not met and returns 1; returns 0 when it is met.
_perf_gate() {
    local hide lock remove cmp
    hide="$(sg HIDE_EXISTING_KERNEL 2>/dev/null || echo false)"
    lock="$(sg LOCK_TESTS 2>/dev/null || echo false)"
    remove="$(sg REMOVE_TESTS 2>/dev/null || echo false)"
    cmp="$(sg PERF_COMPARE 2>/dev/null || echo true)"; [ -n "$cmp" ] || cmp=true
    [ "$cmp" != "false" ]   || { echo "PERF_COMPARE=false"; return 1; }
    [ "$hide" = "true" ]    || { echo "HIDE_EXISTING_KERNEL=${hide:-unset} (need true)"; return 1; }
    [ "$lock" = "true" ]    || { echo "LOCK_TESTS=${lock:-unset} (need true)"; return 1; }
    [ "$remove" != "true" ] || { echo "REMOVE_TESTS=true (the test is regenerated, so nothing is comparable)"; return 1; }
    return 0
}

# One perf measurement. Args: <label> <full|variant> [build root].
# full = the op's whole sweep, variant = the single node id in PERF_TEST_ID.
# Leaves the CSV path in _PERF_CSV (empty if none); returns run_test.sh's exit code.
_PERF_CSV=""
_perf_run() {
    local label="$1" kind="$2" build_root="${3:-}"
    local wt arch module k tid llk mod_stem perf_dir rc
    wt="$(_wt)"; arch="$(sg TARGET_ARCH)"; module="$(sg PERF_MODULE)"; k="$(sg PERF_K)"; tid="$(sg PERF_TEST_ID)"
    llk="$wt/tt_metal/tt-llk"; mod_stem="${module%.py}"; perf_dir="$llk/perf_data/$mod_stem"
    local -a sel=()
    if [ "$kind" = "variant" ] && [ -n "$tid" ]; then sel=(--test-id "$tid")
    elif [ -n "$k" ]; then sel=(--k "$k"); fi
    # perf_data/ is gitignored build output. Wipe it before and after so a stale CSV can
    # never pass as this measurement; the find below handles both harness layouts.
    rm -rf "$llk/perf_data"
    _PERF_CSV=""
    (
        export QSR_SIM_BACKEND; QSR_SIM_BACKEND="$(sg QSR_SIM_BACKEND)"
        [ -n "$build_root" ] && export TT_LLK_LOCAL_ARTIFACT_ROOT="$build_root"
        bash "$llk/.claude/scripts/run_test.sh" run --worktree "$llk" --arch "$arch" \
            --test "$module" "${sel[@]}" --maxfail 0 --log-dir "$_L/perf_${label}"
    ); rc=$?
    local csv
    csv="$(find "$llk/perf_data" -type f -name "${mod_stem}.post.csv" -printf '%T@ %p\n' 2>/dev/null \
            | sort -rn | head -1 | cut -d' ' -f2-)"
    if [ -n "$csv" ] && [ -s "$csv" ]; then
        if _disk_guard cp "$csv" "$_L/perf_${label}.post.csv"; then
            _PERF_CSV="$_L/perf_${label}.post.csv"
            cp "${csv%.post.csv}.csv" "$_L/perf_${label}.raw.csv" 2>/dev/null || true
        fi
    fi
    rm -rf "$llk/perf_data"
    return "$rc"
}

# One line from a perf_eval.py result, 11 '|'-separated fields:
# verdict|median%|worst%|cur_cycles|base_cycles|variants|improved|neutral|regressed|worst_key|verdict_typical
# verdict = strict any-variant-slower rule (drives the loop); verdict_typical = median (reported).
_perf_summary() {
    python - "$1" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("not_measured|||||0|||||"); raise SystemExit(0)
wv = d.get("worst_variant") or {}
def f(x):
    if x is None: return ""
    return "%.2f" % x if isinstance(x, float) else str(x)
key = wv.get("key") or {}
def suf(v): return str(v).split(".")[-1]
parts = []
if key.get("formats.input_A"):
    parts.append(f"{key['formats.input_A']}->{key.get('formats.output') or '?'}")
if key.get("dest_acc"): parts.append("dest_acc=" + suf(key["dest_acc"]))
if key.get("approx_mode"): parts.append("approx=" + suf(key["approx_mode"]))
print("|".join([str(d.get("verdict") or "not_measured"), f(d.get("delta_pct_median")),
                f(d.get("delta_pct_worst")), f(wv.get("current_cycles")),
                f(wv.get("baseline_cycles")), str(d.get("variants_compared") or 0),
                str(d.get("variants_improved", "")), str(d.get("variants_neutral", "")),
                str(d.get("variants_regressed", "")), ",".join(parts),
                str(d.get("verdict_typical") or "")]))
PY
}

# After a full sweep that is still regressed, point PERF_TEST_ID at the worst variant so the
# next attempts measure the slow path. Matches the CSV key columns against the collected
# pytest ids by text, strictest column set first. Arg: <vs_baseline json>.
_perf_reaim() {
    local json="$1" collect="$_L/perf_baseline_collect.log" module arch tid
    module="$(sg PERF_MODULE)"; arch="$(sg TARGET_ARCH)"
    [ -s "$collect" ] || { echo "  re-aim skipped: no collection log"; return 0; }
    tid="$(python - "$json" "$collect" "$module" "$arch" <<'PY'
import json, re, sys
d = json.load(open(sys.argv[1])); key = (d.get("worst_variant") or {}).get("key") or {}
module, arch = sys.argv[3], sys.argv[4]
pat = re.compile(r"^(%s/)?%s::" % (re.escape(arch), re.escape(module)))
ids = [l.strip() for l in open(sys.argv[2], errors="replace") if pat.match(l.strip())]
def need(cols):
    out = []
    for c in cols:
        v = key.get(c)
        if v in (None, ""): continue
        if c == "formats.input_A": out.append(f"A:{v},")
        elif c == "formats.input_B": out.append(f"B:{v},")
        elif c == "formats.output": out.append(f"out:{v}]")
        elif "." in str(v): out.append(f"<{v}")
    return out
for cols in (("mathop", "formats.input_A", "formats.input_B", "formats.output",
              "dest_acc", "approx_mode", "dest_sync", "implied_math_format"),
             ("formats.input_A", "formats.output", "dest_acc", "approx_mode"),
             ("dest_acc", "approx_mode")):
    n = need(cols)
    hits = [i for i in ids if n and all(s in i for s in n)]
    if hits:
        print(hits[0]); break
else:
    print("")
PY
)"
    if [ -n "$tid" ]; then
        ss PERF_TEST_ID "$tid"
        echo "  re-aimed: attempts now measure the worst variant ($(_perf_summary "$json" | cut -d'|' -f10))"
    else
        echo "  re-aim skipped: worst variant not found among collected ids — attempts keep measuring the current variant"
    fi
}

# Evaluate <current csv> vs <reference csv> with the run's metric and the given
# regression threshold; writes <json_out>; prints the _perf_summary line.
_perf_eval() {
    local cur="$1" ref="$2" out="$3" regress="$4" metric kn
    metric="$(sg PERF_METRIC)"; kn="$(sg KERNEL_NAME)"
    python "$_ORCH_SCRIPTS/perf_eval.py" --current "$cur" --baseline "$ref" --op "$kn" \
        --metric "$metric" --regress-pct "$regress" --improve-pct "$regress" \
        --goal no_regress --json-out "$out" >/dev/null 2>&1 || true
    _perf_summary "$out"
}

# The run.json `perf` object, built from state. Requires $_L.
_perf_json() {
    python "$_ORCH_SCRIPTS/state.py" --log-dir "$_L" dump | python -c '
import json, sys
s = json.load(sys.stdin)
def num(k):
    v = s.get(k)
    if v in (None, ""): return None
    try: return float(v)
    except (TypeError, ValueError): return None
def s_(k):
    v = s.get(k); return v if v not in (None, "") else None
print(json.dumps({"perf": {
    "enabled": s.get("PERF_ENABLED") is True,
    "verdict": s_("PERF_VERDICT") or "not_measured",       # follows the median variant
    "verdict_note": s_("PERF_VERDICT_NOTE"),                # names regressed variants, if any
    "verdict_worst_case": s_("PERF_BEST_VS_BASELINE"),      # strict any-variant-slower rule
    "reason": s_("PERF_REASON"),
    "module": s_("PERF_MODULE"), "k": s_("PERF_K"), "metric": s_("PERF_METRIC"),
    "regress_pct": num("PERF_REGRESS_PCT"),
    "best": s_("PERF_BEST_LABEL"),
    "delta_pct_median": num("PERF_BEST_DELTA_MEDIAN_PCT"),
    "delta_pct_worst": num("PERF_BEST_DELTA_WORST_PCT"),
    "current_cycles": num("PERF_BEST_CYCLES"),
    "baseline_cycles": num("PERF_BEST_BASELINE_CYCLES"),
    "variants_compared": int(num("PERF_BEST_VARIANTS") or 0),
    "variants_improved": int(num("PERF_BEST_VARIANTS_IMPROVED") or 0),
    "variants_neutral": int(num("PERF_BEST_VARIANTS_NEUTRAL") or 0),
    "variants_regressed": int(num("PERF_BEST_VARIANTS_REGRESSED") or 0),
    "worst_variant": s_("PERF_BEST_WORST_KEY"),
    "attempts": int(num("PERF_ATTEMPTS") or 0),
    "kept": int(num("PERF_KEPT") or 0),
    "baseline_csv": s_("PERF_BASELINE_CSV"), "best_csv": s_("PERF_BEST_CSV"),
}}))'
}

# What the optimizer does after a keep or revert:
#   attempt N — some variant is still slower than the original and attempts remain
#   final     — best changed since its last full sweep
#   done      — nothing left to do
_perf_next() {
    local attempts max vb kept full
    attempts="$(sg PERF_ATTEMPTS)"; max="$(sg PERF_MAX_ATTEMPTS)"; vb="$(sg PERF_BEST_VS_BASELINE)"
    kept="$(sg PERF_KEPT)"; full="$(sg PERF_BEST_FULL)"
    if [ "$vb" = "regressed" ] && [ "${attempts:-0}" -lt "${max:-3}" ]; then echo "attempt $((attempts + 1))"
    elif [ "${kept:-0}" -gt 0 ] && [ "$full" != "true" ]; then echo "final"
    else echo "done"; fi
}

# ===========================================================================
# Step 2a — measure the original kernel's perf before Step 2b hides it.
# No-op (prints why) unless the perf gate is met and a perf module collects the op.
# Finds the module (SFPU: the shared unary/binary sweep with -k; others:
# perf_<op>_<arch>.py), records the first collected node id as the representative
# variant, runs the full sweep in a throwaway build root and deletes that root so
# no ELF of the hidden kernel survives. Never fails the run: any problem leaves
# PERF_ENABLED=false with PERF_REASON set.
# Run after write_initial_run_json and before hide_existing_kernel.
# ===========================================================================
execute_step_perf_baseline() {
    local _L; _L="$(_LOG)"
    ss PERF_ENABLED false --json
    ss PERF_VERDICT  not_measured
    ss PERF_ATTEMPTS 0 --json
    ss PERF_KEPT     0 --json
    local why
    if ! why="$(_perf_gate)"; then
        ss PERF_REASON "gate not met: $why"
        echo "perf_baseline: not applicable ($why) — skipping"
        return 0
    fi
    local wt kn kt arch llk regress max; wt="$(_wt)"; kn="$(sg KERNEL_NAME)"; kt="$(sg KERNEL_TYPE)"; arch="$(sg TARGET_ARCH)"
    llk="$wt/tt_metal/tt-llk"
    regress="$(sg PERF_REGRESS_PCT)"; [ -n "$regress" ] || { regress=2.0; ss PERF_REGRESS_PCT "$regress" --json; }
    max="$(sg PERF_MAX_ATTEMPTS)";   [ -n "$max" ]     || { max=3;       ss PERF_MAX_ATTEMPTS "$max" --json; }

    # Candidate "<module>|<-k token>" pairs.
    local -a cands=()
    if [ "$kt" = "sfpu" ]; then
        cands+=("perf_eltwise_unary_sfpu_${arch}.py|$(printf '%s' "$kn" | tr 'A-Z' 'a-z')")
        cands+=("perf_eltwise_binary_sfpu_${arch}.py|$(printf '%s' "$kn" | tr 'a-z' 'A-Z')")
    else
        local f
        for f in "$llk/tests/python_tests/${arch}"/perf_*"${kn}"*_"${arch}".py; do
            [ -e "$f" ] && cands+=("$(basename "$f")|")
        done
    fi
    local module="" k="" count=0 cand m t collect="$_L/perf_baseline_collect.log"
    for cand in "${cands[@]}"; do
        m="${cand%%|*}"; t="${cand#*|}"
        [ -f "$llk/tests/python_tests/${arch}/$m" ] || continue
        local -a sel=(); [ -n "$t" ] && sel=(--k "$t")
        count="$(bash "$llk/.claude/scripts/run_test.sh" count --worktree "$llk" --arch "$arch" \
                    --test "$m" "${sel[@]}" 2>"$collect" | tail -1 | tr -dc '0-9')"
        count="${count:-0}"
        if [ "$count" -gt 0 ]; then module="$m"; k="$t"; break; fi
    done
    if [ -z "$module" ]; then
        local tried=""; for cand in "${cands[@]}"; do tried="${tried}${tried:+, }${cand%%|*}"; done
        ss PERF_REASON "no perf test collects ${kn} variants (tried: ${tried:-none})"
        echo "perf_baseline: no perf test collects ${kn} variants (tried: ${tried:-none}) — perf comparison disabled"
        return 0
    fi
    # Representative variant: the first node id the -k selection collected.
    local tid; tid="$(grep -m1 -E "^(${arch}/)?${module}::" "$collect" | tr -d '\r' | sed 's/^[[:space:]]*//')"
    ss PERF_MODULE  "$module"
    ss PERF_K       "$k"
    ss PERF_TEST_ID "$tid"
    rj message --message "Perf baseline: measuring the original ${kn} with ${module} (${count} variants) before the hide"

    local build_root="${TMPDIR:-/tmp}/codegen-perf-baseline-$(sg RUN_ID)" rc
    rm -rf "$build_root"
    _perf_run baseline full "$build_root"; rc=$?
    rm -rf "$build_root"   # no ELF of the hidden kernel may survive
    if [ "$rc" -ne 0 ] || [ -z "$_PERF_CSV" ]; then
        ss PERF_REASON "baseline perf run failed (run_test.sh exit ${rc}, csv=${_PERF_CSV:-missing})"
        echo "perf_baseline: FAILED (run_test.sh exit ${rc}, csv=${_PERF_CSV:-missing}) — perf comparison disabled; log: $_L/perf_baseline/run.log"
        return 0
    fi
    # SFPU work runs on the math thread, so MATH_ISOLATE is the kernel's own cost.
    local metric="mean(L1_TO_L1)"
    if [ "$kt" = "sfpu" ] && head -1 "$_PERF_CSV" | grep -q 'mean(MATH_ISOLATE)'; then metric="mean(MATH_ISOLATE)"; fi
    ss PERF_METRIC       "$metric"
    ss PERF_BASELINE_CSV "$_PERF_CSV"
    ss PERF_REASON       ""
    ss PERF_ENABLED      true --json
    rj message --message "Perf baseline captured: ${module} --k '${k}' (${count} variants, metric ${metric}) → perf_baseline.post.csv"
    echo "PERF_BASELINE: module=${module} k='${k}' variants=${count} metric=${metric} regress_pct=${regress} test_id='${tid}' csv=${_PERF_CSV}"
}

# ===========================================================================
# Step 6 — measure the kernel in the tree. Args: <label> [full|variant] (default variant).
# Labels: entry (tester's kernel, full), attempt_N (candidate, variant), final (best, full).
# Prints one line; the caller acts on its action=keep|revert field. cur/base are the
# worst variant's cycles per tile. A failed run prints status=run_failed action=revert.
# ===========================================================================
execute_step_perf_measure() {
    local _L; _L="$(_LOG)"
    local label="${1:?label required}" kind="${2:-variant}"
    if [ "$(sg PERF_ENABLED)" != "true" ]; then echo "PERF label=${label} status=disabled ($(sg PERF_REASON)) action=none"; return 0; fi
    case "$kind" in full|variant) ;; *) echo "PERF label=${label} status=bad_args (kind must be full|variant) action=none"; return 0 ;; esac
    case "$label" in attempt*) ss PERF_ATTEMPTS "$(( $(sg PERF_ATTEMPTS) + 1 ))" --json ;; esac
    ss PERF_LAST_LABEL "$label"
    ss PERF_LAST_KIND  "$kind"
    local rc; _perf_run "$label" "$kind"; rc=$?
    ss PERF_LAST_RC "$rc" --json
    if [ "$rc" -ne 0 ] || [ -z "$_PERF_CSV" ]; then
        ss PERF_LAST_VS_BEST run_failed; ss PERF_LAST_VS_BASELINE run_failed
        echo "PERF label=${label} kind=${kind} status=run_failed exit=${rc} action=revert log=$_L/perf_${label}/run.log"
        return 0
    fi
    local regress base best metric vb vbest action
    regress="$(sg PERF_REGRESS_PCT)"; base="$(sg PERF_BASELINE_CSV)"; best="$(sg PERF_BEST_CSV)"; metric="$(sg PERF_METRIC)"
    vb="$(_perf_eval "$_PERF_CSV" "$base" "$_L/perf_${label}_vs_baseline.json" "$regress")"
    local vb_verdict vb_med vb_worst vb_cur vb_base vb_n vb_imp vb_neu vb_reg vb_key vb_typ
    IFS='|' read -r vb_verdict vb_med vb_worst vb_cur vb_base vb_n vb_imp vb_neu vb_reg vb_key vb_typ <<<"$vb"
    local vbest_verdict="n/a" vbest_med="" vbest_worst="" vbest_cur="" vbest_base="" vbest_n="" _x
    case "$label" in
        entry|final) action=keep ;;   # entry: nothing to beat yet; final: re-measures the kernel already kept
        *)
            if [ -n "$best" ] && [ -s "$best" ]; then
                vbest="$(_perf_eval "$_PERF_CSV" "$best" "$_L/perf_${label}_vs_best.json" 0.5)"
                IFS='|' read -r vbest_verdict vbest_med vbest_worst vbest_cur vbest_base vbest_n _x _x _x _x _x <<<"$vbest"
            fi
            case "$vbest_verdict" in improved|neutral) action=keep ;; *) action=revert ;; esac
            ;;
    esac
    ss PERF_LAST_CSV "$_PERF_CSV"
    ss PERF_LAST_VS_BASELINE "$vb_verdict"; ss PERF_LAST_VS_BASELINE_DELTA_PCT "$vb_med"
    ss PERF_LAST_VS_BEST "$vbest_verdict";  ss PERF_LAST_VS_BEST_DELTA_PCT "$vbest_med"
    ss PERF_LAST_WORST_KEY "$vb_key"
    # Full sweeps also print the per-variant tally and the worst variant.
    local tally=""
    [ "$kind" = "full" ] && tally="; ${vb_imp:-?} improved/${vb_neu:-?} neutral/${vb_reg:-?} regressed) typical=${vb_typ:-?} worst_variant=${vb_key:-?}" || tally=")"
    echo "PERF label=${label} kind=${kind} metric=${metric} cur=${vb_cur:-?} best=${vbest_base:-none} base=${vb_base:-?} vs_best=${vbest_verdict}${vbest_med:+(${vbest_med}%)} vs_baseline=${vb_verdict}(median ${vb_med:-?}%, worst ${vb_worst:-?}%${tally} variants=${vb_n} action=${action}"
}

# ===========================================================================
# Step 6 — make the kernel in the tree (measured as <label>) best-so-far: snapshot it
# to $LOG_DIR/perf_best_*, record its standing vs the original. Prints next=.
# ===========================================================================
execute_step_perf_keep() {
    local _L; _L="$(_LOG)"
    local label="${1:?label required}" wt algo gen csv
    if [ "$(sg PERF_ENABLED)" != "true" ]; then echo "PERF_KEEP: disabled next=done"; return 0; fi
    csv="$_L/perf_${label}.post.csv"
    [ -s "$csv" ] || { echo "PERF_KEEP: no measurement for '${label}' — run execute_step_perf_measure ${label} first next=$(_perf_next)"; return 1; }
    wt="$(_wt)"; algo="$(_algo_file)"; gen="$(sg GENERATED_KERNEL)"
    [ -f "$wt/$algo" ] && { _disk_guard cp "$wt/$algo" "$_L/perf_best_$(basename "$algo")" || return $?; }
    [ "$algo" != "$gen" ] && [ -f "$wt/$gen" ] && { _disk_guard cp "$wt/$gen" "$_L/perf_best_wrapper_$(basename "$gen")" || return $?; }
    local vb vb_verdict vb_med vb_worst vb_cur vb_base vb_n vb_imp vb_neu vb_reg vb_key vb_typ full=false
    vb="$(_perf_summary "$_L/perf_${label}_vs_baseline.json")"
    IFS='|' read -r vb_verdict vb_med vb_worst vb_cur vb_base vb_n vb_imp vb_neu vb_reg vb_key vb_typ <<<"$vb"
    [ "$(sg PERF_LAST_LABEL)" = "$label" ] && [ "$(sg PERF_LAST_KIND)" = "full" ] && full=true
    ss PERF_BEST_CSV             "$csv"
    ss PERF_BEST_LABEL           "$label"
    ss PERF_BEST_FULL            "$full" --json   # measured with the full sweep
    ss PERF_BEST_VS_BASELINE     "$vb_verdict"   # strict rule, drives the loop
    ss PERF_BEST_VERDICT_TYPICAL "$vb_typ"       # median rule, reported
    ss PERF_BEST_DELTA_MEDIAN_PCT "$vb_med"
    ss PERF_BEST_DELTA_WORST_PCT  "$vb_worst"
    ss PERF_BEST_CYCLES          "$vb_cur"
    ss PERF_BEST_BASELINE_CYCLES "$vb_base"
    ss PERF_BEST_VARIANTS        "$vb_n"
    ss PERF_BEST_VARIANTS_IMPROVED  "${vb_imp:-0}"
    ss PERF_BEST_VARIANTS_NEUTRAL   "${vb_neu:-0}"
    ss PERF_BEST_VARIANTS_REGRESSED "${vb_reg:-0}"
    ss PERF_BEST_WORST_KEY       "$vb_key"
    case "$label" in attempt*) ss PERF_KEPT "$(( $(sg PERF_KEPT) + 1 ))" --json ;; esac
    local tally=""
    [ "$full" = "true" ] && tally="; ${vb_imp:-?} improved/${vb_neu:-?} neutral/${vb_reg:-?} regressed"
    rj message --message "Perf: kept ${label} as best — typical ${vb_typ:-?}, worst-variant ${vb_verdict} vs original (median ${vb_med:-?}%, worst ${vb_worst:-?}%${tally} on $(sg PERF_METRIC))"
    # Still regressed after a full sweep: aim the next attempts at the worst variant.
    [ "$full" = "true" ] && [ "$vb_verdict" = "regressed" ] && _perf_reaim "$_L/perf_${label}_vs_baseline.json"
    echo "PERF_KEEP: best=${label} vs_baseline=${vb_verdict}(median ${vb_med:-?}%, worst ${vb_worst:-?}%${tally})${full:+ typical=${vb_typ:-?}}${vb_key:+ worst_variant=${vb_key}} attempts=$(sg PERF_ATTEMPTS)/$(sg PERF_MAX_ATTEMPTS) next=$(_perf_next)"
}

# ===========================================================================
# Step 6 — discard the candidate in the tree and restore best-so-far. Arg: [attempt label].
# With a label, a candidate that died before being measured still consumes its attempt.
# Prints next=.
# ===========================================================================
execute_step_perf_revert() {
    local _L; _L="$(_LOG)"
    local label="${1:-}"
    if [ "$(sg PERF_ENABLED)" != "true" ]; then echo "PERF_REVERT: disabled next=done"; return 0; fi
    case "$label" in
        attempt*)
            if [ "$(sg PERF_LAST_LABEL)" != "$label" ]; then
                ss PERF_ATTEMPTS "$(( $(sg PERF_ATTEMPTS) + 1 ))" --json
                ss PERF_LAST_LABEL "$label"; ss PERF_LAST_KIND none
                ss PERF_LAST_VS_BEST functional_failed; ss PERF_LAST_VS_BASELINE functional_failed
            fi ;;
    esac
    local wt algo gen best; wt="$(_wt)"; algo="$(_algo_file)"; gen="$(sg GENERATED_KERNEL)"; best="$(sg PERF_BEST_LABEL)"
    [ -n "$best" ] || { echo "PERF_REVERT: no best snapshot recorded — nothing restored next=done"; return 1; }
    [ -f "$_L/perf_best_$(basename "$algo")" ] && cp "$_L/perf_best_$(basename "$algo")" "$wt/$algo"
    [ "$algo" != "$gen" ] && [ -f "$_L/perf_best_wrapper_$(basename "$gen")" ] && cp "$_L/perf_best_wrapper_$(basename "$gen")" "$wt/$gen"
    rj message --message "Perf: reverted $(sg PERF_LAST_LABEL) — kept best=${best} ($(sg PERF_LAST_VS_BEST) vs best)"
    echo "PERF_REVERT: restored best=${best} attempts=$(sg PERF_ATTEMPTS)/$(sg PERF_MAX_ATTEMPTS) next=$(_perf_next)"
}

# ===========================================================================
# Step 6b — record the perf verdict in state and run.json. Never touches STATUS.
# ===========================================================================
execute_step_perf_finalize() {
    local _L; _L="$(_LOG)"
    local verdict vb
    if [ "$(sg PERF_ENABLED)" != "true" ]; then
        ss PERF_VERDICT not_measured
        rj metric --patch-json "$(_perf_json)" >/dev/null || true
        echo "PERF_FINAL: not measured ($(sg PERF_REASON))"
        return 0
    fi
    # The verdict follows the median variant; regressed variants are still named.
    vb="$(sg PERF_BEST_VS_BASELINE)"
    local typ; typ="$(sg PERF_BEST_VERDICT_TYPICAL)"; [ -n "$typ" ] || typ="$vb"
    case "$typ" in
        regressed) verdict=PERF_REGRESSED ;;
        improved)  verdict=PERF_IMPROVED ;;
        neutral)   verdict=PERF_NEUTRAL ;;
        *)         verdict=not_measured ;;
    esac
    local nreg note=""; nreg="$(sg PERF_BEST_VARIANTS_REGRESSED)"
    [ "${nreg:-0}" -gt 0 ] 2>/dev/null && note=" (${nreg} of $(sg PERF_BEST_VARIANTS) variants regressed: $(sg PERF_BEST_WORST_KEY))"
    ss PERF_VERDICT "$verdict"
    ss PERF_VERDICT_NOTE "$note"
    rj metric --patch-json "$(_perf_json)" >/dev/null || true
    echo "PERF_FINAL: verdict=${verdict}${note} best=$(sg PERF_BEST_LABEL) typical=${typ} median=$(sg PERF_BEST_DELTA_MEDIAN_PCT)% worst=$(sg PERF_BEST_DELTA_WORST_PCT)% tally=$(sg PERF_BEST_VARIANTS_IMPROVED) improved/$(sg PERF_BEST_VARIANTS_NEUTRAL) neutral/$(sg PERF_BEST_VARIANTS_REGRESSED) regressed of $(sg PERF_BEST_VARIANTS) metric=$(sg PERF_METRIC) attempts=$(sg PERF_ATTEMPTS) kept=$(sg PERF_KEPT) rule=verdict-follows-median;attempts-continue-while-any-variant-slower-by-more-than-$(sg PERF_REGRESS_PCT)%"
}

# ===========================================================================
# Step 6 — print the facts that route the optimizer stage.
# ===========================================================================
execute_step_perf_status() {
    local _L; _L="$(_LOG)"
    echo "PERF_ENABLED=$(sg PERF_ENABLED) KERNEL_TYPE=$(sg KERNEL_TYPE) PERF_METRIC=$(sg PERF_METRIC) PERF_MAX_ATTEMPTS=$(sg PERF_MAX_ATTEMPTS) PERF_REASON=$(sg PERF_REASON)"
}

# ===========================================================================
# Step 2 — write the initial run.json, capture the session, seed all counters,
#          and snapshot the agent playbooks.
# ===========================================================================
execute_step_write_initial_run_json() {
    local _L; _L="$(_LOG)"
    local S="$_ORCH_SCRIPTS" wt; wt="$(_wt)"

    local RUN_ID START_TIME GIT_COMMIT CODEGEN_VERSION PROMPT BATCH_ID MODEL RUN_TYPE
    local KERNEL_NAME KERNEL_TYPE TARGET_ARCH QSR_BACKEND QUEUE_ITEM_ID REF_ARCH KERNEL_PATH GENERATED_KERNEL
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
    QSR_BACKEND="$(sg QSR_SIM_BACKEND)"
    QUEUE_ITEM_ID="$(sg QUEUE_ITEM_ID)"
    REF_ARCH="$(sg REF_ARCH)"
    KERNEL_PATH="$(sg KERNEL_PATH)"
    GENERATED_KERNEL="$(sg GENERATED_KERNEL)"

    if [ -f "$_L/run.json" ]; then
        # begin_setup already created run.json at step=setup. Patch in the kernel
        # identity now known, then advance setup → analyzer (setup stays in the
        # history as a completed stage).
        local patch
        patch="$(python - "$KERNEL_TYPE" "$REF_ARCH" "$KERNEL_PATH" "$GENERATED_KERNEL" "$GIT_COMMIT" "$QSR_BACKEND" "$QUEUE_ITEM_ID" <<'PY'
import json, sys
kt, ra, rf, gf, gc, qsb, queue_item_id = sys.argv[1:8]
patch = {"kernel_type": kt, "reference_arch": ra, "reference_file": rf,
         "generated_file": gf, "git_commit": gc, "quasar_backend": qsb}
if queue_item_id:
    patch["queue_item_id"] = queue_item_id
print(json.dumps(patch))
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
        local initial_patch
        initial_patch="$(python - "$QSR_BACKEND" "$QUEUE_ITEM_ID" <<'PY'
import json, sys
backend, queue_item_id = sys.argv[1:3]
patch = {"quasar_backend": backend}
if queue_item_id:
    patch["queue_item_id"] = queue_item_id
print(json.dumps(patch))
PY
)"
        rj metric --patch-json "$initial_patch" || return $?
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
    ss PERF_ENABLED         false    --json   # perf comparison vs the original kernel (Step 2a decides)
    ss PERF_VERDICT         not_measured      # PERF_IMPROVED | PERF_NEUTRAL | PERF_REGRESSED | not_measured
    ss PERF_ATTEMPTS        0        --json   # optimizer perf attempts measured
    ss PERF_KEPT            0        --json   # optimizer perf attempts kept

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
    local sfpi kn cycle msg perf
    sfpi="$(sg SFPI_MODE)"; kn="$(sg KERNEL_NAME)"; cycle="$(sg CYCLE)"; perf="$(sg PERF_ENABLED)"
    if [ "$sfpi" = "true" ]; then
        msg="Reimplementing ${kn} in SFPI and comparing instruction count vs the TTI baseline"
    else
        msg="Applying replay-buffer optimization to ${kn}"
    fi
    [ "$perf" = "true" ] && msg="${msg}; perf loop vs the original kernel on $(sg PERF_METRIC) (up to $(sg PERF_MAX_ATTEMPTS) attempts)"
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
    export PRETTIFIED FORMATTED OPTIMIZED_KERNEL_FILE PERF_JSON
    export STATUS FINAL_RESULT KERNEL_NAME TARGET_ARCH LOG_DIR WORKTREE_BRANCH
    END_TIME="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    LOG_DIR="$_L"
    PERF_JSON="$(_perf_json 2>/dev/null || echo '{"perf": null}')"
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
    # Perf comparison vs the original kernel (Step 2a baseline, Step 6 loop).
    # Soft outcome: `enabled` false with a `reason` when the gate was not met.
    "perf": json.loads(os.environ.get("PERF_JSON") or '{"perf": null}').get("perf"),
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
    # Files hidden for blind regeneration and never rewritten (HIDE_EXISTING_KERNEL) are not
    # this run's output — they were removed on the branch so the pipeline could not read them.
    # Exclude them, so the patch carries no deletion and applies onto origin/main without
    # removing the prior implementation. HIDDEN_FILES holds repo-root-relative paths, hence
    # :(top,exclude); it is empty unless the hide step ran, and never contains
    # GENERATED_KERNEL.
    local hidden h; hidden="$(sg HIDDEN_FILES 2>/dev/null || echo '')"
    for h in $hidden; do PATHSPEC="$PATHSPEC :(top,exclude)$h"; done
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

    # Collect the git-guard log alongside the transcripts and record how many commands it
    # denied. GUARD_BLOCKS=0 means the guard ran and this run never reached for git history;
    # higher means git-guard.log names the commands. No log means nothing to count, so
    # GUARD_BLOCKS is null — "not measured" must stay distinguishable from a measured zero, or
    # an unenforced run looks like a clean one.
    local guard_src="${TMPDIR:-/tmp}/codegen-git-guard-${sid}.log" blocks=null
    if [ -n "$sid" ] && [ -f "$guard_src" ]; then
        _disk_guard cp "$guard_src" "$_L/git-guard.log" || return $?
        # `grep -c` prints 0 AND exits 1 on no match, so the fallback goes on the assignment —
        # `|| echo 0` inside the substitution would append a second zero, making GUARD_BLOCKS
        # "0\n0", which `ss --json` rejects. Match the tab-delimited verdict, not a column
        # index, so the count survives a change to the log's columns.
        blocks="$(grep -c $'\tBLOCK\t' "$_L/git-guard.log" 2>/dev/null)" || blocks=0
        rm -f "${TMPDIR:-/tmp}/codegen-blind-run-${sid}"
        echo "git-guard: $(wc -l < "$_L/git-guard.log") command(s) logged, $blocks blocked"
    elif [ "$(sg GUARD_ARMED 2>/dev/null || echo false)" = "true" ]; then
        # Armed at hide time but no log at the end: the hook never ran, so the removals went
        # ahead unenforced.
        echo "  WARNING: git-guard was armed but produced no log for session ${sid:-<unknown>}." >&2
        echo "  The hook did not run, so this run's blindness is UNVERIFIED (GUARD_BLOCKS=null)." >&2
    else
        echo "git-guard: no log for session ${sid:-<unknown>} (hook not active this run)"
    fi
    ss GUARD_BLOCKS "$blocks" --json
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
