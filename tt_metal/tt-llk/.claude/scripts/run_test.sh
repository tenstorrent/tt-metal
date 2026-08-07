#!/bin/bash
# run_test.sh — synchronous LLK test runner for codegen agents and humans.
#
# Invoke it like pytest: one blocking call, wait for the verdict. There is no
# timeout to set and no resume loop. Build entry and device locks are first come,
# first served and unbounded; once a run starts, a watcher bounds it — a hang is
# detected from a log stall and killed gracefully, so the call always returns.
#
# Two device paths:
#   quasar              Aether VCS/emulator — pytest --run-simulator --port;
#                       tt-exalens boots, runs, tears down. HANG = post-ready
#                       log stall. QSR_SIM_BACKEND selects vcs|emu.
#   blackhole/wormhole  real silicon (/dev/tenstorrent). HANG = TENSIX TIMED OUT
#                       or log stall; recovered with llk_triage.py + tt-smi -r.
#
# ONE global device lock (/tmp/tt-llk-test.lock) serialises execution. Build
# artifacts are isolated by run/worktree + full build-input digest, with a
# per-entry build lock, so independent producers never wipe each other's files.
#
# BUILD STAMP (why simulate can rebuild): `compile` and `simulate` are separate
# script invocations but derive the same attempt-owned artifact directory and
# stamp. If source/compiler/selection changes, `simulate` derives a new digest
# and rebuilds before execution.
#
# Usage:
#   run_test.sh <COMMAND> --worktree DIR --arch ARCH --test FILE [OPTIONS]
#
# Commands:
#   count     Count test variants (collection-only; prints an integer). Uses its
#             own artifact entry, separate from compiled outputs.
#   compile   Compile-producer step (parallel, -x). Locks only its artifact entry.
#   simulate  Run the pre-built variants. Takes the lock; rebuilds under it when
#             the build stamp is not ours; then runs on the device/emulator.
#   run       compile + run in one held-lock session (always rebuilds).
#
# Required:
#   --worktree DIR    LLK working dir (contains tests/ and tt_llk_<arch>/).
#   --arch     ARCH   quasar | blackhole | wormhole.
#   --test     FILE   Test file, e.g. test_sfpu_where_quasar.py
#
# Optional:
#   --maxfail  N      Stop after N failures (default 10). simulate/run only — lets a
#                     few variants fail so their tile dumps reveal the pattern, then
#                     pytest ends cleanly.
#   --k        EXPR   pytest -k filter (applied to compile AND run).
#   --test-id  ID     Full parametrize id (quotes/brackets safe). A leading
#                     "<arch>/" rootdir prefix is stripped automatically.
#   --no-split        Combined compile+run in one pytest invocation.
#   --jobs     N      compile parallelism (default 15).
#   --port     PORT   tt-exalens server port (default 5556).
#   --sim-path PATH   Override TT_UMD_SIMULATOR_PATH.
#   --lock     FILE   Global lock file (default /tmp/tt-llk-test.lock).
#   --log-dir  DIR    Append the run's output to <DIR>/run.log (compile output to
#                     <DIR>/compile.log).
#   --result-json-out FILE  Write the exact structured verification result here.
#   --stall    SECS   Log-stall seconds that mark a hang (default 180 emulator,
#                     300 silicon). Also settable via HANG_STALL.
#   --verbose         Print step headers to stderr.
#
# Exit codes:
#   0  PASS   1  FAIL   2  COMPILE_FAIL   3  ENV_ERROR   4  BAD_ARGS   5  HANG
#
# Verdict line (always last, on stderr):
#   === RUN_LLK_TESTS_VERDICT === <V> (exit N, phase=<cmd>, test=<f>, arch=<a>)

# ── Args ─────────────────────────────────────────────────────────────────────

CMD="${1:-}"
shift 2>/dev/null || true

WORKTREE="" ARCH="" TEST_FILE=""
MAXFAIL="10" K_FILTER="" TEST_ID=""
PORT="5556" JOBS="15"
LOCKFILE="" SIM_PATH="" LOG_DIR=""
RESULT_JSON_OUT=""
NO_SPLIT="false" VERBOSE="false"
STALL=""

# Tunables (rarely overridden).
WATCH_INTERVAL="${WATCH_INTERVAL:-5}"   # seconds between log-stall checks
GRACE_SECS="${GRACE_SECS:-30}"          # wait after SIGINT before SIGKILL
# tt-exalens readiness marker as it appears in the PYTEST log (the "[4B MODE]"
# string lives only in the separate tt-exalens.log, not here). The helper logs
# "tt-exalens ready (PID …)" to the pytest stream; match that (keep [4B MODE] as
# a fallback for 4B-mode configs that surface it).
READY_RE='tt-exalens ready|\[4B MODE\]'
QSR_SIM_BACKEND="${QSR_SIM_BACKEND:-emu}"
EMU_HOST="${EMU_HOST:-${QSR_AETHER_HOST:-${SSH_MACHINE_NAME:-soc-l-12}}}"
NNG_LOCAL_BASE="5555"                   # local NNG bind (infra-forwarded; fixed)
DBD_BASE="54910"                        # non-Docker legacy debuda port

while [[ $# -gt 0 ]]; do
  case "$1" in
    --worktree)  WORKTREE="$2";  shift 2 ;;
    --arch)      ARCH="$2";      shift 2 ;;
    --test)      TEST_FILE="$2"; shift 2 ;;
    --maxfail)   MAXFAIL="$2";   shift 2 ;;
    --k)         K_FILTER="$2";  shift 2 ;;
    --test-id)   TEST_ID="$2";   shift 2 ;;
    --port)      PORT="$2";      shift 2 ;;
    --jobs)      JOBS="$2";      shift 2 ;;
    --lock)      LOCKFILE="$2";  shift 2 ;;
    --sim-path)  SIM_PATH="$2";  shift 2 ;;
    --log-dir)   LOG_DIR="$2";   shift 2 ;;
    --result-json-out) RESULT_JSON_OUT="$2"; shift 2 ;;
    --stall)     STALL="$2";     shift 2 ;;
    --no-split)  NO_SPLIT="true"; shift ;;
    --verbose|-v) VERBOSE="true"; shift ;;
    # Deprecated no-ops: the watcher bounds the run, so there is no timeout or
    # poll budget. Accepted (and ignored) so pre-rewrite callers don't hard-fail.
    --timeout|--poll-budget) shift 2 ;;
    --help|-h)   sed -n 's/^# \{0,1\}//p' "$0" | head -70; exit 0 ;;
    *) echo "ERROR: unknown option: $1" >&2; echo "Run with --help for usage." >&2; exit 4 ;;
  esac
done

# ── Helpers ──────────────────────────────────────────────────────────────────

_vlog() { [[ "$VERBOSE" == "true" ]] && echo "[run_test] $*" >&2; return 0; }

_resolve_nng_channel() {
  local callback_host dbd_port

  NNG_LOCAL="${NNG_SOCKET_LOCAL_PORT:-$NNG_LOCAL_BASE}"
  if [[ -n "${NNG_SOCKET_ADDR:-}" ]]; then
    NNG_ADDR="$NNG_SOCKET_ADDR"
    return 0
  fi

  if [[ -f /.dockerenv ]]; then
    dbd_port="${P_USER_DBD_PORT:-}"
    if [[ -z "$dbd_port" ]]; then
      dbd_port="$(bash -lc 'printf "%s" "${P_USER_DBD_PORT:-}"' 2>/dev/null)"
    fi
    if [[ ! "$dbd_port" =~ ^[0-9]+$ ]]; then
      echo "ERROR: NNG_SOCKET_ADDR is unset and IRD did not provide a valid P_USER_DBD_PORT" >&2
      return 3
    fi

    callback_host="$(hostname)"
    callback_host="${callback_host%%-special-*}"
    NNG_ADDR="tcp://${callback_host}:${dbd_port}"
    return 0
  fi

  # Non-container legacy flow: the host is directly reachable, so retain the
  # historical fixed debuda port unless the caller supplied an explicit address.
  NNG_ADDR="tcp://$(hostname):${DBD_BASE}"
}

# Activate the venv only if it exists (external setup); else use the ambient
# python (tt-metal Docker image, deps installed system-wide).
# shellcheck disable=SC1091
_activate_venv() { [[ -f "${VENV}/bin/activate" ]] && source "${VENV}/bin/activate"; return 0; }

# Validate args and derive VENV, TEST_DIR, SIM_PATH, LOCKFILE, MODE, STALL, the
# NNG channel, and the build-stamp identity. Exits on error.
_validate() {
  local errors=0
  [[ -z "$WORKTREE"  ]] && { echo "ERROR: --worktree is required" >&2; ((errors++)); }
  [[ -z "$ARCH"      ]] && { echo "ERROR: --arch is required"     >&2; ((errors++)); }
  [[ -z "$TEST_FILE" ]] && { echo "ERROR: --test is required"     >&2; ((errors++)); }
  [[ $errors -gt 0 ]] && exit 4

  VENV="${WORKTREE}/tests/.venv"
  case "$ARCH" in
    blackhole|wormhole) TEST_DIR="${WORKTREE}/tests/python_tests" ;;
    *)                  TEST_DIR="${WORKTREE}/tests/python_tests/${ARCH}" ;;
  esac
  [[ -d "$TEST_DIR" ]] || { echo "ERROR: test directory not found: ${TEST_DIR}" >&2; exit 3; }
  [[ -f "${TEST_DIR}/${TEST_FILE}" ]] || { echo "ERROR: test file not found: ${TEST_DIR}/${TEST_FILE}" >&2; exit 3; }

  case "$ARCH" in
    blackhole|wormhole) MODE="hardware"  ;;
    *)                  MODE="simulator" ;;
  esac

  if [[ -z "$SIM_PATH" ]]; then
    if [[ "$ARCH" == "quasar" ]]; then
      case "$QSR_SIM_BACKEND" in
        emu|emulator)
          QSR_SIM_BACKEND="emu"
          SIM_PATH="${QSR_EMU_SIM_PATH:-/proj_sw/user_dev/${USER}/tt-umd-simulators/build/emu-quasar-1x3}"
          ;;
        vcs)
          SIM_PATH="${QSR_VCS_SIM_PATH:-/proj_sw/user_dev/${USER}/tt-umd-simulators/build/vcs-quasar-1x3}"
          ;;
        *)
          echo "ERROR: QSR_SIM_BACKEND must be emu or vcs, got '$QSR_SIM_BACKEND'" >&2
          exit 3
          ;;
      esac
    else
      SIM_PATH="/proj_sw/user_dev/${USER}/tt-umd-simulators/build/emu-${ARCH}-1x3"
    fi
  fi

  # Quasar's Aether resource is remote and shared by both compute runners. Use
  # QSR_AETHER_LOCK (a shared-filesystem path) when configured so tensix-l-04
  # and tensix-l-05 cannot start or reap each other's runs. Other paths retain
  # the historical node-local lock.
  if [[ -z "$LOCKFILE" ]]; then
    if [[ "$ARCH" == "quasar" && -n "${QSR_AETHER_LOCK:-}" ]]; then
      LOCKFILE="$QSR_AETHER_LOCK"
    else
      LOCKFILE="/tmp/tt-llk-test.lock"
    fi
  fi

  # Hang threshold: emulator gets the post-ready stall; silicon keeps the larger
  # default. Explicit --stall / HANG_STALL wins.
  if [[ -z "$STALL" ]]; then
    if [[ -n "${HANG_STALL:-}" ]]; then STALL="$HANG_STALL"
    elif [[ "$MODE" == "simulator" ]]; then STALL=180
    else STALL=300; fi
  fi

  # --collect-only / count emit node-ids relative to the pytest rootdir, so a
  # quasar id carries a leading "quasar/". We cd into the arch subdir before
  # running, so strip that prefix or pytest collects 0 items.
  [[ -n "$TEST_ID" ]] && TEST_ID="${TEST_ID#${ARCH}/}"

  # Codegen infra (symlinked into each worktree).
  REAP="${WORKTREE}/codegen/scripts/reap_stale_emu.sh"
  TRIAGE="${WORKTREE}/.claude/scripts/llk_triage.py"
  RUN_TAG="ttllk_${ARCH}_$$"

  if [[ "$MODE" == "simulator" ]]; then
    _resolve_nng_channel || exit $?
    _vlog "NNG callback ${NNG_ADDR} -> local port ${NNG_LOCAL}"
  fi
}

# SFPI (the RISC-V toolchain) is mandatory to compile. Fetch it if absent.
_ensure_sfpi() {
  [[ -d "${WORKTREE}/tests/sfpi/compiler/bin" ]] && return 0
  echo "[run_test] SFPI missing — fetching (CHIP_ARCH=${ARCH} ./setup_testing_env.sh)" >&2
  ( cd "${WORKTREE}/tests" && CHIP_ARCH="$ARCH" ./setup_testing_env.sh ) >&2 2>&1
  [[ -d "${WORKTREE}/tests/sfpi/compiler/bin" ]] || { echo "[run_test] SFPI still missing after setup" >&2; return 3; }
  return 0
}

# Hash every tracked or candidate-created, nonignored tt-llk file by content.
# Paths, kinds, executable bits, sizes, and hashes are framed with NUL bytes so
# names cannot alias one another. This deliberately matches the dashboard's
# source policy instead of relying on size/mtime fingerprints.
_source_tree_sha256() {
  local -a files=()
  local relative path kind executable size digest target
  git -C "$WORKTREE" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
    echo "ERROR: worktree is not a git checkout: ${WORKTREE}" >&2
    return 3
  }
  mapfile -d '' files < <(
    git -C "$WORKTREE" ls-files -z --cached --others --exclude-standard -- .
  )
  [[ ${#files[@]} -gt 0 ]] || {
    echo "ERROR: worktree contains no tracked or candidate-created files: ${WORKTREE}" >&2
    return 3
  }
  (
    set -o pipefail
    {
      printf '%s\0' "tt-llk-git-files-v1"
      for relative in "${files[@]}"; do
        path="${WORKTREE}/${relative}"
        if [[ -L "$path" ]]; then
          kind="symlink"
          executable="false"
          target="$(readlink "$path")"
          size="$(printf '%s' "$target" | wc -c)"
          digest="$(printf '%s' "$target" | sha256sum | cut -d' ' -f1)"
        elif [[ -f "$path" ]]; then
          kind="file"
          [[ -x "$path" ]] && executable="true" || executable="false"
          size="$(stat -c %s "$path")"
          digest="$(sha256sum "$path" | cut -d' ' -f1)"
        else
          echo "ERROR: unsupported source-tree entry: ${relative}" >&2
          return 3
        fi
        printf '%s\0%s\0%s\0%s\0%s\0' \
          "$relative" "$kind" "$executable" "$size" "$digest"
      done
    } | sha256sum | cut -d' ' -f1
  )
}

# Hash the candidate delta separately from the complete post-patch source tree.
# The tracked diff and every untracked, nonignored file are NUL-framed so a
# local run has the same base/patch distinction as a dashboard-dispatched run.
_patch_sha256() {
  local relative path size digest tracked_digest
  (
    set -o pipefail
    {
      printf '%s\0' "tt-llk-local-patch-v1"
      tracked_digest="$(git -C "$WORKTREE" diff --binary --full-index HEAD -- . | sha256sum | cut -d' ' -f1)"
      printf '%s\0' "$tracked_digest"
      while IFS= read -r -d '' relative; do
        path="${WORKTREE}/${relative}"
        [[ -f "$path" && ! -L "$path" ]] || {
          echo "ERROR: unsupported untracked patch entry: ${relative}" >&2
          return 3
        }
        size="$(stat -c %s "$path")"
        digest="$(sha256sum "$path" | cut -d' ' -f1)"
        printf '%s\0%s\0%s\0' "$relative" "$size" "$digest"
      done < <(git -C "$WORKTREE" ls-files -z --others --exclude-standard -- .)
    } | sha256sum | cut -d' ' -f1
  )
}

# Derive the artifact path only after SFPI setup, so the selected compiler's
# bytes are part of the identity. `count` uses a separate purpose namespace and
# therefore cannot clear a compiled entry while collection initializes pytest.
_prepare_artifact_identity() {
  local purpose="${1:-build}"
  local compiler owner_scope selector
  compiler="${WORKTREE}/tests/sfpi/compiler/bin/riscv-tt-elf-g++"
  SOURCE_TREE_SHA256="$(_source_tree_sha256)" || return $?
  ACTUAL_BASE_SHA="$(git -C "$WORKTREE" rev-parse HEAD 2>/dev/null)" || return 3
  EXPECTED_BASE_SHA="${CODEGEN_BASE_COMMIT:-$ACTUAL_BASE_SHA}"
  [[ "$ACTUAL_BASE_SHA" =~ ^[0-9a-f]{40}$ && "$EXPECTED_BASE_SHA" =~ ^[0-9a-f]{40}$ ]] || {
    echo "ERROR: local verification requires exact base SHAs" >&2
    return 3
  }
  [[ "$ACTUAL_BASE_SHA" == "$EXPECTED_BASE_SHA" ]] || {
    echo "ERROR: checked-out base ${ACTUAL_BASE_SHA} does not match expected ${EXPECTED_BASE_SHA}" >&2
    return 3
  }
  if [[ "${CODEGEN_PATCH_SHA256:-}" =~ ^[0-9a-f]{64}$ ]]; then
    PATCH_SHA256="$CODEGEN_PATCH_SHA256"
  else
    PATCH_SHA256="$(_patch_sha256)" || return $?
  fi
  if [[ -f "$compiler" ]]; then
    COMPILER_SHA256="$(sha256sum "$compiler" | cut -d' ' -f1)"
  elif [[ "$purpose" == "count" ]]; then
    COMPILER_SHA256="unavailable-for-collection"
  else
    echo "ERROR: compiler is unavailable after SFPI setup: ${compiler}" >&2
    return 3
  fi

  selector="${ARCH}|${TEST_FILE}|${TEST_ID}|${K_FILTER}|${NO_SPLIT}"
  ARTKEY="$(printf '%s' "$selector" | sha256sum | cut -d' ' -f1)"
  BUILD_INPUT_DIGEST="$({
    printf '%s\0' "tt-llk-local-build-input-v2"
    printf '%s\0' "$purpose" "$SOURCE_TREE_SHA256" "$COMPILER_SHA256"
    printf '%s\0' "$ARCH" "$TEST_FILE" "$TEST_ID" "$K_FILTER" "$NO_SPLIT"
  } | sha256sum | cut -d' ' -f1)"

  owner_scope="${CODEGEN_RUN_ID:-${RUN_ID:-manual}}|${CODEGEN_ATTEMPT_ID:-manual}|${LOG_DIR:-no-log}|$(realpath -m "$WORKTREE")"
  ARTIFACT_OWNER="$(printf '%s' "$owner_scope" | sha256sum | cut -d' ' -f1)"
  MANAGED_ARTIFACT_ROOT="$(realpath -m "${TT_LLK_LOCAL_ARTIFACT_ROOT:-/tmp/tt-llk-build-v2}")"
  local worktree_root
  worktree_root="$(realpath -m "$WORKTREE")"
  case "$MANAGED_ARTIFACT_ROOT" in
    /|"$HOME"|"$worktree_root"|"$worktree_root"/*)
      echo "ERROR: unsafe managed artifact root: ${MANAGED_ARTIFACT_ROOT}" >&2
      return 3
      ;;
  esac
  ARTIFACT_DIR="${MANAGED_ARTIFACT_ROOT}/v2/${ARTIFACT_OWNER}/${BUILD_INPUT_DIGEST}"
  ARTIFACT_LOCK="${MANAGED_ARTIFACT_ROOT}/locks/${ARTIFACT_OWNER}-${BUILD_INPUT_DIGEST}.lock"
  STAMP_DIR="${MANAGED_ARTIFACT_ROOT}/stamps/${ARTIFACT_OWNER}"
  STAMP="${STAMP_DIR}/${ARTKEY}"
  mkdir -p "$(dirname "$ARTIFACT_DIR")" "$(dirname "$ARTIFACT_LOCK")" "$STAMP_DIR" || {
    echo "ERROR: cannot create managed artifact namespace: ${MANAGED_ARTIFACT_ROOT}" >&2
    return 3
  }
  EVIDENCE_DIR="${MANAGED_ARTIFACT_ROOT}/evidence/${ARTIFACT_OWNER}/${BUILD_INPUT_DIGEST}/${CMD}-$$"
  COLLECTION_JSON="${EVIDENCE_DIR}/collection.json"
  PRODUCER_JUNIT="${EVIDENCE_DIR}/producer.junit.xml"
  CONSUMER_JUNIT="${EVIDENCE_DIR}/consumer.junit.xml"
  CONSUMER_LOG="${EVIDENCE_DIR}/consumer.log"
  ARTIFACT_MANIFEST="${EVIDENCE_DIR}/artifact-manifest.json"
  mkdir -p "$EVIDENCE_DIR" || {
    echo "ERROR: cannot create evidence directory: ${EVIDENCE_DIR}" >&2
    return 3
  }
  if [[ -z "$RESULT_JSON_OUT" ]]; then
    if [[ -n "$LOG_DIR" ]]; then
      RESULT_JSON_OUT="${LOG_DIR}/verification-results/${CMD}-${ARCH}-${BUILD_INPUT_DIGEST:0:16}-$$.json"
    else
      RESULT_JSON_OUT="${EVIDENCE_DIR}/verification-result.json"
    fi
  fi
  RUN_IDENTITY="${CODEGEN_RUN_ID:-${RUN_ID:-manual-${ARTIFACT_OWNER:0:16}}}"
  ATTEMPT_IDENTITY="${CODEGEN_ATTEMPT_ID:-manual}"
  JOB_IDENTITY="${CODEGEN_JOB_ID:-local-${ARTIFACT_OWNER:0:12}-$$}"
  REQUIREMENT_IDENTITY="${CODEGEN_REQUIREMENT_ID:-${ARCH}:llk:1}"
  RUN_JSON_WRITER="${WORKTREE}/codegen/scripts/run_json_writer.py"
  [[ -f "$RUN_JSON_WRITER" ]] || {
    echo "ERROR: result writer is missing: ${RUN_JSON_WRITER}" >&2
    return 3
  }
  export TT_LLK_ARTEFACTS_DIR="$ARTIFACT_DIR"
  SRC_ID="${SOURCE_TREE_SHA256:0:16}"
  _vlog "artifact owner=${ARTIFACT_OWNER} build=${BUILD_INPUT_DIGEST} root=${ARTIFACT_DIR}"
}

_lock_artifact_entry() {
  exec 8>>"$ARTIFACT_LOCK" || {
    echo "ERROR: cannot open artifact lock ${ARTIFACT_LOCK}" >&2
    return 3
  }
  _vlog "waiting for artifact lock ${ARTIFACT_LOCK}"
  flock 8 || {
    echo "ERROR: cannot acquire artifact lock ${ARTIFACT_LOCK}" >&2
    return 3
  }
}

# The pytest target selector (file, -k filter, or a single parametrize id).
_build_target() {
  TARGET=()
  if   [[ -n "$TEST_ID"  ]]; then TARGET=("$TEST_ID")
  elif [[ -n "$K_FILTER" ]]; then TARGET=(-k "$K_FILTER" "$TEST_FILE")
  else                            TARGET=("$TEST_FILE"); fi
}

# Perform a pure collection pass. conftest explicitly skips device/runtime
# initialization when pytest is collecting only and atomically writes the
# selected, pre-filter collected, error, and process-exit counts.
_collect_tests() {
  local collection_log="${EVIDENCE_DIR}/collection.log"
  rm -f "$COLLECTION_JSON" "$collection_log"
  ( CHIP_ARCH="$ARCH" pytest --collect-only -q \
      --codegen-collection-json "$COLLECTION_JSON" "${TARGET[@]}" ) \
      >"$collection_log" 2>&1
  local rc=$?
  cat "$collection_log" >&2
  [[ -f "$COLLECTION_JSON" ]] || {
    echo "ERROR: pytest collection produced no structured result" >&2
    return 3
  }
  return "$rc"
}

_seal_artifacts() {
  python3 "$RUN_JSON_WRITER" artifact-manifest \
    --output "$ARTIFACT_MANIFEST" \
    --artifact-root "$ARTIFACT_DIR" \
    --owner-id "$ARTIFACT_OWNER" \
    --build-input-digest "$BUILD_INPUT_DIGEST" \
    --source-tree-sha256 "$SOURCE_TREE_SHA256" \
    --compiler-sha256 "$COMPILER_SHA256" >&2
}

_emit_structured_result() {
  local backend="${CODEGEN_VERIFICATION_BACKEND:-}"
  if [[ -z "$backend" ]]; then
    if [[ "$SIM_PATH" == *.so ]]; then backend="ttsim"
    elif [[ "$ARCH" == "quasar" ]]; then backend="quasar"
    elif [[ "$MODE" == "hardware" ]]; then backend="silicon"
    else backend="local"; fi
  fi
  local -a args=(
    verification-result
    --output "$RESULT_JSON_OUT"
    --collection-json "$COLLECTION_JSON"
    --junit "$CONSUMER_JUNIT"
    --output-log "$CONSUMER_LOG"
    --artifact-manifest "$ARTIFACT_MANIFEST"
    --artifact-root "$ARTIFACT_DIR"
    --requirement-id "$REQUIREMENT_IDENTITY"
    --run-id "$RUN_IDENTITY"
    --attempt-id "$ATTEMPT_IDENTITY"
    --job-id "$JOB_IDENTITY"
    --architecture "$ARCH"
    --suite llk
    --backend "$backend"
    --test "$TEST_FILE"
    --expected-base-sha "$EXPECTED_BASE_SHA"
    --actual-base-sha "$ACTUAL_BASE_SHA"
    --patch-sha256 "$PATCH_SHA256"
    --returncode "${CONSUMER_RETURN_CODE:-$_rc}"
  )
  [[ -n "$TEST_ID" ]] && args+=(--test-id "$TEST_ID")
  [[ -n "$K_FILTER" ]] && args+=(--k "$K_FILTER")
  [[ "${CONSUMER_TIMED_OUT:-false}" == "true" ]] && args+=(--timed-out)
  [[ -n "${CONSUMER_SIGNAL:-}" ]] && args+=(--signal "$CONSUMER_SIGNAL")
  [[ -n "${CONSUMER_INFRA_CODE:-}" ]] && args+=(--infrastructure-code "$CONSUMER_INFRA_CODE")
  [[ "$NO_SPLIT" == "true" ]] && args+=(--infrastructure-code artifact_not_presealed)

  local writer_rc=0
  python3 "$RUN_JSON_WRITER" "${args[@]}" >&2 || writer_rc=$?
  if [[ ! -s "$RESULT_JSON_OUT" ]]; then
    echo "ERROR: structured verification result was not written: ${RESULT_JSON_OUT}" >&2
    return 3
  fi
  _vlog "structured result ${RESULT_JSON_OUT}"
  return "$writer_rc"
}

_emit_verdict() {
  local code="$1" phase="$2" v
  case "$code" in
    0) v=PASS ;; 1) v=FAIL ;; 2) v=COMPILE_FAIL ;; 3) v=ENV_ERROR ;; 4) v=BAD_ARGS ;; 5) v=HANG ;; *) v="EXIT_${code}" ;;
  esac
  echo "=== RUN_LLK_TESTS_VERDICT === ${v} (exit ${code}, phase=${phase}, test=${TEST_FILE:-?}, arch=${ARCH:-?})" >&2
}

# ── Producer (compile) ─────────────────────────────────────────────────────────

# Parallel compile with the transient parallel-build-setup retry. The producer's
# xdist workers share only this attempt-owned entry; the entry flock prevents a
# second producer from clearing it concurrently.
# Retry on that signature; treat anything else as a genuine compile failure.
# Returns pytest's exit code.
_producer() {
  local plog; plog="$(mktemp "${TMPDIR:-/tmp}/tt-llk-prod.XXXXXX")"
  local prc=1 attempt
  for attempt in 1 2 3; do
    : > "$plog"
    ( CHIP_ARCH="$ARCH" pytest --compile-producer -n "$JOBS" -x \
        --junitxml "$PRODUCER_JUNIT" "${TARGET[@]}" ) >"$plog" 2>&1
    prc=$?
    [[ -n "$LOG_DIR" ]] && { mkdir -p "$LOG_DIR"; cat "$plog" >> "${LOG_DIR}/compile.log"; }
    cat "$plog" >&2
    [[ $prc -eq 0 ]] && break
    if grep -q "create_build_directories" "$plog" && grep -q "FileNotFoundError" "$plog"; then
      echo "[run_test] transient parallel-build-setup race (attempt ${attempt}/3); retrying" >&2
      sleep 3; continue
    fi
    break   # genuine compile error — do not retry
  done
  rm -f "$plog"
  return "$prc"
}

# ── Silicon (BH/WH) hang cleanup ───────────────────────────────────────────────

# Free the device handle, dump LLK triage while the Tensix is still wedged, then
# reset. Triage is skipped if the script is absent.
_hw_hang_cleanup() {
  pkill -9 -f "pytest.*--compile-consumer" 2>/dev/null || true
  if [[ -f "$TRIAGE" ]]; then
    echo "--- llk-triage ---" >&2
    timeout 60 python3 "$TRIAGE" --arch "$ARCH" >&2 2>&1 || true
    echo "--- end llk-triage ---" >&2
  fi
  command -v tt-smi >/dev/null 2>&1 && { echo "[run_test] tt-smi -r" >&2; tt-smi -r >&2 2>&1 || true; }
}

# ── Consumer (run on device/emulator) with hang watchdog ───────────────────────

# Run pytest in the background, watch its log for a stall, and on a hang send
# SIGINT so conftest's handler tears tt-exalens down gracefully (releasing the
# remote emulator), escalating to SIGKILL if pytest ignores it. Sets/clears the
# global CONSUMER_PID (the EXIT trap uses it). Returns the classified exit code.
_run_consumer() {
  local log="$CONSUMER_LOG"
  local hangflag="${log}.hang"
  rm -f "$log" "$hangflag" "$CONSUMER_JUNIT"
  : > "$log"
  CONSUMER_RETURN_CODE="" CONSUMER_TIMED_OUT="false"
  CONSUMER_SIGNAL="" CONSUMER_INFRA_CODE=""

  local -a flags=(-rN "--maxfail=${MAXFAIL}" --junitxml "$CONSUMER_JUNIT")
  [[ "$MODE" == "simulator" ]] && flags+=(--run-simulator "--port=${PORT}")
  [[ "$NO_SPLIT" == "false" ]] && flags+=(--compile-consumer)

  if [[ "$MODE" == "simulator" ]]; then
    export NNG_SOCKET_ADDR="$NNG_ADDR" NNG_SOCKET_LOCAL_PORT="$NNG_LOCAL" NNG_SOCKET_NAME="$RUN_TAG"
    export TT_UMD_SIMULATOR_PATH="$SIM_PATH"
    export SSH_MACHINE_NAME="$EMU_HOST"
    # Free this run's port before booting tt-exalens.
    local stale; stale="$(lsof -ti :"$PORT" 2>/dev/null || true)"
    [[ -n "$stale" ]] && echo "$stale" | xargs -r kill -9 2>/dev/null || true
    pkill -9 -f "tt-exalens.*--port=${PORT}" 2>/dev/null || true
    sleep 1
  fi

  ( CHIP_ARCH="$ARCH" pytest "${flags[@]}" "${TARGET[@]}" ) >>"$log" 2>&1 &
  CONSUMER_PID=$!

  # Watchdog: a healthy run keeps emitting progress lines in EVERY phase — during
  # boot the server prints "still waiting" every 10s, then per-variant results — so
  # a log that stops advancing for STALL seconds means it wedged, no matter the
  # phase. Armed from the start (boot wedges included); the readiness marker is used
  # only to CLASSIFY the outcome afterwards (pre-ready stall = ENV), not to arm.
  (
    while kill -0 "$CONSUMER_PID" 2>/dev/null; do
      sleep "$WATCH_INTERVAL"
      local now mtime; now="$(date +%s)"; mtime="$(stat -c %Y "$log" 2>/dev/null || echo "$now")"
      if [[ $((now - mtime)) -ge "$STALL" ]]; then
        : > "$hangflag"
        # Graceful: conftest turns SIGINT/SIGTERM into KeyboardInterrupt →
        # pytest_sessionfinish/atexit → ExalensServer.stop() sends `exit`.
        kill -INT "$CONSUMER_PID" 2>/dev/null
        local waited=0
        while kill -0 "$CONSUMER_PID" 2>/dev/null && [[ $waited -lt $GRACE_SECS ]]; do
          sleep 1; waited=$((waited + 1))
        done
        # Ignored the signal (wedged in a C call) → hard-kill the tree.
        if kill -0 "$CONSUMER_PID" 2>/dev/null; then
          for p in $(pgrep -P "$CONSUMER_PID" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
          kill -9 "$CONSUMER_PID" 2>/dev/null
        fi
        break
      fi
    done
  ) &
  local watch_pid=$!

  wait "$CONSUMER_PID"; local rc=$?
  CONSUMER_RETURN_CODE="$rc"
  if [[ $rc -ge 128 && $rc -le 255 ]]; then
    CONSUMER_SIGNAL="$((rc - 128))"
  fi
  kill "$watch_pid" 2>/dev/null; wait "$watch_pid" 2>/dev/null

  [[ -n "$LOG_DIR" ]] && { mkdir -p "$LOG_DIR"; cat "$log" >> "${LOG_DIR}/run.log" 2>/dev/null; }
  tail -80 "$log" >&2

  # Classify. Order matters: never-ready (infra) is checked before FAIL because a
  # kernel cannot run before the emulator is up.
  local code
  if [[ -f "$hangflag" ]] && [[ "$MODE" == "simulator" ]] && ! grep -qE "$READY_RE" "$log" 2>/dev/null; then
    # Stalled before tt-exalens ever reported ready → a boot wedge, not a kernel
    # hang. Transient (emulator congestion) → ENV so the caller may retry.
    echo "[run_test] ENV: stalled before tt-exalens became ready (boot wedge)" >&2
    [[ -x "$REAP" ]] && bash "$REAP" --arch "$ARCH" --emu-host "$EMU_HOST" --lock "$LOCKFILE" --force >&2 2>&1 || true
    CONSUMER_TIMED_OUT="true" CONSUMER_INFRA_CODE="emulator_boot_stalled"
    code=3
  elif [[ -f "$hangflag" ]]; then
    echo "[run_test] HANG: no output for ${STALL}s" >&2
    if [[ "$MODE" == "simulator" ]]; then
      [[ -x "$REAP" ]] && bash "$REAP" --arch "$ARCH" --emu-host "$EMU_HOST" --lock "$LOCKFILE" --force >&2 2>&1 || true
    else
      _hw_hang_cleanup
    fi
    CONSUMER_TIMED_OUT="true" CONSUMER_INFRA_CODE="execution_stalled"
    code=5
  elif [[ "$MODE" == "simulator" && $rc -ne 0 ]] && ! grep -qE "$READY_RE" "$log" 2>/dev/null; then
    echo "[run_test] ENV: tt-exalens never became ready" >&2
    [[ -x "$REAP" ]] && bash "$REAP" --arch "$ARCH" --emu-host "$EMU_HOST" --lock "$LOCKFILE" --force >&2 2>&1 || true
    CONSUMER_INFRA_CODE="emulator_not_ready"
    code=3
  elif [[ "$MODE" == "hardware" && $rc -ne 0 ]] && grep -qF "TENSIX TIMED OUT" "$log" 2>/dev/null; then
    echo "[run_test] HANG: TENSIX TIMED OUT" >&2
    _hw_hang_cleanup
    CONSUMER_TIMED_OUT="true" CONSUMER_INFRA_CODE="tensix_timed_out"
    code=5
  elif [[ $rc -ne 0 ]] && grep -qiE "No Tenstorrent devices? (were|was)? ?detected|No Tenstorrent devices" "$log" 2>/dev/null; then
    echo "[run_test] ENV: no Tenstorrent device detected (CHIP_ARCH / device access)" >&2
    CONSUMER_INFRA_CODE="no_device_available"
    code=3
  elif [[ $rc -eq 0 ]]; then
    code=0
  else
    code=1
  fi

  CONSUMER_PID=""
  rm -f "$hangflag" 2>/dev/null
  return "$code"
}

# ── count / compile (per-entry artifact lock) ─────────────────────────────────

_do_count() {
  _validate; _activate_venv
  _prepare_artifact_identity count || { echo "0"; return 3; }
  _lock_artifact_entry || { echo "0"; return 3; }
  _build_target
  cd "$TEST_DIR" || { echo "0"; return 3; }
  local rc=0
  _collect_tests || rc=$?
  [[ $rc -ne 0 ]] && { echo "0"; return "$rc"; }
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"])' \
    "$COLLECTION_JSON" || { echo "0"; return 3; }
}

_do_compile() {
  _validate; _activate_venv; _ensure_sfpi || return 3
  _prepare_artifact_identity build || return 3
  _lock_artifact_entry || return 3
  _build_target; cd "$TEST_DIR" || return 3
  local collection_rc=0
  _collect_tests || collection_rc=$?
  [[ $collection_rc -eq 4 ]] && return 4
  [[ $collection_rc -ne 0 ]] && return 2
  _vlog "compile ${TEST_FILE} (arch=${ARCH}, -n ${JOBS})"
  _producer; local rc=$?
  if [[ $rc -eq 0 ]]; then
    _seal_artifacts || return 3
    printf '%s' "$BUILD_INPUT_DIGEST" > "$STAMP"
    return 0
  fi
  return 2
}

# ── simulate / run (under the global lock) ─────────────────────────────────────

# Acquire the global lock, rebuild under it when the stamp is not ours (or forced),
# then run the consumer. The producer and consumer run back-to-back without
# releasing the lock, so the ELFs consumed are exactly the ones just produced.
_run_under_lock() {
  local force="$1"
  _validate; _activate_venv; _ensure_sfpi || return 3
  _prepare_artifact_identity build || return 3
  _build_target; cd "$TEST_DIR" || return 3

  mkdir -p "$(dirname "$LOCKFILE")" 2>/dev/null ||
    { echo "ERROR: cannot create lock directory for ${LOCKFILE}" >&2; return 3; }
  exec 9>>"$LOCKFILE" || { echo "ERROR: cannot open lock ${LOCKFILE}" >&2; return 3; }
  _vlog "waiting for global lock ${LOCKFILE}"
  flock 9                       # unbounded — wait in line
  _vlog "acquired lock"
  _lock_artifact_entry || return 3

  # Pre-flight reap under the lock: any live emu job now is an orphan from a run
  # whose peer died non-gracefully. Clear it before booting ours.
  if [[ "$MODE" == "simulator" && -x "$REAP" ]]; then
    bash "$REAP" --arch "$ARCH" --emu-host "$EMU_HOST" --lock "$LOCKFILE" --force >&2 2>&1 || true
  fi

  local collection_rc=0
  _collect_tests || collection_rc=$?
  [[ $collection_rc -eq 4 ]] && return 4
  [[ $collection_rc -ne 0 ]] && return 2

  # Build under the lock if forced or the stamp is not ours (a peer recompiled).
  # --no-split compiles inside the consumer, so it is skipped here.
  if [[ "$NO_SPLIT" == "false" ]]; then
    local need="$force"
    [[ "$(cat "$STAMP" 2>/dev/null)" != "$BUILD_INPUT_DIGEST" ]] && need=1
    if [[ "$need" == "1" ]]; then
      _vlog "building under lock (have=$(cat "$STAMP" 2>/dev/null) want=${BUILD_INPUT_DIGEST} force=${force})"
      _producer || return 2
      printf '%s' "$BUILD_INPUT_DIGEST" > "$STAMP"
    else
      _vlog "reusing build (stamp matches ${SRC_ID})"
    fi
    _seal_artifacts || return 3
  fi

  local execution_rc=0
  _run_consumer || execution_rc=$?
  if [[ "$NO_SPLIT" == "true" ]]; then
    _seal_artifacts || return 3
  fi
  return "$execution_rc"
  # The lock (fd 9) is released when the script exits.
}

# ── Cleanup trap ───────────────────────────────────────────────────────────────

CONSUMER_PID=""
# On any script exit — including a harness SIGTERM/SIGINT — if the consumer is
# still alive we are dying abnormally: tear it down gracefully so tt-exalens
# releases the remote emulator, escalate + reap if it ignores the signal. Normal
# completion clears CONSUMER_PID, so this no-ops.
_cleanup() {
  if [[ -n "${CONSUMER_PID:-}" ]] && kill -0 "$CONSUMER_PID" 2>/dev/null; then
    kill -INT "$CONSUMER_PID" 2>/dev/null
    local waited=0
    while kill -0 "$CONSUMER_PID" 2>/dev/null && [[ $waited -lt $GRACE_SECS ]]; do sleep 1; waited=$((waited + 1)); done
    kill -9 "$CONSUMER_PID" 2>/dev/null
    if [[ "${MODE:-}" == "simulator" && -x "${REAP:-}" ]]; then
      bash "$REAP" --arch "$ARCH" --emu-host "$EMU_HOST" --lock "$LOCKFILE" --force >/dev/null 2>&1 || true
    fi
  fi
}
trap _cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

# ── Dispatch ──────────────────────────────────────────────────────────────────

_rc=0
case "$CMD" in
  count)    _do_count       ; _rc=$? ;;
  compile)  _do_compile     ; _rc=$? ;;
  simulate) _run_under_lock 0 ; _rc=$? ;;
  run)      _run_under_lock 1 ; _rc=$? ;;
  help|--help|-h) sed -n 's/^# \{0,1\}//p' "$0" | head -70; exit 0 ;;
  "") echo "ERROR: no command. Use: count | compile | simulate | run" >&2; exit 4 ;;
  *)  echo "ERROR: unknown command '${CMD}'. Use: count | compile | simulate | run" >&2; exit 4 ;;
esac

# Once a consumer ran, the structured classification is authoritative. This can
# tighten an exit-0 process to coverage/infra failure but never turn a failing
# process into success.
if [[ "$CMD" == "simulate" || "$CMD" == "run" ]] && [[ -n "${CONSUMER_RETURN_CODE:-}" ]]; then
  _emit_structured_result
  _rc=$?
fi

# count's stdout contract is "just the integer" — no verdict line.
case "$CMD" in compile|simulate|run) _emit_verdict "$_rc" "$CMD" ;; esac
exit "$_rc"
