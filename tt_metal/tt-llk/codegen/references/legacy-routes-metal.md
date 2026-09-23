# Fallback execution routes — Metal tester

Reference for `metal-tester.md` (suite `metal`, `unit_tests_llk` gtest). Read this **only** when the sealed dispatch path in that
playbook does not apply, that is when any of these hold:

- `HW_TEST_DISPATCH_CMD` is unset, or its `--help` does not advertise
  `--requirement-id`;
- `TEST_BACKEND=ttsim`;
- the architecture is `quasar`, which never uses the silicon queue;
- local silicon execution without a dispatcher.

For Blackhole or Wormhole with a sealed dispatcher, the playbook's sealed path
is authoritative and this file is not needed. Every coverage, scope, manifest,
identity and verdict rule in the playbook still applies to these routes.

---

## Step A — Build `unit_tests_llk` locally

This is the early compile gate for every backend, including queued silicon. Do
not submit a hardware job when it fails. The queue intentionally rebuilds in an
isolated workspace; both paths must use the narrow target and warm caches below.

Require `dashboard.hw_test.builder` from the companion `llk_code_gen` checkout
to be importable by `python`. Preserve the `PYTHONPATH` supplied by the dashboard;
for standalone runs, add the `llk_code_gen` checkout root to `PYTHONPATH`.
Pick the strategy that matches what the environment provides.

Install one cleanup trap before either local strategy. It preserves the warm
tree rollback and removes only a cache directory created by this invocation:

```bash
FRESH_CACHE=
FRESH_CACHE_ARCH=
cleanup_fresh_cache() {
  local cache="${FRESH_CACHE:-}" root="${TTCACHE_ROOT:-}" cache_arch="${FRESH_CACHE_ARCH:-}"
  [ -z "$cache" ] && return 0
  case "$root" in
    /*) ;;
    *) echo "ENV_ERROR: TTCACHE_ROOT must be absolute"; return 1 ;;
  esac
  local expected_prefix="${root%/}/ttcache_${cache_arch}."
  case "$cache" in
    "$expected_prefix"*) rm -rf -- "$cache" ;;
    *) echo "ENV_ERROR: refusing to remove unexpected cache path: $cache"; return 1 ;;
  esac
  FRESH_CACHE=
  FRESH_CACHE_ARCH=
}
cleanup_metal_tester() {
  cleanup_fresh_cache || true
  if [ "${VERIFY_STRATEGY:-}" = warm ] && [ -n "${FIX_PATCH:-}" ]; then
    git -C "$METAL_VERIFY_HOME" apply -R "$FIX_PATCH" 2>/dev/null || true
  fi
}
trap cleanup_metal_tester EXIT
```

### Strategy 1: warm tree

Use this only when the warm tree is clean, the fix changes tracked files only,
and the worktree diff applies cleanly. Create `FIX_PATCH` from the worktree's
binary diff against `HEAD`. Otherwise use Strategy 2.

Run Strategy 1 and the execution step in the same Bash process. The cleanup
trap must stay active until verification finishes; exiting after the build
would reverse the patch before local JIT compilation.

```bash
set -euo pipefail
: "${METAL_VERIFY_HOME:?warm tree not provided}"
BUILD_DIR="${METAL_VERIFY_BUILD_DIR:-$METAL_VERIFY_HOME/build}"
BIN="$BUILD_DIR/test/tt_metal/unit_tests_llk"
VERIFY_STRATEGY=warm
FIX_PATCH="$LOG_DIR/metal_fix.patch"

if git -C "$WORKTREE_DIR" status --porcelain | rg -q '^\?\?'; then
  echo "Warm strategy cannot carry untracked fix files; use Strategy 2."
  exit 3
fi
git -C "$WORKTREE_DIR" diff --binary HEAD > "$FIX_PATCH"
[ -s "$FIX_PATCH" ] || { echo "ENV_ERROR: fix patch is empty"; exit 3; }

git -C "$METAL_VERIFY_HOME" status --porcelain | rg -q . &&
  { echo "ENV_ERROR: verification tree is dirty"; exit 3; }
git -C "$METAL_VERIFY_HOME" apply --check "$FIX_PATCH" ||
  { echo "ENV_ERROR: fix does not apply to the verification tree base"; exit 3; }
git -C "$METAL_VERIFY_HOME" apply "$FIX_PATCH"
cd "$METAL_VERIFY_HOME"
python -m dashboard.hw_test.builder --prepare-workspace "$METAL_VERIFY_HOME" --kind metal \
  2>&1 | tee -a "$LOG_DIR/metal_build.log" \
  || { echo "ENV_ERROR: workspace preparation failed"; exit 3; }

# Incremental build. Fast/no-op for a pure Compute-API (JIT-side) header change; a real
# rebuild only when host-compiled metal code changed. Build failure => COMPILE_FAILED.
if [ ! -f "$BUILD_DIR/CMakeCache.txt" ] ||
   ! rg -q '^ENABLE_CCACHE:BOOL=(1|ON|TRUE|YES)$' "$BUILD_DIR/CMakeCache.txt" ||
   ! rg -q '^TT_METAL_BUILD_TESTS:BOOL=(1|ON|TRUE|YES)$' "$BUILD_DIR/CMakeCache.txt"; then
  ./build_metal.sh --enable-ccache --build-metal-tests \
    --build-dir "$BUILD_DIR" --configure-only 2>&1 \
    | tee -a "$LOG_DIR/metal_build.log" \
    || { echo "COMPILE_FAILED"; exit 2; }
fi
if ! cmake --build "$BUILD_DIR" --target unit_tests_llk 2>&1 | tee -a "$LOG_DIR/metal_build.log"; then
  echo "COMPILE_FAILED"; exit 2
fi
```

### Strategy 2: build the issue worktree

Use when no suitable warm tree exists or the fix adds files:

```bash
set -euo pipefail
cd "$WORKTREE_DIR"
python -m dashboard.hw_test.builder --prepare-workspace "$WORKTREE_DIR" --kind metal \
  2>&1 | tee -a "$LOG_DIR/metal_build.log" \
  || { echo "ENV_ERROR: workspace preparation failed"; exit 3; }
CACHE_USER="${USER:-$(id -un)}"
export CCACHE_DIR="${CCACHE_DIR:-/localdev/$CACHE_USER/ccache}"
export CCACHE_BASEDIR="$WORKTREE_DIR"
mkdir -p "$CCACHE_DIR" || { echo "ENV_ERROR: cannot create $CCACHE_DIR"; exit 3; }
BUILD_DIR="$WORKTREE_DIR/build"
if [ ! -f "$BUILD_DIR/CMakeCache.txt" ] ||
   ! rg -q '^ENABLE_CCACHE:BOOL=(1|ON|TRUE|YES)$' "$BUILD_DIR/CMakeCache.txt" ||
   ! rg -q '^TT_METAL_BUILD_TESTS:BOOL=(1|ON|TRUE|YES)$' "$BUILD_DIR/CMakeCache.txt"; then
  ./build_metal.sh --enable-ccache --build-metal-tests \
    --build-dir "$BUILD_DIR" --configure-only 2>&1 \
    | tee -a "$LOG_DIR/metal_build.log" \
    || { echo "COMPILE_FAILED"; exit 2; }
fi
cmake --build "$BUILD_DIR" --target unit_tests_llk 2>&1 \
  | tee -a "$LOG_DIR/metal_build.log" \
  || { echo "COMPILE_FAILED"; exit 2; }
BIN="$BUILD_DIR/test/tt_metal/unit_tests_llk"
VERIFY_STRATEGY=worktree
```

Build only the `unit_tests_llk` target — a plain `--build-metal-tests` builds the
whole metal test suite (~1750 targets). `CCACHE_BASEDIR` is not a storage path; it
strips the per-run worktree prefix so a rebuild can match a previous run's cache.
On retries in the same worktree, skip explicit configuration only when the
CMake cache exists with `ENABLE_CCACHE` and `TT_METAL_BUILD_TESTS` enabled;
`cmake --build` regenerates the graph automatically when CMake inputs changed.

Report the strategy and local build wall-time in the self-log. After queued
dispatch, also derive the isolated producer duration from `build_started_at` to
`built_at` in the returned job when both timestamps exist.

Before requesting hardware, confirm that the locally built binary exists and
that the filter selects at least one test:

```bash
[ -x "$BIN" ] || { echo "ENV_ERROR: missing $BIN"; exit 3; }
listed_tests="$("$BIN" --gtest_list_tests --gtest_filter="$METAL_FILTER" 2>&1)"
if ! printf '%s\n' "$listed_tests" |
    rg -q '^[[:space:]]+[^[:space:]]'; then
  echo "MISSING_TEST_COVERAGE: gtest filter selected zero tests: $METAL_FILTER"
  exit 1
fi
```

## Step B — Execute on queued silicon

Use this route only for Blackhole/Wormhole with `TEST_BACKEND=local` and
`HW_TEST_DISPATCH_CMD`, after the local compile gate passes. The queue rebuilds
in its own warm workspace, then its card executor consumes that artifact. Quasar
is excluded even when the dispatch command is present. Dispatch captures
tracked, modified, deleted, and untracked worktree files in the submitted binary
diff. The queue uses the same narrow `unit_tests_llk` target, a normalized
node-local ccache, and skips explicit CMake configuration when its session
workspace already has a valid generation.

```bash
for arch in "${ARCHES[@]}"; do
  [ "$arch" = quasar ] && continue
  # Resolve this architecture's one sealed metal leaf before dispatch and set
  # CODEGEN_RUN_ID/CODEGEN_ATTEMPT_ID/CODEGEN_REQUIREMENT_ID from it.
  mkdir -p "$LOG_DIR/verification-results/${CODEGEN_ATTEMPT_ID}"
  RESULT_JSON_OUT="$LOG_DIR/verification-results/${CODEGEN_ATTEMPT_ID}/${CODEGEN_REQUIREMENT_ID}.json"
  result_args=()
  if [ "${CODEGEN_RUNNER_POOL:-prod}" = audit ]; then
    result_args+=(--result-json-out "$RESULT_JSON_OUT")
  fi
  set +e
  $HW_TEST_DISPATCH_CMD --kind metal --arch "$arch" \
    --test "$METAL_FILTER" --dispatch "${METAL_DISPATCH:-fast}" \
    --worktree "$WORKTREE_DIR" \
    --base "$(sg GIT_COMMIT)" \
    --session "${HW_TEST_SESSION:-issue-${ISSUE_NUMBER}}-${arch}" \
    "${result_args[@]}" \
    --timeout "${TIMEOUT:-1800}" 2>&1 | tee -a "$LOG_DIR/metal_run.log"
  dispatch_exit=${PIPESTATUS[0]}
  set -e
  # Record this architecture's marker/result before dispatching the next leaf.
done
```

Require exactly one final `HW_TEST_RESULT arch=<arch>` marker for each queued
Blackhole/Wormhole invocation and record its `job` value. For an audit run,
also require the exact protocol-v2 result at `RESULT_JSON_OUT`, validate its
sealed identity, and derive the suite verdict and counts from its
`classification`, `collection`, and `execution` records exactly as in
`tester.md`. The strict reducer is authoritative; the marker and dispatch exit
are supporting evidence only.

A marker with `failure_stage=build` is `ENV_ERROR`: the local compile already
passed, so a queue rebuild failure means the isolated runner could not reproduce
that build. No structured execution result exists because the job correctly
never reached a card.

For production compatibility, do not request a protocol-v2 result copy and use
the legacy marker:

| Marker | Verdict |
|---|---|
| `ok=true ran=true passed=true` | `SUCCESS` |
| `ok=false ran=true` | `TESTS_FAILED` |
| `failure_stage=build ran=false` | `ENV_ERROR` |
| missing, malformed, or `ran=false` | `ENV_ERROR` |

If legacy counts are absent, use zero and state that the queue did not report
them; never infer a passing count. The overall dispatch exit is supporting
evidence only because one failed architecture makes a multi-arch call
non-zero.

Do not set `TT_METAL_SIMULATOR`, `TT_METAL_CACHE`,
`TT_METAL_SLOW_DISPATCH_MODE`, or card locks on this route. Return after
recording the queued architecture results unless `ARCHES` also contains
Quasar; for a mixed solve, continue to Step C for Quasar only.

## Step C — Execute locally on ttsim, silicon, or Quasar Aether

Use this route for:

- every architecture on ttsim;
- local silicon when `HW_TEST_DISPATCH_CMD` is unset;
- Quasar on `TEST_BACKEND=local`, even when the dispatch command is set.

Use the same gtest binary for every backend. Ttsim uses its `.so`; local Quasar
uses the selected UMD simulator directory and slow dispatch.

Set `TT_METAL_HOME` to the tree containing the fix and use a fresh
`TT_METAL_CACHE`. For ttsim, use the arch library through
`TT_METAL_SIMULATOR`; its directory must contain `soc_descriptor.yaml`.
If the mapped test uses SFPU but does not verify `SFPLOADMACRO` itself, also
set `TT_METAL_DISABLE_SFPLOADMACRO=1`; that instruction is unavailable on
ttsim.

```bash
for arch in "${ARCHES[@]}"; do
  # In a mixed local solve, Step B already handled every card architecture.
  if [ "$TEST_BACKEND" = local ] &&
     [ -n "${HW_TEST_DISPATCH_CMD:-}" ] &&
     [ "$arch" != quasar ]; then
    continue
  fi

if [ "${VERIFY_STRATEGY:-worktree}" = warm ]; then
  HOME_TREE="$METAL_VERIFY_HOME"
  BIN="${METAL_VERIFY_BUILD_DIR:-$METAL_VERIFY_HOME/build}/test/tt_metal/unit_tests_llk"
else
  HOME_TREE="$WORKTREE_DIR"
  BIN="$WORKTREE_DIR/build/test/tt_metal/unit_tests_llk"
fi
CACHE_USER="${USER:-$(id -un)}"
TTCACHE_ROOT="${TTCACHE_ROOT:-/localdev/$CACHE_USER/ttcache}"
case "$TTCACHE_ROOT" in
  /*) ;;
  *) echo "ENV_ERROR: TTCACHE_ROOT must be absolute"; exit 3 ;;
esac
mkdir -p "$TTCACHE_ROOT"
FRESH_CACHE="$(mktemp -d "$TTCACHE_ROOT/ttcache_${arch}.XXXXXX")"
FRESH_CACHE_ARCH="$arch"
env_args=( TT_METAL_HOME="$HOME_TREE" TT_METAL_CACHE="$FRESH_CACHE" )
qsr_executed=0
# Opt-in: verify with device-side LLK asserts + Watcher so a firing assert prints a readable
# message to the run log instead of ebreak-hanging the kernel until the gtest timeout.
[ "${TT_METAL_LLK_ASSERTS:-0}" = 1 ] && env_args+=( TT_METAL_LLK_ASSERTS=1 TT_METAL_WATCHER=1 )

if [ "$TEST_BACKEND" = ttsim ]; then
  if [ "$(sg RUN_MODE)" = multi ]; then
    SIM_SO="$(
      python - "$(bg TTSIM_SO_PATHS)" "$arch" <<'PY'
import json
import sys

print(json.loads(sys.argv[1]).get(sys.argv[2], ""))
PY
    )"
  else
    SIM_SO="$(bg TTSIM_SO_PATH)"
  fi
  case "$SIM_SO" in "~/"*) SIM_SO="$HOME/${SIM_SO#\~/}" ;; esac
  [ -f "$SIM_SO" ] ||
    { echo "ENV_ERROR: missing ttsim .so for $arch"; exit 3; }
  [ -f "$(dirname "$SIM_SO")/soc_descriptor.yaml" ] || { echo "ENV_ERROR: no soc_descriptor.yaml beside $SIM_SO"; exit 3; }
  env_args+=( TT_METAL_SIMULATOR="$SIM_SO" TT_METAL_SLOW_DISPATCH_MODE=1 )
elif [ "$arch" = quasar ]; then
  # Quasar has no local card. The wrapper resolves QSR_SIM_BACKEND=emu|vcs,
  # resolves the IRD callback, shares run_test.sh's reservation lock, and
  # limits cleanup to this run's NNG tag.
  set +e
  bash "$WORKTREE_DIR/tt_metal/tt-llk/.claude/scripts/run_qsr_metal_test.sh" \
    --bin "$BIN" \
    --gtest-filter "$METAL_FILTER" \
    --tt-metal-home "$HOME_TREE" \
    --cache "$FRESH_CACHE" \
    --log-dir "$LOG_DIR" \
    --timeout "${TIMEOUT:-1200}"
  gtest_exit=$?
  set -e
  qsr_executed=1
else
  [ "$METAL_DISPATCH" = slow ] &&
    env_args+=( TT_METAL_SLOW_DISPATCH_MODE=1 )
fi
# Local Blackhole/Wormhole silicon reaches this point only when the queue
# command is unset.

set +e
if [ "${qsr_executed:-0}" != 1 ]; then
  env "${env_args[@]}" timeout "${TIMEOUT:-1200}" \
    "$BIN" --gtest_filter="$METAL_FILTER" 2>&1 | tee -a "$LOG_DIR/metal_run_${arch}.log"
  gtest_exit=${PIPESTATUS[0]}
fi
set -e
cleanup_fresh_cache || exit 3
done
```

Use a new cache path that does not already exist. Clean up only the exact path
created by `mktemp`; never reuse or delete an unknown cache directory. The exit
trap handles interruptions and `cleanup_fresh_cache` prevents accumulation
between architectures.
