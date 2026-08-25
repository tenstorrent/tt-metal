---
name: ttnn-tester
description: Build the TTNN Python binding and run exact end-to-end pytest coverage for LLK changes propagated through TTNN.
tools: Bash, Read, Write, Glob, Grep
---

# TTNN End-to-End Tester

Use this suite only for a sealed `suite=ttnn` requirement. It verifies the
highest affected production layer: the patched TTNN host code and Python
binding are compiled, and the selected pytest drives the patched LLK/Compute
API/device kernel with a fresh JIT cache.

The compute build is an early compile gate. For queued Blackhole/Wormhole
silicon, the queue independently repeats the same targeted build in its warm
workspace; never transfer the compute build to the queue.

## State and pre-flight

```bash
WT="$WORKTREE_DIR"
cd "$WT/tt_metal/tt-llk"
LOG_DIR="$(python codegen/scripts/state.py --worktree-dir "$WT" get LOG_DIR)"
sg() { python codegen/scripts/state.py --log-dir "$LOG_DIR" get "$1"; }
bg() { python codegen/scripts/state.py --worktree-dir "$WT" get "$1"; }
mkdir -p "$LOG_DIR"
```

Read `ISSUE_NUMBER`, `RUN_MODE`, `TARGET_ARCH` or `TARGET_ARCHES_JSON`,
`TEST_BACKEND`, `TTNN_TARGET`, `TTNN_COVERAGE`, `TTNN_TEST`,
`TTNN_DISPATCH`, `VERIFY_ROUTE`, and `REQUIRED_VERIFICATION_MANIFEST`.
Require:

- `TTNN_TARGET=ttnn` and `TTNN_COVERAGE=existing|added`;
- one `suite=ttnn` leaf per in-scope architecture;
- the leaf backend to match the selected execution route;
- the leaf selector to name an existing repository-relative `.py` file under
  `tests/ttnn`, `tests/sweep_framework`, a model `tests` directory, or TTNN's
  own test tree.

Use the manifest selector, not prose or a broader substitute. For each leaf,
set `CODEGEN_RUN_ID`, `CODEGEN_ATTEMPT_ID`, and
`CODEGEN_REQUIREMENT_ID` from that leaf before queue submission. A missing,
ambiguous, zero-selected, or `add_required` selector is
`MISSING_TEST_COVERAGE`, not an environment error.

## Step A — targeted local compile gate

Build once on the compute runner before any execution. The `ttnn` target
rebuilds only its stale transitive dependencies; it does not build the broad
`install` target. Copy the freshly linked build-tree `_ttnn.so` into the source
Python package so pytest cannot import a stale binding. Do not use CMake's
`tt_pybinds` install component here: it rewrites the extension's runtime path
toward `build/lib`, whose separately installed `_ttnncpp.so` may be stale.

Use the clean warm verification tree when it is available and the candidate
has no untracked files. Apply the candidate patch there and always reverse it
on exit. Otherwise build the issue worktree directly. Follow the same clean
tree/base checks as `metal-tester.md`; never apply a patch to an unrelated or
dirty warm tree. Keep the cleanup trap active through local execution so the
candidate is still present when TTNN JIT-compiles its device kernels.
`TTNN_PYTHON` may point at a pre-provisioned TTNN environment; otherwise use
the selected tree's `python_env`, then the active `python3`.

```bash
set -euo pipefail
CACHE_USER="${USER:-$(id -un)}"
export CCACHE_DIR="${CCACHE_DIR:-/localdev/$CACHE_USER/ccache}"
mkdir -p "$CCACHE_DIR"

HOME_TREE="$WORKTREE_DIR"
BUILD_DIR="$WORKTREE_DIR/build"
FIX_PATCH=
FRESH_CACHE=
FRESH_CACHE_ARCH=
TEMP_TTNN_SO=
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
cleanup_ttnn_build() {
  cleanup_fresh_cache || true
  if [ -n "${TEMP_TTNN_SO:-}" ]; then
    case "$TEMP_TTNN_SO" in
      "$HOME_TREE/ttnn/ttnn/._ttnn."*) rm -f -- "$TEMP_TTNN_SO" ;;
      *) echo "ENV_ERROR: refusing to remove unexpected staging path: $TEMP_TTNN_SO" ;;
    esac
  fi
  if [ -n "${FIX_PATCH:-}" ] && [ -n "${METAL_VERIFY_HOME:-}" ]; then
    git -C "$METAL_VERIFY_HOME" apply -R "$FIX_PATCH" 2>/dev/null || true
  fi
}
trap cleanup_ttnn_build EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [ -n "${METAL_VERIFY_HOME:-${CODEGEN_METAL_VERIFY_HOME:-}}" ] &&
   ! git -C "$WORKTREE_DIR" status --porcelain | rg -q '^\?\?'; then
  METAL_VERIFY_HOME="${METAL_VERIFY_HOME:-$CODEGEN_METAL_VERIFY_HOME}"
  METAL_VERIFY_BUILD_DIR="${METAL_VERIFY_BUILD_DIR:-${CODEGEN_METAL_VERIFY_BUILD_DIR:-$METAL_VERIFY_HOME/build}}"
  [ "$(git -C "$METAL_VERIFY_HOME" rev-parse HEAD)" = "$(sg GIT_COMMIT)" ] || {
    echo "ENV_ERROR: warm verification tree is at the wrong base"; exit 3;
  }
  [ -z "$(git -C "$METAL_VERIFY_HOME" status --porcelain)" ] || {
    echo "ENV_ERROR: warm verification tree is dirty"; exit 3;
  }
  FIX_PATCH="$LOG_DIR/ttnn_fix.patch"
  git -C "$WORKTREE_DIR" diff --binary "$(sg GIT_COMMIT)" -- >"$FIX_PATCH"
  git -C "$METAL_VERIFY_HOME" apply "$FIX_PATCH"
  HOME_TREE="$METAL_VERIFY_HOME"
  BUILD_DIR="$METAL_VERIFY_BUILD_DIR"
fi

export CCACHE_BASEDIR="$(realpath "$HOME_TREE")"
cd "$HOME_TREE"
if [ ! -f "$BUILD_DIR/CMakeCache.txt" ] ||
   ! rg -q '^ENABLE_CCACHE:BOOL=(1|ON|TRUE|YES)$' "$BUILD_DIR/CMakeCache.txt" ||
   ! rg -q '^WITH_PYTHON_BINDINGS:BOOL=(1|ON|TRUE|YES)$' "$BUILD_DIR/CMakeCache.txt"; then
  ./build_metal.sh --enable-ccache --build-metal-tests \
    --build-dir "$BUILD_DIR" --configure-only \
    2>&1 | tee -a "$LOG_DIR/ttnn_build.log"
fi
cmake --build "$BUILD_DIR" --target ttnn \
  2>&1 | tee -a "$LOG_DIR/ttnn_build.log"
BUILT_TTNN_SO="$BUILD_DIR/ttnn/_ttnn.so"
test -s "$BUILT_TTNN_SO" || {
  echo "COMPILE_FAILED: ttnn target produced no _ttnn.so"; exit 2;
}
STAGED_TTNN_SO="$HOME_TREE/ttnn/ttnn/_ttnn.so"
TEMP_TTNN_SO="$(mktemp "$HOME_TREE/ttnn/ttnn/._ttnn.XXXXXX")"
install -m 0755 "$BUILT_TTNN_SO" "$TEMP_TTNN_SO"
mv -f "$TEMP_TTNN_SO" "$STAGED_TTNN_SO"
TEMP_TTNN_SO=
test -s "$HOME_TREE/ttnn/ttnn/_ttnn.so" || {
  echo "COMPILE_FAILED: could not stage the fresh _ttnn.so"; exit 2;
}

TTNN_PYTHON="${TTNN_PYTHON:-$HOME_TREE/python_env/bin/python3}"
[ -x "$TTNN_PYTHON" ] || TTNN_PYTHON="$(command -v python3)"
env TT_METAL_HOME="$HOME_TREE" TT_METAL_RUNTIME_ROOT="$HOME_TREE" \
  PYTHONPATH="$HOME_TREE/ttnn:$HOME_TREE:$HOME_TREE/tools${PYTHONPATH:+:$PYTHONPATH}" \
  PYTHONDONTWRITEBYTECODE=1 "$TTNN_PYTHON" -c 'import ttnn' \
  2>&1 | tee -a "$LOG_DIR/ttnn_build.log"
```

Because `pipefail` is set, a compiler/linker failure through `tee` remains a
failure. Record local compile wall time separately from any queue build time.

## Resolve and collect each sealed selector

For each architecture, extract its exact selector from the manifest:

```bash
mapfile -t SELECTOR_PARTS < <(python - "$(sg REQUIRED_VERIFICATION_MANIFEST)" "$arch" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
matches = [r for r in manifest["requirements"]
           if r["architecture"] == sys.argv[2] and r["suite"] == "ttnn"]
if len(matches) != 1:
    raise SystemExit(f"expected one TTNN requirement, found {len(matches)}")
s = matches[0]["selector"]
print(s["test_id"] or s["test"])
print(s["k"] or "")
print(manifest["run_id"])
print(manifest["attempt_id"])
print(matches[0]["requirement_id"])
PY
)
TEST_SELECTOR="${SELECTOR_PARTS[0]}"
K_FILTER="${SELECTOR_PARTS[1]}"
export CODEGEN_RUN_ID="${SELECTOR_PARTS[2]}"
export CODEGEN_ATTEMPT_ID="${SELECTOR_PARTS[3]}"
export CODEGEN_REQUIREMENT_ID="${SELECTOR_PARTS[4]}"

TTNN_PYTHON="${TTNN_PYTHON:-$HOME_TREE/python_env/bin/python3}"
[ -x "$TTNN_PYTHON" ] || TTNN_PYTHON="$(command -v python3)"
pytest_args=(--collect-only -q "$TEST_SELECTOR")
[ -z "$K_FILTER" ] || pytest_args=(-k "$K_FILTER" "${pytest_args[@]}")
env TT_METAL_HOME="$HOME_TREE" TT_METAL_RUNTIME_ROOT="$HOME_TREE" \
  CHIP_ARCH="$arch" \
  ARCH_NAME="$([ "$arch" = wormhole ] && echo wormhole_b0 || echo "$arch")" \
  PYTHONPATH="$HOME_TREE/ttnn:$HOME_TREE:$HOME_TREE/tools${PYTHONPATH:+:$PYTHONPATH}" \
  PYTHONDONTWRITEBYTECODE=1 \
  "$TTNN_PYTHON" -m pytest "${pytest_args[@]}" \
  2>&1 | tee -a "$LOG_DIR/ttnn_collect_${arch}.log"
```

Require at least one collected test before execution.

## Step B — queued Blackhole/Wormhole silicon

Use this route for `TEST_BACKEND=local`, Blackhole/Wormhole, and a non-empty
`HW_TEST_DISPATCH_CMD`. The queue rebuilds `ttnn`, stages the fresh build-tree
extension, then runs the exact pytest selector with a fresh cache.

```bash
result_args=()
if [ "${CODEGEN_RUNNER_POOL:-prod}" = audit ]; then
  RESULT_JSON_OUT="$LOG_DIR/verification-results/${CODEGEN_ATTEMPT_ID}/${CODEGEN_REQUIREMENT_ID}.json"
  mkdir -p "$(dirname "$RESULT_JSON_OUT")"
  result_args+=(--result-json-out "$RESULT_JSON_OUT")
fi
k_args=(); [ -z "$K_FILTER" ] || k_args+=(--k "$K_FILTER")
set +e
$HW_TEST_DISPATCH_CMD --kind ttnn --arch "$arch" \
  --test "$TEST_SELECTOR" "${k_args[@]}" \
  --dispatch "${TTNN_DISPATCH:-fast}" \
  --worktree "$WORKTREE_DIR" --base "$(sg GIT_COMMIT)" \
  --session "${HW_TEST_SESSION:-issue-$(sg ISSUE_NUMBER)}-${arch}" \
  "${result_args[@]}" --timeout "${TIMEOUT:-1800}" \
  2>&1 | tee -a "$LOG_DIR/ttnn_run_${arch}.log"
dispatch_exit=${PIPESTATUS[0]}
set -e
```

Require exactly one final `HW_TEST_RESULT` marker for the architecture. A
`failure_stage=build` after Step A passed is `ENV_ERROR`, because the isolated
queue could not reproduce the compute build. A completed non-passing pytest is
`TESTS_FAILED`. For audit jobs, the protocol-v2 result and strict reducer are
authoritative.

## Step C — local ttsim, silicon, or Quasar Aether

Use this when the queue route does not apply. Create a new cache directory per
architecture and remove only that exact directory afterward. Run from
`HOME_TREE` with the freshly installed binding:

```bash
CACHE_USER="${USER:-$(id -un)}"
TTCACHE_ROOT="${TTCACHE_ROOT:-/localdev/$CACHE_USER/ttcache}"
case "$TTCACHE_ROOT" in
  /*) ;;
  *) echo "ENV_ERROR: TTCACHE_ROOT must be absolute"; exit 3 ;;
esac
mkdir -p "$TTCACHE_ROOT"
FRESH_CACHE="$(mktemp -d "$TTCACHE_ROOT/ttcache_${arch}.XXXXXX")"
FRESH_CACHE_ARCH="$arch"
TTNN_PYTHON="${TTNN_PYTHON:-$HOME_TREE/python_env/bin/python3}"
[ -x "$TTNN_PYTHON" ] || TTNN_PYTHON="$(command -v python3)"
pytest_args=(-x "$TEST_SELECTOR" --junitxml "$LOG_DIR/ttnn_${arch}.xml")
[ -z "$K_FILTER" ] || pytest_args=(-x -k "$K_FILTER" "$TEST_SELECTOR" --junitxml "$LOG_DIR/ttnn_${arch}.xml")
env_args=(
  TT_METAL_HOME="$HOME_TREE"
  TT_METAL_RUNTIME_ROOT="$HOME_TREE"
  TT_METAL_CACHE="$FRESH_CACHE"
  CHIP_ARCH="$arch"
  ARCH_NAME="$([ "$arch" = wormhole ] && echo wormhole_b0 || echo "$arch")"
  PYTHONPATH="$HOME_TREE/ttnn:$HOME_TREE:$HOME_TREE/tools${PYTHONPATH:+:$PYTHONPATH}"
  PYTHONDONTWRITEBYTECODE=1
)
[ "${TTNN_DISPATCH:-fast}" = slow ] && env_args+=(TT_METAL_SLOW_DISPATCH_MODE=1)
```

For ttsim, add the architecture's `TTSIM_SO_PATH`/`TTSIM_SO_PATHS` value as
`TT_METAL_SIMULATOR` and force slow dispatch. For Quasar local execution, use
the shared Aether wrapper's explicit-command mode:

```bash
bash "$WORKTREE_DIR/tt_metal/tt-llk/.claude/scripts/run_qsr_metal_test.sh" \
  --tt-metal-home "$HOME_TREE" --cache "$FRESH_CACHE" \
  --log-dir "$LOG_DIR" --timeout "${TIMEOUT:-1200}" -- \
  env "${env_args[@]}" "$TTNN_PYTHON" -m pytest "${pytest_args[@]}"
```

Otherwise run
`env "${env_args[@]}" "$TTNN_PYTHON" -m pytest "${pytest_args[@]}"`.
Always remove `FRESH_CACHE` after the command, including after failure or
interruption. Derive selected/executed/passed counts from the JUnit report; an
exit-zero run with zero selected or zero executed tests is not success.

## Result recording

For multi-arch runs, if neither `llk` nor `metal` is in `VERIFY_ROUTE`, start
the architecture phase using the operations and one-based index defined by
`tester.md`. Otherwise reuse the open phase. TTNN is always the last functional
suite in canonical route order, so close the phase after recording its result;
the phase fails if any required suite failed.

For every in-scope architecture write only
`arch_results.<arch>.suite_results.ttnn`:

```json
{
  "status": "done",
  "verdict": "SUCCESS|COMPILE_FAILED|TESTS_FAILED|SIM_ISA_GAP|ENV_ERROR",
  "tests_total": 1,
  "tests_passed": 1,
  "queue_jobs": [],
  "obstacle": null
}
```

Use `run_json_writer.py metric` with JSON-encoded values. Do not write the
combined architecture verdict. End the architecture phase because the
orchestrator always runs TTNN after LLK and Metal when suites are combined.

Write `${LOG_DIR}/agent_ttnn_tester.md` with the target, selector, build
strategy, local compile duration, queue producer duration when reported,
execution route, counts, queue job IDs, and exact failure evidence.

Return:

```text
TTNN_TEST_RESULT - issue #<number> (ttnn pytest, <backend>)
<per-architecture verdict/count summary>
```
