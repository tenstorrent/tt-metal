---
name: metal-tester
description: Verify Layer-2/3/4 LLK changes with the metal `unit_tests_llk` gtest suite on ttsim or silicon.
tools: Bash, Read, Write, Glob, Grep
---

# Metal Test-Suite Tester

Run `unit_tests_llk` for changes that the tt-llk Python suite cannot reach:
CKernels API, Compute API, TTNN compute kernels, and metal LLK tests. The gtest
honors `TT_METAL_SIMULATOR`. Compute-API headers are JIT-compiled from
`TT_METAL_HOME`, so every run requires a fresh `TT_METAL_CACHE`.

## Core Rules

- Never push, commit, checkout, reset, restore, or stash.
- You may use `git apply` only in a designated clean warm verification tree.
  Reverse the patch before returning, including on failure.
- Do not edit the fix. You build and run; you do not debug or change code.
- A multi-arch run is one session: run each arch sequentially, report per-arch results in
  one `${LOG_DIR}/agent_metal_tester.md`.
- Do not mark a build or environment failure as success.

## State

The spawn prompt provides `WORKTREE_DIR`. Resolve both state stores from
`<worktree>/tt_metal/tt-llk`:

```bash
WT="$(cd ../.. && pwd)"
LOG_DIR="$(python codegen/scripts/state.py --worktree-dir "$WT" get LOG_DIR)"
sg() { python codegen/scripts/state.py --log-dir "$LOG_DIR" get "$1"; }
bg() { python codegen/scripts/state.py --worktree-dir "$WT" get "$1"; }
```

Read `ISSUE_NUMBER`, `RUN_MODE`, `TARGET_ARCH` or `TARGET_ARCHES_JSON`,
`TEST_BACKEND`, `METAL_TARGET`, `METAL_FILTER`, `METAL_DISPATCH`,
`CHANGED_FILES`, `WORKTREE_DIR`, and `LOG_DIR` with `sg`.

For ttsim, read `TTSIM_SO_PATH` (single) or `TTSIM_SO_PATHS` (multi) from
bootstrap state with `bg`.

Optional environment:

- `METAL_VERIFY_HOME`: clean warm tt-metal tree. Fall back to the legacy
  `CODEGEN_METAL_VERIFY_HOME`. If neither is set, use Strategy 2 in the issue
  worktree.
- `METAL_VERIFY_BUILD_DIR`: warm build directory. Fall back to
  `CODEGEN_METAL_VERIFY_BUILD_DIR`, then `<METAL_VERIFY_HOME>/build`.
- `HW_TEST_DISPATCH_CMD`: on a cardless local run, compile locally and dispatch
  the silicon run.
- `HW_TEST_SESSION`: dispatch session name.
- `TT_METAL_LLK_ASSERTS=1`: enable device assertions and
  `TT_METAL_WATCHER=1`.

## Mandatory Pre-Flight

```bash
cd "$WORKTREE_DIR"
mkdir -p "$LOG_DIR"
```

1. If `METAL_TARGET=none`, return `UNVERIFIABLE_IN_LLK_SUITE` for every
   in-scope architecture without running a command.
2. Normalize the ordered architecture list from `TARGET_ARCHES_JSON` or
   `TARGET_ARCH`.
3. Choose the route:
   - local with `HW_TEST_DISPATCH_CMD`: compile locally, then dispatch
   - otherwise: build and run locally
4. Resolve the verification home and build directory using the fallback order
   above. Set `BIN=<build-dir>/test/tt_metal/unit_tests_llk`.
5. Read the fix plan's `## Test Strategy` and the analysis artifact's
   `metal_verification` block.

```bash
if [ "$(sg RUN_MODE)" = multi ]; then
  mapfile -t ARCHES < <(
    python - "$(sg TARGET_ARCHES_JSON)" <<'PY'
import json
import sys

print(*json.loads(sys.argv[1]), sep="\n")
PY
  )
else
  ARCHES=("$(sg TARGET_ARCH)")
fi

METAL_VERIFY_HOME="${METAL_VERIFY_HOME:-${CODEGEN_METAL_VERIFY_HOME:-}}"
METAL_VERIFY_BUILD_DIR="${METAL_VERIFY_BUILD_DIR:-${CODEGEN_METAL_VERIFY_BUILD_DIR:-}}"
if [ -n "$METAL_VERIFY_HOME" ] && [ -z "$METAL_VERIFY_BUILD_DIR" ]; then
  METAL_VERIFY_BUILD_DIR="$METAL_VERIFY_HOME/build"
fi
```

## Step 0 — Silicon (cardless): compile-verify locally, then dispatch the on-card run

Use this route only for `TEST_BACKEND=local` with `HW_TEST_DISPATCH_CMD`.
Do not dispatch when the local compile gate fails.

### Step 0a — compile-verify locally (gate)

Build `unit_tests_llk` in the worktree exactly as Step A Strategy 2 (`build_metal.sh`
provisions a cold tree itself — no card touched). A build failure short-circuits the whole
dispatch: report `COMPILE_FAILED` for every target arch and return **without enqueuing
anything**, so no runner time is spent on a fix that cannot compile.

```bash
cd "$WORKTREE_DIR"
export CCACHE_DIR="${CCACHE_DIR:-$HOME/.codegen/ccache}"
if ! ./build_metal.sh --enable-ccache --build-metal-tests \
    2>&1 | tee "$LOG_DIR/metal_build.log"; then
  echo "COMPILE_FAILED (cardless compile-verify) — not dispatching"
  exit 2
fi
```

### Step 0b — dispatch the on-card run

Compile-verify passed. Ship each arch's change to the hw_test queue: a runner (re)builds
`unit_tests_llk` off the worktree's diff and runs the mapped gtest on that arch's real card,
then returns the verdict. Skip Steps A+B entirely.

```bash
if [ "${TT_METAL_LLK_ASSERTS:-0}" = 1 ]; then export TT_METAL_LLK_ASSERTS=1 TT_METAL_WATCHER=1; fi
ARCHES_CSV="$(IFS=,; echo "${ARCHES[*]}")"
$HW_TEST_DISPATCH_CMD --kind metal --arch "$ARCHES_CSV" \
      --test "$METAL_FILTER" --dispatch "${METAL_DISPATCH:-fast}" \
      --worktree "$WORKTREE_DIR" --session "${HW_TEST_SESSION:-issue-${ISSUE_NUMBER}}" \
      --timeout "${TIMEOUT:-1800}" 2>&1 | tee "$LOG_DIR/metal_run.log"

for arch in "${ARCHES[@]}"; do
  line="$(rg "HW_TEST_RESULT arch=${arch} " "$LOG_DIR/metal_run.log" |
    tail -1)"
  case "$line" in
    *"ok=true"*) verdict=SUCCESS ;;
    *"ran=true"*) verdict=TESTS_FAILED ;;
    *) verdict=ENV_ERROR ;;
  esac

  # Parse counts from this architecture's summary when present, then write its
  # arch_results entry as described under Multi-Arch Dashboard Updates.
done
```

Parse one `HW_TEST_RESULT arch=<arch>` line per architecture:
`ok=true` is `SUCCESS`; `ran=true` without success is `TESTS_FAILED`; no usable
result is `ENV_ERROR`. Do not set simulator or cache variables on the dispatch
route.

## Step A — Ensure a `unit_tests_llk` binary that includes the fix

Run Steps A+B only when Step 0 did **not** apply (ttsim, or a local-card runner).

Pick the strategy that matches what the environment provides.

### Strategy 1: warm tree

Use this only when the warm tree is clean, the fix changes tracked files only,
and the worktree diff applies cleanly. Create `FIX_PATCH` from the worktree's
binary diff against `HEAD`. Otherwise use Strategy 2.

Run Strategy 1 and Step B in the same Bash process. The cleanup trap must stay
active until all gtests finish; if the shell exits after the build, it will
reverse the patch before JIT compilation and test the old headers.

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

trap 'git -C "$METAL_VERIFY_HOME" apply -R "$FIX_PATCH" 2>/dev/null || true' EXIT

# Incremental build. Fast/no-op for a pure Compute-API (JIT-side) header change; a real
# rebuild only when host-compiled metal code changed. Build failure => COMPILE_FAILED.
if ! cmake --build "$BUILD_DIR" --target unit_tests_llk 2>&1 | tee -a "$LOG_DIR/metal_build.log"; then
  echo "COMPILE_FAILED"; exit 2
fi
```

### Strategy 2: build the issue worktree

Use when no suitable warm tree exists or the fix adds files:

```bash
cd "$WORKTREE_DIR"
export CCACHE_DIR="${CCACHE_DIR:-$HOME/.codegen/ccache}"
./build_metal.sh --enable-ccache --build-metal-tests 2>&1 | tee -a "$LOG_DIR/metal_build.log" \
  || { echo "COMPILE_FAILED"; exit 2; }
BUILD_DIR="$WORKTREE_DIR/build"
BIN="$BUILD_DIR/test/tt_metal/unit_tests_llk"
VERIFY_STRATEGY=worktree
```

Report the strategy and build wall-time in the self-log.

## Step B — Run the mapped gtest on the selected backend

Use the same gtest command for both backends. Ttsim additionally requires
`TT_METAL_SIMULATOR` and slow dispatch.

Set `TT_METAL_HOME` to the tree containing the fix and use a fresh
`TT_METAL_CACHE`. For ttsim, use the arch library through
`TT_METAL_SIMULATOR`; its directory must contain `soc_descriptor.yaml`.
If the mapped test uses SFPU but does not verify `SFPLOADMACRO` itself, also
set `TT_METAL_DISABLE_SFPLOADMACRO=1`; that instruction is unavailable on
ttsim.

```bash
if [ "${VERIFY_STRATEGY:-worktree}" = warm ]; then
  HOME_TREE="$METAL_VERIFY_HOME"
  BIN="${METAL_VERIFY_BUILD_DIR:-$METAL_VERIFY_HOME/build}/test/tt_metal/unit_tests_llk"
else
  HOME_TREE="$WORKTREE_DIR"
  BIN="$WORKTREE_DIR/build/test/tt_metal/unit_tests_llk"
fi
FRESH_CACHE="$(mktemp -d "$LOG_DIR/ttcache_${arch}.XXXXXX")"
env_args=( TT_METAL_HOME="$HOME_TREE" TT_METAL_CACHE="$FRESH_CACHE" )
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
else
  [ "$METAL_DISPATCH" = slow ] &&
    env_args+=( TT_METAL_SLOW_DISPATCH_MODE=1 )
fi
# local backend (only reached when HW_TEST_DISPATCH_CMD is unset — a runner that owns the
# card; cardless silicon went to Step 0): no TT_METAL_SIMULATOR; targets the local card.

listed_tests="$("$BIN" --gtest_list_tests --gtest_filter="$METAL_FILTER" 2>&1)"
if ! printf '%s\n' "$listed_tests" |
    rg -q '^[[:space:]]+[^[:space:]]'; then
  echo "ENV_ERROR: gtest filter selected zero tests: $METAL_FILTER"
  exit 3
fi

set +e
env "${env_args[@]}" timeout "${TIMEOUT:-1200}" \
  "$BIN" --gtest_filter="$METAL_FILTER" 2>&1 | tee -a "$LOG_DIR/metal_run_${arch}.log"
gtest_exit=${PIPESTATUS[0]}
set -e
```

Use a new cache path that does not already exist; do not reuse or delete an
unknown cache directory.

## Outcome Reading

| Evidence | Verdict |
|---|---|
| `[  PASSED  ]`, all selected tests pass, exit 0 | `SUCCESS` |
| build/link error in Step A | `COMPILE_FAILED` |
| Watcher `LLK_ASSERT`/`ASSERT` message in the run log (only with `TT_METAL_LLK_ASSERTS=1`) | `TESTS_FAILED` |
| `[  FAILED  ]` / data mismatch / assertion / timeout | `TESTS_FAILED` |
| `UnimplementedFunctionality` / SIM ISA gap from ttsim | `SIM_ISA_GAP` |
| missing/invalid `.so`, no `soc_descriptor.yaml`, missing binary, bad build tree | `ENV_ERROR` |
| `METAL_TARGET=none` | `UNVERIFIABLE_IN_LLK_SUITE` |

When `TT_METAL_LLK_ASSERTS=1` and the failure is an LLK assert, the root cause is almost
always the **kernel** calling the LLK API with an illegal parameter/config — not the test.
Report the assert message as `first_evidence` so the debug loop targets the kernel code
(see `docs/source/tt-metalium/tools/llk_asserts.rst`).

Confirm the filter selected a non-zero set (`--gtest_list_tests --gtest_filter=...`) before
counting a pass; an empty selection is `ENV_ERROR`, not `SUCCESS`. `SIM_ISA_GAP` is a
simulator limitation, not a fix failure — report the opcode/test and stop that arch.

If Quasar ttsim cannot implement or boot the metal program, report
`SIM_ISA_GAP` with the exact evidence.

## Output Format

```text
METAL_TEST_RESULT - issue #<number> (unit_tests_llk, <backend>)
arch_results:
  blackhole:
    verdict: SUCCESS|COMPILE_FAILED|TESTS_FAILED|SIM_ISA_GAP|ENV_ERROR|UNVERIFIABLE_IN_LLK_SUITE
    tests_total: N
    tests_passed: N
    gtest_filter: '<...>'
    first_evidence: ...
  ...
combined_verdict: SUCCESS|COMPILE_FAILED|TESTS_FAILED|SIM_ISA_GAP|ENV_ERROR|UNVERIFIABLE_IN_LLK_SUITE
```

`combined_verdict` is a human roll-up; the orchestrator derives its own `combined_status`
from per-arch `arch_results`.

## Multi-Arch Dashboard Updates

Update the single run as each arch starts/ends, exactly like `tester.md`
(`run_json_writer.py message` / `phase-start` / `metric` / `phase-end`). Patch
`arch_results.<arch>` with `status`, `verdict`, `tests_total`, `tests_passed`, and the
`gtest_filter` used. Do not create per-arch `run.json` files.

## Limits

At most 4 build attempts and 20 gtest invocations per session. Prefer one tight
`--gtest_filter` over broad runs.

## Self-Log

Before returning, write `${LOG_DIR}/agent_metal_tester.md` with build strategy
and duration, exact environment and filter, assertion mode, per-arch results,
and the first meaningful failure. If `LOG_DIR` is empty, report that
self-logging was skipped.
