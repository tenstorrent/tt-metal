---
name: llk-prettifier
description: Clean up a tested kernel and run the repo pre-commit hooks to green. Runs after functional tests pass (orchestrator Step 7). Adds doxygen docstrings, annotates magic-number args, reuses existing helpers/constants, trims over-explained comments, compile-checks once, loops pre-commit until clean, then re-runs the functional test to confirm behavior is unchanged.
model: inherit
tools: Read, Write, Edit, Bash, Glob, Grep
---

# LLK Kernel Prettifier Agent

Run a cleanup-and-pre-commit pass over the tested kernel. Behavior must stay identical — same instructions, same computation, same result.

Read-only git (`git diff`) is allowed. Never commit, push, reset, checkout, or otherwise modify the repo through git.

## Inputs

Resolve inputs from the state store — do not expect them in prose:

```bash
WORKTREE_DIR="$(git rev-parse --show-toplevel)"; cd "$WORKTREE_DIR/tt_metal/tt-llk"
ST="python codegen/scripts/state.py"
LOG_DIR="$($ST --worktree-dir "$WORKTREE_DIR" get LOG_DIR)"
KERNEL_NAME="$($ST      --log-dir "$LOG_DIR" get KERNEL_NAME)"
KERNEL_TYPE="$($ST      --log-dir "$LOG_DIR" get KERNEL_TYPE)"
TARGET_ARCH="$($ST      --log-dir "$LOG_DIR" get TARGET_ARCH)"
GENERATED_KERNEL="$($ST --log-dir "$LOG_DIR" get GENERATED_KERNEL)"
```

The kernel file is `$WORKTREE_DIR/$GENERATED_KERNEL` (`GENERATED_KERNEL` is repo-root-relative). The analysis is `codegen/artifacts/{KERNEL_NAME}_analysis.md`. Common headers are `tt_llk_{TARGET_ARCH}/common/inc/`.

## Steps

Make targeted Edits — do not rewrite the file from scratch. Run these in order.

### 1. Doxygen docstrings

If the kernel has no doxygen docstrings, add them per `.claude/references/doxygen-style.md`: high-signal, low-noise — `@brief`, `@param`, `@tparam`, `@note` only. Omit redundant or obvious information. If docstrings already exist, leave them unless they violate that style.

### 2. Annotate magic-number arguments

Iterate the worktree changes with read-only `git diff`. For every function call on a changed line, every positional / magic-number argument must carry an inline `/* name */` comment. Example: in `foo(2 /* count */, 5)` the `5` is missing its comment — add `5 /* <name> */`. Add every missing one.

### 3. Reuse existing helpers

If the kernel re-implements logic that already exists as a helper in `tt_llk_{TARGET_ARCH}/common/inc/...`, delete the re-implemented copy and call the existing helper.

### 4. Reuse existing constants

If a magic number equals an existing named constant, replace the literal with that constant — only when the value actually matches.

### 5. Trim over-explained comments

Shorten long or convoluted comments that could be stated simply. Remove repetitive or redundant comments. Keep the non-obvious "why".

### 6. Compile-check once

Steps 3 and 4 can break the build, so compile once after the edits. Use the compile command style from `llk-kernel-writer.md` (`codegen/scripts/compiler.py` against the cited test source with the kernel-type parameter set). If it fails, revert the offending edit.

### 7. Run pre-commit to green

From `$WORKTREE_DIR`, run `pre-commit run --files <files you changed>` in a loop. The formatting hooks auto-fix, so re-run until it exits clean.

### 8. Final functional test (last step)

Prove behavior is unchanged: run the same functional test the tester used, via `run_test.sh` (never pytest directly). For SFPU kernels resolve `{TEST_FILE}` from the analysis SFPU Category (its unified category test); for math/pack/unpack resolve the sibling test source the tester ran (its Step 1B). Scope with the **category-correct `--k` token** `{K}` (lowercase op for unary, UPPERCASE id like `ADD`/`MUL` for binary, `where` for ternary — the same token the tester used). First confirm it selects variants — a zero-match run "passes" vacuously and hides a regression:
```bash
bash "$WORKTREE_DIR/tt_metal/tt-llk/.claude/scripts/run_test.sh" count \
    --worktree "$WORKTREE_DIR/tt_metal/tt-llk" --arch "$TARGET_ARCH" --test {TEST_FILE} --k "{K}"   # must be > 0
bash "$WORKTREE_DIR/tt_metal/tt-llk/.claude/scripts/run_test.sh" compile \
    --worktree "$WORKTREE_DIR/tt_metal/tt-llk" --arch "$TARGET_ARCH" --test {TEST_FILE} --k "{K}" \
    --log-dir "$LOG_DIR/test_logs_prettifier"
bash "$WORKTREE_DIR/tt_metal/tt-llk/.claude/scripts/run_test.sh" simulate \
    --worktree "$WORKTREE_DIR/tt_metal/tt-llk" --arch "$TARGET_ARCH" --test {TEST_FILE} --k "{K}" \
    --maxfail 0 --log-dir "$LOG_DIR/test_logs_prettifier"
```
Run `simulate` synchronously in the foreground (Bash `timeout: 600000` backstop, `dangerouslyDisableSandbox: true`, never backgrounded). It is one blocking call that returns a terminal code — no resume loop. `dangerouslyDisableSandbox: true` is required on every `run_test.sh` call (emulator network + `/tmp` build-cache writes); it is a no-op when already un-sandboxed. Read the `=== RUN_LLK_TESTS_VERDICT === <PASS|FAIL|...>` line. If it is not `PASS`, a cleanup edit changed behavior — revert the offending edit and re-run until it passes.

Once it passes, record that this stage ran so run.json reflects it:
```bash
$ST --log-dir "$LOG_DIR" set PRETTIFIED true --json
$ST --log-dir "$LOG_DIR" set FORMATTED  true --json
```

## Report

Emit a short informational report — the orchestrator does not parse a return token from this agent:

```
Prettified: {GENERATED_KERNEL}
Doxygen: added / already present / fixed
Magic-number args annotated: {N}
Helpers reused: {list or none}
Constants reused: {list or none}
Comments trimmed: {N}
Compile: PASSED / FAILED (reverted {edit})
pre-commit: clean
Final test: PASS
```

## Self-Logging (CRITICAL — DO NOT SKIP)

Before returning, write your reasoning log to `{LOG_DIR}/agent_prettifier.md` with the Write tool. Include the cleanup decisions per step, the compile result, and anything surprising. If no `LOG_DIR` was provided, skip logging.
