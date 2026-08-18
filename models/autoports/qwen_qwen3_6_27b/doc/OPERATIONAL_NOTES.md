# Operational notes: what broke in the tooling, and what it cost

Written 2026-08-18. The rest of this directory documents findings about the model and the
pipeline. This file documents failures in **my own instrumentation** during that work,
because three of them produced conclusions that were wrong, and several others silently
degraded runs. Anyone repeating this work will hit the same traps.

## A. Traps that produced a WRONG conclusion

These matter most: in each case the tool reported something confidently and the report was
false.

### A1. A smoke test that always passed

`mesh_smoke.sh` ended its device check with a pipeline and then read `$?`:

```bash
timeout 300 python - <<'PY' ... PY 2>&1 | grep -E "MESH_|Error" | head -12
echo "SMOKE_EXIT=$?"        # <- status of `head`, always 0
```

`$?` is the exit status of the **last element of the pipeline** (`head`), not of python. So
the script printed `SMOKE_EXIT=0` on a mesh that was throwing
`TT_THROW: Timed out while waiting for active ethernet core 29-25 to become active again`.

**Cost:** a wedged mesh was reported healthy. **Fixed:** the check now captures python's own
status, asserts on the `MESH_OPEN_OK` / `SEM_OK` / `MESH_CLOSE_OK` markers, greps explicitly
for the ethernet-core timeout, and `exit`s non-zero. Callers were changed to abort rather
than continue when it fails.

**General form of the bug:** never read `$?` after a pipeline unless you mean the last
stage. Use `PIPESTATUS`, or redirect to a file and check separately. The same mistake
appeared a second time in a status check whose `||` fallback never fired because `sed`
succeeded on empty input.

### A2. A monitor that reported a previous run's failure as the current run's

A watcher picked the newest `workflow_logs/local_server/*.log` with `ls -t`. When a new
attempt had not yet created its log, that resolved to the **previous** attempt's file, whose
tail contained a `TT_FATAL`. The monitor duly reported `server:FAILED` for a run that was
still in its reset phase and had not started a server at all.

**Cost:** one false failure verdict, acted on before being caught. **Fixed:** the watcher
now parses the run's own start timestamp out of `run.log` and selects only logs newer than
it (`find -newermt "$START"`). Prefer a positive freshness guard over "newest file".

### A3. An idempotency check that matched the wrong entry

The script inserting the `r1_gpqa_diamond` task into `eval_config.py` guarded against
double-application by searching a fixed 8,000-character window after the model's anchor:

```python
tail = src[i:i + 8000]
if 'task_name="r1_gpqa_diamond"' in tail:
    print("already applied, nothing to do")
```

The window ran past the end of the Qwen3.6-27B `EvalConfig` and into the **Qwen3.8-27B**
entry that follows, which legitimately contains `r1_gpqa_diamond`. So the script reported
"already applied" and did nothing, and the subsequent `git commit` reported "nothing to
commit" — a clean-looking no-op.

**Cost:** the config change silently did not happen; caught only because the script also
printed the resulting task list, which showed `terminal_bench_2` alone. **Fixed:** the
window is now bounded by the next `hf_model_repo=` occurrence, the insertion point is
asserted to fall inside that range, and the script prints the tasks it found before acting.

**Lesson worth keeping:** a verification step that prints what it *actually* observed
caught this; a step that only printed "done" would not have.

### A4. A probe verdict that did not control for its own configuration

`stop_token_probe.py` concluded "stop tokens are NOT honoured" when a trivial prompt hit its
token cap. But the probe sent no `chat_template_kwargs`, so thinking mode was on and the
budget was spent on a reasoning preamble. The conclusion did not follow. **Fixed** in the
committed test: that branch now reports INCONCLUSIVE and points at the
`enable_thinking=false` control arm.

### A5. Baselines in the wrong unit

`batch_isolation_probe.py` compared word counts against baselines (2 / 472 / 1849) that were
actually **completion-token** counts from an earlier probe. Every `WORD-COUNT COLLAPSE` flag
and every "ratio ~0.5" in its raw output is therefore meaningless — that ratio is just
words-per-token. The conclusions in `BATCH32_DEGRADATION.md` rest on repetition rate, boxed
answer and `finish_reason`, which are unaffected, and the mistake is recorded there too.

### A6. Process matching that matched the wrong process

`pgrep -f <pattern>` matched a `timeout` wrapper rather than the python child, which once
supported a wrong claim that a process was wedged. Later, `pkill -f "<pattern>"` matched the
**shell issuing the pkill**, because the pattern appeared in that shell's own command line —
killing it mid-script (exit 143) and, twice, killing the monitors with it, which then
surfaced as spurious "failure" notifications. This happened three times before I stopped
using `pkill -f` with patterns drawn from my own command text.

**Use instead:** resolve PIDs in one step, kill by number in the next; or use the `[p]attern`
bracket trick *and* verify it cannot match the invoking command.

## B. Traps that cost time but not correctness

### B1. `tt-smi` is not on PATH outside the venv

The chain scripts called bare `tt-smi -r`. It exists only at
`~/tt-metal/python_env/bin/tt-smi`, so every inter-stage board reset **silently did nothing**
while printing a failure line the chain ignored. Runs that were supposed to start from a
freshly reset mesh did not. **Fixed:** absolute path everywhere, and a failed smoke now
aborts the chain.

### B2. `exec > >(tee file)` fails when detached

Scripts using process substitution for logging died before writing anything when launched
without a controlling terminal (`setsid nohup ... &`, and `docker exec -d`). The symptom was
maddening: no process, no log, no error — the log file still held the *previous* run's
content because `tee` never ran. Three launch attempts were lost to this. **Fixed:** plain
`exec >"$OUT/run.log" 2>&1` for anything that runs detached; keep process substitution for
foreground use only.

### B3. Heredocs inside `docker exec bash -c '...'`

Apostrophes in prose (`user's`, `Here's`) terminated the outer single-quoted string, which
truncated a commit message mid-word and required an amend; nested `<<'EOF'` inside the same
construction produced `unexpected EOF while looking for matching`. **Fixed:** write files
locally with the editor, `docker cp` them in, and use `git commit -F <file>`. No prose
through shell quoting.

### B4. Wrong assumption about a data structure

`EVAL_CONFIGS` is a **dict keyed by `model_spec.model_name`**, not a list of configs.
Iterating it yielded strings, so a verification script reported "0 entries for
Qwen/Qwen3.6-27B" for a config that was present and correct. Worth knowing because that dict
is built only from models that exist in `MODEL_SPECS`, so a hit there proves spec↔eval
wiring — it is the better check once used correctly.

## C. Device hygiene, learned the hard way

Killing a server while it holds the mesh leaves the fabric in a state where the next run
dies in `RiscFirmwareInitializer::assert_active_ethernet_cores_to_reset` or times out
waiting for an ethernet core. The recovery sequence that works:

1. stop the client, then the runner, then the server — by **PID**, in that order;
2. wait for the server process to actually exit (poll, do not assume);
3. `tt-smi -r` by absolute path;
4. **settle before opening the mesh** — a reset returns before the ethernet cores finish
   re-training. `mesh_smoke.sh` now sleeps 60 s by default before opening, plus 20 s after
   closing, on top of whatever the caller slept. Opening too early yields the ethernet-core
   timeout and reads as a hardware fault rather than impatience.

This cost one wedged mesh and one misdiagnosis before the settle was added.

## D. Configuration traps that belong to the release flow, not to me

Cross-referenced here so this file is a single index of "things that will waste your time":

- **`TT_MESH_GRAPH_DESC_PATH` is relative** and only resolves when tt-metal is nested inside
  the tt-inference-server tree — `CI_FAITHFUL_RUN.md`.
- **`trace_region_size: 1 GB` OOMs** with this implementation — `CI_FAITHFUL_RUN.md`.
- **`--vllm-dir` is deprecated and ignored**; the plugin is picked up via an *editable*
  install into tt-metal's `python_env`, which is what actually determines which tree serves
  — `CI_FAITHFUL_RUN.md`.
- **lm-eval's default request timeout is 1800 s** and tt-inference-server never overrides it
  — `CI_FAITHFUL_RUN.md`.
- **A per-model `.gitignore` does not survive a branch switch**, leaving another model's
  ignored artifacts untracked *and* unignored — 1,293 MiB of `.tracy` captures in one
  observed case.

## E. What is still not instrumented

Stated so it is not mistaken for coverage:

- The monitors' "server ready" detection never matched this server's log format, so they
  reported `server:loading` while an eval was demonstrably running. Harmless because
  completion and failure detection were separate, but it means "loading" in those logs
  cannot be trusted as a state.
- No check anywhere asserts that a generation is *well formed*. Every quality judgement in
  this directory came from reading samples by hand after the fact. That gap is the same one
  `BATCH32_DEGRADATION.md` identifies in CI, and it applies to my own tooling too.
