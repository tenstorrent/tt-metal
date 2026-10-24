# perf_automation — fixes plan

Scope: 5 verified bugs + the SDK→Claude Code migration.
All findings below were verified against code and/or reproduced on 2026-07-25 (llama3_1_8b_p150 full-pipeline optimize, ACE-Step module-level optimize as the working control).

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done

## Method (non-negotiable)

**Test first, then implement.** For every item below, in this order:
1. Write the failing test that reproduces the bug against current code (it must FAIL for the right reason, and the failure message must name the actual defect).
2. Only then change the code.
3. Re-run: the new test passes, and the whole existing suite still passes.
4. Then stress it (see "Stress testing" at the end) before calling the item done.

Do not mark an item `[x]` on a static read of the diff — every item needs its test green plus its stress scenario clean. Tests must be hermetic (stubbed agent, stubbed device) so they run without hardware; the on-device checks are separate and listed under Validation.

---

## BUG 1 — `· wedged` is an overloaded bucket (mislabels non-wedges, resets the board)

**Where:** `cc_optimize/perf_mcp.py:1072` (catch-all in `measure_candidate`)

```python
except Exception as exc:
    ...only dram-overflow / L1-overflow handled specially...
    _note_device_crash("measure_candidate")      # 2 in a row -> tt-smi reset
    _autorecord_wedge(f"wedged/crashed when tried: {_msg[-300:]}")
```

**Evidence:** the llama RUN_REPORT prints `· wedged  wedged/crashed when tried: unexpected CSV header ... '\n'` — a host-side CSV parse failure recorded as a device wedge. The same `· wedged` label also covers watchdog timeouts (`round killed (UNPRODUCTIVE 2400s)`) and genuine device wedges, so three unrelated failures are indistinguishable in the report.

**Impact:** lever is burned and never retried; board is reset for a host-side parse error; report reads as "tried and failed on merit" when nothing was measured.

**Fix**
- [ ] Catch `TracyRunError` / CSV-parse failures separately from device crashes.
- [ ] New verdict `MEASUREMENT_FAILED` (reason: `csv_unparseable` | `zero_ops` | `no_csv`) that: does not call `_note_device_crash`, does not `_autorecord_wedge`, does not consume the lever (stays untried), and does not feed the "no lever beat baseline" conclusion.
- [ ] Reserve `wedged` for real device wedges (crash signatures / hang / freeze detector).
- [ ] Report: render unmeasured attempts in a separate "unmeasured" section, not as `· wedged` with `ms`/`gain` dashes.

---

## BUG 2 — `knob:*` levers mis-bucketed into the `host` column  [x] IMPLEMENTED 2026-07-25

**Where:** `cc_optimize/summary.py:73`

```python
def _level_of(kind: str) -> str:
    k = (kind or "").lower()
    if k in ("grid", "dtype", "fidelity", "shard", "tt-lang", "cpp"):
        return k
    if k in _HOST_KINDS:
        return "host"
    return "host"          # <-- anything unrecognised silently becomes "host"
```

**Evidence:** in the same report, the per-attempt row says lever `knob:grid` while the lever matrix marks `·wedge` under **host** — the report contradicts itself. `"knob:grid" != "grid"`, so the exact-match fails and the catch-all misfiles it. Same class as the ACE report crediting 5 `knob:shard`/dtype/fusion wins to `tt-lang`.

**Impact:** wrong lever attribution; `host` shows activity that never happened; `grid`/`dtype`/`fidelity`/`shard` show `—` ("not attempted") when they were attempted; typos and new lever names are swallowed.

**Fix — the Claude Code agent classifies EVERY attempt (no string matching)**

Decided 2026-07-25. String matching is unfixable here for two reasons the tool cannot self-detect:
1. A wrong match looks like a successful one — `knob:grid` resolved to `host` and nothing flagged it, so a "classify only unknown names" scheme never fires.
2. A label can match a real column and still be semantically wrong — ACE rows carried `tt-lang` in the lever column while the note read `knob:shard — L1-pinned the (hidden,2I) weight`. The label lied; only the *description of the change* held the truth.

So the column must be derived from what the attempt actually DID, not from its name.

- [ ] Replace `_level_of()` with an agent classifier: input = `(kernel_kind, note/description, op_signature)` per attempt; output = one of `_LEVEL_COLS` or `other` + a one-line reason. The `kernel_kind` string is a HINT only, never authoritative.
- [ ] **One batched call per report render** — classify all attempt rows in a single `claude -p` invocation (a report has tens of rows, not thousands), so cost is one call per render, not per row.
- [ ] Cache by content hash of `(kernel_kind, note)` so re-renders and resumed runs reuse prior decisions and stay deterministic; cache lives with the knob catalog.
- [ ] Give the classifier the column semantics explicitly (grid = core-grid occupancy/sharding of work across cores; dtype = weight/activation precision; fidelity = math fidelity; shard = memory sharding/L1 pinning; host = host-side/dispatch work; tt-lang / cpp = custom kernel authoring) so it can disambiguate overlapping cases.
- [ ] Emit the classifier's reason into the report row, so a human can audit any surprising attribution.
- [ ] Fallback when the agent is unavailable: `other` + log the raw `kernel_kind`. Never silently return `host`; delete that catch-all.
- [ ] Regression (with a stubbed classifier so tests stay hermetic): a row whose label says `tt-lang` but whose note describes L1 weight-pinning must land in `shard`, not `tt-lang`; `knob:grid` must land in `grid`; an invented lever name must be classified from its note rather than defaulted; a second render must hit the cache with no agent call.

---

## BUG 3 — false "no net speedup / may already be at its ttnn floor"  [x] IMPLEMENTED 2026-07-25

**Where:** report writer (`cc_optimize/summary.py`, limitations block)

**Evidence:** ACE's `ace_step_audio_tokenizer` section reports `baseline 33.08 -> final 11.82 ms (+64.3%, 2.80x)` and `committed wins: 21`, and still prints `- No net speedup recorded — the model may already be at its ttnn floor, or the dominant op needs a custom kernel.` The llama report printed the same line while at **22% of floor with 521.85 ms of headroom** and zero valid measurements.

**Impact:** the single most misleading line in the artifact — it tells a reader the model is optimised when it was never measured (or was improved 2.8x).

**Fix**
- [ ] Only emit the "no net speedup" line when there was at least one **valid** measurement and none beat baseline.
- [ ] When there were zero valid measurements, emit instead: "no lever was successfully measured — results inconclusive" plus the count of unmeasured attempts and why.
- [ ] Never emit the "at its ttnn floor" clause when `at-floor %` is low or the floor is stale/absent.

---

## BUG 4 — round watchdog is NOT adaptive (all timers must be)

**Owner ask: every timer in optimize should be adaptive so small AND big models are covered.**

**Where:**
- adaptive already: `agent/probes.py:30 adaptive_backstop(floor_default=3600, mult=3, env_key="PERF_MCP_MEASURE_BACKSTOP")` → `min(ceil, max(floor, 3*base))`, `base` = `tracy_baseline` seconds from `events.jsonl`, `ceil` = manifest `config.timeout`. Used by `agent/pcc_runner.py:55` and `cc_optimize/perf_mcp.py:1187`.
- NOT adaptive: `cc_optimize/run.py:1783 stall_sec = int(os.environ.get("PERF_MCP_ROUND_STALL_SEC", "600"))`; the `UNPRODUCTIVE 2400s (hard cap)` derives from it.

**Evidence (reproduced 2/2 rounds, 2026-07-25):** `WATCHDOG: round UNPRODUCTIVE 2400s — alive but no commit/kernel attempt in that time (hard cap) — killed the round ... (killed holders none)`. Nothing failed: 0 CSV errors, no crash, no device holders. One `check_pcc` on llama-8B shells out to the whole `simple_text_demo` (minutes/call), so edit -> check_pcc -> measure_candidate -> commit cannot fit in 2400 s. The two timers contradict each other: a measurement may legitimately take 3600 s while the round dies at 2400 s. ACE never hit this because per-module PCC gates are seconds (401 tool calls on one module vs ~3 per 20 min on llama).

**Fix — minimum**
- [ ] Derive `stall_sec` (and the 4x hard cap) from the same `base` the backstop uses, e.g. `stall_sec = clamp(k * base, floor, ceil)`.
- [ ] Audit every remaining constant timeout in `run.py` / `perf_mcp.py` / `probes.py` and route them all through one adaptive helper.
- [ ] Scale automatically when the run is full-pipeline with an expensive e2e-only gate — discovery already emits `no_component_level_tests`.
- [ ] Document the trap: passing `PERF_MCP_MEASURE_BACKSTOP=<n>` hits the override branch and **disables adaptivity entirely** (setting `=900` on llama-8B made every PCC call report `timed out after 900 seconds`).

**Audit result (2026-07-25). Two separate gaps: EFFECTIVENESS and COVERAGE.**

*Correction to earlier notes: the round cap is NOT hard-coded — `run.py:1388 _round_hard_cap` calls `_adaptive_cap(repo_root, max(stall_sec*4, 2400))`, and the watchdog already separates `stall_sec` (FROZEN: no sign of life) from `max_no_progress` (UNPRODUCTIVE: alive, no commit). That design is sound.*

**(a) EFFECTIVENESS — adaptivity is inert on most models.** All four adaptive paths compute `min(ceil, max(floor, mult * base))` with `base` = the **tracy baseline profile** duration. For llama: `3 * 146.72 = 440 s`, which loses to every floor (2400 / 3600), so all four return their floor. Adaptivity therefore does nothing unless a model's baseline profile exceeds ~800 s. Worse, `base` is the wrong quantity: a *round* is edit -> `check_pcc` (full 8B demo, minutes) -> `measure_candidate` -> commit, so scaling a round budget from a profiling measurement compares unlike things.
- [ ] Derive each timer from the SAME operation class it governs: round cap from observed cycle cost (`check_pcc` + `measure_candidate`, both already timed), measure backstop from observed profile durations, build stall from observed build durations.
- [ ] Also scale from HARDWARE facts already in the manifest `env` block (`worker_cores`, `device_count`, `arch`, host cpu count) — e.g. rebuild time ~ 1/host_cores, tt-smi ~ chip count, collect ~ model import cost.

**(b) COVERAGE — 13 gates never routed through the helper.** Pattern: **backstops are adaptive, stalls are not.** Every `*_STALL_SEC` is a plain constant while its paired backstop is adaptive.

| Site | Value | Gates |
|---|---|---|
| `sdk_retry.py:34` | `AGENT_CALL_TIMEOUT_S=300` | EVERY SDK agent call (plan/select/strategist/progress/promote) — worst offender |
| `perf_test_agent.py:24` | `COMPONENT_RUN_TIMEOUT_S=240` | each perf-test build run on device |
| `run.py:284,353,411` | `MEASURE_STALL_SEC=600` | gate / coverage / full-pipeline stall |
| `run.py` (`stall_sec`) | `ROUND_STALL_SEC=600` | round FROZEN threshold |
| `run.py:183` | `DISCOVER_STALL_SEC=1200` | perf-test build stall |
| `structural_agent.py:18` | `AGENT_DEVICE_CALL_TIMEOUT_S=3600` | device-tool agent call |
| `matmul_sweep.py:265` | `900` | sweep per-candidate |
| `probes.py:240` | `1200` literal | cc discovery subprocess |
| `tracy_tool.py:257` | `600` literal | tracy post-process (scales with trace size — 450 MB device log here) |
| `probes.py:728,799` | `120` literal | pytest collect / preflight (scales with model import cost) |
| `profiler_heal.py:130` | `5400` literal | libtt_metal rebuild (scales with host cores) |
| `gitio.py:20` | `300` literal | every git subprocess (scales with repo/artifact size, disk) |
| `probes.py:88,493,564,614` | `30-300` literal | tt-smi (scales with chip count) |
| `probes.py:575` | `COOL_MAX_S=120` | thermal cooldown (scales with board) |

**Principle to adopt: NO bare constants anywhere.** Every timeout derives from a measured workload duration and/or a probed hardware fact, with a floor, and is logged once with its inputs. Nothing is "model-independent" in practice — tt-smi scales with chip count, rebuild with host cores, collect with import cost, git with repo size.

**Safety rule that makes this robust: a timeout must never decide correctness.** Make timeouts generous and pair them with a PROGRESS signal (no output + no CPU + no transcript growth), which today only the round watchdog has. Then an over-long timeout costs a little wall clock, never a wrong verdict — and a genuine hang is still caught quickly by the progress check.

**(c) SMALL models are covered WORSE than big ones — flag + must-check.** The absolute floors are the de-facto policy at every realistic size, and one constant cannot serve both ends. Measured with the real formula `min(ceil, max(floor, 3*base))`:

| module / model | base | 3*base | measure backstop (floor 3600) | round cap (floor 2400) | backstop / work |
|---|---|---|---|---|---|
| ACE `ace_step_encoder_layer` | 3.16 s | 9 s | **3600 s (FLOOR)** | 2400 s (FLOOR) | **1139x** |
| ACE `ace_step_di_t_model` | 5.41 s | 16 s | 3600 s (FLOOR) | 2400 s (FLOOR) | 665x |
| ACE `ace_step_di_t_layer` | 13.06 s | 39 s | 3600 s (FLOOR) | 2400 s (FLOOR) | 276x |
| ACE `ace_step_audio_tokenizer` | 42.61 s | 127 s | 3600 s (FLOOR) | 2400 s (FLOOR) | 84x |
| llama3_1_8b_p150 full pipeline | 146.72 s | 440 s | 3600 s (FLOOR) | 2400 s (FLOOR) | 25x |
| hypothetical huge model | 900 s | 2700 s | 3600 s (FLOOR) | 2700 s (adapts) | 4x |

Consequences, opposite at each end:
- **Small models:** a 3 ms module gets a **1-hour** measurement backstop (1139x the real work), so a genuine hang wastes an hour before detection. ACE survived only because the **FROZEN 600 s** progress detector caught its real freeze — the backstop never would have.
- **Big models:** the same floor is too tight — llama's round needed more than 2400 s and was killed twice.
- `mult * base` never wins below ~800 s baseline, so adaptivity is inert for essentially everything currently run.

- [ ] Make the FLOOR proportional too, not just the multiplier: `timeout = clamp(mult * base, k_lo * base, k_hi * base)` plus a small absolute minimum for measurement noise — so a 3 s module gets tens of seconds (hang caught fast) and an 8B round gets what it needs.
- [x] RED TEST WRITTEN + RUN (2026-07-25): `models/experimental/perf_automation/tests/test_adaptive_timers_coverage.py` exercises the REAL `probes.adaptive_backstop` and `run._round_hard_cap` across 7 model sizes with synthetic manifest/events fixtures. Result: **5 of 7 model kinds mis-served** — base 3.16/5.41/13.06/42.61 s all get the 3600 s floor (1139x/665x/276x/84x the work); base 146.72 s (llama) gets round cap stuck at 2400 s. Adaptivity only engages at base >~1200 s. This test must go GREEN after the fix.
- [ ] Keep the progress signal (FROZEN) as the true hang detector at every size, so a generous timeout never costs correctness — only a little wall clock.

---

## BUG 4 — AGREED DESIGN (validated 2026-07-25): Claude Code watchdog on full evidence, zero hardcoded constants

Benchmarked on generated scenarios with a held-out split (36 calibration kept unused, **84 held-out** scored). Ground truth from the generator, deciders never see labels. Classes: healthy / hung_flat (dead) / zombie (tiny CPU trickle, no progress) / spin (log grows, same action repeated).

| decider | correct | acc | false KILL | false WAIT |
|---|---|---|---|---|
| current fixed timers | 59/84 | 70% | 0 | 25 |
| agent, curated stats only | 71/84 | 85% | 0 | 13 |
| agent, curated + derived bounds | 82/84 | 98% | 0 | 2 |
| **agent, FULL evidence** | **84/84** | **100%** | **0** | **0** |
| agent, FULL evidence + derived bounds | 84/84 | 100% | 0 | 0 |

Per class with full evidence: healthy 28/28, hung_flat 18/18, spin 19/19, zombie 19/19.
Earlier 30-scenario suite (snapshot evidence) for reference: fixed 22/30 with **4 false kills**; agent 23/30; agent+guardrails 26/30 with 0 false kills.
Cost: ~0.9 s per decision (8 parallel); a watchdog check runs every few minutes, so cost is negligible.

### Specification

1. **The Claude Code agent decides continue/kill.** No numeric threshold decides a verdict.
2. **It receives the FULL raw evidence, not summaries.** This was worth +15 points (85% -> 100%) on its own:
   - the actual observed duration samples for this op on this model (sorted), plus p50/p95/p99
   - device CPU jiffies per window and transcript bytes per window (5 windows, oldest -> newest)
   - **the literal action sequence** (e.g. `["Bash(ninja)"] x 11`) — not just counts
   - **the last log lines verbatim** (a repeated `retrying: shard width must match per core N` is self-evident)
   - run context: round number, commits this round, prior attempts on this op, prior wedges
   - hardware: chip count, host cores; profile op_count; device fds held by the run
   - the operator ceiling
   Summary statistics destroy the signal that separates working from stuck: **repetition**. Pass raw evidence.
3. **Derived bounds as a fallback safety net only** — used when the agent is unavailable/errors, and as a backstop: `grace = p95(observed)`, `flat = p99(observed) * (p95/p50)`, `ceiling = manifest config.timeout`. All computed from logged history; the ceiling is the ONLY operator-supplied number. With full evidence the net changed nothing (84/84 either way), so it must never override a confident agent decision — only cover its absence.
4. **No hardcoded timer constants anywhere.** Delete the fixed values catalogued in (b); every bound is derived from the observed duration history for that operation class on that model (+ hardware facts where relevant).
5. **Novelty is a first-class signal.** Hash each tool call / edit; report total vs distinct actions in the window AND the sequence. This alone fixed all spin-loop misses (21/24 -> 20/20 in the intermediate run).
6. **Log the decision and its evidence** so any kill is auditable after the fact.

### Why not deterministic arithmetic (settled, with numbers)
- Fixed timers: 45 of 105 (timer x model-size) combinations broken — simultaneously too tight for big models (240 s `component_run` cap vs llama's real 872 s build; 600 s FROZEN vs its 1400 s PCC gate) and absurdly loose for small ones (3600 s for 6 s of work = 600x).
- A proportional-arithmetic variant I tried scored **6/30 with 5 false kills** — worse than what ships. Rejected.
- Arithmetic cannot distinguish host-bound-quiet (compile / weight load / thermal cooldown / device reset / git op / API backoff / JIT) from dead, nor zombie trickle and spin loops from progress. Those four classes are where fixed timers lose 25 of their 25 misses.

### Tests (must exist before implementing)
- [ ] Port the harness to `tests/` with a stubbed agent so it is hermetic; assert **0 false kills** as a hard gate and >=95% accuracy on the held-out split.
- [ ] Keep `tests/test_adaptive_timers_coverage.py` (already red) and make it green.
- [ ] Regression: assert no bare numeric timeout is reintroduced at the catalogued call sites.
- [ ] Small-model case: a 3 s baseline must NOT receive a 3600 s budget; a hang on a micro module must be caught in tens of seconds.
- [ ] Big-model case: llama's round must survive a legitimate 1400 s `check_pcc` without being killed.

**Order of work (important):** fix (a) and (c) first — they are the same root issue (floors dominate, `base` is the wrong quantity). Routing the 14 sites onto a helper whose `base`/floors are wrong changes nothing.

- [ ] Sweep the codebase for EVERY hard-coded timeout/cap and route each through the existing adaptive helper. Known so far: `run.py:1783` `PERF_MCP_ROUND_STALL_SEC=600` + its 4x cap; `perf_test_agent.py:24` `_COMPONENT_RUN_TIMEOUT_S=240`; `sdk_retry.py:34` `_DEFAULT_TIMEOUT_S=300`; `structural_agent.py:18` `_DEVICE_CALL_TIMEOUT_S=3600`; discover/discover-stall/measure-stall defaults in `run.py`. Produce the full list first, then convert.
- [ ] Add a regression test that FAILS if any new bare numeric timeout is introduced (grep/AST check over the timer call sites), so coverage cannot silently regress again.
- [ ] Log each derived timer once at round start so the operator can see what was chosen and from what `base`.
- [ ] Never silently disable adaptivity: if an env override pins a timer, log it as a warning (the `PERF_MCP_MEASURE_BACKSTOP=900` trap above).

**Deferred (only if the above proves insufficient — do NOT do preemptively):** per-operation-class budgets, in-run p95 learning, progress-based liveness replacing the wall-clock cap, persisted per-model timings. These are refinements to a design that is otherwise sound; revisit only if wiring adaptivity through still starves a real run.

---

## BUG 5 — zero-op profile is fatal instead of retried (+ upstream tt-metal defect)

**Upstream (tt-metal):** `tools/tracy/process_ops_logs.py:1328-1344`

```python
csv_row_headers = set()
for row in rowDicts: ...            # zero rows -> stays empty
with open(allOpsCSVPath, "w") as allOpsCSV:
    allHeaders = []
    for header in OPS_CSV_HEADER + PERF_COUNTER_CSV_HEADERS:
        if header in csv_row_headers: allHeaders.append(header)
    writer = csv.DictWriter(allOpsCSV, fieldnames=allHeaders)
    writer.writeheader()            # empty fieldnames -> writes exactly b'\r\n'
logger.info(f"Device only OPs csv generated at: {allOpsCSVPath}")   # reports SUCCESS
```

**REPRODUCED offline (2026-07-25):** device log present with zero op rows produced `ops_perf_results.csv: 2 bytes, first line = b'\r\n'` while logging "generated at" and returning OK. In text mode `readline()` returns `'\n'` — byte-for-byte the `unexpected CSV header ... : '\n'` in the llama report. Also falsified the neighbouring case: with **no** device log it writes no CSV at all (different error path).

**Downstream (perf_automation):** `agent/probes.py:946 _validate_csv(...)` receives that file and raises, which BUG 1 then mislabels.

**Fix**
- [ ] perf_automation: gate on op-row count > 0 after each profile; if zero, **re-profile** (1-2 retries, fresh out_dir) before treating it as anything. This alone would have saved all 9 llama attempts.
- [ ] perf_automation: if it persists, return `MEASUREMENT_FAILED(zero_ops)` per BUG 1 and tell the agent plainly ("profiler captured no ops; your edit was NOT measured") instead of an opaque CSV error.
- [ ] Pre-flight per profile: no stale `tracy-capture` holding a port; fresh output dir.
- [ ] Consider explicit tracy signposts around the measured region (this run warned `no tracy signposts ... using default 'start'/'stop' (full capture)`).
- [ ] Test the drain knob: the generated perf test drains the device profiler every 32 ops (`TT_PERF_FLUSH_EVERY=32`); try `0` for candidate measurements. HYPOTHESIS, not established.
- [ ] Upstream tt-metal: write the full header unconditionally (valid CSV, zero rows) or fail loudly with a non-zero exit; never log success on an empty report.

### OPEN QUESTION (blocked on evidence)
Why did those 9 candidate profiles yield zero op rows at all? Eliminated: Tracy compiled in (`ENABLE_TRACY=ON`, on by default), `TT_METAL_DEVICE_PROFILER=1` set per profile (`probes.py:843`), fresh out_dir per attempt, no `PERF_MCP_PROFILE_ENV` injection in that run (depth bridge said "ignoring"), nothing killed by the operator. The deciding artifact is the `run0_tracy.log` from a zero-op attempt; those were deleted with their `/tmp/perf_mcp_*` dirs, and the failure has not recurred (0 in ~2 h on a clean tree).
- [ ] Add capture-on-failure: on CSV/zero-op failure, snapshot the malformed CSV + full `run0_tracy.log` + the `.logs/` inputs to a durable dir before cleanup, so one recurrence is diagnosable and replayable offline.

---

## CLEANUP (separate from the 5 bugs) — post-rebase test failures

After rebasing the branch onto latest tt-metal (991 commits, 2026-07-25), the perf_automation suite is **606 passed / 21 failed**. Proven NOT caused by the BUG 2/3 fixes: A/B with and without the changed `summary.py` gives an identical **9 failed / 45 passed** on the affected files. Two distinct classes:

**(i) ~9 genuine rebase fallout — fail in isolation too**
- [ ] `tests/test_tp_ladder.py` (2) · `tests/test_gap_a_host_signal.py` (2) · `tests/test_exit_policy.py` (1) · `tests/test_apply.py` (1) · `tests/test_model_files.py` (1) · `tests/test_engine.py` (KeyError at :314) · `tests/test_trace_fix_retry.py` (7)
- Upstream changed behaviour these assert on; each needs its expectation re-derived against the new tt-metal, not silenced.

**(ii) ~12 test-isolation pollution — pass alone, fail in full-suite order**
- [ ] `tests/test_config.py` passes **12/12 alone** but fails at :30 and :107 in a full run. Some earlier test leaves global state behind (env var / imported module singleton).
- Root pattern already seen once and fixed: a test set `PERF_MCP_MANIFEST` to a temp path that was then deleted, and `cc_optimize/perf_mcp.py:45` reads that env var **at import time**, so 18 later files failed collection with `FileNotFoundError`.
- [ ] Harden the source of the fragility too: `perf_mcp.py` should not read/parse the manifest at module import; make it lazy so a stale env var cannot break unrelated collection.
- [ ] Add an isolation guard: an autouse fixture that snapshots/restores `PERF_MCP_*`, `TT_PERF_*`, `AGENT_*` env vars around every test.

**Housekeeping**
- [ ] Benchmark harnesses must live in `models/experimental/perf_automation/benchmarks/`, never `tests/` — pytest imports everything under `tests/` and would execute them. (Already moved: 9 scripts.)

---

## MIGRATION — claude_agent_sdk -> Claude Code (`claude -p`)

**Already on `claude -p`:** `cc_optimize/run.py`, `cc_optimize/perf_mcp.py`, `agent/perf_test_gen.py` (the optimize engine itself — `run.py:1787` `[_resolve_claude_bin(), "-p", prompt, "--mcp-config", ..., "--strict-mcp-config", "--allowedTools", ..., "--output-format", "stream-json"]`).

**Still on the SDK (13 files):** `plan_agent.py`, `edit_agent.py`, `perf_test_agent.py`, `structural_agent.py`, `select_agent.py`, `strategist.py`, `progress_agent.py`, `promote.py`, `probes.py` (discovery sub-agent), `perf_check.py`, `edit_check.py`, `sdk_health.py`, `before_loop.py`.

**Key design blocker:** `perf_test_agent.py:83` and `edit_check.py` expose **in-process** MCP servers via `create_sdk_mcp_server` (`run_perf_test`, `check_candidate_edit`). `claude -p` can only reach **external** MCP servers via `--mcp-config`, so those tools must be extracted into standalone servers (the pattern `perf_mcp.py` already uses).

**Phase 1 — the two that cost us today**
- [ ] `perf_test_agent.py` (the perf-test builder): extract `run_perf_test` into an external MCP server; spawn via `claude -p ... --mcp-config`; parse `stream-json`. Removes the hardcoded haiku default (`config.py:25 "edit": "claude-haiku-4-5-20251001"`) that produced the truncated 2-layer perf test.
- [ ] `plan_agent.py`: read-only (Read/Grep/Glob), no MCP — straight swap to `claude -p`. Also fix the failure path so a PLAN error **skips the lever** instead of letting APPLY improvise.

**Phase 2 — tool-less swaps (mechanical)**
- [ ] `select_agent.py`, `strategist.py`, `progress_agent.py`, `promote.py`, `probes.py` discovery sub-agent.
- [ ] Pass `select_reasoning` through to PLAN while here — SELECT correctly diagnosed "TopK 620 ms single-core, apply TILE->ROW_MAJOR" and only the bare lever name reached PLAN.

**Phase 3 — MCP-bearing**
- [ ] `edit_agent.py` + `edit_check.py` (`check_candidate_edit` -> external server).
- [ ] `structural_agent.py`, `perf_check.py`.

**Cross-cutting**
- [ ] Single spawn helper (bin resolution, `--mcp-config`, `--strict-mcp-config`, allowed tools, `stream-json` parsing, retry/timeout) so all agents share one path.
- [ ] Keep `sdk_retry.run_with_retry`'s bounded-wait semantics (per-attempt timeout + transient retry) in the new helper; wire its timeout to the adaptive helper from BUG 4.
- [ ] Delete `config.py` model-ladder defaults that no longer apply once agents run on the operator's `claude` login; keep env overrides.
- [ ] Retire `sdk_health.py`'s SDK check in favour of a `claude` CLI + auth check.

---

## Suggested order

1. BUG 2 (one function, no device) and BUG 3 (report honesty) — smallest, make artifacts trustworthy immediately.
2. BUG 1 + BUG 5 retry — together they convert a transient profiling failure from "0 wins, board reset" into a retried measurement.
3. BUG 4 adaptive timers — unblocks big-model full-pipeline runs (currently 2/2 rounds lost).
4. Capture-on-failure instrumentation — closes the open question on the next recurrence.
5. MIGRATION phases 1 -> 3.

## Validation

- Unit-testable without hardware: BUG 2, BUG 3, BUG 5 upstream repro (`process_ops_logs.py` with a header-only device log), BUG 1 verdict routing.
- Needs device: BUG 4 (a full-pipeline round must reach a commit), MIGRATION phase 1 (builder must reach `PASS_TRACE`).
- Control case to re-run for regression: ACE-Step `--module-level` (35 committed wins, 0 CSV errors) must stay green.

---

## Test-first: the failing test to write BEFORE each fix

| Item | Test that must FAIL on current code |
|---|---|
| BUG 1 | Feed `measure_candidate` a `TracyRunError("unexpected CSV header ... '\n'")`; assert verdict is `MEASUREMENT_FAILED(csv_unparseable)`, `_note_device_crash` NOT called, no `_autorecord_wedge`, lever still untried. Currently returns a wedge + marks a device crash. |
| BUG 2 | Row with `kernel_kind="tt-lang"` + note `"knob:shard — L1-pinned the (hidden,2I) weight"` must classify as `shard`. And `kernel_kind="knob:grid"` must classify as `grid`. Currently -> `tt-lang` and `host`. |
| BUG 3 | Render a report with (a) zero valid measurements and (b) one 2.80x win; assert the "no net speedup / may be at its ttnn floor" line is absent in both. Currently printed in both. |
| BUG 4 | With `base=146s` (llama baseline), assert derived round `stall_sec` >> 600 and that a simulated cycle of `edit + check_pcc(600s) + measure(300s) + commit` is NOT killed. Currently killed at the fixed 2400s cap. |
| BUG 5 | Point the profile path at a header-only device log (repro already built); assert the tool RE-PROFILES on zero rows instead of raising `unexpected CSV header`, and that after N retries it returns `MEASUREMENT_FAILED(zero_ops)` with the agent told "your edit was NOT measured". |
| MIGRATION | For each migrated agent: stubbed `claude -p` returning well-formed `stream-json` -> parsed result equals the old SDK path's result on the same input (golden-output equivalence), and a malformed/truncated stream is handled as a retryable transient, not a crash. |

---

## Stress testing (required before any item is `[x]`)

**BUG 1 + BUG 5 — measurement-failure robustness**
- [ ] 200 synthetic profile results mixing valid CSVs, header-only CSVs, 0-byte CSVs, missing CSVs, truncated CSVs, and CSVs whose write is still in flight (the mid-write race that produced my own false positive). Assert: no false `wedged`, no `tt-smi` reset from any parse failure, every zero-op case retried, and every lever left retryable.
- [ ] Interleave a REAL device crash signature among them; assert it is still classified as a genuine wedge (no over-correction).

**BUG 2 — classifier stress**
- [ ] 500 lever names/notes including: `knob:*` and `rung:*` prefixes, invented names, labels that contradict their note, empty note, 8 KB note, unicode, near-duplicates (`grid` vs `core-grid` vs `full-grid`). Assert every row lands in a real column or `other` — never `host` by default.
- [ ] Determinism: classify the same 500 twice; second pass must be 100% cache hits with zero agent calls and identical output.
- [ ] Adversarial: a note that plausibly fits two columns must still produce a stable choice plus a reason string.

**BUG 3 — report honesty matrix**
- [ ] Cross-product render: {0, 1, many} valid measurements x {0, some} wins x {absent, stale, fresh} roofline floor. Assert the conclusion line is correct in all cells and never claims "at floor" when `at-floor %` is low.

**BUG 4 — adaptive timers across model sizes**
- [ ] Sweep `base` from 0.5 s (tiny module) to 600 s (8B full pipeline); assert every derived timer scales monotonically, clamps to `[floor, config.timeout]`, and that a small model keeps a tight watchdog while a big one gets a proportionally long round.
- [ ] Assert `PERF_MCP_MEASURE_BACKSTOP=<n>` still pins (override wins) and that this is logged loudly as "adaptivity disabled".

**MIGRATION — agent-layer soak**
- [ ] Run each migrated agent 20x back-to-back; assert no leaked `claude` processes, no orphaned MCP servers, no fd/port leaks, and bounded memory.
- [ ] Kill the `claude` child mid-stream; assert the parent recovers as a transient and retries rather than hanging (the old SDK path froze here).
- [ ] External MCP servers (`run_perf_test`, `check_candidate_edit`): assert clean startup/shutdown per invocation and no port collisions when two agents run in sequence.

**End-to-end soak (on device, after unit + stress are green)**
- [ ] ACE-Step `--module-level` regression: must still reach comparable committed wins with 0 CSV errors and 0 false wedges.
- [ ] llama3_1_8b_p150 full-pipeline: at least one round must reach a real `measure_candidate` -> commit/revert decision (i.e. the timer no longer starves it), with every attempt in the report carrying either a real `ms` or an explicit "unmeasured" reason.
- [ ] Long soak: repeat the full optimize twice in a row on the same tree; assert no state leakage between runs (stale `/tmp/perf_mcp_*` baselines, profile cache, leftover worktrees).
