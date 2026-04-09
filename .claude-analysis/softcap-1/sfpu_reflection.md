# SFPU Reflection: softcap

## Metadata
| Field | Value |
|-------|-------|
| Operation | `softcap` |
| Math Definition | `cap * tanh(x / cap)` |
| Output Folder | `.claude-analysis/softcap-1/` |
| Pipeline Phases Executed | 1, 2, 3, 4, 5 |
| Agents Invoked | generator, discoverer, 5x analyzer, implementor, tester, impl-notes |
| Total Git Commits | 9 (f7a0d964e4 through 25dbf41523) |
| Total Pipeline Duration | ~64 minutes (18:13:53 to 19:18:08 UTC) |
| Overall Result | SUCCESS |

---

## 1. Pipeline Execution Summary

### Phase Timeline

| Phase | Agent(s) | Duration | Status | Key Observations |
|-------|----------|----------|--------|------------------|
| 1: Reference Discovery | discoverer | ~5m 28s | OK | 5 references selected; clean run |
| 2: Reference Analysis | 5x analyzer | ~13m 22s (wall) | OK | 5/5 succeeded; 2 agents did not commit (orchestrator committed on their behalf) |
| 3: Implementation | implementor | ~16m 31s | OK | 12 layers completed; single pass |
| 4: Testing & Debugging | tester | ~24m 25s | OK | 1 iteration; tester self-fixed build errors + ICE before running tests; 25/25 PASS |
| 5: Documentation | impl-notes + generator | ~2m 30s (notes) + ~50s (final) | OK | Notes enriched with full source code |
| **Total** | | **~64m** | | |

### Agent Duration Breakdown

| Agent | Start Time | End Time | Wall Duration | Iterations | Notes |
|-------|------------|----------|---------------|------------|-------|
| generator (orchestrator) | 18:13:53 | 19:18:08 | ~64m 15s | - | Entire pipeline |
| discoverer | 18:15:08 | 18:19:12 | ~4m 4s | - | Clean run |
| analyzer (tanhshrink) | 18:20:53 | ~18:31:24 (commit) | ~10m 31s | - | Committed by agent |
| analyzer (swish) | 18:21:05 | 18:28:06 | ~7m 1s | - | Committed by agent |
| analyzer (hardshrink) | 18:21:04 | ~18:33:59 (commit) | ~12m 55s | - | Committed by orchestrator |
| analyzer (atanh) | ~18:20:00 | ~18:23:59 (commit) | ~4m | - | Committed by agent (earliest) |
| analyzer (sinh) | ~18:20:01 | ~18:32:47 (commit) | ~12m 46s | - | Committed by orchestrator |
| implementor | 18:33:24 | 18:49:55 | ~16m 31s | - | Single pass, all 12 layers |
| tester | 18:50:06 | 19:14:31 | ~24m 25s | 1 test iteration | Significant self-repair before tests ran |
| impl-notes | ~19:14:44 | ~19:16:47 (commit) | ~2m 3s | - | Enriched notes with source code |

**Duration calculation method**: Primary source is orchestrator breadcrumb timestamps (`phase_start` / `phase_complete`). Agent-level start/end from individual breadcrumb files (`start` / `complete` events). Git commit timestamps used to cross-check and fill gaps where agent breadcrumbs are absent (implementor, tester).

### Duration Visualization

```
Phase 1  |████|                                                          (~5m)
Phase 2       |████████████|                                             (~13m)
Phase 3                     |███████████████|                            (~16m)
Phase 4                                      |███████████████████████|   (~24m)
Phase 5                                                               |██| (~3m)
         0    5    10   15   20   25   30   35   40   45   50   55   60  65 min

Longest phase: Phase 4 (~24m) -- tester self-repair dominated (stub headers, SfpuType enum stubs, kernel simplification for register pressure ICE)
```

### Time Distribution

| Category | Duration | % of Total | Notes |
|----------|----------|------------|-------|
| Discovery (Phase 1) | ~5m | 8% | |
| Analysis (Phase 2) | ~13m | 21% | 5 parallel analyzers; wall = slowest (hardshrink ~13m) |
| Implementation (Phase 3) | ~16m | 26% | 12 layers |
| Testing (Phase 4) | ~24m | 38% | 1 iteration but heavy pre-test repair |
| -- Productive (first run) | ~24m | 38% | All time productive (no retries) |
| -- Debugging/retries | ~0m | 0% | Tests passed on first actual run |
| Documentation (Phase 5) | ~3m | 5% | Notes enrichment + final report |
| **Total** | **~62m** | **100%** | |

---

## 2. Implementation Coverage Audit

### Math Definition Fidelity

| Aspect | Status | Details |
|--------|--------|---------|
| Core formula | MATCH | Kernel computes `cap * tanh(x / cap)` using `u = x * inv_cap`, Taylor/exp-based tanh(u), then `result * cap` |
| Conditional branches | CORRECT | `v_if(abs_u < 1.0f)` selects Taylor regime; `v_if(x < 0.0f)` applies sign for exp regime; `v_if(z < -127.0f)` clamps underflow |
| Parameter handling | CORRECT | `cap` parameter decoded via union reinterpretation from `uint32_t param0`; `inv_cap = 1.0f / cap` computed once per SFPU call |
| Edge cases | MATCH | At x=0, Taylor returns 0; at large |x|, exp underflows and tanh saturates to 1.0, result = cap; sign correctly preserved via v_if |

**Math definition from orchestrator**: `cap * tanh(x / cap)`
**Kernel implementation summary**: Dual-regime tanh approximation: degree-5 Taylor series for |u| < 1.0 (where u = x/cap), and 2-term geometric series from exp(-2|u|) for |u| >= 1.0. Result multiplied by cap. Sign applied via `v_if` conditional.

**Note**: The implementation notes document degree-7 Taylor + 3-term geometric as the original design, but the tester simplified this to degree-5 Taylor + 2-term geometric to resolve a register pressure internal compiler error (ICE). The on-disk kernel confirms the simplified version. This is an acceptable deviation -- the degree-5 Taylor with 2/15 coefficient provides sufficient accuracy for bfloat16 precision, as confirmed by the 25/25 test pass with ULP threshold of 10.

### 12-Layer Completeness

| Layer | Description | Expected File(s) | Status | Notes |
|-------|-------------|-------------------|--------|-------|
| 1 | SFPU Kernel | `ckernel_sfpu_softcap.h` (WH+BH) | PRESENT | Both files exist, identical content, 119 lines |
| 2 | LLK Dispatch | `llk_math_eltwise_unary_sfpu_softcap.h` (WH+BH) | PRESENT | Both files exist, identical content, 26 lines |
| 3 | Compute API Header | `softcap.h` | PRESENT | Exists at expected path, 38 lines with documentation |
| 4 | SFPU Include Guard | `sfpu_split_includes.h` | PRESENT | `#if SFPU_OP_SOFTCAP_INCLUDE` with include at line 24-26 |
| 5 | SfpuType Enum | `llk_sfpu_types.h` (WH+BH) | PRESENT | `softcap` enum value present in both arch files |
| 6 | UnaryOpType Enum | `unary_op_types.hpp` | PRESENT | `SOFTCAP` at line 127 |
| 7 | Op Utils Registration | `unary_op_utils.cpp` | PRESENT | 2 of 3 functions: `get_macro_definition` (line 24) and `get_op_init_and_func_parameterized` (lines 43-45). `get_op_approx_mode` falls through to default `return false`, which is correct (no approx mode branching in kernel) |
| 8 | Op Utils Header | `unary_op_utils.hpp` | PRESENT | `is_parametrized_type` returns true for SOFTCAP (line 48) |
| 9 | C++ API Registration | `unary.hpp` | PRESENT | `REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(softcap, SOFTCAP)` at line 173 |
| 10 | Python Nanobind | `unary_nanobind.cpp` | PRESENT | Full binding at lines 1940-1981 with documentation, default cap=50.0 |
| 11 | Python Golden | `unary.py` | PRESENT | Golden function `_golden_function_softcap` at lines 68-76; also registered in `golden_functions.py` |
| 12 | Test File | `test_softcap.py` | PRESENT | 97 lines, 5 test functions, 25 parametrized cases |

**Layer completeness**: 12/12 layers present

### Reference Utilization

| Reference | Analysis Produced? | Cited by Implementor? | Usefulness |
|-----------|-------------------|----------------------|------------|
| tanhshrink | YES | YES (but noted as "not directly used") | MEDIUM -- provided tanh_tile usage patterns, but softcap uses a custom tanh, not tanh_tile |
| swish | YES | YES (SFPI piecewise pattern, v_if workflow) | MEDIUM -- provided SFPI style reference for piecewise computation |
| hardshrink | YES | YES (parameterized float handling context) | LOW -- softcap uses standard eltwise_sfpu.cpp path, not custom compute kernel; parameter pattern from hardshrink was informational but not directly applied |
| atanh | YES | YES (abstraction layer pattern, SfpuType registration) | HIGH -- provided the standard ckernel -> LLK -> API -> split-include registration pattern that softcap followed exactly |
| sinh | YES | YES (exp_21f helper, dual-regime pattern, v_if override) | HIGH -- most useful reference; exp_21f algorithm copied directly, dual-regime (exp + Taylor override) pattern adopted, `#pragma GCC unroll 0` pattern followed |

**References wasted**: 0. All 5 references had analysis produced and were cited in implementation notes. However, hardshrink's utility was marginal since softcap took the standard dispatch path rather than a custom compute kernel.

### Test Coverage

| Metric | Value |
|--------|-------|
| Test file created | YES |
| bfloat16 parametrization | PASS (default dtype; 25/25 tests) |
| fp32 parametrization | NOT RUN (no fp32-specific test cases in test file) |
| Max ULP (bfloat16) | <= 10 (threshold; exact max not logged) |
| Max ULP (fp32) | N/A (not tested) |
| allclose (bfloat16) | PASS (rtol=5e-2, atol=0.35) |
| allclose (fp32) | N/A (not tested) |
| Total test iterations | 1 |
| Final result | PASS (25/25) |

**Finding -- MEDIUM severity**: No fp32-specific test cases. The test file uses `data_gen_with_range` which produces bfloat16 tensors by default. While the implementation notes acknowledge fp32 precision is bfloat16-quality on SFPU hardware, an explicit fp32 parametrization would verify the fp32 code path compiles and runs correctly.

---

## 3. Breadcrumb & Logging Compliance Audit

### Per-Agent Breadcrumb Compliance

| Agent | File Exists? | Event Count | Min Expected | Mandatory Events Present? | Timestamps? | Ordering? | Compliance |
|-------|-------------|-------------|--------------|--------------------------|-------------|-----------|------------|
| generator | YES | 30 | ~27 | PARTIAL -- missing `pipeline_complete` | YES (all entries have `ts`) | YES | PARTIAL |
| discoverer | YES | 5 | 4 | YES -- `start`, `files_read`, `ranking_complete`, `complete` all present | YES | YES | FULL |
| analyzer(s) | YES | 19 | 30 (6x5) | PARTIAL -- see details below | PARTIAL (most have `ts`, some use `"ts"` key inconsistently) | YES | PARTIAL |
| implementor | NO | 0 | 15 | NO -- file does not exist | N/A | N/A | ABSENT |
| tester | NO | 0 | 4+ | NO -- file does not exist | N/A | N/A | ABSENT |
| impl-notes | YES | 1 | 3 | PARTIAL -- single JSON object (not JSONL), missing `notes_read`, `files_collected`, `complete` events | PARTIAL (has `timestamp` not `ts`) | N/A | PARTIAL |

#### Detailed Analyzer Breadcrumb Analysis

The analyzer breadcrumb file contains events from 3 of the 5 analyzers (tanhshrink, hardshrink, swish). The atanh and sinh analyzers did not write their own breadcrumbs (their analysis files were committed by the orchestrator on their behalf, and no breadcrumb entries appear for them in the shared file).

Per-analyzer event coverage:
- **tanhshrink**: `start` (line 1), `nuke_impact_analysis`, `tracing_complete`, `analysis_written` -- missing `dispatch_traced`, `kernel_source_read`, `instruction_analysis_complete`, `complete`
- **hardshrink**: `start` (line 2), `read_unary_op_utils`, `found_compute_kernels`, `traced_sfpu_layers`, `analysis_written` -- missing `dispatch_traced`, `kernel_source_read`, `instruction_analysis_complete`, `complete`
- **swish**: `start` (line 3, 9), `dispatch_traced` (line 10), `kernel_source_read` (line 11), `instruction_analysis_complete` (line 12), `analysis_written` (line 14), `complete` (line 16) -- FULL compliance
- **atanh**: No breadcrumbs at all (analysis committed by agent but no breadcrumb entries)
- **sinh**: No breadcrumbs at all (analysis committed by orchestrator but no breadcrumb entries)

Only swish fully complied. The tanhshrink and hardshrink analyzers used non-standard event names (e.g., `nuke_impact_analysis`, `found_compute_kernels`) instead of the mandatory events. Atanh and sinh produced analysis files but no breadcrumbs whatsoever.

#### Detailed Generator Breadcrumb Analysis

The generator has 30 entries covering:
- `start` (line 1), `pipeline_start` (line 2)
- `phase_start` x6 (phases 1-6, including self-reflection phase 6)
- `subagent_launched` x9 (1 discoverer + 5 analyzers + 1 implementor + 1 tester + 1 self-reflection)
- `subagent_completed` x8 (1 discoverer + 5 analyzers + 1 implementor + 1 tester)
- `phase_complete` x5 (phases 1-5)

**Missing**: `pipeline_complete` event (pipeline was still running when self-reflection was launched, so this is expected for phase 6 being in-progress).

### Execution Log Compliance

| Agent | Log Exists? | Sections Present | Notes |
|-------|------------|------------------|-------|
| generator | NO | N/A | No execution log file exists |
| discoverer | NO | N/A | No execution log file exists |
| analyzer | YES | Session Summary, Execution Steps, Key Findings (x3 sessions: hardshrink, tanhshrink, swish) | Good quality; covers 3 of 5 analyzers. Missing atanh and sinh execution logs. |
| implementor | NO | N/A | No execution log file exists |
| tester | NO | N/A | No execution log file exists |
| impl-notes | NO | N/A | No execution log file exists |

### Logging Infrastructure Issues

| Issue | Severity | Description |
|-------|----------|-------------|
| Missing implementor breadcrumbs | HIGH | `ttnn-unary-sfpu-operation-implementor_breadcrumbs.jsonl` does not exist. The implementor logging spec (`.claude/references/logging/sfpu-operation-implementor.md`) exists and is well-specified, but the implementor agent did not create any breadcrumbs. This means we have zero visibility into which of the 12 layers were implemented in what order, what design decisions were made, and what difficulties were encountered. |
| Missing tester breadcrumbs | HIGH | `ttnn-unary-sfpu-operation-tester_breadcrumbs.jsonl` does not exist. The tester logging spec (`.claude/references/logging/sfpu-operation-tester.md`) exists and is well-specified. Without breadcrumbs, we cannot trace: (1) what build errors occurred, (2) what hypotheses were formed, (3) what fixes were applied (stub headers, kernel simplification), (4) the exact test results per attempt. The final report mentions ~6 fixes applied by the tester, but the fix-by-fix timeline is lost. |
| Impl-notes wrong format | MEDIUM | The impl-notes breadcrumb file contains a single multi-line JSON object instead of JSONL entries. The mandatory events (`notes_read`, `files_collected`, `complete`) are not present as separate entries. Instead, there is one monolithic status object with `"status": "completed"`. This makes programmatic parsing and timeline analysis impossible. |
| 2 of 5 analyzers have no breadcrumbs | MEDIUM | The atanh and sinh analyzers produced analysis files but wrote zero breadcrumb entries to the shared file. The swish analyzer is the only one with full mandatory event coverage. |
| Most agents lack execution logs | MEDIUM | Only the analyzer agent produced an execution log. The other 5 agent types (generator, discoverer, implementor, tester, impl-notes) produced no execution logs. This limits post-mortem analysis. |

### Breadcrumb-to-Git Correlation

| Agent | Breadcrumb Commit | Git Commit | Match? |
|-------|-------------------|------------|--------|
| discoverer | (no commit field) | (no standalone commit) | N/A -- discoverer output committed as part of first analyzer commit (8bb3ee4249) |
| analyzer (atanh) | `f7a0d964e4` (in orchestrator breadcrumb line 13) | `f7a0d964e4` | YES |
| analyzer (swish) | `8bb3ee4249` (in orchestrator breadcrumb line 14; also in analyzer `complete` event) | `8bb3ee4249` | YES |
| analyzer (tanhshrink) | `faadb69dae` (in orchestrator breadcrumb line 15) | `faadb69dae` | YES |
| analyzer (sinh) | `090a329bdd` (in orchestrator breadcrumb lines 16-17) | `090a329bdd` | YES |
| analyzer (hardshrink) | `090a329bdd` (shared with sinh) + `6f043e5c73` (separate commit) | `6f043e5c73` | PARTIAL -- orchestrator logged 090a329bdd but there is also a separate hardshrink-only commit 6f043e5c73 |
| implementor | `d57642b293` (in orchestrator breadcrumb line 21) | `d57642b293` | YES |
| tester | (no commit field in orchestrator breadcrumb) | `d6f8d170f9` | MISSING -- orchestrator `subagent_completed` for tester has no `commit` field |
| impl-notes | (no field in breadcrumb) | `25dbf41523` | MISSING -- no commit field in breadcrumb |

---

## 4. SFPI Code Enforcement Audit

### Kernel Style Classification

| Check | Result | Evidence |
|-------|--------|---------|
| SFPI indicators present? | YES | `sfpi::vFloat` (20+ occurrences), `sfpi::vInt` (10+ occurrences), `sfpi::dst_reg[0]` (2 occurrences), `v_if`/`v_endif` (4 pairs), `sfpi::setsgn`, `sfpi::addexp`, `sfpi::exexp`, `sfpi::exman8`, `sfpi::exman9`, `sfpi::vConst1`, `sfpi::setexp`, `sfpi::int32_to_float`, `sfpi::reinterpret` |
| Raw TTI indicators present? | NO | No `TT_SFP`, `TTI_SFP`, `SFPLOADI`, `SFPLOAD`, `SFPSTORE`, `SFPSETCC`, `SFPENCC`, `SFPMAD`, `SFPMUL`, or `SFPIADD` patterns found |
| **Kernel style** | **SFPI** | Pure SFPI abstractions throughout |

### SFPI Quality Checks

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
| `#pragma GCC unroll` | Present on inner loop | `#pragma GCC unroll 0` at line 80 | OK -- unroll 0 intentionally used to reduce register pressure, consistent with sinh kernel pattern |
| DEST register pattern | `dst_reg[0]` read, compute, write, `dst_reg++` | `sfpi::dst_reg[0]` read at line 82, write at line 108, `sfpi::dst_reg++` at line 109 | OK |
| ITERATIONS template | `int ITERATIONS = 8` in template params | `template <bool APPROXIMATION_MODE, int ITERATIONS = 8>` at line 67 | OK |
| fp32 handling | `is_fp32_dest_acc_en` template param | Not present | OK -- consistent with reference kernels (swish, sinh) which also omit this |
| Parameter reconstruction | Union or Converter for param0 | Union reinterpretation at lines 70-73: `conv.u = param0; const float cap = conv.f;` | OK -- correct approach for this codebase |
| WH/BH identical | Both architecture files same content | IDENTICAL (confirmed via Read tool) | OK |

### Reference Style Comparison

| Reference | Reference Style | New Kernel Style | Assessment |
|-----------|----------------|-----------------|------------|
| tanhshrink | N/A (tanh kernel nuked; binary subtraction uses raw SFPMAD) | SFPI | N/A -- softcap does not use tanh_tile or binary subtraction |
| swish | A_sfpi (per analyzer breadcrumb `kernel_source_read`) | SFPI | Consistent -- both use pure SFPI abstractions |
| hardshrink | Mixed (comp functions use SFPI v_if; binary ops use SFPMAD) | SFPI | Correct -- softcap avoids the binary SFPMAD patterns |
| atanh | A_sfpi (uses vFloat, vConstFloatPrgm, dst_reg) | SFPI | Consistent |
| sinh | A_sfpi (uses vFloat, dst_reg, exp_21f helper) | SFPI | Consistent -- softcap adopted the exp_21f helper which is itself pure SFPI |

**Verdict**: COMPLIANT -- uses SFPI. The kernel is a clean, well-structured SFPI implementation with no raw TTI instructions.

---

## 5. What Went Well

### 1. First-pass test success

**Phase/Agent**: Phase 4 -- Tester
**Evidence**: Orchestrator breadcrumb `phase_complete` for phase 4 shows `"result":"PASS","iterations":1`. All 25 tests passed on the first actual test execution.
**Why it worked**: The tester proactively identified and resolved build environment issues (stub headers, SfpuType enum stubs, register pressure) before running tests, rather than discovering them through repeated test failures. This is a mature debugging strategy.

### 2. Strong reference selection

**Phase/Agent**: Phase 1 -- Discoverer
**Evidence**: All 5 references were analyzed and cited by the implementor. The sinh reference provided the exp_21f algorithm directly, and atanh provided the registration pattern. The discoverer's rationale (in `reference_selection.md`) accurately predicted which aspect of each reference would be useful.
**Why it worked**: The discoverer correctly identified that softcap's core challenge is implementing tanh from scratch (since tanh was nuked), and selected references covering: (1) tanh usage patterns (tanhshrink), (2) SFPI composite activation (swish), (3) float parameter pipeline (hardshrink), (4) registration pattern (atanh), and (5) exp_21f helper + dual-regime pattern (sinh).

### 3. Clean 12-layer implementation

**Phase/Agent**: Phase 3 -- Implementor
**Evidence**: All 12 layers present on disk. Implementor committed in a single pass (commit d57642b293, 18:49:41 UTC). No revisiting or layer-fixing was required downstream.
**Why it worked**: The 5 reference analyses provided comprehensive patterns for every layer. The implementor used the standard `eltwise_sfpu.cpp` dispatch path (simpler than custom compute kernel), avoiding the complexity that hardshrink and tanhshrink demonstrated.

### 4. Orchestrator handled analyzer commit failures gracefully

**Phase/Agent**: Phase 2 -- Generator (orchestrator)
**Evidence**: Orchestrator breadcrumb line 18 notes `"issues":["sinh and hardshrink agents did not commit; orchestrator committed on their behalf"]`. Git log confirms commit 090a329bdd was made by the orchestrator.
**Why it worked**: The orchestrator detected that 2 of 5 analyzer agents failed to commit and compensated by committing their output files. This prevented the pipeline from stalling.

---

## 6. Issues Found

### Issue 1: Implementor produced zero breadcrumbs

| Field | Value |
|-------|-------|
| Severity | HIGH |
| Phase | Phase 3 -- Implementation |
| Agent | implementor |
| Verification Dimension | Logging Compliance |
| Retries Consumed | 0 |
| Time Cost | 0 (no pipeline time lost, but post-mortem analysis severely impaired) |

**Problem**: The file `ttnn-unary-sfpu-operation-implementor_breadcrumbs.jsonl` does not exist in `agent_logs/`. The implementor logging spec (`.claude/references/logging/sfpu-operation-implementor.md`) clearly defines mandatory events: `references_parsed`, 12x `layer_implemented`, `implementation_complete`, and `complete`. None were produced.

**Root Cause**: The implementor agent likely did not read or follow the logging spec despite it being referenced in its instructions. This is a recurring issue -- the agent prioritized implementation over observability.

**Fix for agents**:
- **Implementor**: The SubagentStart hook should inject the breadcrumb file path and a reminder to read the logging spec as the very first action. Consider making the first breadcrumb (`references_parsed`) a gating requirement before implementation begins.
- **Generator (orchestrator)**: After implementor completes, verify that the breadcrumb file exists and contains the expected minimum event count before marking phase 3 as complete.

### Issue 2: Tester produced zero breadcrumbs

| Field | Value |
|-------|-------|
| Severity | HIGH |
| Phase | Phase 4 -- Testing |
| Agent | tester |
| Verification Dimension | Logging Compliance |
| Retries Consumed | 0 |
| Time Cost | 0 (no pipeline time lost, but debugging timeline is invisible) |

**Problem**: The file `ttnn-unary-sfpu-operation-tester_breadcrumbs.jsonl` does not exist in `agent_logs/`. The tester logging spec (`.claude/references/logging/sfpu-operation-tester.md`) defines mandatory events: `notes_parsed`, `test_created`, `test_run`, `hypothesis`, `fix_applied`, and `complete`. None were produced.

The tester applied at least 6 fixes (stub headers, SfpuType enum stubs, kernel simplification) and ran tests, but the entire debug timeline is lost. We know from the final report that tests passed 25/25, but we do not know: how many build attempts preceded the test run, what the exact build errors were, or how the kernel simplification was decided.

**Root Cause**: Same as Issue 1 -- the tester agent did not follow its logging spec.

**Fix for agents**:
- **Tester**: Same as implementor -- inject breadcrumb requirement as first action via SubagentStart hook.
- **Generator (orchestrator)**: Verify tester breadcrumb file exists before marking phase 4 complete.

### Issue 3: Implementation notes agent used wrong breadcrumb format

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase | Phase 5 -- Documentation |
| Agent | impl-notes |
| Verification Dimension | Logging Compliance |
| Retries Consumed | 0 |
| Time Cost | 0 |

**Problem**: The impl-notes breadcrumb file contains a single multi-line JSON object (55 lines) instead of JSONL entries. The mandatory events (`notes_read`, `files_collected`, `complete`) do not appear as separate entries. Instead, there is one monolithic object with keys like `"timestamp"`, `"agent"`, `"task"`, `"stage"`, `"status"`, `"files_processed"`, `"sections_added"`, and `"enrichment_details"`. The timestamp key is `"timestamp"` instead of the standard `"ts"`.

**Root Cause**: The impl-notes agent either did not read the logging spec or chose to produce a summary object instead of incremental breadcrumbs. The spec clearly requires JSONL format with `notes_read`, `files_collected`, and `complete` as separate events.

**Fix for agents**:
- **Impl-notes**: Enforce JSONL format -- each breadcrumb must be a single line with a `"ts"` field. Use the `append_breadcrumb.sh` helper instead of writing a JSON blob directly.

### Issue 4: No fp32-specific test cases

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase | Phase 4 -- Testing |
| Agent | tester |
| Verification Dimension | Implementation Coverage |
| Retries Consumed | 0 |
| Time Cost | 0 |

**Problem**: The test file `test_softcap.py` uses `data_gen_with_range` which produces bfloat16 tensors by default. No test case explicitly creates fp32 input tensors (e.g., via `dtype=ttnn.float32`). While the SFPU hardware computes at bfloat16-equivalent precision for both dtypes, the fp32 code path may have different behavior in the pack/unpack stages.

**Root Cause**: The tester focused on functional correctness in the default dtype and did not add fp32-parametrized tests.

**Fix for agents**:
- **Tester**: The tester instructions should mandate at least one test case with `dtype=ttnn.float32` to verify the fp32 code path.

### Issue 5: 2 of 5 analyzers produced no breadcrumbs

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase | Phase 2 -- Analysis |
| Agent | analyzer (atanh, sinh) |
| Verification Dimension | Logging Compliance |
| Retries Consumed | 0 |
| Time Cost | 0 |

**Problem**: The atanh and sinh analyzers produced analysis files (committed) but zero breadcrumb entries in the shared `ttnn-unary-sfpu-operation-analyzer_breadcrumbs.jsonl` file. This is despite the analyzer logging spec requiring 6 mandatory events per operation.

These same two analyzers also failed to commit their own output (the orchestrator committed on their behalf). This suggests a pattern: agents that fail to commit also fail to produce breadcrumbs, likely because both require using the `append_breadcrumb.sh` helper and `git commit`.

**Root Cause**: The atanh and sinh analyzer instances either (1) did not read the logging spec, or (2) encountered an error that prevented breadcrumb writing (e.g., file path issue with parallel writes to the same JSONL file).

**Fix for agents**:
- **Analyzer**: Ensure the `start` breadcrumb is written as the very first action. If parallel analyzers share a breadcrumb file, consider using file-level locking (the append_breadcrumb.sh helper may already handle this).

### Issue 6: Tester spent significant time on environment repair

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase | Phase 4 -- Testing |
| Agent | tester |
| Verification Dimension | Implementation Coverage |
| Retries Consumed | 0 (all fixes applied before first test run) |
| Time Cost | Estimated ~15-20 minutes of the ~24 minute testing phase |

**Problem**: The tester had to create multiple stub headers (trigonometry.h, rpow.h, rdiv.h, fill.h, ckernel_sfpu_conversions.h, ckernel_sfpu_mul_int32.h, mul_int_sfpu.h, llk_math_eltwise_binary_sfpu_params.h), add ~35 stub SfpuType enum values, and simplify the kernel from degree-7 Taylor + 3-term geometric to degree-5 Taylor + 2-term geometric. This consumed the majority of Phase 4 time.

**Root Cause**: The "deeply nuked" codebase environment requires extensive stub creation for any new SFPU operation. This is an inherent challenge of the evaluation environment, not an agent design flaw. However, the implementor could have anticipated some of these needs.

**Fix for agents**:
- **Implementor**: Before committing, attempt a compile check (or at least verify that all transitive includes exist) to catch missing header issues before handing off to the tester.
- **Pipeline infrastructure**: Pre-populate common stub headers in the nuked environment to reduce tester repair overhead.

---

## 7. Efficiency Analysis

### Per-Phase Breakdown

| Phase | Duration | Status | Bottleneck |
|-------|----------|--------|------------|
| 1: Discovery | ~5m | OK | Clean |
| 2: Analysis | ~13m (wall) | OK | hardshrink analyzer was slowest (~13m); atanh was fastest (~4m) |
| 3: Implementation | ~16m | OK | Clean single pass |
| 4: Testing | ~24m | OK | Environment repair (stub headers, SfpuType stubs, kernel simplification) |
| 5: Documentation | ~3m | OK | Clean |

### Tester Iteration Breakdown

| Attempt | Result | Error Type | Fix Applied | Duration |
|---------|--------|-----------|-------------|----------|
| Pre-test repair | N/A | build_error (multiple) | Created 8+ stub headers, added ~35 SfpuType enum stubs, simplified kernel (degree-7 -> degree-5 Taylor, 3-term -> 2-term geometric) | ~15-20m (estimated) |
| 1 (actual test) | PASS (25/25) | - | - | ~4-5m |

### Time Sinks

| Rank | Area | Agent | Duration | % of Total | Description |
|------|------|-------|----------|------------|-------------|
| 1 | Environment repair | tester | ~18m | ~29% | Creating stub headers and simplifying kernel for nuked environment; unavoidable given the evaluation setup |
| 2 | Analysis (hardshrink) | analyzer | ~13m | ~21% | Slowest of 5 parallel analyzers; investigated custom compute kernel path with two-pass algorithm and multiple SFPU functions |
| 3 | Implementation (all layers) | implementor | ~16m | ~26% | Full 12-layer implementation; reasonable for the complexity |

---

## 8. Inter-Agent Communication

| Handoff | Source -> Target | Artifact | Quality | Issues | Suggestion |
|---------|-----------------|----------|---------|--------|------------|
| 1 | Generator -> Discoverer | Math definition | GOOD | None; `cap * tanh(x / cap)` clearly communicated with parameter info | None |
| 2 | Discoverer -> Analyzers | Reference list | GOOD | All 5 references were relevant and well-justified in reference_selection.md | None |
| 3 | Analyzers -> Implementor | Analysis files | GOOD | All 5 analysis files produced; quality is thorough (dispatch chains, kernel source, instruction analysis) | None |
| 4 | Implementor -> Tester | Impl notes | ADEQUATE | Implementation notes were complete but did not warn about nuked-environment transitive include issues; tester had to discover and resolve these independently | Implementor should include a "build verification" section noting any anticipated include issues |
| 5 | Tester -> Impl-Notes | File manifest | GOOD | All new and modified files correctly listed; impl-notes agent enriched with full source code | None |

---

## 9. Comparison with Known Issues

### Known Issues Encountered

| Issue # | Title | Encountered? | Notes |
|---------|-------|-------------|-------|
| 1 | Kernel writer burns massive context on numerical debugging | NO | Tests passed first time; no numerical debugging needed |
| 13 | Phase 1/2 overlap | NO | All phases ran strictly sequentially per breadcrumb timestamps |
| 15 | Kernel writer missing execution logs | YES (analogous) | The implementor and tester (analogous to "kernel writer" in older pipeline) produced no execution logs. This is the same class of issue. |
| 18 | Agent relaunch loses debugging context | NO | No relaunches occurred; single-pass success |

### New Issues Discovered

| Title | Description | Suggested Priority |
|-------|-------------|-------------------|
| Implementor and tester agents produce zero breadcrumbs | Both agents have well-defined logging specs but completely ignored them. The implementor breadcrumb file does not exist; the tester breadcrumb file does not exist. This eliminates post-mortem visibility into the two longest pipeline phases. | HIGH |
| Impl-notes agent uses wrong breadcrumb format | Produces a single JSON object instead of JSONL entries. Uses `"timestamp"` instead of `"ts"`. Does not emit the mandatory events (`notes_read`, `files_collected`, `complete`) as separate entries. | MEDIUM |
| Parallel analyzers have inconsistent breadcrumb compliance | Of 5 parallel analyzer instances, only 1 (swish) achieved full compliance with the mandatory event contract. 2 (tanhshrink, hardshrink) used non-standard event names, and 2 (atanh, sinh) produced zero breadcrumbs. Suggests that parallel agent instances have lower logging reliability. | MEDIUM |
| No fp32 test coverage | Test file lacks explicit fp32-dtype test cases. Should be a standard requirement for all SFPU operations. | MEDIUM |

---

## 10. Actionable Recommendations

### Recommendation 1: Enforce breadcrumb creation via SubagentStart hook validation

- **Type**: pipeline_change
- **Target**: SubagentStart hooks for implementor and tester agents
- **Change**: Add a mandatory first-action requirement: the agent must call `append_breadcrumb.sh` with a `start` event before doing any other work. The orchestrator should verify the breadcrumb file exists (and has >= 1 entry) after agent completion; if missing, log a warning in the issues log.
- **Expected Benefit**: Eliminates the "zero breadcrumbs" problem for the two highest-impact agents
- **Priority**: HIGH
- **Effort**: SMALL

### Recommendation 2: Add fp32 test parametrization to tester instructions

- **Type**: instruction_change
- **Target**: Tester agent instructions
- **Change**: Add a mandatory test case with `dtype=ttnn.float32` in addition to the default bfloat16 tests. Template: `@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])`.
- **Expected Benefit**: Verifies fp32 code path compiles and runs; catches pack/unpack issues that only manifest with fp32
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 3: Standardize impl-notes breadcrumb format

- **Type**: logging_fix
- **Target**: Impl-notes agent instructions / SubagentStart hook
- **Change**: Ensure the impl-notes agent uses `append_breadcrumb.sh` for each mandatory event (`notes_read`, `files_collected`, `complete`) instead of writing a single JSON blob. The SubagentStart hook should explicitly mention the helper script path and JSONL format requirement.
- **Expected Benefit**: Enables timeline analysis for the documentation phase; consistent with other agents' format
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 4: Orchestrator should verify breadcrumb file existence post-agent

- **Type**: new_validation
- **Target**: Generator (orchestrator) agent instructions
- **Change**: After each `subagent_completed` event, the orchestrator should verify that the agent's breadcrumb file exists in `agent_logs/` and log a warning if missing. This catches logging failures early.
- **Expected Benefit**: Creates accountability for logging compliance without blocking the pipeline
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 5: Improve parallel analyzer breadcrumb reliability

- **Type**: pipeline_change
- **Target**: Analyzer agent launch mechanism
- **Change**: Ensure each parallel analyzer instance receives a unique reminder to write breadcrumbs as its first action. Consider whether the shared breadcrumb file creates contention issues for parallel writes; if so, use per-operation breadcrumb files and merge after completion.
- **Expected Benefit**: Raises analyzer breadcrumb compliance from 1/5 (full) to 5/5
- **Priority**: MEDIUM
- **Effort**: MEDIUM

---

## 11. Overall Assessment

### Pipeline Maturity Score

| Dimension | Score | Notes |
|-----------|-------|-------|
| Reference discovery accuracy | 5 | All 5 references relevant; sinh and atanh proved directly useful for the core algorithm and registration pattern |
| Reference analysis quality | 4 | All 5 analyses produced and thorough; -1 for 2 analyzers failing to commit (orchestrator compensated) |
| Implementation completeness | 5 | 12/12 layers present, correct math, clean single-pass implementation |
| SFPI compliance | 5 | Pure SFPI kernel; no raw TTI; all quality checks pass; WH/BH identical |
| Testing thoroughness | 3 | 25/25 pass is excellent, but fp32 coverage missing; ULP thresholds are generous (10 ULP, rtol=5e-2) |
| Inter-agent communication | 4 | Handoff quality was good throughout; -1 for implementor not warning tester about transitive include issues |
| Logging/observability | 2 | Only 2 of 6 agents achieved FULL breadcrumb compliance (discoverer, generator-partial); implementor and tester have zero observability; this was the weakest dimension |

### Top 3 Things to Fix

1. **Enforce breadcrumb creation for implementor and tester agents** -- these are the two longest phases and currently have zero logging, making post-mortem analysis of failures impossible.
2. **Add fp32 test parametrization** -- every SFPU operation should verify both bfloat16 and fp32 code paths.
3. **Standardize breadcrumb format across all agents** -- the impl-notes agent used the wrong format; analyzers had inconsistent compliance; a consistent enforcement mechanism would solve both.

### What Worked Best

The reference selection and utilization pipeline was the strongest aspect of this run. The discoverer correctly identified sinh as the most useful reference (providing the exp_21f algorithm and dual-regime pattern), and the implementor adopted these patterns directly, resulting in a mathematically sound kernel that passed all tests on the first attempt. The end-to-end flow from discovery through analysis to implementation was remarkably efficient, completing a novel SFPU operation in ~62 minutes with no pipeline-level retries.
