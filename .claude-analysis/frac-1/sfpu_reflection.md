# SFPU Reflection: frac

## Metadata
| Field | Value |
|-------|-------|
| Operation | `frac` |
| Math Definition | `frac(x) = x - floor(x)` (orchestrator) / `frac(x) = x - trunc(x)` (corrected) |
| Output Folder | `.claude-analysis/frac-1/` |
| Pipeline Phases Executed | 1, 2, 3, 4, 5, 6 |
| Agents Invoked | generator, discoverer, 5x analyzer, implementor, tester, impl-notes, self-reflection |
| Total Git Commits | 18 (in `.claude-analysis/frac-1/` scope) |
| Total Pipeline Duration | ~71 min (09:20:48 to 10:32:07) |
| Overall Result | SUCCESS |

---

## 1. Pipeline Execution Summary

### Phase Timeline

| Phase | Agent(s) | Duration | Status | Key Observations |
|-------|----------|----------|--------|------------------|
| 1: Reference Discovery | discoverer | 4m 45s | OK | 5 references selected; discoverer identified trunc/floor ambiguity in `reference_selection.md` |
| 2: Reference Analysis | 5x analyzer | 16m 10s (wall) | OK (4/5 on time) | cbrt analyzer still running when orchestrator proceeded; hardswish analyzer produced NO breadcrumbs |
| 3: Implementation | implementor | 8m 37s | OK | 12 layers completed; initial kernel used floor(x) semantics |
| 4: Testing & Debugging | tester | 39m 45s | OK | 3 test runs across 2 issues: semantics fix + build cache |
| 5: Documentation | impl-notes + generator | 0m 30s | OK | Notes enriched with embedded source code |
| **Total** | | **~71m** | | |

### Agent Duration Breakdown

| Agent | Start Time | End Time | Wall Duration | Iterations | Notes |
|-------|------------|----------|---------------|------------|-------|
| generator (orchestrator) | 09:20:48 | 10:32:07 | 71m 19s | - | Entire pipeline |
| discoverer | 09:21:43 | 09:25:18 | 3m 35s | - | Clean run |
| analyzer (cbrt) | 09:26:49 | 09:46:42 | 19m 53s | - | Slowest analyzer; finished after Phase 2 declared complete |
| analyzer (hardtanh) | 09:26:59 | 09:43:52 | 16m 53s | - | Re-analysis committed at 09:43:23 |
| analyzer (hardsigmoid) | 09:27:04 | 09:41:14 | 14m 10s | - | |
| analyzer (hardswish) | ~09:25:56 | ~09:30:04 | ~4m (git) | - | NO breadcrumb events; only git commit evidence |
| analyzer (softshrink) | 09:27:24 | 09:41:44 | 14m 20s | - | |
| implementor | 09:30:22 | 09:46:46 | 16m 24s | - | Started before Phase 3 declared; productive time ~5m (09:42 to 09:47) |
| tester | 09:48:41 | 10:12:01 | 23m 20s | 3 attempts | Orchestrator Phase 4 spanned 39m 45s (gap unexplained) |
| impl-notes | 10:16:27 | 10:17:12 | 0m 45s | - | Clean enrichment |

**Duration calculation method**: Breadcrumb `ts` fields as primary source, git commit timestamps (`%ai`) as corroboration. Where breadcrumbs are missing (hardswish analyzer), git commit timestamps are sole source.

### Duration Visualization

```
Phase 1  |████|                                                              (~5m)
Phase 2       |████████████████|                                             (~16m)
Phase 3                         |████████|                                   (~9m)
Phase 4                                   |████████████████████████████████████████| (~40m)
Phase 5                                                                            || (~1m)
         0    5    10   15   20   25   30   35   40   45   50   55   60   65   70 min

Longest phase: Phase 4 (40m) -- floor-vs-trunc semantics fix + build cache debugging
```

### Time Distribution

| Category | Duration | % of Total | Notes |
|----------|----------|------------|-------|
| Discovery (Phase 1) | 4m 45s | 6.7% | |
| Analysis (Phase 2) | 16m 10s | 22.7% | 5 parallel analyzers |
| Implementation (Phase 3) | 8m 37s | 12.1% | 12 layers |
| Testing (Phase 4) | 39m 45s | 55.8% | 3 iterations |
| -- Productive (test creation + first run) | ~1m 32s | 2.2% | Test created at 09:49:22, first run result at 09:50:13 |
| -- Debugging/retries | ~38m 13s | 53.6% | Hypothesis, kernel fix, build cache issue, re-run |
| Documentation (Phase 5) | 0m 30s | 0.7% | |
| Orchestrator overhead / gaps | ~1m 12s | 1.7% | Time between phases |
| **Total** | **~71m** | **100%** | |

---

## 2. Implementation Coverage Audit

### Math Definition Fidelity

| Aspect | Status | Details |
|--------|--------|---------|
| Core formula | MATCH (after fix) | Kernel computes `x - trunc(x)` via IEEE 754 mantissa bit masking, matching `torch.frac()` semantics |
| Conditional branches | CORRECT | Three-case branching: exp < 0 (result = x), exp >= 23 (result = 0), 0 <= exp < 23 (mask + subtract) |
| Parameter handling | N/A | frac is parameterless |
| Edge cases | MATCH | x=0 returns 0; integers return 0; negative fractionals preserve sign; subnormals return x (correct since |x| < 1) |

**Math definition from orchestrator**: `frac(x) = x - floor(x)`
**Kernel implementation**: `frac(x) = x - trunc(x)` (corrected by tester in Phase 4)

**IMPORTANT**: The orchestrator's `pipeline_start` breadcrumb and `issues_log.md` both state `frac(x) = x - floor(x)`. This is mathematically incorrect for `torch.frac()`, which uses truncation toward zero, not floor. For negative inputs: `floor(-1.5) = -2` but `trunc(-1.5) = -1`, so `frac(-1.5) = -1.5 - (-1) = -0.5` (trunc) vs `-1.5 - (-2) = 0.5` (floor). The implementor faithfully implemented the wrong definition; the tester caught and corrected it. The final kernel on disk is correct.

### 12-Layer Completeness

| Layer | Description | Expected File(s) | Status | Notes |
|-------|-------------|-------------------|--------|-------|
| 1 | SFPU Kernel | `ckernel_sfpu_frac.h` (WH+BH) | PRESENT | Identical files on disk; trunc semantics (post-fix) |
| 2 | LLK Dispatch | `llk_math_eltwise_unary_sfpu_frac.h` (WH+BH) | PRESENT | Verified on disk, both architectures |
| 3 | Compute API Header | `frac.h` | PRESENT | `tt_metal/hw/inc/api/compute/eltwise_unary/frac.h` exists |
| 4 | SFPU Include Guard | `sfpu_split_includes.h` | PRESENT | `SFPU_OP_FRAC_INCLUDE` at line 23 |
| 5 | SfpuType Enum | `llk_sfpu_types.h` (WH+BH) | PRESENT | `frac` entry in both architectures |
| 6 | UnaryOpType Enum | `unary_op_types.hpp` | PRESENT (pre-existing) | `FRAC` at line 103; implementor correctly skipped (already existed) |
| 7 | Op Utils Registration | `unary_op_utils.cpp` + `unary_ng_op_utils.cpp` | PRESENT | `get_macro_definition` + `get_op_init_and_func_default` in both legacy and NG utils |
| 8 | Op Utils Header | `unary_op_utils.hpp` | N/A | frac is not parameterized; no changes needed |
| 9 | C++ API Registration | `unary.hpp` | PRESENT (pre-existing) | `REGISTER_UNARY_OPERATION(frac, FRAC)` at line 154; implementor correctly skipped |
| 10 | Python Nanobind | `unary_nanobind.cpp` | PRESENT (pre-existing) | `bind_unary_operation<"frac", &ttnn::frac>` at line 1830; implementor correctly skipped |
| 11 | Python Golden | `unary.py` | PRESENT | `"frac": torch.frac` at line 44 and `ttnn.frac` at line 65 |
| 12 | Test File | `test_frac.py` | PRESENT | 3 test functions (basic, negative, integers) with bfloat16 |

**Layer completeness**: 12/12 layers present. The implementor correctly identified that layers 6, 9, and 10 already existed (frac was partially wired in the codebase but missing the SFPU kernel and dispatch) and skipped them.

**Additional file**: `sources.cmake` was also updated (line 52) to register `frac.h` in the compute kernel header list. This is not a standard 12-layer item but is required for correct compilation.

### Reference Utilization

| Reference | Analysis Produced? | Cited by Implementor? | Usefulness |
|-----------|-------------------|----------------------|------------|
| cbrt | YES (197 lines) | YES -- `exexp`, `reinterpret<vInt>`, bit manipulation patterns | HIGH |
| hardtanh | YES (170 lines) | YES -- standard `ckernel_sfpu` boilerplate and `v_if/v_endif` | HIGH |
| hardsigmoid | YES (143 lines) | YES -- parameterless init pattern | HIGH |
| hardswish | YES (308 lines) | YES -- intermediate-then-subtract pattern | MEDIUM |
| softshrink | YES (157 lines) | YES -- three-case conditional structure | MEDIUM |

**References wasted**: 0. All 5 references were analyzed and cited in the implementation notes. The discoverer's selection was well-targeted.

**Notable discrepancy**: The implementor's `references_parsed` breadcrumb lists `["trunc","floor","ceil","softshrink","rpow"]` as references, which does NOT match the discoverer's selected references `["cbrt","hardtanh","hardsigmoid","hardswish","softshrink"]`. This suggests the implementor may have read different references than what the discoverer selected. However, the implementation notes "Reference Operations Used" section correctly lists all 5 discoverer-selected references, indicating the implementor also read these. The `references_parsed` breadcrumb appears to reflect existing operations the implementor found in the codebase (trunc, floor, ceil) rather than the analysis files it was given.

### Test Coverage

| Metric | Value |
|--------|-------|
| Test file created | YES |
| bfloat16 parametrization | PASS |
| fp32 parametrization | PASS (from tester execution log; final test file only parametrizes bfloat16) |
| Max ULP (bfloat16) | 0.0 |
| Max ULP (fp32) | 0.0 |
| allclose (bfloat16) | PASS (rtol=1.6e-2, atol=1e-2) |
| allclose (fp32) | PASS (rtol=1e-3, atol=1e-4) |
| Total test iterations | 3 |
| Final result | PASS |

**Test quality note**: The final test file on disk (`test_frac.py`) includes 3 test functions (`test_frac`, `test_frac_negative`, `test_frac_integers`) but only `test_frac` is parametrized with `dtype=[ttnn.bfloat16]`. There is no explicit `fp32` parametrization in the file. The tester's execution log reports fp32 passed, which suggests it ran an earlier version of the test or a separate fp32 test that was not persisted. The on-disk test covers bfloat16 only. This is a MEDIUM gap -- fp32 was tested but not preserved in the final test file.

---

## 3. Breadcrumb & Logging Compliance Audit

### Per-Agent Breadcrumb Compliance

| Agent | File Exists? | Event Count | Min Expected | Mandatory Events Present? | Timestamps? | Ordering? | Compliance |
|-------|-------------|-------------|--------------|--------------------------|-------------|-----------|------------|
| generator | YES (on disk) | 32 | ~27 | YES: `pipeline_start`, 5x `phase_start`, 4x `phase_complete` (missing Phase 6 complete), 8x `subagent_launched`, 7x `subagent_completed` | YES | YES | PARTIAL |
| discoverer | YES (on disk) | 5 | 4 | YES: `start`, `files_read`, `ranking_complete`, `complete` | YES | YES | FULL |
| analyzer(s) | YES (on disk) | 29 | 30 (6 per op x 5) | PARTIAL: hardswish has 0 events; cbrt/hardtanh/hardsigmoid/softshrink each have 6-7 | YES | YES | PARTIAL |
| implementor | YES (in git) | 16 | 15 | YES: `start`, `references_parsed`, 12x `layer_implemented`, `implementation_complete`, `complete` | YES | YES | FULL |
| tester | YES (in git) | 9 | 4 | YES: `start`, `notes_parsed`, `test_created`, 3x `test_run`, `hypothesis`, `fix_applied`, `complete` | YES | YES | FULL |
| impl-notes | YES (in git) | 4 | 3 | YES: `start`, `notes_read`, `files_collected`, `complete` | YES | YES | FULL |

**Key findings**:

1. **Hardswish analyzer**: ZERO breadcrumb events. The hardswish analysis file exists (308 lines, the longest of all 5 analyses) and was committed at `09:30:04`, but the analyzer produced no breadcrumbs at all. This is a complete logging failure for one of the 5 analyzer instances.

2. **Generator Phase 6 incomplete**: The generator breadcrumbs end with `phase_start` for Phase 6 (Self-Reflection) and `subagent_launched` but no `phase_complete` or `pipeline_complete` event. This is expected since the self-reflection agent is the one reading these breadcrumbs.

3. **Implementor and tester breadcrumbs**: These files exist in git history (committed to `.claude-analysis/frac-1/agent_logs/`) but are NOT present on the working tree disk. The git commit `d1775c1662` (tester) and `79d6124930` (implementor) added them, but a subsequent operation (the "Deep family nuke" commit `3954826071` or the worktree transition) deleted them from the working tree. The breadcrumb data was recovered from the `vignjatijevic/sfpu-agent-codegen_kernel_bench` branch where these commits still exist.

4. **Generator breadcrumbs on disk are modified**: The git status shows `M .claude-analysis/frac-1/agent_logs/ttnn-unary-sfpu-operation-generator_breadcrumbs.jsonl` -- the orchestrator continued appending events after its last commit (`230cc0658c`), including Phase 5 completion and Phase 6 start.

### Execution Log Compliance

| Agent | Log Exists? | Sections Present | Notes |
|-------|------------|------------------|-------|
| generator | NO | - | Generator does not produce an execution log (not required by spec) |
| discoverer | NO | - | Discoverer does not produce an execution log (not required by spec) |
| analyzer | YES (on disk) | Metadata, Input Interpretation (4 ops), Execution Timeline (4 ops), Deviations (2 ops), Artifacts (4 ops), Key Findings (4 ops), Instruction Recommendations (1) | Covers hardsigmoid, softshrink, hardtanh, cbrt. Hardswish is MISSING from execution log. |
| implementor | NO | - | No execution log found in agent_logs or git history |
| tester | YES (in git) | Metadata, Input Interpretation, Upstream Feedback, Execution Timeline, Test Attempt Details, Debugging Narrative, Numerical Accuracy, Test Infrastructure Notes, Recovery Summary, Deviations, Artifacts, Handoff Notes, Instruction Recommendations, Raw Logs | Comprehensive -- 267 lines, all standard sections present |
| impl-notes | NO | - | Impl-notes agent does not produce an execution log (not required by spec) |

### Logging Infrastructure Issues

| Issue | Severity | Description |
|-------|----------|-------------|
| Hardswish analyzer zero breadcrumbs | HIGH | The hardswish analyzer instance produced a 308-line analysis file and a git commit but logged zero breadcrumb events. All other 4 analyzer instances logged 6-8 events each. Root cause unclear -- possibly the agent crashed before breadcrumb finalization, or the breadcrumb file was overwritten by another analyzer instance sharing the same output file. |
| Implementor missing execution log | MEDIUM | The implementor logging spec exists (`sfpu-operation-implementor.md`) and defines expected breadcrumb events (which were produced), but no execution log was generated. The tester (which has a similar spec) produced a comprehensive 267-line execution log. |
| Agent breadcrumbs not on working tree | MEDIUM | Implementor and tester breadcrumbs were committed to git but are not present on the current working tree. This is a cross-worktree artifact loss issue: the "Deep family nuke" commit or worktree transition deleted these files. The data is recoverable from the `sfpu-agent-codegen_kernel_bench` branch. |

### Breadcrumb-to-Git Correlation

| Agent | Breadcrumb Commit | Git Commit | Match? |
|-------|-------------------|------------|--------|
| discoverer | `complete` -- no commit field | N/A (no own commit) | N/A |
| analyzer (hardsigmoid) | `a0788e4da1` | `a0788e4da1` (09:40:05) | YES |
| analyzer (softshrink) | `pending_orchestrator` | `beb7424885` (09:40:32) | PARTIAL -- breadcrumb says "pending", git has the commit |
| analyzer (hardtanh) | `7486964b1f` | `7486964b1f` (09:43:23) | YES |
| analyzer (cbrt) | `add7a366e5` | `add7a366e5` (09:45:11, by orchestrator) | YES |
| analyzer (hardswish) | (no breadcrumb) | `c48ffe76c8` (09:30:04) | MISSING -- no breadcrumb to correlate |
| implementor | `c3b0f1b8e1` (in `complete` event, not explicit commit field) | `c3b0f1b8e1` (09:50:27) | YES |
| tester | (no explicit commit field in `complete`) | `d1775c1662` (10:15:11) | PARTIAL -- no commit hash in breadcrumb |
| impl-notes | (no explicit commit field in `complete`) | `e4d174f886` (10:17:16) | PARTIAL -- no commit hash in breadcrumb |

---

## 4. SFPI Code Enforcement Audit

### Kernel Style Classification

| Check | Result | Evidence |
|-------|--------|---------|
| SFPI indicators present? | YES | `sfpi::vFloat`, `sfpi::vInt`, `sfpi::vUInt`, `sfpi::dst_reg[0]`, `sfpi::exexp()`, `sfpi::reinterpret<>`, `v_if`/`v_endif` all present |
| Raw TTI indicators present? | NO | No `TT_SFP*`, `TTI_SFP*`, `SFPLOAD`, `SFPMAD`, `SFPSETCC`, or any raw instruction macros found |
| **Kernel style** | **SFPI** | Pure SFPI abstractions throughout |

### Exception Check

Not applicable -- no raw TTI indicators detected.

**Verdict**: COMPLIANT -- uses SFPI abstractions exclusively.

### SFPI Quality Checks

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
| `#pragma GCC unroll 8` | Present on inner loop | Present at line 24 | OK |
| DEST register pattern | `dst_reg[0]` read, compute, write, `dst_reg++` | `sfpi::vFloat x = sfpi::dst_reg[0]` (read) ... `sfpi::dst_reg[0] = x - trunc_x` (write) ... `sfpi::dst_reg++` (advance) | OK |
| ITERATIONS template | `int ITERATIONS = 8` in template params | `template <bool APPROXIMATION_MODE, int ITERATIONS = 8>` at line 22 | OK |
| fp32 handling | `is_fp32_dest_acc_en` template param | NOT PRESENT | MEDIUM |
| Parameter reconstruction | `Converter::as_float(param0)` | N/A (not parameterized) | N/A |
| WH/BH identical | Both architecture files same content | Byte-for-byte identical (verified on disk) | OK |

**fp32 handling note**: The kernel does not have `is_fp32_dest_acc_en` template specialization. For frac, the bit-manipulation algorithm operates on the IEEE 754 representation which is the same in fp32 mode, so the lack of explicit fp32 handling is unlikely to cause issues (ULP 0.0 confirms this). However, some reference operations (like cbrt) provide a separate fp32 path. This is a MEDIUM quality gap -- the kernel works correctly but does not follow the full pattern.

### Reference Style Comparison

| Reference | Reference Style | New Kernel Style | Assessment |
|-----------|----------------|-----------------|------------|
| cbrt | A_sfpi | SFPI | Consistent. Used `exexp`/`reinterpret` pattern from cbrt |
| hardtanh | A_sfpi | SFPI | Consistent. Used `v_if/v_endif` boilerplate from hardtanh |
| hardsigmoid | A_sfpi | SFPI | Consistent. Used parameterless init pattern from hardsigmoid |
| hardswish | A_sfpi | SFPI | Consistent. Used intermediate-computation pattern from hardswish |
| softshrink | A_sfpi | SFPI | Consistent. Used three-case conditional pattern from softshrink |

All 5 references use SFPI abstractions, and the new kernel correctly follows the same style. No style translation was needed.

---

## 5. What Went Well

### 1. Reference selection was excellent

**Phase/Agent**: Phase 1 -- discoverer
**Evidence**: All 5 selected references (cbrt, hardtanh, hardsigmoid, hardswish, softshrink) were cited in the implementation notes' "Reference Operations Used" section. The discoverer identified the exact SFPI primitives needed (`exexp`, `reinterpret<vInt>`, bit masking) and selected cbrt specifically for this pattern. The discoverer also correctly noted in `reference_selection.md` that `torch.frac()` uses `x - trunc(x)` semantics, not `x - floor(x)`.
**Why it worked**: The discoverer read the existing `ckernel_sfpu_frac.h` build artifact (listed as `ckernel_sfpu_frac.h(build)` in `files_read`) and analyzed the component operations before selecting references.

### 2. All 12 implementation layers completed correctly

**Phase/Agent**: Phase 3 -- implementor
**Evidence**: The implementor's breadcrumbs show 12 sequential `layer_implemented` events from `09:42:31` to `09:45:54` (under 4 minutes for all 12 layers). The implementor correctly identified that layers 6 (UnaryOpType), 9 (C++ API), and 10 (Python Nanobind) already existed and skipped them. It also added the NG op utils registration (a bonus not in the standard 12 layers).
**Why it worked**: The implementor had 5 thorough analysis files to work from, and the frac operation is structurally simple (parameterless, no approximation mode branching).

### 3. Tester produced exceptional debugging documentation

**Phase/Agent**: Phase 4 -- tester
**Evidence**: The tester's execution log (267 lines) contains a detailed debugging narrative with exact mismatch counts (17024), root cause analysis (floor vs trunc for negative inputs), and a precise breakdown of mismatch categories (16128 negative with exp < 0 + 769 with exp 0-6 + 127 subnormals). Hypothesis H1 was classified HIGH confidence and was correct.
**Why it worked**: The tester wrote a diagnostic script to analyze the failure pattern systematically rather than guessing at the cause.

### 4. Bit-exact numerical accuracy (ULP 0.0)

**Phase/Agent**: Phase 4 -- tester (verification)
**Evidence**: Both bfloat16 and fp32 tests achieved Max ULP = 0.0, meaning the hardware output is bit-identical to the PyTorch golden reference. This is the best possible numerical result.
**Why it worked**: The IEEE 754 bit-manipulation approach to computing trunc(x) is exact -- it directly masks mantissa bits rather than using floating-point arithmetic that could introduce rounding errors.

### 5. All 5 analysis files produced

**Phase/Agent**: Phase 2 -- analyzers
**Evidence**: 5 analysis files in the output folder totaling 975 lines. All 5 reference operations were analyzed, with detailed dispatch tracing, kernel source reading, and instruction analysis.
**Why it worked**: Despite the hardswish analyzer not producing breadcrumbs, it still produced a thorough 308-line analysis (the longest of all 5).

---

## 6. Issues Found

### Issue 1: Incorrect math definition propagated through pipeline

| Field | Value |
|-------|-------|
| Severity | HIGH |
| Phase | Phase 0 (input) through Phase 4 (testing) |
| Agent | Orchestrator (generator) and implementor |
| Verification Dimension | Implementation Coverage |
| Retries Consumed | 1 hard retry (semantics fix) + 1 free retry (build cache) |
| Time Cost | ~15m of tester debugging time |

**Problem**: The orchestrator's `pipeline_start` breadcrumb and `issues_log.md` both specify `frac(x) = x - floor(x)`. This is wrong for `torch.frac()`, which computes `x - trunc(x)`. The implementor faithfully implemented the floor-based definition. The tester discovered 17024 mismatches on negative inputs (exactly the inputs where floor and trunc differ) and had to fix the kernel.

**Root Cause**: The orchestrator (or its human invoker) provided an incorrect math definition. The discoverer actually noted the correct semantics in `reference_selection.md` line 7: "PyTorch torch.frac() semantics use x - trunc(x) (sign-preserving)", but this information was not back-propagated to correct the definition before implementation.

**Fix for agents**:
- **Orchestrator**: Add a validation step after Phase 1: cross-check the math definition against the golden function (`torch.frac`) by evaluating a few test values (e.g., -1.5, 1.5, -0.5) and verifying consistency.
- **Implementor**: Before implementing, verify the math definition against the golden function by computing a few examples. If `x - floor(-1.5) = 0.5` but `torch.frac(-1.5) = -0.5`, flag the discrepancy.
- **Discoverer**: When identifying semantics corrections (as it did in `reference_selection.md`), explicitly flag them as `upstream_feedback` to the orchestrator so the definition can be corrected before Phase 3.

### Issue 2: Hardswish analyzer produced zero breadcrumbs

| Field | Value |
|-------|-------|
| Severity | HIGH |
| Phase | Phase 2 -- Reference Analysis |
| Agent | Analyzer (hardswish instance) |
| Verification Dimension | Logging Compliance |
| Retries Consumed | 0 |
| Time Cost | 0 (analysis file was still produced) |

**Problem**: The hardswish analyzer instance produced a 308-line analysis file (`hardswish_analysis.md`) and committed it to git (`c48ffe76c8` at 09:30:04), but it logged zero entries to the shared analyzer breadcrumb file. All other 4 analyzer instances logged 6-8 events each. The execution log also omits hardswish entirely, covering only hardsigmoid, softshrink, hardtanh, and cbrt.

**Root Cause**: Likely a concurrency issue. All 5 analyzers share a single breadcrumb file (`ttnn-unary-sfpu-operation-analyzer_breadcrumbs.jsonl`). The hardswish analyzer may have started writing, encountered a file lock conflict with another instance, and silently failed to append. Alternatively, the hardswish analyzer completed very quickly (~4 minutes vs 14-20 minutes for others) and its breadcrumb writes may have been overwritten by later instances.

**Fix for agents**:
- **Orchestrator**: Use per-operation breadcrumb files (e.g., `analyzer_hardswish_breadcrumbs.jsonl`) instead of a single shared file to eliminate write conflicts.
- **Analyzer**: Add a breadcrumb verification step before completing: read back the breadcrumb file and confirm own events are present.

### Issue 3: Build cache prevented kernel fix from taking effect

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase | Phase 4 -- Testing |
| Agent | Tester |
| Verification Dimension | Implementation Coverage |
| Retries Consumed | 1 free retry |
| Time Cost | ~1-2m |

**Problem**: After the tester fixed the kernel source files (changing floor to trunc semantics), the second test run showed the same failure. The runtime JIT compiler reads kernel headers from `tt-metal-1/build_Release/libexec/`, not from the worktree source tree. Source edits had no effect until the tester manually copied files to the build paths.

**Root Cause**: The worktree's `python_env` symlinks to `tt-metal-1/python_env`, and the runtime loads `.so` and kernel headers from `tt-metal-1`. Worktree kernel source edits are invisible to the runtime.

**Fix for agents**:
- **Tester**: Add to instructions: "After modifying kernel source files, also copy them to the build path used by the runtime. Check `python_env -> ../tt-metal-1/python_env` to find the installation root."
- **Pipeline infrastructure**: Consider adding a `sync_kernels.sh` script that copies worktree kernel headers to the active build directory.

### Issue 4: Implementor started before Phase 3 was declared

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase | Phase 2/3 boundary |
| Agent | Implementor |
| Verification Dimension | Logging Compliance |
| Retries Consumed | 0 |
| Time Cost | 0 (no negative impact) |

**Problem**: The implementor's breadcrumb `start` event timestamp is `09:30:22`, but the orchestrator did not declare Phase 3 start until `09:42:04` -- a 12-minute gap. The implementor's `references_parsed` event at `09:41:45` is just before Phase 3 start. This suggests the implementor was launched during Phase 2 (while analyzers were still running) and spent ~11 minutes reading references before the orchestrator officially started Phase 3.

**Root Cause**: The orchestrator likely launched the implementor early to overlap with the final minutes of Phase 2 analysis. The implementor's `references_parsed` event lists `["trunc","floor","ceil","softshrink","rpow"]` -- these are existing codebase operations, not the discoverer's selected references. This suggests the implementor started by exploring the codebase independently while waiting for analysis files.

**Fix for agents**:
- **Orchestrator**: Either (a) document the early-launch strategy as intentional, or (b) hard-gate Phase 3 on Phase 2 completion to ensure all analysis files are available before the implementor starts.

### Issue 5: Tester Phase 4 duration gap (39m 45s vs 23m 20s)

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase | Phase 4 -- Testing |
| Agent | Tester / Orchestrator |
| Verification Dimension | Logging Compliance |
| Retries Consumed | 0 |
| Time Cost | ~16m unaccounted |

**Problem**: The orchestrator's Phase 4 spans `09:50:46` to `10:30:31` (39m 45s), but the tester's breadcrumbs span `09:48:41` to `10:12:01` (23m 20s). There is a ~16-minute gap between the tester's `complete` event (`10:12:01`) and the orchestrator's `phase_complete` (`10:30:31`). The git log shows the tester's PASS commit at `10:15:11` and then another PASS commit at `10:30:42`. This suggests the tester was launched TWICE -- once completing at `10:12:01`/`10:15:11`, and again completing at `10:30:42`.

**Root Cause**: The orchestrator may have run the tester a second time (perhaps to produce the final test file with the simplified pytest-style tests visible in `test_frac.py` on disk). The first tester run used an exhaustive bfloat16/fp32 ULP test; the second run may have produced the simpler pytest-style test file.

**Fix for agents**:
- **Orchestrator**: Log a `subagent_launched` event for every tester launch, including relaunches. This would make the second tester invocation visible in breadcrumbs.

### Issue 6: Implementor breadcrumb references mismatch

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase | Phase 3 -- Implementation |
| Agent | Implementor |
| Verification Dimension | Implementation Coverage |
| Retries Consumed | 0 |
| Time Cost | 0 |

**Problem**: The implementor's `references_parsed` breadcrumb lists `["trunc","floor","ceil","softshrink","rpow"]` as references, but the discoverer selected `["cbrt","hardtanh","hardsigmoid","hardswish","softshrink"]`. Only `softshrink` overlaps. The implementation notes correctly cite all 5 discoverer-selected references.

**Root Cause**: The `references_parsed` event logged the operations the implementor found during its initial codebase exploration (trunc, floor, ceil -- existing operations with similar semantics), not the analyzer-provided references. This is a breadcrumb accuracy issue, not a functional problem.

**Fix for agents**:
- **Implementor**: The `references_parsed` event should log the references from the analysis files, not from codebase exploration. Add a field distinguishing `analysis_references` from `codebase_references`.

---

## 7. Efficiency Analysis

### Per-Phase Breakdown

| Phase | Duration | Status | Bottleneck |
|-------|----------|--------|------------|
| 1: Discovery | 4m 45s | OK | Clean -- discoverer finished promptly |
| 2: Analysis | 16m 10s | OK | cbrt analyzer was slowest (19m 53s, finished 5m after Phase 2 declared complete) |
| 3: Implementation | 8m 37s | OK | Layer 1 (SFPU kernel) was the core work; layers 2-12 took <4m combined |
| 4: Testing | 39m 45s | OK | floor-vs-trunc semantics fix (H1) consumed most time; build cache issue consumed 1 extra run |
| 5: Documentation | 0m 30s | OK | Clean |

### Tester Iteration Breakdown

| Attempt | Result | Error Type | Fix Applied | Duration |
|---------|--------|-----------|-------------|----------|
| 1 | FAIL | numerical_error | Identified floor-vs-trunc mismatch (H1, HIGH confidence) | ~1m (09:49:22 to 09:50:13) |
| 2 | FAIL | infrastructure | Kernel source edits not picked up by runtime; copied to build paths | ~2m (09:50:13 to ~10:06:18) -- includes ~14m hypothesis + fix time |
| 3 | PASS | - | - | ~4m (10:06:18 to 10:10:43) |

**Note**: The tester's hypothesis H1 was formulated at `10:04:35` (14 minutes after the first failure). This 14-minute gap was spent writing and running a diagnostic script to analyze the 17024 mismatches. While this is a significant time investment, the resulting diagnosis was precise and correct.

### Time Sinks

| Rank | Area | Agent | Duration | % of Total | Description |
|------|------|-------|----------|------------|-------------|
| 1 | Semantics debugging | Tester | ~15m | 21% | Diagnosing floor-vs-trunc mismatch: writing diagnostic script, analyzing 17024 mismatches, formulating H1, applying fix |
| 2 | Analysis wall time | Analyzers | 16m 10s | 23% | 5 parallel analyzers, limited by cbrt (19m 53s) |
| 3 | Unaccounted Phase 4 gap | Tester/Orchestrator | ~16m | 23% | Gap between tester `complete` (10:12) and orchestrator `phase_complete` (10:30); possibly a second tester launch |

---

## 8. Inter-Agent Communication

| Handoff | Source -> Target | Artifact | Quality | Issues | Suggestion |
|---------|-----------------|----------|---------|--------|------------|
| 1 | Generator -> Discoverer | Math definition | POOR | Math definition `x - floor(x)` is incorrect for `torch.frac()` | Validate math definition against golden function before passing to pipeline |
| 2 | Discoverer -> Analyzers | Reference list | GOOD | All 5 references well-selected with clear rationale; discoverer noted correct trunc semantics | Discoverer should flag definition corrections as `upstream_feedback` |
| 3 | Analyzers -> Implementor | Analysis files | GOOD | 5/5 analysis files produced; thorough dispatch tracing and kernel source analysis | Ensure all 5 are available before implementor starts (not early-launched) |
| 4 | Implementor -> Tester | Implementation notes | ADEQUATE | Notes contained file paths and algorithm description but the math definition was wrong (floor-based) | Include a "Verification Examples" section with 3-5 input/output pairs |
| 5 | Tester -> Impl-Notes | File manifest | GOOD | Tester updated implementation notes with test results and debug log; impl-notes agent enriched with embedded source code | No issues |

---

## 9. Comparison with Known Issues

### Known Issues Encountered

| Issue # | Title | Encountered? | Notes |
|---------|-------|-------------|-------|
| 1 | Numerical debugging burns context | YES | Tester spent ~15m debugging floor-vs-trunc mismatch, though with a systematic diagnostic approach |
| 13 | Phase 1/2 overlap | YES (variant) | Implementor started during Phase 2 (09:30:22 vs Phase 3 start 09:42:04) |
| 15 | Kernel writer missing execution logs | YES (analogous) | Implementor agent produced no execution log |
| 18 | Agent relaunch loses debugging context | NO | Not applicable -- no agent relaunches |

### New Issues Discovered

| Title | Description | Suggested Priority |
|-------|-------------|-------------------|
| Incorrect math definition from orchestrator | `frac(x) = x - floor(x)` should be `x - trunc(x)` for `torch.frac()`. Caused 1 hard retry. | HIGH |
| Shared analyzer breadcrumb file causes data loss | 5 parallel analyzers share one breadcrumb file. Hardswish analyzer logged 0 events (likely write conflict). | HIGH |
| Worktree kernel edits invisible to runtime | JIT compiler reads headers from `tt-metal-1/build_Release/libexec/`, not worktree source tree. Caused 1 free retry. | MEDIUM |
| Agent breadcrumbs lost from working tree | Implementor and tester breadcrumbs were committed but later lost from the working tree (cross-worktree transition). | MEDIUM |

---

## 10. Actionable Recommendations

### Recommendation 1: Validate math definition against golden function

- **Type**: new_validation
- **Target**: Orchestrator (generator) agent instructions
- **Change**: After defining the math formula, compute 5 test values using both the formula and the golden function (e.g., `torch.frac(-1.5)`, `torch.frac(1.5)`, `torch.frac(0)`, `torch.frac(-0.3)`, `torch.frac(100.7)`). If any mismatch, update the math definition before proceeding.
- **Expected Benefit**: Eliminates the most expensive tester debugging cycle (semantics mismatch)
- **Priority**: HIGH
- **Effort**: SMALL

### Recommendation 2: Use per-operation analyzer breadcrumb files

- **Type**: logging_fix
- **Target**: Orchestrator and analyzer agent instructions
- **Change**: Instead of all 5 analyzers writing to `ttnn-unary-sfpu-operation-analyzer_breadcrumbs.jsonl`, each instance writes to `analyzer_{operation}_breadcrumbs.jsonl` (e.g., `analyzer_hardswish_breadcrumbs.jsonl`).
- **Expected Benefit**: Eliminates write conflicts that caused the hardswish analyzer to log 0 events
- **Priority**: HIGH
- **Effort**: SMALL

### Recommendation 3: Add worktree kernel deployment documentation

- **Type**: instruction_change
- **Target**: Tester agent instructions and CLAUDE.md
- **Change**: Add note: "In worktree development, kernel source files must be deployed to the build directory used by the runtime. Check where `python_env` symlinks to, and copy modified kernel headers to `{runtime_root}/build_Release/libexec/...` and `{runtime_root}/tt_metal/hw/ckernels/...`."
- **Expected Benefit**: Saves 1 free retry per run when kernel changes are made during testing
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 4: Require implementor execution log

- **Type**: instruction_change
- **Target**: Implementor agent instructions
- **Change**: Add mandatory execution log generation at session end, matching the tester's format: Input Interpretation, per-layer Execution Timeline, Deviations, Artifacts, Handoff Notes, Instruction Recommendations.
- **Expected Benefit**: Provides structured implementation context for self-reflection analysis
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 5: Preserve agent breadcrumbs across worktree transitions

- **Type**: pipeline_change
- **Target**: Worktree management scripts and orchestrator
- **Change**: Either (a) copy agent_logs to a persistent location outside the worktree before cleanup, or (b) ensure the orchestrator's final commit includes all agent breadcrumb files.
- **Expected Benefit**: Prevents breadcrumb data loss for post-pipeline analysis
- **Priority**: MEDIUM
- **Effort**: MEDIUM

---

## 11. Overall Assessment

### Pipeline Maturity Score

| Dimension | Score | Notes |
|-----------|-------|-------|
| Reference discovery accuracy | 5 | All 5 references were useful; discoverer identified the correct semantics |
| Reference analysis quality | 4 | 5/5 analyses produced; 4/5 had full breadcrumbs; hardswish logging gap |
| Implementation completeness | 5 | All 12 layers present; correct identification of pre-existing layers |
| SFPI compliance | 5 | Pure SFPI with all quality checks passing except fp32 specialization |
| Testing thoroughness | 4 | Both dtypes tested; ULP 0.0; but fp32 parametrization not in final test file |
| Inter-agent communication | 3 | Incorrect math definition propagated from orchestrator through implementor; discoverer noted the correction but it was not acted on |
| Logging/observability | 3 | 4/6 agents have breadcrumbs on working tree; hardswish analyzer 0 events; implementor no execution log; some breadcrumbs lost from working tree |

### Top 3 Things to Fix

1. **Validate math definitions against golden functions before implementation** -- the floor-vs-trunc mismatch cost 15+ minutes of debugging and was entirely preventable. The discoverer even identified the correct semantics but the correction was not propagated.

2. **Use per-operation analyzer breadcrumb files** -- the shared breadcrumb file caused complete data loss for the hardswish analyzer. This is a systemic risk that affects every parallel analyzer run.

3. **Ensure agent breadcrumb files persist on the working tree** -- 3 of 6 agent breadcrumb files were lost from the working tree, requiring git archaeology to reconstruct. The orchestrator's final commit should include all agent log files.

### What Worked Best

The reference discovery phase was the strongest aspect of this pipeline run. The discoverer selected 5 references that were all directly useful to the implementor, correctly identified the exact SFPI primitives needed (exexp, reinterpret, bit masking), and even noted the trunc-vs-floor semantics correction in its output. The resulting implementation achieved bit-exact accuracy (ULP 0.0) on both bfloat16 and fp32, demonstrating that the IEEE 754 bit-manipulation approach suggested by the cbrt reference was the ideal algorithm for this operation.
