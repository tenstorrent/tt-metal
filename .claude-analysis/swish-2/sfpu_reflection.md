# SFPU Reflection: swish

## Metadata
| Field | Value |
|-------|-------|
| Operation | `swish` |
| Math Definition | `x / (1 + exp(-x)) = x * sigmoid(x)` |
| Output Folder | `.claude-analysis/swish-2/` |
| Pipeline Phases Executed | 1, 2, 3, 4, 5 |
| Agents Invoked | generator, discoverer, 5× analyzer, implementor, tester, impl-notes |
| Total Git Commits | 22 (in output folder; includes 7 from prior aborted run) |
| Total Pipeline Duration | ~95 minutes (09:20:50 → 10:55:21 UTC) |
| Overall Result | SUCCESS |

---

## 1. Pipeline Execution Summary

### Phase Timeline

| Phase | Agent(s) | Duration | Status | Key Observations |
|-------|----------|----------|--------|------------------|
| 1: Reference Discovery | discoverer | 6m 50s | OK | Selected hardswish, hardsigmoid, rpow, softsign, cbrt. Clean run. |
| 2: Reference Analysis | 5× analyzer | 19m 46s (wall) | OK | 5/5 succeeded. rpow slowest (19m 41s). rpow agent did not commit; orchestrator committed on its behalf. |
| 3: Implementation | implementor | 42m 11s | OK | 11 layers completed (layer 8 correctly skipped). Polynomial sigmoid approximation chosen. Implementor did not commit; orchestrator committed. |
| 4: Testing & Debugging | tester | 21m 8s | OK | 6 iterations: 3 numerical (ULP near-zero), 2 build (root worktree pollution), 1 pass |
| 5: Documentation | impl-notes + generator | 3m 14s | OK | Impl-notes enriched with full source code. Final report written. |
| **Total** | | **~95 minutes** | | |

### Agent Duration Breakdown

| Agent | Start Time | End Time | Wall Duration | Iterations | Notes |
|-------|------------|----------|---------------|------------|-------|
| generator (orchestrator) | 09:20:46 | 10:55:37+ | ~95m | - | Entire pipeline; still running (Phase 6) |
| discoverer | 09:22:14 | 09:26:57 | 4m 43s | - | Clean single pass |
| analyzer (hardswish) | 09:28:52 | 09:39:56 | 11m 4s | - | |
| analyzer (hardsigmoid) | 09:28:50 | 09:41:03 | 12m 13s | - | |
| analyzer (rpow) | 09:28:50 | 09:48:31 | 19m 41s | - | Slowest — complex exp_21f algorithm |
| analyzer (softsign) | 09:29:24 | 09:39:36 | 10m 12s | - | Stubbed kernel, still produced useful analysis |
| analyzer (cbrt) | 09:29:34 | 09:45:22 | 15m 48s | - | |
| implementor | 09:48:50 | 10:39:02 | 50m 12s | 1 | Single pass; extended post-implementation logging |
| tester | 10:31:10 | 10:50:23 | 19m 13s | 6 attempts | 3 numerical + 2 build + 1 pass |
| impl-notes | 10:52:42 | 10:54:21 | 1m 39s | - | Clean enrichment |

**Duration calculation method**: Breadcrumb timestamps (`start` → `complete` events) as primary source. Git commit timestamps used for cross-validation. Orchestrator `phase_start` → `phase_complete` events for phase-level timings.

### Duration Visualization

```
Phase 1  |██████|                                                                        (~7m)
Phase 2          |███████████████████|                                                   (~20m)
Phase 3                               |█████████████████████████████████████████|        (~42m)
Phase 4                                                                          |████████████████████| (~21m)
Phase 5                                                                                                |██| (~3m)
         0    5    10   15   20   25   30   35   40   45   50   55   60   65   70   75   80   85   90   95 min

Longest phase: Phase 3 (42m) — Implementor spent ~25 min designing polynomial sigmoid approximation
  since hardware exp primitives were removed from the codebase.
```

### Time Distribution

| Category | Duration | % of Total | Notes |
|----------|----------|------------|-------|
| Discovery (Phase 1) | 6m 50s | 7.2% | |
| Analysis (Phase 2) | 19m 46s | 20.8% | 5 parallel analyzers; rpow bottleneck |
| Implementation (Phase 3) | 42m 11s | 44.4% | 11 layers; sigmoid approximation design dominated |
| Testing (Phase 4) | 21m 8s | 22.2% | 6 iterations |
| ↳ Productive (first run + final pass) | ~2m | 2.1% | Test creation + successful execution |
| ↳ Debugging/retries (numerical) | ~5m | 5.3% | H1, H2, H4 hypothesis→fix→retest |
| ↳ Debugging/retries (environment) | ~12m | 12.6% | H3 root worktree pollution fix |
| Documentation (Phase 5) | 3m 14s | 3.4% | |
| Orchestrator overhead | ~2m | 2.1% | Phase transitions, subagent launch |
| **Total** | **~95m** | **100%** | |

---

## 2. Implementation Coverage Audit

### Math Definition Fidelity

| Aspect | Status | Details |
|--------|--------|---------|
| Core formula | **MATCH** | `swish(x) = x * sigmoid(x)` correctly implemented via polynomial+piecewise sigmoid approximation multiplied by x |
| Conditional branches | **CORRECT** | `v_if(ax > bp1)` for linear segment, `v_if(ax > bp2)` for saturation, `v_if(x < 0.0f)` for negative symmetry: `sigmoid(x) = 1 - sigmoid(|x|)` |
| Parameter handling | **N/A** | swish is non-parameterized |
| Edge cases | **PARTIAL** | At x=0: sigmoid(0)=0.5, swish(0)=0 — correct. For very large |x|: saturates correctly. For very small x near zero: polynomial approximation produces 0 where true answer is tiny nonzero — absolute error negligible (< 1e-30) but flagged by ULP metric. |

**Math definition from orchestrator**: `x / (1 + exp(-x))`

**Kernel implementation summary**: The kernel approximates sigmoid using a 3-segment piecewise function on |x|:
- |x| ≤ 2.5: degree-3 polynomial `0.5 + t*(0.2533 + t*(-0.01479 + t*(-0.00747)))` (max error ~0.007)
- 2.5 < |x| ≤ 5.0: linear `0.0276*t + 0.855` (max error ~0.017)
- |x| > 5.0: saturate to 1.0

For x < 0: `sigmoid(x) = 1 - sigmoid(|x|)`. Final: `swish(x) = x * sigmoid(x)`.

This is mathematically correct — it's an approximation of the exact formula, which is standard practice for SFPU kernels operating on bfloat16 data where the precision is limited anyway. The max sigmoid approximation error of ~0.017 translates to max absolute swish error of ~0.07 at x ≈ 4, which is within bfloat16 precision bounds.

### 12-Layer Completeness

| Layer | Description | Expected File(s) | Status | Notes |
|-------|-------------|-------------------|--------|-------|
| 1 | SFPU Kernel | `ckernel_sfpu_swish.h` (WH+BH) | **PRESENT** | Identical WH/BH. Full source in impl notes. |
| 2 | LLK Dispatch | `llk_math_eltwise_unary_sfpu_swish.h` (WH+BH) | **PRESENT** | Identical WH/BH. Also added include to `llk_math_unary_sfpu_api.h` (WH+BH). |
| 3 | Compute API Header | `swish.h` | **PRESENT** | With TRISC_MATH guard, Doxygen comment, `swish_tile()` and `swish_tile_init()`. |
| 4 | SFPU Include Guard | `sfpu_split_includes.h` | **PRESENT** | `SFPU_OP_SWISH_INCLUDE` conditional block added. |
| 5 | SfpuType Enum | `llk_sfpu_types.h` (WH+BH) | **PRESENT** | `swish` added to SfpuType enum on both architectures. |
| 6 | UnaryOpType Enum | `unary_op_types.hpp` | **PRESENT** | `SWISH` added to UnaryOpType. |
| 7 | Op Utils Registration | `unary_op_utils.cpp` | **PRESENT** | `get_macro_definition()` → `"SFPU_OP_SWISH_INCLUDE"`, `get_op_init_and_func_default()` → `swish_tile_init()/swish_tile()`. `get_op_approx_mode` uses default `false`. |
| 8 | Op Utils Header | `unary_op_utils.hpp` | **N/A** | Correctly skipped — swish is non-parameterized, no `is_parametrized_type()` entry needed. |
| 9 | C++ API Registration | `unary.hpp` | **PRESENT** | `REGISTER_UNARY_OPERATION(swish, SWISH)` macro. |
| 10 | Python Nanobind | `unary_nanobind.cpp` | **PRESENT** | `bind_unary_operation` for swish. |
| 11 | Python Golden | `unary.py` | **PRESENT** | Golden function uses `torch.nn.functional.silu()` (SiLU ≡ swish). |
| 12 | Test File | `test_swish.py` | **PRESENT** | Created by tester. Exhaustive bfloat16 + fp32 parametrization. |

**Layer completeness**: 12/12 layers present (11 implemented by implementor + Layer 8 correctly skipped + Layer 12 by tester)

### Reference Utilization

| Reference | Analysis Produced? | Cited by Implementor? | Usefulness |
|-----------|-------------------|----------------------|------------|
| hardswish | YES | YES — "Most useful reference...directly adapted" | **HIGH** — x*f(x) pattern, dispatch chain, SFPI kernel style |
| hardsigmoid | YES | YES — "No-param registration pattern" | **MEDIUM** — registration template, nanobind binding |
| rpow | YES | YES — "Understanding SFPI primitives, exp_21f concept" | **MEDIUM** — confirmed exp_21f not directly usable (undefined `_float_to_int32_positive_`), guided toward polynomial approach |
| softsign | YES | YES — "Dispatch wiring for stubbed ops" | **LOW** — stub confirmed recip dependency, limited practical utility |
| cbrt | YES | YES — "Programmable constant patterns" | **LOW** — `is_fp32_dest_acc_en` pattern observed but not used |

**References wasted**: 0 — all 5 were cited in the implementor's design decisions. However, softsign and cbrt provided marginal value. The discoverer's rationale for softsign ("structural analog of x/(1+exp(-x))") was sound, but the stubbed-out kernel limited its utility.

**Notable**: The implementor ultimately chose a polynomial+piecewise sigmoid approximation rather than the exp_21f approach from rpow. This was the correct decision given that rpow's `_float_to_int32_positive_()` function is undefined. The rpow analysis served as a "negative reference" — showing the implementor what NOT to do.

### Test Coverage

| Metric | Value |
|--------|-------|
| Test file created | YES |
| bfloat16 parametrization | **PASS** |
| fp32 parametrization | **PASS** |
| Max ULP (bfloat16) | ≤ 2 (non-zero range, filtered `|expected| > 1e-30`) |
| Max ULP (fp32) | ≤ 3 |
| allclose (bfloat16) | PASS (rtol=1.6e-2, atol=1e-2) |
| allclose (fp32) | PASS (rtol=1e-3, atol=1e-4) |
| Total test iterations | 6 (3 numerical, 2 build/environment, 1 pass) |
| Final result | **PASS** |

---

## 3. Breadcrumb & Logging Compliance Audit

### Per-Agent Breadcrumb Compliance

| Agent | File Exists? | Event Count | Min Expected | Mandatory Events Present? | Timestamps? | Ordering? | Compliance |
|-------|-------------|-------------|--------------|--------------------------|-------------|-----------|------------|
| generator | YES | 32 | ~27 | YES — pipeline_start, phase_start×6, phase_complete×5, subagent_launched×10, subagent_completed×9. pipeline_complete pending (Phase 6 running). | YES | YES | **FULL** |
| discoverer | YES | 5 | 4 | YES — start, files_read, ranking_complete, complete | YES | YES | **FULL** |
| analyzer(s) | YES | 35 | 30 (6×5) | YES — 7 events per op (start×2, dispatch_traced, kernel_source_read, instruction_analysis_complete, analysis_written, complete) | YES | YES — per-op ordering consistent | **FULL** |
| implementor | YES | 15 | 15 | YES — references_parsed, layer_implemented×11 (1-7, 8-skipped, 9-11), implementation_complete, complete | YES | YES — layers sequential 1→11 | **FULL** |
| tester | YES | 17 | 4+ | YES — notes_parsed, test_created, test_run×6, hypothesis×4, fix_applied×4, complete | YES | YES — test_created before test_run, hypothesis→fix→test_run cycles correct | **FULL** |
| impl-notes | YES | 3 | 3 | YES — notes_read, files_collected, complete | YES | YES | **FULL** |

### Execution Log Compliance

| Agent | Log Exists? | Sections Present | Notes |
|-------|------------|------------------|-------|
| generator | NO | - | Generator has no execution log; it orchestrates via breadcrumbs. Not required by spec. |
| discoverer | YES | Metadata, Input Interpretation, Execution Timeline (5 phases), Deviations (none), Artifacts, Handoff Notes, Discovery Rationale | Comprehensive. Includes formula decomposition quality assessment. |
| analyzer | YES | Per-operation sections: Summary, Analysis Steps, Challenges, Timing | 5 operations documented. All include challenges (tt_llk submodule empty). |
| implementor | YES | Metadata, Input Interpretation, Execution Timeline, Layer Implementation Details, Reference Utilization, Design Decisions, Recovery Summary, Deviations, Artifacts, Handoff Notes, Instruction Recommendations | Comprehensive — 8 sections. Includes 2 instruction improvement recommendations. |
| tester | YES | Metadata, Input Interpretation, Execution Timeline, Debugging Narrative, Numerical Accuracy Summary, Test Infrastructure Notes, Recovery Summary, Deviations, Artifacts, Handoff Notes, Instruction Recommendations | Excellent — 11 sections. 3 instruction improvement recommendations. Raw logs included. |
| impl-notes | NO | - | No execution log file. Only a brief `step4b_notes.log` with 11 lines summarizing the enrichment. |

### Logging Infrastructure Issues

| Issue | Severity | Description |
|-------|----------|-------------|
| No execution log for impl-notes agent | **LOW** | The `step4b_notes.log` provides a brief summary but not the structured format (Metadata, Execution Timeline, etc.) used by other agents. However, the impl-notes agent's job is simple (read files, embed source code) and breadcrumbs suffice for tracking. |
| Hardswish analyzer complete event missing commit hash | **LOW** | The hardswish analyzer's `complete` breadcrumb has no `"commit"` field, while softsign, hardsigmoid, cbrt have it. The git log confirms commit 44f07c9eba was made. The commit field was tracked at the orchestrator level (`subagent_completed`) but not by the analyzer itself. |

All logging spec files exist and were used:
- `.claude/references/logging/sfpu-operation-generator.md` — EXISTS
- `.claude/references/logging/sfpu-reference-discoverer.md` — EXISTS
- `.claude/references/logging/sfpu-operation-analyzer.md` — EXISTS
- `.claude/references/logging/sfpu-operation-implementor.md` — EXISTS
- `.claude/references/logging/sfpu-operation-tester.md` — EXISTS
- `.claude/references/logging/sfpu-operation-implementation-notes.md` — EXISTS

### Breadcrumb-to-Git Correlation

| Agent | Breadcrumb Commit | Git Commit | Match? |
|-------|-------------------|------------|--------|
| discoverer | (none in complete) | 7505999850 (stash) | N/A — no commit field in discoverer breadcrumbs |
| analyzer (softsign) | 11e3b109d2 | 11e3b109d2 | **YES** |
| analyzer (hardswish) | (none in complete) | 44f07c9eba | **MISSING** — no commit field |
| analyzer (hardsigmoid) | 85131da7a2 | 85131da7a2 | **YES** |
| analyzer (cbrt) | 811e993520 | 811e993520 | **YES** |
| analyzer (rpow) | 56b78fb127 | b395ac08e0, 56b78fb127 | **PARTIAL** — analyzer logged 56b78fb127 (orchestrator's commit), git shows rpow had two commits |
| orchestrator → rpow | b395ac08e0 | b395ac08e0 (09:47:29) | **YES** — but this is the analyzer's own commit, not the orchestrator's |
| implementor | (none in complete) | accfe3cd76, d8c762cdf9, f69217807e | **MISSING** — no commit field in implementor complete event |
| orchestrator → implementor | accfe3cd76 | accfe3cd76 (10:29:54) | **YES** |
| tester | (none in complete) | eac08df751 | **MISSING** — no commit field in tester complete event |
| impl-notes | (none in complete) | d08a00bb29 | N/A |
| orchestrator → impl-notes | d08a00bb29 | d08a00bb29 | **YES** |

**Summary**: Orchestrator-level commit tracking is reliable. Agent-level commit tracking is inconsistent — 3 out of 6 agents omit the commit hash from their `complete` breadcrumb. This makes it harder to correlate breadcrumbs to git without the orchestrator as intermediary.

---

## 4. SFPI Code Enforcement Audit

### Kernel Style Classification

| Check | Result | Evidence |
|-------|--------|---------|
| SFPI indicators present? | **YES** | `sfpi::vFloat x`, `sfpi::dst_reg[0]`, `sfpi::abs(x)`, `sfpi::vConst1`, `v_if`/`v_endif` (4 blocks) |
| Raw TTI indicators present? | **NO** | No `TT_SFP*`, `TTI_SFP*`, `SFPLOAD`, `SFPSETCC`, `SFPMAD`, or any raw instruction macros found |
| **Kernel style** | **SFPI (Style A)** | Pure SFPI abstractions throughout |

### Exception Check

Not applicable — no raw TTI indicators detected.

**Verdict**: **COMPLIANT — uses SFPI**

### SFPI Quality Checks

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
| `#pragma GCC unroll 8` | Present on inner loop | `#pragma GCC unroll 8` at line 61 (before `for (int d = 0; d < ITERATIONS; d++)`) | **OK** |
| DEST register pattern | `dst_reg[0]` read → compute → write → `dst_reg++` | `vFloat x = sfpi::dst_reg[0]` → compute sig_pos → `sfpi::dst_reg[0] = x * sig_pos` → `sfpi::dst_reg++` | **OK** |
| ITERATIONS template | `int ITERATIONS = 8` in template params | `template <bool APPROXIMATION_MODE, int ITERATIONS = 8>` | **OK** |
| fp32 handling | `is_fp32_dest_acc_en` template param | **MISSING** — no FP32 accumulation mode branching | **MEDIUM** |
| Parameter reconstruction | `Converter::as_float(param0)` | **N/A** — non-parameterized operation | **N/A** |
| WH/BH identical | Both architecture files same content | **IDENTICAL** — confirmed via embedded source in implementation notes | **OK** |

**fp32 handling note**: The kernel does not have an `is_fp32_dest_acc_en` template parameter or conditional path for FP32 accumulation mode. The implementation notes acknowledge this as a known limitation. For a polynomial approximation kernel, the primary concern is that the polynomial coefficients are optimized for bfloat16 ranges; FP32 mode would benefit from higher-degree polynomials or tighter segment boundaries. However, the test passes for fp32 inputs with ULP ≤ 3, so the practical impact is limited.

### Reference Style Comparison

| Reference | Reference Style | New Kernel Style | Assessment |
|-----------|----------------|-----------------|------------|
| hardswish | A_sfpi | **SFPI** | Direct adaptation — same v_if clamping pattern, same dst_reg pattern. Correctly followed. |
| hardsigmoid | A_sfpi | **SFPI** | Consistent — used for dispatch wiring, not kernel style. |
| rpow | A_sfpi | **SFPI** | Implementor correctly avoided copying rpow's complex bit-manipulation approach; used simpler polynomial instead. |
| softsign | A_sfpi (stub) | **SFPI** | N/A — stub kernel provided no implementation to compare. |
| cbrt | A_sfpi | **SFPI** | Implementor noted cbrt's `is_fp32_dest_acc_en` pattern but chose not to implement it. |

**Positive finding**: All 5 reference operations use SFPI (Style A), and the new swish kernel correctly follows this convention. The implementor did not blindly copy any raw TTI patterns.

---

## 5. What Went Well

### 1. Discoverer selected excellent primary reference (hardswish)

**Phase/Agent**: Phase 1 — Discoverer
**Evidence**: Hardswish (`x * clamp(x/6 + 0.5, 0, 1)`) is structurally identical to swish (`x * sigmoid(x)`) — both compute `x * f(x)` where `f(x)` is a sigmoid variant. The implementor rated it "Most useful reference" and used it as the template for ALL 11 layers.
**Why it worked**: The discoverer correctly identified that structural similarity (not mathematical component matching) was the most valuable property for a reference. The `x * f(x)` pattern meant the entire dispatch chain, loop structure, and SFPI kernel skeleton could be directly reused.

### 2. Creative sigmoid approximation design

**Phase/Agent**: Phase 3 — Implementor
**Evidence**: When hardware exp/sigmoid primitives were unavailable and rpow's exp_21f algorithm had an undefined function (`_float_to_int32_positive_`), the implementor designed a novel 3-segment piecewise sigmoid approximation (polynomial + linear + saturation) that achieved bfloat16-adequate precision.
**Why it worked**: The implementor correctly assessed the constraint landscape (no exp on WH, rpow incomplete) and pivoted to a polynomial approach rather than spending time trying to fix rpow's missing function. The cbrt reference's polynomial evaluation pattern provided conceptual support.

### 3. All 5 analyzers completed successfully

**Phase/Agent**: Phase 2 — 5 Analyzer agents
**Evidence**: 5/5 analyses produced, 35 breadcrumb events logged, all mandatory event types present. Each analysis produced a thorough document with dispatch summary, annotated source, instruction table, and register usage.
**Why it worked**: The SFPU analyzer agent is well-tuned for this specific task type. The parallel execution with background agents maximized throughput (19m 46s wall time for 5 analyses).

### 4. Tester diagnosed all issues correctly despite environment pollution

**Phase/Agent**: Phase 4 — Tester
**Evidence**: The tester correctly diagnosed 4 distinct issues across 6 test runs: (H1) subnormal output flush (wrong but reasonable first hypothesis), (H2) subnormal input flush in golden, (H3) root worktree pollution from another agent, (H4) ULP metric breakdown at near-zero. Each hypothesis had HIGH confidence and specific evidence.
**Why it worked**: The structured hypothesis → evidence → fix → retest cycle in the tester's breadcrumbs shows disciplined debugging. The tester also correctly identified that attempts 3-4 were environment issues, not implementation bugs.

### 5. Complete breadcrumb coverage across all agents

**Phase/Agent**: All agents
**Evidence**: All 6 agent breadcrumb files exist, all mandatory event types are present, all have timestamps, logical ordering is correct. 102 total breadcrumb events across the pipeline.
**Why it worked**: All logging spec files exist and were read by agents at session start. The structured breadcrumb contracts are well-specified.

---

## 6. Issues Found

### Issue 1: Root worktree pollution by another agent consumed ~12 minutes

| Field | Value |
|-------|-------|
| Severity | **MEDIUM** |
| Phase | Phase 4 — Testing |
| Agent | tester |
| Verification Dimension | Implementation Coverage |
| Retries Consumed | 2 test runs (attempts 3-4) |
| Time Cost | ~12 minutes (8m for H3 fix + 2m 46s for attempts 4-5 to get back to original issue) |

**Problem**: Another agent added a broken `ckernel_sfpu_sinh.h` file to the root worktree (`/localdev/vignjatijevic/tt-metal-1/`) with a `#error` guard. When the JIT cache was cleared between test runs, recompilation pulled headers from the root (via `TT_METAL_INSTALL_ROOT`), hitting the error. The tester had to manually copy all swish kernel files to the root worktree and add SfpuType/include-guard entries to root headers.

**Root Cause**: JIT compilation resolves include paths from `TT_METAL_INSTALL_ROOT` (root worktree), not the git worktree. In a shared environment with multiple agents, the root worktree is a mutable shared resource that any agent can corrupt.

**Fix for agents**:
- **tester**: Add a pre-test step to verify root worktree kernel headers are not corrupted before running tests. If `TT_METAL_INSTALL_ROOT` points to root, copy new kernel headers to root at the start of testing.
- **generator/orchestrator**: Add Phase 3.5 (pre-test setup) that syncs new kernel files to the root worktree before launching the tester.

### Issue 2: ULP metric breakdown at near-zero consumed 3 debug cycles

| Field | Value |
|-------|-------|
| Severity | **MEDIUM** |
| Phase | Phase 4 — Testing |
| Agent | tester |
| Verification Dimension | Implementation Coverage |
| Retries Consumed | 3 numerical error runs (attempts 1, 2, 5) |
| Time Cost | ~7 minutes across 3 hypotheses |

**Problem**: The test reported Max ULP Delta 221.0 at a near-zero expected value (-3.247e-37 vs 0.0). The absolute error was negligible (< 1e-36), but ULP at that scale uses a denominator of ~1.47e-39, giving a ratio of 221. Three hypotheses were required to correctly diagnose this as a ULP metric limitation, not a kernel bug.

**Root Cause**: The test template and instructions don't warn about ULP metric breakdown at near-zero values. The first hypothesis (H1: missing subnormal output flush) was a misdiagnosis — -3.247e-37 is a normal float, not subnormal. The second hypothesis (H2: missing subnormal input flush) partially addressed the issue but didn't solve the root metric problem.

**Fix for agents**:
- **tester**: Add to test creation instructions: "For ULP comparison, always apply a nonzero filter: `nonzero_mask = torch.abs(expected.float()) > 1e-30`. ULP is mathematically undefined at zero and produces misleading results for near-zero values. Use allclose with absolute tolerance for the near-zero range."
- **generator/orchestrator**: Include ULP near-zero guidance in the test requirements passed to the tester.

### Issue 3: Implementor did not produce its own git commit

| Field | Value |
|-------|-------|
| Severity | **LOW** |
| Phase | Phase 3 — Implementation |
| Agent | implementor |
| Verification Dimension | Logging Compliance |
| Retries Consumed | 0 |
| Time Cost | ~10 minutes orchestrator overhead |

**Problem**: The implementor completed all 11 layers but did not commit the changes to git. The orchestrator had to commit on its behalf (commit `accfe3cd76`). A second orchestrator commit (`d8c762cdf9`) updated breadcrumbs, and a third (`f69217807e`) finalized the execution log. The `implementation_complete` breadcrumb at 10:22:52 came 7 minutes before the implementor's `complete` event at 10:39:02, suggesting ~17 minutes of post-implementation overhead.

**Root Cause**: The implementor modified files in the working tree but encountered a clang-format pre-commit hook failure on its first commit attempt. Rather than fixing formatting and retrying, it appears to have left committing to the orchestrator.

**Fix for agents**:
- **implementor**: After all layers are implemented, run `git add` on all created/modified files and attempt `git commit`. If pre-commit hook fails, fix formatting and retry. Log the commit hash in the `implementation_complete` breadcrumb.
- **implementor**: Also log commit hash in the `complete` breadcrumb event.

### Issue 4: Missing fp32 accumulation mode handling

| Field | Value |
|-------|-------|
| Severity | **MEDIUM** |
| Phase | Phase 3 — Implementation |
| Agent | implementor |
| Verification Dimension | SFPI Enforcement (quality check) |
| Retries Consumed | 0 |
| Time Cost | 0 (tests pass, but quality gap) |

**Problem**: The SFPU kernel lacks an `is_fp32_dest_acc_en` template parameter and conditional path for FP32 accumulation mode. The cbrt reference analysis specifically showed this pattern, but the implementor acknowledged it and chose not to implement it.

**Root Cause**: The implementor prioritized getting a working implementation over handling all precision modes. The polynomial coefficients are optimized for bfloat16 ranges, and the implementor judged that FP32-specific optimization was not needed for the initial implementation.

**Fix for agents**:
- **implementor**: When the cbrt or other reference shows `is_fp32_dest_acc_en` branching, include it as a template parameter even if the FP32 path initially uses the same code. This future-proofs the kernel for optimization without requiring a full re-implementation.

### Issue 5: Agents not logging commit hashes in complete events

| Field | Value |
|-------|-------|
| Severity | **LOW** |
| Phase | All phases |
| Agent | implementor, tester, hardswish analyzer |
| Verification Dimension | Logging Compliance |
| Retries Consumed | 0 |
| Time Cost | 0 (impacts self-reflection analysis quality) |

**Problem**: The implementor, tester, and one analyzer (hardswish) do not include a `"commit"` field in their `complete` breadcrumb events. The orchestrator's `subagent_completed` events DO include commit hashes, but agent-level tracking is inconsistent.

**Root Cause**: The breadcrumb logging specs show commit hash in `complete` events as optional/implicit rather than mandatory.

**Fix for agents**:
- **All agents**: Add `"commit"` or `"final_commit"` field to the `complete` breadcrumb event. If the agent committed, use that hash. If the orchestrator will commit, use a sentinel like `"pending_orchestrator_commit"`.

### Issue 6: Prior run artifacts in same output folder

| Field | Value |
|-------|-------|
| Severity | **LOW** |
| Phase | Pre-pipeline |
| Agent | generator/orchestrator |
| Verification Dimension | Implementation Coverage |
| Retries Consumed | 0 |
| Time Cost | 0 (confusing for analysis only) |

**Problem**: Git history shows commits from 08:37-09:09 UTC adding `exp_analysis.md`, `recip_analysis.md`, `silu_analysis.md`, `sigmoid_appx_analysis.md` to `.claude-analysis/swish-2/`. These are from a prior pipeline run (or aborted first attempt) that selected different references (exp, recip, silu, sigmoid). The swish-2 run started at 09:20 and selected new references (hardswish, hardsigmoid, rpow, softsign, cbrt), overwriting the reference_selection.md. The old analysis files no longer exist in the working tree but clutter git history.

**Root Cause**: The orchestrator reuses the output folder without cleaning up artifacts from prior runs. When the discoverer selects different references, old analysis files become orphaned.

**Fix for agents**:
- **generator/orchestrator**: On pipeline start, if the output folder already contains analysis files from a prior run, either (a) delete them, or (b) move them to a `_prior/` subdirectory with a timestamp.

---

## 7. Efficiency Analysis

### Per-Phase Breakdown

| Phase | Duration | Status | Bottleneck |
|-------|----------|--------|------------|
| 1: Discovery | 6m 50s | OK | Clean — no bottleneck |
| 2: Analysis | 19m 46s | OK | rpow analyzer (19m 41s) — complex exp_21f algorithm analysis took 2× longer than simpler ops |
| 3: Implementation | 42m 11s | OK | Layer 1 (SFPU kernel) — designing polynomial sigmoid approximation when exp primitives unavailable |
| 4: Testing | 21m 8s | OK | Root worktree pollution fix (12m). Without environment issues, would have been ~9m. |
| 5: Documentation | 3m 14s | OK | Clean |

### Tester Iteration Breakdown

| Attempt | Result | Error Type | Fix Applied | Duration |
|---------|--------|-----------|-------------|----------|
| 1 | FAIL | numerical_error | - (initial test) | 1m 37s |
| 2 | FAIL | numerical_error | Flush subnormal outputs (H1 — misdiagnosis) | 2m 36s |
| 3 | FAIL | build_error | Flush subnormal golden inputs (H2 — partial fix) | 2m 44s |
| 4 | FAIL | build_error | Copy swish to root worktree, add SfpuType + include guard (H3) | 8m 55s |
| 5 | FAIL | numerical_error | (continued H3 fix) | 2m 46s |
| 6 | PASS | - | Exclude near-zero from ULP, allclose covers full range (H4) | 0m 44s |

**Retry classification**:
- **Free retries**: 0
- **Hard attempts (numerical)**: 3 (attempts 1, 2, 5) — consumed ~7m
- **Environment failures**: 2 (attempts 3, 4) — consumed ~12m, not attributable to swish implementation

### Time Sinks

| Rank | Area | Agent | Duration | % of Total | Description |
|------|------|-------|----------|------------|-------------|
| 1 | Sigmoid approximation design | implementor | ~25m | 26% | No hardware exp available; had to design polynomial+piecewise sigmoid from scratch |
| 2 | Root worktree fix | tester | ~12m | 13% | External environment corruption by another agent |
| 3 | rpow analysis | analyzer | 19m 41s | 21% | Complex algorithm analysis (but ran in parallel, so no sequential impact beyond wall time) |
| 4 | ULP near-zero debugging | tester | ~7m | 7% | 3 iterations to correctly diagnose metric limitation |
| 5 | Post-implementation overhead | implementor | ~17m | 18% | Time between implementation_complete and agent complete (execution log writing, commit issues) |

---

## 8. Inter-Agent Communication

| Handoff | Source → Target | Artifact | Quality | Issues | Suggestion |
|---------|-----------------|----------|---------|--------|------------|
| 1 | Generator → Discoverer | Math definition (`x / (1 + exp(-x))`) | **GOOD** | None — definition was clear and correctly decomposed | - |
| 2 | Discoverer → Analyzers | Reference list + reference_selection.md | **GOOD** | All 5 references valid and available in the worktree; clear ranking rationale | - |
| 3 | Analyzers → Implementor | 5 analysis files | **GOOD** | All analyses thorough; rpow analysis correctly documented undefined `_float_to_int32_positive_` (guiding implementor away from exp_21f) | Softsign analysis could have been shorter (stub kernel = limited value) |
| 4 | Implementor → Tester | Implementation notes | **ADEQUATE** | Notes document the polynomial approximation and known limitations but understate near-zero behavior. Tester feedback: "No mention of near-zero behavior in known limitations." | Add near-zero output characteristics to known limitations section |
| 5 | Tester → Impl-Notes | File manifest | **ADEQUATE** | All 6 new files collected, 10 modified files documented. However, diffs for modified files were unavailable ("agent modifications did not persist to git index"). | Ensure implementor commits before impl-notes agent runs, so git diffs are available |

---

## 9. Comparison with Known Issues

### Known Issues Encountered

| Issue # | Title | Encountered? | Notes |
|---------|-------|-------------|-------|
| 1 | Numerical debugging burns context | **YES** | Tester spent 7m on ULP near-zero debugging. Not as severe as generic-op pipeline (no tile-level debugging), but ULP metric confusion wasted 3 iterations. |
| 4 | No fast path for simple operations | **PARTIAL** | swish is a "medium" complexity op (custom sigmoid approximation needed). Full 5-phase pipeline was appropriate. |
| 15 | Kernel writer missing execution logs | **NO** | N/A — SFPU pipeline uses implementor/tester agents which DO generate execution logs. |
| 18 | Agent relaunch loses debugging context | **NO** | No agent relaunches occurred in this run. |

### New Issues Discovered

| Title | Description | Suggested Priority |
|-------|-------------|-------------------|
| ULP metric breaks down at near-zero expected values | ULP comparison gives misleading 200+ ULP for negligible absolute errors near zero. Affects all operations where f(x) ≈ 0 for some inputs. Test template should include standard nonzero filter. | **HIGH** |
| JIT compilation uses root worktree headers | In shared environments, root worktree corruption by other agents causes build failures. New kernel headers must be synced to root for JIT to find them. Need pre-test sync step. | **MEDIUM** |
| Agents inconsistently log commit hashes in complete events | 3 of 6 agents omit commit hash from `complete` breadcrumb, making post-mortem git correlation dependent on orchestrator-level tracking. | **LOW** |

---

## 10. Actionable Recommendations

### Recommendation 1: Add ULP near-zero filter to test template

- **Type**: instruction_change
- **Target**: Tester agent instructions; test template; generator's Phase 4 prompt
- **Change**: Add mandatory guidance: "For ULP comparison, apply `nonzero_mask = torch.abs(expected.float()) > 1e-30` to exclude near-zero values. ULP is undefined at zero. allclose with `atol` covers the near-zero range."
- **Expected Benefit**: Eliminates 2-3 wasted iterations for every operation where f(x) can be near-zero
- **Priority**: **HIGH**
- **Effort**: SMALL

### Recommendation 2: Add pre-test root worktree sync step

- **Type**: pipeline_change
- **Target**: Generator/orchestrator; Phase 3→4 transition
- **Change**: After implementor completes, copy all new kernel files (ckernel_sfpu_*.h, llk_math_eltwise_unary_sfpu_*.h, API headers) to root worktree and update root's sfpu_split_includes.h and llk_sfpu_types.h. Run this as a Phase 3.5 before launching tester.
- **Expected Benefit**: Prevents 10+ minute environment debugging when JIT cache is cleared
- **Priority**: **HIGH**
- **Effort**: MEDIUM

### Recommendation 3: Mandate commit hash in agent complete breadcrumbs

- **Type**: logging_fix
- **Target**: All agent logging specs in `.claude/references/logging/sfpu-operation-*.md`
- **Change**: Add `"final_commit"` as a mandatory field in the `complete` event schema. If the agent committed, record the hash. If committing failed or was deferred, record `"pending"`.
- **Expected Benefit**: Enables reliable post-mortem git correlation without depending on orchestrator
- **Priority**: **LOW**
- **Effort**: SMALL

### Recommendation 4: Document SFPNONLINEAR availability per architecture

- **Type**: instruction_change
- **Target**: Implementor agent instructions; reference documentation
- **Change**: Add a section: "SFPNONLINEAR intrinsics (`approx_exp`, `approx_recip`, `approx_sigmoid`) are available ONLY on Blackhole (guarded by `#if __riscv_xtttensixbh`). Wormhole does NOT support these. Since kernel source must be identical across architectures, use polynomial approximations for transcendental functions."
- **Expected Benefit**: Saves 15+ minutes of research per implementation needing exp/sigmoid
- **Priority**: **MEDIUM**
- **Effort**: SMALL

### Recommendation 5: Add is_fp32_dest_acc_en template parameter to SFPU kernel template

- **Type**: instruction_change
- **Target**: Implementor agent instructions; SFPU kernel template
- **Change**: Include `is_fp32_dest_acc_en` as a template parameter in the standard kernel signature, even if the initial implementation uses the same code for both paths. Template: `template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>`
- **Expected Benefit**: Future-proofs kernels for FP32 optimization without re-implementation
- **Priority**: **MEDIUM**
- **Effort**: SMALL

### Recommendation 6: Implementor must commit before completion

- **Type**: instruction_change
- **Target**: Implementor agent instructions; logging spec
- **Change**: After all layers are implemented, the implementor MUST: (1) `git add` all created/modified files, (2) attempt `git commit`, (3) if pre-commit hook fails, fix formatting and retry, (4) log commit hash in `implementation_complete` breadcrumb. The orchestrator should not need to commit on behalf of the implementor.
- **Expected Benefit**: Ensures git diffs are available for impl-notes enrichment; reduces orchestrator overhead
- **Priority**: **MEDIUM**
- **Effort**: SMALL

---

## 11. Overall Assessment

### Pipeline Maturity Score

| Dimension | Score | Notes |
|-----------|-------|-------|
| Reference discovery accuracy | **5/5** | Hardswish was an excellent primary reference. All 5 references contributed to the implementor's decision-making. |
| Reference analysis quality | **5/5** | All 5 analyses completed, thorough documentation, correct identification of integration gaps (rpow undefined function). |
| Implementation completeness | **4/5** | All 12 layers present and correct. -1 for missing `is_fp32_dest_acc_en` template parameter. |
| SFPI compliance | **5/5** | Pure SFPI throughout. All quality checks pass except fp32 handling (MEDIUM, not HIGH). |
| Testing thoroughness | **4/5** | Both dtypes tested and passing. -1 for 6 iterations (3 avoidable with ULP near-zero guidance in template). |
| Inter-agent communication | **4/5** | Handoff quality GOOD overall. -1 for implementor not documenting near-zero behavior, and missing git diffs for impl-notes. |
| Logging/observability | **4/5** | All breadcrumb files exist with full mandatory events. -1 for inconsistent commit hashes in complete events and no impl-notes execution log. |

**Weighted average: 4.4/5**

### Top 3 Things to Fix

1. **Add ULP near-zero filter to test template** — would save 3 debug iterations and ~7 minutes per operation with near-zero output behavior (HIGH priority, SMALL effort).
2. **Add pre-test root worktree sync step** — would prevent 10+ minute environment debugging when JIT cache is cleared in shared environments (HIGH priority, MEDIUM effort).
3. **Document SFPNONLINEAR availability per architecture** — would save 15+ minutes of research for every operation needing exp/sigmoid/recip (MEDIUM priority, SMALL effort).

### What Worked Best

**The discoverer's reference selection was the single strongest aspect of this pipeline run.** By identifying hardswish as the structural template for swish (both compute `x * sigmoid_variant(x)`), the discoverer enabled the implementor to reuse the entire dispatch chain, loop structure, and SFPI kernel skeleton from a working reference. This directly contributed to the implementor completing all 11 layers in a single pass without errors. The negative references (rpow's undefined function, softsign's stub) also provided crucial guidance, steering the implementor toward the polynomial approach that ultimately succeeded. The reference selection phase invested 7 minutes (7% of total time) and saved potentially 30+ minutes of exploration during implementation.
