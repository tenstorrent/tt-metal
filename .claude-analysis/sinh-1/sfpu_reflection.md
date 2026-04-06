# SFPU Reflection: sinh

## Metadata
| Field | Value |
|-------|-------|
| Operation | `sinh` |
| Math Definition | `sinh(x) = (exp(x) - exp(-x)) / 2` |
| Output Folder | `.claude-analysis/sinh-1/` |
| Pipeline Phases Executed | 1, 2, 3, 4, 5 |
| Agents Invoked | generator, discoverer, 5x analyzer, implementor, tester, impl-notes |
| Total Git Commits | 10 (this run: 60a55a24..90083ba1) |
| Total Pipeline Duration | ~87 min (5223s from 09:20:51 to 10:47:54) |
| Overall Result | SUCCESS |

---

## 1. Pipeline Execution Summary

### Phase Timeline

| Phase | Agent(s) | Duration | Status | Key Observations |
|-------|----------|----------|--------|------------------|
| 1: Reference Discovery | discoverer | ~5m 25s | ok | Selected rpow, cbrt, hardsigmoid, hardswish, softshrink |
| 2: Reference Analysis | 5x analyzer | ~16m 04s (wall) | ok | 3/5 completed on time; cbrt + hardswish late-completed during Phase 3 |
| 3: Implementation | implementor | ~19m 58s | ok | All layers implemented, single commit |
| 4: Testing & Debugging | tester | ~38m 13s | ok | 2 iterations: catastrophic cancellation fix + clang-format |
| 5: Documentation | impl-notes + generator | ~6m 31s | ok | Enriched notes with full source; final report written |
| **Total** | | **~87m** | | |

### Agent Duration Breakdown

| Agent | Start Time | End Time | Wall Duration | Iterations | Notes |
|-------|------------|----------|---------------|------------|-------|
| generator (orchestrator) | 09:20:48 | 10:48:09+ | ~87m | - | Entire pipeline; still running at self-reflection launch |
| discoverer | 09:21:43 | 09:26:03 | ~4m 20s | - | Clean single pass |
| analyzer (rpow) | 09:26:43 | 09:36:13* | ~9m 30s | - | First to commit; *git commit ts |
| analyzer (hardsigmoid) | 09:27:36 | 09:40:19 | ~12m 43s | - | Breadcrumb start->complete |
| analyzer (softshrink) | 09:28:16 | 09:41:23 | ~13m 07s | - | Had separate execution log file |
| analyzer (hardswish) | 09:27:41 | 09:45:57 | ~18m 16s | - | Late; completed during Phase 3 |
| analyzer (cbrt) | 09:27:40 | 09:48:01 | ~20m 21s | - | Latest; completed during Phase 3 |
| implementor | 09:43:00 | 10:02:58 | ~19m 58s | - | Single commit 5f8262b6 |
| tester | 10:03:10 | 10:41:23 | ~38m 13s | 2 | Taylor fix + clang-format |
| impl-notes | 10:42:48 | 10:46:36 | ~3m 48s | - | Collected 8 new + 7 modified files |

**Duration calculation method**: Breadcrumb timestamps used as primary source. Git commit timestamps used for rpow analyzer (no breadcrumbs for rpow) and for cross-validation.

### Duration Visualization

Phase durations (rounded): d1=5m, d2=16m, d3=20m, d4=38m, d5=7m. Total=86m.
Cumulative offsets: s1=0, s2=5, s3=21, s4=41, s5=79.

```
Phase 1  |####|                                                                       (~5m)
Phase 2       |###############|                                                        (~16m)
Phase 3                       |###################|                                    (~20m)
Phase 4                                           |#####################################| (~38m)
Phase 5                                                                                 |######| (~7m)
         0    5    10   15   20   25   30   35   40   45   50   55   60   65   70   75   80   85  min

Longest phase: Phase 4 (38m) -- testing + debugging, dominated by numerical cancellation diagnosis and fix
```

### Time Distribution

| Category | Duration | % of Total | Notes |
|----------|----------|------------|-------|
| Discovery (Phase 1) | ~5m 25s | 6.2% | |
| Analysis (Phase 2) | ~16m 04s | 18.4% | 5 parallel analyzers; 2 late completions |
| Implementation (Phase 3) | ~19m 58s | 22.9% | 12 layers in single pass |
| Testing (Phase 4) | ~38m 13s | 43.9% | 2 iterations |
| -- Productive (first run) | ~20m (est) | 23.0% | Initial test creation + run |
| -- Debugging/retries | ~18m (est) | 20.7% | Taylor approximation fix, clang-format fix |
| Documentation (Phase 5) | ~6m 31s | 7.5% | impl-notes enrichment + final report |
| **Total** | **~87m** | **100%** | |

---

## 2. Implementation Coverage Audit

### Math Definition Fidelity

| Aspect | Status | Details |
|--------|--------|---------|
| Core formula | MATCH | Kernel computes `(2^(x*log2e) - 2^(-x*log2e)) / 2` via two `exp_21f` calls, which correctly implements `(exp(x) - exp(-x)) / 2` |
| Conditional branches | CORRECT | `v_if(z_pos < v_low_threshold)` clamps to -127.0f to prevent exp underflow; `v_if(abs_x < v_half)` switches to Taylor for small inputs |
| Parameter handling | N/A | sinh has no parameters |
| Edge cases | MATCH | Taylor branch `sinh(x) ~ x + x^3/6` for `|x| < 0.5` avoids catastrophic cancellation; `z` clamped at -127.0f handles large magnitude inputs; explicit `float_to_fp16b` rounding ensures deterministic bfloat16 output |

**Math definition from orchestrator**: `sinh(x) = (exp(x) - exp(-x)) / 2`
**Kernel implementation summary**: The kernel converts `exp(x)` to `2^(x * log2(e))` and uses the Moroz et al. 2022 `exp_21f` algorithm to compute both `exp(x)` and `exp(-x)`. For small `|x| < 0.5`, a Taylor polynomial `x + x^3/6` is used to avoid catastrophic cancellation in the subtraction `exp(x) - exp(-x)`. The result is scaled by 0.5 and rounded to bfloat16.

### 12-Layer Completeness

| Layer | Description | Expected File(s) | Status | Notes |
|-------|-------------|-------------------|--------|-------|
| 1 | SFPU Kernel | `ckernel_sfpu_sinh.h` (WH+BH) | PRESENT | Both files on disk, verified identical; well-commented with exp_21f helper |
| 2 | LLK Dispatch | `llk_math_eltwise_unary_sfpu_sinh.h` (WH+BH) | PRESENT | Both files on disk, verified identical |
| 3 | Compute API Header | `sinh.h` | PRESENT | Includes Doxygen docs, `sinh_tile()` and `sinh_tile_init()` |
| 4 | SFPU Include Guard | `sfpu_split_includes.h` | PRESENT | `SFPU_OP_SINH_INCLUDE` guard added at line 23 |
| 5 | SfpuType Enum | `llk_sfpu_types.h` (WH+BH) | PRESENT | `sinh` added to both WH and BH enum files |
| 6 | UnaryOpType Enum | `unary_op_types.hpp` | PRESENT | `SINH` at line 35 (pre-existing enum value preserved) |
| 7 | Op Utils Registration | `unary_op_utils.cpp` | PRESENT | Both `get_macro_definition` and `get_op_init_and_func_default` registered; also registered in `unary_ng_op_utils.cpp` |
| 8 | Op Utils Header | `unary_op_utils.hpp` | N/A | sinh is non-parameterized; no header changes needed |
| 9 | C++ API Registration | `unary.hpp` | PRESENT | `REGISTER_UNARY_OPERATION(sinh, SINH)` at line 116 |
| 10 | Python Nanobind | `unary_nanobind.cpp` | PRESENT | `bind_unary_operation<"sinh", &ttnn::sinh>` at line 1791 |
| 11 | Python Golden | `unary.py` | PRESENT | `"sinh": torch.sinh` in golden dict (line 44), `ttnn.sinh` in function list (line 65) |
| 12 | Test File | `test_sinh.py` | PRESENT | Exhaustive bfloat16 bitpattern test with bfloat16 + fp32 parametrization |

**Layer completeness**: 11/12 layers present (Layer 8 correctly N/A for non-parameterized op). Effective: 12/12.

### Reference Utilization

| Reference | Analysis Produced? | Cited by Implementor? | Usefulness |
|-----------|-------------------|----------------------|------------|
| rpow | YES (rpow_analysis.md, 336 lines) | YES -- exp_21f algorithm extracted | HIGH |
| cbrt | YES (cbrt_analysis.md, 200 lines) | NO -- setsgn pattern used but cbrt not explicitly cited | LOW |
| hardsigmoid | YES (hardsigmoid_analysis.md, 151 lines) | YES -- loop/init template pattern | MEDIUM |
| hardswish | YES (hardswish_analysis.md, 157 lines) | NO -- composite pattern not needed once exp_21f was factored | LOW |
| softshrink | YES (softshrink_analysis.md, 180 lines) | YES -- confirmed include guard mechanism | MEDIUM |

**References wasted**: 2 (cbrt, hardswish). The cbrt analysis was primarily useful for its `setsgn` pattern which the implementor did use (line 85 of kernel), but the analysis itself was not explicitly cited. Hardswish was selected for its composite-subexpression pattern, but the implementor's cleaner approach of factoring `exp_21f` into a helper function made this pattern unnecessary. The cbrt and hardswish analyzers were also the two that completed late (during Phase 3), reducing their potential value to the implementor.

### Test Coverage

| Metric | Value |
|--------|-------|
| Test file created | YES |
| bfloat16 parametrization | PASS |
| fp32 parametrization | PASS |
| Max ULP (bfloat16) | <=2 (ULP threshold 2) |
| Max ULP (fp32) | N/A (allclose-only for fp32) |
| allclose (bfloat16) | PASS (rtol=1.6e-2, atol=1e-2) |
| allclose (fp32) | PASS (rtol=1.6e-2, atol=1e-2) |
| Total test iterations | 2 (initial fail + Taylor fix pass) |
| Final result | PASS |

---

## 3. Breadcrumb & Logging Compliance Audit

### Per-Agent Breadcrumb Compliance

| Agent | File Exists? | Event Count | Min Expected | Mandatory Events Present? | Timestamps? | Ordering? | Compliance |
|-------|-------------|-------------|--------------|--------------------------|-------------|-----------|------------|
| generator | YES | 34 | ~27 | PARTIAL -- missing `pipeline_complete` | YES | YES | PARTIAL |
| discoverer | YES | 5 | 4 | YES (start, files_read, ranking_complete, complete) | YES | YES | FULL |
| analyzer(s) | YES | 28 | 30 (6x5) | PARTIAL -- rpow missing all 6 events | YES | YES | PARTIAL |
| implementor | NO | 0 | 15 | NO -- file absent | N/A | N/A | ABSENT |
| tester | NO | 0 | 4+ | NO -- file absent | N/A | N/A | ABSENT |
| impl-notes | YES | 5 | 3 | YES (notes_read, files_collected, complete) | YES | YES | FULL |

**Detailed findings**:

**Generator (PARTIAL)**: 34 events logged, well above the minimum of 27. All phase_start/phase_complete pairs present for phases 1-5, plus a phase_start for Phase 6 (self-reflection). All subagent_launched/completed events present. However, the mandatory `pipeline_complete` event is missing -- the pipeline was still running when the self-reflection agent was launched, which is expected. The generator also logged late cbrt and hardswish analyzer completions (lines 21-22) that arrived during Phase 3 -- good observability.

**Discoverer (FULL)**: 5 events (1 extra `start` event with agent metadata). All 4 mandatory events present with correct structure. Timestamps are monotonically increasing.

**Analyzer (PARTIAL)**: 28 events across 4 operations (hardsigmoid: 8, softshrink: 4, hardswish: 3, cbrt: 5). The rpow analyzer produced NO breadcrumbs in this file -- it was the first to complete but did not write to the shared breadcrumb file. This means rpow has 0/6 mandatory events. For the other 4 operations:
- hardsigmoid: all 6 mandatory events (start, dispatch_traced, kernel_source_read, instruction_analysis_complete, analysis_written, complete)
- softshrink: 4 of 6 -- missing `analysis_written` and has `dispatch_traced` but no explicit `instruction_analysis_complete` before `complete` (wait -- re-checking: softshrink has start, dispatch_traced, kernel_source_read, instruction_analysis_complete, analysis_written, complete -- 6 events but 2 are agent-metadata starts)
- hardswish: 3 of 6 -- missing `start` (has only agent-metadata start), has kernel_source_read and instruction_analysis_complete, has analysis_written and complete but no dispatch_traced in later entries
- cbrt: 5 of 6 -- has start, dispatch_traced, kernel_source_read, instruction_analysis_complete, analysis_written, complete

Expected 30 events (6 per operation x 5 operations), got 28. Rpow contributing 0 events is a significant gap.

**Implementor (ABSENT)**: No breadcrumb file exists at all. The implementor logging spec (`.claude/references/logging/sfpu-operation-implementor.md`) exists and is well-defined with clear requirements for 15 minimum breadcrumbs (references_parsed + 12x layer_implemented + implementation_complete + complete). The absence of any breadcrumb file means zero observability into the implementation phase -- we cannot determine which layers were implemented in what order, what references were actually consulted, or what issues were encountered. This is a **HIGH severity** gap.

**Tester (ABSENT)**: No breadcrumb file exists at all. The tester logging spec (`.claude/references/logging/sfpu-operation-tester.md`) exists and is well-defined with clear requirements for at minimum 4 breadcrumbs (notes_parsed, test_created, test_run, complete) plus hypothesis/fix_applied for failures. The absence means we cannot reconstruct the debugging narrative from breadcrumbs alone -- we rely entirely on the issues_log.md and git commits. This is a **HIGH severity** gap.

**Impl-Notes (FULL)**: 5 events including 2 agent-metadata `start` events, plus `notes_read`, `files_collected`, and `complete`. All mandatory events present. Note: timestamp format uses `"timestamp"` key for some events instead of `"ts"` (lines 3-5 use `"timestamp"` while lines 1-2 use `"ts"`), which is a minor format inconsistency.

### Execution Log Compliance

| Agent | Log Exists? | Sections Present | Notes |
|-------|------------|------------------|-------|
| generator | NO | N/A | No execution log; however, the generator's role is orchestration and its breadcrumbs + issues_log serve this purpose |
| discoverer | NO | N/A | No execution log produced |
| analyzer | YES (2 files) | Metadata, Input Interp, Exec Timeline, Recovery, Artifacts, Handoff Notes, SFPU Sections | 3 analyzers in main file (hardsigmoid, hardswish, cbrt); softshrink in separate file. Rpow has no execution log. |
| implementor | NO | N/A | No execution log -- **HIGH severity** combined with absent breadcrumbs |
| tester | NO | N/A | No execution log -- **HIGH severity** combined with absent breadcrumbs |
| impl-notes | NO | N/A | No execution log; breadcrumbs provide minimal coverage |

### Logging Infrastructure Issues

| Issue | Severity | Description |
|-------|----------|-------------|
| Implementor breadcrumbs absent | HIGH | `ttnn-unary-sfpu-operation-implementor_breadcrumbs.jsonl` does not exist. The spec file exists but the agent did not produce breadcrumbs. Zero observability into 12-layer implementation sequence. |
| Tester breadcrumbs absent | HIGH | `ttnn-unary-sfpu-operation-tester_breadcrumbs.jsonl` does not exist. The spec file exists but the agent did not produce breadcrumbs. Debugging narrative cannot be reconstructed from breadcrumbs. |
| Implementor execution log absent | HIGH | `ttnn-unary-sfpu-operation-implementor_execution_log.md` does not exist. Combined with absent breadcrumbs, implementation phase is a black box. |
| Tester execution log absent | HIGH | `ttnn-unary-sfpu-operation-tester_execution_log.md` does not exist. Combined with absent breadcrumbs, the 38-minute debugging phase is only documented via the issues_log (3 lines). |
| Rpow analyzer breadcrumbs absent | MEDIUM | The rpow analyzer completed first (git commit 60a55a24) and produced a detailed analysis file (336 lines), but wrote zero entries to the shared analyzer breadcrumb file. |
| Timestamp key inconsistency in impl-notes | LOW | Some events use `"timestamp"` key, others use `"ts"`. Parser must handle both. |

### Breadcrumb-to-Git Correlation

| Agent | Breadcrumb Commit | Git Commit | Match? |
|-------|-------------------|------------|--------|
| discoverer | (none logged) | (no separate commit) | N/A |
| analyzer (rpow) | (none in breadcrumbs) | 60a55a2487 | MISSING breadcrumb |
| analyzer (hardsigmoid) | (none in complete event) | b1d0e49171 | MISSING -- `complete` event has no commit field |
| analyzer (softshrink) | 5f380066b2 | 5f380066b2 | YES |
| analyzer (hardswish) | f4a907d02d | f4a907d02d | YES |
| analyzer (cbrt) | "pending" | 05c1747504 | PARTIAL -- breadcrumb says "pending", git has actual hash |
| implementor | (no breadcrumbs) | 5f8262b65f | MISSING |
| tester | (no breadcrumbs) | e799a2002d | MISSING |
| impl-notes | (no commit in breadcrumbs) | 90083ba173 | MISSING |
| generator -> implementor | 5f8262b65f | 5f8262b65f | YES |
| generator -> tester | (no commit field) | e799a2002d | MISSING |
| generator -> impl-notes | 90083ba173 | 90083ba173 | YES |

---

## 4. SFPI Code Enforcement Audit

### Kernel Style Classification

| Check | Result | Evidence |
|-------|--------|---------|
| SFPI indicators present? | YES | `sfpi::vFloat`, `sfpi::vInt`, `sfpi::dst_reg[0]`, `v_if`/`v_endif`, `sfpi::addexp`, `sfpi::exexp`, `sfpi::exman9`, `sfpi::setsgn`, `sfpi::reinterpret`, `sfpi::setexp`, `sfpi::float_to_fp16b`, `sfpi::int32_to_float` |
| Raw TTI indicators present? | NO | No `TT_SFP*`, `TTI_SFP*`, `SFPLOADI`, `SFPLOAD`, `SFPSTORE`, `SFPSETCC`, `SFPMAD`, or `SFPMUL` macro patterns found |
| **Kernel style** | **SFPI** | Fully SFPI-based implementation |

### Exception Check

Not applicable -- no raw TTI indicators detected. The kernel is fully SFPI-compliant.

**Verdict**: COMPLIANT -- uses SFPI abstractions exclusively.

### SFPI Quality Checks

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
| `#pragma GCC unroll` | Present on inner loop | `#pragma GCC unroll 0` at line 59 | OK -- unroll 0 is intentional due to heavy loop body (two exp_21f calls); matches rpow reference pattern |
| DEST register pattern | `dst_reg[0]` read -> compute -> write -> `dst_reg++` | Line 61: `x = sfpi::dst_reg[0]`; line 95: `sfpi::dst_reg[0] = y`; line 96: `sfpi::dst_reg++` | OK |
| ITERATIONS template | `int ITERATIONS = 8` in template params | `template <bool APPROXIMATION_MODE, int ITERATIONS = 8>` at line 51 | OK |
| fp32 handling | `is_fp32_dest_acc_en` template param | NOT PRESENT | MEDIUM -- The standalone `ckernel_sfpu_sinh.h` does not have `is_fp32_dest_acc_en`. However, the trig-family version in `ckernel_sfpu_trigonometry.h` (which is what the JIT actually compiles) does have `template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>`. The standalone kernel forces bfloat16 rounding via `float_to_fp16b` even in fp32 mode. |
| Parameter reconstruction | `Converter::as_float(param0)` | N/A -- sinh is non-parameterized | N/A |
| WH/BH identical | Both architecture files same content | Verified via `diff` -- IDENTICAL | OK |

**Note on fp32 handling**: The standalone `ckernel_sfpu_sinh.h` lacks `is_fp32_dest_acc_en` and unconditionally rounds to bfloat16 via `float_to_fp16b`. The trig-family kernel in `ckernel_sfpu_trigonometry.h` (lines 857-876 in the impl notes) does NOT have this unconditional rounding and instead just writes `result` directly to `dst_reg[0]`. This means the actual runtime path (via trigonometry.h) may differ from the standalone kernel. The test still passes with relaxed fp32 tolerances (rtol=1.6e-2) which accounts for bfloat16-level precision from the SFPU, so this is not a functional issue but represents a code inconsistency between the two kernel copies.

### Reference Style Comparison

| Reference | Reference Style | New Kernel Style | Assessment |
|-----------|----------------|-----------------|------------|
| rpow | A_sfpi | SFPI | Correctly reused exp_21f algorithm; factored into helper function |
| cbrt | A_sfpi | SFPI | setsgn(x, 0) pattern borrowed for absolute value |
| hardsigmoid | A_sfpi | SFPI | Loop structure template followed |
| hardswish | A_sfpi | SFPI | Composite pattern not needed; clean SFPI maintained |
| softshrink | A_sfpi | SFPI | Include guard pattern followed |

All references used SFPI style (A_sfpi). The new kernel correctly maintains SFPI style throughout. No style regression.

---

## 5. What Went Well

### 1. Reference Discovery Quality

**Phase/Agent**: Phase 1 -- discoverer
**Evidence**: The discoverer selected rpow as the #1 reference, which proved to be the critical building block. The rpow analysis provided the `exp_21f` algorithm that became the core of the sinh kernel. The discoverer's rationale ("rpow contains the only complete working implementation of the 2^z exponential function in this codebase") was exactly correct.
**Why it worked**: The discoverer read actual kernel source files and understood the mathematical relationships between operations.

### 2. Implementation Completeness on First Pass

**Phase/Agent**: Phase 3 -- implementor
**Evidence**: All 12 layers (11 applicable + 1 N/A) were implemented in a single commit (5f8262b65f) with no revisits. The implementation notes show clean registration across both `unary_op_utils.cpp` and `unary_ng_op_utils.cpp`, plus Python bindings.
**Why it worked**: The reference analyses provided clear patterns for each layer, and the implementor correctly adapted the rpow exp_21f algorithm into a reusable helper.

### 3. Taylor Approximation Fix for Catastrophic Cancellation

**Phase/Agent**: Phase 4 -- tester
**Evidence**: The initial implementation failed for small `|x|` due to `exp(x) - exp(-x)` producing inaccurate results when both terms are close to 1.0. The tester diagnosed this as catastrophic cancellation and implemented the Taylor series `sinh(x) ~ x + x^3/6` for `|x| < 0.5`. This is a mathematically sound fix.
**Why it worked**: The tester recognized the root cause (floating-point cancellation) rather than trying to tune tolerances or ignore the error.

### 4. All 5 Reference Analyses Completed Successfully

**Phase/Agent**: Phase 2 -- analyzers
**Evidence**: All 5 analysis files were produced (rpow: 336 lines, cbrt: 200 lines, hardsigmoid: 151 lines, hardswish: 157 lines, softshrink: 180 lines). Even though cbrt and hardswish were initially marked as "failed" at the Phase 2 deadline (09:42:41), they completed during Phase 3 and their commits were correctly tracked by the orchestrator (breadcrumb lines 21-22).
**Why it worked**: The orchestrator did not discard the late analyzers; it tracked their eventual completion.

### 5. Dual Op Utils Registration

**Phase/Agent**: Phase 3 -- implementor
**Evidence**: The implementor registered sinh in both `unary_op_utils.cpp` (legacy path) and `unary_ng_op_utils.cpp` (next-gen path). This ensures the operation works through both dispatch mechanisms.
**Why it worked**: The implementor read the codebase structure and discovered both registration points.

---

## 6. Issues Found

### Issue 1: Implementor and Tester Produced Zero Breadcrumbs

| Field | Value |
|-------|-------|
| Severity | HIGH |
| Phase | Phase 3 and Phase 4 |
| Agent | implementor, tester |
| Verification Dimension | Logging Compliance |
| Retries Consumed | 0 |
| Time Cost | N/A (lost observability, not lost time) |

**Problem**: The implementor and tester agents produced no breadcrumb files (`ttnn-unary-sfpu-operation-implementor_breadcrumbs.jsonl` and `ttnn-unary-sfpu-operation-tester_breadcrumbs.jsonl` are both absent from `agent_logs/`). The logging specs for both agents exist at `.claude/references/logging/sfpu-operation-implementor.md` and `.claude/references/logging/sfpu-operation-tester.md` and clearly define mandatory events (15 minimum for implementor, 4+ for tester). Both agents also failed to produce execution logs.

This means the two longest phases (Phase 3: 20m, Phase 4: 38m, totaling 58m or 67% of pipeline time) have zero structured observability. We cannot determine:
- For the implementor: layer implementation order, which references were consulted, design decisions made
- For the tester: test creation details, numerical error magnitudes, hypothesis/fix cycle details, attempt-by-attempt results

**Root Cause**: The agents likely did not read or follow their logging spec files. The spec files are referenced in the agent prompts but compliance is not enforced by the orchestrator. The orchestrator does not verify breadcrumb file existence after subagent completion.

**Fix for agents**:
- **Orchestrator (generator)**: After each `subagent_completed` event, verify that the expected breadcrumb file exists and contains at minimum the `complete` event. If not, log a `logging_gap` warning event.
- **Implementor**: Add a pre-completion checklist that reads the breadcrumb spec and verifies all mandatory events were logged before reporting completion.
- **Tester**: Same as implementor.

### Issue 2: Rpow Analyzer Missing from Shared Breadcrumb File

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase | Phase 2 |
| Agent | analyzer (rpow) |
| Verification Dimension | Logging Compliance |
| Retries Consumed | 0 |
| Time Cost | N/A |

**Problem**: The rpow analyzer was the first to complete (git commit 60a55a2487 at 09:36:13) and produced a thorough 336-line analysis file. However, it contributed zero events to the shared `ttnn-unary-sfpu-operation-analyzer_breadcrumbs.jsonl` file. All other analyzers wrote their events to this file.

**Root Cause**: The rpow analyzer appears to have run in the same commit that also created the initial `reference_selection.md`, discoverer breadcrumbs, generator breadcrumbs, and `issues_log.md` (commit 60a55a24). This suggests the rpow analysis may have been done inline with the initial pipeline setup or the breadcrumb file was created after rpow wrote its events. Without rpow's breadcrumbs, we lose timing data and the `kernel_source_read` event that would tell us the kernel style.

**Fix for agents**:
- **Analyzer**: Ensure breadcrumb file append is atomic and does not depend on file pre-existence. Each analyzer should create the file if absent.
- **Orchestrator**: Initialize the shared breadcrumb file before launching parallel analyzers.

### Issue 3: Catastrophic Cancellation in Initial Implementation

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase | Phase 4 -- Testing |
| Agent | tester (fix), implementor (root cause) |
| Verification Dimension | Implementation Coverage |
| Retries Consumed | 1 hard retry |
| Time Cost | ~18m additional testing time |

**Problem**: The initial sinh kernel computed `(exp(x) - exp(-x)) / 2` directly for all x values. For small `|x| < 0.5`, where `exp(x) ~ 1 + x` and `exp(-x) ~ 1 - x`, the subtraction `exp(x) - exp(-x)` suffers catastrophic cancellation (subtracting two nearly equal numbers), producing results with far fewer significant bits than the individual terms.

**Root Cause**: The implementor did not anticipate the well-known numerical issue of catastrophic cancellation in hyperbolic functions near zero. This is a standard textbook concern for sinh/cosh implementations.

**Fix for agents**:
- **Implementor**: When implementing any operation involving subtraction of exponentials (sinh, expm1, log1p, etc.), proactively add a small-argument approximation branch. The math definition handoff should include a "numerical pitfalls" field.
- **Generator (orchestrator)**: Include a `numerical_pitfalls` field in the `pipeline_start` breadcrumb for operations known to have cancellation issues.

### Issue 4: Dual Kernel Copies with Inconsistency

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase | Phase 3 -- Implementation, Phase 4 -- Testing |
| Agent | implementor, tester |
| Verification Dimension | Implementation Coverage |
| Retries Consumed | 0 |
| Time Cost | Minimal |

**Problem**: Two versions of the sinh kernel exist in the codebase:
1. **Standalone**: `ckernel_sfpu_sinh.h` -- uses `exp_21f` helper defined locally, no `is_fp32_dest_acc_en`, unconditional `float_to_fp16b` rounding
2. **Trig family**: `ckernel_sfpu_trigonometry.h` -- uses `_sfpu_exp_21f_bf16_` from the exp library, has `is_fp32_dest_acc_en`, no forced bfloat16 rounding

The `sinh_final.md` states: "The JIT compiler resolves kernel headers from the runtime install root, not the worktree. The sinh operation dispatches through SFPU_OP_TRIG_FAMILY_INCLUDE -> trigonometry.h -> ckernel_sfpu_trigonometry.h." This means the trig-family version is what actually runs, while the standalone `ckernel_sfpu_sinh.h` is what the implementor originally wrote.

The tester discovered this during testing and had to modify `ckernel_sfpu_trigonometry.h` to add the Taylor approximation fix there as well. The two kernel implementations are not identical and serve different purposes.

**Root Cause**: The pipeline creates a standalone kernel file per the 12-layer pattern, but the actual runtime dispatch may go through a different code path (the trig family header). This is specific to operations that have pre-existing dispatch infrastructure.

**Fix for agents**:
- **Implementor**: Before implementing Layer 1, verify the actual JIT dispatch path. If the operation already has a slot in an existing family header (trigonometry, activations, etc.), implement there instead of creating a standalone file.
- **Tester**: Document which kernel file is actually compiled by the JIT in the test output.

### Issue 5: Clang-Format Pre-Commit Hook Failure

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase | Phase 4 -- Testing |
| Agent | tester |
| Verification Dimension | Implementation Coverage |
| Retries Consumed | 1 free retry |
| Time Cost | ~2m |

**Problem**: The clang-format pre-commit hook failed on the tester's commit, requiring re-staging of formatted files.

**Root Cause**: Code was written without running clang-format first.

**Fix for agents**:
- **Tester**: Run `clang-format` on modified C++ files before attempting to commit.

---

## 7. Efficiency Analysis

### Per-Phase Breakdown

| Phase | Duration | Status | Bottleneck |
|-------|----------|--------|------------|
| 1: Discovery | ~5m 25s | ok | Clean -- no issues |
| 2: Analysis | ~16m 04s | ok | cbrt (20m 21s) and hardswish (18m 16s) exceeded Phase 2 deadline; completed during Phase 3 |
| 3: Implementation | ~19m 58s | ok | Single pass, 12 layers; no bottleneck visible without breadcrumbs |
| 4: Testing | ~38m 13s | ok | 2 iterations; catastrophic cancellation diagnosis + Taylor fix was the main cost |
| 5: Documentation | ~6m 31s | ok | Clean |

### Tester Iteration Breakdown

| Attempt | Result | Error Type | Fix Applied | Duration |
|---------|--------|-----------|-------------|----------|
| 1 | FAIL | numerical_error | Catastrophic cancellation for small |x|; exp(x)-exp(-x) near 0 loses precision | ~20m (est.) |
| 2 | PASS (after clang-format retry) | N/A | Taylor approximation sinh(x) ~ x + x^3/6 for |x|<0.5 added to kernel; clang-format fix | ~18m (est.) |

### Time Sinks

| Rank | Area | Agent | Duration | % of Total | Description |
|------|------|-------|----------|------------|-------------|
| 1 | Testing + debugging | tester | ~38m | 43.9% | Two iterations; catastrophic cancellation fix required modifying trigonometry.h (the actual JIT-compiled kernel) |
| 2 | Implementation | implementor | ~20m | 22.9% | All 12 layers in single pass; no breakdowns available |
| 3 | Analysis (late) | analyzer (cbrt, hardswish) | ~18-20m each | Overlapped with Phase 3 | Both exceeded the Phase 2 deadline; completed during implementation |

---

## 8. Inter-Agent Communication

| Handoff | Source -> Target | Artifact | Quality | Issues | Suggestion |
|---------|-----------------|----------|---------|--------|------------|
| 1 | Generator -> Discoverer | Math definition | GOOD | Clean: `sinh(x) = (exp(x) - exp(-x)) / 2` | None |
| 2 | Discoverer -> Analyzers | Reference list | GOOD | 5 references selected with clear rationale; rpow proved critical | None |
| 3 | Analyzers -> Implementor | Analysis files | ADEQUATE | 3/5 available at Phase 3 start; rpow (most important) available; cbrt and hardswish arrived late | Orchestrator should hard-gate Phase 3 on all analyzer completions, or explicitly pass the list of available analyses |
| 4 | Implementor -> Tester | Impl notes | ADEQUATE | Notes listed files created/modified but lacked numerical pitfall warnings | Implementor should include a "known numerical risks" section in notes |
| 5 | Tester -> Impl-Notes | File manifest | GOOD | Impl-notes agent collected 8 new + 7 modified files successfully; enriched with full source | None |

---

## 9. Comparison with Known Issues

### Known Issues Encountered

| Issue # | Title | Encountered? | Notes |
|---------|-------|-------------|-------|
| 1 | Numerical debugging burns context | YES | The tester spent ~38m on Phase 4, with catastrophic cancellation being a numerical debugging challenge. However, the fix was found in a single iteration -- better than the multi-iteration grinds described in issue #1. |
| 13 | Phase 1/2 overlap | YES (variant) | Phase 2 analyzers (cbrt, hardswish) overlapped into Phase 3. The orchestrator launched the implementor at 09:43:00 while cbrt/hardswish were still running. The implementor still succeeded, likely because the 3 completed analyses were sufficient. |
| 15 | Kernel writer missing execution logs | YES (analogous) | Both implementor and tester agents produced no execution logs, mirroring the kernel-writer issue from the softmax run. This SFPU pipeline variant has the same gap. |

### New Issues Discovered

| Title | Description | Suggested Priority |
|-------|-------------|-------------------|
| Implementor and tester produce zero breadcrumbs | Both agents failed to generate any breadcrumb files despite well-defined specs existing. This affects 67% of pipeline time. | HIGH |
| Dual kernel copy inconsistency | Standalone `ckernel_sfpu_sinh.h` and trig-family `ckernel_sfpu_trigonometry.h` have different implementations of the same function (fp32 handling differs). JIT uses the trig-family version. | MEDIUM |
| Rpow analyzer breadcrumbs missing from shared file | First-completing analyzer wrote no breadcrumbs, suggesting a race condition or initialization issue with the shared breadcrumb file. | MEDIUM |

---

## 10. Actionable Recommendations

### Recommendation 1: Enforce Breadcrumb Generation for Implementor and Tester

- **Type**: logging_fix
- **Target**: Orchestrator (generator) agent instructions; implementor and tester agent instructions
- **Change**: (a) Orchestrator should verify breadcrumb file existence and minimum event count after each `subagent_completed` event. If missing, log a `logging_gap` warning. (b) Add a pre-completion gate in implementor/tester: before reporting `complete`, verify own breadcrumb file exists and contains the minimum mandatory events.
- **Expected Benefit**: Restore observability for 67% of pipeline time; enable per-layer timing analysis and debugging narrative reconstruction.
- **Priority**: HIGH
- **Effort**: SMALL

### Recommendation 2: Add Numerical Pitfalls to Math Definition Handoff

- **Type**: instruction_change
- **Target**: Generator (orchestrator) `pipeline_start` breadcrumb; implementor agent instructions
- **Change**: Include a `numerical_pitfalls` field in the pipeline start event for operations known to have cancellation or overflow issues (sinh, cosh, expm1, log1p, tanh for large args). The implementor should proactively implement small-argument approximations when this field is present.
- **Expected Benefit**: Avoid the most common Phase 4 failure mode (catastrophic cancellation) by front-loading the fix to Phase 3.
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 3: Verify JIT Dispatch Path Before Layer 1 Implementation

- **Type**: instruction_change
- **Target**: Implementor agent instructions
- **Change**: Before creating `ckernel_sfpu_{op_name}.h`, the implementor should verify the actual JIT dispatch path by checking if the operation already has a slot in an existing family header (trigonometry.h, activations.h). If so, implement in the family header directly rather than creating a standalone file.
- **Expected Benefit**: Eliminate dual-kernel inconsistency; reduce the risk of fixes being applied to the wrong file.
- **Priority**: MEDIUM
- **Effort**: MEDIUM

### Recommendation 4: Initialize Shared Breadcrumb File Before Parallel Analyzers

- **Type**: pipeline_change
- **Target**: Orchestrator (generator) agent; breadcrumb append script
- **Change**: Before launching the 5 parallel analyzers, create an empty `ttnn-unary-sfpu-operation-analyzer_breadcrumbs.jsonl` file. This prevents race conditions where the first analyzer to complete cannot find the file.
- **Expected Benefit**: Prevent rpow-style breadcrumb loss.
- **Priority**: LOW
- **Effort**: SMALL

### Recommendation 5: Mandate Execution Logs for All Agents

- **Type**: logging_fix
- **Target**: All agent instructions (particularly implementor and tester)
- **Change**: Add mandatory execution log generation to implementor and tester agent instructions, matching the pattern already established for the analyzer agent. The log should follow the template in `.claude/references/agent-log-template.md` with agent-specific sections.
- **Expected Benefit**: Structured narrative for self-reflection analysis; debugging insights preserved for future runs.
- **Priority**: MEDIUM
- **Effort**: SMALL

---

## 11. Overall Assessment

### Pipeline Maturity Score

| Dimension | Score | Notes |
|-----------|-------|-------|
| Reference discovery accuracy | 5 | Rpow selection was critical and correct; all references relevant |
| Reference analysis quality | 4 | All 5 analyses produced; 2 late completions but thorough content |
| Implementation completeness | 5 | All 12 layers implemented correctly on first pass; dual registration (legacy + ng) |
| SFPI compliance | 5 | Fully SFPI-based kernel; no raw TTI; proper register pattern; WH/BH identical |
| Testing thoroughness | 4 | Both dtypes tested; exhaustive bfloat16 bitpatterns; Taylor fix found; -1 for needing 2 iterations |
| Inter-agent communication | 4 | Handoffs generally good; missing numerical pitfall warnings in impl notes |
| Logging/observability | 2 | Implementor and tester (67% of pipeline time) produced zero breadcrumbs or execution logs; rpow analyzer breadcrumbs also missing |

### Top 3 Things to Fix

1. **Implementor and tester must produce breadcrumbs and execution logs.** These two agents cover 67% of pipeline time and currently have zero structured observability. This is the single most impactful improvement.
2. **Front-load numerical pitfall warnings** (catastrophic cancellation, overflow) into the math definition handoff to prevent the most common Phase 4 debugging cycles.
3. **Resolve the dual-kernel inconsistency** by having the implementor verify the JIT dispatch path before creating standalone kernel files.

### What Worked Best

The reference discovery and utilization pipeline performed excellently. The discoverer correctly identified rpow as the critical reference (containing the exp_21f algorithm), the rpow analyzer produced a thorough 336-line analysis, and the implementor successfully extracted and refactored the exp_21f algorithm into a reusable helper function. This chain -- from mathematical insight ("sinh needs exp") to codebase search ("rpow has exp_21f") to detailed analysis to clean implementation -- represents the pipeline working exactly as designed.
