---
name: LLK Device Validation Reference
description: >
  Reference document for Phase B of Stage 5 (Review-Fix + Device Validation Loop).
  Describes how to generate and run device-side LLK sequence validation tests.
  This is NOT a standalone agent — it is reference material used by the review-fix
  agent (llk_review_fix_agent.md) during its Phase B execution.
type: reference
---

## Usage

This document is referenced by the Stage 5 review-fix agent during Phase B.
The review-fix agent reads this for the test generation pattern. Placeholders
are substituted by the orchestrator when invoking Stage 5:
- `{{LLK_CATEGORY}}` — the operation category (e.g. elementwise unary)
- `{{PROPOSAL_FILE}}` — path to the Stage 4 proposal (e.g. `elementwise_unary_helper_proposal.md`)
- `{{EXISTING_TEST_REFERENCE}}` — path to an existing helper test to use as structural template (e.g. `tests/tt_metal/tt_metal/llk/test_sfpu_unary_helpers.cpp`)
- `{{EXISTING_KERNEL_REFERENCE}}` — path to the compute kernel used by that test (e.g. `tests/tt_metal/tt_metal/test_kernels/compute/sfpu_unary_helper_test.cpp`)

## Prompt Template

```
You are generating and running device-side LLK sequence validation tests for proposed {{LLK_CATEGORY}} helpers.

BREADCRUMB LOGGING — do this first:
Derive CATEGORY_SLUG from {{LLK_CATEGORY}} (lowercase, spaces → underscores).
BCRUMB="agent_logs/${CATEGORY_SLUG}_device_validation_breadcrumbs.jsonl"
Run at start:
  mkdir -p agent_logs
  echo '{"ts":"'"$(date -Iseconds)"'","event":"start","agent":"device_validation","category":"{{LLK_CATEGORY}}"}' >> $BCRUMB

## Purpose

Stage 4 proposes LLK call sequences that the helper will execute internally. Stage 3 verified
these sequences exist in the codebase via code reading. This stage goes further: it runs the
EXACT proposed sequences on actual hardware and checks numerical correctness against golden
references.

This catches:
- Sequences that look correct in code but produce wrong results due to hardware state interactions
- Init ordering issues that cause silent data corruption
- DEST management bugs (wrong acquire/release placement)
- Mutual exclusion violations that don't crash but produce garbage

## Inputs

Read the proposal: {{PROPOSAL_FILE}}

Extract from it:
1. The **LLK Sequence Validation** table — each row has a helper name, the internal LLK
   call sequence (init and exec calls in order), and the claimed codebase precedent.
2. The **Op Struct Designs** section — for each op, the wrapper signatures (`*_tile_init()`,
   `*_tile()`) and their parameters.
3. The **Before/After** examples — the "Before" code shows the raw LLK pattern that the
   helper will replace. This IS the sequence to test.

Read the reference test and kernel to understand the pattern:
- {{EXISTING_TEST_REFERENCE}}
- {{EXISTING_KERNEL_REFERENCE}}

## What to Generate

For each UNIQUE LLK sequence in the proposal (not per-op — per-sequence, since many ops
share the same sequence with different tile() calls):

### 1. Compute Kernel (device-side)

File: `tests/tt_metal/tt_metal/test_kernels/compute/{CATEGORY_SLUG}_llk_validation.cpp`

The kernel exercises the proposed LLK sequences using raw API calls (NOT the helper, which
doesn't exist yet). Use preprocessor defines to select which op/sequence to test.

**How to derive the kernel**: Read the proposal's "Before" code blocks for each helper.
These show the raw LLK call pattern that the helper will replace. Copy each "Before" block
into a `#if defined(VALIDATE_OP_X)` section. The proposal's LLK Sequence Validation table
specifies the exact init→exec order — the kernel must match it.

The includes, initialization, and overall structure depend on the category:
- **Unary SFPU**: `api/compute/eltwise_unary/eltwise_unary.h`, `init_sfpu(cb_in, cb_out)`, copy+sfpu pattern
- **Binary eltwise**: `api/compute/eltwise_binary/eltwise_binary.h`, two input CBs, `*_tiles_init()`/`*_tiles()` pattern
- **Reduce**: `api/compute/reduce/reduce.h`, scaler CB, accumulation loops, `reduce_tile()` pattern
- **Tilize/Untilize**: `api/compute/tilize/tilize.h`, block-width loops, fast path dispatch
- **Matmul**: `api/compute/matmul/matmul.h`, inner-dim loops, accumulation
- **Other**: Derive from the proposal's Before blocks and the compute API dir in `llk_helpers_hq.md`

Read {{EXISTING_KERNEL_REFERENCE}} for the structural pattern (how to read compile-time args,
how to organize `#if defined` blocks, overall kernel_main structure), then adapt for the
current category.

```cpp
// Generic pattern — actual content is category-specific
#include <cstdint>
#include "api/compute/common.h"
// ... category-specific compute API includes (from proposal's Before blocks)

void kernel_main() {
    uint32_t num_tiles = get_compile_time_arg_val(0);
    // Additional compile-time args for ops with parameters
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    // Additional CBs as needed (e.g., cb_in1 for binary, cb_scaler for reduce)

    // Category-specific initialization
    // (init_sfpu, or binary init, or reduce init, etc. — from proposal)

#if defined(VALIDATE_OP_A)
    // --- EXACT sequence from proposal's LLK Sequence Validation table ---
    // Copy the proposal's "Before" code for this op verbatim.
    // Include all init, exec, CB, and DEST calls in the proposed order.
    // ...
#elif defined(VALIDATE_OP_B)
    // ... next op's sequence from proposal
#endif
}
```

CRITICAL RULES for the compute kernel:
- The LLK call sequence MUST match EXACTLY what the proposal says the helper will do internally.
  Do NOT write a "correct" kernel independently — write what the PROPOSAL DESCRIBES.
  If the proposal's sequence is wrong, this test should FAIL, catching the error.
- Include ALL init/exec calls in the exact order proposed, including re-inits after disruptive operations.
- Include the CB management (wait/pop/reserve/push) and DEST management (acquire/commit/wait/release)
  exactly as proposed.
- Use `#if defined(VALIDATE_OP_X)` blocks for each op so one kernel file covers all ops.

### 2. Host Test (C++ GTest)

File: `tests/tt_metal/tt_metal/llk/test_{CATEGORY_SLUG}_llk_validation.cpp`

Follow the exact structure of {{EXISTING_TEST_REFERENCE}}:

```cpp
// Key sections to include:

// Key sections — adapt all for the specific category:

// 1. Golden reference function — derive from mathematical definition of each op
//    Unary: bfloat16 golden(op_name, input) → e.g., std::exp, 1/x, std::sqrt
//    Binary: bfloat16 golden(op_name, a, b) → e.g., a+b, a*b, std::max(a,b)
//    Reduce: vector golden(op_name, input_matrix, dim) → e.g., row sum, col max
//    Signature depends on category — match the op's mathematical definition.
golden_function(...)

// 2. Input generation — avoid problematic values per op
//    Every category has dangerous inputs: zeros for division, negatives for sqrt,
//    large values for exp, etc. Use the op's domain constraints.
generate_input(...)

// 3. Comparison with op-specific tolerances
//    Transcendental ops (exp, log, sin) need wider tolerances than linear ops (add, sub).
//    Reduce ops accumulate error with tile count.
is_close_output(...)

// 4. Op defines mapping — map op name to preprocessor defines
//    Include any JIT include guards required by the category's compute API.
//    Read the reference test ({{EXISTING_TEST_REFERENCE}}) for the pattern of which
//    include guards are needed, then adapt for the current category.
get_op_defines(...)

// 5. Test runner — create program, buffers, CBs, kernels, run, compare
//    Adapt the buffer/CB setup for the category:
//    - Unary: 1 input CB, 1 output CB, reader_unary + writer_unary
//    - Binary: 2 input CBs, 1 output CB, reader_binary or custom + writer
//    - Reduce: input CB + scaler CB + output CB, may need custom reader
//    - Tilize: input CB (row-major) + output CB (tiled), specialized reader
//    Follow the reference test's structure for program/buffer/kernel creation.
run_llk_validation_test(...)

// 6. Parameterized test registration
//    Minimum: 1 tile, 4 tiles, 16 tiles per op
//    For reduce/matmul: also vary dimensions (e.g., wide row, tall column)
//    For binary: also test broadcast shapes if applicable
INSTANTIATE_TEST_SUITE_P(...)
```

CRITICAL RULES for the host test:
- Prefer standard reader/writer kernels from `tt_metal/kernels/dataflow/` when they fit.
  For unary: `reader_unary.cpp` + `writer_unary.cpp`. For binary: check for `reader_binary*.cpp`.
  Only write a custom reader/writer if no standard one matches the category's CB layout.
- Golden reference must be computed from the MATHEMATICAL definition of each op.
  Do NOT copy golden from existing test output.
- Include at minimum: 1 tile, 4 tiles, and 16 tiles per op (tests single, multi, and
  DEST-overflow scenarios). For multi-dimensional ops (reduce, matmul), also vary shapes.
- Include all JIT include guards needed by the compute kernel. These vary by category —
  determine them by reading the reference test and the category's compute API headers.
  Missing include guards cause cryptic compilation failures.
- For ops with parameters, test with at least 2 representative parameter values.

### 3. CMakeLists entry

Check if `tests/tt_metal/tt_metal/llk/CMakeLists.txt` exists. If so, add the new test file
to the appropriate target. If the test file pattern doesn't use CMake (e.g., uses a different
build system), follow whatever pattern the reference test uses.

## Running the Tests

After generating the files:

1. Build:
```bash
cd /localdev/astancov/tt-metal
./build_metal.sh
```

2. Run the tests:
```bash
source python_env/bin/activate
# Run with the project's test runner
./build/test/tt_metal/test_{CATEGORY_SLUG}_llk_validation
```

If any test hangs (output stops for > 60 seconds), STOP IMMEDIATELY and report it as a
SEQUENCE_HANG. Do NOT retry. Do NOT reset the device. The hang itself is diagnostic — it
means the proposed LLK sequence deadlocks the hardware.

Log each test result:
  echo '{"ts":"'"$(date -Iseconds)"'","event":"test_result","op":"OP_NAME","tiles":N,"status":"PASS/FAIL/HANG","details":"..."}' >> $BCRUMB

## Output Format

Write `{CATEGORY_SLUG}_device_validation.md` with this structure:

```markdown
# LLK Device Validation Results: {CATEGORY}

## Test Environment
- Device: {from tt-smi or device fixture output}
- Build: {commit hash}
- Date: {timestamp}

## Generated Files
- Compute kernel: `tests/tt_metal/tt_metal/test_kernels/compute/{CATEGORY_SLUG}_llk_validation.cpp`
- Host test: `tests/tt_metal/tt_metal/llk/test_{CATEGORY_SLUG}_llk_validation.cpp`

## Results

| Op | Tiles | Proposed Sequence | Status | Max Error | Notes |
|----|-------|-------------------|--------|-----------|-------|
| op_a | 1 | sequence_1 | PASS | 0.03 | |
| op_a | 16 | sequence_1 | PASS | 0.03 | DEST batching validated |
| op_b | 1 | sequence_2 | FAIL | 0.85 | Init ordering wrong — see details |
| op_c | 4 | sequence_3 | HANG | — | Sequence deadlocks — see details |

## Failures

### op_b (1 tile)
**Proposed sequence**: (exact init→exec sequence from proposal)
**Expected**: (mathematically correct result)
**Got**: (actual device output)
**Diagnosis**: (root cause — e.g., init ordering, mutual exclusion, DEST management)
**Proposed fix**: (corrected sequence)

### op_c (4 tiles)
**Status**: HANG after tile 2
**Diagnosis**: (which RISC-V is stuck, on what call, root cause)

## Sequence Verdict Summary

| Proposed Sequence | Ops Using It | Device Result | Verdict |
|-------------------|--------------|---------------|---------|
| sequence_1 | op_a, op_d, op_e | ALL PASS | VALIDATED |
| sequence_2 | op_b, op_f | 2/2 FAIL | INVALID — needs redesign |

## Impact on Proposal

List specific changes needed in the Stage 4 proposal based on test results:
- [what sequences need to be changed]
- [what op groupings need to be reconsidered]
- [what init orderings need to be fixed]
```

## Handling Failures

The review-fix agent handles failures AUTONOMOUSLY (no human intervention):

If a test FAILS (wrong numerical output):
1. Diagnose WHY the sequence fails (init ordering, DEST management, parameter encoding).
2. Fix the proposal's LLK Sequence Validation table with the corrected sequence.
3. Regenerate the test kernel with the corrected sequence.
4. Re-run. Up to 5 attempts per op before marking as UNVALIDATED.

If a test HANGS:
1. Diagnose from the hang pattern (which RISC-V is stuck, on what call).
2. Fix the proposal's CB/DEST management.
3. Regenerate kernel, re-run.

If ALL tests pass:
- The proposed sequences are VALIDATED for device execution.
- The review-fix agent can exit with confidence that the LLK internals are correct.

## Handling Parameterized Ops

For ops that take runtime parameters (e.g., elu with alpha, power with exponent):
- Pass parameters as additional compile-time args to the compute kernel
- Test with at least 2 representative parameter values
- Use `get_compile_time_arg_val()` in the kernel, OR pass via `#define` (whichever the
  reference test uses for similar cases)
- Golden function must accept the same parameters

## Handling Multi-Input Ops

For ops that need multiple input tensors (e.g., binary ops):
- Create additional DRAM buffers and CBs as needed
- Use `reader_binary.cpp` instead of `reader_unary.cpp` if available
- If no standard reader exists, write a minimal custom reader

At completion:
  echo '{"ts":"'"$(date -Iseconds)"'","event":"complete","total_ops":N,"passed":P,"failed":F,"hangs":H}' >> $BCRUMB
  Write agent_logs/${CATEGORY_SLUG}_device_validation_execution_log.md: per-op test trail, failures with diagnosis, generated file paths.
```

## How This Is Used

This document is NOT invoked as a standalone agent. It is reference material for
the Stage 5 review-fix agent (`llk_review_fix_agent.md`), which reads this during
Phase B (Device Validation). The review-fix agent's prompt template includes
`{{EXISTING_TEST_REFERENCE}}` and `{{EXISTING_KERNEL_REFERENCE}}` placeholders
that the orchestrator substitutes when invoking Stage 5.

The autonomous debug/fix loop (diagnose → fix proposal → regenerate kernel → re-run)
is implemented directly in the review-fix agent's Phase B protocol, following the
same pattern as `ttnn-kernel-writer` but focused on validating individual LLK
sequences rather than full operations.
