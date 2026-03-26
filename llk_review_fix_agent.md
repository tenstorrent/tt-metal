---
name: LLK Review-Fix Loop Agent Prompt
description: >
  General-purpose review-fix loop agent. Reads any pipeline output document,
  finds issues (BLOCKER / CONFUSING / MINOR), fixes them, re-reviews, and
  repeats until the exit condition is met. Can be applied to any stage output:
  proposals, op struct designs, investigation reports, or verification results.
type: reference
---

## Usage

Invoke with `subagent_type: general-purpose`. Replace placeholders:
- `{{TARGET_FILE}}` — the document to review and fix (path from repo root)
- `{{REFERENCE_FILES}}` — newline-separated list of ground-truth files to verify claims against
- `{{REVIEW_CRITERIA}}` — domain-specific checklist items (in addition to the general criteria baked in)
- `{{DONE_CONDITION}}` — exit predicate, e.g. "0 blockers and 0 confusing issues"

## Prompt Template

```
You are running a review-fix loop on {{TARGET_FILE}}.

BREADCRUMB LOGGING — do this first:
Derive SLUG from {{TARGET_FILE}} filename (stem, underscores).
BCRUMB="agent_logs/${SLUG}_review_breadcrumbs.jsonl"
mkdir -p agent_logs
echo '{"ts":"'"$(date -Iseconds)"'","event":"start","agent":"review_fix","target":"{{TARGET_FILE}}"}' >> $BCRUMB

Your job: read the document, find issues, fix them, re-review, and repeat
until the exit condition is met. Run autonomously — do not stop after one
round and do not ask for confirmation between rounds.

## Target file

{{TARGET_FILE}}

## Reference files

Read these to verify claims in the document against ground truth:

{{REFERENCE_FILES}}

## Review criteria

Classify every issue as:
  BLOCKER   — factually wrong, inconsistent with referenced code, or missing
              something a reader NEEDS to use the design correctly
  CONFUSING — correct but unclear; a careful reader would misunderstand or
              need to re-read to get the right answer
  MINOR     — style, completeness of examples, polish; document works without it

### General criteria (apply to every document type)

**Accuracy**
- Do code examples match actual implementation signatures (exact names, template
  params, argument types)?
- Are names spelled exactly as they appear in the referenced code?
- Do Before/After blocks show the ORIGINAL code before migration, not the
  already-migrated version?
- Do stated invariants (e.g. "empty body", "called once per tile") match
  what the .inl / implementation actually does?
- Are enum values, flag names, and constant names correct?

**Completeness**
- Does every proposed API have: a signature, at least one Before/After example,
  a list of what it absorbs, and a note on prerequisites?
- Are design decisions recorded with rationale (not just "we chose X")?
- Are excluded cases (ops or patterns that don't fit) documented with reasons?
- Are open questions listed?
- Are all pipeline stages that produced this document's inputs cited?

**Clarity**
- Can a developer use the API without reading the implementation?
- Are non-obvious parameter semantics (operand order, template values, enum
  meanings) explained with concrete examples?
- Is there a decision guide for choosing between alternative APIs?
- Are footguns (places where a plausible wrong choice compiles but produces
  incorrect results) explicitly called out?

**Architectural soundness**
- Does any "caller must" or "caller is responsible for" requirement ask the
  caller to manage hardware state that the helper is supposed to abstract?
  Example failure mode: a helper that acquires DST but does not release it,
  forcing the caller to call tile_regs_release() — this leaks internal state
  and defeats the purpose of the helper. Flag as BLOCKER.
- If the helper manages some but not all of a paired hardware operation
  (e.g. acquire without release, commit without wait), verify this is
  intentional AND that the document explicitly states why the pairing is
  split and who owns the other half. If there is no clear rationale, flag
  as BLOCKER.
- Check: can a developer use the helper in all documented modes without
  understanding the underlying hardware sync model? If a mode requires the
  caller to know about tile_regs, DST slot assignment, or CB reservation
  internals, the abstraction is leaky — flag as CONFUSING or BLOCKER
  depending on whether incorrect usage would corrupt state.

**LLK sequence validity**
- Does the proposal include an "LLK Sequence Validation" section? If not,
  flag as BLOCKER — every helper must have its internal LLK call sequence
  validated against codebase evidence.
- For each helper: write out the internal init→exec sequence. Does each
  init immediately precede its own exec (no intervening init from a
  different LLK function that could overwrite hardware state)? If not,
  flag as BLOCKER.
- Does the proposal combine init functions that the investigation's
  "Init Mutual Exclusion" table shows never coexist in any kernel?
  If yes, flag as BLOCKER.
- Is every claimed "Codebase Precedent" file:line actually a real kernel
  that uses the sequence? Spot-check at least one per helper by reading
  the cited file. If the citation is wrong or the sequence doesn't match,
  flag as BLOCKER.

### Domain-specific criteria

{{REVIEW_CRITERIA}}

## Loop protocol

Stage 5 has two phases that run as an autonomous loop: document review (Phase A) followed
by device validation (Phase B). No human intervention between phases.

### Phase A: Document Review

Each round:
1. Read the current state of {{TARGET_FILE}}.
2. Read the relevant reference files to verify claims — do NOT skip this step.
3. Build a complete issue list classified as BLOCKER / CONFUSING / MINOR.
4. Fix ALL issues found in priority order: BLOCKERs first, then CONFUSING, then MINOR.
5. Write the updated file with all fixes applied.
6. Log the round:
     echo '{"ts":"'"$(date -Iseconds)"'","event":"round","phase":"A","n":N,"blockers":B,"confusing":C,"minor":M,"status":"continuing/phase_b"}' >> $BCRUMB
7. Print: ROUND N (Phase A) — [B blockers, C confusing, M minor] → [continuing / entering Phase B]

When writing fixes:
- Preserve all correct content; only change what is wrong or missing.
- When a Before block shows migrated code, replace it with the original code.
- Never truncate correct sections to save space.
- Add missing sections in full — do not reference them as "left as exercise".
- Do NOT re-introduce issues that were fixed in a previous round.

Phase A exits when: 0 document blockers and 0 confusing issues.

### Phase B: LLK Device Validation

After Phase A produces a clean proposal, validate the proposed LLK call sequences on
actual hardware. This catches sequences that look correct in code but produce wrong
results or hang due to hardware state interactions.

**What to test**: Each UNIQUE LLK call sequence from the proposal's "LLK Sequence Validation"
table. The test exercises the EXACT sequence the helper will execute internally, using raw
LLK API calls (the helper does not exist yet).

**Test generation strategy**: Read the reference test ({{EXISTING_TEST_REFERENCE}}) and its
compute kernel ({{EXISTING_KERNEL_REFERENCE}}) to learn the structural pattern. Then adapt
that pattern for the current category. The reference test provides:
- How to set up DRAM buffers, CBs, reader/writer/compute kernels
- How to use preprocessor defines for op selection
- How to generate golden references and compare with tolerances
- How to register parameterized GTests

For each proposed helper sequence, generate two files:

1. **Compute kernel** at `tests/tt_metal/tt_metal/test_kernels/compute/{{CATEGORY_SLUG}}_llk_validation.cpp`

   Use preprocessor defines (`#if defined(VALIDATE_OP_X)`) to select which op/sequence to
   test. Each `#if` block must replicate the EXACT init→exec sequence from the proposal,
   including all CB management and DEST management calls.

   What goes in the kernel depends on the category:
   - **Unary SFPU**: `init_sfpu()`, `copy_tile_to_dst`, SFPU `*_tile_init()` / `*_tile()`, `pack_tile`
   - **Binary eltwise**: `init_sfpu()` or FPU init, two input CB waits, `*_tiles_init()` / `*_tiles()`, pack
   - **Reduce**: `reduce_init()`, scaler CB, accumulation loops, `reduce_tile()`, optional post-reduce op
   - **Tilize/Untilize**: `tilize_init()` / `untilize_init()`, block-width loops, fast path dispatch
   - **Matmul**: `mm_init()`, inner-dim loops, accumulation
   - **Other categories**: Follow the Before/After examples in the proposal — the "Before" block
     IS the raw sequence to test

   The includes also vary by category:
   - Read the proposal's "Before" code blocks to determine which `api/compute/` headers are needed
   - Read the reference kernel ({{EXISTING_KERNEL_REFERENCE}}) for the include pattern

   CRITICAL: Write what the PROPOSAL DESCRIBES, not what you think is correct. If the
   proposal's sequence is wrong, the test must FAIL to catch the error.

2. **Host-side GTest** at `tests/tt_metal/tt_metal/llk/test_{{CATEGORY_SLUG}}_llk_validation.cpp`

   Follow the structure of {{EXISTING_TEST_REFERENCE}}, adapting for the category:

   - `golden_function()` — mathematical reference per op. Derive from the op's definition
     (e.g., exp→std::exp, add→a+b, reduce_sum→row/col sum). Each category has different
     golden functions — read the proposal's op descriptions to implement them.
   - `generate_input()` — safe inputs per op. Avoid domain-specific dangerous values:
     - Unary SFPU: no zeros for recip, no negatives for sqrt
     - Binary: consider broadcast shapes, avoid division by zero for div
     - Reduce: ensure accumulation stays in representable range
     - General: use small magnitude values to avoid overflow in bfloat16
   - `is_close_output()` — op-specific tolerances. Transcendental functions (exp, log, sin)
     need wider tolerances than linear ops (add, sub).
   - `get_op_defines()` — map op name → preprocessor defines for the compute kernel.
     Include any JIT include guards required by the category's compute API headers.
     Determine these by reading the reference test and the category's compute API headers.
   - `run_llk_validation_test()` — create program, buffers, CBs, kernels, run, compare.
     Choose reader/writer kernels based on category:
     - Unary ops: `tt_metal/kernels/dataflow/reader_unary.cpp` + `writer_unary.cpp`
     - Binary ops: `tt_metal/kernels/dataflow/reader_binary_diff_lengths.cpp` or similar + writer
     - Reduce: may need scaler CB + scaler data in addition to input
     - If no standard reader/writer fits, write a minimal custom one
   - Parameterized test registration: 1 tile, 4 tiles, 16 tiles per op at minimum.
     For reduce/matmul, also test different shapes (e.g., wide rows, tall columns).

   For ops with runtime parameters, pass them as additional compile-time args and test with
   at least 2 representative parameter values.

3. **Build and run**:
   ```bash
   cd /localdev/astancov/tt-metal
   ./build_metal.sh
   ./build/test/tt_metal/test_{{CATEGORY_SLUG}}_llk_validation
   ```

   If the test binary doesn't exist after build, check CMakeLists.txt at
   `tests/tt_metal/tt_metal/llk/CMakeLists.txt` and add the test source.

**Handling test results**:

- **ALL PASS**: Log and proceed to exit.
  ```
  echo '{"ts":"'"$(date -Iseconds)"'","event":"device_validation","status":"ALL_PASS","ops_tested":N}' >> $BCRUMB
  ```

- **TEST FAILURE** (wrong numerical output):
  Log the failure with expected vs actual values:
  ```
  echo '{"ts":"'"$(date -Iseconds)"'","event":"device_validation_fail","op":"OP","tiles":N,"expected":"...","got":"...","hypothesis":"..."}' >> $BCRUMB
  ```
  Then autonomously debug and fix:
  1. **Diagnose**: What is the proposed sequence doing wrong? Common causes vary by category:
     - **Init ordering**: one init overwrites hardware state needed by a subsequent exec
       (e.g., copy reconfigures unpack, so a compute init may need re-calling after copy)
     - **Missing re-init**: a disruptive init (copy, reduce, matmul) reconfigures shared
       hardware, and the subsequent operation's init was not re-called
     - **Init mutual exclusion**: two inits that configure the same hardware in incompatible
       ways are called in the same kernel
     - **Parameter encoding**: bit_cast or derived parameter computation wrong
     - **DEST slot mismatch**: writing to or reading from wrong DEST index
     - **CB sync**: wrong number of tiles waited/popped/reserved/pushed
     - **Accumulation**: DEST not cleared between accumulation phases (reduce/matmul)
  2. **Fix the proposal**: Update the LLK Sequence Validation table in {{TARGET_FILE}} with the
     corrected sequence. Document WHY the original sequence failed.
  3. **Regenerate the test kernel** with the corrected sequence.
  4. **Re-run**. Repeat until the sequence passes.
  5. Each fix attempt is one "device round":
     ```
     echo '{"ts":"'"$(date -Iseconds)"'","event":"round","phase":"B","n":N,"op":"OP","attempt":M,"status":"PASS/FAIL","fix_applied":"..."}' >> $BCRUMB
     ```

- **HANG** (output stops for > 60 seconds):
  If using `scripts/tt-test.sh`, the hang is auto-detected with triage.
  For GTest binaries, watch for stalled output.
  ```
  echo '{"ts":"'"$(date -Iseconds)"'","event":"device_validation_hang","op":"OP","tiles":N,"diagnosis":"..."}' >> $BCRUMB
  ```
  Diagnose from the hang pattern:
  - **Compute stuck on cb_wait_front**: Producer (reader) never pushed, or wrong CB index
  - **Writer stuck on cb_wait_front**: Compute never pushed to output CB
  - **Compute stuck on tile_regs_acquire**: Previous acquire block never released
  - **Reader/Writer stuck on noc_async_*_barrier**: NoC transfer never completed (address alignment issue)
  Fix the proposal's CB/DEST management, regenerate kernel, re-run.

- **COMPILE FAILURE**:
  Fix includes, template params, or API signatures. Log and retry.

**Max attempts**: 5 per op. If an op fails 5 times, mark the sequence as UNVALIDATED in the
proposal and flag for human review. Do not loop forever.

**Device validation output**: Write `{{CATEGORY_SLUG}}_device_validation.md`:
```markdown
# LLK Device Validation Results: {{LLK_CATEGORY}}

## Generated Files
- Compute kernel: `tests/tt_metal/tt_metal/test_kernels/compute/{{CATEGORY_SLUG}}_llk_validation.cpp`
- Host test: `tests/tt_metal/tt_metal/llk/test_{{CATEGORY_SLUG}}_llk_validation.cpp`

## Results
| Op | Tiles | Proposed Sequence | Status | Attempts | Notes |
|----|-------|-------------------|--------|----------|-------|

## Sequence Verdict Summary
| Proposed Sequence | Ops Using It | Device Result | Verdict |
|-------------------|--------------|---------------|---------|
```

### Outer Loop: Phase A ↔ Phase B

If Phase B finds failures that require changes to the proposal:
1. Fix the proposal (update LLK Sequence Validation, Before/After, etc.)
2. Re-enter Phase A to verify the document is still consistent after changes.
3. Re-enter Phase B to validate the corrected sequences.

This loop continues until:
- Phase A: 0 document blockers
- Phase B: ALL LLK sequences pass on device (or are marked UNVALIDATED with justification)

```
Phase A (doc review) ──[0 blockers]──▶ Phase B (device validation)
    ▲                                         │
    │         [FAIL/HANG: sequence invalid]    │
    └─────────────────────────────────────────┘
                                              │
                                   [ALL PASS] ▼
                                            done
```

## Exit condition

{{DONE_CONDITION}}

Additionally, if device validation is enabled ({{EXISTING_TEST_REFERENCE}} is provided):
ALL proposed LLK sequences must pass on device, or be explicitly marked UNVALIDATED.

Stop only when both conditions are met. Output the final round log and a
one-paragraph summary of all changes made across all rounds.

## Feedback summary (mandatory — always output at the end)

After the final round, print this block so the orchestrator can route next steps:

```
FEEDBACK_SUMMARY_START
FEEDBACK_TARGET: <stage5_proposal | stage6_op_structs | stage4_verification | stage3_investigation | implementation_hpp | none>
BLOCKING_ISSUES_REMAINING: <count>
UPSTREAM_CHANGES_NEEDED: <yes | no>
DEVICE_VALIDATION: <ALL_PASS | N_FAILED | NOT_RUN>
UNVALIDATED_SEQUENCES: <count or 0>
DETAILS:
  - <one line per upstream change needed — e.g. "Exp<true,true> struct missing from .hpp">
  - <one line per UNVALIDATED sequence — e.g. "elu init→exec: failed 5 attempts, needs manual investigation">
  - or "none" if no upstream changes needed
FEEDBACK_SUMMARY_END
```

Rules for selecting FEEDBACK_TARGET:
- `none` — all issues were fixable within the document itself and all sequences validated
- `implementation_hpp` — document was correct but described ops/signatures not yet in the `.hpp`/`.inl`
- `stage6_op_structs` — op struct definitions need rework (re-run Stage 6)
- `stage5_proposal` — proposal structure or design decisions need rework (re-run Stage 5)
- `stage4_verification` — a verification verdict was wrong and affected the proposal (re-run Stage 4)
- `stage3_investigation` — investigation data was incomplete or wrong (re-run Stage 3)

The orchestrator reads FEEDBACK_SUMMARY to decide: advance to next pipeline stage,
re-enter an earlier stage, or fix the implementation directly.
```

## When to use this agent

- After **Stage 5** (Helper Proposal) to ensure the proposal is accurate and
  usable before implementation begins.
- After **Stage 6** (Op Struct Design) to verify struct definitions match the
  raw compute API signatures.
- After **Stage 3** (Investigation) if the consolidated investigation report
  needs a consistency pass before verification claims are written.
- Any time a pipeline output document has been partially written and needs
  a correctness check before the next stage consumes it.

## How to invoke

```python
# Pseudocode — illustrative only
Agent(
    subagent_type="general-purpose",
    prompt=review_fix_template
        .replace("{{TARGET_FILE}}", "{category}_helper_proposal.md")
        .replace("{{REFERENCE_FILES}}", """
- ttnn/cpp/ttnn/kernel_lib/{category}_helpers.hpp
- ttnn/cpp/ttnn/kernel_lib/{category}_helpers.inl
- ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp  (style reference)
- {category}_investigation.md
- {category}_verification.md
""")
        .replace("{{REVIEW_CRITERIA}}", """
- Does every helper have a signature, Before/After, and what-it-absorbs?
- Are all migrated kernels listed with their correct After code?
- Is the Combine policy interface (init + apply) documented if applicable?
- Are design decisions (e.g. per-tile init, cb_pop_front timing) explained?
""")
        .replace("{{DONE_CONDITION}}", "0 blockers and 0 confusing issues")
)
```
