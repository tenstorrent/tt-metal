# Codegen Review Learnings

Source: review comments on tt-metal PR #42645 (Quasar `abs` SFPU kernel) + migrated thread `tt_llk_quasar/PR1562_review_comments.md`.

Each entry captures a concrete defect an agent produced, the root-cause reasoning, and the rule the agent must follow next time. Rules are tagged with the agent that owns them so they can be merged into the matching playbook.

---

## 1. Template arguments on LLK API entry points

**Defect (writer/test-writer):** Emitted `_llk_unpack_dest_dvalid_section_done_()` and `_llk_math_set_dvalid_<p_cleardvalid::FPU>()` without the required `<dest_sync>` / `<DstSync>` template arg. Fails to compile against `llk_unpack_common.h`'s `template <DstSync DST>` form.

**Root cause:** Agent matched call site syntax against the function name without re-reading the target header's signature. LLK functions frequently require `<dest_sync>` (or equivalent) and the requirement changes between arches.

**Rule → kernel-writer, test-writer:**
- Before emitting any `_llk_*` call, **read its declaration in the target arch header** (`llk_lib/llk_*.h` for pack/unpack/math). If it is a function template, you must pass every non-defaulted template parameter.
- For Quasar, functions taking `DstSync` typically expect `<dest_sync>` as the template arg — never emit the bare `()` form.

**Rule → debugger:**
- "no matching function for call" with one candidate that is a template → suspect missing template arg first.

---

## 2. `params.DST_INDEX` must offset every Dest-touching loop

**Defect (test-writer):** Datacopy and SFPU loops passed raw loop index `i` to `_llk_math_eltwise_unary_datacopy_(num_rows, i)` and `_llk_math_eltwise_unary_sfpu_params_(..., i, ...)`. Other Quasar unary SFPU tests (e.g., `sfpu_rsqrt_quasar_test.cpp`, `sfpu_square_quasar_test.cpp`) use `params.DST_INDEX + i`. With non-zero runtime `DST_INDEX`, the bare `i` form corrupts the wrong Dest region.

**Root cause:** Agent copied a simplified pattern (without DST_INDEX offset) instead of mirroring the closest existing test. Test-writer's playbook calls this out in general terms — the failure is that it wasn't enforced on the loop-body level.

**Rule → test-writer:**
- For Quasar unary SFPU tests, when emitting a MATH-side loop that calls `_llk_math_eltwise_unary_datacopy_` or `_llk_math_eltwise_unary_sfpu_params_`, the tile index argument **must be `params.DST_INDEX + i`, never bare `i`**.
- Before writing any tile-indexed LLK call, grep the closest existing target test (`sfpu_square_quasar_test.cpp` or `sfpu_rsqrt_quasar_test.cpp`) for the same LLK function and copy the index expression verbatim.
- The PACK side already uses `params.DST_INDEX` — if the MATH side uses a different offset expression, they are mismatched by construction.

---

## 3. No duplicate SFPU header under `llk_lib/`

**Defect (writer):** Created the kernel header twice — in `tt_llk_quasar/common/inc/experimental/ckernel_sfpu_abs.h` **and** `tt_llk_quasar/llk_lib/experimental/ckernel_sfpu_abs.h`. Both dirs are on the include path for kernel tests, so which copy wins depends on include-dir ordering. The two copies already diverged (the `common/inc/` one picked up extra commentary about `SFPABS` `instr_mod1`).

**Root cause:** Agent invented a duplicate location. All other Quasar SFPU kernels live only in `common/inc/sfpu/ckernel_sfpu_*.h` (no `llk_lib/sfpu/` twin).

**Rule → planner, writer:**
- Quasar SFPU kernel headers live in exactly **one** location: `tt_llk_quasar/common/inc/sfpu/ckernel_sfpu_{op}.h` (or `common/inc/experimental/` for AI-generated kernels not yet validated). There is no `llk_lib/sfpu/` or `llk_lib/experimental/` twin. **Never create one.**
- When the planner emits the target file path, it must grep for similar existing kernels first (`find tt_llk_quasar -name 'ckernel_sfpu_*.h'`) and match the single observed location.

**Rule → debugger:**
- If a generated kernel already has a duplicate header, treat the `llk_lib/` copy as the spurious one and delete it; the test `#include "experimental/ckernel_sfpu_{op}.h"` resolves through `common/inc/` for all other SFPU kernels.

---

## 4. Comment quality: explain *why*, not *what*

**Defect (writer/test-writer):** Comments describe the mechanical action (`// Setup dvalid for MATH kernel`, `// Datacopy all tiles from SRC to DEST`, `// Apply SFPU absolute value`) instead of the rationale a reviewer needs. Reviewer nvelickovicTT asked specifically for:

- Why `set_up_dest_dvalid_per_thread` is called this way — which clients are involved and why those three.
- Why `_llk_math_upk_to_dest_hw_configure_` is needed on MATH even when UNPACK writes Dest directly (SFPU still reads Dest in the right format).
- Why the buffer descriptor dims line up (`x=cols, y=rows/face, z=num_faces`) — current code does have this line, keep it.
- Why a `wait_*_idle()` sequence covers SFPU/FPU/MOP but not REPLAY (this kernel uses straight-line SFPU, no replay buffer → no REPLAY wait needed). The updated version does include this reasoning — **retain the pattern**.

**Rule → kernel-writer, test-writer:**
- Each non-trivial LLK call site gets a one-line comment answering **"why is this here and why in this form?"** — not a restatement of the function name.
- Specific required explanations on Quasar SFPU tests:
  - **dvalid setup**: name the producer/consumer chain (e.g., "Dest clients for this kernel: UNPACK writes, SFPU reads+writes, PACK reads") and why the chain differs between the `unpack_to_dest` path and the FPU path.
  - **MATH hw configure on unpack_to_dest path**: "MATH still needs format configuration because SFPU reads Dest in that format, even though FPU datacopy is bypassed."
  - **Idle waits**: enumerate which units could still be running and why. If there is no REPLAY wait, say so and explain (no replay buffer).
- Never write `ELWADD`-style comments on Quasar. `ELWADD` was a WH/BH workaround for an 8-row MOVA2D bug; Quasar does not have it. If porting from BH/WH, drop any references to that workaround.
- No abbreviations like "MOV2D" — always write `MOVA2D` or `MOVB2D`.

---

## 5. Explain loop iteration count when it's subtle

**Defect (test-writer):** The line `const std::uint32_t num_sfpu_iterations = params.TEST_FACE_R_DIM / ckernel::math::SFP_ROWS;` followed by an outer `for (i < params.TILE_CNT)` reads as "one SFPU invocation per tile" but a reader cannot tell from the code whether `_llk_math_eltwise_unary_sfpu_params_` advances by tile or by face. Reviewer (nvelickovicTT) asked exactly that; the resolution was that it's fine **because Quasar SFPU only supports `VectorMode::RC` (the default and only mode), so iterating a single face is equivalent to iterating the whole tile.**

**Rule → test-writer:**
- Describe what each LLK call does **per invocation**, not just what the loop counts. Per `abs_arch_research.md`: `_llk_math_eltwise_unary_sfpu_params_` processes one whole tile per call — for each face in the tile it invokes the sfpu helper with `num_sfpu_iterations`, and each iteration covers `SFP_ROWS` rows (so `num_sfpu_iterations * SFP_ROWS == TEST_FACE_R_DIM` rows per face).
- **Reviewer-supplied domain facts may be quoted** in the code comments when they resolve an ambiguity a reader would otherwise trip on — for example, the `VectorMode::RC` note above. The guardrail is:
  - Prefix such comments with `Note:` (or equivalent) so a future reader recognises it as domain knowledge, not something derivable from the surrounding code.
  - Do not stretch the reviewer fact to cover claims it didn't make.
  - File a corresponding arch-lookup entry (see rule below) so the pipeline can verify the fact itself next time.
- Careful about `params.num_faces`: it **is** used on the FPU path for `num_rows = num_faces * TEST_FACE_R_DIM` (passed to the datacopy init). Do not claim it is unused in MATH. If it appears unused in a particular code path, point at the path that does consume it.

**Rule → arch-lookup:**
- For SFPU kernels on Quasar, the arch brief must answer: **"Which `VectorMode` values does Quasar SFPU support, and what is the default?"** If only one is supported, call that out explicitly — it has direct implications for how tile-vs-face loops are written, and it is the kind of fact that tends to live only in review threads today.

**Rule → analyzer / planner:**
- Treat review threads (e.g., `PR1562_review_comments.md`) as a reviewed source of domain knowledge. When they assert a constraint the pipeline doesn't currently capture (like "VectorMode::RC is the only mode"), flag it in the analysis/spec so downstream agents know it's available and so arch-lookup can promote it to first-class research next time.

---

## 6. Don't carry forward reference-arch artifacts

**Defect (analyzer/planner, implicitly):** Comments in the Blackhole reference (about ELWADD as a datacopy workaround) were preserved in rewritten form even though the workaround does not apply to Quasar. Reviewer flagged this as misleading.

**Rule → analyzer, planner:**
- The analysis output must explicitly list **"BH/WH workarounds that do NOT apply to Quasar"** so the writer can drop them. Known entries to watch for on SFPU kernels:
  - `ELWADD`-as-datacopy (WH/BH 8-row MOVA2D bug) — not applicable on Quasar.
  - Any comment that says "we do X because of bug Y" — verify the bug still exists on target before carrying the comment forward.

---

## 7. Future improvement (not urgent): merge unary SFPU tests

Reviewer ldjurovicTT suggested folding `test_sfpu_abs_quasar.py` into a combined SFPU sweep (as WH/BH do), once multiple unary SFPU kernels land.

**Rule → test-writer (deferred):**
- When generating the N-th unary SFPU test for Quasar (N > 2), default to extending the existing sweep file (`test_sfpu_nonlinear_quasar.py` or its successor) rather than adding a new per-op file. Only add a new file if the op has test semantics that cannot be parameterized into the sweep.

---

## Summary: where each rule lands

| Rule | Agent playbook(s) to update |
|------|-----------------------------|
| 1. Template args on LLK calls | `llk-kernel-writer.md`, `llk-test-writer.md`, `llk-debugger.md` |
| 2. `params.DST_INDEX + i` everywhere | `llk-test-writer.md` |
| 3. No duplicate SFPU header locations | `llk-planner.md`, `llk-kernel-writer.md`, `llk-debugger.md` |
| 4. Comment the *why* (dvalid / hw_configure / waits, no abbreviations) | `llk-kernel-writer.md`, `llk-test-writer.md` |
| 5. VectorMode::RC + unused-param rationale | `llk-test-writer.md` |
| 6. Drop reference-arch workarounds | `llk-analyzer.md`, `llk-planner.md` |
| 7. Merge into SFPU sweep (deferred) | `llk-test-writer.md` |
