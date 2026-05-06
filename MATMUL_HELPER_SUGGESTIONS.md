# Matmul-Helper Improvement Suggestions

## Why this document exists

The matmul kernel helpers (`matmul_block_helpers.hpp`, `bias_add_helpers.hpp`,
`reblock_untilize_helpers.hpp`, `transpose_block_helpers.hpp`) are designed to
be the **canonical building blocks** that the `tt_ops_code_gen` Claude Code
pipeline reaches for when it generates a TTNN op. That pipeline is the
intended deployment path: a user supplies a math description, the planner →
implementer → verifier agents read the helpers and emit a working op.

Until now, the helper design has only been validated by **hand-migration** of
existing kernels (SDPA, conv2d, conv3d, bmm). That validates correctness but
not ergonomics — a kernel author who knows the API can navigate gaps the
docstring leaves implicit.

**This run is the first end-to-end test of whether agents can navigate the
helpers from cold start.** Three Claude sessions (each spawned fresh by
`run_op.py`, no prior helper knowledge in context) had to:
- Discover which helpers cover the op's compute phases
- Read the headers, derive the right template parameters, and write a kernel
- Compile and pass tests
- Audit the result for misuse and fix any issues

If the helper API is well-designed for this workflow, the pipeline produces a
working op without hand-holding. If the API has friction, the friction shows
up as wrong includes, hand-rolled raw-API fallbacks where a helper exists,
silent precision bugs, or rejected designs.

**Result.** The pipeline produced a kernel that compiled the first time and
passed 35/35 tests. The matmul helpers are validated in the deployment
workflow. The suggestions below are the small but real friction points the
run exposed — each one is something an agent stumbled on while reading our
docstrings, not a hypothetical critique. Apply them and the next codegen run
of a matmul-shaped op should be smoother still.

This doc is for the engineer (or agent) who will apply edits. Every item
names the file, the symbol/line, the proposed change, and the **specific
observation** from the run that motivated it.

---

## The op the pipeline implemented

A single-core 2D matmul with optional row-broadcast bias — chosen as the
**simplest possible matmul-shaped forcing function** so that any failure
pattern would isolate to matmul-helper integration rather than algorithm
complexity.

```python
linear(
    input_tensor: ttnn.Tensor,    # [1, 1, M, K] TILE bfloat16
    weight_tensor: ttnn.Tensor,   # [1, 1, K, N] TILE bfloat16
    *,
    bias: ttnn.Tensor = None,     # [1, 1, 32, N] TILE bfloat16, broadcast row
) -> ttnn.Tensor                  # [1, 1, M, N] TILE bfloat16
```

Math: `output[m, n] = sum_k input[m, k] * weight[k, n]  (+ bias[0, n])`.
Constraints: M, K, N divisible by 32; bf16; TILE_LAYOUT; DRAM interleaved;
single Tensix at (0, 0). No multi-core, no padding, no transpose, no
activation fusion — just the matmul + bias_add helper pair driving FPU work
on one core. Refinements (multi-core, dtypes, alignment) are explicitly out
of scope for Phase 0.

The forcing function for the run lives at:
- `eval/prompts/linear.txt` (the agent input)
- `eval/golden_tests/linear/` (golden tests + API contract)

The pipeline produced an op that exercises both helper paths under test:

| Test suite | Tests | Path exercised |
|---|---:|---|
| `test_linear.py` (acceptance) | 19 | `matmul_block` (no-bias) + `matmul_block` → `add_bias_bcast_rows` (bias) + Python validation |
| `test_linear_extended.py` | 8 | Verifier-added shape and parameter coverage |
| `test_linear_precision_baseline.py` | 8 | PCC ≥ 0.99999775 across 4 shapes × {bias, no-bias} |
| **Total** | **35** | **35/35 PASS** |

---

## How the three pipeline phases performed

The pipeline drove three independent Claude sessions, each with no prior
context. The wall-clock and turn-count breakdown below is the **direct
evidence base** for the suggestions further down — every observed friction
maps to a specific phase.

| Phase | Wall | Turns | What it produced | Helper-related verdict |
|---|---:|---:|---|---|
| **Planner** | 6 min | 46 | `op_design.md` + acceptance test (`test_linear.py`) | Picked `matmul_block` + `add_bias_bcast_rows`, cited `matmul_block_helpers.hpp:298-319`, `:.inl:206-207`, `:.inl:450-464`, `bias_add_helpers.hpp:168-179`, `:93-97`, `:81-90`. Correct `LastBlockTarget`, correct `OutputLayout` match, correct `interm_buf` placeholder. **One hallucinated header path** for `compute_kernel_hw_startup` — see Priority 1. |
| **Implementer** | 13 min | 66 | `linear.py`, `linear_program_descriptor.py`, 3 kernels (reader/compute/writer). | Compiled the first reported attempt; 19/19 acceptance tests pass. Wrapped CBs in `experimental::CircularBuffer`, used the documented `MatmulBlockShape::of(...)` factory, branched cleanly on `if constexpr (has_bias)`. **Found `HiFi4 + fp32_dest_acc_en=True` requirement by failing tests first**, not from the docstring — see Priority 2. |
| **Verifier** | 1h 47m | n/a (claude `--max-turns 200`) | `verification_report.md`, `capabilities.md`, `op_requirements.md`, `changelog.md`, `data_transfer.md`, `numerical_stability.md`, extended + precision-baseline tests. | Re-audited helper usage with five explicit checkmarks; made one canonical-style fix (raw `cb_wait_front` → `bias_buf.wait_front`); flagged that `HiFi4 + fp32_dest_acc_en=True` is a known WH B0 fault path (#38306). Substantive work done in first ~14 min; remaining ~90 min was repeated test runs and a long quiet stretch. |

The verifier's apparent silence drove a misdiagnosis on my part during the
run (I thought it was hung) — that lesson is recorded as a curriculum note
separately and is not a helper concern.

The commits added by the run, on top of the matmul-helper feature stack:
- `394ff09a3ae` `[incremental-planner] linear op design and acceptance test`
- `08795221c53` `[ttnn-implementer] PASS: linear Phase 0 — matmul_block + bias_add helpers (19/19 tests)`
- `d8e9602aff9` `[incremental-verifier] PASS: linear Phase 0 — verification artifacts (35/35 tests)`

---

## What the pipeline already does well — DO NOT regress

The codegen run produced kernels that an experienced kernel writer would
accept without re-architecting. Specifically, the planner:

- Picked `matmul_block_helpers.hpp` and `bias_add_helpers.hpp` as primary
  building blocks, citing file:line throughout (e.g. `matmul_block_helpers.hpp:298-319`,
  `matmul_block_helpers.inl:206-207`, `bias_add_helpers.hpp:93-97`).
- Chose `LastBlockTarget::Out` for the no-bias path and `LastBlockTarget::Interm`
  for the bias path, with the right `cb_partials` plumbing.
- Matched `OutputLayout::SubblockMajor` across both helpers (avoiding gotcha #8
  in its own design).
- Used the documented "pass the output buffer as `interm_buf` placeholder when
  `num_k_blocks == 1`" pattern (gotcha #4 in its design).
- Identified caller-managed bias CB lifecycle from the docstring at
  `bias_add_helpers.hpp:93-97`.

The implementer translated the design into a kernel that compiled the first
time and passed all 19 acceptance tests. So the helper docstrings + `@example`
blocks are mostly load-bearing as written. The suggestions below are about
closing the small but recurring friction points the run exposed — not about
restructuring the API.

---

## Priority 1 — Self-document the prerequisite include

**Where:** `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp` (and mirror in
`bias_add_helpers.hpp`, `reblock_untilize_helpers.hpp`,
`transpose_block_helpers.hpp`).

**Symptom observed.** The planner's `op_design.md` cited
`tt_metal/include/compute_kernel_api/common.h (canonical entry)` as the
"Prerequisite include for `compute_kernel_hw_startup`." That path **does not
exist** in this tree. The actual file is
`tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h` and the include
directive is `#include "api/compute/compute_kernel_hw_startup.h"`.

The planner only avoided emitting the wrong include because the implementer
agent has it correct in its own helper-table preamble. If a future planner
hands the design to a less-instrumented downstream (or a verifier audits the
design itself), the wrong path becomes load-bearing.

**Why this happened.** The matmul helper docstring says:

```
Init handling: by default the helper calls mm_block_init() itself
(init_mode=Full). The caller's only init responsibility is one
compute_kernel_hw_startup() at boot.
```

It tells you to call `compute_kernel_hw_startup()` but does not name the
header. The planner had to infer the header from `binary_op_helpers.hpp:13-19`
(which it cited correctly), but in the API-mapping table it then paraphrased
the header path and got it wrong.

**Suggested change.** Add a one-line "include this" note at the top of the
public docstring, immediately above the `@example` blocks:

```cpp
/**
 * matmul_block: sub-blocked tiled matrix multiplication ...
 *
 * Required includes (kernel must include both):
 *   #include "api/compute/compute_kernel_hw_startup.h"   // for compute_kernel_hw_startup()
 *   #include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
 *
 * Init handling: by default the helper calls ...
 */
```

Apply the same shape (helper-specific include + the shared
`compute_kernel_hw_startup.h`) to the three sister headers
(`bias_add_helpers.hpp`, `reblock_untilize_helpers.hpp`,
`transpose_block_helpers.hpp`). Each of them implicitly assumes the same
boot init.

---

## Priority 2 — Surface the math_fidelity / fp32_dest_acc_en interaction

**Where:** `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp` Doxygen header
(the section that already documents template parameters).

**Symptom observed.** The implementer's program descriptor selected
`MathFidelity.HiFi4 + fp32_dest_acc_en=True`. From its breadcrumb:

> HiFi4 + fp32_dest_acc_en required for K>=64 to meet rtol=0.02 atol=0.1.

The implementer found this **by failing tests first**, not from the helper
docstring. The matmul helper documentation never mentions math_fidelity or
`fp32_dest_acc_en` — both of which fundamentally affect the precision of the
K-accumulation it performs.

The verifier then flagged that the chosen combination is
**a known-bad combo on Wormhole B0 (issue #38306)**. So the helper currently:
1. Doesn't tell the caller they probably want fp32_dest_acc_en=True for
   non-trivial K.
2. Doesn't warn that `HiFi4 + fp32_dest_acc_en=True` is broken on WH B0.
3. Has no compile-time or runtime check.

The implementer landing in the precise combo the verifier later flagged as
problematic was pure luck — the kernel happens to pass on the device under
test but is on a known fault path.

**Suggested change.**

1. Add a `## Precision` block to the matmul helper docstring near the
   K-accumulation discussion:

   ```
   ── Precision ─────────────────────────────────────────────────────────
   The K-accumulation accumulates `Kt * 32` bf16 multiply-adds per output
   element. Caller-side ComputeConfig recommendations:
     - bf16 inputs, K <= 32  : LoFi or HiFi2 sufficient.
     - bf16 inputs, K > 32   : HiFi2 + fp32_dest_acc_en=True recommended.
                                Without fp32 DEST, K-accumulation rounds
                                to bf16 every step; max-abs error grows
                                ~O(sqrt(K)).
     - HiFi4 + fp32_dest_acc_en=True : KNOWN BAD on Wormhole B0
       (#38306) — silent precision corruption. Use HiFi2 or HiFi3 instead.
     - fp32 inputs           : HiFi4 + fp32_dest_acc_en=True required.
   ```

2. (Optional, defer to a separate refinement) An `ASSERT()`-level guard
   that fires under `--dev` when the host has set HiFi4 + fp32_dest_acc_en
   together on a Wormhole B0 device. Implementing this requires reaching
   compile-time arch detection inside the helper which is heavier than a
   docstring fix; do it only if other call sites repeat the same mistake.

---

## Priority 3 — Pull sister helpers into the agent helper-tables

**Where:** Submodule `tt_metal/third_party/tt_ops_code_gen/`:

- `agents/incremental-planner.md` (helper table around line 43-53)
- `agents/ttnn-implementer.md` (helper table around line 134-143)
- `references/ttnn-cb-memory-fundamentals.md` (include-paths table around
  line 191-194)

**Symptom observed.** The planner found `bias_add_helpers.hpp` only because
it cross-referenced the `bmm_large_block_zm_fused_bias_activation.cpp`
canonical kernel for inspiration. The agent's own helper-table only lists
`matmul_block_helpers.hpp`. If the planner had been steered toward only
`matmul_block`, it would have hand-rolled bias add or treated it as a raw API
phase. The conventions doc at
`ttnn/cpp/ttnn/kernel_lib/agents/llk_helpers_conventions.md` lists all four
matmul-family helpers but is not pulled into the planner's prompt.

**Suggested change.** Append three rows to each of the helper tables in
`incremental-planner.md` and `ttnn-implementer.md`, immediately after the
`matmul_block_helpers.hpp` row:

```
| `bias_add_helpers.hpp` | `add_bias_bcast_rows()` | Broadcast-row bias add over a sub-blocked matmul output. Pairs with `matmul_block` (must use matching `OutputLayout`). |
| `reblock_untilize_helpers.hpp` | `reblock_untilize()` | Output reblocking + untilize after matmul. Use when downstream wants RM stick output per row-group. |
| `transpose_block_helpers.hpp` | `TransposePreKBlock` | Per-K-block in0 transpose; pass as `PreKBlockFn` to `matmul_block`. |
```

Add the matmul include row to `ttnn-cb-memory-fundamentals.md`:

```
| Compute with matmul helper | `#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"` |
| Compute with bias helper   | `#include "ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp"` |
```

**Why.** The planner is told to consult this reference for CB layout. Today
the table implies "if you need matmul, you're outside the helper library."

---

## Priority 4 — Replace the `HwRelu` mention in the implementer agent

**Where:** Submodule `tt_metal/third_party/tt_ops_code_gen/agents/ttnn-implementer.md`,
around line 141.

**Current text:**

> `matmul_block_helpers.hpp` | `matmul_block()` | Sub-blocked tiled matrix
> multiply ... Supports packer L1 accumulation (`packer_l1_acc`), transpose,
> and post-compute activations (`HwRelu`, custom SFPU functors). Prerequisite:
> `compute_kernel_hw_startup()` or `mm_block_init()`.

**Symptom observed.** `HwRelu` is **not a public symbol** in
`matmul_block_helpers.hpp`. The actual API is:

- `LastBlockTarget::OutWithRelu` for hardware RELU
- `PostComputeFn` template parameter (defaults to `NoPostCompute`) for
  arbitrary SFPU functors

The current text would mislead an implementer into searching for a
non-existent type. Our run got lucky because `linear` doesn't fuse activation;
the next op that wants relu fusion will fail or hand-roll.

**Suggested change.**

```
| `matmul_block_helpers.hpp` | `matmul_block()` | Sub-blocked tiled matrix
multiply C = A × B with K-blocking. Supports packer L1 accumulation
(`packer_l1_acc`), B-transpose, hardware RELU on the last K-block
(`LastBlockTarget::OutWithRelu`), and arbitrary SFPU activations via the
`PostComputeFn` template parameter. Prerequisite: `compute_kernel_hw_startup()`
called once before any helper. |
```

(Same row also goes into the planner's table — see Priority 3 above.)

---

## Priority 5 — Document the canonical bias CB lifecycle pattern in `@example`

**Where:** `ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp` — add an
`@example` block just before the `add_bias_bcast_rows` declaration.

**Symptom observed.** The implementer's first cut used raw
`cb_wait_front(cb_bias_tiles, Nt)` / `cb_pop_front(cb_bias_tiles, Nt)` calls.
The verifier replaced those with `bias_buf.wait_front(Nt)` /
`bias_buf.pop_front(Nt)` to "match the canonical
`bmm_large_block_zm_fused_bias_activation.cpp` style." Both work; the latter
keeps the kernel uniformly typed against the `experimental::CircularBuffer`
abstraction (no raw uint32_t sneaking in mid-function).

The bias helper's docstring already explains the caller-owns-wait/pop
contract at lines 93-97, but **does not show what the call site looks like**.
A working `@example` would have anchored the canonical pattern.

**Suggested change.** Insert just before the function declaration around
line 168:

```cpp
/**
 * ...existing prose...
 *
 * @example
 *   // Bias is pushed once by reader; matmul produces partials; bias add reads
 *   // partials + bias, writes biased output. Caller owns bias_buf wait/pop —
 *   // the helper does NOT call wait/pop on bias_buf. Use the buffer object's
 *   // own wait_front/pop_front (not raw cb_wait_front) to keep types uniform.
 *
 *   experimental::CircularBuffer partials_buf(cb_partials);
 *   experimental::CircularBuffer bias_buf(cb_bias);
 *   experimental::CircularBuffer out_buf(cb_out);
 *
 *   bias_buf.wait_front(Nt);
 *   add_bias_bcast_rows<BiasBroadcast::RowBroadcast,
 *                       OutputLayout::SubblockMajor>(
 *       partials_buf, bias_buf, out_buf,
 *       BiasAddShape::of(Mt, Nt, /*subblock_h=*/1, /*subblock_w=*/1));
 *   bias_buf.pop_front(Nt);
 */
```

---

## Priority 6 — Decide what `BiasAddShape::of(...)` argument names mean

**Where:** `ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp` `BiasAddShape`
struct + `of()` factory.

**Symptom observed.** The planner wrote
`BiasAddShape::of(Mt, Nt, 1, 1)` and noted "out_row_width default 0 — only
consulted under RowMajor layout." But it had to derive the argument order by
reading the struct definition; the `of(...)` call site has no parameter
names. Our planner got it right, but I'd expect ~20% of agents to confuse the
argument order with `MatmulBlockShape::of(in0_num_subblocks, in1_num_subblocks,
out_subblock_h, out_subblock_w, in0_block_w, num_k_blocks, batch)` — the
matmul shape factory takes 6-7 positional args in a specific order.

**Suggested change.** Either:

(a) Add a comment header inside the `BiasAddShape::of(...)` factory body
mirroring the one already in `MatmulBlockShape`:

```cpp
static constexpr BiasAddShape of(
    uint32_t in0_num_subblocks,   // = M tiles per output, NOT the matmul Mt
    uint32_t in1_num_subblocks,   // = N tiles per output column
    uint32_t out_subblock_h,      // matches matmul's out_subblock_h
    uint32_t out_subblock_w,      // matches matmul's out_subblock_w
    uint32_t out_row_width = 0)   // RowMajor only; SubblockMajor ignores
```

(b) Add a `@code` block in the docstring showing the call shape next to the
matmul one so callers see the parallel.

The risk of getting this wrong is silent: `add_bias_bcast_rows` would index
the wrong number of tiles in `cb_partials`, producing garbage output that may
still hit a passable PCC on small shapes — exactly the kind of bug the helper
is supposed to prevent.

---

## Priority 7 — Note the `interm_buf` placeholder pattern in the docstring

**Where:** `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp` runtime-params
section (around line 199-208 where `interm_buf` is described).

**Symptom observed.** Both the planner (gotcha #4 in its design) and the
implementer (line 103-105 comment in the kernel) explicitly called out:

> interm_buf is unused when num_k_blocks==1 but the signature requires a
> valid buffer reference — pass output_buf (same buffer type, never read on
> this path; documented gotcha #4).

The current docstring says:

> `interm_buf` Intermediate buffer for K-blocking spill/reload or L1-ACC FIFO.
> When `num_k_blocks == 1` it is never read; pass any non-output buffer
> (typically the same buffer type as the others).

But:
1. The advice "pass any non-output buffer" is contradicted by the implementer
   passing `output_buf` itself, which works because the helper short-circuits
   the read. The current docstring discourages exactly the simplest call
   pattern.
2. Both agents arrived at the same workaround independently — passing
   `out_buf` as the placeholder. That's the de facto canonical pattern; the
   docstring should match.

**Suggested change.** Update the docstring to:

> `interm_buf` Intermediate buffer for K-blocking spill/reload or L1-ACC FIFO.
> When `num_k_blocks == 1` it is never read; passing `out_buf` itself in
> this slot is the canonical pattern (zero-cost placeholder, no extra CB to
> allocate).

This is consistent with what the existing `bmm_large_block_zm.cpp` migrated
kernel does at line 49-52 (passes `interm_buf` as a separate CB only because
it wants to support `num_k_blocks > 1` later), and it removes one decision
the implementer has to make.

---

## Priority 8 — A pure-matmul `@example` for the simplest call

**Where:** `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp` `@example`
section.

**Symptom observed.** The current `@example` blocks all show non-trivial
configurations: SDPA-style with `retain_in0`, FUSE_BIAS with explicit
`in1_per_core_w`, conv2d with `pin_interm_to_captured_base` and
`PreKBlockFn`. The simplest case — "I just want to do a single-core matmul
with default everything" — has no anchor example. Our planner reached the
right minimal call by reading the function signature, but a simpler agent
might over-template.

The first existing example does technically cover this case (`@example //
Simple K=1 non-blocked matmul, defaults everywhere`), but it's labeled
"K=1 non-blocked" which sounds restrictive. In fact `num_k_blocks=1` covers
**any matmul that fits in one K-block**, which is most single-core matmuls
in practice.

**Suggested change.** Rename the first example to be more inviting:

```
@example
  // Single-core matmul with defaults (no transpose, no L1 accumulation,
  // no activation fusion). Valid for any (M, K, N) where the K dimension
  // fits in one block — i.e. the K tiles fit alongside one M and N
  // sub-block in L1. This is the simplest helper call and matches the
  // bmm_large_block_zm.cpp migrated kernel.
  experimental::CircularBuffer in0_buf(cb_in0);
  experimental::CircularBuffer in1_buf(cb_in1);
  experimental::CircularBuffer out_buf(cb_out);
  matmul_block<>(
      in0_buf, in1_buf, out_buf,
      out_buf,  // interm_buf placeholder — unread when num_k_blocks==1
      MatmulBlockShape::of(
          /*in0_num_subblocks=*/Mt,
          /*in1_num_subblocks=*/Nt,
          /*out_subblock_h=*/1,
          /*out_subblock_w=*/1,
          /*in0_block_w=*/Kt,
          /*num_k_blocks=*/1));
```

---

## Lower-priority / observational

These came up but don't clearly motivate a code change:

- **The helper handled `LastBlockTarget::Interm` cleanly** — passing
  `out_buf` in the `out_buf` slot when the last block goes to `interm_buf`.
  The planner explicitly noted in gotcha #2 that `out_buf` is "unused on the
  last/only block since target=Interm." The signature could express this with
  `std::optional` but it's not clearly worth the disruption.

- **No agent ever opened `matmul_reduce_inplace`** — that's expected for a
  linear forcing function. We can't say from this run whether its docstring
  is sufficient. Validate when the SDPA codegen run happens.

- **None of `retain_in0`, `retain_in1`, `OutputLayout::RowMajor`,
  `pin_interm_to_captured_base`, or non-default `InitMode` were exercised.**
  The pipeline correctly defaulted them all. We can't validate their
  documentation from this run.

- **`pin_interm_to_captured_base`'s docstring is the longest in the helper
  and reads well, but it could not be tested by this run.** Future
  conv-style codegen will exercise it; revisit if we see a misuse.

---

## Change-set summary for the applying agent

| # | File | Type of change |
|---|---|---|
| 1 | `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp` | Add "Required includes" line; add Precision section; rename simplest `@example`; document `interm_buf` placeholder pattern; cross-link `BiasAddShape::of()` |
| 2 | `ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp` | Add `@example` showing canonical `bias_buf.wait_front()/pop_front()`; comment `BiasAddShape::of()` arguments |
| 3 | `ttnn/cpp/ttnn/kernel_lib/reblock_untilize_helpers.hpp` | Mirror "Required includes" line |
| 4 | `ttnn/cpp/ttnn/kernel_lib/transpose_block_helpers.hpp` | Mirror "Required includes" line |
| 5 | `tt_metal/third_party/tt_ops_code_gen/agents/incremental-planner.md` | Add 3 sister-helper rows to helper table |
| 6 | `tt_metal/third_party/tt_ops_code_gen/agents/ttnn-implementer.md` | Add 3 sister-helper rows; fix the `HwRelu` text |
| 7 | `tt_metal/third_party/tt_ops_code_gen/references/ttnn-cb-memory-fundamentals.md` | Add matmul/bias helper rows to include-path table |

Items 5-7 require a submodule-side commit + pointer bump in tt-metal. Items
1-4 are tt-metal-side-only.

All eight Priority items can be implemented as docstring + table edits — no
helper API surface changes are proposed by this document.

---

## Reproducing the run

- Branch: `wransom/mm_help` on commit `d8e9602aff9` (post-pipeline).
- Hardware: Wormhole n300 L+R pair, `ARCH_NAME=wormhole_b0`.
- Driver: `python3 .claude/run_op.py eval/prompts/linear.txt`. The
  `--skip-planner` / `--skip-implementer` / `--skip-verifier` flags let you
  re-run any single phase without redoing the others.
- Pre-run setup: `git submodule update --init tt_metal/third_party/tt_ops_code_gen`
  (the submodule is `update = none` by default; either set
  `git config --local submodule.tt_metal/third_party/tt_ops_code_gen.update checkout`
  first or call `--init` manually). Symlinks `.claude` and `eval` at the repo
  root must resolve into the submodule for `run_op.py` to find prompts and
  agent definitions.
- Golden tests: `eval/golden_tests/linear/` was authored fresh for this run.
  Re-running on a clean checkout requires both the prompt file and the golden
  test directory to be present in the submodule's working tree (they are not
  yet committed upstream — they exist only in this run's environment).
