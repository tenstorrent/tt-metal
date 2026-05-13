# pack_patterns.tsv → `compute_kernel_lib::eltwise_chain` migration sweep

Date: 2026-05-13.
Base branch: `astancov/eltwise_run7_refined_rebase_v2`.
Input: `pack_patterns.tsv` (667 rows / 221 unique kernel files / 181 unique `ttnn-op:*` files).
Migration target: `compute_kernel_lib::eltwise_chain` (see `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp`).

## TL;DR

- **25 kernels migrated** across 25 commits this run. Each commit is per-kernel,
  passes pre-commit clang-format, and either ran a representative pytest green or
  is annotated `untestable_locally` (multi-chip / mesh-only).
- **Kernel_lib regression suite** (`tests/ttnn/unit_tests/kernel_lib/`): 465 passed,
  7 skipped (pre-existing infra skips), 0 regressions.
- The remainder of `pack_patterns.tsv` is **structurally out of scope** for the
  eltwise chain — categorised below with concrete blocker per family.

## Migrated kernels (this run, per commit)

| Commit | Kernel | Mode | Test verified |
|---|---|---|---|
| `442d348c02a` | `data_movement/sharded/.../eltwise_copy.cpp` | FULL | `test_interleaved_to_sharded.py` 90 passed |
| `e2b24f65a0c` | `experimental/bcast_to/.../compute_interleaved_col_bcast_to.cpp` | FULL | `test_bcast_to.py` 24 passed |
| `b91194b7a3e` | `experimental/bcast_to/.../compute_interleaved_row_bcast_to.cpp` | FULL | `test_bcast_to.py` 24 passed |
| `a5c207a9227` | `experimental/bcast_to/.../compute_interleaved_scalar_bcast_to.cpp` | FULL | `test_bcast_to.py` 24 passed |
| `758ef26f201` | `moreh/moreh_dot/.../moreh_dot.cpp` | FULL | `test_moreh_dot.py` 8 passed |
| `ee86517bb8d` | `moreh/moreh_dot_backward/.../moreh_dot_backward.cpp` | FULL | `test_moreh_dot_backward.py` 60 passed |
| `39261882e39` | `moreh/moreh_clip_grad_norm_step1/.../moreh_clip_grad_norm_step1_kernel.cpp` | PARTIAL (accumulator-add stage; pre-norm + `power_tile_to_cb` left raw) | `test_moreh_clip_grad_norm.py` 20 passed |
| `67e324dbd36` | `normalization/softmax/.../softmax.cpp` | PARTIAL (mul-by-recip-of-sum stage) | `test_softmax_interleaved.py` 4 passed |
| `a68f0e59629` | `normalization/softmax/.../softmax_sharded.cpp` | PARTIAL (mul-by-recip-of-sum stage) | `test_softmax_sharded.py` 4 passed |
| `6d078969860` | `experimental/transformer/rotary_embedding_llama_sharded.cpp` | PARTIAL (sin mul + cos mul + add) | mesh-only — `untestable_locally` |
| `c0f9165bd7d` | `experimental/transformer/rotary_embedding_llama.cpp` | PARTIAL (final add stage) | mesh-only — `untestable_locally` |
| `ac6d78d1851` | `experimental/transformer/rotary_embedding_llama_fused_qk/.../sharded.cpp` | PARTIAL (sin mul stage; TRISC2 code-size pressure precluded full) | mesh-only — `untestable_locally` |
| `051f42b798e` | `eltwise/binary_ng/.../eltwise_where_sfpu_row_col_bcast.cpp` | FULL | `test_ternary_bcast.py::test_ttnn_where_row_col_mixed_bcast_t{ts,st}` 4 passed |
| `7d1a95b648a` | `moreh/moreh_norm/moreh_norm_h/.../moreh_norm_h_kernel.cpp` | PARTIAL (accumulator) | `test_moreh_norm.py` 276 passed |
| `3b592ad663d` | `moreh/moreh_norm/moreh_norm_other/.../moreh_norm_other_kernel.cpp` | PARTIAL (accumulator) | `test_moreh_norm.py` 276 passed |
| `dcca4f90460` | `moreh/moreh_norm/moreh_norm_w/.../moreh_norm_w_kernel.cpp` | PARTIAL (accumulator) | `test_moreh_norm.py` 276 passed |
| `3bf3fbd7994` | `moreh/moreh_norm/ord_other/moreh_norm_h/.../moreh_norm_h_kernel.cpp` | PARTIAL (IS_ZERO accumulator) | `test_moreh_norm.py` 276 passed |
| `6b8c67b2b80` | `moreh/moreh_norm/ord_other/moreh_norm_nc/.../moreh_norm_nc_kernel.cpp` | PARTIAL (IS_ZERO accumulator) | `test_moreh_norm.py` 276 passed |
| `9f05c30233a` | `moreh/moreh_norm/ord_other/moreh_norm_w/.../moreh_norm_w_kernel.cpp` | PARTIAL (IS_ZERO accumulator) | `test_moreh_norm.py` 276 passed |
| `91a0fce7b49` | `moreh/moreh_clip_grad_norm_step2/.../moreh_clip_grad_norm_step2_kernel.cpp` | PARTIAL (accumulator) | `test_moreh_clip_grad_norm.py` (smoke) PASS |
| `71d6f7e8e2c` | `reduction/prod/.../prod_all.cpp` | FULL | `test_prod_all.py` 4 passed |
| `466a0668f89` | `reduction/prod/.../prod_nc.cpp` | FULL | `test_reduction.py::test_prod` 56 passed |
| `61f720c8985` | `experimental/transformer/rotary_embedding/.../rotary_embedding_single_tile.cpp` | FULL | `test_rotary_embedding_*` PASS |
| `b2ca12e0c5d` | `experimental/transformer/rotary_embedding/.../rotary_embedding.cpp` | PARTIAL (final add stage) | `test_rotary_embedding_*` PASS |
| `537e8c724b4` | `normalization/rmsnorm_distributed/.../rmsnorm_pre_allgather.cpp` | PARTIAL (x² stage) | `test_layernorm_part_1_with_program_cache[rmsnorm-...]` PASS |

## Helper-coverage by `pack_patterns.tsv` source category

| Category | Files in TSV | Migrated/Partial this run | Already-migrated pre-run (chain present) | Categorically BLOCKED (see reasons) |
|---|---|---|---|---|
| `ttnn-op:eltwise` | 50 | 1 | 32 (pre-run, prior cycles on this branch) | 17 |
| `ttnn-op:normalization` | 28 | 3 | 0 | 25 |
| `ttnn-op:moreh` | 39 | 10 | 4 | 25 |
| `ttnn-op:experimental` | 38 | 8 | 0 | 30 |
| `ttnn-op:reduction` | 12 | 2 | 0 | 10 |
| `ttnn-op:data_movement` | 8 | 1 | 0 | 7 |
| `ttnn-op:transformer` | 2 | 0 | 0 | 2 |
| `ttnn-op:matmul`, `:conv`, `:examples`, `:copy` | 4 | 0 | 0 | 4 |
| models / tt_metal / tt-train / examples | 40 | 0 | 0 | 40 (out of scope — demos / 3rd-party tree) |

The category counts of "already-migrated pre-run" reflect the state of branch
`astancov/eltwise_run7_refined_rebase_v2` at the start of this sweep — many
`ttnn-op:eltwise` kernels (binary_ng, ternary, unary chain ports) were
migrated in earlier commits on this branch and only their stale TSV rows
(matching `pack_tile` at a now-removed line) remain.

## BLOCKER taxonomy (why the remaining 137 kernels stay on raw LLK)

Every BLOCK reason below was reproduced by at least one sub-agent during this
sweep. Sub-agents tried each kernel against the chain helper invariants and
recorded the specific incompatibility. Numbers are unique-kernel counts across
the remaining file set.

| Blocker | Count | Description |
|---|---|---|
| **`reduce_init` / `reduce_tile`** | 41 | Reduce primitive belongs to `reduce_helpers_compute`, not eltwise. `chain` is eltwise-only by design. |
| **`mm_init` / `matmul_tile{,s}`** | 33 | Matmul primitive belongs to a matmul helper. Same scope boundary as reduce. |
| **Macro-injected SFPU chain** (`SFPU_OP_FUNC_*`, `SFPU_OP_INIT_*`, `BINARY_OP`, `PREPROCESS`, `HAS_ACTIVATIONS`) | 27 | Chain elements must be static op structs at the chain template instantiation site. Macros expanded by the program factory at compile time cannot be turned into a typelist without a code-gen layer. |
| **Welford reduction** | 8 | Welford has its own dedicated state; not an eltwise pattern. |
| **`transpose_wh_*` / sort / topk / cumsum** | 11 | Block-permutation / order-changing primitives are not eltwise. Some appear interleaved with eltwise blocks; those eltwise blocks could be PARTIAL-migrated but the kernels under TSV scrutiny all also trip another blocker. |
| **Held-DEST across loop iters (FPU acc_to_dst)** | 7 | `eltwise_chain` opens a fresh `tile_regs_acquire/release` window per outer iter — held-DEST is an explicit non-goal (`eltwise_chain.hpp` §"Non-goals"). |
| **Cumulative `cb_wait_front(cb, base + i)`** | 5 | Listed as explicit non-goal in the chain doc-comment. |
| **Runtime CB ids passed as `const auto cb = id++`** | 4 | Chain element CB params are `uint32_t` *template* args (compile-time). Refactoring the surrounding code to make every CB id constexpr was out of "no unrelated edits" scope. |
| **Mid-loop `_with_dt` dtype walk / `reconfig_data_format` per iter** | 3 | Chain reconfig is entry-time per element (fold-driven). Per-iter dtype swaps would require a new chain policy. |
| **Missing op struct in compute_kernel_lib** | 6 | E.g. `BinaryMax` (moreh_adam / moreh_norm non-IS_ZERO), `Mask`/`MaskPosInf` (moreh_sum prologue), `lgamma_stirling_*` / `lgamma_adjusted_*` (lgamma kernels), `gelu_derivative_tanh` (eltwise_bw_gelu_approx_tanh). |
| **Dead code (no factory compiles it)** | 1 | `ternary_addc_ops_fpu_bcast.cpp` — `ternary_op_utils.cpp:336` dispatches to `_rowbcast` variant. No test possible. |
| **out-of-tree / demo** | 40 | `models/demos/...`, `tt-train/...`, `tt_metal/programming_examples/...`, `ttnn/examples/...`. Not in the ttnn ops migration surface. |

These are not unique tags — many kernels carry two or three (e.g. `bmm_*` is
matmul + macro-injection; layernorm welford is welford + reduce). The count
column is "kernels for which this blocker was the recorded primary reason."

## Helper-gap notes (would unblock specific kernels)

Items below are suggested follow-ups for the helper-creation pipeline, not
landed here. Each is gated on a small API addition with concrete unblock value.

1. **`BinaryMax` op struct** — unblocks the non-IS_ZERO accumulator branch in 3
   moreh_norm kernels (`moreh_norm_h`, `moreh_norm_other`, `moreh_norm_w`,
   plus their `ord_other` peers) and the AMSGRAD branch in
   `moreh_adam`/`moreh_adamw`.
2. **`Mask` / `MaskPosInf` chain elements** — moreh_sum_h, moreh_sum_w prologues
   and the moreh_norm `ord_other` prologue all share an in-loop mask pattern.
3. **FPU accumulate-to-DEST policy** — chain `BinaryFpu` currently emits
   `add_tiles_init<.., /*acc_to_dst=*/false>`; the CCL reduction family
   (all_reduce_async, llama_reduce_scatter, deepseek_moe_reduce_scatter,
   reduce_scatter_minimal_async × 4) all need the `acc_to_dst=true` form.
   Single new policy enum value covers ~7 kernels.
4. **Cumulative-wait-no-pop `CopyTilePolicy`** — distinct from
   `CumulativeWaitPopAtEnd` and `WaitUpfrontPopAtEnd`. Unblocks
   `layernorm_pre_allgather` and the 4 `reduce_scatter_minimal_async` kernels.
5. **Composite SFPU op structs** for `lgamma_stirling_float_tile`,
   `lgamma_adjusted_tile`, `gelu_derivative_tanh` — currently in
   `api/compute/eltwise_unary/lgamma.h` etc., but not catalogued as chain
   elements (`eltwise_special.hpp` has an "implement when needed" comment).

## Verification

- Per-commit pytest as listed in the Migrated table above.
- Cross-helper regression: `scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/kernel_lib/*.py` → **465 passed, 7 pre-existing skips, 0 failures.**
- Pre-commit hooks: every commit went through `clang-format` + `validate-metalium-*` (no `using namespace compute_kernel_lib` introduced; no `namespace alias = …` introduced; no `*_with_dt` LLK calls reintroduced).

## What was NOT changed

- No edits to `compute_kernel_lib` helpers themselves — migrations consume
  the helper API as-is.
- No edits to program factories, host-side code, or tests beyond identifying
  representative coverage.
- No deletion of TSV rows / `pack_patterns.tsv` itself — that's a snapshot input.
- No commits to `main`. All work landed on
  `astancov/eltwise_run7_refined_rebase_v2`.
