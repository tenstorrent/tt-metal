# `matmul_block` migration status

Inventory of every `matmul_block` invocation in TTNN compute kernels and
their migration status against `compute_kernel_lib::matmul_block` (and
sister helpers in `ttnn/cpp/ttnn/kernel_lib/`). Updated each iteration of
the wransom/mm_help branch.

**As of 2026-05-07, every TTNN compute kernel calling `ckernel::matmul_block`
has been migrated** onto one of two orthogonal public helpers:
`compute_kernel_lib::matmul_block` (the K-loop helper that owns iteration
and pack lifecycle) or `compute_kernel_lib::matmul_kloop_pack` (a unified
entry point for the four ring-aware MoE / DeepSeek MLA kernels: it folds
the DST scope, the segmented K-loop body, an optional post-K compute body,
and the pack body into a single call, with three overloads picked by the
kernel's pack shape). The DST scope itself
(`detail::matmul_subblock_pack`) and its token-injected FMA primitive
(`detail::MatmulSubblockStep`) are internal building blocks consumed only
by `matmul_kloop_pack`; no kernel calls them directly. See the "Two
orthogonal helper abstractions" section below for the design split. The
"Unmigrated" section is empty; per-phase detail of what each helper
extension delivered and which kernels it absorbed is preserved below as a
record of the project arc.

## Migrated

| Kernel | Helpers used |
|---|---|
| `matmul/device/kernels/compute/bmm_large_block_zm.cpp` | `matmul_block` (batch loop in helper) |
| `matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp` | `matmul_block` + `bias_add` + `reblock_untilize` (in0_transpose handled by a kernel-local lambda PreKBlockFn) |
| `conv/conv2d/device/kernels/conv_bmm_tilize.cpp` | `matmul_block` (`InitMode::None`, `pin_interm_to_captured_base`) + `bias_add` + `reblock_untilize` + `tilize` |
| `experimental/conv3d/device/kernels/compute.cpp` | `matmul_block` (`In1Policy::WaitAndRetainOnLastBlock`) |
| `transformer/sdpa/device/kernels/compute/compute_common.hpp` | `matmul_block` (`InitMode::None`, `In0Policy::WaitAndRetainOnLastBlock`) — wrapper used by every SDPA prefill / decode / ring-joint variant |
| `transformer/sdpa/device/kernels/compute/compute_streaming.hpp` | `matmul_block` (`InitMode::None`, PostFn=RecipPost) for the per-row 1×1 sum × col_identity → recip → 1/sum fusion |
| `experimental/deepseek/mla/matmul_wo/device/kernels/compute.cpp` | `matmul_kloop_pack` Form 1 (in0/in1/out + dst_count); single-line replacement per outer iter |
| `experimental/deepseek/moe/moe_gate_mm/device/kernels/compute.cpp` | `matmul_kloop_pack` Form 2 (send-core: K-loop + custom multi-CB pack, partial-last-block) and Form 3 (non-send: K-loop + post-K SFPU/bias on MATH thread + custom multi-CB pack, `last_block_no_pop`) |
| `experimental/ccl/moe_compute/device/kernels/compute.cpp` | `matmul_kloop_pack` Form 2 in both cycles. W0/W1 dispatches `KStepWithBias` / `KStepDefault` via `if constexpr (has_bias)`; pack body is STALL_CFG semwait + PACK-thread SFPU. W2 uses `KStepWithRing` with monotonic ring step callback; pack body is `pack_untilize_dest` |
| `experimental/ccl/moe_gpt/device/kernels/compute.cpp` | `matmul_kloop_pack` Form 2 across `TILIZE_FUSED` + non-fused branches. Same shape as moe_compute except the W2 ring step callback wraps `(step_idx % 6) * tiles_per_step` for the 6-buffer cyclic in2 layout |
| `experimental/ccl/llama_all_gather_matmul_async/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp` | `matmul_block` with `PreKBlockFn` (ring rd_ptr capture + relu-on-untilize), `PostKBlockFn` (rd_ptr commit), `In1BaseOffsetFn` (Case 4 offset shift), `In1Policy` ternary on `in1_is_dram` (`WaitAndPopPerKBlock` / `NoWaitNoPop`), `OutWithUntilize` / `OutWithRelu` / `Out` (Phase A migration; single helper call covers all four `(ENABLE_GLOBAL_CB, in1_is_dram)` cases) |
| `matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp` | `matmul_block` with `PreKBlockFn` (ring rd_ptr capture + block-0 self-prime + relu-on-untilize), `PostKBlockFn` (rd_ptr commit), `KBlockInnerDimFn` (per-K-block unpadded inner-dim from runtime ring-shard-width array), `In0SourceFn` (block 0 → in0_cb_id, blocks 1..N → in2_cb_id), `In1BaseOffsetFn` (Case 4 offset shift, `(ring_idx + block) % ring_size`), `In1Policy` ternary on `in1_is_dram` (`WaitAndPopPerKBlock` / `NoWaitNoPop`), `OutWithUntilize` / `OutWithRelu` / `Out` (Phase C migration; uses every K-loop helper extension shipped) |
| `experimental/ccl/all_gather_minimal_matmul_async/device/kernels/compute.cpp` | `matmul_block` with `LastBlockTarget::Interm` + `OutputLayout::RowMajor` + `packer_l1_acc=true` + `caller_owns_pack_target=true`; outer kernel pre-reserves the partials CB once and pushes once after the K-loop, helper packs every K-block at absolute row-major offsets with L1_ACC accumulating into the same L1 region. `In0Policy` is selected at runtime per outer (m,n) iter (`WaitAndRetainOnLastBlock` when reusing across the next n iter, `WaitAndPopPerKBlock` on the last n iter). Bias / addcmul / copy-with-activation phase outside helper scope (eltwise downstream). |

## Two orthogonal helper abstractions

The migrations land on one of two helper shapes, picked by the kernel's
loop structure — not by the op family. Both live in
`ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp`:

| Helper | Owns | Caller drives |
|---|---|---|
| `matmul_block` | K-loop iteration, per-K pack lifecycle, `mm_block_init` dispatch | Outer iter, init-mode choice via templates, optional callbacks for ring-aware extensions |
| `matmul_kloop_pack` | DST scope + segmented K-loop + (optional) post-K compute body + pack body, all in one call | Per-step functor (`KStepDefault` / `KStepWithBias` / `KStepWithRing`), pack body lambda, optional post-K compute body |

`matmul_block` fits kernels whose K-loop has a regular shape — every
K-block reads one in1 CB lifecycle, one in0 CB, and produces one output
write. The 11 kernels in the upper table consume it.

The four ring-aware MoE / DeepSeek MLA kernels (`mla/matmul_wo`,
`moe_gate_mm`, `ccl/moe_compute`, `ccl/moe_gpt`) have a fundamentally
different shape: caller-driven K-loop with multiple in1 CB lifecycles per
DST cycle, bias-as-matmul step interleaving, custom semwait variants,
`pack_untilize_dest` packs, and out-of-order multi-CB packs.
Extending `matmul_block` to fit them would mean teaching it to give up
loop control to the caller — at which point it stops being the K-loop
helper. `matmul_kloop_pack` complements it: it bundles the segmented
K-loop pattern shared by all four kernels (cb_wait/pop on in1 + inner
stride loop), the DST acquire→commit→pack→release scope, an optional
post-K compute body, and the kernel's pack body into a single call. The
per-step FMA logic lives in mutable functors (`KStepDefault` for the
simple case, `KStepWithBias` for bias-as-matmul interleaving,
`KStepWithRing<HasBias, Buf, RingStepFn>` for ring-aware in0 cycling) so
each kernel's body collapses to a functor declaration plus one or two
pack lambdas. The DST scope helper itself
(`detail::matmul_subblock_pack`) and the FMA primitive
(`detail::MatmulSubblockStep`) are internal building blocks consumed
only by `matmul_kloop_pack`.

## Unmigrated

None. All TTNN compute kernels invoking `ckernel::matmul_block` have been
moved onto `compute_kernel_lib::matmul_block` (the K-loop helper) or
`compute_kernel_lib::matmul_kloop_pack` (the unified DST + K-loop +
pack helper, three forms). The production-tree
`matmul/.../bmm_large_block_zm_fused_bias_activation_gathered.cpp` and the
experimental-tree `all_gather_minimal_matmul_async/.../compute.cpp` were
the last K-loop migrations; both shipped 2026-05-06 alongside the
`caller_owns_pack_target` extension that absorbed the all_gather_minimal
Class-B pack lifecycle. The MLA / MoE family migrated onto
`matmul_subblock_pack` on 2026-05-06; on 2026-05-07 the API was reshaped
so the four ring kernels go through `matmul_kloop_pack` exclusively (no
kernel sees `matmul_subblock_pack` directly — it lives in `detail::`).

`experimental/matmul/group_attn_matmul/device/kernels/compute/transformer_group_attn_matmul.cpp`
is a comment-only reference — no live `matmul_block` call — and is omitted
from the migrated table for that reason.

## Helper extensions shipped

All five extensions originally forecast to "unlock the unmigrated work"
shipped — three as new template parameters/callbacks, two as no-ops
because the existing helper already covered them. Two further extensions
surfaced during the Phase A llama migration. One last extension surfaced
during the Class-B `all_gather_minimal_matmul_async` migration. The full
list of extensions added to the K-loop helper this project, in approximate
payoff order:

1. ✅ **`KBlockInnerDimFn`** (Phase B) — per-K-block inner-dim step count.
   Replaces the original "Variable per-K-block inner-dim" forecast.
   Unblocked the ring-aware matmuls (production gathered).
2. ✅ **`PreKBlockFn` + `PostKBlockFn`** (Phase A) — symmetric
   pre/post UNPACK-friendly callbacks fired around the helper's per-K-block
   wait/pop. `PostKBlockFn` was the new one this project; `PreKBlockFn`
   already existed. Replaces the "Per-K-block rd_ptr hooks" forecast.
3. ✅ **`matmul_subblock_pack` DST scope** with **token-injected
   `MatmulSubblockStep` FMA primitive** (Phase D) — the primary helper
   for kernels whose custom outer loops don't fit the K-loop abstraction.
   `matmul_subblock_pack` owns the DST acquire→commit→pack→release scope
   and injects a stateless `step` token into the body lambda; calling
   `step(...)` fires one `ckernel::matmul_block` FMA with helper hygiene
   (DST_AUTO_LIMIT assert + SKIP_COMPUTE guard + buf abstraction). The
   FMA primitive itself lives in the detail namespace as
   `detail::MatmulSubblockStep` and has no public free-function entry —
   it can only be reached via the injected token, which forces every FMA
   call to live inside a DST scope. Two overloads on
   `matmul_subblock_pack`: Form 1 `(out_buf, dst_count, dst_body)` for
   the simple `pack_tile_block` case (mla_wo); Form 2
   `(dst_body, pack_body)` for kernels with custom wait variants
   (STALL_CFG semwait), out-of-order multi-CB packs, or
   `pack_untilize_dest` sequences (the three MoE kernels). Replaces the
   "Fine-grained primitive helper" forecast — the original public
   `matmul_block_step` free function (a one-line wrapper around
   `ckernel::matmul_block`) was replaced by the token-injection design
   so the FMA primitive cannot be called outside a DST scope.
4. ✅ **`LastBlockTarget::OutWithUntilize`** + `untilize_block_ct_dim`
   (Phase A) — last-K-block pack through `pack_untilize_dest`. Used by
   the gathered variants' `untilize_out` path.
5. ✅ **`In0SourceFn`** (Phase C) — per-K-block in0 CB swap. Unblocked
   the production-gathered `block == 0 ? in0_cb : in2_cb` pattern.
6. ✅ **`In1Policy::NoWaitNoPop`** (surfaced during Phase A) — enum value
   that skips the helper's per-K-block in1 `wait_front` / `pop_front` for
   callers managing in1 externally (global-CB receivers, pre-populated
   L1-sharded CBs).
7. ✅ **`In1BaseOffsetFn`** (surfaced during Phase A) — per-K-block in1
   starting tile offset, for ring-aware kernels that rotate between ring
   positions via offset arithmetic instead of rd_ptr advance.
8. ✅ **`caller_owns_pack_target`** (surfaced during Class B) — bool
   that skips ALL helper-side `reserve_back` / `push_back` / inter-block
   drain on the pack target for callers that pre-reserve the pack target
   once around the entire K-loop and pack via L1_ACC into the same L1
   region (single-buffer accumulation).

Two forecast bullets did **not** ship as new APIs — they were already
covered:

- **"Reload reconfig hook"** — the helper already issued
  `copy_tile_to_dst_init_short_with_dt` + `copy_block_matmul_partials` +
  `mm_block_init_short_with_dt` inline as part of its K-loop reload. The
  gathered variants' `_with_dt` reload dance maps onto this directly.
- **"`InitMode::ShortWithDt` / `ShortWithBothDt`"** — the gathered's only
  `_with_dt` use is inside the reload (already covered above); conv2d's
  `_with_both_dt` lives in its `PreKBlockFn`.

The original Class-B extension forecast ("`LastBlockTarget::SingleBufferAccum`,
partial-shape support on `MatmulBlockShape`, outer-loop in0 reuse distinct
from the per-call `In0Policy`") collapsed at execution time into the single
`caller_owns_pack_target` bool above, plus two existing surfaces — partial
shape via `MatmulBlockShape`'s runtime fields + `out_row_width`, and
outer-loop in0 reuse via the existing `In0Policy::WaitAndRetainOnLastBlock`
enum value selected at runtime per outer `(m, n)` iter.

## Phase summary

The project followed an A → B → D → C → Class-B order. Each phase
was independently releasable; helpers stayed backwards-compatible because
every new template parameter took a no-op default. Helper-side changes
folded into the `helper implementation` commit on the four-commit stack
(`helper implementation` → `tests addition` →
`migration / kernel implementation` → `status`); migrations into the
third commit; doc updates into the fourth.

Per-phase detail of what shipped (helper changes, tests, migrations, and
their validation gates) is in the per-phase sections below.

### Phase A — `PostKBlockFn` + `LastBlockTarget::OutWithUntilize`

**Helper changes** ✅ shipped 2026-05-06 (`ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.{hpp,inl}`):

- `NoPostKBlock` struct + `PostKBlockFn` template parameter
  (default `NoPostKBlock`), symmetric to the existing `PreKBlockFn`.
  Signature `(uint32_t block, uint32_t num_k_blocks, bool last_out)`.
  Fires inside the K-loop body **after** the in0/in1 `pop_front` and
  after the `packer_l1_acc` drain.
- `LastBlockTarget::OutWithUntilize` enumerator. On the last
  K-block, dispatches `pack_untilize_dest_init<untilize_block_ct_dim>(pack_target_id)`
  before `tile_regs_commit`, replaces the `pack_tile_block` call with
  `pack_untilize_dest<untilize_block_ct_dim>(pack_target_id)`, and emits
  `pack_untilize_uninit(pack_target_id)` after `tile_regs_release`.
  `untilize_block_ct_dim` is a new compile-time template parameter
  (default 0; required > 0 when this enum value is selected).
- **Scope deviation from the original plan**: only `SubblockMajor` is
  covered. The original plan ("Cover both `SubblockMajor` and `RowMajor`
  paths") would have routed the row-major caller through
  `pack_subblock_row_major_strided` with per-row pack_untilize_dest, but
  pack_untilize_dest's compile-time `block_ct_dim` plus its DST-offset
  semantics don't compose with the row-major path's per-row fifo_wr_ptr
  manipulation. Row-major callers route via `Interm` +
  `reblock_and_untilize` (the existing pattern in
  `bmm_large_block_zm_fused_bias_activation.cpp`). A `static_assert`
  enforces `OutWithUntilize ⇒ layout == SubblockMajor`.

**Tests** ✅ shipped 2026-05-06
(`tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_tile_pack_row_major.py`):

- New `test_mcast_1d_mcast_in0_tile_pack_row_major_untilize_out`
  parametrization (subblk_1x4, subblk_1x2) that compiles
  `bmm_large_block_zm_fused_bias_activation.cpp` with
  `untilize_out=True` + `tile_pack_row_major=True` and PCC-checks
  against the un-untilized reference. This exercises the helper's
  `Interm` + `RowMajor` last-block pack chained into
  Phase-3 `reblock_and_untilize`. Serves as the regression guard for
  the `pack_untilize.h` include added to the helper's `.inl`.
- `OutWithUntilize` itself is exercised by the migrated llama and
  production-gathered kernels (build-validated; end-to-end gates on
  TG / T3K hardware listed in the per-phase sections below).

**Migration** ✅ shipped 2026-05-06
(`ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp`):

- Single `compute_kernel_lib::matmul_block` call replaces the entire
  `for block` body. Outer `for b` preserves fabric sync (sync_cb2 wait
  + sync_cb push/reset on ENABLE_GLOBAL_CB), CB-array indexing
  (mm_out_cb_ids[b], mm_partials_cb_ids[b]), and the batch-start
  `pack_reconfig_data_format` + `llk_pack_relu_config(NO_RELU)` resets.
  No legacy K-loop branch — all four `(ENABLE_GLOBAL_CB, in1_is_dram)`
  combinations route through the same helper invocation, with case
  selection driven by template parameters and functor compile-time
  switches.
- `RingPreKBlock` carries (a) `calculate_next_block_index_and_update_rd_ptr`
  for `ENABLE_GLOBAL_CB` (UNPACK-only; computes the next ring rd_ptr
  without applying it), and (b) `llk_pack_relu_config(ZERO_RELU)` on the
  last K-block when `PACK_RELU && untilize_out` — the helper's
  `OutWithUntilize` target doesn't auto-enable relu the way `OutWithRelu`
  does, so the PreKBlockFn picks it up there. Both behaviors are gated
  by compile-time template bools so non-ring / non-relu callers compile
  to no-ops.
- `RingPostKBlock` commits the next ring rd_ptr via
  `update_local_cb_rd_ptr(in1_cb_id, next_in1_rd_ptr_addr)` (UNPACK-only;
  gated on `ENABLE_GLOBAL_CB`).
- `RingIn1BaseOffsetFn` returns `in1_block_num_tiles * curr_ring_idx` on
  the `!ENABLE_GLOBAL_CB && !in1_is_dram` path (Case 4) and 0 elsewhere,
  threaded through the new `In1BaseOffsetFn` helper extension. This is
  the per-K-block in1 base offset shift the original Phase A plan
  flagged as needing a separate extension.
- `In1Policy` is `WaitAndPopPerKBlock` when `in1_is_dram` and `NoWaitNoPop`
  otherwise (compile-time): the helper does per-K-block wait_front/pop_front
  when in1 is DRAM-backed, and skips both when in1 is a global-CB receiver
  or a pre-populated L1 shard.
- `LastBlockTarget` selection: `OutWithUntilize` if `untilize_out`,
  else `OutWithRelu` if `PACK_RELU`, else `Out`. FUSE_BIAS is host-side
  dead code in this kernel — never set in `llama_1d_mm_fusion.cpp` —
  so the original plan's FUSE_BIAS branch is moot.
- The local `reload_from_cb_to_dst` is dropped — the helper does its
  own `copy_tile_to_dst_init_short_with_dt` +
  `copy_block_matmul_partials` + `mm_block_init_short_with_dt` reload
  inline as part of its K-loop.

**Validation:** end-to-end correctness requires Galaxy / T3K hardware
that this op runs on; ttsim cannot reproduce the fabric / multi-chip
path (`docs/simulator.md`). The validation gate before publishing the
branch is at least one `tests/ttnn/unit_tests/operations/ccl/test_llama_all_gather_matmul.py`
parametrization on TG. Build-only validation passed in the migration
session; default-callback regressions covered via the matmul tile-pack
nightly (34/34), SDPA non-causal (1/1), and auto-tune (29/29) sweeps,
which transitively re-exercise every other matmul_block consumer.

### Phase B — `KBlockInnerDimFn`

**Helper changes** ✅ shipped 2026-05-06 (`ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.{hpp,inl}`):

- `NoKBlockInnerDimFn` struct: `(uint32_t block, uint32_t block_w) → uint32_t`,
  default returns `block_w` so existing callers' loop semantics are
  unchanged.
- `KBlockInnerDimFn` template parameter (default `NoKBlockInnerDimFn`) +
  runtime parameter `k_block_inner_dim` — appended to the helper's
  template + function signatures (slot ordering kept stable for
  existing callers; both have defaults so existing call sites need no
  edits).
- `.inl`: hoisted `const uint32_t inner_steps = k_block_inner_dim(block, block_w);`
  before the in0/in1 wait calls (so the value is computed once per
  K-block and used by every subblock's inner FMA loop), and replaced
  the loop bound `inner_dim < block_w` with `inner_dim < inner_steps`.
  The inner LLK call's `kt_dim` argument stays `block_w` — that
  parameter is the in1 row stride in L1, NOT the FMA step count, and
  the LLK still strides through in1's tile geometry as if the K-tile
  span were full-width.

**Tests** — backwards-compat regression coverage shipped via the existing
`test_matmul_tile_pack_row_major.py` sweep (34/34 pass) and the SDPA
prefill non-causal test. The default-callback path is the only
production path today; the unpadded-width path will land its own
parametrization with the Phase C migration when a real factory emits a
non-default callback. Adding a parametrization that returns
`block_w - 1` ahead of that migration would require a new test-only
factory or a custom op that exposes the callback as a runtime arg
(neither of which exists today and writing one for a single test isn't
worth the maintenance cost).

**Migration**: none directly in this phase — the helper extension is
prerequisite-only. Phase C's production-gathered migration (shipped same
day) is the first caller; it threads
`unpadded_in0_shard_widths_in_tiles[(ring_idx + block) % ring_size]`
through `KBlockInnerDimFn` to shrink the FMA loop on K-blocks whose
unpadded width is < `in0_block_w`.

### Phase D — `matmul_subblock_pack` DST scope + `matmul_segmented_kloop` K-loop primitive

**Helper changes** ✅ shipped 2026-05-06 / 2026-05-07 (`ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.{hpp,inl}`):

Three layers, picked by the kernel's loop / pack shape:

1. **`matmul_subblock_pack`** — DST scope helper. Owns
   acquire → commit → pack → release; injects a token-bound FMA primitive
   into the body lambda. Two overloads (Form 1 simple pack /
   Form 2 custom pack). Use directly when the kernel's loop body is
   irregular (custom semwait, mid-DST SFPU, multi-CB packs).
2. **`matmul_segmented_kloop`** — segmented K-loop primitive. Drives the
   shared shape across all four ring-aware kernels: outer loop of
   `num_blocks` iterations with `cb_wait_front` / `cb_pop_front` per
   block on in1, inner stride loop calling a per-step functor at
   each FMA position. Must be invoked from inside a `matmul_subblock_pack`
   body — that's how the `step` token comes into scope.
3. **`matmul_kloop_pack`** — convenience helper combining (1) Form 1 +
   (2) + a default per-step functor. Single-call replacement for the
   simplest use case (mla_wo).

The split lets the two complex MoE kernels keep their custom pack /
SFPU bodies (Form 2) while having the K-loop body collapse to a single
`matmul_segmented_kloop` call plus a per-step functor declaration.

- **DST scope helper** (two overloads):
  ```cpp
  // Form 1: simple pack (helper handles tile_regs_wait + pack_tile_block)
  template <typename DstBody, typename Buf>
  ALWI void matmul_subblock_pack(Buf& out_buf, uint32_t dst_count, DstBody dst_body);

  // Form 2: custom pack (caller drives pack_body — incl. tile_regs_wait variants,
  //                       multi-CB packs, pack_untilize_dest sequences, mid-DST SFPU)
  template <typename DstBody, typename PackBody>
  ALWI void matmul_subblock_pack(DstBody dst_body, PackBody pack_body);
  ```
  - Form 1 implementation: `tile_regs_acquire → dst_body(step) →
    tile_regs_commit → tile_regs_wait → cb_reserve_back → pack_tile_block
    → cb_push_back → tile_regs_release`. ASSERTs
    `dst_count <= DEST_AUTO_LIMIT`. The `dst_body` lambda takes one
    parameter (the `step` token).
  - Form 2 implementation: `tile_regs_acquire → dst_body(step) →
    tile_regs_commit → pack_body() → tile_regs_release`. The `pack_body`
    must call `tile_regs_wait()` (or a custom variant — see moe_compute's
    STALL_CFG semwait) at its start before any pack work. The `pack_body`
    does not receive `step` — it has no FMAs to fire.

- **Detail-only FMA primitive** injected as `step`:
  ```cpp
  namespace detail {
  struct MatmulSubblockStep {
      template <matmul_config::InitMode init_mode = matmul_config::InitMode::None,
                typename Buf>
      ALWI void operator()(Buf& in0_buf, Buf& in1_buf,
                            uint32_t in0_index, uint32_t in1_index, uint32_t dst_index,
                            bool transpose, uint32_t ct_dim, uint32_t rt_dim,
                            uint32_t kt_dim) const;
  };
  }
  ```
  Token injection: there is no public `matmul_block_step` free function —
  the only way to reach an FMA is through the injected token, so a
  future kernel cannot fire `ckernel::matmul_block` via the helper
  without entering a DST scope. `InitMode::Full` is `static_assert`-
  rejected (no `out_cb_id` to bind on a single-FMA primitive).

- **Segmented K-loop primitive**:
  ```cpp
  struct SegmentedKLoopShape {
      uint32_t num_blocks;
      uint32_t tiles_per_block;
      uint32_t ct_dim;
      uint32_t rt_dim = 1;
      uint32_t kt_dim = 1;
      uint32_t last_block_tiles = 0;     // optional partial-last-block bound
      bool last_block_no_pop = false;    // skip cb_pop on partial-last
  };

  template <typename Buf, typename KStepFn>
  ALWI void matmul_segmented_kloop(
      detail::MatmulSubblockStep& step,
      Buf& in1_cb_buf,
      SegmentedKLoopShape shape,
      KStepFn& k_step);
  ```
  - Drives `for (block = 0; block < num_blocks) { wait_front; for (k = 0;
    k < tiles_per_block; k += ct_dim) { k_step(...); } pop_front; }`.
    Optional partial last block (`last_block_tiles > 0`) waits for a full
    `tiles_per_block` from the producer but iterates only the populated
    prefix; `last_block_no_pop` lets the kernel keep that block fronted
    after the helper returns (e.g. for a post-K-loop bias `copy_tile`).
  - The `k_step` functor is taken by lvalue reference so its mutable
    state (in0_index cursor, k_tracker, ring counters) persists across
    stride iterations within a single matmul_subblock_pack invocation.

- **Per-step functors**:
  ```cpp
  template <typename Buf>
  struct KStepDefault {
      Buf& in0_buf; Buf& in1_buf;
      uint32_t in0_index = 0;
      bool transpose = false;
      ALWI void operator()(detail::MatmulSubblockStep&, SegmentedKLoopShape,
                            uint32_t /*block*/, uint32_t k);
  };

  template <typename Buf>
  struct KStepWithBias {
      Buf& in0_buf; Buf& in1_buf; Buf& bias_buf;
      uint32_t in0_index = 0;
      uint32_t k_tracker = 0;
      uint32_t bias_at;          // K-tile count where bias FMA fires
      bool transpose = false;
      ALWI void operator()(detail::MatmulSubblockStep&, SegmentedKLoopShape,
                            uint32_t, uint32_t k);
  };

  struct RingStepResult {
      uint32_t in0_index;        // where the next ring slot starts in in0
      uint32_t tiles_remaining;  // tile budget for that slot
  };

  template <bool HasBias, typename Buf, typename RingStepFn>
  struct KStepWithRing { /* fields incl. ring_cb_buf, ring_step_fn */ };
  ```
  - **`KStepDefault`** — single in0/in1 pair, auto-incrementing
    in0_index. Used by mla_wo, moe_gate_mm (both branches),
    moe_compute W0/W1 when has_bias=false.
  - **`KStepWithBias`** — fires regular FMAs up to `bias_at` K-tiles,
    then ONE bias FMA (`bias_buf × in1_buf` at the bias K position),
    then skips remaining padding K-slots in the producer's block. Used
    by moe_compute / moe_gpt W0/W1 with bias.
  - **`KStepWithRing<HasBias, Buf, RingStepFn>`** — adds ring CB sync
    on top of bias / padding skip. When the active ring slot's tile
    budget hits 0, pops the ring CB, waits for the next slot, and
    invokes `ring_step_fn(step_idx)` (caller-supplied lambda) which
    returns a `RingStepResult` with the new in0 read index and tile
    budget. The callback owns the indexing convention — moe_compute's
    monotonic `step_idx * tiles_per_step`, moe_gpt's 6-buffer cyclic
    `(step_idx % 6) * tiles_per_step`, etc. The kernel cb_wait_fronts
    the FIRST ring slot before the matmul_subblock_pack call; the
    final balancing pop fires inside the dst_body after
    matmul_segmented_kloop returns.

- **Convenience wrapper**:
  ```cpp
  template <typename Buf>
  ALWI void matmul_kloop_pack(
      Buf& in0_buf, Buf& in1_buf, Buf& out_buf,
      SegmentedKLoopShape shape,
      uint32_t dst_count,
      uint32_t in0_index_start = 0,
      bool transpose = false);
  ```
  Combines `matmul_subblock_pack` Form 1 + `matmul_segmented_kloop` +
  `KStepDefault` into a single call. Used by mla_wo — collapses each
  outer iter to one helper call.

**Tests** — coverage distributed across the migrated kernels:

- **mla_wo** on Wormhole n150 — JIT-compiles + runs to completion via
  `test_mla_wo[check_accuracy_False-...-dispatch_row]` (2/2 pass). This
  is the single-card validation that exercises `matmul_kloop_pack` end-
  to-end on real silicon. The `check_accuracy_True` variant has a
  preexisting BH-incompatibility (test scaffolding hardcoded for 12
  Wormhole DRAM banks; on BH's 7 banks the K dimension is truncated →
  PCC mismatch by design, independent of this migration).
- **moe_gate_mm** on Wormhole n150 — `test_moe_mm[...-check_accuracy_True-...]`
  passes (PCC=0.977). Validates Form 2 + `matmul_segmented_kloop` with
  `KStepDefault` + the `last_block_no_pop` partial-last-block path.
- **moe_compute** / **moe_gpt** — multi-chip CCL ring ops; build-only
  validation locally. Form 2 + `KStepWithBias` (W0/W1) and
  `KStepWithRing` (W2) end-to-end validation gates on the target
  multi-chip systems via CI.

The orthogonal `test_matmul_tile_pack_row_major.py` (34/34 pass) gates
against regressions in the existing `matmul_block` K-loop helper.

**Migrations** — 1:1 substitutions; `init_mode` defaults to `None` at
every `step(...)` call site since each kernel keeps its own outer
`mm_block_init`. Each DST cycle is wrapped in the appropriate
`matmul_subblock_pack` form; the K-loop body itself is absorbed by
`matmul_segmented_kloop` + a per-step functor:

- `experimental/deepseek/mla/matmul_wo/device/kernels/compute.cpp` —
  one outer iter loop, each iter collapsed to a single
  `matmul_kloop_pack(in0_buf, in1_buf, out_buf, shape, dst_count,
  in0_index_base)` call (the original 18-line dst_body lambda is gone).
- `experimental/deepseek/moe/moe_gate_mm/device/kernels/compute.cpp` —
  Form 2 `matmul_subblock_pack` with `matmul_segmented_kloop` +
  `KStepDefault` for both branches. The send-core branch uses a
  partial last block (`last_block_tiles = w_tiles_per_block_last`).
  The non-send branch uses `last_block_no_pop = true` so the bias
  `copy_tile` from the still-fronted in1 CB stays valid; the kernel
  pops the last block after the matmul_subblock_pack scope. Custom
  pack body unchanged (multi-CB, mid-DST SFPU).
- `experimental/ccl/moe_compute/device/kernels/compute.cpp` — Form 2
  in both cycles. W0/W1 cycle uses `KStepWithBias` (or `KStepDefault`
  via `if constexpr (has_bias)`). W2 cycle uses
  `KStepWithRing<HasBias, ..., RingStepFn>` with a monotonic
  `step_idx * tiles_per_step` `ring_step_fn`. STALL_CFG semwait +
  PACK-thread SFPU activation pack body unchanged on W0/W1;
  pack_untilize_dest pack body unchanged on W2.
- `experimental/ccl/moe_gpt/device/kernels/compute.cpp` — same shape
  as moe_compute across `TILIZE_FUSED` + non-fused branches. The
  W2 `ring_step_fn` differs: `(step_idx % 6) * tiles_per_step` for
  the 6-buffer cyclic in2 layout (vs moe_compute's monotonic).

**Validation gate for resuming**: re-run the three single-chip tests
above (mla_wo, moe_mm, matmul_tile_pack_row_major) on Wormhole n150;
trigger the moe_compute / moe_gpt CI lanes for the multi-chip CCL
ring ops. All four migrated kernels are 1:1 LLK → helper substitutions
inside the K-loop body; a regression in any of them is a helper-side
issue first (inspect the relevant `KStep*` operator()) rather than a
per-kernel logic bug.

### Phase C — `In0SourceFn` (production gathered)

**Helper changes** ✅ shipped 2026-05-06 (`ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.{hpp,inl}`):

- `NoIn0Source` struct: `(uint32_t block, uint32_t in0_cb_id) → uint32_t`,
  default returns the bound `in0_cb_id` so existing callers' single-source
  behavior is preserved.
- `In0SourceFn` template parameter (default `NoIn0Source`) + runtime
  parameter `in0_source_fn` — appended to the helper's template + function
  signatures (slot ordering kept stable; both have defaults so existing call
  sites need no edits).
- `.inl`: hoisted `const uint32_t active_in0_cb_id = in0_source_fn(block, in0_cb_id); Buf active_in0_buf(active_in0_cb_id);`
  at the top of the K-block body (alongside the Phase B `inner_steps`
  hoist). Substituted `active_in0_buf` for `in0_buf` on the per-K-block
  `wait_front` and both `pop_front` branches (the
  `In0Policy::WaitAndRetainOnLastBlock` and `WaitAndPopPerKBlock` paths),
  and the LLK call's first arg switched to `active_in0_cb_id`. The
  kernel-entry `mm_block_init` and the reload's
  `mm_block_init_short_with_dt` keep using the bound `in0_cb_id` — the
  alternate in0 CBs MUST share that CB's dataformat, documented as a
  hard invariant in the `In0SourceFn` docstring.
- The block-0 self-prime that the production gathered kernel needs
  (`reserve_back` / `push_back` on the local in0 CB before the K-loop's
  first iteration) lands inside the caller's `PreKBlockFn` when
  `block == 0` — no new helper API needed.

**Tests** — backwards-compat regression coverage shipped via
`test_matmul_tile_pack_row_major.py` (34/34), `test_sdpa_prefill.py::test_sdpa_noncausal`
(1/1), and `test_matmul_auto_tune.py` (29/29). The non-default callback
is exercised by the production-gathered migration (the first and currently
only caller emitting `block == 0 ? in0_cb_id : in2_cb_id`); end-to-end
correctness is gated on the T3K validation listed in the migration block
below. The "two physically-distinct in0 CBs filled with different data,
PCC against a manually-spliced reference" parametrization in the original
plan would have needed a test-only factory or custom op to expose the
callback as a runtime arg, so it was not added — production-gathered's
own sweep covers the path.

**Migration** ✅ shipped 2026-05-06
(`ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp`):

- Single `compute_kernel_lib::matmul_block` call replaces the entire `for block`
  body. Outer `for b` preserves fabric sync (sync_cb2 wait + sync_cb push/reset
  on ENABLE_GLOBAL_CB), CB-array indexing via the named `cb_mm_out_*` /
  `cb_mm_partials_*` compile-time arg tables, and the batch-start
  `pack_reconfig_data_format` + `llk_pack_relu_config(NO_RELU)` resets. No
  legacy K-loop branch — the kernel uses every K-loop helper extension shipped
  to express its full (ring-aware, partial-bias-padding, dual-source in0)
  matmul pattern through one helper invocation:
  - **`GatheredPreKBlock`** carries (a) the block-0 self-prime
    (`reserve_back` + `push_back` on the local in0 CB on block 0), (b)
    `calculate_next_block_index_and_update_rd_ptr` for ENABLE_GLOBAL_CB, and
    (c) `llk_pack_relu_config(ZERO_RELU)` on the last K-block when
    `PACK_RELU && untilize_out` (since `OutWithUntilize` doesn't auto-enable
    relu the way `OutWithRelu` does).
  - **`GatheredPostKBlock`** commits the next ring rd_ptr via
    `update_local_cb_rd_ptr` (UNPACK-only, gated on ENABLE_GLOBAL_CB).
  - **`GatheredInnerDimFn`** (Phase B) returns
    `unpadded_in0_shard_widths_in_tiles[(ring_idx + block) % ring_size]` so
    the helper's per-K-block FMA loop runs only the unpadded inner-dim count.
  - **`GatheredIn0Source`** (Phase C) returns
    `block == 0 ? in0_cb_id : in2_cb_id`. The helper's static
    dataformat-invariant assertion is satisfied by the host factory's CB
    setup (both CBs allocated with the same in0 dtype).
  - **`GatheredIn1BaseOffsetFn`** (Phase E) returns
    `in1_block_num_tiles * curr_ring_idx` on the
    `!ENABLE_GLOBAL_CB && !in1_is_dram` Case 4 path; 0 elsewhere. Note the
    formula uses `(ring_idx + block) % ring_size` (forward rotation) here,
    distinct from the llama variant's `(ring_idx - block + ring_size) % ring_size`.
  - **`In1Policy`** = `WaitAndPopPerKBlock` when `in1_is_dram` else
    `NoWaitNoPop` (Phase E).
  - **`LastBlockTarget`** = `OutWithUntilize` if `untilize_out`, else
    `OutWithRelu` if `PACK_RELU`, else `Out`. FUSE_BIAS dropped (host-side
    dead code in this kernel — same as the llama variant).
- The local `reload_from_cb_to_dst` is dropped; the helper does its own
  reload via `copy_tile_to_dst_init_short_with_dt` +
  `copy_block_matmul_partials` + `mm_block_init_short_with_dt` inline.

**Validation:** end-to-end correctness requires T3K hardware that this op
runs on; ttsim cannot reproduce the multi-chip / fabric / CCL path
(`docs/simulator.md`). Validation gate before publishing the branch is at
least one production-gathered T3K sweep parametrization. Build-only
validation passed in the migration session; default-callback regressions
covered via the matmul tile-pack nightly (34/34), SDPA non-causal (1/1),
auto-tune (29/29), and the deepseek MLA `check_accuracy_False`
parametrization (2/2 — transitive `matmul_subblock_pack` + `step`
consumer).

### Class B — `all_gather_minimal_matmul_async` ✅ shipped 2026-05-06

Originally deferred as needing three orthogonal helper extensions
(`LastBlockTarget::SingleBufferAccum`, partial-shape support on
`MatmulBlockShape`, outer-loop in0 reuse distinct from the per-call
`In0Policy`). Re-examination at completion time collapsed those into a
single new extension and two existing surfaces:

- **`caller_owns_pack_target` template bool** (new) — when true the helper
  skips ALL its own `reserve_back` / `push_back` / inter-block drain calls
  on the pack target buffer (and the spill `interm_buf`). The caller pre-
  reserves once before the helper call and pushes once after; every
  K-block packs at the same absolute row-major offsets with L1_ACC
  accumulating into the same L1 cells. Replaces the proposed
  `LastBlockTarget::SingleBufferAccum` cleanly without a new enum value.
- **Partial-shape support** turned out to already exist via
  `MatmulBlockShape`'s runtime fields (`in0_num_subblocks`,
  `in1_num_subblocks`, `out_subblock_h`, `out_subblock_w`) and the
  `out_row_width` helper parameter. Class-B passes
  `current_M_block_tiles / current_subblock_h` etc. as runtime args; no
  new helper API needed.
- **Outer-loop in0 reuse** turned out to map exactly onto the existing
  `In0Policy` enum — `In0Policy::WaitAndRetainOnLastBlock` skips popping
  in0 on the last K-block, which is exactly what Class-B's
  `reuse_in0_block` flag does. The kernel selects between two helper
  template instantiations at runtime per outer `(m, n)` iter
  (`WaitAndRetainOnLastBlock` when reusing for the next n iter,
  `WaitAndPopPerKBlock` on the final n iter).

The kernel-side migration replaces the private `void matmul_blocks(...)`
function and the K-loop calling it with one helper invocation per outer
`(m, n)` iter. The bias / addcmul / copy-with-activation phase is
unchanged (it's an eltwise downstream pipeline that consumes from the
helper's accumulated intermediate CB). Validation gate: any of the
op's existing test parametrizations on its target multi-chip system.

### Cross-cutting

- ttsim is single-chip only — fabric/CCL workloads (T3K, Galaxy/TG, the
  all_gather variants) are out of scope. The CCL/multi-chip migrations
  (llama gathered, production gathered, moe_gate_mm, moe_compute,
  moe_gpt, all_gather_minimal) ship build-validated on Blackhole p100a
  with end-to-end gates listed in their per-phase sections.
- Default-callback regression coverage on Blackhole p100a:
  `test_matmul_tile_pack_row_major.py` (34/34),
  `test_sdpa_prefill.py::test_sdpa_noncausal` (1/1),
  `test_matmul_auto_tune.py` (29/29), `test_mla_wo[check_accuracy_False-...]`
  (2/2 — transitive `matmul_subblock_pack` + `step` JIT consumer). These
  transitively re-exercise every other helper-using compute kernel
  through the default-callback path.
- The `matmul_block_helpers` row in
  `ttnn/cpp/ttnn/kernel_lib/agents/llk_helpers_conventions.md` was kept
  up to date as each helper extension shipped — final row lists every
  template parameter, callback, and enum value above.

## Test suite and verification status

Single-source map of the validation surface for this branch — read this
*first* on resume before re-running anything. Run S1–S5 always; S6+
scope-by-scope. **Pre-flight: `unset TT_METAL_DPRINT_CORES` before any
perf tier (S7 gtests / S8 demos / S6 sweep `--perf`).** Build:
`./build_metal.sh --release --build-tests --enable-ccache`.

Status legend per row: `n/n ✅` = pass, `m/n ⚠️` = partial, `n/n ❌` = fail
(see notes below table), `running` = currently executing, `—` = not yet
run on the current 4-commit shape. `[stale yyyy-mm-dd · arch]` = pass on a
prior commit shape, re-run before relying. Pass counts come from
`tests addition` validation runs against the 2026-05-06 4-commit form;
the 2026-05-07 reshape was metadata-only (commit boundaries, no file
content changes), so those passes carry forward but are not
authoritative on the current shape until re-verified.

### Test matrix — WH n150 + BH p100a

Each tier's row carries both arches' latest result + date so a single matrix is the running ledger across both stages.

| Tier | Entry | Cases | WH n150 | BH p100a |
|---|---|---|---|---|
| S1.1 | `tests/ttnn/unit_tests/operations/matmul/test_matmul_auto_tune.py` | ~28 host + 1 device | 29/29 ✅ (2026-05-07) | 29/29 ✅ (2026-05-08, 6.4s) |
| S1.2 | `tests/ttnn/nightly/.../matmul/test_matmul_tile_pack_row_major.py -k "subblk_2x2 and dram_out"` | 3 | 3/3 ✅ (2026-05-07) | 3/3 ✅ (2026-05-08, 4.0s) |
| S3.1 | `tests/ttnn/nightly/.../matmul/test_matmul_tile_pack_row_major.py` (full) | 34 | 34/34 ✅ (2026-05-07, 30.7s) | 34/34 ✅ (2026-05-08) |
| S3.2 | `tests/ttnn/nightly/.../experimental/test_mla_wo.py` | 4 | 4/4 ✅ (2026-05-07, 2:46) | 2/2 ✅ + 2 preexisting BH-incompat (`check_accuracy_True`: scaffolding hardcoded for 12 WH DRAM banks; BH p100a has 7 → K truncates → PCC drift; not a migration regression) |
| S3.3 | `tests/ttnn/nightly/.../experimental/test_mla_wo_perf.py` | 1 | 1/1 ✅ (2026-05-07, 63.5µs/layer; target <65µs) | preexisting BH-incompat (wraps the failing `check_accuracy_True` case) |
| S3.4 | `tests/ttnn/nightly/.../experimental/test_moe_mm.py` | 1 | 1/1 ✅ (2026-05-07) | preexisting BH-incompat (`TT_FATAL @ moe_gate_mm_program_factory.cpp:37: num_cores == required_cores` — host-side op-level core-count assertion fails on BH p100a; never reaches compute, so kloop refactor is not at fault) |
| S3.5 | `tests/ttnn/nightly/.../experimental/test_moe_mm_perf.py` | 1 | 1/1 ✅ (2026-05-07, 25.2µs/layer; target <27µs) | preexisting BH-incompat (same as S3.4) |
| S4.1 | `tests/ttnn/unit_tests/operations/matmul/` (excl. auto_tune) | 9 files, 1275 | 1134/1134 ✅ + 140 skipped + 1 xpassed (2026-05-07, 20:45) | 1071/1071 ✅ + 203 skipped + 1 xpassed (2026-05-08, 19:42) |
| S4.2 | `tests/ttnn/unit_tests/operations/conv/test_conv2d.py` | 209 | 161/161 ✅ + 48 skipped (2026-05-07, 3:53) | 161/161 ✅ + 48 skipped (2026-05-08, 3:58) |
| S4.3 | `tests/ttnn/unit_tests/operations/conv/test_conv3d.py` | 1551 | 1551/1551 ✅ (2026-05-07, 10:27) | combined with S4.4 below: 1857/1857 ✅ + 10 skipped (2026-05-08, 21:43) |
| S4.4 | `tests/ttnn/unit_tests/operations/conv/test_conv1d.py` + `test_conv_transpose2d.py` + `test_prepare_conv_weights.py` | 316 | 307/307 ✅ + 9 skipped (2026-05-07, 7:42) | (batched with S4.3 on BH) |
| S4.5 | `tests/ttnn/unit_tests/operations/sdpa/test_sdpa_prefill.py` | 6 | 4/4 ✅ + 2 skipped (2026-05-07) | combined with S4.6+4.7 below: 29/29 ✅ + 3 skipped (2026-05-08, 4:38) |
| S4.6 | `tests/ttnn/unit_tests/operations/sdpa/test_sdpa_decode.py` | 12 | 11/11 ✅ + 1 skipped (2026-05-07, 1:37) | (batched with S4.5+4.7 on BH) |
| S4.7 | `tests/ttnn/unit_tests/operations/sdpa/test_mla_prefill.py` + `test_mla_decode.py` + `test_mla_prefill_v_embedding_space.py` | 14 | 13/13 ✅ + 1 skipped (2026-05-07, 2:36) | (batched with S4.5+4.6 on BH) |
| S5.A | `./tests/scripts/single_card/nightly/run_wh_b0_only.sh` (WH B0 nightly wrapper) | broad | skipped (CI-only) | n/a (WH-only path) |
| S5.B | `tests/ttnn/nightly/.../matmul/test_matmul_dram_sharded.py` (DRAM reload-deadlock gate) | 41 | 41/41 ✅ (2026-05-07, 1:31) | combined with S5.C+D+E below: 1147/1147 ✅ + 994 skipped (2026-05-08, 19:41) |
| S5.C | `tests/ttnn/nightly/.../matmul/test_matmul2.py` | 290 | 193/193 ✅ + 97 skipped (2026-05-07, 2:58) | (batched on BH) |
| S5.D | `tests/ttnn/nightly/.../matmul/test_attn_matmul.py` | 753 | 603/603 ✅ + 150 skipped (2026-05-07, 4:10) | (batched on BH) |
| S5.E | `tests/ttnn/nightly/.../matmul/test_matmul.py` + `test_matmul_1d_2d.py` + `test_bert_matmuls.py` + `test_tilize_matmul.py` + `test_tilize_hpadding_matmul.py` + `test_matmul_block_sharded_1d_grid.py` (+ T3K-only `test_matmul_1d_gather_in0.py` + `test_rs_matmul_1d_gather_in0.py`) | 1057 | 553/553 ✅ + 504 skipped (2026-05-07, 12:30) | (batched on BH) |
| S5.F | `tests/ttnn/nightly/.../matmul/test_matmul_tile_pack_row_major.py` (full nightly form) | dup of S3.1 | skipped (dup) | skipped (dup) |
| S5.G | `tests/ttnn/nightly/.../conv/test_conv2d.py` (60+ funcs) | 2399 | 1722/1722 ✅ + 677 skipped (2026-05-08, 42:17) | combined with S5.H below: 5417/5417 ✅ + 633 skipped + 2 xfailed (2026-05-08, 1:49:52) |
| S5.H | `tests/ttnn/nightly/.../conv/test_conv3d.py` + `test_conv2d_ulp.py` + `test_conv2d_sweeps.py` (1550 + 494 specs) | 3653 | 3646/3646 ✅ + 5 skipped + 2 xfailed (2026-05-08, 1:03:36) | (batched with S5.G on BH) |
| S5.I | `tests/ttnn/nightly/.../sdpa/test_sdpa_prefill.py` (extended) | 889 | 681/681 ✅ + 208 skipped (2026-05-08, 38:55) | combined with S5.J+K below: 1028/1028 ✅ + 271 skipped (2026-05-08, 56:05) |
| S5.J | `tests/ttnn/nightly/.../sdpa/test_sdpa_decode.py` + `test_sdpa_decode_cache.py` + `test_sdpa_decode_sink.py` + `test_sdpa_chunked.py` + `test_sdpa_joint.py` | 346 | 297/297 ✅ + 49 skipped (2026-05-08, 17:54) | (batched with S5.I on BH) |
| S5.K | `tests/ttnn/nightly/.../sdpa/test_mla_prefill.py` + `test_mla_decode.py` + `test_mla_prefill_chunked_vs_not.py` | 64 | 64/64 ✅ (2026-05-08, 6:53) | (batched with S5.I on BH) |
| S5.L | `tests/ttnn/nightly/.../sdpa/test_mla_prefill_stress.py` + `test_mla_decode_stress.py` | ~10 | skipped (stress oos) | skipped (stress oos) |
| S6.1–5 | sweep_framework (matmul.{short,full,generality,sparse} + conv2d.short + model_traced) | ~5K | skipped (CI-only setup) | skipped (CI-only setup) |
| S7.1 | `./tests/scripts/run_cpp_unit_tests.sh` | dispatch + legacy gtests | skipped (binaries not built) | skipped (binaries not built) |
| S7.2 | `./build/test/ttnn/unit_tests_ttnn --gtest_filter='*Matmul*\|*Conv2D*\|*Sdpa*'` | matmul + conv2d + sdpa_reduce_c gtests | skipped (binaries not built) | skipped (binaries not built) |
| S8 | `pytest models/demos/{resnet50,distilbert,metal_BERT_large_11,vgg}/demo/demo.py` | 4 ran | distilbert 2/2 ✅; bert 1+ ✅; resnet50 1+ ✅; vgg HF-auth (2026-05-08) | distilbert 2/2 ✅ (2026-05-08, 38.6s); others HF-auth same as WH |
| S9 | `tests/ttnn/stress_tests/...` | hammer set | skipped (stress oos) | skipped (stress oos) |
| **Post-refactor regression** | auto-tune + tile_pack RM full + mla_wo + moe_mm | 68 | 68/68 ✅ (2026-05-08) | — |

#### BH-only adds (Stage 2 unique surfaces)

| Tier | Entry | Cases | WH n150 | BH p100a |
|---|---|---|---|---|
| BH.1 | `tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py` + `test_exp_ring_joint_sdpa.py` + `test_ring_joint_attention.py` + `test_scaled_dot_product_attention_sprint.py` | ~13 | n/a (BH-only tree) | 11 skipped on single-chip p100a (all parametrizations need 2×2 / 2rpx2up mesh) — covered when a multi-chip BH reservation arrives |
| BH.2 | `tests/ttnn/nightly/.../conv/test_conv2d.py -k "panoptic or panoptic_110_cores"` | — | n/a (110-core BH p150 grid) | folded into S5.G conv2d nightly batch (above) — passed within the 5417 passes |
| BH.3 | `tests/ttnn/unit_tests/ -k "force_full_grid or grid_override"` | — | n/a (BH 64-core) | n/a — no tests on this branch use `force_full_grid` / `grid_override` markers; nothing to run |
| BH.4 | `tests/didt/test_mla_sdpa.py` + `test_sdpa_op.py` | — | n/a (BH multi-chip determinism) | skipped (stress-equivalent — `test_mla_sdpa[blackhole-1chips]` is a 1000+ iter determinism check on a 4K×128K×32-head×576-dim attention; timed out at 900s on p100a) — per "skip stress" policy |

### Multi-chip (CI lanes; cannot run locally on n150; ttsim N/A — single-chip only)

| System | Suite |
|---|---|
| T3K (8x WH) | `tests/scripts/t3000/run_t3000_unit_tests.sh` + `tests/nightly/t3000/ccl/test_{minimal_all_gather_matmul_async, strided_all_gather_minimal_matmul_async, minimal_matmul_strided_reduce_scatter_async, ring_joint_attention, ring_attention_all_gather, moe_routing_remap, moe_expert_token_remap, moe_ccl_end_to_end, deepseek_moe_reduce_scatter}.py` + `tests/ttnn/unit_tests/operations/ccl/test_new_matmul_reduce_scatter.py` |
| TG / Galaxy | `tests/scripts/tg/run_tg_frequent_tests.sh` + `tests/ttnn/unit_tests/operations/ccl/test_llama_all_gather_matmul.py` (Galaxy ONLY) + `tests/nightly/tg/ccl/test_{minimal_all_gather_matmul_async_nightly, minimal_matmul_strided_reduce_scatter_async, ring_joint_attention}.py` + `tests/nightly/tg/ccl/moe/test_moe_compute_6U.py` (96 cases) |
| BH 4×8 | `tests/nightly/tg/ccl/test_all_gather_minimal_matmul_addcmul_async.py` (requires BH 12×9 grid) |
| Multi-host | `tests/scripts/multihost/run_{dual_galaxy, quad_galaxy}_tests.sh` (Llama-405B distributed) |
| Model lanes | T3K Llama2-70B / Mixtral; TG Llama3-70B Galaxy / GPT-OSS / DeepSeek-V3-D-P / DIT (Wan2.2) |

### Recommended session order (per arch)

S1 → S3 (gates; stop on red) · S4 · S5 (small batches) · `unset TT_METAL_DPRINT_CORES` · S8 (demos). On BH, also run BH.1–BH.4 (BH-only adds: blackhole/sdpa, panoptic 110-core, force_full_grid, didt). Each tier is independently fail-investigatable; a regression at S5 doesn't invalidate S1–S4. If anything hangs, the `tt-hang-debugger` agent owns the diagnostic loop. (S2 LLK Python, S6 sweep_framework, S7 gtests are out of scope / CI-only; S9 stress is out of scope.)

### Resume protocol

1. Read this section's status column — anything with `n/n ✅` is done; skip it.
2. `pgrep -af pytest` and `pgrep -af unit_tests` to find any stragglers from a prior session. Decide: resume (if it's still making progress — `pgrep` + `tt-smi -s` show device active) or kill (`pkill -f pytest`, then `tt-smi -r` to recover device).
3. Pick the next `—` row and run it; update the row inline before moving on.
4. **Don't** mass-rerun rows that already passed on this 4-commit shape; trust them and move forward.

## How this file is maintained

This file moves alongside the four-commit stack on `wransom/mm_help`:
`helper implementation` → `tests addition` → `migration / kernel
implementation` → **`status`**. Each pass on the project updates these four
commits in place rather than appending new ones.
