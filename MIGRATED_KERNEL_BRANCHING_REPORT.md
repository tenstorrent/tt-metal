# Migrated kernel branching and code-size audit

Date: 2026-07-23

Raw LLK baseline: `d59e06f1c51211089bd34705099745b0b6b05833`

Migration HEAD: `d1922c23aa55ecab4c0c6b591ea76799d39028e1`

Fixed state: local worktree after the branching audit and Moreh helper migration

## Result

The audit covered the 89 kernel source paths in the migration range (88 current
production paths). It compared raw setup lifetimes, preprocessor structure,
`ckl::runtime_if` predicate grouping, and emitted TRISC1 code for shortlisted
candidates.

Four kernels have confirmed unnecessary branching or repeated guarded setup:

| Kernel | Regression introduced by migration | Fix |
|---|---|---|
| `experimental/transformer/rotary_embedding_llama/.../rotary_embedding_llama.cpp` | One raw `mul_tiles_init` was split across two self-initializing CKL calls. The migration also duplicated both multiply bodies under new `#if RELOAD_IMPL` / `#else` blocks. | One caller-owned raw multiply setup and two direct setup-free `eltwise_chain` calls. The multiply body uses compile-time-selected lifecycle and offset policies, so the added preprocessor branch is gone. The single add remains a self-contained `ckl::add`. |
| `experimental/transformer/rotary_embedding_llama/.../rotary_embedding_llama_sharded.cpp` | One raw `mul_bcast_rows_init_short` was split across two self-initializing CKL calls. | Reuse one caller-owned setup across the two direct multiply chains. The single add remains a self-contained `ckl::add`. |
| `experimental/transformer/rotary_embedding_hf/.../rotary_embedding_hf_sharded.cpp` | Same broadcast-row setup duplication as the Llama sharded kernel. | Reuse one caller-owned setup across the two direct multiply chains. The single add remains a self-contained `ckl::add`. |
| `moreh/moreh_clip_grad_norm/.../moreh_clip_grad_norm_step1_kernel.cpp` | Two independent raw mask decisions became a four-way `H+W` / `H` / `W` / neither dispatch. | Restore two independent ordered `runtime_if` elements: H mask followed by W mask. |

The convenience API is unchanged. Caller-owned setup and runtime tile offsets are
escape-hatch controls, so the three exceptional rotary sites use
`eltwise_chain<SetupOwner::Caller>` directly. Only the pair of multiplications that
shares one raw setup uses caller ownership. The lone add uses the ordinary
self-contained `ckl::add`; its inputs explicitly disable redundant data-format
reconfiguration.

HEAD had six conditional directives versus five in the raw Llama source; the fixed
source has four. More importantly, no fixed conditional encloses an alternative CKL
operation body: the remaining directives only manage the mode-specific outer CB
lifetime and counter updates. No other migrated kernel increased operation-body
preprocessor branching.

Two source-level SFPU candidates (`ternary_addcmul_int_sfpu` and batch-norm running
statistics) were investigated separately. A slot-independent init-dedup experiment
produced no `.text` or conditional-branch reduction for the integer addcmul kernel
(HEAD and experiment were both 5,376 bytes and 37 branches), so that generalized
planner change was discarded rather than included without emitted-code evidence.

## Moreh helper migration follow-up

The affected Moreh kernels were revisited as a separate pass. The `_to_cb` helpers
now express their one-tile operations with eltwise chains while preserving the
existing Moreh API and concise call sites:

- CB IDs are compile-time template arguments.
- The helpers own their one-tile output reserve/push lifecycle.
- The existing wait/pop knobs remain helper-owned; callers retain only genuinely
  external or multi-operation input lifetimes.
- `power_tile_to_cb_impl` uses optional chain elements for absolute value and final
  reciprocal, with chain-owned intermediate CB lifetimes.
- FP32 data-format reconfiguration remains conditional on `FP32_DEST_ACC_EN`.
- `moreh_input` and `moreh_output` are implementation details of
  `moreh_common.hpp`; kernel files describe their actual lifetimes directly.

Raw LLK call sites were migrated only where the operation mapped cleanly to the
eltwise helpers. Moreh mean-W deliberately retains its raw matmul sequence: it is
an accumulation into a retained destination register, and the existing
`mm_init_short_with_dt` helper only reconfigures SrcA rather than the two-input
format transition needed there. Adding a broad convenience helper for this one
case would hide more lifecycle than it clarifies.

The follow-up audit also established two invariants for subsequent migrations:

1. A kernel that used `DataflowBuffer` before chain migration must not regress to
   `CircularBuffer`. All affected Moreh kernels were checked against the raw
   baseline; none introduces `CircularBuffer`.
2. Conditions sourced from compile-time arguments use `if constexpr`. Remaining
   ordinary `if` statements depend on runtime arguments or loop indices.

## Three-state measurements

### Rotary embedding Llama, prefill

Workload: BF16, prefill sequence 32, head dimension 64; Q has eight heads and K has
one head. Device results are the maximum kernel duration across participating
cores. Compile results are medians of five direct, ccache-disabled TRISC1 compile
commands.

| State | Variant | ELF file (bytes) | `.text` (bytes) | Conditional branches | TRISC1 compile | Device time |
|---|---|---:|---:|---:|---:|---:|
| Raw LLK baseline | Q, 8 heads | 340,048 | 1,844 | 28 | 260.648 ms | 17.557 us |
| Raw LLK baseline | K, 1 head | 338,092 | 1,804 | 26 | 257.907 ms | 9.110 us |
| Migration HEAD | Q, 8 heads | 535,340 | 2,284 | 36 | 333.742 ms | 18.469 us |
| Migration HEAD | K, 1 head | 534,188 | 2,256 | 35 | 333.785 ms | 9.131 us |
| HEAD plus fix | Q, 8 heads | 450,588 | 1,836 | 26 | 341.506 ms | 17.686 us |
| HEAD plus fix | K, 1 head | 442,300 | 1,812 | 25 | 338.749 ms | 9.228 us |

Relative to HEAD, the fix removes 19.6% of Q `.text` and ten conditional branch
instructions; K removes 19.7% and ten branches. Executable code is back at raw
LLK size: Q is 0.4% smaller than raw and K is 0.4% larger. Q device time improves
4.2% versus HEAD and is 0.7% above raw; K measurements are within 1.3% of both
other states.

The full ELF remains larger than raw because it includes template-heavy DWARF debug
information. `.text` is therefore the meaningful device code-size comparison; the
full filesystem ELF size is included because it was explicitly requested.

### Rotary embedding HF sharded

Workload: decode, head dimension 128, more than one batch row per core. Device time
is the target program's median over three launches; compile time is the median of
five direct TRISC1 compiles.

| State | ELF file (bytes) | `.text` (bytes) | Conditional branches | TRISC1 compile | Device time |
|---|---:|---:|---:|---:|---:|
| Raw LLK baseline | 361,908 | 1,704 | 23 | 355.967 ms | 4.660 us |
| Migration HEAD | 535,372 | 2,104 | 32 | 467.618 ms | 4.662 us |
| HEAD plus fix | 435,260 | 1,624 | 22 | 481.199 ms | 4.620 us |

The fixed emitted code has one fewer branch than raw and is 4.7% smaller than raw
`.text`. Relative to HEAD it removes 22.8% of `.text`, ten branch instructions, and
18.7% of full ELF size. Device time improves 0.9%. Direct compile time is 2.9% above
HEAD; compile time is not claimed as an improvement.

### Moreh clip-grad-norm step 1

This table is retained for the other confirmed branch family. Its workload is one
BF16 tiled `[1, 1, 1023, 1023]` tensor with `max_norm=2.0` and `norm_type=2.2`, which
exercises H-only, W-only, H+W, and unmasked tiles. Compile results are medians of
five direct, ccache-disabled TRISC1 compile commands, using each state's own source
tree and headers.

| State | TRISC0 ELF | TRISC1 ELF | TRISC1 `.text` | TRISC2 ELF | Total ELF | TRISC1 compile | Device time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw LLK baseline | 659,620 | 1,508,572 | 7,012 | 566,776 | 2,734,968 | 488 ms | 4,223.463 us |
| Migration HEAD | 1,116,232 | 2,037,900 | 7,328 | 1,011,540 | 4,165,672 | 756 ms | 4,144.716 us |
| Post-Moreh helper fix | 901,540 | 1,873,408 | 7,064 | 896,428 | 3,671,376 | 718 ms | 4,287.6 us |

The post-Moreh state reduces total ELF size by 11.9% and TRISC1 `.text` by 3.6%
versus migration HEAD. Its emitted code is within 0.7% of the raw LLK `.text`;
the 34.2% full-ELF gap to raw is predominantly template-heavy debug information.
Direct compile time improves 5.0% versus HEAD but remains 47.1% above raw. Device
time is 3.4% slower than HEAD and 1.5% slower than raw, so no device-time
improvement is claimed.

## Method

- Hardware: one N150 L, Wormhole B0; firmware bundle 18.12.1.
- Software: `build_Release`. The HEAD host runtime was held constant and
  `TT_METAL_KERNEL_PATH` selected raw or migration-HEAD kernel sources.
- ELF size: filesystem size of `trisc1.elf`, including debug information. `.text`
  comes from `riscv-tt-elf-size -A`.
- Branch count: conditional RISC-V branch mnemonics in `riscv-tt-elf-objdump -d`;
  calls, jumps, and returns are excluded.
- Rotary compile time: the exact successful TRISC1 compiler command emitted by the
  JIT was replayed five times with ccache disabled; the median is reported. Link time
  is excluded.
- Rotary device time: maximum `*-KERNEL` duration across participating cores for the
  target run-host ID, converted at 1,000 MHz.
- Clip compile time: the exact successful TRISC1 compiler command was replayed five
  times with ccache disabled and state-local headers; the median is reported.
- Clip device time: mean of ten launches after two warmups.

## Validation

- Llama prefill focused test: passed after the final change for both generated
  variants, PCC approximately 0.9999965.
- HF sharded decode tests: head dimensions 32 and 128 passed; the 128 case exercises
  `rotary_embedding_hf_sharded.cpp`.
- Llama sharded and HF sharded compute sources also passed direct TRISC1 compilation.
- Integer addcmul validation: four int32 shape cases passed during candidate analysis.
- Clip-grad-norm suite: 20 tests passed.
- Moreh softmax/log-softmax BF16 matrix: 185 tests passed, 64 deselected.
- Moreh layer norm BF16 forward/backward matrix: 48 tests passed, 51 deselected.
- Moreh Adam/AdamW, SGD, norm, mean-W, and sum-H focused configurations passed,
  including FP32 destination accumulation, AMSGrad, momentum/Nesterov, masks, and
  forward/backward paths; one unsupported SGD case was skipped by the test.
- Source audit: affected raw kernels that used `DataflowBuffer` still use it (or
  delegate that lifecycle to a Moreh helper); no affected kernel includes or uses
  `CircularBuffer`. Compile-time conditionals use `if constexpr`.

The Llama decode integration test currently fails before reaching the patched
sharded compute kernel because an unrelated `eltwise_typecast` invocation is missing
`CHAIN_TYPECAST_IN_DF` / `CHAIN_TYPECAST_OUT_DF` defines. That failure is not counted
as validation of this change; the target compute source was compiled directly.
