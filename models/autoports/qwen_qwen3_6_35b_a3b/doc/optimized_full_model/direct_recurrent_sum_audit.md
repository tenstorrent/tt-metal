# Direct recurrence contraction candidate

This is a new experimental component, not a production replacement or a qualified performance result. No hardware was opened by its implementing agent. CPU fixture preparation passed; device compilation, bit equality, trace safety, and latency remain for the parent's serialized hardware run.

## Preserved sequence

Current `LinearTP4Mixin._recurrent_sum` receives native local FP32 `[1,8,128,128]` products. Its HC permutation expands the physical footprint fourfold. `fast_reduce_nc(dims=[0,1])` processes dimension 1 before dimension 0: eight contiguous groups of sixteen inputs, each starting from its first operand; then eight partials, also starting from the first. FP32 SFPU adds retain the arithmetic order.

`tt/direct_recurrent_sum_probe.py` assigns 32 workers, one per head and 32-column tile. Each worker reads four original product tiles (16 KiB). It unpacks directly into FP32 destination registers and performs the same ordered reductions. Four-row local register transposes follow the existing Blackhole column-scan mapping. Every dependent add retains an explicit SFPNOP. The intermediate accumulated group total is stored in FP32 destination scratch, not recomputed or rounded to BF16. No FPU matmul, TF32, multiply, FMA, product change, or pointwise recurrence fusion is used.

The direct kernel returns either the existing head-major `[1,8,1,128]` layout or terminal rows `[1,1,8,128]`. Head-major output writes 128 KiB physically. Terminal output writes only 4 KiB of valid rows into its 16 KiB allocation, using disjoint 64-byte half-row ownership. Padded rows are unspecified and must not be mistaken for logical values. Any wider consumer that exposes padding must be separately checked. These are traffic counts, not measured speedups.

The optional `install` helper changes only `_recurrent_sum` when explicitly invoked by a probe. Production imports and callers are untouched. `_recurrent_step` still performs BF16-on-consumption state rounding when requested, FP32 products, live FP32 state update, and current-output consumption before the next decode rounds state. No cache, activation, CCL, residual, or projection policy changes follow from this candidate.

## Checks and commands

`tests/run_stage7_recurrent_sum_probe.py` builds rank-distinct real-derived products from saved full-head operands at positions 128 and 8157, for rounded and unrounded state consumption, both memory and output contractions. CPU reconstruction is only a fixture source; current TT contraction is the device oracle. Additional random, cancellation, and signed-zero fixtures discriminate reassociation and first-operand behavior. All four physical ranks, both output layouts, eager outputs, and changed-input trace replays are compared bitwise. All output readback layouts and copies are warmed before the first capture. Program cache count must remain unchanged.

The timing loop uses nonblocking replay and synchronization only at measurement boundaries. Each trace includes the complete old conversion/reduction/output sequence or complete new direct operation and an identical stable-output copy. Host transfers and readbacks are excluded from the clock. Acceptance-mode timings are diagnostic only.

Parent-only commands, after sourcing the lane environment and checking exclusive device ownership:

```bash
env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 timeout 300 python -u -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_stage7_recurrent_sum_probe --mode acceptance --output models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_full_model/recurrent_sum_acceptance.json
```

After a pass, rerun with tracking 0, no watcher/profiler, `--mode timing`, and a distinct output path. `recurrent_sum_cpu.json` records CPU fixture hashes and finiteness/reassociation diagnostics; it does not establish hardware correctness. `logs/recurrent_sum_hooks.log` records repository formatting/lint hooks.

A component pass remains insufficient for production selection. Require complete recurrence state/output and next-consumer equality, changing-position traces, real layer34/long-stream diagnostics, full-model accuracy, and clean all40 timing after any installation. If the candidate misses equality, retains only unverified speed, or fails the full-model speed floor, the stage remains unfinished.

## Independent source review

The independent source reviewer found that the first draft initialized address modifier 3 while explicit SFPU loads/stores used modifier 7. `compute_kernel_hw_startup` and `copy_init` do not establish the required invariant modifier 7. Fixed before any device execution: `exact_sum` now explicitly initializes modifier 7 to zero increments, matching the invariant SFPU initialization contract. This finding is not a hardware failure or evidence of numerical equivalence. The remainder of independent review found no demonstrated arithmetic, circular-buffer, or trace issue for L1 inputs. The harness now asserts exactly four ranks plus equal FP32 shapes before comparing, and restores the first real-derived fixture before timing. The helper now restricts inputs to the actual interleaved L1 contract; DRAM is not advertised or validated. Device checks remain required.
