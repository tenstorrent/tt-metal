# AutoFix: batched public-layout L1 pressure in a decoder stack

Source-only diagnosis dated 2026-09-12. No device commands or implementation edits were made by this investigator. The frozen `optimized_decoder.py` is unchanged. Hardware verification of the policy override is pending; the parent will append its result.

## Starting evidence

`stress_stack_batch32.log:25` fails during the **first normal decode**, before the queued stress loop. The second, full-attention decoder reaches `_finish` -> `_linear(n, "mlp.gate_up")` -> the native DRAM-sharded matmul. Its static circular-buffer region ends at **1,338,368**, while the lowest occupied L1 allocation is **921,344**, an overlap of **417,024 bytes**. The failing program covers core range `[0-0 - 9-9]`.

The original command, recorded in `commands.log`, is:

```bash
bash models/autoports/qwen_qwen3_8_27b/tests/run_multichip_experiment.sh stress_stack_batch32 --layer 0 --batch 32 --length 257 --stack --repeats 10 --stress-iterations 100
```

The parent reports that both TP4 B32 single-layer kinds, TP4 B1/B3 stacks, and the corrected B32 baseline stack pass. The failing TP4 source revision is the parent's recorded `e360872...` snapshot. This failure is a host-side allocation-region validation, not observed all-reduce data corruption, trace replay failure, or an unsupported batch assertion.

## Causal chain

1. **A public batched tensor expands its tile storage.** `optimized_decoder.py:442-452` documents the packed/public distinction: compact decode `[1,1,B,H]` has `ceil(B/32)*32` physical rows, while public `[B,1,H]` has `B*32`. `_public_rows` converts to interleaved L1 by default and then reshapes. For B32/H5120, compact BF16 storage is **327,680 bytes**, but public tiled `[32,32,5120]` needs **10,485,760 bytes = 10 MiB per device**. This is actual row repacking, not a free view. Native `reshape_view/reshape.cpp:387-391` computes different padded source and destination shapes.
2. **The first layer's expanded result stays live in the second layer.** The carried `_finish` branch returns `_public_rows(..., B,5120)` (`optimized_decoder.py:569`). The runner evaluates `next_decoder.decode_forward(result, ...)` before replacing `result` (`tests/run_multichip_decoder.py:210-217`). The second decoder's `decode_forward.x` reference also retains this public L1 input through attention and `_finish` (`optimized_decoder.py:894-906`). Moving only a local callee reference cannot release a caller-owned allocation.
3. **Other public boundaries retain similar tensors.** Input normalization repacks through `_public_rows` (`optimized_decoder.py:496-520`); `decode_forward.n` then stays live through `_finish`. Attention projection output already has its narrower `attention_dram_batch` rule, but that does not relocate the final layer result or public normalization. The packed residual/post-attention norm inside `_finish` is a separate compact `[1,1,B,5120]` path; its small L1 tensors are useful and should remain compact.
4. **The packed MLP has a large fixed scratch requirement.** It reaches the same compact B32 DRAM-sharded matmul used by a single layer, but with additional live public tensors and different allocation history. The stack does not require a larger logical matmul. Its extra persistent public L1 input and other public intermediates tighten the allocator budget until the existing static region overlaps occupied L1.

The log does not identify the tensor owning address 921,344. The first result alone must not be claimed to explain the entire 417,024-byte per-core shortfall: distribution across L1 banks, other live allocations, and allocation history matter. A buffer allocation snapshot would identify the exact owner. The source supports a focused placement control without inventing that attribution.

## Native scratch arithmetic matches the failure

`ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:159-191` assigns the 272 output-width tiles across eight readers, yielding 34 weight tiles per reader and a padded compute width of 36 tiles. Lines 223-239 double-buffer the activation block, triple-buffer the weight block, and allocate separate BF16 output / FP32 accumulation storage. For the current gate block 16 and one compact row tile:

| Static CB | Bytes/core |
| --- | ---: |
| Activation: `1 * 16 * 2 * 2048` | 65,536 |
| BFP4 weights: `34 * 16 * 3 * 576` | 940,032 |
| BF16 output: `36 * 2048` | 73,728 |
| FP32 intermediate: `36 * 4096` | 147,456 |
| Total static payload | **1,226,752** |

Adding the **111,616-byte base implied by the log** gives exactly **1,338,368**. Factory lines 540-603 place these static CBs across the program's rectangular core grid. `tt_metal/impl/program/program.cpp:1925-1932` rejects any occupied L1 address below that end. This explains why a large public tensor can cause a failure despite an unchanged compact matmul shape and successful single-layer execution.

## Smallest model-local experiment and threshold rationale

Use the existing inherited `_public_rows` policy before adding an override implementation:

```bash
bash models/autoports/qwen_qwen3_8_27b/tests/run_multichip_experiment.sh stress_stack_batch32_dram --layer 0 --batch 32 --length 257 --stack --repeats 10 --stress-iterations 100 --policy '{"public_dram_batch":2}'
```

`optimized_decoder.py:445-452` already converts the compact tensor to the selected memory target **before** reshaping into the expanded public form. Thus `public_dram_batch=2` sends every B>1 public boundary to DRAM without first allocating its large public L1 form. Logical shape, row order, dtype, projection weights/fidelity, native attention, state updates, direct all-reduce, and supported batch/context remain the same. Compact packed normalization, residual arithmetic, and matmul inputs still use their existing L1 layouts. No host transfer is introduced.

Threshold **2** follows the structural point where one tile per public batch element exceeds the compact row count. It covers the public **B1-B32 stack contract** consistently; B1 takes the unchanged existing branch. Threshold **32** is a narrower empirical repair for the reported failure, but leaves B2-B31 stack memory behavior to the same expanding public-L1 path. The two thresholds execute identical branches at B32, so a B32 pass alone cannot establish which threshold has the better overall tradeoff. Threshold 2 may add DRAM traffic at small batched boundaries; B2/B3/B8/B16 stack correctness, trace and timing controls must measure that cost.

After the focused override passes, the proposed implementation is only `public_dram_batch=2` in the **multichip** construction defaults, before explicit policy overrides. The frozen single-chip default remains unchanged. Do not first disable packed MLP, change precision, shrink the batch, or add forced tensor deallocation; those would mix different hypotheses into this placement test. If placement alone still fails, use the new failing allocation site to choose a narrower follow-up, such as reducing the packed gate block's static weight buffering.

## Verification status

The parent completed reset 13 and device listing successfully and is running the override control after its mesh smoke. No passing override result is claimed here. Acceptance requires the original B32 linear/full stack to complete normal eager decode, changed-input replay and **100 queued replays**, with output/state equality, finite values, per-user/baseline PCC and cache ownership checks. B2/B3/B8/B16 stack controls are planned; B1 should retain its performance path. Existing saved single-layer results are supporting context, not proof of the repaired stack.

**Verdict:** expanded public-L1 storage and caller-held stack lifetimes are verified by source; native scratch arithmetic exactly matches the failure. Their contribution to the reproduced collision is the focused hypothesis under test. The one-line multichip placement default should be retained only after the parent records the passing override and neighboring regressions. No public capability reduction is justified.

## Hardware resolution

`stress_stack_batch32_dram.json` passes with public_dram_batch=2, including
100 evolving eager calls and100 queued trace replays without host barriers
between calls. Shared workspace identity is asserted. Full raw state and
outputs are bitwise equal; final trajectory output PCC0.99998122 and recurrent
PCC0.99982654 versus the frozen optimized single-chip baseline. This supersedes
the pending stress status above. Final default adopts the batched DRAM boundary;
expanded B2/B3/B8/B16 stack regressions and final default stress are rerun.
