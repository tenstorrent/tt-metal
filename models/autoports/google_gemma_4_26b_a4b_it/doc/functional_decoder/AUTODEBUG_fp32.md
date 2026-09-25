# AutoDebug: remaining Float32 diagnostic drift

Source-only investigation, 2026-09-25. No hardware access, runtime experiments,
implementation edits, or acceptance changes. Source citations below are
repository-relative `path:line` locations. The findings identify real precision
boundaries; their responsibility for slot 17 still needs a focused control.

## Evidence and conclusion

The corrected residual diagnostic fails sliding slot 17 at final PCC 0.98250818.
`batch32_sliding_residual_routes.json` records residual PCC 0.999952197 and both
CPU-on-TT-residual and TT routing replace HF expert 76 with 108. The exact-head-norm
retry raises that residual PCC to 0.999985218, still replaces 76 with 108, and
introduces route changes at slots 5 and 27. Slot 8 in the former run is a separate
warning: CPU-on-TT-residual replaces expert 5 with 41 while TT retains 5.
`AUTOFIX_decode.md` and `HF_PRECISION_CONTROLS.md` establish that oracle prefix
cache does not repair slot 17, while HF with only BF16 cache/RoPE-table rounding
preserves all 64 decode route sets.

The diagnostic is **not an IEEE-FP32 attention/residual oracle**. Its strongest
unresolved source issue is RoPE: Float32 storage and `--rope-exact` do not remove
its BF16 compute/product boundaries. Matmul and RMSNorm also narrow Float32 FPU
inputs. These are narrower explanations to test before invoking unavoidable MoE
instability.

## Source findings

1. **Multi-tile RoPE ignores FP32 accumulation on core group 1 and stores products
   in the table dtype.**

   `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/rotary_embedding_program_factory.cpp:812`
   explicitly preserves `ComputeConfigDescriptor{}` for group 1; only group 2
   receives requested fidelity/FP32 accumulation at line 828. The default has
   `fp32_dest_acc_en=false` in `tt_metal/api/tt-metalium/program_descriptors.hpp:102`.
   The multi-tile factory is selected whenever width exceeds 32 at factory
   line 897. Its `c_25`/`c_26` sine/cosine product buffers use the respective table
   dtype at lines 602–621. The compute kernel actually packs both products there
   before adding them:
   `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/compute/rotary_embedding.cpp:149–170`.
   Thus BF16 tables cause BF16 **products**, beyond the table-only rounding tested
   by HF. Float32 tables alone still leave group 1 with FP32 accumulation disabled.

   This group is relevant to the diagnostic's small per-slot decode: work is
   partitioned by padded tile rows at factory line 51, and work that fits on the
   grid is assigned entirely to group 1 in `tt_metal/common/work_split.cpp:348–364`.
   `tests/probe_policy.py:146–155` only sets the public compute config, so it cannot
   override the factory's group-1 default. This is a concrete lowering defect,
   although its contribution to the remaining route change is unmeasured here.

2. **FP32 destination accumulation preserves sums, not full Float32 FPU operands.**

   `tt_metal/jit_build/genfiles.cpp:822–828` selects `Tf32` as the conditional
   unpack destination when FP32 accumulation is enabled and the buffers are
   Float32 or B-family. `tt_metal/jit_build/data_format.cpp:151–158,215–221` maps
   Float32 to that format unless an explicit nondefault unpack-to-DEST mode is
   supplied. This is not merely a Wormhole comment: Blackhole documents SrcA/B
   as TF32 and direct DEST as FP32 in
   `tt_metal/tt-llk/tt_llk_blackhole/common/inc/cunpack_common.h:283–327`.
   Matmul feeds operands through SrcB/SrcA:
   `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_unpack_AB_matmul_api.h:21–24,78–79`.
   The LLK golden model truncates the lower 13 mantissa bits for Float32→TF32 at
   `tt_metal/tt-llk/tests/python_tests/helpers/golden_generators.py:854–860`.

   This applies to the diagnostic's Float32 normalized-input QKV projection,
   Float32 Q·BF16 K scores, Float32 probability·BF16 V, and Float32-input O
   projection (`tests/probe_policy.py:85–103`,
   `tests/probe_fp32_attention.py:60–68`). HiFi4 does not recover mantissa bits
   already lost on unpack. BF16 weights/cache are already representable in
   TF32; the newly computed Float32 activations/probabilities are the relevant
   operands. This boundary alone is a hypothesis, not a demonstrated repair.

3. **“Exact” RMSNorm still crosses Src-register precision boundaries.**

   Unmodified per-head normalization supplies no compute config
   (`models/demos/gemma4/tt/attention/operations.py:149–152`); RMSNorm defaults to
   approximate math and FP32 accumulation disabled regardless of Float32 input
   (`ttnn/cpp/ttnn/operations/normalization/rmsnorm/rmsnorm.cpp:16–19,62`). The
   explicit diagnostic config fixes those defaults, but its ordinary RMSNorm
   path still uses FPU square, reduction, variance-plus-epsilon and two scale
   multiplications (`ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm.cpp:260–301,334–387`).
   Its factory deliberately pins Float32 consumer buffers to UnpackToSrc, with
   direct-DEST exceptions for Welford aliases and a large-tensor accumulator:
   `ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_op_multi_core.cpp:883–926`. Welford is explicitly
   excluded for RMSNorm at line 437. Consequently FP32 intermediates are narrowed
   again when consumed. The epsilon buffer itself is BF16 at line 511.

   Do not generalize this to every Float32 TTNN operation. Equal-Float32 binary
   ADD/SUB/MUL select SFPU in
   `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp:49–63`
   and use direct-DEST unpack in `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp:1205–1241`.
   Generic Float32 sum/mean/max also default to accurate SFPU with FP32
   accumulation (`ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op.cpp:123–168`,
   `ttnn/cpp/ttnn/operations/reduction/generic/generic_reductions_nanobind.hpp:179–203`). Therefore the custom attention's
   Float32 max/sum are not automatically implicated by the FPU matmul finding.

## Focused next controls, in order

1. **Same-input RoPE:** capture Q/K immediately before/after RoPE, then compare
   the existing op with CPU HF arithmetic on those exact inputs and exact BF16
   table values. A device diagnostic can cast both input and tables to Float32,
   repeat tables to identical logical shapes, and compute
   `x*cos + concat(-x[..., D/2:], x[..., :D/2])*sin` with equal-dtype SFPU binaries.
   Preserve table values by device-casting BF16 rather than uploading new tables.
   Start with decode only, then apply the verified control to prefill/cache.
   Leave the fast flag unset for Float32 add/sub: explicitly passing `False`
   is rejected for non-BF16 outputs at `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp:35–41`.

2. **Same-input norm localization:** compare input norm, per-head norm and
   post-attention norm individually against CPU RMSNorm on the exact TT input.
   If causal, test an all-Float32 diagnostic composition using SFPU square,
   accurate `mean`, rsqrt and equal-dtype scaling; compare rsqrt independently.
   Repeat weights/scales as needed to remove mixed-dtype/broadcast ambiguity.
   This tests the FPU path; another fused RMSNorm compute flag does not.

3. **TF32 operand control:** for the earliest remaining discrepant matmul,
   compare TT against CPU FP32 using identical operands, and against CPU with
   Float32 operands bit-masked by `0xffffe000` before multiplication. Record max
   error, relative L2 and routing-logit margins, not PCC alone. Follow with a CPU
   substitution at that single boundary; do not replace the entire residual and
   infer a specific cause. Preserve cache/page-table policy throughout.

4. **Confirm broad behavior after localization:** rerun the original batch32
   sliding/full checks and request-reuse/trace checks unchanged. Slot-17 recovery
   alone is insufficient because the present policy already contains route-error
   cancellation (slot 8) and exact-head-norm changes moved failures.

The paged diagnostic's scale omission is not a mismatch for this caller:
`tt/decode_attention.py:87` passes 1.0. At S33 its causal/sliding mask and GQA
grouping in `tests/probe_fp32_attention.py:52–68` match the intended semantics.
The optional BF16 attention output remains another explicit rounding boundary
at line 69; its existing Float32-output A/B did not repair the failure.

No build was needed: only this report was added. All proposed controls remain
unrun by this investigator; hardware ownership stays with the runtime agent.
