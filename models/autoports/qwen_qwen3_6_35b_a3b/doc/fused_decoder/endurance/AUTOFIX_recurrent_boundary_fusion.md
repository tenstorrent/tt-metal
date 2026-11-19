# Fold recurrent public-boundary rounding into state decay

Source-only candidate, no live-source edits or hardware execution. Patch: `recurrent_boundary_cast_fusion.patch`. Base fused source SHA256: `f7b052549c01fa1a2cf6c6c074d65d0589fe71ecd308eaa9b5e32ed0994cc193`. The patch changes only `_recurrent_step` and its decode caller, preserving the current projection-product L1 placement.

## Exact boundary retained

Existing public decode computes `S_bf = BF16(S_fp32)`, `S_wide = FP32(S_bf)`, then `decayed = S_wide * decay`. The proposed multiply has the unchanged FP32 state as input A, with the existing `round_widen_chain` as input-A preactivations. It therefore evaluates `decayed = round_widen(S_fp32) * decay` before any recurrence subtraction, product or reduction. It does not fold rounding into the output or round the new state at a different point.

The new `round_state=False` keyword keeps existing direct callers and internal non-boundary decode behavior unchanged. The decode caller passes `cache_boundary`. Prefill's explicit sequential state loop is unchanged. Canonical recurrent cache allocation, final copy, and public BF16-between-calls semantics are unchanged.

## Source proof

- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp:573` emits the same `typecast_tile<FP32,BF16>` / `typecast_tile<BF16,FP32>` functions for TYPECAST chain elements.
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h:273` (target Blackhole), also `wormhole_b0/.../ckernel_sfpu_typecast.h:285`, implements FP32→BF16 with `(bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000` and stores the result in FP32 destination. This is round-to-nearest-even with BF16 mantissa bits, not a BF16 output-pack approximation. The argument applies to the finite state values required by the correctness gates; it is not a new claim about arbitrary NaN payload preservation.
- `tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h:158` makes BF16→FP32 a no-op in SFPU registers; the already rounded numerical value is preserved. The normal separate typecast path uses the same Typecast primitive (`copy/typecast/device/kernels/compute/eltwise_typecast.cpp`) with FP32 preservation enabled by `copy/typecast/typecast.cpp`.
- `eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp:19` copies A into FP32 destination, executes all A activations, then packs the activated operand into the intermediate CB before the binary multiplication consumes it.
- `eltwise/binary_ng/device/binary_ng_program_factory.cpp:1197` selects the intermediate c3 format from the actual A dtype for SFPU. A is FP32 here, so the rounded/widened values stay FP32. Lines1370–1400 configure FP32 unpack-to-destination for both original and intermediate operands. This avoids the BF16-input activation packing issue previously demonstrated for Z/SiLU.

These are source-level reasons for numerical equivalence, not substitute device evidence. Installed binary and current source can differ; the retained native fingerprint and actual full-stream gates remain authoritative.

## Traffic and existing evidence

State shape `[1,32,128,128]` contains 524288 elements: 2 MiB FP32 and 1 MiB BF16. The two removed casts read/write 6 MiB total at their external tensor boundaries (2+1 MiB and 1+2 MiB). The decay multiply still reads 2 MiB of state and writes 2 MiB of decayed state; the proposed extra activation CB traffic is local, so 6 MiB is removed external traffic, not a guaranteed latency saving.

In the older `profile_qz_norm/tracy/layer_0/decode_op_sequence.csv`, the two state casts were steps129/130 at9.561/9.062us, totaling18.623us per replay. The root is collecting the current profile; do not project this historical time directly onto the current candidate. Fused preactivation can add compute overhead and must beat the best correct traced baseline.

Require the existing three-seed endurance, finite/cache-boundary/determinism checks and paired warmed traced timing. If outputs differ, first compare preactivation round/widen values and decayed state from identical inputs against the original two-cast path. Do not change thresholds.

## Separate follow-up: recurrent intermediates in L1

This placement experiment is source-legal but should be tested separately after cast fusion:

1. Place the decay multiply output in interleaved L1. It reads canonical state from DRAM; no persistent state movement is needed.
2. `_recurrent_sum` must pass `memory_config=products.memory_config()` to `fast_reduce_nc`; otherwise its intermediate/final stages default to DRAM. Permute and reshape inherit the product placement.
3. `decayed * k` inherits L1 from decayed. However `v - memory` inherits DRAM from v, and `k * delta` inherits DRAM from k. For a fully L1 recurrence, explicitly set L1 on those two operations (and on `new` if clarity is preferred). The final `new * q` inherits L1 from new.
4. Leave the canonical state in DRAM and retain the final `ttnn.copy(new, state["recurrent"])`. Its shape already matches the destination. Decode output can stay L1 through its small norm/gate temporaries, but any downstream placement dispatch change needs measured validation.

Each large state/product is2MiB across the interleaved banks; several are simultaneously live. `_recurrent_sum` geometry `[8,16,32,128]` has no added padding: first reduction output is128KiB and final output16KiB. No dtype/reduction-order change is needed. L1 capacity is not proven by these sizes: current projection products also use L1, and trace allocation/liveness may overlap or fragment. Scope this control to `_recurrent_step` (decode), not the large prefill sequential loop. A helper memory argument with defaultDRAM or inference from the product can preserve current prefill behavior.

This candidate has no global datatype sweep, relaxed accuracy, host fallback, shared-native changes, or context-contract change. Like the rejected norm-L1 experiment, merely reducing DRAM traffic is insufficient: keep only if traced latency improves.

## Standalone precision probe

`tests/recurrent_boundary_precision_probe.py` compares the old two-cast multiply and fused input-A chain bitwise on full `[1,32,128,128]` FP32 tensors, with `[1,32,1,1]` decay broadcast. Cases are archived raw `tt_values.new_state` from seed154 position5991, random FP32 values scaled per captured head, and both signs/even-odd BF16 mantissas at and immediately around halfway points. Each case uses captured decay and a second decay containing exact ones plus varied fractions. The captured new state and decay form a same-input arithmetic diagnostic, not a claim of replaying the captured token recurrence.

It also compares standalone round/widen chains with separate casts and CPU BF16 RNE. FP32 product CPU comparisons are diagnostic; the final hard local equivalence requirement is zero complete old-vs-fused multiply FP32 code differences and finite outputs. Standalone intermediate comparisons remain reported diagnostics; see the signed-zero finding below. No latency claims or decoder acceptance substitution.

CPU command `python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.recurrent_boundary_precision_probe --oracle-only --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/recurrent_boundary_precision_cpu.json` passed. All three cases contain524288 finite values, explicit integer RNE matches CPU BF16, and TTNN was not imported. Device fields are null. Provenance is written at start; device execution opens only physical chip0 and closes in finally.

Root device command (remove oracle-only):

```sh
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.recurrent_boundary_precision_probe --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/recurrent_boundary_precision_device.json
```

## Initial device finding and diagnostic-contract correction

Preserved initial failure byte-for-byte as `recurrent_boundary_precision_device.initial_intermediate_gate_failure.json` plus its original provenance; SHA256 `18330d21e6c46d3e54d92a286b725193a6a17f6dc3d0ff6ab5be263eb18255c0`. The original `recurrent_boundary_precision_device.json` was also left unchanged. Its `device_comparison_pass=false` came from a stronger standalone intermediate assertion than the proposed rewrite's complete expression contract.

All six complete old-vs-fused multiply comparisons had **zero FP32 code differences**. Captured and random standalone round comparisons were exact. The adversarial tie case had exactly one standalone code difference at inserted negative zero: separate typecasts produced +0 (`00000000`), while the round/widen chain retained -0 (`80000000`). Both subsequent multiply paths produced the same output bits. Therefore do not claim every intermediate bit is identical, even though the complete tested rewrite is bitwise identical.

CPU product differences were also identical for old and fused paths: captured decay gave15848/16383/14154 differences for captured/random/tie inputs respectively, and unit/varied decay gave0/0/1. Their examples contain flushed subnormal values and the signed-zero distinction; CPU arithmetic is not the equivalence oracle for this device-to-device rewrite. Updated metrics explicitly count signed-zero and reference-subnormal-to-zero differences, and add a separately labeled CPU model flushing subnormal inputs/outputs and canonicalizing zero positive. That model is diagnostic and does not waive any old-vs-fused difference. Its full explanation remains to be verified on the new device report.

The revised diagnostic gate checks only the requested complete expression: every old-vs-fused multiply output must be finite and FP32 bitwise identical. Standalone round-chain and CPU comparisons remain visible without being mislabeled as rewrite failures. Decoder PCC, full-stream, context and determinism gates are unchanged. CPU-only v2 validation passed without importing TTNN; the root will run a fresh device artifact filename.
