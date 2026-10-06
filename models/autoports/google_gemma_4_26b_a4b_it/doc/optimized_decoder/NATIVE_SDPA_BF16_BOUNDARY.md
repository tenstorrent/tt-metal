# Native SDPA BF16 output boundary candidate

Status: **unapplied, source-legal candidate**. No runtime edits or hardware
execution. `native_sdpa_bf16_boundary.patch` changes exactly one line:

```diff
-        return ttnn.typecast(result, ttnn.float32)
+        return result
```

Base runtime SHA256:
`365be21918879bdf94514fd344c352db2e0e898979d059880ef9cd261de5457c`.
Candidate SHA256 for that base:
`1259635843c1227699e6d9c56b817d48e2f0ce0996a32b02b233b18d0e9c4520`.
Exact metadata and CPU wiring checks: `native_sdpa_bf16_boundary.json`.

## Source constraints and complete consumer path

Both model attention kinds use GQA. Native SDPA requires BF16 Q
(`ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_device_operation.cpp:412-418`)
and creates output with the query's dtype (`:491-501`). The producer therefore
already returns BF16. `NativePagedAttention.__call__` currently promotes that
result to FP32 only after SDPA has completed
(`tt/optimized_decoder.py:934-948`). Neither the precision nor the accumulation
inside SDPA changes in this candidate.

The only runtime consumer is the inherited decode method, which immediately
passes that result to `project(..., True)`
(`tt/fused_decoder.py:642-643`; equivalent base path in
`tt/decode_attention.py:92-93`). There is no intervening norm, residual add,
cache update or other FP32 consumer. All reachable projection branches support
BF16 input and explicitly retain FP32 linear output:

| Projection branch | Source and dtype handling |
| --- | --- |
| Selected explicit output program | `tt/optimized_decoder.py:1034-1058`: height-shard, concat heads, reshape, convert to L1 interleaved, linear with `dtype=ttnn.float32`. No input cast. The output compute config keeps `fp32_dest_acc_en=True` at`:998-1005`. |
| Inherited sharded-head projection | `tt/fused_decoder.py:538-564`: the same height-shard/concat/reshape structure, followed by linear with FP32 output and the original compute config. |
| Generic decode projection | `tt/decode_attention.py:57-66`: `concat_heads` then linear with FP32 output. `models/demos/gemma4/tt/attention/operations.py:535-574` reshards, concatenates and converts memory without a dtype cast. |

`nlp_concat_heads_decode` explicitly admits **both BF16 and FP32**, requires
tiled height-sharded input, and checks the shard's dimensions against padded
heads and head width
(`ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/nlp_concat_heads_decode_device_operation.cpp:30-68`).
Both head widths256/512 use16 logical heads padded to32 and one logical user,
matching the current one-core input shard. Its output spec preserves input dtype
(`:105-110`). The program derives tile size and row byte offsets from the input
format/element size, rather than hard-coding FP32
(`.../nlp_concat_heads_decode_program_factory.cpp:30-38,82-89,119-124`). It is a
byte-copy head rearrangement; the candidate does not insert numerical work there.

The linear/matmul contract accepts floating-point tiled inputs independently
(`ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:34-57`),
including BF16 activations with the selected BFP8 projection weights. Explicit
`output_dtype` survives independently of input dtype (`:2823-2860`). The selected
program and compute config remain supplied by the caller. The1D factory sizes
input buffers using the input format while retaining FP32 intermediates when
FP32 destination accumulation is enabled
(`device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp:141-145,776-810`).
No required API/layout adaptation or compensating cast was found.

## Expected semantics and measurement scope

Before: native BF16 result → FP32 promotion → FP32 head concat/layout transfers
→ FP32-input output projection → FP32 output.

Candidate: native BF16 result → BF16 head concat/layout transfers → BF16-input
output projection → FP32 output.

The current promotion cannot recover precision lost before native SDPA produced
BF16. Removing it adds **no new rounding of those values** before the matmul;
the represented input values are the same finite BF16 values. Output storage,
FP32 accumulation, chosen fidelity, weight dtype, program geometry, residual
interface, positions and cache handling stay unchanged. Hardware input formats
and kernel selection can still affect arithmetic, so bitwise equality or a
passing whole-layer PCC is not asserted from source alone. Keep the unchanged
.995 actual-input gate and normal final-policy stress/contract checks.

This candidate removes one typecast and retains BF16 across the real native
attention-to-projection boundary. The existing output probe's BF16 control at
`tests/probe_optimized_output.py:209-214` first receives the already promoted
FP32 attention result, concatenates FP32 heads, then casts the projection operand
back to BF16. That control can inspect input-format accuracy but does **not**
measure this movement change. Test the one-line production candidate without
that compensating probe wrapper. No latency or bandwidth improvement is claimed.

## Apply and reproduce

After the parent completes the active hardware run, inspect the base hash and
apply the checked patch from the repository root:

```bash
sha256sum models/autoports/google_gemma_4_26b_a4b_it/tt/optimized_decoder.py
git apply --check models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/native_sdpa_bf16_boundary.patch
git apply models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/native_sdpa_bf16_boundary.patch
```

Use the same audited actual-input command and cumulative policy as the immediate
baseline, changing only result/log destinations. Rewarm and recapture because
the dtype/program signatures change. Final profiler rows should show BF16 head
concat and output-projection input, FP32 projection output, and no standalone
native-result FP32 typecast. The prior final result remains the baseline until
these checks pass.

If rejected, restore just this change with:

```bash
git apply -R models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/native_sdpa_bf16_boundary.patch
```

Validation performed: syntax compilation, `git apply --check`, and12 CPU-only
before/after wiring checks covering both head widths and all three projection
branches. Every candidate branch carries BF16 through concat and linear input,
returns FP32, and removes exactly the result-promotion call. These are mocked
call-graph checks, not a device test. Black produces the same pre-existing
normalization-conditional formatting delta in both complete files; the one-line
candidate introduces no additional formatting difference. The active runtime
was left unchanged.

## Hardware decision

Applied and measured on actual 4096/128 inputs for both kinds; exact commands in `bf16_boundary_commands.json`, paired results in `native_sdpa_boundary_results.json`. Prefill and every decode PCC are unchanged. Full attention improves from1170.943 to1160.806us traced host median. Sliding1073.880 versus1074.636us is within observed timing variation. Retain the BF16 boundary: no additional rounding, one conversion removed, full-attention gain. Final combined default validation remains required. These are host timings, not device telemetry.
