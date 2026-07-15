# AutoDebug: Stage 08 BFP8 activation/residual cache-update failure

## Headline finding

The `residual_activation_bfp8` smoke exposes a caller-boundary dtype bug, not a limitation of BFP8 KV-cache storage. The model passes native BFP8 K/V tensors to `paged_update_cache`, but that operation intentionally accepts only FLOAT32/BFLOAT16 update inputs and performs the final repack into a FLOAT32/BFLOAT16/BFLOAT8_B/BFLOAT4_B cache itself.

This finding is high confidence from the exact runtime validator failure and matching source contracts. No hardware reproduction was attempted during this inspection.

## Evidence-ranked causal chain

1. **Observed failure (direct evidence).** The smoke fails during the 149-token prefill in the non-aligned sliding-cache tail. The stack reaches `MultichipDecoder._prefill_attention_tp`, inherited `OptimizedDecoder._fill_bounded_sliding_cache_exact`, and then the first `paged_update_cache` call. The validator reports that the input is neither FLOAT32 nor BFLOAT16 (`doc/datatype_sweep/smokes/group_a.log:82-116`).

2. **The cache dtype is valid; the update-input dtype is not (direct contract evidence).** Non-fused update accepts cache dtype FLOAT32, BFLOAT16, BFLOAT8_B, or BFLOAT4_B (`ttnn/cpp/ttnn/operations/experimental/paged_cache/device/update_cache/paged_update_cache_device_operation.cpp:41-48`) but separately requires update input FLOAT32 or BFLOAT16 (`:293-297`). Fused decode update has the same cache set and update-input restriction (`device/fused_update_cache/paged_fused_update_cache_device_operation.cpp:59-65,203-207`). The binding documentation explicitly states that the op owns the final low-precision cache repack and update K/V must not be pre-cast to the cache dtype (`paged_cache_nanobind.cpp:24-41`). The program factory derives separate input and cache formats and uses the cache format for its output CB (`paged_update_cache_program_factory.cpp:89-105,228-235`).

3. **Stage 08 deliberately introduces BFP8 at the model boundary (direct model evidence).** The candidate independently sets `activation_dtype` and `residual_dtype` to BFP8 (`configs/residual_activation_bfp8.json:1-6`). `embed_tokens` casts the gathered embedding to `activation_dtype` (`tt/model.py:473-508`). Each multichip layer casts its final combined residual to `residual_dtype` (`tt/multichip_decoder.py:1240-1246`).

4. **BFP8 reaches K/V without an explicit widening conversion (direct lowering evidence).** The input RMSNorm does not request an output dtype (`tt/multichip_decoder.py:1192-1215`); layernorm defaults its output to the input dtype (`ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_device_operation.cpp:423-445`). Prefill QKV projection likewise does not request an output dtype (`models/demos/gemma4/tt/attention/operations.py:43-45`), and matmul defaults output dtype to input A (`ttnn/cpp/ttnn/operations/matmul/device/config/matmul_program_config.cpp:995-1006`). Head splitting, per-head normalization, and RoPE have no widening request before cache fill (`tt/multichip_decoder.py:1094-1126`). The observed validator failure confirms that this static path produced a packed low-precision update tensor.

5. **The earliest inconsistent calculation is in the bounded-tail caller (direct source contradiction).** `_fill_bounded_sliding_cache_exact` says it uses “BF16 token updates,” but it slices, permutes, shards, and submits K/V without a BF16 cast (`tt/optimized_decoder.py:695-748`). The validator is correctly rejecting an input that contradicts the helper's stated contract.

6. **Fixing only prefill leaves a latent decode failure (direct path evidence).** Multichip decode submits post-norm/post-RoPE K/V directly to `paged_fused_update_cache` for full attention and `paged_update_cache` for sliding attention (`tt/multichip_decoder.py:819-962`). Both callees enforce the same BF16/FLOAT32 input contract, so a BFP8 residual path that survives prefill would fail when traced decode reaches either branch.

## Smallest valid intervention

Keep activation/residual and KV-cache precision policies unchanged, and widen only K/V cache-update operands:

- In `OptimizedDecoder._fill_bounded_sliding_cache_exact`, create BF16 tail-update copies once after the tail permutation and slice/shard those copies for `paged_update_cache`. Keep the original low-precision K/V used by prefill SDPA unchanged. Deallocate both original and converted temporaries explicitly.
- In `MultichipDecoder._decode_attention_tp`, after K normalization/RoPE and immediately before cache-update sharding, typecast K and V to BF16 when needed. Use those tensors for both fused and modulo update branches. Q does not feed cache update and does not need widening.

This intervention matches the TTNN op boundary, covers non-aligned prefill and both decode branches, preserves the BFP8 activation/residual experiment, and does not change KV-cache allocation or advertised context capacity. Casting the whole attention input or all Q/K/V to BF16 would be legal but broader and more likely to erase the performance effect under evaluation. Changing the TTNN C++ validator would violate the documented kernel contract and is not warranted.

## Can activation and residual dtype be separated?

Yes. They are distinct resolved fields and are consumed at distinct boundaries: activation controls the embedding-to-stack tensor, while residual controls every decoder-layer return. The grouped runner also mutates them independently (`tests/run_datatype_same_weights_group.py:60-98`). However, the current combined candidate cannot attribute performance or accuracy to either field alone.

Without the cache-boundary repair, activation-only BFP8 can fail in layer 0, while residual-only BFP8 can propagate into attention from layer 1 onward. After the repair, separate activation-only and residual-only candidates would be valid if individual attribution is desired; the combined candidate remains a valid lower-precision policy.

## Focused static regression test

Prefer a small helper such as `_cache_update_input(tensor)` that returns BF16 unchanged, typecasts other floating inputs to BF16, and is called by both bounded-tail prefill and multichip decode. A no-device test can monkeypatch `ttnn.typecast` and use dtype-bearing fakes to verify:

- BFP8 K/V each trigger exactly one BF16 conversion;
- BF16 K/V remain identity/no-copy;
- bounded-tail prefill and both decode branches submit the helper result, not the original tensor;
- Q and the cache tensor are not converted.

An `inspect.getsource` ordering assertion can supplement this, but testing the extracted conversion helper is less brittle. Hardware validation must still rerun the reduced non-aligned 149-token prefill plus traced teacher-forcing decode because a static test cannot prove sharding/layout preservation or trace capture safety.

## Other observations and remaining uncertainty

- The grouped runner resets traces/cache before applying the next policy and then updates `model.config.activation_dtype` plus every layer's `residual_dtype` (`tests/run_datatype_same_weights_group.py:60-98`, `tt/generator.py:636-647`); stale candidate state does not explain this failure.
- The current smoke proves the bounded-prefill branch failure. The predicted decode failure is source-contract certain but was not separately observed because prefill aborts first.
- Static inspection does not establish the performance cost or accuracy of widening only update operands. Those require the authorized serialized hardware smoke after implementation.

## Second pass: independent decode QKV-head-split failure

### New observed symptom

After applying the cache-update boundary repair, the refreshed smoke passes the previous prefill blocker and reaches decode preparation. It then fails in `_prewarm_split_sampling_workloads` before trace capture, at `ttnn.experimental.nlp_create_qkv_heads_decode`, with `Unsupported data format` (`doc/datatype_sweep/smokes/group_a_autofix.log:74-118`). This is an independent, earlier decode boundary than the paged cache update.

### Exact contract and earliest divergence

`NLPCreateQKVHeadsDecodeDeviceOperation` accepts only FLOAT32 or BFLOAT16 tile-layout input (`ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/nlp_create_qkv_heads_decode_device_operation.cpp:27-42`). Its Q, K, and V output specs inherit the input tensor dtype (`:114-182`), so a BF16 fused-QKV input produces BF16 head tensors without another conversion.

The Stage 08 Multichip lowering currently creates the decode QKV projection without an explicit output dtype (`tt/multichip_decoder.py:835-879`). With a BFP8 inter-layer activation, the sharded input is BFP8, and matmul defaults output dtype to input A (`ttnn/cpp/ttnn/operations/matmul/device/config/matmul_program_config.cpp:995-1006`). The resulting fused QKV remains BFP8 through the interleaved copy/concat and is rejected when `split_qkv_heads_decode` calls the BF16/FLOAT32-only op (`models/demos/gemma4/tt/attention/operations.py:48-69`). The validator is again the messenger; the earliest inconsistent boundary is the omitted output dtype on the decode QKV projection.

### Revised smallest intervention

Make each Multichip decode QKV projection produce BF16 directly by supplying an explicit BF16 `dtype` to the `ttnn.linear` call in the `decode_weights` loop. This is preferable to producing BFP8 and allocating a separate full-width typecast immediately before the split: both satisfy the consumer, but the direct output contract avoids an extra copy and applies equally to packed and split projection topologies.

Widening this derived QKV temporary does **not** widen inter-layer residual storage. `_forward_device` retains the original `hidden_states` as the residual, passes a separately produced normalized tensor into attention, and independently casts the final layer result back to `self.residual_dtype` (`tt/multichip_decoder.py:1193-1254`). Attention weights also remain at their configured physical dtype; only the projection output tensor changes.

Because the head-split op propagates the BF16 input dtype to Q/K/V, downstream per-head norm and RoPE remain on a BF16 path: plain layernorm preserves input dtype (`ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_device_operation.cpp:441-445`), as does rotary embedding (`ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/rotary_embedding_device_operation.cpp:99-138`). Therefore `_prepare_cache_update_input(k/v)` in Multichip decode becomes an identity/no-copy check after this repair. It should remain as the cache-operation boundary guard and is still required by bounded-tail prefill, whose QKV projection remains intentionally low precision. Thus the second finding supersedes the first report's decode ordering and removes the expected decode conversion cost, but it does not invalidate the original bounded-prefill cache-update diagnosis.

### Focused static check for this pass

Add a source-level contract test that inspects `MultichipDecoder._decode_attention_tp` and asserts the QKV `ttnn.linear` explicitly requests the BF16 split-input dtype before `split_qkv_heads_decode`. Also retain the helper test from the first pass and assert that BF16 inputs return by identity. The serialized reduced smoke remains necessary to prove the explicit output dtype is accepted by the tuned DRAM-sharded matmul program and is trace-capture safe.

## Third pass: BFP8 packed-MLP circular-buffer overflow

### New observed symptom

The independent `mlp_bfp8_lofi` smoke reaches token-out decode prewarm and fails while compiling the packed gate/up `ttnn.linear`. The static circular-buffer end is 1,937,280 bytes, beyond Blackhole's 1,572,864-byte worker L1 (`doc/datatype_sweep/smokes/group_f.log:64-111`). The stack identifies `_TPOptimizedSharedMLP.__call__` packed M=1 projection at `tt/multichip_decoder.py:502-517`; this is not the down projection or a trace-replay failure.

### Concrete BFP4 versus BFP8 geometry

The TP-local Gemma MLP is square: hidden and local intermediate are both 5,376 elements, or 168 32-wide tiles (`tt/multichip_decoder.py:315-321`). The packed gate/up weight therefore has K=168 tiles and N=336 tiles. Construction fixes this MLP to 14 input/output storage cores with `gate_up_in0_block_w=12`; the resulting per-core K shard is 168/14=12 tiles and packed output storage is 336/14=24 tiles (`:711-724`, `_decode_program_config` at `:415-425`).

The DRAM-sharded matmul factory does not divide its weight-reader CB by those 14 storage cores. It assigns readers by the eight Blackhole DRAM banks and computes `per_core_N_in1_sender = ceil(336/8) = 42` (`ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:91-145`). With K block 12 there are 168/12=14 blocks, so input A is double-buffered and weight input B is triple-buffered (`:181-212`). BFP8_B, BFP4_B, and BF16 tile sizes are 1,088, 576, and 2,048 bytes respectively (`tt_metal/tt-llk/tests/python_tests/helpers/llk_params.py:469-473`). The packed output remains BFP8, while `packer_l1_acc` makes the multi-block intermediate BF16, forcing separate output/intermediate CBs (`matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:181-192,553-594`).

The failing BFP8 static payload is therefore exact:

| CB | Calculation | Bytes |
|---|---:|---:|
| input A | 2 x 12 x 2,048 | 49,152 |
| BFP8 packed weight | 3 x 42 x 12 x 1,088 | 1,645,056 |
| BFP8 output | 42 x 1,088 | 45,696 |
| BF16 accumulator | 42 x 2,048 | 86,016 |
| static payload | sum | 1,825,920 |

`ProgramImpl` reports the CB region end rather than payload size (`tt_metal/impl/program/program.cpp:1481-1492`). The observed end implies the 111,360-byte default-unreserved base: 111,360 + 1,825,920 = 1,937,280, exactly matching the failure.

For baseline BFP4, only the packed-weight CB changes: 3 x 42 x 12 x 576 = 870,912 bytes. Its payload is 1,051,776 bytes and region end is 1,163,136 bytes, leaving 409,728 bytes below the L1 limit. This accounts for the BFP4 baseline passing with identical cores, K block, BFP8 output, and accumulator policy. The failure is therefore dtype-dependent program geometry, not a physical prohibition on BFP8 MLP weights.

### Smallest dtype-aware adjustment and prediction

Keep 14 cores and packed topology, but use `gate_up_in0_block_w=6` when `mlp_gate_up_weight_dtype` is BFP8; retain 12 for BFP4. Six is the largest fitting legal divisor of the 12 K tiles/core. Values 7-9 would fit the byte budget but violate the model helper's per-core divisibility check, while 12 overflows (`tt/multichip_decoder.py:415-425`); TTNN independently requires total K tiles to divide the block width (`matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:1006-1013`).

For BFP8 and block 6, the predicted region end is:

- input A: 2 x 6 x 2,048 = 24,576 bytes;
- packed weight: 3 x 42 x 6 x 1,088 = 822,528 bytes;
- unchanged output plus intermediate: 45,696 + 86,016 bytes;
- payload: 978,816 bytes;
- region end: 111,360 + 978,816 = **1,090,176 bytes**, leaving **482,688 bytes** of L1 headroom.

This doubles K-block iterations from 14 to 28 and may reduce speed, but it preserves one packed gate/up matmul and changes less topology than falling back to two separate projections. Increasing `decode_num_cores` alone does not address the dominant CB because weight-reader width is determined by DRAM banks, not storage-core count.

The BFP8 down projection does not need adjustment: N=168 gives 21 weight tiles per DRAM reader, and its activated input is also BFP8, so its existing block-12 predicted region end is 1,025,856 bytes. HiFi2 does not change these CB formats or sizes because FP32 destination accumulation remains disabled, so the same block-6 policy should unblock `mlp_bfp8_hifi2` and the canonical BFP8 control as well.

If block 6 exposes an unrelated kernel constraint in hardware validation, the evidence-backed fallback is separate gate and up projections at N=168, each of which has the same fitting geometry as the down projection. Static inspection finds no current validator contradiction for block 6, so this is not a rigorous physical blocker.

### Focused static regression test

Factor the dtype decision into a pure helper and assert BFP4 selects block 12 while BFP8 selects block 6. Instantiate `_decode_program_config(k=5376, n=10752, num_cores=14, in0_block_w=6)` without a device and assert K tiles/core divide exactly. A small host-side CB estimator using the factory formulas should assert the four exact region ends above: BFP4 packed 1,163,136; BFP8 packed block 12 at 1,937,280; BFP8 packed block 6 at 1,090,176; and BFP8 down block 12 at 1,025,856. The serialized reduced smoke remains necessary to confirm compile, trace capture, accuracy, and the actual performance cost.

## Fourth pass: precision-policy consumption validity

### Headline finding and scope

The shared single-device `OptimizedDecoder` violates its expanded precision
policy: prefill and decode projections all use the legacy shared attention
dtype, and both decode matmuls use one legacy attention fidelity. This would
misconstruct candidates that separate QKV/output dtype or fidelity.

The measured Stage 08 TP4 path does **not** share the defect.
`MultichipDecoder.from_state_dict` intentionally gives the demo prefill loader
`attention_weight_dtype`, then constructs decode QKV/split tensors from
`resolved_attention_qkv_weight_dtype` and decode output from
`resolved_attention_o_weight_dtype`. It also builds distinct resolved QKV and
output compute configs. Therefore the existing TP4 numeric rows are not
invalidated by the single-device loader bug.

### Full material-field audit

- Attention prefill, decode QKV, and decode output: TP4 physical loaders are
  distinct and correct; single-device decode was wrong as described above.
- MLP gate/up and down: `_TPOptimizedSharedMLP` constructs prefill, decode, and
  optional packed tensors from their respective policy dtypes. Its separate
  gate/up and down compute configs use their resolved fidelities.
- LM head and logits: `Gemma4FullModel` constructs physical LM-head shards from
  `lm_head_weight_dtype`, creates its compute config from LM-head fidelity, and
  requests `logits_dtype` at both sharded and unsharded projection boundaries.
- Layer exceptions: model construction selects the per-layer policy before
  calling `MultichipDecoder.from_state_dict`; the grouped runner likewise
  selects overrides by layer before updating compute/runtime policy.
- Activation/residual and CCL: embedding output reads `activation_dtype`, each
  layer return reads `residual_dtype`, and phase-specific communication fields
  reach their distinct prefill/decode all-reduce calls.
- KV cache: allocation reads `layer.policy.kv_cache_dtype`; incompatible cache
  dtypes are separated by the grouped runner's physical signature.
- Sampling: the primary sampler consumes the configured dtype. A secondary
  discrepancy was found in lazily created non-default-batch eager samplers,
  which hard-coded FP32. It does not affect the batch-1 Stage 08 rows or any
  current policy (all select FP32), but violates the policy API.

### Minimal intervention and evidence

Keep single-device prefill on the shared prefill field. Change only its decode
QKV/split and output tensor loaders to the corresponding resolved dtypes,
create separate resolved compute configs, and use them at their respective
matmuls. Replace the eager-sampler hard-code with the model-config dtype.

Policy-name summaries alone cannot prove physical consumption. Add an
additive TP4 runtime-summary map derived from constructed tensor `.dtype`
values for prefill/decode attention, prefill/decode MLP, and LM-head shards.
Source-only AST tests must separately assert the loader assignments and every
other material runtime boundary without importing TTNN while hardware is busy.

No measured TP4 config requires a numeric rerun for this defect. If
single-device optimized evidence existed, the exact affected policies would be
`attention_bfp8_hifi2`, `attention_qkv_bfp4_lofi`,
`attention_qkv_bfp4_hifi2`, `attention_output_bfp4_lofi`,
`attention_output_bfp4_hifi2`, and
`canonical_accuracy_bfp8_hifi2_bf16commcache`.
