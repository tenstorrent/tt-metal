# AutoFix diagnosis: rare 512-step GDN failures

Source/CPU inspection only; no device execution or main decoder edits. Evidence uses immutable `candidates/combined30.py` and `candidates/boundary32.py`, not the changing main module. All gates remain PCC >= .995 and BFP8 default weights.

## Observed failure pattern

| Artifact | Minimum PCC | Worst step,user | Cells below .995 |
|---|---:|---|---:|
| `combined30_sequence.json`, 128 steps | .995126681 | 63,16 | 0/4096 |
| `combined30_512.json` | .983038151 | 240,7 | 13/16384 |
| `boundary32_sequence.json`, 128 steps | .995382186 | 63,16 | 0/4096 |
| `boundary32_512.json` | .980315624 | 240,7 | 11/16384 |
| `state34_round.json`, boundary32 + cache rounding | .988453439 | 240,7 | 2/16384 |
| `state34_round_unsharded.json`, also unsharded explicit FP32 hidden norms | .992493177 | 209,25 | 1/16384 |

The first 128 per-user PCC arrays in each 512-step run are **exactly identical** to its corresponding 128-step run. This extends the same fixture; the failures are not caused by a different prefix or RNG. Every listed prefix PCC is .9999525037506221. The worst boundary32 user's PCC at steps 239/240/241 is .99996994/.98031562/.99996607; user29 at 410/411/412 is .99997486/.98583194/.99995587. This fits amplification at rare nearly cancelled readouts, not a continuous output collapse. It does not prove absence of accumulated state error.

## Ranked corrections and falsifiable controls

1. **Round persistent state at public call boundaries, after computing the current output.** HF `cache_utils.py:925-937` obtains cache dtype from the BF16 convolution state, allocates recurrent storage in that dtype, and `:989` copies the FP32 recurrence result into BF16. `modeling_qwen3_5.py:334-349` widens that rounded initial state on the next call. TT candidates persist FP32. The existing `_round_recurrent_state` hook implements BF16 -> FP32 into the original stable buffer after prefix and decode; enabling it is the correct isolated control. Do not round before computing this token's readout or after each internal prefill chunk: neither matches HF. The measured `state34_round` improvement from 11 failures to 2 supports a material contribution but disproves sufficiency. A fused writer could eventually implement BF16 rounding followed by FP32 widening without the two whole-state operations; keep the state identity and current-output ordering.

2. **Use actual unsharded FP32 hidden-norm control, then inspect the remaining failed head.** `accurate_sharded_norm=True` bypasses the explicit unsharded `_norm` branch for compact `[1,B,5120]`, despite `explicit_fp32_norm=True`. Thus that flag combination does not control against unsharded norm arithmetic. The new round+unsharded measurement leaves just one failed cell at 209,25 (.992493). This is stronger evidence than policy names that hidden-norm differences matter. Compare captured actual `input_norm`, `post_attention_norm`, and projections; the diagnostic now hooks `_norm`'s final return, not only its intermediate RMSNorm output. Retain sharded layouts only when their numerical implementation and timing pass the same sequence.

3. **HF q/k normalization has real intermediate BF16 boundaries, but the policy can be a no-op.** HF `l2norm` at modeling lines 240-243 executes BF16 product, returned sum, epsilon addition, rsqrt, and final multiply before widening. Boundary32 raw compact compute performs those stages in FP32. With `sfpu_normalized=False`, `_gdn_compact` never calls `_l2`: toggling `hf_qk_norm` alone changes nothing. A legal existing control sets `sfpu_normalized=True`, `hf_qk_norm=True`, `conv_packed_rm=False`, and `conv_rm=False`; the normalized interface requires TILE FP32 `[B,16,128]` q/k and TILE BF16 value. Packed RM `[B,1,10240]` cannot be passed to it. Alternatively implement the BF16 boundaries inside the raw compact kernel with exact RNE semantics. Do not silently change query scaling: HF normalizes first, widens, then multiplies by `128**-.5` in FP32. This control is pending and is not claimed to fix the sequence.

4. **Prefill state needs its own attribution; high prefix-output PCC is insufficient.** Both candidates use the same noncompact `_gdn` prefix path: native convolution applies SiLU before its BF16 output, native chunk GDN performs internal normalization, and final state stays FP32 unless the boundary hook is enabled. The boundary32 `conv_round_output` and `hf_gated_norm` changes affect compact decode, not this prefix. Native convolution's HF sum->BF16->SiLU boundary and HF q/k normalization therefore remain unmatched during the prefix. Prefix state mismatch can persist in slowly decaying heads while prefix outputs pass. First measure HF vs TT prefix state per head, including RMS gain, then compare the failed step using `HF inputs + TT old state` versus `TT inputs + HF old state`. A diagnostic-only run initializing TT from the exact HF prefix state can isolate inherited prefix error; it is not an acceptable runtime repair. If this removes failures, repair prefix convolution/normalization/state publication in the decoder and rerun the untouched fixture.

5. **Only after these controls, distinguish remaining SFPU recurrence math from input/state error.** The diagnostic's `tt_inputs_tt_state_fp32_norm` CPU oracle predicts the mathematical FP32 update from identical raw TT inputs/state. Its core/state error against TT isolates kernel approximation/reduction ordering, while `tt_inputs_hf_state_hf_norm` and `hf_inputs_tt_state_hf_norm` distinguish local input and stored-state contributions. Repeated SFPU `exp(g)` error could affect slow-state gain; source alone does not establish its size. Retaining DST is a performance/topology change unless measurements show a numerical change. The earlier explicit `UnpackToDestFp32` correction must remain; FP32 CB/DST declarations alone did not prevent prior TF32 operand conversion.

## Updated selected-step capture

`tests/run_decode_sequence_diagnostic.py` now supports both old `_delta_chunks` and module-level `_recurrent_sfpu_compact`. It splits packed RM Q/K/V, captures raw vectors before any external normalization, captures FP32 state before/after the kernel separately from published cache state, records no-affine RMSNorm and its following actual BF16 weight multiply, and records final hidden `_norm` returns. It uses eager diagnostic replay only, restores traced state, checks stable state identity, and retains exact RNG ordering and .995 assertions. Captured host reads are not performance evidence.

Use the existing serialized device wrapper and the exact boundary32 policy in `capture36_jobs.py`:

```bash
python -m models.autoports.qwen_qwen3_6_27b.tests.run_decode_sequence_diagnostic \
  --layer 0 --batch 32 --steps 512 --target 240:7 --target 411:29 \
  --dump-intermediates models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/capture36 \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/capture36.json
```

Also target 209:25 when diagnosing the round+unsharded candidate, since the worst cell changed. The parent has queued hardware captures; this report does not claim their completion. CPU validation: `python -m py_compile` passed, and an AST-isolated fake-tensor harness exercised packed recurrence with both affine and no-affine RMS branches, checking required captured keys and `[selected_users,1,48,128]` vector / `[selected_users,48,128,128]` state shapes. That harness imported Torch only, not TTNN or its device runtime.
