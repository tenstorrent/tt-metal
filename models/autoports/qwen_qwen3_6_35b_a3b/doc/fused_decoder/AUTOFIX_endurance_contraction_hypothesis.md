# Endurance hypothesis: recurrence contraction versus sharded post-attention norm

Inspection/CPU-only report. No TTNN import, device operation, implementation edit, or test edit was performed. `D` denotes `models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder`. The inspected candidate is `D/mlp_norm_shard_dram.py.snapshot`; the comparison is `D/profile_carry.py.snapshot`.

**Finding:** the new `kᵀ @ state` contraction changes FP32 operand precision and reduction order in a persistent state update. It is a credible regression source, but the current final-output evidence cannot attribute the seven failures to it. The sharded post-attention norm is a distinct local-output hypothesis; it cannot independently cause recurrent-state drift in this fixed-input, single-layer endurance test. A contraction-only revert and a same-input norm comparison distinguish them without changing convolution or cache semantics.

The snapshot diff contains three changes: moving decode A bias addition from the complete packed projection into its A slice; replacing the recurrence memory contraction; and width-sharding the post-attention norm input. There is no shape/head-layout discrepancy in the replacement: `k` is `[1,32,128,1]`, transposed to `[1,32,1,128]`, multiplied with `[1,32,128,128]` state, producing `[1,32,1,128]` memory. The A-bias move uses the same FP32 add on A's 32 elements and is not an evident numerical-order change for those values.

## Supported arithmetic distinction

Candidate lines 271-286 compute:

```
S_decay = S * exp(g)
memory_new = matmul(kᵀ, S) * exp(g)
delta = beta * (v - memory_new)
S_new = S_decay + k * delta
o = sum(S_new * q)
```

`profile_carry.py.snapshot:282-285` instead computes `memory_old = sum((S * exp(g)) * k)` with elementwise multiplication and reduction. This matches the order in the installed HF fallback (`python_env/lib/python3.12/site-packages/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py:357-361`). Moving decay outside the sum is algebraically valid but changes floating-point rounding. Replacing the sum/product with matmul also puts the normalized FP32 key through the FPU operand path. FP32 destination accumulation does not preserve full FP32 input significands there: Blackhole LLK documents SrcA/SrcB as TF32-capable, while full FP32 is available in destination registers (`tt_metal/tt-llk/tt_llk_blackhole/common/inc/cunpack_common.h:276-286,311-340`). The matmul compute sources invoke `matmul_block`/`matmul_tiles`, rather than an SFPU FP32 multiply/reduce (`ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp:373-382`; alternative simple kernel `bmm_large_block_zm.cpp:76`). No device program report was collected here to identify the exact selected matmul factory.

An important limit on the hypothesis: **public decode already rounds incoming state through BF16** (`mlp_norm_shard_dram.py.snapshot:330-335`). Therefore the state operand is BF16-representable even though stored as FP32; ordinary TF32 input narrowing loses no additional state mantissa bits in this contraction. The normalized key remains FP32 (`313-320`), so it can lose precision. This is narrower than the previous native-FDR issue, which narrowed full-precision intermediate state. Internal prefill-tail calls with `cache_boundary=False` can have truly FP32 incoming state, but the 8192 sequential public decode calls do not have that condition.

For a one-step comparison starting with identical state/Q/K/V/g/beta, the first changed tensor is `memory`. If its difference is `dm = memory_new - memory_old`, the corresponding update difference, before rounding, is approximately `-beta * k * dm`, a rank-one correction per head. The immediate core-output difference is approximately `-beta * (qᵀk) * dm`. Later decode calls may differ after BF16 cache rounding; values straddling a rounding boundary can make a very small memory discrepancy persistent. This gives a precise signature and explains why globally small state error can still affect weak heads. The observed sparse failures alone do not establish this chain.

## Why the norm hypothesis has a different signature

Candidate lines 673-678 move only the post-attention residual into width-sharded memory before normalization and back to DRAM afterward. The shards are 32 cores with width 64 for the 2048-channel activation (`145-150`). This changes the reduction from the interleaved RMSNorm path to a cross-core sharded reduction, and can change intermediate rounding/order. The sharded kernel squares input (`layernorm_sharded.cpp:432-444`), normalizes with broadcast multiply (`577-593`), then applies gamma (`636-652`). The interleaved kernel uses a row-wise mean reduction (`layernorm.cpp:266-290`). There is no source evidence here of a missing width, wrong epsilon, or wrong gamma indexing for this aligned 2048-channel shape; arithmetic equivalence does not imply bit identity.

If this change causes a failure, input norm, Q/K/V/g/beta, updated recurrent state, core output, attention output, and the residual **must agree** between the candidate and a norm-only reverted control. The earliest difference is post-attention norm output, followed potentially by router logits/top-8 expert IDs and MoE output. Router discontinuities can turn a small norm difference into an isolated final-output PCC failure.

Crucially, the test supplies the same prerecorded random hidden-state token to each decode (`tests/stress_fused_decoder.py:31-37,92-107`); it does not feed the previous decoder output into the next call. Post-attention norm/MoE therefore cannot modify the recurrent state or future token inputs in this test. A norm-only change has no temporal error accumulation path here. It may expose rare routing-sensitive inputs later in a longer random sequence, but cannot explain a measured state divergence.

## What the existing CPU evidence establishes

I parsed `D/endurance/mlp_norm_shard_8192_0_results.json` and inspected its `.tensors.pt` with CPU PyTorch. The artifact contains prefix, tokens, HF final outputs, and TT final outputs; **no attention/state/norm/router intermediates**. Consequently no first-divergence attribution can be extracted from this artifact alone.

The seven failing positions are 3979, 5011, 6595, 7551, 7668, 7703, and 8157; all three repeats are identical. At the worst position 7703, contribution PCC is 0.983477155, but the immediately adjacent values are 0.9999107 and 0.9998434. Every failing position has passing immediate neighbors. Mean contribution PCC over the first 1024 decode tokens is 0.9997933; subsequent 1024-token block means are 0.9997407, 0.9997225, 0.9997487, 0.9997091, 0.9997014, 0.9997183, and 0.9996606. The final block's 1st-percentile PCC remains 0.9978343. This looks like sparse sensitivity on top of a small overall error shift, not wholesale state corruption. Both weak-head recurrent error and a local router threshold can produce that pattern. These statistics are descriptive, not a causal test.

## Minimal discriminating controls

1. **Contraction-only control:** keep the current candidate's carry handling, A-bias placement, input norm, post-attention sharding, and cache-boundary casts. Replace only the two `memory` statements at 281-283 with `ttnn.sum(ttnn.multiply(decayed, k), dim=2, keepdim=True)`. Run one identical 8192-token replay. Compare its full output series to the existing candidate and HF, rather than only the worst token. A global old-snapshot baseline changes too many factors to identify this contraction independently.
2. If the contraction-only control helps, capture one warmed step with identical incoming state and Q/K/V/g/beta, and record `memory_old`, `memory_new`, updated state before/after BF16 rounding, and core output. Compare CPU FP32 `sum((S*exp(g))*k)` to CPU FP32 `sum(S*k)*exp(g)` on the captured inputs to separate decay reassociation from device matmul input precision. If reassociation is negligible, test SFPU `sum(S*k)*exp(g)` as a precise control retaining the new ordering. No need to rerun MoE for this micro-comparison.
3. **Norm-only control:** retain the candidate recurrence and restore only the two pre/post memory-layout operations around post-attention norm to the `profile_carry` behavior. Alternatively, at position 7703, run both norm paths on the *same saved residual*, then compare post-norm values, CPU router logits, and native top-8 IDs. This can establish a local norm/router cause without restarting a second recurrence history. Confirm that attention and recurrent state remain identical; otherwise the control changed more than the norm.

If the parent's older 8192 baseline also fails at the same positions with comparable attention/state error, the new contraction and post-norm changes cannot be claimed as the root cause merely because the shorter 1024 run passed. They might change the magnitude of a pre-existing sensitivity. If the older baseline passes, use the one-factor controls above to assign the regression before retaining a repair.

No device result or verified repair is claimed in this report.
