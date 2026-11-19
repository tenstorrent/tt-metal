# AutoFix input-norm order and prefill follow-up

CPU/source only. No runtime, test, or native edits and no device execution. The parent's position130 device diagnostic establishes the first difference at variance (one FP32 ULP), with exact rsqrt on the canonical shifted variance. The proposed two-level TTNN reduction fixes that one case; broader equivalence remains unproved.

## Thread scheduling is not the explanation

Installed torch is2.11.0+cpu, commit70d99e998b4955e0049d13a98d77ae1b14db1f45; routing_probe uses8 CPU threads. On position130, thread counts1/2/4/8/16 give identical mean bits991798430 and inverse bits1101206690. Reconstructed seed913 prefix128 plus decode8192 produces identical mean outputs at1 versus8 threads for all8320 rows. Thus switching CPU thread count does not fix this observed discrepancy.

The HF RMSNorm at installed `modeling_qwen3_5_moe.py:817–833` squares FP32 input, averages, adds epsilon, takes rsqrt, multiplies input, multiplies effective gamma, then casts BF16. Current TT keeps these boundaries but changes reduction order. Its SFPU row reduction folds corresponding columns across32-wide tiles before horizontally reducing the accumulated tile (`ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl:499–555`); FP32 mean applies the scale after summing (`reduction/generic/device/reduce_op.cpp:235–243`).

The matching PyTorch source uses CPU sum then divide for mean. Its contiguous floating sum uses four interleaved vector accumulators with a multilevel cascade and finally accumulates the vector lanes serially. This differs from TT's tile folding. The package reports AVX512 capability, but the **eight-lane** source-algorithm emulation matches all8320 observed CPU means; the16-lane emulation already misses six of the nine saved cases. Treat the empirically verified eight-lane schedule as the relevant contract, not the capability label alone. Sources: [ReduceOps.cpp at installed commit](https://raw.githubusercontent.com/pytorch/pytorch/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/ReduceOps.cpp), [SumKernel.cpp at installed commit](https://raw.githubusercontent.com/pytorch/pytorch/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/cpu/SumKernel.cpp).

## Broader CPU evidence

| Alternative mean computation | Different means /8320 | Max mean ULP difference | Final BF16 norm differing elements | Affected rows |
|---|---:|---:|---:|---:|
|Eight-lane CPU schedule emulation|0|0|0|0|
|FP64 mean, rounded to FP32|3197|2|55|53|
|CPU reshape64x32, sum32 then sum64|4054|3|70|66|

The BF16 comparisons keep CPU rsqrt and both following multiplications unchanged. Most mean differences disappear at the BF16 boundary, but dozens of rare rows do not. This is enough to disprove the idea that more real-number precision necessarily matches HF. It does **not** prove those55–70 differences cause the endurance failures.

The CPU64x32 reduction is not a simulator for the TT64x32 reduction: at130 CPU64x32 gives mean bits991798429, while the parent's TT64x32 result gives991798430, exactly HF. Both use32-column logical chunks but their internal reduction orders differ. Evaluate the TT candidate on the actual stream; do not infer all-token equivalence from its shape or its one-case success.

## Model-only alternatives

The least invasive candidate is the parent's two-level TTNN reduction, with a broad same-input intermediate sweep to measure its FP32 mean and final BF16 differences. Keep it only on evidence.

If canonical mean matching is necessary, an explicit TTNN SFPU add schedule can reproduce the empirically verified CPU order without native edits. For K2048, reshape squared inputs into four512-element blocks, each sixteen32-element rows. In parallel for all four blocks, add those sixteen rows sequentially, producing32 lanes per block. Add the four block results left-to-right. Interpret the32 lanes as four groups of8; add the four groups left-to-right, then add the8 resulting lanes serially. Multiply by1/2048 afterward. Keep every add FP32 SFPU; use slicing/reshape and explicit `ttnn.add(..., fast_and_approximate_mode=False)`, not generic reductions that choose another order. This layout is a derivation validated by CPU replay over8320 rows. Device slicing, cost and primitive exactness still need testing. It has roughly28 add stages and is a correctness control before a performance choice.

No epsilon shift, unconditional nextafter, or midpoint bias is justified: the required correction changes sign with input. Likewise, compensated or higher-precision sums improve numerical accuracy but need not match the canonical CPU FP32 oracle.

## Minimal residue-mask prefill projection experiment

The frozen source dispatches precise packed projection only for S=1 (`linear`, lines221–229). Prefill still uses ordinary packed native linear, so decode improvements leave its QKV history and recurrent prefix state inaccurate.

A minimal **projection-only** model patch would pre-upload eight BF16 packed matrices `W_r[k,n] = W[k,n] if k % 8 == r else 0` during setup, then replace the prefill `attention_packed` linear with eight native HiFi4/FP32-destination matmuls and accurate FP32 SFPU additions. Preserve S and output shape[1,S,12352], use DRAM output, and cast only after the final sum. The existing decode residue-mask probe gives~5e-7 relative error for8 masks; this is promising but not a prefill measurement. Weight storage is386MiB for eight full matrices (versus48.25MiB original); setup can reuse a masked host buffer. Output accumulation needs only a running total and current partial.

**Bias must be added exactly once after combining all partials.** To isolate the projection hypothesis, retain current prefill's one-time attention_bias semantics initially; never pass attention_bias to each of the eight calls. Compare prefix QKV/history and recurrent state before repeating endurance. For128-token prefill, inspect incoming conv cache at decode128 and the next two steps, where the existing cache discrepancy is directly observed.

This one patch cannot promise canonical prefill exactness. Current prefill also uses native input RMSNorm, native causal conv without the validated decode BF16-preactivation/SFPU-SiLU sequence, native chunk Q/K normalization, and FP32 A/B gate projection/bias boundaries. A subsequent canonical prefill frontend would reuse the validated vectorized decode operations over S: precise norm, SFPU four-tap conv using slices of history+projection, explicit BF16 conv boundary, and rounded A/B before gate evaluation. Move dt_bias out of the packed prefill output when making A/B boundaries canonical, or it will be rounded with A rather than added afterward. Leave native chunk recurrence initially to keep frontend and recurrence experiments separate.

Mask8 is the smaller measured candidate; mask16 is the stronger single-live-product-per16-wide-MVMUL control. Neither has a measured prefill/endurance result from this investigation.
