# v5 attention input-norm gamma folding candidate

Prepared isolated candidate `v5_inputnorm_weight_fold.py.snapshot` (SHA256 `e7caf965eafcb8f9ebb3f27cd0f65e5721ae6171039a5df69d77f32ebd5074fd`) against immutable `v5_cache_writers_native_norm.py.snapshot` (`1a593ec9a2522dd579194d5082ffe5e987a81fc87be287590bdec086e067d455`). The adjacent patch and provenance bind both sources and the CPU proof. No production, native, core-harness or hardware edits/execution occurred. This candidate is ready for complete decoder timing and unchanged quality gates; no selection or quality result is claimed.

## Complete graph change

For linear-attention layers only, setup shallow-copies the caller dictionary and replaces exactly `linear_attn.in_proj_{qkv,z,a,b}.weight` with `(original.float() * gamma[None,:]).to(original.dtype)`, where `gamma = input_layernorm.weight.float() + 1.0`. No checkpoint tensor is modified in place. The checkpoint's weights remain BF16; this is a scale fold into their existing storage, with new rounding at setup.

This runs before any upload or packing, so every representation receives the same folded source. The generic per-key uploads are folded; native `output_only_packed` receives Q's first2048 rows plus all4096 Z rows; ordered `scan_attention_weight` receives the remaining6144 QKV rows (K/V), then32 A and32 B rows, followed by the original odd/even K permutation. Full-attention Q/K/V packed weights and all nonprojection weights remain unchanged. Native QZ and ordered KVAB use their existing kernels, K policies, projection order and layouts.

The linear-attention inputnorm branch changes its final two multiplies:

```python
normalized = multiply(wide, inverse, dtype=float32, fast_and_approximate_mode=False)
result = multiply(normalized, gamma, dtype=bfloat16, fast_and_approximate_mode=False)
```

to one existing-style SFPU multiply with direct BF16 output:

```python
result = multiply(wide, inverse, dtype=bfloat16, fast_and_approximate_mode=False)
```

Thus one runtime gamma multiply and its FP32 normalized intermediate are actually removed; no identity multiply, replacement typecast operation or runtime host work is left. Original output memory restoration remains. Setup also skips the now-unused linear inputnorm gamma upload. The norm's mean, epsilon, inverse corrections and all other norm branches are unchanged.

`_forward` sends this normalized tensor only into `_linear_attention`, which consumes it solely through the two projections above. The residual retains original hidden states. Post-attention norm, gated output norm, recurrent updates, convolution weights, A_log/dt_bias and their LUTs, MLP/router weights and output projection remain untouched. Full-attention layers take neither the fold nor changed norm branch. Only `from_state_dict` and `norm` differ in the AST; their hashes are in provenance.

## Correct scale and numerical contract

Canonical `Qwen3_5MoeRMSNorm.forward` in installed `modeling_qwen3_5_moe.py:823–831` computes FP32 normalization, multiplies by `(1.0 + self.weight.float())`, then casts to the input dtype. The fold uses that exact scale convention, not checkpoint weight alone. Layer0 actual gamma spans0.919921875–1.330078125.

In real arithmetic, `(u * gamma) @ W.T == u @ (W * gamma).T`. BF16 arithmetic differs: the baseline rounds `u*gamma` before projection, while this candidate rounds `u` and the scaled weights separately. Direct BF16 normalization output also replaces the previous FP32-normalization product boundary. These are explicit candidate changes; neither exact-bit equivalence nor inevitable quality failure is asserted. The actual decoder's PCC, deterministic replay and long-state behavior must decide acceptance under unchanged thresholds.

## Scoped CPU proof

`v5_inputnorm_weight_fold_cpu.py` imports no TTNN and executes the candidate's actual setup fold AST on real layer0 checkpoint weights. It proves:

- Caller tensors are unchanged; only the four intended matrices are replaced, retaining original dtypes.
- Full-attention setup is a no-op; norm/output weights retain their original objects.
- Native QZ and ordered KVAB partitions reconstruct every intended folded weight, including all A/B columns and the odd/even K permutation.
- Sampled FP64 pre-rounding identities agree within8.89e-16 on real weights.

The report `v5_inputnorm_weight_fold_cpu.json` also honestly records sampled BF16 projected-output differences (max0.001953125–0.0048828125 across the four projections on synthetic normalized rows). These are scope/rounding diagnostics, not model quality measurements and not grounds for rejection by themselves. CPU execution completed in about one second with one Torch thread.

Reproduce CPU proof:

```sh
python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v5_inputnorm_weight_fold_cpu.py
```

The parent's existing candidate-snapshot whole-decoder route can load this file directly. Complete timing must include the updated norm plus both projections; no projection-only gain should be treated as whole-path gain. Fresh-HF8192 streams remain required if short correctness/performance makes this candidate useful.

## Subsequent parent-run qualification

The BF16-fold candidate passed the short benchmark but failed seed1548192+33 at5991,6062,6425 under unchanged PCC thresholds. It is not selected. Full evidence and the single setup-only FP32-fold follow-up are recorded in `AUTOFIX_v5_inputnorm_weight_fold_fp32.md`; original snapshot/proof/results remain preserved.

## Measured closure

The FP32-fold short complete-path benchmark `v5_inputnorm_weight_fold_fp32_s256.json` passed short PCC and deterministic replay: prefill PCC0.999933442088, decode PCC0.999939257990. It recorded prefill116.623264970 ms / decode1.380870817 ms versus matched selected107.813201146 / 1.285217237 ms (`v5_prefill_pair_blocks_baseline_s256.json`). That is 8.17% slower prefill and 7.44% slower decode. **Reject this FP32 adaptation on complete-path speed. No8192 FP32 quality run or quality verdict is claimed.** The BF16 fold separately remains rejected on its actual seed154 long-stream failures.

`v5_inputnorm_weight_fold.decision.json` binds the matched baseline and all three candidate evidence artifacts/sidecars by SHA256. FP32 short artifact SHA256:`fa26fb01ca69829b201a322d1cff87ce70d2f936ec741a0ccb784b27373fc92a`. No new candidate or runtime change is proposed.
