# v3 conditional RMSNorm identity-cast removal

Status: root device diagnostic passed all 30 cases exactly; full-path controls remain pending. No production edit.

The [candidate snapshot](v3_norm_identity_cast.py.snapshot), SHA256 `8c5410ea2885170378296da8418c07178e8f02b62e48ff06efe0d549d1ecdbd8`, is based on [v3](v3_movement_baseline.py.snapshot), SHA256 `73ba9aabb10a2c34f3b10b5ff1fc93418470c0a624dd863e87a8c8c2af2901ba`. It changes only the generic final return in `FusedDecoder.norm`: return the RMSNorm output when its dtype is already BF16; otherwise retain the explicit BF16 typecast. The special linear-attention inputnorm branch is AST-identical. This remains separate from carry/router movement candidates.

## Native contract and preserved behavior

- `ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_device_operation.cpp:88–92` explicitly accepts FP32 and BF16 input. Gamma accepts FP32/BF16 at lines 165–167; production uploads FP32 `1 + checkpoint_weight` for these norms.
- `ttnn/cpp/ttnn/operations/normalization/rmsnorm/rmsnorm.cpp:77–83` selects the native norm program config and passes no dtype override. For the nondistributed paths used here, output specification in `layernorm_device_operation.cpp:531–544` uses the input dtype. An FP32 input therefore still needs the final BF16 cast. The candidate checks the actual result dtype, rather than assuming the caller always uses BF16.
- `layernorm/device/layernorm_common.cpp:11–24` derives the sharded configuration and sets `inplace=false`. Skipping a BF16 identity cast does not introduce input mutation.
- Existing memory-config selection, native RMSNorm compute configuration, epsilon and weights remain unchanged. The generic method still returns BF16. The skipped operation does not round to a new dtype.
- Sharded RMSNorm requires matching input/output memory layouts and buffer types (`layernorm_device_operation.cpp:349–360`). Multirow postnorm intentionally returns interleaved L1, so the probe uses the actual production interleaved input for that case rather than inventing an invalid sharded multirow call.

The accurate [v2 decode profile](../profile_v2/tracy/layer_0/decode_op_sequence.csv) has one such BF16 identity cast in linear attention, step 152, at 0.719 us. Layer 3 has steps 3/13/15/46 at 0.827/0.686/0.650/0.748 us. These are dispatch-removal opportunities, not measured v3 savings.

## Actual-method proof and complete-adapter timing

[tests/v3_norm_identity_probe.py](../../../tests/v3_norm_identity_probe.py) loads the entire actual `norm` methods from frozen snapshots. CPU setup asserts no other method changed and the special linear inputnorm branch is identical. It loads real layer-0 postnorm and layer-3 input/post/Q/K checkpoint weights, applying the same FP32 setup transform as production.

Each of five norm roles is exercised with BF16 and genuinely FP32 seeded inputs at:

1. S1, interleaved DRAM;
2. S1, production-shaped width sharding: hidden norms use 32 cores with `(32,64)` shards; Q/K use one `(32,256)` shard and actual head shapes `[1,1,16,256]` / `[1,1,2,256]`;
3. S33, interleaved input, including actual PF Q/K shapes `[1,16,33,256]` / `[1,2,33,256]` and postnorm's L1 output selection.

The diagnostic checks native output dtype and codes, final BF16 codes and input preservation. Warmed traces include the complete adapter: optional DRAM-to-width-sharded conversion, actual norm method and all its output/cast operations, and output-to-DRAM conversion. Common layout conversions remain timed. First and last replay outputs must match baseline codes. Host inspection is outside device-only and timed regions.

[CPU setup evidence](v3_norm_identity_probe_cpu.json) records source/dependency/snapshot hashes and all 30 planned cases. It proves setup and source scope only; legal native contracts still require root's device run.

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v3_norm_identity_probe \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v3_norm_identity_probe.json
```

After exact adapter results, run paired complete decoder controls for both attention kinds:

```bash
for layer in 0 3; do
  for candidate in movement_baseline norm_identity_cast; do
    python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.benchmark_fused_snapshot \
      --snapshot "models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v3_${candidate}.py.snapshot" \
      --layer "$layer" --length 256 --context 512 --prefill-repeats 10 --decode-repeats 60 \
      --output "models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v3_${candidate}_layer${layer}_s256.json"
  done
done
```

Keep/reject remains pending exact device results, full-path latency and required regression coverage. No acceptance gate is waived.

## Root device results

[v3_norm_identity_probe.json](v3_norm_identity_probe.json) reports `device_exact=true` for all 30 cases. Native outputs, final BF16 outputs, input preservation and first/last traced outputs passed. All BF16 adapter cases improve; FP32 retains the original path and differs only by small timing variation.

| BF16 role | Rows | Input layout | Baseline ms | Identity removed ms |
| --- | ---: | --- | ---: | ---: |
| L0 post_attention_layernorm.weight | 1 | interleaved | 0.030465191 | 0.028470438 |
| L0 post_attention_layernorm.weight | 1 | width_sharded | 0.025837356 | 0.023364695 |
| L0 post_attention_layernorm.weight | 33 | interleaved | 0.032304786 | 0.030494295 |
| L3 input_layernorm.weight | 1 | interleaved | 0.030321022 | 0.028467365 |
| L3 input_layernorm.weight | 1 | width_sharded | 0.025751302 | 0.023338711 |
| L3 input_layernorm.weight | 33 | interleaved | 0.031018304 | 0.028611720 |
| L3 post_attention_layernorm.weight | 1 | interleaved | 0.030333037 | 0.028564595 |
| L3 post_attention_layernorm.weight | 1 | width_sharded | 0.025825296 | 0.023352588 |
| L3 post_attention_layernorm.weight | 33 | interleaved | 0.032280711 | 0.030572480 |
| L3 self_attn.q_norm.weight | 1 | interleaved | 0.010695495 | 0.009248685 |
| L3 self_attn.q_norm.weight | 1 | width_sharded | 0.012268499 | 0.011046231 |
| L3 self_attn.q_norm.weight | 33 | interleaved | 0.020431029 | 0.013159215 |
| L3 self_attn.k_norm.weight | 1 | interleaved | 0.010698475 | 0.009202585 |
| L3 self_attn.k_norm.weight | 1 | width_sharded | 0.012209313 | 0.011024112 |
| L3 self_attn.k_norm.weight | 33 | interleaved | 0.011279527 | 0.009443983 |

These are complete-adapter measurements, including the common layout conversions described above. Full decoder latency and required endurance coverage still determine selection; exact primitive checks are not final acceptance.
