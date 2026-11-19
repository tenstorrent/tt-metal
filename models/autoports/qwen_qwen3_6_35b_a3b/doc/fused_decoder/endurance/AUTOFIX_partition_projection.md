# Partition precise state projection from native Q/Z projection

CPU/source-only investigation, with no TTNN import, hardware execution, or fused-source changes. This is a model-local arithmetic partition candidate, not a datatype sweep: checkpoint weights, normalized input and public projection results remain BF16.

## Semantic dependency

The recurrent update in `_recurrent_step` is `decayed = state * decay(A)`, `memory = reduce(decayed*K)`, `delta = (V-memory)*beta(B)`, `new = decayed + K*delta`. Neither Q nor Z enters `new`. Q is used only for `reduce(new*Q)`; Z enters the subsequent output gate/norm. The prefill state-only correction likewise reads only K/V/A/B, replacing the native chunk's final state independently of Q.

Q **does** enter the convolution history. Because this is depthwise four-tap convolution, a Q projection error can affect Q at the current token and next three tokens, but cannot cross into K/V or enter the long-lived recurrent state. Z has no convolution/history. Therefore Q/Z projection discrepancies do not accumulate through this decoder's recurrent cache on an identical external input stream. This is not a guarantee about downstream full-model autoregressive feedback, which is outside this stage.

Maintain the existing exact shared input norm. Preserve precise K/V/A/B and all their BF16 boundaries. With identical initialization, the recurrent cache should remain bit-identical to the best correct full-precision-projection candidate; K/V convolution cache channels should too. Q convolution cache channels may differ. Check these distinctions in diagnostics rather than requiring the whole convolution cache to be bit-identical.

## Proposed packing

- Precise ordered output: K `[2048:4096]`, V `[4096:8192]`, A `[12288:12320]`, B `[12320:12352]`, concatenated to 6208 columns. Pad by 960 to 7168, represented as `32×224`, preserving the validated odd/even K order and K512/K1024 accumulation schedules.
- Native BF16 matmul output: Q `[0:2048]`, Z `[8192:12288]`, concatenated to 6144 columns; native weights `2048×6144` BF16. Start with HiFi4, nonapproximate, FP32 destination accumulation and BF16 output, without bias.
- Reconstruct the existing logical `[Q,K,V,Z,A,B]` 12352-column result to keep all downstream slices and native prefill gate-column offset unchanged. Extra concat/slice cost must be included in traced latency.

The ordered FP32 product falls from `2048×13312×4 = 109,051,904` bytes to `2048×7168×4 = 58,720,256` bytes, a 46.15% decrease. One write plus one read saves 100,663,296 bytes per decode before added native-matmul/packing costs. Total padded BF16 projection weight bytes are unchanged: `7168 + 6144 = 13312`. Reduction latency may not fall proportionally because fewer output tiles reduce core occupancy. Only measured warmed traced latency can establish benefit.

## Existing numerical evidence

`partition_projection_cpu.py` loads only the real checkpoint QKV/Z weights with safetensors and examines all 13 `endurance_exact_frontend_probe.json.pos*.pt` captures on four CPU threads. The archived fused-source hash is `adf5dcf0e489249738161c0119429960f447e59d9130702f501a76c06540f543`, matched to `endurance_gates_norm_l2_conv.py.snapshot`, whose packed projection is the native sharded matmul. CPU BF16 projection uses the identical archived **TT normalized input**, isolating projection arithmetic from input-norm errors. Per-capture file hashes, metrics and normalization mismatches are in `partition_projection_cpu.json`.

Across positions 128,129,256,1024,3032,3979,4904,5011,6595,7551,7668,7703,8157:

- Q PCC: 0.9999990863–0.9999995063; relative L2: 0.0009935–0.0013524; max absolute difference at most 0.015625. Per-token BF16 code differences: 301–364 of 2048.
- Z PCC: 0.9999992717–0.9999994876; relative L2: 0.0010077–0.0012071; max absolute difference at most 0.015625. Per-token BF16 code differences: 538–635 of 4096.

`endurance/projection_precision.json` provides corroborating same-input native matmul evidence at position3979: generic Z PCC 0.9999993908, relative L2 0.0011062, 596/4096 differing BF16 codes. Its QKV aggregate cannot isolate Q.

Do **not** cite `endurance/linear_frontend_stream_probe.json` as native matmul evidence: matching its source hash shows it used the later explicit FP32 SFPU multiply plus generic sum, not native matmul. Likewise the old `projection_probe.json` fused-bias failure is a separate rejected packing bug.

These archived outputs use the old 12352-column native packing. The proposed 6144-column native matmul may schedule differently, and favorable projection PCC alone is not final-output acceptance. Root must measure the new actual path, preserve unchanged full 8192-token/continuation gates and prefill contribution PCC, check recurrent/KV cache invariance, and compare warmed traced latency with the best correct candidate before keeping it.

Reproduce CPU evidence:

```
python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/partition_projection_cpu.py
```
