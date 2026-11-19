# Decode gated RMSNorm through promoted padding: source audit

Current production source is `b1633d51621b4c45d9514a66fad21faa57ad562185502342c3d87bd5ebc2701c`. This audit reads source and historical artifacts only. No hardware, production, test, native-source or current README changes were made. No new candidate is prepared in this boundary-preserving audit. The existing head-as-token alternatives have measured latency rejections; the proposed true-head metadata adapter remains unmeasured. Its missing intermediate BF16 boundaries are an arithmetic-contract difference, not evidence that it fails the model acceptance bar.

## Follow-up: current model-gated experiments

The parent requested current-source experiments after this boundary-preserving source audit. [Four current candidates and a bounded probe](AUTOFIX_v4_decode_native_norm_candidates.md) now exist. Intermediate BF16 differences are not an additional model acceptance gate. Old measured losses do not reject a candidate on the current precise graph. The earlier statement that no candidate was prepared describes the initial audit only; the isolated current-source probe now passes its trace/padding checks and measures a faster true-head adapter. Current whole-decoder measurements now retain heads-BF16 after three cached long-stream passes, reject heads-FP32 on an actual seed154/position5991 model PCC failure, and reject compact rows variants on measured current-source latency. Combined v5 qualification remains pending. See the linked candidate memo for raw measurements; the old timing losses do not override this new evidence.

## What the metadata idea could actually remove

The old `native_gated_norm_all.py.snapshot` constructs head-first input, explicitly pads both it and the full packed gate operand from T1 to T32, invokes native gated RMSNorm, then slices its output back to T1. Its artifact `native_gated_norm_all_0.json` measures 0.968656 ms decode versus 0.964374 ms for the prefill-only native-norm control. Those timings belong to the old source, not today's exact decoder.

If tensors already possess the required physical head-first padding, explicit logical+padded reshape can promote T1 to T32 without filling/copying those bytes, then crop the native output logically. This could remove the two explicit pads and final slice; it is a different adapter than the old padded experiment. The new smaller Q/Z gate operand can also replace the old 12352-column gate buffer. Therefore the historical padded latency alone does **not** prove this revised adapter is slower.

However, the current recurrent core packs 32 heads as 32 rows of 128 values: `[1,1,32,128]`, only four physical tiles. Native 32-head/T32 input is `[32,32,128]`, 128 physical tiles. Promoting metadata cannot create the missing per-head tile planes. The existing permutation/reshape into head-first `[32,1,128]` must still establish that expanded physical layout; metadata promotion only removes the subsequent pad. This is not the same geometry as cropping an already-allocated 32-row head-concat output.

The native reader in `device/kernels/dataflow/reader_sigmoid_gated_rms_norm.cpp:45–65` sets `x_base = work_item * Vt`; with 32 heads and T32 it reads four tiles for every head. Reinterpreting the four-tile head-packed core as the 128-tile head-first operand would read beyond valid backing storage, even if a permissive metadata API accepted the shape. The valid head-as-token alternative uses `[1,32,128]`, one native head and 32 independent rows; that is exactly the geometry already exercised by the prior `native_gated_norm_head_rows.py.snapshot` and its L1 variant.

## Padding and row independence

Native validation requires positive T divisible by 32, rank-three input/gate and tile-aligned gate offset (`sigmoid_gated_rms_norm_device_operation.cpp:64–93`). The gate may have extra columns; current `[Q,Z]` with offset 2048 is legal. The kernel reduces across V for each row, then uses row-broadcast gamma and elementwise gate operations. It does not intentionally mix the 32 token rows in one statistic. Unused physical rows therefore need not be zero merely to protect the live row's RMS calculation, and metadata promotion is not rejected on an invented cross-row statistics concern. A real adapter would still need sentinel-padding and consumed-output tests; source independence is not a device proof.

## Actual blocker: unavailable intermediate BF16 boundaries

Current decode explicitly computes:

```
core_bf16 = BF16(core)
norm_bf16 = BF16(unweighted_rms_norm(core_bf16))
weighted_bf16 = BF16(float(norm_bf16) * float(gamma))
output_bf16 = BF16(float(weighted_bf16) * silu(float(z)))
```

Gamma round/widen is folded into the multiply postactivation but remains a real numerical boundary. Z is explicitly widened before binary SiLU, keeping its activation intermediate FP32. These boundaries were added after the seed154/5991 router-tie failure; removing them is not a neutral layout rewrite. [The operand analysis](AUTOFIX_output_gate_operand_precision.md) retains the failure and binding/kernel explanation.

The native factory at `sigmoid_gated_rms_norm_program_factory.cpp:78–88` fixes both `NORM_DFB` and `TMP_DFB` to Float32. Its compute sequence is:

1. Scale core by inverse RMS and pack normalized values into FP32 `norm` (`compute/sigmoid_gated_rms_norm.cpp:51–67`).
2. Multiply by BF16 gamma and pack into FP32 `tmp` (lines 70–86).
3. Compute sigmoid of BF16 gate into FP32 `norm` (lines 89–106).
4. Multiply weighted normalization by that sigmoid and only then pack the chosen output dtype (lines 109–125).

Selecting BF16 output does not insert BF16 rounding after unweighted normalization or gamma multiplication. An explicit BF16 input can preserve the core boundary only. The external multiplication by Z also gives a different rounding/association path: with native BF16 output it rounds after weighted-normalization-times-sigmoid, then multiplies Z; with native FP32 output it still computes `(weighted * sigmoid(z)) * z` rather than the current `weighted * (z * sigmoid(z))`. Neither choice restores the missing intermediate boundaries. There are no public flags to set these internal buffer dtypes, return pre-gate intermediates or skip normalization/weighting stages.

Using identity gamma or a constant gate to extract one stage would require retaining the external gamma/SiLU work and compensating the native sigmoid scaling. That no longer supplies the intended fused expression and introduces its own precision/adapter questions. It is not a concrete boundary-preserving optimization from the current source.

## Prior decode evidence, with its limits

| Historical adapter and exact source | Complete decode ms | Matched control ms | Decode PCC | Raw result |
|---|---:|---:|---:|---|
| [Explicit T32 pads, 32 heads](../native_gated_norm_all.py.snapshot) | 0.968656 | 0.964374 | 0.999960009 | [Candidate](../native_gated_norm_all_0.json), [prefill-only control](../native_gated_norm_prefill_0.json), [control source](../native_gated_norm_prefill.py.snapshot) |
| [Heads interpreted as 32 rows, one head](../native_gated_norm_head_rows.py.snapshot) | 0.903477 | 0.900932 | 0.999967562 | [Candidate](../native_gated_norm_head_rows_0.json), [control](../recurrent_memory_matmul_0.json), [control source](../recurrent_memory_matmul.py.snapshot) |
| [Same head-as-row geometry, L1](../native_gated_norm_head_l1.py.snapshot) | 0.903046 | 0.900932 | 0.999967562 | [Candidate](../native_gated_norm_head_l1_0.json), [control](../recurrent_memory_matmul_0.json) |

All three results report deterministic eager/traced outputs and passing fallback audits. The measured latency differences are +4.282 µs (+0.444%), +2.545 µs (+0.283%) and +2.114 µs (+0.235%), respectively. These are small losses, not a claim of a large or statistically established slowdown. None beats its measured correct control, so none was selected. Their short model PCC **passed**; the rejection is measured performance, not intermediate-bit equality or a demonstrated quality failure.

These controls passed their historical short PCC checks and did not improve measured speed. They predate the later exact output-boundary repair and do not establish current 8192-token robustness. The head-as-token/L1 alternatives already avoid the true 32-head expansion and explicit T32 padding, so repeating those unchanged is not a new graph-fusing control.

Conclusion: the head-as-token variants already provide the compact four-tile geometry and have measured performance rejections. A true-head metadata adapter could remove pads from the old expanded geometry, but still needs the head-plane expansion and does not preserve the requested internal precision boundaries. This source audit therefore supplies no new boundary-preserving candidate. If the arithmetic boundary requirement is relaxed for a separate model-PCC experiment, this particular metadata adapter remains unmeasured, not quality-rejected; it would need current-source full latency and unchanged model acceptance tests. Historical short passes do not establish current long-stream robustness, and historical latency losses do not prove the new adapter is slower. No new acceptance exemption or Stop Rule conclusion follows.

## Artifact identity

SHA-256 hashes below identify the exact historical inputs to this audit.

| Artifact | SHA-256 |
|---|---|
| [native_gated_norm_all.py.snapshot](../native_gated_norm_all.py.snapshot) | `1f29b862e8d22396575fd72002443435fff9bb47f3dc42c1aab36465da7fadbe` |
| [native_gated_norm_all_0.json](../native_gated_norm_all_0.json) | `2257b09d7c60279df0490e0af2ae8a9495d0603016e5ca04f30cb47b39de856c` |
| [native_gated_norm_prefill.py.snapshot](../native_gated_norm_prefill.py.snapshot) | `1a190ca2b698ab00db3a05716842176c929996965c32e9031eeae7b12693bf78` |
| [native_gated_norm_prefill_0.json](../native_gated_norm_prefill_0.json) | `13b148eefe881d2e131fc032043be99790615c8ca81e2c5bccc88056f5309f08` |
| [native_gated_norm_head_rows.py.snapshot](../native_gated_norm_head_rows.py.snapshot) | `1e3e120ba59bfda356ea024f4a67bbc9065088d20008a057081a394f02e46427` |
| [native_gated_norm_head_rows_0.json](../native_gated_norm_head_rows_0.json) | `44f64e005defbf129732fbfe219253b1fffc80ae62203317a45363f75d5ce486` |
| [native_gated_norm_head_l1.py.snapshot](../native_gated_norm_head_l1.py.snapshot) | `1368274b5f4b1d9920f9511ad8a106d2666db662f82596c1bfc70fe9dace41b7` |
| [native_gated_norm_head_l1_0.json](../native_gated_norm_head_l1_0.json) | `c5372ecd36103b15431858121de6b26e4f132f831b9e0c685450ba7872029de7` |
| [recurrent_memory_matmul.py.snapshot](../recurrent_memory_matmul.py.snapshot) | `5028b78571c58a2802be3b844f94bb51c51f0f6fba71d24ab2a93fe0a602405f` |
| [recurrent_memory_matmul_0.json](../recurrent_memory_matmul_0.json) | `b0a6fb7c7336da55e557e82370b06274c49d815c34a012a6e1bebb0b26e849e9` |
