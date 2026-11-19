# Prefill canonical projection order

**M=128 uses two K=1024 partial dots, not decode's four K=512 partial dots.** Both use odd-product then even-product FP32 accumulation inside each partial, followed by FP32 addition of the partial results and one BF16 cast.

CPU-only proof uses the entire canonical seed-913 prefix `[1,128,2048]`, real checkpoint weights, canonical inputnorm, and full `F.linear(prefix_norm, weight)` calls with original QKV/Z/A/B widths. All 128 rows were then independently emulated. Artifacts: `prefill_projection_cpu_dot_order.py`, `.json`, `.log`. No TTNN imports, devices, F or test edits.

| Projection | Outputs checked | Significant BF16 differences, K=1024 schedule | Significant BF16 differences, K=512 schedule |
|---|---:|---:|---:|
| QKV | 1,048,576 | 0 | 259 |
| Z | 524,288 | 0 | 106 |
| A | 4,096 | 0 | 0 |
| B | 4,096 | 0 | 0 |

“Significant” here means the canonical value has magnitude above `1e-30`, to distinguish the already known instruction denormal behavior. All A/B/Z outputs match bitwise as values under K=1024. QKV retains only extremely tiny differences (see JSON); no tolerance or acceptance thresholds were altered. A/B alone do not discriminate the two schedules on this prefix, but their actual emitted kernel has K=1024 as well.

The verbose log reports `brg_matmul:avx512_core_bf16`, `src:bf16`, `wei:bf16`, `dst:bf16`, for `128x2048:2048x8192`, `128x2048:2048x4096`, and `128x2048:2048x32`. CPU JIT dumps under `/tmp/prefill_projection_jit` show the QKV M=128 kernel `.0.bin` and A/B M=128 kernel `.12.bin` load loop count `0x80` (128); each iteration consumes four consecutive BF16 pairs, hence 1024 K elements. The BF16 dot instruction is `vdpbf16ps`, partial combination is `vaddps`, and final conversion is `vcvtneps2bf16`. This doubles the K-block length observed in the decode kernel. The existing oneDNN source references and SFPU cumsum precision audit are in `AUTOFIX_projection_cpu_dot_order.md`.

A concrete discriminator is prefix row 127, QKV channel 1818: canonical HF is `0.2099609375`; the otherwise identical K=512 schedule gives `0.208984375`. Row 126 channel 6205 gives HF `-0.2353515625` versus K=512 `-0.236328125`.

## Model-only prefill control

Reuse the parent's already device-validated odd/even K permutation and SFPU product/cumsum path, but explicitly select **two K=1024 blocks for public M=128 prefill**. Do not select the decode schedule merely because the implementation processes one prefill row at a time.

For a small token batch T (1–4 initially), form FP32 SFPU outer products `[T,2048,12352]` with adjacent K pairs reversed. Reshape to `[T,2,1024,12352]`, run `ttnn.cumsum(..., dim=-2, dtype=ttnn.float32, disable_compensated_sum=True)`, take row 1023 of each block, and add the two partial rows using FP32 SFPU add. Reshape the result to `[1,T,12352]`; concatenate token batches in their original order before convolution/gates. Cast each projection at its canonical existing BF16 boundary. T=1 products are about 96.5 MiB, so a small T bounds temporary storage while preserving the whole-prefix oracle's numeric order.

This changes only projection arithmetic. Keep the public 128-token prefill boundary intact: convolution sees the complete ordered projection sequence, recurrence processes the public prefix as before, and the public recurrent cache is BF16-rounded once, not after each projection microbatch. Do not run the public decode state-update routine for these projection microbatches.

The packed projection control should first compare the whole-prefix QKV/Z/A/B outputs, then conv/Q/K/V and final state. This source/CPU result does not establish native GDN recurrence exactness or eliminate the independently observed native prefill-state discrepancy. M=256 has not been audited and must not silently inherit this M=128 schedule.

Reproduce from the checkout:

```sh
ONEDNN_VERBOSE=1 python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/prefill_projection_cpu_dot_order.py > models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/prefill_projection_cpu_dot_order.log 2>&1
```
