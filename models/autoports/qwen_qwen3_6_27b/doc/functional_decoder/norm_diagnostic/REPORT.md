# CPU real-input rounding-boundary discriminator

Folded-scale FP32 storage is supported as the first repair to test: in this CPU simulation of the original combined rounding boundaries, it raises minimum decoder-contribution row PCC from **0.9962499942 to 0.9993809729**, both at row 350. This does **not** reproduce the actual TT minimum 0.99407857 or prove a passing device repair. No TTNN module was imported and no device was touched.

The exact saved length-511 input has raw BF16 SHA256 `50a69ebec5278c4e507cc340f4b5e50559d8946fca5d7546b3295715465bdc55`. The script loads real layer-0 weights at pinned revision `8f33b98937c6031bb49e713dee3e8e185aafd26f`, uses actual `[1,511,5120]` activations and complete projections, recurrent heads and MLP, and compares against the actual installed `Qwen3_5DecoderLayer` forward. The all-HF-boundaries control is **bit identical** to that forward. FP32 accumulation of the four BF16 convolution taps, followed by one BF16 output conversion, is also bit identical to the HF convolution path on this input; the all-parity-repairs final decoder control is bit identical too.

Every variant uses the same HF 64-token chunk recurrence/row-substitution algorithm, so these tests isolate rounding boundaries. They do not simulate TT tensor layouts, approximation kernels, operand intake precision, the TT 32-token inverse, or device matmul numerics. The inverse has a separate diagnostic. Projections and MLP remain actual CPU BF16 HF modules. Input/output residual-add BF16 rounding remains in every variant. Contribution is `decoder_output.float() - input.float()`; correlations are computed in FP64 per row. Every reported row metric is finite.

| Case | Minimum contribution row PCC | Worst row (zero based) | Contribution relative L2 |
|---|---:|---:|---:|
| `all_hf_boundaries_control` | 1.0000000000 | 0 | 0.00000000 |
| `only_input_folded_scale_bf16` | 0.9992983402 | 460 | 0.00711485 |
| `only_post_folded_scale_bf16` | 0.9999810141 | 42 | 0.00168767 |
| `both_folded_scales_bf16` | 0.9992961695 | 460 | 0.00716348 |
| `only_l2_fp32` | 0.9996038312 | 480 | 0.00316212 |
| `only_conv_bf16_taps` | 0.9996537388 | 480 | 0.00530704 |
| `only_conv_fp32_taps` | 1.0000000000 | 0 | 0.00000000 |
| `only_gamma_before_round` | 0.9998599051 | 19 | 0.00314601 |
| `only_bf16_silu_gate` | 0.9999115991 | 196 | 0.00285247 |
| `combined_original_boundaries` | 0.9962499942 | 350 | 0.00827658 |
| `combined_with_folded_scale_repair` | 0.9993809729 | 350 | 0.00574305 |
| `combined_scale_and_l2_repair` | 0.9996563852 | 480 | 0.00554357 |
| `combined_scale_and_conv_repair` | 0.9997875286 | 480 | 0.00394228 |
| `combined_scale_and_gatednorm_repair` | 0.9994470861 | 190 | 0.00506786 |
| `all_parity_repairs` | 1.0000000000 | 0 | 0.00000000 |

Isolated cases enable only the named mismatch, preserving HF boundaries elsewhere. Combined-original enables both folded scales rounded to BF16, BF16 multiplication/addition after each convolution tap, FP32 Q/K L2, gamma application before the normalized BF16 round, and BF16 SiLU gate. Combined repairs remove only the named discrepancies from that baseline. No padding, decay, state, or reference threshold is modified.

The input folded-scale change alone alters 606,418 normalized activation elements. Its normalized-hidden relative L2 error is 0.00269290, maximum absolute error 0.015625, and minimum normalized-hidden row PCC 0.9999958271. The resulting decoder error is more sensitive than this early activation correlation suggests. Rounding the post-attention folded scale alone is much smaller in these controls.

After fixing folded scales, the tested convolution boundary repair gives the largest further improvement in minimum contribution PCC and relative L2 among the individual additional repairs. L2 parity also improves the minimum; gated norm parity improves relative L2 and changes the worst row. These results support testing the earliest folded-scale repair first, then using actual device intermediates or one-change device runs to distinguish convolution, L2, and gated norm. They do not support declaring any remaining mismatch harmless or conflating these source repairs into one causal experiment.

The worst combined-original CPU row is 350, with reference contribution norm 40.79696297 and error norm 3.54352288; after folded-scale repair its error norm is 1.44595032. The original TT artifact did not retain the failed row index, so row 350 is **not identified as the TT failing row**. All row PCCs, worst-row norms, core and gated-output metrics, options, and input provenance are retained in `results.json` (compact JSON, below 500 KB). `run.log` contains the complete compact command results.

Reproduction from the checkout root (CPU only, final run exit 0):

```bash
source python_env/bin/activate
source /home/ttuser/qwen36-27b/lane-env.sh
timeout 900 python models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/norm_diagnostic/discriminate.py > models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/norm_diagnostic/run.log 2>&1
```

The script keeps upstream/core calculations cached across related variants. No decoder source, test runner, weights, or sibling diagnostic artifacts were edited. The saved input remains in `../inverse_diagnostic/input_length511.pt` and is reproducible using that directory's script. This is diagnostic evidence only, not stage acceptance evidence.
