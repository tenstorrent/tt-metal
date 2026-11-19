# Real-weight two-BFP8 projection precision probe

Runner: `tests/run_projection_precision_probe.py`. Authored and CPU-checked without device access. It requires the existing frozen `QWEN_DECODER=optimized`, `QWEN_OPT_CANDIDATE=packer53`, and `QWEN_OPT_POLICY` environment used by `packer53_jobs.py`; it does not import or execute that job script. Parent device owner must use the existing serial device wrapper.

```bash
python -m models.autoports.qwen_qwen3_6_27b.tests.run_projection_precision_probe \
  --capture models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/capture41/layer0_step209.pt \
  --variant hifi4_pack --variant hifi4_nopack \
  --variant hifi2_pack --variant hifi2_nopack \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/projection_precision_probe.json
```

Default is all four variants and both captured TT/HF inputs. `--input-source tt` limits the input control. `--user-index` selects within the capture's saved users, not the original batch index. Each projection receives one saved logical row `[1,1,K]`; its program retains the legal32-core/4-reader/block1 geometry and tile-height32. No HF model is instantiated.

For each variant, policy is set before constructing the frozen decoder. Only attention fidelity and `packer_l1_acc` change; BFP8 base/correction weights, DRAM sharding, L1 partial addition, and explicit geometry are asserted. Actual `_linear` runs with its production partial/add/slice path. A temporary hook holds its two FP32 matmul outputs for later diagnostic reads; it neither changes results nor replaces arithmetic. An additional FP32 TTNN add is evaluated after the production result to isolate addition from BF16 output packing.

Readbacks include the actual device-dequantized W0/W1, FP32 partials, separate FP32 sum, and production BF16 final result. CPU references accumulate the captured BF16 activation and dequantized weights in FP64, using512-output-column chunks to bound temporary storage. Original checkpoint BF16 `F.linear` and FP64 original-weight reference are reported separately. Packed Q/K/V/Z/B/A layout exactly follows frozen setup; padding columns are omitted from the aggregate metrics.

Interpretation:

- `dequantized_pair_vs_original_fp64`: residual quantization alone.
- `base_partial_vs_dequantized_fp64` / `correction_partial_vs_dequantized_fp64`: TT matmul/accumulation precision on identical stored operands.
- `tt_fp32_add_vs_cpu_sum_of_tt_partials`: standalone TT addition precision.
- `tt_final_vs_cpu_bf16_sum_of_tt_partials`: production add/packing boundary.
- `tt_final_vs_tt_fp32_sum_then_bf16`: whether combining the add with BF16 output differs from explicit FP32 addition then cast.
- `tt_final_vs_dequantized_fp64_then_bf16`: complete TT arithmetic on identical quantized weights.
- `tt_final_vs_hf_bf16`: total same-input precision difference.

Every comparison reports PCC, fitted gain (descriptive only), relative RMS error, maximum absolute error, exact equality, and BF16 equality fraction. Device weight and activation hashes allow checking the four variants used identical operands. The JSON records full policies and decoder/harness provenance and is saved after each completed variant.

All readbacks and additional operations are diagnostic. There are no perf rows, new acceptance thresholds, threshold waivers, or runtime host fallbacks. Validation completed in the child: `py_compile`, plus AST-isolated CPU checks of FP64 chunked projection, metrics, channel aggregation, and fingerprints without importing TTNN. Kernel/device execution remains for the parent.
