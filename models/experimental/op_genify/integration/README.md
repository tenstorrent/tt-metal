# genify_matmul in the models (#65150)

The generated matmul from #65149 (`ttnn/ttnn/operations/genify_matmul/`) replaces the stock single-chip
matmuls of Qwen3.8-27B, FLUX.2-dev and Panoptic-DeepLab. `TT_GENIFY_MATMUL=1` routes them through
`genify_matmul_adapter.py`; unset, the stock ops run unchanged. CCL-fused matmuls (#65143) are not routed.

Device kernel time per chip from Tracy, Blackhole QuietBox 2, stock and generated measured on this branch.
Roofline from #65148 (`../matmul_shape_set/65148_comment.md`; it assumes 70% of peak compute, so a
compute-bound op can beat it).

| Model | Per | Stock matmuls | Generated | Roofline | Model device time, stock -> generated |
|---|---|---:|---:|---:|---|
| Qwen3.8-27B (1x4, TP4) | decode token | 10.12 ms | 9.32 ms (1.09x) | 8.17 ms | 22.38 -> 21.73 ms |
| | prefill M = 128 | 16.20 | 16.58 (even) | 8.40 | 45.75 -> 44.45 |
| | prefill M = 512 | 31.04 | 29.14 (1.07x) | 15.01 | 87.74 -> 80.02 |
| | prefill M = 1024 | 44.16 | 36.82 (1.20x) | 29.42 | 131.5 -> 111.0 |
| | prefill M = 2048 | 80.82 | 63.29 (1.28x) | 58.66 | 241.5 -> 202.4 |
| | prefill M = 4096 | 155.85 | 121.25 (1.29x) | 117.32 | 477.9 -> 386.3 |
| FLUX.2-dev (2x2, SP2 x TP2) | denoising step | 318.06 ms | 272.38 (1.17x); 298.02 (1.07x) with the split/concat ops genify needs | 303.10 | 370.4 -> 350.6 ms without collectives |
| | prompt encoding | 69.57 | 70.44 (even) | 53.69 | 99.2 -> 100.4 ms without collectives |
| Panoptic-DeepLab (1 chip) | inference | 0.990 ms | 0.967 (1.02x); 1.042 vs 1.171 with the data movement around them | 0.33 | 5.759 -> 5.635 ms |

Accuracy, genify vs stock:
- Qwen3.8 (real weights, layers 0 and 3): logits PCC 0.99997 (prefill M = 128, 512), 0.99999 (decode);
  same top-1 token at every position. `qwen38_test_genify_logits.py`
- FLUX.2 (random weights, 1 double + 2 single blocks): PCC vs diffusers 0.997519 (stock 0.997518),
  genify vs stock 0.999993; tt_dit unit tests 142/142 in both modes. `flux2_test_transformer_pcc.py`
- PDL (real weights): test_tt_model passes; semantic / center / offset PCC 0.98538 / 0.95384 / 0.99254
  (stock 0.98500 / 0.95974 / 0.99277).

Details: `qwen38_results.md`, `flux2_perf_table.md`, `pdl_profile_table.md`; running logs `flux2_notes.md`,
`pdl_notes.md`. genify's kernels keep MaybeDeviceZoneScope zones, active under the profiler; compiled
out, Qwen's genify matmul time drops by 0.3% (decode) to 1.5% (prefill).

## Gaps in the op (for a code-gen follow-up)
1. Hang: bf16 batched-B matmul `[1,32,512,128] @ [1,32,128,1024]`, HiFi2, fp32_dest_acc_en=False (writer
   tail split waits on the reader's K-walk). Repro: `pdl_test_upsample_hang.py`. PDL's bf16 upsample keeps ttnn.matmul.
2. No multi-output (split over N) and no two-segment A (concat over K): FLUX.2 adds 25.7 ms of chunk/concat per step.
3. Split choices: tall interleaved PDL shapes on few cores (32768x256x128: 41 us vs stock 9.4 + 24.7 us I2S),
   M = 1 MLPs (FLUX.2 linear_2 213 vs 195 us), text-encoder M = 512 bfp8 (80-88 cores), Qwen qkvg and o_proj
   at M = 128 (82 vs 53 us, 38 vs 26 us), decode qkvg (32 vs 29 us).

## Run
```
source python_env/bin/activate; export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
# accuracy
MODEL_WEIGHTS_DIR=<Qwen3.8-27B> scripts/run_safe_pytest.sh models/experimental/op_genify/integration/qwen38_test_genify_logits.py -s
TT_GENIFY_MATMUL=1 scripts/run_safe_pytest.sh models/experimental/panoptic_deeplab/tests/pcc/test_tt_model.py -k panoptic_deeplab_110_cores
TT_GENIFY_MATMUL=1 scripts/run_safe_pytest.sh models/experimental/op_genify/integration/flux2_test_transformer_pcc.py
# perf: profile with and without TT_GENIFY_MATMUL=1, then compare
TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=5000 scripts/run_safe_pytest.sh --profile models/experimental/op_genify/matmul_shape_set/test_qwen38_profile.py -s
python models/experimental/op_genify/integration/compare_qwen38.py STOCK.csv GENIFY.csv
```
FLUX.2: `test_flux2_profile.py` / `test_flux2_text_encoder_profile.py` + `flux2_perf_table.py`;
PDL: `test_tt_model.py --profile` + `pdl_profile_table.py`.
