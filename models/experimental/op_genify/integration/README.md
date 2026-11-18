# genify_matmul in the models (#65150)

The generated matmul (`ttnn/ttnn/operations/genify_matmul/`) replaces the stock single-chip matmuls of
Qwen3.8-27B, FLUX.2-dev and Panoptic-DeepLab. `TT_GENIFY_MATMUL=1` routes them through
`genify_matmul_adapter.py`; unset, the stock ops run unchanged. CCL-fused matmuls (#65143) are not routed.

The op on this branch is the output of the follow-up code-gen run (eval run 1151, tt-metal branch
`avelickov/2026_11_18_genify_matmul_roofline`, da2d3e5), which continued #65149's run 1137:
- Refinement 12 fixed a hang in bf16 batched-B matmuls.
- Refinement 13 added split output (`chunks=`).
- Refinement 14 added A as a list of K segments.
- Refinements 15-18 and two perf tournaments moved the op toward the #65148 roofline, which is now the
  golden suite's target on every model case.

FLUX.2 uses the split output and the K-segment input directly. The chunk / concat ops genify needed before
(25.7 ms per step) are gone.

Device kernel time per chip from Tracy, Blackhole QuietBox 2. Stock and generated were both measured on this
branch. Roofline from #65148 (`../matmul_shape_set/65148_comment.md`): max(compute at 70% of peak, DRAM
traffic at 90% of 512 GB/s). Because it assumes 70% of peak, a compute-bound op can beat it.

| Model | Per | Stock matmuls | Generated | Roofline | Generated / roofline | Model device time, stock -> generated |
|---|---|---:|---:|---:|---:|---|
| Qwen3.8-27B (1x4, TP4) | decode token | 10.12 ms | 8.85 (1.14x faster) | 8.17 | 1.08x | 22.38 -> 21.04 ms |
| | prefill M = 128 | 16.20 | 11.97 (1.35x) | 8.40 | 1.42x | 45.75 -> 40.74 |
| | prefill M = 512 | 31.04 | 22.80 (1.36x) | 15.01 | 1.52x | 87.74 -> 73.83 |
| | prefill M = 1024 | 44.16 | 34.80 (1.27x) | 29.42 | 1.18x | 131.5 -> 110.1 |
| | prefill M = 2048 | 80.82 | 61.69 (1.31x) | 58.66 | 1.05x | 241.5 -> 196.4 |
| | prefill M = 4096 | 155.85 | 119.80 (1.30x) | 117.32 | 1.02x | 477.9 -> 389.1 |
| FLUX.2-dev (2x2, SP2 x TP2) | denoising step | 318.06 ms | 268.75 (1.18x) | 303.10 | 0.89x | 370.4 -> 321.3 ms without collectives |
| | prompt encoding | 69.57 | 52.66 (1.32x) | 53.69 | 0.98x | 99.2 -> 82.7 ms without collectives |
| Panoptic-DeepLab (1 chip) | inference | 0.990 ms | 0.907 (1.09x); 0.981 vs 1.171 with the data movement around them (1.19x) | 0.33 | 2.75x | 5.759 -> 5.565 ms |

Every Qwen3.8 matmul is faster than stock. Every FLUX.2 matmul is faster except the guidance MLP
linear_1 (M = 1, 14.7 vs 10.7 us). PDL's sharded 1x1 convs are at stock (0.93-1.03x), and its upsample
H passes run at 0.55x stock.

Accuracy, genify vs stock (unchanged from the first integration):
- Qwen3.8 (real weights, layers 0 and 3): logits PCC 0.99997 (prefill M = 128, 512) and 0.99999 (decode);
  same top-1 token at every position. Test: `qwen38_test_genify_logits.py`.
- FLUX.2 (random weights, 1 double + 2 single blocks): PCC vs diffusers 0.997519 (stock 0.997518), with
  no call falling back to stock. Test: `flux2_test_transformer_pcc.py`.
- PDL (real weights): test_tt_model passes; semantic / center / offset PCC 0.98538 / 0.95384 / 0.99254
  (stock 0.98500 / 0.95974 / 0.99277). PDL's bf16 upsample module tests now run on genify too: 45 passed,
  34 skipped, and the former hang repro `pdl_test_upsample_hang.py` passes 6/6.

Details: `qwen38_results.md`, `flux2_perf_table.md`, `pdl_profile_table.md`; running logs `flux2_notes.md`,
`pdl_notes.md`. genify's kernels keep MaybeDeviceZoneScope zones, active under the profiler. In the first
integration, compiling them out cut Qwen's genify matmul time by 0.3% (decode) to 1.5% (prefill).

## Remaining gaps
1. Qwen3.8 prefill at M = 128 / 512 is 1.42x / 1.52x its roofline.
   - At M = 128 the bank-local weight stream is bound by multicasting the 1.3 MB activation (Refinement 15).
   - At M = 512, the 16 tile rows cap the 2-D split at 80-88 of the 110 cores (Refinement 16). A
     non-rectangular split is the next lever.
2. PDL: `genify_conv.py` reads some interleaved inputs in place. The op splits those tall, narrow shapes
   over too few cores:

   | Call | genify us | stock us | stock's input sharding us |
   |---|---:|---:|---:|
   | decoder project convs | 21.6 | 3.6 | 10.8 |
   | ASPP 1x1 | 28.6 | 13.8 | 6.6 |
   | res3.0 conv1 | 41.3 | 9.4 | 24.7 |

   The golden suite has only PDL's sharded layouts. Sharding these inputs first, as stock does, would save
   about 50 us per inference.
3. The upsample H pass, at 66.8 us, sits on its measured output-write floor of 62.6 us (17.8 MB at
   ~285 GB/s).

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
FLUX.2: `test_flux2_profile.py` / `test_flux2_text_encoder_profile.py` (MISTRAL3_CONFIG_DIR = the
Mistral-Small-3.1-24B config) + `flux2_perf_table.py`. PDL: `test_tt_model.py --profile` + `pdl_profile_table.py`.
Profiles behind the tables: stock 2026_11_17_20_25_44 (Qwen), 21_01_23 / 21_09_40 (DiT), 21_05_56 / 21_12_15
(TE), 20_32_48 (PDL); genify 2026_11_18_11_35_25 (Qwen), 11_39_12 / 11_41_22 (DiT), 11_40_08 / 11_42_16 (TE),
11_43_27 (PDL).
