# Prefill output-projection selection

The selected integration plan is **grid `(11,8)`, K block 16, LoFi, inherited DRAM input for both attention kinds**. All 28 recorded cases pass real-input 4096-prefill/128-decode checks. Maximum-context all-sampled-row validation remains pending. This report reads existing artifacts only; it performs no hardware execution or runtime edit.

All cases use runtime SHA `b513a1b40988b33a359acb7d7809696eebf81a7197756e3ec3f943692182ab83`, checkpoint revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, and the same recorded text-derived fixture per representative layer (0 sliding, 5 full). The [machine-readable summary](prefill_output_results.json) preserves exact PCCs, every timing sample, fixture/source hashes, commands, observed tensor/program settings, and matched-baseline checks.

## What was tested

The [probe](../../tests/probe_optimized_prefill_output.py), SHA `561fd4a9fff0747e065dcea9e62921753fa1e5838919e681b9e0ae4543ab63cd`, replaces only the prefill output projection. Its decode branch delegates to the original `project`; attention, cache, expert, and router policies remain unchanged. It retains BF16 concatenated-head input, BFP8 weights, FP32 accumulation/output, DRAM output, and disabled approximation/packer accumulation. All explicit programs are created at setup, with grid `(11,8)` and legal subblocks. At 1024 rows the selected program uses per-core M4/N8 and subblock 1x4.

The matched baseline uses the same probe with `--prefill-output-mode auto`, HiFi4, DRAM input, and `program_config=None`. It is the controlled automatic-program baseline, not a separate fused-decoder baseline. Runtime/probe hashes, fixture hashes, and all non-probe precision-policy fields match within each layer. All 128 decode-position checks match the corresponding baseline exactly, and repeated traced decode is equal.

The [initial journal](prefill_output_commands.json) records 18 successful commands: per layer, automatic baseline; HiFi4 K2/K4/K8/K16/K32; HiFi2/LoFi K4; and HiFi4 K4 with L1 input. The [cross-fidelity journal](prefill_output_cross_commands.json) adds eight successful K16/K32 HiFi2/LoFi cases. The [L1 journal](prefill_output_l1_commands.json) adds the two K16 LoFi L1-input controls, each naming its DRAM baseline.

## Exact PCC results

Within each layer and fidelity, all tested K blocks/input-memory choices have the same recorded prefill PCC. Every prefill and decode check passes the 0.995 threshold; runtime audits and program-cache guards pass.

| Output-projection fidelity | Sliding prefill PCC | Full prefill PCC |
| --- | ---: | ---: |
| HiFi4 | 0.9991598414846561 | 0.9991318001817636 |
| HiFi2 | 0.9991603933468373 | 0.9991324146537388 |
| LoFi | 0.99915285133721 | 0.9991236270227776 |

Minimum decode PCC is `0.9953353194067491` for sliding and `0.9951361141534202` for full attention in every case. These headline results do not close the pending maximum/near-maximum-context gate.

## Whole-prefill host timing

Each value below is the median of three warmed, synchronized whole-decoder prefill wall-time samples in microseconds. `run_decoder.py:179-189` times device-only prefill through the final synchronization; preceding synchronization and output deallocation are outside the interval. Initial and guarded prefill runs precede the samples. These are not isolated GEMM times or device-profiler measurements.

| Candidate | Sliding median, µs | Full median, µs |
| --- | ---: | ---: |
| Auto HiFi4, DRAM | 225,132.786 | 195,118.534 |
| K2 HiFi4, DRAM | 224,074.701 | 193,287.808 |
| K4 HiFi4, DRAM | 223,806.149 | 192,112.119 |
| K8 HiFi4, DRAM | 223,556.383 | 191,832.315 |
| K16 HiFi4, DRAM | 223,320.102 | 191,573.212 |
| K32 HiFi4, DRAM | 223,338.428 | 191,282.605 |
| K4 HiFi2, DRAM | 223,535.893 | 191,395.536 |
| K16 HiFi2, DRAM | 222,919.430 | 190,953.907 |
| K32 HiFi2, DRAM | 222,805.344 | 190,520.606 |
| K4 LoFi, DRAM | 223,032.645 | 191,045.919 |
| **K16 LoFi, DRAM — selected** | 222,689.914 | 190,090.823 |
| K32 LoFi, DRAM | 222,756.139 | 190,267.300 |
| K4 HiFi4, L1 | 223,746.935 | 192,185.830 |
| K16 LoFi, L1 | 222,674.718 | 190,323.226 |

Relative to the matched automatic HiFi4 baseline, selected K16 LoFi/DRAM reduces the observed full-prefill median by **2,442.872 µs (1.085081%) sliding** and **5,027.711 µs (2.576747%) full**. K16 has the lowest DRAM-input median among these measured settings. The K32 LoFi medians are only 66.225 µs and 176.477 µs higher, respectively, within the observed spreads; this short sweep does not establish an absolute optimum.

## L1 input decision

| Attention kind / K16 LoFi input | Three samples, µs | Median, µs | Range width, µs |
| --- | --- | ---: | ---: |
| Sliding, DRAM | 223275.888, 222338.546, 222689.914 | 222689.914 | 937.342 |
| Sliding, L1 | 223356.976, 222674.718, 222598.864 | 222674.718 | 758.112 |
| Full, DRAM | 190994.180, 190090.823, 189985.102 | 190090.823 | 1009.078 |
| Full, L1 | 191011.375, 190323.226, 190142.923 | 190323.226 | 868.452 |

Sliding L1 is 15.196 µs (0.006824%) below DRAM at the median, while the ranges overlap and span 937.342/758.112 µs. The sliding median absolute deviations are 351.368/75.854 µs. The median difference is smaller than the variation visible in these samples, so it does not demonstrate a useful L1 advantage. Full L1 is 232.403 µs (0.122259%) slower at the median; those ranges also overlap. These are descriptive comparisons, not a statistical significance test.

The L1 probe explicitly calls `to_memory_config(combined, L1_MEMORY_CONFIG)` after head concatenation; inherited input is already DRAM in the observed records. Retaining DRAM avoids adding that movement and does not sacrifice a demonstrated timing advantage. The K4 HiFi4 L1 controls show similarly small mixed timing differences. The names `selected_l1_*` identify trials, not the final selected input-memory policy.

## Integration and reproduction

The [candidate patch](prefill_output_projection.patch), SHA `8a011638a61c0cf92c3a0d88ad23456da699d90eb329fd94af0c8ee5e5239371`, is the opt-in implementation proposal used for subsequent integration. Its initial defaults (`grid=None`, `fidelity=None`, K4, `input_l1=False`) preserve the existing path. Applying it alone does not select this result: the selected setup is `prefill_output_grid=(11,8)`, `prefill_output_block_w=16`, `prefill_output_fidelity=ttnn.MathFidelity.LoFi`, and `prefill_output_l1=False`. The selected projection is now integrated in runtime SHA `3d51014f98128dfb21bb484fcece50993524b967754f68f6dfe825ae7f472ba9`; this later source also selects all sharded norm sites for full attention. The candidate measurements above remain bound to `b513a1b4…`. Do not reapply the patch. Cumulative public contracts, representative profiles, and maximum-context gates remain pending for the new source.

The exact argv arrays and return codes are preserved in the linked journals and summary. One selected candidate command is:

```bash
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_optimized_prefill_output --prefill-output-block 16 --prefill-output-fidelity LoFi --defaults --layer 0 --length 4096 --real --input-fixture models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/actual_text_layer0_4096_128.pt --decode --steps 128 --timing --prefill-timing --verify-program-cache --output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/prefill_output_k16_lofi_layer0.json
```

The full-attention command replaces layer/fixture/output suffix 0 with 5. The L1 control adds `--prefill-output-input-memory l1`; the automatic baseline uses `--prefill-output-mode auto` with default HiFi4. Reproduction requires unused output paths because the probe refuses to overwrite evidence. No additional environment settings are inferred from the command journals.
