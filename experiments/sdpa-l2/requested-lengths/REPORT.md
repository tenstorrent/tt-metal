# SDPA L2 at 25,920 and 75,600 tokens, five heads

## Follow-up: BF16 compensation explicitly enabled at both lengths

The user requested compensation ON for the improved measurement, including 25,920 tokens. A measurement-only host-factory override removed only `k_num_chunks >= 64` from the BF16 compensation guard. All remaining guards, Q/K chunk sizes, input buffering, and numerical kernels were unchanged. No Q preprocessing is used for improved BF16.

The following supersedes the improved-BF16 column of the original results below. Each cell is **relative L2 % / PCC**, averaged over seeds 1234, 1235, and 1236. The other three columns reuse the original measurements with identical inputs and reference protocol.

| Sequence length | Main BF16 dst | Main FP32 dst | Improved BF16 dst, compensation ON | Improved FP32 dst |
|---|---:|---:|---:|---:|
| 25,920 | 2.679217 / 0.99966285 | 2.397222 / 0.99971536 | 2.535934 / 0.99969993 | 0.492571 / 0.99998786 |
| 75,600 | 3.100203 / 0.99958334 | 3.048141 / 0.99964771 | 2.678271 / 0.99970345 | 0.492790 / 0.99998785 |

Compensation-enabled BF16 L2 ranges: **2.519416–2.547470%** at 25,920 and **2.660240–2.711533%** at 75,600. All three sampled-output hashes changed at 25,920; all three at 75,600 exactly match the previous compensated results, as expected. Full square device operations were run, with FP64 reference comparison on the last 128 query rows per head. All trace replay equality checks passed.

Raw results: `forced-comp-bf16-{1234,1235,1236}.jsonl`, with corresponding logs. `forced-comp-build.log` confirms recompilation of the transformer host object. The exact override is [force-bf16-compensation.patch](force-bf16-compensation.patch), applied on top of the retained improved source for measurement only, not a production guard change.

After measurement, the remote factory was restored and explicitly rebuilt. All four operator-source SHA256 checksums match the retained local source. Both seed-1234 BF16 smoke reruns match the original as-is improved-build sampled-output hashes and L2 exactly; see `post-forced-comp-restored.jsonl` and its build/run logs. Thus the allocated machine is left on the retained default guards, not the measurement override.

### Are improved FP32 streaming and non-streaming numerics the same?

**They are not guaranteed identical.** Both retain FP32 recurrent state and the improved Q preprocessing, but the current implementations differ:

- The padded non-streaming fallback used by both lengths here calls `calculate_sdpa_exp_hifi2`: a cubic exp refinement followed by explicit rounding of weights to six fraction bits before reduction and PV.
- The specialized streaming path calls `calculate_sdpa_exp_stream_effective`: a different cubic with a bias targeting the effective six-fraction-bit weights consumed by the FPU. Its denominator is accumulated using a LoFi P-times-ones matmul, so it sees the same effective weights as HiFi2 PV. QK and PV themselves remain HiFi2.
- Reduction and online-update scheduling also differ, so FP32 state alone does not imply bit-identical outputs.

There is also an unpadded non-streaming effective-weight specialization, using the same exp-fit coefficients as streaming but an explicit effective-weight reduction. Do not conflate that specialization with the padded fallback measured here. This table is **not** a controlled FP32 streaming-versus-non-streaming comparison: both requested lengths require generated padding masks, which the current FP32 streaming guard excludes. Similar aggregate L2 on other shapes is not proof of numerical equivalence.

## Results

Relative L2 percentages, **mean over seeds 1234, 1235, and 1236**:

| Sequence length | Main BF16 dst | Main FP32 dst | Improved BF16 dst | Improved FP32 dst |
|---|---:|---:|---:|---:|
| 25,920 | 2.679217% | 2.397222% | 2.679217% | 0.492571% |
| 75,600 | 3.100203% | 3.048141% | 2.678271% | 0.492790% |

Ranges across the three seeds:

| Sequence length | Main BF16 dst | Main FP32 dst | Improved BF16 dst | Improved FP32 dst |
|---|---:|---:|---:|---:|
| 25,920 | 2.664515–2.702276% | 2.379254–2.432646% | 2.664515–2.702276% | 0.490411–0.496878% |
| 75,600 | 3.082241–3.119938% | 3.032424–3.067920% | 2.660240–2.711533% | 0.491102–0.495264% |

All six improved-FP32 cases are below 0.5% L2. At 75,600, BF16 compensation reduces mean L2 from 3.100203% to 2.678271%. At 25,920, the BF16 compensation guard is inactive, so the improved build gives **identical sampled-output hashes to main for all three seeds**.

## Protocol

- Noncausal, batch 1, **heads=5, d=128, Q length = K/V length = requested sequence length**.
- Q/K/V are independently generated standard normal N(0,1), then rounded to BF16. All four versions use the same original tensors for each seed and length.
- HiFi2 for both matmuls, approximate-math/exp configuration enabled, BF16 inputs/output; only destination mode and the existing implementation changes differ.
- Fixed **Q/K chunks 128/512**, unchanged input buffering. No source guards were modified for these tests.
- Run the entire square device operation. Compute relative L2 against FP64 online softmax on the **last 128 query rows of each head**, not every output row: 100 * norm(output-reference) / norm(reference).
- Only improved FP32 applies its existing six-fraction-bit Q bit-ceil preprocessing with scale compensation 1.0027. Reference still uses the original BF16 Q. Main BF16, main FP32, and improved BF16 receive untouched BF16 Q.
- Device: one Blackhole P100A on yyzo-bh-26, 11x10 compute grid, reservation 215262. This is not a Galaxy measurement.
- Main is the original unmodified base commit `2ba6fc2339d53300ae87c5202f335ef56492cfb3`. Improved source is the retained [BF16 optimization](../bf16-perf-v1/REPORT.md), with the previously improved FP32 implementation unchanged.

The repro verifies finite outputs, its FP64 reference against dense softmax, and exact sampled-output equality between ordinary execution and trace replay. Two warmup and two measured trace replays are correctness checks here; no steady-state performance claim is made.

## Which paths execute?

| Build / mode | 25,920 | 75,600 |
|---|---|---|
| Main BF16 | Original BF16 streaming | Original BF16 streaming |
| Main FP32 | Original non-streaming FP32 dst | Original non-streaming FP32 dst |
| Improved BF16 | Original BF16 streaming: compensation guard off | Compensated BF16 streaming with replay helper |
| Improved FP32 | Improved non-streaming FP32 fallback | Improved non-streaming FP32 fallback |

The BF16 guard in [sdpa_program_factory.cpp](../../../ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp) requires at least **64 K chunks**. Length 25,920 has 51, while 75,600 has 148.

Both lengths require the generated padding mask with 128/512 chunks. The current improved FP32 streaming and effective-weight-denominator specializations exclude generated padding masks. Their fallback still retains FP32 recurrent state and the improved exp/Q-rounding algorithm. Thus these are **as-is improved-build results**, not measurements of the specialized 256K FP32 streaming path. Logical sequence lengths were preserved; padding was not silently counted as valid context.

## Raw results and reproduction

The 24 primary results are in `main-{bf16,fp32}-{1234,1235,1236}.jsonl` and `improved-{bf16,fp32}-{1234,1235,1236}.jsonl`, with matching logs. Each file contains both lengths.

[run.sh](run.sh) runs the matrix against the currently installed source snapshot:

```bash
# With the matching source snapshot built, and PYTHONPATH / ARCH_NAME / TT_METAL_HOME configured:
bash experiments/sdpa-l2/requested-lengths/run.sh main 1236 1234 1235
# Restore and rebuild the improved snapshot, then:
bash experiments/sdpa-l2/requested-lengths/run.sh improved 1236 1234 1235
```

Use a fresh output directory before rerunning: the script refuses to overwrite results. The initial improved seed-1236 pair was run with equivalent direct commands before the driver was added.

Main and restored-improved host builds explicitly touch the SDPA host translation unit to prevent stale-object reuse after snapshot restoration. The operator sources were not edited for this experiment. Final restored-source rechecks and checksums are recorded alongside the raw results.
