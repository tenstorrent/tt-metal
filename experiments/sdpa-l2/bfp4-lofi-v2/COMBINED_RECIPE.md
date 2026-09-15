# Combined targeted repairs: isolated device qualification driver

`combined_recipe_fullchip.py` adds no kernel changes and does not mutate imported modules. It reuses the V-transposed driver primitives/kernels, the existing signed Hadamard preprocessor, and the existing adaptive BFP4 preprocessor. Only the new driver and its stdlib tests are added. Completed hardware qualification and its conditional benefits/regressions are reported below.

The scope is **full-compensated BF16 LoFi streaming**, fixed Q256/K512/D128, two K slots and two V slots. K/V formats remain independently selectable. Full-chip geometry, chain communication and recurrent state are identical to the V-axis driver.

| Independent axis | Off/control | Enabled |
|---|---|---|
| `--h16` | Original Q/K into existing quantizers | Same signed unnormalized H16 on Q and K, real HiFi4/FP32-DST dense matmuls with BF16 outputs; SDPA scale divided by 16 |
| `--v-transposed` | V groups across 16 D channels | Exact BF16 V transpose, groups across 16 N tokens, existing transposed-PV wrapper |
| `--adaptive-v` | `none`: existing shared-exponent RNE | `baseline`, `minus`, or `pm` exponent search, **V only**, requires V4; K quantizer unchanged |
| `--grid7-exp` | Existing native grid | Existing direct seven-significant-bit grid |

Default format is K4/V4; all optional axes are off. The CLI always enables preprocessing checks. No fallback, K/Q centering, adaptive K search, changed chunk size, or additional KV slots are introduced. H16 and V-axis group changes are not algebraically interchangeable: the first rotates features of Q/K while preserving exact logits in real arithmetic; the second changes which V values share a lossy quantization exponent.

## Gates and accounting

- Original immutable BF16 Q/K/V is always the FP64 attention reference. At N1024 all output rows are referenced; longer lengths reference explicitly recorded rows and all KV.
- Actual device BF16 H16 outputs feed Q/K quantizer oracles. H16 error against its FP64 transform is measured, not required to equal ideally rounded BF16; that hardware/spill error remains in end-to-end L2/PCC.
- V transpose, original-input immutability, and two combined trace replays compare BF16 storage bits, including signed zero. Quantizers require exact decoded-value equality against the actual input tensor; adaptive V uses the established FP32 arithmetic/selection oracle.
- Every output must be finite before timing. Optional `--max-l2` adds an operator gate. Sources are hashed before and after, including imported helpers, selected frozen `.h`/`.hpp`, original reference, Hadamard matmul sources, transpose/no-MOP sources and adaptive quantizer.
- Combined timing recomputes every selected transform and quantizer on device. Disjoint stage timers are Q rotation, K rotation, V transpose, Q quantization, K quantization and V quantization; disabled transforms are absent. Preprocessing and combined totals are separately measured, not assembled by adding stage medians. Host constants/uploads are excluded; the H16 implementation is dense matmul, not a free or butterfly transform.

## Focused inputs and smoke plan

In addition to original distributions, `channel_v` multiplies every sixteenth BF16 V channel by 32. `k_outliers_channel_v` keeps normal Q, uses the existing sparse-outlier K generator, and adds that persistent V channel imbalance. `outliers_channel_v` uses sparse outliers in Q/K/V and the same V channel imbalance. The latter two directly test composition. Quiet-channel output error is reported separately for all three.

Start with a 2×2 ablation on `--h16` and `--v-transposed`, adaptive V off. Example combined corner:

```sh
python_env/bin/python experiments/sdpa-l2/bfp4-lofi-v2/combined_recipe_fullchip.py \
  --label recipe-h16-vn-1024-v1 --length 1024 --heads 2 --cores 4 \
  --kv-formats b4_b4 --grid7-exp --h16 --v-transposed \
  --distributions normal constant_v k_outliers_channel_v --iters 0
```

Repeat with fresh labels and neither flag, H16 only, and V-transpose only. Then rerun the combined corner with `--adaptive-v pm`; an optional `--adaptive-v baseline` control should preserve quantized values relative to the existing RNE path. Do not assume group-MSE optimization must improve attention error.

Only after those gates pass, use N32768/seed1240 normal, outliers, channel V, and combined stress for a small qualification. Include common Q/K as explicit negative controls before any broader robustness claim: existing H16 experiments can substantially worsen them, and this recipe intentionally does not repair their common modes. No model-quality claim follows from operator tests.

## Static verification

Python compilation and three stdlib tests pass. Eighty combinations cover K4/K8, V4/V8, H16 on/off, V-axis on/off, native/grid7 and all valid adaptive V choices. They verify unchanged CBs/jobs/two slots, correct scale division, V-only adaptive routing, actual source shapes, and the exact preprocessing invocation order. Additional tests reject adaptive V8, MAIN/denominator-only modes, and verify every source-pin path. These are not C++ compilation or device execution.

## Measured combined-recipe qualification

Five complete device ledgers: [Plain D](recipe-32768-h0-vD-none-v1.jsonl), [H16 + D](recipe-32768-h16-vD-none-v1.jsonl), [Plain N](recipe-32768-h0-vN-none-v1.jsonl), [H16 + N](recipe-32768-h16-vN-none-v1.jsonl), [H16 + N + adaptive](recipe-32768-h16-vN-pm-v1.jsonl). The following tables are generated from their JSON; raw records are authoritative.

All 25 rows use N32768/H10/D128, 110 cores, Q256/K512, two KV slots, full BF16 compensation, K4/V4, **native exp, not grid7**, barrier 2 and seed 1240. References use original BF16 Q/K/V, all KV and 128 recorded Q rows per head. Exact preprocessing, all-output finite, original-input immutability, bitwise replay and unchanged-source checks pass; all five runs share 68 source hashes. H16 comparisons include actual device BF16 transformed spills. For normal input the Q/K transform L2 against FP64 is about 0.167% each; millions of differences from ideally rounded BF16 are measured rather than substituted away.

Each accuracy cell is **operator L2 % / PCC**. No gain fitting or reference substitution.

| Input | Plain D | H16 + D | Plain N | H16 + N | H16 + N + adaptive |
|---|---:|---:|---:|---:|---:|
| normal | 16.938 / 0.985741 | 16.945 / 0.985717 | 16.970 / 0.985675 | 16.966 / 0.985674 | 16.542 / 0.986274 |
| outliers | 32.444 / 0.947066 | 17.184 / 0.985317 | 32.786 / 0.945868 | 17.590 / 0.984577 | 17.122 / 0.985271 |
| k_outliers_channel_v | 44.206 / 0.896984 | 23.895 / 0.971048 | 43.775 / 0.899194 | 23.888 / 0.971165 | 23.377 / 0.972292 |
| common_k | 80.354 / 0.595110 | 95.724 / 0.465839 | 80.373 / 0.594874 | 95.739 / 0.464947 | 95.409 / 0.465496 |
| common_q | 33.974 / 0.944506 | 48.651 / 0.881649 | 33.842 / 0.944946 | 48.369 / 0.882791 | 48.175 / 0.882563 |

For the combined K-sparse-outlier / V-channel-imbalance input, the quiet-channel result is more revealing than global L2:

| Recipe | Quiet L2 % | Quiet PCC | Quiet gain |
|---|---:|---:|---:|
| Plain D | 83.940 | 0.572559 | 0.430881 |
| H16 + D | 80.744 | 0.619715 | 0.501635 |
| Plain N | 42.264 | 0.906366 | 0.831463 |
| H16 + N | 23.055 | 0.973073 | 0.951767 |
| H16 + N + adaptive | 23.143 | 0.972865 | 0.941641 |

The two repairs **do compose for their targeted failure mechanisms**: H16 reduces operator error on sparse-outlier Q/K inputs, and N-group V protects quiet value channels. Neither alone achieves the combined quiet-channel result. But this is not an unconditional recipe: H16 worsens common K from about 80% to 96% L2 and common Q from 34% to 48–49%. V transposition does not repair those score errors. Normal K4/V4 accuracy remains around 17%.

Adaptive V slightly improves global L2 in all five measured rows, but the improvement is conditional at finer granularity: on combined stress it moves quiet L2 23.055→23.143%, slightly worse, despite global 23.888→23.377%. Group reconstruction MSE is not the attention-error objective. Its extra preprocessing cost must therefore be justified by a real workload, not selected just because a local quantizer score improves.

### Measured preprocessing and total time

Normal-input milliseconds, three measured trace replays after two warmups. Disabled stages are marked —. Each stage and total is separately measured; stage medians must not be added to the already-inclusive combined total. Other distributions have very similar kernel/stage timings in these ledgers.

| Stage / total | Plain D | H16 + D | Plain N | H16 + N | H16 + N + adaptive |
|---|---:|---:|---:|---:|---:|
| Q rotation | — | 0.682 | — | 0.681 | 0.679 |
| K rotation | — | 0.680 | — | 0.681 | 0.682 |
| V transpose | — | — | 0.535 | 0.522 | 0.518 |
| Q quantization | 0.526 | 0.526 | 0.527 | 0.527 | 0.527 |
| K quantization | 0.376 | 0.376 | 0.377 | 0.375 | 0.375 |
| V quantization | 0.375 | 0.377 | 0.375 | 0.376 | 1.427 |
| Attention | 27.678 | 27.614 | 27.623 | 27.613 | 27.606 |
| Preprocessing total | 1.174 | 2.437 | 1.642 | 2.902 | 3.961 |
| Combined total | 28.759 | 30.018 | 29.203 | 30.464 | 31.565 |

Normal combined overhead versus plain D: H16+N 1.705 ms (5.930%); adding adaptive V then costs another 1.101 ms (3.614%). Attention itself remains about 27.6 ms. These are sequential measurements, not the forthcoming interleaved timing experiment; small differences should not be assigned to an optimization without paired evidence. No clock/power overrides were used.

## B8 V-axis control changes the practical comparison

Separate K8/V8 full-compensated BF16/native-exp controls use the same H10/N32768 geometry and seed 1240, exact gates, and source hashes as the earlier K8/V4 V-axis tests. Five timed replays after three warmups. Sources: [B8 D](vt-b8-full-32768-D-v1.jsonl), [B8 N](vt-b8-full-32768-N-v1.jsonl), [B4 D](vt-full-32768-D-v1.jsonl), [B4 N](vt-full-32768-N-v1.jsonl). K remains B8 in this table; it is **not** a direct K4/V4 recipe comparison.

| V format/axis | Normal L2 % | Normal PCC | Channel-V global L2 % | Quiet L2 % | Quiet PCC | Normal attention ms | Normal combined ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| B4 / D | 12.007 | 0.992816 | 11.965 | 79.662 | 0.628216 | 27.642 | 28.777 |
| B4 / N | 12.038 | 0.992791 | 12.444 | 12.012 | 0.992818 | 27.649 | 29.221 |
| B8 / D | 3.151 | 0.999533 | 3.478 | 11.454 | 0.993935 | 27.754 | 28.850 |
| B8 / N | 3.148 | 0.999534 | 3.244 | 3.142 | 0.999536 | 27.709 | 29.300 |

[All-row N1024 B8 D](vt-b8-smoke-D-v1.jsonl) and [B8 N](vt-b8-smoke-N-v1.jsonl) smokes also pass exact transpose/quantizer/replay gates: quiet-channel L2 improves 11.514→3.064%. Normal L2 is 3.092% for both. Constant V remains 0.633% L2 for both, with PCC undefined, so this is not a normalization/recurrence fix; these controls use native exp, not grid7.

The V-axis benefit is not specific to BFP4. B8 N preserves quiet channels much better than B4 N and has almost the same observed time here. Existing LoFi PV uses the same matmul fidelity/replay count for both formats; this experiment does not receive an automatic extra compute-rate multiplier from choosing BFP4. B4 instead saves storage and transfer bytes: 576 versus 1088 bytes per tile, 47.1% of V storage, or 23.5% of combined K8+V bytes. With two 64-tile V slots it also saves 64 KiB of L1 per core. Those are real capacity/bandwidth advantages, but they have not produced a material throughput advantage in this tested compute/compensation-heavy geometry.

Engineering implication: prioritize B8 N as the reference low-precision candidate for this workload, and retain B4 only where a measured memory-capacity/bandwidth advantage or actual-model tolerance justifies its extra error. This is not a proof of universal B8 dominance: B8 N has only normal/channel-V 32K tests and small constant-V smokes here, not matched broad/256K qualification or a paired B8/B4 timing sweep. Before wider claims, test B8 N on the same outlier/common-mode suite and relevant real activation/KV-cache workloads; compare total prep+attention cost, context capacity and downstream model quality. For K4/H16, either constrain the admissible activation regime or independently qualify the necessary centering/correction path; do not silently enable a transform known to regress common modes.
