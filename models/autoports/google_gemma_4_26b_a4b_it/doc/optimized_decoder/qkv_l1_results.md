# Full prefill QKV L1 controls

Full attention only, exact recorded real activations, FP32 input/output, BFP8 weights and HiFi2. Same-process alternating whole-prefill host timing includes input movement. Device profiling and integrated-policy validation are separate.

| Control | Tokens | Baseline median µs | Candidate median µs | Median paired delta µs | Faster pairs | Prefill PCC | Minimum decode PCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| m2_l1_vs_m4_dram | 4096 | 186594.618 | 185772.553 | -809.140 | 8/8 | 0.999122336917 | 0.995105687605 |
| n4_l1_vs_m4_dram | 4096 | 186562.724 | 186102.644 | -384.858 | 8/8 | 0.999122360056 | 0.995105687605 |
| m2_placement | 4096 | 186461.423 | 185809.753 | -664.925 | 8/8 | 0.999122336917 | 0.995105687605 |
| m2_placement_65 | 65 | 4696.310 | 4685.770 | -2.149 | 5/8 | 0.998991233581 | 0.999415889190 |
| m2_grid | 4096 | 185784.666 | 185816.137 | +2.836 | 4/8 | 0.999122336917 | 0.995105687605 |
| m2_producer | 4096 | 185804.629 | 185690.111 | -128.295 | 22/32 | 0.999122336917 | 0.995105687605 |

The first M4 L1 failure is a config/allocation overlap, not rejection of the entire L1 family. M2 and N4 retain K16 and change bounded geometry to fit.
Matched DRAM/M2 versus copied-L1/M2 isolates the coupled input/output placement family from the M4-to-M2 geometry change. It is not evidence of an input-only benefit: unspecified output memory inherits the input memory config.
The 65-token control checks three tiled M rows with an M2 block; its timings are a boundary sample, not headline performance.
Copy-L1/M2 versus direct-producer-L1/M2 holds the matmul program and L1 input/output tensor specs fixed. Only the final input-normalization gamma multiply output placement changes; other sites and decode remain original.
M2 copied-L1 grid88 versus110 is independent of the earlier M4 grid test. Both candidate and baseline use the same precision and L1 input spec.
No absolute geometry optimum or final stage acceptance is inferred from this bounded matrix.

Original M4/K16/N8 static CBs end at 1,307,648 bytes, beyond the lowest observed live L1 tensor allocation beginning at 1,114,112. The exception does not identify that allocation as input versus output. With the same 111,616-byte static base, M2/K16/N8 ends at 848,896 and M4/K16/N4 at 971,776. Raw failure and successful adapted runs are preserved separately.

The probes forward memory_config=None. Minimal matmul inherits the input memory config for its output: DRAM input yields DRAM output; L1 input yields L1 output. The matched placement control changes both boundaries together. Copy-L1 versus producer-L1 and grid88 versus110 preserve both L1 boundaries. Native v8 rows verify L1 input and output; historical raw reports are unchanged.

The JSON links exact commands, fixture/source hashes, all paired samples, quartiles, observed tensor memory and configured programs. The median of paired deltas is reported directly; it need not equal the difference between independently computed medians.

Historical paired-helper hash13f71dfb resolves to the exact preserved probe_optimized_prefill_pairs_before_format.py.txt snapshot. The later live-helper formatting is AST-equivalent; historical report hashes remain unchanged and are not rebound to the formatted helper.

Pending controls: none. Selected integration validation and final native profiles remain separate.
