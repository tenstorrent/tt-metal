# Repair the high-core geometry family with exact K divisors

V5's `geometry120.json` measured all C8/16 high-reader cases slower than the approximately 0.425 ms control (best candidate approximately 0.455 ms). Its two C64 cases failed before execution with the exact validation:

```text
matmul_device_operation.cpp:3115: K % per_core_K == 0
in DRAM sharded Matmul we don't have support for un-even sharding currently.
K: 160, per_core_K: 3.
```

The earlier V5 source audit established valid excess-capacity input tensor construction and safe factory K addressing, but missed an additional `compute_output_specs` condition. That output-spec branch unconditionally requires `K % per_core_K == 0` before the otherwise-capable multicore factory is reached. V5's exact activation checks do not bypass this assertion. The C64 result is an API validation blocker, not proof that larger storage grids are unhelpful.

The immutable repair is `tests/run_attention_geometry_probe_v6.py`. It retains K5120 (160 tiles) and the original prequantized weight elements, choosing C40 with four K tiles/shard and C80 with two. Both satisfy the output-spec divisibility condition and the factory capacity rule. Every grid core has real input data, so there are no trailing empty shards. K block1 and CK1 preserve the existing K grouping; output equality remains a measured question because prior higher-reader candidates changed output bits.

Eight candidates follow the original native C32/R4 control:

| Storage cores | Readers/bank | K tiles/shard | Physical N | Compute workers | DST passes |
| --- | --- | --- | --- | --- | --- |
| 32 | 4 | 5 | 17408 | 32 | 3 |
| 8 | 5 | 20 | 16640 | 40 | 2 |
| 8 | 6 | 20 | 16896 | 48 | 2 |
| 16 | 6 | 10 | 16896 | 48 | 2 |
| 40 | 5 | 4 | 16640 | 40 | 2 |
| 40 | 10 | 4 | 17920 | 80 | 1 |
| 80 | 5 | 2 | 17920 | 40 | 2 |
| 80 | 10 | 2 | 17920 | 80 | 1 |

The intermediate R5/R6 controls test less bank/NoC contention while reducing N padding. C40/R5 versus C8/R5 isolates activation/output storage placement at the same physical weight size and worker count. C40/R10 versus C80/R10 likewise isolates storage layout. These are hypotheses, not inferred speedups.

Padding logical K to 6144 is deliberately excluded: it adds 20% weight traffic. Even using the old N17408, the two BFP8 matrices' ideal 512 GB/s read floor rises from approximately 0.370 to approximately 0.444 ms, already above the measured 0.424 ms corrected-projection control before other work. Exact-divisor layouts test the family without that penalty.

V6 preserves V5's immediate host capture snapshots, exact activation round-trip and logical/padded K checks, actual active-core verification, exact stored-weight checks, HiFi4/packerFalse FP32 partials, BF16 addition, full corrected-projection timing, references, and per-row error recovery. It does not edit delivery or previous measured probes. Native and transformed C32/R4 controls remain available in every run.

Under the parent's current final decoder environment and device lock:

```sh
python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.run_attention_geometry_probe_v6 \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/geometry_divisors_results.json \
  > models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/geometry_divisors_results.log 2>&1
```

Syntax and hardware-free plan generation pass. `candidates/check_geometry_divisors_v6_cpu.py` checks both output-spec divisibility and factory capacity/ownership, unchanged timed-function ASTs, and no empty input shards. Results and source hash are in `geometry_divisors_v6_cpu.json`; plan is `attention_geometry_v6_plan.json`. No hardware run was performed while creating V6. This repair continues testing the material higher-core hypothesis after the first API error rather than dismissing it.
