# Full prefill QKV advice predicates and matched grid control

The installed report has no remaining fidelity recommendation for FP32×BFP8→FP32 at HiFi2. Moving the input to L1 removes its placement advice. A sufficiently faster operator can instead trigger the110-core recommendation; the matched M2 grid control tests that possibility before selection. This is source/CPU evidence, not a prediction of measured latency or a new native profile.

[The structured audit](prospective_qkv_advice_v7.json) records source hashes, six CPU-executed advice cases, formulas and configuration metadata. The measured runtime snapshot is `daa82a4a5197a007ccd5d29d912f082e695fd37a596578f25fba96a6694e625b`. The installed `tt_perf_report/perf_report.py` SHA is `4a73ba4cb93cf84a8f6f5d2efde25c09ad71a120a878f169713f67dee6773983`.

| Predicate | Exact source behavior | Disposition |
| --- | --- | --- |
| Fidelity | `evaluate_fidelity`, lines845–855: FP32 input/BFP8 weight/FP32 output/HiFi2 returns sufficient and no advice | No lower/higher fidelity recommendation follows from this candidate |
| Input placement | `generate_matmul_advice`, lines1743–1746: SLOW rows advise L1 only if input memory lacks L1 | Copy-L1 and producer-L1 satisfy this operand predicate; their whole-layer costs still require measurement |
| Minimal config parsing | Lines1374–1386 only inspect `program_config` and ordinary `in0_block_w`/`out_subblock_*` names | Minimal `config=MinimalMatmulConfig` still produces the false missing-config message. Actual K16/subblock1×4 is explicit; supplying these fields to the CPU predicate returns that they look good |
| Grid | Lines1413–1420 classify FLOP when FLOPs≥65% and DRAM<65%; lines1729–1740 advise growth below110 cores | For1024×2816×9216 at88 HiFi2 cores this is336.082µs or faster, subject to the DRAM predicate. Measure grid88 versus110 at matched M2 before any new selection |
| DRAM sharding | Lines1720–1724 advise DRAM sharding for DRAM/BOTH rows | With both input and output in L1 the report threshold is77.982µs. This is its nominal-byte heuristic; it omits BFP exponent bytes and is not the final roofline numerator |

The current v7 full QKV native rows are525–560µs, classified SLOW. The adapted M2 control improves whole-prefill timing, but that does not establish its native operator duration. No fabricated candidate-native row is used in this audit.

## Geometry and controls

The minimal program factory, lines246–305, partitions M across grid Y and N across grid X because M<N. For1024×2816×9216,11×8 gives4 Mtiles/core and27 Ntiles/core;11×10 still gives4×27 because M tiles round32→40. At M2/K16/N8 both grids execute2 Mblocks×4 Nblocks per core and6 Kblocks. The larger grid adds padded M work rather than reducing this per-core partition. This explains why a larger grid need not win; it does not replace a matched measurement.

The earlier M4 grid comparison is not relabeled as M2 evidence. [The new probe](../../tests/probe_optimized_prefill_qkv_m2_grid.py) compares copied-L1 M2/K16/N8/sub1×4/HiFi2 grid11×8 against11×10. Both variants preserve the same BFP8 weight/compute objects, FP32 activation/output, implicit L1 output and decode path. Both warm twice before eight alternating timing pairs; cache misses are forbidden. The final candidate output/cache feeds the ordinary real-input prefill and128-step HF gates. CPU mocks verified the grid-only change, both copies and setup-only configuration creation; pinned formatting and pre-commit passed. Parent-run hardware results are CPU-verified below; the probe remains frozen.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_optimized_prefill_qkv_m2_grid --pairs 8 --defaults --real --layer 5 --length 4096 --decode --steps 128 --input-fixture models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/actual_text_layer5_4096_128.pt --verify-program-cache --timing --output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/prefill_qkv_m2_grid_v7_layer5.json
```

No installed-report source, runtime, existing probe or acceptance threshold was changed.

## Completed matched grid result

[The command journal](prefill_qkv_m2_grid_v7_command.json) completed with return code0 at `daa82`, now retained exactly in [the pre-placement source snapshot](runtime_v7_before_placement.py.txt). [The actual-input report](prefill_qkv_m2_grid_v7_layer5.json) passes prefill PCC0.999122336916838 and every128-step decode gate (minimum0.9951056876047375), with exact repeated trace outputs. All16 timed calls run with cache misses forbidden; entries remain176 after both warmups and samples. Source hashes, actual FP32 L1 inputs, per-variant M2/K16/N8/subblock1×4 grids and alternating order were checked from the report.

| Whole-prefill host statistic |88-core baseline |110-core candidate |
| --- | ---: | ---: |
| Median, µs |185784.6665 |185816.1365 |
| Candidate faster pairs | — |4/8 |

The median paired candidate-minus-baseline delta is **+2.8355µs**. Retain88 cores: the larger grid shows no resolved gain. These are same-process whole-prefill host measurements, including the same input-copy boundary and final synchronization; they are not native operator timings. Both sides use copied L1 input. Direct producer-L1 selection is a separate control and is not claimed resolved here. The result is never rebound to later runtime source hashes.

## Output-memory interpretation correction

The frozen probe narrative said DRAM output, but its actual recorded request is `memory_config=None`. `minimal_matmul_device_operation.cpp:297` uses `output_mem_config.value_or(input.memory_config())`; both copied-L1 grid variants therefore produce L1 output. Current v8 native rows independently confirm the same rule for producer-L1 input. The grid comparison still matches input/output placement, dtype and compute exactly. Its measured conclusion is unchanged; frozen probe/report bytes are preserved. The DRAM recommendation threshold above is corrected to77.982µs because only the BFP8 weight is in DRAM for this operator.
