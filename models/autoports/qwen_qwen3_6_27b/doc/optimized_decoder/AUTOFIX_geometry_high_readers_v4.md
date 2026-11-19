# Packed projection: higher reader count

V4 is `tests/run_attention_geometry_probe_v4.py`, an immutable packed-only derivative of the successful V2 harness. It tests C32 activation/output storage cores, K block1, and R4/8/9/10/12/13 reader/compute workers per DRAM bank. It measures the original native C32/R4/B1 control first, then includes a reformatted R4 matched control. The entire corrected projection is measured: two FP32-output matmuls plus BF16 add and movement. The parent's reported V2 approximately 0.424 ms refers to this entire corrected projection, **not each matmul**.

## Source-supported opportunity

`matmul_device_operation.cpp` permits any positive reader count when `use_multicore_pipeline=True`; the later tensor-aware eligibility check determines whether the pipeline is selected. `matmul_utilities.cpp::use_dram_sharded_multicore` requires full 32×32 tiles, no bias/fused activation/untilize, tile-aligned K, L1 activation and width-sharded L1 output. These are the captured packed-projection conditions. The legacy non-pipeline reader cap of three does not apply.

`get_dram_bank_adjacent_workers` requires `cores_per_bank * num_banks <= grid.x * grid.y`. At eight banks and 110 worker cores, R13 uses 104 and R14 would require 112. The pipeline factory additionally requires fewer than 256 role-table entries. V4 lets the real API enforce device bounds per row and captures exact recoverable exceptions; it does not abort the sweep because one case fails.

For B32, padded M is one tile. The pipeline retains up to eight output-column tiles in FP32 DST per column pass. Factory values are `CN=min(R,Nbt)`, `CK=1` whenever R≤Nbt, `ncmax=ceil(Nbt/CN)`, and `P=ceil(ncmax/8)`. Increasing R changes only column ownership here: all cases keep CK=1 and K block1, so no K splitting/reduction or precision-policy change is introduced. Existing actual-operand references and native comparison must establish device numerical equivalence; source inspection alone is insufficient.

| C32 readers/bank | Workers | Physical N | Max column tiles/worker | DST passes | Pair weight-read floor at 512 GB/s |
| --- | --- | --- | --- | --- | --- |
| 4 | 32 | 17408 | 17 | 3 | 0.370 ms |
| 8 | 64 | 18432 | 9 | 2 | 0.392 ms |
| 9 | 72 | 18432 | 8 | 1 | 0.392 ms |
| 10 | 80 | 20480 | 8 | 1 | 0.435 ms |
| 12 | 96 | 18432 | 6 | 1 | 0.392 ms |
| 13 | 104 | 26624 | 8 | 1 | 0.566 ms |

The floor counts the two packed BFP8 weight matrices as `2 * (K*N/1024) * 1088` bytes and assumes perfect 512 GB/s reads. It excludes other movement, compute and dispatch. More workers can reduce compute passes while increasing padding, NoC contention, and bank sharing. Therefore fewer passes do not establish a speedup. R13/C32 is especially padding-heavy and remains an explicit measurement control rather than an assumed winner.

## Probe and evidence

The numerical kernels, captured precision configuration, trace measurement, exact prequantized-weight preservation checks, and immediate host capture snapshots are unchanged from V2. Only the selected key, plan, and report metadata change. Existing geometry errors remain recorded per row with full exception and traceback. Each candidate must preserve actual stored BFP8 values before timing; setup retries supported by V2 remain available. JSON/CSV/`.inputs.pt` artifacts and low-volume console output remain unchanged.

Under the parent's current delivery118 environment and serial device ownership:

```sh
python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.run_attention_geometry_probe_v4 \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/geometry_high_readers_results.json \
  > models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/geometry_high_readers_results.log 2>&1
```

Syntax and hardware-free plan generation pass. `candidates/check_geometry_high_readers_v4_cpu.py` checks unchanged capture/timing/projection/metric ASTs, per-column ownership without gaps/overlap, unchanged K groups, and predicted worker/pass counts. `geometry_high_readers_v4_cpu.json` records results, source hash and calculated byte floors. No hardware was used to create this probe.

## Further parallelism notes

The existing policy builder cannot directly use C64 for K5120 because `k//cores=80` is not a tile-aligned shard width. This is not a hard hardware limit: `TensorSpec::get_shape_fits_shard_grid_error` allows excess shard capacity, and the pipeline checks `x_shard_w * x_cores.size() >= Kt`. An explicit C64 grid with 96-value/3-tile shards could cover K160 with unused trailing shard capacity, without padding the logical K or original weights. It needs a separate decoder-local shard-shape configuration and actual input round-trip/geometry validation. V4 does not modify this.

A further R13/C8 storage-layout candidate would require only N16640 (520 tiles), yielding 104 workers with five columns each and one pass, with an ideal pair read floor around 0.354 ms. R13/C32 pads N much more because its output storage grid requires N tiles divisible by 32. C8 output shards also hold more live output tiles per core, so real L1 coexistence must be checked. This remains an untried hand-off hypothesis; V4 stays at the explicitly requested C32.
