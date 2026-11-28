# g6_sharded_to_interleaved — precision matrix results

Last run: 2026-11-27, Blackhole p150b (11 x 10 worker grid), through the public entry point.
Test: `test_g6_sharded_to_interleaved_precision_baseline.py::test_g6_sharded_to_interleaved_precision_matrix`.

The op is an exact copy (`helpers.TOLERANCES`: "exact" for both dtypes). The gate is 0 bit mismatches,
0 ULP, max_abs 0, relative RMS 0 and PCC 1. A bfloat8_b reference is the host readback of the input as
stored. There is no compute-config axis (`math_fidelity`, `fp32_dest_acc_en`): the op does no
arithmetic, and the contract fixes a two-argument signature.

Axes: 4 geometries (the two production shapes, each on its production and its other shard spec, BB-22)
x every SUPPORTED dtype x {uniform `rand`, normal `randn`}, each tile scaled by 2^(t % 16 - 8).

| Geometry | Scheme -> output | dtype | dist | PCC | max abs | rel RMS | max ULP | bit mismatches |
|---|---|---|---|---|---|---|---|---|
| p2 (1,5120), shard (32,160), (0,0)-(7,3), row-major | width -> DRAM | bf16 | uniform | 1.0 | 0 | 0 | 0 | 0 |
| p2 | width -> DRAM | bf16 | normal | 1.0 | 0 | 0 | 0 | 0 |
| p2 | width -> DRAM | bf8b | uniform | 1.0 | 0 | 0 | 0 | 0 |
| p2 | width -> DRAM | bf8b | normal | 1.0 | 0 | 0 | 0 | 0 |
| p2 other spec, shard (32,96), (0,0)-(8,5), column-major | width -> DRAM | bf16 | uniform | 1.0 | 0 | 0 | 0 | 0 |
| p2 other spec | width -> DRAM | bf16 | normal | 1.0 | 0 | 0 | 0 | 0 |
| p2 other spec | width -> DRAM | bf8b | uniform | 1.0 | 0 | 0 | 0 | 0 |
| p2 other spec | width -> DRAM | bf8b | normal | 1.0 | 0 | 0 | 0 | 0 |
| p1 (1,128,256,32), shard (320,32), (0,0)-(10,8)+(0,9)-(3,9), row-major | height -> L1 | bf16 | uniform | 1.0 | 0 | 0 | 0 | 0 |
| p1 | height -> L1 | bf16 | normal | 1.0 | 0 | 0 | 0 | 0 |
| p1 | height -> L1 | bf8b | uniform | 1.0 | 0 | 0 | 0 | 0 |
| p1 | height -> L1 | bf8b | normal | 1.0 | 0 | 0 | 0 | 0 |
| p1 other spec, shard (352,32), (0,0)-(8,9)+(9,0)-(9,3), column-major | height -> L1 | bf16 | uniform | 1.0 | 0 | 0 | 0 | 0 |
| p1 other spec | height -> L1 | bf16 | normal | 1.0 | 0 | 0 | 0 | 0 |
| p1 other spec | height -> L1 | bf8b | uniform | 1.0 | 0 | 0 | 0 | 0 |
| p1 other spec | height -> L1 | bf8b | normal | 1.0 | 0 | 0 | 0 | 0 |

got/true ratio (p5, p50, p95) = (1.0, 1.0, 1.0) on every row.

Skipped combinations: none.
