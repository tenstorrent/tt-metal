# Round 1 proposals (G2-r1; to be approved with G1, D3)

Source: the round-0 reduced-layer profile of paper / accuracy (`perf_report.txt.gz`, `ops_perf_results.csv.gz`,
`profile_zones.json`): layer 0 profiled once and counted 5 times, the last layer once. Op IDs below are the
report's. Bandwidth floors use the report's own DRAM peak (127 GB/s = 24.8 %, so about 512 GB/s). Each saving is
for one forward at paper and assumes about half of the gap to the floor is recovered, unless stated.

FlowState has no attention: the F1/F9 table and the SDPA table of v1.20, v1.21 and v1.23 do not apply
(CONTRACT:403). Round-0 device time: 374.9 ms (bench), 375.9 ms (profile estimate).

| # | item | profile rows (per mid layer, x5) | ms now | floor | saving (ms per forward) |
|---|---|---|---|---|---|
| 1 | matmul program configs per role | 45 B (10.58), 47 C (13.56), 48 gate (4.30), 54 w1 (4.70), 56 w2 (4.71), 58 w3 (7.41); last layer 64 B (10.70) | 45.26 / layer | 2.6 + 2.6 + 1.05 + 1.57 x 3 = 11.0 / layer | (45.26 - 11.0) / 2 x 5 + (10.70 - 2.6) / 2 = 85.7 + 4.1 = **about -90** |
| 2 | fused SwiGLU (w1, silu, w2, mul in one op) | 54 (4.70), 55 silu (3.71), 56 (4.71), 57 mul (3.58) | 16.70 / layer | one read of h, one write: about 8.7 | (3.71 + 3.58) + half of one in0 read (about 0.8) = 8.1 x 5 = **about -40** |
| 3 | time-segmented route-K scan over all cores | 46 / 65 scan (5.05 / 5.01), 32 of 110 cores | 30.3 (x6) | 2.15 GB / 512 GB/s = 4.2 / layer | paper: (5.05 - 4.2) x 6 = **about -5** (already at 425 GB/s); long_context (16 cores, 20.2 ms / layer, floor 8.4): **about -70** |
| 4 | elementwise fusion | 49 sigmoid (2.08), 50 gate mul (1.80), 51 D·u (1.85), 52 add (1.80), 59 z + y (1.81) | 9.34 / layer | sigmoid into the mul; D·u + y in one op; z + y in the w3 epilogue | 2.08 + 1.80 + 1.81 = 5.7 x 5 = **about -28** |
| 5 | padding for B < 32 (single, single_long, long_context): time segments folded into rows | guard shapes only, builds on #3 | | | guard shapes only |

Total of #1-#4 at paper: 375 - 90 - 40 - 5 - 28 = **about 212 ms** (range 200-230 ms).

**Experiment and gate per item** (each one candidate, committed before its job, session-stamped):
1. One reduced-layer bench job sweeps per-role configs (in0_block_w, out_subblock, per_core_M/N, fp32 in0 in L1
   where it fits). Gate: bench paper device ms down at least 3 % (above the run-to-run noise), and check_gates 0
   failures against `iter_00/fixed_record.json` in both precisions; bit-identical outputs are expected (no
   precision change), so any drift is a finding.
2. Micro-benchmark `fuse_swiglu` (with the b1 / b2 biases) on 262144 x 512 x 1024; if it is not available, the
   silu as the w1 program config's fused activation (about -18 ms). Gate: the MLP module tests (test_modules.py)
   >= 0.999 both precisions, then the bench and check_gates as in #1.
3. Micro-benchmark the segmented scan against op 46 at paper and long_context, with the B0 bounds of test_s5.py
   (fused path, both L tiers, the bf16-Lbar control) first. Gate: test_s5.py passes, then bench and check_gates.
4. One candidate per fusion, bench screened. Gate: as #1.
5. After #3. Gate: the guard shapes' records in check_gates.

**Deferred:** bf16 x into C, and bf16 B / C weights (precision changes: per-row table, X gate if they change the
policy); the last layer's scan writing only rows >= r0.

**Performance class:** the re-raise proposes retiring it for now (v1.12). At paper it is 375.8 ms against 374.9 ms
(job 1088): it changes only the gate / MLP weights and fidelity, and those matmuls are bound neither by fidelity
nor by weight bandwidth at the default configs. Revisit after #1, if the dense matmuls become compute-bound.
