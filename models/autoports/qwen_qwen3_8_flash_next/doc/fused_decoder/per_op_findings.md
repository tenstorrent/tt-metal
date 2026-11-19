# Per-operation profiler findings

The six final selected-path captures are complete. Each report directory below
contains the original `ops.csv`, filtered `{mode}_perf_report.csv`, human-readable
`{mode}_perf_report.txt`, official advice, provenance and latency summary. They
supersede the preliminary packed-early capture. Exact QKV, residual writer and
compact-count-2 routing are integrated in the final runtime; no integration is pending.

All captures use real weights, one process on chip 0, and no Watcher. Prefill
has 32 logical rows and is segmented to drain profiler buffers. Its kernel sum
is diagnostic: gaps and complete prefill latency are not measured by this capture.
Decode is uninterrupted traced replay of one logical row after the same 32-row
prefill. Reported matmul height 32 includes tile padding. The exact source paths
and machine-readable totals are in `wrap_profile_summary.json`.

| Layer / phase | Active experts | Ops | Kernel sum ms | Measured gaps ms | Kernel + gaps ms |
|---|---:|---:|---:|---:|---:|
| 0 / prefill | 104 | 984 | 78.003737 | unmeasured | unmeasured |
| 0 / decode | 10 | 206 | 4.966203 | 0.133837 | 5.100040 |
| 1 / prefill | 156 | 1164 | 92.397247 | unmeasured | unmeasured |
| 1 / decode | 10 | 398 | 6.290691 | 0.247841 | 6.538532 |
| 3 / prefill | 185 | 1747 | 183.022748 | unmeasured | unmeasured |
| 3 / decode | 10 | 330 | 9.404568 | 0.252566 | 9.657134 |

Report directories: `tracy/wrap_final_ops_layer{0,1,3}_{prefill,decode}/report/`.

## Measured bottlenecks

| Layer / phase | Sparse gate/up + down ms | Binary ms | Reduction ms | Generic kernels ms |
|---|---:|---:|---:|---:|
| 0 / prefill | 21.727768 | 26.760087 | 12.902121 | 10.751961 |
| 0 / decode | 2.038813 | 0.964101 | 0.535178 | 0.528850 |
| 1 / prefill | 32.258770 | 27.133432 | 12.960035 | 11.318328 |
| 1 / decode | 2.034486 | 1.235715 | 0.603582 | 0.720882 |
| 3 / prefill | 38.149214 | 42.206838 | 19.123358 | 61.975417 |
| 3 / decode | 2.037547 | 1.640832 | 0.778130 | 2.415970 |

Operation-family totals aggregate multiple mathematical roles. In particular,
`GenericOpDeviceOperation` includes custom device kernels; its total is not a
single projection or a host fallback. QSA remains dominated by its precise
indexer/projection arithmetic and associated binary/reduction work. Sparse
experts take 41.1%, 32.3% and 21.7% of decode kernel time for layers 0, 1 and 3.
The final layer-0 capture has 206 operations and 4.966203 ms kernel time versus
279 operations and 8.075147 ms in the historical pre-exact-QKV packed-early
capture. This is a topology diagnostic; the clean same-harness ranking is the
latency authority for the final speedup.

## Dtype and movement audit

The actual sparse weight operands are BF16 for layers 0/3 and BFP8 (`BFLOAT8_B`)
for layer 1; activations/output remain BF16 and sparse matmuls report HiFi4.
Packed gate/up uses 40 cores, down uses 8. All three decode reports use exactly
10 indexed experts. Prefill unions are 104, 156 and 185 experts, read from the
actual warmup sparsity operands outside measured execution. No whole-resident
512-expert traffic assumption is used.

| Layer / phase | Tilize/untilize ops | Their kernel sum ms | Reshard ops |
|---|---:|---:|---:|
| 0 / prefill | 17 | 0.160943 | 0 |
| 0 / decode | 13 | 0.061559 | 0 |
| 1 / prefill | 47 | 0.728142 | 0 |
| 1 / decode | 45 | 0.247678 | 0 |
| 3 / prefill | 404 | 2.034171 | 0 |
| 3 / decode | 28 | 0.100364 | 0 |

The remaining conversions serve actual consumer representation changes: precise
projection reduction outputs, PLE selected packed lookup features, integer
QSA cache/position interfaces and selected RoPE tables, and MoE sparse masks
and weighted-reduction scores. Read-side four-stream grouping changes axis
order. `runtime_boundary_audit.md` maps those boundaries. The final writer
eliminates write-side permutation intermediates; compact GDN recurrence eliminates
the prior repeated layout round trips. Remaining reshape/transpose operation
families are listed in the summary JSON. Zero reshard rows is observed, not a
claim that all representation-changing operations vanished. The forward runtime
audits passed in each capture and exclude Torch/host conversions in measured calls.

## Official advice: Not tried (stop rule)

The owner wall cap ends tuning. The following official advice is retained for
the subsequent optimized-decoder stage, with no claimed speedup or rejection:

- Sparse gate/up and down: try L1 input, larger legal `in0_block_w`, larger output
  subblocks, and HiFi2 under the same real-weight correctness controls. Current
  sparse rows remain `SLOW`; packed-width/core variants were measured earlier,
  but these additional geometry/fidelity combinations have not been run.
- Other `SLOW` dense projections: L1 operands, larger subblocks and HiFi2.
- Packed GDN z/a/b, QSA QKV and the wide PLE projection: official advice proposes
  a DRAM-sharded matmul program. These rows achieve roughly 76–79% of the
  report bandwidth ceiling individually; that is not whole-decoder attainment.

Native norm, convolution and TopK substitutions have separate measured/source
rejections in `pattern_assessment.json`. Their numeric boundaries cannot be
discarded merely because per-op advice suggests a cheaper primitive. No new
optimized-decoder implementation or device experiment is included in this audit.

Collection note: inline generic kernel sources are materialized unchanged as
C++ files to avoid CSV newline corruption. Raw CSV/logs are retained; this final
audit creates no per-artifact archives.
