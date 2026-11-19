# Profile accounting

Accepted decoder SHA256 `082c82faa6b9500f654dcc3f0e26f1c4fab656e987d3fd079bc1c79ec19a3c44`. These are instrumented runs; [performance.md](performance.md) reports uninstrumented warmed latency. Each device row uses the slowest rank. Category times are microseconds.

## Decode

| Layer / B / S | Kernel µs | Compute µs | Explicit CCL µs | Tensor movement µs | Data movement µs | Other µs | Host ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| linear_attention / 1 / 512 | 553.4 | 309.6 | 80.9 | 22.8 | 23.9 | 116.2 | 0.614 |
| linear_attention / 32 / 33 | 715.9 | 356.9 | 76.4 | 54.5 | 26.2 | 201.8 | 0.774 |
| linear_attention / 32 / 513 | 717.0 | 356.7 | 78.0 | 54.3 | 26.1 | 201.8 | 0.771 |
| full_attention / 1 / 512 | 463.0 | 303.0 | 80.2 | 24.3 | 43.0 | 12.5 | 0.538 |
| full_attention / 32 / 33 | 675.0 | 274.3 | 95.8 | 89.8 | 44.3 | 170.7 | 0.732 |
| full_attention / 32 / 513 | 722.7 | 326.3 | 91.6 | 90.0 | 44.7 | 170.2 | 0.783 |

## Prefill

| Layer / B / S | Kernel µs | Compute µs | Explicit CCL µs | Tensor movement µs | Data movement µs | Other µs | Host ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| linear_attention / 1 / 512 | 1723.8 | 561.3 | 456.5 | 135.5 | 1.8 | 568.7 | 2.680 |
| linear_attention / 32 / 33 | 10192.9 | 914.4 | 441.2 | 5349.8 | 10.7 | 3476.9 | 11.512 |
| linear_attention / 32 / 513 | 55976.9 | 17598.8 | 12297.3 | 13326.2 | 120.7 | 12634.0 | 59.134 |
| full_attention / 1 / 512 | 1444.0 | 656.2 | 427.4 | 51.7 | 0.0 | 308.8 | 1.923 |
| full_attention / 32 / 33 | 7717.2 | 551.2 | 444.8 | 4469.8 | 6.5 | 2244.8 | 12.496 |
| full_attention / 32 / 513 | 42562.8 | 13856.6 | 12493.2 | 8967.0 | 114.7 | 7131.4 | 45.606 |

Prefill native MMRS includes matmul, communication and reduction in one operation. The tool places its time in the Other category above, so explicit CCL alone is not total communication time.

| Layer / B / S | Fused matmul + CCL µs |
|---|---:|
| linear_attention / 1 / 512 | 0.0 |
| linear_attention / 32 / 33 | 1398.0 |
| linear_attention / 32 / 513 | 0.0 |
| full_attention / 1 / 512 | 0.0 |
| full_attention / 32 / 33 | 1368.3 |
| full_attention / 32 / 513 | 0.0 |

Decode projections achieve 293–443 GB/s versus 512 GB/s nominal per chip. The runtime rows retain BFP8 weights, LoFi attention, HiFi2 GDN MLP and LoFi full-attention MLP. The narrow correction has physical width3072. Initial B32 kernel sums were2745.5µs/GDN and1533.2µs/full attention; the accepted compact decode totals are above.

The weight+KV lower bound is about0.237ms/GDN and0.194–0.212ms/full attention. It omits intermediate movement and recurrent-state traffic. Each accounting.json reconciles the profiled device time with its same-run host latency; profiler timing is not substituted for the uninstrumented comparison. Prefill device rows select the first of three warmed signposted calls, whereas host prefill is their mean; instrumented dispatch gaps and that statistic mismatch can produce a larger device interval.

Prefill advice was investigated through the ordinary K/grid sweep, precision-locked native SwiGLU configurations and row-projection decomposition. The accepted path uses 110-core/K8 ordinary matmuls,88-core small SwiGLU and110-core large/tail SwiGLU. Intermediate rows use128-row MMRS pieces; small and large rows use separate matmul/RS. This avoids the proven native NoC ownership defect, with a measured dispatch/weight-traffic cost relative to the watcher-rejected unchunked path. Native MMRS exposes no public program_config, block/grid or read-NoC override; its generic missing-program_config advice is not an absent configuration. MinimalMatmul uses its explicit MinimalMatmulConfig. The source diagnosis and measured alternatives are in AUTODEBUG_PREFILL_GEOMETRY.md and safe/native/integrated candidate CSV/JSON artifacts.

The remaining B32/S33 prefill bottleneck is physical reshape movement: ReshapeViewDeviceOperation totals 4.043ms/GDN and 3.686ms/full attention. Logical33-row batches require retiling when flattened; these are measured data moves, not zero-cost views. At B32/S513, matmuls, slices and about12.3–12.5ms of explicit CCL dominate. The safe separate-row large-M path beats128-row fused pieces in the matched crossover controls. These costs remain visible in the accepted numbers; a persistent flat-prefill representation is a future optimization opportunity.

tt-perf-report uses the P150 nominal compute model for Blackhole; this machine is P300c. Tool compute utilization above100% is not a physical-efficiency claim. Other includes local native math and structural operations. Raw CSV, exact tool commands/version, hashes and human-readable tables are under perf/<prefix>_profile_l{0,3}_b{1,32}/. The optional disabled-WASM-preview warning does not prevent capture/CSV/report generation.
