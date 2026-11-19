| Layer | B | S | Prefill single → TP4 ms | Decode single → TP4 ms | Prefill speedup / efficiency | Decode speedup / efficiency |
|---|---:|---:|---:|---:|---:|---:|
| GDN | 1 | 512 | 5.866 → 2.054 | 1.158 → 0.574 | 2.86× / 71.4% | 2.02× / 50.4% |
| GDN | 32 | 33 | 23.818 → 11.102 | 1.671 → 0.733 | 2.15× / 53.6% | 2.28× / 57.0% |
| GDN | 32 | 513 | 185.305 → 59.051 | 1.671 → 0.733 | 3.14× / 78.5% | 2.28× / 57.0% |
| Full attention | 1 | 512 | 3.311 → 1.689 | 1.022 → 0.490 | 1.96× / 49.0% | 2.09× / 52.2% |
| Full attention | 32 | 33 | 12.909 → 10.621 | 1.174 → 0.701 | 1.22× / 30.4% | 1.67× / 41.9% |
| Full attention | 32 | 513 | 113.010 → 44.846 | 1.283 → 0.748 | 2.52× / 63.0% | 1.72× / 42.9% |

Speedup includes TP4 sharding, compact local kernels and PCC-validated LoFi attention; the frozen single-chip GDN baseline uses HiFi4 attention.
