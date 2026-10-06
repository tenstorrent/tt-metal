# Isolated DRAM reader results

All28 legal reader/K/role cases pass the0.995 operand/production PCC gate. Both QKV K11 reader1 cases fail the same exact static-L1 bound:1,641,728B exceeds1,572,864B. The six capture/profile/summary commands return0. Each kind has84 complete windows and1,680 native matmul samples. Runtime169c0d97 remains the recorded source. [Source-hash incident](dram_readers_v5/provenance_incident.md) distinguishes the formatting-only split; measured files were not rewritten.

| Layer | Role | K block | Reader1 µs | Reader2 µs | Reader3 µs | Isolated fastest |
| --- | --- | ---: | ---: | ---: | ---: | ---: |

| 0 | qkv | 1 | 235.9875 | 134.7765 | 90.6390 | 3 |

| 0 | qkv | 11 | L1 invalid | 77.7470 | 93.5680 | 2 |

| 0 | output | 16 | 85.0730 | 52.3170 | 47.4430 | 3 |

| 0 | shared_gate_up | 11 | 32.9655 | 38.2690 | 45.6635 | 1 |

| 0 | shared_down | 11 | 18.2145 | 21.4330 | 23.4375 | 1 |

| 5 | qkv | 1 | 209.6375 | 124.6200 | 90.3315 | 3 |

| 5 | qkv | 11 | L1 invalid | 75.4375 | 92.1670 | 2 |

| 5 | output | 16 | 58.1370 | 70.2175 | 84.7960 | 1 |

| 5 | shared_gate_up | 11 | 26.8005 | 21.9305 | 25.0795 | 2 |

| 5 | shared_down | 11 | 15.0060 | 13.9585 | 13.6150 | 3 |


[JSON](dram_reader_results.json) retains exact precision, per-reader physical/logical GB/s and peak percentage, round-median spread, PCC, padding/header bytes, bank/reader rows, NoC page/burst estimates and BRISC/NCRISC/TRISC medians. All shared results exactly match production output; all within-family reader-pair PCCs are1.0. The QKV/output records retain their nonzero production differences and do not claim exact equality.

Three readers are slower than two for QKV K11 despite smaller rows. Sliding shared gate/down and full output also slow with extra readers. Reader/compute-kernel durations rise together in these cases; those intervals include waits and do not isolate a NoC or pure-math cause. The payload/multicast/request evidence is recorded instead of assigning an unmeasured root cause. Full shared down reader3 is only0.3435µs faster than reader2 in the isolated device median; this is explicitly a whole-layer control, not an automatic policy change.

[The22-case whole-layer plan](reader_layer_v5/plan.json) includes current baselines, legal QKV reader1/K1 and reader2/3K11, output readers1/2/3 at the real BF16 boundary, and each shared role independently. Full down-reader3 retains gate-reader2. Its wrapper preserves new minimal-prefill defaults and common QKV N9216/output N3072. [All22 whole-layer controls](reader_layer_results.md) have now completed on b585a21f: all pass, no material layer gain is observed, and defaults are retained. The published wrapper/driver are frozen after pinned Black23.10.1, pre-commit, syntax and CPU control-matrix checks.
