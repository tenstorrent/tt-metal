# Native migration: all_to_all_matmul

- **Outcome:** ported
- **Branch:** port/all_to_all_matmul-nm-20261126T165418Z from 8a2349d09bdd
- **Flow:** 418f33841cdf
- **Cost:** $35.06 over 20 agent turn(s), 119.0 min
- **Rounds:** 7 of 8; open findings 0
- **Parity:** unavailable (unsupported by this checkout)
- **Unchecked:** descriptor parity: the checkout has no per-process parity switch; core arrangement and dispatch bytes of tests/ttnn/unit_tests/operations/all_to_all_matmul/pre_migration/test_all_to_all_matmul_host_perf.py::test_host_perf[deepseek_bf8b]: the generic capture failed; core arrangement and dispatch bytes of tests/ttnn/unit_tests/operations/all_to_all_matmul/pre_migration/test_all_to_all_matmul_host_perf.py::test_host_perf[non_aligned_40_rows]: the generic capture failed; core arrangement and dispatch bytes of tests/ttnn/unit_tests/operations/all_to_all_matmul/pre_migration/test_all_to_all_matmul_host_perf.py::test_host_perf[kimi_model]: the generic capture failed; core arrangement and dispatch bytes of tests/ttnn/unit_tests/operations/all_to_all_matmul/pre_migration/test_all_to_all_matmul_host_perf.py::test_host_perf[smallest]: the generic capture failed
- **Edited outside the op:** nothing
- **Preallocated output:** the run asked for the exact public surface
- **Generic route deleted:** ttnn/ttnn/operations/all_to_all_matmul/all_to_all_matmul_program_descriptor.py

| case | route | cold us | hot us | steady us |
|---|---|---|---|---|
| deepseek_bf8b | native | 10273.4 | 372.3 | 362.9 |
| deepseek_bf8b | generic | 12779.4 | 6848.7 | 6918.3 |
| non_aligned_40_rows | native | 5855.9 | 361.4 | 358.0 |
| non_aligned_40_rows | generic | 11073.5 | 6100.6 | 6179.7 |
| kimi_model | native | 6026.7 | 422.2 | 424.3 |
| kimi_model | generic | 11469.8 | 6972.6 | 6726.7 |
| smallest | native | 3250.3 | 198.3 | 174.8 |
| smallest | generic | 5391.4 | 2029.6 | 1999.8 |

| spec | route | dispatch bytes per call (rtargs / cb_config / semaphore / binary) |
|---|---|---|
| test_host_perf[deepseek_bf8b] | native | 42996 / 1344 / 128 / 227080 |
| test_host_perf[non_aligned_40_rows] | native | 36304 / 2000 / 128 / 240600 |
| test_host_perf[smallest] | native | 7368 / 1408 / 128 / 204976 |

| golden spec | cases timed | device, port/source (median) | slowest on the device | host, port/source (median) | traced wall, port/source (median) |
|---|---|---|---|---|---|
| test_all_to_all_matmul.py | 10 of 10 | 23.441 | 23.441 (test_all_to_all_matmul[single_tile_per_head]) | 0.032 | 1.038 |
| test_all_to_all_matmul_non_aligned.py | 6 of 6 | 21.192 | 21.192 (test_non_aligned_row_provenance_bit_exact[b2_12_rows]) | 0.039 | 1.012 |
| test_all_to_all_matmul_precision_baseline.py | 20 of 20 | 19.501 | 19.501 (test_precision_baseline[small-bf8bw_bf16acc_HiFi2_L1]) | 0.034 | 0.988 |
| test_golden.py | 148 of 148 | 13.446 | 19.006 (test_op[1x24x640x128-w1x96x128x512-alignment=tile_aligned-cl) | 0.029 | 1.021 |
| test_regression.py | 8 of 8 | 12.821 | 12.821 (test_row_provenance) | 0.022 | 0.983 |

| stage | started (UTC) | min |
|---|---|---|
| translation (beside the test writer) | 17:03:31 | 8.6 |
| golden baselines | 16:59:29 | 19.2 |
| test writer | 16:59:29 | 24.2 |
| test writer | 17:26:42 | 6.4 |
| golden baselines | 17:26:42 | 12.3 |
| translation (beside the test writer) | 17:29:44 | 9.6 |
| source check | 17:39:00 | 4.7 |
| round 2: build, gates, acceptance, perf | 19:37:05 | 20.4 |
| round 2: review | 19:57:29 | 1.4 |
| round 3: build, gates, acceptance, perf | 20:04:08 | 13.5 |
| round 3: review | 20:17:36 | 9.3 |
| round 4: build, gates, acceptance, perf | 20:27:02 | 12.2 |
| round 4: review | 20:39:11 | 8.2 |
| round 6: build, gates, acceptance, perf | 20:55:11 | 11.7 |
| round 6: review | 21:06:51 | 8.7 |
| round 7: build, gates, acceptance, perf | 21:15:37 | 13.7 |
| round 7: review | 21:29:17 | 8.6 |
| final: every perf case and the whole golden suite | 21:38:29 | 46.9 |

Commits:
- `c5edd6a16c53` [codegen-port] all_to_all_matmul pre-migration tests (nm-20261126T165418Z)
- `2281c77966aa` [codegen-port] all_to_all_matmul translation (nm-20261126T165418Z)
- `f83b0c5e31cd` [codegen-port] all_to_all_matmul round 3 (nm-20261126T165418Z)
- `76e44656194c` [codegen-port] all_to_all_matmul round 6 (nm-20261126T165418Z)
