# Compare main (base) vs new_288afd2 (test)

generated 2026-11-04T15:59:18+00:00 from /home/aagarwal/tt-metal/fabric_exp/testplan/results

## Steps

| group | step | base | test | note |
| --- | --- | --- | --- | --- |
| G-L0-a | G0-fix-1d | PASS | PASS |  |
| G-L0-a | G0-fix-2d | PASS | PASS |  |
| G-L0-a | G0-fix-mux | PASS | PASS |  |
| G-L0-a | G0-health-cv | ENV | ENV | known-fail on base: not scored |
| G-L0-a | G0-health-stab | PASS | PASS |  |
| G-L0-a | G0-night | TIMEOUT,PASS | PASS |  |
| G-L0-a | G0-torus-n | PASS | PASS |  |
| G-L0-b | G0-d2d | PASS | PASS |  |
| G-L0-b | G0-eth | PASS | FAIL | **step regressed**; cases only in base 35, only in test 0 |
| G-L0-b | G0-mm-2x8-pt | KNOWN_FAIL | KNOWN_FAIL | known-fail on base: not scored |
| G-L0-b | G0-mm-2x8-stab | PASS | PASS |  |
| G-L0-b | G0-mm-4x4 | PASS | FAIL | **step regressed** |
| G-L0-b | G0-mm-hetero | PASS | PASS |  |
| G-L0-b | G0-mm-nsew | PASS | PASS |  |
| G-L0-b | G0-mm-trim-nsew | PASS | PASS |  |
| G-L0-b | G0-mm-trim-z | PASS | PASS |  |
| G-L0-b | G0-mm-z | PASS | PASS |  |
| G-L0-b | G0-sock | PASS | PASS |  |
| G-L0-b | G0-torus-s | PASS | PASS |  |
| G-L0-mpiab | G0-mm-4x4-ci | PASS | - |  |
| G-L0-mpiab | G0-mm-4x4-lo | FAIL | - |  |
| G-L1 | G1-dp-combine | PASS | PASS |  |
| G-L1 | G1-dp-hybrid | PASS | PASS |  |
| G-L1 | G1-mh | NOT_RUN | NOT_RUN |  |
| G-L1 | G1-night | PASS | PASS |  |
| G-L1P | G1-hbw | PASS,PASS | - |  |
| G-L2-a | G2-mm3 | KNOWN_FAIL | - |  |
| G-L2-a | G2-oss | PASS | - |  |
| G-L2-a | G2-qwen | PASS | TIMEOUT | **step regressed**; cases only in base 1, only in test 0 |
| G-L2-b | G2-dsb1-attn | KNOWN_FAIL | KNOWN_FAIL | known-fail on base: not scored |
| G-L2-b | G2-dsb1-f2d-ag | PASS | PASS |  |
| G-L2-b | G2-dsb1-f2d-ar | PASS | PASS |  |
| G-L2-b | G2-dsb1-f2d-bc | PASS | PASS |  |
| G-L2-b | G2-dsb1-f2d-brms | PASS | PASS |  |
| G-L2-b | G2-dsb1-f2d-dzf | PASS | PASS |  |
| G-L2-b | G2-dsb1-f2d-lmh | KNOWN_FAIL | KNOWN_FAIL | known-fail on base: not scored |
| G-L2-b | G2-dsb1-f2d-mhp | PASS | PASS |  |
| G-L2-b | G2-dsb1-f2d-model | PASS | PASS |  |
| G-L2-b | G2-dsb1-f2d-moe | KNOWN_FAIL | KNOWN_FAIL | known-fail on base: not scored |
| G-L2-b | G2-dsb1-f2d-post | PASS | PASS |  |
| G-L2-b | G2-dsb1-f2d-pre | KNOWN_FAIL | KNOWN_FAIL | known-fail on base: not scored |
| G-L2-b | G2-dsb1-f2d-pss | PASS | PASS |  |
| G-L2-b | G2-dsb1-f2d-pw | PASS | PASS |  |
| G-L2-b | G2-dsb1-f2d-r1 | PASS | PASS |  |
| G-L2-b | G2-dsb1-f2d-samp | PASS | PASS |  |
| G-L2-b | G2-dsb1-f2d-sdpa | KNOWN_FAIL | KNOWN_FAIL | known-fail on base: not scored |
| G-L2-b | G2-dsb1-hyb | PASS | PASS |  |
| G-L2-b | G2-dsb1-more | PASS | PASS |  |
| G-L2-b | G2-dsb1-sweep | NOT_RUN | NOT_RUN |  |
| G-P | G0-bench | PASS,PASS | - |  |
| LB-L0-a | B0-1d | PASS | PASS |  |
| LB-L0-a | B0-1e | PASS | PASS |  |
| LB-L0-a | B0-2d | PASS | PASS |  |
| LB-L0-a | B0-mux | PASS | PASS |  |
| LB-L0-a | B0-trim | PASS | PASS |  |
| LB-L0-a | B0-udm | PASS | PASS |  |
| LB-L0-a | B0-udm-ttnn | PASS | PASS |  |
| LB-L0-b | B0-eth | PASS,PASS | PASS |  |
| LB-L0-b | B0-health | PASS | PASS |  |
| LB-L0-b | B0-infra | PASS | PASS |  |
| LB-L0-b | B0-night | PASS | PASS |  |
| LB-L0-b | B0-stab | PASS | PASS |  |
| LB-L1 | B1-apc | PASS | PASS |  |
| LB-L1 | B1-dp-d2d | PASS | PASS |  |
| LB-L1 | B1-dp-host | PASS | PASS |  |
| LB-L1 | B1-dp-ops-a | PASS | PASS |  |
| LB-L1 | B1-dp-ops-b | PASS | PASS |  |
| LB-L1 | B1-dp-ops-c | PASS | PASS |  |
| LB-L1 | B1-night-a | PASS | PASS |  |
| LB-L1 | B1-night-b | PASS | PASS |  |
| LB-L1 | B1-sock | PASS | PASS |  |
| LB-L1 | B1-watch | PASS | PASS |  |
| LB-L1P | B1-ccl-ag | PASS,PASS,PASS | - |  |
| LB-L1P | B1-ccl-ar | PASS,PASS,PASS | - |  |
| LB-L1P | B1-ccl-bc | PASS,PASS,PASS | - |  |
| LB-L1P | B1-ccl-sdpa | KNOWN_FAIL,KNOWN_FAIL,KNOWN_FAIL | - |  |
| LB-L2 | B2-dp-k3 | PASS | - |  |
| LB-L2 | B2-dp-perf-ds | NOT_RUN | - |  |
| LB-L2 | B2-dp-perf-kda | PERF_WINDOW | - |  |
| LB-L2 | B2-dp-perf-m4 | PERF_WINDOW | - |  |
| LB-P | B0-bench | PASS,PASS | - |  |
| LB-X-a09fails | B0-eth-sd | FAIL | - |  |
| LB-X-a09fails | B1-night-1x4 | FAIL | - |  |
| LB-X-a09fails | B1-watch-1x4 | FAIL | - |  |
| LB-X-a09fails | hostcheck | PASS | - |  |
