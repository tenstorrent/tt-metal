# Pass-to-pass noise: main, router main

| group/step | kind | key | per pass | spread |
| --- | --- | --- | --- | ---: |
| LB-L1P/B1-ccl-ar | metric | ccl=all_reduce cfg=num_links=2,  payload=2048 | 1.415 / 1.35 | 4.70% |
| LB-L1P/B1-ccl-ar | metric | ccl=all_reduce cfg=num_links=2,  payload=15232 | 1.309 / 1.275 | 2.63% |
| LB-L1P/B1-ccl-ar | metric | ccl=all_reduce cfg=num_links=1,  payload=4352 | 1.314 / 1.286 | 2.15% |
| LB-L1P/B1-ccl-ar | metric | ccl=all_reduce cfg=num_links=1,  payload=15232 | 1.306 / 1.28 | 2.01% |
| LB-L1P/B1-ccl-ar | metric | ccl=all_reduce cfg=num_links=2,  payload=4352 | 1.299 / 1.283 | 1.24% |
| LB-L1P/B1-ccl-bc | metric | ccl=broadcast cfg=num_links=1,  payload=8192 | 4.263 / 4.305 | 0.98% |
| LB-L1P/B1-ccl-ar | metric | ccl=all_reduce cfg=num_links=1,  payload=8192 | 1.274 / 1.284 | 0.78% |
| LB-L1P/B1-ccl-ar | metric | ccl=all_reduce cfg=num_links=1,  payload=4096 | 1.279 / 1.289 | 0.78% |
| LB-L1P/B1-ccl-ag | metric | ccl=all_gather cfg=num_links=1,  payload=2048 | 3.014 / 2.992 | 0.73% |
| LB-L1P/B1-ccl-bc | metric | ccl=broadcast cfg=num_links=1,  payload=2048 | 6.377 / 6.422 | 0.70% |
| LB-L1P/B1-ccl-ar | metric | ccl=all_reduce cfg=num_links=1,  payload=2048 | 1.481 / 1.491 | 0.67% |
| LB-L1P/B1-ccl-bc | metric | ccl=broadcast cfg=num_links=2,  payload=4096 | 7.965 / 8.014 | 0.61% |
| LB-L1P/B1-ccl-ag | metric | ccl=all_gather cfg=num_links=1,  payload=4352 | 2.647 / 2.637 | 0.38% |
| LB-L1P/B1-ccl-ag | metric | ccl=all_gather cfg=num_links=1,  payload=8192 | 2.639 / 2.631 | 0.30% |
| LB-L1P/B1-ccl-bc | metric | ccl=broadcast cfg=num_links=2,  payload=2048 | 9.229 / 9.208 | 0.23% |
| LB-L1P/B1-ccl-bc | metric | ccl=broadcast cfg=num_links=2,  payload=8192 | 7.098 / 7.082 | 0.23% |
| LB-L1P/B1-ccl-bc | metric | ccl=broadcast cfg=num_links=2,  payload=4352 | 8.021 / 8.005 | 0.20% |
| LB-L1P/B1-ccl-bc | metric | ccl=broadcast cfg=num_links=2,  payload=15232 | 4.171 / 4.163 | 0.19% |
| LB-L1P/B1-ccl-ag | metric | ccl=all_gather cfg=num_links=1,  payload=15232 | 2.628 / 2.632 | 0.15% |
| LB-L1P/B1-ccl-ar | metric | ccl=all_reduce cfg=num_links=2,  payload=8192 | 1.291 / 1.292 | 0.08% |
| LB-L1P/B1-ccl-bc | metric | ccl=broadcast cfg=num_links=1,  payload=4352 | 5.143 / 5.141 | 0.04% |
| LB-L1P/B1-ccl-bc | metric | ccl=broadcast cfg=num_links=1,  payload=4096 | 5.199 / 5.197 | 0.04% |
| LB-L1P/B1-ccl-ag | metric | ccl=all_gather cfg=num_links=1,  payload=4096 | 2.634 / 2.633 | 0.04% |
| LB-L1P/B1-ccl-bc | metric | ccl=broadcast cfg=num_links=1,  payload=15232 | 3.92 / 3.919 | 0.03% |
| LB-L1P/B1-ccl-ar | metric | ccl=all_reduce cfg=num_links=2,  payload=4096 | 1.266 / 1.266 | 0.00% |

25 rows with >= 2 passes: median spread 0.38%, max 4.70%
