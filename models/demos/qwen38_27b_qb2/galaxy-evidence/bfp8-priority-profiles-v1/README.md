# BFP8 B32/32K stage comparison

Warm eager real-weight two-layer diagnostic with synthetic caches; kernel-duration sums, not traced full-model TPOT or accuracy.

Four physical ranks; table shows median rank kernel-duration sums. All ranks and raw CSVs are retained.

| Stage | Native (us) | Shared Q/K (us) | Reduction |
|---|---:|---:|---:|
| L0__delta_recurrence | 1161.0 | 375.6 | 67.6% |
| L0_decode_forward | 1999.4 | 1214.5 | 39.3% |
| L0_packed_decode_conv | 149.7 | 149.7 | 0.0% |
| L3_decode_forward | 2012.9 | 2018.3 | -0.3% |
| L3_paged_decode | 1505.1 | 1504.9 | 0.0% |

Input hashes and precision match; only recurrence selection differs. No full-model tok/s/user uplift is claimed. Firmware/RISC intervals include waits; nested stages overlap and must not be summed together.

The native and candidate runs each passed the hardware test, clean closure, repeated-output checks and timing completeness on all four ranks. These checks do not replace full GPQA.
