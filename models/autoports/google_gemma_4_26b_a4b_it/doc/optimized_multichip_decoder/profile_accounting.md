# Signpost-scoped profile accounting

The final default was captured on the 1x4 P300C mesh with a single warmed
trace replay for device accounting. Host acceptance latency is measured
separately over 30 replays after five warmups. `tt-perf-report` is delimited
between the matching `PERF_DECODE_*` and `*_END` signposts; it no longer mixes
prefill, setup, and repeated replay rows.

| Layer | Scoped rows | Merged device-op sum (us) | Reported op-gap sum (us) | Warmed host decode (us) | Approx. local weight payload/token | Ideal 512 GB/s weight floor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sliding | 88 | 1043.127 | 350.650 | 1070.437 | 25.1 MB | 49 us |
| full | 93 | 1088.442 | 335.140 | 1113.634 | 37.4 MB | 73 us |

Device rows are merged from four concurrently executing devices, so the
device-op and op-gap columns are diagnostics, not quantities to add into a
single wall-clock total. Their proximity to warmed host latency shows that
dispatch is hidden by trace replay; the large apparent gap is dominated by
one merged `LayerNormDeviceOperation` boundary (261.880 us sliding, 244.914 us
full) and overlaps work on other ranks. The byte floor counts TP-local QKV/O,
dense, router, and top-eight active-expert weights at their selected packed
precision; packed-format metadata and cache/activation traffic make it a lower
bound. Three BF16 hidden-width ring collectives carry 25,344 payload bytes per
rank/layer (`3 * 1.5 * 2816 * 2`) before protocol overhead.

## Cumulative decode contract

| Family | Sliding evidence | Full evidence | Final configuration / boundary |
| --- | --- | --- | --- |
| SDPA decode | 31.637 us, row 6247 | 38.791 us, row 6414 | height-sharded L1 input; 8x4 grid, q-chunk 32, k-chunk 64, exact exp |
| QKV matmul | 51.808 us, row 6494 | 64.510 us, row 6581 | DRAM-interleaved input/packed BFP8 weight, block K=2, 1x1 subblock; DRAM block-11 retry failed sliding PCC |
| Attention O | 14.536 us, row 6428 | 17.165 us, row 6512 | width-sharded L1 input, DRAM-sharded BFP8 weight, block K=4; HiFi2 sliding/LoFi full |
| Dense packed gate/up | 10.725 us, row 6437 | 10.834 us, row 6615 | width-sharded L1, DRAM-sharded BFP4 decode copy, block K=11 |
| Dense down | 8.772 us, row 6356 | 8.755 us, row 6436 | width-sharded L1, DRAM-sharded BFP8, block K=17 |
| Expert gate/up | 75.531 + 75.226 us, rows 6560/6563 | 75.477 + 75.256 us, rows 6465/6561 | sparse active=8/128, L1 interleaved, BFP8 LoFi, block K=44, 1x2 subblock |
| Expert down | 18.755 us, row 6481 | 18.760 us, row 6381 | sparse active=8/128, L1 interleaved, BFP8 LoFi, block K=6, 1x2 subblock |
| Decode norms | L1 head norms 5.329/6.174 us; replicated residual norms 34.739-44.505 us | L1 head norms 8.040/9.556 us; replicated residual norms 34.748-45.424 us | BF16; head tensors L1-sharded, inter-layer residual DRAM-interleaved |
| Residual adds | eight `BinaryNg` rows, 24.33 us total | eight rows, comparable 2.17-5.54 us each | BF16 DRAM-interleaved output; no boundary collective/layout conversion |
| Layout conversions | I2S/S2I rows total 17.71 us | scoped I2S/S2I rows visible individually | conversions remain inside attention/dense families; final boundary is replicated BF16 DRAM |
| Collectives | three RS+AG ring implementations | three persistent `AllReduceAsync`, 17.116/17.272/17.436 us | two links; no decoder-to-decoder gather/reshard/all-reduce |

Exact rows and geometry fields are in each `decode_analyzed.csv`; signposted
prefill tables live beside them as `prefill_analyzed.csv`.
