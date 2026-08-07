# moe_fused_swiglu performance — 8 rows × 11 columns

Device-kernel duration on Blackhole p150 at 1.35 GHz. Each value is the median of three Tracy-profiled dispatches. The op API spells this grid `core_grid=(11, 8)` because its tuple is `(columns, rows)`; every CSV row reports 88 cores.

Inputs: `x_rm` = BF16 row-major; `x_tile` = BFP8 tiled. Weights are BFP4 tiled. `w_ndshard` is the op-aware DRAM ND-sharded placement and `w_interleaved` is DRAM interleaved.

The upstream columns come from the supplied `perf_table_upstream.txt`. Delta is `(this kernel / upstream - 1)`: negative is faster, positive is slower.

## x_rm · kimi_k26 (K=7168, N=2048)

| M | this interleaved (us) | upstream interleaved (us) | delta | this ND-shard (us) | upstream ND-shard (us) | delta |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.450 | 2.884 | +19.6% | 3.556 | 2.975 | +19.5% |
| 64 | 89.059 | 121.366 | -26.6% | 79.616 | 94.900 | -16.1% |
| 128 | 97.333 | 122.925 | -20.8% | 84.832 | 97.532 | -13.0% |
| 256 | 122.758 | 127.024 | -3.4% | 112.866 | 110.242 | +2.4% |
| 512 | 200.458 | 162.893 | +23.1% | 190.726 | 157.987 | +20.7% |
| 1024 | 350.379 | 298.172 | +17.5% | 342.844 | 297.718 | +15.2% |
| 2048 | 654.405 | 585.216 | +11.8% | 642.654 | 585.567 | +9.7% |
| 4096 | 1250.894 | 1163.201 | +7.5% | 1242.881 | 1158.800 | +7.3% |
| 5120 | 1554.195 | 1454.667 | +6.8% | 1542.840 | 1448.190 | +6.5% |

## x_rm · glm_51 (K=6144, N=2048)

| M | this interleaved (us) | upstream interleaved (us) | delta | this ND-shard (us) | upstream ND-shard (us) | delta |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.353 | 2.803 | +19.6% | 3.542 | 2.973 | +19.1% |
| 64 | 77.844 | 106.450 | -26.9% | 69.666 | 85.815 | -18.8% |
| 128 | 86.011 | 108.279 | -20.6% | 76.973 | 85.939 | -10.4% |
| 256 | 112.384 | 111.831 | +0.5% | 104.833 | 97.253 | +7.8% |
| 512 | 186.275 | 144.427 | +29.0% | 175.508 | 139.347 | +26.0% |
| 1024 | 325.177 | 261.148 | +24.5% | 313.991 | 260.572 | +20.5% |
| 2048 | 602.647 | 512.956 | +17.5% | 592.540 | 515.509 | +14.9% |
| 4096 | 1157.525 | 1017.936 | +13.7% | 1148.025 | 1014.213 | +13.2% |
| 5120 | 1435.502 | 1267.576 | +13.2% | 1425.448 | 1266.821 | +12.5% |

## x_tile · kimi_k26 (K=7168, N=2048)

| M | this interleaved (us) | upstream interleaved (us) | delta | this ND-shard (us) | upstream ND-shard (us) | delta |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.424 | 2.971 | +15.2% | 3.761 | 2.899 | +29.7% |
| 64 | 86.933 | 121.399 | -28.4% | 78.336 | 92.016 | -14.9% |
| 128 | 93.718 | 123.069 | -23.8% | 84.053 | 95.773 | -12.2% |
| 256 | 118.980 | 126.712 | -6.1% | 108.730 | 102.804 | +5.8% |
| 512 | 197.600 | 145.491 | +35.8% | 185.744 | 141.113 | +31.6% |
| 1024 | 347.793 | 269.984 | +28.8% | 335.295 | 268.513 | +24.9% |
| 2048 | 648.661 | 531.124 | +22.1% | 635.624 | 530.594 | +19.8% |
| 4096 | 1246.521 | 1054.732 | +18.2% | 1234.264 | 1053.151 | +17.2% |
| 5120 | 1546.092 | 1315.377 | +17.5% | 1533.001 | 1306.567 | +17.3% |

## x_tile · glm_51 (K=6144, N=2048)

| M | this interleaved (us) | upstream interleaved (us) | delta | this ND-shard (us) | upstream ND-shard (us) | delta |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.337 | 2.886 | +15.6% | 3.637 | 2.802 | +29.8% |
| 64 | 77.699 | 106.844 | -27.3% | 70.625 | 83.205 | -15.1% |
| 128 | 84.741 | 108.238 | -21.7% | 74.234 | 85.717 | -13.4% |
| 256 | 109.924 | 111.109 | -1.1% | 98.414 | 90.936 | +8.2% |
| 512 | 180.723 | 129.564 | +39.5% | 171.309 | 124.310 | +37.8% |
| 1024 | 321.119 | 237.256 | +35.3% | 309.333 | 235.378 | +31.4% |
| 2048 | 599.718 | 467.307 | +28.3% | 584.960 | 463.863 | +26.1% |
| 4096 | 1156.350 | 923.307 | +25.2% | 1136.921 | 917.145 | +24.0% |
| 5120 | 1435.991 | 1150.249 | +24.8% | 1412.650 | 1147.513 | +23.1% |

## Reproduction inputs

- CSV: `/localdev/mstaletovic/2026_07_31/1406_mstaletovic_moe_matmul_codegen/clones/moe_fused_swiglu_run1/tt-metal/generated/perf_table_8x11_raw/ops_k7168_bf16_rm.csv`
- Manifest: `/localdev/mstaletovic/2026_07_31/1406_mstaletovic_moe_matmul_codegen/clones/moe_fused_swiglu_run1/tt-metal/generated/perf_table_8x11_raw/manifest_k7168_bf16_rm.json`
- CSV: `/localdev/mstaletovic/2026_07_31/1406_mstaletovic_moe_matmul_codegen/clones/moe_fused_swiglu_run1/tt-metal/generated/perf_table_8x11_raw/ops_k7168_bfp8_tile.csv`
- Manifest: `/localdev/mstaletovic/2026_07_31/1406_mstaletovic_moe_matmul_codegen/clones/moe_fused_swiglu_run1/tt-metal/generated/perf_table_8x11_raw/manifest_k7168_bfp8_tile.json`
- CSV: `/localdev/mstaletovic/2026_07_31/1406_mstaletovic_moe_matmul_codegen/clones/moe_fused_swiglu_run1/tt-metal/generated/perf_table_8x11_raw/ops_k6144_bf16_rm.csv`
- Manifest: `/localdev/mstaletovic/2026_07_31/1406_mstaletovic_moe_matmul_codegen/clones/moe_fused_swiglu_run1/tt-metal/generated/perf_table_8x11_raw/manifest_k6144_bf16_rm.json`
- CSV: `/localdev/mstaletovic/2026_07_31/1406_mstaletovic_moe_matmul_codegen/clones/moe_fused_swiglu_run1/tt-metal/generated/perf_table_8x11_raw/ops_k6144_bfp8_tile.csv`
- Manifest: `/localdev/mstaletovic/2026_07_31/1406_mstaletovic_moe_matmul_codegen/clones/moe_fused_swiglu_run1/tt-metal/generated/perf_table_8x11_raw/manifest_k6144_bfp8_tile.json`
