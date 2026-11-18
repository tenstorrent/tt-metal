# PDL stock vs genify_matmul, one inference (test_tt_model.py -k panoptic_deeplab_110_cores)

Stock CSV: `generated/profiler/reports/2026_11_17_20_32_48/ops_perf_results_2026_11_17_20_32_48.csv`
Genify CSV: `generated/profiler/reports/2026_11_18_14_59_14/ops_perf_results_2026_11_18_14_59_14.csv`

## Totals (device kernel time, µs)

| | stock | genify | genify - stock |
|---|---:|---:|---:|
| 55 matmuls (kernel only) | 989.8 | 855.9 | -133.8 |
| glue charged to the matmuls (pre + post) | 181.0 | 73.7 | -107.2 |
| matmuls + their glue | 1170.7 | 929.6 | -241.1 |
| all ops: matmul | 989.8 | 855.9 | -133.8 |
| all ops: glue (data movement) | 857.2 | 766.5 | -90.7 |
| all ops: other | 3912.1 | 3891.5 | -20.6 |
| **whole inference (sum of all ops)** | **5759.0** | **5514.0** | **-245.1** |
| device ops | 313 | 292 | -21 |

Stride-2 subsample (genify pre-glue of res3.0 / res4.0 shortcut): res3.0.shortcut (stride 2): Slice 17.6 + Select 8.7 = 26.3 µs; res4.0.shortcut (stride 2): Slice 9.2 + Select 4.9 = 14.1 µs

## Per matmul (µs)

| matmul | shape | A stock | A genify | stock us | genify us | genify/stock | stock pre-glue | stock post-glue | genify pre-glue | genify post-glue |
|---|---|---|---|---:|---:|---:|---|---|---|---|
| res2.0.shortcut | 32768x128x256 | L1-HS bfp8 | L1-HS bfp8 | 9.8 | 10.0 | 1.02 | Reshard 0.6 | - | - | - |
| res2.0.conv1 | 32768x128x64 | L1-HS bfp8 | L1-HS bfp8 | 4.0 | 3.8 | 0.96 | Reshard 0.6 + Move 0.8 | - | - | - |
| res2.0.conv3 | 32768x64x256 | L1-HS bfp8 | L1-HS bfp8 | 8.7 | 8.2 | 0.94 | - | - | - | - |
| res2.1.conv1 | 32768x256x64 | L1-HS bfp8 | L1-HS bfp8 | 5.0 | 5.2 | 1.03 | - | - | - | - |
| res2.1.conv3 | 32768x64x256 | L1-HS bfp8 | L1-HS bfp8 | 8.7 | 8.1 | 0.93 | - | - | - | - |
| res2.2.conv1 | 32768x256x64 | L1-HS bfp8 | L1-HS bfp8 | 5.0 | 5.1 | 1.01 | - | - | - | - |
| res2.2.conv3 | 32768x64x256 | L1-HS bfp8 | L1-HS bfp8 | 8.7 | 8.1 | 0.93 | - | - | - | - |
| res3.0.shortcut (stride 2) | 8192x256x512 | L1-HS bf16 | L1 bfp8 | 13.1 | 16.6 | 1.27 | I2S 24.8 + Halo 8.1 + Move 3.3 | - | Slice 17.6 + Select 8.7 | I2S 5.6 |
| res3.0.conv1 | 32768x256x128 | L1-HS bfp8 | DRAM bfp8 | 9.4 | 27.7 | 2.95 | I2S 24.7 | - | - | I2S 5.5 + Move 1.0 |
| res3.0.conv3 | 8192x128x512 | L1-HS bfp8 | L1-HS bfp8 | 8.2 | 8.4 | 1.03 | - | - | - | - |
| res3.1.conv1 | 8192x512x128 | L1-HS bfp8 | L1-HS bfp8 | 7.0 | 6.5 | 0.94 | - | - | - | - |
| res3.1.conv3 | 8192x128x512 | L1-HS bfp8 | L1-HS bfp8 | 8.2 | 8.4 | 1.03 | - | - | - | - |
| res3.2.conv1 | 8192x512x128 | L1-HS bfp8 | L1-HS bfp8 | 6.9 | 6.6 | 0.96 | - | - | - | - |
| res3.2.conv3 | 8192x128x512 | L1-HS bfp8 | L1-HS bfp8 | 8.2 | 8.3 | 1.02 | - | - | - | - |
| res3.3.conv1 | 8192x512x128 | L1-HS bfp8 | L1-HS bfp8 | 6.9 | 6.5 | 0.94 | - | - | - | - |
| res3.3.conv3 | 8192x128x512 | L1-HS bfp8 | L1-HS bfp8 | 8.2 | 8.3 | 1.01 | - | - | - | - |
| res4.0.shortcut (stride 2) | 2048x512x1024 | L1-BS bf16 | L1 bfp8 | 16.2 | 14.9 | 0.92 | I2S 12.1 + Halo 5.3 + Move 1.7 | Reshard 1.9 | Slice 9.2 + Select 4.9 | I2S 3.3 |
| res4.0.conv1 | 8192x512x256 | L1-BS bfp8 | DRAM bfp8 | 16.6 | 20.8 | 1.26 | I2S 11.9 | - | - | I2S 3.4 + Move 0.8 |
| res4.0.conv3 | 2048x256x1024 | L1-BS bfp8 | L1-BS bfp8 | 13.6 | 12.9 | 0.95 | - | - | - | - |
| res4.1.conv1 | 2048x1024x256 | L1-BS bfp8 | L1-BS bfp8 | 9.5 | 9.8 | 1.04 | - | - | - | - |
| res4.1.conv3 | 2048x256x1024 | L1-BS bfp8 | L1-BS bfp8 | 13.7 | 12.9 | 0.95 | - | - | - | - |
| res4.2.conv1 | 2048x1024x256 | L1-BS bfp8 | L1-BS bfp8 | 9.4 | 9.8 | 1.04 | - | - | - | - |
| res4.2.conv3 | 2048x256x1024 | L1-BS bfp8 | L1-BS bfp8 | 13.6 | 12.8 | 0.94 | - | - | - | - |
| res4.3.conv1 | 2048x1024x256 | L1-BS bfp8 | L1-BS bfp8 | 9.4 | 9.6 | 1.03 | - | - | - | - |
| res4.3.conv3 | 2048x256x1024 | L1-BS bfp8 | L1-BS bfp8 | 13.6 | 12.8 | 0.94 | - | - | - | - |
| res4.4.conv1 | 2048x1024x256 | L1-BS bfp8 | L1-BS bfp8 | 9.4 | 9.6 | 1.03 | - | - | - | - |
| res4.4.conv3 | 2048x256x1024 | L1-BS bfp8 | L1-BS bfp8 | 13.6 | 12.8 | 0.94 | - | - | - | - |
| res4.5.conv1 | 2048x1024x256 | L1-BS bfp8 | L1-BS bfp8 | 9.3 | 9.7 | 1.03 | - | - | - | - |
| res4.5.conv3 | 2048x256x1024 | L1-BS bfp8 | L1-BS bfp8 | 13.5 | 12.8 | 0.95 | - | - | - | - |
| res5.0.shortcut | 2048x1024x2048 | L1-BS bfp8 | DRAM bfp8 | 72.3 | 33.4 | 0.46 | I2S 7.2 | - | - | I2S 6.2 |
| res5.0.conv1 | 2048x1024x512 | L1-BS bfp8 | DRAM bfp8 | 10.6 | 17.5 | 1.65 | I2S 7.4 | - | - | I2S 2.0 + Move 0.6 |
| res5.0.conv3 | 2048x512x2048 | L1-BS bfp8 | L1-BS bfp8 | 24.2 | 22.7 | 0.94 | - | - | - | - |
| res5.1.conv1 | 2048x2048x512 | L1-BS bfp8 | L1-BS bfp8 | 16.1 | 16.7 | 1.04 | - | - | - | - |
| res5.1.conv3 | 2048x512x2048 | L1-BS bfp8 | L1-BS bfp8 | 24.1 | 22.6 | 0.94 | - | - | - | - |
| res5.2.conv1 | 2048x2048x512 | L1-BS bfp8 | L1-BS bfp8 | 16.0 | 16.7 | 1.04 | - | - | - | - |
| res5.2.conv3 | 2048x512x2048 | L1-BS bfp8 | L1-BS bfp8 | 24.2 | 22.6 | 0.94 | - | - | - | - |
| sem aspp.convs.0 | 2048x2048x256 | L1-BS bfp8 | L1 bfp8 | 13.8 | 28.7 | 2.08 | I2S 6.6 | - | - | - |
| sem aspp.convs.4 (pool, M=1) | 1x2048x256 | L1-BS bf16 | L1 bf16 | 6.9 | 6.6 | 0.96 | TilizeVP 2.5 + I2S 0.7 | - | TilizeVP 2.5 | - |
| sem aspp.project | 2048x1280x256 | L1-BS bfp8 | L1 bfp8 | 16.8 | 18.8 | 1.12 | I2S 4.9 + Move 1.6 | - | - | - |
| sem decoder.res3.project_conv | 8192x512x64 | L1-HS bfp8 | L1 bfp8 | 4.6 | 9.8 | 2.13 | I2S 5.6 + Move 1.2 | - | - | - |
| sem decoder.res2.project_conv | 32768x256x32 | L1-HS bfp8 | L1 bfp8 | 3.6 | 12.4 | 3.46 | I2S 10.8 + Move 1.5 | - | - | - |
| sem predictor (19/32) | 32768x256x32 | L1-HS bfp8 | L1-HS bfp8 | 3.6 | 3.3 | 0.92 | - | - | - | - |
| sem upsample W | 32x128x256x1024 | L1 bfp8 | L1 bfp8 | 15.2 | 11.2 | 0.74 | - | - | - | - |
| sem upsample H | 32x512x128x1024 | DRAM bfp8 | DRAM bfp8 | 122.4 | 67.1 | 0.55 | - | - | - | - |
| ins aspp.convs.0 | 2048x2048x256 | L1-BS bfp8 | L1 bfp8 | 13.8 | 28.6 | 2.08 | I2S 6.4 | - | - | - |
| ins aspp.convs.4 (pool, M=1) | 1x2048x256 | L1-BS bf16 | L1 bf16 | 6.9 | 6.6 | 0.95 | TilizeVP 2.5 + I2S 0.7 | - | TilizeVP 2.5 | - |
| ins aspp.project | 2048x1280x256 | L1-BS bfp8 | L1 bfp8 | 16.9 | 18.9 | 1.12 | I2S 4.9 + Move 1.6 | - | - | - |
| ins decoder.res3.project_conv | 8192x512x64 | L1-HS bfp8 | L1 bfp8 | 4.7 | 9.8 | 2.10 | I2S 5.7 + Move 1.2 | - | - | - |
| ins decoder.res2.project_conv | 32768x256x32 | L1-HS bfp8 | L1 bfp8 | 3.6 | 12.3 | 3.43 | I2S 10.9 + Move 1.4 | - | - | - |
| ins center_predictor (1/32) | 32768x32x32 | L1-HS bfp8 | L1-HS bfp8 | 3.5 | 2.6 | 0.75 | - | - | - | - |
| ins offset_predictor (2/32) | 32768x32x32 | L1-HS bfp8 | L1-HS bfp8 | 3.5 | 2.5 | 0.73 | - | - | - | - |
| ins center upsample W | 32x128x256x1024 | L1 bfp8 | L1 bfp8 | 15.3 | 11.3 | 0.74 | - | - | - | - |
| ins center upsample H | 32x512x128x1024 | DRAM bfp8 | DRAM bfp8 | 122.4 | 67.3 | 0.55 | - | - | - | - |
| ins offset upsample W | 32x128x256x1024 | L1 bfp8 | L1 bfp8 | 15.3 | 11.5 | 0.75 | - | - | - | - |
| ins offset upsample H | 32x512x128x1024 | DRAM bfp8 | DRAM bfp8 | 122.3 | 67.3 | 0.55 | - | - | - | - |

## Per op code (count, µs)

| op code | stock n | stock µs | genify n | genify µs | delta µs |
|---|---:|---:|---:|---:|---:|
| Conv2dDeviceOperation | 43 | 2614.7 | 41 | 2584.6 | -30.2 |
| MatmulDeviceOperation | 53 | 960.5 | 0 | 0.0 | -960.5 |
| HaloDeviceOperation | 44 | 627.3 | 42 | 611.7 | -15.6 |
| UpsampleOperation | 6 | 205.4 | 6 | 205.2 | -0.3 |
| InterleavedToShardedDeviceOperation | 28 | 205.3 | 18 | 87.2 | -118.1 |
| Pool2D | 1 | 148.5 | 1 | 148.5 | -0.0 |
| MoveDeviceOperation | 39 | 147.5 | 35 | 141.2 | -6.2 |
| BinaryNgDeviceOperation | 21 | 134.7 | 21 | 132.1 | -2.6 |
| CopyDeviceOperation | 6 | 131.6 | 8 | 136.8 | +5.2 |
| ShardedToInterleavedDeviceOperation | 21 | 104.8 | 13 | 90.7 | -14.1 |
| TilizeDeviceOperation | 7 | 93.3 | 7 | 94.9 | +1.6 |
| ConcatDeviceOperation | 6 | 82.5 | 6 | 83.1 | +0.6 |
| TypecastDeviceOperation | 11 | 79.1 | 11 | 79.0 | -0.1 |
| PermuteDeviceOperation | 3 | 78.9 | 3 | 77.3 | -1.6 |
| ReduceDeviceOperation | 2 | 41.4 | 2 | 41.3 | -0.2 |
| SliceDeviceOperation | 4 | 27.3 | 6 | 54.1 | +26.8 |
| TilizeWithValPaddingDeviceOperation | 4 | 25.8 | 4 | 25.8 | +0.0 |
| UntilizeCodegenDeviceOperation | 3 | 21.5 | 3 | 21.6 | +0.1 |
| UntilizeDeviceOperation | 2 | 13.3 | 2 | 13.1 | -0.1 |
| UnaryDeviceOperation | 2 | 7.9 | 2 | 7.8 | -0.1 |
| UntilizeWithUnpaddingDeviceOperation | 4 | 4.6 | 4 | 8.4 | +3.9 |
| ReshardDeviceOperation | 3 | 3.1 | 0 | 0.0 | -3.1 |
| GenericOpDeviceOperation | 0 | 0.0 | 55 | 855.9 | +855.9 |
| GenericOpDeviceOperation (stride-2 select) | 0 | 0.0 | 2 | 13.6 | +13.6 |
