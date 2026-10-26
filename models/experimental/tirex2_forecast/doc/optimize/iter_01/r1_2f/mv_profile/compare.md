# multivariate per-op device ms, reduced-layer profile (jobs 748 base / 749 R1-2f)

total base 2098.1  r12f 1227.7

| op | in0 W | in1 W | base ms (n) | r12f ms (n) | delta |
|---|---|---|---|---|---|
| MatmulDeviceOperation | 1[1] | 1[1] | 1280.4 (190) | 237.3 (158) | -1043.1 |
| MinimalMatmulDeviceOperation | 1[1] | 1[1] | 41.6 (4) | 253.7 (36) | +212.0 |
| UnaryDeviceOperation | 1[1] |  | 57.9 (292) | 18.8 (286) | -39.1 |
| PermuteDeviceOperation | 1024[1024] |  | 94.0 (6) | 94.4 (6) | +0.4 |
| BinaryNgDeviceOperation | 20480[20480] | 20480[20480] | 90.8 (900) | 90.5 (900) | -0.3 |
| PermuteDeviceOperation | 320[320] |  | 50.0 (6) | 49.8 (6) | -0.1 |
| BinaryNgDeviceOperation | 320[320] | 1[1] | 7.4 (4) | 7.6 (4) | +0.1 |
| UnaryDeviceOperation | 20480[20480] |  | 26.6 (744) | 26.6 (744) | -0.1 |
| NlpCreateHeadsDeviceOperation | 1024[1024] |  | 9.4 (2) | 9.4 (2) | -0.1 |
| SoftmaxDeviceOperation | 320[320] |  | 4.7 (4) | 4.8 (4) | +0.1 |
| BinaryNgDeviceOperation | 4096[4096] | 4096[4096] | 22.7 (186) | 22.6 (186) | -0.0 |
| TransposeDeviceOperation | 20480[20480] |  | 5.1 (6) | 5.0 (6) | -0.0 |
| ReduceDeviceOperation | 20480[20480] |  | 7.8 (8) | 7.8 (8) | -0.0 |
| BinaryNgDeviceOperation | 20480[20480] |  | 8.3 (4) | 8.3 (4) | -0.0 |
