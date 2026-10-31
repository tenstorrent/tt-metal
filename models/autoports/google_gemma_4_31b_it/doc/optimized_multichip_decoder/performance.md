| Layer | Batch | Before prefill / decode device us | Final prefill / decode device us | Speedup pre / de |
|---|---:|---:|---:|---:|
| sliding_attention | 1 | 11422.54 / 484.77 | 11150.01 / 475.22 | 1.024 / 1.020 |
| sliding_attention | 32 | 314966.53 / 681.66 | 313771.09 / 674.86 | 1.004 / 1.010 |
| sliding_attention | 16 | 157937.56 / 569.14 | 156120.41 / 561.57 | 1.012 / 1.013 |
| sliding_attention | 8 | 79235.19 / 526.38 | 78891.10 / 519.63 | 1.004 / 1.013 |
| full_attention | 1 | 13748.47 / 534.77 | 13488.43 / 527.82 | 1.019 / 1.013 |
| full_attention | 32 | 378371.63 / 993.08 | 376513.98 / 985.63 | 1.005 / 1.008 |
| full_attention | 16 | 189592.45 / 747.95 | 189059.04 / 742.39 | 1.003 / 1.007 |
| full_attention | 8 | 95672.66 / 626.47 | 94807.44 / 620.93 | 1.009 / 1.009 |
