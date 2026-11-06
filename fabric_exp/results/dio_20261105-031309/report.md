## sender main (runs: a/inline, a/noinline, b/inline, b/noinline)

| row | size | ntype | noinline a | noinline b | inline a | inline b | inline - noinline a | inline - noinline b | mean |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EW_Atomic | 4352 | atomic_inc | 130.2 | 130.2 | 134.0 | 133.9 | +3.8 (+2.9%) | +3.7 (+2.9%) | +3.78 |
| EW_Fused | 4352 | fused_atomic_inc | 149.2 | 149.2 | 151.8 | 151.8 | +2.6 (+1.7%) | +2.6 (+1.7%) | +2.60 |
| EW_Scatter | 4352 | unicast_scatter_write | 142.4 | 142.5 | 148.1 | 148.1 | +5.6 (+4.0%) | +5.6 (+3.9%) | +5.61 |
| EW_Write | 16 | unicast_write | 120.1 | 120.1 | 120.6 | 120.6 | +0.5 (+0.4%) | +0.5 (+0.4%) | +0.49 |
| EW_Write | 4352 | unicast_write | 128.2 | 128.2 | 129.0 | 129.0 | +0.8 (+0.6%) | +0.8 (+0.6%) | +0.79 |

## sender loop (runs: a/inline, a/noinline, b/inline, b/noinline)

| row | size | ntype | noinline a | noinline b | inline a | inline b | inline - noinline a | inline - noinline b | mean |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EW_Atomic | 4352 | atomic_inc | 129.9 | 129.9 | 133.6 | 133.6 | +3.7 (+2.8%) | +3.7 (+2.8%) | +3.69 |
| EW_Fused | 4352 | fused_atomic_inc | 149.0 | 149.0 | 151.4 | 151.4 | +2.4 (+1.6%) | +2.4 (+1.6%) | +2.41 |
| EW_Scatter | 4352 | unicast_scatter_write | 142.2 | 142.3 | 147.8 | 147.8 | +5.5 (+3.9%) | +5.5 (+3.9%) | +5.51 |
| EW_Write | 16 | unicast_write | 114.7 | 114.8 | 115.8 | 115.9 | +1.2 (+1.0%) | +1.1 (+1.0%) | +1.12 |
| EW_Write | 4352 | unicast_write | 126.9 | 126.9 | 127.3 | 127.3 | +0.4 (+0.3%) | +0.4 (+0.3%) | +0.37 |

