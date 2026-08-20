# Outlier characterisation -- Wormhole, PACK_ISOLATE

Rule: more than 2% AND more than 30 cycles.
Scope: PACK_ISOLATE -- **1,457** flagged of **99,414** points.

## 1. Shape -- how many distinct values do the five runs take?

| distinct values across the 5 runs | points | share |
|---|--:|--:|
| 2 | 1,108 | 76.0% |
| 3 | 230 | 15.8% |
| 4 | 47 | 3.2% |
| 5 | 72 | 4.9% |

## 2. Size of the deviation

| statistic | odd run vs the other four (%) |
|---|--:|
| min | -19.82% |
| 5% | -15.61% |
| 25% | -7.63% |
| 50% | +2.34% |
| 75% | +7.85% |
| 95% | +18.33% |
| max | +24.27% |
| mean | +0.56% |
| std | 10.72 |

## 3. Is the deviation per-run or per-point?

Under independence each run would be the odd one about 291 times (20%).

| run | times it was the odd one | share |
|---|--:|--:|
| run_1 | 355 | 24.4% |
| run_2 | 333 | 22.9% |
| run_3 | 296 | 20.3% |
| run_4 | 219 | 15.0% |
| run_5 | 254 | 17.4% |

Largest share on any single run: **24.4%**. A run-level state predicts this near 100%; independent per-point events predict near 20%.

## 4. Does any sweep parameter separate flagged from unflagged?

Restricted to `perf_matmul` -- 1,246 of 1,457 flagged points -- so a test's own sweep does not masquerade as a parameter effect. 29,376 points in that test.

| parameter | flag rate by value | spread in rate |
|---|---|--:|
| `formats.output` | Float32=0%, Bfp8_b=0%, Float16=3%, Float16_b=13% | 13% |
| `formats.register_A` | Bfp8_b=2%, Float16_b=4%, Float16=5%, Tf32=10% | 8% |
| `formats.register_B` | Bfp8_b=2%, Float16_b=4%, Float16=5%, Tf32=10% | 8% |
| `formats.sfpu_math` | Bfp8_b=2%, Float16_b=4%, Float16=5%, Tf32=10% | 8% |
| `dest_acc` | DestAccumulation.No=2%, DestAccumulation.Yes=9% | 6% |
| `tile_cnt` | 40=1%, 10=1%, 256=1%, 15=2%, 20=2%, 5=2%, 14=2%, 7=2%, 21=2%, 48=2%, 28=2%, 192=2%, 18=2%, 56=3%, 224=3%, 24=3%, 160=4%, 6=4%, 2=4%, 1=4%, 64=4%, 3=4%, 12=5%, 8=5%, 32=5%, 16=6%, 4=6%, 96=6%, 9=7%, 128=7% | 6% |

**This is association, not cause.** Sweep parameters are correlated with each other, so a high spread here identifies where to look, and nothing more.
