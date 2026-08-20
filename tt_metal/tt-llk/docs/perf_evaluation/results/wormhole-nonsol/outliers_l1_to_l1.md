# Outlier characterisation -- Wormhole, L1_TO_L1

Rule: more than 2% AND more than 30 cycles.
Scope: all run types -- **53** flagged of **100,971** points.

## 1. Shape -- how many distinct values do the five runs take?

| distinct values across the 5 runs | points | share |
|---|--:|--:|
| 2 | 11 | 20.8% |
| 3 | 9 | 17.0% |
| 4 | 17 | 32.1% |
| 5 | 16 | 30.2% |

## 2. Size of the deviation

| statistic | odd run vs the other four (%) |
|---|--:|
| min | -4.62% |
| 5% | -3.58% |
| 25% | -2.65% |
| 50% | +2.01% |
| 75% | +3.12% |
| 95% | +3.42% |
| max | +3.63% |
| mean | +0.19% |
| std | 2.88 |

## 3. Is the deviation per-run or per-point?

Under independence each run would be the odd one about 11 times (20%).

| run | times it was the odd one | share |
|---|--:|--:|
| run_1 | 8 | 15.1% |
| run_2 | 18 | 34.0% |
| run_3 | 14 | 26.4% |
| run_4 | 4 | 7.5% |
| run_5 | 9 | 17.0% |

Largest share on any single run: **34.0%**. A run-level state predicts this near 100%; independent per-point events predict near 20%.

## 4. Does any sweep parameter separate flagged from unflagged?

Restricted to `perf_math_matmul` -- 45 of 53 flagged points -- so a test's own sweep does not masquerade as a parameter effect. 59,760 points in that test.

| parameter | flag rate by value | spread in rate |
|---|---|--:|
| `dst_index` | 1.0=0%, 5.0=0%, 13.0=0%, 12.0=0%, 11.0=0%, 10.0=0%, 9.0=0%, 8.0=0%, 0.0=0%, 4.0=0%, 7.0=0%, 2.0=0%, 14.0=0%, 6.0=0%, 3.0=0%, 15.0=1% | 1% |
| `tile_cnt` | 3=0%, 8=0%, 7=0%, 6=0%, 12=0%, 11=0%, 10=0%, 9=0%, 13=0%, 14=0%, 15=0%, 16=0%, 4=0%, 2=0%, 5=0%, 1=0% | 0% |
| `c_dimm` | 3.0=0%, 8.0=0%, 7.0=0%, 6.0=0%, 12.0=0%, 11.0=0%, 10.0=0%, 9.0=0%, 13.0=0%, 14.0=0%, 15.0=0%, 16.0=0%, 4.0=0%, 2.0=0%, 5.0=0%, 1.0=0% | 0% |
| `dest_sync` | DestSync.Full=0%, DestSync.Half=0% | 0% |
| `formats.input_A` | Float32=0%, Bfp8_b=0%, Float16=0%, Float16_b=0% | 0% |
| `formats.input_B` | Float32=0%, Bfp8_b=0%, Float16=0%, Float16_b=0% | 0% |

**This is association, not cause.** Sweep parameters are correlated with each other, so a high spread here identifies where to look, and nothing more.
