# Outlier characterisation -- Wormhole, L1_TO_L1, 10 runs

Rule: more than 2% AND more than 30 cycles.
Scope: L1_TO_L1 -- **83** flagged of **100,971** points.

## 1. Shape -- how many distinct values do the five runs take?

| distinct values across the 5 runs | points | share |
|---|--:|--:|
| 2 | 23 | 27.7% |
| 3 | 14 | 16.9% |
| 4 | 14 | 16.9% |
| 5 | 3 | 3.6% |
| 6 | 3 | 3.6% |
| 7 | 6 | 7.2% |
| 8 | 10 | 12.0% |
| 9 | 2 | 2.4% |
| 10 | 8 | 9.6% |

## 2. Size of the deviation

| statistic | odd run vs the other four (%) |
|---|--:|
| min | -6.05% |
| 5% | -4.61% |
| 25% | -3.05% |
| 50% | +2.35% |
| 75% | +3.31% |
| 95% | +4.41% |
| max | +8.83% |
| mean | +0.91% |
| std | 3.59 |

## 3. Is the deviation per-run or per-point?

Under independence each run would be the odd one about 8 times (10%).

| run | times it was the odd one | share |
|---|--:|--:|
| run_1 | 10 | 12.0% |
| run_2 | 10 | 12.0% |
| run_3 | 6 | 7.2% |
| run_4 | 14 | 16.9% |
| run_5 | 8 | 9.6% |
| run_6 | 9 | 10.8% |
| run_7 | 6 | 7.2% |
| run_8 | 6 | 7.2% |
| run_9 | 8 | 9.6% |
| run_10 | 6 | 7.2% |

Largest share on any single run: **16.9%**. A run-level state predicts this near 100%; independent per-point events predict near 10%.

## 4. Does any sweep parameter separate flagged from unflagged?

Restricted to `perf_math_matmul` -- 60 of 83 flagged points -- so a test's own sweep does not masquerade as a parameter effect. 59,760 points in that test.

| parameter | flag rate by value | spread in rate |
|---|---|--:|
| `dst_index` | 5.0=0%, 10.0=0%, 13.0=0%, 12.0=0%, 11.0=0%, 8.0=0%, 9.0=0%, 4.0=0%, 7.0=0%, 0.0=0%, 2.0=0%, 3.0=0%, 1.0=0%, 6.0=0%, 14.0=0%, 15.0=1% | 1% |
| `tile_cnt` | 7=0%, 8=0%, 14=0%, 13=0%, 12=0%, 11=0%, 10=0%, 9=0%, 15=0%, 16=0%, 3=0%, 6=0%, 5=0%, 4=0%, 1=0%, 2=0% | 0% |
| `c_dimm` | 7.0=0%, 8.0=0%, 14.0=0%, 13.0=0%, 12.0=0%, 11.0=0%, 10.0=0%, 9.0=0%, 15.0=0%, 16.0=0%, 3.0=0%, 6.0=0%, 5.0=0%, 4.0=0%, 1.0=0%, 2.0=0% | 0% |
| `math_fidelity` | MathFidelity.HiFi2=0%, MathFidelity.HiFi3=0%, MathFidelity.HiFi4=0%, MathFidelity.LoFi=0% | 0% |
| `dest_sync` | DestSync.Full=0%, DestSync.Half=0% | 0% |
| `formats.input_A` | Float32=0%, Bfp8_b=0%, Float16=0%, Float16_b=0% | 0% |

**This is association, not cause.** Sweep parameters are correlated with each other, so a high spread here identifies where to look, and nothing more.

## 5. What threshold would cover everything?

| cycle floor | points above it | smallest % that flags nothing |
|--:|--:|--:|
| 0 | 77,578 | 8.83% |
| 10 | 40,110 | 8.83% |
| 20 | 27,026 | 8.83% |
| 30 | 19,427 | 8.83% |
| 50 | 11,401 | 8.83% |
| 100 | 6,216 | 8.83% |
| 200 | 3,273 | 8.83% |
| 500 | 968 | 8.83% |
