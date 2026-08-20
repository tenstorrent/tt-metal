# Outlier characterisation -- Blackhole, PACK_ISOLATE

Rule: more than 2% AND more than 30 cycles.
Scope: PACK_ISOLATE -- **2** flagged of **106,748** points.

## 1. Shape -- how many distinct values do the five runs take?

| distinct values across the 5 runs | points | share |
|---|--:|--:|
| 2 | 2 | 100.0% |

## 2. Size of the deviation

| statistic | odd run vs the other four (%) |
|---|--:|
| min | -19.75% |
| 5% | -19.47% |
| 25% | -18.34% |
| 50% | -16.94% |
| 75% | -15.53% |
| 95% | -14.40% |
| max | -14.12% |
| mean | -16.94% |
| std | 3.99 |

## 3. Is the deviation per-run or per-point?

Under independence each run would be the odd one about 0 times (20%).

| run | times it was the odd one | share |
|---|--:|--:|
| run_1 | 2 | 100.0% |
| run_2 | 0 | 0.0% |
| run_3 | 0 | 0.0% |
| run_4 | 0 | 0.0% |
| run_5 | 0 | 0.0% |

Largest share on any single run: **100.0%**. A run-level state predicts this near 100%; independent per-point events predict near 20%.

## 4. Does any sweep parameter separate flagged from unflagged?

Restricted to `perf_pack_dest_bank` -- 2 of 2 flagged points -- so a test's own sweep does not masquerade as a parameter effect. 54 points in that test.

| parameter | flag rate by value | spread in rate |
|---|---|--:|
| `loop_factor` | 128=0%, 512=0%, 8=11% | 11% |
| `input_num_blocks` | 1.0=0%, 2.0=11% | 11% |
| `num_blocks` | 1.0=0%, 2.0=11% | 11% |
| `output_num_blocks` | 1.0=0%, 2.0=11% | 11% |
| `l1_acc` | L1Accumulation.No=0%, L1Accumulation.Yes=7% | 7% |
| `tile_cnt` | 4=0%, 8=6% | 6% |

**This is association, not cause.** Sweep parameters are correlated with each other, so a high spread here identifies where to look, and nothing more.
