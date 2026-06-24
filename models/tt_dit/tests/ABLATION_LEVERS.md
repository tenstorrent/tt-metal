# Ablation lever sweep — `wan_fused_distributed_rmsnorm`

Goal: on the **worst-speedup-over-baseline** shapes, find which component is on the critical
path. Method: traced end-to-end fused µs (`test_bench`, fused-only) with one component stubbed
per run (`WAN_ABLATION=N`, timing-only — not bit-correct). **Δµs = `fused(0) − fused(N)`** =
wall-clock removed by skipping component N = its contribution to the critical path. WH 4×8
galaxy, ring, 4 links. Same 9 worst-speedup shapes as the original sweep.

`WAN_ABLATION`: 1=rope read, 2=input read, 3=output write, 4=fabric (mcast+sem),
5=gather/scatter, 6=weight/bias, 7=skip-compute, 8=all-IO (pure compute).

---

## Current: fabric-forwarder fused (`15a76d8ee6d`)

`base µs` = `WAN_ABLATION=0` (full forwarder fused). `cmp µs` = ABL8 pure-compute floor (all
I/O + fabric + g/s stubbed, compute runs full-speed on garbage); `cmp%` = `cmp/base` = how
much of wall-clock is pure compute. Δ columns are µs removed by skipping that one component.

| config | base µs | rope | input | output | fabric | g/s | wgt | **all-IO** | **cmp µs (cmp%)** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flux_tp4_N64_phn0   (1.0×) | 107.6 | −0.1 | +1.5 | +1.4 | +6.8 | +1.4 | +0.2 | +11.0 | **96.6 (90%)** |
| flux_tp4_N2048_phn0 (1.3×) | 135.8 | +1.3 | +9.7 | +10.2 | +6.7 | +1.5 | −4.4 | +29.3 | **106.5 (78%)** |
| flux_tp8_N4096_phn0 (1.7×) | 163.0 | +2.6 | +6.5 | +8.5 | **+25.2** | +4.9 | −4.8 | +47.2 | **115.8 (71%)** |
| tp4_v_textcross_q_s1 (1.3×)| 67.5  | +0.2 | +3.2 | +5.7 | +7.6 | +1.6 | +0.4 | +18.0 | **49.5 (73%)** |
| tp4_v_textcross_k (1.5×)   | 55.7  | +0.6 | +6.1 | +11.7 | +4.6 | +1.1 | +0.4 | +22.3 | **33.4 (60%)** |
| tp4_a_textcross_k (1.8×)   | 40.1  | +0.0 | +2.2 | +5.3 | +3.9 | +0.2 | −0.1 | +12.6 | **27.4 (68%)** |
| self_sp32_N2368 (1.2×)     | 153.6 | +0.2 | +5.2 | +5.1 | +11.9 | +2.6 | −3.4 | +24.8 | **128.8 (84%)** |
| cross_q_sp32_N2368 (1.3×)  | 112.4 | −0.1 | +9.9 | +17.2 | +9.1 | +1.6 | +0.6 | +38.2 | **74.1 (66%)** |
| cross_k_prompt_L512 (1.6×) | 46.3  | +0.1 | +2.7 | +4.0 | +3.9 | +0.7 | −0.1 | +11.7 | **34.6 (75%)** |

`COMPUTE` (skip-compute, `WAN_ABLATION=7`) **hangs by design** in the forwarder model: the
worker blocks on `cb_wait_front(stats_transposed_local_cb)` for compute's PRE sum-of-squares
tile before pushing its stick to the forwarder, so stubbing compute deadlocks the AG handshake.
The pure-compute floor (`cmp µs`, ABL8) is the proxy for the compute cost.

## What changed vs the transpose-MUX fused (the original sweep)

The original sweep found **COMPUTE #1, FABRIC #2 (and #1 at TP=8, +81.8µs)**. The fabric
forwarder was built to attack exactly that. It worked — the **fabric and gather/scatter levers
collapsed**, and the op is now firmly **compute-bound everywhere**:

| config | fabric Δ: old → new | g/s Δ: old → new |
|---|---:|---:|
| flux_tp8_N4096 | 81.8 → **25.2** (−69%) | 19.9 → **4.9** |
| self_sp32_N2368 | 19.7 → **11.9** | 7.4 → **2.6** |
| cross_q_sp32_N2368 | 19.0 → **9.1** | 6.9 → **1.6** |
| flux_tp4_N2048 | 15.4 → **6.7** | 3.5 → **1.5** |
| tp4_v_textcross_q_s1 | 14.8 → **7.6** | 4.4 → **1.6** |
| tp4_a_textcross_k | 14.5 → **3.9** | 4.5 → **0.2** |
| cross_k_prompt_L512 | 7.8 → **3.9** | 2.2 → **0.7** |

The TP=8 fabric cost — previously the single biggest lever in the whole study — dropped from
**82µs to 25µs**. gather/scatter is now sub-5µs everywhere (transpose g/s + coalesced packets).

## Levers, ranked (forwarder)

1. **COMPUTE is now THE lever, by a wide margin** — pure compute is **60–90%** of wall-clock on
   every shape (90% on the tiny flux_tp4_N64, which is why it doesn't speed up at all). With
   fabric/IO shrunk, everything that's left is the math: PRE sum-of-squares + POST norm/weight/
   RoPE. **To go faster, optimize the compute kernel.**
2. **output-write and input-read are the #2/#3 levers** (each ~5–17µs; output is biggest on
   cross_q_sp32 +17.2, v_textcross_k +11.7, flux_tp4_N2048 +10.2). These are the next DRAM
   targets once compute is addressed — e.g. wider write coalescing / better overlap.
3. **fabric is no longer dominant** — only flux_tp8 still shows a real fabric cost (+25µs, the
   8-wide ring); on TP=4 it's now 4–12µs and well-hidden. The forwarder spent this lever.
4. **gather/scatter, rope, weight/bias are spent / negligible** — all ≤5µs, several within
   run-to-run noise (the negative wgt entries).

## Bottom line
The forwarder did its job: it converted these from fabric-bound to **compute-bound**. The next
optimization target is unambiguously **the compute kernel** (PRE sum-of-squares + POST math),
then the **output-write / input-read** DRAM paths. Fabric and gather/scatter are no longer worth
further tuning except possibly the 8-wide-ring fabric at TP=8.
