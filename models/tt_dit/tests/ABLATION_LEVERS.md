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

---

# Full-config ablation — the 28 configs outside the minimal set

Same method, run on every config NOT in the 9-shape minimal set (WAN tp4, LTX tp4,
FLUX tp4/tp8). Ablation 7 (skip-compute) is omitted (deadlocks the forwarder
handshake). `cmp%` = ABL8 pure-compute floor / base. Δµs = `base − ablated`.

| config | base µs | rope | input | output | fabric | g/s | wgt | all-IO | cmp% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tp4_v_block_s1       | 75.0  | +0.2 | +2.7 | +5.4 | +6.7 | +0.8 | +0.1 | +17.6 | 77% |
| tp4_v_block_s2       | 173.9 | +0.3 | +14.4 | +26.8 | +14.1 | +4.4 | +1.2 | +56.0 | 68% |
| tp4_v_selfattn_qk_s1 | 91.5  | +0.8 | +3.2 | +1.5 | +7.9 | +1.5 | +0.3 | +15.2 | 83% |
| tp4_v_selfattn_qk_s2 | 223.5 | +20.5 | +15.5 | +27.1 | +6.4 | +0.9 | +5.9 | +61.7 | 72% |
| tp4_a2v_videoQ_s1    | 68.9  | +0.4 | +1.4 | +4.9 | +6.9 | +0.9 | −0.5 | +16.8 | 76% |
| tp4_a2v_videoQ_s2    | 137.9 | +4.1 | +1.8 | +8.6 | +14.3 | +2.9 | +0.4 | +35.8 | 74% |
| tp4_a2v_audioK       | 38.6  | +0.2 | +0.3 | +0.8 | +3.8 | +0.4 | +0.0 | +5.8 | 85% |
| tp4_v_textcross_q_s2 | 153.6 | −0.2 | +13.8 | +26.1 | +13.0 | +3.7 | +0.4 | +55.5 | 64% |
| flux_tp4_N512_phn0   | 70.0  | +0.3 | +3.2 | +1.1 | +4.2 | +0.8 | −0.9 | +9.4 | 87% |
| flux_tp4_N8192_phn0  | 438.1 | −1.0 | +20.2 | +19.8 | +18.6 | +7.1 | −4.2 | +75.3 | 83% |
| flux_tp8_N1024_phn0  | 61.0  | +0.4 | −10.0 | +4.6 | +7.2 | +1.0 | −3.5 | +16.9 | 72% |
| flux_tp8_N128_phn0   | 48.4  | −0.5 | −0.1 | +0.8 | +6.6 | +0.7 | −0.1 | +8.2 | 83% |
| flux_tp8_N16384_phn0 | 586.6 | +5.0 | +26.1 | +28.7 | **+104.9** | +27.3 | −4.8 | +183.9 | 69% |
| self_sp4_N18944      | 836.6 | +1.2 | +13.7 | +29.9 | +55.4 | +14.0 | +0.1 | +125.7 | 85% |
| self_sp8_N9472       | 448.6 | +0.1 | +8.1 | +11.9 | +32.5 | +7.2 | −0.3 | +64.6 | 86% |
| cross_q_sp4_N18944   | 582.2 | −0.1 | +58.5 | **+107.0** | +55.2 | +30.7 | +1.2 | +202.6 | 65% |
| cross_q_sp8_N9472    | 315.8 | −0.2 | +29.8 | +54.8 | +27.7 | +11.8 | +1.4 | +108.2 | 66% |
| flux_tp4_N512_phn1   | 81.6  | +0.5 | +0.1 | +1.4 | +0.5 | +0.4 | +0.3 | +1.4 | 98% |
| flux_tp4_N64_phn1    | 79.8  | +0.4 | +0.3 | +1.2 | +0.5 | +0.4 | +0.1 | +1.2 | 99% |
| flux_tp4_N2048_phn1  | 126.3 | +0.3 | +12.7 | +21.8 | −0.0 | −0.2 | −1.4 | +40.0 | 68% |
| flux_tp4_N8192_phn1  | 346.3 | +6.9 | +22.6 | +26.5 | +0.1 | +0.3 | +4.3 | +50.4 | 85% |
| flux_tp8_N1024_phn1  | 53.5  | +0.2 | +1.8 | +3.1 | +0.2 | +0.1 | −1.0 | +5.9 | 89% |
| flux_tp8_N128_phn1   | 45.3  | +0.7 | +0.6 | +1.4 | +0.4 | +0.5 | +0.5 | +1.3 | 97% |
| flux_tp8_N4096_phn1  | 115.6 | +7.4 | +15.5 | +9.2 | −0.1 | −0.2 | +3.5 | +28.7 | 75% |
| flux_tp8_N16384_phn1 | 366.7 | +46.5 | +55.6 | +39.9 | −1.6 | −0.2 | +6.7 | +69.8 | 81% |

(32-row audio configs `tp4_a_block`, `tp4_a_selfattn_qk`, `tp4_a_textcross_q` omitted —
dispatch-bound, ablation deltas are pure noise: negative all-IO, cmp%>100%.)

Findings:
- **Still compute-bound** (cmp 64–89%) — matches the minimal set.
- **At large scale the I/O levers are huge in absolute µs.** No-rope giant `cross_q_sp4_N18944`:
  **output-write +107µs**, input +58, fabric +55. So output-write is the single biggest I/O
  lever on the large no-rope WAN configs.
- **Fabric is NOT spent at the extremes.** `flux_tp8_N16384_phn0` still pays **+105µs** to the
  8-wide ring; large rope-WAN `self_sp4` +55µs. The forwarder shrank fabric on the mid shapes
  but the widest ring × most rows still leans on it.
- **per-head RoPE (phn1)** carries a real rope-read cost at large N (`flux_tp8_N16384_phn1`
  rope **+46µs**); broadcast RoPE (WAN) is ~0. phn1 fabric/g-s ≈ 0 (no all-gather), confirming
  the local path.

---

# Worker-count sweep — shapes with >2 tile-rows/worker

`WAN_RMSNORM_FORCE_WORKERS` swept over {32, 48, 64, 68} on the 13 phn0/AG shapes whose
tile-row count exceeds 2×32 (so the default 32-worker cap forces ≥3 rounds). Fused-only
µs; `↑base` = composite baseline (REBENCH) / best-worker µs. Forwarders stay at 4 (=
num_links); higher worker counts grow each forwarder's coalescing group. Max grid = 72
cores (workers + 4 forwarders ≤ 72, so 68 is the ceiling).

| shape | w=32 | w=48 | w=64 | w=68 | best | **vs w=32** | ↑base |
|---|---:|---:|---:|---:|:--:|---:|---:|
| flux_tp8_N16384_phn0 | 587.2 | 447.8 | **407.8** | 435.5 | 64 | **+44%** | 2.33× |
| self_sp4_N18944      | 836.3 | 685.4 | **634.6** | 683.7 | 64 | **+32%** | 1.82× |
| self_sp8_N9472       | 448.5 | 379.3 | **346.6** | 355.6 | 64 | **+29%** | 1.65× |
| flux_tp4_N8192_phn0  | 438.5 | 402.1 | **356.3** | 367.8 | 64 | **+23%** | 1.63× |
| flux_tp8_N4096_phn0  | 163.3 | 160.4 | **134.5** | 137.7 | 64 | **+21%** | 2.08× |
| self_sp32_N2368      | 153.5 | **128.1** | 129.6 | 133.2 | 48 | **+20%** | 1.46× |
| cross_q_sp4_N18944   | 582.0 | 530.4 | **516.7** | 555.3 | 64 | +13% | 1.82× |
| cross_q_sp8_N9472    | 315.9 | 283.2 | **278.4** | 287.7 | 64 | +13% | 1.70× |
| cross_q_sp32_N2368   | 112.3 | **100.0** | 102.2 | 105.8 | 48 | +12% | 1.41× |
| tp4_v_block_s2       | 173.4 | 164.1 | **159.2** | 166.8 | 64 | +9% | 2.72× |
| tp4_v_textcross_q_s2 | 153.8 | 148.9 | **146.1** | 153.8 | 64 | +5% | 1.69× |
| tp4_a2v_videoQ_s2    | 137.8 | 163.7 | **135.9** | 139.6 | 64 | +1% | 2.13× |
| tp4_v_selfattn_qk_s2 | 224.1 | **223.4** | 231.4 | 235.0 | 48 | +0% | 2.03× |

Findings:
- **The 32-worker cap leaves big gains on the table.** Best is **w=64** almost everywhere,
  worth **+20–44%** over w=32 on the large AG-bound shapes (TP=8-large biggest). The forwarder
  removed the shared-MUX contention that originally motivated `kMaxMuxWorkersPerChip = 32`, so
  more workers now keep shrinking the latency-bound wall.
- **w=68 (the full 72-core grid) regresses vs w=64** everywhere — leave the last cores for
  dispatch. 64 is the sweet spot; 48 is marginally better than 64 only on the two 74-tile-row
  configs (within ~2%).
- **Compute-bound LTX shapes barely move** (selfattn_qk_s2 +0%, a2v +1%) — more workers can't
  help where the bottleneck is the math, consistent with the ablation.
- **No downside for small shapes:** `num_workers = min(tile_rows, cap)`, so raising the cap
  only affects shapes with >cap tile-rows (rows > 32·cap); sub-1024-row shapes are untouched.

## Recommendation
**Raise `kMaxMuxWorkersPerChip` from 32 to 64** (or make it size/arch-conditional via the
existing `WAN_RMSNORM_WORKER_CAP` knob). It is a free +20–44% on the large AG-bound shapes
with zero effect on small ones. Cap at 64, not 68 (dispatch-core headroom). Combined with the
"compute is the #1 lever" finding, the two near-term wins are: (1) bump the worker cap to 64,
(2) optimize the compute kernel (PRE sum-of-squares + POST math).
