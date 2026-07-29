# Ablation lever sweep — `wan_fused_distributed_rmsnorm`

> **Platform: all numbers below are Wormhole 4×8 galaxy (PROXY)** — 4 links, 8×9 grid, worker
> cap 64. The target is Blackhole (2 links, 12×10 grid, torus), whose BW/FLOP ratio will shift
> the compute-vs-fabric-vs-IO balance — **re-run the ablation on BH** (`WAN_ABLATION=N`,
> `WAN_GALAXY_LINKS=2`) and add a parallel "Blackhole" section before drawing BH conclusions.
> See `RMSNORM_FUSION_FINDINGS.md` for the porting checklist.

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

---

# Fabric re-ablation at the grid-derived worker cap (cap=64)

After the worker cap change (`510cc544204`, cap=64), re-ran **skip-fabric** (`WAN_ABLATION=4`)
on all AG-path shapes. `fabric Δ` = `baseline(cap64) − skip_fabric(cap64)` = the exposed fabric
still on the critical path = the **ceiling** for a fabric-overlap optimization (e.g. overlapping
the all-gather behind the next row's PRE). All `phn1` shapes ≈ 0 (no AG — control, omitted).

| config | base µs | no-fabric µs | **fabric Δ µs** | Δ% |
|---|---:|---:|---:|---:|
| self_sp4_N18944 | 634.3 | 594.3 | **+40.0** | +6% |
| cross_q_sp4_N18944 | 516.8 | 502.8 | +13.9 | +3% |
| self_sp8_N9472 | 346.6 | 329.6 | **+16.9** | +5% |
| cross_q_sp8_N9472 | 278.5 | 268.1 | +10.3 | +4% |
| flux_tp8_N16384_phn0 | 407.5 | 368.3 | **+39.2** | +10% |
| flux_tp4_N8192_phn0 | 356.0 | 343.2 | +12.9 | +4% |
| flux_tp8_N4096_phn0 | 134.9 | 120.5 | +14.4 | +11% |
| tp4_v_block_s2 | 159.6 | 151.3 | +8.2 | +5% |
| tp4_v_selfattn_qk_s2 | 231.4 | 225.8 | +5.6 | +2% |
| tp4_a2v_videoQ_s2 | 135.7 | 131.5 | +4.1 | +3% |
| tp4_v_textcross_q_s2 | 146.4 | 136.6 | +9.8 | +7% |
| self_sp32_N2368 | 129.5 | 123.7 | +5.8 | +4% |
| cross_q_sp32_N2368 | 102.2 | 95.9 | +6.3 | +6% |
| flux_tp4_N2048_phn0 | 124.5 | 119.0 | +5.5 | +4% |
| tp4_v_block_s1 | 68.5 | 64.2 | +4.3 | +6% |
| tp4_v_selfattn_qk_s1 | 78.7 | 75.7 | +3.0 | +4% |
| tp4_a2v_videoQ_s1 | 54.4 | 50.6 | +3.8 | +7% |
| tp4_v_textcross_q_s1 | 64.3 | 59.6 | +4.6 | +7% |
| flux_tp8_N1024_phn0 | 61.0 | 53.8 | +7.2 | +12% |
| flux_tp4_N512_phn0 | 69.7 | 65.6 | +4.1 | +6% |
| flux_tp8_N128_phn0 | 48.5 | 41.9 | +6.6 | +14% |
| flux_tp4_N64_phn0 | 107.2 | 100.5 | +6.7 | +6% |
| cross_k_prompt_L512 | 46.2 | 42.4 | +3.8 | +8% |
| tp4_v_textcross_k | 55.2 | 51.0 | +4.2 | +8% |
| tp4_a_textcross_k | 40.4 | 36.1 | +4.3 | +11% |
| tp4_a2v_audioK | 38.6 | 34.5 | +4.1 | +11% |

Findings:
- **The worker cap already harvested most of the fabric cost.** flux_tp8_N16384 fabric was
  **+105µs at the old 32 cap → +39µs at cap=64**; more workers = fewer rounds = less serial
  fabric. The fabric lever is now the "remaining slice," not the dominant cost.
- **Ceiling is modest:** the most exposed fabric is **~40µs (self_sp4) / ~39µs (flux_tp8_N16384)**;
  everything else is ≤17µs absolute, mostly **3–11%** even on the giants. Small shapes show a
  higher *fraction* (8–14%) but tiny absolute (4–7µs) — and they're compute/dispatch-bound, so
  an overlap can't help much there.
- **Verdict on fabric-overlap-behind-PRE:** worthwhile **only if the large TP=8 / SP=4 shapes
  dominate** — there it's the biggest remaining structural lever (~6–10%, up to ~40µs). It needs
  a real pipeline restructure (decouple PRE(N+1) from POST(N), double-buffer stats across rows),
  so scope it to the AG path and validate on flux_tp8_N16384 + self_sp4 where the payoff sits.

---

# Blackhole IO ablation (TARGET — 2 links, 12×10 grid, worker cap 48)

Same method on the **BH 4×8 galaxy** (`WAN_GALAXY_LINKS=2`, cap48 = the BH perf optimum;
forwarder fused, origin `2d56fbdd64e`). Δµs = `base − ablated`, fused-only, min-of-2.
`no-fabric` = `WAN_ABLATION=4`, `no-output` = `3`, `no-reads` (input+weight+rope together) =
new combined `WAN_ABLATION=9`.

| config | base µs | no-fabric | no-output | no-reads | **fabric** | **output write** | **reads** |
|---|---:|---:|---:|---:|---:|---:|---:|
| self_sp4_N18944      | 410.8 | 381.0 | 297.9 | 343.0 | 30µs (7%)  | **113µs (27%)** | 68µs (17%) |
| cross_q_sp4_N18944   | 356.7 | 326.6 | 201.5 | 278.5 | 30µs (8%)  | **155µs (44%)** | 78µs (22%) |
| flux_tp8_N16384_phn0 | 275.9 | 237.9 | 215.5 | 233.0 | 38µs (14%) | **60µs (22%)**  | 43µs (16%) |
| flux_tp8_N16384_phn1 | 208.9 | 208.2 | 171.5 | 114.2 | 0µs (0%)   | 37µs (18%)      | **95µs (45%)** |
| flux_tp4_N8192_phn0  | 272.2 | 260.2 | 176.6 | 218.2 | 12µs (4%)  | **96µs (35%)**  | 54µs (20%) |

**Confirms the WH balance on BH:** the op is **DRAM-bound, output-write first** (22–44% on every
AG config; biggest on the low-compute `cross_q_sp4`), **reads second** (16–22%), **fabric spent**
(≤14%, 0% on phn1 — the forwarder did its job; split-sender is moot). vs the WH proxy the % are
close; BH fabric is a slightly bigger fraction on tp8 (14% vs 10%, 2 links).

**BH-specific:** **per-head `phn1` is read-bound, not output-bound** — reads **45%** on BH (vs 15%
on WH). No AG → input + per-head cos/sin reads dominate. So per-head FLUX.2 wants the **read**
path; every other config wants the **output-write** path.

**Next levers (DRAM):** (1) **output write** — biggest, on every AG config. (2) **reads** — #2,
and #1 for per-head. Fabric/gather-scatter are spent. Output-write experiments (barrier/flush
granularity to cut NoC contention across the ~48 writing cores; dual-NoC for full DRAM BW) follow.

### DRAM-speed tricks — both NON-LEVERS (2026-06-26, BH 4×8, cap48, fused-only, min-of-2)

Two common "go-faster on DRAM" tricks were tried against the output-write bottleneck. Both are
**no-ops** — confirming the writer is **DRAM-bandwidth-bound, not NoC-bound**.

1. **Flush/barrier granularity** (barrier every N tiles vs once per row to cut NoC contention):
   no change. Reverted.
2. **Dual-NoC output drain** (`WAN_RMSNORM_DUAL_NOC`, alternate `noc_async_write_tile` between
   `noc_index` and `1-noc_index` per tile so the output stream uses both NoCs):

   | config | dual=0 µs | dual=1 µs |
   |---|---:|---:|
   | self_sp4_N18944      | 367.84 | 367.81 |
   | cross_q_sp4_N18944   | 321.40 | 321.56 |
   | flux_tp8_N16384_phn0 | 252.10 | 252.17 |
   | flux_tp4_N8192_phn0  | 249.08 | 249.35 |

   Dead flat (≤0.1%). Implementation note: tt-metal requires **all NoC kernels in a program to
   share one `noc_mode`** (`TT_FATAL noc_modes.size()<=1`), so dual-NoC forces reader + writer +
   forwarder all to `DM_DYNAMIC_NOC` (fabric forwarder included — correctness held, PCC 99.99%+,
   so fabric-under-dynamic-NoC is safe). Even so, zero speedup. Reverted.

**Takeaway:** a single NoC already saturates DRAM write BW for this access pattern — extra NoC
injection bandwidth is not the limiter. (Tile order is also irrelevant for locality — tiles are
pages, round-robin across DRAM banks — so the head-interleaved scatter costs nothing extra.) The
output write is genuinely **DRAM-bound**; the only way to win against it is to **overlap** it (or
the fabric) with compute, i.e. software-pipelining.

### Software-pipelined AG, idea #3 (writer deferred-drain) — NO-OP (2026-06-26, BH cap48)

`WAN_RMSNORM_PIPELINE` (writer half): defer each row's output drain one iteration so its DRAM
writes overlap the next row's go-sem wait (`W_AGWAIT`, the fabric round-trip).

| config | pipe=0 µs | pipe=1 µs |
|---|---:|---:|
| self_sp4_N18944      | 428.54 | 428.66 |
| cross_q_sp4_N18944   | 323.08 | 322.92 |
| flux_tp8_N16384_phn0 | 277.71 | 278.08 |
| flux_tp4_N8192_phn0  | 269.41 | 269.39 |

**Dead flat.** First thought was "the op is compute-bound, so only the compute-side pipeline can
help" — so idea #2 was implemented next.

### Software-pipelined AG, idea #2 (compute look-ahead PRE) — ALSO NO-OP (2026-06-26, BH cap48)

Full software pipeline: compute PREs row r+1 (parity-split input CBs `input_cb`/`input_cb_b`,
each read at its front, wrap-safe; stats double-buffered) **during** row r's AG gather wait, plus
the idea-#3 writer deferred-drain. Correctness verified (the pipelined path executed — parity CBs,
look-ahead PRE, deferred drain — still bit-exact, PCC 99.99%+). Perf (min of 2 runs):

| config | pipe=0 µs | pipe=1 µs |
|---|---:|---:|
| self_sp4_N18944      | 467.48 | 467.70 |
| cross_q_sp4_N18944   | 303.86 | 303.63 |
| flux_tp8_N16384_phn0 | 274.33 | 274.78 |
| flux_tp4_N8192_phn0  | 247.82 | 248.00 |

**Dead flat (≤0.2%), rock-stable across runs — a true no-op, not noise.**

### Conclusion: the forwarder code is at its DRAM floor on BH

Three independent overlap experiments (dual-NoC, writer deferred-drain, full compute pipeline) all
give **zero**. The consistent explanation: the **fabric AG is already fully hidden** — the
dedicated forwarder cores run the ring mcast concurrently with the workers' DRAM traffic, so there
is **no exposed stall** for compute/writer/NoC pipelining to recover. The ablation's "no-fabric
7–14%" was the **DRAM-staggering confound** (removing the sem-wait reshuffles DRAM timing), not a
real exposed fabric cost. BH has 2× the FLOPs of WH but the **same** DRAM BW, so this op is
**DRAM-bound** here (more so than on WH) and already pipelined (reader prefetch + async writes).
That is why the old MUX branch's −13.6% (WH, fabric exposed at 47%) does **not** reproduce: the
forwarder rewrite already captured that win. **The remaining lever would have to reduce DRAM bytes**
(output is already bf16; tile order is irrelevant) — there is no traffic-shaping win left. All three
experiments reverted; the op ships at its DRAM/compute floor.
