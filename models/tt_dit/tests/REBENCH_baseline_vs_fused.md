# Re-bench: baseline (composite) vs fused — `wan_fused_distributed_rmsnorm`

> **Platform: Wormhole 4×8 galaxy (PROXY).** Blackhole 4×8 torus (2 links, 12×10 grid) is the
> real target — re-run with `WAN_GALAXY_LINKS=2` and paste the tables under the **Blackhole stub
> at the bottom** so BH numbers sit next to these WH ones. BH porting checklist:
> `RMSNORM_FUSION_FINDINGS.md`.

## Wormhole 4×8 galaxy (proxy — 4 links, 8×9 grid, worker cap 64)

Traced bench (`test_bench`) on a Wormhole **4×8 galaxy**. RING = full 4×8 mesh, TP on a closed
axis (4-wide for TP=4, 8-wide for TP=8), replicate the other axis. 4 links. WAN bench = 100
iters, LTX/FLUX = 50 iters. **`fused-F` columns re-swept on `cglagovich/fused_layernorm_bringup`
(post-rebase, worker-cap sweep 2026-07-01)** — cap 64 re-confirmed optimal for every WH shape
(RMS + LN); numbers within run-to-run noise of the prior measurement. `baseline` and `fused-T`
(historical transpose-MUX op) are unchanged from the original run.

Columns:
- **`baseline µs`** — on-device composite reference (separate all-gather + rmsnorm ops),
  measured at `43841c2caaa`. Composite code unchanged since, so reused (it's flaky under
  traced bench on the heavy WAN N≥9472 configs, so not re-measured).
- **`fused-T µs`** — prior fused op, transpose-based gather/scatter + MUX all-gather
  (`0ad69fc8b9b`). Reference.
- **`fused-F⁽ʷ³²⁾ µs`** — fabric-forwarder all-gather (`15a76d8ee6d`) at the **old 32-worker
  cap**.
- **`fused-F⁽ʷ⁶⁴⁾ µs`** — same forwarder op at the **grid-derived worker cap** (`510cc544204`:
  workers = floor((grid − forwarders)/grid.x)·grid.x; **64** on this 8×9 galaxy). This is the
  **current op state**.
- **`w32→w64`** = the worker-cap effect = `fused-F⁽ʷ³²⁾ / fused-F⁽ʷ⁶⁴⁾ − 1`.
- **`↑F`** = `baseline / fused-F⁽ʷ⁶⁴⁾` = current forwarder fused speedup over composite.

The worker cap only affects the all-gather path (`phn0` / qk / block) with >32 tile-rows
(rows > 1024); **`phn1` (per-head norm) takes the local `is_tp_1` path — never 32-capped, so
`w32→w64` ≈ 0 there** (a clean control), as does any shape with ≤32 tile-rows.

## Wan2.2 — TP=4 (ring), feat 1280/dev, 10 heads, head_dim 128

| config | rows | pattern | baseline µs | fused-T µs | fused-F⁽ʷ³²⁾ µs | **fused-F⁽ʷ⁶⁴⁾ µs** | **w32→w64** | ↑F |
|---|---|---|---:|---:|---:|---:|---:|---:|
| self_sp4_N18944 | 18944 | qk+rope | 1153.62 | 896.13 | 847.0 | **631.7** | **+34%** | **1.83×** |
| self_sp8_N9472 | 9472 | qk+rope | 572.20 | 503.61 | 453.8 | **347.0** | **+31%** | **1.65×** |
| self_sp32_N2368 | 2368 | qk+rope | 187.10 | 191.09 | 154.3 | **130.4** | **+18%** | **1.43×** |
| cross_q_sp4_N18944 | 18944 | qk | 940.36 | 581.33 | 584.7 | **520.2** | **+12%** | **1.81×** |
| cross_q_sp8_N9472 | 9472 | qk | 472.45 | 333.42 | 316.6 | **278.8** | **+14%** | **1.69×** |
| cross_q_sp32_N2368 | 2368 | qk | 141.12 | 138.13 | 113.3 | **102.8** | **+10%** | **1.37×** |
| cross_k_prompt_L512 | 512 | qk | 73.36 | 67.00 | 46.2 | **46.3** | -0% | **1.58×** |

## LTX-2.3 AV — TP=4 (ring)

| config | rows | feat | heads | hd | pattern | baseline µs | fused-T µs | fused-F⁽ʷ³²⁾ µs | **fused-F⁽ʷ⁶⁴⁾ µs** | **w32→w64** | ↑F |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| tp4_v_block_s1 | 1216 | 1024 | 1 | – | block+addcmul | 139.59 | 110.23 | 75.0 | **68.4** | **+10%** | **2.04×** |
| tp4_v_block_s2 | 4864 | 1024 | 1 | – | block+addcmul | 432.91 | 203.93 | 173.7 | **159.5** | **+9%** | **2.71×** |
| tp4_a_block | 32 | 512 | 1 | – | block+addcmul | 33.63 | 24.68 | 27.1 | **27.1** | +0% | **1.24×** |
| tp4_v_selfattn_qk_s1 | 1216 | 1024 | 8 | 128 | qk+rope | 144.71 | 128.64 | 91.5 | **78.5** | **+17%** | **1.84×** |
| tp4_v_selfattn_qk_s2 | 4864 | 1024 | 8 | 128 | qk+rope | 453.47 | 254.88 | 223.2 | **230.2** | **-3%** | **1.97×** |
| tp4_a_selfattn_qk | 32 | 512 | 8 | 64 | qk+rope | 51.64 | 29.22 | 31.8 | **31.9** | -0% | **1.62×** |
| tp4_a2v_videoQ_s1 | 1216 | 512 | 8 | 64 | qk+rope | 106.51 | 104.43 | 68.7 | **54.3** | **+27%** | **1.96×** |
| tp4_a2v_videoQ_s2 | 4864 | 512 | 8 | 64 | qk+rope | 289.29 | 184.93 | 138.9 | **135.1** | **+3%** | **2.14×** |
| tp4_a2v_audioK | 256 | 512 | 8 | 64 | qk+rope | 79.67 | 58.01 | 37.9 | **38.0** | -0% | **2.10×** |
| tp4_v_textcross_q_s1 | 1216 | 1024 | 8 | 128 | qk | 88.66 | 100.83 | 67.3 | **64.0** | **+5%** | **1.39×** |
| tp4_v_textcross_q_s2 | 4864 | 1024 | 8 | 128 | qk | 246.58 | 184.81 | 153.6 | **146.8** | **+5%** | **1.68×** |
| tp4_v_textcross_k | 1024 | 1024 | 8 | 128 | qk | 82.48 | 85.37 | 54.7 | **54.8** | -0% | **1.51×** |
| tp4_a_textcross_q | 32 | 512 | 8 | 64 | qk | 32.37 | 22.46 | 25.0 | **25.0** | +0% | **1.29×** |
| tp4_a_textcross_k | 1024 | 512 | 8 | 64 | qk | 71.88 | 75.26 | 39.8 | **39.7** | +0% | **1.81×** |

## FLUX — TP=4 + TP=8 (ring), feat 1536 (TP4) / 768 (TP8)

| config | tp | rows | heads | pattern | baseline µs | fused-T µs | fused-F⁽ʷ³²⁾ µs | **fused-F⁽ʷ⁶⁴⁾ µs** | **w32→w64** | ↑F |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| flux_tp4_N512_phn0 | 4 | 512 | 12 | qk+rope | 117.96 | 91.85 | 70.3 | **70.1** | -0% | **1.68×** |
| flux_tp4_N512_phn1 | 4 | 512 | 12 | perhead+rope | 117.79 | 81.18 | 81.3 | **81.3** | -0% | **1.45×**\* |
| flux_tp4_N64_phn0 | 4 | 64 | 12 | qk+rope | 92.93 | 102.97 | 108.0 | **108.2** | -0% | **0.86×** |
| flux_tp4_N64_phn1 | 4 | 64 | 12 | perhead+rope | 92.46 | 102.91 | 80.2 | **79.6** | +1% | **1.16×**\* |
| flux_tp4_N2048_phn0 | 4 | 2048 | 12 | qk+rope | 175.20 | 167.41 | 136.2 | **125.0** | **+9%** | **1.40×** |
| flux_tp4_N2048_phn1 | 4 | 2048 | 12 | perhead+rope | 174.92 | 126.12 | 126.0 | **126.2** | -0% | **1.39×**\* |
| flux_tp4_N8192_phn0 | 4 | 8192 | 12 | qk+rope | 579.60 | 480.37 | 440.6 | **356.7** | **+24%** | **1.62×** |
| flux_tp4_N8192_phn1 | 4 | 8192 | 12 | perhead+rope | 577.75 | 345.77 | 346.0 | **346.3** | -0% | **1.67×**\* |
| flux_tp8_N1024_phn0 | 8 | 1024 | 6 | qk+rope | 122.94 | 108.23 | 60.8 | **61.2** | -1% | **2.01×** |
| flux_tp8_N1024_phn1 | 8 | 1024 | 6 | perhead+rope | 123.02 | 53.42 | 53.3 | **53.5** | -0% | **2.30×**\* |
| flux_tp8_N128_phn0 | 8 | 128 | 6 | qk+rope | 95.87 | 64.19 | 48.5 | **48.3** | +0% | **1.98×** |
| flux_tp8_N128_phn1 | 8 | 128 | 6 | perhead+rope | 95.83 | 44.70 | 44.8 | **44.8** | +0% | **2.14×**\* |
| flux_tp8_N4096_phn0 | 8 | 4096 | 6 | qk+rope | 279.12 | 250.86 | 163.7 | **135.2** | **+21%** | **2.06×** |
| flux_tp8_N4096_phn1 | 8 | 4096 | 6 | perhead+rope | 278.27 | 115.22 | 115.7 | **115.6** | +0% | **2.41×**\* |
| flux_tp8_N16384_phn0 | 8 | 16384 | 6 | qk+rope | 951.93 | 770.42 | 589.3 | **407.6** | **+45%** | **2.34×** |
| flux_tp8_N16384_phn1 | 8 | 16384 | 6 | perhead+rope | 959.65 | 366.43 | 366.6 | **367.4** | -0% | **2.61×**\* |

\* `phn1` baseline is the full-row composite (per-head norm has no apples-to-apples
composite); phn1 does not use the all-gather / worker-cap path.

## Notes
- **Worker cap (w32→w64) is a clean win on the large all-gather-bound shapes** — biggest where
  the gather dominates: flux_tp8_N16384 **+44%** (587→408µs), self_sp4 **+32%**, self_sp8 **+29%**,
  flux_tp4_N8192 **+23%**, flux_tp8_N4096 **+21%**, a2v_videoQ_s1 **+27%**. Small / ≤32-tile-row
  shapes are unchanged (≈0%), and one compute-bound shape (v_selfattn_qk_s2) gives back ~3%
  (more workers can't help where the bottleneck is the math).
- **vs composite (↑F):** the current op is **1.4–2.7×** on the all-gather shapes (and 2.0–2.6×
  on TP=8). Only the tiny 64-row flux_tp4_N64 sits at 0.87× — compute/dispatch-bound, fabric
  was never its bottleneck.
- **vs the prior fused-T:** the forwarder + worker cap together cut the large shapes by up to
  ~47% (flux_tp8_N16384 770→408µs).

## FLUX LayerNorm — TP=4 + TP=8 (ring), WH proxy — whole-row, weight+bias, no RoPE

Fused Welford LayerNorm (`norm_type=LAYERNORM`) vs the composite `dit_layernorm` chain
(`dit_layernorm_pre_allgather` → `all_gather_persistent_buffer` → `dit_layernorm_post_allgather`,
weight+bias). This is the adaLN block-norm path — separate from the RMSNorm (`qk`) tables above.
`↑LN` = baseline/fused. µs/iter (traced, 4 links). Bench: `test_layernorm_module_bench -k flux`.

| config | tp | rows | feat | baseline µs | fused µs | **↑LN** |
|---|---:|---:|---:|---:|---:|---:|
| flux_tp4_N64    | 4 | 64    | 1536 | 105.3 | 154.0 | 0.68× |
| flux_tp4_N512   | 4 | 512   | 1536 | 114.5 | 102.6 | **1.12×** |
| flux_tp4_N2048  | 4 | 2048  | 1536 | 179.4 | 170.6 | 1.05× |
| flux_tp4_N8192  | 4 | 8192  | 1536 | 571.8 | 401.5 | **1.42×** |
| flux_tp8_N128   | 8 | 128   | 768  | 108.9 | 69.0  | **1.58×** |
| flux_tp8_N1024  | 8 | 1024  | 768  | 140.3 | 90.2  | **1.56×** |
| flux_tp8_N4096  | 8 | 4096  | 768  | 306.0 | 170.7 | **1.79×** |
| flux_tp8_N16384 | 8 | 16384 | 768  | 986.0 | 527.6 | **1.87×** |

Correctness: `pcc(fused:torch)` 100.0003–100.0010%, bit-exact deterministic, 5/5 LN corr.
**↑LN is below the RMSNorm `↑F` above** because a *stable* LayerNorm needs 2-stat Welford
(mean+var) where RMSNorm needs only 1-stat sum-of-squares — the LN cross-shard combine and PRE
are inherently heavier. The big lever was the cross-shard merge: profiling showed
`combine_welford_partials` was 62.8% of LN compute, replaced by an equal-count Welford combine
(numerically identical, stable deviation form) — TP8 large 1.44× → **1.87×**. Full write-up:
`FLUX_LN_SPEEDUP_PROGRESS.md`.

---

## Blackhole 4×8 torus (TARGET — 2 links, 12×10 grid, ring+row-aware worker knee)

Traced `test_bench` on the **BH 4×8 galaxy** (`bh4x8links2`, 32 chips), forwarder fused op,
`WAN_GALAXY_LINKS=2`. `baseline µs` = on-device composite (measured fresh on BH); `fused µs` =
forwarder fused; `↑BH` = baseline/fused; `↑WH` = the Wormhole-proxy `↑F` above for the same
config. Correctness re-validated on BH first (det=OK, PCC 99.99–100%, det_ndiff=0/9 across all
ring configs). **chunk=1. Numbers below are the current default heuristic (see below).**

> **BH worker-count heuristic (re-swept 2026-07-01, `derive_worker_cap`):**
> 1. **Validity clamp:** workers_per_forwarder ≤ sticks_per_packet, i.e. cap ≤ sticks_per_packet ×
>    num_forwarders = **64 for RMS** (128 B sticks) / **32 for LayerNorm** (256 B, 2-stat Welford).
>    The raw grid budget (~108 on 12×10, 2 forwarders → 54/fwd) would `TT_FATAL`; this clamps it.
> 2. **The perf optimum is workload-dependent — a fixed cap is wrong.** The full sweep (cap
>    32/48/64) shows two competing effects: per-round DRAM/NoC+fabric contention (favours fewer
>    workers, scales with ring_size and rows) vs round count = ceil(rows/workers) (favours more).
>    The knee: **ring_size ≥ 8 → 48** (heavy 8-hop fabric: FLUX tp8 N16384 275@48 vs 335@64);
>    **ring_size ≤ 4 & rows > 448 → 48** (contention on the two most expensive shapes: Wan
>    self/cross_sp4 592-row, 410/357@48 vs 477/434@64); **ring_size ≤ 4 & rows ≤ 448 → 64**
>    (round-bound; biggest wins the 152-row LTX s2 shapes, e.g. videoQ_s2 **143→75µs, 1.9×**, and
>    Wan sp8 296-row 220→205, FLUX tp4 N2048/N8192). (Round-*balancing* to fewer even workers was
>    tried and did NOT help — it is a round-count win, not a remainder-balance one.) LayerNorm is
>    validity-capped at 32 and monotonically wants that max, so the ring/row branch never binds it.
>    Cost of the knee: the 74-row Wan sp32 shapes lose ~3–4% (79→82µs) — negligible vs the wins.

> **`eff GB/s` = effective per-chip DRAM bandwidth of the fused op** = (input read + output
> write) ÷ fused µs = `N × feat_local × 4` (bf16 activation in **and** out, 2 B each) ÷ `t`.
> It is a *lower bound* on total DRAM traffic — it excludes the RoPE cos/sin and weight/bias
> reads (which add ~10–20% on the `+rope` configs) and the tiny AG stat sticks. **BH per-chip
> DRAM peak ≈ 512 GB/s**, so the `% peak` in parens is `eff GB/s / 512`. The large all-gather
> configs land at ~35–53% of peak; small/low-row shapes are latency/dispatch-bound (low GB/s).

### Wan2.2 — TP=4 ring (feat 1280/dev)

| config | rows | pattern | baseline µs | fused µs | eff GB/s | ↑BH | ↑WH |
|---|---:|---|---:|---:|---:|---:|---:|
| self_sp4_N18944    | 18944 | qk+rope | 584.4 | 410.3 | 236 (46%) | **1.42×** | 1.82× |
| self_sp8_N9472     | 9472  | qk+rope | 313.3 | 204.6 | 237 (46%) | **1.53×** | 1.65× |
| self_sp32_N2368    | 2368  | qk+rope | 104.2 | 82.0  | 148 (29%) | 1.27× | 1.45× |
| cross_q_sp4_N18944 | 18944 | qk      | 512.2 | 356.6 | 272 (53%) | **1.44×** | 1.82× |
| cross_q_sp8_N9472  | 9472  | qk      | 272.6 | 175.1 | 277 (54%) | **1.56×** | 1.70× |
| cross_q_sp32_N2368 | 2368  | qk      | 93.4  | 68.6  | 177 (35%) | 1.36× | 1.38× |
| cross_k_prompt_L512| 512   | qk      | 51.3  | 29.3  | 90 (17%)  | **1.75×** | 1.59× |

### LTX-2.3 AV — TP=4 ring

| config | rows | feat | pattern | baseline µs | fused µs | eff GB/s | ↑BH | ↑WH |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| tp4_v_block_s1       | 1216 | 1024 | block+addcmul | 86.9  | 38.7  | 129 (25%) | **2.25×** | 2.04× |
| tp4_v_block_s2       | 4864 | 1024 | block+addcmul | 253.3 | 94.0  | 212 (41%) | **2.69×** | 2.71× |
| tp4_a_block          | 32   | 512  | block+addcmul | 28.1  | 18.6  | 4 (1%) | **1.51×** | 1.21× |
| tp4_v_selfattn_qk_s1 | 1216 | 1024 | qk+rope       | 94.0  | 44.5  | 112 (22%) | **2.11×** | 1.84× |
| tp4_v_selfattn_qk_s2 | 4864 | 1024 | qk+rope       | 264.4 | 126.2 | 158 (31%) | **2.10×** | 1.96× |
| tp4_a_selfattn_qk    | 32   | 512  | qk+rope       | 38.1  | 20.7  | 3 (1%) | **1.84×** | 1.59× |
| tp4_a2v_videoQ_s1    | 1216 | 512  | qk+rope       | 73.8  | 31.2  | 80 (16%) | **2.37×** | 1.96× |
| tp4_a2v_videoQ_s2    | 4864 | 512  | qk+rope       | 171.2 | 75.5  | 132 (26%) | **2.27×** | 2.13× |
| tp4_a2v_audioK       | 256  | 512  | qk+rope       | 54.9  | 24.7  | 21 (4%) | **2.23×** | 2.06× |
| tp4_v_textcross_q_s1 | 1216 | 1024 | qk            | 62.5  | 36.2  | 138 (27%) | 1.72× | 1.38× |
| tp4_v_textcross_q_s2 | 4864 | 1024 | qk            | 153.8 | 87.0  | 229 (45%) | **1.77×** | 1.68× |
| tp4_v_textcross_k    | 1024 | 1024 | qk            | 59.9  | 34.9  | 120 (23%) | 1.72× | 1.49× |
| tp4_a_textcross_q    | 32   | 512  | qk            | 54.4  | 17.8  | 4 (1%) | **3.05×** | 1.26× |
| tp4_a_textcross_k    | 1024 | 512  | qk            | 51.9  | 26.8  | 78 (15%) | **1.94×** | — |

The earlier `tp4_a_block` regression (fused 57.4 > base 28.1, 0.49×) is **fixed** — it now runs
18.6 µs (**1.51×**), and `tp4_a_selfattn_qk` likewise improved (27.4 → 20.7 µs, **1.84×**). Both
are the tiny 32-row audio shapes; the fixes came in with the small-shape work integrated from
`kevinmi/fused_rms_norm` (re-measured 2026-06-26 after the rebase onto main).

### FLUX — TP=4 + TP=8 ring (`phn0` = whole-row qk+rope; `phn1` = per-head QK-norm, no AG)

| config | tp | rows | feat | baseline µs | fused µs | eff GB/s | ↑BH |
|---|---:|---:|---:|---:|---:|---:|---:|
| flux_tp4_N512_phn0   | 4 | 512   | 1536 | 69.1  | 39.5  | 80 (16%) | **1.75×** |
| flux_tp4_N512_phn1   | 4 | 512   | 1536 | 69.1  | 43.8  | 72 (14%) | 1.58× |
| flux_tp4_N64_phn0    | 4 | 64    | 1536 | 65.5  | 58.1  | 7 (1%) | 1.13× |
| flux_tp4_N64_phn1    | 4 | 64    | 1536 | 65.5  | 41.1  | 10 (2%) | 1.60× |
| flux_tp4_N2048_phn0  | 4 | 2048  | 1536 | 104.2 | 70.0  | 180 (35%) | **1.49×** |
| flux_tp4_N2048_phn1  | 4 | 2048  | 1536 | 104.1 | 70.2  | 179 (35%) | 1.48× |
| flux_tp4_N8192_phn0  | 4 | 8192  | 1536 | 310.7 | 255.1 | 197 (39%) | **1.22×** |
| flux_tp4_N8192_phn1  | 4 | 8192  | 1536 | 310.8 | 197.1 | 255 (50%) | 1.58× |
| flux_tp8_N1024_phn0  | 8 | 1024  | 768  | 77.3  | 36.7  | 86 (17%) | **2.10×** |
| flux_tp8_N1024_phn1  | 8 | 1024  | 768  | 76.9  | 30.3  | 104 (20%) | **2.54×** |
| flux_tp8_N128_phn0   | 8 | 128   | 768  | 58.8  | 28.2  | 14 (3%) | **2.08×** |
| flux_tp8_N128_phn1   | 8 | 128   | 768  | 58.7  | 24.1  | 16 (3%) | **2.43×** |
| flux_tp8_N4096_phn0  | 8 | 4096  | 768  | 175.9 | 82.8  | 152 (30%) | **2.12×** |
| flux_tp8_N4096_phn1  | 8 | 4096  | 768  | 175.8 | 79.4  | 158 (31%) | **2.22×** |
| flux_tp8_N16384_phn0 | 8 | 16384 | 768  | 552.5 | 276.6 | 182 (36%) | **2.00×** (WH 2.34×) |
| flux_tp8_N16384_phn1 | 8 | 16384 | 768  | 550.3 | 208.7 | 241 (47%) | **2.64×** |

### FLUX LayerNorm (BH) — TP=4 + TP=8 (ring)

Benched on the **BH 4×8 galaxy** (`WAN_GALAXY_LINKS=2`, `test_layernorm_module_bench -k flux`),
branch `cglagovich/fused_layernorm_bringup`, 2026-07-01. `↑BH` = baseline/fused; `↑WH` = the WH
LayerNorm `↑LN` above for the same shape. Correctness first (`test_layernorm_corr`): **5/5 pass,
PCC 100.0003–100.0010%, bit-exact (det_ndiff=0/3)**.

| config | tp | rows | feat | baseline µs | fused µs | ↑BH | ↑WH |
|---|---:|---:|---:|---:|---:|---:|---:|
| flux_tp4_N64    | 4 | 64    | 1536 | 60.81  | 90.43  | 0.67× | 0.68× |
| flux_tp4_N512   | 4 | 512   | 1536 | 69.50  | 60.20  | **1.15×** | 1.12× |
| flux_tp4_N2048  | 4 | 2048  | 1536 | 115.47 | 111.40 | 1.04× | 1.05× |
| flux_tp4_N8192  | 4 | 8192  | 1536 | 380.71 | 370.40 | 1.03× | 1.42× |
| flux_tp8_N128   | 8 | 128   | 768  | 63.80  | 41.88  | **1.52×** | 1.58× |
| flux_tp8_N1024  | 8 | 1024  | 768  | 93.44  | 53.17  | **1.76×** | 1.56× |
| flux_tp8_N4096  | 8 | 4096  | 768  | 224.36 | 164.65 | **1.36×** | 1.79× |
| flux_tp8_N16384 | 8 | 16384 | 768  | 707.17 | 635.26 | 1.11× | 1.87× |

**Reading it:** small/mid shapes track WH closely and TP=8 mid even beats it (N1024 **1.76×** vs
1.56×, from BH's ~2× compute), but the **large fabric-bound shapes drop hard** — N8192 tp4 1.03×
(WH 1.42×), N16384 tp8 1.11× (WH 1.87×). This is a *bigger* BH-vs-WH gap than RMSNorm (~15–25%)
because LayerNorm's Welford all-gather carries **2 stats (mean+var)** vs RMSNorm's 1, so it's
~2× the fabric payload — and BH has **2 fabric links vs WH's 4**. The absolutes confirm it: on
N16384 tp8 the BH *baseline* is faster than WH (707 vs 986 µs, more compute) yet the BH *fused*
path is slower than WH (635 vs 528 µs) — the fused op is fabric-bound there and BH's half-BW
fabric dominates, collapsing the ratio. Net: fused LN still wins on every shape except the tiny
64-row tp4 (0.67×, dispatch-bound, same as WH), strongest on TP=8 mid-size (1.4–1.8×).

### Bottom line (BH)
The forwarder fused op gives a **strong, real speedup on BH**: **~2× on all FLUX TP=8**,
**1.4–2.5× across Wan/LTX/FLUX**, far above the old transpose-MUX op. BH runs ~15–25% below the
WH proxy ↑ (2 links vs 4), as expected, but the goal — significant speedup over composite on the
real target — is met. The earlier open items are now all resolved: `derive_worker_cap` is
arch- and workload-aware (BH ring+row knee — 64 for round-bound ring≤4 mid-size shapes, 48 for
fabric/contention-bound ring≥8 or >448-row shapes, clamped to the fabric-packet validity limit),
and the `tp4_a_block` regression is fixed (now 1.51×).

**Re-swept + re-verified 2026-07-01 (on `cglagovich/fused_layernorm_bringup`):** full worker
sweep (cap 32/48/64) over all RMS + LN shapes → the ring+row-aware knee above. All Wan/LTX/FLUX
RMS shapes re-run at the new default: PCC 99.99–100% (F:torch), bit-exact (`det_ndiff=0/9`),
flagged NONE. The knee moved the mid-size ring≤4 shapes to 64 workers: **LTX s2 shapes 8–47%
faster** (videoQ_s2 143.7→75.5, selfattn_qk_s2 169.9→126.2, textcross_q_s2 109.0→87.0,
v_block_s2 102.5→94.0), **Wan sp8 7%** (220→205 / 187→175), **FLUX tp4 N2048/N8192 6–15%**
(82→70 / 272→255); the 592/512-row and ring8 shapes are unchanged (stay at 48), and the 74-row
Wan sp32 pair regress ~3–4% (the knee's only cost). LN unchanged (validity-capped at 32, already
optimal). All updated in the tables above.

**Effective DRAM bandwidth (`eff GB/s`):** the largest all-gather-bound configs reach
**~46–53% of the ~512 GB/s per-chip peak** (cross_q_sp4 **272 GB/s / 53%**, self_sp4 236/46%,
flux_tp8_N16384_phn1 241/47%, flux_tp4_N8192_phn1 255/50%), with the no-AG `phn1` per-head path
consistently higher than its `phn0` twin (no fabric round-trip diluting the BW). This is the
**DRAM-balanced ceiling** the device profiling pinned down: at these sizes read, compute, and
write are all co-busy at ~42–44 µs/tile-row, so the op is moving bytes at roughly half of peak
DRAM BW and is **not** leaving a large overlap win on the table (software-pipelining the AG was a
measured no-op — see `ABLATION_LEVERS.md`). Smaller / low-row shapes are latency- and
dispatch-bound (single-digit % of peak), which is expected — they don't have enough rows to
amortize fixed costs, not a bandwidth problem.
