# ND-sharded expert weights + hybrid (SP=2), new MoE ops on the 4x4 sub-torus (SP=4), ND A/B on SP=8

Branch `vmelnykov/m3_moe_fabric2d` @ 817cff5 (main 1143f58 incl. #57850 + #57859/#57654 + local bf8 combine v2 + M3
knobs). M3-tokenized trace `longbook_56320`. 64 runs (`runs.csv` rows `nd_*`), zone profiles in `nd/profiles/`.
No hangs. Knobs: M3_MOE_W_NDSHARD (0 interleaved = default, 1 ND-sharded), M3_MOE_HYBRID_THRESHOLD (128).

## A. SP=2 (2,4), S8' (layers 8–15), 1d fabric, v1 ops — wall ms, median of 5

| cfg | W=2048: h = 0 / 16k / 141k / 549k | W=4096: h = 0 / 139k / 549k |
|---|---|---|
| N0 interleaved | 70.22 / 68.12 / 76.69 / 92.91 | 130.21 / 129.12 / 146.50 |
| N1 ND-sharded | 68.19 / 66.08 / 74.61 / 90.20 | 128.00 / 126.70 / 142.38 |
| N1H ND + hybrid 128 | 66.76 / 64.90 / 73.08 / 88.21 | 126.61 / 125.48 / 141.42 |

D2 (dense, W=4096 cold): N0 16.94, N1 17.37 ms (noise level).

Zones, sparse layer 3, W=2048, device ms (worst chip / mean of 8):

| zone | N0 | N1 | N1H |
|---|---|---|---|
| experts_mm | 2.390 / 1.578 | 2.078 / 1.348 | 1.807 / 1.176 (2 ops) |
| dispatch | 0.619 / 0.474 | 0.606 / 0.474 | 0.606 / 0.477 |
| combine | 1.304 / 0.650 | 1.222 / 0.623 | 1.191 / 0.527 |
| moe_reduce | 1.892 / 1.208 | 1.737 / 1.130 | 1.544 / 1.010 |
| mlp | 5.918 / 5.251 | 5.030 / 4.644 | 4.625 / 4.216 |
| layer | 8.731 / 8.671 | 8.470 / 8.081 | 8.185 / 7.655 |

Expert matmul (worst chip): ND −13%, ND + hybrid −24%. Layer: −3.0% / −6.3%. Wall: N1H −4.7…−5.1% (W=2048),
−2.8…−3.5% (W=4096), ~0.45 ms per sparse layer.

## B. SP=4 on the middle-rows 4x4 torus (all ND-sharded) — wall ms

M4A = carved (4,4) rows 0–3, 1d, v1. M4R = middle 4x4 torus, Ring, v1. M4E3 = M4R + dispatch v2 + combine v2.

| cfg | W=4096: h = 4k / 139k / 549k | W=8192: h = 8k / 139k / 549k |
|---|---|---|
| M4A | 72.06 / 83.34 / 106.88 | 131.40 / 144.86 / 170.35 |
| M4R | 74.04 / 82.22 / 95.96 | 136.36 / 146.46 / 162.41 |
| M4E3 | 64.35 / 72.27 / 83.42 | 120.37 / 130.35 / 144.11 |

h=0 (first, cold-compiling point): W=4096 80.30 / 82.52 / 72.63, W=8192 146.81 / 149.49 / 133.84. W=4096 rows use the
`_r2`/`_r3` repeats with a 4096 warm-up point: the first M4E3 W=4096 run fell into the "slow mode" after h=0
(93.56 / 107.02 ms at 139k / 549k), the repeat stayed fast. Torus fabric cost (M4R vs M4A) +2.7…+3.8% shallow, but Ring
helps at depth (−10% at 549k W=4096, −4.7% W=8192). New ops net of the torus (M4E3 vs M4R): −11…−13% everywhere.
Total vs M4A: −11…−22% (W=4096), −8…−15% (W=8192). No ND=0 control on SP=4.

The sub-torus opened with the default reliability mode (TORUSXY, degree 4 on all 16 chips); M3's hard-coded Linear TP
collectives did not hang. First M4R attempt failed only because TT_VISIBLE_DEVICES leaked into `tt-smi -glx_reset`
(POST_RESET on 16 devices); run_budget.sh now resets with `env -u TT_VISIBLE_DEVICES`.

## C. SP=8 (8x4), E3, W=4096 — ND A/B, two runs each, order reversed

| ND | h = 0 | h = 549k |
|---|---|---|
| 0 | 73.03, 74.85 | 75.47, 76.66 |
| 1 | 73.90, 73.41 | 77.81, 75.32 |

No effect beyond ±1.5% run-to-run (4 experts per chip: not weight-read bound).

## Chip-µs per token-layer, sparse (wall × chips / (W × 8))

| layout | h ≈ 0 | ~141k | 549k |
|---|---|---|---|
| SP=2 (2,4) N1H, W=4096 | 30.9 | 30.6 | 34.5 |
| SP=2 N0, W=4096 | 31.8 | 31.5 | 35.8 |
| SP=4 4x4 torus M4E3, W=8192 | 29.4 (h=8k) | 31.8 | 35.2 |
| SP=4 carved M4A, W=8192 | 32.1 | 35.4 | 41.6 |
| SP=8 E3, W=4096 | ~72 | – | ~74 |
| SP=8 E3, W=8192 (earlier, ND=0) | ~38 | ~41 | ~48 |

## Conclusions

- SP=2 with ND + hybrid and SP=4 on a real 4x4 torus with the v2 ops are tied on chip efficiency: SP=4 M4E3 5% ahead
  shallow, SP=2 N1H 2–4% ahead at depth.
- ND-sharded weights + the hybrid path are worth −24% on SP=2's expert matmul and ~−5% wall per stage.
- The v2 ops are worth −11…−13% on a real SP=4 torus, but only one 4x4 torus exists per galaxy.
- SP=8 stays least efficient (≈2× at W=4096, ≈1.3× at W=8192); ND sharding does not change it.
