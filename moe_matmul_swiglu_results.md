# MoE routed-expert FFN — fused SwiGLU results

## Setup

| | |
|---|---|
| Op | MoE routed-expert FFN (fused SwiGLU): `h = SiLU(x@Wg) * (x@Wu)`, `out = h@Wd` |
| Hardware | Blackhole P150 (card 0), single device |
| Cores | **88** (11x8) — all implementations measured on the same grid |
| Weights dtype | **bfloat4_b** (bfp4) |
| Activations | bfloat16, ROW_MAJOR |
| Hidden dim | 2048 |
| Buffer | 5120 tokens allocated; M = tokens actually routed to the expert |
| Metric | Device kernel duration, microseconds (lower is better) |

## emb 7168 [kimi, deepseek]

| M | Reference<br>on branch,<br>DRAM sharded weights | Fused SwiGLU<br>(Perf 2 / self-reflection) | SwiGLU vs ref |
|---:|---:|---:|:--:|
| 0 | 3.9 | 5.0 | 0.78x |
| 128 | 140.3 | 112.9 | **1.24x** |
| 256 | 159.2 | 155.3 | **1.02x** |
| 512 | 179.2 | 254.4 | 0.70x |
| 1024 | 308.1 | 456.6 | 0.67x |
| 2048 | 590.9 | 863.7 | 0.68x |
| 4096 | 1167.9 | 1679.0 | 0.70x |
| 5120 | 1472.5 | 2086.6 | 0.71x |

## emb 6144 [GLM]

| M | Reference<br>on branch,<br>DRAM sharded weights | Fused SwiGLU<br>(Perf 2 / self-reflection) | SwiGLU vs ref |
|---:|---:|---:|:--:|
| 0 | 4.0 | 5.1 | 0.79x |
| 128 | 127.3 | 95.1 | **1.34x** |
| 256 | 144.6 | 136.1 | **1.06x** |
| 512 | 159.7 | 230.4 | 0.69x |
| 1024 | 269.8 | 418.5 | 0.64x |
| 2048 | 517.0 | 797.2 | 0.65x |
| 4096 | 1022.2 | 1550.8 | 0.66x |
| 5120 | 1284.7 | 1924.8 | 0.67x |

## Takeaway

Fused SwiGLU wins at short sequences (M <= 256) and loses by ~1.4-1.5x from M = 512 up.
The crossover sits between M = 256 and M = 512 in both shapes.

---

## Measurement notes

- Reference: `UnifiedRoutedExpertFfnDeviceOperation`, branch `mbezulj/2607-routed-expert-dram`
  (commit `85fe7d883f2`), `w_ndshard` layout. Hardwired to an 11x8 grid. All 32 cases of
  `test_single_routed_expert_perf.py` passed against committed baselines.
- Fused SwiGLU: commit `89f6a84a28` (self-reflection). Op code is byte-identical to Perf 2
  (`92703b6a6a`) — both self-reflection commits touch only `self_reflection.md`.
- SwiGLU natively takes the full 11x10 = 110-core grid; it was clamped to 11x8 to match the
  reference. Cost of that clamp: ~2% (ratios 1.008-1.040 across M, both shapes).
- SwiGLU 88-core points are single dispatches (the op destabilises the device under the profiler
  at this grid, so each point ran in a fresh session after a board reset). Reference points and
  the 110-core SwiGLU points are multi-sample. Single vs multi-sample agreed to 0.1% where both
  were available, but small-M points carry a few percent of run-to-run noise either way.
- Measured 2026-08-03. Device times are DDR-speed dependent — re-baseline on other hardware.
