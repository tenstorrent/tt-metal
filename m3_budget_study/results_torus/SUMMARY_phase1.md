# M3 prefill MoE on the full 8x4 BH galaxy — torus / Ring / combine v2, phase 1

Host bh-glx-120-b09u02, branch `vmelnykov/m3_moe_torus` @ 59a5723, whole 8x4 mesh (SP=8, TP=4, EP=32), bf4 experts,
untraced, tt-smi reset before every process. No hangs. Raw data: `runs.csv`, `logs/`, `profiles/*/zones.txt`.

Configs: A = 1d fabric, default descriptor, MoE Linear, combine v1, CCL Linear (today). B = 2d_torus_xy + torus
descriptor, MoE Linear, v1, CCL Linear. C = torus, MoE Ring, v1, CCL Ring. D = C + combine v2 (combine_fabric2d).
Extra: Cm = torus, MoE Ring, v1, CCL Linear; Cc = torus, MoE Linear, v1, CCL Ring; Dm = Cm + v2.

## Wall, S8 (layers 8–15), h = 0, B = 1 (median of 5; ms, ms/layer)

| cfg | W = 8192 | W = 16384 |
|---|---|---|
| A | 106.25 (13.28) | 196.66 (24.58) |
| B | 115.65 (14.46) | 214.82 (26.85) |
| C | ERROR | 207.93 (25.99) |
| D | ERROR | 216.09 (27.01) |
| Cm | 115.78 | 214.85 |
| Dm | 120.21 | 227.81 |

Spread per run 0.1–0.3 ms (C@16384 1.8 ms).

## Zones, sparse layer 3, h = 0, worst chip, device ms (ops in zone)

| zone | A 8192 | Cm 8192 | Dm 8192 (r1/r2) | A 5120 | C 5120 | D 5120 |
|---|---|---|---|---|---|---|
| dispatch | 2.58 | 2.88 | 2.72/2.74 | 1.68 | 1.91 | 1.80 |
| experts_mm | 2.92 | 2.92 | 2.92/2.93 | 1.76 | 1.92 | 1.93 |
| combine | 5.82 (1) | 6.52 (1) | 6.44/6.44 (3) | 3.54 | 4.11 | 3.94 (3) |
| of which combine_v2_prep (typecast + offsets all_gather) | – | – | 4.89 (2) | – | – | 3.14 (2) |
| moe_reduce | 7.46 | 8.51 | 6.82 | 4.22 | 4.92 | 4.29 |
| mlp | 10.28 (19) | 11.50 (19) | 11.30/11.33 (21) | 6.12 | 6.87 | 7.00 (21) |
| layer | 14.51 (73) | 15.54 (73) | 15.91/16.01 (75) | 9.38 | 9.93 | 9.96 (75) |

Each row is its own worst chip, so rows do not add. **The combine_fabric2d op alone is ≈ 1.5 ms vs 5.8–6.5 ms for
v1 combine**; v2's typecast over the worst-case dispatch buffer (4.9 ms) cancels it. Across-chip skew at A 8192:
experts_mm 2.92, combine 5.62, moe_reduce 7.03 ms — most combine/moe_reduce time is waiting on the busiest chip.

Host gap (profiled chunk wall / device / gap / %): A 8192 25.0 / 15.69 / 9.3 / 37%; Cm 8192 26.9 / 16.83 / 10.1 / 37%;
Dm 8192 38.7 / 17.47 / 21.2 / 55% (repeat 41.6 / 17.74 / 23.9 / 57%); A 5120 22.0 / 10.28 / 11.7 / 53%;
C 5120 22.5 / 10.70 / 11.8 / 52%; D 5120 23.4 / 10.77 / 12.6 / 54%. Ops per sparse layer: 73 (A/B/C), 75 (v2).
In Dm@8192 the op-to-op gaps sum to 21–25 ms over the chunk vs 8–9 ms for Cm, spread over every op.

## KV PCC (6 layers, longbook_5120 one-shot, chunk 5120; K / V / index_k)

A: L3 0.99972 / 0.99925 / 0.99975; L4 0.99900 / 0.99747 / 0.99927; L5 0.99902 / 0.99579 / 0.99933 (min 0.99579)
D: L3 0.99972 / 0.99925 / 0.99975; L4 0.99892 / 0.99715 / 0.99921; L5 0.99891 / 0.99534 / 0.99927 (min 0.99534)

## Failures

- `M3_CCL_TOPOLOGY=ring` at W = 8192 (runs C, D, Cc and the C profile at CHUNK = 8192):
  `TT_FATAL moe_grouped_topk_device_operation.cpp:36: scores.dtype() == FLOAT32 || scores.dtype() == BFLOAT16`, in the
  router (tt/mlp.py:237). Ring CCLs work at 5120 and 16384. The C profile wrapper printed exit=0 anyway; its partial
  CSV is parked in `profiles/*_INVALID_TTFATAL`.
- Torus init logs 8 non-fatal "4 eth channels, but only 2 routing planes" warnings.

## Interpretation

1. The torus fabric works on this galaxy (one topology match, degree 4 on every chip).
2. Opening the 2D torus costs ≈ 9% with unchanged Linear algorithms (B vs A). Likely cause: prefill dispatch turns
   sparse multicast off on any non-1D fabric.
3. Ring for MoE dispatch/combine alone gains nothing (Cm = B); the Ring CCL gain at 16384 comes from the legacy CCLs,
   which break at 8192.
4. combine v2 is a net loss today (+4.4 ms at 8192, +8…13 ms at 16384) although the op itself is ~4 ms cheaper than v1:
   the bfp8→bf16 typecast over the worst-case buffer and the offsets all_gather eat it, and v2 widens op-to-op gaps.
5. The 8x4 wall is not host-launch bound in steady state (wall scales 1.85× with 2× W; wall per layer ≤ device per
   layer); the single-layer profile's host gap is a fixed per-forward cost. EP load skew dominates combine/moe_reduce.
