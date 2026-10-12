# GLM 5.3 prefill MoE: fp8 wire with native 1×128 scales — plan

Port the K3 fp8 dispatch/combine wire (`K3_MOE_FP8_WIRE_PLAN.md`, branch `pjosipovic/k3-moe-fp8-wire`) to GLM 5.3,
with GLM's native activation scaling: **one scale per 128 values** (DeepSeek-style 1×128 blocks, matching the
checkpoint's `weight_block_size [128, 128]` / dynamic activation quant), instead of K3's 1×32.

## 0. What is different from K3

| | K3 (done) | GLM 5.3 |
|---|---|---|
| emb (routed width on the wire) | 3584 after the latent projection | **6144**, no latent projection |
| experts / top-k / per chip on 8×4 | 896 / 16 / 28 | **256 / 8 / 8** |
| routed expert | SiTU-GLU, 7168→... | 6144 × 2048, SiLU; `ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD = 320` (unified kernel + `moe_fused_swiglu`) |
| shared expert | split across two overlap windows | 1 shared expert, hidden 2048 |
| scale block | 1×32 (H/32 = 112 scales/row) | **1×128 (H/128 = 48 scales/row)** |
| packed row | 3584 + 448 = 4032 B | **6144 + 192 = 6336 B** (bf16 row: 12288 B → 52%) |
| fabric payload | `KimiK3Config.FABRIC_PAYLOAD_SIZE` | `GLM53Config.FABRIC_PAYLOAD_SIZE = 6144` < 6336 → **must raise** (≥ 6336 + metadata, e.g. 7168) or every row splits into 2 packets |

Scale format: keep **power-of-two fp32 scales** (as K3): exact dequant by exponent arithmetic, 1/scale exact in bf16,
and every K3 kernel trick (pow2 via integer exponent ops, scale→bf16 by taking the high half) carries over. Option to
evaluate: exact amax/448 fp32 scales (DeepSeek reference) — better numerics, but costs a real divide/multiply; only if
pow2 1×128 misses the PCC bar.


## Status (2026-10-12, 2×4 LoudBox, GLM dims, 256 experts = 32 per chip — the device gate requires exactly 256)

| | bf16 wire | fp8 1×128 | Δ |
|---|---|---|---|
| **traced T_moe p50** (payload 7168) | **4.507 ms** | **4.138 ms** (RNE: 4.164) | **−0.35 ms (−7.8%)** |
| dispatch | 358 µs | 176 µs (315 µs at payload 6144: rows split) | −182 |
| combine | 524 µs | 127 µs | −397 |
| post_combine_reduce | 97 µs | 71 µs (DQ2 fused) | −26 |
| routed expert | 2773 µs (bf16 rows, tilizes inside) | 2731 µs (bf8_b tiles from DQ1) | −42 |
| Q1 / DQ1 / Q2 | — | 39 / 71 / 70 µs | +180 |
| final_output PCC (`test_glm_moe` 2×4 pcc) | 0.988194 | 0.986259 truncation, **0.987805 RNE** (default for 1×128) | −0.0004 |

Done:
- P3 DQ1 `tile_output` for 1×64 / 1×128 (one scale per scale_block_w/32 tiles; unit G = 24 at H = 6144): 284 → 71 µs.
- P4 fused DQ2 for 1×64 / 1×128 in post_combine_reduce (removes a 406 µs standalone dequant; PCR 174 → 71 µs).
- P1 `FAST_BLOCK4`: 1×128 on the fast quantize path (per-tile SFPU row max of |x|, fold of the 4 tiles' maxima + pow2
  in DEST, one 1/scale tile per chunk); Q2 88 → 70 µs; Q1 unchanged at 39 µs (near its byte floor).
- P5 / G5: fp8 1×128 with the 320-token hybrid threshold passes PCC (0.986425 trunc).
- P6: 1×32 and 1×128 give identical PCC on this test (same pow2 buckets); the gap to bf16 was truncation — RNE closes
  it to 0.0004 for ~3 µs per Q (`MOE_FP8_RNE=auto` → on for 1×128).
- Combine fix found on the way: the pipelined sender sent a row into one EDM slot, which overruns a slot when the row is
  wider than one packet (GLM bf16, 12 KB) — hung the GLM bf16 PCC test; now pipelining only applies to single-packet rows.

Open:
- **Fabric payload (model-level decision):** `GLM53Config.FABRIC_PAYLOAD_SIZE = 6144` < 6336 B fp8 row. fp8 still works
  (two packets per row) but dispatch costs 315 vs 176 µs. The config is shared with migration code ("must stay in sync"),
  so it was not changed here; set it to ≥ 6400 (e.g. 7168, as DSv3) when the GLM model enables the fp8 wire.
- Galaxy-equivalent load cannot be emulated on the LB (gate needs 256 experts → 32/chip vs 8/chip on 8×4); G6 on Galaxy.

## 1. Where the code stands for block = 128 today (from a code survey; not run)

- Op validation already accepts `scale_block_w = 128` in `per_token_cast_to_fp8` and `per_token_cast_back`; TtMoe
  asserts `fp8_scale_block in (0, 32, 64, 128)`. Nothing in Q1/DQ1/Q2/DQ2 requires latent, so the wire runs in
  principle with `MOE_FP8_SCALE_BLOCK=128` — but only on the **slow generic paths**:
  - Q (`per_token_cast_to_fp8`): `FAST_1X32`, `Q_SFPU_AMAX`, count-aware units, write groups are all 1×32-only.
  - DQ1: `tile_output` (bf8_b tiles straight into the expert) is 1×32-only → 128 produces bf16 rows (expert tilizes).
  - DQ2: `post_combine_reduce` fuses only 0 / 32 (hard-codes the `H + H/8` row) → 128 runs a standalone
    `per_token_cast_back` (K3 v1 measured that at ~546 µs — must not ship).
  - Q1-before-gather / `row_segments` are latent-only; GLM has no latent gather, so plain Q1 before dispatch.
- The combine changes (4-row ROW_MAJOR batches, pipelined sender) and post_combine_reduce L1 source are generic and
  already apply to GLM (bf16 and fp8).
- **Hybrid-threshold gap:** with `hybrid_token_threshold = 320`, `x_row_scales` reaches only the unified routed-expert
  kernel, not `moe_fused_swiglu` (`tt_routed_expert.py:732-768`). Block-scale DQ1 is unaffected (it produces bf16/bf8
  first); per-row fp8 + fused DQ1 would feed raw e4m3 to the fused half. `test_glm_moe` runs under the DSv3 variant
  (threshold None), so it would not catch this.

## 2. Goals

| Goal | Target |
|---|---|
| **G0** baseline | GLM bf16 traced T_moe and per-op times on the LB at Galaxy-equivalent load (8 experts/chip: `num_routed_experts=64` on 2×4) and on 8×4 |
| **G1 (headline)** | fp8 1×128 traced T_moe < bf16 at the same load; final_output PCC ≥ bf16 − 0.001 (set the absolute bar from G0) |
| **G2** Q1 + Q2 | ≤ 1.2× a plain DRAM copy of the same bytes each (K3 lesson: these ops end up NoC-bound) |
| **G3** DQ1 | bf8_b tile output for 1×128 (no bf16 round trip); ≤ K3-equivalent per byte |
| **G4** DQ2 | fused into post_combine_reduce for 1×128; PCR fp8 ≤ PCR bf16 |
| **G5** correctness | GLM variant with the 320 hybrid threshold on, fp8 wire, PCC passes |
| **G6** Galaxy | 8×4 numbers (PCC + traced), as for K3 |

## 3. Work items (in order)

**P0 — baseline and harness (no kernel work).**
- Scratch perf test like `test_scratch_k3_moe_perf_lb.py` for GLM: `K3_NUM_EXPERTS`-style override (64 on LB), trace,
  `--profile`, payload override. Run under the **GLM variant** (threshold 320), not DSv3.
- Fabric payload: sweep 6144 (split) vs 7168 for bf16 and fp8; expect fp8 to need ≥ 6336 + metadata in one packet.
- Get G0 numbers; run `MOE_FP8_SCALE_BLOCK=128` once on the generic paths to see the starting fp8 cost per op.
- Always confirm the wire env vars took effect (fp8 PCC must differ from bf16) — see the K3 env-split lesson.

**P1 — Q for 1×128 (`FAST_1X128`, both Q1 and count-aware Q2).**
- Generalize the FAST_1X32 compute: one scale per 4 tiles. Per tile, the fused SFPU pass already produces the row
  max of |x| in registers; extend it to keep a running max across the 4 tiles of a block (store per-tile maxima to a
  DEST scratch column and fold, like `max_first_columns_across_tiles`, or fold in registers if LREGs allow), then
  one pow2 per row group; multiply all 4 tiles by the same column-broadcast 1/scale tile.
- Fewer scales per row (48 vs 112): writer staging shrinks 4×; packed row = 6144 + 192.
- Keep the K3 Q2 machinery: valid-row multicast, even split, write groups (re-tune: GLM rows are 6144 B, so 2-chunk
  units are 4 tiles of 128 = 512 B runs), L1 output where it fits.

**P2 — dispatch.**
- One packet per token-expert with fused metadata (`DISPATCH_FUSED_META` needs P + M ≤ payload).
- No pre-gather Q1 (no latent); Q1 runs on each chip's 640×6144 tokens before dispatch.

**P3 — DQ1 tile output for 1×128.**
- `per_token_cast_back(tile_output=True)` with `sbw = 128`: each scale covers 4 consecutive tiles, so the writer
  builds one col-0 scale tile per 4 output tiles (or the compute reuses one scale tile for 4 multiplies) — less
  staging than K3. Keep the K3 fixes: valid-row multicast, gathered plain-load staging.
- Unit size: H = 6144 → Wt = 192; pick G as a multiple of 4 dividing 192 (24 / 32 / 48) and re-check work balance
  (K3: ~2.7 units/core was too coarse).

**P4 — fused DQ2 for 1×128 in post_combine_reduce.**
- Generalize the fp8_block reader/compute from 1×32 to 1×128: the row is `H + 4·H/128`; each scale applies to 4
  tiles of a part; per-part scale windows shrink. Keep the gathered staging and the L1 source.

**P5 — routed expert hybrid path.**
- Block-scale DQ1 feeds bf8/bf16 to both halves, so no change expected; add the GLM-variant fp8 PCC test (G5) to prove
  it. Per-row fp8 (block 0) + fused DQ1 stays unsupported with the threshold unless `moe_fused_swiglu` gets scales.

**P6 — numerics study (parallel to P1).**
- On host / reference: GLM routed and final PCC for per-row, 1×128, 1×32, pow2 vs exact scales, on real GLM
  activations if available (dynamic quant, so no checkpoint scales are involved). Pick 1×128 pow2 unless it misses G1's
  PCC bar.

**P7 — Galaxy (G6).** Handoff README like `K3_MOE_FP8_GALAXY_HANDOFF.md`.

## 4. Expected savings (rough, to be replaced by G0)

- Dispatch / combine bytes: 6336 vs 12288 B per row (−48%). K3 showed time does not follow bytes once the ops are
  per-row / imbalance bound — measure per-core spans first (combine fixes already in).
- New costs: Q1 (640×6144 per chip), DQ1, Q2 (count-aware, valid rows only); with GLM's 8 experts/chip the routed row
  count per chip is smaller than K3's, so Q2/DQ1 should be cheaper per chip than K3's 69 / 73 µs, but rows are 1.7×
  wider.
- PCR reads half the bytes (K3: 196 → 101 µs with fused DQ2 + L1 source).

## 5. K3 lessons to apply from day one

- Profile per core (kernel start/end per RISC, DeviceZoneScopedN in the hot loops) before optimizing; K3's biggest
  wins were imbalance (combine untilizer batches, Q2/DQ1 startup table reads) and per-row/per-element software cost
  (volatile staging loops, flush-blocking sends), not bandwidth.
- Routing tables: read once per chip and multicast (240 kernels reading the same pages cost 8–15 µs).
- L1 placement: L1-interleaved intermediates relieve DRAM-endpoint NoC congestion, but check the fit against every
  concurrent kernel's CBs (shared expert, routed expert); only one of Q2-output / combine-output fit on K3.
- Verify env knobs reach the kernel (log line / `-D` flags); zsh needs `${=var}` to split multi-var configs.
