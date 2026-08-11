# The cell's real defect: v3 shards the norm in decode only, and reads a cache built without it

**One line of guard decides gemma-4-26B `-onA`'s sliding kind — −5,919 µs/model, 39 % of v2's corpus total.**

Measured on device 2026-08-11 with the model's own oracle (`test_optimized_real_weights_prefill_decode`, layer 0,
`shared_physical_cache=true`, real weights via `GEMMA4_RANGE_DOWNLOAD=1`, bar 0.995), in two isolated worktrees —
one at v2's tag `ad3ca71d8`, one at the v3 stage tag. Nothing varied but the core count and the guard.

| sliding cores | v2 tree | v3 tree + v2-style guard | v3 tree **as shipped** (decode-only) |
|---:|---:|---:|---:|
| 0 — interleaved | 0.9996280142258483 ✅ | — | 0.9996280142258483 ✅ |
| 11 | 0.9943331194625922 ❌ | **0.9943331194625922** ❌ | 0.9945729603715616 ❌ |
| 22 | 0.9942874693564726 ❌ | **0.9942874693564726** ❌ | 0.9944099795374435 ❌ |
| 44 | 0.9941146130802025 ❌ | **0.9941146130802025** ❌ | 0.9945729603715616 ❌ |
| **88** | **0.9996293363224806 ✅** | **0.9996293363224806 ✅** | 0.9943716809625597 ❌ |

*(v2's tree only admits multiples of 11 that divide 2816 evenly, so 11/22/44/88 is its whole legal ladder;
33/55/66/77 raise on its own width check.)*

## What this says

1. **The two trees are numerically identical to sixteen digits at every rung, once the guard matches.** So the
   difference I chased for two rounds — v2 leaves the norm weight interleaved, v3 reshards it — **does not
   matter**, exactly as the isolated op test said.
2. **The single difference that decides the cell is the guard.**

   | | v2 | v3 |
   |---|---|---|
   | condition | `x.shape[-2] > TILE_SIZE` → skip | `self._executing_decode` → else skip |
   | effect | sharded norm in **prefill *and* decode** | sharded norm in **decode only** |

   At seq_len 32 v2's shape test passes in prefill too; v3's phase test does not. So **v3 builds its KV cache with
   interleaved norms and then reads it with sharded ones** — the two phases disagree about how the residual stream
   is normalised.
3. **That inconsistency costs ~5 × 10⁻³ of layer PCC at every grid, and at 88 cores it turns a pass into a fail.**
   v3's shipped tree scores 0.9943717 at 88 where v2 scores 0.9996293.
4. **88 cores is genuinely special, and it is the only rung that passes.** At 2816/88 = 32 — **exactly one tile per
   core** — the reduction is a pure cross-core tree with no intra-core sequential accumulation, and it is *as
   accurate as interleaved* (0.9996293 vs 0.9996280) while being 13 % faster on the layer. Every other rung has
   ≥ 2 tiles per core, mixes intra- and cross-core accumulation, and lands at 0.9941–0.9946. Prefill shows the same
   shape: **0.9988100 at 88 against an unsharded 0.9986203**, versus 0.9978658 / 0.9981087 / 0.9981995 at
   11/22/44. **The ladder is non-monotonic, with the optimum at the maximum** — which the advisor advised.

## Why the isolated sweep missed it

[`NORM-GRID-SWEEP`](ADVCHAL-V3-NORM-GRID-SWEEP.md) ran 79 configurations and found the op grid-insensitive to
7.3 × 10⁻⁷. That was true, **and it used the decode-shaped input throughout — `[1,1,1,2816]`, one real row and
thirty-one of padding.** The effect lives in the **prefill** norm, `[1,32,2816]`, thirty-two real rows, which the
reconstruction never ran. → [`PITFALLS`](ADVCHAL-V3-ANALYST-PITFALLS.md) ERROR 15.

## So v3 lost this cell to two of its own defects, independently

| | |
|---|---|
| **88 was not on v3's ladder** — it swept 2/4/8/11/22/44 and stopped | the only passing rung was the one never measured |
| **v3's guard is decode-only** | even with 88 on the ladder, v3's tree scores 0.9943717 there and fails |

Either defect alone loses the cell. **The veto was correct for the tree it was applied to, and the tree was
wrong.** And because the verdict was hardcoded as `passed = kind == "full_attention"` with no oracle log committed,
none of this was visible from the artefacts.

## Actions

1. **Change v3's guard to shard the norm in prefill and decode** — v2's condition, phase-consistent. One line, and
   worth **−5,919 µs/model** on this cell alone.
2. **Put the advised grid on the ladder.** 88 was both the advised value and the only passing rung.
3. **Any placement knob gated on execution phase must have its cross-phase consistency asserted**, because the KV
   cache carries the disagreement into the measurement. Generalises beyond this op.
4. **`oracle_passed` computed from a parsed, provenanced oracle artefact** — unchanged, and it is what would have
   surfaced 1–2 during the run instead of a week later.
5. **Re-examine every other cell for the same phase asymmetry.** phi and north-mini also gained decode-only knobs
   in v3, and the same class of defect would be invisible in exactly the same way.
