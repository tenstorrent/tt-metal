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


---

# The isolation, continued: mechanism confirmed, defect localised to ONE call, and the win measured

Three further experiments on 2026-08-11, same oracle, same worktree method.

## 1. The mechanism is phase *mismatch*, in either direction

`NORM_PHASE` parameterised so the sharded norm can be applied in prefill only, decode only, or both:

| cores | phase | decode PCC | prefill PCC | |
|---:|---|---:|---:|:--|
| 88 | **decode only** *(v3 as shipped)* | 0.9943716809625597 | 0.9986202564547553 | ❌ |
| 88 | **prefill only** | **0.9943996010696218** | 0.9988099561825600 | ❌ |
| 88 | **both** | **0.9996293363224806** | 0.9988099561825600 | ✅ |
| 11 | decode only | 0.9945729603715616 | 0.9986202564547553 | ❌ |
| 11 | prefill only | 0.9942125864050061 | 0.9978658021847042 | ❌ |
| 11 | both | 0.9943331194625922 | 0.9978658021847042 | ❌ |

**Sharding either phase alone costs ~5.2 × 10⁻³ — the same, in both directions.** Only *agreement* between the
phases recovers it. That is the mismatch hypothesis measured rather than inferred, and it needed the prefill-only
arm to be a test rather than a story: had prefill-only passed, the explanation would have been "sharding the decode
norm is harmful", which is a different defect with a different fix.

**And 88 needs both conditions.** Phase agreement alone is not enough — 11/22/44 fail at `both` too, because
≥ 2 tiles per core is genuinely less accurate. Only **88, with agreement**, clears the bar.

## 2. `full_attention` does not suffer it — which is why nobody noticed

| kind | cores | phase | decode PCC | |
|---|---:|---|---:|:--|
| full (layer 5, natural cache) | 8 | decode only *(v3 shipped this)* | 0.9997999978731844 | ✅ |
| full | 8 | both | 0.9997928485426096 | ✅ |
| full | 88 | decode only | **0.9997999978731844 — identical to 8 cores** | ✅ |
| full | 88 | both | 0.9997872958305739 | ✅ |

**The mismatch costs `full_attention` 7 × 10⁻⁶ and `sliding_attention` 5.2 × 10⁻³ — a factor of 740.** So the cell
shipped the mismatched guard on the kind where it is free and was vetoed on the kind where it is not, and the
`full` result that *passed* is itself sitting on the same latent defect with an accidental margin.

Note also that at `decode only` **full is identical at 8 and 88 cores to sixteen digits** — the op really is
grid-insensitive, exactly as [`NORM-GRID-SWEEP`](ADVCHAL-V3-NORM-GRID-SWEEP.md) measured. Sliding's grid
sensitivity comes from downstream discontinuity, not from the reduction.

## 3. One of eight norm sites carries the entire cost

`residual_norm_drop_index` leaves a single site interleaved. At 88 cores, decode-only (the failing config):

| dropped site | decode PCC | |
|---|---:|:--|
| none | 0.9943716809625597 | ❌ |
| **1** | **0.9996226684246500** | ✅ **recovered** |
| 2 | 0.9944871257200655 | ❌ |
| 3 | 0.9943649965713889 | ❌ |
| 4 | 0.9943964814206211 | ❌ |
| 5 | 0.9943716809625597 | ❌ |
| 6 | 0.9943819709107029 | ❌ |
| 7 | 0.9943574112542729 | ❌ |
| 8 | 0.9943952829240805 | ❌ |

**Site 1 is `self._rms_norm(hidden_states, self.weights.input_ln)`** — the first call in `_decode`, feeding
`_attention_decode`, i.e. the QKV projection and **the KV cache write**. Leaving that one site interleaved recovers
the whole 5.2 × 10⁻³; the other seven are numerically free.

That is the mechanism nailed shut: prefill filled the cache from an **interleaved** `input_ln`; decode computes the
new token's K/V from a **sharded** one; the new K/V is inconsistent with every cached entry. Nothing else in the
layer cares.

## 4. Both fixes measured, and there is no correctness/latency trade

`test_optimized_repeated_perf`, layer 0 sliding, batch 1, `decode_trace_host_ms`, 5 samples:

| config | decode PCC | bar | ms/layer | vs incumbent | µs/model (25 L) |
|---|---:|:--|---:|---:|---:|
| incumbent, interleaved | 0.9996280142258483 | ✅ | 1.835632 | — | — |
| **88 cores, phase = both, all 8 sites** | **0.9996293363224806** | ✅ | **1.610552** | **−12.27 %** | **−5,627** |
| **88 cores, decode-only, drop site 1** | **0.9996226684246500** | ✅ | **1.625231** | **−11.47 %** | **−5,260** |
| 11 cores, decode-only — *v3's vetoed best* | 0.9945729603715616 | ❌ | 1.597167 | −13.00 % | — |

**The best passing configuration is more accurate than the incumbent *and* 12.3 % faster.** `0.9996293` against the
incumbent's `0.9996280` — so the earlier framing of this as a *trade* ("spend 5 × 10⁻³ of PCC to buy 13 % of
latency") was wrong too: **nothing has to be spent.** The trade only appeared to exist because every configuration
v3 measured had the mismatch baked in.

**−5,627 µs/model recoverable on this cell**, against v2's booked −5,919 and v3's shipped 0. And the alternative fix
— drop site 1, keep the decode-only guard — is worth **−5,260 µs** and needs **no change to the prefill path at
all**, only a policy field.

## Revised actions

1. **Ship 88 cores with a phase-consistent guard** (`x.shape[-2] <= TILE_SIZE`, v2's condition). Measured:
   **−12.27 %/layer, PCC 0.9996293 — above the incumbent.** Worth −5,627 µs/model.
2. **Or, if the prefill path must not change: 88 cores, decode-only, `drop_index=1`.** −11.47 %/layer, PCC
   0.9996227, −5,260 µs/model, one policy field.
3. **Audit `full_attention`'s shipped config for the same latent mismatch.** It passes by a 7 × 10⁻⁶ margin that is
   accidental, not designed.
4. **The ladder must include the advised grid.** 88 was the advised value and the only passing rung, on both fixes.
5. **A phase-gated placement knob needs a cross-phase consistency assertion.** The KV cache carries the
   disagreement into the measurement, and nothing in the artefacts showed it. Generalise to phi and north-mini,
   which also gained decode-only knobs in v3.
