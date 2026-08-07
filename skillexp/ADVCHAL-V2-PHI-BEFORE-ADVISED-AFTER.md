# phi-3.5-mini FN — BEFORE / ADVISED / AFTER, per op

Every op in one table, from the raw artefacts: the **original** placement, what the **advisor advised**, what
**shipped**, the stage's own label, and whether the shipped result actually follows the advice.

## Sources joined

| column | comes from |
|---|---|
| `µs`, `BEFORE cores·mem` | `profiles/dense_incumbent.csv` (`Device Time`, `Cores`, `Input 0 Memory`) |
| `ADVISOR ADVISED` | `shard_advise/dense/report.json` → `ops[].layout`, plus `advised_cores` from the reconciliation |
| `AFTER cores·mem` | `profiles/dense_winner.csv` |
| `stage label` | `reconciliation_dense.json` → `disagreements[].bucket` — **the stage's own verdict** |
| `in/out shape` | the executed ttnn trace (`phi_AFTER_rope_on.txt`) |

`intlv` = interleaved, `shrd` = sharded, `L1_H` = L1 height-sharded, `L1_W` = L1 width-sharded.
`· 32c` after an advised layout is the reconciliation's `advised_cores` field — **note it often disagrees with
the number in the layout string** (`1x96 · 96c`, `32x1 · 32c`); that mismatch is the grid-vs-DS-family parsing
quirk recorded as action C5.

## The answer to "does it follow the advice?"

| actually follows advice? | rows | `why` codes |
|---|---|---|
| — no advice exists (`boundary` 17 + `untraced` 8) | 25 | `—`, `G·b` |
| **no** | 12 | `O` ×9, `G` ×2, `L` ×1 |
| **undecidable** — the op↔advice pairing is a positional guess | 7 | `P` ×4, `O·P` ×2, `T·P` ×1 |
| **buffer only** — got L1, not the sharding | 6 | `U` ×2, `U·P` ×4 |
| **yes** | 4 | `W` ×4 |
| **family only** — right space and layout family, wrong grid | 2 | `O` ×1, `D` ×1 |

Of the 31 ops that carry advice at all: **4 follow it, 6 take the buffer type only, 2 match the family but not
the grid, 12 do not follow it, and for 7 the mapping is too weak to say.**

**And there is one dominant reason.** Nine of the 12 "no" rows, plus one "family only" and two "undecidable",
are the *same single decision*: chains `dense:0` and `dense:11` were screened, **measured 7.6 % faster
standalone and 13.3 % faster combined with the rope win**, and then rejected because a differential PCC came
back `0.9999910667` against a `0.999999` bar. **Twelve of the 27 `chain` ops were left unimproved by one oracle
call, not by anything about the advice.** See code `O` in the legend and §3.4/E1 in the other docs.

### The 6 "buffer only" rows are the interesting ones

`neg` ×2, `concat` ×2, `multiply` ×2 — the advisor advised `l1/height_sharded/32x1` and the cell shipped
`l1/interleaved`. So the shipped win took the **buffer type** (DRAM→L1, the advisor's first-ranked criterion)
and left the **sharding** half unimplemented. **Measured below: that half cannot be implemented at all** — so
this is not a shortcut the cell took, it is the only legal placement.
### Measured: what happens if you *do* implement the advised sharding — it is the fastest form

⚠ **An earlier version of this section claimed the advised sharding could not run. That was wrong, and the
error was mine** — I used shard `(32,48)`; the advisor specified `(32,64)`. Corrected below.

The advised layout, from `final_ir.mlir` — the authoritative artefact, because `report.json` prints no shard
shape at all and truncates the core range:

| | shard shape | cores |
|---|---|---|
| 48-wide `neg` and the halves — `#ttnn_layout24` | `memref<1x2x tile<32x32,bf16>>` = **(32, 64)**, tile-aligned | **32** — `[(0,0)-(10,1), (0,2)-(9,2)]` = 22 + 10 |
| 96-wide `concat` out, `multiply`, `add` — `#ttnn_layout16` | `memref<1x3x tile<32x32,bf16>>` = **(32, 96)** | **32**, same set |

**Isolated single-op test of that exact config, one op at a time on device:**

| op | config | result |
|---|---|---|
| `ttnn.neg` (1,32,32,48) | shard (32,64), 32 cores | **OK** |
| `ttnn.concat` 2×(…,48) → (…,96) | shard (32,96), 32 cores | **OK** — the case I claimed was impossible |
| `ttnn.multiply` (…,96) | shard (32,96), 32 cores | **OK** |
| `ttnn.add` (…,96) | shard (32,96), 32 cores | **OK** |
| `ttnn.slice` → (…,48) | shard (32,64), 32 cores | **OK** |
| *control:* `ttnn.neg` shard **(32,48)** — my old probe | | **FAIL** `TT_FATAL tensor_layout.cpp:162` |

**And the whole chain implemented verbatim**, with the cell's own unmodified `harness.py`, one fresh process per
form, WARMUP 10 / REPEATS 5 / ITERS 50:

| form | median ms | Δ vs incumbent | differential PCC | repeats span |
|---|---|---|---|---|
| frozen incumbent | 0.807535 | — | — | 0.807144 – 0.808512 |
| **what this cell shipped** — L1 interleaved | 0.768104 | **−4.88 %** | 1.0 | 0.767374 – 0.768156 |
| what phi B shipped — sharded `multiply`/`add` | 0.751277 | **−6.97 %** | 1.0 | 0.751010 – 0.752433 |
| **the advisor's IR, verbatim** | **0.723320** | **−10.43 %** | **1.0** | 0.722631 – 0.723988 |

Strict non-overlap between all three candidates; all bit-identical to the incumbent at the cell's own 0.999999
bar, so the oracle never blocked the sharding — nothing did, it was simply never tried.

**So the 6 "buffer only" rows are not the only legal option. They are an unproven deviation that cost
5.55 pp ≈ 1.43 ms/model.** The `follows advice?` column below still describes what shipped; what changed is the
*reason* — not "impossible", but "never tried".

Phi's own `_apply_rope` comment explains why the widths are awkward, and it remains true — it just does not
prevent the sharding:

> *`ttnn.experimental.rotary_embedding` requires a width divisible by 64, whereas Phi-3.5's 96-wide heads split
> at 48. The explicit topology is the exact HF operation and has no host fallback.*

→ [`ADVICE-FAITHFULNESS`](ADVCHAL-V2-ADVICE-FAITHFULNESS.md) §4–§5,
[`COUNTERFACTUALS`](ADVCHAL-V2-COUNTERFACTUALS.md) §E25

## `chain` and the other stage labels — what they mean

The `stage label` column is `reconciliation_dense.json` → `disagreements[].bucket`. The five values are assigned
by `reconcile.py` (`scripts/reconcile.py:429-460`) in this order, first match wins:

| label | assigned when | what it means for you |
|---|---|---|
| `boundary` | the device op is a **movement op** (`ShardedToInterleaved`, `Tilize`, `Permute`, …) | The advisor lists conversions in `reshards[]`, not `ops[]`, so **no per-op advice exists**. Classified first, otherwise these fall through to `untraced` and the ranking loses its input. |
| `untraced` | the op is in the profile but **absent from the advisor's graph** | No advice to follow. |
| `dram_resident` | the advisor put the op in **DRAM** while the shipped op is sharded | That *is* advice, and it disagrees. |
| `agrees_with_shipped` | advised cores **==** shipped cores, **or** both sides are DRAM-sharded matmuls | "Nothing to screen, we already do what the advice says." |
| **`chain`** | **everything else: the advisor wants it in L1, and the advised core count differs from the shipped one** | **This op is part of a candidate the stage considers screenable.** |

**So `chain` = "the advisor and the shipped code disagree about this op's placement, in L1, and that disagreement
is actionable."** Consecutive `chain` ops are then grouped into a *chain* — a maximal run of L1-resident
disagreements, broken by any conversion, DRAM placement or untraced op — and **the chain, not the op, is the
unit the stage measures and ships**. That is why the ids look like `dense:0`, `dense:4`: `<layer_kind>:<index>`.
Ids of the form `dense:b14` are *boundary-derived* chains — a lone conversion the advisor said not to do,
promoted to a candidate on its own (`reconcile.py:506-511`).

Each chain carries its own measured verdict, and **that verdict is the reason column below**:

| chain | ops | µs | verdict | measured ms |
|---|---|---|---|---|
| `dense:1` `dense:2` `dense:6` `dense:7` | `slice_static` ×4 | 1.3–6.8 | **kept** | 0.79345 / 0.79263 |
| `dense:3` `dense:8` | `neg` | 1.5–2.1 | **kept** | 0.79345 / 0.79263 |
| `dense:4` `dense:9` | `concat`, `multiply` | 19.8–20.1 | **kept** | 0.79345 / 0.79263 |
| `dense:5` `dense:10` | `multiply`, `add` | 6.8–7.0 | **kept** | 0.79345 / 0.79263 |
| `dense:b14` `dense:b29` | (boundary) | 0.0 | **kept** | 0.79345 / 0.79263 |
| `dense:0` | `rms_norm`, `linear`, `nlp_create_qkv_heads_decode`, `embedding` ×2 | **178.4** | **rejected** | **0.745905** |
| `dense:11` | `linear`, `add`, `rms_norm`, `linear`, `slice_static` ×2, `multiply` | **196.1** | **rejected** | **0.745905** |
| `dense:b43` | (the `InterleavedToSharded` between SDPA and concat-heads) | 0.0 | **rejected** | 0.807152 |
| `dense:12` | `add` | 2.5 | **below_threshold** | — |
| `dense:b38` `dense:b39` | (boundary) | 0.0 | **below_threshold** | — |

## The `why` legend

Every code below is read off an artefact, not inferred. Where I could not establish a reason I say so.

| code | reason | evidence |
|---|---|---|
| **O** | **Oracle veto.** The op is in chain `dense:0` or `dense:11`. Both were screened together as `advisor_norm_cores=11` and **measured faster** — 0.745905 ms against a 0.807152 ms incumbent (−7.6 %), and 0.700267 ms combined with the rope win (−13.3 %). The cell rejected them anyway: the differential real-weight oracle scored **0.9999910667** against a bar of **0.999999**. | `final.json` → `rejected_knobs.advisor_norm_cores` (`"reason": "real-weight differential PCC moved to 0.9999910666979231; placement change rejected"`), `oracle_combined.json` (`oracle_pcc_bar: 0.999999`, `oracle_passed: false`), `measurements/norm_11c.json`, `measurements/rope_l1_query_key_norm_11c.json`, `reconciliation_dense.json` → `chains[].verdict` |
| **W** | **Shipped win.** Chain `kept`. The advice was `l1/interleaved`, which specifies no grid, so the advice is *fully* met. | `chains[].verdict = kept`, `profiles/dense_winner.csv` |
| **U** | **Never tried.** Chain `kept`, but only the L1 half — the advised `l1/height_sharded/32x1`, shard (32,64)/(32,96) on 32 cores, was not attempted by the cell. It **runs**, it is **bit-identical**, and it is **−10.43 % against the −4.88 % this cell shipped**. *(This code read "Unreachable" in an earlier revision, on the strength of my own mis-shaped probe. Retracted.)* | `exp-rope-faithful/rope_advisor_full.json`, `oracle_modes.json`, the isolated single-op test, `final_ir.mlir` `#ttnn_layout24`/`#ttnn_layout16` |
| **G** | **Hard error.** `advisor_sdpa_concat_l1` shipped **default-false** — `TT_FATAL: Sharded output not supported for GQA`. The paired boundary chain `dense:b43` was **rejected** and fell back to the incumbent time exactly (0.807152). | `final.json` → `rejected_knobs.advisor_sdpa_concat_l1`, `chains[dense:b43].hard_error`, `measure_sdpa_concat_l1.log` |
| **T** | **Below the noise floor.** Chain `dense:12` is 2.461 µs of a 725 µs window, against a harness noise floor of **1.064 µs**. Not measurable, so never screened. | `chains[dense:12].verdict = below_threshold`, `incumbent.json` → `noise_floor_ms: 0.0010642` |
| **L** | **Label defect, not a placement decision.** `agrees_with_shipped` is decided by `advised_cores == shipped_cores` — **`reconcile.py` never compares the memory space**. Advised `l1/…/1c`, shipped `1 core DRAM interleaved`: cores match, so it is labelled "agrees" while the buffer type disagrees. | `reconcile.py:446-448`, row 4 `pair_confidence: name` |
| **D** | **DS-family match.** `agrees_with_shipped` via the second branch: the shipped op has `DRAM Sharded = True` and the advice's `program_config` is DRAM-sharded, so **the grid is deliberately not compared** — 88 advised against 12 shipped. | `reconcile.py:446-447`, `profiles/dense_incumbent.csv` row 54 (`dram_sharded=True`, `cores=12`) |
| **P** | **The op↔advice pairing is a guess.** `pair_confidence: position`, not `name` — the tool's own documented limitation ("*a positional pair is a guess*"). For these rows **I cannot establish whether the advice was followed**, and I am not going to claim one. | `reconciliation_dense.json` → `disagreements[].pair_confidence`, `limitations[0]` |
| **·b** | this boundary row is the one the `G` knob targeted | `chains[dense:b43]` |
| **—** | no advice exists for this op (`boundary` or `untraced`) | see the label table above |

`U·P` / `O·P` / `T·P` mean the reason holds but the individual row's pairing is positional. For the six `U`
rows that is not a problem: the two `concat` rows are **name**-paired, and my measurement covers the rope body
as a whole, so the group conclusion stands even where a single row's pairing is a guess.

### Which rows are positional

Eleven of the 31 paired rows are positional guesses: **13, 19, 21, 22, 28, 34, 36, 37, 47, 52, 56**.
Twenty are name-paired: 1, 2, 3, 4, 5, 6, 9, 11, 18, 24, 26, 33, 43, 45, 46, 48, 49, 50, 51, 54.

## Known artefact: the positional pairings

Row 19 pairs `multiply` with a `PermuteDeviceOperation` — the reconciliation's own pairing being wrong, not a
real mapping. It is **not an isolated case**: 11 of the 31 paired rows are positional, i.e. the tool's own
documented guess ("*a positional pair is a guess*"), and they are marked `P` in the `why` column. This is the
same soft-pairing limitation qwen FN and nm B both caught against the IR.

---

| # | op | device op | µs | in shape | out shape | BEFORE cores·mem | ADVISOR ADVISED | AFTER cores·mem | stage label | follows advice? | why |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `rms_norm` | `LayerNorm` | 44.3 | (1, 1, 32, 3072) interleaved | (1, 1, 32, 3072) interleaved | 1·DRAM·intlv | `l1/block_sharded/1x11 · 11c` | 1·DRAM·intlv | `chain` | no | O |
| 2 | `linear` | `Matmul` | 72.6 | (1, 1, 32, 3072) interleaved | (1, 1, 32, 9216) interleaved | 96·DRAM·intlv | `l1/width_sharded/1x96 · 96c` | 96·DRAM·intlv | `chain` | no | O |
| 3 | `nlp_create_qkv_heads_decode` | `NLPCreateQKVHeadsDecode` | 56.6 | — | — | 32·DRAM·intlv | `l1/height_sharded/32x1 · 32c` | 32·DRAM·intlv | `chain` | no | O |
| 4 | `typecast` | `Typecast` | 0.9 | (32,) interleaved | — | 1·DRAM·intlv | `l1/height_sharded/1x1 · 1c` | 1·DRAM·intlv | `agrees_with_shipped` | no | L |
| 5 | `embedding` | `Embeddings` | 2.5 | (32,) interleaved | (32, 96) interleaved | 1·DRAM·intlv | `l1/interleaved/10x11` | 1·DRAM·intlv | `chain` | no | O |
| 6 | `embedding` | `Embeddings` | 2.5 | (32,) interleaved | (32, 96) interleaved | 1·DRAM·intlv | `l1/interleaved/10x11` | 1·DRAM·intlv | `chain` | no | O |
| 7 | `*(conversion)*` | `ShardedToInterleaved` | 1.5 | — | — | 32·L1_H·shrd | `—` | 32·L1_H·shrd | `boundary` | — | — |
| 8 | `*(conversion)*` | `ShardedToInterleaved` | 1.4 | — | — | 32·L1_H·shrd | `—` | 32·L1_H·shrd | `boundary` | — | — |
| 9 | `slice_static` | `Slice` | 1.6 | (1, 32, 32, 96) interleaved | — | 64·DRAM·intlv | `l1/interleaved/10x11` | 64·L1·intlv | `chain` | **yes** | W |
| 10 | `*(conversion)*` | `Untilize` | 4.4 | — | — | 32·DRAM·intlv | `—` | 32·L1·intlv | `boundary` | — | — |
| 11 | `slice_static` | `Slice` | 6.8 | (1, 32, 32, 96) interleaved | — | 110·DRAM·intlv | `l1/interleaved/10x11` | 110·L1·intlv | `chain` | **yes** | W |
| 12 | `*(conversion)*` | `TilizeWithValPadding` | 4.3 | — | — | 32·DRAM·intlv | `—` | 32·L1·intlv | `boundary` | — | — |
| 13 | `neg` | `Unary` | 2.1 | (1, 32, 32, 48) interleaved | — | 110·DRAM·intlv | `l1/height_sharded/32x1 · 32c` | 110·L1·intlv | `chain` | **buffer only** | U·P |
| 14 | `*(conversion)*` | `UntilizeWithUnpadding` | 10.2 | — | — | 32·DRAM·intlv | `—` | 32·L1·intlv | `boundary` | — | — |
| 15 | `*(conversion)*` | `UntilizeWithUnpadding` | 10.1 | — | — | 32·DRAM·intlv | `—` | 32·L1·intlv | `boundary` | — | — |
| 16 | `*(conversion)*` | `Permute` | 5.8 | — | — | 64·DRAM·intlv | `—` | **gone**·— | `untraced` | — | — |
| 17 | `*(conversion)*` | `Permute` | 5.9 | — | — | 64·DRAM·intlv | `—` | **gone**·— | `untraced` | — | — |
| 18 | `concat` | `Concat` | 12.2 | — | (1, 32, 32, 96) interleaved | 110·DRAM·intlv | `l1/height_sharded/32x1 · 32c` | 110·L1·intlv | `chain` | **buffer only** | U |
| 19 | `multiply` | `Permute` | 7.9 | — | — | 96·DRAM·intlv | `l1/height_sharded/32x1 · 32c` | **gone**·— | `chain` | undecidable | P |
| 20 | `*(conversion)*` | `TilizeWithValPadding` | 3.6 | — | — | 32·DRAM·intlv | `—` | 32·DRAM·intlv | `boundary` | — | — |
| 21 | `multiply` | `BinaryNg` | 3.4 | (1, 32, 32, 96) interleaved | — | 110·DRAM·intlv | `l1/height_sharded/32x1 · 32c` | 110·L1·intlv | `chain` | **buffer only** | U·P |
| 22 | `add` | `BinaryNg` | 3.4 | (1, 32, 32, 96) interleaved | — | 110·DRAM·intlv | `l1/height_sharded/32x1 · 32c` | 110·DRAM·intlv | `chain` | undecidable | P |
| 23 | `*(conversion)*` | `BinaryNg` | 2.1 | (1, 32, 32, 96) interleaved | — | 110·DRAM·intlv | `—` | 110·L1·intlv | `untraced` | — | — |
| 24 | `slice_static` | `Slice` | 1.3 | (1, 32, 32, 96) interleaved | — | 64·DRAM·intlv | `l1/interleaved/10x11` | 64·L1·intlv | `chain` | **yes** | W |
| 25 | `*(conversion)*` | `Untilize` | 3.6 | — | — | 32·DRAM·intlv | `—` | 32·L1·intlv | `boundary` | — | — |
| 26 | `slice_static` | `Slice` | 6.6 | (1, 32, 32, 96) interleaved | — | 110·DRAM·intlv | `l1/interleaved/10x11` | 110·L1·intlv | `chain` | **yes** | W |
| 27 | `*(conversion)*` | `TilizeWithValPadding` | 3.9 | — | — | 32·DRAM·intlv | `—` | 32·L1·intlv | `boundary` | — | — |
| 28 | `neg` | `Unary` | 1.5 | (1, 32, 32, 48) interleaved | — | 110·DRAM·intlv | `l1/height_sharded/32x1 · 32c` | 110·L1·intlv | `chain` | **buffer only** | U·P |
| 29 | `*(conversion)*` | `UntilizeWithUnpadding` | 10.1 | — | — | 32·DRAM·intlv | `—` | 32·L1·intlv | `boundary` | — | — |
| 30 | `*(conversion)*` | `UntilizeWithUnpadding` | 10.1 | — | — | 32·DRAM·intlv | `—` | 32·L1·intlv | `boundary` | — | — |
| 31 | `*(conversion)*` | `Permute` | 5.8 | — | — | 64·DRAM·intlv | `—` | **gone**·— | `untraced` | — | — |
| 32 | `*(conversion)*` | `Permute` | 6.1 | — | — | 64·DRAM·intlv | `—` | **gone**·— | `untraced` | — | — |
| 33 | `concat` | `Concat` | 12.3 | — | (1, 32, 32, 96) interleaved | 110·DRAM·intlv | `l1/height_sharded/32x1 · 32c` | 110·L1·intlv | `chain` | **buffer only** | U |
| 34 | `multiply` | `Permute` | 7.6 | — | — | 96·DRAM·intlv | `l1/height_sharded/32x1 · 32c` | **gone**·— | `chain` | undecidable | P |
| 35 | `*(conversion)*` | `TilizeWithValPadding` | 3.5 | — | — | 32·DRAM·intlv | `—` | 32·DRAM·intlv | `boundary` | — | — |
| 36 | `multiply` | `BinaryNg` | 3.7 | (1, 32, 32, 96) interleaved | — | 110·DRAM·intlv | `l1/height_sharded/32x1 · 32c` | 110·L1·intlv | `chain` | **buffer only** | U·P |
| 37 | `add` | `BinaryNg` | 3.3 | (1, 1, 32, 8192) interleaved | — | 110·DRAM·intlv | `l1/height_sharded/32x1 · 32c` | 110·DRAM·intlv | `chain` | undecidable | P |
| 38 | `*(conversion)*` | `BinaryNg` | 2.3 | — | — | 110·DRAM·intlv | `—` | 110·L1·intlv | `untraced` | — | — |
| 39 | `*(conversion)*` | `InterleavedToSharded` | 1.0 | — | — | 32·DRAM·intlv | `—` | 32·L1·intlv | `boundary` | — | — |
| 40 | `*(conversion)*` | `InterleavedToSharded` | 1.0 | — | — | 32·DRAM·intlv | `—` | 32·L1·intlv | `boundary` | — | — |
| 41 | `*(conversion)*` | `PagedUpdateCache` | 34.1 | — | — | 32·DRAM·intlv | `—` | 32·DRAM·intlv | `untraced` | — | — |
| 42 | `*(conversion)*` | `PagedUpdateCache` | 34.9 | — | — | 32·DRAM·intlv | `—` | 32·DRAM·intlv | `untraced` | — | — |
| 43 | `paged_scaled_dot_product_attention_decode` | `SdpaDecode` | 50.4 | — | — | 110·L1_H·shrd | `dram/interleaved/1x1` | 110·L1_H·shrd | `dram_resident` | no | G |
| 44 | `*(conversion)*` | `InterleavedToSharded` | 1.6 | — | — | 32·DRAM·intlv | `—` | 32·DRAM·intlv | `boundary` | — | G·b |
| 45 | `nlp_concat_heads_decode` | `NLPConcatHeadsDecode` | 4.7 | — | — | 32·L1_H·shrd | `dram/interleaved/1x1` | 32·L1_H·shrd | `dram_resident` | no | G |
| 46 | `linear` | `Matmul` | 30.4 | (1, 1, 32, 3072) shard(32, 96)/32c | (1, 1, 32, 3072) interleaved | 32·L1_W·shrd | `l1/width_sharded/1x96 · 96c` | 32·L1_W·shrd | `chain` | **family only** | O |
| 47 | `add` | `BinaryNg` | 2.4 | — | — | 110·DRAM·intlv | `l1/width_sharded/1x96 · 96c` | 110·DRAM·intlv | `chain` | undecidable | O·P |
| 48 | `rms_norm` | `LayerNorm` | 44.5 | (1, 1, 32, 3072) interleaved | (1, 1, 32, 3072) interleaved | 1·DRAM·intlv | `l1/block_sharded/1x11 · 11c` | 1·DRAM·intlv | `chain` | no | O |
| 49 | `linear` | `Matmul` | 103.9 | (1, 1, 32, 3072) interleaved | (1, 1, 32, 16384) interleaved | 103·DRAM·intlv | `l1/width_sharded/1x103 · 103c` | 103·DRAM·intlv | `chain` | no | O |
| 50 | `slice_static` | `Slice` | 3.1 | (1, 1, 32, 16384) interleaved | — | 110·DRAM·intlv | `l1/interleaved/10x11` | 110·DRAM·intlv | `chain` | no | O |
| 51 | `slice_static` | `Slice` | 3.3 | (1, 1, 32, 16384) interleaved | — | 110·DRAM·intlv | `l1/interleaved/10x11` | 110·DRAM·intlv | `chain` | no | O |
| 52 | `multiply` | `BinaryNg` | 8.5 | — | — | 110·DRAM·intlv | `l1/width_sharded/1x86 · 86c` | 110·DRAM·intlv | `chain` | undecidable | O·P |
| 53 | `*(conversion)*` | `InterleavedToSharded` | 2.0 | — | — | 16·DRAM·intlv | `—` | 16·DRAM·intlv | `boundary` | — | — |
| 54 | `linear` | `Matmul` | 48.0 | (1, 1, 32, 8192) shard(32, 512)/16c | (1, 1, 32, 3072) shard(32, 192)/16c | 12·L1_W·shrd | `l1/width_sharded/1x96 · 96c` | 12·L1_W·shrd | `agrees_with_shipped` | **family only** | D |
| 55 | `*(conversion)*` | `ShardedToInterleaved` | 1.5 | — | — | 16·L1_W·shrd | `—` | 16·L1_W·shrd | `boundary` | — | — |
| 56 | `add` | `BinaryNg` | 2.5 | — | — | 110·DRAM·intlv | `l1/width_sharded/1x96 · 96c` | 110·DRAM·intlv | `chain` | undecidable | T·P |

| stage `bucket` | follows advice? | rows | why |
|---|---|---|---|
| `boundary` | — | 17 | no per-op advice exists (`reshards[]`, not `ops[]`) |
| `chain` | **no** | 9 | **`O` — the oracle veto** |
| `untraced` | — | 8 | absent from the advisor's graph |
| `chain` | **undecidable** | 7 | `P` — positional pairing |
| `chain` | **buffer only** | 6 | `U` — the advised sharding does not run |
| `chain` | **yes** | 4 | `W` — shipped, chain `kept` |
| `dram_resident` | **no** | 2 | `G` — `TT_FATAL` GQA |
| `chain` | **family only** | 1 | `O` — grid 88 advised vs 32 shipped |
| `agrees_with_shipped` | **family only** | 1 | `D` — DS branch, grid not compared |
| `agrees_with_shipped` | **no** | 1 | `L` — label compares cores only |


## Corrections made in this revision

| was | now | why |
|---|---|---|
| every advised core count — `· 22c`, `· 88c`, `· 99c`, `· 77c` | **`· 32c`, `· 96c`, `· 103c`, `· 86c`** | The reconciliation parses `cores=` from `report.json`, which prints only the first range of a multi-range `CoreRangeSet`. The grid-string product is correct — validated 22/22 against the decision trace. |
| "the advised sharding is unreachable; `l1/interleaved` is the only legal placement" | **it runs, and it is −10.43 % at PCC 1.0** | My probe used shard (32,48); the advisor specified (32,64). The isolated single-op test passes on every op, including the `concat` I claimed was impossible. |

### Corrections from the previous revision

Adding the `why` column forced three of my own earlier labels to change. Recorded rather than quietly fixed:

| was | now | why |
|---|---|---|
| row 46 `linear` — "**yes**" | **family only** | It was *already* `L1 width-sharded` before the change, and the advised grid (88c) differs from the shipped one (32c). The chain that would have changed it, `dense:11`, was rejected. Calling that "follows the advice" gave the advisor credit for the incumbent's own choice. |
| row 54 `linear` — "**yes**" | **family only** | Same: `agrees_with_shipped` here comes from the DS branch, which does not compare grids (88 advised vs 12 shipped). |
| rows 19, 22, 34, 37, 47, 52, 56 — "no" / "— (op removed)" | **undecidable** | All seven are `pair_confidence: position`. I had been reading positional guesses as findings. |

The headline counts moved with them: **"5 yes / 14 no" → "4 yes / 12 no / 7 undecidable / 2 family only"**.

---

## The files, and where they are

All on branch `mvasiljevic/qb2/skillexp/advchal-v2-results`, path `skillexp/`, and locally under
`/home/mvasiljevic/skillexp-logs/exp-advisor-probe/`:

| file | what it is |
|---|---|
| `phi_BEFORE_rope_off.txt` | executed ttnn call sequence, `ROPE_L1=''` — 43 calls, shapes + shard specs |
| `phi_AFTER_rope_on.txt` | executed ttnn call sequence, `ROPE_L1='query_key'` — 43 calls |
| **`phi_TERMINAL_BEFORE.txt`** | **real `tt-perf-report` terminal output, incumbent** — 62 op rows |
| **`phi_TERMINAL_AFTER.txt`** | **real `tt-perf-report` terminal output, winner** — 58 op rows |
| `trace_ttnn.py` | the tracer, to regenerate for any policy |

The two terminal outputs are what you asked for: the tool's own rendering, with `Total %`, `Bound`, `Cores`,
`DRAM %`, `FLOPs %`, math fidelity and its per-op advice, for both sides. I had to generate them — the cells
saved only the stdout *preamble* (`profiles/dense_incumbent.txt`, `dense_winner.txt`, 9 lines each), because
the table went to `--csv`. Those preambles do carry one useful number each:

> **DRAM roofline: 15.9 % (81 GB/s) before → 17.6 % (90 GB/s) after.**

⚠ The BEFORE terminal output is generated from `profiles/incumbent_large/` rather than `profiles/incumbent/`,
because the latter's signposted window contains **no device rows** — the Tracy trace-replay limitation. The
`incumbent_large` dump covers the same configuration.
