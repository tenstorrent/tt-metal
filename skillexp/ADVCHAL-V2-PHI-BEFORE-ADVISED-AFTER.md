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
`· 22c` after an advised layout is the reconciliation's `advised_cores` field — **note it often disagrees with
the number in the layout string** (`1x96 · 88c`, `32x1 · 22c`); that mismatch is the grid-vs-DS-family parsing
quirk recorded as action C5.

## The answer to "does it follow the advice?"

| stage `bucket` label | actually follows advice? | rows |
|---|---|---|
| `chain` | **no** | 14 |
| `chain` | **buffer only** — got L1, not the sharding | **6** |
| `chain` | **yes** | 5 |
| `chain` | — (op removed) | 2 |
| `boundary` | — (conversion, no advice) | 17 |
| `untraced` | — | 8 |
| `dram_resident` | no | 2 |
| `agrees_with_shipped` | yes | 1 |
| `agrees_with_shipped` | **no** | 1 |

**Of 27 ops the stage put in the `chain` bucket, 5 ended up matching the advice, 6 took the buffer type but
not the sharding, and 14 did not follow it at all.** And note the last two rows: the stage labels 2 ops
`agrees_with_shipped` — its term for "nothing to screen, we already do what the advice says" — and **one of
those two does not actually match** (`typecast`, advised `l1/height_sharded/1x1`, shipped `1·DRAM·intlv`).

### The 6 "buffer only" rows are the interesting ones

`neg` ×2, `concat` ×2, `multiply` ×2 — the advisor advised `l1/height_sharded/32x1` and the cell shipped
`l1/interleaved`. So the shipped win took the **buffer type** (DRAM→L1, the advisor's first-ranked criterion)
and left the **sharding** half unimplemented. **Measured below: that half cannot be implemented at all** — so
this is not a shortcut the cell took, it is the only legal placement.

### Measured: what happens if you *do* implement the advised sharding

I implemented the advice faithfully — slices `l1/interleaved`, then `neg`/`concat`/`multiply`/`add` on
`l1/height_sharded` over 32 cores, shard `(32, width)`, matching the incoming
`l1/height_sharded, shard=(32,96), cores=32` the trace shows. Two variants, because the ops have two widths:

| variant | what it shards | result |
|---|---|---|
| `partial` | only the **96-wide** ops (`concat` output, `multiply` ×2, `add`) — shard `(32,96)` = 3 tiles, tile-aligned | **`TT_FATAL: Cannot concat interleaved inputs into a sharded output. Either shard the inputs first or use an interleaved output memory config.`** |
| `full` | also the **48-wide** `neg`, which the advice requires — shard `(32,48)` | **`TT_FATAL: Physical shard shape (32, 48) must be tile {32, 32} sized!`** |
| *(control)* `shipped` — `l1/interleaved` | — | **runs: 0.768758 / 0.768047 ms** |

Both failures reproduced twice. They **chain**: to give `concat` a sharded output you must shard its inputs;
its inputs are the 48-wide halves; a 48-wide shard is not tile-aligned. So **there is no way to reach the
advised placement for this rope body**, and the shipped `l1/interleaved` form is not a shortcut — it is the only
legal option. **The time cannot be measured because the configuration does not run.**

Phi's own source says why the widths are awkward, in `_apply_rope`:

> *`ttnn.experimental.rotary_embedding` requires a width divisible by 64, whereas Phi-3.5's 96-wide heads split
> at 48. The explicit topology is the exact HF operation and has no host fallback.*

### The real finding: the advisor's validation and the runtime disagree

This is worth separating from the outcome. The advisor **validated** the configuration it advised — from its own
decision trace:

| op | evaluations | valid | is `height_sharded/32x1` among the valid? |
|---|---|---|---|
| `ttnn.neg` op10 | 296 | **296 (all valid)** | **yes** — one of 112 valid height-sharded candidates |
| `ttnn.concat` op11 | 512 | 256 | **yes** |

**So the op model accepts a `(32,48)` height shard and the runtime rejects it.** The advisor validates against
the op model on a mock device (`op_constraint_validation::validateOperation`), and that check does not enforce
the runtime's tile-sized-shard rule. That is a genuine consistency gap between tt-mlir's validation and
tt-metal's runtime, and it is the reason a validated plan is unimplementable.

*(A note on my own trail here: I first called the advice illegal on the basis of a probe that did not implement
it — that reasoning was wrong and I retracted it. With a faithful implementation the conclusion holds, but the
substantive result is the validation gap above, not the illegality per se.)*

## Known artefact in row 19

Row 19 pairs `multiply` with a `PermuteDeviceOperation`. That is the reconciliation's own name/position
pairing being wrong, not a real mapping — the same soft-pairing limitation the tool documents in its
`limitations[]` and that qwen FN and nm B both caught against the IR.

---

| # | op | device op | µs | in shape | out shape | BEFORE cores·mem | ADVISOR ADVISED | AFTER cores·mem | stage label | follows advice? |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `rms_norm` | `LayerNorm` | 44.3 | (1, 1, 32, 3072) interleaved | (1, 1, 32, 3072) interleaved | 1·DRAM·intlv | `l1/block_sharded/1x11 · 11c` | 1·DRAM·intlv | `chain` | no |
| 2 | `linear` | `Matmul` | 72.6 | (1, 1, 32, 3072) interleaved | (1, 1, 32, 9216) interleaved | 96·DRAM·intlv | `l1/width_sharded/1x96 · 88c` | 96·DRAM·intlv | `chain` | no |
| 3 | `nlp_create_qkv_heads_decode` | `NLPCreateQKVHeadsDecode` | 56.6 | — | — | 32·DRAM·intlv | `l1/height_sharded/32x1 · 22c` | 32·DRAM·intlv | `chain` | no |
| 4 | `typecast` | `Typecast` | 0.9 | (32,) interleaved | — | 1·DRAM·intlv | `l1/height_sharded/1x1 · 1c` | 1·DRAM·intlv | `agrees_with_shipped` | no |
| 5 | `embedding` | `Embeddings` | 2.5 | (32,) interleaved | (32, 96) interleaved | 1·DRAM·intlv | `l1/interleaved/10x11` | 1·DRAM·intlv | `chain` | no |
| 6 | `embedding` | `Embeddings` | 2.5 | (32,) interleaved | (32, 96) interleaved | 1·DRAM·intlv | `l1/interleaved/10x11` | 1·DRAM·intlv | `chain` | no |
| 7 | `*(conversion)*` | `ShardedToInterleaved` | 1.5 | — | — | 32·L1_H·shrd | `—` | 32·L1_H·shrd | `boundary` | — |
| 8 | `*(conversion)*` | `ShardedToInterleaved` | 1.4 | — | — | 32·L1_H·shrd | `—` | 32·L1_H·shrd | `boundary` | — |
| 9 | `slice_static` | `Slice` | 1.6 | (1, 32, 32, 96) interleaved | — | 64·DRAM·intlv | `l1/interleaved/10x11` | 64·L1·intlv | `chain` | **yes** |
| 10 | `*(conversion)*` | `Untilize` | 4.4 | — | — | 32·DRAM·intlv | `—` | 32·L1·intlv | `boundary` | — |
| 11 | `slice_static` | `Slice` | 6.8 | (1, 32, 32, 96) interleaved | — | 110·DRAM·intlv | `l1/interleaved/10x11` | 110·L1·intlv | `chain` | **yes** |
| 12 | `*(conversion)*` | `TilizeWithValPadding` | 4.3 | — | — | 32·DRAM·intlv | `—` | 32·L1·intlv | `boundary` | — |
| 13 | `neg` | `Unary` | 2.1 | (1, 32, 32, 48) interleaved | — | 110·DRAM·intlv | `l1/height_sharded/32x1 · 22c` | 110·L1·intlv | `chain` | **buffer only** |
| 14 | `*(conversion)*` | `UntilizeWithUnpadding` | 10.2 | — | — | 32·DRAM·intlv | `—` | 32·L1·intlv | `boundary` | — |
| 15 | `*(conversion)*` | `UntilizeWithUnpadding` | 10.1 | — | — | 32·DRAM·intlv | `—` | 32·L1·intlv | `boundary` | — |
| 16 | `*(conversion)*` | `Permute` | 5.8 | — | — | 64·DRAM·intlv | `—` | **gone**·— | `untraced` | — |
| 17 | `*(conversion)*` | `Permute` | 5.9 | — | — | 64·DRAM·intlv | `—` | **gone**·— | `untraced` | — |
| 18 | `concat` | `Concat` | 12.2 | — | (1, 32, 32, 96) interleaved | 110·DRAM·intlv | `l1/height_sharded/32x1 · 22c` | 110·L1·intlv | `chain` | **buffer only** |
| 19 | `multiply` | `Permute` | 7.9 | — | — | 96·DRAM·intlv | `l1/height_sharded/32x1 · 22c` | **gone**·— | `chain` | — |
| 20 | `*(conversion)*` | `TilizeWithValPadding` | 3.6 | — | — | 32·DRAM·intlv | `—` | 32·DRAM·intlv | `boundary` | — |
| 21 | `multiply` | `BinaryNg` | 3.4 | (1, 32, 32, 96) interleaved | — | 110·DRAM·intlv | `l1/height_sharded/32x1 · 22c` | 110·L1·intlv | `chain` | **buffer only** |
| 22 | `add` | `BinaryNg` | 3.4 | (1, 32, 32, 96) interleaved | — | 110·DRAM·intlv | `l1/height_sharded/32x1 · 22c` | 110·DRAM·intlv | `chain` | no |
| 23 | `*(conversion)*` | `BinaryNg` | 2.1 | (1, 32, 32, 96) interleaved | — | 110·DRAM·intlv | `—` | 110·L1·intlv | `untraced` | — |
| 24 | `slice_static` | `Slice` | 1.3 | (1, 32, 32, 96) interleaved | — | 64·DRAM·intlv | `l1/interleaved/10x11` | 64·L1·intlv | `chain` | **yes** |
| 25 | `*(conversion)*` | `Untilize` | 3.6 | — | — | 32·DRAM·intlv | `—` | 32·L1·intlv | `boundary` | — |
| 26 | `slice_static` | `Slice` | 6.6 | (1, 32, 32, 96) interleaved | — | 110·DRAM·intlv | `l1/interleaved/10x11` | 110·L1·intlv | `chain` | **yes** |
| 27 | `*(conversion)*` | `TilizeWithValPadding` | 3.9 | — | — | 32·DRAM·intlv | `—` | 32·L1·intlv | `boundary` | — |
| 28 | `neg` | `Unary` | 1.5 | (1, 32, 32, 48) interleaved | — | 110·DRAM·intlv | `l1/height_sharded/32x1 · 22c` | 110·L1·intlv | `chain` | **buffer only** |
| 29 | `*(conversion)*` | `UntilizeWithUnpadding` | 10.1 | — | — | 32·DRAM·intlv | `—` | 32·L1·intlv | `boundary` | — |
| 30 | `*(conversion)*` | `UntilizeWithUnpadding` | 10.1 | — | — | 32·DRAM·intlv | `—` | 32·L1·intlv | `boundary` | — |
| 31 | `*(conversion)*` | `Permute` | 5.8 | — | — | 64·DRAM·intlv | `—` | **gone**·— | `untraced` | — |
| 32 | `*(conversion)*` | `Permute` | 6.1 | — | — | 64·DRAM·intlv | `—` | **gone**·— | `untraced` | — |
| 33 | `concat` | `Concat` | 12.3 | — | (1, 32, 32, 96) interleaved | 110·DRAM·intlv | `l1/height_sharded/32x1 · 22c` | 110·L1·intlv | `chain` | **buffer only** |
| 34 | `multiply` | `Permute` | 7.6 | — | — | 96·DRAM·intlv | `l1/height_sharded/32x1 · 1c` | **gone**·— | `chain` | — |
| 35 | `*(conversion)*` | `TilizeWithValPadding` | 3.5 | — | — | 32·DRAM·intlv | `—` | 32·DRAM·intlv | `boundary` | — |
| 36 | `multiply` | `BinaryNg` | 3.7 | (1, 32, 32, 96) interleaved | — | 110·DRAM·intlv | `l1/height_sharded/32x1 · 22c` | 110·L1·intlv | `chain` | **buffer only** |
| 37 | `add` | `BinaryNg` | 3.3 | (1, 1, 32, 8192) interleaved | — | 110·DRAM·intlv | `l1/height_sharded/32x1 · 1c` | 110·DRAM·intlv | `chain` | no |
| 38 | `*(conversion)*` | `BinaryNg` | 2.3 | — | — | 110·DRAM·intlv | `—` | 110·L1·intlv | `untraced` | — |
| 39 | `*(conversion)*` | `InterleavedToSharded` | 1.0 | — | — | 32·DRAM·intlv | `—` | 32·L1·intlv | `boundary` | — |
| 40 | `*(conversion)*` | `InterleavedToSharded` | 1.0 | — | — | 32·DRAM·intlv | `—` | 32·L1·intlv | `boundary` | — |
| 41 | `*(conversion)*` | `PagedUpdateCache` | 34.1 | — | — | 32·DRAM·intlv | `—` | 32·DRAM·intlv | `untraced` | — |
| 42 | `*(conversion)*` | `PagedUpdateCache` | 34.9 | — | — | 32·DRAM·intlv | `—` | 32·DRAM·intlv | `untraced` | — |
| 43 | `paged_scaled_dot_product_attention_decode` | `SdpaDecode` | 50.4 | — | — | 110·L1_H·shrd | `dram/interleaved/1x1` | 110·L1_H·shrd | `dram_resident` | no |
| 44 | `*(conversion)*` | `InterleavedToSharded` | 1.6 | — | — | 32·DRAM·intlv | `—` | 32·DRAM·intlv | `boundary` | — |
| 45 | `nlp_concat_heads_decode` | `NLPConcatHeadsDecode` | 4.7 | — | — | 32·L1_H·shrd | `dram/interleaved/1x1` | 32·L1_H·shrd | `dram_resident` | no |
| 46 | `linear` | `Matmul` | 30.4 | (1, 1, 32, 3072) shard(32, 96)/32c | (1, 1, 32, 3072) interleaved | 32·L1_W·shrd | `l1/width_sharded/1x96 · 88c` | 32·L1_W·shrd | `chain` | **yes** |
| 47 | `add` | `BinaryNg` | 2.4 | — | — | 110·DRAM·intlv | `l1/width_sharded/1x96 · 88c` | 110·DRAM·intlv | `chain` | no |
| 48 | `rms_norm` | `LayerNorm` | 44.5 | (1, 1, 32, 3072) interleaved | (1, 1, 32, 3072) interleaved | 1·DRAM·intlv | `l1/block_sharded/1x11 · 11c` | 1·DRAM·intlv | `chain` | no |
| 49 | `linear` | `Matmul` | 103.9 | (1, 1, 32, 3072) interleaved | (1, 1, 32, 16384) interleaved | 103·DRAM·intlv | `l1/width_sharded/1x103 · 99c` | 103·DRAM·intlv | `chain` | no |
| 50 | `slice_static` | `Slice` | 3.1 | (1, 1, 32, 16384) interleaved | — | 110·DRAM·intlv | `l1/interleaved/10x11` | 110·DRAM·intlv | `chain` | no |
| 51 | `slice_static` | `Slice` | 3.3 | (1, 1, 32, 16384) interleaved | — | 110·DRAM·intlv | `l1/interleaved/10x11` | 110·DRAM·intlv | `chain` | no |
| 52 | `multiply` | `BinaryNg` | 8.5 | — | — | 110·DRAM·intlv | `l1/width_sharded/1x86 · 77c` | 110·DRAM·intlv | `chain` | no |
| 53 | `*(conversion)*` | `InterleavedToSharded` | 2.0 | — | — | 16·DRAM·intlv | `—` | 16·DRAM·intlv | `boundary` | — |
| 54 | `linear` | `Matmul` | 48.0 | (1, 1, 32, 8192) shard(32, 512)/16c | (1, 1, 32, 3072) shard(32, 192)/16c | 12·L1_W·shrd | `l1/width_sharded/1x96 · 88c` | 12·L1_W·shrd | `agrees_with_shipped` | **yes** |
| 55 | `*(conversion)*` | `ShardedToInterleaved` | 1.5 | — | — | 16·L1_W·shrd | `—` | 16·L1_W·shrd | `boundary` | — |
| 56 | `add` | `BinaryNg` | 2.5 | — | — | 110·DRAM·intlv | `l1/width_sharded/1x96 · 88c` | 110·DRAM·intlv | `chain` | no |

| stage `bucket` label | follows advice? | rows |
|---|---|---|
| `boundary` | — | 17 |
| `chain` | no | 14 |
| `untraced` | — | 8 |
| `chain` | **buffer only** | 6 |
| `chain` | **yes** | 5 |
| `chain` | — | 2 |
| `dram_resident` | no | 2 |
| `agrees_with_shipped` | no | 1 |
| `agrees_with_shipped` | **yes** | 1 |


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
