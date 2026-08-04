# phi-3.5-mini — the shipped win, op by op

**One file, both sides aligned.** This is what the advisor's `rope_l1` change actually did to a phi decoder
layer, every op, before and after — from the cells' own `tt-perf-report` CSVs.

## The headline: it is a *movement* win, not a compute win

Two independent phi arms shipped the same class of change. Both were profiled before and after:

| | phi FN | phi B |
|---|---|---|
| profiled window | **725.2 → 654.0 µs** (−71.2, **−9.82 %**) | **699.8 → 624.0 µs** (−75.7, **−10.82 %**) |
| ops | 56 → 50 | 60 → 54 |
| **Compute** | 434.1 → 430.6 (**−3.5**) | 462.3 → 456.0 (**−6.4**) |
| **TM** (tensor manipulation) | 212.1 → **146.0** (**−66.2**) | 224.4 → **154.9** (**−69.5**) |
| DM (data movement) | 78.9 → 77.4 (−1.5) | 13.0 → 13.2 (+0.2) |

**93 % of phi FN's win and 92 % of phi B's is TM reduction. Compute is essentially flat.** Attribution by op
class is near-identical across the two arms, which is a strong reproducibility signal:

| op class | phi FN Δ µs | share | phi B Δ µs | share |
|---|---|---|---|---|
| **`Permute`** | **−38.95** | **55 %** | **−38.56** | **51 %** |
| `Concat` | −14.02 | 20 % | −14.21 | 19 % |
| `Slice` | −8.44 | 12 % | −8.40 | 11 % |
| `UntilizeWithUnpadding` | −3.79 | 5 % | −4.05 | 5 % |
| `BinaryNg` | −3.06 | 4 % | −5.15 | 7 % |

**Mechanically: six `Permute` ops disappear entirely** (4 × 64-core at ~5.8–6.1 µs, 2 × 96-core at ~7.6–7.9 µs)
**and `Concat` on 110 cores halves** (12.24 → 5.27 µs, twice). Keeping the RoPE chain L1-resident removes the
permute/concat round-trips around it. No matmul got faster; no op got more cores.

> Note how this lands against the corpus's framing: the stage measures the advisor as a *placement* tool, and
> the placement change it shipped bought its entire win by deleting **movement** ops. That is the same signal
> the movement-budget rubric picks up — see
> [`ADVCHAL-V2-PERF-REPORT-AUDIT.md`](ADVCHAL-V2-PERF-REPORT-AUDIT.md).

## Reading the table

- `cat` — the tool's own `Op Category`, with the four missing op codes patched in (see the audit §5).
- `cores` — `before→after` when it changed, a single number when it did not.
- ⬇ / ⬆ mark a change of more than 1 µs.
- **REMOVED / ADDED** — the op is present on only one side.

⚠ **One alignment artefact to ignore.** Rows 42–49 show `PagedUpdateCache` / `SdpaDecode` as ADDED and then
REMOVED. That block simply moved position between the two runs and my greedy aligner does not re-order; the

---

## The sharding view — and the win is *not* about sharding

`tt-perf-report` gives three columns that describe placement: **`Cores`**, **`Input 0 Memory`**,
**`DRAM Sharded`**. Aggregating phi FN's device time by `Input 0 Memory`:

| `Input 0 Memory` | BEFORE | AFTER |
|---|---|---|
| `DEV_0_DRAM_INTERLEAVED` | **587.2 µs, 81.0 %, 49 ops** | 433.7 µs, 66.3 %, **21 ops** |
| `DEV_0_L1_INTERLEAVED` | — | **83.6 µs, 12.8 %, 22 ops** *(new bucket)* |
| `DEV_0_L1_WIDTH_SHARDED` | 79.9 µs, 11.0 %, 3 ops | 79.9 µs, 12.2 %, 3 ops — **unchanged** |
| `DEV_0_L1_HEIGHT_SHARDED` | 58.0 µs, 8.0 %, 4 ops | 56.7 µs, 8.7 %, 4 ops — **unchanged** |

**28 ops moved from DRAM-interleaved to L1-interleaved**, DRAM time fell 587.2 → 433.7 µs (−153.5), the new L1
bucket costs 83.6 µs, net −69.9 µs — which is the whole −71.2 µs win.

**The two genuinely *sharded* buckets do not change at all.** Width-sharded is 79.9 µs on both sides; height-sharded
is 58.0 → 56.7. `DRAM Sharded` is `False` on every row of both files, and no shard spec or core count changes
except where an op disappears.

So despite being produced by a *shard* advisor and screened as a placement candidate, **this win is a change of
buffer type — DRAM → L1 — not a change of sharding.** Worth holding next to the fact that the advisor's score
ranks `isL1` as its very first criterion (see [`ADVISOR-INTERNALS`](ADVCHAL-V2-ADVISOR-INTERNALS.md) §2): the one
thing it puts first is the thing that paid.

### Per-op sharding deltas (only rows that changed)

| op | cores B→A | Input 0 Memory B→A | DRAM-sharded B→A | µs B→A |
|---|---|---|---|---|
| `ShardedToInterleaved` **REMOVED** | 32 | DEV_0_L1_HEIGHT_SHARDED | False | 1.4→— |
| `Slice` | 64 | **DEV_0_DRAM_INTERLEAVED→DEV_0_L1_INTERLEAVED** | False | 1.6→1.3 |
| `Untilize` | 32 | **DEV_0_DRAM_INTERLEAVED→DEV_0_L1_INTERLEAVED** | False | 4.4→4.2 |
| `Slice` | 110 | **DEV_0_DRAM_INTERLEAVED→DEV_0_L1_INTERLEAVED** | False | 6.8→3.2 |
| `TilizeWithValPadding` | 32 | **DEV_0_DRAM_INTERLEAVED→DEV_0_L1_INTERLEAVED** | False | 4.3→4.4 |
| `Unary` | 110 | **DEV_0_DRAM_INTERLEAVED→DEV_0_L1_INTERLEAVED** | False | 2.1→1.7 |
| `UntilizeWithUnpadding` | 32 | **DEV_0_DRAM_INTERLEAVED→DEV_0_L1_INTERLEAVED** | False | 10.2→9.2 |
| `UntilizeWithUnpadding` | 32 | **DEV_0_DRAM_INTERLEAVED→DEV_0_L1_INTERLEAVED** | False | 10.1→9.2 |
| `Permute` **REMOVED** | 64 | DEV_0_DRAM_INTERLEAVED | False | 5.8→— |
| `Permute` **REMOVED** | 64 | DEV_0_DRAM_INTERLEAVED | False | 5.9→— |
| `Concat` | 110 | **DEV_0_DRAM_INTERLEAVED→DEV_0_L1_INTERLEAVED** | False | 12.2→5.3 |
| `Permute` **REMOVED** | 96 | DEV_0_DRAM_INTERLEAVED | False | 7.9→— |
| `BinaryNg` | 110 | **DEV_0_DRAM_INTERLEAVED→DEV_0_L1_INTERLEAVED** | False | 3.4→2.8 |
| `BinaryNg` | 110 | **DEV_0_DRAM_INTERLEAVED→DEV_0_L1_INTERLEAVED** | False | 2.1→1.7 |
| `InterleavedToSharded` **ADDED** | 32 | DEV_0_L1_INTERLEAVED | False | —→0.7 |
| `ShardedToInterleaved` **ADDED** | 32 | DEV_0_L1_HEIGHT_SHARDED | False | —→0.7 |
| `Slice` | 64 | **DEV_0_DRAM_INTERLEAVED→DEV_0_L1_INTERLEAVED** | False | 1.3→1.0 |
| `Untilize` | 32 | **DEV_0_DRAM_INTERLEAVED→DEV_0_L1_INTERLEAVED** | False | 3.6→3.3 |
| `Slice` | 110 | **DEV_0_DRAM_INTERLEAVED→DEV_0_L1_INTERLEAVED** | False | 6.6→2.2 |
| `TilizeWithValPadding` | 32 | **DEV_0_DRAM_INTERLEAVED→DEV_0_L1_INTERLEAVED** | False | 3.9→3.7 |
| `Unary` | 110 | **DEV_0_DRAM_INTERLEAVED→DEV_0_L1_INTERLEAVED** | False | 1.5→1.2 |
| `UntilizeWithUnpadding` | 32 | **DEV_0_DRAM_INTERLEAVED→DEV_0_L1_INTERLEAVED** | False | 10.1→9.2 |
| `UntilizeWithUnpadding` | 32 | **DEV_0_DRAM_INTERLEAVED→DEV_0_L1_INTERLEAVED** | False | 10.1→9.2 |
| `Permute` **REMOVED** | 64 | DEV_0_DRAM_INTERLEAVED | False | 5.8→— |
| `Permute` **REMOVED** | 64 | DEV_0_DRAM_INTERLEAVED | False | 6.1→— |
| `Concat` | 110 | **DEV_0_DRAM_INTERLEAVED→DEV_0_L1_INTERLEAVED** | False | 12.3→5.2 |
| `Permute` **REMOVED** | 96 | DEV_0_DRAM_INTERLEAVED | False | 7.6→— |
| `BinaryNg` | 110 | **DEV_0_DRAM_INTERLEAVED→DEV_0_L1_INTERLEAVED** | False | 3.7→2.8 |
| `BinaryNg` | 110 | **DEV_0_DRAM_INTERLEAVED→DEV_0_L1_INTERLEAVED** | False | 2.3→1.6 |
| `InterleavedToSharded` | 32 | **DEV_0_DRAM_INTERLEAVED→DEV_0_L1_INTERLEAVED** | False | 1.0→0.6 |
| `PagedUpdateCache` **ADDED** | 32 | DEV_0_DRAM_INTERLEAVED | False | —→34.1 |
| `PagedUpdateCache` **ADDED** | 32 | DEV_0_DRAM_INTERLEAVED | False | —→34.7 |
| `SdpaDecode` **ADDED** | 110 | DEV_0_L1_HEIGHT_SHARDED | False | —→49.7 |
| `PagedUpdateCache` **REMOVED** | 32 | DEV_0_DRAM_INTERLEAVED | False | 34.1→— |
| `PagedUpdateCache` **REMOVED** | 32 | DEV_0_DRAM_INTERLEAVED | False | 34.9→— |
| `SdpaDecode` **REMOVED** | 110 | DEV_0_L1_HEIGHT_SHARDED | False | 50.4→— |
| `InterleavedToSharded` **REMOVED** | 32 | DEV_0_DRAM_INTERLEAVED | False | 1.6→— |
| `Matmul` | 103.0 | DEV_0_DRAM_INTERLEAVED | False | 103.9→105.0 |


---

## The raw ttnn code — two files to open in split view

The change is a policy knob, so before/after is not two source files — it is two different **methods**.

**BEFORE** (`advisor_rope_l1=""`) takes the base-class path, `FunctionalDecoder._decode_rope`
(`tt/functional_decoder.py:400`). **AFTER** (`advisor_rope_l1="query_key"`) takes the override,
`OptimizedDecoder._decode_rope` (`tt/optimized_decoder.py:90`), whose entire substance is these six lines:

```python
def apply(value, enabled):
    memory_config = value.memory_config()
    staging = ttnn.L1_MEMORY_CONFIG if enabled else ttnn.DRAM_MEMORY_CONFIG   # <-- the whole change
    value = ttnn.to_memory_config(value, staging)
    value = self._apply_rope(value, cos, sin)
    return ttnn.to_memory_config(value, memory_config)

return apply(query, "query" in targets), apply(key, "key" in targets)
```

So the shipped optimisation is literally **`L1_MEMORY_CONFIG` instead of `DRAM_MEMORY_CONFIG` as the staging
buffer around `_apply_rope`**, applied to query and key.

### And the executed call sequence, traced

Reading source only tells you what *might* run. These two files are the **actual ttnn calls a decode executed**,
one per policy, with each tensor's shape, dtype, layout, buffer type and shard spec:

| file | policy |
|---|---|
| `phi_BEFORE_rope_off.txt` | `CHALLENGER_ADVISOR_ROPE_L1=''` |
| `phi_AFTER_rope_on.txt` | `CHALLENGER_ADVISOR_ROPE_L1='query_key'` |

**43 traced calls each, 165 lines each, 121 lines of diff.** Open them side by side. The rope region is calls
9–20; here is the same region on both sides:

```
BEFORE                                              AFTER
[  9] ttnn.to_memory_config()                       [  9] ttnn.to_memory_config()
        in   l1/height_sharded, cores=32                    in   l1/height_sharded, cores=32
        out  dram/interleaved          <-- to DRAM          out  l1/interleaved         <-- stays in L1
[ 10] ttnn.to_memory_config()          <-- second   [ 10] ttnn.slice()
        in   l1/height_sharded, cores=32                    in   l1/interleaved
        out  dram/interleaved              conversion       out  l1/interleaved
[ 11] ttnn.slice()   dram -> dram                   [ 11] ttnn.slice()   l1 -> l1
[ 13] ttnn.neg()     dram -> dram                   [ 12] ttnn.neg()     l1 -> l1
[ 15] ttnn.multiply() in dram                       [ 14] ttnn.multiply() in l1
```

Two things are visible that no summary conveys: the **second `to_memory_config` at [10] disappears**, and every
op in the rope body changes its input buffer from `dram/interleaved` to `l1/interleaved`.

To regenerate, or to trace any other policy:

```bash
TRACE_ROPE=query_key TRACE_NORM_CORES=11 TRACE_OUT=/tmp/x.txt python trace_ttnn.py
```

The tracer (`trace_ttnn.py`, in this directory) wraps ~29 ttnn entry points, logs one decode, and drops the
construction phase. Setting `TRACE_NORM_CORES=11` additionally traces the **discarded** combined candidate from
[`EXPERIMENTS`](ADVCHAL-V2-EXPERIMENTS.md) §E1.
