# advchal-v3 — every core grid that moved PCC, and the candidate tt-metal defects in it

The v2 corpus said this explicitly and nothing collected it: *"some ops compute the wrong answer under
particular shard specs, and changing placement is exactly what triggers such a bug… report the op and the
shard spec as a tt-metal bug. That is a real result of this stage, not a nuisance."*

This is that collection, from all 11 cells of the v3 run. It answers three questions asked of the run.

## 1. When is PCC checked?

**Per candidate, after it is timed.** Every measurement record carries `median_ms` and, where an oracle was
built, `oracle_pcc`. So the order is **measure → oracle → verdict**, and a materially wrong grid gets timed
before it is caught. That is the right order for device cost — you do not pay oracle time on candidates you
will not ship — and it is why the timing data below exists at all.

## 2. When a grid produces a bad PCC, is another grid tried?

**Not reliably — and on the cell where it mattered most, no.** gemma-4-26B `-onA` tried **one** rung on the
`sliding_attention` kind: 1 → 11 cores, PCC 0.99457, `rejected_correctness`. It then stopped searching that
kind and shipped nothing there — while v2 had shipped **88 cores on that same kind for −12.98 %**, and 88 was
**never tried**. On the `full_attention` kind the same cell went on to try 8 cores and kept it.

So a correctness rejection at one rung terminated the ladder for that kind. That is a **more precise
explanation of g26onA's miss** than the oracle-clause defect: the rule that vetoes is one thing, and stopping
the search on a veto is another. Both need fixing, and the skill says nothing about the second.

## 3. Do we have every case collected? Not yet, and the reason is a gate of mine

| | |
|---|---|
| measurements in the run | **187** |
| carrying `oracle_pcc` | **60 (32 %)** |
| carrying `op_under_test` | **0 (0 %)** |

`op_under_test {name, incumbent_grid, candidate_grid, legal_ladder}` is the field added specifically to make
"which op, which grid" mechanical — and it is populated in **none** of the 187 measurement records, because
my gate made it **advisory**. The table below is recoverable only because `reconcile.py` fills the same field
inside `cliff_candidates`, which gives **49 rows with an op name and 42 with a PCC**.

**Required change: `op_under_test` and `oracle_pcc` become CRITICAL per screened candidate.** Without them a
grid-dependent correctness bug is visible as a number with nothing attached to it.

---

# 4. The finding: `rms_norm` widened off 1 core degrades PCC on gemma-4-26B's *sliding* kind, reproducibly

**Three independent cells, three different arms, same model, same op, same layer kind, same result:**

| cell | arm | kind | grid | **PCC** | verdict |
|---|---|---|---|---:|---|
| gemma-4-26B `-onA` | nofuse-noadvise-onA | `sliding_attention` | 1 → **11** | **0.99457296** | `rejected_correctness` |
| gemma-4-26B `fuse-noadvise` | fuse-noadvise | `sliding_attention` | 1 → *(unrecorded)* | **0.99467277** | `rejected` |
| gemma-4-26B `nofuse-noadvise` | nofuse-noadvise | `sliding_attention` | 1 → **22** | **0.99469421** | `rejected_absolute_oracle` |

**Clustered within 0.00012 of each other, all just below the model's own 0.995 bar.** Three cells that never
saw each other's artefacts (blob-identity checked, 156 refs parked) produced the same number.

**And the same op, on the same model, on the *other* layer kind, is clean:**

| cell | kind | grid | PCC |
|---|---|---|---:|
| gemma-4-26B `-onA` | `full_attention` | 1 → 8 | 0.99980000 |
| gemma-4-26B `nofuse-noadvise` | `full_attention` | 1 → 22 | 0.99985746 |
| gemma-4-26B `fuse-noadvise` | `full_attention` | 1 → *(unrecorded)* | 0.99989244 |

So it is **not** the widening as such, and **not** the model. It is specific to the **sliding-attention path's
norm**, which points at that path's input — mask, shape or dtype — rather than at the core count.

**Why this is a candidate defect and not just a rejection.** v2 measured its shipped 88-core sliding config at
**0.999629** on this same cell. So 88 cores is clean and 11/22 are not, on the same op and kind. A correctness
result that depends on the *core count* of a reduction, non-monotonically, is the shard-spec bug signature the
corpus described.

**What is needed to file it**, and none of it is expensive: an isolated single-op test of gemma-4-26B's
sliding-attention `rms_norm` at 1, 11, 22, 44 and 88 cores against a fixed reference, on this host, with the
shard shape from `final_ir.mlir`. If 11 and 22 degrade while 1 and 88 do not, it is a tt-metal report with a
reproduction. If all of them degrade, then **v2's 0.999629 is the number in question** and the −12.98 % it
shipped needs re-examining.

# 5. The other materially wrong case

| cell | candidate | PCC | notes |
|---|---|---:|---|
| phi-3.5 `nofuse-noadvise` | `rope_l1_compatible_geometry` | **0.91731302** | recorded **twice** (candidate + confirmation), so reproducible |

**0.917 is not reassociation, it is wrong output** — and it was the cell's *fastest* measurement at −4.11 %,
so clause 1 of the oracle earned its place here. But the name is the finding: *`compatible_geometry`* means
the advised geometry was **not expressible**, so the cell built a nearby one — and no `op_under_test` was
recorded, so **the op and grid that produced 0.917 are not in the artefacts.** This is the case the §3 gap
costs us most.

# 6. Everything else, for completeness

The remaining 30-odd rows sit at 0.9995–0.99999 — reassociation, not error. Two are worth noting as the
counter-example that keeps the bar honest:

| cell | op | grid | PCC | outcome |
|---|---|---|---:|---|
| north-mini `-onA` | `rms_norm` | 1 → 16 | 0.99500000 | **kept** — exactly at the bar |
| north-mini `-onA` | `rms_norm` | 1 → 16 | 0.99597765 | kept |
| phi-3.5 `fuse-noadvise` | `rms_norm` | 1 → 11 | 0.99890780 | kept, shipped |
| north-mini `fuse-noadvise` | `rms_norm` | 1 → 22 | 0.99952112 | kept, shipped |

So the same transformation — a norm off one core — is **clean on north-mini and phi at every grid tried, and
degraded on gemma-4-26B's sliding kind at every grid tried.** That contrast is what makes the gemma case worth
filing rather than writing off as a tight bar.

# 7. Actions

1. **`op_under_test` and `oracle_pcc` CRITICAL per screened candidate** — without them this document cannot be
   built from the artefacts, and I had to reconstruct it from a different file.
2. **A correctness rejection must not end the ladder for that kind.** Record the remaining rungs as
   `not_screened_after_correctness_rejection` so the gap is visible; g26onA never tried the grid v2 shipped.
3. **Isolated single-op test** for §4, on this host, at 1/11/22/44/88 cores.
4. Until §4 is settled, gemma-4-26B's sliding-attention norm results — v2's and v3's — are **mutually
   inconsistent**, and neither should be quoted as settled.
