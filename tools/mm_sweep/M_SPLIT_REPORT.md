# M-split (Sm>1) in1 delivery — placement + forwarding

Placement and forwarding are treated as **two separate experiments**; gains are attributed and landed
separately. Only the production Sm>1 shape today is **256×2048×1024 (Sm=2)** (the picker uses Sm=1
everywhere else), so that is the production primary.

## Part 1 — audit of the current M-split path

**Indexing / placement.** Core index `i = bank*preaders + slice`, `slice = kk*(Ns*Sm) + nn*Sm + mm` (mm
innermost). `mm==0` is the in1 DRAM reader; `mm>0` are slaves; all `Sm` members of a `(kk,nn)` group share
one reader NoC (`noc = (kk*Ns+nn)&1`). Placement is a `b`-outer/`p`-inner `find_near()` **logical-Manhattan
spiral** around `opt0/opt1[bank]` (the bank's preferred worker for the group's NoC). Because groups are
allocated **reader-then-slaves contiguously**, early groups' slaves consume bank-adjacent cores before later
groups' actual DRAM readers are placed — pushing later readers farther from their bank. The factory builds
reader/slave runtime args from **contiguous indices** (`reader→i+s`, `slave→i-mm`), so any placement change
may move only `cp.coord`; indices/ownership must stay fixed.

**in1 synchronization protocol.** Per in1 block (there are `N_bpc*G*W`), reader (mrole=1):
`cb_reserve_back` → DRAM read + `noc_async_read_barrier` → **group-wide** `noc_semaphore_wait_min(in1ready,
(mbc+1)*mpeers)` (wait ALL slaves have a free slot) → `noc_async_write` payload to each slave →
`noc_async_writes_flushed()` → `noc_semaphore_inc(valid)` to each slave → `cb_push_back`. Slave (mrole=0):
`cb_reserve_back` → `noc_semaphore_inc(reader_ready)` → `noc_semaphore_wait_min(valid,b+1)` → `cb_push_back`.
Payload is **flushed before the valid signal** (correct visibility). Waits are **group-wide** (reader waits
for all slaves before forwarding a block), not per-peer. Per block, as a function of Sm (`mpeers=Sm-1`):
`Sm-1` payload writes, `1` flush, `Sm-1` valid atomics (reader) + `1` ready atomic (each slave). **At Sm=2
this is exactly one forwarded copy per block** — so multicast/tree cannot reduce Sm=2's payload count; a
Sm=2 gain must come from placement, synchronization, or overlap.

**Watcher warning (investigated).** The pre-existing Sm>1 warning is **real but a drain-hygiene issue, not an
intra-op visibility bug**: the reader's `valid` incs and the slaves' `ready` incs are **non-posted atomics
that are never drained before kernel exit** (`noc_async_writes_flushed()` drains writes, not atomics; there
is no `noc_async_atomic_barrier()`). Within the op, correctness holds (payload flushed before valid). The gap
is cross-op safety (a pending atomic could touch a semaphore address a later op reuses). It fires identically
under the current placement and any Sm>1 config. **Fix belongs to the forwarding part** (drain atomics at
exit); tracked there.

## Part 2 — placement experiment

Cache-hashed internal placement diagnostics (host-side factory override of `cp.coord` before the PARETO ring
reorder, which then recomputes on the new coords; logical indices/ownership/forward-arg math unchanged):
- **current** (`DIAG_PLACE_CURRENT`, baseline) — the planner's reader-then-slaves logical-Manhattan spiral.
- **readers_first** (`DIAG_PLACE_READERS_FIRST`) — place every `mm==0` reader first (same targets/NoC/spiral),
  then slaves by bank spiral.
- **in1_near** (DEFAULT) — readers-first, then each slave at the free worker minimizing the directed
  reader→slave hop on the group's in1-reader NoC.

`TOPOLOGY_AWARE` (jointly optimizing reader→slave distance, ring route, reduction links, multicast geometry)
was **not** pursued: `in1_near` already captures the dominant lever (forward distance) with a clean win; a
fuller multi-objective placement risks over-tuning for marginal additional gain and is left as future work.

**Route metrics (PLACECOST, Sm=2 primary).** Under readers_first the noc=1 groups' slaves still land far
(reader→slave `maxfwd` ≈ 15–17 hops); in1_near cuts them to ≈ 3 (op-level `maxfwd` 17→6).

**Performance** — median device-profiler kernel µs, 3 interleaved relaunches × 2 independent runs, Δ vs
current. Raw: `regime_a_placement_run1.json`, `regime_a_placement_bench.json` (run2).
| shape | Sm | cfg | readers_first Δ | **in1_near Δ** |
|---|---|---|---|---|
| 256×2048×1024 (production) | 2 | 1,4,2,2,2 | −2.7 / −4.3% | **−6.3 / −7.2%** |
| 128×6144×4608 | 2 | 1,6,2,2,1 | +1.2 / +1.2% | −3.5 / −3.6% |
| 256×2048×1024 | 3 | 1,1,3,2,2 | +0.7 / +0.4% | −0.9 / −1.2% |
| 256×2048×1024 | 4 | 1,1,4,2,2 | −10.5 / −11.1% | −13.8 / −13.8% |
| 256×6144×4608 | 4 | 1,3,4,2,1 | −3.6 / −3.4% | −8.0 / −7.6% |
| deep-K/wide-N/M=32/64/128 (Sm=1) | 1 | picker | ~0 (no-op) | ~0 (no-op) |

**Decision: in1_near** (single default). It clears the ≥2% gate on the production Sm=2 primary
(**−6.3/−7.2%**, two runs), wins every Sm>1 shape (−0.9% to −13.8%), and is a **no-op at Sm=1** (controls
within noise). `readers_first` alone is rejected (inconsistent: +1.2% on 128×6144×4608 both runs — it seats
readers but leaves slaves far). The win tracks the reader→slave `maxfwd` reduction. Landed as the placement
decision separately from forwarding.

**Correctness.** `PlacementCorrectness` gtest: PCC ≥ 0.999 vs CPU golden fresh+cached for all variants across
Sm=2/3/4 + Sm=1; **bit-identical to current for the Sm=1 no-op**. For Sm>1, placement recomputes the PARETO
ring order on the new coords → the K-shard accumulation order can change → a benign **bf16 output-ULP**
rounding difference (ab_maxdiff ≈ one ULP at the output magnitude; PCC 0.99999) — correct, not a bug (the
2048×1024 shapes, whose ring order didn't change, stayed bit-identical). Public 20/20 pass on the in1_near
default. Watcher: the in1_near Sm>1 path still trips the **pre-existing** in1_reader atomic-drain warning
(identical under current placement) — addressed in the forwarding part.

## Part 3 — forwarding (in progress)
On top of in1_near, evaluating: the atomic-drain safety fix (required for watcher-clean Sm>1), and forwarding
candidates (per-peer cumulative, pipelined unicast, multicast, tree/chain). Details appended when complete.
