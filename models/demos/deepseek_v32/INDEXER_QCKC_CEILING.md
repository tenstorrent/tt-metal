# indexer_score — best QC/KC for the COMPUTE CEILING (heads8 bfp8 HiFi2)

Production case: **8 heads, bfp8 k, HiFi2, sp_rank 7, 110 cores**. This doc answers a
single question with hardware proof: *what QC/KC maximizes the compute ceiling, and why is
a bigger QC (and is a bigger KC) not better?* Companion to `INDEXER_COMPUTE_CEILING.md`
(what the ceiling is made of) and `INDEXER_DATAMOVEMENT.md` (why production nonetheless
ships QC=2 — a DMA win, not a compute win).

All numbers are `INDEXER_DMA_OFF=1` (reader/writer skip NoC, still push/pop CBs → compute
runs unstarved = the pure compute ceiling), from `test_indexer_score_sp7_math_util
[heads8_k_bfp8]`, this board, tracy device-kernel min.

## Result — QC=1 wins; the ceiling falls monotonically with QC

math_util (%), DMA off:

| QC \ KC | 4    | 8    | 16   | 32       |
|---------|------|------|------|----------|
| **1**   | 62.8 | 64.5 | 65.3 | **65.7** |
| 2       | 61.1 | 62.1 | 62.0 | L1 overflow |
| 4       | 58.7 | 59.0 | 57.9 | L1 overflow |

**Best compute-ceiling config: QC=1, KC=32 → 65.7 % (0.367 ms).** QC=1 beats QC=2 beats
QC=4 at every KC. Bigger KC is *also* better within QC=1 (62.8 → 65.7, diminishing). So
"bigger QC" is not best (it is strictly worst); "bigger KC" *is* best up to L1/divisibility.

> Production still ships **QC=2, KC=8** because with DMA *on* the op is reader-bandwidth-
> bound and QC=2 cuts redundant K reads ~2× (`INDEXER_DATAMOVEMENT.md`). That is a data-
> movement win that does not exist at the compute ceiling — hence the two answers differ.

## Why bigger QC is worse — proof: cross-core load imbalance, not extra work

The op is **work-conserving in QC**: the dense schedule computes the full `[0, Tt)`
rectangle for every q-row, and the slow "diagonal + future" masked tiles (per-tile `W=1`
untilize + `add_mask`, no fast-strip) are a **per-row** quantity. Grouping rows into a
QC-tall unit cannot change the *total* masked-tile count. The per-core profiler confirms
the **average per-core work is QC-invariant**; only the **slowest core** (which sets the
kernel duration) grows.

Per-core kernel span, cycles (one op, 110 cores, KC=8):

| config        | min   | avg   | max   | spread (max−min) | max/avg |
|---------------|-------|-------|-------|------------------|---------|
| QC=1 KC=8     | 489.6k| 491.4k| 504.9k| 15.3k            | 1.027   |
| QC=2 KC=8     | 495.5k| 497.3k| 524.9k| 29.4k            | 1.056   |
| QC=4 KC=8     | 495.1k| 496.8k| 552.6k| 57.5k            | 1.112   |

- **avg is flat (~491–497k) across QC** → total compute does not grow. ✔ work-conserving.
- **spread doubles each QC step (15.3k → 29.4k → 57.5k)** and `max/avg` climbs 1.027 →
  1.056 → 1.112. The kernel = slowest core, so the doubling imbalance *is* the slowdown.
- The slow-core **excess over avg scales linearly in QC**: 14k → 28k → 56k cyc = exactly
  1×/2×/4× of one q-row's diagonal slow-work.

### Mechanism — the diagonal slow-work stacks on the grid-tail column

The masked tiles live in the **high k-columns near Tt** (the causal diagonal sits at
col ≈ 1740 of 1760 at sp7). The 12 slowest cores, by physical coord:

- **QC=2**: 10 of 12 are physical **x=15** = logical x=10 = the grid's **last column**,
  across grid-rows y=2..11. With QC=2 the deal is *grid-aligned* (`groups==grid_y==10`),
  so core (x=10, y) owns group y's **highest k-band** — i.e. that group's whole diagonal.
  Both of the group's 2 q-rows put their diagonal slow-work on that one core.
- **QC=4**: top-5 are all x=15, now carrying **4 rows'** diagonal work, with a ~57k-cyc
  gap to the rest of the field (which sit flat at ~495k).
- **QC=1**: not grid-aligned (`groups=20 ≠ grid_y=10` → plain flat deal), so the diagonal
  units of each group fall on whatever core the 40-unit flat range happens to cover — split
  across the boundary columns (x=6 and x=15), **never stacked** → smallest spread.

So: grid-aligned QC concentrates `QC` rows of slow diagonal tiles onto a single tail core
per group. Average work is unchanged; the critical path lengthens by `QC × (one row's
diagonal slow-work ≈ 14k cyc)`. That is the entire QC penalty.

### Direct attribution — wrap the slow path in a zone

A `DeviceZoneScopedN("IDX_PTILE")` around *only* the non-fast-strip path (the W=1 untilize
prefix + the per-tile `add_mask` suffix) shows, per op, how many of the 110 cores ever
touch the slow path and where:

| QC | distinct cores running IDX_PTILE | which cores |
|----|----------------------------------|-------------|
| 1  | **20**  (= groups = Sqt/QC)      | spread (plain flat deal, x=6 & x=15) |
| 2  | **10**  (= groups)               | **all physical x=15** (logical x=10 tail column) |
| 4  | **5**   (= groups)               | all physical x=15 |

The whole diagonal band (cols 1740–1760 at sp7) lives inside the single x=10 k-band, so on
the grid-aligned deals (QC=2,4) **only the tail column executes the slow path and the other
100 cores run it exactly zero times.** The count of cores sharing the (constant) slow work
is exactly `groups = Sqt/QC` — halve the groups (double QC) and each slow core inherits 2×
the masked tiles. This is the mechanism in one line: **bigger QC = fewer groups = the fixed
pool of slow diagonal tiles is shared over fewer cores = a hotter critical-path core.**

## Why bigger KC IS better (the other axis)

KC acts on a different quantity: the **fast-strip fixed overhead**. Each full-width row is
one `W=KC` fast untilize plus one `mm_block_init` resync (~326 cyc, the half-sync re-init,
see `INDEXER_FAST_UNTILIZE.md`). Bigger KC = fewer, wider strips = that bracket amortized
over more columns, so the **average** per-core work drops:

| QC=1 | avg per-core (cyc) | max (cyc) | math_util |
|------|--------------------|-----------|-----------|
| KC=8 | 491.4k             | 504.9k    | 64.5 %    |
| KC=32| 479.0k             | 495.8k    | 65.7 %    |

avg falls 491k → 479k (−12k) going KC=8 → KC=32; imbalance is roughly unchanged (QC=1 is
not grid-aligned at any KC), so the lower average drops the max core and lifts the ceiling.
Diminishing returns + `Tt`-divisibility + L1 cap the useful KC; KC=32 is the practical best
(KC=64 leaves partial units since 1760 ∤ 64, and grows the strip CBs).

## FIX LANDED — stamp the mask via packer L1-acc on the strip (SDPA idiom)

Root cause above: a single masked tile dropped the whole row off the fast `pack_untilize` strip onto
the per-tile W=1 untilize + eltwise `add_mask` (CB pop/repush) path, and grid-aligned QC stacked that
slow work onto the tail column. The SDPA fix (`compute_common.hpp::apply_causal_mask_lightweight`):
keep the row on the strip and **stamp the −inf mask onto the accumulated strip slots in-place via
packer L1-accumulate** (`copy_tile(mask)` + `pack_tile<true>`), then untilize the whole strip fast.

Implemented in `compute_indexer_score.cpp` (`stamp_strip_mask`, called from `produce_full_strip`) and
mirrored in `writer_indexer_score.cpp`; both kernels now branch on **full unit** (`k_tiles_in_unit ==
KC`) instead of **fully-valid row**, so every row of a full unit takes the strip and the masked suffix
is just a cheap stamp. The per-tile path remains only for partial edge units / KC==1.

Results (heads8 bfp8 sp7, this board):

| metric                          | before fix | after fix |
|---------------------------------|------------|-----------|
| ceiling QC=1/2/4 @ KC=8         | 64.5/62.1/59.0 % | **65.9/65.7/65.5 %** (QC-flat) |
| ceiling QC=1 KC=32 (best)       | 65.7 %     | **68.0 %** |
| per-core spread (DMA off, QC=2) | 29.4k cyc  | **2.68k cyc** (max/avg 1.056→1.005) |
| full kernel QC=2/KC=8 + mcast   | 0.430 ms / 56.1 % | **0.416 ms / 58.1 %** |

The QC penalty is gone (ceiling is now ~flat in QC) and the overall ceiling rose ~2 pts. Correctness:
25/25 accuracy tests pass (exact −inf map + PCC at the full multi-core shape); no perf regression at
heads16/64 (matmul-bound, unchanged). Note QC=2/KC=8 stays the production config (it enables the
grid-aligned K multicast that wins the DMA-on, reader-bound kernel); the fix just removes its compute
penalty.

## One-line summary

- **avg per-core work** = f(KC), independent of QC → raise KC.
- **slowest-core excess** = QC × (per-row diagonal slow-work), via grid-tail stacking →
  keep QC=1 for the ceiling.
- Compute-ceiling optimum: **QC=1, KC=32 (65.7 %)**. Production ships QC=2/KC=8 only because
  the DMA-on kernel is reader-bound, where QC=2's K-reuse outweighs the −3.6-pt ceiling hit.

## Reproduce

A temporary env-gated override in `production_config()` (`tests/.../test_indexer_score.py`)
reads `INDEXER_SWEEP_QC` / `INDEXER_SWEEP_KC` (in **tiles**); it is a no-op when unset, so
committed defaults are unchanged. Remove it after the study.

```bash
# sweep one cell (math_util + ms):
INDEXER_DMA_OFF=1 INDEXER_SWEEP_QC=1 INDEXER_SWEEP_KC=32 \
  scripts/run_safe_pytest.sh \
  tests/nightly/blackhole/sdpa/test_indexer_score.py::"test_indexer_score_sp7_math_util[heads8_k_bfp8]"

# per-core spread (the imbalance proof): tracy → profile_log_device.csv, then the awk in
#   /tmp/percore.awk (per (op,core): span = max(KERNEL end) − min(KERNEL start)).
flock /tmp/tt-device.lock python -m tracy -r -p -o generated/profiler/idx_qc2 \
  -m "pytest .../test_indexer_score_sp7_perf_impl[heads8_k_bfp8]"   # with the env vars set
```
