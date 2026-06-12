# indexer_score — grid-aligned scheduling & multicast

How the dense output deal maps to the physical Tensix grid, why that mapping makes cores in a grid
**row** share Q/W and cores in a grid **column** share K, and how the reader multicasts each input
once per line instead of once per core. Companion to `INDEXER_DATAMOVEMENT.md` (the reader/writer
bandwidth analysis that motivated this) and `INDEXER_OP.md` (op design). All concrete numbers are
the GLX production shape, sp_rank 7, on the 11×10 (=110) Blackhole worker grid.

## 1. The work-unit deal (recap)

The op is output-stationary. The output is `Sqt × Tt` score tiles (production: **20 × 1760**). It is
carved into **work units** of `QC × KC` tiles (the atomic piece dealt to one core), and the units
are dealt **flat, row-major** across cores: unit 0 = (group 0, chunk 0), then chunk 1 … across
group 0, then group 1, … Core `i` gets a contiguous run `[i·base, i·base+count)` where
`base = V/num_cores`, the first `rem = V%num_cores` cores get one extra.

- `groups = Sqt/QC`, `units_in_group(g) = ceil(vmax(g)/KC)`, `V = Σ units_in_group`.
- **Dense schedule** (`dense_schedule = true` in `indexer_score_work_split.hpp`): `vmax(g) = Tt` for
  every group, so the full rectangle is dealt and `V = groups · ceil(Tt/KC)`. Future (−∞) tiles are
  computed + masked in-band; the writer's tail-fill becomes a no-op. This makes the deal **uniform**
  (no causal triangles to balance). See `INDEXER_OP.md` / `INDEXER_DATAMOVEMENT.md`.

Production (QC=2, KC=8): `groups=10`, `units_in_group=220`, **V=2200**, `num_cores=110`, `base=20`,
`rem=0` → every core gets exactly **20 units = 320 output tiles**.

## 2. Grid alignment — the key property

Cores enumerate **row-major** (`num_cores_to_corerangeset(..., true)`), so logical core index
`i = y·grid_x + x`, i.e. core `(x, y)` with `x ∈ [0,11)`, `y ∈ [0,10)`. With the production
numbers the flat deal lands **each group on exactly one grid row**:

```
core (x, y)  →  units [i·base, i·base+base)  with i = y·11 + x
             →  group = i·base / units_in_group = y          (q-rows [2y, 2y+2))
             →  chunks [x·base, x·base+base) within the group  (k-cols [160x, 160x+160))
```

So **core (x, y) owns q-rows `[2y, 2y+2)` × k-cols `[160x, 160x+160)`** — a clean 2×160-tile block.
Verified across all 110 cores. Two consequences:

- **Same grid ROW `y` (vary x) ⇒ same q-rows ⇒ same Q (and W).** Q/W can be shared **along a row**.
- **Same grid COLUMN `x` (vary y) ⇒ same k-cols ⇒ same K band.** K can be shared **down a column**.

### Why it lands perfectly (the alignment conditions)

It is not luck — the dims factor onto the grid:

- `groups == grid_y` ⟺ `QC == Sqt/grid_y` (production 20/10 = 2). → group `g` is grid row `g`.
- `units_in_group == grid_x · base` with `rem == 0` ⟺ `grid_x | (Tt/KC)`, i.e. `KC | (Tt/grid_x)`
  (Tt/grid_x = 1760/11 = 160; KC=8 | 160). → the row's `grid_x` cores split the group's chunks
  into equal contiguous k-bands.

The factory's eligibility gate is exactly:
`dense_schedule && groups==grid_y && num_cores==grid_x*grid_y && rem==0 && units_per_group==grid_x*base`.
Cases that fail it (e.g. **heads64 QC=1** → 20 groups ≠ grid_y=10) keep the plain flat deal and read
every input from DRAM (no mcast) — correct, just no sharing.

### Per-core load is invariant to the knobs

`tiles/core ≈ QC·Σvmax / num_cores ≈ 320` regardless of KC (because `V·KC ≈ Σvmax` is fixed). KC
only sets the **column quantum** (where core boundaries fall, snapped to KC-multiples) and the unit
count `V`; QC sets which **rows** group together and (via `groups==grid_y`) the row alignment. Too
large QC·KC can drop `V` below 110 and leave cores idle.

## 3. Multicast — read each input once per line

Each input is read from DRAM by **one sender** per line and multicast L1→L1 to the line's
**receivers**, so DRAM traffic for that input drops ~line-length fold:

| input | shared over        | sender      | receivers      | redundancy removed |
|-------|--------------------|-------------|----------------|--------------------|
| **K** | grid column (10)   | `(x, 0)`    | `(x, 1..9)`    | ~10× (down column) |
| **Q** | grid row (11)      | `(0, y)`    | `(1..10, y)`   | ~11× (along row)   |
| **W** | grid row (11)      | `(0, y)`    | `(1..10, y)`   | ~11× (along row)   |

K is read **per unit** (each column core processes the same 20 k-chunks in the same order — same
k-cols, different q-rows — so they are lockstep on chunk index). Q/W are read **once per group** at
group start (each core does one group). The two directions are **independent** (decoupled): a core
can be a K sender and a Q receiver, etc. Roles: `(0,0)` is both K-sender (col 0) and Q-sender
(row 0); `(0,y>0)` is Q-sender (row y) + K-receiver; `(x>0,0)` is K-sender (col x) + Q-receiver;
`(x>0,y>0)` is a pure receiver.

### Handshake (mirrors SDPA `chain_link`)

Three semaphores per direction — `send`, `recv`, `valid` (init 0, 0, 1):

- **Receiver**: reserve the CB slot (→ `addr`); `recv.set(0)`; remote-inc the sender's `send` by 1;
  `recv.wait(1)`; push.
- **Sender**: read the block from DRAM into its CB slot (→ `addr`); `send.wait(num_dests)`;
  `send.set(0)`; `async_write_multicast` the block to the receiver rectangle (`.addr = addr`, the
  **same** L1 offset on every core — homogeneous program + lockstep rings); `valid.relay_multicast`
  into the receivers' `recv` (back-to-back after the linked data write, **no flush between**);
  `async_writes_flushed()`; push.

Receiver sets `recv=0` **before** signaling, and the sender relays valid **after** `send.set(0)` and
**after** all `send` increments — so there is no lost-update or stale-valid race. The slot address
matches because all cores have identical CB layouts and advance their rings in lockstep.

### Eligibility & fallback

A direction is enabled only if every one of its lines maps to a **contiguous NoC rectangle**
(K: shared phys-x, contiguous phys-y; Q/W: shared phys-y, contiguous phys-x). Q/W additionally
needs all heads resident (`HB == Hi`, one q-block). If a line isn't a clean rectangle (harvested
grid) the direction falls back to **per-core DRAM reads**. On the BH board here the logical→physical
map is contiguous, so mcast is always eligible. (A **unicast forwarding chain** for non-rectangle
lines is *not* implemented — it would be dead/untestable on this clean grid; the DRAM fallback
stands.)

Reader roles are runtime args (`0`=DRAM read, `1`=sender, `2`=receiver) + rectangle + sender coord;
the on/off flags and 6 semaphore ids are compile-time args. Env A/B kill-switches:
`INDEXER_NO_KMCAST`, `INDEXER_NO_QMCAST`. Mcast is auto-disabled under the `INDEXER_DMA_OFF` ceiling
diagnostic (so that stays a pure compute measurement).

## 4. Results (sp7, tracy device-kernel min)

| config        | dense, no mcast | + multicast | win   | math_util |
|---------------|-----------------|-------------|-------|-----------|
| heads8  bfp8  | 0.480 ms        | **0.430**   | 10.4% | 50.3 → **56.1%** |
| heads16 bf16  | 0.866 ms        | **0.794**   | 8.3%  | 55.8 → 60.8% |
| heads16 bfp8  | 0.826 ms        | **0.794**   | 3.9%  | 58.5 → 60.8% |
| heads64       | ~2.9 ms         | unchanged   | —     | not grid-aligned (no mcast) |

**K mcast is the entire win** (heads8: K-only 0.432, both 0.430). **Q/W mcast is net-neutral**:
once K stops being the bandwidth bottleneck, the once-per-group q/w reads already hide under
compute, so sharing them saves nothing measurable (Q/W-only = 0.481 ≈ baseline). Q/W mcast is kept
(correct, decoupled) but is not a perf lever — this corrects the earlier prediction (which assumed
q/w were exposed) and supersedes the old "parallel-pull q/w share regressed 3.7×" finding: the real
fix was the grid-aligned clean-rectangle geometry + true multicast.

The production case sits at **0.430 ms / 56.1% util**, vs the **~66.8% compute ceiling** (DMA off).
The remaining ~11 pts are irreducible data movement (one core per column still reads K from DRAM and
multicasts it) + fixed per-core costs.

### Provenance of the production config knobs

`QC=2` (=Sqt/grid_y, also the K-bandwidth knee), `KC=8` (divides Tt/grid_x=160; load-balances units),
`HB=0` (all heads resident — head streaming is ~24× slower). These are chosen by the **caller**
(`production_config(heads)` in the test) — the factory no longer auto-tunes; it only rejects configs
whose CBs overflow L1 (per-core L1 1464 KiB − 320 KiB reserve). Production L1 footprint ≈ **522 KB**
(cb_q 128 + cb_qk 128 + cb_k 68 + cb_acc 64 + cb_w 32 + out/strip 98 + misc).

## 5. How to reproduce / measure

```
# perf (needs profiler build = default build_metal.sh):
scripts/run_safe_pytest.sh --run-all \
  tests/nightly/blackhole/sdpa/test_indexer_score.py::test_indexer_score_sp7_math_util
# A/B a direction:
INDEXER_NO_KMCAST=1  ...::test_indexer_score_sp7_math_util -k heads8    # Q/W only
INDEXER_NO_QMCAST=1  ...                                                # K only
INDEXER_NO_KMCAST=1 INDEXER_NO_QMCAST=1 ...                             # dense baseline
# accuracy (mcast exercised by grid-aligned heads16; fallback by heads64/corner/qc2):
scripts/run_safe_pytest.sh --run-all tests/nightly/blackhole/sdpa/test_indexer_score.py \
  -k "production or bfp8_k or knobs or corner_shapes or multicore_qc2 or glx_chunked or invalid"
```

Interactive: `indexer_core_map_viz.html` colors the full 20×1760 output by owning Tensix (with a
dense/causal toggle) — each core's clean 2×160 block makes the row=Q / column=K sharing visible.

## 6. Status / what's left

- [x] dense schedule, grid-aligned deal, **K-column + Q/W-row multicast** (decoupled), committed.
      Accuracy 42/42; heads8 production 0.480 → **0.430 ms** (10.4%).
- [ ] **unicast fallback** for non-rectangle (harvested) lines — currently DRAM-read fallback.
- [ ] deeper K pipeline (cb_k > 2) now that K isn't bandwidth-bound — may hide the per-chunk
      handshake latency on the column sender; untried (L1 budget + risk).
- [ ] heads64 (QC=1) doesn't grid-align (20 groups ≠ 10 rows); a QC=2 64-head config overflows L1,
      so it can't use this scheme as-is.
