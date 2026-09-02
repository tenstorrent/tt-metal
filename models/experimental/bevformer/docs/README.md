# BEVFormer on Tenstorrent — what was done, why, and what is left

This is the entry point for the BEVFormer optimization work on branch `ctr-mmicic/bev-former`.
It is written to be read start to finish by someone who has not touched this code.

- **Measured results per stage:** [`PERF.md`](PERF.md) (Wormhole) and the table below (Blackhole).
- **One report per landed change:** [`perf_reports/`](perf_reports/) — `00` through `10`.
- **Backlog of untried ideas:** [`perf_optimization_candidates.md`](perf_optimization_candidates.md).

---

## 1. What the model does, in one paragraph

BEVFormer turns six camera images into a top-down "bird's eye view" map. The expensive part is the
**encoder**, six identical layers stacked. Each layer runs two attentions:

- **TSA** (temporal self-attention) — the BEV grid attends to its own previous frame.
- **SCA** (spatial cross-attention) — each BEV cell projects itself into all six camera images and
  samples the image features at those points.

Both are built on **MSDA** — *multi-scale deformable attention*. MSDA does not attend to everything.
For each query it predicts a handful of `(x, y)` sampling points, reads the feature map at those
points with bilinear interpolation, and combines them with learned weights. "Multi-scale" means it
does this on four feature-map resolutions (*levels*) and sums the results.

That sampling step — thousands of small, scattered, unpredictable DRAM reads — is where nearly all
the time goes, and it is what almost every change below is about.

---

## 2. How performance is measured here

**Harness:** [`tests/perf/test_layer_perf.py`](../tests/perf/test_layer_perf.py) — **one** encoder
layer, config `nuscenes_base`, `bev_size=(100, 100)`, batch 1, 4 levels, 6 cameras.

One layer and not all six, because the 6-layer harness emits more device ops than Tracy's per-device
buffer holds and the report comes back truncated. The layer is the repeated unit anyway, so
encoder ≈ 6 × layer + one point-sampling pass.

**Correctness gate:** PCC ≥ 0.997 against the torch reference, asserted inside the perf test itself.
PCC ("Pearson correlation coefficient") is the standard similarity check here — 1.0 is identical.
On top of that, `tests/pcc/` and the MSDA op suite must pass. **No perf number in this document
was accepted from a run that failed its gate.**

### Which column to trust

The profiler CSV gives two per-op durations, and the difference between them matters:

| column | meaning |
|---|---|
| `DEVICE KERNEL DURATION` | the op's kernels actually running |
| `DEVICE FW DURATION` | the firmware window around them — includes time a core sat waiting |

**Summing `FW DURATION` over-counts, because the FW windows of consecutive ops overlap.** A core
that finishes its part of op N early enters op N+1's firmware immediately and waits there for the
slower cores to finish op N. That wait is inside *both* ops' FW windows and gets counted twice.

Measured on the current build: summed FW is **139.08 ms**, summed kernel is **101.43 ms** — the
37.65 ms difference is almost entirely this double count. **Use `DEVICE KERNEL DURATION`.** Section
7.1 explains why the gap is so large; it is a real problem, just not 37 ms of real *work*.

### What one CSV actually contains

Do not read the CSV total as "one layer". The harness compiles, warms up, then runs the measured
iteration, so **the layer body appears three times**, and point sampling — which the encoder runs
once for all layers, not once per layer — appears once. Split by signpost:

| segment | kernel |
|---|---:|
| point sampling, once | 9.31 ms |
| layer body × 3 | 30.67 / 30.73 / 30.73 ms |
| CSV total | 101.43 ms |

**One layer is 30.7 ms.** A six-layer encoder is `9.31 + 6 × 30.7` ≈ **193.6 ms** of kernel. Those
two are the numbers worth quoting; the CSV total is an artifact of the harness.

---

## 3. Timeline — Wormhole phase (2026-08-25 → 08-28)

Full numbers and per-stage reasoning live in [`PERF.md`](PERF.md). Summary:

| # | date | author | commit | change | kernel |
|--:|---|---|---|---|---:|
| 0 | 08-25 | Milos Micic | `f0977be8102` | baseline + perf harness | 655.6 ms |
| 1 | 08-26 | Milos Micic | `4048ef2bbf1` | SCA rebatch and scatter-back moved onto the device | 682.0 ms |
| 2 | 08-27 | Ilija Kasic | `828c3315149` | MSDA through the fused ttnn op | 487.4 ms |
| 3 | 08-27 | Ilija Kasic | `65ae83ff009` | camera fold without tiling a batch-of-one | 450.6 ms |
| 4 | 08-27 | Ilija Kasic | `87e2e9f2de0` | flat, tile-clean sampling chain | 310.1 ms |
| 5 | 08-27 | Ilija Kasic | `3ebd77d25c0` | hoisted head permute, untilize before the head split | 263.6 ms |
| 6 | 08-28 | Ilija Kasic | `192cdf21916` | sampling geometry moved onto the SFPU | 125.3 ms |
| 7 | 08-28 | Ilija Kasic | `1bc7a7c997b` | grid's point axis folded into its page | 112.4 ms |
| 8 | 08-28 | Ilija Kasic | `a63ca3582c7` | value heads addressed by byte offset | 91.9 ms |
| 9 | 08-28 | Ilija Kasic | `978fc933566` | attn level runs addressed by byte offset | 74.0 ms |
| 10 | 08-28 | Ilija Kasic | `96aa157b2a6` | rank-3 grid packing head and level | 69.3 ms |

**Stage 1 was the single biggest win** and it was not a kernel change. SCA was rebatching its
per-camera work on the *host*, which meant the device idled waiting for Python between ops. Moving
that on-device took wall clock from 3072 ms to 900 ms — a −70.7% cut where device time barely moved.
The lesson is at the top of the list in section 5.

**Stages 2–10 cut kernel time by 89.8%**, from 681.5 ms to 69.3 ms, with the PCC gate held at 0.9996
throughout.

### The three ideas those stages keep re-using

**Stage 6 — do float maths where the float hardware is.** A Tensix core has five RISC-V cores. The
two *dataflow* cores (BRISC, NCRISC) move data and **have no FPU**. The compute cores (TRISC) reach
the SFPU, a vector float unit. The MSDA reader was computing per-point sampling geometry on a
dataflow core at roughly 140 cycles per operation, and the compute kernel sat idle the whole call
waiting for it. Moving that maths to the SFPU collapsed the op from 167.8 ms to 29.5 ms **without
touching the sampling kernel at all**. Stage 5 had read the same op as "78× above its DRAM roof" and
concluded the sampling was slow. It was not. *Measure which unit is idle before deciding what is
slow.*

**Stages 7–10 — an axis you only ever index is an address, not data.** The original code physically
rearranged tensors so each head/level/point sat in its own tensor. That is a full copy of the data
through DRAM. But the reader already computes a page id and a byte offset for every read — so a head
is just `head_index * head_bytes` added to that offset. Four separate stages (grid point axis, value
heads, attn level runs, packed grid) replaced copies with arithmetic. Stage 8 alone deleted **92.6
MB** of DRAM traffic per layer.

**Stages 3–4, 7 — a ROW_MAJOR page is the last dimension, rounded up to the DRAM alignment.** A
tensor whose last dimension is 2 `bfloat16` values has 4 bytes of data in a 32-byte page (64 on
Blackhole). It moves at 2 GB/s against a 288 GB/s roof — **not because it is big, but because its
rows are narrow.** Sorted by measured bandwidth, the layout ops sorted exactly by page width:
512 B → 38 GB/s, 64 B → 14 GB/s, 4 B → 2 GB/s. TILE_LAYOUT has the mirror-image version of this
problem: trailing dims pad up to 32, so a `(4, 2)` tail becomes `(32, 32)` — **128× its own data**.
*Padding and page width are separate faults with opposite fixes.*

---

## 4. Timeline — Blackhole phase (2026-08-31 → 09-01)

The work moved to a Blackhole card. **Nothing in stages 0–10 had been validated there**, and the
first result was that the op was completely broken on it.

| date | commit | change |
|---|---|---|
| 08-31 | `341d3cb00bd` | take the NoC read alignment from the arch, not from Wormhole |
| 08-31 | `af602149ddd` | scatter the SCA aggregation back with `embedding_bw` |
| 09-01 | `ea02d94be10` | keep the point axis folded through point sampling |
| 09-01 | `a32e9b8ae38` | scale and validate point sampling at full width |
| 09-01 | `d023e7a7581` | let the unpacker tilize the gathered value block |
| 09-01 | `6d0da1c747f` | barrier once per point instead of once per corner |

### 4.1 The Blackhole breakage — `341d3cb00bd`

**Symptom:** 128 of 128 correctness tests failing on Blackhole, PCC ≈ 0. On Wormhole, all green.

**Cause:** stage `38263c1ed1c` had the reader gather value data straight into tile *face halves* —
32-byte pieces. A tile is 32×32, stored as four 16×16 faces of 512 bytes; a face half is 32 bytes.
That works on Wormhole, where DRAM alignment is 32 bytes. **On Blackhole DRAM alignment is 64
bytes, and the transfer is 64-byte granular in both address and size.** A 32-byte read is not a slow
read there — it is an illegal one.

**Found by:** `git bisect`. Commit `5c5e26a8204` gave 131 passed; `38263c1ed1c` was the first bad
one.

**Fix:** stop hard-coding 32 and ask the hardware abstraction layer:
`tt::tt_metal::hal::get_dram_alignment()`. The face-half gather itself was removed entirely later,
by `d023e7a7581`.

### 4.2 The largest Blackhole win — `d023e7a7581`

The reader had been placing gathered rows into tile face halves by hand — a scalar loop on a
dataflow core, doing byte shuffling that the **unpacker** does in hardware for free.

Now the reader writes the sticks down contiguously, one row per query, and the compute kernel calls
`compute_kernel_lib::tilize<...>()`
([`ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`](../../../../ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp)),
which drives the unpacker. This both removed the scalar work **and** made the Blackhole alignment
problem disappear, since a whole row is 64-byte legal.

This change came directly out of a one-line instruction: *if something that is not compute is
spending time, that means something is wrong and must be fixed.* It was worth more than any
micro-optimization tried that day.

### 4.3 The last change — `6d0da1c747f`

Bilinear interpolation needs four corners per sampling point. The reader was issuing 32 DRAM reads
for one corner, then **waiting** (`noc_async_read_barrier`), then doing the next corner. Four full
DRAM round trips per point.

Investigating why some cores were slower than others produced this, measured per core row `y`:

| y | issue loop | barrier |
|---|---:|---:|
| 2 | 1.09 ms | **1.69 ms** |
| 5 | 1.05 | 1.20 |
| 8 | 0.83 | 0.56 |
| 11 | 0.81 | **0.36** |
| spread | 1.35× | **4.7×** |

The **barrier** carries the variation, the issue loop does not. Slow cores are *waiting for reads to
land*, not stalling while posting them — read latency, not backpressure. Latency varies by row
because Blackhole's DRAM sits in columns (x = 0 and x = 9) shared by all 110 worker cores, and a
core's distance along that shared column depends on its `y`.

Latency cannot be removed, but it can be paid less often. All four corners are now reserved and
issued as one group under a single barrier. `input_rm_cb` is 8 blocks deep so a 4-block reservation
never wraps mid-group. **MSDA: 3965 → 3828 µs per call.**

### 4.4 Blackhole results

Same test throughout, 318 device ops, 15 MSDA calls, all runs green.

Kernel time, split by signpost as in §2:

| | point sampling | **one layer** | **encoder (6 layers)** |
|---|---:|---:|---:|
| pre-optimization code (stages 0–10 reverted, measured 09-01) | 22.72 ms | **104.65 ms** | **650.6 ms** |
| start of the Blackhole phase | 14.03 ms | 35.75 ms | 228.5 ms |
| **now** | **9.31 ms** | **30.71 ms** | **193.6 ms** |

**Total speedup: 3.41× per layer, 3.36× per encoder.** Of that, roughly 2.9× came from the Wormhole
stages and a further 1.16× from the Blackhole phase.

The "pre-optimization" row is a real measurement, not an estimate — the stages were reverted and the
old code was run on the same card, same test.

---

## 5. The principles, extracted

These are the things worth remembering; the individual commits are just applications of them.

1. **Find which unit is idle before deciding what is slow.** Stage 5 blamed the sampling kernel;
   the compute kernel was idle waiting on a reader without an FPU (§3, stage 6).
2. **An axis you only index is an address, not data.** Replace copies with offset arithmetic
   (§3, stages 7–10).
3. **A ROW_MAJOR page is the last dimension.** Narrow rows are slow regardless of tensor size.
   Padding (TILE) and page width (ROW_MAJOR) are separate faults with opposite fixes (§3).
4. **Host round-trips cost more than kernels.** The single biggest win in the project moved work
   on-device without making any kernel faster (§3, stage 1).
5. **If something that is not compute is spending time, something is wrong.** This is what produced
   §4.2.
6. **Ask the hardware, do not hard-code the architecture.** `hal::get_dram_alignment()`, not `32`
   (§4.1).
7. **Sum only what does not overlap.** Concurrent zones on three RISCs do not add up to op time
   (§2, and §8).

---

## 6. Current state

Everything below is committed on `ctr-mmicic/bev-former`, tests green:

- MSDA op suite (`tests/ttnn/unit_tests/operations/experimental/test_multi_scale_deformable_attn.py`)
  — passing, job 038
- `tests/pcc/` for SCA and point sampling — 15 passing, job 038
- perf gate PCC 0.999590

Where the time goes now on Blackhole:

| | |
|---|---|
| one encoder layer | 30.71 ms |
| six-layer encoder | ~193.6 ms |
| pure layout ops | 22.7% of kernel |
| MSDA op, per call | ~3.5–3.8 ms |
| — fastest core finishes at | 1565 µs |
| — slowest core finishes at | **3796 µs** |
| — compute active | ~42% |

---

## 7. What is left, ranked by measured evidence

### 7.1 Core imbalance inside MSDA — the largest item

Same op, same work per core: **the fastest core finishes at 1565 µs, the slowest at 3796 µs.** The
op takes as long as the slowest. If all cores finished at the average (~2505 µs), each call would be
**~1.25 ms shorter — a 33% cut.**

This was proven by reading per-core firmware timestamps out of `profile_log_device.csv`:

| | FW start (min→max) | kernel start | kernel end (min→med→max) |
|---|---|---|---|
| MSDA (idx 254) | 0.0 → 6.4 µs | 5.8 | 1564.6 → 2505.1 → **3796.3** |
| MSDA (idx 255) | 0.0 → **2231.7** | all at 2231.1 | 3778.9 → 4701.2 → **5967.8** |
| add (idx 256) | 0.0 → **2188.9** | all at 2188.2 | 2350.5 → 2515.7 → 2659.0 |

Op 254 finishes across a **2231.7 µs** spread; op 255's cores enter firmware across exactly
**2231.7 µs** and all its kernels then start together. The numbers match to the decimal. **The idle
time reported against one op is the previous op's imbalance.** This is also the whole explanation
for two things that looked like separate bugs: an "add that is 5× slower than the identical add
before it" (its kernel is a constant 470 µs) and "two MSDA calls that are 60% slower" (their kernels
are a normal 3737 µs).

**Cause:** the DRAM read latency gradient in §4.3. Cores on far rows read slower.

**Proposed fix — self-scheduling, not a weight table.** Round-robin tile dealing was tried and
reverted (18.36 → 18.15 ms, within noise): it equalizes *how many* tiles a core gets, not *how fast
that core reads*. A per-row weight table would work but hard-codes a topology assumption that breaks
on another device or harvesting configuration. Instead, put a single atomic counter in L1 and have
each core claim its next tile index via `noc_fast_atomic_increment`. Fast cores naturally take more.
Topology-agnostic, self-balancing, and one atomic per tile over ~30–60 tiles per core is negligible
against 3.8 ms.

### 7.2 Fold the level axis into the op

MSDA is currently called **once per level** (4 calls), with `value` sliced per level by
`ttnn.split`, and the four outputs summed on the host side:

```
4 × SliceDeviceOperation          0.57 ms
4 × MSDAOperation                14.4  ms
3 × BinaryNg (accumulate)         1.4  ms   ← kernel time; see the warning below
```

The compute kernel **already accumulates** across `reduction_size = 4 * P` groups using
`pack_reconfig_l1_acc(1)`. Levels are the same loop. Taking all levels in one call would remove the
4 slices, the 3 adds, 3 of 4 kernel launches, and 3 write-then-read-back round trips through DRAM.

**Estimated ~2 ms per SCA call.** The one real design obstacle: `h_in` / `w_in` are compile-time
arguments today, and each level has different dimensions, so the kernel needs a per-level geometry
table. Everything else is offset arithmetic.

> **Correction on record.** This item was first sized at ~5.8 ms, from summing `FW DURATION` on the
> three accumulate ops (482 + 2405 + 2377 µs). Their *kernels* are 470 µs each. The rest was
> MSDA's imbalance tail (§7.1) billed to the next op. The real figure is ~2 ms.

### 7.3 The prep chain before MSDA

Between the `MS Deformable Attn Module Start` signpost and the MSDA op there are 21 ops totalling
3.81 ms, of which **2.40 ms is pure layout** — 3 reshapes (0.96), 2 untilizes (0.67), 4 slices
(0.57), and repeat/tilize (0.20). The op's own cost is 3.74 ms, so **preparation costs as much as
the operation.**

A dedicated "prep op" is the **wrong** fix: fusing several layout ops into one still writes an
intermediate to DRAM, and the point is that the intermediate should not exist. The right fix is the
same one used four times already (§3, stages 7–10) — absorb the addressing into the reader.

### 7.4 Smaller and unverified

- **`writer_msda.cpp`** still does a scalar face-half gather into `output_scratch_cb` with a
  flush per row. Measured at 1.12 ms (6%) — but that was **before** the tilize rework, so re-measure
  before acting.
- **`input_rm_cb` depth stall.** A 0.40 ms/call stall was measured when the CB was 2 deep. It is now
  8 deep (`6d0da1c747f`); assume nothing, re-measure.

### 7.5 Documentation and validation debt

- **[`PERF.md`](PERF.md) is entirely Wormhole numbers.** There is no Blackhole column. §4.4 of this
  file is currently the only record of the Blackhole results.
- **[`perf_reports/06-sfpu-geometry.md`](perf_reports/06-sfpu-geometry.md) still claims "both halves
  are 32-byte aligned"** — that claim is exactly what broke Blackhole (§4.1). It needs a correction
  note.
- **No Blackhole reports exist** for the six commits in §4. Stages 0–10 each have one; 11–16 do not.
- **Wormhole has not been re-run** since any of the Blackhole-phase changes. They are all expected to
  be neutral or positive there, but that is an expectation, not a measurement.

---

## 8. Traps encountered, so they are not re-encountered

- **Never sum concurrent zones against an op's duration.** Zones on three RISCs run at the same
  time. Doing this once "found" 11 ms that did not exist; a whole-body sanity zone showed the reader
  body was 2.83 ms of a 3.99 ms op. A mean had been compared against a max.
- **`pack_reconfig_l1_acc` is global to the packer.** It affects *every* pack including the one
  inside `tilize`. Leaving accumulate armed across a tilize makes it accumulate into its own input
  buffer. Cost: 154 failing tests until spotted.
- **`TensorAccessorArgs<N>` must track the compile-time arg count.** Removing one CT arg and leaving
  `<28>` instead of `<27>` gives PCC 0 with no other symptom.
- **A partial-page DRAM read on Blackhole is illegal, not slow** (§4.1).
- **Reading past a page end hangs the chip off the PCIe bus.** Applying a `point_offset` to an
  unpacked grid did this; recovery needed `tt-smi -r`. An offset is only valid on an input that is
  actually packed.
- **Do not pipe device jobs through `| tail`.** Output gets buffered so the broker reaps the job as
  hung, and the pipeline's exit code is `tail`'s, so `&&` stops gating on the test result.

### Dead ends, established by measurement rather than argument

- **Sharding** — the op is not latency-bound in the way sharding helps; latency is 3% of it.
- **`async_read_with_state`** — requires the same NoC location for every read (only the low 32 bits
  of the address are passed). The pages here are bank-scattered.
- **Merging adjacent corner reads** — consecutive page ids land on different DRAM banks.
- **Removing `volatile`** — no measurable change.
- **Round-robin tile dealing** — flattens the column variation, leaves the row gradient (§7.1).

---

## 9. Reproducing the measurements

Device runs go through `tt-device-mcp`, never bare `pytest`.

```
# correctness
pytest tests/ttnn/unit_tests/operations/experimental/test_multi_scale_deformable_attn.py
pytest models/experimental/bevformer/tests/pcc/

# device profile
TT_METAL_PROFILER_SUM=1 python tools/tracy/profile_this.py \
    -c 'pytest models/experimental/bevformer/tests/perf/test_layer_perf.py'
```

Output lands in `generated/profiler/reports/<timestamp>/`:

- `ops_perf_results_*.csv` — one row per op. Use `DEVICE KERNEL DURATION`, not `DEVICE FW DURATION`
  (§2).
- `profile_log_device.csv` — one row per zone marker per core. This is where per-core start/end
  timestamps live, and it is the only way to see the imbalance in §7.1.

**Kernel `.cpp` files are JIT-compiled from the source tree** — editing one needs no rebuild.
Host-side files (program factory, device operation) do need `./build_metal.sh`.
