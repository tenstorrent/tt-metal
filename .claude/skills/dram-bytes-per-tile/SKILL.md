---
name: dram-bytes-per-tile
description: >
  Measure how many bytes a TTNN op reads from DRAM per input tensor (role), on a SPECIFIC
  input shape, empirically — by temporarily instrumenting the reader kernel with per-role byte
  counters and DPRINT (NOT the profiler, which silently drops bursty reads). Produces, per input
  arg, both reads-per-output-byte and read-redundancy (reads / unique-input-bytes ≈ 1 ideal,
  > 1 = the same input re-read from DRAM many times). Use to find redundant-DRAM-read
  optimization opportunities (cross-core forwarding / multicast / input reuse) before proposing
  one, or to validate one. DRAM traffic is shape- and work-distribution-dependent — measure per
  shape. Temporary edits are journalled and reverted.
---

# dram-bytes-per-tile

Empirically measure the **DRAM-sourced read bytes per input tensor** for a TTNN op **on one
input shape**, then normalize two ways so the number points at an optimization. The measurement
is done by adding tiny temporary counters to the op's reader kernel and printing them with
`DPRINT` — deliberately **not** the profiler (see *Why DPRINT, not the profiler*).

## Unit of measurement: one test cell

DRAM read traffic is a property of **this op × this shape/config × this grid** — not of the op
in general. So one measurement = **one test cell**:

- The agent (oriented by this skill) supplies **one command that runs the op exactly once** on
  the target shape/config — a single pytest node id, or a small "run this op once on shape X"
  script. For an op without such a driver, write a minimal one (invoke the op once; correctness
  check off).
- One run → one per-role table for that cell.
- Changing anything that alters work distribution — shape, chunk/block parameters, dtype, grid,
  or the parallelization scheme — is a **different cell**; re-measure.

## The metric

For each **input tensor role** (every DRAM input the op reads: e.g. the activations, the
weights, a second operand, a mask, …), from the measured `dram_read_bytes(role)`:

```
reads_per_output_byte(role) = dram_read_bytes(role) / total_output_bytes
read_redundancy(role)       = dram_read_bytes(role) / unique_input_bytes(role)
```

- `dram_read_bytes(role)` — **measured** (summed over all cores from the DPRINT counters).
- `total_output_bytes`, `unique_input_bytes(role)` — **static**, computed from the tensor shapes
  and dtypes (`unique_input_bytes(role)` = that input tensor's full size = read-once cost).

Read both columns, per role:

- **`read_redundancy`** is the **detector**. `≈ 1.0` = the tensor is read from DRAM exactly once
  (no waste). `> 1.0` = it is re-read from DRAM N times — across cores that share it, and/or
  across a core's own work-units. That surplus is the optimization target.
- **`reads_per_output_byte`** is the **impact / magnitude** — how much DRAM read traffic this
  tensor costs per unit of useful output. It ranks which tensor to attack first, and is
  comparable across ops and shapes.

Report a **table with one row per input arg**, plus an aggregate `total reads/output`. Always
report per-role — an aggregate alone hides which tensor carries the redundancy.

> Note: when the output happens to be the same size as one input tensor, that input's
> `reads_per_output_byte` will numerically equal its `read_redundancy`. That is a coincidence of
> equal denominators, not a duplicated column — the two still mean different things.

## Why DPRINT, not the profiler

The Tracy NoC-event profiler (`--collect-noc-traces`) **silently drops** a busy core's bursty
DRAM reads: it is drop-don't-stall by design (stalling would perturb the timing it measures), and
it additionally refuses to flush while a linked multicast transaction is in flight. The cores
doing the heaviest reads are exactly the ones whose reads vanish — so a trace-based byte count is
untrustworthy for this signal, and no runtime knob fixes it. A single per-core counter emitted
**once** at kernel end is immune (it is one marker, not thousands). **The profiler and DPRINT are
mutually exclusive** — this measurement runs with no profiler.

## Prerequisites

- A source build (DPRINT is only supported on source builds).
- A way to **invoke the op exactly once on the target shape** — a pytest node id or a small
  script. This is the only op-specific input beyond the kernel edits.
- A device (the op's `device` fixture auto-opens the default one). Never reset it.
- Read `docs/source/tt-metalium/tools/device_print.rst` for the DPRINT API and env vars.

## Procedure

### 1. Find the reader kernel and the per-role DRAM-read sites

Locate the op's **reader / data-movement-in kernel(s)** (the ones that `noc_async_read` /
`noc.async_read` input tiles from DRAM into circular buffers). Reader kernels run on **NCRISC**
(`ReaderDataMovementConfig`; DPRINT RISC = `NC`); writers run on **BRISC** (`BR`). Confirm which
kernel file is the reader from the program factory. For **each input tensor role**, find the call
site(s) where its tiles are read **from DRAM**. Be careful to identify:

- Which read call corresponds to which input tensor (match the `TensorAccessor` / address / CB).
- Reads whose **source is DRAM** vs reads/receives whose source is another core's L1 (forwarded,
  multicast-received, or sharded-peer data). **Only DRAM-sourced reads count** — counting L1
  receives would erase the very redundancy signal you are measuring.
- How many tiles each site reads **for real** (exclude padded / zero-filled tiles — those are not
  DRAM reads) and the tile size in bytes.
- A role may have **mutually-exclusive `constexpr` read paths** (only one compiles into the
  kernel for a given config). Instrument **all** of them — the counter on the non-compiled path
  costs nothing, and it prevents under-counting if the config later selects the other path.

**Instrument at the op-specific reader's call sites — NOT inside shared dataflow helpers.** The
actual `noc.async_read` calls often live in shared headers (e.g. a `read_chunk_*` helper in a
`dataflow_common.hpp`) used by many kernels/ops. Editing those (a) perturbs other kernels
compiled in the same run and (b) forces threading an accumulator through templates. Instead,
count at the reader's own call sites using a **real-tile expression** (the tile count the call
will actually fetch × tile bytes), read from the loop variables already in scope.

### 2. Journal, then instrument (temporary edits)

Before the first edit, snapshot every file you will touch per
[`../shared/revert-temp-edits.md`](../shared/revert-temp-edits.md) (backups + manifest in the
**scratchpad**). Tag every added/changed line with `// [DRAMPROBE temp]`.

In the reader kernel:

- Add one `uint32_t` accumulator per role, e.g. `uint32_t dbg_<role>_bytes = 0;`.
- At each DRAM-read site, add `dbg_<role>_bytes += (uint32_t)<real_tiles_read> * <tile_bytes>;`,
  guarded so it fires **only when the data is read from DRAM** on this core (skip the
  received/forwarded/L1 branch).
- **When a role fans out into several DRAM code paths** (e.g. paged, forward, plain) **plus one
  L1-receive path**, don't add a counter to each branch — add a **single** accumulator once
  **after the whole read block**, gated by the negation of the L1-receive condition
  (`if (!<received_from_L1>) dbg_<role>_bytes += ...;`). Simpler and less error-prone.
- At the very end of the kernel, emit **one** line per core (see `device_print.rst`):

  ```cpp
  #include "api/debug/dprint.h"   // [DRAMPROBE temp]  (printf-style API)
  ...
  DPRINT("DRAMCOUNT core={} <role1>={} <role2>={}\n", <core_id_arg>, dbg_<role1>_bytes, dbg_<role2>_bytes);
  ```

  Use the **printf-style** API (`DPRINT("{} {}\n", a, b)`) — the old `DPRINT << ... << ENDL()`
  streaming API is a hard compile error now. **The format string must end with `\n`** or the
  line is buffered and dropped at device detach. Prefer a per-core identifier the kernel already
  has (a runtime arg) so you can attribute rows.

Kernel edits are JIT-compiled — **no `./build_metal.sh` needed** (`needs_rebuild: false`). If you
must edit host/program-factory C++ (e.g. to force a scheme on/off for a before/after), that is
`needs_rebuild: true` — rebuild with `./build_metal.sh` and again on revert.

### 3. Run with DPRINT, no profiler

Run the single cell as **plain pytest (no `tracy`, no `TT_METAL_DEVICE_PROFILER`)** with DPRINT
enabled. Capture all worker cores so the per-role totals are complete:

```
TT_METAL_DPRINT_CORES=worker \
TT_METAL_DPRINT_RISCVS=<reader RISC, e.g. NC> \
TT_METAL_DPRINT_FILE=<scratchpad>/dramcount.log \
  <the run-once command for this cell>          # e.g. pytest '<node-id>'
```

- `TT_METAL_DPRINT_CORES` uses **logical** coords; `worker`/`all`/`(x1,y1)-(x2,y2)` all work.
  Capturing all worker cores is required to **sum** the per-role totals (a single core is only a
  spot check). Restricting `TT_METAL_DPRINT_RISCVS` to the reader's RISC cuts noise.
- `TT_METAL_DPRINT_FILE` writes to a file instead of stdout — cleaner to parse.
- Watch for hangs (device op). Do not reset the device.

### 4. Aggregate and normalize

- Parse the `DRAMCOUNT` lines; **sum each role across all cores** → `dram_read_bytes(role)`.
- Get `total_output_bytes` and `unique_input_bytes(role)`. **Prefer logging these host-side** in
  the run-once driver — one line per tensor with `tensor.buffer_address()` and
  `tensor.volume() * <dtype_bytes>` — rather than hand-computing from logical shapes, which is
  easy to get wrong for tiled tensors (padded vs logical volume). This host-side log is itself a
  temporary, journalled edit, but `needs_rebuild: false`.
- Build the per-role table: `dram_read_bytes`, `reads_per_output_byte`, `read_redundancy`, plus
  aggregate `total reads/output`.
- Sanity-check: a role read exactly once should give `read_redundancy ≈ 1.0`; less than 1.0 means
  you missed a read site (or counted L1 receives as non-DRAM incorrectly); a huge value means the
  role is re-read many times (real, and interesting).

### 5. Revert

Restore every touched file from the journal per
[`../shared/revert-temp-edits.md`](../shared/revert-temp-edits.md) (hash-check first; rebuild if
any `needs_rebuild` file was touched), then confirm `grep -rn "DRAMPROBE temp"` over the touched
files returns nothing. **Never `git checkout`** — the file may hold the user's uncommitted work.

"Touched files" means **only the files your journal owns**. Other files may be dirty in
`git status` for reasons unrelated to you (the user's own uncommitted work, the run-once driver
they already had) — leave those alone; do not expect a globally clean `git status`, only that
*your* files are back to their snapshot.

## Gotchas / universal knowledge

- **Count DRAM-sourced reads only.** Forwarded / multicast-received / sharded-peer tiles arrive
  over the NoC from another core's L1, not DRAM — counting them destroys the redundancy signal.
- **Count real tiles, not padded.** Zero-filled / padding tiles are not DRAM reads.
- **Profiler ↔ DPRINT are mutually exclusive.** This runs with no profiler, by design (the
  profiler drops the bursty reads this signal is about).
- **DPRINT API currency:** include `api/debug/dprint.h`; use `DPRINT("... {} ...\n", ...)`;
  every line must end with `\n` (unterminated tails are dropped). See `device_print.rst`.
- **Capture all worker cores** to get true totals; one core is only a spot check. Keep the print
  to one short line per core (the per-core DPRINT buffer is small).
- **Multi-device systems:** DPRINT may attach to every device even though the op ran on one.
  Rows are prefixed with the device id (`<dev>:(x,y):<RISC>:`) — filter to the device the op ran
  on before summing, or you will double-count.
- **uint32 per-core counters** are almost always enough (one core rarely reads > 4 GB); use
  uint64 accumulation if a single core could exceed that.
- **Kernel edits = JIT (no build); host C++ edits = `./build_metal.sh`.** Mark the manifest
  `needs_rebuild` accordingly.
- **Per-cell.** DRAM traffic depends on shape and work distribution — re-measure per shape/config.
- **Revert via the shared journal, never `git checkout`.**

## Interpreting the result

Read the two columns per role — they map to different levers:

| read_redundancy(role) | reads_per_output_byte(role) | reading | lever |
|-----------------------|-----------------------------|---------|-------|
| ≈ 1 | any | read once, no waste | nothing to do for this tensor |
| > 1, and the tensor is **shared across cores** | high | many cores each re-read the same tiles from DRAM | **cross-core forwarding / multicast** — one core reads, shares over the NoC |
| > 1, from **one core re-reading across its own work-units** | high | a tile is re-fetched per work-unit | **reuse / cache the tile across work-units** (reorder the loop, keep it resident) |
| ≈ 1 | high | read once but still dominates traffic | inherently bandwidth-heavy input — reuse won't help; consider dtype/layout |

The **highest `reads_per_output_byte`** row is where DRAM read traffic concentrates; its
`read_redundancy` tells you whether the fix is *sharing/reuse* (redundancy > 1) or something else
(redundancy ≈ 1). When evaluating a specific optimization, measure the same cell **before and
after** the change — the per-role redundancy moving toward 1.0 is the confirmation.
