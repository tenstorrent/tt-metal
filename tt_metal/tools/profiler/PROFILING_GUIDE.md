# Profiling in tt-metal — Device Zone Scopes, Enabling, and Reading Output

The device profiler is built on **Tracy**. Markers ("zones") are timestamped on-device
per RISC-V core, dumped L1 → DRAM → host, written to a raw CSV, then post-processed into a
per-op performance CSV.

This guide covers four things:

1. Device zone scopes — what they are and how to place them
2. Turning the profiler on (build flag + runtime env + wrapper)
3. Reading the output CSVs
4. Why per-RISC (BRISC/NCRISC/TRISC) times are **not** a balance metric

---

## 1. Device zone scopes

A **device zone** is a timed region inside a kernel. It records two wall-clock timestamps
(a `ZONE_START` on entry, a `ZONE_END` on exit) tagged with the RISC that ran it.

Defined in: `tt_metal/tools/profiler/kernel_profiler.hpp`

The macros are RAII-based — drop one at the top of a `{ }` block and the destructor closes
the zone when the block exits.

| Macro | Use |
|---|---|
| `DeviceZoneScopedN("NAME")` | The main one. Times the enclosing scope. `NAME` is a string literal. |
| `DeviceZoneScopedMainN("NAME")` | "Guaranteed" marker at fixed index 0 (always written even if the buffer is full). Used for the firmware/main span. |
| `DeviceZoneScopedMainChildN("NAME")` | Guaranteed marker at fixed index 1 (child program). |
| `DeviceZoneScopedSumN1/2("NAME")` | Accumulating timer — sums durations of a repeated block instead of recording each instance (this is how `CB-COMPUTE-WAIT-FRONT` totals are built). |
| `DeviceTimestampedData("NAME", u64)` | Records a timestamp + a 64-bit data value (e.g. a command ID), not a duration. |
| `DeviceRecordEvent(id)` | A single point-in-time event. |

### Placing one

```cpp
void kernel_main() {
    {
        DeviceZoneScopedN("MY-REDUCE-PHASE");
        // ... reduce work ...
    }   // ZONE_END recorded here
    {
        DeviceZoneScopedN("MY-NORM-PHASE");
        // ... normalize work ...
    }
}
```

- **No explicit `#include`** is needed in the kernel — the build system injects
  `kernel_profiler.hpp` when the profiler build flag is set.
- The macros compile to **nothing** when the profiler is off, so they're safe to leave in.

### Naming mechanics

The string is turned into a compile-time 16-bit FNV-1a hash (`Hash16_CT`,
`kernel_profiler.hpp:107`) which becomes the `timer_id` in the CSV. A pragma also emits
`name,file,line,KERNEL_PROFILER`, harvested into `zone_src_locations.log` so a row can be
traced back to the exact source line.

### Zone → RISC mapping

Every timestamp is tagged with `myRiscID = PROCESSOR_INDEX` (`kernel_profiler.hpp:91`).
On Wormhole/Blackhole Tensix: `0=BRISC, 1=NCRISC, 2=TRISC0, 3=TRISC1, 4=TRISC2`.

You don't choose the RISC — it's wherever the kernel runs. A zone in your reader kernel
shows up as BRISC/NCRISC; a zone in compute shows up as one of the TRISCs. The framework
already wraps every kernel in built-in zones: `BRISC-FW`/`BRISC-KERNEL`, `NCRISC-KERNEL`,
`TRISC-KERNEL`, etc.

---

## 2. Turning the profiler on

### Build flag

The profiler is compiled in **by default** via `-DENABLE_TRACY=ON`
(`cmake/project_options.cmake:112`). You only act to turn it **off**:

```bash
./build_metal.sh                    # profiler compiled in (default)
./build_metal.sh --disable-profiler # -DENABLE_TRACY=OFF
```

### Runtime env vars

Declared in `tt_metal/llrt/rtoptions.cpp`:

| Var | Effect |
|---|---|
| `TT_METAL_DEVICE_PROFILER=1` | **Master switch** for device profiling. |
| `TT_METAL_DEVICE_PROFILER_DISPATCH=1` | Also profile dispatch cores (CQ-DISPATCH/PREFETCH). |
| `TT_METAL_PROFILER_SYNC=1` | Host↔device clock sync for accurate cross-core alignment. |
| `TT_METAL_PROFILER_DIR=<path>` | Where artifacts land. |
| `TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1` | NoC event tracing. |
| `TT_METAL_PROFILE_PERF_COUNTERS` | FPU/pack/unpack/L1 perf counters. |

### Recommended way to run

The `tracy` wrapper sets the env, runs your workload, reads results, and post-processes
into the ops CSV in one shot:

```bash
python3 -m tracy -v -r -p -o <out_dir> -m 'pytest <your_test.py>'
```

(`-r` = generate ops report, `-p` = only profile enabled zones, `-m` = module/command.)

The host pulls device results via `tt::tt_metal::detail::ReadDeviceProfilerResults(device)`
(`api/tt-metalium/tt_metal_profiler.hpp:67`), which fires automatically at device close /
`Finish`. You can also just `export TT_METAL_DEVICE_PROFILER=1` and run pytest directly,
but then you post-process manually.

---

## 3. Reading the output CSVs

Two files matter, both under `generated/profiler/` (overridable by `TT_METAL_PROFILER_DIR`).

### Raw device log — `profile_log_device.csv`

One row **per marker event** (so two rows per zone: a START and an END).
Written at `tt_metal/impl/profiler/profiler.cpp:1087`. Columns:

```
PCIe slot, core_x, core_y, RISC processor type, timer_id,
time[cycles since reset], data, run host ID, trace id, trace id counter,
zone name, type, source line, source file, meta data
```

Key columns:
- `RISC processor type` — BRISC/NCRISC/TRISC_0/1/2/ERISC
- `zone name` — your marker
- `type` — `ZONE_START`/`ZONE_END`/`ZONE_TOTAL`/`TS_DATA`
- `time[cycles…]` — raw device cycles
- `run host ID` — which op call

To get a duration: subtract the START cycle from the END cycle for a matching
(core, RISC, zone) tuple. This file is verbose and is the ground truth.

### Final per-op report — `ops_perf_results_<timestamp>.csv`

This is what you actually read. Produced by `tools/tracy/process_ops_logs.py`, one row
**per op**, durations already converted to **ns**. The analyses that fill the columns are
defined in `tools/tracy/device_post_proc_config.py`. Important columns:

- **`DEVICE FW DURATION [ns]`** — full firmware span (earliest `*-FW` start → latest `*-FW` end across all RISCs).
- **`DEVICE KERNEL DURATION [ns]`** — **the headline number.** Computed as `op_first_last`: earliest `*-KERNEL` ZONE_START on *any* RISC → latest `*-KERNEL` ZONE_END on *any* RISC. This is the op's true wall-clock on the core grid.
- **`DEVICE {BRISC,NCRISC,TRISC0,TRISC1,TRISC2,ERISC} KERNEL DURATION [ns]`** — each individual RISC's own first-to-last span.
- **`DEVICE KERNEL DURATION PER CORE MIN/MAX/AVG [ns]`** — spread across cores (load-balance signal across the grid).
- **`DEVICE COMPUTE CB WAIT FRONT [ns]`** / **`CB RESERVE BACK [ns]`** — accumulated time compute spent *blocked* on circular buffers. **These are the real bottleneck-finding columns** (see §4).
- **`OP TO OP LATENCY [ns]`** — gap between consecutive ops (dispatch overhead / pipeline bubbles).
- Plus op metadata: `OP CODE`, `CORE COUNT`, `MATH FIDELITY`, `PROGRAM CACHE HIT`, `INPUTS`/`OUTPUTS`, etc.

Map a row back to a marker via `zone name` + `timer_id`, cross-referenced against
`zone_src_locations.log` (gives `file:line`).

---

## 4. Why per-RISC times are NOT a balance metric

This is the most important interpretation point, and `device_post_proc_config.py` makes
the mechanism explicit.

**A zone wraps the entire kernel body, including time the RISC spends blocked.** The five
RISCs of a Tensix core don't run a single program — they run reader (BRISC/NCRISC), compute
(TRISC0/1/2), and writer concurrently, and they hand data between each other **only through
circular buffers** with backpressure:

- A consumer calls `cb_wait_front` and **stalls** until the producer pushes tiles.
- A producer calls `cb_reserve_back` and **stalls** until the consumer frees space.

That stall time is *inside* the `*-KERNEL` zone. So a RISC's measured "kernel duration" is
its **lifetime / occupancy**, not its **active work**. It starts when the kernel launches
and ends when it finishes — and in between it may be parked on a CB the whole time.

**Consequence:** in any pipelined steady state, every RISC begins at roughly the same
go-signal and ends when the slowest stage drains. The fast stages simply sit in
`cb_wait_front`/`cb_reserve_back` waiting for the bottleneck. So:

> **BRISC ≈ NCRISC ≈ TRISC durations does NOT mean the work is balanced.** It means they're
> all gated by the same critical path — the slowest stage's latency propagates to everyone
> through backpressure, so all the lifetimes converge to the bottleneck's length. Equal
> per-RISC times is the *expected* signature of a pipeline, regardless of whether work is
> evenly distributed.

This is baked into the metrics themselves: `DEVICE KERNEL DURATION` is deliberately defined
as `op_first_last` *across all RISCs* (earliest-any-start → latest-any-end) — i.e. the
wall-clock envelope — precisely because the per-RISC numbers individually are dominated by
mutual waiting and aren't independently meaningful.

### What to read instead, to actually find the bottleneck

1. **`DEVICE KERNEL DURATION [ns]`** — the single honest "how long did this op take" number.
2. **`DEVICE COMPUTE CB WAIT FRONT` / `CB RESERVE BACK`** — how much of compute's life was
   spent *starved* (waiting for input) vs *backpressured* (waiting for the writer to drain
   output). The stage with **near-zero** CB-wait is your bottleneck; large CB-wait elsewhere
   means those RISCs are idle, waiting on it. High WAIT_FRONT → reader/upstream too slow;
   high RESERVE_BACK → writer/downstream too slow.
3. **`PER CORE MIN/MAX/AVG`** — for grid-level (across cores) imbalance, a separate axis
   from per-RISC.
4. For finer detail, drop your own `DeviceZoneScopedN` markers around individual compute
   phases — the wall-clock deltas of *those* (between explicit CB sync points) tell you
   where active compute time actually goes.

**In short:** treat per-RISC kernel durations as occupancy windows, not workloads. Use the
total kernel duration for "how slow," and the CB wait/reserve sums to answer "why."

---

## Key file reference

| What | Path |
|---|---|
| Zone macros + hashing | `tt_metal/tools/profiler/kernel_profiler.hpp` |
| Raw CSV writer | `tt_metal/impl/profiler/profiler.cpp:1087` |
| Host read API | `api/tt-metalium/tt_metal_profiler.hpp:67` |
| Per-op post-processing | `tools/tracy/process_ops_logs.py` |
| Analysis definitions (duration metrics) | `tools/tracy/device_post_proc_config.py` |
| Build flag | `cmake/project_options.cmake:112` (`ENABLE_TRACY`) |
| Runtime env vars | `tt_metal/llrt/rtoptions.cpp` |
| Tracy wrapper | `python3 -m tracy` (`tools/tracy/__main__.py`) |
| Output location | `generated/profiler/` (CSV: `profile_log_device.csv`, `ops_perf_results_<ts>.csv`) |
