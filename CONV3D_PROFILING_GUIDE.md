# Conv3D Device Profiling with Tracy

References:
- [Tracy Profiler](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/tracy_profiler.html)
- [Device Program Profiler](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/device_program_profiler.html)

## Setup

Profiling is enabled by default in Metalium builds (`./build_metal.sh`). No special build flags needed.

**Profiling, kernel debug print, and watcher cannot be used at the same time** — they share SRAM. Do not set `TT_METAL_DPRINT_CORES`, `TT_METAL_WATCHER`, and `TT_METAL_DEVICE_PROFILER` simultaneously.

### Running

```bash
# Terminal 1: start Tracy capture (binary at build/tools/profiler/bin/)
./build/tools/profiler/bin/capture-release -o output.tracy -s 30

# Terminal 2: run with device profiler enabled
TT_METAL_DEVICE_PROFILER=1 python your_script.py
```

Results go to:
- **Tracy file**: `output.tracy` (open in Tracy GUI)
- **Device CSV**: `generated/profiler/.logs/profile_log_device.csv`

### Tracy GUI

Tracy GUI connects on port 8086. For remote machines, forward the port:

```bash
ssh -L 8086:localhost:8086 user@remote-machine
```

The capture binary at `build/tools/profiler/bin/capture-release` is the CLI alternative to the GUI — it saves `.tracy` files directly.

### Python host profiling

```bash
# Profile entire script
python -m tracy your_script.py

# Profile specific pytest
python -m tracy -m pytest test_file.py::test_function

# Add signposts for op-level markers
from tracy import signpost
signpost(header="Run 5", message="Post warmup")
```

## Adding Device Zones

```cpp
#include "tools/profiler/kernel_profiler.hpp"
```

```cpp
{
    DeviceZoneScopedN("zone-name");
    // ... code to profile ...
}
```

Each zone needs its own `{}` scope — multiple zones in the same scope cause `conflicting declaration` errors.

**Zone every wait/stall explicitly.** CB waits and semaphore waits are where kernels block on each other. If you hide a `cb_wait_front` inside a "work" zone, you can't tell how much is compute vs waiting:

```cpp
// BAD: stall hidden inside work zone
{
    DeviceZoneScopedN("c-tilize-matmul");
    cb_wait_front(cb_weight_tiled, weight_tiles);  // stall hidden
    // ... tilize + matmul ...
}

// GOOD: stall visible
{
    DeviceZoneScopedN("c-wait-weights");
    cb_wait_front(cb_weight_tiled, weight_tiles);
}
{
    DeviceZoneScopedN("c-tilize-matmul");
    // ... tilize + matmul ...
}
```

### Limits

- **125 scopes max per core** (hardware buffer limit)
- More than ~20,000 zone events per run can crash the host-side profiler readback
- RISCs on the same core have perfect clock sync; different cores have minor skew; different devices are not synchronized
- Zones add overhead — use selectively, remove before final measurements

### Zone naming

Use short prefixes: `r-` reader (NCRISC), `c-` compute (TRISC), `w-` writer (BRISC).

Conv3d zones that give a complete picture:
- Reader: `r-dram-gather`, `r-vol2col`
- Compute: `c-wait-weights`, `c-tilize-matmul`, `c-reduce`, `c-untilize`
- Writer: `w-weight-read`, `w-wait-compute`, `w-wait-workers`, `w-reducer-gather`, `w-wait-untilize`, `w-output-dram`

## Analyzing the CSV

The device CSV has one row per zone event. Key columns: `core_x`, `core_y`, `RISC processor type`, `time[cycles since reset]`, `zone name`, `type` (ZONE_START/ZONE_END).

Parse to get per-core per-zone durations:

```python
import csv
from collections import defaultdict

FREQ_MHZ = 1350  # BH clock

rows = []
with open('generated/profiler/.logs/profile_log_device.csv') as f:
    reader = csv.reader(f)
    next(reader); next(reader)  # skip header lines
    for row in reader:
        rows.append(row)

events = defaultdict(list)
for r in rows:
    if len(r) < 12: continue
    zone, typ = r[10].strip(), r[11].strip()
    if 'FW' in zone or 'KERNEL' in zone: continue
    if typ not in ('ZONE_START', 'ZONE_END'): continue
    cx, cy, risc = int(r[1]), int(r[2]), r[3].strip()
    tc, run_id = int(r[5]), r[7].strip()
    events[(cx, cy, risc, zone, run_id)].append((tc, typ))

last_run = sorted(set(k[4] for k in events))[-1]
core_zone = defaultdict(lambda: defaultdict(int))
for key, evts in events.items():
    cx, cy, risc, zone, run_id = key
    if run_id != last_run: continue
    evts.sort()
    stack = []
    for tc, typ in evts:
        if typ == 'ZONE_START': stack.append(tc)
        elif typ == 'ZONE_END' and stack:
            core_zone[(cx, cy, risc)][zone] += (tc - stack.pop())

for risc in ['BRISC', 'NCRISC', 'TRISC_0', 'TRISC_1', 'TRISC_2']:
    cores = [(k, v) for k, v in core_zone.items() if k[2] == risc]
    if not cores: continue
    cores.sort(key=lambda x: -sum(x[1].values()))
    key, zones = cores[0]
    total = sum(zones.values())
    print(f'{risc} slowest ({key[0]},{key[1]}): {total/FREQ_MHZ:.0f} us')
    for z, cyc in sorted(zones.items(), key=lambda x: -x[1]):
        print(f'  {z:30s} {cyc/FREQ_MHZ:>8.1f} us ({cyc/total*100:5.1f}%)')
```

### Aggregate breakdown table

The per-core view shows the slowest core. The aggregate view shows **where the entire device's time budget goes** — sum zone cycles across all cores and all RISCs, rank by share. This produces a before/after comparison table:

```
Zone                       Before (%)   After (%)   Change
w-wait-compute               22.6%       37.8%      Writer waits more (pipeline tighter)
c-tilize-matmul              20.7%       25.7%      Same absolute, higher share
c-matmul                     13.5%       23.6%      Same absolute, higher share
r-dram-gather                21.6%       gone       Merged into h-loop
w-output-dram                 1.4%        2.4%      Same absolute
```

Script to generate this:

```python
import csv
from collections import defaultdict

FREQ_MHZ = 1350

def parse_zones(csv_path):
    rows = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader); next(reader)
        for row in reader:
            rows.append(row)

    last_run = sorted(set(r[7].strip() for r in rows if len(r) > 7 and r[7].strip()))[-1]

    # Pair START/END per (core, risc, zone) and sum durations
    events = defaultdict(list)
    for r in rows:
        if len(r) < 12: continue
        zone, typ, run_id = r[10].strip(), r[11].strip(), r[7].strip()
        if run_id != last_run: continue
        if 'FW' in zone or 'KERNEL' in zone: continue
        if typ not in ('ZONE_START', 'ZONE_END'): continue
        key = (int(r[1]), int(r[2]), r[3].strip(), zone)
        events[key].append((int(r[5]), typ))

    zone_totals = defaultdict(int)
    for key, evts in events.items():
        zone = key[3]
        evts.sort()
        stack = []
        for tc, typ in evts:
            if typ == 'ZONE_START': stack.append(tc)
            elif typ == 'ZONE_END' and stack:
                zone_totals[zone] += (tc - stack.pop())

    grand = sum(zone_totals.values())
    return {z: cyc / grand * 100 for z, cyc in zone_totals.items()}, grand / FREQ_MHZ

# Single run:
pcts, total_us = parse_zones('generated/profiler/.logs/profile_log_device.csv')
print(f'Total device-cycles: {total_us:.0f} us')
print(f'{"Zone":30s} {"Share":>8s}')
for z, pct in sorted(pcts.items(), key=lambda x: -x[1]):
    print(f'{z:30s} {pct:>7.1f}%')

# Before/after comparison: run parse_zones on two CSVs (copy the .logs dir between runs)
# before_pcts, _ = parse_zones('before/profile_log_device.csv')
# after_pcts, _ = parse_zones('after/profile_log_device.csv')
# all_zones = sorted(set(before_pcts) | set(after_pcts), key=lambda z: -max(before_pcts.get(z,0), after_pcts.get(z,0)))
# for z in all_zones:
#     print(f'{z:30s} {before_pcts.get(z,0):>7.1f}% → {after_pcts.get(z,0):>7.1f}%')
```

**Important**: the percentages are meaningful only if zones cover 100% of each kernel's time. Unzoned gaps appear as missing percentage. If the percentages sum to <90%, you have significant unzoned regions — add zones to cover the gaps.

### Identifying the bottleneck

The three RISCs run as a pipeline with CB-based handoffs:

```
BRISC (writer): read weights → push cb_weight → [wait for compute] → write output
NCRISC (reader): DRAM gather → vol2col → push cb_vol2col
TRISC (compute): wait cb_weight → wait cb_vol2col → tilize+matmul → reduce → untilize
```

The wall time is NOT simply the max RISC time — it's the **serial chain of dependencies**. A RISC can appear fast but still be on the critical path if it blocks another. For example, TRISC may show 750 us total, but 300 us of that is `c-wait-weights` — stalled on BRISC. The actual compute work is only 450 us, and the critical path is: BRISC weight read (200 us) → TRISC compute (450 us) → BRISC output (50 us) = 700 us.

**The zone breakdown tells you where time is spent. Ablations tell you what's on the critical path.** Always use both.

### Reducer vs worker cores

Conv3d uses C_in_block parallelism: one reducer + N workers per reduction group. In the writer:
- Reducer cores have `w-wait-workers`, `w-reducer-gather`
- Worker cores have `w-wait-compute`, `w-wait-reducer-ack`

## Ablation Testing

Tracy zones show where time is spent. Ablations show **what actually matters** — comment out work and measure the wall-time delta.

### Ablation 1: Comment out compute

Tests whether compute is hidden behind the reader pipeline.

```cpp
// compute.cpp: replace the fused tilize+matmul block with CB pass-through
for (uint32_t t_block = t_out_start; ...) {
    for (uint32_t h_block = ...) {
        for (uint32_t w_block = ...) {
            // ABLATION: skip tilize+matmul, just drain vol2col and produce dummy output
            {
                constexpr uint32_t row_tiles = matmul_K_t;
                uint32_t patches_left = num_patches;
                for (uint32_t m = 0; m < matmul_M_t; m++) {
                    const uint32_t patches_this_row = (patches_left >= TILE_HEIGHT)
                                                          ? TILE_HEIGHT : patches_left;
                    // Consume the RM patches (unblocks reader)
                    cb_wait_front(cb_vol2col_rm, patches_this_row);
                    cb_pop_front(cb_vol2col_rm, patches_this_row);
                    patches_left -= patches_this_row;
                }
            }
            // Produce dummy output (unblocks writer)
            cb_reserve_back(cb_matmul_interm_tiled, output_tiles);
            cb_push_back(cb_matmul_interm_tiled, output_tiles);

            // ... rest of reducer/worker logic unchanged ...
```

**If wall time barely changes** → compute is fully hidden behind the reader. The reader DRAM gather is the true bottleneck.

**If wall time drops significantly** → compute is on the critical path.

### Ablation 2: Comment out writer NOC transactions

Tests the cost of weight reads, output writes, and reducer gather.

```cpp
// writer.cpp: skip ALL NOC reads/writes, keep CB handshakes

// Weight read — skip the NOC reads, keep cb_push_back to unblock compute
cb_reserve_back(cb_weight_tiled, weight_tiles);
// for (row...) for (col...) noc_async_read_tile(...);  // COMMENTED OUT
// noc_async_read_barrier();                             // COMMENTED OUT
cb_push_back(cb_weight_tiled, weight_tiles);

// Bias read — same pattern
if constexpr (use_bias) {
    if (is_reducer) {
        cb_reserve_back(cb_bias_tiled, matmul_N_t);
        // for (...) noc_async_read_tile(...);  // COMMENTED OUT
        // noc_async_read_barrier();             // COMMENTED OUT
        cb_push_back(cb_bias_tiled, matmul_N_t);
    }
}

// Reducer gather — skip reading worker partials
cb_reserve_back(cb_reduction_tiled, output_tiles);
// for (tile...) noc_async_read(...);  // COMMENTED OUT
// noc_async_read_barrier();           // COMMENTED OUT
cb_push_back(cb_reduction_tiled, output_tiles);

// Output DRAM write — skip
// for (t...) for (h...) for (w...) noc_async_write(...);  // COMMENTED OUT
// noc_async_write_barrier();                                // COMMENTED OUT
cb_pop_front(cb_matmul_result_rm, output_tiles);
```

**If wall time drops 25%+** → writer NOC is on the critical path (weight reads blocking compute start, or output writes serializing the pipeline).

You can also comment out subsets (just weights, just output writes) to isolate which writer phase dominates.

### Ablation 3: Comment out reader DRAM gather

Tests the cost of the per-position DRAM reads in the shard gather.

```cpp
// reader_vol2col.cpp: skip the DRAM gather, keep h_rows_gathered update
if (h_needed > h_rows_gathered) {
    // ABLATION: skip all noc_async_read calls in the gather loop
    // for (t_local...) for (h_local...) for (w_local...)
    //     noc_async_read(...);
    // noc_async_read_barrier();
    h_rows_gathered = h_needed;
}
```

**If wall time drops significantly** → the DRAM gather dominates. Reducing NOC transaction count (larger reads, fewer positions) would help.

**If wall time barely changes** → gather is overlapped with compute.

### Interpreting ablation results together

| Compute ablation | Writer ablation | Gather ablation | Conclusion |
|---|---|---|---|
| No change | No change | Big drop | **Reader DRAM-gather bound** |
| Big drop | No change | No change | **Compute bound** |
| No change | Big drop | No change | **Writer weight-read bound** |
| No change | Some drop | Some drop | **Mixed reader + writer** |

## Pitfalls

### NOC cmd buf state is shared

`noc_async_read_one_packet_set_state()` writes source address + size into hardware registers. Any `noc_async_read()` call overwrites them. The `set_state` must be called immediately before `_with_state` reads — never before a gather that uses `noc_async_read`:

```cpp
// WRONG — 60% regression. Gather clobbers the state.
noc_async_read_one_packet_set_state(shard_base, size);
for (...) noc_async_read(...);           // overwrites cmd buf registers
noc_async_read_barrier();
noc_async_read_one_packet_with_state(...); // uses wrong state

// CORRECT
for (...) noc_async_read(...);
noc_async_read_barrier();
noc_async_read_one_packet_set_state(shard_base, size);
noc_async_read_one_packet_with_state(...); // uses correct state
```

### noc_async_read_barrier waits for ALL reads

The global barrier waits for every outstanding NOC read regardless of purpose. You cannot overlap DRAM prefetch with L1 vol2col reads — the barrier at the chunk push stalls until both complete. TRID-based barriers exist but the `_with_trid` API has per-read overhead and flow control issues that can hang.

### Interleaved pages are on different DRAM banks

Each page in an interleaved tensor goes to bank `page_idx % NUM_BANKS`. Consecutive W positions within the same (t,h) row are on different banks — you cannot coalesce them into a single NOC read. This is the fundamental reason the conv3d gather uses one transaction per (t,h,w) position.

### Profiler + watcher + dprint conflict

These three tools share the same per-core SRAM region. Only one can be active at a time. If a profiled run hangs, reset devices with `tt-smi -r 0,1` before switching to watcher for debugging.

### Manual profiler readback for long runs

Profiling data is collected automatically on `CloseDevice`. For runs with >1000 kernel iterations before device closure, manually trigger readback:

```cpp
tt::tt_metal::detail::ReadDeviceProfilerResults(device);
```

## Full VAE Op-Type Breakdown

To get the per-op breakdown table (NeighborPadAsync, Conv3d, LayerNorm, etc.) across the full
decoder, combine two measurements: **non-conv ops** (timed once) and **conv3d** (timed with
ablation). The script `run_vae_decoder_ablation.py` runs the full decoder with a warmup + timed
pass; `CONV3D_ABLATE=tilize_dm` zeroes out tilize and DM overhead in conv3d so you can isolate
pure matmul time.

### Scripts

| Script | Purpose |
|---|---|
| `run_vae_decoder_ablation.py` | Full decoder wall time (warmup + timed run) |
| `run_vae_all_ops.py` | Conv3d-only: 35 ops in decoder order, reports device kernel time |
| `TT_METAL_DEVICE_PROFILER=1 run_vae_all_ops.py` | Conv3d device time (68.2ms baseline) |
| `CONV3D_ABLATE=tilize python run_vae_all_ops.py` | Conv3d with tilize zeroed |
| `CONV3D_ABLATE=dm python run_vae_all_ops.py` | Conv3d with DRAM gather zeroed |
| `CONV3D_ABLATE=tilize_dm python run_vae_all_ops.py` | Conv3d with both zeroed (pure matmul) |

### Collecting the breakdown

```bash
# 1. Baseline full decoder (includes NP, LayerNorm, Pad, etc.)
python run_vae_decoder_ablation.py                        # note wall time

# 2. Conv3d breakdown (device kernel time)
TT_METAL_DEVICE_PROFILER=1 python run_vae_all_ops.py      # 68.2ms device
CONV3D_ABLATE=tilize_dm TT_METAL_DEVICE_PROFILER=1 \
    python run_vae_all_ops.py                             # 43.6ms = pure matmul

# 3. Non-conv op times
# Measured once from the decoder run:
#   total_wall = run_vae_decoder_ablation output
#   conv3d_wall = run_vae_all_ops wall time
#   nonconv = total_wall - conv3d_wall  (then profiled per-op via tracy below)
```

### Getting per-op host-side times with Tracy

The table (NeighborPadAsync, Conv3d, LayerNorm, BinaryNg ...) comes from host-side Tracy op
zones. Each ttnn op records its dispatch-to-completion time as a host zone named after the op.

```bash
# In terminal 1, start tracy capture:
./build/tools/profiler/bin/capture-release -o vae_baseline.tracy -s 60

# In terminal 2, run the decoder:
python -m tracy run_vae_decoder_ablation.py

# Open vae_baseline.tracy in Tracy GUI (port-forward 8086 for remote machines).
# In the Statistics view, group by zone name — sum all occurrences of each op to get total ms.
# Or export as CSV and parse with the script below.
```

### Parsing Tracy host CSV

When Tracy GUI exports a CSV (Statistics → Export), each row is an op type with total duration.
Parse to build the breakdown table:

```python
import csv

# Tracy Statistics CSV export (File → Export Statistics)
ops = {}
with open('tracy_stats.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = row['Name'].strip()
        total_ms = float(row['Total time (ms)'])
        ops[name] = ops.get(name, 0) + total_ms

# Bucket op names → canonical category
BUCKETS = {
    'NeighborPadAsync': 'NeighborPadAsync',
    'Conv3d':           'Conv3d',
    'RMSNorm':          'LayerNorm',
    'Pad':              'Pad',
    'BinaryNg':         'BinaryNg',
    'AllGatherAsync':   'AllGather',
    'TilizeWithValPadding': 'TilizeWithValPadding',
    'UntilizeWithUnpadding': 'UntilizeWithUnpadding',
    'SDPA':             'SDPA',
    'Upsample':         'Upsample',
    'Permute':          'Permute',
    'MinimalMatmul':    'MinimalMatmul',
    'ConcatDevice':     'ConcatDevice',
}
bucketed = {}
other = 0.0
for name, ms in ops.items():
    matched = next((v for k, v in BUCKETS.items() if k in name), None)
    if matched:
        bucketed[matched] = bucketed.get(matched, 0) + ms
    else:
        other += ms
bucketed['Other'] = other

total = sum(bucketed.values())
print(f"{'Op':<30} {'ms':>8} {'%':>7}")
for op, ms in sorted(bucketed.items(), key=lambda x: -x[1]):
    print(f"{op:<30} {ms:>8.1f} {ms/total*100:>6.1f}%")
print(f"{'TOTAL':<30} {total:>8.1f}")
```

### Reference numbers — bh_4x32 720p uncached, 2×4 BH LB

| Op | Baseline (ms) | Baseline (%) | Ablate Both (ms) | Saved |
|---|---|---|---|---|
| NeighborPadAsync | 203.5 | 56.4% | 203.5 | 0.0 |
| Conv3d | 68.4 | 19.0% | 43.6 | **24.8** |
| LayerNorm | 24.0 | 6.7% | 24.0 | 0.0 |
| Pad | 16.4 | 4.5% | 16.4 | 0.0 |
| Host overhead | 13.7 | 3.8% | 13.7 | 0.0 |
| TilizeWithValPadding | 10.1 | 2.8% | 10.1 | 0.0 |
| UntilizeWithUnpadding | 8.3 | 2.3% | 8.3 | 0.0 |
| BinaryNg | 7.2 | 2.0% | 7.2 | 0.0 |
| Other | 9.2 | 2.5% | 9.2 | 0.0 |
| **TOTAL** | **360.8** | 100% | **336.0** | **24.8** |

Ablate Both = `CONV3D_ABLATE=tilize_dm` (tilize + DRAM gather zeroed, only matmul remains).
Conv3d = 24.8ms of the 68.4ms is overhead that is hidden in the reader pipeline. The remaining
43.6ms is pure matmul time that cannot be further reduced without changing the arithmetic.
