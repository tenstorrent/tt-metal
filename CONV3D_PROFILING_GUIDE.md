# Conv3D Profiling Guide

## Quick reference

| Goal | Command |
|---|---|
| Device kernel time per op (~70ms baseline) | `TT_METAL_DEVICE_PROFILER=1 python run_vae_all_ops.py` |
| Full VAE op breakdown (NP/Conv3d/LayerNorm/...) | `python -m tracy -r run_vae_decoder_ablation.py` |
| Conv3d ablation (tilize=0, DM=0, or both) | `CONV3D_ABLATE=tilize_dm python run_vae_all_ops.py` |
| Per-layer wall-clock timing (isolated, cold cache) | `python models/tt_dit/tests/models/wan2_2/profile_conv3d.py` |
| Per-layer device time | `TT_METAL_DEVICE_PROFILER=1 python models/tt_dit/tests/models/wan2_2/profile_conv3d.py` |

**Do not combine** `TT_METAL_DEVICE_PROFILER`, `TT_METAL_DPRINT_CORES`, and `TT_METAL_WATCHER` — they share SRAM.

---

## Device profiler

```bash
TT_METAL_DEVICE_PROFILER=1 python your_script.py
# CSV written to: generated/profiler/.logs/profile_log_device.csv
# (written on device close, not during execution)
```

Parse to get per-dispatch device wall time:

```python
FREQ_MHZ = 1350  # BH
starts, ends = {}, {}
with open("generated/profiler/.logs/profile_log_device.csv") as f:
    next(f); next(f)
    for line in f:
        p = line.strip().split(",")
        if len(p) < 12 or "KERNEL" not in p[10]: continue
        try: cycles, run_id = int(p[5]), int(p[7])
        except ValueError: continue
        starts[run_id] = min(starts.get(run_id, cycles), cycles)
        ends[run_id] = max(ends.get(run_id, cycles), cycles)
dispatches = sorted([(rid, (ends[rid]-starts[rid])/FREQ_MHZ) for rid in starts if rid in ends],
                    key=lambda x: x[0])
# Last 35 entries = timed run (after warmup)
total_ms = sum(d for _, d in dispatches[-35:]) / 1000
print(f"Device kernel time: {total_ms:.1f} ms")
```

## Per-dispatch breakdown + zone analysis

Rank the 35 timed conv3d calls by device time:

```python
FREQ_MHZ = 1350
starts, ends = {}, {}
with open("generated/profiler/.logs/profile_log_device.csv") as f:
    next(f); next(f)
    for line in f:
        p = line.strip().split(",")
        if len(p) < 12 or "KERNEL" not in p[10]: continue
        try: cycles, run_id = int(p[5]), int(p[7])
        except ValueError: continue
        starts[run_id] = min(starts.get(run_id, cycles), cycles)
        ends[run_id] = max(ends.get(run_id, cycles), cycles)
dispatches = sorted([(rid, (ends[rid]-starts[rid])/FREQ_MHZ) for rid in starts if rid in ends],
                    key=lambda x: x[0])
for i, (rid, us) in enumerate(sorted(dispatches[-35:], key=lambda x: -x[1])):
    print(f"  [{i+1:2d}] {us:.0f} us")
```

**Note:** use `dispatches[-35:]` (last 35 by run_id order) not top-35 by duration — other ops
(norms, pads) also appear in the CSV and the top-35 approach picks them up incorrectly.

### Zone-level breakdown (CONV3D_ABLATE=profile)

```bash
CONV3D_ABLATE=profile TT_METAL_DEVICE_PROFILER=1 python run_vae_all_ops.py
```

Parse the slowest TRISC core for the last dispatch (the bottleneck layer):

```python
import csv; from collections import defaultdict
FREQ_MHZ = 1350
rows = [r for r in csv.reader(open("generated/profiler/.logs/profile_log_device.csv"))][2:]
last_run = str(max(int(r[7]) for r in rows if r[7].strip().isdigit()))
events = defaultdict(list)
for r in rows:
    if r[7].strip() != last_run or 'FW' in r[10] or 'KERNEL' in r[10]: continue
    if r[11].strip() not in ('ZONE_START', 'ZONE_END'): continue
    events[(int(r[1]), int(r[2]), r[3].strip(), r[10].strip())].append((int(r[5]), r[11].strip()))
core_zone = defaultdict(lambda: defaultdict(int))
for (cx, cy, risc, zone), evts in events.items():
    evts.sort(); stack = []
    for tc, typ in evts:
        if typ == 'ZONE_START': stack.append(tc)
        elif stack: core_zone[(cx, cy, risc)][zone] += tc - stack.pop()
(cx, cy, risc), zones = max(
    [(k, v) for k, v in core_zone.items() if k[2].startswith('TRISC')],
    key=lambda x: sum(x[1].values()))
total = sum(zones.values())
for z, cyc in sorted(zones.items(), key=lambda x: -x[1]):
    print(f"  {z:20s}  {cyc/FREQ_MHZ:5.0f} us  {cyc/total*100:4.1f}%")
```

---

## Tracy host profiling — full VAE op breakdown

```bash
python -m tracy -r run_vae_decoder_ablation.py
# Writes CSV to: generated/profiler/reports/<timestamp>/ops_perf_results_<timestamp>.csv
```

Parse to get per-op device kernel time for the timed run (filter to last ~1.6s of host timestamps):

```python
import csv, collections

CSV = "generated/profiler/reports/<timestamp>/ops_perf_results_<timestamp>.csv"

NAME_MAP = {
    "Conv3dDeviceOperation": "Conv3d",
    "NeighborPadAsyncDeviceOperation": "NeighborPad",
    "LayerNormDeviceOperation": "LayerNorm/RMSNorm",
    "PadDeviceOperation": "Pad",
    "TilizeWithValPaddingDeviceOperation": "Tilize",
    "UntilizeWithUnpaddingDeviceOperation": "Untilize",
    "PermuteDeviceOperation": "Permute",
    "BinaryNgDeviceOperation": "Binary",
    # ... add others as needed
}

rows = []
with open(CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            rows.append((
                int(row["HOST START TS"]),
                int(row["HOST END TS"]),
                float(row.get("DEVICE KERNEL DURATION [ns]", 0) or 0),
                row["OP CODE"].strip(),
                int(row.get("DEVICE ID", 0) or 0),
            ))
        except: continue

rows.sort()
last_ts = rows[-1][0]
TIMED_WINDOW_NS = 1_600_000_000  # 1.6s — adjust to match your timed run duration
timed = [r for r in rows if r[0] > last_ts - TIMED_WINDOW_NS and r[4] == 0]  # device 0 only

op_dev = collections.defaultdict(float)
op_cnt = collections.defaultdict(int)
for _, _, dev_ns, op_code, _ in timed:
    label = NAME_MAP.get(op_code, op_code[:20])
    op_dev[label] += dev_ns / 1e6
    op_cnt[label] += 1

total = sum(op_dev.values())
print(f"{'Op':<25} {'N':>4}  {'Dev ms':>8}  {'Dev%':>6}")
for label, ms in sorted(op_dev.items(), key=lambda x: -x[1]):
    print(f"{label:<25} {op_cnt[label]:>4}  {ms:>8.1f}  {ms/total*100:>5.1f}%")
print(f"{'TOTAL (serial sum)':<25} {'':>4}  {total:>8.1f}")
print("Note: serial sum > wall clock because NP+Conv3d overlap (T-slice pipelining)")
```

**Important:** filter to the last N seconds (matching your timed run duration) and to device 0 only.
Midpoint-based filtering does not work when warmup is dominated by JIT compilation time.

---

## Identifying the critical path

Wall time is the **serial chain of pipeline stalls**, not the slowest RISC in isolation:

```
BRISC: weight DRAM read → push cb_weight ─────────────────────────────► output write
NCRISC: DRAM gather → vol2col → push cb_vol2col
TRISC: ░░░░c-wait-weights░░░░ → ░c-wait-vol2col░ → tilize → matmul → reduce → untilize
```

Key non-obvious cases:
- **`c-wait-weights ≈ 0` but still slow**: deferred wait is working, bottleneck is elsewhere (reader or matmul)
- **NCRISC shows 5% of cycles but ablating DM saves 17%**: reader causes startup stall on the first spatial block — overlapped for later blocks but dominates small layers with few blocks
- **Increasing T_out_block doesn't help**: weight stall already hidden; bottleneck shifted to reader DRAM gather (192B reads at 4.6 GB/s, can't coalesce — fundamental limit)

**Use ablations to confirm** — zone breakdown shows where time is spent, ablations prove causality.

---

## Why Tilize/Untilize are NOT fused into Conv3d

Conv3d's writer outputs `ROW_MAJOR` (required by the halo buffer read path). The surrounding
residual/norm structure creates a layout mismatch:

```
Conv3d(ROW_MAJOR out) → Tilize → Binary/residual-add(TILE) → LayerNorm(TILE) → Untilize → Pad → Conv3d
```

The conv3d kernel itself has a fused internal tilize (vol2col → tilize → matmul → untilize in
the compute kernel), but the external writer goes back to ROW_MAJOR. The standalone Tilize (26 ops,
55ms) and Untilize (35 ops, 51ms) are format conversions for the residual and norm ops which
require TILE layout. To eliminate them: conv3d output_layout=TILE_LAYOUT, or fuse into the conv3d
writer — not implemented yet.

---

## Conv3d ablation — where to add code

Set `CONV3D_ABLATE` env var before running. Maps to kernel `#define` flags set in
`conv3d_program_factory.cpp` (already wired up — just use the env var):

| Value | Effect |
|---|---|
| `tilize` | Skips tilize+matmul in compute — measures reader+writer cost |
| `dm` | Skips all DRAM reads in reader — measures matmul+writer cost |
| `tilize_dm` | Both zeroed → pure overhead baseline |

### compute.cpp — ablate tilize+matmul

File: `ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/compute.cpp`

Inside the triple nested loop (t_block → h_block → w_block), replace the tilize+matmul with:

```cpp
#if defined(ABLATE_TILIZE)
{
    uint32_t patches_left = num_patches;
    for (uint32_t m = 0; m < matmul_M_t; m++) {
        uint32_t n = (patches_left >= TILE_HEIGHT) ? TILE_HEIGHT : patches_left;
        cb_wait_front(cb_vol2col_rm, n);
        cb_pop_front(cb_vol2col_rm, n);
        patches_left -= n;
    }
}
cb_reserve_back(cb_matmul_interm_tiled, output_tiles);
cb_push_back(cb_matmul_interm_tiled, output_tiles);
#else
// normal tilize+matmul
#endif
```

### reader_vol2col.cpp — ablate DRAM gather

File: `ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/reader_vol2col.cpp`

```cpp
#if defined(ABLATE_DM) || defined(ABLATE_READER_DM)
h_rows_gathered = h_needed;
#else
GATHER_ROWS(...);
#endif
```

### Interpreting results

```
baseline - tilize_dm  = reader + writer overhead (overlapped with matmul in pipeline)
baseline - tilize     = compute cost
baseline - dm         = DRAM gather cost
tilize_dm             = pure writer cost (weight reads + output writes)
```

- `baseline ≈ dm` → DRAM-bound (reader is critical path)
- `baseline ≈ tilize` → compute-bound (matmul is critical path)
- `baseline >> tilize_dm` → both matter

---

## Reference numbers

### BH 2×4 480p uncached, our branch (halo + T-slice pipelining)

Tracy op breakdown (`python -m tracy -r run_vae_decoder_ablation.py`), device 0, timed run.
Wall clock: **1.36s** (22ms upload + 1.02s compute + 0.32s readback).

| Op | Count | Device ms | Dev% | Notes |
|---|---|---|---|---|
| **Conv3d** | 35 | **421ms** | **34.4%** | Fused vol2col+tilize+matmul inside kernel |
| **NeighborPad** | 33 | **376ms** | **30.8%** | Fabric halo exchange + compact buffer write |
| Pad | 33 | 110ms | 9.0% | conv_pad_height + internal paddings |
| LayerNorm/RMSNorm | 30 | 105ms | 8.6% | |
| Tilize | 26 | 55ms | 4.5% | ROW_MAJOR→TILE for residual adds (not fused) |
| Untilize | 35 | 51ms | 4.2% | TILE→ROW_MAJOR after LayerNorm (not fused) |
| Binary | 15 | 30ms | 2.4% | |
| Permute | 13 | 38ms | 3.1% | |
| Other | — | 36ms | 3.0% | AllGather, Upsample, Concat, etc. |
| **Serial total** | | **1222ms** | | |
| *(NP‖Conv3d overlap)* | | *−160ms* | | T-slice pipelining saves this |
| **Actual compute** | | **~1062ms** | | Matches wall clock − upload − readback |

Main branch (no halo, no T_out_block>1): **1.67s wall clock**.

### BH 4×32 720p uncached (device profiler, pre-halo reference)

| Op | Baseline (ms) | Baseline (%) | Ablate tilize+DM (ms) | Saved |
|---|---|---|---|---|
| NeighborPadAsync | 203.5 | 56.4% | 203.5 | 0 |
| **Conv3d** | **68.4** | **19.0%** | **43.6** | **24.8** |
| LayerNorm | 24.0 | 6.7% | 24.0 | 0 |
| Pad | 16.4 | 4.5% | 16.4 | 0 |
| Host overhead | 13.7 | 3.8% | 13.7 | 0 |
| TilizeWithValPadding | 10.1 | 2.8% | 10.1 | 0 |
| UntilizeWithUnpadding | 8.3 | 2.3% | 8.3 | 0 |
| Other | 16.4 | 4.6% | 16.4 | 0 |
| **TOTAL** | **360.8** | 100% | **336.0** | **24.8** |

Conv3d = 43.6ms pure matmul + 24.8ms reader/writer overhead hidden by pipelining.

Reproduce:
```bash
TT_METAL_DEVICE_PROFILER=1 python run_vae_all_ops.py
CONV3D_ABLATE=tilize_dm TT_METAL_DEVICE_PROFILER=1 python run_vae_all_ops.py
```

---

## Pitfalls

**NOC cmd buf registers are shared.** `noc_async_read_one_packet_set_state()` writes source
address + size into hardware registers. Any `noc_async_read()` call after it overwrites them.
Always call `set_state` immediately before `_with_state` — never hoist it past a gather loop.
Violating this causes silent wrong data and a ~60% perf regression on BH.

**`noc_async_read_barrier()` waits for ALL outstanding NOC reads**, not just the current gather.
DRAM prefetch cannot be overlapped with L1 vol2col reads because the barrier at the CB push
stalls until both finish.

**Profiler CSV is written on `CloseDevice()`**, not during execution. Parsing before close gives
an empty file.

**`profile_conv3d.py` shows higher time (~82ms) than in-decoder (~68ms)** because it measures
each layer in isolation (cold cache). In the real decoder, consecutive calls to the same layer
benefit from warm instruction caches. Always use the in-decoder measurement for accurate numbers.

**Tracy serial sum > wall clock** when NP and Conv3d run concurrently (T-slice pipelining):
both are counted separately but execute simultaneously. Expect serial_sum ≈ wall_clock + NP_time.
The difference is the overlap saved by pipelining.

**Filter Tracy by last N seconds, not by midpoint.** Warmup is dominated by JIT compilation
(30+ seconds) not compute, so the midpoint approach incorrectly splits the data. Use
`host_start > last_ts - TIMED_WINDOW_NS` where TIMED_WINDOW_NS matches your timed run duration.
