# 09 · Profiling — Tracy Capture, tt-perf-report, and Op Analysis

07_METHODOLOGY covers *when* to trust a measurement. This file is the *mechanical
workflow*: how to capture a Tracy device profile, turn it into a report, and read the
op-level CSV to find the ops worth optimizing — by **total device time** and by **op
count**. This is the loop that drove the Swin-L + DyHead and BGE-M3 campaigns.

---

## 1. Capture a device profile

Build Metal normally — the profiler is included automatically:

```bash
./build_metal.sh
```

Then capture a single traced forward, signpost-bounded (see 07 section 6 for why):

```bash
# Preferred: Tracy harness
TT_METAL_DEVICE_PROFILER=1 python -m tracy -r -p -v --no-runtime-analysis \
    -m pytest path/to/perf_test.py -k <your_case> -sv

# Fallback if tracy wrapper misbehaves:
TT_METAL_DEVICE_PROFILER=1 pytest path/to/perf_test.py -k <your_case>
python tools/tracy/process_ops_logs.py --date
```

Both produce an ops-perf CSV under
`generated/profiler/<model>/reports/<timestamp>/ops_perf_results_<timestamp>.csv`.

> [!WARNING]
> Profile with a **single trace execution step**. Profiler + tracing across multiple
> replays raises `Device data mismatch`. Skip the compile iteration; capture one clean forward.

---

## 2. The human-readable report

```bash
pip install tt-perf-report
tt-perf-report generated/profiler/.../ops_perf_results_<ts>.csv \
    --start-signpost start --end-signpost stop
```

`tt-perf-report` gives per-op rows with device time, a **DRAM %** and **FLOP %** (math
utilization) tag, and matmul advice. Use `--id-range` to isolate a single layer (an
N-layer model repeats the same ops N times; one layer is enough to tune).

**Read the utilization tags first** — they tell you the *kind* of bottleneck:
- **DRAM % > ~60%** → the op is **DRAM-bandwidth-bound**. Lever: smaller dtype (bf8b→bf4b
  weights, PCC-gated), DRAM-sharded matmul (decode), or eliminate the read (L1 handoff).
- **FLOP % high, DRAM % low** → **compute-bound**. Lever: lower math fidelity, wider
  subblock (`fp32_dest_acc_en=False`), more cores.
- **Both low** → the op is **too small to matter** or **dispatch-bound** (op-to-op gap) —
  look at fusion / op-count reduction (06), not the op config.

---

## 3. Bucket the CSV by op — the core analysis

The single most useful view is **aggregate device time and call count per op bucket**.
A `MatmulDeviceOperation` row appears dozens of times; you want the *sum*. Minimal script:

```python
import csv, re
from collections import defaultdict

def tile(s):  # parse "80" from "80 (padded)" style cells
    m = re.match(r"(\d+)", (s or "").strip()); return int(m.group(1)) if m else 0

agg = defaultdict(lambda: {"t": 0.0, "c": 0})
for r in csv.DictReader(open(CSV)):
    op = r.get("OP CODE", "")
    if not op: continue
    try: t = float(r.get("DEVICE KERNEL DURATION [ns]", 0)) / 1000.0  # -> us
    except ValueError: continue
    agg[op]["t"] += t
    agg[op]["c"] += 1

total = sum(v["t"] for v in agg.values())
print(f"{'op':32} {'ms':>8} {'%':>6} {'count':>6} {'us/call':>8}")
for op, v in sorted(agg.items(), key=lambda kv: -kv[1]["t"]):
    print(f"{op:32} {v['t']/1000:8.2f} {100*v['t']/total:6.1f} {v['c']:6d} {v['t']/v['c']:8.1f}")
```

This is exactly the table that drove every Swin-L decision. Example (abridged) output:

```
op                                    ms      %  count  us/call
GridSampleOperation                34.86   22.8    150    232.4
MatmulDeviceOperation              23.09   15.1    381     60.6
BinaryNgDeviceOperation            22.52   14.7    931     24.2
ReshapeViewDeviceOperation         13.84    9.1    482     28.7
TilizeWithValPaddingDeviceOp        8.46    5.5    372     22.7
...
```

---

## 4. The two rankings you act on

Read the bucket table **two ways**:

### (a) Top by total device time — where the milliseconds are
The biggest `ms` rows are your optimization budget. Attack them in order:
- A big matmul bucket → fidelity walk + subblock unlock (03/05).
- A big norm/softmax bucket → sharding + HiFi2 + fused residual (02/04).
- A big data-movement bucket (Reshape, Tilize, Untilize, reshard) → it's often *not*
  doing math; see section 6.

### (b) Top by op count — where the dispatch overhead is
A bucket with a **huge count but tiny us/call** (e.g. 931 BinaryNg at 24 us, or hundreds of
sub-5 us ops) is a **fusion / op-count** target, not an op-config target. In the host-bound
regime each op also costs ~0.6 us of dispatch turnaround, so 900 tiny ops = ~0.5 ms of pure
turnaround on top of their device time. Levers: `unary_chain`, residual fusion, fused
activation, load-time folds (06).

> Rule of thumb: **time ranking → tune the op; count ranking → remove the op.**

---

## 5. Drill into a single bucket by shape and neighbors

Once a bucket is in your sights, split it further. Two refinements that repeatedly paid off:

### Filter by (K, N) / tensor shape
`MatmulDeviceOperation` mixes QKV, attn-out, FF1, FF2. Group by the weight shape to see
which *family* dominates (see 07 section 6). Same for any op that runs at multiple shapes:

```python
# add shape to the key:
z = tile(r.get("INPUT_0_Z_PAD[LOGICAL]")); y = tile(r.get("INPUT_0_Y_PAD[LOGICAL]"))
x = tile(r.get("INPUT_0_X_PAD[LOGICAL]")); imem = r.get("INPUT_0_MEMORY","")[:12]
key = f"{op[:22]} ({z},{y},{x}) {imem}"
```

This immediately surfaces, e.g., "the P3-size grid_sample is 16 ms of the 35 ms bucket" or
"the DRAM-input variant of this multiply is the slow one."

### Identify a mystery op by its neighbors
The op CODE alone (`ReshapeViewDeviceOperation`) doesn't say *which* reshape. Print the
preceding 2-3 ops to locate it in the graph:

```python
rows = list(csv.DictReader(open(CSV)))
for i, r in enumerate(rows):
    if r["OP CODE"] == TARGET and tile(r["INPUT_0_Y_PAD[LOGICAL]"]) == SUSPECT_DIM:
        print("prev:", [rows[i-j]["OP CODE"][:24] for j in (3,2,1)])
        break
```

Swin-L used this to discover that a 3.5 ms reshape bucket was the DCN K-fold, and that a
"DRAM-input multiply" was the modulation step — neither was obvious from the op code.

---

## 6. Reading the data-movement buckets

Reshape / Tilize / Untilize / InterleavedToSharded / ShardedToInterleaved / Permute /
Concat / Slice are **pure data movement** — they do no math. Big buckets here mean:

| Symptom | Likely cause | Lever |
|---|---|---|
| large `ReshapeView` (DRAM→DRAM) | page-size change that ttnn materializes (not a view) | restructure so the reshape is a byte-view; or accept it |
| many `Tilize`/`Untilize` | layout bridges between TILE-matmul and RM-consumer ops | keep a consistent layout across the run; fuse dtype into the (un)tilize |
| many `Interleaved<->Sharded` | reshards between mismatched producer/consumer layouts | match layouts (02 section 7); a reshard the next op redoes anyway is not yours to remove |
| `Typecast` present at all | unnecessary dtype conversion | fuse into a neighboring reshard, or remove (the Q/K/V→bf16 trap, 04 section 5) |
| `Slice`+`Concat` pairs | a `roll`/window op decomposed into slices | usually structural; check for a single-op alternative |

The headline lesson across campaigns: **after the math ops are tuned, the next 20-30% of
device time is usually data movement.** Treat reshards and typecasts as first-class
optimization targets, not plumbing.

---

## 7. Sanity-check the capture before trusting it

| Check | Expected | If wrong |
|---|---|---|
| op counts | `4N` matmuls, `N` SDPA, `~2N` norms for an N-layer model | doubled → warmup/2 replays in range; fix signposts |
| your tuned op's device time | matches the standalone sweep within noise | mismatch → harness bug (`packer_l1_acc`, wrong grid — 07 section 4) |
| total device time vs wall | device ≤ wall; gap = host+dispatch | gap ≫ device → host-bound, enable trace (01 section 8) |
| a "landed" change moved the bucket | the targeted bucket shrank | unchanged → a downstream guard reverted it (07 section 8); trace the actual op |

---

## 8. The end-to-end loop

```
1. capture (section 1)  →  2. report + util tags (section 2)  →  3. bucket by op (section 3)
        ↑                                                                    │
        │                                                                    ▼
6. re-capture, confirm the bucket moved  ←  5. apply lever (02-06,08)  ←  4. rank by time AND count (section 4)
                                                                             │
                                                                  (drill by shape/neighbors, section 5)
```

Stop when the top buckets are either (a) irreducible compute at the math/bandwidth ceiling,
or (b) ops that only kernel-level C++ work can fuse (06 close, 07). That's the floor for the
Python / config / memory-layout surface.

---

## 9. Quick reference

| Goal | Command / view |
|---|---|
| Build with profiler | `./build_metal.sh` |
| Capture | `TT_METAL_DEVICE_PROFILER=1 python -m tracy -r -p -v -m pytest ... -k case` |
| Report | `tt-perf-report ops_perf_*.csv --start-signpost start --end-signpost stop` |
| One layer only | `--id-range` or run a 1-layer test |
| Bottleneck kind | DRAM % (bandwidth) vs FLOP % (compute) vs both-low (dispatch/small) |
| Where the ms are | bucket by OP CODE, sort by total `ms` → tune those ops |
| Where the dispatch is | bucket by OP CODE, sort by `count` (tiny us/call) → fuse/remove |
| Which matmul family | filter by (K, N) |
| Identify a mystery op | print preceding 2-3 op codes |
| Data-movement buckets | Reshape/Tilize/Untilize/reshard/Typecast — first-class targets |
| Trust check | op counts `4N/N/2N`; tuned-op time matches sweep; bucket moved after change |
