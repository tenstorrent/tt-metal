# Ring joint SDPA — per-(ring iter, device) math util measurement

How to reproduce a `iter × ring_index` math util table like the one below:

```
| iter \ ring_index | 0 (mesh 2)   | 1 (mesh 0)   | 2 (mesh 3)   | 3 (mesh 1)   |
|-------------------|--------------|--------------|--------------|--------------|
| 0                 | 40.55  DIAG  | 40.55  DIAG  | 40.55  DIAG  | 40.55  DIAG  |
| 1                 | 69.14  DOWN  | 69.16  DOWN  | 69.17  DOWN  | 70.71  UP    |
| 2                 | 69.29  DOWN  | 70.68  UP    | 70.71  UP    | 70.65  UP    |
| 3                 | 66.49  DOWN  | 66.41  DOWN  | 68.03  UP    | 67.96  UP    |
```

Configuration here is `mla_100k` (q=160, k=320) on a Blackhole 4-device
single ring. The same procedure works for any ring topology + balanced+causal
config — only the ring_size and the mesh_dev↔ring_index mapping change.

## What's measured

For each ring iter `N` in `0..ring_size-1`, the SDPA op runs **only that one
iter** on every device simultaneously (AllGather skipped, non-target iters
short-circuited). The reported math util is the per-device kernel duration
turned into a percent of theoretical peak FLOPs for the per-iter local work.

Each cell is classified by comparing the iter's `ring_id` against the device's
own `ring_index`:

| comparison              | label | reason                                              |
|-------------------------|-------|-----------------------------------------------------|
| `ring_id == ring_index` | DIAG  | local Q × local K (causal mask, full diagonal block)|
| `ring_index < ring_id`  | DOWN  | first-half local Q chunks fully causally-masked → skipped (writer's `balanced_skip_q`)|
| `ring_index > ring_id`  | UP    | second-half local K chunks fully causally-masked → skipped (reader symmetric) |

UP and DOWN both do half-block work — same FLOPs, different skip pattern.

## Prerequisites

1. Branch with `TT_METAL_RING_ITER_ONLY` measurement hook + the kernel DPRINT
   for `ring_index` discovery (this commit).
2. Built repo (`./build_metal.sh`).
3. Activated python env (`source python_env/bin/activate`).
4. A ring SDPA topology of size ≥ 2.

## Procedure

### Step 1 — discover `mesh_dev → ring_index` mapping (one-time per topology)

The DEVICE_ID column in the tracy ops CSV is the mesh device id, **not** the
ring_index. The kernel DPRINT prints `rix` (= `ring_index`) so you can build
the mapping for the current run.

```bash
unset TT_METAL_RING_ITER_ONLY
TT_METAL_DPRINT_CORES="(0,0)" \
TT_METAL_DPRINT_RISCVS="TR0" \
  scripts/run_safe_pytest.sh \
  tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_sweep_perf_impl[mla_100k-q160-k320] \
  2>&1 | grep "\[RING\]"
```

Expected output (one line per device per iter from logical core (0,0), TR0
only) — for ring_size=4 you get 16 lines. The first 4 (iter 0) reveal the
mapping:

```
0:0-0:TR0: [RING] rix=1 iter=0 rid=1
1:0-0:TR0: [RING] rix=3 iter=0 rid=3
2:0-0:TR0: [RING] rix=0 iter=0 rid=0
3:0-0:TR0: [RING] rix=2 iter=0 rid=2
```

`<mesh_dev>:<core>: [RING] rix=<ring_index> ...` — read off the mapping.

### Step 2 — run per-iter perf loop

Driver script (save as `/tmp/per_iter_perf.py`):

```python
#!/usr/bin/env python3
"""Per-(ring_iter, device) math util sweep for ring joint SDPA."""
import os, sys, subprocess
from pathlib import Path
from statistics import mean, pstdev

REPO = Path("/localdev/skrstic/tt-metal")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tt_metal/third_party/tracy/python"))

import pandas as pd
from tests.nightly.sdpa_perf_utils import MeshConfig, compute_math_utilization
from tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler
from tests.nightly.blackhole.sdpa.test_ring_joint_sdpa import MODEL_CONFIGS

mesh = MeshConfig.detect()
ring_size = mesh.sp_size
arch = (os.environ.get("ARCH_NAME") or os.environ.get("IRD_ARCH_NAME") or "blackhole").lower()
model_name = os.environ.get("PERF_MODEL", "mla_100k")
q_chunk = int(os.environ.get("PERF_Q", "160"))
k_chunk = int(os.environ.get("PERF_K", "320"))
repeats = int(os.environ.get("PERF_REPEATS", "3"))

model = MODEL_CONFIGS[model_name]
local_seqlen = model.seq_len
local_nhq = model.nhq // mesh.tp_size
config_id = f"{model_name}-q{q_chunk}-k{k_chunk}"

cells = {}  # (iter, mesh_dev) -> [util across reps]
for ring_iter in range(ring_size):
    os.environ["TT_METAL_RING_ITER_ONLY"] = str(ring_iter)
    for rep in range(repeats):
        subdir = f"ring_iter_only_perf/iter_{ring_iter}_rep_{rep}"
        cmd = (f"pytest tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::"
               f"test_ring_joint_attention_sdpa_sweep_perf_impl[{config_id}]")
        run_device_profiler(cmd, subdir, device_analysis_types=["device_kernel_duration"])
        df = pd.read_csv(get_latest_ops_log_filename(subdir))
        df = df[df["OP CODE"].str.contains("RingJointSDPA", case=False, na=False)]
        iter_is_causal = model.is_causal if ring_iter == 0 else False
        for _, row in df.iterrows():
            dev = int(row["DEVICE ID"])
            dur_ns = int(row["DEVICE KERNEL DURATION [ns]"])
            cores = int(row["CORE COUNT"])
            util = compute_math_utilization(
                local_seqlen, local_seqlen, model.d_q, model.d_v, local_nhq,
                dur_ns, cores - cores % 10, iter_is_causal, arch=arch)
            if ring_iter > 0 and model.is_causal and model.is_balanced:
                util /= 2
            cells.setdefault((ring_iter, dev), []).append(util)

# Print mean per (iter, mesh_dev)
for (it, dev), vals in sorted(cells.items()):
    print(f"iter={it} mesh_dev={dev} mean_util={mean(vals):.2f}% std={pstdev(vals):.3f}")
```

Run it:

```bash
source python_env/bin/activate
python3 /tmp/per_iter_perf.py
```

This produces a `(iter, mesh_dev) → math util` mapping. ~3 reps × ring_size
runs, ≈15 s each = ~3 min for ring_size=4.

### Step 3 — assemble the final table

Combine the rix mapping (Step 1) with the per-cell utils (Step 2). For each
cell:

1. Look up `ring_index` for the cell's `mesh_dev`.
2. Compute the iter's `ring_id` using the sequencer formula. For ring_size=4
   the per-rix iter→ring_id sequence is `d, d+1, d−1, d−2` (mod 4) where
   `d = ring_index`. For other ring sizes, derive from
   `RingIdSequencer` in `ring_utils.hpp` with the appropriate
   `(forward_writes_expected, backward_writes_expected)` split.
3. Classify DIAG / UP / DOWN by comparing `ring_index` to `ring_id` using the
   table at the top of this doc.
4. Sort columns by `ring_index` (with `mesh_dev` in parens) so the row pattern
   `d, d+1, d−1, d−2` reads cleanly across the table.

## Notes / gotchas

- DPRINT is gated to `UCK_CHLKC_UNPACK` only — printing on all 3 TRISCs
  inflates the kernel binary past the kernel-config buffer cap (program
  size 70704 vs cap 70656).
- Per-rep std-dev is < 0.15 pp — small inter-cell differences (~1.5 pp UP vs
  DOWN) are reproducible signal, not noise.
- `TT_METAL_RING_ITER_ONLY=N` skips the AllGather host-side AND the reader's
  CCL semaphore wait, so per-iter timings do **not** include any AG/CCL
  contention. The K tensor must be pre-populated (already gathered).
