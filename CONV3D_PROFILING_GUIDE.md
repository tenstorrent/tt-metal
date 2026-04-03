# Conv3D Profiling Guide

## Quick reference

| Goal | Command |
|---|---|
| Device kernel time per op (68ms baseline) | `TT_METAL_DEVICE_PROFILER=1 python run_vae_all_ops.py` |
| Full VAE op breakdown (NP/Conv3d/LayerNorm/...) | `python -m tracy run_vae_decoder_ablation.py` |
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

## Tracy host profiling

```bash
# Terminal 1 — start capture before running
./build/tools/profiler/bin/capture-release -o output.tracy -s 60

# Terminal 2 — run
python -m tracy run_vae_decoder_ablation.py

# Open output.tracy in Tracy GUI (ssh -L 8086:localhost:8086 for remote)
# Statistics view → sort by total time → groups by op name (NeighborPadAsync, Conv3d, ...)
```

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
    // Drain vol2col CB to unblock reader, push dummy output to unblock writer
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
// ... rest of reducer/worker logic unchanged ...
#else
// normal tilize+matmul
#endif
```

### reader_vol2col.cpp — ablate DRAM gather

File: `ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/reader_vol2col.cpp`

Inside the L1 prefetch shard gather (inside h_rows loop), skip the NOC reads:

```cpp
#if defined(ABLATE_DM) || defined(ABLATE_READER_DM)
// Skip all NOC reads; just advance the h_rows counter
h_rows_gathered = h_needed;
#else
GATHER_ROWS(...);
#endif
```

Already wired for the non-prefetch path too — gated by `ABLATE_DM` at the w_block level.

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

## Reference numbers — bh_4x32 720p uncached, 2×4 BH LB

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

Conv3d = 43.6ms pure matmul + 24.8ms reader/writer overhead that is hidden by pipelining.

Reproduce:
```bash
TT_METAL_DEVICE_PROFILER=1 python run_vae_all_ops.py               # 68.2ms
CONV3D_ABLATE=tilize_dm TT_METAL_DEVICE_PROFILER=1 python run_vae_all_ops.py  # 43.6ms
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
benefit from warm instruction caches.
