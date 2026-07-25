# Dispatch op2op investigation — warm-iteration telemetry (#50932 / #50772)

Kimi K2.6 L10 chunk0, `two_iters`, 8x4 Blackhole galaxy, **no profiler in the path**.
Collected with `TT_METAL_DISPATCH_TELEMETRY=1` (see fd_mesh_command_queue.cpp / dispatch.cpp).
Warm iteration isolated by the test's own `iter 0 done` / `iter 1 done` timestamps.

## Root cause

- TENSIX kernel-config ring: **70656 B (69.0 KB)**, sized as `worker_l1_unreserved_start - KERNEL_CONFIG_base`
- Per-op TENSIX config demand: median **10848 B**, p90 **29888 B**, max **43312 B**
- Ops whose configs fit simultaneously: **~6.5**
- `need_sync` over all 1413 reservations: **835 (59%)**

### Dose-response: sync rate rises with config demand

| TENSIX demand | ops | need_sync |
|---|---:|---:|
| 0-4096 B | 271 | 45.4% |
| 4096-8192 B | 274 | 39.4% |
| 8192-16384 B | 552 | 63.2% |
| 16384-32768 B | 296 | 82.4% |
| 32768-inf B | 20 | 55.0% |

## Warm iteration budget

- wall: **768.0 ms**, enqueues: **703**
- inside `EnqueueMeshWorkload`: **436.3 ms (56.8%)**
- between enqueues: **231.6 ms (30.2%)** → **87.0% host-busy**
- device kernel (from #50932 report): 175.5 ms (22.9%)
- `stall_first=1`: **458 / 703 (65%)**
- of which full drains (`sync_count == exp_workers_done`): **0**
- median t_enqueue **678.8 us**, p90 1171.9, max 3865.3

## Hypotheses tested and rejected

| hypothesis | verdict | evidence |
|---|---|---|
| host-side "32 programs = 32x enqueue = 3.5 ms" | DENIED | host enqueue 62-156 us; 32-vs-1 delta 84-183 us at every footprint |
| `kernel_config_entry_count = 8` too small | DENIED | raised to 64: stall rate unchanged (73/81 both) -> it is L1 space, not the table |
| ACTIVE_ETH special-case sync | DENIED | 0 of 81 ops use active ethernet while 73 stalled |
| prefetcher cache bypass | DENIED | all ops `prefetch_cache=true`, kernels 10-72 KB vs 1 MB ring |
| binaries-in-flight full drain | cold only | `binstatus=Committed` throughout the warm iteration |
| `--sync-host-device` manufactures the gap | DENIED | gates only syncDeviceHost/Device at open+CLOSE_DEVICE; no per-op barrier |

## Fixability

`worker_l1_size` (a `device_params` / `open_mesh_device` argument) sets the ring:

| worker_l1_size | TENSIX config ring | result |
|---|---:|---|
| default (0 = auto) | 70656 B | runs; 65% of warm ops sync |
| 1 MB | 483456 B (6.8x) | throws at program.cpp:1724 — too aggressive for this model |

Next step: bisect `worker_l1_size` and record ring size, need_sync rate and warm wall clock at each point.

## Caveat

`t_enqueue` mixes host CPU work with the host *blocking* on a full command queue
(`wait_for_fetch_q_space`). Since 65% of ops carry a sync, much of it is likely blocking —
a symptom of the device stalling rather than independent host cost. Separating them needs a
timer inside `wait_for_fetch_q_space` (zone exists under `--build-perf-debug dispatch`).
The 65% sync rate and the dose-response curve do not depend on that split.

