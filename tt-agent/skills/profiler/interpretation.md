# Reading tt-perf-report output

## Headline metric

`DEVICE FW DURATION [ns]` — time from first RISC firmware start to last RISC
firmware end for the op. This is the on-device cost and the canonical number
to minimize.

Rank ops by this column. The top 1-3 dominate the device time budget — any
optimization push targets one of these unless there is a specific reason
otherwise.

## Key CSV columns

| Column | What it tells you |
|---|---|
| `OP CODE` | C++ class name of the op |
| `OP TYPE` | `tt_dnn_device` = ran on device; `python_fallback` / `tt_dnn_cpu` = host code (not profilable at kernel level) |
| `CORE COUNT` | Number of Tensix cores the op dispatched to. Compare against the grid available on the part (64 on Wormhole B0). Low count under a heavy op = under-parallelized. |
| `PARALLELIZATION STRATEGY` | How the op shards work across cores. Names differ per op; a missing or fallback strategy often means sub-optimal placement. |
| `MATH FIDELITY` | LoFi / HiFi2 / HiFi3 / HiFi4. Lower fidelity = fewer cycles per tile in the FPU. Over-specified fidelity is a common silent cost. |
| `DEVICE FW DURATION [ns]` | Headline. |
| `DEVICE KERNEL DURATION [ns]` | Compute-kernel-only window. `FW - KERNEL` ≈ setup + teardown + CB flush. A large gap on a short op means setup dominates. |
| `DEVICE BRISC KERNEL DURATION [ns]` | Reader (DM0) kernel time — usually dominated by `noc_async_read` traffic and barriers. |
| `DEVICE NCRISC KERNEL DURATION [ns]` | Writer (DM1) kernel time — `noc_async_write` and output draining. |
| `DEVICE TRISC0/1/2 KERNEL DURATION [ns]` | Unpack / Math / Pack compute stages. |
| `HOST DURATION [ns]` | Includes queue time + host dispatch overhead. Inflated on first iteration (cache-miss populates the program cache). |

## Bottleneck tags

After reading the per-RISC columns for a top op, tag the bottleneck. Rule of
thumb: the RISC with the **largest kernel duration relative to its peers** is
the bound.

| Tag | Signal |
|---|---|
| `reader-bound` | BRISC duration ≳ TRISC1 duration. Compute waits on tiles from DRAM/L1. Often fixable by batching `noc_async_read` before the barrier, increasing CB depth, or switching to a sharded input layout. |
| `compute-bound` | TRISC1 (math) dominates. Either a legitimate compute ceiling, or math fidelity is higher than the model requires, or the op could use a lower-cost variant. |
| `writer-bound` | NCRISC dominates. Compute waits to push output. Uncommon in isolation — usually means downstream pressure or under-sized output CB. |
| `under-parallelized` | Low `CORE COUNT` relative to device grid, combined with high per-core duration. Sharding or a larger dispatch grid can help. |
| `NOC-stall` | BRISC and NCRISC both high but TRISC low. Data movement is saturating NOC bandwidth — batching, routing, or layout changes help. |
| `host-dominated` | `FW` is short but `HOST DURATION` is large on runs past the first. Rare for device ops; points to Python or dispatch overhead, not kernel code. |

Use the tag only — do not prescribe the fix in a profile note. Fix selection
belongs to the caller (developer or `tt:optimizer`).

## First-run caveat

The first iteration of any test populates the program cache. Host times on
that iteration are inflated. If the test runs the layer only once,
`HOST DURATION` is unreliable. The device-side columns are still valid.

## When numbers look wrong

- All device durations = 0 → profiler did not run. Check for env conflicts
  (`TT_METAL_DPRINT_CORES`, `TT_METAL_WATCHER`, `TTNN_CONFIG_PATH` still set).
- Only a handful of ops captured but the test ran many → buffer overflow;
  test needs periodic `ReadDeviceProfilerResults(device)` calls.
- Wildly different numbers between two otherwise-identical runs → program
  cache miss on the measured iteration, or another agent's workload
  interleaving. Re-run; the MCP queue serializes device access but
  build/compile caches do not.
