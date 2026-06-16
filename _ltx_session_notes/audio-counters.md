# LTX-2 audio vocoder — measured HW counter profile (bounded scope)

Blackhole Galaxy 4x8 (32 chips), branch `ltx-video`. Timestamps host=UTC, user=PT.
Captured 2026-06-15 ~02:38 UTC / 19:38 PT.

## What was measured, and the honest scope

Goal: per-op HW counters (fpu/sfpu/pack/unpack/instrn) for the audio vocoder, to classify
each dominant op COMPUTE- vs DATA-MOVEMENT-bound. The full audio decode counter capture
**overflows the on-device marker buffer** (`bufferEndIndex = 191992`, the per-RISC
`PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC=768000 B` ≈ 192k markers) — that is Deliverable 2.

Bounded scope used here: **one AMPBlock1** (`test_prof_single_ampblock`, bh_4x8, the real
T-shard=8 / C-shard=4 sharded input captured by intercepting `resblocks[0]` inside a genuine
`Vocoder.forward`). The single block's profiled window captured cleanly — the only dropped
markers (32, one per device) were the one-time cold mesh-assembly burst, NOT the block.
`--profiler-capture-perf-counters fpu,sfpu,pack,unpack,instrn` → bitfield 39 (note: `sfpu` is
not a valid group in tools/tracy and is silently ignored; fpu|pack|unpack|instrn captured, and
the device RTL emits its full counter block regardless).

### HARD BLOCKER on the per-OP-CODE table (be explicit)

The per-OP-CODE `ops_perf_results` CSV could NOT be produced. With
`--profiler-capture-perf-counters`, tracy disables the fast C++ post-processing
("Skipping runtime analysis ... Falling back to legacy Python processing") and
`process_ops_logs` uses `import_log_run_stats()` over the raw device log. On 32 chips with
counters this is pathological: the one-block device log is ~23 GB / ~hundreds-of-millions of
rows, and the legacy parser climbed past 140 GB RSS over 25+ min without finishing (broker
2100 s timeout). The whole-vocoder variant additionally OOM-Killed the tracy capture server on
a 1.3–8 GB trace. So I report the MEASURED counters straight from the raw device log
(`profile_log_device.csv`) aggregated by counter type — which still answers the
compute-vs-DM question — and the device-time op ranking from the prior warm 4x8 profile
(optimizer state note), rather than fabricate a per-op-code counter join I could not compute.

## MEASURED counters — one AMPBlock1 window (sum & per-readback mean over 76,955 counter samples)

All perf-counter datapoints are emitted from the Tensix-AGG zone (logged under the BRISC slot),
so they are per-zone aggregates across the block's ops, not split by op-code. The VALUES carry
the binding verdict:

| Counter | mean / readback | sum | what it measures |
|---|---:|---:|---|
| WAITING_FOR_NONZERO_SEM_2 | **1,449,722** | 111,563,365,835 | cycles a thread waits on a semaphore (cross-RISC / cross-chip sync) |
| FPU_INSTRN_AVAILABLE_1 | 1,513,394 | 116,463,254,213 | FPU instruction-slot availability (mostly idle-available) |
| THREAD_INSTRUCTIONS_2 (pack thread) | 53,436 | 4,112,156,243 | TRISC2/packer thread instrs |
| PACKER_BUSY | 40,546 | 3,120,173,695 | packer (output writeback) busy cycles |
| ANY_THREAD_STALL | 22,374 | 1,721,773,513 | any compute thread stalled |
| PACKER_DEST_READ_AVAILABLE | 20,491 | 1,576,830,942 | packer waiting on DEST |
| UNPACK0_BUSY_THREAD0 | 19,036 | 1,464,866,443 | unpacker (input read) busy |
| MATH_COUNTER | 5,864 | 451,223,760 | **actual MATH (FPU) ops retired** |
| FPU_COUNTER | 5,828 | 448,469,523 | **actual FPU ops** |
| SFPU_COUNTER | 36 | 2,754,237 | actual SFPU ops (negligible) |

## Compute-vs-DM verdict (backed by the ratios above)

**The AMPBlock1 is overwhelmingly DATA-MOVEMENT / SYNCHRONIZATION-bound, not compute-bound.**

- `WAITING_FOR_NONZERO_SEM_2` mean **1,449,722** vs `MATH_COUNTER`/`FPU_COUNTER` mean **~5,830** →
  the block spends **~250× more cycles waiting on semaphores than doing FPU math**. Semaphore
  waits on this sharded path are the cross-chip `neighbor_pad_async` halo barriers + CCL — i.e.
  data movement / sync, exactly what dominates wall time.
- `PACKER_BUSY` (40,546) and `UNPACK0_BUSY` (19,036) — packer/unpacker (L1↔DEST movement) — are
  each ~3–7× the actual `MATH_COUNTER` (5,864). Movement of operands/results outweighs the math.
- `SFPU_COUNTER` ~36 confirms the SnakeBeta activation is a fused, FPU/SFPU-light op here; it is
  not where cycles go.

This corroborates the existing optimizer finding that vocoder+bwe is **~58% dispatch/sync-bound**
(865 ms wall vs 362 ms device-active) with NeighborPad cross-chip halo barriers as the lever —
now backed by an actual on-device counter ratio (sem-wait ≫ math) rather than only the wall-vs-
device gap.

## Per-op DEVICE-time ranking (warm 4x8, prior eager profile; optimizer state note)

Not re-measured this session (the counter capture blocked the per-op CSV), carried as the
device-time context the counters refine:

| Op (vocoder) | share of vocoder device time | binding (from counters + structure) |
|---|---:|---|
| Conv1dDepthwise (DilatedConv1d→Conv3d) | ~21.6% | DM-bound: needs cross-chip halo (neighbor_pad) before each conv; sem-wait dominates |
| Conv3d (Conv1dViaConv3d) | ~14.5% | mixed; blockings tuned; packer/unpacker-heavy (PACKER_BUSY ≫ MATH) |
| BinaryNg (residual/bias adds, tail masks) | ~14.2% | DM/dispatch-bound: tiny broadcast MUL/ADD, dispatch-cheap, lever exhausted |
| NeighborPad (per-conv causal halo) | ~13.8% | DM-bound by definition: fabric/ethernet CCL, GlobalSemaphores — the sem-wait source |
| AllGather (CCL) | ~12% | DM-bound: CCL |

## The real lever

**NeighborPad / cross-chip halo synchronization is the real lever**, confirmed by the counter
ratio (`WAITING_FOR_NONZERO_SEM` ≫ `MATH`/`FPU`). The block is not waiting on the FPU — it is
waiting on semaphores guarding cross-chip halo exchanges. The grounded fix is op-GRAPH-level, not
in the conv kernel (the in-kernel halo fold was already closed as infeasible on the sharded path):
fewer cross-chip halo barriers — the PROVEN win is the `LTX_AUDIO_SUBMESH` route (smaller T-shard
→ −20% vocoder+bwe). Trace replay is net-negative here, so op-count / barrier-count reduction is
the durable fix.

## Reproduce

    # bounded single-block counter capture (device-clean; per-op-code CSV still blocked by the
    # legacy process_ops_logs OOM described above):
    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILE_PERF_COUNTERS=39 \
    TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=16000 \
    python -m pytest models/tt_dit/tests/models/ltx/prof_vocoder_forward.py::test_prof_single_ampblock -k bh_4x8 -s
    # then aggregate counters straight from the raw device log:
    awk -F',' '/counter type/{...}' generated/profiler/.logs/profile_log_device.csv
