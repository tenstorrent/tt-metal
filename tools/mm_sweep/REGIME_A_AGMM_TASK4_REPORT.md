# AGMM Task 4 — DRAM-staged phase: measurement & characterization (partial; profiler env-blocked)

Fused op: `ttnn.experimental.all_gather_regime_a_matmul_async` (streaming DRAM-staged, Phase A).
Machine: single Blackhole galaxy, container `trusting_borg`, worktree `resilient-marinating-piglet`.
Harnesses (committed): `tools/mm_sweep/agmm_measure.py` (wall A/B), `tools/mm_sweep/agmm_profile.py` (per-RISC).

## Status

Task 3 (streaming correctness + overlap mechanism) is complete: D=2/4/8, ring+linear, fresh+cached,
PCC ≥ 0.999. Task 4 asks for per-RISC characterization (first-compute latency, fabric span, in1 BRISC
span, compute span, first-remote stall, effective BW) and a streaming-vs-full-gather-vs-unfused A/B with a
direct-L1 ceiling estimate. **The per-RISC profiling required for those numbers is blocked in this container
(see "Environment blockers").** What is measurable here is reported below.

## Host-wall A/B (coarse; overhead-bound)

Config: `32x6144x3072`, D=4, ring, 1 link / 1 worker, kb=4, Pk=3, Ns=Sm=1, num_buffers_per_channel=2.
30 synchronized launches/mode, GlobalSemaphores reset each launch (real usage), separate process per mode
(the stream/full-gather choice is a build-time flag, so programs must not share a cache).

| mode                       | median (µs) | min (µs) | max (µs) | PCC     |
|----------------------------|-------------|----------|----------|---------|
| streaming (production)     | 1261        | 1184     | 1376     | 0.99999 |
| full-gather diagnostic     | 1226        | 1173     | 1298     | 0.99999 |
| unfused (all_gather → mm)  | (harness AG-arg issue; not captured this run) | | | |

**Interpretation.** Single-chip regime_a compute for this exact shape is ~88 µs (Task 1 GLX baseline). The
fused wall is ~1200–1260 µs, i.e. **>90% is host dispatch + per-launch GlobalSemaphore reset + fabric
setup**, not compute/gather. Streaming and full-gather are within each other's spread. **Host wall therefore
cannot resolve overlap here** — exactly the plan's warning ("A host wall time without kernel spans is
insufficient for overlap claims"). No overlap conclusion is drawn from these numbers.

## Environment blockers (per-RISC evidence deferred)

Both mechanisms the plan requires for overlap evidence are blocked by the container/build, not the op:

1. **Watcher** (`TT_METAL_WATCHER=10`): fabric-router program overflows the ACTIVE_ETH kernel-config buffer
   (27856 B > 25600 B) at fabric init, so the op cannot run under the watcher on this build.
2. **Device profiler** (`TT_METAL_DEVICE_PROFILER=1`): `ttnn.ReadDeviceProfiler(mesh|submesh)` writes no
   `profile_log_device.csv` for a MeshDevice (0 rows); there is no `ReadMeshDeviceProfiler`. Tracy also
   targets `<TT_METAL_HOME>/build/profiler/build_wasm` for artifacts. Per-RISC kernel-span timelines
   (compute-start vs injector/gather-done) can't be captured here.

`agmm_profile.py` already implements the overlap metric (per device: injector BRISC-KERNEL end vs earliest
compute TRISC-KERNEL start → compute-started-before-gather-done). It will produce the timeline as soon as it
runs on a profiler-capable environment.

## What the overlap mechanism does (implemented, correctness-verified)

- Injector marks each device's shard present via per-shard GlobalSemaphores `shard_landed[e]` (local + fabric
  atomic-inc, ordered after the payload on the same mux channel).
- It advances a monotonic `gather_ready` on every compute core in global-K order, so `gather_ready == k` iff
  shards 0..k-1 are all present.
- The regime_a in0 reader (copied, `AGMM_STREAM`) gates each in0 DRAM read of global tile `kg` on
  `gather_ready > kg/Kt_local`, so a core whose K-region is already present computes while later shards are
  still arriving. The device owning shard 0 (and every core once shard 0 lands ~1 hop away) begins matmul
  before the gather completes. `AGMM_FULL_GATHER_BARRIER` (env `TT_AGMM_FULL_GATHER=1`) is the same-binary
  no-overlap A/B.

## Recommendation

Run `agmm_profile.py` and the watcher on a profiler/watcher-capable bare-metal galaxy or CI to obtain the
per-RISC spans, first-compute latency, fabric/in1/compute spans, and the direct-L1 ceiling estimate. Tasks 5–8
and 10 (direct-L1 selection, placement sweeps, fabric-aware picker) are gated on that evidence and should not
be adopted from host-wall numbers. Correctness and the streaming mechanism are done and committed.
