# AGMM Task 4 — DRAM-staged phase: measurement & characterization

Fused op: `ttnn.experimental.all_gather_regime_a_matmul_async` (streaming DRAM-staged, Phase A).
Machine: single Blackhole galaxy, container `trusting_borg`, worktree `resilient-marinating-piglet`.
Harnesses (committed): `tools/mm_sweep/agmm_profile.py` (per-RISC device-profiler spans + overlap A/B),
`tools/mm_sweep/agmm_measure.py` (host-wall A/B).

Profiler enablement on this container required pre-creating the tracy artifact dirs
(`build_Release/profiler/build_wasm/traces`, `generated/profiler/.logs`) and a CLEAN submesh teardown
(`reset_sub_device_stall_group` + `clear_loaded_sub_device_manager` + `remove_sub_device_manager` +
`close_mesh_device(submesh)` before the parent) — otherwise `ReadMeshDeviceProfilerResults` never flushes
`profile_log_device.csv`. Watcher runs require `TT_METAL_WATCHER_DISABLE_ETH=1` (the fabric-router ETH
program otherwise overflows the ACTIVE_ETH kernel-config buffer; our Tensix kernels are still fully watched).

## Per-RISC device-profiler A/B (D=4, ring, 1 link/1 worker, config=None; median over the 4 profiled devices)

| shape                | mode        | total device span | injector (fabric) span | compute span | PCC    |
|----------------------|-------------|-------------------|------------------------|--------------|--------|
| 32x6144x3072 (wide N)| streaming   | 143.9 µs          | 65.1 µs                | 143.2 µs     | 0.9999 |
| 32x6144x3072         | full-gather | 142.7 µs          | 61.4 µs                | 142.0 µs     | 0.9999 |
| 32x15360x768 (narrow N)| streaming | **176.5 µs**      | 127.3 µs               | 176.0 µs     | 0.9999 |
| 32x15360x768         | full-gather | **180.2 µs**      | 126.3 µs               | 179.7 µs     | 0.9999 |

`streaming` = production progressive gather; `full-gather` = same binary, `TT_AGMM_FULL_GATHER=1`, reader
waits for the whole gather before any matmul (no-overlap A/B).

## Findings

1. **Overlap works, and is bounded by which stage binds.** Streaming reduces the total device span on the
   fabric-heavier narrow-N shape (176.5 vs 180.2 µs, ~2%) and is neutral on the wide-N shape. Reason: at these
   Mt=1 corpus shapes the **compute span (143–176 µs) exceeds the injector/gather span (61–127 µs)**, so the
   DRAM-staged gather is largely hidden behind compute in BOTH modes; streaming only shaves the part of the
   gather that would otherwise serialize ahead of compute. The overlap win grows as T_gather → T_compute
   (smaller N, smaller M, larger D, lower effective fabric BW).
2. **DRAM staging adds a second critical in0 read.** The fused compute span (~143 µs at 32x6144x3072) is well
   above single-chip regime_a for the same shape (~88 µs, Task 1 GLX baseline). The gather buffer is read from
   DRAM by the regime_a in0 path AND written by the injector, competing with the in1 bank bandwidth. This is
   the concrete motivation for **Phase B (direct remote-L1 streaming, Tasks 5–6)**: removing the DRAM
   round-trip for remote A should recover most of that ~55 µs gap on compute-bound shapes and improve the
   fabric-bound tail.
3. **Kernel-span granularity caveat.** The device profiler emits one span per RISC KERNEL that starts at
   kernel launch (including CB waits), so it cannot isolate "first matmul math"; the total-device-span A/B
   above is the meaningful overlap signal. A finer first-compute-latency number needs custom in-kernel zones
   (small follow-up).

## Direct-L1 ceiling estimate (input to Task 5/6)

Upper bound on the Phase-B win ≈ the DRAM-staging overhead exposed above: ~55 µs on 32x6144x3072 (143→~88 µs
compute if the second in0 DRAM read is eliminated) plus the ~2–4 µs serial-gather tail that streaming already
removes. Net: direct-L1 is worth pursuing primarily to cut the extra in0 DRAM read, not the overlap (overlap
is already near-complete at these shapes). Do not adopt direct-L1 without confirming this on the corpus.

## Status / next

Task 4 measurable objectives are met on this machine. Tasks 5–8 and 10 (direct-L1 implementation + selection,
placement sweeps, fabric-aware picker) build on this evidence and remain to be done; they are large but no
longer environment-blocked (profiler + watcher now usable via the setup above). Correctness (Task 3) is
complete and committed.
