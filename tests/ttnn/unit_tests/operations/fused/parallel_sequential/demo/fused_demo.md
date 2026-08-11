# Fusion Infrastructure Demo Suite

Eight demos showcasing different fusion capabilities on Tenstorrent Wormhole hardware.

**Test file:** `tests/ttnn/unit_tests/operations/fused/parallel_sequential/demo/test_fused_demo.py`

Normal pytest runs exercise correctness only (`perf_mode="none"`). The current
slide benchmarks for Parallel Chains and Balanced Tree are separate manual
tests gated by `TT_METAL_RUN_FUSION_SLIDE_PERF=1`:

```bash
# Run correctness tests:
python -m pytest tests/ttnn/unit_tests/operations/fused/parallel_sequential/demo/test_fused_demo.py -xvs

# Run one steady-state slide benchmark mode:
TT_METAL_RUN_FUSION_SLIDE_PERF=1 python -m pytest \
  tests/ttnn/unit_tests/operations/fused/parallel_sequential/demo/test_fused_demo.py \
  -k "slide_parallel_chains_e2e and persistent" -q -s

# Capture one mode-specific device profile:
TT_METAL_RUN_FUSION_SLIDE_PERF=1 \
TT_METAL_DEVICE_PROFILER=1 \
TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
python -m pytest \
  'tests/ttnn/unit_tests/operations/fused/parallel_sequential/demo/test_fused_demo.py::TestPerfDemos::test_slide_parallel_chains_fused_device_fw[mode=inline]' \
  -q
```

The older demo methods retain disabled `cold_start`, `e2e`, and `device_fw`
branches for historical measurements. Add those values back to their local
`perf_mode` parametrization only when reproducing an older table.

All timing measured on Wormhole n300 (single chip), BF16.

## Perf Modes

### `device_fw` — What does the hardware actually cost?

Single dispatch, no timing loops. Designed for Tracy device profiling (`TT_METAL_DEVICE_PROFILER=1`). Tracy's CSV reports `DEVICE FW DURATION [ns]` (firmware setup + kernel + teardown) and `DEVICE KERNEL DURATION [ns]` (pure Tensix execution). Fused ops appear as one `FusionDispatchOpDeviceOperation` row; unfused ops appear as one row per `ttnn.*` dispatch (sum for total). This is the ground truth for whether fusion saves device cycles — it strips away all host overhead.

### `e2e` — What does the user see in steady state?

Measured by the slide benchmark tests: 5 warmup iterations (discarded), then the median of seven 100-iteration trials, all caches warm. This captures steady-state pipelined throughput, including host descriptor/container work, dispatch overhead, `fusion_dispatch_op` argument patching, and device execution. It is not single-request latency.

The August 2026 refresh reports both current fusion usage modes:

- **Inline:** recreate the descriptors and container each iteration, then use the warm fusion build cache.
- **Persistent:** create descriptors and the container once, call `update()` for the activation, and reuse the container's hot dispatch path.

Both modes use the production command-lifetime semaphore bank. All logical
barrier words are packed into one lockstep-sharded L1 tensor, initialized by
one queued write, and released after dispatch submission. No semaphore tensor
is retained in the fusion cache or between forward passes.

Each mode is benchmarked in an isolated pytest invocation so persistent
intermediate allocations from one mode cannot affect another mode's available
L1.

### `cold_start` — How long until first output?

All caches cleared (JIT disk + in-memory + program + fusion build), then one full execution including JIT compilation. For fused tests this is `run()` (which internally calls `build()` + `launch()`) plus `synchronize()`; for unfused tests it's the ttnn op calls which trigger JIT internally. This matters for model loading and first-inference latency — fusion JIT-compiles one kernel instead of N.

Cold-start values belong to the older per-demo workflow and are not part of
the August 2026 slide refresh.

### Apples-to-apples configs

Unfused `ttnn.matmul` calls use the same `program_config` and `compute_kernel_config` as their fused counterparts. Unfused `ttnn.rms_norm` / `ttnn.layer_norm` calls use `LayerNormShardedMultiCoreProgramConfig` with matching core counts and shard specs where possible.

---

## Demo 1: Linear Chain — RMS -> Matmul -> RMS

Basic sequential chaining of heterogeneous ops (norm + matmul + norm) into a single fused kernel dispatch. Intermediates round-trip through DRAM between phases.

```
  RMS ──> Matmul ──> RMS
         (8 cores)
```

**API:**
```python
[out] = Sequential(rms1, matmul, rms2).run(results=[rms2])
```

**Setup (parameterized on H):**
| | Shape | Memory |
|-|-------|--------|
| Input | `(1, 1, 256, H)` | BF16, DRAM interleaved |
| Weight (norm) | `(1, 1, 1, H)` | BF16, DRAM interleaved |
| Weight B (matmul) | `(1, 1, H, H)` | BF16, DRAM interleaved |
| Output | `(1, 1, 256, H)` | BF16, DRAM interleaved |

**Program configs:** RMS + Matmul on `(0,0)-(3,1)` = 8 cores. `fp32=False`, `math_approx=False`, `HiFi4`.

### H=128 (dispatch-dominated)

Unfused breakdown: RMS #1 FW=7.9 us + matmul FW=9.0 us + RMS #2 FW=8.1 us = 25.0 us.

| Metric | Fused | Unfused | Speedup |
|--------|------:|--------:|--------:|
| Device FW | 26.6 us | 25.0 us (3 ops) | 0.94x |
| Device kernel | 26.0 us | 22.7 us | 0.87x |
| E2E steady state | 0.047 ms | 0.070 ms | **1.49x** |
| Cold start | 2004 ms | 2651 ms | **1.32x** |

At H=128, each op takes 8-9 us. The fused kernel is ~2 us slower than the unfused sum due to inter-phase barrier overhead. The **1.49x E2E speedup** comes from reducing per-op overhead: each unfused dispatch has device firmware setup/teardown and host-side RT arg prep that fast dispatch can't fully hide when ops are this short.

### H=1536 (compute-dominated)

Unfused breakdown: RMS #1 FW=30.6 us + matmul FW=905.8 us + RMS #2 FW=607.8 us = 1544.2 us. (RMS #2 FW inflated by cold program loading; kernel = 30.9 us.)

| Metric | Fused | Unfused | Speedup |
|--------|------:|--------:|--------:|
| Device FW | 1007.3 us | 1544.2 us (3 ops) | 1.53x |
| Device kernel | 1006.7 us | 965.8 us | 0.96x |
| E2E steady state | 0.994 ms | 0.995 ms | 1.00x |
| Cold start | 2106 ms | 2732 ms | **1.30x** |

At H=1536, the matmul dominates (~906 us). Device kernel times are essentially identical (the 1.53x FW speedup is an artifact of cold program loading in the unfused RMS #2 dispatch). **No structural device speedup** for a linear chain of DRAM-interleaved ops.

**PCC:** H=128: 0.9999, H=1536: 0.9996

---

## Demo 2: Sharded Chain — RMS -> LN

Fusion with block-sharded memory layout. The CB allocator detects pinned buffer addresses from the shard spec and preserves them while pool-allocating other CB slots.

```
  RMS ──> LN
     (16 cores, block-sharded L1)
```

**API:**
```python
[out] = Sequential(rms, ln).run(results=[ln])
```

**Setup (parameterized on H):**
| | Shape | Memory |
|-|-------|--------|
| Input | `(1, 1, H, 512)` | BF16, block-sharded L1, 4x4 = 16 cores |
| Output | `(1, 1, H, 512)` | BF16, block-sharded L1, 4x4 = 16 cores |

**Program configs:** `LayerNormShardedMultiCoreProgramConfig`, 16 cores. `fp32=True`, `math_approx=False`, `HiFi4`.

### H=128 (dispatch-dominated)

Unfused breakdown: RMS FW=9.0 us + LN FW=13.7 us = 22.7 us.

| Metric | Fused | Unfused | Speedup |
|--------|------:|--------:|--------:|
| Device FW | 22.2 us | 22.7 us (2 ops) | 1.02x |
| Device kernel | 21.1 us | 20.4 us | 0.97x |
| E2E steady state | 0.059 ms | 0.072 ms | **1.22x** |
| Cold start | 1948 ms | 3188 ms | **1.64x** |

At the device level, fused and unfused FW are nearly identical. The **1.22x E2E speedup** comes from `fusion_dispatch_op`'s selective slot-patching dispatch, which is lightweight enough that the fused single-dispatch path is faster than two individual `ttnn.rms_norm`/`ttnn.layer_norm` dispatches with their native program-cache fast paths.

### H=1536 (compute-dominated)

Unfused breakdown: RMS FW=39.9 us + LN FW=64.0 us = 103.9 us.

| Metric | Fused | Unfused | Speedup |
|--------|------:|--------:|--------:|
| Device FW | 107.7 us | 103.9 us (2 ops) | 0.96x |
| Device kernel | 106.7 us | 101.7 us | 0.95x |
| E2E steady state | 0.123 ms | 0.105 ms | 0.85x |
| Cold start | 1912 ms | 3115 ms | **1.63x** |

The fused kernel is ~5 us slower than the unfused kernel sum (~107 vs ~102 us) — inter-phase barrier overhead. E2E is 0.85x because at these longer kernel durations, the per-dispatch host overhead is a smaller fraction of total time, and the barrier overhead on the device side tips the balance slightly toward unfused.

**PCC:** 1.000000 (both H values)

---

## Demo 3: Parallel Chains — (LN->MM) + (RMS->MM)

Two independent 2-op chains running on disjoint 1x8 core columns within a single kernel dispatch. No inter-chain synchronization needed.

```
  Chain A (col 0):   LN ──> Matmul
                                       (parallel, disjoint cores)
  Chain B (col 1):   RMS ──> Matmul
```

**API:**
```python
out_a, out_b = Parallel(
    Sequential(ln_a, mm_a),
    Sequential(rms_b, mm_b),
).run(results=[mm_a, mm_b])
```

**Setup:**
| | Shape | Memory |
|-|-------|--------|
| Input A | `(1, 1, 1024, 256)` | BF16, block-sharded L1 on `(0,0)-(0,7)` |
| Input B | `(1, 1, 1024, 256)` | BF16, block-sharded L1 on `(1,0)-(1,7)` |
| Weight B | `(1, 1, 256, 128)` | BF16, DRAM interleaved |

**Program configs:** LN/RMS on 1x8 grids, matmul `MatmulMultiCoreReuseProgramConfig`. `fp32=True`, `math_approx=False`, `HiFi4`.

**August 11, 2026 refresh.** Unfused breakdown: LN FW=42.236 us + matmul FW=17.762 us + RMS FW=27.008 us + matmul FW=17.929 us = 104.935 us. Unfused runs four sequential dispatches on a single `(0,0)`-based grid.

| Metric | Fused Inline | Fused Persistent | Unfused | Inline speedup | Persistent speedup |
|--------|-------------:|-----------------:|--------:|---------------:|-------------------:|
| Device FW | 68.313 us | 68.357 us | 104.935 us (4 ops) | **1.536x** | **1.535x** |
| Device kernel | 67.561 us | 67.614 us | 102.369 us | **1.515x** | **1.514x** |
| E2E | 0.691 ms | 0.180 ms | 0.274 ms | 0.40x | **1.52x** |

Inline and Persistent were captured in independent single-dispatch device-profiler runs. Their FW durations differ by only 0.044 us (0.064%) and their kernel durations by 0.053 us (0.078%), confirming that both modes execute the same device program within run-to-run noise.

The **1.54x device speedup** demonstrates the intended parallelism: both chains overlap on disjoint core columns. The command-lifetime semaphore bank removes per-semaphore blocking initialization, allowing Persistent E2E throughput to realize the device benefit and beat the mature unfused program-cache path by **1.52x**. Inline remains host-bound because descriptor and container construction occurs every iteration.

**PCC:** Chain A = 1.0000, Chain B = 1.0000

---

## Demo 4: Sharded Tree — LN -> Slice -> Matmul -> Slice -> LN

Full block-sharded tree topology with 13 ops across 5 levels on a 2x8 core grid with hierarchical core subset splitting.

```
                        LN_stem (2x8 = 16 cores)
                                |
                +---------------+---------------+
                |                               |
           sl_top (1x8=8)                  sl_bot (1x8=8)
          row 0, cols 0-7                 row 1, cols 0-7
                |                               |
          mm_left (1x8=8)                mm_right (1x8=8)
                |                               |
          +-----+-----+                   +-----+-----+
          |           |                   |           |
     sl_tl (1x4)  sl_bl (1x4)       sl_tr (1x4)  sl_br (1x4)
          |           |                   |           |
     ln_ll (1x4)  ln_lr (1x4)       ln_rl (1x4)  ln_rr (1x4)
```

**API:**
```python
ll, lr, rl, rr = Sequential(
    ln_stem,
    Parallel(
        Sequential(sl_top, mm_left, Parallel(Sequential(sl_tl, ln_ll), Sequential(sl_bl, ln_lr))),
        Sequential(sl_bot, mm_right, Parallel(Sequential(sl_tr, ln_rl), Sequential(sl_br, ln_rr))),
    ),
).run(results=[ln_ll, ln_lr, ln_rl, ln_rr])
```

**Setup:**
| | Shape | Memory |
|-|-------|--------|
| Input | `(1, 1, 2048, 256)` | BF16, block-sharded L1, 2x8 |
| B_left, B_right | `(1, 1, 256, 128)` | BF16, DRAM interleaved |

**Program configs:** LN sharded, matmul `MatmulMultiCoreReuseProgramConfig`, slice tile-path with named CT args. `fp32=True`, `math_approx=False`, `HiFi4`.

**August 11, 2026 refresh.** Unfused breakdown (13 dispatches): stem LN FW=39.744 us + two slices FW=12.020 us + two matmuls FW=35.850 us + four leaf slices FW=11.586 us + four leaf LNs FW=97.590 us = 196.790 us.

| Metric | Fused Inline | Fused Persistent | Unfused | Inline speedup | Persistent speedup |
|--------|-------------:|-----------------:|--------:|---------------:|-------------------:|
| Device FW | 119.810 us | 119.795 us | 196.790 us (13 ops) | **1.642x** | **1.643x** |
| Device kernel | 118.329 us | 118.293 us | 186.730 us | **1.578x** | **1.579x** |
| E2E | 1.669 ms | 0.264 ms | 0.965 ms | 0.58x | **3.65x** |

Inline and Persistent were captured independently here as well. Their FW durations differ by only 0.015 us (0.013%) and their kernel durations by 0.036 us (0.030%).

The **1.64x device speedup** comes from branch parallelism across disjoint core subsets. Persistent mode also collapses 13 host submissions into one and, with semaphore initialization reduced to one queued bank write, reaches a **3.65x E2E throughput speedup**. Inline is still slower because rebuilding 13 descriptors and the nested container dominates.

**PCC:** 1.000000 (leaf LN output vs unfused reference)

---

## Demo 5: Asymmetric Branches — LN -> Parallel(Slice->RMS->RMS, Slice->LN)

A common stem LN fans out into two asymmetric branches: a lightweight chain (Slice + RMS + RMS) on the left, and a heavier single LN on the right. The lightweight branch runs hidden behind the heavy branch on disjoint core columns. Unfused, all 6 ops serialize.

```
            LN_stem (4x8 = 32 cores)
                    |
      +-------------+-------------+
      |                           |
 Slice_L (2x8=16)          Slice_R (2x8=16)
 cols 0-1                   cols 2-3
      |                           |
 RMS (2x8=16)                LN (2x8=16)
      |
 RMS (2x8=16)
```

**API:**
```python
left_out, right_out = Sequential(
    ln_stem,
    Parallel(
        Sequential(sl_left, rms1, rms2),
        Sequential(sl_right, ln_right),
    ),
).run(results=[rms2, ln_right])
```

**Setup:**
| | Shape | Memory |
|-|-------|--------|
| Input | `(1, 1, 2048, 512)` | BF16, block-sharded L1, 4x8 = 32 cores |
| Weight (RMS) | `(1, 1, 1, 512)` | BF16, L1 width-sharded |

**Program configs:** LN/RMS `LayerNormShardedMultiCoreProgramConfig`, slice tile-path. `fp32=True`, `math_approx=False`, `HiFi4`.

Unfused breakdown (6 dispatches): stem LN FW=41.0 us + 2 slices FW=20.5 us + 2 branch RMS FW=51.6 us + branch LN FW=35.6 us = 148.6 us.

| Metric | Fused | Unfused | Speedup |
|--------|------:|--------:|--------:|
| Device FW | 120.6 us | 148.6 us (6 ops) | **1.23x** |
| Device kernel | 118.6 us | 141.7 us | **1.19x** |
| E2E steady state | 0.160 ms | 0.268 ms | **1.68x** |
| Cold start | 2265 ms | 5588 ms | **2.47x** |

The **1.23x device speedup** comes from the left branch (Slice+RMS+RMS ~ 10+26+26 us) running in parallel with the right branch (Slice+LN ~ 10+36 us). Fused time ~ stem (41 us) + max(62, 46) = 103 us kernel + barriers. The **1.68x E2E speedup** additionally saves per-dispatch overhead (6 dispatches → 1) with `fusion_dispatch_op`'s cache-hit path handling the single fused dispatch efficiently.

**PCC:** Left chain = 1.0000, Right LN = 0.9998

---

## Demo 6: GlobalCircularBuffer Mid-Kernel Write

Data exfiltration from the middle of a fused kernel via `GlobalCircularBuffer`. The sender pushes data to an external consumer core during kernel execution, before finishing all phases.

```
  Sender core (0,0):                    Receiver core (1,0):
    Phase 0: DRAM(A) -> GlobalCB push     GlobalCB -> DRAM(output_recv)
    Phase 1: DRAM(B) -> DRAM(output_b)
```

**API:**
```python
out_b, out_recv = Parallel(
    Sequential(gcb_sender, identity_phase1),
    gcb_consumer,
).run(results=[identity_phase1, gcb_consumer])
```

All three ops use hand-written SOURCE_CODE kernels (not ttnn ops):

- **`gcb_sender`** (core 0,0): Reader loads tiles from DRAM into a local CB. Compute copies tiles from the input CB to an output CB (tile copy via `copy_tile`/`pack_tile`). Writer pushes tiles from the output CB into the `GlobalCircularBuffer`, which transfers them to the receiver core over NOC.
- **`identity_phase1`** (core 0,0): A standard DRAM-to-DRAM identity op (read tiles from DRAM tensor B, tile-copy through compute, write back to DRAM output_b). This runs as the sender's second phase, demonstrating that the sender core continues executing after the GlobalCB push.
- **`gcb_consumer`** (core 1,0): Reader waits for tiles to arrive via the `GlobalCircularBuffer` and makes them available in a local CB. Writer drains the local CB to DRAM. No compute kernel — data passes through unmodified.

This is a proof of concept demonstrating how producer and consumer ops can run in parallel, with the producer sending data to the consumer mid-kernel via GlobalCB.

---

## Demo 7: Non-Contiguous Core Grid ("Swiss Cheese")

Validates that the unicast barrier release works correctly on a `CoreRangeSet` with gaps. The stem op runs on rows 0-1, 3, and 5 (24 cores, skipping rows 2 and 4). Data flows into two parallel branches. This mimics core patterns like DRAM-sharded matmul, where the compute cores are spread throughout the device.

```
      col0  col1  col2  col3  col4  col5
row0   X     X     X     X     X     X    +
row1   X     X     X     X     X     X    | branch A (18 cores)
row2   .     .     .     .     .     .    |  <- gap
row3   X     X     X     X     X     X    +
row4   .     .     .     .     .     .      <- gap
row5   X     X     X     X     X     X    <- branch B (6 cores)
```

**API:**
```python
out_a, out_b = Sequential(stem, Parallel(op_a, op_b)).run(results=[op_a, op_b])
```

All three ops are hand-written DRAM-to-DRAM identity ops (same `_build_identity_op` helper used by the barrier benchmark). Each op has three kernels: reader loads tiles from a DRAM tensor into a local CB, compute does a tile copy (`copy_tile`/`pack_tile`) from input CB to output CB, and writer drains the output CB back to DRAM. The data content is trivial — the point is exercising the barrier and core-grid mechanics, not the compute.


---

## Demo 8: Barrier Overhead Microbenchmark

Measures pure barrier mechanism cost by chaining N no-op phases (empty `kernel_main()`) so the only work is the inter-phase barrier synchronization itself.

```
  Phase 0 (no-op) ──barrier──> Phase 1 (no-op) ──barrier──> ... ──barrier──> Phase N (no-op)
```

**API:**
```python
Sequential(*[noop_op for _ in range(N)]).run()
```

**Setup:**
- Each phase has 3 kernels (reader, compute, writer) with empty bodies
- No CBs, no DRAM I/O, no compute -- barrier transitions dominate
- A single dummy DRAM tensor satisfies the `fusion_dispatch_op` tensor requirement
- Parametrized over `num_phases` (2-6) and `num_cores` (1, 8, 16, 64)

**Methodology:** All numbers are E2E steady-state (`_time_e2e`: 5 warmup + 100 timed iterations, host-side wall clock including dispatch and device sync). Per-barrier cost = `(fused_N - fused_1) / (N - 1)`, where `fused_1` is a 1-phase baseline that captures fixed kernel launch overhead. At low N the baseline noise is a large fraction of the total, so per-barrier appears inflated; at high N (5-6) it converges to the true mechanism cost.

**Per-barrier results (Wormhole n300, unicast barrier release):**

| Cores | Grid | 2 phases | 3 phases | 4 phases | 5 phases | 6 phases | Converged |
|------:|:-----|-------:|-------:|-------:|-------:|-------:|----------:|
| 1 | (0,0) | 5.0 us | 1.4 us | 2.1 us | 1.8 us | 1.4 us | ~1.5 us |
| 8 | 1x8 | 4.9 us | 3.2 us | 1.8 us | 1.5 us | 1.5 us | ~1.5 us |
| 16 | 2x8 | 5.1 us | 2.3 us | 2.2 us | 1.7 us | 1.5 us | ~1.5 us |
| 64 | 8x8 | 6.3 us | 3.1 us | 2.8 us | 3.0 us | 1.6 us | ~1.7 us |

**Per-barrier cost is ~1.5 us** regardless of core count (1 to 64). The barrier has two levels:

- **`local::sync()`** -- per-core 3-RISC rendezvous via L1 semaphores (compute_done, writer_done, reset_done). Coordinator (NCRISC) waits for followers, resets CB state, signals reset_done.
- **`group::sync()`** -- cross-core synchronization. Each core sends `noc_semaphore_inc` to core 0; core 0 waits for all arrivals, then unicast-releases each core individually.

At 64 cores the unicast release loop sends 63 individual `noc_async_write` packets, yet converged cost is only ~0.2 us more than 1 core.

**Multicast was benchmarked and removed.** Replacing the unicast release loop with a single `noc_async_write_multicast` for rectangular grids showed no measurable improvement (64-core 6-phase: 1.7 us unicast vs 1.5 us multicast -- within noise). The release loop was never the bottleneck. Unicast is kept because it works with non-contiguous core grids (e.g., DRAM-sharded matmul's scattered cores) without special-casing.
