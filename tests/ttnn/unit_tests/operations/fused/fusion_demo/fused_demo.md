# Fusion Infrastructure Demo Suite

Eight demos showcasing different fusion capabilities on Tenstorrent Wormhole hardware.

**Test file:** `tests/ttnn/unit_tests/operations/fused/fusion_demo/test_fused_demo.py`

```bash
# Run all tests:
python -m pytest tests/ttnn/unit_tests/operations/fused/fusion_demo/test_fused_demo.py -xvs

# Run perf demos only:
python -m pytest tests/ttnn/unit_tests/operations/fused/fusion_demo/test_fused_demo.py -xvs -k TestPerfDemos

# Run with Tracy device profiler (device_fw mode, skip GlobalCB demo):
export TT_METAL_DEVICE_PROFILER=1
python -m tracy -r -m pytest tests/ttnn/unit_tests/operations/fused/fusion_demo/test_fused_demo.py -xvs -k "device_fw and not global_circular"
```

All timing measured on Wormhole n150, BF16.

## How Timing Was Measured

**Device FW / kernel** come from Tracy's profiler CSV (`DEVICE FW DURATION [ns]` and `DEVICE KERNEL DURATION [ns]`), run with `perf_mode=device_fw` (one dispatch per op, no timing loops). FW includes firmware setup + kernel + teardown (~0.7 us overhead). Kernel is pure Tensix execution time. Fused ops appear as a single `GenericOpDeviceOperation` row; unfused ops appear as one row per `ttnn.*` dispatch. Device FW totals for unfused are the sum of all per-op FW durations.

**E2E steady state** is measured with `_time_steady_state()`: 5 warmup iterations followed by 100 timed iterations, all with JIT caches populated. This captures the full host-to-device-to-host round-trip per iteration.

**Cold start** is measured with `_time_fused()` / `_time_cold_warm()`: all caches cleared (JIT disk + in-memory + program + fusion build), then `build() + launch() + synchronize()`. This measures end-to-end cold dispatch including JIT compilation of the kernel binary.

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
Sequential(rms1, matmul, rms2).build(device)
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

Unfused breakdown: RMS #1 FW=8.0 us + matmul FW=8.7 us + RMS #2 FW=7.9 us = 24.6 us.

| Metric | Fused | Unfused | Speedup |
|--------|------:|--------:|--------:|
| Device FW | 27.1 us | 24.6 us (3 ops) | 0.91x |
| Device kernel | 26.4 us | 22.6 us | 0.86x |
| E2E steady state | 0.043 ms | 0.067 ms | **1.56x** |
| Cold start | 1714 ms | 2136 ms | **1.25x** |

At H=128, each op takes 7-8 us. The fused kernel is ~3 us slower than the unfused sum due to inter-phase barrier overhead. The **1.56x E2E speedup** comes from eliminating host dispatch gaps between ops.

### H=1536 (compute-dominated)

Unfused breakdown: RMS #1 FW=30.0 us + matmul FW=949.4 us + RMS #2 FW=572.5 us = 1551.9 us. (RMS #2 FW inflated by cold program loading; kernel = 30.3 us.)

| Metric | Fused | Unfused | Speedup |
|--------|------:|--------:|--------:|
| Device FW | 993.4 us | 1551.9 us (3 ops) | 1.56x |
| Device kernel | 992.8 us | 1008.4 us | 1.02x |
| E2E steady state | 0.998 ms | 1.005 ms | 1.01x |
| Cold start | 1758 ms | 2192 ms | **1.25x** |

At H=1536, the matmul dominates (~949 us). Device kernel times are essentially identical (the 1.56x FW speedup is an artifact of cold program loading in the unfused RMS #2 dispatch). **No structural device speedup** for a linear chain of DRAM-interleaved ops.

**PCC:** 0.9999 (both H values)

---

## Demo 2: Sharded Chain — RMS -> LN

Fusion with block-sharded memory layout. The CB allocator detects pinned buffer addresses from the shard spec and preserves them while pool-allocating other CB slots.

```
  RMS ──> LN
     (16 cores, block-sharded L1)
```

**API:**
```python
Sequential(rms, ln).build(device)
```

**Setup (parameterized on H):**
| | Shape | Memory |
|-|-------|--------|
| Input | `(1, 1, H, 512)` | BF16, block-sharded L1, 4x4 = 16 cores |
| Output | `(1, 1, H, 512)` | BF16, block-sharded L1, 4x4 = 16 cores |

**Program configs:** `LayerNormShardedMultiCoreProgramConfig`, 16 cores. `fp32=True`, `math_approx=False`, `HiFi4`.

### H=128 (dispatch-dominated)

Unfused breakdown: RMS FW=8.9 us + LN FW=13.4 us = 22.3 us.

| Metric | Fused | Unfused | Speedup |
|--------|------:|--------:|--------:|
| Device FW | 22.1 us | 22.3 us (2 ops) | 1.01x |
| Device kernel | 21.3 us | 20.5 us | 0.96x |
| E2E steady state | 0.088 ms | 0.060 ms | 0.68x |
| Cold start | 1699 ms | 2654 ms | **1.56x** |

At the device level, fused and unfused FW are nearly identical. The **E2E is 0.68x (slower fused)** because `generic_op` dispatch overhead (`override_runtime_arguments`) is larger than the native program-cache fast path used by individual `ttnn.rms_norm`/`ttnn.layer_norm` calls.

### H=1536 (compute-dominated)

Unfused breakdown: RMS FW=39.7 us + LN FW=63.6 us = 103.3 us.

| Metric | Fused | Unfused | Speedup |
|--------|------:|--------:|--------:|
| Device FW | 107.5 us | 103.3 us (2 ops) | 0.96x |
| Device kernel | 106.7 us | 101.6 us | 0.95x |
| E2E steady state | 0.121 ms | 0.104 ms | 0.86x |
| Cold start | 1676 ms | 2645 ms | **1.58x** |

The fused kernel is ~5 us slower than the unfused kernel sum (~107 vs ~103 us) — inter-phase barrier overhead. E2E is 0.86x for the same reason as H=128: `generic_op` dispatch overhead exceeds the saved dispatch gap.

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
Parallel(
    Sequential(ln_a, mm_a),
    Sequential(rms_b, mm_b),
).build(device)
```

**Setup:**
| | Shape | Memory |
|-|-------|--------|
| Input A | `(1, 1, 1024, 256)` | BF16, block-sharded L1 on `(0,0)-(0,7)` |
| Input B | `(1, 1, 1024, 256)` | BF16, block-sharded L1 on `(1,0)-(1,7)` |
| Weight B | `(1, 1, 256, 128)` | BF16, DRAM interleaved |

**Program configs:** LN/RMS on 1x8 grids, matmul `MatmulMultiCoreReuseProgramConfig`. `fp32=True`, `math_approx=False`, `HiFi4`.

Unfused breakdown: LN FW=44.8 us + matmul FW=18.2 us + RMS FW=26.8 us + matmul FW=18.0 us = 107.8 us. (Unfused runs 4 sequential dispatches on a single (0,0)-based grid.)

| Metric | Fused | Unfused | Speedup |
|--------|------:|--------:|--------:|
| Device FW | 70.0 us | 107.9 us (4 ops) | **1.54x** |
| Device kernel | 69.4 us | 105.4 us | **1.52x** |
| E2E steady state | 0.077 ms | 0.141 ms | **1.83x** |
| Cold start | 1613 ms | 3517 ms | **2.18x** |

The **1.54x device speedup** comes from parallelism — both chains overlap on disjoint core columns. Chain A (LN+MM ~ 45+18 us) and Chain B (RMS+MM ~ 27+18 us) run simultaneously, so fused time ~ `max(63, 45)` = 63 us. The **1.83x E2E speedup** additionally benefits from host dispatch elimination (4 dispatches -> 1).

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
Sequential(
    ln_stem,
    Parallel(
        Sequential(sl_top, mm_left, Parallel(Sequential(sl_tl, ln_ll), Sequential(sl_bl, ln_lr))),
        Sequential(sl_bot, mm_right, Parallel(Sequential(sl_tr, ln_rl), Sequential(sl_br, ln_rr))),
    ),
).build(device)
```

**Setup:**
| | Shape | Memory |
|-|-------|--------|
| Input | `(1, 1, 2048, 256)` | BF16, block-sharded L1, 2x8 |
| B_left, B_right | `(1, 1, 256, 128)` | BF16, DRAM interleaved |

**Program configs:** LN sharded, matmul `MatmulMultiCoreReuseProgramConfig`, slice tile-path with named CT args. `fp32=True`, `math_approx=False`, `HiFi4`.

Unfused breakdown (13 dispatches): ln_stem FW=42.6 us + 2 slices FW=11.9 us + 2 matmuls FW=36.0 us + 4 leaf slices FW=11.3 us + 4 leaf LNs FW=98.9 us = 200.8 us.

| Metric | Fused | Unfused | Speedup |
|--------|------:|--------:|--------:|
| Device FW | 120.9 us | 200.8 us (13 ops) | **1.66x** |
| Device kernel | 119.4 us | 191.1 us | **1.60x** |
| E2E steady state | 0.195 ms | 0.452 ms | **2.32x** |
| Cold start | 1917 ms | 5035 ms | **2.63x** |

The **1.66x device speedup** comes from branch parallelism — the fused kernel runs independent tree branches simultaneously on disjoint core subsets. The **2.32x E2E speedup** additionally eliminates 12 host dispatch gaps. The **2.63x cold start speedup** comes from JIT-compiling 1 fused kernel instead of 13 individual kernels.

**PCC:** 0.993 (leaf LN output vs unfused reference)

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
Sequential(
    ln_stem,
    Parallel(
        Sequential(sl_left, rms1, rms2),
        Sequential(sl_right, ln_right),
    ),
).build(device)
```

**Setup:**
| | Shape | Memory |
|-|-------|--------|
| Input | `(1, 1, 2048, 512)` | BF16, block-sharded L1, 4x8 = 32 cores |
| Weight (RMS) | `(1, 1, 1, 512)` | BF16, L1 width-sharded |

**Program configs:** LN/RMS `LayerNormShardedMultiCoreProgramConfig`, slice tile-path. `fp32=True`, `math_approx=False`, `HiFi4`.

Unfused breakdown (6 dispatches): stem LN FW=40.7 us + 2 slices FW=20.6 us + 2 branch RMS FW=51.0 us + branch LN FW=35.3 us = 147.6 us.

| Metric | Fused | Unfused | Speedup |
|--------|------:|--------:|--------:|
| Device FW | 120.8 us | 147.6 us (6 ops) | **1.22x** |
| Device kernel | 119.0 us | 142.3 us | **1.20x** |
| E2E steady state | 0.187 ms | 0.213 ms | **1.14x** |
| Cold start | 1998 ms | 4778 ms | **2.39x** |

The **1.22x device speedup** comes from the left branch (Slice+RMS+RMS ~ 10+25+25 us) running in parallel with the right branch (Slice+LN ~ 10+35 us). Fused time ~ stem (41 us) + max(60, 45) = 101 us kernel + barriers. The speedup is more modest than demos 3 and 4 because the stem LN (41 us FW, 32 cores) is a large serial component that cannot be parallelized.

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
Parallel(
    Sequential(gcb_sender, identity_phase1),
    gcb_consumer,
).build(device)
```

| Metric | Value |
|--------|------:|
| Cold start | 930 ms |
| Warm dispatch | 0.14 ms |

No unfused comparison (GlobalCB mid-kernel write has no unfused equivalent). No Tracy profiling (GlobalCB is incompatible with device profiler).

**PCC:** Receiver=1.000000, Phase 1=1.000000

---

## Demo 7: Non-Contiguous Core Grid ("Swiss Cheese")

Validates that the unicast barrier release works correctly on a `CoreRangeSet` with gaps. The stem op runs on rows 0-1, 3, and 5 (24 cores, skipping rows 2 and 4). Data flows into two parallel branches.

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
Sequential(stem, Parallel(op_a, op_b)).build(device)
```

| Metric | Value |
|--------|------:|
| Cold start | 910 ms |
| E2E steady state | 0.034 ms |

No unfused comparison (custom `SOURCE_CODE` identity ops). No Tracy profiling (identity ops have no `ttnn.*` equivalent for comparison).

**PCC:** A=1.0000, B=1.0000 (identity ops -- exact copy)

---

## Demo 8: Barrier Overhead Microbenchmark

Measures pure barrier mechanism cost by chaining N no-op phases (empty `kernel_main()`) so the only work is the inter-phase barrier synchronization itself.

```
  Phase 0 (no-op) ──barrier──> Phase 1 (no-op) ──barrier──> ... ──barrier──> Phase N (no-op)
```

**API:**
```python
Sequential(*[noop_op for _ in range(N)]).build(device)
```

**Setup:**
- Each phase has 3 kernels (reader, compute, writer) with empty bodies
- No CBs, no DRAM I/O, no compute -- barrier transitions dominate
- A single dummy DRAM tensor satisfies the `generic_op` tensor requirement
- Parametrized over `num_phases` (2-6) and `num_cores` (1, 8, 16, 64)

**Methodology:** Per-barrier cost = `(fused_N - fused_1) / (N - 1)`, where `fused_1` is a 1-phase baseline that captures fixed kernel launch overhead. At low N the baseline noise is a large fraction of the total, so per-barrier appears inflated; at high N (5-6) it converges to the true mechanism cost.

**Results (Wormhole n150, unicast barrier release):**

| Cores | Grid | 2 phases | 3 phases | 4 phases | 5 phases | 6 phases | Converged |
|------:|:-----|-------:|-------:|-------:|-------:|-------:|----------:|
| 1 | (0,0) | 4.6 us | 2.6 us | 1.9 us | 1.5 us | 1.6 us | ~1.5 us |
| 8 | 1x8 | 3.7 us | 2.1 us | 1.2 us | 1.6 us | 1.3 us | ~1.4 us |
| 16 | 2x8 | 4.0 us | 2.6 us | 1.9 us | 1.9 us | 1.5 us | ~1.5 us |
| 64 | 8x8 | 4.8 us | 1.8 us | 2.4 us | 1.9 us | 1.7 us | ~1.7 us |

**Per-barrier cost is ~1.5 us** regardless of core count (1 to 64). The barrier has two levels:

- **`local::sync()`** -- per-core 3-RISC rendezvous via L1 semaphores (compute_done, writer_done, reset_done). Coordinator (NCRISC) waits for followers, resets CB state, signals reset_done.
- **`group::sync()`** -- cross-core synchronization. Each core sends `noc_semaphore_inc` to core 0; core 0 waits for all arrivals, then unicast-releases each core individually.

At 64 cores the unicast release loop sends 63 individual `noc_async_write` packets, yet converged cost is only ~0.2 us more than 1 core. The NOC writes are pipelined and the per-write cost is ~30 ns.

**Multicast was benchmarked and removed.** Replacing the unicast release loop with a single `noc_async_write_multicast` for rectangular grids showed no measurable improvement (64-core 6-phase: 1.7 us unicast vs 1.5 us multicast -- within noise). The release loop was never the bottleneck. Unicast is kept because it works with non-contiguous core grids (e.g., DRAM-sharded matmul's scattered cores) without special-casing.

**Dispatch overhead comparison** (fused vs unfused, 6 phases):

| Cores | Fused | Unfused | Per-dispatch overhead |
|------:|------:|--------:|---------------------:|
| 1 | 21.7 us | 77.9 us | ~11.2 us |
| 8 | 21.6 us | 92.1 us | ~14.1 us |
| 16 | 24.5 us | 100.5 us | ~15.2 us |
| 64 | 36.2 us | 168.5 us | ~26.4 us |

Fusion eliminates ~11-26 us of host dispatch overhead per op (scales with core count due to RT arg setup).

---

## Summary

### Device-level timing (Tracy)

| Demo | Ops | Fused FW | Unfused FW | FW speedup | Source of speedup |
|------|----:|---------:|-----------:|--------:|:-----------------|
| 1 (H=128): Linear chain | 3 | 27.1 us | 24.6 us | 0.91x | No device speedup (linear chain) |
| 1 (H=1536): Linear chain | 3 | 993.4 us | 1008.4 us (K) | 1.02x | No device speedup (compute-dominated) |
| 2 (H=128): Sharded chain | 2 | 22.1 us | 22.3 us | 1.01x | No device speedup (linear chain) |
| 2 (H=1536): Sharded chain | 2 | 107.5 us | 103.3 us | 0.96x | No device speedup (compute-dominated) |
| 3: Parallel chains | 4 | 70.0 us | 107.9 us | **1.54x** | Parallelism (2 chains overlap) |
| 4: Sharded tree | 13 | 120.9 us | 200.8 us | **1.66x** | Branch parallelism (5-level tree) |
| 5: Asymmetric branches | 6 | 120.8 us | 147.6 us | **1.22x** | Branch parallelism (asymmetric) |

### Host-level timing (E2E steady state + cold start)

| Demo | Fused E2E | Unfused E2E | E2E speedup | Fused cold | Unfused cold |
|------|----------:|------------:|--------:|-----------:|-------------:|
| 1 (H=128) | 0.043 ms | 0.067 ms | **1.56x** | 1714 ms | 2136 ms |
| 1 (H=1536) | 0.998 ms | 1.005 ms | 1.01x | 1758 ms | 2192 ms |
| 2 (H=128) | 0.088 ms | 0.060 ms | 0.68x | 1699 ms | 2654 ms |
| 2 (H=1536) | 0.121 ms | 0.104 ms | 0.86x | 1676 ms | 2645 ms |
| 3 | 0.077 ms | 0.141 ms | **1.83x** | 1613 ms | 3517 ms |
| 4 | 0.195 ms | 0.452 ms | **2.32x** | 1917 ms | 5035 ms |
| 5 | 0.187 ms | 0.213 ms | **1.14x** | 1998 ms | 4778 ms |

**Key takeaways:**
- **Barrier overhead is ~1.5 us per transition** (demo 8), independent of core count (1-64 cores). Multicast release provides no measurable benefit over unicast.
- **Linear chains** (demos 1, 2): No device speedup -- fused FW ~ sum of unfused kernels + barrier overhead. Demo 1 gets E2E speedup from eliminating 2 dispatch roundtrips; demo 2 is slower fused because `generic_op` dispatch overhead exceeds the single saved dispatch gap.
- **Branching topologies** (demos 3, 4, 5): Device speedup from overlapping independent branches on disjoint core subsets. Demo 3 (1.54x) from 2 parallel chains; demo 4 (1.66x) from a full binary tree; demo 5 (1.22x) from asymmetric branches with a large serial stem.
- **Dispatch overhead is ~11-26 us per op** (demo 8), scaling with core count. Fusion eliminates this for all but the first dispatch.
- **Cold start** (including JIT compilation): Fused is **1.25-2.63x faster** because it JIT-compiles 1 fused kernel instead of N individual kernels. The advantage grows with op count: demo 4 (13 ops) sees 2.63x, demo 1 (3 ops) sees 1.25x. JIT compilation dominates cold start cost (~1-2s fused vs 2-5s unfused).

---

## Implementation Notes

1. **Compute configs must match across all phases.** `fp32_dest_acc_en`, `math_approx_mode`, and `math_fidelity` must be identical. Default configs differ by op type:
   - RMS norm: `fp32=False`, `math_approx=True`, `HiFi4`
   - Layer norm: `fp32=True`, `math_approx=True`, `HiFi4`
   - Matmul: `fp32=False`, `math_approx=False`, `LoFi`

   Pass explicit `WormholeComputeKernelConfig` to ensure consistency.

2. **Branching output order** depends on Python set hashing of core coordinates. Reference outputs through the original op descriptors (`op.output_tensors[0]`) rather than relying on positional indexing in `fused.output_tensors`.
