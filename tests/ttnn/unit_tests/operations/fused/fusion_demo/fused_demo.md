# Fusion Infrastructure Demo Suite

Seven demos showcasing different fusion capabilities on Tenstorrent Wormhole hardware.

**Test file:** `tests/ttnn/unit_tests/operations/fused/fusion_demo/test_fused_demo.py`

```bash
# Run tests:
python -m pytest tests/ttnn/unit_tests/operations/fused/fusion_demo/test_fused_demo.py -xvs

# Run with Tracy device profiler (single-run mode, skip GlobalCB demo):
export TT_METAL_DEVICE_PROFILER=1
python -m tracy -r -m pytest tests/ttnn/unit_tests/operations/fused/fusion_demo/test_fused_demo.py -xvs -k "single_run_only=True and not demo4"
```

All timing measured on Wormhole n150, BF16.

## How Timing Was Measured

**Device FW / kernel** come from Tracy's profiler CSV (`DEVICE FW DURATION [ns]` and `DEVICE KERNEL DURATION [ns]`), run with `single_run_only=True` (one dispatch per op, no timing loops). FW includes firmware setup + kernel + teardown (~0.7 us overhead). Kernel is pure Tensix execution time. Fused ops appear as a single `GenericOpDeviceOperation` row; unfused ops appear as one row per `ttnn.*` dispatch. Device FW totals for unfused are the sum of all per-op FW durations.

**E2E steady state** is measured with `_time_steady_state()`: 5 warmup iterations followed by 100 timed iterations, all with JIT caches populated. This captures the full host-to-device-to-host round-trip per iteration.

**Cold start** is measured with `_time_fused()` / `_time_cold_warm()`: `build() + launch() + synchronize()` with JIT caches populated. This measures Python-side fusion build cost (CB pool allocation, source generation, barrier setup), NOT JIT compilation.

### Apples-to-apples configs

Unfused `ttnn.matmul` calls use the same `program_config` and `compute_kernel_config` as their fused counterparts. Unfused `ttnn.rms_norm` / `ttnn.layer_norm` calls use `LayerNormShardedMultiCoreProgramConfig` with matching core counts and shard specs where possible.

---

## Demo 1: RMS -> Matmul -> RMS

Basic sequential chaining of heterogeneous ops (norm + matmul + norm) into a single fused kernel dispatch. Intermediates round-trip through DRAM between phases.

**Setup (parameterized on H):**
| | Shape | Memory |
|-|-------|--------|
| Input | `(1, 1, 256, H)` | BF16, DRAM interleaved |
| Weight (norm) | `(1, 1, 1, H)` | BF16, DRAM interleaved |
| Weight B (matmul) | `(1, 1, H, H)` | BF16, DRAM interleaved |
| Output | `(1, 1, 256, H)` | BF16, DRAM interleaved |

**Program configs:** RMS + Matmul on `(0,0)-(3,1)` = 8 cores. `fp32=False`, `math_approx=True`, `HiFi4`.

### H=128 (dispatch-dominated)

Unfused breakdown: RMS #1 FW=7.8 us + matmul FW=8.9 us + RMS #2 FW=7.8 us = 24.5 us.

| Metric | Fused | Unfused | Speedup |
|--------|------:|--------:|--------:|
| Device FW | 27.2 us | 24.5 us (3 ops) | 0.90x |
| Device kernel | 26.6 us | 22.6 us | 0.85x |
| E2E steady state | 0.043 ms | 0.065 ms | **1.51x** |
| Cold start | 39.89 ms | 21.90 ms | 0.55x |

At H=128, each op takes 7-8 us. The fused kernel is ~3 us slower than the unfused sum due to inter-phase barrier overhead. The **1.51x E2E speedup** comes from eliminating host dispatch gaps between ops.

### H=1536 (compute-dominated)

Unfused breakdown: RMS #1 FW=30.0 us + matmul FW=929.9 us + RMS #2 FW=248.3 us = 1208.2 us. (RMS #2 FW inflated by cold program loading; kernel = 30.3 us.)

| Metric | Fused | Unfused | Speedup |
|--------|------:|--------:|--------:|
| Device FW | 1020.8 us | 1208.2 us (3 ops) | 1.18x |
| Device kernel | 1020.2 us | 988.9 us | 0.97x |
| E2E steady state | 0.993 ms | 0.998 ms | 1.01x |
| Cold start | 38.61 ms | 21.41 ms | 0.55x |

At H=1536, the matmul dominates (~930 us). Device kernel times are essentially identical (the 1.18x FW speedup is an artifact of cold program loading in the unfused RMS #2 dispatch). **No structural device speedup** for a linear chain of DRAM-interleaved ops.

**PCC:** 0.9999 (both H values)

## Demo 2: RMS -> LN (Block-Sharded)

Fusion with block-sharded memory layout. The CB allocator detects pinned buffer addresses from the shard spec and preserves them while pool-allocating other CB slots.

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
| Device FW | 22.2 us | 22.3 us (2 ops) | 1.00x |
| Device kernel | 21.4 us | 20.6 us | 0.96x |
| E2E steady state | 0.086 ms | 0.066 ms | 0.77x |
| Cold start | 49.58 ms | 23.11 ms | 0.47x |

At the device level, fused and unfused FW are nearly identical. The **E2E is 0.77x (slower fused)** because `generic_op` dispatch overhead (`override_runtime_arguments`) is larger than the native program-cache fast path used by individual `ttnn.rms_norm`/`ttnn.layer_norm` calls.

### H=1536 (compute-dominated)

Unfused breakdown: RMS FW=39.6 us + LN FW=63.7 us = 103.3 us.

| Metric | Fused | Unfused | Speedup |
|--------|------:|--------:|--------:|
| Device FW | 107.6 us | 103.3 us (2 ops) | 0.96x |
| Device kernel | 106.9 us | 101.6 us | 0.95x |
| E2E steady state | 0.121 ms | 0.105 ms | 0.87x |
| Cold start | 45.21 ms | 19.43 ms | 0.43x |

The fused kernel is ~5 us slower than the unfused kernel sum — inter-phase barrier overhead. E2E is 0.87x for the same reason as H=128: `generic_op` dispatch overhead exceeds the saved dispatch gap.

**PCC:** 1.000000 (both H values)

## Demo 3: Parallel Independent Chains (Block-Sharded)

Two independent 2-op chains running on disjoint 1x8 core columns within a single kernel dispatch. No inter-chain synchronization needed.

**Setup:**
| | Shape | Memory |
|-|-------|--------|
| Input A | `(1, 1, 1024, 256)` | BF16, block-sharded L1 on `(0,0)-(0,7)` |
| Input B | `(1, 1, 1024, 256)` | BF16, block-sharded L1 on `(1,0)-(1,7)` |
| Weight B | `(1, 1, 256, 128)` | BF16, DRAM interleaved |

**Program configs:** LN/RMS on 1x8 grids, matmul `MatmulMultiCoreReuseProgramConfig`. `fp32=True`, `math_approx=False`, `HiFi4`.

**API:**
```python
Parallel(
    Sequential(ln_a, mm_a),
    Sequential(rms_b, mm_b),
).build(device)
```

Unfused breakdown: LN FW=44.7 us + matmul FW=18.0 us + RMS FW=26.9 us + matmul FW=17.9 us = 107.5 us. (Unfused runs 4 sequential dispatches on a single (0,0)-based grid.)

| Metric | Fused | Unfused | Speedup |
|--------|------:|--------:|--------:|
| Device FW | 70.0 us | 107.5 us (4 ops) | **1.54x** |
| Device kernel | 69.3 us | 105.0 us | **1.52x** |
| E2E steady state | 0.078 ms | 0.151 ms | **1.94x** |
| Cold start | 50.47 ms | 33.86 ms | 0.67x |

The **1.54x device speedup** comes from parallelism — both chains overlap on disjoint core columns. Chain A (LN+MM ~ 45+18 us) and Chain B (RMS+MM ~ 27+18 us) run simultaneously, so fused time ~ `max(63, 45)` = 63 us. The **1.94x E2E speedup** additionally benefits from host dispatch elimination (4 dispatches -> 1).

**PCC:** Chain A = 1.0000, Chain B = 1.0000

## Demo 4: GlobalCircularBuffer Mid-Kernel Write

Data exfiltration from the middle of a fused kernel via `GlobalCircularBuffer`. The sender pushes data to an external consumer core during kernel execution, before finishing all phases.

**Architecture:**
```
Sender core (0,0):                    Receiver core (1,0):
  Phase 0: DRAM(A) -> GlobalCB push     GlobalCB -> DRAM(output_recv)
  Phase 1: DRAM(B) -> DRAM(output_b)
```

| Metric | Value |
|--------|------:|
| Cold start | 907 ms |
| Warm dispatch | 0.19 ms |

No unfused comparison (GlobalCB mid-kernel write has no unfused equivalent). No Tracy profiling (GlobalCB is incompatible with device profiler).

**PCC:** Receiver=1.000000, Phase 1=1.000000

## Demo 5: Block-Sharded Branching Tree (LN -> Slice -> Matmul -> Slice -> LN)

Full block-sharded tree topology with 13 ops across 5 levels on a 2x8 core grid with hierarchical core subset splitting.

**Topology:**
```
                            ln_stem (2x8 = 16 cores)
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

**Setup:**
| | Shape | Memory |
|-|-------|--------|
| Input | `(1, 1, 2048, 256)` | BF16, block-sharded L1, 2x8 |
| B_left, B_right | `(1, 1, 256, 128)` | BF16, DRAM interleaved |

**Program configs:** LN sharded, matmul `MatmulMultiCoreReuseProgramConfig`, slice tile-path with named CT args. `fp32=True`, `math_approx=False`, `HiFi4`.

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

Unfused breakdown (13 dispatches): ln_stem FW=42.6 us + 2 slices FW=11.9 us + 2 matmuls FW=35.8 us + 4 leaf slices FW=11.3 us + 4 leaf LNs FW=99.0 us = 200.6 us.

| Metric | Fused | Unfused | Speedup |
|--------|------:|--------:|--------:|
| Device FW | 119.9 us | 200.6 us (13 ops) | **1.67x** |
| Device kernel | 118.5 us | 191.0 us | **1.61x** |
| E2E steady state | 0.199 ms | 0.453 ms | **2.28x** |
| Cold start | 100.40 ms | 38.31 ms | 0.38x |

The **1.67x device speedup** comes from branch parallelism — the fused kernel runs independent tree branches simultaneously on disjoint core subsets. The **2.28x E2E speedup** additionally eliminates 12 host dispatch gaps. The **cold start** is slower fused because the fusion build (source gen + CB allocation for 13 ops) takes longer than dispatching 13 individual cached programs.

**PCC:** 0.993 (leaf LN output vs unfused reference)

## Demo 6: Asymmetric Parallel Branches with Common Stem

A common stem LN fans out into two asymmetric branches: a lightweight chain (Slice + RMS + RMS) on the left, and a heavier single LN on the right. The lightweight branch runs hidden behind the heavy branch on disjoint core columns. Unfused, all 6 ops serialize.

**Topology:**
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

**Setup:**
| | Shape | Memory |
|-|-------|--------|
| Input | `(1, 1, 2048, 512)` | BF16, block-sharded L1, 4x8 = 32 cores |
| Weight (RMS) | `(1, 1, 1, 512)` | BF16, L1 width-sharded |

**Program configs:** LN/RMS `LayerNormShardedMultiCoreProgramConfig`, slice tile-path. `fp32=True`, `math_approx=False`, `HiFi4`.

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

Unfused breakdown (6 dispatches): stem LN FW=40.7 us + 2 slices FW=20.6 us + 2 branch RMS FW=51.0 us + branch LN FW=35.3 us = 147.6 us.

| Metric | Fused | Unfused | Speedup |
|--------|------:|--------:|--------:|
| Device FW | 121.3 us | 147.6 us (6 ops) | **1.22x** |
| Device kernel | 119.6 us | 142.4 us | **1.19x** |
| E2E steady state | 0.190 ms | 0.224 ms | **1.18x** |
| Cold start | 89.57 ms | 33.49 ms | 0.37x |

The **1.22x device speedup** comes from the left branch (Slice+RMS+RMS ~ 10+25+25 us) running in parallel with the right branch (Slice+LN ~ 10+35 us). Fused time ~ stem (41 us) + max(60, 45) = 101 us kernel + barriers. The speedup is more modest than demos 3 and 5 because the stem LN (41 us FW, 32 cores) is a large serial component that cannot be parallelized.

**PCC:** Left chain = 1.0000, Right LN = 0.9998

## Demo 7: Non-Contiguous Core Grid ("Swiss Cheese")

Validates that the unicast barrier release works correctly on a `CoreRangeSet` with gaps. The stem op runs on rows 0-1, 3, and 5 (24 cores, skipping rows 2 and 4). Data flows into two parallel branches.

**Core grid:**
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
| Cold start | 969 ms |
| E2E steady state | 0.034 ms |

No unfused comparison (custom `SOURCE_CODE` identity ops). No Tracy profiling (identity ops have no `ttnn.*` equivalent for comparison).

**PCC:** A=1.0000, B=1.0000 (identity ops — exact copy)

---

## Summary

### Device-level timing (Tracy)

| Demo | Ops | Fused FW | Unfused FW | FW speedup | Source of speedup |
|------|----:|---------:|-----------:|--------:|:-----------------|
| 1 (H=128): RMS->MM->RMS | 3 | 27.2 us | 24.5 us | 0.90x | No device speedup (linear chain) |
| 1 (H=1536): RMS->MM->RMS | 3 | 1020.8 us | 988.9 us (K) | 0.97x | No device speedup (compute-dominated) |
| 2 (H=128): RMS->LN sharded | 2 | 22.2 us | 22.3 us | 1.00x | No device speedup (linear chain) |
| 2 (H=1536): RMS->LN sharded | 2 | 107.6 us | 103.3 us | 0.96x | No device speedup (compute-dominated) |
| 3: Parallel chains | 4 | 70.0 us | 107.5 us | **1.54x** | Parallelism (2 chains overlap) |
| 5: Sharded tree | 13 | 119.9 us | 200.6 us | **1.67x** | Branch parallelism (5-level tree) |
| 6: Asymmetric branches | 6 | 121.3 us | 147.6 us | **1.22x** | Branch parallelism (asymmetric) |

### Host-level timing (E2E steady state + cold start)

| Demo | Fused E2E | Unfused E2E | E2E speedup | Fused cold | Unfused cold |
|------|----------:|------------:|--------:|-----------:|-------------:|
| 1 (H=128) | 0.043 ms | 0.065 ms | **1.51x** | 39.89 ms | 21.90 ms |
| 1 (H=1536) | 0.993 ms | 0.998 ms | 1.01x | 38.61 ms | 21.41 ms |
| 2 (H=128) | 0.086 ms | 0.066 ms | 0.77x | 49.58 ms | 23.11 ms |
| 2 (H=1536) | 0.121 ms | 0.105 ms | 0.87x | 45.21 ms | 19.43 ms |
| 3 | 0.078 ms | 0.151 ms | **1.94x** | 50.47 ms | 33.86 ms |
| 5 | 0.199 ms | 0.453 ms | **2.28x** | 100.40 ms | 38.31 ms |
| 6 | 0.190 ms | 0.224 ms | **1.18x** | 89.57 ms | 33.49 ms |

**Key takeaways:**
- **Linear chains** (demos 1, 2): No device speedup — fused FW ~ sum of unfused kernels + barrier overhead. Demo 1 gets E2E speedup from eliminating 2 dispatch roundtrips; demo 2 is slower fused because `generic_op` dispatch overhead exceeds the single saved dispatch gap.
- **Branching topologies** (demos 3, 5, 6): Device speedup from overlapping independent branches on disjoint core subsets. Demo 3 (1.54x) from 2 parallel chains; demo 5 (1.67x) from a full binary tree; demo 6 (1.22x) from asymmetric branches with a large serial stem.
- **Cold start**: Fused is slower than unfused (with JIT caches populated) because the Python-side fusion build (source gen, CB allocation, barrier setup) adds 40-100 ms vs 20-40 ms for dispatching individual cached programs. Without JIT caches (first process invocation), fused is faster because it JIT-compiles 1 kernel instead of N.

---

## Implementation Notes

1. **Compute configs must match across all phases.** `fp32_dest_acc_en`, `math_approx_mode`, and `math_fidelity` must be identical. Default configs differ by op type:
   - RMS norm: `fp32=False`, `math_approx=True`, `HiFi4`
   - Layer norm: `fp32=True`, `math_approx=True`, `HiFi4`
   - Matmul: `fp32=False`, `math_approx=False`, `LoFi`

   Pass explicit `WormholeComputeKernelConfig` to ensure consistency.

2. **Branching output order** depends on Python set hashing of core coordinates. Reference outputs through the original op descriptors (`op.output_tensors[0]`) rather than relying on positional indexing in `fused.output_tensors`.
