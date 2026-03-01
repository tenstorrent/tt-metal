# Fusion Infrastructure Demo Suite

Five demos showcasing different fusion capabilities on Tenstorrent Wormhole hardware.

**Test file:** `tests/ttnn/unit_tests/operations/fused/fusion_demo/test_fused_demo.py`

```bash
# Run tests:
python -m pytest tests/ttnn/unit_tests/operations/fused/fusion_demo/test_fused_demo.py -xvs

# Run with Tracy device profiler (skip GlobalCB demo):
TT_METAL_DEVICE_PROFILER=1 python -m tracy -r -m "pytest tests/ttnn/unit_tests/operations/fused/fusion_demo/test_fused_demo.py -xvs -k 'not demo4'"

# Run in Tracy single-run mode (for device profiling without timing loops):
# Change single_run_only parametrize from [False] to [True] in the test file.
```

All timing measured on Wormhole n150, BF16.

## How Timing Was Measured

All device-side timing comes from Tracy's profiler CSV, generated at:

```
generated/profiler/reports/<timestamp>/ops_perf_results_<timestamp>.csv
```

Three CSV columns are used:

| Column | What it measures |
|--------|-----------------|
| `DEVICE FW DURATION [ns]` | Time from firmware start to firmware end on the device. Includes kernel execution **plus** firmware setup before the kernel and teardown after it (~0.7 us overhead per op). In the tables below this is **FW**. |
| `DEVICE KERNEL DURATION [ns]` | Time from the earliest RISC kernel start cycle to the latest RISC kernel end cycle across all cores. This is the pure Tensix compute time, a subset of `FW DURATION`. In the tables below this is **Kernel**. |
| `OP TO OP LATENCY [ns]` | Gap between the previous op's firmware end and the current op's firmware start. This is the **host-side latency**: Python ttnn framework, C++ operation dispatch, command queue enqueue, and the time for the command to reach the device. In the tables below this is **Host gap**. |

The device timeline for consecutive unfused ops looks like:

```
|--FW₁: [setup|kernel₁|teardown]--|--host gap--|--FW₂: [setup|kernel₂|teardown]--|
                                  ^            ^
                             FW₁ end      FW₂ start
                                  |____________|
                                  OP TO OP LATENCY
```

**End-to-end** is computed as total device time from first FW start to last FW end:

```
E2E = FW₁ + host_gap₁₂ + FW₂ + host_gap₂₃ + ... + FWₙ
```

For the fused op, E2E is just its single `FW DURATION` (one firmware window, no host gaps). The initial dispatch of the first op is excluded from both sides because it cancels out.

**Compile fused** is measured with `time.perf_counter()` around `Sequential.build(device)` in Python. This is the Python-side fusion cost (CB pool allocation, source generation, barrier setup). It does NOT include JIT compilation.

### Warmup

The first invocation of each op+config variant in a new process incurs ~15 ms of JIT cache deserialization (loading compiled programs from the on-disk cache into per-process memory). This one-time cost dominates the `OP TO OP LATENCY` column and obscures the true steady-state dispatch overhead.

To avoid this, the **steady-state measurement** uses `_time_steady_state()` with 5 warmup iterations followed by 100 timed iterations, all with caches populated. In the Tracy CSV, warmup ops appear with large `OP TO OP LATENCY` values (millions of ns); the timed ops appear as later rows with small values (30–60 us). Only the steady-state rows are reported below.

Fused ops appear in the CSV as `GenericOpDeviceOperation`.

### Apples-to-apples configs

Unfused `ttnn.matmul` calls use the same `program_config` and `compute_kernel_config` as their fused counterparts so both the core count and compute config match. Without explicit configs, `ttnn.matmul` auto-selects a multi-cast program that uses up to 64 cores.

Unfused `ttnn.rms_norm` and `ttnn.layer_norm` calls use the default interleaved grid because the ttnn Python API does not expose a `core_range` parameter for interleaved inputs. This means unfused norm ops may use more cores than their fused counterparts. Where this occurs, it makes the unfused kernel times a **lower bound** (more cores = faster kernel), so the comparison is **conservative** — the actual speedup from fusion is at least as large as reported.

---

## Demo 1: RMS -> Matmul -> RMS

Basic sequential chaining of heterogeneous ops (norm + matmul + norm) into a single fused kernel dispatch. Each phase still writes its output to DRAM and the next phase reads from DRAM — intermediates are **not** kept in L1. At small H (dispatch-dominated), the speedup comes from eliminating host-side dispatch gaps. At large H (compute-dominated), there is no device speedup.

**Setup (parameterized on H):**
| | Shape | Memory |
|-|-------|--------|
| Input | `(1, 1, 256, H)` | BF16, DRAM interleaved |
| Weight (norm) | `(1, 1, 1, H)` | BF16, DRAM interleaved |
| Weight B (matmul) | `(1, 1, H, H)` | BF16, DRAM interleaved |
| Output | `(1, 1, 256, H)` | BF16, DRAM interleaved |

**Program configs:**
- **RMS #1, RMS #2:** Interleaved, `(0,0)-(3,1)` = 4x2 = 8 cores
- **Matmul:** `MatmulMultiCoreReuseProgramConfig(compute_with_storage_grid_size=(4, 2), in0_block_w=4, out_subblock_h=1, out_subblock_w=4, per_core_M=1, per_core_N=H/32)`
- **Compute:** `fp32=False`, `math_approx=True`, `HiFi4` (all 3 phases)

**Dataflow:**
```
Input (DRAM) -> [RMS norm -> Matmul -> RMS norm] -> Output (DRAM)
                  L1 intermediate   L1 intermediate
```

### H=128 (small — dispatch-dominated)

**Timing (Tracy):**

| | FW | Kernel | Cores |
|--|---:|------:|------:|
| **Fused (1 dispatch)** | **26.9 us** | **26.3 us** | 8 |
| Unfused RMS #1 | 8.4 us | 7.7 us | 8 |
| Unfused matmul | 7.9 us | 7.2 us | 8 |
| Unfused RMS #2 | 8.1 us | 7.4 us | 8 |
| **Unfused total (3 dispatches)** | **24.3 us** | **22.3 us** | |

| Metric | Fused | Unfused | Speedup |
|--------|------:|--------:|--------:|
| Cold start | 1,681 ms | 2,075 ms | **1.23x** |
| E2E steady state | 0.042 ms | 0.066 ms | **1.57x** |

At H=128, each op takes 7-8 us on device. The **1.57x E2E speedup** comes entirely from eliminating host dispatch gaps between ops. The fused kernel is ~3 us slower than the unfused sum due to inter-phase barrier overhead.

### H=1536 (large — compute-dominated)

**Timing (Tracy):**

| | FW | Kernel | Cores |
|--|---:|------:|------:|
| **Fused (1 dispatch)** | **957.7 us** | **957.0 us** | 8 |

| Metric | Fused | Unfused | Speedup |
|--------|------:|--------:|--------:|
| Cold start | 1,688 ms | 2,119 ms | **1.26x** |
| E2E steady state | 0.993 ms | 0.994 ms | 1.00x |

At H=1536, the matmul dominates (~930 us). Host dispatch gaps are noise compared to compute time. **Intermediates still round-trip through DRAM** in the fused path — each phase's writer writes to DRAM and the next phase's reader reads from DRAM, identical to unfused. There is **no structural device speedup** for a linear chain of DRAM-interleaved ops.

**PCC:** 0.9999 (both H values)

## Demo 2: RMS -> LN (Block-Sharded)

Fusion with block-sharded memory layout. The CB allocator detects pinned buffer addresses from the shard spec and preserves them while pool-allocating other CB slots. Rebind addresses are passed as runtime args so the JIT cache is not busted when tensor allocations change.

**Setup (parameterized on H):**
| | Shape | Memory |
|-|-------|--------|
| Input | `(1, 1, H, 512)` | BF16, block-sharded L1 |
| Weight | `(1, 1, 1, 512)` | BF16, L1 interleaved |
| Output | `(1, 1, H, 512)` | BF16, block-sharded L1 |

- **Shard spec:** `(H/4, 128)` per core, `ROW_MAJOR` orientation, 4x4 = 16 cores

**Program configs:**
- **RMS, LN:** `LayerNormShardedMultiCoreProgramConfig(compute_with_storage_grid_size=(4, 4), subblock_w=4, block_h=H/128, block_w=4, inplace=False)`, 4x4 = 16 cores
- **Compute:** `fp32=True`, `math_approx=False`, `HiFi4` (both phases)

### H=128 (small — dispatch-dominated)

**Timing (Tracy):**

| | FW | Kernel | Cores |
|--|---:|------:|------:|
| **Fused (1 dispatch)** | **22.2 us** | **21.3 us** | 16 |
| Unfused RMS | 13.4 us | 12.5 us | 16 |
| Unfused LN | 9.2 us | 8.2 us | 16 |
| **Unfused total (2 dispatches)** | **22.6 us** | **20.7 us** | |

| Metric | Fused | Unfused | Speedup |
|--------|------:|--------:|--------:|
| Cold start | 1,609 ms | 2,572 ms | **1.60x** |
| E2E steady state | 0.085 ms | 0.059 ms | 0.69x |
| Device FW (Tracy) | 22.2 us | 22.6 us | 1.02x |
| Device kernel (Tracy) | 21.3 us | 20.7 us | 0.97x |

At H=128, each op takes 9-13 us. At the Tracy device level, fused and unfused FW are nearly identical (22.2 vs 22.6 us). However, the **E2E steady state is 0.69x (slower fused)** because `generic_op` dispatch overhead (RT-arg rebuild via `override_runtime_arguments`) is larger than the native program-cache fast path used by individual `ttnn.rms_norm`/`ttnn.layer_norm` calls. The build cache is NOT in the E2E timing loop — this is purely dispatch overhead.

### H=1536 (large — compute-dominated)

**Timing (Tracy):**

| | FW | Kernel | Cores |
|--|---:|------:|------:|
| **Fused (1 dispatch)** | **107.7 us** | **106.8 us** | 16 |
| Unfused RMS | 42.2 us | 40.5 us | 16 |
| Unfused LN | 66.0 us | 64.3 us | 16 |
| **Unfused total** | **108.2 us** | **104.8 us** | |

| Metric | Fused | Unfused | Speedup |
|--------|------:|--------:|--------:|
| Cold start | 1,599 ms | 2,565 ms | **1.60x** |
| E2E steady state | 0.120 ms | 0.105 ms | 0.88x |
| Device FW (Tracy) | 107.7 us | 108.2 us | 1.00x |
| Device kernel (Tracy) | 106.8 us | 104.8 us | 0.98x |

At H=1536, the sharded norm ops take 42-66 us each. The fused kernel (106.8 us) is 2.0 us slower than the unfused kernel sum (104.8 us) — the inter-phase barrier overhead. The E2E is 0.88x (slower fused) for the same reason as H=128: `generic_op` dispatch overhead exceeds the saved dispatch gap.

**PCC:** 1.000000 (both H values)

## Demo 3: Parallel Independent Chains (Block-Sharded)

Two completely independent 2-op chains running on disjoint 1×8 core columns within a single kernel dispatch. No inter-chain synchronization needed. Inputs are block-sharded in L1; weight and norm parameters are L1 interleaved.

**Setup:**
| | Shape | Memory |
|-|-------|--------|
| Input A | `(1, 1, 1024, 256)` | BF16, block-sharded L1 on `(0,0)-(0,7)` |
| Input B | `(1, 1, 1024, 256)` | BF16, block-sharded L1 on `(1,0)-(1,7)` |
| Weight B | `(1, 1, 256, 128)` | BF16, DRAM interleaved |
| Weight (norm) | `(1, 1, 1, 256)` | BF16, L1 width-sharded |
| Output A | `(1, 1, 1024, 128)` | BF16, block-sharded L1 on `(0,0)-(0,7)` |
| Output B | `(1, 1, 1024, 128)` | BF16, block-sharded L1 on `(1,0)-(1,7)` |

- **Shard spec (input):** `(128, 256)` per core, `ROW_MAJOR`, 1×8 grid
- **Shard spec (output):** `(128, 128)` per core, `ROW_MAJOR`, 1×8 grid

**Program configs:**
- **Chain A — LN:** `LayerNormShardedMultiCoreProgramConfig`, `(0,0)-(0,7)` = 1×8 = 8 cores
- **Chain A — Matmul:** `MatmulMultiCoreReuseProgramConfig(compute_with_storage_grid_size=(1, 8), in0_block_w=8, out_subblock_h=1, out_subblock_w=4, per_core_M=4, per_core_N=4)`
- **Chain B — RMS:** `LayerNormShardedMultiCoreProgramConfig`, `(1,0)-(1,7)` = 1×8 = 8 cores
- **Chain B — Matmul:** Same config as Chain A
- **Compute:** `fp32=True`, `math_approx=False`, `HiFi4` (all 4 ops)

**API:**
```python
Parallel(
    Sequential(ln_a, mm_a),
    Sequential(rms_b, mm_b),
).build(device)
```

**Fused vs unfused comparison:**

The fused kernel runs both chains **simultaneously** on 16 cores (8 per chain). The fused kernel time is approximately `max(chain_a, chain_b)` since the chains overlap. The unfused comparison runs 4 sequential `ttnn` dispatches on a single (0,0)-based 1×8 grid (since `ttnn.matmul` with `MatmulMultiCoreReuseProgramConfig` requires (0,0)-origin grids).

**Timing (Tracy):**

| | FW | Kernel | Cores |
|--|---:|------:|------:|
| **Fused (1 dispatch)** | **65.0 us** | **64.2 us** | 16 (8+8 parallel) |
| Unfused LN (chain A) | 26.8 us | 26.2 us | 8 |
| Unfused matmul (chain A) | 18.0 us | 17.4 us | 8 |
| Unfused RMS (chain B) | 45.5 us | 44.8 us | 8 |
| Unfused matmul (chain B) | 19.0 us | 18.4 us | 8 |
| **Unfused total (4 ops)** | **109.3 us** | **106.8 us** | |
| **FW speedup** | | | **1.68x** |

| Metric | Fused | Unfused | Speedup |
|--------|------:|--------:|--------:|
| Cold start | 1,521 ms | 3,408 ms | **2.24x** |
| E2E steady state | 0.077 ms | 0.146 ms | **1.90x** |
| Device FW (Tracy) | 65.0 us | 109.3 us | **1.68x** |
| Device kernel (Tracy) | 64.2 us | 106.8 us | **1.66x** |

The **1.68x device speedup** comes from parallelism — both chains overlap on disjoint core columns. Chain A (LN+MM ≈ 45 us) and Chain B (RMS+MM ≈ 64 us) run simultaneously, so fused time ≈ `max(45, 64)` = 64 us. The **1.90x E2E speedup** additionally benefits from host dispatch elimination (4 dispatches → 1).

**PCC:** Chain A (LN->MM) = 1.0000, Chain B (RMS->MM) = 1.0000

## Demo 4: GlobalCircularBuffer Mid-Kernel Write

Data exfiltration from the middle of a fused kernel via `GlobalCircularBuffer`. The sender pushes data to an external consumer core during kernel execution, before finishing all phases.

**Architecture:**
```
Sender core (0,0):                    Receiver core (1,0):
  Phase 0: DRAM(A) -> GlobalCB push     GlobalCB -> DRAM(output_recv)
  Phase 1: DRAM(B) -> DRAM(output_b)
```

**Setup:**
| | Details |
|-|---------|
| Tiles | 8 tiles of 32x32 BF16 (shape `(1, 1, 32, 256)`) |
| Sender | core `(0,0)`, 2 phases (GlobalCB push + identity copy) |
| Receiver | core `(1,0)`, standalone consumer program |
| GlobalCB | 2-tile double-buffer (`4096` bytes) |

**Program configs:**
- **gcb_sender (phase 0):** Custom reader + compute + GlobalCB writer on `(0,0)`, 1 core
- **identity (phase 1):** Custom reader + compute + DRAM writer on `(0,0)`, 1 core
- **gcb_consumer:** Custom GlobalCB reader + DRAM writer on `(1,0)`, 1 core
- All kernels are `SOURCE_CODE` (not factory descriptors, no `ProgramConfig` objects)

- **Compute:** Identity tile copy (unary `copy_tile`)

**Timing:**

| Metric | Value |
|--------|------:|
| Cold start | 907 ms |
| Warm dispatch | 0.19 ms |

No unfused comparison (GlobalCB mid-kernel write has no unfused equivalent). No Tracy profiling (GlobalCB is incompatible with device profiler).

**PCC:** Receiver=1.000000, Phase 1=1.000000

## Demo 5: Block-Sharded Branching Tree (LN → Slice → Matmul → Slice → LN)

Full block-sharded tree topology with 13 ops across 5 levels. Exercises all three descriptor-based op types (LayerNorm, Slice, Matmul) on a 2×8 core grid with hierarchical core subset splitting. The unfused comparison uses `ttnn.*` API calls with matching sharded memory configs on (0,0)-based grids.

**Topology:**
```
                            ln_stem (2×8 = 16 cores)
                                    |
                    +───────────────+───────────────+
                    |                               |
               sl_top (1×8=8)                  sl_bot (1×8=8)
              row 0, cols 0-7                 row 1, cols 0-7
                    |                               |
              mm_left (1×8=8)                mm_right (1×8=8)
                    |                               |
              +─────+─────+                   +─────+─────+
              |           |                   |           |
         sl_tl (1×4)  sl_bl (1×4)       sl_tr (1×4)  sl_br (1×4)
        r0,c0-3      r0,c4-7           r1,c0-3      r1,c4-7
              |           |                   |           |
         ln_ll (1×4)  ln_lr (1×4)       ln_rl (1×4)  ln_rr (1×4)
```

**Setup:**
| | Shape | Tiles | Memory |
|-|-------|-------|--------|
| Input | `(1, 1, 2048, 256)` | 64×8 | BF16, block-sharded L1 |
| B_left | `(1, 1, 256, 128)` | 8×4 | BF16, DRAM interleaved |
| B_right | `(1, 1, 256, 128)` | 8×4 | BF16, DRAM interleaved |

**Core assignments:**
| Cores | Range | Count | Used by |
|-------|-------|------:|---------|
| `stem_cores` | `(0,0)-(1,7)` | 16 | `ln_stem` |
| `left_cores` | `(0,0)-(0,7)` | 8 | `sl_top`, `mm_left` |
| `right_cores` | `(1,0)-(1,7)` | 8 | `sl_bot`, `mm_right` |
| `ll_cores` | `(0,0)-(0,3)` | 4 | `sl_tl`, `ln_ll` |
| `lr_cores` | `(0,4)-(0,7)` | 4 | `sl_bl`, `ln_lr` |
| `rl_cores` | `(1,0)-(1,3)` | 4 | `sl_tr`, `ln_rl` |
| `rr_cores` | `(1,4)-(1,7)` | 4 | `sl_br`, `ln_rr` |

**Shard specs (all block-sharded, ROW_MAJOR):**
| Config key | Tensor shape | Shard shape | Grid | Used after |
|------------|-------------|-------------|------|-----------|
| `stem` | `2048×256` | `[256, 128]` | 2×8 | Input, `ln_stem` output |
| `left` / `right` | `1024×256` | `[128, 256]` | 1×8 | `sl_top` / `sl_bot` output |
| `mm_left` / `mm_right` | `1024×128` | `[128, 128]` | 1×8 | `mm_left` / `mm_right` output |
| `ll`/`lr`/`rl`/`rr` | `512×128` | `[128, 128]` | 1×4 | Leaf slice output, leaf LN input |

**Program configs:**
- **ln_stem, leaf LNs:** `LayerNormShardedMultiCoreProgramConfig`, auto-detected from block-sharded input grid. Multicast sender (row 0) / receiver (row 1) kernels for `ln_stem`; single-row for leaf LNs.
- **Matmul:** `MatmulMultiCoreReuseProgramConfig(compute_with_storage_grid_size=(1, 8), in0_block_w=8, out_subblock_h=1, out_subblock_w=4, per_core_M=4, per_core_N=4)` — `in0_block_w = 256/32 = 8` (full K in one block).
- **Slice:** Tile-path slice with named compile-time args (`cb_in`, `cb_out`) for fusion CB remapping.
- **Compute:** `fp32=True`, `math_approx=False`, `HiFi4` (all 13 ops)

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

**Timing (Tracy):**

| | FW | Kernel | Cores |
|--|---:|------:|------:|
| **Fused (1 dispatch)** | **117.3 us** | **115.6 us** | 16 |
| **Unfused total (13 dispatches)** | **189.3 us** | **179.3 us** | |
| **FW speedup** | | | **1.61x** |

**Summary:**

| Metric | Fused | Unfused | Speedup |
|--------|------:|--------:|--------:|
| Cold start (build + first run) | 1,866 ms | 4,879 ms | **2.61x** |
| E2E steady state (host→device→host) | 0.197 ms | 0.434 ms | **2.21x** |
| Device FW duration (Tracy) | 117.3 us | 189.3 us | **1.61x** |
| Device kernel duration (Tracy) | 115.6 us | 179.3 us | **1.55x** |

The **~1.6x device speedup** comes from branch parallelism — the fused kernel runs independent tree branches simultaneously on disjoint core subsets, while the unfused path dispatches each op sequentially. At each level of the tree:

- **Level 0:** `ln_stem` runs on all 16 cores (no parallelism possible — same in both).
- **Level 1:** `sl_top` and `sl_bot` overlap on 8 cores each.
- **Level 2:** `mm_left` and `mm_right` overlap on 8 cores each.
- **Level 3:** 4 leaf slices overlap on 4 cores each.
- **Level 4:** 4 leaf LNs overlap on 4 cores each.

The **E2E speedup (2.21x)** is larger than the device speedup because the unfused path uses `ttnn.*` API calls (not `generic_op`), which have higher per-op dispatch overhead, accumulated across 13 sequential dispatches.

The **cold start speedup (2.61x)** reflects that fusion JIT-compiles 1 kernel program instead of 13 separate ones.

**PCC:** 0.993 (leaf LN output vs unfused reference). Lower than demos 1-3 due to 5 levels of accumulated numerical error across block-sharded ops.

---

## Summary

### Device-level timing (Tracy, H=128 / small variants)

For small, fast ops on Wormhole, **host gaps between dispatches dominate per-op firmware time**. Fusion eliminates all inter-op host gaps, and for parallel topologies also overlaps independent chains.

| Demo | Fused FW | Unfused FW sum | FW speedup | Source of speedup |
|------|----------:|------------:|--------:|:-----------------|
| 1: RMS→MM→RMS (3 ops, 8 cores) | 26.9 us | 24.3 us | 0.90x | No device speedup (linear chain) |
| 2: RMS→LN sharded (2 ops, 16 cores) | 22.2 us | 22.6 us | 1.02x | No device speedup (linear chain) |
| 3: Parallel chains (4 ops, 16 cores) | 65.0 us | 109.3 us | 1.68x | Parallelism (2 chains overlap) |
| 5: Sharded tree (13 ops, 16 cores) | 117.3 us | 189.3 us | 1.61x | Branch parallelism |

### Device-level timing (Tracy, H=1536 / large variants)

At larger H, compute dominates and host gaps become irrelevant. **Linear chains show no device speedup at large H** — intermediates still round-trip through DRAM in the fused path, so there is no I/O elimination. Fusion only adds barrier overhead.

| Demo | Fused FW | Unfused FW sum | FW speedup | Why |
|------|----------:|------------:|--------:|:---|
| 1: RMS→MM→RMS | 957.7 us | — | ~1.00x | Compute-dominated; DRAM intermediates |
| 2: RMS→LN sharded | 107.7 us | 108.2 us | 1.00x | Sharded, ~2 us barrier overhead |

### Host-level timing (cold start + steady-state E2E)

| Demo | H | Fused cold | Unfused cold | Cold | Fused E2E | Unfused E2E | E2E |
|------|--:|----------:|-----------:|--------:|----------:|-----------:|--------:|
| 1: RMS→MM→RMS | 128 | 1,681 ms | 2,075 ms | 1.23x | 0.042 ms | 0.066 ms | **1.57x** |
| 1: RMS→MM→RMS | 1536 | 1,688 ms | 2,119 ms | 1.26x | 0.993 ms | 0.994 ms | 1.00x |
| 2: RMS→LN sharded | 128 | 1,609 ms | 2,572 ms | 1.60x | 0.085 ms | 0.059 ms | 0.69x |
| 2: RMS→LN sharded | 1536 | 1,599 ms | 2,565 ms | 1.60x | 0.120 ms | 0.105 ms | 0.88x |
| 3: Parallel chains | — | 1,521 ms | 3,408 ms | 2.24x | 0.077 ms | 0.146 ms | **1.90x** |
| 4: GlobalCB | — | 907 ms | — | — | 0.19 ms | — | — |
| 5: Sharded tree | — | 1,866 ms | 4,879 ms | 2.61x | 0.197 ms | 0.434 ms | **2.21x** |

**Key takeaways:**
- **Small H, linear chains** (demos 1, 2 at H=128): No device speedup — fused FW ≈ sum of unfused kernels + small barrier overhead. Speedup comes entirely from host gap elimination, visible at the Tracy device timeline level. At the Python E2E level, demo 1 (3 ops) shows 1.57x from eliminating 2 dispatch roundtrips; demo 2 (2 ops) is slower fused because `generic_op` dispatch overhead exceeds the single saved dispatch gap.
- **Large H, linear chains** (demos 1, 2 at H=1536): Compute dominates, host gaps are negligible. Neither demo sees device speedup — intermediates still round-trip through DRAM (Demo 1) or stay in sharded L1 (Demo 2) in both fused and unfused paths. Fusion only adds ~2 us of barrier overhead per inter-phase transition.
- **Branching topologies** (demos 3, 5): Device speedup from overlapping independent branches on disjoint core subsets. Demo 3 gets 1.68x from running two chains simultaneously; Demo 5 gets 1.61x from a full binary tree with 4-way leaf parallelism.
- **Cold start**: Always faster fused (1 JIT compilation vs N), with the gap growing with op count (1.23x for 3 ops → 2.61x for 13 ops).

---

## Implementation Notes

1. **Compute configs must match across all phases.** `fp32_dest_acc_en`, `math_approx_mode`, and `math_fidelity` must be identical. Default configs differ by op type:
   - RMS norm: `fp32=False`, `math_approx=True`, `HiFi4`
   - Layer norm: `fp32=True`, `math_approx=True`, `HiFi4`
   - Matmul: `fp32=False`, `math_approx=False`, `LoFi`

   Pass explicit `WormholeComputeKernelConfig` to ensure consistency.

2. **Branching output order** depends on Python set hashing of core coordinates. Reference outputs through the original op descriptors (`op.output_tensors[0]`) rather than relying on positional indexing in `fused.output_tensors`.
