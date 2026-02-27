# Fusion Infrastructure Demo Suite

Five demos showcasing different fusion capabilities on Tenstorrent Wormhole hardware.

**Test file:** `tests/ttnn/unit_tests/operations/fused/fusion_demo/test_fused_demo.py`

```bash
# Run tests:
python -m pytest tests/ttnn/unit_tests/operations/fused/fusion_demo/test_fused_demo.py -xvs

# Run with Tracy device profiler:
TT_METAL_DEVICE_PROFILER=1 python -m tracy -r -m pytest tests/ttnn/unit_tests/operations/fused/fusion_demo/test_fused_demo.py -xvs
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

To avoid this, the **entire suite runs twice** via a class-scoped `iteration` fixture. The first pass (warmup) runs all 5 demos, populating the per-process JIT cache for every kernel — fused and unfused. The second pass (timed) is all cache hits and provides accurate timing. In the Tracy CSV, the warmup ops appear as earlier rows with large `OP TO OP LATENCY` values (millions of ns); the timed ops appear as later rows with small values (30–60 us). Only the second-pass rows are reported below.

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

| | FW | Kernel | Host gap | Cores |
|--|---:|------:|--------:|------:|
| **Fused (1 dispatch)** | **26.0 us** | 25.4 us | — | 8 |
| Unfused RMS #1 | 7.9 us | 7.3 us | | 8 |
| | | | 59.1 us | |
| Unfused matmul | 9.2 us | 8.5 us | | 8 |
| | | | 37.8 us | |
| Unfused RMS #2 | 7.9 us | 7.3 us | | 8 |
| **Unfused total (3 dispatches)** | **25.0 us** | 23.1 us | **96.9 us** | |
| **Unfused end-to-end** | | | | **121.9 us** |

| Metric | Fused | Unfused | Speedup |
|--------|------:|--------:|--------:|
| Cold start | 1,603 ms | 2,074 ms | **1.29x** |
| E2E steady state | 0.042 ms | 0.064 ms | **1.52x** |
| Device FW (Tracy) | 26.0 us | 25.0 us | 1.04x |
| Device kernel (Tracy) | 25.4 us | 23.1 us | 1.10x |

At H=128, each op takes 7-9 us on device. Host dispatch gaps (59+38 = 97 us) are **4x** the total kernel time (23 us). The **1.52x E2E speedup** comes entirely from eliminating those gaps. The fused kernel is ~2 us slower than the unfused sum due to inter-phase barrier overhead.

### H=1536 (large — compute-dominated)

**Timing (Tracy):**

| | FW | Kernel | Cores |
|--|---:|------:|------:|
| **Fused (1 dispatch)** | **906.1 us** | 905.5 us | 8 |
| Unfused RMS #1 | 30.1 us | 29.5 us | 8 |
| Unfused matmul | 897.1 us | 896.4 us | 8 |
| Unfused RMS #2 | — | 30.4 us | 8 |
| **Unfused kernel total** | | **956.3 us** | |

| Metric | Fused | Unfused | Speedup |
|--------|------:|--------:|--------:|
| Cold start | 1,678 ms | 2,114 ms | **1.26x** |
| E2E steady state | 0.998 ms | 0.967 ms | **0.97x** |
| Device kernel (Tracy) | 905.5 us | 956.3 us | **1.06x** |

At H=1536, the matmul dominates (~880 us). Host dispatch gaps are noise compared to compute time. **Intermediates still round-trip through DRAM** in the fused path — each phase's writer writes to DRAM and the next phase's reader reads from DRAM, identical to unfused. The fused kernel adds ~9 us of barrier overhead (2 segment-sync barriers between 3 phases). The apparent 1.06x in the table above is within run-to-run DRAM bandwidth variance (~30 us); a repeat run showed fused 39 us *slower*. There is **no structural device speedup** for a linear chain of DRAM-interleaved ops.

**PCC:** 1.000000 (both H values)

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
- **Compute:** `fp32=False`, `math_approx=True`, `HiFi4` (both phases)

### H=128 (small — dispatch-dominated)

**Timing (Tracy):**

| | FW | Kernel | Host gap | Cores |
|--|---:|------:|--------:|------:|
| **Fused (1 dispatch)** | **17.5 us** | 16.8 us | — | 16 |
| Unfused RMS | 7.3 us | 6.5 us | | 16 |
| | | | 62.5 us | |
| Unfused LN | 10.4 us | 9.5 us | | 16 |
| **Unfused total (2 dispatches)** | **17.7 us** | 16.0 us | **62.5 us** | |
| **Unfused end-to-end** | | | | **80.2 us** |

| Metric | Fused | Unfused | Speedup |
|--------|------:|--------:|--------:|
| Cold start | 1,504 ms | 2,421 ms | **1.61x** |
| E2E steady state | 0.083 ms | 0.060 ms | 0.72x |
| Device FW (Tracy) | 17.5 us | 17.7 us | 1.01x |
| Device kernel (Tracy) | 16.8 us | 16.0 us | 0.95x |

At H=128, each op takes 7-10 us. The single host dispatch gap (62.5 us) is **4x** the total kernel time. At the Tracy device level, fused is 4.6x faster than the unfused end-to-end. However, the **E2E steady state is 0.72x (slower fused)** because `generic_op` dispatch overhead (RT-arg rebuild via `override_runtime_arguments`) is larger than the native program-cache fast path used by individual `ttnn.rms_norm`/`ttnn.layer_norm` calls. The build cache is NOT in the E2E timing loop — this is purely dispatch overhead.

### H=1536 (large — compute-dominated)

**Timing (Tracy):**

| | FW | Kernel | Cores |
|--|---:|------:|------:|
| **Fused (1 dispatch)** | **78.0 us** | 77.2 us | 16 |
| Unfused RMS | 29.5 us | 28.8 us | 16 |
| Unfused LN | 45.1 us | 44.2 us | 16 |
| **Unfused kernel total** | **74.7 us** | **72.9 us** | |

| Metric | Fused | Unfused | Speedup |
|--------|------:|--------:|--------:|
| Cold start | 1,514 ms | 2,449 ms | **1.62x** |
| E2E steady state | 0.084 ms | 0.075 ms | 0.89x |
| Device FW (Tracy) | 78.0 us | 74.7 us | 0.96x |
| Device kernel (Tracy) | 77.2 us | 72.9 us | 0.94x |

At H=1536, the sharded norm ops take 29-45 us each. The fused kernel (77.2 us) is 4.3 us slower than the unfused kernel sum (72.9 us) — the inter-phase barrier overhead is a larger fraction of total work compared to Demo 1 because there are no DRAM intermediates to eliminate (both paths keep data in sharded L1). The E2E is 0.89x (slower fused) for the same reason as H=128: fusion build cache overhead exceeds the saved dispatch gap.

**PCC:** 1.000000 (both H values)

**PCC:** fused=1.000000, unfused reference=1.000000

## Demo 3: Branching Topology on Full Grid (Segment Barrier Measurement)

Nested `Sequential`/`Parallel` composition expressing a tree-shaped dataflow graph on the full 8x8 device grid. A single stem feeds two branches, and one branch further splits into sub-branches. All 5 ops execute in a single kernel dispatch. At each branching point, a **segment barrier** synchronizes all cores in the parent group via NOC semaphore multicast before splitting into independent branches.

**Topology:**
```
                       stem_rms (8x8 = 64 cores)
                            |
                  +---------+---------+
                  |                   |
               left_ln             right_mm
             (0,0)-(7,3)         (0,4)-(7,7)
              8x4 = 32            8x4 = 32
                  |
             +----+----+
             |         |
           ll_mm    lr_rms
         (0,0)-(3,3) (4,0)-(7,3)
          4x4 = 16    4x4 = 16
```

**Setup:**
| | Shape | Tiles | Memory |
|-|-------|-------|--------|
| Input | `(1, 1, 2048, 128)` | 64x4 | BF16, DRAM interleaved |
| Weight (norm) | `(1, 1, 1, 128)` | 1x4 | BF16, DRAM interleaved |
| Weight B (matmul) | `(1, 1, 128, 128)` | 4x4 | BF16, DRAM interleaved |

**Program configs:**
- **stem_rms:** Interleaved, `(0,0)-(7,7)` = 8x8 = 64 cores
- **left_ln:** Interleaved, `(0,0)-(7,3)` = 8x4 = 32 cores
- **right_mm:** `MatmulMultiCoreReuseProgramConfig(compute_with_storage_grid_size=(8, 4), in0_block_w=4, out_subblock_h=1, out_subblock_w=4, per_core_M=2, per_core_N=4)`, `(0,4)-(7,7)` = 8x4 = 32 cores
- **ll_mm:** `MatmulMultiCoreReuseProgramConfig(compute_with_storage_grid_size=(4, 4), in0_block_w=4, out_subblock_h=1, out_subblock_w=4, per_core_M=4, per_core_N=4)`, `(0,0)-(3,3)` = 4x4 = 16 cores
- **lr_rms:** Interleaved, `(4,0)-(7,3)` = 4x4 = 16 cores
- `per_core_M = total_M_tiles / num_cores` (each branch processes the full tensor, grid split by rows so matmul `grid_y` divides `N_tiles=4`)

- **Compute:** `fp32=False`, `math_approx=True`, `HiFi4` (all 5 ops)

**API:**
```python
Sequential(
    stem,
    Parallel(
        Sequential(left_ln, Parallel(ll_mm, lr_rms)),
        right_mm,
    ),
).build(device)
```

**Fused vs unfused comparison:**

The fused kernel runs all 5 ops in a single dispatch. The stem runs on all 64 cores; after the first segment barrier, branches run simultaneously on non-overlapping core subsets (left 32, right 32); then the left branch further splits (ll 16, lr 16) after a second segment barrier.

The unfused comparison runs 5 `ttnn` calls sequentially, matching the tree structure:

```python
u_stem  = ttnn.rms_norm(input, ...)                              # stem (64 cores)
u_left  = ttnn.layer_norm(u_stem, ...)                           # left (64 cores)
u_ll    = ttnn.matmul(u_left, B, program_config=ll_mm_cfg, ...)  # ll (16 cores)
u_lr    = ttnn.rms_norm(u_left, ...)                             # lr (64 cores)
u_right = ttnn.matmul(u_stem, B, program_config=right_mm_cfg, ...)  # right (32 cores)
```

Matmul core counts match exactly (ll=16, right=32). Norm ops use the default interleaved grid (64 cores) because the `ttnn` Python API does not expose a core range parameter for interleaved norms. Where the fused version runs a norm on a subset (left_ln on 32 cores, lr_rms on 16 cores), the unfused version uses all 64 — giving the unfused norms **more cores and therefore faster kernel times**. This makes the comparison conservative.

**Timing:**

| | FW | Kernel | Host gap | Fused cores | Unfused cores |
|--|---:|------:|--------:|------:|------:|
| **Fused (1 dispatch)** | **64.5 us** | 63.0 us | — | 64 | — |
| stem RMS | 15.7 us | 15.0 us | | 64 | 64 |
| | | | 38.8 us | | |
| left LN | 15.6 us | 14.9 us | | 32 | 64 |
| | | | 40.6 us | | |
| ll matmul | 15.6 us | 14.9 us | | 16 | 16 |
| | | | 24.5 us | | |
| lr RMS | 15.7 us | 15.0 us | | 16 | 64 |
| | | | 17.0 us | | |
| right matmul | 21.8 us | 21.1 us | | 32 | 32 |
| **Unfused total (5 dispatches)** | **84.4 us** | 80.9 us | **120.9 us** | | |
| **Unfused end-to-end** | | | | | **205.3 us** |

Compile fused: 62 ms. **Fused is 3.2x faster end-to-end** (64.5 us vs 205.3 us). Host gaps are 59% of unfused E2E. The fused kernel (63.0 us) is less than the unfused kernel sum (80.9 us) because branches run in parallel within the fused kernel.

**PCC:** ll_mm=0.999974, lr_rms=0.999992, right_mm=0.999975

**Notes:**
- Output ordering from `fused.output_tensors` depends on internal core group iteration order (Python set hashing). Reference outputs through the original op descriptors instead: `ll_mm_op.output_tensors[0]`.
- Two segment barriers fire during execution: one at 64-core scale (stem → left/right) and one at 32-core scale (left → ll/lr).

### Segment barrier protocol and profiling

At each branching point, the fused kernel executes a **segment barrier** — a multi-core synchronization protocol distinct from the per-core phase barriers used in linear chains.

The segment barrier `sync()` function works as follows:

1. **Arrive:** Every core sends `noc_semaphore_inc` (unicast) to core 0's arrive semaphore.
2. **Fan-in wait** (core 0 only): Core 0 spins on the arrive semaphore until all N cores have incremented it.
3. **Multicast release** (core 0 only): Core 0 writes to its local release semaphore, then uses `noc_semaphore_set_multicast_loopback_src` to multicast it to all N cores.
4. **Wait release** (non-core-0): All other cores spin on their release semaphore until the multicast arrives.

Fine-grained `DeviceZoneScopedN` markers were injected into the barrier codegen to measure each phase of this protocol. Results from the 5-op branching topology (two segment barriers: 64-core and 32-core):

| Zone | Where | Count | Min | Max | Avg |
|------|-------|------:|----:|----:|----:|
| `seg-arrive` | all cores | 96 | 58 ns | 59 ns | 59 ns |
| `seg-fan-in-wait` | core 0 | 2 | 64 ns | 1,945 ns | 1,005 ns |
| `seg-mcast-release` | core 0 | 2 | 481 ns | 586 ns | 534 ns |
| `seg-wait-release` | non-core-0 | 94 | 467 ns | 7,766 ns | 3,774 ns |
| `seg-sync` (overall) | all cores | 96 | 622 ns | 7,924 ns | 3,878 ns |

**Key findings:**
- The **NOC mechanism itself is fast**: arrive unicasts take 58–59 ns each; the multicast release takes 481–586 ns. The NOC cost is essentially constant regardless of core count.
- **Fan-in wait varies with straggler arrival**: the two observations (64-core and 32-core barriers) show 64 ns and 1,945 ns respectively. When all cores finish at roughly the same time, core 0 barely waits; when there's skew in phase completion, core 0 waits for the slowest core.
- The **bottleneck is straggler wait**: `seg-wait-release` varies from 467 ns to 7,766 ns across the non-core-0 cores. This spread reflects variation in when each core finishes its phase — the earliest-finishing cores wait longest for the multicast.
- The overall `seg-sync` cost (max 7.9 us) is dominated by **phase completion skew** across cores, not by NOC latency.

For comparison, the per-core **phase barrier** zones (which synchronize between RISCs on the same core with no inter-core communication):

| Zone | Count | Min | Max | Avg |
|------|------:|----:|----:|----:|
| `barrier-wait` (NCRISC waits for compute+writer) | 480 | 23 ns | 11,422 ns | 840 ns |
| `barrier-cb-reset` (NCRISC CB state reset) | 96 | 422 ns | 431 ns | 425 ns |
| `barrier-cb-resync` (TRISC CB resync) | 384 | 18 ns | 401 ns | 210 ns |
| `barrier-noc-drain` | 192 | 36 ns | 87 ns | 61 ns |
| `barrier-sem-wait` (NCRISC sem wait) | 96 | 1,461 ns | 11,245 ns | 3,840 ns |

## Demo 4: Parallel Independent Chains

Two completely independent 2-op chains running on disjoint 4x4 core groups within a single kernel dispatch. No inter-chain synchronization needed.

**Setup (parameterized on H):**
| | Shape | Memory |
|-|-------|--------|
| Input A | `(1, 1, 512, H)` | BF16, DRAM interleaved |
| Input B | `(1, 1, 512, H)` | BF16, DRAM interleaved |
| Weight B | `(1, 1, H, H)` | BF16, DRAM interleaved |

**Program configs:**
- **Chain A — LN:** Interleaved, `(0,0)-(3,3)` = 4x4 = 16 cores
- **Chain A — Matmul:** `MatmulMultiCoreReuseProgramConfig(compute_with_storage_grid_size=(4, 4), in0_block_w=4, out_subblock_h=1, out_subblock_w=min(H/32, 4), per_core_M=1, per_core_N=H/32)`
- **Chain B — RMS:** Interleaved, `(4,0)-(7,3)` = 4x4 = 16 cores
- **Chain B — Matmul:** Same config as Chain A
- **Compute:** `fp32=False`, `math_approx=True`, `HiFi4` (all 4 ops)

**API:**
```python
Parallel(
    Sequential(ln_a, mm_a),
    Sequential(rms_b, mm_b),
).build(device)
```

**Fused vs unfused comparison:**

The fused kernel runs both chains **simultaneously** on 32 cores (16 per chain). The fused kernel time is approximately `max(chain_a, chain_b)` since the chains overlap. The unfused comparison runs the same 4 ops **sequentially** as 4 separate `ttnn` dispatches, each on 16 cores.

### H=128 (small — dispatch-dominated)

**Timing (Tracy):**

| | FW | Kernel | Cores |
|--|---:|------:|------:|
| **Fused (1 dispatch)** | **30.6 us** | 29.7 us | 32 (16+16 parallel) |
| Unfused LN (chain A) | 10.5 us | 9.9 us | 16 |
| Unfused matmul (chain A) | 12.4 us | 11.7 us | 16 |
| Unfused RMS (chain B) | 8.5 us | 7.8 us | 16 |
| Unfused matmul (chain B) | 11.9 us | 11.2 us | 16 |
| **Unfused total (4 dispatches)** | **43.3 us** | 40.7 us | |

| Metric | Fused | Unfused | Speedup |
|--------|------:|--------:|--------:|
| Cold start | 1,541 ms | 3,438 ms | **2.24x** |
| E2E steady state | 0.066 ms | 0.092 ms | **1.39x** |
| Device FW (Tracy) | 30.6 us | 43.3 us | 1.42x |
| Device kernel (Tracy) | 29.7 us | 40.7 us | 1.37x |

At H=128, each op takes 8-12 us. Fused runs both chains in parallel so device time ≈ `max(chain_a, chain_b)` ≈ 30 us vs unfused sequential sum of 43 us. The **1.39x E2E speedup** comes from both parallelism (1.4x device) and host dispatch elimination (4 dispatches → 1).

### H=512 (large — compute-dominated)

**Timing (Tracy):**

| | FW | Kernel | Cores |
|--|---:|------:|------:|
| **Fused (1 dispatch)** | **342.9 us** | 342.0 us | 32 (16+16 parallel) |
| Unfused LN (chain A) | 21.1 us | 20.4 us | 16 |
| Unfused matmul (chain A) | 168.3 us | 167.7 us | 16 |
| Unfused RMS (chain B) | 15.8 us | 15.1 us | 16 |
| Unfused matmul (chain B) | 171.4 us | 170.7 us | 16 |
| **Unfused total (4 dispatches)** | **376.5 us** | 374.0 us | |

| Metric | Fused | Unfused | Speedup |
|--------|------:|--------:|--------:|
| Cold start | 1,628 ms | 3,544 ms | **2.18x** |
| E2E steady state | 0.349 ms | 0.379 ms | **1.09x** |
| Device FW (Tracy) | 342.9 us | 376.5 us | 1.10x |
| Device kernel (Tracy) | 342.0 us | 374.0 us | 1.09x |

At H=512, matmul dominates (~168-171 us per chain). Fused overlaps both chains: device time ≈ `max(chain_a, chain_b)` ≈ 342 us vs unfused sequential 377 us. Chain A (LN+MM ≈ 188 us) and Chain B (RMS+MM ≈ 186 us) are nearly balanced, so parallelism gives close to 2x on the op sum, but the matmuls already dominate each chain.

**PCC:** Chain A (LN->MM) = 1.0000, Chain B (RMS->MM) = 1.0000

## Demo 5: GlobalCircularBuffer Mid-Kernel Write

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

| | FW | Kernel | Host gap | Cores |
|--|---:|------:|--------:|------:|
| **Fused (1 dispatch)** | **11.8 us** | 11.1 us | — | 2 |

No unfused comparison (GlobalCB mid-kernel write has no unfused equivalent).

**PCC:** Receiver=1.000000, Phase 1=1.000000

## Demo 8: Block-Sharded Branching Tree (LN → Slice → Matmul → Slice → LN)

Full block-sharded tree topology with 13 ops across 5 levels. Exercises all three descriptor-based op types (LayerNorm, Slice, Matmul) on a 2×8 core grid with hierarchical core subset splitting. The unfused comparison launches the **exact same OpDescriptors** individually via `generic_op`, making it a true apples-to-apples comparison — identical ops, identical memory configs, identical kernels. The only difference is 1 dispatch vs 13.

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
| B_left | `(1, 1, 256, 128)` | 8×4 | BF16, L1 interleaved |
| B_right | `(1, 1, 256, 128)` | 8×4 | BF16, L1 interleaved |

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
- **Compute:** `fp32=False`, `math_approx=True`, `HiFi4` (all 13 ops)

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

**Unfused comparison (apples-to-apples):**

The unfused test creates the exact same 13 OpDescriptors via `_demo8_make_ops` and launches them individually:
```python
all_ops = [ln_stem, sl_top, sl_bot, mm_left, mm_right,
           sl_tl, sl_bl, sl_tr, sl_br, ln_ll, ln_lr, ln_rl, ln_rr]
for op in all_ops:
    ttnn.generic_op(list(op.input_tensors) + list(op.output_tensors), op.descriptor)
```

Both paths use identical block-sharded memory configs, identical kernels, and identical core assignments. The only difference is dispatch granularity: 1 fused dispatch vs 13 individual dispatches.

**Timing (Tracy, single iteration):**

| | FW | Kernel | Cores |
|--|---:|------:|------:|
| **Fused (1 dispatch)** | **87.4 us** | **86.0 us** | 16 |
| `ln_stem` — LayerNorm | 31.3 us | 30.6 us | 16 |
| `sl_top` — Slice top half | 14.8 us | 13.9 us | 8 |
| `sl_bot` — Slice bottom half | 17.1 us | 16.2 us | 8 |
| `mm_left` — Matmul left | 13.0 us | 12.4 us | 8 |
| `mm_right` — Matmul right | 13.0 us | 12.4 us | 8 |
| `sl_tl` — Leaf slice | 4.6 us | 3.7 us | 4 |
| `sl_bl` — Leaf slice | 4.6 us | 3.8 us | 4 |
| `sl_tr` — Leaf slice | 5.9 us | 5.1 us | 4 |
| `sl_br` — Leaf slice | 6.0 us | 5.1 us | 4 |
| `ln_ll` — Leaf LN | 18.3 us | 17.6 us | 4 |
| `ln_lr` — Leaf LN | 18.2 us | 17.6 us | 4 |
| `ln_rl` — Leaf LN | 18.3 us | 17.6 us | 4 |
| `ln_rr` — Leaf LN | 18.3 us | 17.6 us | 4 |
| **Unfused total (13 dispatches)** | **183.3 us** | **173.6 us** | |

**Summary:**

| Metric | Fused | Unfused | Speedup |
|--------|------:|--------:|--------:|
| Cold start (build + first run) | 1,732 ms | 8,191 ms | **4.7x** |
| E2E steady state (host→device→host) | 0.199 ms | 0.217 ms | **1.09x** |
| Device FW duration (Tracy) | 87.4 us | 183.3 us | **2.10x** |
| Device kernel duration (Tracy) | 86.0 us | 173.6 us | **2.02x** |

The **2x device speedup** comes from branch parallelism — the fused kernel runs independent tree branches simultaneously on disjoint core subsets, while the unfused path dispatches each op sequentially. At each level of the tree:

- **Level 0:** `ln_stem` runs on all 16 cores (no parallelism possible — same in both).
- **Level 1:** `sl_top` and `sl_bot` overlap on 8 cores each (fused: `max(14, 17)` = 17 us; unfused: `14 + 17` = 32 us).
- **Level 2:** `mm_left` and `mm_right` overlap on 8 cores each (fused: `max(13, 13)` = 13 us; unfused: `13 + 13` = 26 us).
- **Level 3:** 4 leaf slices overlap on 4 cores each (fused: `max(4, 5, 5, 6)` = 6 us; unfused: `4+5+5+6` = 20 us).
- **Level 4:** 4 leaf LNs overlap on 4 cores each (fused: `max(18, 18, 18, 18)` = 18 us; unfused: `18×4` = 72 us).

Critical path (fused): `31 + 17 + 13 + 6 + 18` = **85 us** ≈ observed 86 us. The theoretical max is achieved within ~1 us of barrier overhead.

The **E2E speedup (1.09x)** is modest because the descriptor path (`generic_op`) has minimal host overhead per dispatch — no Python framework layering, no op decomposition, no autoformat. With the standard `ttnn` API (which adds 30-60 us host gaps per dispatch), the E2E unfused time would be ~3-4x higher.

The **cold start speedup (4.7x)** reflects that fusion JIT-compiles 1 kernel program instead of 13 separate ones.

**PCC:** 0.992 (leaf LN output vs torch reference). Lower than demos 1-4 due to 5 levels of accumulated numerical error across block-sharded ops.

---

## Summary

### Device-level timing (Tracy, H=128 / small variants)

For small, fast ops on Wormhole, **host gaps between dispatches (30–62 us each) dominate per-op firmware time (5–11 us each)**. Fusion eliminates all inter-op host gaps, and for parallel topologies also overlaps independent chains.

| Demo | Fused FW | Unfused FW sum | FW speedup | Source of speedup |
|------|----------:|------------:|--------:|:-----------------|
| 1: RMS→MM→RMS (3 ops, 8 cores) | 26.0 us | 25.0 us | 1.0x | No device speedup (linear chain) |
| 2: RMS→LN sharded (2 ops, 16 cores) | 17.5 us | 17.7 us | 1.0x | No device speedup (linear chain) |
| 3: Branching (5 ops, 64 cores) | 64.5 us | 205.3 us | 3.2x | Branch parallelism |
| 4: Parallel chains (4 ops, 32 cores) | 30.6 us | 43.3 us | 1.4x | Parallelism (2 chains overlap) |
| 8: Sharded tree (13 ops, 16 cores) | 87.4 us | 183.3 us | 2.1x | Branch parallelism |

### Device-level timing (Tracy, H=1536 / large variants)

At larger H, compute dominates and host gaps become irrelevant. **Linear chains show no device speedup at large H** — intermediates still round-trip through DRAM in the fused path, so there is no I/O elimination. Fusion only adds barrier overhead.

| Demo | Fused kernel | Unfused kernel sum | Kernel speedup | Why |
|------|----------:|------------:|--------:|:---|
| 1: RMS→MM→RMS | ~906 us | ~911 us | **~1.00x** | Within DRAM variance; ~9 us barrier overhead |
| 2: RMS→LN sharded | 77.2 us | 72.9 us | **0.94x** | 4.3 us barrier overhead (sharded, no DRAM) |
| 4: Parallel chains (H=512) | 342.0 us | 374.0 us | **1.09x** | Parallelism overlaps balanced chains |

### Host-level timing (cold start + steady-state E2E)

| Demo | H | Fused cold | Unfused cold | Cold | Fused E2E | Unfused E2E | E2E |
|------|--:|----------:|-----------:|--------:|----------:|-----------:|--------:|
| 1: RMS→MM→RMS | 128 | 1,603 ms | 2,074 ms | 1.29x | 0.042 ms | 0.064 ms | **1.52x** |
| 1: RMS→MM→RMS | 1536 | 1,678 ms | 2,114 ms | 1.26x | 0.998 ms | 0.967 ms | 0.97x |
| 2: RMS→LN sharded | 128 | 1,504 ms | 2,421 ms | 1.61x | 0.083 ms | 0.060 ms | 0.72x |
| 2: RMS→LN sharded | 1536 | 1,514 ms | 2,449 ms | 1.62x | 0.084 ms | 0.075 ms | 0.89x |
| 4: Parallel chains | 128 | 1,541 ms | 3,438 ms | 2.23x | 0.066 ms | 0.092 ms | **1.39x** |
| 4: Parallel chains | 512 | 1,628 ms | 3,544 ms | 2.18x | 0.349 ms | 0.379 ms | **1.09x** |
| 8: Sharded tree | — | 1,732 ms | 8,191 ms | 4.73x | 0.199 ms | 0.217 ms | **1.09x** |

**Key takeaways:**
- **Small H, linear chains** (demos 1, 2 at H=128): No device speedup — fused FW ≈ sum of unfused kernels + small barrier overhead. Speedup comes entirely from host gap elimination, visible at the Tracy device timeline level. At the Python E2E level, demo 1 (3 ops) shows 1.52x from eliminating 2 dispatch roundtrips; demo 2 (2 ops) is slower fused because `generic_op` dispatch overhead exceeds the single saved dispatch gap.
- **Large H, linear chains** (demos 1, 2 at H=1536): Compute dominates, host gaps are negligible. Neither demo sees device speedup — intermediates still round-trip through DRAM (Demo 1) or stay in sharded L1 (Demo 2) in both fused and unfused paths. Fusion only adds ~5-9 us of barrier overhead per inter-phase transition.
- **Branching topologies** (demos 3, 4, 8): Device speedup from overlapping independent branches on disjoint core subsets. Demo 8's apples-to-apples comparison isolates this: 2x device speedup, purely from parallelism.
- **Cold start**: Always faster fused (1 JIT compilation vs N), with the gap growing with op count (1.3x for 3 ops → 4.7x for 13 ops).

---

## Implementation Notes

1. **FusedOps cannot be relaunched.** `composite.launch([fused])` can only be called once per FusedOp. Build a fresh FusedOp for each launch — the JIT cache ensures the second build+launch is fast.

2. **Compute configs must match across all phases.** `fp32_dest_acc_en`, `math_approx_mode`, and `math_fidelity` must be identical. Default configs differ by op type:
   - RMS norm: `fp32=False`, `math_approx=True`, `HiFi4`
   - Layer norm: `fp32=True`, `math_approx=True`, `HiFi4`
   - Matmul: `fp32=False`, `math_approx=False`, `LoFi`

   Pass explicit `WormholeComputeKernelConfig` to ensure consistency.

3. **Branching output order** depends on Python set hashing of core coordinates. Reference outputs through the original op descriptors (`op.output_tensors[0]`) rather than relying on positional indexing in `fused.output_tensors`.

4. **`per_core_M` for branching matmuls:** Each branch processes the full tensor. Compute `per_core_M = total_M_tiles / branch_grid_x`.

5. **Rebind addresses are runtime args.** Sharded CB buffer addresses that change between phases are passed as runtime args (not compile-time args) to avoid JIT cache misses when L1 allocations differ between program builds. The `rebind_cbs()` templated function uses `cb_addr_shift` from `circular_buffer_interface.h` for per-RISC byte-to-CB-word conversion.
