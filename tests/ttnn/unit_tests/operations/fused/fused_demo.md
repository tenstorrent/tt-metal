# Fusion Infrastructure Demo Suite

Five demos showcasing different fusion capabilities on Tenstorrent Wormhole hardware.

**Test file:** `tests/ttnn/unit_tests/operations/fused/test_fused_demo.py`

```bash
# Run tests:
python -m pytest tests/ttnn/unit_tests/operations/fused/test_fused_demo.py -xvs

# Run with Tracy device profiler:
TT_METAL_DEVICE_PROFILER=1 python -m tracy -r -m pytest tests/ttnn/unit_tests/operations/fused/test_fused_demo.py -xvs
```

All timing measured on Wormhole n150, BF16, `l1_small_size=24576`.

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

To avoid this, each test runs its unfused op chain **twice**. The first pass (warmup) populates the per-process JIT cache. The second pass (measured) reflects steady-state dispatch. In the Tracy CSV, the warmup ops appear as earlier rows with large `OP TO OP LATENCY` values (millions of ns); the measured ops appear as later rows with small values (30–60 us). Only the second-pass rows are reported below.

Fused ops appear in the CSV as `GenericOpDeviceOperation`.

### Apples-to-apples configs

Unfused `ttnn.matmul` calls use the same `program_config` and `compute_kernel_config` as their fused counterparts so both the core count and compute config match. Without explicit configs, `ttnn.matmul` auto-selects a multi-cast program that uses up to 64 cores.

Unfused `ttnn.rms_norm` and `ttnn.layer_norm` calls use the default interleaved grid because the ttnn Python API does not expose a `core_range` parameter for interleaved inputs. This means unfused norm ops may use more cores than their fused counterparts. Where this occurs, it makes the unfused kernel times a **lower bound** (more cores = faster kernel), so the comparison is **conservative** — the actual speedup from fusion is at least as large as reported.

---

## Demo 1: RMS -> Matmul -> RMS

Basic sequential chaining of heterogeneous ops (norm + matmul + norm) into a single fused kernel dispatch. Each phase still writes its output to DRAM (matching the original op's output buffer); the speedup comes from eliminating host-side dispatch gaps between ops.

**Setup:**
| | Shape | Tiles | Memory |
|-|-------|-------|--------|
| Input | `(1, 1, 256, 128)` | 8x4 | BF16, DRAM interleaved |
| Weight (norm) | `(1, 1, 1, 128)` | 1x4 | BF16, DRAM interleaved |
| Weight B (matmul) | `(1, 1, 128, 128)` | 4x4 | BF16, DRAM interleaved |
| Output | `(1, 1, 256, 128)` | 8x4 | BF16, DRAM interleaved |

- **Grid:** 4x2 = 8 cores, `per_core_M=1`, `per_core_N=4`, `in0_block_w=4`
- **Compute:** `fp32=False`, `math_approx=True`, `HiFi4` (all 3 phases)

**Dataflow:**
```
Input (DRAM) -> [RMS norm -> Matmul -> RMS norm] -> Output (DRAM)
                  L1 intermediate   L1 intermediate
```

**Timing:**

| | FW | Kernel | Host gap | Cores |
|--|---:|------:|--------:|------:|
| **Fused (1 dispatch)** | **26.4 us** | 25.8 us | — | 8 |
| Unfused RMS #1 | 7.9 us | 7.2 us | | 8 |
| | | | 59.1 us | |
| Unfused matmul | 8.7 us | 8.0 us | | 8 |
| | | | 37.8 us | |
| Unfused RMS #2 | 7.9 us | 7.2 us | | 8 |
| **Unfused total (3 dispatches)** | **24.5 us** | 22.4 us | **96.9 us** | |
| **Unfused end-to-end** | | | | **121.4 us** |

Compile fused: 23 ms. **Fused is 4.6x faster end-to-end** (26.4 us vs 121.4 us). Host gaps are 80% of unfused E2E.

**PCC:** fused=0.999973, unfused=0.999976

## Demo 2: RMS -> LN (Block-Sharded)

Fusion with block-sharded memory layout. The CB allocator detects pinned buffer addresses from the shard spec and preserves them while pool-allocating other CB slots. Rebind addresses are passed as runtime args so the JIT cache is not busted when tensor allocations change.

**Setup:**
| | Shape | Tiles | Memory |
|-|-------|-------|--------|
| Input | `(1, 1, 128, 512)` | 4x16 | BF16, block-sharded L1 |
| Weight | `(1, 1, 1, 512)` | 1x16 | BF16, DRAM interleaved |
| Output | `(1, 1, 128, 512)` | 4x16 | BF16, block-sharded L1 |

- **Grid:** 4x4 = 16 cores
- **Shard spec:** `(32, 128)` = 1x4 tiles per core, `ROW_MAJOR` orientation
- **Program config:** `LayerNormShardedMultiCoreProgramConfig(subblock_w=4, block_h=1, block_w=4)`
- **Compute:** `fp32=False`, `math_approx=True`, `HiFi4` (both phases)

**Timing:**

| | FW | Kernel | Host gap | Cores |
|--|---:|------:|--------:|------:|
| **Fused (1 dispatch)** | **17.5 us** | 16.7 us | — | 16 |
| Unfused RMS | 7.3 us | 6.3 us | | 16 |
| | | | 62.5 us | |
| Unfused LN | 10.4 us | 9.4 us | | 16 |
| **Unfused total (2 dispatches)** | **17.7 us** | 15.8 us | **62.5 us** | |
| **Unfused end-to-end** | | | | **80.2 us** |

Compile fused: 45 ms. **Fused is 4.6x faster end-to-end** (17.5 us vs 80.2 us). Host gaps are 78% of unfused E2E.

**PCC:** fused=0.999993, unfused=0.999993

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

- **Grid:** 8x8 = 64 cores total, subdivided per branch (split by rows so matmul grid_y divides N_tiles=4)
- **Matmul configs:** `right_mm` 8x4=32 cores (`per_core_M=2`), `ll_mm` 4x4=16 cores (`per_core_M=4`)
- **`per_core_M` rule:** Each branch processes the full tensor. `per_core_M = total_M_tiles / num_cores`
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

**Setup:**
| | Shape | Tiles | Memory |
|-|-------|-------|--------|
| Input A | `(1, 1, 512, 128)` | 16x4 | BF16, DRAM interleaved |
| Input B | `(1, 1, 512, 128)` | 16x4 | BF16, DRAM interleaved |
| Weight B | `(1, 1, 128, 128)` | 4x4 | BF16, DRAM interleaved |

- **Chain A:** LN -> Matmul on `(0,0)-(3,3)` = 4x4 = 16 cores, `per_core_M=1`, `per_core_N=4`, `in0_block_w=4`
- **Chain B:** RMS -> Matmul on `(4,0)-(7,3)` = 4x4 = 16 cores, same matmul config
- **Compute:** `fp32=False`, `math_approx=True`, `HiFi4` (all 4 ops)

**API:**
```python
Parallel(
    Sequential(ln_a, mm_a),
    Sequential(rms_b, mm_b),
).build(device)
```

**Fused vs unfused comparison:**

The fused kernel runs both chains **simultaneously** on 32 cores (16 per chain). The fused kernel time is approximately `max(chain_a, chain_b)` since the chains overlap.

The unfused comparison runs the same 4 ops **sequentially** as 4 separate `ttnn` dispatches (LN -> matmul -> RMS -> matmul), each on 16 cores. There is no way to express cross-core parallelism with standard `ttnn` op calls.

Fused benefits from both dispatch elimination AND parallelism.

**Timing:**

| | FW | Kernel | Host gap | Cores |
|--|---:|------:|--------:|------:|
| **Fused (1 dispatch)** | **31.0 us** | 30.0 us | — | 32 (16+16 parallel) |
| Unfused LN (chain A) | 10.7 us | 10.0 us | | 16 |
| | | | 34.7 us | |
| Unfused matmul (chain A) | 11.9 us | 11.3 us | | 16 |
| | | | 25.6 us | |
| Unfused RMS (chain B) | 8.5 us | 7.8 us | | 16 |
| | | | 28.0 us | |
| Unfused matmul (chain B) | 11.8 us | 11.2 us | | 16 |
| **Unfused total (4 dispatches)** | **42.9 us** | 40.3 us | **88.3 us** | |
| **Unfused end-to-end** | | | | **131.2 us** |

Compile fused: 35 ms. **Fused is 4.2x faster end-to-end** (31.0 us vs 131.2 us). Two sources of speedup: host-gap elimination (88.3 us of inter-op overhead removed) and parallelism (unfused runs 42.9 us of firmware sequentially; fused overlaps both chains into 31.0 us).

**PCC:** Chain A (LN->MM)=0.999973, Chain B (RMS->MM)=0.999975

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

- **Compute:** Identity tile copy (unary `copy_tile`)
- **Kernels:** Custom `SOURCE_CODE` kernels (not factory descriptors)

**Timing:**

| | FW | Kernel | Host gap | Cores |
|--|---:|------:|--------:|------:|
| **Fused (1 dispatch)** | **11.8 us** | 11.1 us | — | 2 |

No unfused comparison (GlobalCB mid-kernel write has no unfused equivalent).

**PCC:** Receiver=1.000000, Phase 1=1.000000

---

## Summary

For small, fast ops on Wormhole, **host gaps between dispatches (30–62 us each) dominate per-op firmware time (5–11 us each)**. Fusion eliminates all inter-op host gaps, and for parallel topologies also overlaps independent chains.

| Demo | Fused FW | Unfused E2E | Speedup | Host gap % of unfused |
|------|----------:|------------:|--------:|----------------------:|
| 1: RMS->MM->RMS (3 ops, 8 cores) | 26.4 us | 121.4 us | 4.6x | 80% |
| 2: RMS->LN sharded (2 ops, 16 cores) | 17.5 us | 80.2 us | 4.6x | 78% |
| 3: Branching (5 ops, 64 cores) | 64.5 us | 205.3 us | 3.2x | 59% (+ parallelism) |
| 4: Parallel chains (4 ops, 32 cores) | 31.0 us | 131.2 us | 4.2x | 67% (+ parallelism) |

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
