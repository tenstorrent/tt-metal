# Fusion Infrastructure Demo Suite

Eight demos showcasing different fusion capabilities on Tenstorrent Wormhole hardware.

**Test file:** `tests/ttnn/unit_tests/operations/fused/parallel_sequential/demo/test_fused_demo.py`

Normal pytest runs exercise correctness only (`perf_mode="none"`). The tables
below report E2E and device-profiler measurements taken on Wormhole n150
(single chip).

```bash
python -m pytest tests/ttnn/unit_tests/operations/fused/parallel_sequential/demo/test_fused_demo.py -xvs
```

## Perf Modes

### `device_fw` — What does the hardware actually cost?

Designed for Tracy device profiling (`TT_METAL_DEVICE_PROFILER=1`). Each entry
point emits one synchronized warmup forward followed by five synchronized
measured forwards. Tables report the median forward: one fused row, or the sum
of the unfused rows in that forward. Tracy reports `DEVICE FW DURATION [ns]`
(firmware setup + kernel + teardown) and `DEVICE KERNEL DURATION [ns]` (pure
Tensix execution). Device kernel duration is the more reliable comparison when
FW attribution spans an idle gap between unfused dispatches.

### `e2e` — What does the user see in steady state?

Measured with 5 warmup iterations (discarded), then the median of seven 100-iteration trials, all caches warm. This captures steady-state pipelined throughput, including host descriptor/container work, dispatch overhead, `fusion_dispatch_op` argument patching, and device execution. It is not single-request latency.

Results are reported for both fusion usage modes:

- **Inline:** recreate the descriptors and container each iteration, then use the warm fusion build cache.
- **Persistent:** create descriptors and the container once, call `update()` for the activation, and reuse the container's hot dispatch path.

Both modes use the production command-lifetime semaphore bank. All logical
barrier words occupy 16-byte-aligned slots in one lockstep-sharded L1 tensor,
initialized by one queued write, and released after dispatch submission. No
semaphore tensor is retained in the fusion cache or between forward passes.

Each mode is benchmarked in an isolated pytest invocation so persistent
intermediate allocations from one mode cannot affect another mode's available
L1.

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

Unfused median-position breakdown: RMS #1 FW=7.917 us + matmul FW=8.868 us + RMS #2 FW=7.822 us; median total FW=24.607 us.

| Metric | Fused Inline | Fused Persistent | Unfused | Inline speedup | Persistent speedup |
|--------|-------------:|-----------------:|--------:|---------------:|-------------------:|
| Device FW interval sum | 27.994 us | 27.978 us | 24.607 us (3 ops) | 0.879x | 0.879x |
| Device kernel | 27.335 us | 27.287 us | 22.608 us | 0.827x | 0.829x |
| E2E | 0.421 ms | 0.118 ms | 0.135 ms | 0.321x | **1.144x** |

The three short kernels are faster unfused on device because the fused chain adds
two phase transitions. Persistent mode still wins E2E by 1.144x by replacing
three host submissions with one; Inline descriptor/container construction
dominates and is not competitive.

### H=1536 (compute-dominated)

Unfused median-position breakdown: RMS #1 FW/kernel=27.781/27.141 us,
matmul=964.576/963.874 us, RMS #2=954.730/27.840 us. The final RMS FW
interval consistently spans an idle attribution gap, so the raw 1947.087 us
total FW is not an execution-time speedup; the 1018.855 us kernel total is the
meaningful device comparison.

| Metric | Fused Inline | Fused Persistent | Unfused | Inline speedup | Persistent speedup |
|--------|-------------:|-----------------:|--------:|---------------:|-------------------:|
| Device FW interval sum | 998.410 us | 988.899 us | 1947.087 us (3 ops, attribution artifact) | 1.950x* | 1.969x* |
| Device kernel | 997.742 us | 988.221 us | 1018.855 us | **1.021x** | **1.031x** |
| E2E | 0.998 ms | 0.998 ms | 1.009 ms | **1.011x** | **1.011x** |

At H=1536, the matmul dominates and all three paths are effectively device
bound. Fusion provides only a small kernel/E2E advantage. `*` The FW ratios are
shown for completeness but must not be interpreted as physical speedups because
of the unfused attribution gap.

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

Unfused median-position breakdown: RMS FW=9.188 us + LN FW=13.506 us = 22.694 us.

| Metric | Fused Inline | Fused Persistent | Unfused | Inline speedup | Persistent speedup |
|--------|-------------:|-----------------:|--------:|---------------:|-------------------:|
| Device FW interval sum | 23.910 us | 23.882 us | 22.685 us (2 ops) | 0.949x | 0.950x |
| Device kernel | 23.093 us | 23.094 us | 20.912 us | 0.906x | 0.906x |
| E2E | 0.449 ms | 0.164 ms | 0.129 ms | 0.287x | 0.787x |

This short sequential workload does not amortize the fused phase transition and
command-lifetime bank setup. Persistent is much faster than Inline, but the
mature two-op unfused path remains 1.27x faster E2E.

### H=1536 (compute-dominated)

Unfused median-position breakdown: RMS FW=38.513 us + LN FW=58.094 us; median total FW=96.623 us.

| Metric | Fused Inline | Fused Persistent | Unfused | Inline speedup | Persistent speedup |
|--------|-------------:|-----------------:|--------:|---------------:|-------------------:|
| Device FW interval sum | 101.807 us | 101.774 us | 96.623 us (2 ops) | 0.949x | 0.949x |
| Device kernel | 100.990 us | 100.968 us | 94.855 us | 0.939x | 0.939x |
| E2E | 0.443 ms | 0.168 ms | 0.122 ms | 0.275x | 0.726x |

The fused kernel is ~6 us slower than the unfused kernel sum due to the phase
transition. That device cost plus per-forward bank setup leaves Persistent
slower than unfused; Inline is again dominated by Python construction.

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

Unfused median-position breakdown: LN FW=42.114 us + matmul FW=17.935 us +
RMS FW=26.980 us + matmul FW=17.893 us; median total FW=104.952 us. Unfused
runs four sequential dispatches on a single `(0,0)`-based grid.

| Metric | Fused Inline | Fused Persistent | Unfused | Inline speedup | Persistent speedup |
|--------|-------------:|-----------------:|--------:|---------------:|-------------------:|
| Device FW interval sum | 68.201 us | 68.029 us | 104.952 us (4 ops) | **1.539x** | **1.543x** |
| Device kernel | 67.289 us | 67.140 us | 102.387 us | **1.522x** | **1.525x** |
| E2E | 0.694 ms | 0.180 ms | 0.280 ms | 0.403x | **1.552x** |

Inline and Persistent median FW durations differ by 0.172 us (0.25%) and
kernel durations by 0.149 us (0.22%), confirming equivalent device execution
within run noise.

The **1.54x device speedup** demonstrates the intended parallelism: both chains
overlap on disjoint core columns. Persistent E2E throughput realizes that benefit
and beats the mature unfused program-cache path by **1.55x**. Inline remains
host-bound because descriptor and container construction occurs every iteration.

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

Unfused median-position breakdown (13 dispatches): stem LN FW=39.817 us +
two slices FW=11.876 us + two matmuls FW=35.688 us + four leaf slices
FW=11.640 us + four leaf LNs FW=97.562 us; median total FW=196.600 us.

| Metric | Fused Inline | Fused Persistent | Unfused | Inline speedup | Persistent speedup |
|--------|-------------:|-----------------:|--------:|---------------:|-------------------:|
| Device FW interval sum | 119.683 us | 118.656 us | 196.600 us (13 ops) | **1.643x** | **1.657x** |
| Device kernel | 117.990 us | 116.994 us | 186.433 us | **1.580x** | **1.594x** |
| E2E | 1.672 ms | 0.264 ms | 0.978 ms | 0.585x | **3.701x** |

Inline and Persistent median FW durations differ by 1.027 us (0.86%) and
kernel durations by 0.996 us (0.84%), consistent with the same generated
program under run-to-run memory/system noise.

The **1.64-1.66x device speedup** comes from branch parallelism across disjoint
core subsets. Persistent mode also collapses 13 host submissions into one and
reaches a **3.70x E2E throughput speedup**. Inline is still slower because
rebuilding 13 descriptors and the nested container dominates.

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

This fused case is skipped: its generated kernel configuration is 75,520 bytes,
exceeding the 70,656-byte kernel-config buffer.

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

## Demo 8: Barrier Dispatch Overhead Microbenchmark

Measures the complete steady-state cost of adding fused phase transitions by
chaining N no-op phases (empty `kernel_main()`). This includes device
synchronization plus allocation and queued initialization of the
command-lifetime semaphore bank.

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

**Methodology:** Each table cell is the median of seven paired 100-iteration
trials after warmup. The reported amortized transition cost is
`(fused_N - fused_1) / (N - 1)`, where `fused_1` captures fixed launch
overhead. The one-phase baseline has no barrier bank, so the difference also
contains the fixed cost of allocating and initializing that bank. Consequently,
the value decreases as more transitions amortize setup; it is not a pure
device-barrier latency.

**Amortized E2E transition cost (Wormhole n300):**

| Cores | Grid | 2 phases | 3 phases | 4 phases | 5 phases | 6 phases |
|------:|:-----|---------:|---------:|---------:|---------:|---------:|
| 1 | (0,0) | 41.3 us | 20.8 us | 14.0 us | 10.5 us | 8.5 us |
| 8 | 1x8 | 58.5 us | 29.4 us | 19.7 us | 14.9 us | 12.2 us |
| 16 | 2x8 | 75.3 us | 38.0 us | 26.4 us | 19.5 us | 15.6 us |
| 64 | 8x8 | 191.4 us | 96.6 us | 65.1 us | 49.0 us | 39.5 us |

The fixed bank allocation/write dominates this E2E microbenchmark, especially
at low phase count, and grows with the sharded core footprint. The in-kernel
barrier itself still has two levels:

- **`local::sync()`** -- per-core 3-RISC rendezvous via L1 semaphores (compute_done, writer_done, reset_done). Coordinator (NCRISC) waits for followers, resets CB state, signals reset_done.
- **`group::sync()`** -- cross-core synchronization. Each core sends `noc_semaphore_inc` to core 0; core 0 waits for all arrivals, then unicast-releases each core individually.

The cross-core release remains unicast so non-contiguous grids work without a
rectangular-grid special case. This E2E table should not be used to compare
unicast with multicast because bank setup, not only the release loop, is in the
measurement.
