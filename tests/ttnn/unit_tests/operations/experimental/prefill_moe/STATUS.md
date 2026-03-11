# Prefill MoE Compute: Status & Path Forward

## What this is

A fused MoE expert compute kernel for GPT-OSS-120b prefill on Tenstorrent Blackhole.
Single op does: dispatch tokens → gate_up matmul → SwiGLU → down matmul → fabric return results.

**Production config:** 128 experts, 32 devices (4 experts/device), K=4, 4096 total tokens → 128 tokens/expert. D=2880, D_FF=5760. BFP4 weights. 36 layers.

## Current state

**Branch (remote):** `sraizada/prefill-moe-compute-phase3` on `/data/sraizada_2/tt-metal/` at `ubuntu@UF-EV-B5-GWH02`

**What works:**
- Fused op runs on 1×4 submesh (4 devices) with fabric return
- 15 compute cores on 5×3 grid, 1 return core, 1 recv core
- Phase 3 writer-side untilize complete (commit `94fd5fc22f`): writers output ROW_MAJOR fragments, return core reads contiguous rows
- PCC ≥ 0.996 on all devices

**Measured performance (P=32, 4 experts, 1×4 mesh):**
- Wall-clock: **1.92 ms/op**
- Return core (NCRISC): 1,505 μs (down from 3,785 μs before Phase 3)

## Bottleneck analysis

### At P=32 (current test config)
| Component | Time | Notes |
|-----------|------|-------|
| Weight reads | ~460 μs | Per-tile barriers, ~38 GB/s effective |
| Compute | ~400 μs | 15 cores, sequential experts |
| Return (fabric send) | ~1,050 μs | 1 core serializes 96 remote sends |
| Recv | ~190 μs | Single core, sequential copy |
| Dispatch overhead | ~100 μs | Host-side |

### At P=128 (production config) — projected
| Component | Time | Notes |
|-----------|------|-------|
| Weight reads | ~460 μs | Same weights, same cores — unchanged |
| Compute | ~1,600 μs | 4× more token rows, compute-bound |
| Return (fabric send) | ~5,000 μs | 512 token sends on 1 core |
| Recv | ~750 μs | 496 remote rows, sequential barriers |
| Dispatch overhead | ~100 μs | |
| **Total** | **~8,000 μs** | Single fused op only |

The return core is the dominant bottleneck. It gets **worse** with more tokens (linear scaling).

## Hard floors (physics limits)

| Floor | Value | Basis |
|-------|-------|-------|
| DRAM bandwidth | ~125 μs/layer | 50 MB weights / 400 GB/s device BW |
| Fabric link | ~118 μs/layer | 2.94 MB remote / 25 GB/s (2 directions) |
| Compute (15 cores) | ~170 μs/layer | 25.5 GFLOP / ~150 TFLOPS aggregate |
| Compute (48 cores) | ~53 μs/layer | 25.5 GFLOP / ~480 TFLOPS aggregate |

At 48 cores, DRAM reads (~125 μs) dominate compute (~53 μs). Triple-buffering cannot hide reads when compute is faster. **More cores hit diminishing returns beyond ~24 cores where compute ≈ DRAM.**

## Three improvements that matter

### 1. Distributed send (5,000 μs → 150-200 μs) — HIGHEST IMPACT

Replace the single return core with all compute cores sending their own results after compute finishes.

- Each core already has output rows in L1 from the writer phase
- Route packets through 2-4 FabricMux cores (dedicated fabric connection holders)
- Hard floor is fabric link bandwidth (~118 μs wire time), not core count
- 15 cores sending is enough to saturate the link

**Files:** New send phase in `expert_writer_multi.cpp`. New FabricMux setup in factory. Remove `fabric_return.cpp`.

### 2. Triple-buffer weight reads (460 μs → hidden behind compute at 15 cores)

Replace per-tile `noc_async_read_page` + `noc_async_read_barrier` with triple-buffered trid pattern (proven in `moe_gpt/dm0.cpp`).

- 3 transaction IDs pipeline reads: issue slot 3 while slot 2 completes while slot 1 is consumed
- At 15 cores, compute (~1,600 μs at P=128) > DRAM reads (~460 μs) → fully hidden
- At 48+ cores, DRAM reads become visible again (~170 μs floor)

**Files:** `expert_reader_multi.cpp` (or whichever RISC-V handles weight reads — currently dm1 in `expert_writer_multi.cpp`).

### 3. Optimize recv (750 μs → 200-300 μs)

Current recv_accumulate is blocking (waits for ALL remote rows) then copies one-by-one with per-row barriers.

- Batch DRAM reads and writes (issue multiple, single barrier)
- Start processing as rows arrive instead of waiting for all
- Or move to compute-core based recv (more parallelism)

**Files:** `recv_accumulate.cpp`

## Realistic performance projection

### Fused op only (compute + return)
| Config | Current | After improvements |
|--------|---------|-------------------|
| P=32, 15 cores | 1.92 ms | ~700-900 μs |
| P=128, 15 cores | ~8 ms | ~2,000-2,500 μs |

### Full layer pipeline (dispatch + compute + recv + combine)
| Phase | P=128 estimate |
|-------|---------------|
| Token dispatch (all-to-all) | 200-400 μs |
| Expert compute + send | 1,600-2,000 μs |
| Recv + accumulate | 200-300 μs |
| Weighted combine | 50-200 μs |
| Dispatch overhead (multiple ops) | 100-250 μs |
| **Total per layer** | **~2,200-3,200 μs** |
| **36 layers** | **~79-115 ms** |

### What about scaling to 48 cores?

Helps compute but hits DRAM wall:

| Cores | Compute | DRAM reads | Visible total | Notes |
|-------|---------|-----------|---------------|-------|
| 15 | 1,600 μs | 460 μs (hidden) | ~1,600 μs | Compute-bound, reads hidden |
| 24 | 1,000 μs | 300 μs (hidden) | ~1,000 μs | Sweet spot |
| 48 | 500 μs | 170 μs (visible!) | ~500 μs | Starting to hit DRAM wall |

Diminishing returns beyond ~24 cores at P=128. Going from 15→24 saves ~600 μs. Going from 24→48 saves ~500 μs but requires 2D multicast matmul rewrite.

## Can we reach 10 ms TTFT?

**36 layers × 2,500 μs/layer = 90 ms MoE. With PP=4: ~22 ms/user. Plus attention.**

To reach 10 ms total TTFT including attention — no, not with this architecture alone.

To reach 10 ms MoE-only: needs ~278 μs/layer. That requires:
- 24+ cores for compute (~1,000 μs → need further optimization)
- Fully pipelined weight reads (hidden)
- Distributed send (150-200 μs)
- Optimized recv (200-300 μs)
- Overlap between layers / pipeline stages

Even with all optimizations, the per-layer floor is roughly **DRAM reads (~125 μs) + fabric send (~120 μs) + recv (~100 μs) + combine (~50 μs) + overhead ≈ 400-500 μs/layer**. At 36 layers: **14-18 ms**. Getting below that requires architectural changes (overlapping dispatch with previous layer's compute, multi-layer pipelining, etc.).

## Known issues

1. **Single return core is catastrophic at P>32.** Must be replaced with distributed send. This is the #1 priority.

2. **Weight reads use per-tile barriers.** Each of 7,200 tile reads per core (across 4 experts) has a full `noc_async_read_barrier`. Triple-buffering would eliminate ~7,000 of these barriers.

3. **Recv kernel is blocking and sequential.** Waits for ALL remote rows before processing any. Per-row DRAM read + write with individual barriers.

4. **Combine kernel is scalar BF16 on a single dm0 core.** Extremely slow for P>32. Needs to be moved to FPU/SFPU on compute cores.

5. **150 TFLOPS BFP4 number is an unverified rough estimate.** Actual per-core throughput needs measurement via device profiler. Real number likely 4-8 TFLOPS per core (standard 2 FLOP/MAC convention) → 60-120 TFLOPS at 15 cores.

6. **`moe_gpt` decode op cannot handle 128 tokens/expert.** Input sharding requires E×M×K in L1 per core. At 4×128×2880×2 bytes = 2.88 MB > 1.5 MB L1. Cannot reuse decode ops for prefill without streaming inputs from DRAM.

7. **Expert pipelining (overlap send[e] with compute[e+1]) is not feasible** with current kernel structure. dm1 (RISCV_1) handles both weight reads and writes — can't do both simultaneously.

## File inventory

### Active kernel sources (cpp_op/prefill_moe_compute/device/kernels/)
| File | Role |
|------|------|
| `compute_expert_multi.cpp` | Compute: gate_up matmul → SwiGLU → down matmul |
| `expert_reader_multi.cpp` | DM0: reads activations, broadcasts SEM_GO |
| `expert_writer_multi.cpp` | DM1: reads weights, writes output fragments (ROW_MAJOR) |
| `fabric_return.cpp` | Return core: reads ROW_MAJOR rows, sends via fabric |
| `recv_accumulate.cpp` | Recv core: receives remote rows, copies to output |
| `fabric_dispatch.cpp` | Dispatch core: routes tokens to experts |
| `dispatch_writer_fused.cpp` | Local dispatch writer |
| `combine_dm_fused.cpp` | Weighted expert combine (scalar BF16) |
| `swiglu_sfpu.h` | SwiGLU SFPU implementation |

### Factory
| File | Role |
|------|------|
| `prefill_moe_compute_program_factory.cpp` | Core layout, CB sizing, kernel setup |
| `prefill_moe_compute_program_factory.hpp` | Factory header |

### Test & profile scripts
| File | Role |
|------|------|
| `test_fabric_return_k4_1x4.py` | Main test: 1×4 mesh, PCC validation |
| `profile_fabric_return_1x4.py` | Profiling with wall-clock + device profiler |
| `profile_clean_1x4.py` | Clean profiling with ReadDeviceProfiler |
| `parse_device_log.py` | Parse device profiler CSV output |
| `test_fabric_return_k4_1x2.py` | 1×2 mesh variant |
| `test_fabric_return_1x2.py` | 1×2 mesh variant |
| `test_local_return.py` | Local-only return test |
| `test_fabric_dispatch_compute.py` | Dispatch + compute test |

### Historical / reference
| File | Role |
|------|------|
| `kernels/` (top-level) | Earlier kernel versions (pre-Phase 3) |
| `step4_*.py` | Incremental patch scripts from development |
| `ANALYSIS_SUMMARY.md` | Earlier deadlock/perf analysis |
| `DEADLOCK_ANALYSIS.md` | Barrier ordering investigation |
| `KERNEL_ARCHITECTURE.txt` | Architecture notes |
| `PLAN.md` | Phase 1-3 optimization plan |

## Remote repo

The actual tt-metal integration is on the remote machine:
- **Host:** `ubuntu@UF-EV-B5-GWH02`
- **Path:** `/data/sraizada_2/tt-metal/`
- **Branch:** `sraizada/prefill-moe-compute-phase3`
- **Key commit:** `94fd5fc22f` (Phase 3 writer-side untilize)
- **Related branch:** `save-moe-gpt-work` (GPT-OSS port of decode moe_gpt op)
