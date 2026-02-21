# GPT-OSS Fused MoE Compute Kernel (`moe_gpt`) — Integration Guide

## Overview

This document describes the fused MoE compute kernel adapted from DeepSeek for GPT-OSS.
The kernel performs the full expert computation (W0/W1 matmul → SwiGLU activation → ring
All-to-All → W2 matmul) in a single fused operation across 12 cores on a single chip.

The next step is integrating this kernel with `all_to_all_dispatch` and `all_to_all_combine`
ops to run the full multi-chip GPT-OSS MoE layer.

## GPT-OSS vs DeepSeek Dimensions

| Parameter | DeepSeek | GPT-OSS |
|---|---|---|
| hidden_dim (K) | 7168 | 2880 |
| intermediate_dim (N) | 2048 | 2880 |
| experts per device (E) | 2 | 4 |
| K tiles (K/32) | 224 | 90 |
| N tiles (N/32) | 64 | 90 |
| W0/W1 shape per expert | [7168, 2048] | [2880, 2880] |
| W2 shape per expert | [2048, 7168] | [2880, 2880] |

Key difference: GPT-OSS has **symmetric** weight matrices (K == N == 2880), so W0/W1
and W2 have identical tile distributions across cores.

## Architecture

### Kernel Structure (3 RISC-V cores per Tensix)

1. **dm0** (BRISC/RISCV_1, NOC_0): Reads W0/W1 and W2 weights from DRAM.
   Triple-buffered pipeline with transaction IDs.

2. **dm1** (NCRISC/RISCV_0, NOC_1): Handles ring All-to-All data movement.
   Sends intermediate results to neighbor cores via NOC1 posted writes.
   Uses semaphore-based signaling for synchronization.

3. **compute** (TRISC): Performs matmul + SwiGLU activation + matmul.
   - Phase 1: `input @ [W0, W1]` → SwiGLU activation → intermediate result
   - Phase 2: `intermediate @ W2` → output (written in-place to input buffer)

### Ring All-to-All (A2A)

- 12 cores form a ring ordered by physical NOC1 coordinates (descending y, descending x)
- Each core computes its local intermediate result, then rotates it around the ring
- 2 A2A iterations × 12 steps = 24 rotations per expert
- 2 NOC packets per step (4 tiles × 2048 bytes = 8192 bytes each)
- 6-buffer cycling scheme (buffers 0..5) with 5 steps of slack before overwrite
- Semaphore-based synchronization: predecessor writes semaphore value to signal data readiness

### Activation: SwiGLU

```
gate_clamped = clamp(gate, max=7.0)
up_clamped   = clamp(up, min=-7.0, max=7.0)
result       = (up_clamped + 1) * gate_clamped * sigmoid(1.702 * gate_clamped)
```

Implemented as an SFPU kernel running on the PACK thread (`swiglu_sfpu.h`).

## Key Constants (moe_gpt_ring_common.h)

| Constant | Value | Derivation |
|---|---|---|
| NUM_CORES | 12 | DRAM banks on Wormhole |
| NUM_W0_W1_TILES_H | 90 | K/32 = 2880/32 |
| NUM_W2_TILES_H | 90 | N/32 = 2880/32 |
| W0_W1_TILES_PER_TXN | 10 | Largest factor of 90 fitting 8KB NOC packet |
| W2_TILES_PER_TXN | 10 | Same |
| IN2_TILES_PER_STEP | 8 | max(tiles per core) = max(7,8) |
| NUM_A2A_ITERS | 2 | max(W2_TILES_PER_CORE) / 4 = 8/4 |
| W2_BLOCKS_PER_EXPERT | 36 | w2_blocks_per_four_mm2_tile × NUM_A2A_ITERS = 18×2 |

Tile distribution across 12 cores (90/12 = 7.5):
- Cores {0,1,4,5,8,9}: 8 tiles (FULL_CORES)
- Cores {2,3,6,7,10,11}: 7 tiles (PAD_CORES)

This distribution applies to BOTH W0/W1 width and W2 width since K == N.

## Performance

| Metric | Value |
|---|---|
| Total kernel time (4 experts) | 272.7 μs |
| Per expert | 68.2 μs |
| DRAM data per core per expert | 1.24 MB |
| DRAM theoretical minimum | 51.8 μs/expert |
| DRAM BW utilization | 76% |
| Accuracy (PCC) | ~0.990 (threshold 0.984) |

## File Inventory

### Kernel files (`ttnn/cpp/ttnn/operations/experimental/moe_gpt/device/kernels/`)

| File | Description |
|---|---|
| `moe_gpt_ring_common.h` | Dimension constants, tile distribution lookup tables |
| `compute.cpp` | Matmul + SwiGLU + matmul compute kernel |
| `dm0.cpp` | DRAM weight reader (BRISC, NOC_0) |
| `dm1.cpp` | Ring A2A data movement (NCRISC, NOC_1) |
| `swiglu_sfpu.h` | SwiGLU SFPU implementation (from `origin/sraizada/gpt-oss-swiglu`) |

### Host files (`ttnn/cpp/ttnn/operations/experimental/moe_gpt/device/`)

| File | Description |
|---|---|
| `moe_gpt_program_factory.cpp` | CB allocation, kernel creation, runtime args setup |

### Test files

| File | Description |
|---|---|
| `tests/ttnn/nightly/unit_tests/operations/experimental/test_moe_gpt.py` | Full test with tensor preparation and accuracy checks |
| `run_moe_gpt_test.py` | Standalone test runner (bypasses conftest.py hook issue) |
| `run_moe_gpt_profile.py` | Tracy profiling script |

## Weight Tensor DRAM Layout

### W0/W1 (interleaved)

Shape per DRAM bank: `(L, E, groups_per_core, K, 4×TILE_SIZE)` in tiles.

- `groups_per_core = 4` (8 max tiles / 2 per group)
- Each group contains 2 paired column tiles: [w0_col_i, w1_col_i, w0_col_j, w1_col_j]
- DRAM stride between experts: `w0_w1_total_size_per_expert = 2 × 90 × 8 × 576 bytes`
- Cores with 7 tiles have their 8th tile padded with zeros

### W2

Shape per DRAM bank: `(L, E, 2, N, 4×TILE_SIZE)` in tiles.

- 2 groups of 4 output tiles each (8 max tiles / 4 per group)
- N tiles are reordered per core to match ring A2A rotation order
- DRAM stride: `w2_total_size_per_expert = 90 × 8 × 576 bytes`
- Cores with 7 tiles have group 2 padded to 4 tiles

### Input/Output (sharded L1)

Shape: `(num_cores, E, M, K)` with `shard_shape = (E×M, K) = (128, 2880)`.

- HEIGHT_SHARDED in L1, one shard per core
- Output is written in-place to the input tensor
- Each expert's result occupies `NUM_W0_W1_TILES_H = 90` tile rows

## Circular Buffer Layout

| CB | Index | Format | Tiles | Bytes | Purpose |
|---|---|---|---|---|---|
| cb_r2c_w0 | c_0 | Bfp4_b | 60 | 34,560 | Triple-buffered weight read (aliased as cb_r2c_w2) |
| cb_s2c_in | c_1 | Float16_b | 360 | 737,280 | Sharded input (aliased as cb_c2s_out for output) |
| cb_c2w_rdy | c_2 | Float32 | 1 | 4 | Compute→dm1 ready signal |
| cb_w2c_rdy | c_3 | Float32 | 1 | 4 | dm1→compute ready signal |
| cb_s2c_in2 | c_4 | Float16_b | 48 | 98,304 | A2A intermediate data (6 buffers × 8 tiles) |

Total L1 per core: ~530 KB (fits in 1.2 MB L1).

## Known Issues and Design Decisions

### Semaphore Reset for Program Caching

The ring semaphore must be explicitly reset to 0 at the start of dm1's `kernel_main()`:
```cpp
noc_semaphore_set(my_semaphore_ptr, 0);
uint32_t semaphore_value = 0;
```
Without this, cached program invocations inherit stale semaphore values and deadlock
intermittently. This was the root cause of 3/20 failure rate before the fix.

### Cross-Expert Boundary Barrier

After each expert's ring loop completes, dm1 waits for the predecessor's final A2A write
(which targets buf 0) before signaling compute to proceed with the next expert's SwiGLU.
This prevents the next expert's output from overwriting buf 0 while the predecessor is
still writing to it.

### Double Flush in dm1

There are two `noc_async_posted_writes_flushed()` calls per A2A step:
1. After data packets, before semaphore write — **required** (data must land before signal)
2. After semaphore write — potentially removable for optimization

### conftest.py Hook Issue

The root `conftest.py` has a broken `pytest_handlecrashitem` hook that prevents pytest
from running. Workaround: use standalone scripts (`run_moe_gpt_test.py`, `run_moe_gpt_profile.py`)
that import the test functions directly.

## Integration with all_to_all_dispatch / all_to_all_combine

### Expected Data Flow (Multi-Chip)

```
all_to_all_dispatch  →  moe_gpt (fused expert compute)  →  all_to_all_combine
[distribute tokens]     [W0/W1 → SwiGLU → A2A → W2]       [gather results]
```

### Input Contract

`moe_gpt` expects:
- **input**: HEIGHT_SHARDED L1 tensor, shape `(num_cores, E, M, K)`, dtype Float16_b
- **w0_w1**: HEIGHT_SHARDED DRAM tensor, prepared by `prepare_w0_w1_tensor()`
- **w2**: HEIGHT_SHARDED DRAM tensor, prepared by `prepare_w2_tensor()`
- **output**: Same tensor as input (in-place)

### Output Contract

- Output is written in-place to the input tensor
- Shape: `(num_cores, E, M, K_padded)` where K_padded = max_tiles_per_core × 32
- Use `prepare_output_tensor()` to extract valid tiles per core

### Build System

**MUST build with `./build_metal.sh`**, not `cmake --build build`.

### Profiling

```bash
source python_env/bin/activate
python -m tracy -v -r -p -o <output_dir> -a device_kernel_duration -- run_moe_gpt_profile.py
```
