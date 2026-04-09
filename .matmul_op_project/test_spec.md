# MatmulOp Test Specification

## Overview

These tests verify that the `MatmulOp` class produces mathematically correct matmul
results across all three usage modes. Tests are written against the design spec
(`design_spec.md`) and are independent of the implementation.

All tests run on Blackhole (p100a), single device, using `Float16_b` data format.

---

## Test Architecture

### Approach: C++ Compute Kernel + Python Harness

Since `MatmulOp` is a compute kernel header, we test it by:

1. Writing a **C++ compute kernel** (`test_matmul_op_kernel.cpp`) that includes
   `api/compute/matmul_op.h` and exercises the MatmulOp class in different modes.
2. Writing a **Python pytest harness** (`test_matmul_op.py`) that creates TT-Metal
   programs using that compute kernel, feeds tilized data, runs the program, and
   validates output against `torch.matmul`.

The compute kernel uses compile-time args to select which test mode to run.
The Python harness parametrizes across test cases, each with specific dimensions
and expected behavior.

### Data Flow

```
Python: torch tensors -> ttnn.from_torch(TILE_LAYOUT) -> device DRAM (interleaved)
Reader kernel: DRAM (interleaved) -> CB_in0, CB_in1
Compute kernel: CB_in0 x CB_in1 -> CB_out (using MatmulOp)
Writer kernel: CB_out -> DRAM (interleaved)
Python: ttnn.to_torch(device output) -> compare with torch.matmul
```

### Reader/Writer Kernels

We use inline reader/writer kernels embedded in the Python test:
- **Reader**: Reads tiles sequentially from interleaved DRAM using `TensorAccessor`
  into CB0 and CB1, block by block.
- **Writer**: Writes tiles sequentially from CB16 to interleaved DRAM using
  `InterleavedAddrGenFast`.

### Tile Ordering Constraint

The reader reads tiles sequentially from DRAM. For the sequential tile order to
match what the compute kernel expects within each CB block, we constrain test
dimensions:
- **in0 (A)**: Use M_tiles=1 when K blocking is needed. With M=1, each inner
  block's tiles are a contiguous run in DRAM row-major tile order.
- **in1 (B)**: Row-major tile order (K rows, N cols) naturally matches the expected
  block layout when read sequentially per inner block.

### CB Layout

| CB Index | Purpose | Format |
|----------|---------|--------|
| `c_0` (0) | Input A (in0) | Float16_b |
| `c_1` (1) | Input B (in1) | Float16_b |
| `c_16` (16) | Output | Float16_b |
| `c_24` (24) | Partials (spill/reload, when needed) | Float16_b |

### Python API

Tests use the `ttnn.generic_op` API with `ttnn.ProgramDescriptor` to create and
run programs from Python, following the pattern established in
`tests/ttnn/unit_tests/operations/debug/test_generic_op.py`.

---

## Test Cases

### Test 1: Mode 3 -- Basic Tile-Mode Matmul (`test_mode3_tile_basic`)

**What it verifies**: `TileMatmulOp::init()` + `TileMatmulOp::run()` with
K accumulation (multiple inner tiles per output tile).

**Spec references**: Mode 3 tile-mode `run()` algorithm; maps to call site T1.

**Input shapes**:
- A = (32, 128), B = (128, 32), C = (32, 32)
- In tiles: M=1, K=4, N=1, batch=1

**Compute kernel args**:
- `test_mode = 0` (mode3_tile_auto)
- Reader streams 1 tile at a time (4 blocks, 1+1 tiles per block)

**CB configuration**:
- CB0: 2 tiles (double-buffered for streaming)
- CB1: 2 tiles
- CB16: 1 tile (single output)

**Expected behavior**: `run()` loops 1 output tile, accumulating K=4 inner tiles
via `matmul_tiles`. Each inner step: wait 1 tile from each CB, matmul, pop.

**Correctness**: PCC >= 0.999 vs `torch.matmul(A, B)`

---

### Test 2: Mode 3 -- Block-Mode with Spill/Reload (`test_mode3_block_multisubblock`)

**What it verifies**: `BlockMatmulOp::init()` + `BlockMatmulOp::run()` with
inner-dimension blocking (num_blocks_inner=2) and automatic spill/reload.

**Spec references**: Mode 3 block-mode `run()` with spill/reload. Maps to B9/B10/B15.

**Input shapes**:
- A = (32, 128), B = (128, 64), C = (32, 64)
- In tiles: M=1, K=4, N=2
- subblock_h=1, subblock_w=2, in0_block_w=2
- in0_num_subblocks=1, in1_num_subblocks=1, num_blocks_inner=2

**CB configuration**:
- CB0: 4 tiles (in0_block=2, double-buffered)
- CB1: 8 tiles (in1_block=4, double-buffered)
- CB16: 2 tiles (output subblock)
- CB24: 2 tiles (partials for spill/reload)

**Expected behavior**: 2 inner blocks. First block: accumulate -> spill partials
to CB24. Second block: reload partials from CB24 -> accumulate -> pack to CB16.

**Correctness**: PCC >= 0.999 vs `torch.matmul(A, B)`

---

### Test 3: Mode 2 -- Block Accumulate with Spill/Reload (`test_mode2_block_spill_reload`)

**What it verifies**: `BlockMatmulOp` semi-automatic methods: `init()`,
`begin_subblock()`, `accumulate()`, `end_to_partials()`, `reload_partials()`,
`end_to_output()`.

**Spec references**: Mode 2 semi-automatic pattern with spill/reload. Maps to B1/B2/B3/B16.

**Input shapes**:
- A = (32, 128), B = (128, 64), C = (32, 64)
- In tiles: M=1, K=4, N=2
- subblock_h=1, subblock_w=2, in0_block_w=2
- in0_num_subblocks=1, in1_num_subblocks=1, num_blocks_inner=2

**CB configuration**: Same as Test 2 (CB0=4, CB1=8, CB16=2, CB24=2).

**Expected behavior**: The kernel manages the loop explicitly:
- Block 0: `begin_subblock() -> accumulate(kt=2) -> end_to_partials(2)`
- Block 1: `begin_subblock() -> reload_partials(2) -> accumulate(kt=2) -> end_to_output(cb16, 2)`

**Correctness**: PCC >= 0.999 vs `torch.matmul(A, B)`

---

### Test 4: Mode 2 -- Single Block, No Spill (`test_mode2_block_no_spill`)

**What it verifies**: `BlockMatmulOp` semi-automatic path WITHOUT spill/reload.
Single inner block: `begin_subblock() + accumulate() + end_to_output()`.

**Spec references**: Mode 2 with num_blocks_inner==1. Maps to B4/B5 (SDPA).

**Input shapes**:
- A = (32, 64), B = (64, 64), C = (32, 64)
- In tiles: M=1, K=2, N=2
- subblock_h=1, subblock_w=2, in0_block_w=2, num_blocks_inner=1

**CB configuration**:
- CB0: 4 tiles, CB1: 8 tiles, CB16: 2 tiles
- CB24: not used

**Expected behavior**: Single pass: `begin_subblock() -> accumulate(kt=2) ->
end_to_output(cb16, 2)`. No spill, no reload.

**Correctness**: PCC >= 0.999 vs `torch.matmul(A, B)`

---

### Test 5: Mode 1 -- Single Tile (`test_mode1_tile_single`)

**What it verifies**: `TileMatmulOp::init()` + `TileMatmulOp::matmul()` with
the caller managing DST acquire/release and pack_tile manually.

**Spec references**: Mode 1 tile-mode `matmul()`. Maps to T5/T6/T7/T8/T9.

**Input shapes**:
- A = (32, 32), B = (32, 32), C = (32, 32)
- Single tile each, K=1

**CB configuration**: CB0=2, CB1=2, CB16=1.

**Expected behavior**: `init() -> tile_regs_acquire() -> cb_wait(in0,1) ->
cb_wait(in1,1) -> matmul(0,0,0) -> cb_pop(in0) -> cb_pop(in1) ->
tile_regs_commit() -> cb_reserve(out,1) -> tile_regs_wait() -> pack_tile(0,out)
-> tile_regs_release() -> cb_push(out,1)`.

**Correctness**: PCC >= 0.999 vs `torch.matmul(A, B)`

---

### Test 6: Mode 1 -- Block Call (`test_mode1_block_call`)

**What it verifies**: `BlockMatmulOp::init()` + `BlockMatmulOp::matmul()` with
ct_dim=2, rt_dim=1, kt_dim=1 and caller managing DST.

**Spec references**: Mode 1 block-mode `matmul()`. Maps to B8/B11/B12.

**Input shapes**:
- A = (32, 32), B = (32, 64), C = (32, 64)
- ct_dim=2, rt_dim=1, kt_dim=1. One matmul_block call processes the 1x1x2 block.

**CB configuration**: CB0=2, CB1=4, CB16=2.

**Expected behavior**: `init() -> tile_regs_acquire() -> cb_wait(in0,1) ->
cb_wait(in1,2) -> matmul(0,0,0) [processes ct_dim*rt_dim*kt_dim block] ->
cb_pop -> tile_regs_commit() -> pack 2 tiles -> tile_regs_release()`.

**Correctness**: PCC >= 0.999 vs `torch.matmul(A, B)`

---

## Compile-Time Arg Layout for Test Kernel

| Index | Name | Description |
|-------|------|-------------|
| 0 | `test_mode` | 0=mode3_tile, 1=mode2_semi, 2=mode1_tile, 3=mode1_block, 4=mode2_no_spill, 5=mode3_block |
| 1 | `batch` | Batch count (for mode 3) |
| 2 | `Mt` | M dimension in tiles (or num_blocks_h for mode 3 block) |
| 3 | `Kt` | K dimension in tiles (or num_blocks_inner for mode 2/3 block) |
| 4 | `Nt` | N dimension in tiles (or num_blocks_w for mode 3 block) |
| 5 | `out_subblock_h` | Subblock row dim in tiles (also rt_dim for mode 1 block) |
| 6 | `out_subblock_w` | Subblock col dim in tiles (also ct_dim for mode 1 block) |
| 7 | `in0_block_w` | Inner dim block size in tiles (kt_dim) |
| 8 | `in0_num_subblocks` | Row subblocks count |
| 9 | `in1_num_subblocks` | Col subblocks count |

---

## PCC Thresholds

All tests use PCC >= 0.999. This is achievable for bfloat16 matmul at the tile
sizes tested (32-128 element dimensions). The HiFi4 math fidelity setting ensures
full precision computation.

---

## Timeout Policy

Each test has a 60-second timeout. If any test hangs:
1. Kill the process
2. Run `tt-smi -r` to reset the device
3. Report the hang

---

## Call Site Coverage Summary

| Test | Modes Exercised | Methods Tested | Call Sites Covered |
|------|----------------|----------------|-------------------|
| Test 1 | Mode 3 (tile) | init(), run() | T1 |
| Test 2 | Mode 3 (block) | init(), run() [with spill/reload] | B9, B10, B15 |
| Test 3 | Mode 2 (block, spill) | init(), begin_subblock(), accumulate(), end_to_partials(), reload_partials(), end_to_output() | B1, B2, B3, B16 |
| Test 4 | Mode 2 (block, no spill) | init(), begin_subblock(), accumulate(), end_to_output() | B4, B5, B6, B7 |
| Test 5 | Mode 1 (tile) | init(), matmul() | T3-T14 |
| Test 6 | Mode 1 (block) | init(), matmul() | B8, B11-B14 |

This gives coverage of all three modes, both `IsBlockMode=true` and `IsBlockMode=false`,
and exercises all key public methods: `init()`, `init_short()` (implicitly via
`reload_partials`), `matmul()`, `begin_subblock()`, `accumulate()`,
`end_to_output()`, `end_to_partials()`, `reload_partials()`, and `run()`.
