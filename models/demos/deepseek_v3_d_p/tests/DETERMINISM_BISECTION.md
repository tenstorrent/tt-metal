# Prefill determinism: is it the code or the box?

`test_det_ccl_micro.py` answers that in ~5 minutes, with no weights, no model, no trace and
no env setup. Run it on any Blackhole Galaxy (mesh 8x4).

The prefill determinism tests (`test_prefill_block.py`, and the CI job
`blaze_models_prefill_tests.yaml` "Blaze - Transformer Determinism") compare full-block
outputs and demand **bit-exactness** — `comp_pcc` returns exactly `1.0` via a `torch.equal`
fast path, so the threshold of 1.0 is not a tolerance. When that fails, all you know is
"something on this box is not reproducible". These tests split that apart.

## Run it

```
mpirun --bind-to none --pernode --tag-output bash -c 'source $TT_METAL_HOME/python_env/bin/activate; export PYTHONPATH=$TT_METAL_HOME; python3 -u -m pytest models/demos/deepseek_v3_d_p/tests/test_det_ccl_micro.py -p no:randomly -s -q'
```

Run from the repo root with `TT_METAL_HOME` set. 32 tests, all of which pass on a healthy box.
Local tests only (the fast hardware check, 14 tests):

```
mpirun --bind-to none --pernode --tag-output bash -c 'source $TT_METAL_HOME/python_env/bin/activate; export PYTHONPATH=$TT_METAL_HOME; python3 -u -m pytest models/demos/deepseek_v3_d_p/tests/test_det_ccl_micro.py -k "test_local_ or test_report_device_mapping or test_matmul_core_sweep" -p no:randomly -s -q'
```

## What each test decides

| Test | Params | What a failure means |
|---|---|---|
| `test_ccl_determinism` | 12 | One TP collective, alone, on a fixed input. Failure implicates the op or fabric. |
| `test_ccl_chain_determinism` | 6 | 7 chained collectives, past the depth-2 semaphore pools, incl. free-immediately and sync-between variants. Failure implicates inter-op sync or handle reuse. |
| `test_local_compute_determinism` | 1 | Local matmul chain, **no collective**. Identical input+weights replicated to all 32 chips, so a failure is a chip. |
| `test_local_op_determinism` | 8 | Which subsystem: `readback` (DRAM), `eltwise` (unpack/SFPU/pack), `matmul1`, `matmul2`, each at seq 3200 (ISL 25600) and 640 (a 5120 chunk). First rung to fail names it. |
| `test_local_matmul_core_locality` | 3 | Whether a bad matmul footprint tracks the core grid or an output address. Three shapes; if the *block index* is invariant while the tile offsets scale, it is one core. |
| `test_matmul_core_sweep` | 1 | Which core, by coordinate. `allowed_worker_cores` confines the matmul to one core at a time and the sweep walks the grid, so a failure names the core instead of implying it. |
| `test_report_device_mapping` | 1 | shard index -> physical device id. Never assume identity. |

Everything from `test_local_compute_determinism` down replicates identical inputs to all 32
chips and uses no fabric, so a chip that disagrees with the other 31 **is** the fault.
A failure there is below tt-metal — one chip, not the code; only the first two tests can
implicate tt-metal. Below tt-metal is not automatically the silicon: firmware owns the
operating point (AICLK, VDD, DVFS, harvesting), so a marginal core can pass under one
firmware and drift under another. Rule out the firmware/KMD pair before calling it a bad die.

Read the results in this order — the first failing row is the answer:

```
CCL tests fail                      -> tt-metal, or fabric. Not a bad chip.
CCL pass, local_compute fails       -> one chip. Continue.
readback fails                      -> DRAM / storage on that chip.
eltwise fails, readback passes      -> unpack/SFPU/pack path.
matmul* fail, readback+eltwise pass -> the matmul (FPU) path.
core_locality block index invariant -> one Tensix core on that chip. Not a code bug.
matmul_core_sweep names one core     -> that core, by logical coordinate.
                                       Next: same box on a newer fw/KMD pair. Still fails
                                       -> the die (harvest or RMA). Passes -> firmware.
```

## Reference: b07u08 (a good box), 2026-07-29

```
27 passed, 3 warnings in 1098.09s (0:18:18)
```

Every local test logs `chips=[]` at every iteration:

```
local matmul chain, no CCL: 10 iterations, seq_local=3200
  iter 1..9: BIT-EXACT ndiff=0/734003200 maxabs=0.000e+00 chips=[]
local readback bit-exact across 8 iterations and all 32 chips
local eltwise  bit-exact across 8 iterations and all 32 chips
local matmul1  bit-exact across 8 iterations and all 32 chips
local matmul2  bit-exact across 8 iterations and all 32 chips
matmul 3200x7168 @ 7168x4608 -> output 100x144 tiles: bit-exact, 6 iterations, all 32 chips
matmul 3200x7168 @ 7168x2304 -> output 100x72  tiles: bit-exact, 6 iterations, all 32 chips
matmul 1600x7168 @ 7168x4608 -> output 50x144  tiles: bit-exact, 6 iterations, all 32 chips
```

Same mesh mapping as b06u02, so the two boxes are directly comparable — row 5 is shards
20-23 -> devices [10, 14, 22, 18] on both. That is the row that fails on b06u02 and passes
here, which rules out anything mapping- or topology-shaped.

**A box can also fail this suite for a reason that has nothing to do with determinism.** On
first contact b07u08 errored all 27 tests in 22 s with `MMIO per-op timeout: 4B load took
55632 us (budget=2 ms)` at `ttnn/ttnn/distributed/distributed.py:671`, and `tt-smi -ls` itself
reported `Read 0xffffffff over PCIe ID 15: the board should be reset`. That is a wedged board
at `open_mesh_device`, before any kernel runs — no test result at all, not a failing one.
`tt-smi -glx_reset_auto` cleared it and the box then passed 27/27. Use `tt-smi -glx_reset_auto`
on these Galaxy boxes, not `tt-smi -r`.

## Reference: b06u02 (a bad box), 2026-07-28/29

Excerpts below are from runs on b06u02 over two days; the assertion totals and the
per-iteration lines in a block therefore come from different runs of the same test. That is
harmless here, and it is the point: **the diff counts are intermittent and will not reproduce
exactly.** What reproduces on every run is the chip, the rectangle, and the block index.
A healthy box logs `bit-exact` for all of these.

**Verdict for this box: physical device 14 (PCI `0000:47:00.0`) has one Tensix core whose
matmul FPU path is intermittently wrong. Its DRAM and SFPU paths are bit-exact.** Not a
tt-metal bug. The same failure is reported on b07u02.

### Collectives are clean

`test_ccl_determinism` 12/12 bit-exact, `test_ccl_chain_determinism` 6/6 bit-exact, at
seq_local 3200 / 640 / 128. That covers `ttnn.reduce_scatter`,
`reduce_scatter_minimal_async` (fresh *and* persistent intermediate) and `all_gather`, plus
chains long enough to wrap the depth-2 semaphore pools. So the fabric, the pools and buffer
aliasing are all exonerated before any hardware claim is made.

### `test_local_compute_determinism` — no collective at all

```
local matmul chain, no CCL: 10 iterations, seq_local=3200
run-to-run divergence by chip (chip: total differing elements): {21: 385005}
chip-vs-chip0 divergence (chip: total differing elements): {21: 452740}
E  AssertionError: local matmul chain is not deterministic without any collective:
       run_to_run={21: 385005} chip_vs_chip0={21: 452740}
```

385005 of 22937600 elements = 1.7%. Shard 21 disagrees with the other 31 chips *and* with
itself between iterations, with no collective anywhere in the program.

### `test_local_op_determinism` — which subsystem

```
local readback bit-exact across 8 iterations and all 32 chips
local eltwise  bit-exact across 8 iterations and all 32 chips
local matmul1: iter 0: disagrees with chip 0: [21] | chip 21: 174 diffs over 38 rows x 24 cols,
    row tiles 10 (first [80, 81, 82, 83, 84, 85]), col tiles 12 (first [72, 73, 74, 75, 76, 77]),
    maxabs=3.540e-03
E  AssertionError: local matmul1 not deterministic: run_to_run={21: 1986} chip_vs_chip0={21: 2093}
E  AssertionError: local matmul2 not deterministic: run_to_run={21: 267031} chip_vs_chip0={21: 277616}
```

DRAM and the SFPU/pack path are bit-exact on all 32 chips; only the matmul path fails, and only
on shard 21. Every `matmul1` disagreement lands in the same tile-aligned rectangle every
iteration.

### `test_local_matmul_core_locality` — a core, not an address

```
matmul 3200x7168 @ 7168x4608 -> output 100x144 tiles, 6 iterations
  iter 0: vs chip 0    -> chip 21: 368 diffs | row tiles 80-89 (10 tall), col tiles 72-83 (12 wide) | block idx (M 8 of 10.0, N 6 of 12.0) | maxabs=6.310e-03
  iter 1: vs chip 0    -> chip 21: 356 diffs | row tiles 80-89 (10 tall), col tiles 72-83 (12 wide) | block idx (M 8 of 10.0, N 6 of 12.0) | maxabs=9.949e-03
  iter 1: run-to-run   -> chip 21: 380 diffs | row tiles 80-89 (10 tall), col tiles 72-83 (12 wide) | block idx (M 8 of 10.0, N 6 of 12.0) | maxabs=5.371e-03
  iter 3: vs chip 0    -> chip 21: 135 diffs | row tiles 80-89 (10 tall), col tiles 72-83 (12 wide) | block idx (M 8 of 10.0, N 6 of 12.0) | maxabs=3.540e-03
  iter 5: vs chip 0    -> chip 21: 376 diffs | row tiles 80-89 (10 tall), col tiles 72-83 (12 wide) | block idx (M 8 of 10.0, N 6 of 12.0) | maxabs=6.310e-03

matmul 3200x7168 @ 7168x2304 -> output 100x72 tiles, 6 iterations   [halfN]
  iter 0: vs chip 0    -> chip 21: 141 diffs | row tiles 80-89 (10 tall), col tiles 36-41 (6 wide) | block idx (M 8 of 10.0, N 6 of 12.0) | maxabs=4.395e-03
  iter 1: run-to-run   -> chip 21: 154 diffs | row tiles 80-89 (10 tall), col tiles 36-41 (6 wide) | block idx (M 8 of 10.0, N 6 of 12.0) | maxabs=1.123e-02
  iter 5: vs chip 0    -> chip 21:  92 diffs | row tiles 80-89 (10 tall), col tiles 36-41 (6 wide) | block idx (M 8 of 10.0, N 6 of 12.0) | maxabs=3.418e-03

matmul 1600x7168 @ 7168x4608 -> output 50x144 tiles, 6 iterations   [halfM]
  iter 0: vs chip 0    -> chip 21: 132 diffs | row tiles 40-44 (5 tall), col tiles 72-83 (12 wide) | block idx (M 8 of 10.0, N 6 of 12.0) | maxabs=3.601e-03
  iter 1: run-to-run   -> chip 21: 159 diffs | row tiles 40-44 (5 tall), col tiles 72-83 (12 wide) | block idx (M 8 of 10.0, N 6 of 12.0) | maxabs=4.456e-03
  iter 4: vs chip 0    -> chip 21: 154 diffs | row tiles 40-44 (5 tall), col tiles 72-83 (12 wide) | block idx (M 8 of 10.0, N 6 of 12.0) | maxabs=8.606e-03

E  AssertionError: matmul 3200x4608 not deterministic: run_to_run={21: 1714} chip_vs_chip0={21: 1891}
E  AssertionError: matmul 3200x2304 not deterministic: run_to_run={21:  704} chip_vs_chip0={21:  722}
E  AssertionError: matmul 1600x4608 not deterministic: run_to_run={21:  887} chip_vs_chip0={21:  993}
```

The rectangle's tile offsets scale with the shape while the block index does not:

| shape | output tiles | bad rectangle | block index |
|---|---|---|---|
| 3200x4608 | 100x144 | rows 80-89, cols 72-83 | (M 8 of 10, N 6 of 12) |
| 3200x2304 | 100x72 | rows 80-89, cols **36-41** | (M 8 of 10, N 6 of 12) |
| 1600x4608 | 50x144 | rows **40-44**, cols 72-83 | (M 8 of 10, N 6 of 12) |

Halving N halves the column offset; halving M halves the row offset. An address-bound or
tile-index-bound fault would have stayed at the same tiles. Block (8,6) of the 10x12 core
grid is invariant across all three, which is one core. Turning that block index into a
coordinate needs `transpose_mcast` from the program config, which is what the next test
removes the need for.

### `test_matmul_core_sweep` — the core, by coordinate

`allowed_worker_cores` sets both the origin and the grid of the 2D matmul factory
(`matmul_multicore_reuse_mcast_2d_program_factory.cpp:3174,3467`), so a 1x1 range runs the whole
matmul on one core. Each core gets the same 10x12 output tiles over the full K that the failing
matmul hands it, since less work per core means fewer chances to trip an intermittent fault.
120 cores, 10 iterations each, ~85 s.

On b06u02 one core fails and 119 are bit-exact: logical **(6,8)**, physical (7,10) per the mesh,
on chip 21 — the coordinate the block index predicted. A 2x2 window over (5,8) and another over
(6,8) also fail, and in both the differing elements sit entirely inside (6,8) while its three
neighbours doing identical work stay bit-exact. That is what makes the result independent of how
the factory resolves a window origin.

A 1x1 window cannot sit on the last grid row or column: the factory reads its neighbours at
`start + 1` unconditionally when building the mcast ranges (same file, lines 1173 and 1176), so
the op throws `No core coordinate found at location`. The sweep covers those 21 cores with 2x2
windows anchored one back.

Magnitudes matter for the diagnosis: `maxabs` is ~5e-3 against 0.02-scale activations, and
only ~0.1-0.3% of the elements *inside* that core's block differ per run. Low-order
accumulation bits, intermittent — a marginal core, not a stuck bit. `matmul2` keeps the same
row tiles but smears across every column, because one bad output row of the first matmul
contaminates its whole row in the second.

### `test_report_device_mapping` — shard index is not a device id

```
mesh shape 8x4, get_device_ids() order:
  [0, 4, 28, 24, 1, 5, 29, 25, 2, 6, 30, 26, 3, 7, 31, 27,
   11, 15, 23, 19, 10, 14, 22, 18, 9, 13, 21, 17, 8, 12, 20, 16]
  row 0: shard idx [0, 1, 2, 3]     -> device ids [0, 4, 28, 24]
  row 1: shard idx [4, 5, 6, 7]     -> device ids [1, 5, 29, 25]
  row 2: shard idx [8, 9, 10, 11]   -> device ids [2, 6, 30, 26]
  row 3: shard idx [12, 13, 14, 15] -> device ids [3, 7, 31, 27]
  row 4: shard idx [16, 17, 18, 19] -> device ids [11, 15, 23, 19]
  row 5: shard idx [20, 21, 22, 23] -> device ids [10, 14, 22, 18]
  row 6: shard idx [24, 25, 26, 27] -> device ids [9, 13, 21, 17]
  row 7: shard idx [28, 29, 30, 31] -> device ids [8, 12, 20, 16]
```

The bad shard 21 is **physical device 14**. Device *21* is a different, healthy chip. Reading
a shard index as a device id names the wrong chip and sends you to the wrong board.

## Why the model-level view misleads

Region-splitting the dense block moves the implicated *op* with sequence length while the
implicated *devices* never move: isl 1024 -> `output` only; isl 5120 -> `post_mla_residual`;
isl 25600 -> already `tt_kv`, amplifying to ~54M differing elements by `output`. All of them
land on mesh row 5. An op bug cannot select one row of the mesh. In a TP reduce-scatter one
chip's bad read feeds every output chunk, so a single bad chip presents as all four chips of
its line — which reads like a CCL bug and is not one.

Two more traps, both already ruled out, so do not re-derive them:

- `ttnn.reduce_scatter` is not a separate legacy op. It resolves to `ttnn::prim::reduce_scatter`,
  whose program factory includes the same line/ring factories `reduce_scatter_minimal_async`
  uses. "Legacy vs async reduce" is not a real axis; only the semaphore source differs.
- `line_reduction.cpp` accumulates with a sequential `for` loop, so arrival-order FP
  accumulation is structurally impossible.
