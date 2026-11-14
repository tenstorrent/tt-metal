# HPCG model-local custom ops

`ttnn.generic_op` kernels and their Python wrappers. `tt/hpcg.py` builds every
program once and calls it in each CG iteration, so the whole iteration can be
captured in one trace.

Building the program from Python means a change to a descriptor or a kernel
needs no `_ttnn.so` rebuild.

## Layout

    custom_ops/
        __init__.py              package surface
        README.md                this file
        sparse_rows/             persistent sparse-row launch
            op.py, plan.py, kernels/
        cg_vector/               fused CG vector passes and the dot-product finish
            op.py, kernels/
        stencil_matrix/          setup: the device writes a level's matrix tables
            op.py, kernels/

## What the solver calls

| Op | Role |
| --- | --- |
| `SparseRowOp` mode `gs` | one symmetric Gauss-Seidel per launch: 16 colour phases, neighbours exchange updated rows on chip; the pre-smoother also runs the residual SpMV and writes `r - A z` for the restriction |
| `SparseRowOp` mode `gs`, SpMV phases only | the CG SpMV `Ap = A p`, cores kept in lockstep phase by phase |
| `SparseRowOp` mode `spmv`, 1-slot plan | restriction gather, and prolongation gather plus add |
| `CgVectorOp` mode `dot` | per-core partial sums of `r.r` (CG start), `r.z` and `p.Ap` |
| `CgVectorOp` mode `axpy` | `p = z + beta p`, with `beta` divided on chip |
| `CgVectorOp` mode `xr` | `x += alpha p`, `r -= alpha Ap` and the per-core `r.r` partial, with `alpha` divided on chip |
| `DotFinishOp` | one core adds the per-core partials into one value; options keep the previous `r.z` or take the square root |
| `StencilMatrixOp` | setup, once per level: writes the 27-slot matrix values and window byte offsets (and, on the fine level, `b`) into DRAM; bit-identical to the host-built tables |

`SparseRowPlan` (`sparse_rows/plan.py`) decides which rows each core owns, the
x window pieces each core loads into L1, every column's L1 byte offset, and
which rows each core pushes to which neighbour. For the Gauss-Seidel launches
it plans from the grid's storage-order formula (`stencil=`), and
`StencilMatrixOp` computes the column offsets on the device. Kernel metadata layout:
`sparse_rows/kernels/common.h`.

## Conventions

- A kernel path in `op.py` is relative to `TT_METAL_HOME`, for example
  `"models/experimental/hpcg/tt/custom_ops/sparse_rows/kernels/reader.cpp"`.
- Each op object is built once and launched by calling it; it takes ttnn
  tensors and writes its outputs in place, so the launches are trace-safe.
- An op validates its inputs and raises when they do not fit, including when
  its circular buffers do not fit L1. It does not fall back to a stock op.

## Tests

| Command | Purpose |
| --- | --- |
| `pytest models/experimental/hpcg/tests/test_hpcg.py` | every op against numpy, then CG iterations against fp64 |
| `python models/experimental/hpcg/demo/op_perf.py 192` | device time of each op against its speed of light |
| `python models/experimental/hpcg/demo/kernel_counters.py 192 gs0` | per-core cycle counters of one launch |
