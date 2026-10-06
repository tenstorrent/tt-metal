# TensorOcean horizontal tracer flux on Blackhole

[TensorOcean](https://github.com/lanl/TensorOcean) (LANL) is a small benchmark taken from the MPAS-Ocean
model: the horizontal tracer flux and its accumulation, i.e. for every cell of a hexagonal mesh and every
depth level, how fast the tracer value (salt, heat, ...) changes because water carries it across the cell's
6 edges. Each edge's flux is a weighted sum over 10 nearby cells; each cell's output is the sum of its 6 edge
fluxes divided by its area.

This folder holds the TTNN port we received from HPE and an optimized version for one Blackhole chip.

| Version | Code | Time per step, 100 × 100 cells, 100 levels (P150) |
|---|---|---|
| baseline | `reference/optimized_ttnn.py`: the TTNN port as received (`horizontal_flux_ttnn`), fp32 | 1,186 ms |
| optimized | `tt/tensorocean.py`: a fused TT-Metalium kernel on 110 Tensix cores, fp32 | 0.523 ms (2,266×) |

Both match the float64 reference to a relative RMS error of about 1e-7 (PCC ≈ 1 − 1e-14).

## What is timed

One step = everything that changes per model time step, on the chip:
- the per-step inputs (tracer values `cell`, normal thickness flux `f`, high-order mask) start in DRAM in their
  natural row-major layout (the shapes of `make_inputs`), and the outputs end as natural `[levels, N/2, N]`
  arrays; any rearranging a version needs happens inside the timed step;
- mesh constants (coefficients, edge signs, edge lengths, cell areas) are prepared once;
- as in LANL's `optimized_gpu.py`, the host → device copy of the inputs is not part of the step (it is reported
  there separately as data movement; here about 1.6 ms for the 28.6 MB of per-step inputs).

The optimized version is timed with a trace (20 replays, median of 3); the baseline cannot be traced (it
synchronizes inside the function) and is timed as plain calls.

## What the optimized version does

- **One TT-Metalium kernel** (`tt/fused_kernel.py`, `tt/kernels/{reader,writer,compute}.cpp`) instead of ~320
  TTNN operations. Core (x, y) of an 11 × 10 grid takes strip x of the mesh and a range of depth levels and
  computes all its edges and cells; intermediate values stay in the core's registers and L1.
- Fluxes are computed in fp32 on the SFPU, 32 edges × 2 levels at a time, using
  `flux = f·mask·(P + 0.25·sign(f)·Q) + 2-cell term` with constant per-edge products prepared once.
- Each core loads its tracer values once and makes the shifted copies the aligned NoC copies need on the SFPU;
  f and mask are streamed in blocks of 128 edges, fetched one block ahead; the per-edge coefficients are read
  once per core column and multicast; fluxes live in a 5-block ring buffer in L1.
- Two data-movement kernels (`tt/kernels/relayout_in.cpp`, `relayout_out.cpp`) convert the natural inputs into
  the kernel's per-core layout and the outputs back, on all cores, in one pass each.

Per step at 100 × 100: input conversion ≈ 306 µs, fused kernel ≈ 172 µs, output conversion ≈ 62 µs.

## Run

From the tt-metal root, with the tt-metal Python environment:

```
python models/experimental/tensorocean/demo/demo.py --n 100 --levels 100 --version both
pytest models/experimental/tensorocean/tests/test_tensorocean_accuracy.py
pytest models/experimental/tensorocean/tests/test_tensorocean_perf.py
```

`--n` is the mesh size (n × n cells, even), `--levels` the number of depth levels.

## Layout

```
reference/optimized_ttnn.py   HPE's TTNN port (unchanged): make_inputs, the float64 reference algorithm, the baseline
tt/tensorocean.py             optimized version: prepare(), run()
tt/fused_kernel.py            the fused kernel's host side (program, buffers, per-core arguments)
tt/fused_plan.py              per-core work split and DRAM layouts
tt/formulation.py             which 10 cells each edge reads, which edges each output adds (derived from the reference)
tt/natural_io.py              the per-step inputs in natural layout
tt/kernels/                   reader, writer, compute (fused kernel); relayout_in, relayout_out (layout conversion)
tests/                        accuracy and perf tests, shared helpers
demo/demo.py                  run, check and time a version
```

## Notes

- Tested on a P150 (single Blackhole); the kernel uses an 11 × 10 core grid (the harvested grid of a QuietBox 2
  chip), also on chips with more columns.
- fp32 only; the accuracy bar is relative RMS error ≤ 1e-6 against float64.
- `reference/optimized_ttnn.py` is HPE's port of LANL's TensorOcean code, included unchanged.
