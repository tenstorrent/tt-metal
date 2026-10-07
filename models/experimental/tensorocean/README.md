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
| optimized | `tt/tensorocean.py`: one TT-Metalium program on 110 Tensix cores, fp32 | 0.331 ms (3,583×) |

Both match the float64 reference to a relative RMS error of about 1e-7 (PCC ≈ 1 − 1e-14).

## How it was optimized

I ran Claude Code with this prompt first:
> /goal We want to run the ttnn for TensorOcean on our p150. Please go through the steps of checking for correctness and checking perf time etc, then fixing and optimizing it. Especially consider optimizations in the form of better parallelization or seeing if DRAM memory could be kept in L1 if possible. Make sure to check correctness and measure perf time as you go to ensure you're on the right track and aren't breaking the model. Try to make sure the perf and accuracy tests don't take too long. Make sure to first calculate a speed of light on p150 and continue iterating until you hit 90% of sol.

Claude stopped once it hit 45% of its calculated speed of light. Afterwards I organized its work and files in its directory, asked it so that I understood the main optimizations made, and made sure I was able to run the accuracy and perf tests that it wrote. From analyzing its work I found out that it had "cheated": one of the optimizations it made involved reformatting the memory, which it did on the host CPU (very slowly) which wasn't measured by its own perf test. After clarifying the guidelines I used Claude Code to fix that issue by moving that slow CPU work onto the P150 device and then setting a goal to optimizing it further.


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

One step is **one TT-Metalium program** (`tt/fused_kernel.py`, `tt/kernels/{reader,writer,compute}.cpp`) instead
of ~320 TTNN operations. Core (x, y) of an 11 × 10 grid owns strip x of the mesh and a range of depth levels
(core row y). The program has three phases:

1. **Natural inputs → per-core layout, in L1** (`tt/kernels/relayout_fused.h`). The kernel needs each strip
   column by column; the natural arrays are row by row. Each core row reads the natural rows of its own depth
   levels from DRAM once; the compute engine transposes them (tilize, transpose, untilize, exact in fp32) so
   that each mesh column becomes one contiguous run; each column is sent over the NoC straight into the L1 of
   the core that owns it. A barrier over all cores ends the phase.
2. **Fluxes and outputs.** Fluxes are computed in fp32 on the SFPU, 32 edges × 2 levels at a time, using
   `flux = f·mask·(P + 0.25·sign(f)·Q) + 2-cell term` with constant per-edge products prepared once. Each core
   makes the shifted tracer copies the aligned NoC copies need on the SFPU; f and mask are read from L1 in
   blocks of 128 edges; the per-edge coefficients are read once per core column and multicast; fluxes live in
   a 5-block ring buffer in L1.
3. **Outputs → natural layout.** Each core sends its output columns to one assembler core per (level, even/odd
   part) in the same core row; the assembler transposes them back into rows and writes the natural
   `[levels, N/2, N]` rows to DRAM.

At 100 × 100 the flux phase takes about 170 µs of the 331 µs; most of the rest is reading the 28.6 MB of
per-step inputs from DRAM and moving every mesh column to its core.

Sizes the single program does not fit (meshes larger than about 100 × 100, or more than 11 depth levels per
core row, i.e. more than about 110 levels) automatically use three programs instead: `tt/kernels/relayout_in.cpp`
(natural inputs → per-core layout in DRAM), the same flux kernel, and `tt/kernels/relayout_out.cpp`.

## Run

From the tt-metal root, with the tt-metal Python environment:

```
python models/experimental/tensorocean/demo/demo.py --n 100 --levels 100 --version both
pytest models/experimental/tensorocean/tests/test_tensorocean_accuracy.py
pytest models/experimental/tensorocean/tests/test_tensorocean_perf.py
pytest models/experimental/tensorocean/tests/test_tensorocean_robust.py
```

`--n` is the mesh size (n × n cells, even), `--levels` the number of depth levels.

## Layout

```
models/experimental/tensorocean/
├── README.md
├── reference/
│   └── optimized_ttnn.py      HPE's TTNN port (unchanged): make_inputs, the float64 reference algorithm, the baseline
├── tt/
│   ├── tensorocean.py         optimized version: prepare(), run()
│   ├── fused_kernel.py        the fused kernel's host side (program, buffers, per-core arguments)
│   ├── fused_plan.py          per-core work split and DRAM layouts
│   ├── formulation.py         which 10 cells each edge reads, which edges each output adds (derived from the reference)
│   ├── natural_io.py          the per-step inputs in natural layout
│   └── kernels/
│       ├── reader.cpp         fused kernel: reads natural rows, L1 → math unit
│       ├── writer.cpp         fused kernel: coefficient multicast, sends columns, outputs → DRAM
│       ├── compute.cpp        fused kernel: transposes, fluxes and outputs on the SFPU
│       ├── common.h           constants shared by the three
│       ├── sfpu_shift.h       SFPU routine for the shifted tracer copies
│       ├── relayout_fused.h   natural inputs ↔ per-core layout inside the program (transposes on the compute engine)
│       ├── relayout_in.cpp    larger sizes: natural inputs → per-core layout in DRAM (separate program)
│       └── relayout_out.cpp   larger sizes: per-core outputs → natural outputs (separate program)
├── tests/
│   ├── common.py              both versions behind one interface, reference, metrics, timing
│   ├── test_tensorocean_accuracy.py
│   ├── test_tensorocean_perf.py
│   └── test_tensorocean_robust.py   new inputs after prepare(), NaN-filled outputs and L1, trace replays
└── demo/
    └── demo.py                run, check and time a version
```

## Notes

- Tested on a P150 (single Blackhole); the kernel uses an 11 × 10 core grid (the harvested grid of a QuietBox 2
  chip), also on chips with more columns.
- fp32 only; the accuracy bar is relative RMS error ≤ 1e-6 against float64.
- `test_tensorocean_robust.py` checks that nothing a step computes comes from `prepare()` or an earlier run: after
  `prepare()` it writes another step's tracer values, f and mask into the same DRAM buffers, fills the outputs
  and all of L1 with NaN, and compares eager runs and trace replays with that step's float64 reference.
- `reference/optimized_ttnn.py` is HPE's port of LANL's TensorOcean code, included unchanged.
