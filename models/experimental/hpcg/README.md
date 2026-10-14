# HPCG on Blackhole P150 (fp32)

The [HPCG](https://github.com/hpcg-benchmark/hpcg) 3.1 conjugate-gradient solve,
with its 4-level multigrid preconditioner, on one P150 card. The matrix is
stored the way the HPCG rules require: every term of the 27-point stencil, an
fp32 value and a 32-bit column index per non-zero, read through the column
index (no stencil structure is exploited).

**This is not an official HPCG result.** It does the same math as HPCG, but it
does not meet HPCG's precision rule or its run and scoring rules (see
[Rules](#rules)). The main gap: the rules require IEEE fp64 in every phase, and
Tensix cores have no fp64 arithmetic, so this port runs in fp32. Within 50
iterations its residuals match an fp64 run of the same algorithm to 4-6 digits.

## Run

From `$TT_METAL_HOME`:

| Command | What it does |
| --- | --- |
| `pytest models/experimental/hpcg/tests/test_hpcg.py` | every device op against numpy, then 3 CG iterations against fp64 (64³ and 160³) |
| `pytest models/experimental/hpcg/tests/test_host_setup.py` | the compiled host setup builds exactly what the numpy code builds (host only) |
| `python models/experimental/hpcg/demo/demo.py 192` | 50 traced CG iterations: ms per iteration, raw GFLOP/s, share of speed of light, residual history; HPCG-style rating where a CPU reference residual is known |
| `python models/experimental/hpcg/demo/op_perf.py 192` | device time of each op against its speed of light |
| `python models/experimental/hpcg/demo/kernel_counters.py 192 gs0 [--per-core]` | per-core cycle counters of one launch |
| `python -m models.experimental.hpcg.reference.speed_of_light 192 23.75` | speed of light of one iteration, and the share a measured time reaches |
| `python -m models.experimental.hpcg.reference.problem 104 64` | the numpy reference solve (fp64 or fp32) |

The grid size nx must be a multiple of 8 (HPCG's default is 104); 304 is the largest that fits L1.

## Results (P150a, 130 cores, 512 GB/s GDDR6)

Two numbers per grid:

- **Raw GFLOP/s**: HPCG's flop count for the timed iterations divided by their
  time. This measures how fast the device runs the solver.
- **HPCG-style rating**: the formula HPCG's `ReportResults.cpp` uses for its
  final score, computed by `demo/demo.py`. It makes two deductions:
  - **extra iterations**: the 8-colour ordering needs 51-53 iterations to reach
    the residual HPCG's reference code reaches in 50, and only 50 are credited;
  - **setup time**: HPCG adds (setup + optimization time) / 10 to every set of
    CG iterations. Here, setup is building the matrix, every core's index
    tables and the launch programs on the host, and uploading them: 0.2 s at
    104³, 0.6 s at 192³, 1.9 s at 288³ (`host/native_setup.cpp`, compiled on
    first use; it was 1.6 s, 10.7 s and 34 s in numpy).

| Grid | ms / iteration | Raw GFLOP/s | Speed of light | Share | Iterations to the reference residual | HPCG-style rating |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 104³ | 4.88 | 84.5 | 3.21 ms | 65.9% | 51 | 77.9 GFLOP/s |
| 128³ | 7.69 | 100.3 | 6.01 ms | 78.1% | 53 | 89.4 GFLOP/s |
| 192³ | 23.68 | 110.6 | 21.13 ms | 89.2% | 53 | 99.7 GFLOP/s |
| 240³ | 45.49 | 112.7 | 41.35 ms | 90.9% | not measured | - |
| 288³ | 77.84 | 114.0 | 71.56 ms | 91.9% | not measured | - |
| 304³ | 91.53 | 114.0 | 84.19 ms | 92.0% | not measured | - |

The iteration count needs the CPU reference residual for that grid size, which
was only run up to 192³.

Without the setup charge (iteration penalty only), the rating would be 82.8,
94.6 and 104.3 GFLOP/s at 104³, 128³ and 192³.

**Comparison with a CPU.** HPCG's fp64 reference code on a 24-thread Ryzen 9
7900X rates 3.83, 4.02 and 4.40 GFLOP/s at 104³, 128³ and 192³. That is
HPCG's final rating, so it already includes the setup charge (the reference
code needs no extra iterations). Compared like for like, the P150's
HPCG-style rating is 20.3x, 22.2x and 22.7x the CPU's.

**Speed of light.** `reference/speed_of_light.py` is each operation's minimum
DRAM traffic under the HPCG rules at 512 GB/s; every operation is
memory-bound. The fastest rate measured with all 130 cores reading DRAM is
about 490 GB/s (95.7%), and at 288³ the fine-grid smoothers move their actual
bytes at 98.6-99.9% of it.

## Rules

Checked against HPCG 3.1's source (`main.cpp`, `ReportResults.cpp`,
`TestCG.cpp`) and the [HPCG FAQ](https://www.hpcg-benchmark.org/faq/index.html).

Followed:

- **Unstructured matrix storage.** Each non-zero stores an fp32 value and a
  32-bit column index, and x is read through that index. The 27-point
  stencil's regular structure is not used.
- **Every term, every sweep.** All 27 slots are multiplied in every pass,
  including zero padding at the grid edges.
- **No symmetry shortcut.** The full matrix is stored and read.
- **Same algorithm.** 4 multigrid levels, one pre- and one post-smoothing
  symmetric Gauss-Seidel per level, HPCG's restriction and prolongation, and
  initial guess zero.
- **Allowed reordering.** The 8-colour ordering and the y-bands are
  reorderings; the iterations they add are measured and deducted in the
  rating above.
- **Flop count.** HPCG's own formula, with true non-zero counts.

Not followed:

- **fp64.** Everything runs in fp32.
- **Run length.** An official run times at least 1800 s of CG sets; these
  runs time 50 iterations (0.25-4.6 s).
- **Problem size.** The FAQ asks for at least a quarter of main memory. 304³
  uses about 7 GB of the card's 32 GB (22%), and it is the largest grid whose
  data fits the cores' L1.
- **Validation tests.** HPCG's TestCG, TestSymmetry and TestNorms are not run
  against the device solver. TestSymmetry measures departures in units of
  fp64 rounding, so an fp32 solver fails it: an all-fp32 build of the
  reference code fails it on a CPU.

## Layout

    host/
        native_setup.cpp     compiled host setup: level matrices, per-core plans (same output as the numpy code)
        native.py            builds it with the system C++ compiler on first use and loads it
    reference/
        problem.py           HPCG problem in device storage order + numpy reference solver
        speed_of_light.py    per-op roofline of one CG iteration
    tt/
        hpcg.py              HPCG: device levels, ops, one CG iteration (trace-capturable)
        layout.py            host <-> device vector layout
        custom_ops/          ttnn.generic_op kernels (see custom_ops/README.md)
            sparse_rows/     smoothers, SpMV, restriction, prolongation
            cg_vector/       fused dot products and vector updates
    tests/test_hpcg.py       device ops vs numpy, CG vs fp64
    tests/test_host_setup.py compiled host setup vs the numpy definitions
    demo/                    demo.py, op_perf.py, kernel_counters.py

## How it works

- **Ordering.** Rows are split into 8 colours by coordinate parity; no two
  neighbours share a colour, so Gauss-Seidel updates one colour at a time on
  all cores (HPCG allows this reordering). Colours run 7 to 0 forward.
- **Ownership.** On grids of 112³ and up, each colour is stored in 5 or 10
  bands along y, and each band is split along z over the cores. A core's rows
  then form a short block in both y and z.
- **x windows.** Each launch loads, once, every x value its rows read into the
  core's L1 (its own rows, one band-plane above and below, and short lines from
  the neighbouring bands; about 1.2x the owned rows). Smoother phases push
  updated rows straight into the neighbours' windows over the NoC.
- **Streaming and gather.** The reader and writer stream matrix values and
  32-bit offsets from DRAM; four processors per core (reader, writer, and the
  pack and unpack threads) gather `x[col]` from the window; the SFPU sums the
  27 products in HPCG's column order.
- **One launch per smoother.** All 16 colour phases of a symmetric
  Gauss-Seidel run in one launch, with per-neighbour semaphores between
  phases. The pre-smoother also runs the residual SpMV. SpMV passes keep the
  same per-phase wait, so cores stream DRAM in lockstep.
- **Fused vector work.** Dot products, `p = z + beta p` and the `x`, `r`
  update with `|r|^2` are one pass each. Each core reduces its share of a dot
  product to one number on the SFPU; a one-core launch adds the 130 numbers
  (and takes the square root for `|r|`), so no TT-NN op runs in the iteration.
