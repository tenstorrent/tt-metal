# HPCG on Blackhole P150 (fp32)

[HPCG](https://github.com/hpcg-benchmark/hpcg) is a supercomputer benchmark. It
solves one large sparse linear system, `A·x = b`, on a 3-D grid: each grid point's
equation involves itself and its 26 neighbours. The solver is conjugate gradient
(CG), helped by a multigrid step that also works on 3 smaller copies of the grid.
The score is the floating-point rate of 50 solver iterations.

This folder runs HPCG 3.1's solver on one P150 card, with custom kernels for the
matrix work. It follows HPCG's rules for how the matrix is stored and used, but
runs in fp32: the rules require fp64, which the P150's compute cores do not have.
So this is not an official HPCG result (see [Rules](#rules)).

## Run

From `$TT_METAL_HOME`:

```
pytest models/experimental/hpcg/tests                  # correctness, about 20 s
python models/experimental/hpcg/demo/demo.py 192       # benchmark the 192³ grid
```

The demo prints the time per iteration, GFLOP/s, the share of the speed of
light reached, and the residual after each iteration. The grid size must be a
multiple of 8; HPCG's default is 104, and 304 is the largest that fits.

## Results

One P150a: 130 cores, 512 GB/s of DRAM bandwidth.

| Grid | ms per iteration | GFLOP/s | Speed of light | Share reached |
| --- | ---: | ---: | ---: | ---: |
| 104³ | 4.88 | 84.5 | 3.21 ms | 65.9% |
| 128³ | 7.69 | 100.3 | 6.01 ms | 78.1% |
| 192³ | 23.68 | 110.6 | 21.13 ms | 89.2% |
| 240³ | 45.49 | 112.7 | 41.35 ms | 90.9% |
| 288³ | 77.84 | 114.0 | 71.56 ms | 91.9% |
| 304³ | 91.53 | 114.0 | 84.19 ms | 92.0% |

The speed of light is the fastest an iteration could run: the least DRAM
traffic each operation needs under HPCG's rules, divided by 512 GB/s. Every
operation is limited by memory traffic, not arithmetic. When all 130 cores read
DRAM at once, the card delivers at most about 490 GB/s (95.7%), and the large
operations already run close to that rate. Small grids sit further from the
limit because their coarsest copies are too small to keep the cores busy.

**HPCG's own score** is lower than the raw GFLOP/s, for two reasons. Our
colouring of the grid (needed to run in parallel) takes 51 to 53 iterations to
reach the accuracy HPCG's reference code reaches in 50, and HPCG credits only 50.
HPCG also charges a tenth of the setup time to every 50 iterations; setup here
takes 0.2 s at 104³ and 0.6 s at 192³. Scored this way, the P150 rates 77.9,
89.4 and 99.7 GFLOP/s at 104³, 128³ and 192³. HPCG's fp64 reference code on a
24-thread Ryzen 9 7900X rates 3.83, 4.02 and 4.40 GFLOP/s, so the P150 is 20 to
23 times faster.

## How it works

One iteration runs about 30 operations. All of them run as custom kernels on
the card; the whole iteration is captured once and replayed, so the host does
nothing while it runs.

- **Matrix-vector products and smoothing.** Each core owns a block of grid
  points. It streams its rows' matrix values and column positions from DRAM,
  and reads the vector it multiplies at those column positions from a copy kept
  in its own on-chip memory. Four of each core's five processors share these
  reads.
- **Parallel smoothing.** The smoother (symmetric Gauss-Seidel) updates each
  point from its neighbours' newest values. The points are split into 8
  groups, by whether each coordinate is even or odd, so no two neighbours share
  a group; each group updates in parallel. All 16 steps of a smoother (8 groups
  forward, 8 back) run in one launch, and after each step a core sends its
  updated values straight to the neighbouring cores that need them.
- **Vector operations.** Dot products and vector updates are single passes over
  the vectors. Each core reduces its share of a dot product to one number, and
  one small launch adds the 130 numbers.
- **Setup.** Building the matrix and each core's tables on the host is C++
  (`host/native_setup.cpp`), compiled automatically the first time it runs.

## Rules

Checked against HPCG 3.1's source and its
[FAQ](https://www.hpcg-benchmark.org/faq/index.html).

Followed: the matrix is treated as unstructured (each non-zero stores its value
and its column position, and the vector is read through that position); every
term is used in every pass; symmetry is not used; the algorithm is the
reference one (4 grid levels, one smoothing before and after each level); the
8-group reordering is allowed, and its extra iterations are deducted in the
score above.

Not followed: fp64 (everything is fp32); the 30-minute run length (these runs
time 50 iterations); the problem size of at least a quarter of memory (304³
uses 22% of the card's 32 GB); and HPCG's own validation tests, one of which
measures errors in units of fp64 rounding and fails for any fp32 solver.

## What is in this folder

    demo/
        demo.py              the benchmark (above)
        op_perf.py           time of each operation against its speed of light
        kernel_counters.py   per-core cycle counts inside one launch, for profiling
    tests/
        test_hpcg.py         every device operation against numpy, then CG against fp64
        test_host_setup.py   the C++ setup builds exactly what the numpy code builds
    reference/
        problem.py           the HPCG problem, and a numpy solver that does the same steps
        speed_of_light.py    the speed of light of each operation
    tt/
        hpcg.py              the solver: builds every launch and runs one iteration
        layout.py            moving vectors between host and card
        custom_ops/          the kernels (see custom_ops/README.md)
    host/
        native_setup.cpp     host setup in C++, and native.py, which compiles and loads it
