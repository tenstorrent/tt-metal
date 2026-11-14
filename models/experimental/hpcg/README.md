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

Two drivers run the same card kernels:

- **The Python demo driver** (`demo/demo.py`) is ours. It runs HPCG's CG and
  multigrid on the card and scores them with HPCG's flop counts and rating
  formula. It does not run HPCG's own program, its validation tests or its
  results file.
- **HPCG's own driver** (`xhpcg/`) is HPCG 3.1's C++ program, downloaded from
  its release at build time. The functions HPCG lets a port replace are
  replaced with C++ versions that call the card kernels. HPCG's own problem
  setup, validation tests and results file run unchanged.

## Run the Python demo driver

From `$TT_METAL_HOME`, in TT-Metal's Python environment:

```
pytest models/experimental/hpcg/tests                  # correctness, about 20 s
python models/experimental/hpcg/demo/demo.py 192       # benchmark the 192³ grid
```

The tests check every card operation against a numpy copy of the same
algorithm, then a multigrid cycle and 3 CG iterations against fp64
(`tests/test_hpcg.py`). They also check that the C++ host setup plans exactly
what the numpy definitions plan (`tests/test_host_setup.py`, host only).

`demo.py nx [iterations]` prints one JSON object per line: the setup time, the
time per iteration, GFLOP/s, the share of the speed of light reached, and the
residual after each iteration. At 104³, 128³ and 192³ it also prints HPCG's
rating, because it stores HPCG's fp64 reference residual for those sizes only.
The grid size must be a multiple of 8; HPCG's default is 104, and 304 is the
largest that fits.

## Run HPCG's own driver

From `$TT_METAL_HOME`, with TT-Metal built (`./build_metal.sh`), clang++-20
and Ninja:

```
cmake -S models/experimental/hpcg/xhpcg -B build_xhpcg -G Ninja -DCMAKE_CXX_COMPILER=clang++-20 \
      "-DCMAKE_PREFIX_PATH=$TT_METAL_HOME/build/lib/cmake;$TT_METAL_HOME/build/share/cmake"
cmake --build build_xhpcg -j
TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME TT_METAL_INSPECTOR=0 build_xhpcg/xhpcg --nx=104 --rt=60
```

- The configure step downloads HPCG 3.1's source. On a machine without
  internet access, unpack `HPCG-release-3-1-0.tar.gz` and add
  `-DFETCHCONTENT_SOURCE_DIR_HPCG=<that folder>`.
- `--nx` is the grid side; `--ny` and `--nz` default to it. The grid must be
  cubic with a side that is a multiple of 8.
- `--rt` is the timed phase's length in seconds. An official run needs 1,800.
- `TT_METAL_RUNTIME_ROOT` tells TT-Metal where this repository is, because the
  kernels are read by their paths in it. (Python's `ttnn` sets this itself.)
- HPCG writes its results file (`HPCG-Benchmark_3.1_*.txt`) and its log
  (`hpcg*.txt`) in the current directory.

With 30 s timed, HPCG rates 67.34 GFLOP/s at 104³ and 87.24 at 192³. It
reports INVALID because of one validation test, TestCG: without the
preconditioner our CG needs 17 iterations at 104³ and 16 at 192³, and HPCG
allows 12. With the preconditioner it needs 2, as allowed. TestSymmetry and the
other checks pass. The rating is below the demo's mainly because HPCG's own
problem setup runs on the host (0.39 s at 104³, 1.95 s at 192³), and HPCG
charges a tenth of it to every CG set. The report's per-kernel times (DDOT,
WAXPBY, SpMV, MG) show 0: each CG iteration runs as one trace, so only the
total is timed. More detail is in `xhpcg/README.md`.

## Results (Python demo driver)

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
takes 0.008 s at 104³ and 128³ and 0.010 s at 192³, which lowers the score by
0.1 to 0.3%. Each timed set also includes the CG start (r = b - A x and its
norm), as HPCG times it. Scored this way, the P150 rates 82.6, 94.4 and 104.2
GFLOP/s at 104³, 128³ and 192³. HPCG's fp64
reference code on a 24-thread Ryzen 9 7900X rates 3.83, 4.02 and 4.40 GFLOP/s,
so the P150 is 21 to 24 times faster.

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
- **Setup.** The card writes the matrix itself: each core fills its rows of
  matrix values and column offsets straight into DRAM, and the same for the
  tables of the transfers between grids, so nothing large crosses from the
  host. The host only plans which rows and pieces of x each core handles, in
  C++ (`host/native_setup.cpp`, compiled automatically the first time it
  runs).

## Host time

Measured at 192³ on the P150's host (Ryzen 9 7900X).

- **During the timed solve: none.** The host queues one trace replay for the
  CG start and one per iteration (0.1 ms for 51 replays) and reads nothing
  back; a 50-iteration set measures 50 × 23.65 ms + 3.95 ms for the start.
- **Setup, charged by HPCG's score (a tenth per 50-iteration set): about
  10 ms**, the first time in a process (7.7 ms when repeated). The card and
  the host work at the same time; at 192³ the host is done after about 9 ms
  and the card about 1 ms later (at 104³ the host finishes last, at 8 ms):
  - the card writes the matrix in two launches, 6.8 ms together: the fine grid
    (1.6 GB of values and column offsets), then the three coarser grids and the
    six grid-transfer tables. Its rate is set by its write traffic: a few cores
    in the top rows wait on congested on-chip network routes to DRAM;
  - a second host thread (started at import) loads the matrix-writing kernel
    and builds its program while the host plans the fine grid (about 3 ms in),
    then enqueues both launches, so the fine grid's launch starts as soon as
    the kernel is loaded. Loading the kernel stays inside the timed setup;
  - the host plans which rows and pieces of x each core handles and which
    updated rows it sends to which neighbour (about 2.5 ms, mostly C++), builds
    the launch descriptions and the small per-core tables the kernels read
    (about 3.5 ms: about 150 µs per sparse-row launch, each table uploaded at
    its rows' used width), and allocates the vectors.
- **Once per process, not charged:** `import ttnn` (1.5 s), opening the card
  (0.2 s), loading the compiled kernels on first use (about 0.02 s) and
  recording the two traces (3 ms). These belong to TT-NN and the runtime, like
  loading a program for a CPU run.
- **Checking results, outside the benchmark:** reading the residual after
  each iteration (only in the run that records the history, 0.07 ms each) and
  reading x once at the end (0.07 s).

## Rules

Checked against HPCG 3.1's source and its
[FAQ](https://www.hpcg-benchmark.org/faq/index.html).

Followed: the matrix is treated as unstructured (each non-zero stores its value
and its column position, and the vector is read through that position); every
term is used in every pass; symmetry is not used; the algorithm is the
reference one (4 grid levels, one smoothing before and after each level); the
8-group reordering is allowed, and its extra iterations are deducted in the
score above.

Not followed: fp64 (everything is fp32); the 30-minute run length (the demo
times 50 iterations); and the problem size of at least a quarter of memory
(304³ uses 22% of the card's 32 GB). The Python demo driver does not run HPCG's
validation tests. HPCG's own driver runs them: TestSymmetry passes, and TestCG
fails its unpreconditioned check (16 to 17 iterations, 12 allowed).

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
    xhpcg/
        CMakeLists.txt       builds HPCG 3.1's own program with the replacements below
        src/                 HPCG's replaceable functions (CG.cpp, OptimizeProblem.cpp, ...)
        tt/                  the solver in C++ on TT-NN, calling the same kernels
