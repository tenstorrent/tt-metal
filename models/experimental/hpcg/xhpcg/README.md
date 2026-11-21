# xhpcg: HPCG 3.1's own program on the P150

This folder builds HPCG 3.1's own benchmark program (`xhpcg`) with the P150 solver plugged into the functions HPCG
lets a port replace. HPCG's driver (`main.cpp`), its problem setup, its reference timing phases, its validation tests
and its results file all run unchanged, from HPCG's release source, downloaded at build time.

## What is replaced

HPCG ships each replaceable function as a pair: `X_ref.cpp` (reference, never changed) and `X.cpp` (what
`main.cpp` calls; by default it calls the reference). This folder's `src/` holds new versions of these `X.cpp`
files; every other HPCG file is HPCG's own.

| HPCG file | Here |
|---|---|
| `OptimizeProblem.cpp` | Opens the card, plans each core's rows, x windows and neighbour exchanges, has the card write its copy of the matrix (all four levels, the grid transfer tables, b), and builds the launches. Timed by HPCG as its optimization phase. |
| `CG.cpp` | HPCG's CG algorithm on the card: the CG start and each iteration are one recorded trace each; alpha and beta stay on the card. Before each solve it checks whether the host changed x, b or the diagonal (HPCG's TestCG does) and updates the card's copy; it copies x back at the end. |
| `ComputeSPMV.cpp`, `ComputeMG.cpp`, `ComputeSYMGS.cpp`, `ComputeDotProduct.cpp` | The same launches on host vectors (copied to the card and back), for HPCG's TestSymmetry. |
| `ComputeWAXPBY.cpp` | The reference version: the CG on the card fuses its vector updates and never calls it. |

`tt/` is the P150 implementation in C++ on TT-NN: the same kernels as the Python package
(`tt/custom_ops/*/kernels`), the same launch descriptions, and the package's compiled planners
(`host/native_setup.cpp`), linked in.

## Build and run

On the P150 host, with TT-Metal built (`build_metal.sh`):

```
cmake -S models/experimental/hpcg/xhpcg -B build_xhpcg -G Ninja -DCMAKE_CXX_COMPILER=clang++-20 \
      "-DCMAKE_PREFIX_PATH=$TT_METAL_HOME/build/lib/cmake;$TT_METAL_HOME/build/share/cmake"
cmake --build build_xhpcg -j
TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME TT_METAL_INSPECTOR=0 build_xhpcg/xhpcg --nx=104 --rt=60
```

The grid is `--nx` points per side (`--ny` and `--nz` default to it); `--rt` is the timed phase's length in seconds
(an official run needs 1,800). `TT_METAL_RUNTIME_ROOT` tells TT-Metal where this repository is (Python's `ttnn` sets it
itself): the kernels are read by their paths in it. HPCG writes its results file (`HPCG-Benchmark_3.1_*.txt`) and its
log (`hpcg*.txt`) in the current directory.

## Limits

- **fp32.** The card computes in fp32 (its compute cores have no fp64); HPCG's rules require fp64.
- **TestCG fails.** Without the preconditioner, CG needs 17 iterations at 104³ and 16 at 192³; HPCG allows 12
  (with the preconditioner it needs 2, as allowed). This is why the report says INVALID. TestSymmetry passes
  (departure 0 at 104³, 0.079 at 192³). The cause is not yet confirmed; fp32 rounding is the likely one.
- **One card, cubic grids.** One process; nx = ny = nz, a multiple of 8.
- **Per-kernel times.** HPCG's report splits the CG time into DDOT, WAXPBY, SpMV and MG. The iteration runs as one
  trace, so only the total is measured; the split entries stay 0.
- **What setup costs here.** HPCG's own problem setup runs on the host as shipped (its `GenerateProblem` builds the
  host matrix that its checks and reference phases read), and its time is charged. The optimization phase includes
  opening the card.
