# HPCG on Blackhole P150 (fp32)

The [HPCG](https://github.com/hpcg-benchmark/hpcg) 3.1 conjugate-gradient solve,
with its 4-level multigrid preconditioner, on one P150 card. The matrix is
stored the way the HPCG rules require: every term of the 27-point stencil, an
fp32 value and a 32-bit column index per non-zero, read through the column
index (no stencil structure is exploited).

**This is not an official HPCG result.** The rules require IEEE fp64 in every
phase, and Tensix cores have no fp64 arithmetic, so this port runs in fp32.
Within 50 iterations its residuals match an fp64 run of the same algorithm to
4-6 digits.

## Run

From `$TT_METAL_HOME`:

| Command | What it does |
| --- | --- |
| `pytest models/experimental/hpcg/tests/test_hpcg.py` | every device op against numpy, then 3 CG iterations against fp64 (64³ and 160³) |
| `python models/experimental/hpcg/demo/demo.py 192` | 50 traced CG iterations: ms per iteration, GFLOP/s, share of speed of light, residual history |
| `python models/experimental/hpcg/demo/op_perf.py 192` | device time of each op against its speed of light |
| `python models/experimental/hpcg/demo/kernel_counters.py 192 gs0 [--per-core]` | per-core cycle counters of one launch |
| `python -m models.experimental.hpcg.reference.speed_of_light 192 23.75` | speed of light of one iteration, and the share a measured time reaches |
| `python -m models.experimental.hpcg.reference.problem 104 64` | the numpy reference solve (fp64 or fp32) |

The grid size nx must be a multiple of 8 (HPCG's default is 104); 304 is the largest that fits L1.

## Results (P150a, 130 cores, 512 GB/s GDDR6)

| Grid | ms / iteration | GFLOP/s | Speed of light | Share |
| --- | ---: | ---: | ---: | ---: |
| 104³ | 4.95 | 83.3 | 3.21 ms | 65% |
| 128³ | 7.76 | 99.4 | 6.01 ms | 77.5% |
| 192³ | 23.75 | 110.2 | 21.13 ms | 89.0% |
| 240³ | 45.54 | 112.5 | 41.35 ms | 90.8% |
| 288³ | 77.90 | 113.9 | 71.56 ms | 91.9% |
| 304³ | 91.60 | 113.9 | 84.19 ms | 91.9% |

GFLOP/s uses HPCG's own flop count. The speed of light
(`reference/speed_of_light.py`) is each operation's minimum DRAM traffic under
the HPCG rules at 512 GB/s; every operation is memory-bound. The fastest rate
measured with all 130 cores reading DRAM is about 490 GB/s (95.7%), and at
288³ the fine-grid smoothers move their actual bytes at 98.6-99.9% of it.
For comparison, the fp64 HPCG reference code on a 24-thread Ryzen 9 7900X runs
the 192³ problem at 4.40 GFLOP/s.

## Layout

    reference/
        problem.py           HPCG problem in device storage order + numpy reference solver
        speed_of_light.py    per-op roofline of one CG iteration
    tt/
        hpcg.py              HPCG: device levels, ops, one CG iteration (trace-capturable)
        layout.py            host <-> device vector layout
        custom_ops/          ttnn.generic_op kernels (see custom_ops/README.md)
            sparse_rows/     smoothers, SpMV, restriction, prolongation
            cg_vector/       fused dot products and vector updates
    tests/test_hpcg.py
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
  update with `|r|^2` are one pass each; TT-NN only reduces the per-core
  partial sums and takes the square root.
