# Audio decode — implementation plan for the remaining 5-10x

Everything here is sized from measurements in this branch. Each script that produced a number is
named, so any figure can be re-derived rather than trusted. `AUDIO_RESULTS.md` covers what is already
achieved (~1.3x); this covers what is not.

## The two facts the design must respect

**1. Cost is per-row, not per-byte, and rows are almost free to widen.** (`row_cost.py`)

Holding total elements constant at the s6 tail size and varying only the channel width:

| C | rows | fp32 ms | ns/row | GB/s | vs C=8 |
|---|---|---|---|---|---|
| 8 | 662 424 | 2.765 | 4.2 | 23.0 | 1.00x |
| 16 | 331 212 | 1.409 | 4.3 | 45.1 | 1.96x |
| 32 | 165 606 | 0.751 | 4.5 | 84.7 | 3.68x |
| 64 | 82 803 | 0.437 | 5.3 | 145.4 | 6.32x |
| 128 | 41 401 | 0.293 | 7.1 | 216.7 | 9.42x |
| 256 | 20 700 | 0.235 | 11.3 | 271.0 | 11.78x |

A row costs ~4.2 ns whatever it contains. The decode's tail runs at C=8/16/32, so it operates at
23-85 GB/s on a part that reaches **271 GB/s** when rows are wide. bf16 does not change the row cost
at all (2.772 ms at C=8, versus fp32's 2.765 ms) -- it only helps by halving bytes once rows are wide
enough for bytes to matter.

**2. Fixed per-op cost is ~142 us as the decode issues them.** (`op_floor.py`, `op_pipeline.py`)

~90 us perfectly pipelined, ~142 us chained (which is what the decode does), flat in tensor size below
~0.4 MB and unaffected by how many chips the cluster opens. 6955 ops x 142 us = 988 ms against a
~1170 ms wall.

**These multiply.** Op-count reduction attacks term 2; row-widening attacks term 1. Doing only one
leaves most of the factor on the table.

## Where the ops are (`band_ops.py`, over the committed Tracy capture)

127 bands hold **6862 of 6955 ops (99 %)**, median 53 ops per band, of which ~8 are convolution:

    Halo / Move / Conv2d               6.8 each
    Slice / PaddedSlice / SliceWrite   5.2-5.4 each
    UntilizeWithUnpadding              4.8
    Concat                             3.7
    ReshapeView / I2S / S2I            1.7-2.3
    Conv3d                             1.0

About 45 of every 53 ops are scaffolding around 8 real convolutions.

## Plan, in dependency order

**Step 1 — widen rows through the tail. No new kernel.**

The snake already folds timesteps into channels and is bit-exact doing it (`SnakeBeta._fold_factor`),
but it caps the fold at one tile width, C=32, taking 3.68x where 11.78x is available. Two changes:

* raise the cap so C reaches 128-256 rather than 32;
* pad T to a multiple of the fold factor. T=165606 factors as 2 x 3 x 27601, so it admits a fold of
  only 2 today; padding to 165632 admits 32. The pad is ~26 rows against a 12x speedup, but it costs
  an op, so it only pays if the folded layout is **held across consecutive ops** rather than folded and
  unfolded around each one.

Applies to every per-channel/elementwise op in the tail: the snake, the residual adds, the scale
multiplies. It does **not** apply to the FIRs as-is -- convolution along T does not survive folding T
into C without a polyphase rewrite.

Expected: the tail's elementwise work drops toward 1/4 to 1/10. Verify with `snake_bench.py`.

**Step 2 — fuse the band.** The real kernel.

One op per `Activation1d` band, implementing the decomposition already proven exact in
`audio_resample.py::Activation1d._forward_fused` (rel_rmse 8.5e-08 against the literal form,
`verify_fusion.py` checks the index algebra on CPU first). The trap it documents: replicate padding
does **not** decompose into per-phase replicate padding, because the pad region is a constant whose
parity alternates.

Sized at the measured 142 us/op:

| target | ops | saved | total |
|---|---|---|---|
| 53 -> 5 per band | 6862 -> 635 | 6227 | ~285 ms (4.1x) |
| 53 -> 1 per band | 6862 -> 127 | 6735 | ~215 ms (5.4x) |

Two design constraints from measurement, both of which would otherwise bite mid-implementation:

* **In-model L1 contention is real.** Shapes that take L1_FULL standalone are rejected inside a decode
  (`verify` mode: 1 of 42). Budget L1 against what is free mid-decode, not against a microbenchmark.
* **Prefer fewer, larger invocations.** Per-op cost is flat in size below ~0.4 MB, so batching the 3
  parallel resblock branches along the channel axis is nearly free -- and by fact 1 it also widens
  rows, so it wins twice.

**Step 3 — bf16.** Already runnable (`14572d860ce`), 0.959 s and 34.93 dB against a 28 dB gate. Worth
taking *after* steps 1-2, because bf16's benefit is halved bytes and bytes only become the binding
term once rows are wide.

## What not to do

Nine levers closed by measurement, listed with their numbers in `AUDIO_KERNELS_BENCH.md`: trace,
conv1d L1_FULL, operand splitting, algebraic band fusion, L1-sharded intermediates, cheaper padding
primitives, conv3d UnpackToDestFp32, the 32-chip hypothesis, and act_block_h tuning. They fail for one
of two reasons, both now understood: they optimise bytes (which barely register) or they rearrange ops
without removing them.

## Honest risk on the projection

The 285/215/130 ms figures are arithmetic: measured op counts times measured per-op cost. They assume
a fused op still costs ~142 us while doing far more compute per invocation. Total arithmetic in the
decode is ~20-40 GFLOP against a machine that should do that in single-digit ms, so compute should
stay negligible -- but that is an estimate, not a measurement, and it is the first thing to check with
a throwaway prototype before committing to the full op.
