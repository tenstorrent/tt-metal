# Wormhole EMA kernel: −23.2 % MATH_ISOLATE cycles/tile

Change measured: `_compute_ema_math_` in
`tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_ema.h`.

| | main | this branch | Δ | |
|---|---|---|---|---|
| **MATH_ISOLATE** cycles/tile | 320.77 | **246.18** | **−74.59** | **−23.25 %** |
| L1_TO_L1 cycles/tile | 338.64 | 272.67 | −65.97 | −19.48 % |
| Math ELF `TEXT_SIZE` | 3015 | 2823 | −192 B | −6.37 % |
| Issue slots, math block | 17 | 11 | −6 | −35.3 % |
| Output bits | — | **unchanged** | — | 0 / 172032 differ |

Measured on Wormhole n300, `--speed-of-light`, `loop_factor=16`, `tile_cnt=8`,
`Float16_b→Float16_b`, `dest_acc=No`. Both configurations were run twice; MATH_ISOLATE
reproduced to the last decimal each time (see §4).

---

## 1. What changed

`EMA_new = alpha * EMA_old + beta * input`, chained across 4 rows per block.

**Before** — two MADs per row, both on the dependency chain, so each needs an `SFPNOP`
behind it (2-cycle `SFPMAD` write latency):

```
LREG7 = alpha * LREG4          ; SFPNOP    (carry in from previous block)
LREG0 = beta * LREG0 + LREG7   ; SFPNOP
LREG7 = alpha * LREG0          ; SFPNOP
LREG1 = beta * LREG1 + LREG7   ; SFPNOP
... x4 rows
SFPMOV LREG3 -> LREG4          (carry out)
```

**After** — scale the inputs by `beta` up front, leaving one fused MAD per row on the
chain. The four scaling multiplies are mutually independent, so three of them deal into
the chain's latency slots instead of stalling behind it:

```
LREG0 = beta * LREG0                    (pre-scale in0)
LREG1 = beta * LREG1                    (pre-scale in1)
LREG0 = alpha * LREG4 + LREG0           (row 0)
LREG2 = beta * LREG2                    (pre-scale in2 — covers row 0's latency)
LREG1 = alpha * LREG0 + LREG1           (row 1)
LREG3 = beta * LREG3                    (pre-scale in3 — covers row 1's latency)
LREG2 = alpha * LREG1 + LREG2           (row 2)
SFPNOP
LREG3 = alpha * LREG2 + LREG3           (row 3)
SFPNOP
SFPMOV LREG3 -> LREG4                   (carry out)
```

Same eight MADs. `LREG7` is no longer needed as a temp; no other register pressure change
(LREG0-3 rows, LREG4 carry, LREG5/6 alpha/beta).

## 2. Reading the numbers

**The win is larger than the slot count predicts, and that is the point.** A tile is 32
rows = 8 blocks, and each block drops 6 NOPs, so pure issue-slot accounting predicts
−48 cycles/tile. The measurement is **−74.59**. The extra ~27 cycles come from the second
half of the change: the per-row critical path drops from two dependent MADs to one, so on a
latency-bound kernel the chain shortens as well as the slot count. That is consistent with
the premise this change was built on — these SFPU kernels are latency-bound, not
issue-bound.

`L1_TO_L1` improves by less (−19.48 % vs −23.25 %) exactly as expected: it includes unpack
and pack, which are untouched, so a math-only change is diluted there. `MATH_ISOLATE` is
the number that attributes to the kernel.

`TEXT_SIZE(MATH_ISOLATE)` drops 192 bytes, consistent with 6 fewer instructions × 8
unrolled blocks × 4 bytes.

## 3. Correctness

The reassociation is not bit-neutral in fp32 (the old form rounded `alpha*prev` alone and
fused `beta*input` into the add; the new one rounds `beta*input` alone and fuses
`alpha*prev`, so results can differ by ~2^-24 relative). It **is** bit-neutral at the
output, because DEST for this kernel is bfloat16, whose resolution is 2^-9 — three orders
coarser than the perturbation.

Verified by dumping raw output bits before and after and diffing:

| axis | values |
|---|---|
| seeds | 0-7 |
| input amplitude | 0.25, 4.0, 64.0 |
| tile counts | 1, 2, 4 |
| outputs compared | **172032** |
| **differing** | **0** |

`test_sfpu_ema.py`: 3 passed.

Wormhole only. The Blackhole copy has no NOPs to remove (BH interlocks) and both forms
issue the same eight MADs, so there is nothing to win there.

## 4. Raw runs

`TILE_LOOP` row of `perf_data/perf_sfpu_ema/perf_sfpu_ema.post.csv`, which is
cycles/tile (`.post.csv` divides the `TILE_LOOP` means by `loop_factor * tile_cnt`).

| run | kernel | mean(MATH_ISOLATE) | mean(L1_TO_L1) | TEXT_SIZE(MATH_ISOLATE) |
|---|---|---|---|---|
| 1 | main | 320.7734375 | 338.640625 | 3015 |
| 2 | main | 320.7734375 | 338.65625 | 3015 |
| 1 | branch | 246.1796875 | 272.671875 | 2823 |
| 2 | branch | 246.1796875 | 272.671875 | 2823 |

MATH_ISOLATE is identical across repeats in both configurations. The only movement anywhere
is 0.0156 cycles on one `L1_TO_L1` baseline sample — four orders below the effect.

`perf_data/` is gitignored, so these CSVs are not in the repo; the commands in §5 regenerate
them.

## 5. Reproducing

```bash
cd tt_metal/tt-llk/tests
rm -rf ../perf_data/perf_sfpu_ema          # so a no-op rerun is visibly empty
CHIP_ARCH=wormhole pytest -q --speed-of-light --compile-producer -m perf ./python_tests/perf_sfpu_ema.py
CHIP_ARCH=wormhole pytest -q --speed-of-light --compile-consumer -m perf ./python_tests/perf_sfpu_ema.py
awk -F, '$21=="TILE_LOOP"' ../perf_data/perf_sfpu_ema/perf_sfpu_ema.post.csv
```

Note the output lands in `tt_metal/tt-llk/perf_data/`, **not** `tests/perf_data/`.

| | |
|---|---|
| Branch | `ldjurovic/sfpu_wh_nop_overlap` |
| Base | `f6b36f3b1be` |
| Silicon | Wormhole n300, `CHIP_ARCH=wormhole` |
| Test | `tests/python_tests/perf_sfpu_ema.py` (module `perf_sfpu_ema`) |
| Source | `tests/sources/sfpu_ema_perf.cpp` |
| Run types | `MATH_ISOLATE`, `L1_TO_L1`; markers `INIT`, `TILE_LOOP`, `KERNEL` |
| Rows | 3 (1 variant × 3 markers) |

## 6. Notes on the harness

The EMA kernel had no perf coverage, so this adds it. Two things cost real time and are
worth knowing:

- **`MATH_ISOLATE` must keep the datacopy.** It is what consumes the SrcA valid bits unpack
  sets; dropping it and trying to retire them with a bare `TTI_CLEARDVALID` hangs the math
  thread. `eltwise_unary_sfpu_perf.cpp` is the reference shape. The datacopy is therefore a
  fixed cost inside the marker for every op measured this way — constant across a
  before/after comparison, so it cancels in the delta, but it means the absolute number is
  not the SFPU block alone.
- **The unpack valid count must match what math retires.** With the datacopy in, that is
  `num_faces * TILE_CNT * LOOP_FACTOR`. A mismatch in either direction hangs the handshake,
  and each hang needs `tt-smi -r` before the next attempt means anything.

`llk_math_ema_sfpu_tile` also brackets itself with
`_llk_math_eltwise_sfpu_start_`/`_done_`, so it must not be wrapped in another pair.

**Why this is a dedicated test rather than wired into `test_sfpu_unary.py` /
`perf_eltwise_unary_sfpu.py`:** EMA does not fit the unary op contract. It is stateful
(carries EMA_old in LREG4 across tiles), it reads dst tile 0 and writes dst tile **1** via
compile-time offsets rather than operating in place, it is bfloat16-DEST only, and its
golden is a sequential recurrence down rows rather than an element-wise function.
Registering it as a `MathOperation` would enter it into format and dest_acc sweeps it cannot
satisfy and into an element-wise golden framework that cannot express it. The perf source
does reuse the unary harness's `PerfRunType` structure, which is where the value was.
