# Blackhole FPU Math Kernel Measurements

Before/after numbers, correctness evidence and reproduction commands for the changes described in
[blackhole_fpu_math_kernel_audit.md](blackhole_fpu_math_kernel_audit.md).

## Run Metadata

| | |
|---|---|
| Board | Blackhole **p300a** (not p100a — do not compare across boards) |
| Baseline commit | `d53cb57e206` (tt-metal `main`) |
| Metric | `mean(MATH_ISOLATE)`, row `marker == "TILE_LOOP"` — cycles per tile on the math thread |
| Formats | `Float16_b -> Float16_b`, `dest_acc` off |
| Build | `--speed-of-light` (compile-time constants, no runtime params) |
| Loop shape | `LOOP_FACTOR = 64`, `TILE_CNT = 16` |
| Tile shape | 32x32, 4 faces |

## What Changed

| # | Change | File | Kind |
|---|---|---|---|
| F1 | Hoist the Src zero-substitution flag out of the per-face-row transpose into `_llk_math_reduce_init_`, for `REDUCE_ROW` + `MAX`; restore the baseline in `_llk_math_reduce_uninit_` | [llk_math_reduce.h](../../tt_llk_blackhole/llk_lib/llk_math_reduce.h) | perf |
| F2 | Track `ALU_ACC_CTRL_INT8_math_enabled` and skip the write when the bit does not move | [cmath_common.h](../../tt_llk_blackhole/common/inc/cmath_common.h), [llk_math_common.h](../../tt_llk_blackhole/llk_lib/llk_math_common.h) | perf |
| F6 | Make `replay_buf_len` mirror the recording lambda's branch structure | [llk_math_matmul.h](../../tt_llk_blackhole/llk_lib/llk_math_matmul.h) | correctness |
| — | Take `MATH_FIDELITY` from the harness instead of hard-coding HiFi4; sweep `[LoFi, HiFi4]` for SUM/AVG | [reduce_perf.cpp](../../tests/sources/reduce_perf.cpp), [perf_reduce.py](../../tests/python_tests/perf_reduce.py) | instrumentation |

## Results

Cycles per tile, math thread. Lower is better. Every row that is not F1 is a **control** and is
expected to be flat — they are listed because a flat control is the evidence that nothing else moved.

| Kernel | Fidelity | Before | After | Δ | Δ% | Unpack bound |
|---|---|---:|---:|---:|---:|---:|
| **`reduce` row · max** | any | **97.94** | **58.06** | **−39.88** | **−40.7%** | 37.28 |
| `reduce` row · sum | LoFi | 12.08 | 12.08 | 0.00 | — | 37.28 |
| `reduce` row · sum | HiFi4 | 49.07 | 49.07 | 0.00 | — | 37.28 |
| `reduce` col · max | any | 16.07 | 16.07 | 0.00 | — | 37.28 |
| `reduce` col · sum | LoFi | 16.10 | 16.10 | 0.00 | — | 37.27 |
| `reduce` col · sum | HiFi4 | 83.07 | 83.07 | 0.00 | — | 37.28 |
| `reduce` scalar · max | any | 41.07 | 41.07 | 0.00 | — | 48.09 |
| `reduce` scalar · sum | LoFi | 41.10 | 41.10 | 0.00 | — | 48.10 |
| `reduce` scalar · sum | HiFi4 | 130.07 | 130.07 | 0.00 | — | 48.09 |
| `matmul` 32x32x32 | LoFi | 19.20 | 19.20 | 0.00 | — | 33.25 |
| `matmul` 32x32x32 | HiFi4 | 68.33 | 68.33 | 0.00 | — | 33.25 |
| `eltwise` add | LoFi | 16.62 | 16.62 | 0.00 | — | 37.91 |
| `eltwise` mul | LoFi | 16.62 | 16.62 | 0.00 | — | 37.91 |
| `eltwise` mul | HiFi4 | 82.52 | 82.52 | 0.00 | — | 37.91 |

REDUCE_ROW MAX is fidelity-independent (GMPOOL only), so the saving applies at every fidelity and to
every format that reaches this path.

### F1 in context

The kernel goes from **2.6x its unpack bound to 1.6x**. It is still the one reduce variant where math
is the limiter, so the remaining 58.06 cycles are the next target: roughly 16 for the four GMPOOLs and
~40 for the two mov/transpose sequences (18 instructions at ~2.2 cycles each). Halving that by landing
both face rows in SrcB before a single `TRNSPSRCB` would bring the kernel to about the bound — see F1's
residual notes in the audit.

The 39.88 cycles came from four writes, so **one cfg write plus its pipe drain costs 9.97 cycles here**.

### F2 — reconfig cost

No in-tree perf harness exercises `_llk_math_reconfig_data_format_*` on the math thread, so this was
measured with a **temporary probe**: one `_llk_math_reconfig_data_format_<is_fp32_dest_acc_en>(formats.math, formats.math)`
inserted into the MATH_ISOLATE tile loop of `sources/eltwise_binary_fpu_perf.cpp`. Same format on both
sides, so the INT8 bit never moves — exactly the case the cache is for. The probe is **not committed**.

| Variant | cycles/tile | Cost attributable to the reconfig |
|---|---:|---:|
| No reconfig (reference) | 16.62 | — |
| Reconfig per tile, before F2 | 23.59 | 6.97 |
| Reconfig per tile, after F2 | 20.83 | **4.21** |

**Δ = −2.76 cycles per reconfig (−39.6% of the per-reconfig cost).**

The absolute saving is context-dependent, because what is removed is a pipe drain: 2.76 cycles behind
an 8-instruction eltwise body, 9.97 cycles behind a dense reduce body (F1, same instruction pair). A
kernel that reconfigs between two deep math phases should expect the higher figure.

### F6

Correctness only. `replay_buf_len` is unchanged for every shape the test suite generates, so no perf
delta is expected and none was observed (matmul rows above are flat).

## Correctness Evidence

Every suite run with a fresh `RUNNER_TEMP`, two-phase producer/consumer, on the p300a.

| Suite | Result | What it covers for these changes |
|---|---|---|
| `test_reduce.py` | **3528 passed** | F1. 8 tile dims x {Float32, Float16_b, Bfp8_b, Bfp4_b} x 3 reduce dims x 3 pool types x fidelities x reduce-to-one |
| `test_matmul.py` | **6784 passed** | F6. Full format/fidelity/dest-acc/dimension sweep |
| `test_experimental_reconfig_escape.py`, `test_matmul_and_unary_sfpu.py`, `test_sdpa_reinits.py`, `test_tilize_transition_reconfig.py`, `test_eltwise_binary.py`, `test_bcast.py`, `test_dest_copy.py` | **4464 passed, 87 skipped** | F2. Reconfig escapes, re-inits, and Int8/UInt8/Int32 boundary crossings |
| Broad math-thread sweep — `test_eltwise_unary_datacopy.py`, `test_transpose_dest.py`, `test_math_matmul.py`, `test_reduce_block_max.py`, `test_sdpa_reduce_row.py`, `test_sdpa_weighted_reduce.py`, `test_sum_reduce_scalar.py`, `test_mul_reduce_scalar.py`, `test_unpack_tilize.py`, `test_matmul_pack_untilize.py`, `test_matmul_unpack_tilize.py`, `test_pack_untilize.py`, `test_tilize_polluter_matmul.py`, `test_tilize_polluter_tiny_matmul.py`, `test_dest_copy.py` | **151775 passed, 164 skipped** | `cmath_common.h` is included by every math kernel, so the F2 tracker is exercised library-wide |

| `test_reduce.py`, `test_reduce_block_max.py`, `test_sum_reduce_scalar.py` (re-run) | **3628 passed, 14 skipped** | Re-verification after `_llk_math_reduce_uninit_` was given the paired flag restore, which landed after the broad sweep's ELFs were built |

`test_reduce.py` passing is load-bearing for F1 beyond regression cover: it is what rules out the
hypothesis that GMPOOL misdetects the zero-substitution flag on any datum with a zero low byte. With
random bf16 stimuli over 16 tiles x 1024 datums, values such as `0x4400` (768.0) are hit constantly; a
MAX reduce under PRESERVE returns them unchanged across the whole sweep.

## Reproducing

Everything runs from `tests/python_tests` with a **fresh `RUNNER_TEMP` per variant** — reusing it
serves a stale ELF and silently reports the previous variant's cycles. Wipe `perf_data/runs` and
`perf_data/latest` between variants too; a failed run leaves the previous CSV in place and it reads as
a fresh result.

```bash
cd tt_metal/tt-llk/tests/python_tests

export RUNNER_TEMP=$(mktemp -d)
rm -rf ../../perf_data/runs ../../perf_data/latest

IDS=(
 "perf_reduce.py::test_perf_reduce[formats:Float16_b->Float16_b-dest_acc:No-reduce_dim:Row-pool_type:Max-math_fidelity:HiFi4]"
 "perf_reduce.py::test_perf_reduce[formats:Float16_b->Float16_b-dest_acc:No-reduce_dim:Column-pool_type:Sum-math_fidelity:LoFi]"
 "perf_reduce.py::test_perf_reduce[formats:Float16_b->Float16_b-dest_acc:No-reduce_dim:Column-pool_type:Sum-math_fidelity:HiFi4]"
)

../.venv/bin/python -m pytest --speed-of-light --compile-producer -n 4 -m perf -q "${IDS[@]}"
../.venv/bin/python -m pytest --speed-of-light --compile-consumer -n 1 -m perf -q "${IDS[@]}"
```

Read the result from `perf_data/latest/perf_reduce/perf_reduce.post.csv`, column
`mean(MATH_ISOLATE)`, rows where `marker == TILE_LOOP`.

Notes that cost time if you rediscover them:

- Pass **full node IDs** as positional arguments. `-k` cannot parse the format ids, which contain `->`.
- `perf_data/latest` is a symlink to the newest `perf_data/runs/local-<UTC>/` directory. The old flat
  `perf_data/<module>/` path is no longer written.
- `TENSIX TIMED OUT` means stop and `tt-smi -r` before believing any later result. A wedged Tensix
  survives process exit and every subsequent test fails naming whatever kernel it happens to be on.
- Correctness suites use the same two-phase flow without `--speed-of-light -m perf`.

## Method Notes and Limits

- Single-op, speed-of-light, one format pair, one board. These numbers isolate the math thread; they
  are not end-to-end op timings. A math-side saving only shows up end to end where math was the
  limiter — which, per the audit's bound column, is REDUCE_ROW MAX, the HiFi4 SUM reduces, and HiFi
  matmul/eltwise-mul.
- Every delta above was produced by A/B-ing the same commit with and without the change (via
  `git stash push -- <files>`), not by comparing against a remembered figure.
- The one number that is not a direct A/B is the per-instruction cost table in the audit; those are
  solved from the LoFi/HiFi4 pair for each kernel, which is exact only if fidelity-phase count is the
  sole difference between the two builds. It is, for these kernels.
- F3, F4 and F5 in the audit are **not implemented** and their quoted prizes are estimates, labelled as
  such. Do not treat them as measured.
