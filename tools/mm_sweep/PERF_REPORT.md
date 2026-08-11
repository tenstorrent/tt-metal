# regime_a_matmul — performance report

## Hardware and software under test

| | |
|---|---|
| host | `bh-glx-120-c02u02` |
| board | `tt-galaxy-bh` — Blackhole Galaxy, 32 chips (`tt-smi -ls`) |
| chip under test | device 0; compute grid **12x10 = 120 cores** (`compute_with_storage_grid_size`) |
| AICLK | 800 MHz idle (`0x320`), max 1350 MHz (`0x546`); **1.350002 GHz measured at load** by the device profiler's own frequency sync — the same 1.35 GHz used for cycles->us |
| L1 budget | 1440 KB/core (`kL1BudgetBytes`) |
| numerics | BFLOAT16 in/out, HiFi2, FP32 dest accumulation (fixed by the op) |
| branch | `cglagovich/regime-a-single-chip-opt` @ **40ceaa47c4e** |
| rebased onto | `origin/main` @ **5a934d9f884** (2026-08-11 12:22:36 +0000) |
| date | 2026-08-11 |

**Board matters.** The optimisation campaign ran on an 11x10 / 110-core single-card part, and its goldens are
not thresholds on this 120-core Galaxy chip — comparing across them reported six false regressions of
+5.3%..+13.8%. Perf goldens are therefore keyed by compute grid; see
`tests/ttnn/unit_tests/operations/matmul/test_regime_a_matmul_perf.py`.

## Measurement method

Device time from the device-profiler CSV demuxed by run-host-id — **not** host wall, which would fold dispatch
overhead into a 7 us shape. Each shape runs in its own subprocess (the CSV is only written when the device
closes), 2 blocks x [2 warmup + 12 timed] iterations on resident inputs, median over all 24. Every run is at
DEFAULTS (`config=None`, no env overrides), so this is what ships. Correctness is checked on the same program
the timing uses: PCC vs a torch reference plus an explicit non-finite count.

    # one shape
    TT_METAL_DEVICE_PROFILER=1 python3 tools/mm_sweep/picker_gen/prod_sweep_worker.py <M> <K> <N> 2 auto

    # the 10-shape golden gate (board-keyed)
    scripts/run_safe_pytest.sh --run-all \
      tests/ttnn/unit_tests/operations/matmul/test_regime_a_matmul_perf.py -q -p no:randomly

DRAM % uses this campaign's accounting: `bytes = Ns*M*K*2 + K*N*2 + M*N*2` against a 512 GB/s ceiling, with
`Ns` read from the picker's own log line (only `Ns` duplicates in0).

## Golden gate, this board (10/10 pass)

Margins are 8% under 30 us (iteration spread there reaches 12%) and 5% above (spread <= 4%).

| shape | median us | margin | covers |
|---|---|---|---|
| 32x2048x512 | 7.76 | 8% | Mt1 chain bank-local 64c; smallest, overhead-dominated |
| 32x6144x9216 | 258.90 | 5% | Mt1 chain bank-local 24c; highest DRAM efficiency |
| 64x2048x1024 | 13.53 | 8% | Mt2 reduce-scatter bank-local 64c |
| 128x6144x4608 | 140.24 | 5% | Mt4 chain bank-local 96c |
| 256x2048x1024 | 19.48 | 8% | Mt8 reduce-scatter in1-near 64c; noisiest shape |
| 256x6144x768 | 37.24 | 5% | Mt8 chain mesh 96c |
| 256x15360x768 | 85.81 | 5% | Mt8 reduce-scatter mesh 80c; deep K |
| 256x6144x6144 | 194.70 | 5% | Mt8 reduce-scatter 96c; large square |
| 512x6144x2304 | 112.63 | 5% | Mt16 chain mesh 96c |
| 512x6144x4608 | 188.45 | 5% | Mt16 chain mesh 96c; compute-floor-bound |

Block-median spread 0.3-1.8%, PCC >= 0.999986, zero non-finite across all ten.

## Model shapes: unfused AG + regime-A MM vs main's fused AGMM

Measured on the same Galaxy at **41afa6004a4** (this branch before the rebase; the `regime_a_matmul` sources
are unchanged by the rebase apart from the error-message commit, so these remain current). LTX = tp4, FLUX =
tp8, ring, `num_links=2`. `fused_agmm`/`ag`/`mm_old` are reused from `cglagovich/agmm_analysis`
(`agmm/comparison.csv`), collected on this same box.

Restricted to Regime-A's domain (**M < N**) and its acceptance scope (**M <= 512**) — 16 shapes, all measured,
no failures:

| set | n | vs main fused (median) | vs AG + existing MM (median) | summed fused -> AG+regime-A |
|---|---|---|---|---|
| LTX | 5 | 2.69x | 1.82x | 281 -> 124 us = **2.27x** |
| FLUX | 11 | 1.39x | 1.12x | 2734 -> 1806 us = **1.51x** |
| **ALL** | **16** | **2.28x** | **1.56x** | 3015 -> 1930 us = **1.56x** |

`mm_ra` beats `mm_old` on every one of the 16 rows, so the win is a real MM improvement rather than an
artifact of the composition. Best cases are Mt=1 (M=32): 2.27-3.98x vs main fused at 76-86% DRAM. The one
sub-parity shape is 512x15360x768 (0.91x), where a 185 us AG leg dominates a 146 us MM — nothing to fix on the
MM side. Full per-shape table: `LTXFLUX_AGMM_MILESTONE.md`.

Two caveats kept visible: 3 LTX rows compare a *fused* `mm_old` against an *unfused* `mm_ra` (regime_a_matmul
has bias/activation/addcmul but no `chunks`), which flatters regime-A; and two in-domain shapes outside this
scope regress — 1216x4096x3072 and 1216x4096x4096 pick `Sm=13` at 6.6-6.9% DRAM and are ~6x slower than the
existing MM, which looks like a cheap picker fix and must be resolved before widening the scope past M=512.

## Correctness at this commit

| suite | result |
|---|---|
| `test_regime_a_matmul.py` | 111 passed |
| `test_regime_a_matmul_audit.py` | 40 passed |
| `test_regime_a_matmul_corpus.py` | 60 passed |
| `test_regime_a_matmul_perf.py` | 10 passed |
| **total** | **221 passed, 0 failed** |

Covers bias, activations, ternary/addcmul (incl. broadcast, full and fp32 gates), output chunks, and ring
reduce-scatter — including the reduce-scatter CB wrap-alignment regression cases, which assert finiteness as
well as PCC.
