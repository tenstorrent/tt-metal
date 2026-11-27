# WA.2: wo_a bank-local reads, op-level probe on the 8x4 mesh (2026-11-27)

**Verdict against the owner's KILL line (measured gain < half WA.1's upper bound, or not exact -> KILL):
KILL at 8192 (narrowly); 5120 passes. Exactness passes everywhere.**

| | 8192 chunk (256 rows a chip) | 5120 chunk (160 rows a chip) |
|---|---|---|
| today (shipped knobs), us a call | 151.32 (bw 2, depth 2) | 131.50 (bw 4, depth 4) |
| bank path, best exact config at today's block_w, us a call (whole mesh, slowest chip) | **125.01** (band 10, depth 4, w_inflight 1, x_ahead 1) | **106.89** (band 10, depth 4, w_inflight 2, x_ahead 1) |
| gain a call | 26.3 us (-17.4 %) | 24.6 us (-18.7 %) |
| gain a chunk, x 21 calls (`woa_gain_ms_c*`) | **0.552 ms** | **0.517 ms** |
| gain a chunk, x 20 calls (WA.1's count, layers 0-20) | 0.526 ms | 0.492 ms |
| WA.1 upper bound / half of it | 1.19 / **0.595 ms** | 1.00 / **0.50 ms** |
| per-call kill threshold (WOA_BANK_DESIGN.md section 7) | > 121.2 us | > 108.1 us |
| verdict | **KILL**: 0.552 < 0.595 ms (125.0 > 121.2 us a call; 3.8 us a call short) | pass: 0.517 >= 0.50 ms (106.9 < 108.1 us) |
| exact: bank vs today on every chip (`woa_mismatch_*`) | 0 (default, best, 435 swept configs) | 0 (default, best, best any bw, 435 swept configs) |
| repeats on fresh buffers (`woa_repeat_diff_*`) | 0 | 0 |
| host us a call, today / bank (20 warm calls, no sync) | 100.2 / 124.2 | 99.6 / 296.0 (one outlier run; 133.6 the run before) |

Numbers: the last gate check (`gate WA.2 --check`, PASS, log runs/galaxy_8x4-full/run1/logs/WA.2.check.log), real-time
profiler, slowest chip, 20 calls a section; random fixed-seed tensors of wo_a's real shapes and dtypes (x and w
bfloat8_b, output the model's bfloat8_b, ATTN_FIDELITY HiFi2 with fp32 DEST, today's knobs from mm.gmm_knobs:
GMM256 at 256 rows, WO_A_GMM_* at 160). Run-to-run spread over the 4 runs of this step: today 151.2-151.8 / 131.4-131.8
us; bank best 123.8-125.3 / 106.3-106.9 us. The realistic roofline of WA.1 (all bytes at 410 GB/s) is 114.2 / 104.0 us:
the bank path is within 9 % / 3 % of it. At 256 rows band 10 is compute-bound (its no-x, no-weight and no-compute stubs
all run 116-121 us; 4-column cores), band 12 (3-column cores) is held by the x stream.

What the owner decides from: the line is missed at 8192 by 0.04 ms a chunk (3.8 us a call), met at 5120. Host: the bank
call costs ~24-34 us more host time than today's (sharded memory-config checks / key and more kernel ranges; not yet
trimmed), ~0.5-0.7 ms a chunk on a host-bound span: WA.3 must remove it before any accept. DS41-390: bit-identical on
random inputs is not proof on real ones; WA.3's off == on check on a real chunk would decide that.

## Step 0: today's op with diagnostic stubs (OV-76, before building)

| stub (today's op, `diag=`) | 256 rows us | x today | 160 rows us | x today |
|---|---|---|---|---|
| today | 151.57 | 1.000 | 131.40 | 1.000 |
| no weight reads | 107.20 | 0.707 | 70.92 | 0.540 |
| no compute | 149.44 | 0.986 | 129.70 | 0.987 |
| no x read / multicast | 117.81 | 0.777 | 118.87 | 0.905 |

no_w 0.707 < 0.9 at 256 rows: not compute-bound, so the build went ahead (the no-compute stub ~ today: reader-bound at
both row counts). Today's chip spread: 142.6-151.6 us (256), 117.8-131.4 us (160).

## The bank path built (ttnn.bringup.grouped_matmul option `w_layout`, grouped_matmul_ttnn/bank.py)

- Layout: bank b = group b (8 groups, 8 banks); each core's columns K-major in its bank (one contiguous read a K block);
  WIDTH_SHARDED DRAM, the experts' bank_host_tensor matrix, replicated (35.65 MB a chip, no padding).
- Cores: a `cores_per_bank / 2 x 2` rectangle a bank ("band") on its anchor's side in the 2-row band nearest the
  anchor, or one grid row a bank ("row"); x sender = the rectangle's first core.
- Reader (kernels/gmm_bank_reader.cpp): bank-local weight reads (`w_inflight` blocks in flight, transaction ids),
  today's READY / VALID x multicast (bounded waits), `x_ahead` non-blocking x read-ahead. Compute and writer: today's.
- What it took (each measured, all exact): (1) the sender at the bank's anchor (inside the rectangle) wrapped NoC-0
  multicasts round the chip: 342 us at 256 rows, 2.3x today (DS41-393); the start-corner sender: 155 us. (2) An x
  read-ahead that blocked on a slot at depth 2 stalled each multicast on the sender's own compute (no_w stub 106 -> 152
  us): made non-blocking (`x_ahead`). (3) `from_torch` straight to the sharded config quantizes bfloat8_b differently
  (0.0625 off, DS41-392): `bank.upload` converts plain and places after. (4) Fewer cores a bank won: band 10 (80 cores)
  beat band 12 (96); band 8 / 6, row 8 / 9 measured no better (8) or far worse (6: 276 us, 6-column cores).

## Sweep (in parallel across chips: one mesh program, a different knob set on each of the 32 chips)

435 configs a row count: layouts band 12, band 10, row 12, row 11, row 10 x block_w {2, 4, 8} x depth {2, 3, 4} x
w_inflight 1..depth x x_ahead 0..min(3, depth - 1); every chip's output compared with today's op at the same block_w
on that chip (0 inexact of 870), its us normalised by that chip's own time of today's call (`norm`; the chips differ by
up to 6 % / 12 %). 32 configs a launch; ~4.5 s a layout. The best then confirmed on the whole mesh (table above).
Single-card processes were not needed (the per-chip mesh program did it with 0 device opens).

Best per layout (norm = chip us / that chip's today):

| layout | 256 rows: best knobs (bw, depth, wi, xa) | us | norm | 160 rows: best knobs | us | norm |
|---|---|---|---|---|---|---|
| band 10 (80 cores) | 2, 4, 1, 1 | 124.12 | 0.823 | 2, 3, 2, 1 | 104.65 | 0.796 |
| band 12 (96 cores) | 2, 4, 1, 1 | 128.81 | 0.854 | 2, 4, 1, 2 | 110.06 | 0.842 |
| row 10 (80 cores) | 4, 3, 1, 1 | 129.25 | 0.858 | 4, 4, 1, 2 | 113.38 | 0.866 |
| row 11 (88 cores) | 2, 4, 1, 2 | 135.40 | 0.898 | 2, 4, 1, 1 | 117.77 | 0.896 |
| row 12 (96 cores) | 4, 4, 1, 2 | 146.41 | 0.971 | 4, 4, 1, 2 | 123.97 | 0.947 |

Best per block_w (any layout): 256 rows bw 2 0.823 (band 10), bw 4 0.853 (band 10), bw 8 0.883 (row 10); 160 rows
bw 2 0.796, bw 4 0.808 (band 10), bw 8 0.872 (row 10). Today's K order is block_w 2 at 256 rows and 4 at 160 rows
(DS41-390): the recorded `woa_us_bank_*` keeps it; at 160 rows block_w 2 (another K order) is 1.8 us faster
(`woa_us_bank_any_bw_160` 105.13 us, exact vs today's op at block_w 2).

**Best knobs** (exact vs today's shipped call): 8192 / 256 rows: band, 10 cores a bank, block_w 2, depth 4,
w_inflight 1, x_ahead 1. 5120 / 160 rows: band, 10 cores a bank, block_w 4, depth 4, w_inflight 2, x_ahead 1.

Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate WA.2 --spec $BRINGUP_SPEC --check` (the probe:
models/demos/deepseek_v41_flash_d_p/tests/woa/test_woa_probe.py on the resident runner's held mesh; the fork tests:
tests/woa/test_woa_fork_bank.py).
