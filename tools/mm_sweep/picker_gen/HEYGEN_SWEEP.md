> **SUPERSEDED by `HEAD_SWEEP_UNIFIED.md`**, which re-measures every shape from this report plus the
> other one on HEAD (7bdab431417) in a single table. These numbers were taken on the PRE-rebase build.
> They still hold (the re-run agrees to a median +0.1%, worst +1.7%, and no pick moved), but the
> `FAILED / not measured` row below for `32x128x30720` is still accurate: Kt=4 < the 8 a k-slice must
> span, which is structural.

| shape | Mt | Pk,Ns,Sm,kb,nsb | core | reduction | placement | dev us | eff GB/s | %pk | TFLOP/s | FPU%grid | FPU%core | sch/val | PCC | blk% | it% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32x256x512 | 1 | 1,2,1,1,1 | 16 | chain | bank-local | 4.29 | 76.4 | 15% | 2.0 | 0.6% | 4.4% | 1.00 | 0.99999 | 0.3 | 5.6 |
| 512x5120x2560 | 16 | 10,1,1,2,1 | 80 | chain | mesh | 109.54 | 311.1 | 61% | 122.5 | 40.3% | 55.4% | 1.00 | 1.00002 | 0.1 | 1.6 |
| 512x5120x5120 | 16 | 10,1,1,2,1 | 80 | chain | mesh | 187.17 | 336.1 | 66% | 143.4 | 47.2% | 64.8% | 1.00 | 1.00012 | 0.2 | 1.0 |
| 512x4096x5120 | 16 | 4,3,1,2,1 | 96 | chain | mesh | 173.27 | 344.9 | 67% | 123.9 | 40.8% | 46.7% | 1.04 | 0.99998 | 0.9 | 8.2 |
| 32x256x5120 | 1 | 1,5,1,1,4 | 40 | chain | bank-local | 7.23 | 419.3 | 82% | 11.6 | 3.8% | 10.5% | 1.00 | 1.00000 | 0.2 | 2.2 |
| 32x5120x1280 | 1 | 10,1,1,2,1 | 80 | chain | bank-local | 30.93 | 437.0 | 85% | 13.6 | 4.5% | 6.1% | 1.00 | 0.99999 | 0.1 | 1.8 |
| 96x5120x2560 | 3 | 10,1,1,2,2 | 80 | chain | bank-local | 60.92 | 454.5 | 89% | 41.3 | 13.6% | 18.7% | 1.00 | 0.99999 | 0.1 | 2.1 |
| 96x2048x5120 | 3 | 4,3,1,2,2 | 96 | reduce-scatter | bank-local | 49.76 | 464.9 | 91% | 40.5 | 13.3% | 15.2% | 1.19 | 1.00000 | 0.2 | 1.8 |
| 96x5120x5120 | 3 | 10,1,1,2,2 | 80 | chain | bank-local | 113.91 | 477.5 | 93% | 44.2 | 14.5% | 20.0% | 1.00 | 0.99999 | 0.0 | 1.4 |
| 96x8192x5120 | 3 | 4,3,1,2,2 | 96 | reduce-scatter | bank-local | 187.47 | 477.9 | 93% | 43.0 | 14.1% | 16.2% | 1.19 | 1.00001 | 0.0 | 1.6 |
| 32x1280x30720 | 1 | 5,2,1,1,2 | 80 | chain | bank-local | 167.06 | 483.5 | 94% | 15.1 | 5.0% | 6.8% | 1.00 | 1.00002 | 0.1 | 1.2 |

11 shapes measured, all at DEFAULTS (config=None, no diagnostic mask).
effective DRAM BW: min 76.4  median 437.0  max 483.5 GB/s   (peak 512)
%peak: median 85%   under 50%: 1   over 80%: 7
FPU (bf16 HiFi2 = 2048 FLOP/cycle/core, 2.765 TFLOP/s per core, 110-core grid = 304 TFLOP/s):
  achieved: min 2.0  median 41.3  max 143.4 TFLOP/s
  FPU util vs FULL 110-core grid: median 13.6%  max 47.2% (512x5120x5120)
  FPU util vs ALLOCATED cores:    median 16.2%  max 64.8% (512x5120x5120)
correctness: 11/11 PCC >= 0.999   (min PCC 0.99998)
stability: median block-to-block spread 0.1%, worst 0.9% (512x4096x5120); median iteration spread 1.8%
reduction: 2 reduce-scatter / 9 chain     placement: 3 mesh / 0 in1-near / 8 bank-local
schedule padding > 1.15x on 2: 96x2048x5120 (1.19), 96x8192x5120 (1.19)

FAILED / not measured (1):
  32x128x30720  runtime  TT_FATAL @ /localdev/cglagovich/tt-metal/ttnn/cpp/ttnn/operations/expe

## Filter applied to the 23 HeyGen shapes

Kept a shape only if it is BOTH memory-bound by roofline AND servable by regime-A. Machine balance on this
board = 304 TFLOP/s (110 cores x 2.765 TFLOP/s, bf16 HiFi2) / 512 GB/s = **594 FLOP/byte**; a shape is
memory-bound when its arithmetic intensity `MNK/(MK+KN+MN)` is below that.

Regime-A structural requirements (not tunable -- they follow from the in0 8-bank ring and the in1 8-bank width
shard):
- **Nt wide enough to width-shard over 8 banks**: `7*ceil(Nt/8) < Nt`.
- **Kt >= 8**: the k-slice is distributed over exactly 8 banks, so the smallest slice is 8 tiles; the picker
  rejects anything needing more than 20% K padding.
- **M < N**: regime A shards in1, so it assumes in1 is the big operand. M >= N is regime B.

| shape | Mt | Kt | Nt | AI | t_dram us | t_comp us | verdict |
|---|---|---|---|---|---|---|---|
| 96x8192x5120 | 3 | 256 | 160 | 93 | 168.8 | 26.5 | KEEP |
| 96x5120x5120 | 3 | 160 | 160 | 93 | 106.2 | 16.5 | KEEP |
| 96x5120x2560 | 3 | 160 | 80 | 91 | 54.1 | 8.3 | KEEP |
| 96x2048x5120 | 3 | 64 | 160 | 90 | 43.6 | 6.6 | KEEP |
| 32x256x512 | 1 | 8 | 16 | 27 | 0.6 | 0.0 | KEEP |
| 32x256x5120 | 1 | 8 | 160 | 28 | 5.8 | 0.3 | KEEP |
| 32x5120x1280 | 1 | 160 | 40 | 31 | 26.4 | 1.4 | KEEP |
| 32x1280x30720 | 1 | 40 | 960 | 31 | 157.6 | 8.3 | KEEP |
| 512x4096x5120 | 16 | 128 | 160 | 418 | 100.4 | 70.6 | KEEP |
| 512x5120x5120 | 16 | 160 | 160 | 427 | 122.9 | 88.3 | KEEP |
| 512x5120x2560 | 16 | 160 | 80 | 394 | 66.6 | 44.1 | KEEP |
| 32x128x30720 | 1 | 4 | 960 | 26 | 19.2 | 0.8 | kept by roofline, then REJECTED BY THE PICKER: Kt=4 < 8 |
| 2656x5120x3840 | 83 | 160 | 120 | 1202 | 169.8 | 343.4 | drop: compute-bound |
| 2656x5120x3456 | 83 | 160 | 108 | 1161 | 158.1 | 309.1 | drop: compute-bound |
| 2656x5120x1280 | 83 | 160 | 40 | 739 | 92.0 | 114.5 | drop: compute-bound, M>=N |
| 10560x5120x3840 | 330 | 160 | 120 | 1817 | 446.4 | 1365.3 | drop: compute-bound, M>=N |
| 10560x5120x3456 | 330 | 160 | 108 | 1726 | 422.9 | 1228.8 | drop: compute-bound, M>=N |
| 10560x3456x5120 | 330 | 108 | 160 | 1726 | 422.9 | 1228.8 | drop: compute-bound, M>=N |
| 10560x5120x1280 | 330 | 160 | 40 | 933 | 289.6 | 455.1 | drop: compute-bound, M>=N |
| 2656x5120x64 | 83 | 160 | 2 | 62 | 55.1 | 5.7 | drop: Nt=2 too narrow, M>=N |
| 10560x5120x64 | 330 | 160 | 2 | 63 | 215.1 | 22.8 | drop: Nt=2 too narrow, M>=N |
| 32x512x128 | 1 | 16 | 4 | 24 | 0.3 | 0.0 | drop: Nt=4 too narrow |
| 10560x224x1280 | 330 | 7 | 40 | 187 | 63.2 | 19.9 | drop: M>=N (also Kt=7 < 8) |

Column definitions are the same as PROD_SWEEP_MT16.md.
