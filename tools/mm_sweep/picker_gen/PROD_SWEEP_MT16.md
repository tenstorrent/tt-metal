| shape | Mt | Pk,Ns,Sm,kb,nsb | core | reduction | placement | dev us | eff GB/s | %pk | sch/val | PCC | blk% | it% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 256x2048x512 | 8 | 4,1,3,2,2 | 96 | reduce-scatter | mesh | 13.76 | 247.6 | 48% | 1.05 | 0.99999 | 0.4 | 7.2 |
| 512x6144x768 | 16 | 12,1,1,2,1 | 96 | chain | mesh | 61.79 | 267.3 | 52% | 1.00 | 0.99999 | 0.1 | 3.6 |
| 128x2048x512 | 4 | 4,1,2,2,2 | 64 | reduce-scatter | in1-near | 10.22 | 269.3 | 53% | 1.00 | 0.99999 | 0.5 | 7.1 |
| 256x2048x1024 | 8 | 4,1,2,2,4 | 64 | reduce-scatter | in1-near | 19.54 | 295.1 | 58% | 1.00 | 0.99999 | 1.3 | 12.0 |
| 256x2048x1536 | 8 | 4,1,3,2,3 | 96 | reduce-scatter | in1-near | 27.48 | 295.7 | 58% | 1.03 | 0.99999 | 0.8 | 4.1 |
| 64x2048x512 | 2 | 4,2,1,2,1 | 64 | chain | bank-local | 8.83 | 304.4 | 59% | 1.00 | 0.99999 | 0.7 | 7.7 |
| 256x2048x2048 | 8 | 4,1,3,2,4 | 96 | reduce-scatter | in1-near | 34.38 | 305.0 | 60% | 1.02 | 0.99999 | 0.2 | 7.6 |
| 512x6144x1536 | 16 | 12,1,1,2,1 | 96 | chain | mesh | 86.66 | 308.6 | 60% | 1.00 | 0.99999 | 0.1 | 2.3 |
| 512x2304x6144 | 16 | 3,4,1,1,1 | 96 | chain | mesh | 141.28 | 311.7 | 61% | 1.00 | 1.00005 | 0.9 | 4.2 |
| 512x3072x6144 | 16 | 6,2,1,2,1 | 96 | chain | mesh | 158.31 | 317.9 | 62% | 1.00 | 1.00011 | 0.9 | 7.4 |
| 32x2048x512 | 1 | 4,2,1,2,1 | 64 | chain | bank-local | 7.32 | 326.9 | 64% | 1.00 | 0.99999 | 0.3 | 6.6 |
| 512x6144x2304 | 16 | 12,1,1,2,1 | 96 | chain | mesh | 109.95 | 336.2 | 66% | 1.00 | 1.00001 | 0.1 | 2.1 |
| 128x2048x1024 | 4 | 4,1,2,2,4 | 64 | reduce-scatter | in1-near | 14.51 | 343.3 | 67% | 1.00 | 0.99999 | 1.0 | 10.7 |
| 256x6144x768 | 8 | 12,1,1,2,1 | 96 | chain | mesh | 35.72 | 363.3 | 71% | 1.00 | 0.99998 | 0.2 | 3.8 |
| 256x15360x768 | 8 | 6,1,2,2,3 | 96 | reduce-scatter | mesh | 86.86 | 366.7 | 72% | 1.00 | 0.99999 | 0.3 | 3.2 |
| 128x6144x768 | 4 | 12,1,1,2,1 | 96 | chain | bank-local | 30.37 | 369.0 | 72% | 1.00 | 0.99999 | 0.1 | 4.3 |
| 512x6144x4608 | 16 | 12,1,1,2,1 | 96 | chain | mesh | 180.16 | 375.4 | 73% | 1.00 | 1.00009 | 0.2 | 1.7 |
| 256x6144x1536 | 8 | 6,1,2,4,2 | 96 | reduce-scatter | in1-near | 60.74 | 375.5 | 73% | 1.00 | 0.99999 | 0.2 | 3.6 |
| 128x2048x1536 | 4 | 4,3,1,2,1 | 96 | chain | bank-local | 21.93 | 376.5 | 74% | 1.00 | 0.99999 | 0.4 | 5.0 |
| 64x2048x1024 | 2 | 4,2,1,2,2 | 64 | reduce-scatter | bank-local | 12.16 | 398.8 | 78% | 1.00 | 0.99999 | 0.2 | 4.3 |
| 64x6144x768 | 2 | 12,1,1,2,1 | 96 | chain | bank-local | 25.56 | 403.9 | 79% | 1.00 | 0.99999 | 0.1 | 3.2 |
| 256x6144x2304 | 8 | 12,1,1,2,1 | 96 | chain | mesh | 79.79 | 409.0 | 80% | 1.00 | 1.00001 | 0.0 | 2.7 |
| 256x15360x1536 | 8 | 6,1,2,2,6 | 96 | reduce-scatter | in1-near | 135.53 | 412.0 | 80% | 1.00 | 0.99999 | 1.1 | 3.7 |
| 256x6080x4640 | 8 | 12,1,1,2,1 | 96 | chain | mesh | 148.77 | 416.2 | 81% | 1.06 | 1.00003 | 0.1 | 2.4 |
| 32x256x6144 | 1 | 1,3,1,1,8 | 24 | chain | bank-local | 8.62 | 416.3 | 81% | 1.00 | 0.99999 | 0.4 | 2.9 |
| 64x2048x1536 | 2 | 4,3,1,2,1 | 96 | chain | bank-local | 17.35 | 419.2 | 82% | 1.00 | 0.99999 | 0.7 | 4.6 |
| 256x2304x6144 | 8 | 3,4,1,1,3 | 96 | chain | bank-local | 85.29 | 424.1 | 83% | 1.00 | 1.00005 | 0.5 | 4.6 |
| 32x6144x768 | 1 | 6,1,1,2,3 | 48 | chain | bank-local | 23.15 | 426.8 | 83% | 1.00 | 0.99999 | 0.3 | 1.2 |
| 128x2048x2048 | 4 | 4,3,1,2,3 | 96 | reduce-scatter | bank-local | 24.44 | 429.0 | 84% | 1.11 | 1.00001 | 0.5 | 6.3 |
| 128x6144x1536 | 4 | 12,1,1,2,2 | 96 | chain | bank-local | 48.42 | 430.4 | 84% | 1.00 | 0.99999 | 0.1 | 4.3 |
| 128x15360x768 | 4 | 6,1,1,2,3 | 48 | reduce-scatter | bank-local | 63.84 | 434.3 | 85% | 1.00 | 0.99999 | 0.1 | 4.4 |
| 64x2048x2048 | 2 | 2,3,1,2,3 | 48 | chain | bank-local | 21.63 | 436.2 | 85% | 1.11 | 0.99999 | 0.2 | 6.1 |
| 256x6144x6144 | 8 | 6,1,2,4,2 | 96 | reduce-scatter | in1-near | 187.48 | 436.3 | 85% | 1.00 | 1.00005 | 0.2 | 3.0 |
| 32x2048x1024 | 1 | 2,4,1,4,1 | 64 | chain | bank-local | 10.95 | 437.0 | 85% | 1.00 | 0.99999 | 0.3 | 5.2 |
| 256x6144x4608 | 8 | 6,1,2,4,2 | 96 | reduce-scatter | in1-near | 141.37 | 439.5 | 86% | 1.00 | 1.00000 | 0.4 | 3.9 |
| 32x2048x1536 | 1 | 2,2,1,4,3 | 32 | chain | bank-local | 15.02 | 442.8 | 86% | 1.00 | 0.99999 | 0.2 | 2.3 |
| 64x15360x768 | 2 | 10,1,1,2,3 | 80 | chain | bank-local | 57.82 | 443.7 | 87% | 1.00 | 0.99999 | 0.7 | 2.4 |
| 128x6080x4640 | 4 | 12,1,1,2,2 | 96 | chain | bank-local | 133.29 | 443.9 | 87% | 1.11 | 1.00000 | 0.1 | 1.2 |
| 128x6144x2304 | 4 | 12,1,1,2,1 | 96 | chain | bank-local | 67.84 | 449.2 | 88% | 1.00 | 0.99999 | 0.5 | 2.1 |
| 32x2048x2048 | 1 | 2,2,1,4,4 | 32 | chain | bank-local | 19.34 | 454.1 | 89% | 1.00 | 0.99999 | 0.0 | 2.2 |
| 128x2304x6144 | 4 | 3,2,1,1,6 | 48 | chain | bank-local | 68.40 | 454.1 | 89% | 1.00 | 0.99997 | 0.6 | 3.3 |
| 64x6080x4640 | 2 | 12,1,1,2,2 | 96 | chain | bank-local | 127.02 | 455.0 | 89% | 1.11 | 0.99999 | 0.0 | 0.9 |
| 64x6144x2304 | 2 | 12,1,1,2,2 | 96 | chain | bank-local | 63.80 | 460.7 | 90% | 1.11 | 0.99999 | 0.3 | 1.1 |
| 128x15360x1536 | 4 | 12,1,1,1,3 | 96 | chain | bank-local | 111.67 | 461.3 | 90% | 1.00 | 0.99999 | 0.0 | 2.8 |
| 64x6144x1536 | 2 | 3,1,1,8,2 | 24 | chain | bank-local | 42.79 | 464.1 | 91% | 1.00 | 0.99999 | 0.4 | 5.4 |
| 32x6080x4640 | 1 | 12,1,1,2,1 | 96 | chain | bank-local | 122.73 | 465.3 | 91% | 1.06 | 0.99999 | 0.2 | 0.9 |
| 128x6144x6144 | 4 | 12,1,1,2,2 | 96 | chain | bank-local | 167.63 | 469.1 | 92% | 1.00 | 0.99999 | 0.1 | 3.0 |
| 128x6144x4608 | 4 | 12,1,1,2,1 | 96 | chain | bank-local | 125.27 | 474.0 | 93% | 1.00 | 0.99999 | 0.1 | 1.7 |
| 32x6144x1536 | 1 | 6,1,1,4,2 | 48 | chain | bank-local | 40.56 | 477.5 | 93% | 1.00 | 0.99999 | 0.1 | 4.3 |
| 32x15360x768 | 1 | 6,1,1,2,3 | 48 | chain | bank-local | 51.24 | 480.5 | 94% | 1.00 | 0.99999 | 0.4 | 2.3 |
| 64x6144x6144 | 2 | 12,1,1,2,2 | 96 | chain | bank-local | 158.72 | 485.6 | 95% | 1.00 | 0.99998 | 0.0 | 1.5 |
| 32x2304x6144 | 1 | 3,2,1,1,6 | 48 | chain | bank-local | 59.63 | 486.3 | 95% | 1.00 | 0.99999 | 0.9 | 2.0 |
| 64x15360x1536 | 2 | 6,1,1,2,3 | 48 | chain | bank-local | 101.40 | 486.7 | 95% | 1.00 | 0.99999 | 0.1 | 0.9 |
| 48x6144x4608 | 2 | 6,1,1,4,2 | 48 | chain | bank-local | 118.38 | 487.0 | 95% | 1.01 | 0.99999 | 0.1 | 0.9 |
| 64x2304x6144 | 2 | 3,4,1,1,3 | 96 | chain | bank-local | 62.01 | 488.3 | 95% | 1.00 | 0.99999 | 0.3 | 1.8 |
| 32x6100x4608 | 1 | 12,1,1,2,1 | 96 | chain | bank-local | 116.34 | 489.1 | 96% | 1.01 | 0.99999 | 0.1 | 0.9 |
| 64x6144x4608 | 2 | 6,1,1,4,2 | 48 | chain | bank-local | 118.38 | 490.0 | 96% | 1.00 | 0.99999 | 0.1 | 0.6 |
| 32x6144x4600 | 1 | 12,1,1,2,1 | 96 | chain | bank-local | 116.73 | 490.1 | 96% | 1.00 | 0.99999 | 0.1 | 0.8 |
| 32x6144x4608 | 1 | 12,1,1,2,1 | 96 | chain | bank-local | 116.91 | 490.2 | 96% | 1.00 | 0.99999 | 0.0 | 1.0 |
| 32x6144x2304 | 1 | 3,1,1,4,5 | 24 | chain | bank-local | 58.59 | 492.4 | 96% | 1.11 | 0.99999 | 0.2 | 1.5 |
| 32x6144x3072 | 1 | 3,1,1,4,6 | 24 | chain | bank-local | 77.80 | 492.8 | 96% | 1.00 | 0.99999 | 0.9 | 1.4 |
| 64x4608x6144 | 2 | 3,2,1,2,3 | 48 | chain | bank-local | 118.90 | 492.8 | 96% | 1.00 | 1.00000 | 0.1 | 0.9 |
| 32x15360x1536 | 1 | 6,1,1,2,3 | 48 | chain | bank-local | 97.92 | 492.9 | 96% | 1.00 | 0.99999 | 0.3 | 1.2 |
| 64x6144x9216 | 2 | 6,1,1,4,2 | 48 | chain | bank-local | 231.81 | 497.0 | 97% | 1.00 | 1.00001 | 0.0 | 1.0 |
| 32x6144x6144 | 1 | 6,1,1,4,2 | 48 | chain | bank-local | 152.99 | 498.6 | 97% | 1.00 | 0.99999 | 0.1 | 0.7 |
| 32x6144x9216 | 1 | 3,1,1,4,6 | 24 | chain | bank-local | 227.89 | 501.3 | 98% | 1.00 | 1.00000 | 0.4 | 1.3 |

66 shapes measured, all at DEFAULTS (config=None, no diagnostic mask).
effective DRAM BW: min 247.6  median 436.6  max 501.3 GB/s   (peak 512)
%peak: median 85%   under 50%: 1   over 80%: 44
correctness: 66/66 PCC >= 0.999   (min PCC 0.99997)
stability: median block-to-block spread 0.2%, worst 1.3% (256x2048x1024); median iteration spread 2.9%
reduction: 14 reduce-scatter / 52 chain     placement: 11 mesh / 9 in1-near / 46 bank-local

FAILED / not measured (1):
  512x15360x768  runtime  TT_FATAL @ /localdev/cglagovich/tt-metal/ttnn/cpp/ttnn/operations/expe

Peak = 512 GB/s (the measured BH DRAM ceiling used throughout this campaign).
eff GB/s = (Ns*M*K + K*N + M*N)*2 / device time. in0 is counted Ns times because each of the Ns
n-slice groups reads all of in0; within an 8-bank ring the 8 cores read DIFFERENT shards, so there is no
duplication across a ring, nor across Pk or Sm. in0 ring FORWARD traffic is NoC-only and excluded.
Padded positions are never DRAM-read (balanced tails), so these bytes are the bytes physically moved.
sch/val = scheduled capacity / logical bytes: schedule padding the op carries in compute+L1 (1.00 = none).
blk% = spread between the two 12-iteration measurement blocks; it% = spread across all 24 iterations.
cfg/reduction/placement are read from the factory's own log, i.e. ground truth, not a host-side mirror.
