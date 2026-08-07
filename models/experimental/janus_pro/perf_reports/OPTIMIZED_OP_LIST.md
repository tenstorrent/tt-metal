# tt-perf-report, one replay of the optimized tower

`tt-perf-report` over a single trace replay of the tower at **9.401 ms** — the op table, its advice
section and its summary, unabridged. Commit `6cdad2cf097` (change 28), Wormhole N150, 295 ops.

One of the ten replays the perf test measures, picked as the one whose kernel time sits closest to
the mean of replays 2-10 — the figure PERF.md's change log carries: 9.402 ms against 9.401.

```
Detected CSV format: v2.1 (with device arch and worker core count)
Using architecture from CSV: wormhole
Architecture: wormhole, Worker cores: 64
Sorting CSV by 'HOST START TS' column...
No signposts found in the file. Using the entire file for analysis.
Detected data from 1 devices. Merging device data...

🚀 Performance Report 🚀
========================

ID   Total %  Bound  OP Code                                  Device   Device Time  Op-to-Op Gap  Cores  DRAM     DRAM %  FLOPs        FLOPs %  Math Fidelity
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  2    0.4 %   SLOW  MatmulDeviceOperation 576 x 768 x 1024         0        44 μs                   48  83 GB/s  29.0 %  20.8 TFLOPs   21.1 %  HiFi2 BF16 x BF16 => BF16
  3    0.2 %         BinaryNgDeviceOperation                        0        21 μs          1 μs     64                                                BF16, BF16 => BF16
  4    0.2 %         BinaryNgDeviceOperation                        0        23 μs          1 μs     64                                                BF16, BF16 => BF16
  5    0.1 %         InterleavedToShardedDeviceOperation            0         8 μs          1 μs     48                                                      BF16 => BF16
  6    0.2 %         LayerNormDeviceOperation                       0        20 μs          1 μs     48                                                BF16, BF16 => BF16
  7    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  66 GB/s  22.7 %  75.5 TFLOPs   43.2 %   LoFi BF16 x BFP8 => BFP8
  8    0.1 %         ShardedToInterleavedDeviceOperation            0         9 μs          1 μs     48                                                      BFP8 => BFP8
  9    0.4 %         NlpCreateHeadsDeviceOperation                  0        36 μs          1 μs     18                                                      BFP8 => BFP8
 10    0.7 %         SDPAOperation                                  0        65 μs          1 μs     64                                                BFP8, BFP8 => BFP8
 11    0.1 %         NLPConcatHeadsDeviceOperation                  0        13 μs          1 μs     18                                                      BFP8 => BFP8
 12    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        19 μs          1 μs     48  88 GB/s  30.7 %  65.2 TFLOPs   37.3 %   LoFi BFP8 x BFP8 => BFP8
 13    0.1 %         BinaryNgDeviceOperation                        0        12 μs          1 μs     64                                                BF16, BFP8 => BF16
 14    0.2 %         LayerNormDeviceOperation                       0        20 μs          1 μs     48                                                BF16, BF16 => BF16
 15    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        73 μs          1 μs     48  58 GB/s  20.1 %  66.6 TFLOPs   38.1 %   LoFi BF16 x BFP8 => BFP8
 16    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.7 %  88.7 TFLOPs   50.8 %   LoFi BFP8 x BFP8 => BFP8
 17    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 18    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
 19    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  75.1 TFLOPs   43.0 %   LoFi BF16 x BFP8 => BFP8
 20    0.1 %         ShardedToInterleavedDeviceOperation            0         9 μs          1 μs     48                                                      BFP8 => BFP8
 21    0.4 %         NlpCreateHeadsDeviceOperation                  0        36 μs          1 μs     18                                                      BFP8 => BFP8
 22    0.7 %         SDPAOperation                                  0        67 μs          1 μs     64                                                BFP8, BFP8 => BFP8
 23    0.1 %         NLPConcatHeadsDeviceOperation                  0        13 μs          1 μs     18                                                      BFP8 => BFP8
 24    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        19 μs          1 μs     48  87 GB/s  30.2 %  64.1 TFLOPs   36.7 %   LoFi BFP8 x BFP8 => BFP8
 25    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 26    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
 27    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
 28    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.7 %  88.7 TFLOPs   50.8 %   LoFi BFP8 x BFP8 => BFP8
 29    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 30    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
 31    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.7 %  75.3 TFLOPs   43.1 %   LoFi BF16 x BFP8 => BFP8
 32    0.1 %         ShardedToInterleavedDeviceOperation            0         9 μs          1 μs     48                                                      BFP8 => BFP8
 33    0.4 %         NlpCreateHeadsDeviceOperation                  0        42 μs          1 μs     18                                                      BFP8 => BFP8
 34    0.7 %         SDPAOperation                                  0        67 μs          1 μs     64                                                BFP8, BFP8 => BFP8
 35    0.1 %         NLPConcatHeadsDeviceOperation                  0        13 μs          1 μs     18                                                      BFP8 => BFP8
 36    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        19 μs          1 μs     48  88 GB/s  30.6 %  65.0 TFLOPs   37.2 %   LoFi BFP8 x BFP8 => BFP8
 37    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 38    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
 39    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.7 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
 40    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.6 %  88.2 TFLOPs   50.5 %   LoFi BFP8 x BFP8 => BFP8
 41    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 42    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
 43    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.7 %  75.3 TFLOPs   43.1 %   LoFi BF16 x BFP8 => BFP8
 44    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
 45    0.4 %         NlpCreateHeadsDeviceOperation                  0        35 μs          1 μs     18                                                      BFP8 => BFP8
 46    0.7 %         SDPAOperation                                  0        65 μs          1 μs     64                                                BFP8, BFP8 => BFP8
 47    0.1 %         NLPConcatHeadsDeviceOperation                  0        13 μs          1 μs     18                                                      BFP8 => BFP8
 48    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        19 μs          1 μs     48  86 GB/s  29.9 %  63.6 TFLOPs   36.4 %   LoFi BFP8 x BFP8 => BFP8
 49    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 50    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
 51    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.8 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
 52    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.7 %  88.7 TFLOPs   50.8 %   LoFi BFP8 x BFP8 => BFP8
 53    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 54    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
 55    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.7 %  75.2 TFLOPs   43.0 %   LoFi BF16 x BFP8 => BFP8
 56    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
 57    0.4 %         NlpCreateHeadsDeviceOperation                  0        35 μs          1 μs     18                                                      BFP8 => BFP8
 58    0.7 %         SDPAOperation                                  0        65 μs          1 μs     64                                                BFP8, BFP8 => BFP8
 59    0.1 %         NLPConcatHeadsDeviceOperation                  0        13 μs          1 μs     18                                                      BFP8 => BFP8
 60    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  90 GB/s  31.2 %  66.3 TFLOPs   38.0 %   LoFi BFP8 x BFP8 => BFP8
 61    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 62    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
 63    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.7 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
 64    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.8 %  88.8 TFLOPs   50.8 %   LoFi BFP8 x BFP8 => BFP8
 65    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 66    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
 67    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        49 μs          1 μs     48  65 GB/s  22.5 %  74.6 TFLOPs   42.7 %   LoFi BF16 x BFP8 => BFP8
 68    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
 69    0.4 %         NlpCreateHeadsDeviceOperation                  0        35 μs          1 μs     18                                                      BFP8 => BFP8
 70    0.7 %         SDPAOperation                                  0        65 μs          1 μs     64                                                BFP8, BFP8 => BFP8
 71    0.2 %         NLPConcatHeadsDeviceOperation                  0        20 μs          1 μs     18                                                      BFP8 => BFP8
 72    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  91 GB/s  31.6 %  67.2 TFLOPs   38.5 %   LoFi BFP8 x BFP8 => BFP8
 73    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 74    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
 75    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.8 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
 76    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          2 μs     48  77 GB/s  26.8 %  88.9 TFLOPs   50.9 %   LoFi BFP8 x BFP8 => BFP8
 77    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 78    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
 79    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  66 GB/s  22.8 %  75.6 TFLOPs   43.3 %   LoFi BF16 x BFP8 => BFP8
 80    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
 81    0.4 %         NlpCreateHeadsDeviceOperation                  0        35 μs          1 μs     18                                                      BFP8 => BFP8
 82    0.7 %         SDPAOperation                                  0        68 μs          1 μs     64                                                BFP8, BFP8 => BFP8
 83    0.1 %         NLPConcatHeadsDeviceOperation                  0        13 μs          1 μs     18                                                      BFP8 => BFP8
 84    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  89 GB/s  31.1 %  66.0 TFLOPs   37.8 %   LoFi BFP8 x BFP8 => BFP8
 85    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 86    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
 87    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
 88    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.8 %  88.9 TFLOPs   50.9 %   LoFi BFP8 x BFP8 => BFP8
 89    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 90    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
 91    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          2 μs     48  65 GB/s  22.6 %  74.9 TFLOPs   42.9 %   LoFi BF16 x BFP8 => BFP8
 92    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
 93    0.4 %         NlpCreateHeadsDeviceOperation                  0        35 μs          1 μs     18                                                      BFP8 => BFP8
 94    0.7 %         SDPAOperation                                  0        65 μs          1 μs     64                                                BFP8, BFP8 => BFP8
 95    0.1 %         NLPConcatHeadsDeviceOperation                  0        13 μs          1 μs     18                                                      BFP8 => BFP8
 96    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  90 GB/s  31.2 %  66.2 TFLOPs   37.9 %   LoFi BFP8 x BFP8 => BFP8
 97    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 98    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
 99    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.7 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
100    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.7 %  88.7 TFLOPs   50.8 %   LoFi BFP8 x BFP8 => BFP8
101    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
102    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
103    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          2 μs     48  65 GB/s  22.6 %  75.0 TFLOPs   42.9 %   LoFi BF16 x BFP8 => BFP8
104    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
105    0.4 %         NlpCreateHeadsDeviceOperation                  0        35 μs          1 μs     18                                                      BFP8 => BFP8
106    0.7 %         SDPAOperation                                  0        65 μs          1 μs     64                                                BFP8, BFP8 => BFP8
107    0.1 %         NLPConcatHeadsDeviceOperation                  0        13 μs          1 μs     18                                                      BFP8 => BFP8
108    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        19 μs          1 μs     48  88 GB/s  30.6 %  65.0 TFLOPs   37.2 %   LoFi BFP8 x BFP8 => BFP8
109    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
110    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
111    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.8 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
112    0.7 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        62 μs          1 μs     48  68 GB/s  23.7 %  78.5 TFLOPs   45.0 %   LoFi BFP8 x BFP8 => BFP8
113    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
114    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
115    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        49 μs          1 μs     48  65 GB/s  22.5 %  74.7 TFLOPs   42.7 %   LoFi BF16 x BFP8 => BFP8
116    0.1 %         ShardedToInterleavedDeviceOperation            0         9 μs          1 μs     48                                                      BFP8 => BFP8
117    0.4 %         NlpCreateHeadsDeviceOperation                  0        35 μs          1 μs     18                                                      BFP8 => BFP8
118    0.7 %         SDPAOperation                                  0        66 μs          1 μs     64                                                BFP8, BFP8 => BFP8
119    0.1 %         NLPConcatHeadsDeviceOperation                  0        13 μs          1 μs     18                                                      BFP8 => BFP8
120    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  89 GB/s  30.8 %  65.4 TFLOPs   37.5 %   LoFi BFP8 x BFP8 => BFP8
121    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
122    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
123    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
124    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.7 %  88.7 TFLOPs   50.8 %   LoFi BFP8 x BFP8 => BFP8
125    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
126    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
127    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          2 μs     48  65 GB/s  22.7 %  75.3 TFLOPs   43.1 %   LoFi BF16 x BFP8 => BFP8
128    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
129    0.4 %         NlpCreateHeadsDeviceOperation                  0        35 μs          1 μs     18                                                      BFP8 => BFP8
130    0.7 %         SDPAOperation                                  0        65 μs          1 μs     64                                                BFP8, BFP8 => BFP8
131    0.1 %         NLPConcatHeadsDeviceOperation                  0        13 μs          1 μs     18                                                      BFP8 => BFP8
132    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        19 μs          1 μs     48  86 GB/s  29.8 %  63.3 TFLOPs   36.2 %   LoFi BFP8 x BFP8 => BFP8
133    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
134    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
135    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.8 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
136    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.5 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
137    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
138    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
139    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.5 %  74.8 TFLOPs   42.8 %   LoFi BF16 x BFP8 => BFP8
140    0.1 %         ShardedToInterleavedDeviceOperation            0         9 μs          1 μs     48                                                      BFP8 => BFP8
141    0.4 %         NlpCreateHeadsDeviceOperation                  0        35 μs          1 μs     18                                                      BFP8 => BFP8
142    0.7 %         SDPAOperation                                  0        65 μs          1 μs     64                                                BFP8, BFP8 => BFP8
143    0.1 %         NLPConcatHeadsDeviceOperation                  0        13 μs          1 μs     18                                                      BFP8 => BFP8
144    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        19 μs          1 μs     48  87 GB/s  30.2 %  64.2 TFLOPs   36.8 %   LoFi BFP8 x BFP8 => BFP8
145    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
146    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
147    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.8 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
148    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.8 %  88.8 TFLOPs   50.8 %   LoFi BFP8 x BFP8 => BFP8
149    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
150    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
151    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  66 GB/s  22.8 %  75.5 TFLOPs   43.2 %   LoFi BF16 x BFP8 => BFP8
152    0.1 %         ShardedToInterleavedDeviceOperation            0         9 μs          2 μs     48                                                      BFP8 => BFP8
153    0.4 %         NlpCreateHeadsDeviceOperation                  0        42 μs          1 μs     18                                                      BFP8 => BFP8
154    0.7 %         SDPAOperation                                  0        64 μs          1 μs     64                                                BFP8, BFP8 => BFP8
155    0.1 %         NLPConcatHeadsDeviceOperation                  0        14 μs          1 μs     18                                                      BFP8 => BFP8
156    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  90 GB/s  31.1 %  66.0 TFLOPs   37.8 %   LoFi BFP8 x BFP8 => BFP8
157    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
158    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
159    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
160    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.6 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
161    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
162    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
163    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        49 μs          1 μs     48  65 GB/s  22.5 %  74.7 TFLOPs   42.7 %   LoFi BF16 x BFP8 => BFP8
164    0.1 %         ShardedToInterleavedDeviceOperation            0         9 μs          1 μs     48                                                      BFP8 => BFP8
165    0.4 %         NlpCreateHeadsDeviceOperation                  0        35 μs          1 μs     18                                                      BFP8 => BFP8
166    0.7 %         SDPAOperation                                  0        65 μs          1 μs     64                                                BFP8, BFP8 => BFP8
167    0.1 %         NLPConcatHeadsDeviceOperation                  0        13 μs          1 μs     18                                                      BFP8 => BFP8
168    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  89 GB/s  31.1 %  66.0 TFLOPs   37.8 %   LoFi BFP8 x BFP8 => BFP8
169    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
170    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
171    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.7 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
172    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.8 %  88.9 TFLOPs   50.9 %   LoFi BFP8 x BFP8 => BFP8
173    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
174    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
175    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  75.0 TFLOPs   43.0 %   LoFi BF16 x BFP8 => BFP8
176    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
177    0.4 %         NlpCreateHeadsDeviceOperation                  0        35 μs          1 μs     18                                                      BFP8 => BFP8
178    0.7 %         SDPAOperation                                  0        65 μs          1 μs     64                                                BFP8, BFP8 => BFP8
179    0.1 %         NLPConcatHeadsDeviceOperation                  0        14 μs          1 μs     18                                                      BFP8 => BFP8
180    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  89 GB/s  30.8 %  65.3 TFLOPs   37.4 %   LoFi BFP8 x BFP8 => BFP8
181    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
182    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
183    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
184    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.5 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
185    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
186    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
187    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.5 %  74.8 TFLOPs   42.8 %   LoFi BF16 x BFP8 => BFP8
188    0.1 %         ShardedToInterleavedDeviceOperation            0         9 μs          1 μs     48                                                      BFP8 => BFP8
189    0.4 %         NlpCreateHeadsDeviceOperation                  0        35 μs          1 μs     18                                                      BFP8 => BFP8
190    0.7 %         SDPAOperation                                  0        68 μs          1 μs     64                                                BFP8, BFP8 => BFP8
191    0.2 %         NLPConcatHeadsDeviceOperation                  0        21 μs          1 μs     18                                                      BFP8 => BFP8
192    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  90 GB/s  31.2 %  66.3 TFLOPs   38.0 %   LoFi BFP8 x BFP8 => BFP8
193    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
194    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
195    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.7 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
196    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.6 %  88.2 TFLOPs   50.5 %   LoFi BFP8 x BFP8 => BFP8
197    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
198    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
199    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  66 GB/s  22.8 %  75.5 TFLOPs   43.2 %   LoFi BF16 x BFP8 => BFP8
200    0.1 %         ShardedToInterleavedDeviceOperation            0         9 μs          2 μs     48                                                      BFP8 => BFP8
201    0.4 %         NlpCreateHeadsDeviceOperation                  0        36 μs          1 μs     18                                                      BFP8 => BFP8
202    0.7 %         SDPAOperation                                  0        65 μs          1 μs     64                                                BFP8, BFP8 => BFP8
203    0.1 %         NLPConcatHeadsDeviceOperation                  0        13 μs          1 μs     18                                                      BFP8 => BFP8
204    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  89 GB/s  31.1 %  66.0 TFLOPs   37.8 %   LoFi BFP8 x BFP8 => BFP8
205    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
206    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
207    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.7 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
208    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  76 GB/s  26.5 %  88.0 TFLOPs   50.4 %   LoFi BFP8 x BFP8 => BFP8
209    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
210    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
211    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        49 μs          2 μs     48  65 GB/s  22.5 %  74.7 TFLOPs   42.8 %   LoFi BF16 x BFP8 => BFP8
212    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
213    0.4 %         NlpCreateHeadsDeviceOperation                  0        35 μs          1 μs     18                                                      BFP8 => BFP8
214    0.7 %         SDPAOperation                                  0        64 μs          1 μs     64                                                BFP8, BFP8 => BFP8
215    0.1 %         NLPConcatHeadsDeviceOperation                  0        13 μs          1 μs     18                                                      BFP8 => BFP8
216    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  89 GB/s  30.8 %  65.4 TFLOPs   37.5 %   LoFi BFP8 x BFP8 => BFP8
217    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
218    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
219    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.7 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
220    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.5 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
221    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
222    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
223    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.7 %  75.4 TFLOPs   43.2 %   LoFi BF16 x BFP8 => BFP8
224    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
225    0.4 %         NlpCreateHeadsDeviceOperation                  0        35 μs          1 μs     18                                                      BFP8 => BFP8
226    0.7 %         SDPAOperation                                  0        65 μs          1 μs     64                                                BFP8, BFP8 => BFP8
227    0.1 %         NLPConcatHeadsDeviceOperation                  0        13 μs          1 μs     18                                                      BFP8 => BFP8
228    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  90 GB/s  31.1 %  66.1 TFLOPs   37.9 %   LoFi BFP8 x BFP8 => BFP8
229    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
230    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
231    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        78 μs          1 μs     48  54 GB/s  18.7 %  61.9 TFLOPs   35.4 %   LoFi BF16 x BFP8 => BFP8
232    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  76 GB/s  26.6 %  88.1 TFLOPs   50.4 %   LoFi BFP8 x BFP8 => BFP8
233    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
234    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
235    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  75.0 TFLOPs   42.9 %   LoFi BF16 x BFP8 => BFP8
236    0.1 %         ShardedToInterleavedDeviceOperation            0         9 μs          1 μs     48                                                      BFP8 => BFP8
237    0.4 %         NlpCreateHeadsDeviceOperation                  0        35 μs          1 μs     18                                                      BFP8 => BFP8
238    0.7 %         SDPAOperation                                  0        65 μs          1 μs     64                                                BFP8, BFP8 => BFP8
239    0.1 %         NLPConcatHeadsDeviceOperation                  0        13 μs          1 μs     18                                                      BFP8 => BFP8
240    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        19 μs          1 μs     48  88 GB/s  30.4 %  64.6 TFLOPs   37.0 %   LoFi BFP8 x BFP8 => BFP8
241    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
242    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
243    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
244    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.8 %  89.0 TFLOPs   50.9 %   LoFi BFP8 x BFP8 => BFP8
245    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
246    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
247    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  75.0 TFLOPs   42.9 %   LoFi BF16 x BFP8 => BFP8
248    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
249    0.4 %         NlpCreateHeadsDeviceOperation                  0        35 μs          1 μs     18                                                      BFP8 => BFP8
250    0.7 %         SDPAOperation                                  0        65 μs          1 μs     64                                                BFP8, BFP8 => BFP8
251    0.1 %         NLPConcatHeadsDeviceOperation                  0        13 μs          1 μs     18                                                      BFP8 => BFP8
252    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        19 μs          1 μs     48  87 GB/s  30.3 %  64.3 TFLOPs   36.8 %   LoFi BFP8 x BFP8 => BFP8
253    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
254    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
255    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          2 μs     48  58 GB/s  20.1 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
256    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.7 %  88.7 TFLOPs   50.8 %   LoFi BFP8 x BFP8 => BFP8
257    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
258    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
259    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          2 μs     48  65 GB/s  22.6 %  75.0 TFLOPs   42.9 %   LoFi BF16 x BFP8 => BFP8
260    0.1 %         ShardedToInterleavedDeviceOperation            0         9 μs          1 μs     48                                                      BFP8 => BFP8
261    0.4 %         NlpCreateHeadsDeviceOperation                  0        35 μs          1 μs     18                                                      BFP8 => BFP8
262    0.7 %         SDPAOperation                                  0        68 μs          1 μs     64                                                BFP8, BFP8 => BFP8
263    0.1 %         NLPConcatHeadsDeviceOperation                  0        12 μs          1 μs     18                                                      BFP8 => BFP8
264    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        19 μs          1 μs     48  85 GB/s  29.4 %  62.5 TFLOPs   35.8 %   LoFi BFP8 x BFP8 => BFP8
265    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
266    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
267    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          2 μs     48  58 GB/s  20.1 %  66.7 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
268    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.6 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
269    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
270    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
271    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  75.1 TFLOPs   43.0 %   LoFi BF16 x BFP8 => BFP8
272    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
273    0.4 %         NlpCreateHeadsDeviceOperation                  0        37 μs          1 μs     18                                                      BFP8 => BFP8
274    0.7 %         SDPAOperation                                  0        65 μs          1 μs     64                                                BFP8, BFP8 => BFP8
275    0.1 %         NLPConcatHeadsDeviceOperation                  0        13 μs          1 μs     18                                                      BFP8 => BFP8
276    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        19 μs          1 μs     48  88 GB/s  30.4 %  64.6 TFLOPs   37.0 %   LoFi BFP8 x BFP8 => BFP8
277    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
278    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
279    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          2 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
280    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.6 %  88.4 TFLOPs   50.6 %   LoFi BFP8 x BFP8 => BFP8
281    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
282    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
283    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  75.0 TFLOPs   42.9 %   LoFi BF16 x BFP8 => BFP8
284    0.1 %         ShardedToInterleavedDeviceOperation            0         9 μs          1 μs     48                                                      BFP8 => BFP8
285    0.4 %         NlpCreateHeadsDeviceOperation                  0        36 μs          1 μs     18                                                      BFP8 => BFP8
286    0.7 %         SDPAOperation                                  0        66 μs          1 μs     64                                                BFP8, BFP8 => BFP8
287    0.1 %         NLPConcatHeadsDeviceOperation                  0        13 μs          1 μs     18                                                      BFP8 => BFP8
288    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        20 μs          1 μs     48  83 GB/s  28.8 %  61.1 TFLOPs   35.0 %   LoFi BFP8 x BFP8 => BFP8
289    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
290    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
291    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.7 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
292    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  76 GB/s  26.3 %  87.2 TFLOPs   49.9 %   LoFi BFP8 x BFP8 => BFP8
293    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
294    0.2 %         LayerNormDeviceOperation                       0        20 μs          1 μs     48                                                BF16, BF16 => BF16
295    2.4 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0       236 μs          1 μs     48  38 GB/s  13.1 %  20.5 TFLOPs   20.8 %  HiFi2 BF16 x BFP8 => BF16
296    3.4 %   SLOW  MatmulDeviceOperation 576 x 4096 x 4096        0       329 μs          1 μs     48  80 GB/s  27.7 %  58.8 TFLOPs   59.6 %  HiFi2 BF16 x BFP8 => BF16
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
     100.0 %         295 device ops, 0 host ops, 0 signposts              9,402 μs        292 μs         38 GB/s  13.1 %

💡 Advice 💡
============

Matmul Optimization
-------------------
  2    0.4 %   SLOW  MatmulDeviceOperation 576 x 768 x 1024         0        44 μs                   48  83 GB/s  29.0 %  20.8 TFLOPs   21.1 %  HiFi2 BF16 x BF16 => BF16
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=4 and output subblock 1x4 look good 🤷
- If your matmuls are not FLOP-bound use HiFi4 with BF16 activations for full accuracy

  7    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  66 GB/s  22.7 %  75.5 TFLOPs   43.2 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 12    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        19 μs          1 μs     48  88 GB/s  30.7 %  65.2 TFLOPs   37.3 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 15    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        73 μs          1 μs     48  58 GB/s  20.1 %  66.6 TFLOPs   38.1 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 16    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.7 %  88.7 TFLOPs   50.8 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 19    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  75.1 TFLOPs   43.0 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 24    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        19 μs          1 μs     48  87 GB/s  30.2 %  64.1 TFLOPs   36.7 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 27    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 28    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.7 %  88.7 TFLOPs   50.8 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 31    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.7 %  75.3 TFLOPs   43.1 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 36    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        19 μs          1 μs     48  88 GB/s  30.6 %  65.0 TFLOPs   37.2 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 39    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.7 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 40    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.6 %  88.2 TFLOPs   50.5 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 43    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.7 %  75.3 TFLOPs   43.1 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 48    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        19 μs          1 μs     48  86 GB/s  29.9 %  63.6 TFLOPs   36.4 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 51    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.8 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 52    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.7 %  88.7 TFLOPs   50.8 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 55    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.7 %  75.2 TFLOPs   43.0 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 60    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  90 GB/s  31.2 %  66.3 TFLOPs   38.0 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 63    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.7 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 64    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.8 %  88.8 TFLOPs   50.8 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 67    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        49 μs          1 μs     48  65 GB/s  22.5 %  74.6 TFLOPs   42.7 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 72    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  91 GB/s  31.6 %  67.2 TFLOPs   38.5 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 75    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.8 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 76    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          2 μs     48  77 GB/s  26.8 %  88.9 TFLOPs   50.9 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 79    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  66 GB/s  22.8 %  75.6 TFLOPs   43.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 84    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  89 GB/s  31.1 %  66.0 TFLOPs   37.8 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 87    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 88    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.8 %  88.9 TFLOPs   50.9 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 91    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          2 μs     48  65 GB/s  22.6 %  74.9 TFLOPs   42.9 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 96    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  90 GB/s  31.2 %  66.2 TFLOPs   37.9 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 99    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.7 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

100    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.7 %  88.7 TFLOPs   50.8 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

103    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          2 μs     48  65 GB/s  22.6 %  75.0 TFLOPs   42.9 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

108    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        19 μs          1 μs     48  88 GB/s  30.6 %  65.0 TFLOPs   37.2 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

111    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.8 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

112    0.7 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        62 μs          1 μs     48  68 GB/s  23.7 %  78.5 TFLOPs   45.0 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

115    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        49 μs          1 μs     48  65 GB/s  22.5 %  74.7 TFLOPs   42.7 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

120    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  89 GB/s  30.8 %  65.4 TFLOPs   37.5 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

123    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

124    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.7 %  88.7 TFLOPs   50.8 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

127    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          2 μs     48  65 GB/s  22.7 %  75.3 TFLOPs   43.1 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

132    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        19 μs          1 μs     48  86 GB/s  29.8 %  63.3 TFLOPs   36.2 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

135    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.8 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

136    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.5 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

139    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.5 %  74.8 TFLOPs   42.8 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

144    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        19 μs          1 μs     48  87 GB/s  30.2 %  64.2 TFLOPs   36.8 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

147    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.8 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

148    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.8 %  88.8 TFLOPs   50.8 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

151    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  66 GB/s  22.8 %  75.5 TFLOPs   43.2 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

156    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  90 GB/s  31.1 %  66.0 TFLOPs   37.8 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

159    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

160    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.6 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

163    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        49 μs          1 μs     48  65 GB/s  22.5 %  74.7 TFLOPs   42.7 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

168    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  89 GB/s  31.1 %  66.0 TFLOPs   37.8 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

171    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.7 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

172    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.8 %  88.9 TFLOPs   50.9 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

175    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  75.0 TFLOPs   43.0 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

180    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  89 GB/s  30.8 %  65.3 TFLOPs   37.4 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

183    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

184    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.5 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

187    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.5 %  74.8 TFLOPs   42.8 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

192    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  90 GB/s  31.2 %  66.3 TFLOPs   38.0 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

195    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.7 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

196    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.6 %  88.2 TFLOPs   50.5 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

199    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  66 GB/s  22.8 %  75.5 TFLOPs   43.2 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

204    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  89 GB/s  31.1 %  66.0 TFLOPs   37.8 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

207    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.7 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

208    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  76 GB/s  26.5 %  88.0 TFLOPs   50.4 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

211    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        49 μs          2 μs     48  65 GB/s  22.5 %  74.7 TFLOPs   42.8 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

216    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  89 GB/s  30.8 %  65.4 TFLOPs   37.5 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

219    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.7 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

220    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.5 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

223    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.7 %  75.4 TFLOPs   43.2 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

228    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  90 GB/s  31.1 %  66.1 TFLOPs   37.9 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

231    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        78 μs          1 μs     48  54 GB/s  18.7 %  61.9 TFLOPs   35.4 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

232    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  76 GB/s  26.6 %  88.1 TFLOPs   50.4 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

235    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  75.0 TFLOPs   42.9 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

240    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        19 μs          1 μs     48  88 GB/s  30.4 %  64.6 TFLOPs   37.0 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

243    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

244    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.8 %  89.0 TFLOPs   50.9 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

247    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  75.0 TFLOPs   42.9 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

252    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        19 μs          1 μs     48  87 GB/s  30.3 %  64.3 TFLOPs   36.8 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

255    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          2 μs     48  58 GB/s  20.1 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

256    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.7 %  88.7 TFLOPs   50.8 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

259    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          2 μs     48  65 GB/s  22.6 %  75.0 TFLOPs   42.9 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

264    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        19 μs          1 μs     48  85 GB/s  29.4 %  62.5 TFLOPs   35.8 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

267    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          2 μs     48  58 GB/s  20.1 %  66.7 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

268    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.6 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

271    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  75.1 TFLOPs   43.0 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

276    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        19 μs          1 μs     48  88 GB/s  30.4 %  64.6 TFLOPs   37.0 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

279    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          2 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

280    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.6 %  88.4 TFLOPs   50.6 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

283    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  75.0 TFLOPs   42.9 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

288    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        20 μs          1 μs     48  83 GB/s  28.8 %  61.1 TFLOPs   35.0 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

291    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.7 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

292    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  76 GB/s  26.3 %  87.2 TFLOPs   49.9 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

295    2.4 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0       236 μs          1 μs     48  38 GB/s  13.1 %  20.5 TFLOPs   20.8 %  HiFi2 BF16 x BFP8 => BF16
- in0_block_w=4 and output subblock 1x4 look good 🤷
- If your matmuls are not FLOP-bound use HiFi4 with BF16 activations for full accuracy

296    3.4 %   SLOW  MatmulDeviceOperation 576 x 4096 x 4096        0       329 μs          1 μs     48  80 GB/s  27.7 %  58.8 TFLOPs   59.6 %  HiFi2 BF16 x BFP8 => BF16
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=4 and output subblock 1x4 look good 🤷
- If your matmuls are not FLOP-bound use HiFi4 with BF16 activations for full accuracy


📊 Stacked report 📊
====================

Total %  Op Code                                                     Device Time Sum  Op Count  Op Category  Min FLOPs  Max FLOPs  Mean FLOPs  Std FLOPs  Weighted Mean FLOPs
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
47.37 %  MatmulDeviceOperation (in0:block_sharded)                       4,453.32 μs        73      Compute    20.75 %    50.93 %     43.53 %     5.80 %              42.10 %
16.72 %  SDPAOperation (in0:l1_interleaved)                              1,571.99 μs        24      Compute
 9.90 %  LayerNormDeviceOperation (in0:block_sharded)                      931.24 μs        49      Compute
 9.19 %  NlpCreateHeadsDeviceOperation (in0:l1_interleaved)                863.64 μs        24           TM
 8.71 %  MatmulDeviceOperation (in0:dram_interleaved)                      818.72 μs        26      Compute    21.06 %    59.58 %     37.44 %     5.56 %              45.32 %
 3.50 %  NLPConcatHeadsDeviceOperation (in0:l1_interleaved)                329.05 μs        24           TM
 2.43 %  ShardedToInterleavedDeviceOperation (in0:block_sharded)           228.55 μs        24           DM
 1.51 %  BinaryNgDeviceOperation (in0:block_sharded)                       142.16 μs        47      Compute
 0.59 %  BinaryNgDeviceOperation (in0:dram_interleaved)                     55.43 μs         3      Compute
 0.08 %  InterleavedToShardedDeviceOperation (in0:dram_interleaved)          7.80 μs         1           DM
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```
