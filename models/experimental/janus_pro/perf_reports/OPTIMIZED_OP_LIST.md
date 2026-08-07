# tt-perf-report, one replay of the optimized tower

`tt-perf-report` over a single trace replay of the tower at **9.841 ms** — the op table, its advice
section and its summary, unabridged. Commit `5ea2daf4c8c` (change 26), Wormhole N150, 295 ops.

One replay of the ten the perf test measures, picked as the one whose kernel time sits closest to
their mean rather than the fastest: 9.840 ms against the 9.841 the ten average to.

```
🚀 Performance Report 🚀
========================

ID   Total %  Bound  OP Code                                  Device   Device Time  Op-to-Op Gap  Cores  DRAM     DRAM %  FLOPs        FLOPs %  Math Fidelity
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  2    0.4 %   SLOW  MatmulDeviceOperation 576 x 768 x 1024         0        44 μs                   48  83 GB/s  28.8 %  20.7 TFLOPs   21.0 %  HiFi2 BF16 x BF16 => BF16
  3    0.2 %         BinaryNgDeviceOperation                        0        21 μs          1 μs     64                                                BF16, BF16 => BF16
  4    0.2 %         BinaryNgDeviceOperation                        0        22 μs          1 μs     64                                                BF16, BF16 => BF16
  5    0.1 %         InterleavedToShardedDeviceOperation            0         8 μs          1 μs     48                                                      BF16 => BF16
  6    0.2 %         LayerNormDeviceOperation                       0        20 μs          1 μs     48                                                BF16, BF16 => BF16
  7    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  66 GB/s  22.7 %  75.5 TFLOPs   43.2 %   LoFi BF16 x BFP8 => BFP8
  8    0.1 %         ShardedToInterleavedDeviceOperation            0         9 μs          1 μs     48                                                      BFP8 => BFP8
  9    0.5 %         NlpCreateHeadsDeviceOperation                  0        48 μs          1 μs     18                                                      BFP8 => BFP8
 10    0.7 %         SDPAOperation                                  0        67 μs          1 μs     64                                                BFP8, BFP8 => BFP8
 11    0.2 %         NLPConcatHeadsDeviceOperation                  0        18 μs          1 μs     18                                                      BFP8 => BFP8
 12    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  89 GB/s  30.8 %  65.3 TFLOPs   37.4 %   LoFi BFP8 x BFP8 => BFP8
 13    0.1 %         BinaryNgDeviceOperation                        0        12 μs          1 μs     64                                                BF16, BFP8 => BF16
 14    0.2 %         LayerNormDeviceOperation                       0        20 μs          1 μs     48                                                BF16, BF16 => BF16
 15    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        71 μs          1 μs     48  59 GB/s  20.4 %  67.6 TFLOPs   38.7 %   LoFi BF16 x BFP8 => BFP8
 16    0.5 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          2 μs     48  78 GB/s  26.9 %  89.3 TFLOPs   51.1 %   LoFi BFP8 x BFP8 => BFP8
 17    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 18    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
 19    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  66 GB/s  22.7 %  75.5 TFLOPs   43.2 %   LoFi BF16 x BFP8 => BFP8
 20    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
 21    0.5 %         NlpCreateHeadsDeviceOperation                  0        48 μs          1 μs     18                                                      BFP8 => BFP8
 22    0.7 %         SDPAOperation                                  0        67 μs          1 μs     64                                                BFP8, BFP8 => BFP8
 23    0.2 %         NLPConcatHeadsDeviceOperation                  0        17 μs          1 μs     18                                                      BFP8 => BFP8
 24    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  91 GB/s  31.6 %  67.1 TFLOPs   38.4 %   LoFi BFP8 x BFP8 => BFP8
 25    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 26    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
 27    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  67.0 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
 28    0.5 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.8 %  88.9 TFLOPs   50.9 %   LoFi BFP8 x BFP8 => BFP8
 29    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 30    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
 31    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  66 GB/s  22.8 %  75.6 TFLOPs   43.3 %   LoFi BF16 x BFP8 => BFP8
 32    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
 33    0.5 %         NlpCreateHeadsDeviceOperation                  0        53 μs          1 μs     18                                                      BFP8 => BFP8
 34    0.7 %         SDPAOperation                                  0        67 μs          1 μs     64                                                BFP8, BFP8 => BFP8
 35    0.2 %         NLPConcatHeadsDeviceOperation                  0        18 μs          1 μs     18                                                      BFP8 => BFP8
 36    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  90 GB/s  31.3 %  66.5 TFLOPs   38.1 %   LoFi BFP8 x BFP8 => BFP8
 37    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 38    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
 39    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
 40    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.6 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
 41    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 42    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
 43    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.7 %  75.2 TFLOPs   43.1 %   LoFi BF16 x BFP8 => BFP8
 44    0.1 %         ShardedToInterleavedDeviceOperation            0        11 μs          1 μs     48                                                      BFP8 => BFP8
 45    0.5 %         NlpCreateHeadsDeviceOperation                  0        48 μs          1 μs     18                                                      BFP8 => BFP8
 46    0.7 %         SDPAOperation                                  0        67 μs          1 μs     64                                                BFP8, BFP8 => BFP8
 47    0.2 %         NLPConcatHeadsDeviceOperation                  0        18 μs          1 μs     18                                                      BFP8 => BFP8
 48    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  90 GB/s  31.4 %  66.6 TFLOPs   38.1 %   LoFi BFP8 x BFP8 => BFP8
 49    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 50    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
 51    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
 52    0.5 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.5 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
 53    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 54    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
 55    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.7 %  75.4 TFLOPs   43.2 %   LoFi BF16 x BFP8 => BFP8
 56    0.1 %         ShardedToInterleavedDeviceOperation            0         9 μs          1 μs     48                                                      BFP8 => BFP8
 57    0.5 %         NlpCreateHeadsDeviceOperation                  0        49 μs          1 μs     18                                                      BFP8 => BFP8
 58    0.7 %         SDPAOperation                                  0        67 μs          1 μs     64                                                BFP8, BFP8 => BFP8
 59    0.2 %         NLPConcatHeadsDeviceOperation                  0        18 μs          1 μs     18                                                      BFP8 => BFP8
 60    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  90 GB/s  31.2 %  66.2 TFLOPs   37.9 %   LoFi BFP8 x BFP8 => BFP8
 61    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 62    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
 63    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
 64    0.5 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.8 %  89.1 TFLOPs   51.0 %   LoFi BFP8 x BFP8 => BFP8
 65    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 66    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
 67    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          2 μs     48  65 GB/s  22.6 %  75.0 TFLOPs   43.0 %   LoFi BF16 x BFP8 => BFP8
 68    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
 69    0.5 %         NlpCreateHeadsDeviceOperation                  0        48 μs          1 μs     18                                                      BFP8 => BFP8
 70    0.7 %         SDPAOperation                                  0        68 μs          1 μs     64                                                BFP8, BFP8 => BFP8
 71    0.2 %         NLPConcatHeadsDeviceOperation                  0        24 μs          1 μs     18                                                      BFP8 => BFP8
 72    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  90 GB/s  31.3 %  66.4 TFLOPs   38.0 %   LoFi BFP8 x BFP8 => BFP8
 73    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 74    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
 75    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.8 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
 76    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.8 %  89.0 TFLOPs   50.9 %   LoFi BFP8 x BFP8 => BFP8
 77    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 78    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
 79    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  66 GB/s  22.8 %  75.6 TFLOPs   43.3 %   LoFi BF16 x BFP8 => BFP8
 80    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
 81    0.5 %         NlpCreateHeadsDeviceOperation                  0        49 μs          1 μs     18                                                      BFP8 => BFP8
 82    0.7 %         SDPAOperation                                  0        67 μs          1 μs     64                                                BFP8, BFP8 => BFP8
 83    0.2 %         NLPConcatHeadsDeviceOperation                  0        18 μs          1 μs     18                                                      BFP8 => BFP8
 84    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  90 GB/s  31.2 %  66.2 TFLOPs   37.9 %   LoFi BFP8 x BFP8 => BFP8
 85    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 86    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
 87    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  67.0 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
 88    0.5 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.8 %  89.1 TFLOPs   51.0 %   LoFi BFP8 x BFP8 => BFP8
 89    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 90    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
 91    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  75.1 TFLOPs   43.0 %   LoFi BF16 x BFP8 => BFP8
 92    0.1 %         ShardedToInterleavedDeviceOperation            0         9 μs          1 μs     48                                                      BFP8 => BFP8
 93    0.5 %         NlpCreateHeadsDeviceOperation                  0        47 μs          1 μs     18                                                      BFP8 => BFP8
 94    0.7 %         SDPAOperation                                  0        67 μs          1 μs     64                                                BFP8, BFP8 => BFP8
 95    0.2 %         NLPConcatHeadsDeviceOperation                  0        18 μs          1 μs     18                                                      BFP8 => BFP8
 96    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  90 GB/s  31.3 %  66.4 TFLOPs   38.0 %   LoFi BFP8 x BFP8 => BFP8
 97    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 98    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
 99    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.7 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
100    0.5 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.9 %  89.1 TFLOPs   51.0 %   LoFi BFP8 x BFP8 => BFP8
101    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
102    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
103    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  75.1 TFLOPs   43.0 %   LoFi BF16 x BFP8 => BFP8
104    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
105    0.5 %         NlpCreateHeadsDeviceOperation                  0        48 μs          1 μs     18                                                      BFP8 => BFP8
106    0.7 %         SDPAOperation                                  0        67 μs          1 μs     64                                                BFP8, BFP8 => BFP8
107    0.2 %         NLPConcatHeadsDeviceOperation                  0        18 μs          1 μs     18                                                      BFP8 => BFP8
108    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        19 μs          1 μs     48  88 GB/s  30.5 %  64.9 TFLOPs   37.1 %   LoFi BFP8 x BFP8 => BFP8
109    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
110    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
111    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        80 μs          1 μs     48  53 GB/s  18.3 %  60.8 TFLOPs   34.8 %   LoFi BF16 x BFP8 => BFP8
112    0.5 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  76 GB/s  26.5 %  88.0 TFLOPs   50.4 %   LoFi BFP8 x BFP8 => BFP8
113    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
114    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
115    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        49 μs          1 μs     48  65 GB/s  22.5 %  74.6 TFLOPs   42.7 %   LoFi BF16 x BFP8 => BFP8
116    0.1 %         ShardedToInterleavedDeviceOperation            0         9 μs          1 μs     48                                                      BFP8 => BFP8
117    0.5 %         NlpCreateHeadsDeviceOperation                  0        48 μs          1 μs     18                                                      BFP8 => BFP8
118    0.7 %         SDPAOperation                                  0        66 μs          1 μs     64                                                BFP8, BFP8 => BFP8
119    0.2 %         NLPConcatHeadsDeviceOperation                  0        18 μs          1 μs     18                                                      BFP8 => BFP8
120    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  89 GB/s  31.0 %  65.9 TFLOPs   37.7 %   LoFi BFP8 x BFP8 => BFP8
121    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
122    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
123    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.8 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
124    0.5 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.8 %  88.8 TFLOPs   50.8 %   LoFi BFP8 x BFP8 => BFP8
125    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
126    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
127    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  66 GB/s  22.7 %  75.5 TFLOPs   43.2 %   LoFi BF16 x BFP8 => BFP8
128    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
129    0.5 %         NlpCreateHeadsDeviceOperation                  0        48 μs          1 μs     18                                                      BFP8 => BFP8
130    0.7 %         SDPAOperation                                  0        67 μs          1 μs     64                                                BFP8, BFP8 => BFP8
131    0.2 %         NLPConcatHeadsDeviceOperation                  0        18 μs          1 μs     18                                                      BFP8 => BFP8
132    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  89 GB/s  30.8 %  65.3 TFLOPs   37.4 %   LoFi BFP8 x BFP8 => BFP8
133    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
134    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
135    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
136    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.6 %  88.2 TFLOPs   50.5 %   LoFi BFP8 x BFP8 => BFP8
137    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
138    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
139    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.5 %  74.8 TFLOPs   42.8 %   LoFi BF16 x BFP8 => BFP8
140    0.1 %         ShardedToInterleavedDeviceOperation            0         9 μs          1 μs     48                                                      BFP8 => BFP8
141    0.5 %         NlpCreateHeadsDeviceOperation                  0        48 μs          1 μs     18                                                      BFP8 => BFP8
142    0.7 %         SDPAOperation                                  0        67 μs          1 μs     64                                                BFP8, BFP8 => BFP8
143    0.2 %         NLPConcatHeadsDeviceOperation                  0        18 μs          1 μs     18                                                      BFP8 => BFP8
144    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  91 GB/s  31.5 %  67.0 TFLOPs   38.3 %   LoFi BFP8 x BFP8 => BFP8
145    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
146    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
147    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
148    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.8 %  88.8 TFLOPs   50.8 %   LoFi BFP8 x BFP8 => BFP8
149    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
150    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
151    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  66 GB/s  22.8 %  75.5 TFLOPs   43.2 %   LoFi BF16 x BFP8 => BFP8
152    0.1 %         ShardedToInterleavedDeviceOperation            0         9 μs          1 μs     48                                                      BFP8 => BFP8
153    0.5 %         NlpCreateHeadsDeviceOperation                  0        53 μs          1 μs     18                                                      BFP8 => BFP8
154    0.7 %         SDPAOperation                                  0        67 μs          1 μs     64                                                BFP8, BFP8 => BFP8
155    0.2 %         NLPConcatHeadsDeviceOperation                  0        18 μs          1 μs     18                                                      BFP8 => BFP8
156    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  91 GB/s  31.5 %  67.0 TFLOPs   38.3 %   LoFi BFP8 x BFP8 => BFP8
157    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
158    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
159    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.7 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
160    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.6 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
161    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
162    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
163    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  75.0 TFLOPs   42.9 %   LoFi BF16 x BFP8 => BFP8
164    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
165    0.5 %         NlpCreateHeadsDeviceOperation                  0        48 μs          1 μs     18                                                      BFP8 => BFP8
166    0.7 %         SDPAOperation                                  0        66 μs          1 μs     64                                                BFP8, BFP8 => BFP8
167    0.2 %         NLPConcatHeadsDeviceOperation                  0        18 μs          1 μs     18                                                      BFP8 => BFP8
168    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        19 μs          1 μs     48  88 GB/s  30.6 %  65.1 TFLOPs   37.3 %   LoFi BFP8 x BFP8 => BFP8
169    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
170    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
171    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          2 μs     48  58 GB/s  20.1 %  66.6 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
172    0.5 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.6 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
173    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
174    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
175    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  66 GB/s  22.7 %  75.5 TFLOPs   43.2 %   LoFi BF16 x BFP8 => BFP8
176    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
177    0.5 %         NlpCreateHeadsDeviceOperation                  0        48 μs          1 μs     18                                                      BFP8 => BFP8
178    0.7 %         SDPAOperation                                  0        67 μs          1 μs     64                                                BFP8, BFP8 => BFP8
179    0.2 %         NLPConcatHeadsDeviceOperation                  0        18 μs          1 μs     18                                                      BFP8 => BFP8
180    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  90 GB/s  31.3 %  66.6 TFLOPs   38.1 %   LoFi BFP8 x BFP8 => BFP8
181    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
182    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
183    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.8 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
184    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.6 %  88.4 TFLOPs   50.6 %   LoFi BFP8 x BFP8 => BFP8
185    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
186    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
187    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  75.1 TFLOPs   43.0 %   LoFi BF16 x BFP8 => BFP8
188    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
189    0.5 %         NlpCreateHeadsDeviceOperation                  0        48 μs          1 μs     18                                                      BFP8 => BFP8
190    0.7 %         SDPAOperation                                  0        67 μs          1 μs     64                                                BFP8, BFP8 => BFP8
191    0.2 %         NLPConcatHeadsDeviceOperation                  0        18 μs          1 μs     18                                                      BFP8 => BFP8
192    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  89 GB/s  31.0 %  65.9 TFLOPs   37.7 %   LoFi BFP8 x BFP8 => BFP8
193    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
194    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
195    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        73 μs          1 μs     48  57 GB/s  19.9 %  65.9 TFLOPs   37.8 %   LoFi BF16 x BFP8 => BFP8
196    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.5 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
197    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
198    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
199    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.7 %  75.4 TFLOPs   43.1 %   LoFi BF16 x BFP8 => BFP8
200    0.1 %         ShardedToInterleavedDeviceOperation            0         9 μs          1 μs     48                                                      BFP8 => BFP8
201    0.5 %         NlpCreateHeadsDeviceOperation                  0        48 μs          1 μs     18                                                      BFP8 => BFP8
202    0.7 %         SDPAOperation                                  0        68 μs          1 μs     64                                                BFP8, BFP8 => BFP8
203    0.2 %         NLPConcatHeadsDeviceOperation                  0        18 μs          1 μs     18                                                      BFP8 => BFP8
204    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  89 GB/s  30.9 %  65.6 TFLOPs   37.6 %   LoFi BFP8 x BFP8 => BFP8
205    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
206    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
207    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.8 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
208    0.5 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.8 %  88.9 TFLOPs   50.9 %   LoFi BFP8 x BFP8 => BFP8
209    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
210    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
211    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.7 %  75.2 TFLOPs   43.0 %   LoFi BF16 x BFP8 => BFP8
212    0.1 %         ShardedToInterleavedDeviceOperation            0         9 μs          2 μs     48                                                      BFP8 => BFP8
213    0.5 %         NlpCreateHeadsDeviceOperation                  0        48 μs          1 μs     18                                                      BFP8 => BFP8
214    0.7 %         SDPAOperation                                  0        67 μs          1 μs     64                                                BFP8, BFP8 => BFP8
215    0.2 %         NLPConcatHeadsDeviceOperation                  0        18 μs          1 μs     18                                                      BFP8 => BFP8
216    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        19 μs          1 μs     48  88 GB/s  30.5 %  64.8 TFLOPs   37.1 %   LoFi BFP8 x BFP8 => BFP8
217    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
218    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
219    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.7 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
220    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.5 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
221    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
222    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
223    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  75.1 TFLOPs   43.0 %   LoFi BF16 x BFP8 => BFP8
224    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
225    0.5 %         NlpCreateHeadsDeviceOperation                  0        48 μs          1 μs     18                                                      BFP8 => BFP8
226    0.7 %         SDPAOperation                                  0        67 μs          1 μs     64                                                BFP8, BFP8 => BFP8
227    0.2 %         NLPConcatHeadsDeviceOperation                  0        18 μs          1 μs     18                                                      BFP8 => BFP8
228    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  89 GB/s  31.0 %  65.9 TFLOPs   37.7 %   LoFi BFP8 x BFP8 => BFP8
229    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
230    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
231    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        73 μs          1 μs     48  58 GB/s  20.1 %  66.5 TFLOPs   38.1 %   LoFi BF16 x BFP8 => BFP8
232    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        61 μs          1 μs     48  69 GB/s  23.9 %  79.4 TFLOPs   45.5 %   LoFi BFP8 x BFP8 => BFP8
233    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
234    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
235    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.5 %  74.7 TFLOPs   42.8 %   LoFi BF16 x BFP8 => BFP8
236    0.1 %         ShardedToInterleavedDeviceOperation            0         9 μs          1 μs     48                                                      BFP8 => BFP8
237    0.5 %         NlpCreateHeadsDeviceOperation                  0        48 μs          1 μs     18                                                      BFP8 => BFP8
238    0.7 %         SDPAOperation                                  0        67 μs          1 μs     64                                                BFP8, BFP8 => BFP8
239    0.2 %         NLPConcatHeadsDeviceOperation                  0        18 μs          1 μs     18                                                      BFP8 => BFP8
240    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  90 GB/s  31.3 %  66.5 TFLOPs   38.0 %   LoFi BFP8 x BFP8 => BFP8
241    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
242    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
243    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
244    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.7 %  88.7 TFLOPs   50.8 %   LoFi BFP8 x BFP8 => BFP8
245    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
246    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
247    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  66 GB/s  22.8 %  75.5 TFLOPs   43.2 %   LoFi BF16 x BFP8 => BFP8
248    0.1 %         ShardedToInterleavedDeviceOperation            0         9 μs          1 μs     48                                                      BFP8 => BFP8
249    0.5 %         NlpCreateHeadsDeviceOperation                  0        48 μs          1 μs     18                                                      BFP8 => BFP8
250    0.7 %         SDPAOperation                                  0        67 μs          1 μs     64                                                BFP8, BFP8 => BFP8
251    0.2 %         NLPConcatHeadsDeviceOperation                  0        18 μs          1 μs     18                                                      BFP8 => BFP8
252    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  90 GB/s  31.4 %  66.6 TFLOPs   38.1 %   LoFi BFP8 x BFP8 => BFP8
253    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
254    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
255    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
256    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.7 %  88.7 TFLOPs   50.8 %   LoFi BFP8 x BFP8 => BFP8
257    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
258    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
259    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  74.9 TFLOPs   42.9 %   LoFi BF16 x BFP8 => BFP8
260    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
261    0.5 %         NlpCreateHeadsDeviceOperation                  0        48 μs          1 μs     18                                                      BFP8 => BFP8
262    0.7 %         SDPAOperation                                  0        67 μs          1 μs     64                                                BFP8, BFP8 => BFP8
263    0.2 %         NLPConcatHeadsDeviceOperation                  0        18 μs          1 μs     18                                                      BFP8 => BFP8
264    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  89 GB/s  31.0 %  65.8 TFLOPs   37.7 %   LoFi BFP8 x BFP8 => BFP8
265    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
266    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
267    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          2 μs     48  58 GB/s  20.1 %  66.8 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
268    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.5 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
269    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
270    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
271    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.7 %  75.3 TFLOPs   43.1 %   LoFi BF16 x BFP8 => BFP8
272    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
273    0.5 %         NlpCreateHeadsDeviceOperation                  0        49 μs          1 μs     18                                                      BFP8 => BFP8
274    0.7 %         SDPAOperation                                  0        67 μs          1 μs     64                                                BFP8, BFP8 => BFP8
275    0.2 %         NLPConcatHeadsDeviceOperation                  0        18 μs          1 μs     18                                                      BFP8 => BFP8
276    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  90 GB/s  31.1 %  66.1 TFLOPs   37.9 %   LoFi BFP8 x BFP8 => BFP8
277    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
278    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
279    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.8 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
280    0.5 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.8 %  88.9 TFLOPs   50.9 %   LoFi BFP8 x BFP8 => BFP8
281    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
282    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
283    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  74.9 TFLOPs   42.9 %   LoFi BF16 x BFP8 => BFP8
284    0.1 %         ShardedToInterleavedDeviceOperation            0         9 μs          1 μs     48                                                      BFP8 => BFP8
285    0.5 %         NlpCreateHeadsDeviceOperation                  0        49 μs          1 μs     18                                                      BFP8 => BFP8
286    0.7 %         SDPAOperation                                  0        67 μs          1 μs     64                                                BFP8, BFP8 => BFP8
287    0.2 %         NLPConcatHeadsDeviceOperation                  0        18 μs          1 μs     18                                                      BFP8 => BFP8
288    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        19 μs          1 μs     48  86 GB/s  29.9 %  63.5 TFLOPs   36.4 %   LoFi BFP8 x BFP8 => BFP8
289    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
290    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
291    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.8 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
292    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  76 GB/s  26.2 %  87.1 TFLOPs   49.8 %   LoFi BFP8 x BFP8 => BFP8
293    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
294    0.2 %         LayerNormDeviceOperation                       0        20 μs          1 μs     48                                                BF16, BF16 => BF16
295    2.3 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0       236 μs          1 μs     48  38 GB/s  13.1 %  20.5 TFLOPs   20.7 %  HiFi2 BF16 x BFP8 => BF16
296    3.3 %   SLOW  MatmulDeviceOperation 576 x 4096 x 4096        0       330 μs          1 μs     48  79 GB/s  27.6 %  58.5 TFLOPs   59.3 %  HiFi2 BF16 x BFP8 => BF16
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
     100.0 %         295 device ops, 0 host ops, 0 signposts              9,840 μs        291 μs         36 GB/s  12.5 %

💡 Advice 💡
============

Matmul Optimization
-------------------
  2    0.4 %   SLOW  MatmulDeviceOperation 576 x 768 x 1024         0        44 μs                   48  83 GB/s  28.8 %  20.7 TFLOPs   21.0 %  HiFi2 BF16 x BF16 => BF16
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=4 and output subblock 1x4 look good 🤷
- If your matmuls are not FLOP-bound use HiFi4 with BF16 activations for full accuracy

  7    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  66 GB/s  22.7 %  75.5 TFLOPs   43.2 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 12    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  89 GB/s  30.8 %  65.3 TFLOPs   37.4 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 15    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        71 μs          1 μs     48  59 GB/s  20.4 %  67.6 TFLOPs   38.7 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 16    0.5 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          2 μs     48  78 GB/s  26.9 %  89.3 TFLOPs   51.1 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 19    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  66 GB/s  22.7 %  75.5 TFLOPs   43.2 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 24    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  91 GB/s  31.6 %  67.1 TFLOPs   38.4 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 27    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  67.0 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 28    0.5 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.8 %  88.9 TFLOPs   50.9 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 31    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  66 GB/s  22.8 %  75.6 TFLOPs   43.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 36    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  90 GB/s  31.3 %  66.5 TFLOPs   38.1 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 39    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 40    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.6 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 43    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.7 %  75.2 TFLOPs   43.1 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 48    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  90 GB/s  31.4 %  66.6 TFLOPs   38.1 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 51    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 52    0.5 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.5 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 55    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.7 %  75.4 TFLOPs   43.2 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 60    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  90 GB/s  31.2 %  66.2 TFLOPs   37.9 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 63    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 64    0.5 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.8 %  89.1 TFLOPs   51.0 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 67    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          2 μs     48  65 GB/s  22.6 %  75.0 TFLOPs   43.0 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 72    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  90 GB/s  31.3 %  66.4 TFLOPs   38.0 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 75    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.8 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 76    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.8 %  89.0 TFLOPs   50.9 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 79    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  66 GB/s  22.8 %  75.6 TFLOPs   43.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 84    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  90 GB/s  31.2 %  66.2 TFLOPs   37.9 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 87    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  67.0 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 88    0.5 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.8 %  89.1 TFLOPs   51.0 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 91    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  75.1 TFLOPs   43.0 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 96    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  90 GB/s  31.3 %  66.4 TFLOPs   38.0 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 99    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.7 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

100    0.5 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.9 %  89.1 TFLOPs   51.0 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

103    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  75.1 TFLOPs   43.0 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

108    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        19 μs          1 μs     48  88 GB/s  30.5 %  64.9 TFLOPs   37.1 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

111    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        80 μs          1 μs     48  53 GB/s  18.3 %  60.8 TFLOPs   34.8 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

112    0.5 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  76 GB/s  26.5 %  88.0 TFLOPs   50.4 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

115    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        49 μs          1 μs     48  65 GB/s  22.5 %  74.6 TFLOPs   42.7 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

120    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  89 GB/s  31.0 %  65.9 TFLOPs   37.7 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

123    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.8 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

124    0.5 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.8 %  88.8 TFLOPs   50.8 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

127    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  66 GB/s  22.7 %  75.5 TFLOPs   43.2 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

132    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  89 GB/s  30.8 %  65.3 TFLOPs   37.4 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

135    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

136    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.6 %  88.2 TFLOPs   50.5 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

139    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.5 %  74.8 TFLOPs   42.8 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

144    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  91 GB/s  31.5 %  67.0 TFLOPs   38.3 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

147    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

148    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.8 %  88.8 TFLOPs   50.8 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

151    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  66 GB/s  22.8 %  75.5 TFLOPs   43.2 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

156    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  91 GB/s  31.5 %  67.0 TFLOPs   38.3 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

159    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.7 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

160    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.6 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

163    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  75.0 TFLOPs   42.9 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

168    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        19 μs          1 μs     48  88 GB/s  30.6 %  65.1 TFLOPs   37.3 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

171    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          2 μs     48  58 GB/s  20.1 %  66.6 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

172    0.5 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.6 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

175    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  66 GB/s  22.7 %  75.5 TFLOPs   43.2 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

180    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  90 GB/s  31.3 %  66.6 TFLOPs   38.1 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

183    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.8 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

184    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.6 %  88.4 TFLOPs   50.6 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

187    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  75.1 TFLOPs   43.0 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

192    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  89 GB/s  31.0 %  65.9 TFLOPs   37.7 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

195    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        73 μs          1 μs     48  57 GB/s  19.9 %  65.9 TFLOPs   37.8 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

196    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.5 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

199    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.7 %  75.4 TFLOPs   43.1 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

204    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  89 GB/s  30.9 %  65.6 TFLOPs   37.6 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

207    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.8 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

208    0.5 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.8 %  88.9 TFLOPs   50.9 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

211    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.7 %  75.2 TFLOPs   43.0 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

216    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        19 μs          1 μs     48  88 GB/s  30.5 %  64.8 TFLOPs   37.1 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

219    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.7 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

220    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.5 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

223    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  75.1 TFLOPs   43.0 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

228    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  89 GB/s  31.0 %  65.9 TFLOPs   37.7 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

231    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        73 μs          1 μs     48  58 GB/s  20.1 %  66.5 TFLOPs   38.1 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

232    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        61 μs          1 μs     48  69 GB/s  23.9 %  79.4 TFLOPs   45.5 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

235    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.5 %  74.7 TFLOPs   42.8 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

240    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  90 GB/s  31.3 %  66.5 TFLOPs   38.0 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

243    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

244    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.7 %  88.7 TFLOPs   50.8 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

247    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  66 GB/s  22.8 %  75.5 TFLOPs   43.2 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

252    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  90 GB/s  31.4 %  66.6 TFLOPs   38.1 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

255    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

256    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.7 %  88.7 TFLOPs   50.8 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

259    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  74.9 TFLOPs   42.9 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

264    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  89 GB/s  31.0 %  65.8 TFLOPs   37.7 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

267    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          2 μs     48  58 GB/s  20.1 %  66.8 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

268    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.5 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

271    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.7 %  75.3 TFLOPs   43.1 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

276    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  90 GB/s  31.1 %  66.1 TFLOPs   37.9 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

279    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.8 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

280    0.5 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.8 %  88.9 TFLOPs   50.9 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

283    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  74.9 TFLOPs   42.9 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

288    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        19 μs          1 μs     48  86 GB/s  29.9 %  63.5 TFLOPs   36.4 %   LoFi BFP8 x BFP8 => BFP8
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

291    0.7 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.8 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

292    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  76 GB/s  26.2 %  87.1 TFLOPs   49.8 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

295    2.3 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0       236 μs          1 μs     48  38 GB/s  13.1 %  20.5 TFLOPs   20.7 %  HiFi2 BF16 x BFP8 => BF16
- in0_block_w=4 and output subblock 1x4 look good 🤷
- If your matmuls are not FLOP-bound use HiFi4 with BF16 activations for full accuracy

296    3.3 %   SLOW  MatmulDeviceOperation 576 x 4096 x 4096        0       330 μs          1 μs     48  79 GB/s  27.6 %  58.5 TFLOPs   59.3 %  HiFi2 BF16 x BFP8 => BF16
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=4 and output subblock 1x4 look good 🤷
- If your matmuls are not FLOP-bound use HiFi4 with BF16 activations for full accuracy


📊 Stacked report 📊
====================

Total %  Op Code                                                     Device Time Sum  Op Count  Op Category  Min FLOPs  Max FLOPs  Mean FLOPs  Std FLOPs  Weighted Mean FLOPs
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
45.22 %  MatmulDeviceOperation (in0:block_sharded)                       4,450.14 μs        73      Compute    20.74 %    51.13 %     43.58 %     5.83 %              42.13 %
16.37 %  SDPAOperation (in0:dram_interleaved)                            1,610.93 μs        24      Compute
11.83 %  NlpCreateHeadsDeviceOperation (in0:l1_interleaved)              1,163.89 μs        24           TM
 9.48 %  LayerNormDeviceOperation (in0:block_sharded)                      932.50 μs        49      Compute
 8.27 %  MatmulDeviceOperation (in0:dram_interleaved)                      813.64 μs        26      Compute    20.95 %    59.32 %     37.94 %     5.48 %              45.60 %
 4.41 %  NLPConcatHeadsDeviceOperation (in0:dram_interleaved)              433.65 μs        24           TM
 2.35 %  ShardedToInterleavedDeviceOperation (in0:block_sharded)           231.32 μs        24           DM
 1.44 %  BinaryNgDeviceOperation (in0:block_sharded)                       141.58 μs        47      Compute
 0.55 %  BinaryNgDeviceOperation (in0:dram_interleaved)                     54.53 μs         3      Compute
 0.08 %  InterleavedToShardedDeviceOperation (in0:dram_interleaved)          8.05 μs         1           DM
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```
