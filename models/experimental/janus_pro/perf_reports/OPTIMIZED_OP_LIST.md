# tt-perf-report, one replay of the optimized tower

`tt-perf-report` over a single trace replay of the tower at **9.316 ms** — the op table, its advice
section and its summary, unabridged. Commit `81f76bd4f65` (change 29), Wormhole N150, 293 ops.

One of the ten replays the perf test measures, picked as the one whose kernel time sits closest to
the mean of replays 2-10 — the figure PERF.md's change log carries: 9.316 ms against 9.316.

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
  2    0.3 %   SLOW  MatmulDeviceOperation 576 x 768 x 1024         0        31 μs                   48  80 GB/s  27.6 %  29.3 TFLOPs   29.7 %  HiFi2 BF16 x BF16 => BF16
  3    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BF16 => BF16
  4    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
  5    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        49 μs          1 μs     48  64 GB/s  22.2 %  73.8 TFLOPs   42.2 %   LoFi BF16 x BFP8 => BFP8
  6    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
  7    0.4 %         NlpCreateHeadsDeviceOperation                  0        35 μs          1 μs     18                                                      BFP8 => BFP8
  8    0.7 %         SDPAOperation                                  0        65 μs          1 μs     64                                                BFP8, BFP8 => BFP8
  9    0.1 %         NLPConcatHeadsDeviceOperation                  0        13 μs          1 μs     18                                                      BFP8 => BFP8
 10    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  60 GB/s  20.8 %  68.9 TFLOPs   39.5 %   LoFi BFP8 x BFP8 => BFP8
 11    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 12    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
 13    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  67.1 TFLOPs   38.4 %   LoFi BF16 x BFP8 => BFP8
 14    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.6 TFLOPs   50.8 %   LoFi BFP8 x BFP8 => BFP8
 15    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 16    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
 17    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  75.0 TFLOPs   43.0 %   LoFi BF16 x BFP8 => BFP8
 18    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
 19    0.4 %         NlpCreateHeadsDeviceOperation                  0        35 μs          1 μs     18                                                      BFP8 => BFP8
 20    0.7 %         SDPAOperation                                  0        67 μs          1 μs     64                                                BFP8, BFP8 => BFP8
 21    0.1 %         NLPConcatHeadsDeviceOperation                  0        13 μs          1 μs     18                                                      BFP8 => BFP8
 22    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BFP8 x BFP8 => BFP8
 23    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 24    0.2 %         LayerNormDeviceOperation                       0        20 μs          1 μs     48                                                BF16, BF16 => BF16
 25    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  67.0 TFLOPs   38.4 %   LoFi BF16 x BFP8 => BFP8
 26    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.7 TFLOPs   50.8 %   LoFi BFP8 x BFP8 => BFP8
 27    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 28    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
 29    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  75.0 TFLOPs   43.0 %   LoFi BF16 x BFP8 => BFP8
 30    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
 31    0.4 %         NlpCreateHeadsDeviceOperation                  0        35 μs          1 μs     18                                                      BFP8 => BFP8
 32    0.7 %         SDPAOperation                                  0        66 μs          1 μs     64                                                BFP8, BFP8 => BFP8
 33    0.2 %         NLPConcatHeadsDeviceOperation                  0        19 μs          1 μs     18                                                      BFP8 => BFP8
 34    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  58 GB/s  20.1 %  66.8 TFLOPs   38.3 %   LoFi BFP8 x BFP8 => BFP8
 35    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 36    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
 37    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.7 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
 38    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.6 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
 39    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 40    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
 41    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          2 μs     48  65 GB/s  22.6 %  74.9 TFLOPs   42.9 %   LoFi BF16 x BFP8 => BFP8
 42    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
 43    0.4 %         NlpCreateHeadsDeviceOperation                  0        35 μs          1 μs     18                                                      BFP8 => BFP8
 44    0.7 %         SDPAOperation                                  0        65 μs          1 μs     64                                                BFP8, BFP8 => BFP8
 45    0.1 %         NLPConcatHeadsDeviceOperation                  0        13 μs          1 μs     18                                                      BFP8 => BFP8
 46    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  60 GB/s  20.7 %  68.7 TFLOPs   39.3 %   LoFi BFP8 x BFP8 => BFP8
 47    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 48    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
 49    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        73 μs          1 μs     48  58 GB/s  20.1 %  66.6 TFLOPs   38.1 %   LoFi BF16 x BFP8 => BFP8
 50    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.8 %  88.8 TFLOPs   50.8 %   LoFi BFP8 x BFP8 => BFP8
 51    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 52    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
 53    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.5 %  74.7 TFLOPs   42.8 %   LoFi BF16 x BFP8 => BFP8
 54    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
 55    0.4 %         NlpCreateHeadsDeviceOperation                  0        36 μs          1 μs     18                                                      BFP8 => BFP8
 56    0.7 %         SDPAOperation                                  0        65 μs          1 μs     64                                                BFP8, BFP8 => BFP8
 57    0.1 %         NLPConcatHeadsDeviceOperation                  0        13 μs          1 μs     18                                                      BFP8 => BFP8
 58    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  60 GB/s  20.7 %  68.8 TFLOPs   39.4 %   LoFi BFP8 x BFP8 => BFP8
 59    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 60    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
 61    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.7 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
 62    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.6 %  88.2 TFLOPs   50.5 %   LoFi BFP8 x BFP8 => BFP8
 63    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 64    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
 65    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  75.1 TFLOPs   43.0 %   LoFi BF16 x BFP8 => BFP8
 66    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
 67    0.4 %         NlpCreateHeadsDeviceOperation                  0        35 μs          1 μs     18                                                      BFP8 => BFP8
 68    0.7 %         SDPAOperation                                  0        66 μs          1 μs     64                                                BFP8, BFP8 => BFP8
 69    0.1 %         NLPConcatHeadsDeviceOperation                  0        13 μs          1 μs     18                                                      BFP8 => BFP8
 70    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  58 GB/s  20.1 %  66.7 TFLOPs   38.2 %   LoFi BFP8 x BFP8 => BFP8
 71    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 72    0.3 %         LayerNormDeviceOperation                       0        24 μs          1 μs     48                                                BF16, BF16 => BF16
 73    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        73 μs          1 μs     48  57 GB/s  19.9 %  66.2 TFLOPs   37.9 %   LoFi BF16 x BFP8 => BFP8
 74    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.6 %  88.2 TFLOPs   50.5 %   LoFi BFP8 x BFP8 => BFP8
 75    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 76    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
 77    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  74.9 TFLOPs   42.9 %   LoFi BF16 x BFP8 => BFP8
 78    0.1 %         ShardedToInterleavedDeviceOperation            0         9 μs          1 μs     48                                                      BFP8 => BFP8
 79    0.4 %         NlpCreateHeadsDeviceOperation                  0        35 μs          1 μs     18                                                      BFP8 => BFP8
 80    0.7 %         SDPAOperation                                  0        65 μs          1 μs     64                                                BFP8, BFP8 => BFP8
 81    0.1 %         NLPConcatHeadsDeviceOperation                  0        14 μs          1 μs     18                                                      BFP8 => BFP8
 82    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        17 μs          1 μs     48  60 GB/s  20.9 %  69.3 TFLOPs   39.7 %   LoFi BFP8 x BFP8 => BFP8
 83    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 84    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
 85    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
 86    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.6 %  88.3 TFLOPs   50.5 %   LoFi BFP8 x BFP8 => BFP8
 87    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 88    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
 89    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  75.1 TFLOPs   43.0 %   LoFi BF16 x BFP8 => BFP8
 90    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
 91    0.4 %         NlpCreateHeadsDeviceOperation                  0        35 μs          1 μs     18                                                      BFP8 => BFP8
 92    0.7 %         SDPAOperation                                  0        66 μs          1 μs     64                                                BFP8, BFP8 => BFP8
 93    0.1 %         NLPConcatHeadsDeviceOperation                  0        12 μs          1 μs     18                                                      BFP8 => BFP8
 94    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        19 μs          1 μs     48  57 GB/s  19.6 %  65.1 TFLOPs   37.3 %   LoFi BFP8 x BFP8 => BFP8
 95    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
 96    0.2 %         LayerNormDeviceOperation                       0        20 μs          1 μs     48                                                BF16, BF16 => BF16
 97    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        73 μs          1 μs     48  58 GB/s  20.0 %  66.5 TFLOPs   38.1 %   LoFi BF16 x BFP8 => BFP8
 98    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.6 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
 99    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
100    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
101    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          2 μs     48  65 GB/s  22.6 %  74.9 TFLOPs   42.9 %   LoFi BF16 x BFP8 => BFP8
102    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
103    0.4 %         NlpCreateHeadsDeviceOperation                  0        35 μs          1 μs     18                                                      BFP8 => BFP8
104    0.7 %         SDPAOperation                                  0        65 μs          1 μs     64                                                BFP8, BFP8 => BFP8
105    0.1 %         NLPConcatHeadsDeviceOperation                  0        13 μs          1 μs     18                                                      BFP8 => BFP8
106    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  59 GB/s  20.3 %  67.5 TFLOPs   38.7 %   LoFi BFP8 x BFP8 => BFP8
107    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
108    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
109    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.8 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
110    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  76 GB/s  26.5 %  88.1 TFLOPs   50.4 %   LoFi BFP8 x BFP8 => BFP8
111    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
112    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
113    0.6 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        53 μs          1 μs     48  59 GB/s  20.5 %  67.9 TFLOPs   38.9 %   LoFi BF16 x BFP8 => BFP8
114    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
115    0.4 %         NlpCreateHeadsDeviceOperation                  0        35 μs          1 μs     18                                                      BFP8 => BFP8
116    0.7 %         SDPAOperation                                  0        68 μs          1 μs     64                                                BFP8, BFP8 => BFP8
117    0.1 %         NLPConcatHeadsDeviceOperation                  0        12 μs          1 μs     18                                                      BFP8 => BFP8
118    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  57 GB/s  19.8 %  65.6 TFLOPs   37.5 %   LoFi BFP8 x BFP8 => BFP8
119    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
120    0.2 %         LayerNormDeviceOperation                       0        20 μs          1 μs     48                                                BF16, BF16 => BF16
121    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
122    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.5 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
123    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
124    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
125    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.7 %  75.2 TFLOPs   43.1 %   LoFi BF16 x BFP8 => BFP8
126    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
127    0.4 %         NlpCreateHeadsDeviceOperation                  0        35 μs          1 μs     18                                                      BFP8 => BFP8
128    0.7 %         SDPAOperation                                  0        65 μs          1 μs     64                                                BFP8, BFP8 => BFP8
129    0.1 %         NLPConcatHeadsDeviceOperation                  0        13 μs          1 μs     18                                                      BFP8 => BFP8
130    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        17 μs          1 μs     48  60 GB/s  20.9 %  69.4 TFLOPs   39.7 %   LoFi BFP8 x BFP8 => BFP8
131    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
132    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
133    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        73 μs          1 μs     48  58 GB/s  20.1 %  66.6 TFLOPs   38.1 %   LoFi BF16 x BFP8 => BFP8
134    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.6 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
135    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
136    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
137    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          2 μs     48  65 GB/s  22.6 %  75.0 TFLOPs   43.0 %   LoFi BF16 x BFP8 => BFP8
138    0.1 %         ShardedToInterleavedDeviceOperation            0         9 μs          1 μs     48                                                      BFP8 => BFP8
139    0.4 %         NlpCreateHeadsDeviceOperation                  0        35 μs          1 μs     18                                                      BFP8 => BFP8
140    0.7 %         SDPAOperation                                  0        68 μs          1 μs     64                                                BFP8, BFP8 => BFP8
141    0.1 %         NLPConcatHeadsDeviceOperation                  0        13 μs          1 μs     18                                                      BFP8 => BFP8
142    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  60 GB/s  20.7 %  68.7 TFLOPs   39.3 %   LoFi BFP8 x BFP8 => BFP8
143    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
144    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
145    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.8 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
146    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.6 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
147    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
148    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
149    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.5 %  74.7 TFLOPs   42.8 %   LoFi BF16 x BFP8 => BFP8
150    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
151    0.4 %         NlpCreateHeadsDeviceOperation                  0        35 μs          1 μs     18                                                      BFP8 => BFP8
152    0.7 %         SDPAOperation                                  0        65 μs          1 μs     64                                                BFP8, BFP8 => BFP8
153    0.2 %         NLPConcatHeadsDeviceOperation                  0        21 μs          1 μs     18                                                      BFP8 => BFP8
154    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        17 μs          1 μs     48  60 GB/s  20.8 %  69.2 TFLOPs   39.6 %   LoFi BFP8 x BFP8 => BFP8
155    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
156    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
157    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
158    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.8 %  89.0 TFLOPs   50.9 %   LoFi BFP8 x BFP8 => BFP8
159    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
160    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
161    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          2 μs     48  65 GB/s  22.7 %  75.2 TFLOPs   43.1 %   LoFi BF16 x BFP8 => BFP8
162    0.1 %         ShardedToInterleavedDeviceOperation            0         9 μs          1 μs     48                                                      BFP8 => BFP8
163    0.4 %         NlpCreateHeadsDeviceOperation                  0        35 μs          1 μs     18                                                      BFP8 => BFP8
164    0.7 %         SDPAOperation                                  0        68 μs          1 μs     64                                                BFP8, BFP8 => BFP8
165    0.1 %         NLPConcatHeadsDeviceOperation                  0        12 μs          1 μs     18                                                      BFP8 => BFP8
166    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        19 μs          1 μs     48  56 GB/s  19.5 %  64.7 TFLOPs   37.0 %   LoFi BFP8 x BFP8 => BFP8
167    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
168    0.2 %         LayerNormDeviceOperation                       0        20 μs          1 μs     48                                                BF16, BF16 => BF16
169    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.8 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
170    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.8 %  89.1 TFLOPs   51.0 %   LoFi BFP8 x BFP8 => BFP8
171    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
172    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
173    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.7 %  75.2 TFLOPs   43.0 %   LoFi BF16 x BFP8 => BFP8
174    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
175    0.4 %         NlpCreateHeadsDeviceOperation                  0        35 μs          1 μs     18                                                      BFP8 => BFP8
176    0.7 %         SDPAOperation                                  0        67 μs          1 μs     64                                                BFP8, BFP8 => BFP8
177    0.1 %         NLPConcatHeadsDeviceOperation                  0        12 μs          1 μs     18                                                      BFP8 => BFP8
178    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  57 GB/s  19.7 %  65.5 TFLOPs   37.5 %   LoFi BFP8 x BFP8 => BFP8
179    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
180    0.2 %         LayerNormDeviceOperation                       0        20 μs          1 μs     48                                                BF16, BF16 => BF16
181    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        73 μs          1 μs     48  58 GB/s  20.1 %  66.5 TFLOPs   38.1 %   LoFi BF16 x BFP8 => BFP8
182    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.4 TFLOPs   50.6 %   LoFi BFP8 x BFP8 => BFP8
183    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
184    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
185    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        49 μs          1 μs     48  65 GB/s  22.5 %  74.6 TFLOPs   42.7 %   LoFi BF16 x BFP8 => BFP8
186    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
187    0.4 %         NlpCreateHeadsDeviceOperation                  0        35 μs          1 μs     18                                                      BFP8 => BFP8
188    0.7 %         SDPAOperation                                  0        64 μs          1 μs     64                                                BFP8, BFP8 => BFP8
189    0.1 %         NLPConcatHeadsDeviceOperation                  0        13 μs          1 μs     18                                                      BFP8 => BFP8
190    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  60 GB/s  20.7 %  68.7 TFLOPs   39.3 %   LoFi BFP8 x BFP8 => BFP8
191    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
192    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
193    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        78 μs          1 μs     48  54 GB/s  18.7 %  61.9 TFLOPs   35.4 %   LoFi BF16 x BFP8 => BFP8
194    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.8 %  88.8 TFLOPs   50.9 %   LoFi BFP8 x BFP8 => BFP8
195    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
196    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
197    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  74.8 TFLOPs   42.9 %   LoFi BF16 x BFP8 => BFP8
198    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
199    0.4 %         NlpCreateHeadsDeviceOperation                  0        35 μs          1 μs     18                                                      BFP8 => BFP8
200    0.7 %         SDPAOperation                                  0        65 μs          1 μs     64                                                BFP8, BFP8 => BFP8
201    0.1 %         NLPConcatHeadsDeviceOperation                  0        13 μs          1 μs     18                                                      BFP8 => BFP8
202    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  59 GB/s  20.6 %  68.3 TFLOPs   39.1 %   LoFi BFP8 x BFP8 => BFP8
203    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
204    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
205    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
206    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.5 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
207    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
208    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
209    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  75.0 TFLOPs   42.9 %   LoFi BF16 x BFP8 => BFP8
210    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
211    0.4 %         NlpCreateHeadsDeviceOperation                  0        35 μs          1 μs     18                                                      BFP8 => BFP8
212    0.7 %         SDPAOperation                                  0        65 μs          1 μs     64                                                BFP8, BFP8 => BFP8
213    0.1 %         NLPConcatHeadsDeviceOperation                  0        13 μs          1 μs     18                                                      BFP8 => BFP8
214    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  59 GB/s  20.4 %  67.7 TFLOPs   38.8 %   LoFi BFP8 x BFP8 => BFP8
215    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
216    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
217    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
218    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.6 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
219    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
220    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
221    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          2 μs     48  65 GB/s  22.6 %  75.1 TFLOPs   43.0 %   LoFi BF16 x BFP8 => BFP8
222    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
223    0.4 %         NlpCreateHeadsDeviceOperation                  0        35 μs          1 μs     18                                                      BFP8 => BFP8
224    0.7 %         SDPAOperation                                  0        68 μs          1 μs     64                                                BFP8, BFP8 => BFP8
225    0.1 %         NLPConcatHeadsDeviceOperation                  0        13 μs          1 μs     18                                                      BFP8 => BFP8
226    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  59 GB/s  20.4 %  67.8 TFLOPs   38.8 %   LoFi BFP8 x BFP8 => BFP8
227    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
228    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
229    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        73 μs          1 μs     48  58 GB/s  20.0 %  66.5 TFLOPs   38.1 %   LoFi BF16 x BFP8 => BFP8
230    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.6 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
231    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
232    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
233    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          2 μs     48  65 GB/s  22.7 %  75.3 TFLOPs   43.1 %   LoFi BF16 x BFP8 => BFP8
234    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
235    0.4 %         NlpCreateHeadsDeviceOperation                  0        36 μs          1 μs     18                                                      BFP8 => BFP8
236    0.7 %         SDPAOperation                                  0        65 μs          1 μs     64                                                BFP8, BFP8 => BFP8
237    0.1 %         NLPConcatHeadsDeviceOperation                  0        13 μs          1 μs     18                                                      BFP8 => BFP8
238    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  59 GB/s  20.6 %  68.4 TFLOPs   39.2 %   LoFi BFP8 x BFP8 => BFP8
239    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
240    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
241    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.8 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
242    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.4 TFLOPs   50.6 %   LoFi BFP8 x BFP8 => BFP8
243    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
244    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
245    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  74.9 TFLOPs   42.9 %   LoFi BF16 x BFP8 => BFP8
246    0.1 %         ShardedToInterleavedDeviceOperation            0         9 μs          1 μs     48                                                      BFP8 => BFP8
247    0.4 %         NlpCreateHeadsDeviceOperation                  0        35 μs          1 μs     18                                                      BFP8 => BFP8
248    0.7 %         SDPAOperation                                  0        65 μs          1 μs     64                                                BFP8, BFP8 => BFP8
249    0.1 %         NLPConcatHeadsDeviceOperation                  0        13 μs          1 μs     18                                                      BFP8 => BFP8
250    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  59 GB/s  20.4 %  67.5 TFLOPs   38.7 %   LoFi BFP8 x BFP8 => BFP8
251    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
252    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
253    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  67.0 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
254    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.7 %  88.7 TFLOPs   50.8 %   LoFi BFP8 x BFP8 => BFP8
255    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
256    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
257    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  75.0 TFLOPs   42.9 %   LoFi BF16 x BFP8 => BFP8
258    0.1 %         ShardedToInterleavedDeviceOperation            0         9 μs          1 μs     48                                                      BFP8 => BFP8
259    0.4 %         NlpCreateHeadsDeviceOperation                  0        36 μs          1 μs     18                                                      BFP8 => BFP8
260    0.7 %         SDPAOperation                                  0        65 μs          1 μs     64                                                BFP8, BFP8 => BFP8
261    0.1 %         NLPConcatHeadsDeviceOperation                  0        13 μs          1 μs     18                                                      BFP8 => BFP8
262    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  60 GB/s  20.7 %  68.7 TFLOPs   39.4 %   LoFi BFP8 x BFP8 => BFP8
263    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
264    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
265    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
266    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.6 %  88.2 TFLOPs   50.5 %   LoFi BFP8 x BFP8 => BFP8
267    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
268    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
269    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  75.0 TFLOPs   42.9 %   LoFi BF16 x BFP8 => BFP8
270    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
271    0.4 %         NlpCreateHeadsDeviceOperation                  0        35 μs          1 μs     18                                                      BFP8 => BFP8
272    0.7 %         SDPAOperation                                  0        65 μs          1 μs     64                                                BFP8, BFP8 => BFP8
273    0.2 %         NLPConcatHeadsDeviceOperation                  0        16 μs          1 μs     18                                                      BFP8 => BFP8
274    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  60 GB/s  20.7 %  68.8 TFLOPs   39.4 %   LoFi BFP8 x BFP8 => BFP8
275    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
276    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
277    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.8 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
278    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.7 %  88.7 TFLOPs   50.8 %   LoFi BFP8 x BFP8 => BFP8
279    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
280    0.2 %         LayerNormDeviceOperation                       0        19 μs          2 μs     48                                                BF16, BF16 => BF16
281    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        49 μs          1 μs     48  65 GB/s  22.5 %  74.7 TFLOPs   42.7 %   LoFi BF16 x BFP8 => BFP8
282    0.1 %         ShardedToInterleavedDeviceOperation            0        10 μs          1 μs     48                                                      BFP8 => BFP8
283    0.4 %         NlpCreateHeadsDeviceOperation                  0        37 μs          1 μs     18                                                      BFP8 => BFP8
284    0.7 %         SDPAOperation                                  0        66 μs          1 μs     64                                                BFP8, BFP8 => BFP8
285    0.1 %         NLPConcatHeadsDeviceOperation                  0        12 μs          1 μs     18                                                      BFP8 => BFP8
286    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        19 μs          1 μs     48  54 GB/s  18.8 %  62.4 TFLOPs   35.7 %   LoFi BFP8 x BFP8 => BFP8
287    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
288    0.2 %         LayerNormDeviceOperation                       0        19 μs          1 μs     48                                                BF16, BF16 => BF16
289    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
290    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  76 GB/s  26.3 %  87.4 TFLOPs   50.0 %   LoFi BFP8 x BFP8 => BFP8
291    0.0 %         BinaryNgDeviceOperation                        0         3 μs          1 μs     64                                                BF16, BFP8 => BF16
292    0.2 %         LayerNormDeviceOperation                       0        20 μs          1 μs     48                                                BF16, BF16 => BF16
293    2.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0       236 μs          1 μs     48  38 GB/s  13.1 %  20.5 TFLOPs   20.8 %  HiFi2 BF16 x BFP8 => BF16
294    3.4 %   SLOW  MatmulDeviceOperation 576 x 4096 x 4096        0       328 μs          1 μs     48  80 GB/s  27.8 %  58.9 TFLOPs   59.7 %  HiFi2 BF16 x BFP8 => BF16
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
     100.0 %         293 device ops, 0 host ops, 0 signposts              9,316 μs        284 μs         36 GB/s  12.7 %

💡 Advice 💡
============

Matmul Optimization
-------------------
  2    0.3 %   SLOW  MatmulDeviceOperation 576 x 768 x 1024         0        31 μs                   48  80 GB/s  27.6 %  29.3 TFLOPs   29.7 %  HiFi2 BF16 x BF16 => BF16
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=4 and output subblock 1x4 look good 🤷
- If your matmuls are not FLOP-bound use HiFi4 with BF16 activations for full accuracy

  5    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        49 μs          1 μs     48  64 GB/s  22.2 %  73.8 TFLOPs   42.2 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 10    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  60 GB/s  20.8 %  68.9 TFLOPs   39.5 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 13    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  67.1 TFLOPs   38.4 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 14    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.6 TFLOPs   50.8 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 17    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  75.0 TFLOPs   43.0 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 22    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 25    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  67.0 TFLOPs   38.4 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 26    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.7 TFLOPs   50.8 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 29    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  75.0 TFLOPs   43.0 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 34    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  58 GB/s  20.1 %  66.8 TFLOPs   38.3 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 37    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.7 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 38    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.6 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 41    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          2 μs     48  65 GB/s  22.6 %  74.9 TFLOPs   42.9 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 46    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  60 GB/s  20.7 %  68.7 TFLOPs   39.3 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 49    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        73 μs          1 μs     48  58 GB/s  20.1 %  66.6 TFLOPs   38.1 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 50    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.8 %  88.8 TFLOPs   50.8 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 53    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.5 %  74.7 TFLOPs   42.8 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 58    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  60 GB/s  20.7 %  68.8 TFLOPs   39.4 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 61    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.7 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 62    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.6 %  88.2 TFLOPs   50.5 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 65    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  75.1 TFLOPs   43.0 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 70    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  58 GB/s  20.1 %  66.7 TFLOPs   38.2 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 73    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        73 μs          1 μs     48  57 GB/s  19.9 %  66.2 TFLOPs   37.9 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 74    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.6 %  88.2 TFLOPs   50.5 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 77    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  74.9 TFLOPs   42.9 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 82    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        17 μs          1 μs     48  60 GB/s  20.9 %  69.3 TFLOPs   39.7 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 85    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 86    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.6 %  88.3 TFLOPs   50.5 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 89    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  75.1 TFLOPs   43.0 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 94    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        19 μs          1 μs     48  57 GB/s  19.6 %  65.1 TFLOPs   37.3 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

 97    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        73 μs          1 μs     48  58 GB/s  20.0 %  66.5 TFLOPs   38.1 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

 98    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.6 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

101    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          2 μs     48  65 GB/s  22.6 %  74.9 TFLOPs   42.9 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

106    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  59 GB/s  20.3 %  67.5 TFLOPs   38.7 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

109    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.8 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

110    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  76 GB/s  26.5 %  88.1 TFLOPs   50.4 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

113    0.6 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        53 μs          1 μs     48  59 GB/s  20.5 %  67.9 TFLOPs   38.9 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

118    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  57 GB/s  19.8 %  65.6 TFLOPs   37.5 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

121    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

122    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.5 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

125    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.7 %  75.2 TFLOPs   43.1 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

130    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        17 μs          1 μs     48  60 GB/s  20.9 %  69.4 TFLOPs   39.7 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

133    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        73 μs          1 μs     48  58 GB/s  20.1 %  66.6 TFLOPs   38.1 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

134    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.6 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

137    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          2 μs     48  65 GB/s  22.6 %  75.0 TFLOPs   43.0 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

142    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  60 GB/s  20.7 %  68.7 TFLOPs   39.3 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

145    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.8 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

146    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.6 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

149    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.5 %  74.7 TFLOPs   42.8 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

154    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        17 μs          1 μs     48  60 GB/s  20.8 %  69.2 TFLOPs   39.6 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

157    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

158    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.8 %  89.0 TFLOPs   50.9 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

161    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          2 μs     48  65 GB/s  22.7 %  75.2 TFLOPs   43.1 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

166    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        19 μs          1 μs     48  56 GB/s  19.5 %  64.7 TFLOPs   37.0 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

169    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.8 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

170    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.8 %  89.1 TFLOPs   51.0 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

173    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.7 %  75.2 TFLOPs   43.0 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

178    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  57 GB/s  19.7 %  65.5 TFLOPs   37.5 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

181    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        73 μs          1 μs     48  58 GB/s  20.1 %  66.5 TFLOPs   38.1 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

182    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.4 TFLOPs   50.6 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

185    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        49 μs          1 μs     48  65 GB/s  22.5 %  74.6 TFLOPs   42.7 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

190    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  60 GB/s  20.7 %  68.7 TFLOPs   39.3 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

193    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        78 μs          1 μs     48  54 GB/s  18.7 %  61.9 TFLOPs   35.4 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

194    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.8 %  88.8 TFLOPs   50.9 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

197    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  74.8 TFLOPs   42.9 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

202    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  59 GB/s  20.6 %  68.3 TFLOPs   39.1 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

205    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

206    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.5 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

209    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  75.0 TFLOPs   42.9 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

214    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  59 GB/s  20.4 %  67.7 TFLOPs   38.8 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

217    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

218    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.6 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

221    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          2 μs     48  65 GB/s  22.6 %  75.1 TFLOPs   43.0 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

226    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  59 GB/s  20.4 %  67.8 TFLOPs   38.8 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

229    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        73 μs          1 μs     48  58 GB/s  20.0 %  66.5 TFLOPs   38.1 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

230    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.6 TFLOPs   50.7 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

233    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          2 μs     48  65 GB/s  22.7 %  75.3 TFLOPs   43.1 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

238    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  59 GB/s  20.6 %  68.4 TFLOPs   39.2 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

241    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.8 TFLOPs   38.2 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

242    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.7 %  88.4 TFLOPs   50.6 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

245    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  74.9 TFLOPs   42.9 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

250    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  59 GB/s  20.4 %  67.5 TFLOPs   38.7 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

253    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  67.0 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

254    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.7 %  88.7 TFLOPs   50.8 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

257    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  75.0 TFLOPs   42.9 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

262    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  60 GB/s  20.7 %  68.7 TFLOPs   39.4 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

265    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

266    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  77 GB/s  26.6 %  88.2 TFLOPs   50.5 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

269    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        48 μs          1 μs     48  65 GB/s  22.6 %  75.0 TFLOPs   42.9 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

274    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        18 μs          1 μs     48  60 GB/s  20.7 %  68.8 TFLOPs   39.4 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

277    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.1 %  66.8 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

278    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        54 μs          1 μs     48  77 GB/s  26.7 %  88.7 TFLOPs   50.8 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

281    0.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 3072        0        49 μs          1 μs     48  65 GB/s  22.5 %  74.7 TFLOPs   42.7 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

286    0.2 %   SLOW  MatmulDeviceOperation 576 x 1024 x 1024        0        19 μs          1 μs     48  54 GB/s  18.8 %  62.4 TFLOPs   35.7 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=8 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

289    0.8 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0        72 μs          1 μs     48  58 GB/s  20.2 %  66.9 TFLOPs   38.3 %   LoFi BF16 x BFP8 => BFP8
- in0_block_w=4 and output subblock 1x4 look good 🤷
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy

290    0.6 %   SLOW  MatmulDeviceOperation 576 x 4096 x 1024        0        55 μs          1 μs     48  76 GB/s  26.3 %  87.4 TFLOPs   50.0 %   LoFi BFP8 x BFP8 => BFP8
- in0_block_w=16 and output subblock 1x4 look good 🤷
- HiFi2 is recommended for accuracy; LoFi discards the lowest 2 bits of the weights

293    2.5 %   SLOW  MatmulDeviceOperation 576 x 1024 x 4096        0       236 μs          1 μs     48  38 GB/s  13.1 %  20.5 TFLOPs   20.8 %  HiFi2 BF16 x BFP8 => BF16
- in0_block_w=4 and output subblock 1x4 look good 🤷
- If your matmuls are not FLOP-bound use HiFi4 with BF16 activations for full accuracy

294    3.4 %   SLOW  MatmulDeviceOperation 576 x 4096 x 4096        0       328 μs          1 μs     48  80 GB/s  27.8 %  58.9 TFLOPs   59.7 %  HiFi2 BF16 x BFP8 => BF16
- If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)
- in0_block_w=4 and output subblock 1x4 look good 🤷
- If your matmuls are not FLOP-bound use HiFi4 with BF16 activations for full accuracy


📊 Stacked report 📊
====================

Total %  Op Code                                                  Device Time Sum  Op Count  Op Category  Min FLOPs  Max FLOPs  Mean FLOPs  Std FLOPs  Weighted Mean FLOPs
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
47.82 %  MatmulDeviceOperation (in0:block_sharded)                    4,454.74 μs        73      Compute    20.76 %    50.99 %     43.52 %     5.88 %              42.08 %
16.94 %  SDPAOperation (in0:l1_interleaved)                           1,578.40 μs        24      Compute
10.07 %  LayerNormDeviceOperation (in0:block_sharded)                   938.44 μs        49      Compute
 9.07 %  NlpCreateHeadsDeviceOperation (in0:l1_interleaved)             845.38 μs        24           TM
 4.61 %  MatmulDeviceOperation (in0:l1_interleaved)                     429.75 μs        24      Compute    35.73 %    39.73 %     38.65 %     1.01 %              38.62 %
 3.85 %  MatmulDeviceOperation (in0:dram_interleaved)                   358.82 μs         2      Compute    29.72 %    59.74 %     44.73 %    21.22 %              57.15 %
 3.51 %  NLPConcatHeadsDeviceOperation (in0:l1_interleaved)             327.22 μs        24           TM
 2.51 %  ShardedToInterleavedDeviceOperation (in0:block_sharded)        233.93 μs        24           DM
 1.61 %  BinaryNgDeviceOperation (in0:block_sharded)                    149.62 μs        49      Compute
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```
