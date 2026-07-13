# Seamless M4T v2 Large — performance

Phase-separated timings from `[scripts/demo_perf_sweep.py](scripts/demo_perf_sweep.py)` with `generate(return_timings=True)`. Inputs:

- **Text** — *A Tale of Two Cities* (`models/tt_transformers/tests/tale-of-two-cities.txt.bz2`)
- **Audio** — concatenated LibriSpeech-dummy utterances (`scripts/outputs/long_speech_input_librispeech.wav`; preamble fallback if the dataset is unavailable)

Sequence lengths double **32 → 4096** (text = source tokens; speech = mel frames). Host pre/post-processing (token decode, WAV I/O) is excluded. Each task opens its own mesh device.

Sweep timing vs `demo.py`:

- Untimed warmups match the demo (text=1, speech=2).
- **1 timed iter** per task (demo uses 2 timed iters for T2ST/S2ST).
- No vocoder `post_warmup_fn` in the sweep (demo prewarms vocoder convs after speech warmups).
- At mel **≥ 1024**, **S2TT / S2ST / ASR** warmups run on a throwaway mesh device; the timed session still prewarms the speech encoder. T2ST does not use split warmups.

Metrics follow the TT model catalog (Whisper / LLM / Qwen3-TTS style):


| Metric           | Meaning                                                                                                                     |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **TTFT**         | Time from `generate()` start to first **new** decoder token (includes encoder + decoder prefill + first decode step)        |
| **Encoder**      | Speech or text encoder only                                                                                                 |
| **Prefill**      | Text-decoder KV prefill on the seed sequence (`[decoder_start, lang]`)                                                      |
| **decode t/s/u** | `1000 / steady_ms_per_tok` — steady text-decoder step rate (**decode steps 2+**, excludes first-step trace/compile outlier) |
| **E2E**          | Full synced `generate()` wall time (includes T2U + vocoder on T2ST/S2ST)                                                    |
| **RTF**          | Real-time factor on speech tasks: `e2e_s / audio_duration_s` (`<1` = faster than real time)                                 |


**Compare decode t/s/u across tasks** — unlike legacy E2E `tokens/s`, it is not penalized by long input encoders (S2TT/ASR) or variable output length. At very short lengths (e.g. 32 mel) decode t/s/u is noisy because only a handful of decoder steps run.

Reproduce:

```bash
export MESH_DEVICE=BH-QB   # or P150
python models/experimental/seamless_m4t_v2_large/scripts/demo_perf_sweep.py
```

---

## Demo wall timings (Blackhole BH QB, 2CQ + decode trace)

Four-chip QuietBox (`MeshShape(1, 4)`, batch-1, TP=4). Tables from `scripts/outputs/perf_sweep.txt` (2026-07-13).

### Sequence length: 32


| Task | TTFT     | Encoder  | Prefill | decode t/s/u | ms/tok (steady) | E2E       | Output                     |
| ---- | -------- | -------- | ------- | ------------ | --------------- | --------- | -------------------------- |
| T2TT | 145.4 ms | 27.9 ms  | 44.2 ms | 97.9         | 10.2            | 416.5 ms  | 27 tok                     |
| T2ST | 158.6 ms | 36.5 ms  | 52.7 ms | 95.1         | 10.5            | 1844.2 ms | 103040 smp (RTF **0.29×**) |
| S2TT | 204.6 ms | 96.7 ms  | 41.3 ms | 29.3         | 34.1            | 308.3 ms  | 4 tok (32 mel)             |
| S2ST | 711.0 ms | 575.9 ms | 64.4 ms | 38.2         | 26.2            | 2296.1 ms | 12480 smp (RTF **2.94×**)  |
| ASR  | 210.1 ms | 98.6 ms  | 44.4 ms | 61.5         | 16.3            | 359.1 ms  | 10 tok (32 mel)            |



| Task | T2U    | Vocoder | RTF   |
| ---- | ------ | ------- | ----- |
| T2ST | 762 ms | 601 ms  | 0.29× |
| S2ST | 830 ms | 574 ms  | 2.94× |


### Sequence length: 64


| Task | TTFT     | Encoder  | Prefill | decode t/s/u | ms/tok (steady) | E2E       | Output                     |
| ---- | -------- | -------- | ------- | ------------ | --------------- | --------- | -------------------------- |
| T2TT | 134.8 ms | 21.4 ms  | 43.4 ms | 117.8        | 8.5             | 595.1 ms  | 54 tok                     |
| T2ST | 166.6 ms | 33.5 ms  | 63.0 ms | 113.5        | 8.8             | 2223.0 ms | 196800 smp (RTF **0.18×**) |
| S2TT | 212.2 ms | 97.2 ms  | 46.3 ms | 40.2         | 24.9            | 338.5 ms  | 6 tok (64 mel)             |
| S2ST | 720.4 ms | 583.5 ms | 65.9 ms | 32.3         | 31.0            | 2320.4 ms | 26240 smp (RTF **1.41×**)  |
| ASR  | 216.5 ms | 98.3 ms  | 42.7 ms | 36.2         | 27.6            | 328.7 ms  | 5 tok (64 mel)             |



| Task | T2U    | Vocoder | RTF   |
| ---- | ------ | ------- | ----- |
| T2ST | 866 ms | 664 ms  | 0.18× |
| S2ST | 848 ms | 574 ms  | 1.41× |


### Sequence length: 128


| Task | TTFT     | Encoder  | Prefill | decode t/s/u | ms/tok (steady) | E2E       | Output                     |
| ---- | -------- | -------- | ------- | ------------ | --------------- | --------- | -------------------------- |
| T2TT | 136.6 ms | 22.6 ms  | 43.9 ms | 126.8        | 7.9             | 1090.1 ms | 119 tok                    |
| T2ST | 173.5 ms | 38.2 ms  | 63.8 ms | 123.0        | 8.1             | 2917.4 ms | 406400 smp (RTF **0.11×**) |
| S2TT | 207.7 ms | 98.7 ms  | 41.7 ms | 69.0         | 14.5            | 370.3 ms  | 12 tok (128 mel)           |
| S2ST | 725.6 ms | 591.1 ms | 64.1 ms | 65.8         | 15.2            | 2400.1 ms | 47360 smp (RTF **0.81×**)  |
| ASR  | 207.0 ms | 98.4 ms  | 42.6 ms | 62.0         | 16.1            | 371.3 ms  | 11 tok (128 mel)           |



| Task | T2U    | Vocoder | RTF   |
| ---- | ------ | ------- | ----- |
| T2ST | 867 ms | 835 ms  | 0.11× |
| S2ST | 864 ms | 575 ms  | 0.81× |


### Sequence length: 256


| Task | TTFT     | Encoder  | Prefill | decode t/s/u | ms/tok (steady) | E2E       | Output                     |
| ---- | -------- | -------- | ------- | ------------ | --------------- | --------- | -------------------------- |
| T2TT | 140.2 ms | 24.5 ms  | 43.9 ms | 133.3        | 7.5             | 1770.3 ms | 213 tok                    |
| T2ST | 173.7 ms | 39.7 ms  | 61.9 ms | 131.7        | 7.6             | 3921.2 ms | 724800 smp (RTF **0.09×**) |
| S2TT | 219.4 ms | 103.0 ms | 45.2 ms | 98.1         | 10.2            | 500.6 ms  | 28 tok (256 mel)           |
| S2ST | 720.9 ms | 585.4 ms | 65.1 ms | 92.6         | 10.8            | 2509.6 ms | 91200 smp (RTF **0.44×**)  |
| ASR  | 214.2 ms | 103.5 ms | 43.2 ms | 88.5         | 11.3            | 445.1 ms  | 21 tok (256 mel)           |



| Task | T2U    | Vocoder | RTF   |
| ---- | ------ | ------- | ----- |
| T2ST | 889 ms | 1143 ms | 0.09× |
| S2ST | 769 ms | 660 ms  | 0.44× |


### Sequence length: 512


| Task | TTFT     | Encoder  | Prefill | decode t/s/u | ms/tok (steady) | E2E       | Output                     |
| ---- | -------- | -------- | ------- | ------------ | --------------- | --------- | -------------------------- |
| T2TT | 141.8 ms | 25.0 ms  | 44.3 ms | 129.6        | 7.7             | 2156.6 ms | 256 tok                    |
| T2ST | 179.4 ms | 38.1 ms  | 66.7 ms | 124.6        | 8.0             | 4432.8 ms | 802880 smp (RTF **0.09×**) |
| S2TT | 172.1 ms | 60.9 ms  | 43.1 ms | 105.5        | 9.5             | 579.5 ms  | 43 tok (512 mel)           |
| S2ST | 675.9 ms | 540.4 ms | 64.2 ms | 99.0         | 10.1            | 2589.4 ms | 145920 smp (RTF **0.28×**) |
| ASR  | 172.9 ms | 58.2 ms  | 45.8 ms | 102.3        | 9.8             | 502.8 ms  | 34 tok (512 mel)           |



| Task | T2U    | Vocoder | RTF   |
| ---- | ------ | ------- | ----- |
| T2ST | 914 ms | 1181 ms | 0.09× |
| S2ST | 791 ms | 666 ms  | 0.28× |


### Sequence length: 1024


| Task | TTFT     | Encoder  | Prefill | decode t/s/u | ms/tok (steady) | E2E       | Output                     |
| ---- | -------- | -------- | ------- | ------------ | --------------- | --------- | -------------------------- |
| T2TT | 157.1 ms | 26.2 ms  | 50.6 ms | 120.7        | 8.3             | 2318.7 ms | 256 tok                    |
| T2ST | 194.0 ms | 46.3 ms  | 66.2 ms | 121.0        | 8.3             | 4529.3 ms | 772480 smp (RTF **0.09×**) |
| S2TT | 247.6 ms | 106.6 ms | 64.4 ms | 118.3        | 8.5             | 913.7 ms  | 78 tok (1024 mel)          |
| S2ST | 245.5 ms | 106.5 ms | 60.7 ms | 117.1        | 8.5             | 2565.3 ms | 265280 smp (RTF **0.15×**) |
| ASR  | 242.0 ms | 106.7 ms | 61.6 ms | 115.1        | 8.7             | 847.1 ms  | 69 tok (1024 mel)          |



| Task | T2U    | Vocoder | RTF   |
| ---- | ------ | ------- | ----- |
| T2ST | 911 ms | 1200 ms | 0.09× |
| S2ST | 880 ms | 761 ms  | 0.15× |


### Sequence length: 2048


| Task | TTFT     | Encoder  | Prefill | decode t/s/u | ms/tok (steady) | E2E       | Output                     |
| ---- | -------- | -------- | ------- | ------------ | --------------- | --------- | -------------------------- |
| T2TT | 161.9 ms | 27.4 ms  | 48.3 ms | 109.0        | 9.2             | 2553.4 ms | 256 tok                    |
| T2ST | 199.0 ms | 43.6 ms  | 63.5 ms | 107.7        | 9.3             | 4784.0 ms | 766720 smp (RTF **0.10×**) |
| S2TT | 801.5 ms | 666.2 ms | 63.1 ms | 121.2        | 8.3             | 1798.5 ms | 119 tok (2048 mel)         |
| S2ST | 824.1 ms | 687.6 ms | 63.3 ms | 123.7        | 8.1             | 3783.3 ms | 423360 smp (RTF **0.14×**) |
| ASR  | 792.2 ms | 656.7 ms | 63.7 ms | 124.7        | 8.0             | 1917.9 ms | 138 tok (2048 mel)         |



| Task | T2U    | Vocoder | RTF   |
| ---- | ------ | ------- | ----- |
| T2ST | 924 ms | 1182 ms | 0.10× |
| S2ST | 894 ms | 897 ms  | 0.14× |


### Sequence length: 4096


| Task | TTFT     | Encoder  | Prefill | decode t/s/u | ms/tok (steady) | E2E       | Output                     |
| ---- | -------- | -------- | ------- | ------------ | --------------- | --------- | -------------------------- |
| T2TT | 224.8 ms | 64.3 ms  | 48.8 ms | 91.3         | 10.9            | 3066.3 ms | 256 tok                    |
| T2ST | 263.6 ms | 84.8 ms  | 67.4 ms | 89.8         | 11.1            | 5268.6 ms | 663680 smp (RTF **0.13×**) |
| S2TT | 830.4 ms | 689.2 ms | 62.9 ms | 121.8        | 8.2             | 2360.4 ms | 183 tok (4096 mel)         |
| S2ST | 828.4 ms | 689.7 ms | 62.3 ms | 123.8        | 8.1             | 4182.5 ms | 559360 smp (RTF **0.12×**) |
| ASR  | 827.3 ms | 689.2 ms | 62.6 ms | 124.2        | 8.1             | 2369.7 ms | 188 tok (4096 mel)         |



| Task | T2U    | Vocoder | RTF   |
| ---- | ------ | ------- | ----- |
| T2ST | 925 ms | 1128 ms | 0.13× |
| S2ST | 895 ms | 979 ms  | 0.12× |


---

## Demo wall timings (Blackhole P150, 2CQ + decode trace)

Single-chip P150 (`MeshShape(1, 1)`). Tables from `scripts/outputs/perf_sweep.txt` (2026-07-13).

**Note:** Several S2ST vocoder times can be cold JIT / first-shape compile outliers; compare against shorter lengths for steady-state. T2ST vocoder is closer to steady across lengths.

### Sequence length: 32

| Task | TTFT     | Encoder  | Prefill | decode t/s/u | ms/tok (steady) | E2E       | Output                     |
| ---- | -------- | -------- | ------- | ------------ | --------------- | --------- | -------------------------- |
| T2TT | 784.7 ms | 30.0 ms | 82.4 ms | 78.4 | 12.8 | 1118.5 ms | 27 tok |
| T2ST | 908.5 ms | 66.1 ms | 103.3 ms | 62.7 | 15.9 | 4729.6 ms | 103360 smp (RTF **0.73×**) |
| S2TT | 902.4 ms | 177.9 ms | 77.7 ms | 13.5 | 74.2 | 1051.7 ms | 3 tok (32 mel) |
| S2ST | 2648.7 ms | 1677.5 ms | 160.9 ms | 16.2 | 61.6 | 82707.0 ms | 20800 smp (RTF **63.62×**) |
| ASR | 901.2 ms | 174.4 ms | 81.0 ms | 42.7 | 23.4 | 1113.5 ms | 10 tok (32 mel) |


| Task | T2U    | Vocoder | RTF   |
| ---- | ------ | ------- | ----- |
| T2ST | 1943 ms | 1348 ms | 0.73× |
| S2ST | 5138 ms | 72052 ms | 63.62× |

### Sequence length: 64

| Task | TTFT     | Encoder  | Prefill | decode t/s/u | ms/tok (steady) | E2E       | Output                     |
| ---- | -------- | -------- | ------- | ------------ | --------------- | --------- | -------------------------- |
| T2TT | 883.3 ms | 71.7 ms | 82.1 ms | 95.4 | 10.5 | 1443.7 ms | 54 tok |
| T2ST | 907.3 ms | 101.2 ms | 109.6 ms | 89.5 | 11.2 | 5392.3 ms | 196800 smp (RTF **0.44×**) |
| S2TT | 913.5 ms | 166.2 ms | 78.7 ms | 29.4 | 34.1 | 1084.9 ms | 6 tok (64 mel) |
| S2ST | 2515.8 ms | 1649.6 ms | 114.2 ms | 29.0 | 34.5 | 81478.8 ms | 29120 smp (RTF **44.77×**) |
| ASR | 930.4 ms | 170.5 ms | 77.9 ms | 23.9 | 41.9 | 1099.1 ms | 5 tok (64 mel) |


| Task | T2U    | Vocoder | RTF   |
| ---- | ------ | ------- | ----- |
| T2ST | 2265 ms | 1531 ms | 0.44× |
| S2ST | 4340 ms | 71713 ms | 44.77× |

### Sequence length: 128

| Task | TTFT     | Encoder  | Prefill | decode t/s/u | ms/tok (steady) | E2E       | Output                     |
| ---- | -------- | -------- | ------- | ------------ | --------------- | --------- | -------------------------- |
| T2TT | 860.5 ms | 115.7 ms | 84.1 ms | 107.6 | 9.3 | 1966.3 ms | 119 tok |
| T2ST | 970.2 ms | 154.4 ms | 117.3 ms | 103.2 | 9.7 | 14196.2 ms | 406400 smp (RTF **0.56×**) |
| S2TT | 888.4 ms | 174.1 ms | 78.1 ms | 50.4 | 19.8 | 1108.2 ms | 12 tok (128 mel) |
| S2ST | 2371.2 ms | 1531.5 ms | 108.6 ms | 42.4 | 23.6 | 53503.7 ms | 47360 smp (RTF **18.08×**) |
| ASR | 921.0 ms | 178.3 ms | 79.8 ms | 45.8 | 21.8 | 1141.2 ms | 11 tok (128 mel) |


| Task | T2U    | Vocoder | RTF   |
| ---- | ------ | ------- | ----- |
| T2ST | 2341 ms | 9610 ms | 0.56× |
| S2ST | 2166 ms | 48566 ms | 18.08× |

### Sequence length: 256

| Task | TTFT     | Encoder  | Prefill | decode t/s/u | ms/tok (steady) | E2E       | Output                     |
| ---- | -------- | -------- | ------- | ------------ | --------------- | --------- | -------------------------- |
| T2TT | 1035.8 ms | 239.9 ms | 85.7 ms | 113.4 | 8.8 | 3301.4 ms | 256 tok |
| T2ST | 1171.6 ms | 285.9 ms | 123.4 ms | 109.3 | 9.1 | 60920.7 ms | 853440 smp (RTF **1.14×**) |
| S2TT | 911.7 ms | 170.3 ms | 83.1 ms | 74.4 | 13.4 | 1250.4 ms | 26 tok (256 mel) |
| S2ST | 2507.1 ms | 1605.2 ms | 127.1 ms | 65.5 | 15.3 | 82800.7 ms | 91200 smp (RTF **14.53×**) |
| ASR | 878.1 ms | 176.1 ms | 82.8 ms | 66.4 | 15.1 | 1181.8 ms | 21 tok (256 mel) |


| Task | T2U    | Vocoder | RTF   |
| ---- | ------ | ------- | ----- |
| T2ST | 2615 ms | 54609 ms | 1.14× |
| S2ST | 11010 ms | 66121 ms | 14.53× |

### Sequence length: 512

| Task | TTFT     | Encoder  | Prefill | decode t/s/u | ms/tok (steady) | E2E       | Output                     |
| ---- | -------- | -------- | ------- | ------------ | --------------- | --------- | -------------------------- |
| T2TT | 1246.5 ms | 420.1 ms | 82.5 ms | 111.0 | 9.0 | 3563.4 ms | 256 tok |
| T2ST | 1469.4 ms | 510.2 ms | 116.9 ms | 108.8 | 9.2 | 60699.8 ms | 802880 smp (RTF **1.21×**) |
| S2TT | 876.2 ms | 110.4 ms | 82.2 ms | 89.7 | 11.1 | 1348.2 ms | 43 tok (512 mel) |
| S2ST | 2437.4 ms | 1532.1 ms | 131.6 ms | 75.8 | 13.2 | 81922.2 ms | 135680 smp (RTF **9.66×**) |
| ASR | 871.9 ms | 116.8 ms | 82.4 ms | 82.5 | 12.1 | 1275.3 ms | 34 tok (512 mel) |


| Task | T2U    | Vocoder | RTF   |
| ---- | ------ | ------- | ----- |
| T2ST | 2668 ms | 54052 ms | 1.21× |
| S2ST | 8914 ms | 67285 ms | 9.66× |

### Sequence length: 1024

| Task | TTFT     | Encoder  | Prefill | decode t/s/u | ms/tok (steady) | E2E       | Output                     |
| ---- | -------- | -------- | ------- | ------------ | --------------- | --------- | -------------------------- |
| T2TT | 2046.0 ms | 1074.3 ms | 88.0 ms | 105.9 | 9.4 | 4470.4 ms | 256 tok |
| T2ST | 1983.8 ms | 971.2 ms | 137.1 ms | 103.5 | 9.7 | 71897.4 ms | 772800 smp (RTF **1.49×**) |
| S2TT | 1073.1 ms | 208.1 ms | 129.6 ms | 92.8 | 10.8 | 1898.5 ms | 77 tok (1024 mel) |
| S2ST | 1042.3 ms | 207.0 ms | 122.2 ms | 92.5 | 10.8 | 6206.7 ms | 263360 smp (RTF **0.38×**) |
| ASR | 1122.4 ms | 208.4 ms | 146.5 ms | 88.8 | 11.3 | 1894.9 ms | 69 tok (1024 mel) |


| Task | T2U    | Vocoder | RTF   |
| ---- | ------ | ------- | ----- |
| T2ST | 2548 ms | 64725 ms | 1.49× |
| S2ST | 2334 ms | 1961 ms | 0.38× |

### Sequence length: 2048

| Task | TTFT     | Encoder  | Prefill | decode t/s/u | ms/tok (steady) | E2E       | Output                     |
| ---- | -------- | -------- | ------- | ------------ | --------------- | --------- | -------------------------- |
| T2TT | 2781.8 ms | 1698.2 ms | 85.2 ms | 95.2 | 10.5 | 5484.3 ms | 256 tok |
| T2ST | 3048.3 ms | 1868.4 ms | 121.0 ms | 94.2 | 10.6 | 11543.2 ms | 767360 smp (RTF **0.24×**) |
| S2TT | 2830.4 ms | 1946.2 ms | 140.1 ms | 97.9 | 10.2 | 3973.9 ms | 112 tok (2048 mel) |
| S2ST | 2889.6 ms | 1988.9 ms | 133.8 ms | 102.7 | 9.7 | 9256.3 ms | 425600 smp (RTF **0.35×**) |
| ASR | 2826.1 ms | 1978.6 ms | 119.5 ms | 102.7 | 9.7 | 4167.3 ms | 138 tok (2048 mel) |


| Task | T2U    | Vocoder | RTF   |
| ---- | ------ | ------- | ----- |
| T2ST | 2632 ms | 2964 ms | 0.24× |
| S2ST | 2478 ms | 2421 ms | 0.35× |

### Sequence length: 4096

| Task | TTFT     | Encoder  | Prefill | decode t/s/u | ms/tok (steady) | E2E       | Output                     |
| ---- | -------- | -------- | ------- | ------------ | --------------- | --------- | -------------------------- |
| T2TT | 4681.9 ms | 3391.3 ms | 92.2 ms | 81.1 | 12.3 | 7842.1 ms | 256 tok |
| T2ST | 5462.2 ms | 4091.8 ms | 119.4 ms | 79.8 | 12.5 | 14229.9 ms | 663680 smp (RTF **0.34×**) |
| S2TT | 3315.1 ms | 2382.2 ms | 127.1 ms | 104.7 | 9.6 | 5358.3 ms | 213 tok (4096 mel) |
| S2ST | 3286.3 ms | 2378.5 ms | 116.5 ms | 104.5 | 9.6 | 104615.3 ms | 592000 smp (RTF **2.83×**) |
| ASR | 3305.0 ms | 2378.6 ms | 139.4 ms | 102.8 | 9.7 | 5170.1 ms | 191 tok (4096 mel) |


| Task | T2U    | Vocoder | RTF   |
| ---- | ------ | ------- | ----- |
| T2ST | 2585 ms | 2751 ms | 0.34× |
| S2ST | 14462 ms | 81141 ms | 2.83× |

---


Task notes:

- **T2TT** — text encoder + traced text-decoder loop.
- **T2ST** — text path + T2U + vocoder; vocoder dominates E2E; use **RTF** for speech QoS.
- **S2TT / ASR** — speech encoder dominates TTFT; **decode t/s/u** isolates the text-decoder steady rate.
- **S2ST** — speech encoder + decoder + T2U + vocoder.

**Cold start:** the first timed speech-synthesis call in a **brand-new process** with a cold on-disk JIT cache can still pay a one-time vocoder compile outlier (~15–20 s on BH QB; often larger on P150 when a new unit-length bucket is hit). Subsequent timed iters in the same session and later demo invocations (warm disk cache) report steady vocoder times (~1.1–1.6 s on BH QB; ~2–6 s on P150 when warm). Speech warmups + vocoder prewarm (in `demo.py`) keep the **reported** timed iter near steady state.
