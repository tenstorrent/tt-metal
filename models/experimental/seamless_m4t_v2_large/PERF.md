# Seamless M4T v2 Large — performance

Phase-separated timings from `[scripts/demo_perf_sweep.py](scripts/demo_perf_sweep.py)` with `generate(return_timings=True)`. Inputs:

- **Text** — *A Tale of Two Cities* (`models/tt_transformers/tests/tale-of-two-cities.txt.bz2`)
- **Audio** — concatenated LibriSpeech-dummy utterances (`scripts/outputs/long_speech_input_librispeech.wav`; preamble fallback if the dataset is unavailable)

Sequence lengths double **32 → 4096** (text = source tokens; speech = mel frames). Host pre/post-processing (token decode, WAV I/O) is excluded. Each task opens its own mesh device.

Sweep timing vs `demo.py`:

- Untimed warmups match the demo (text=1, speech=2).
- **1 timed iter** per task (report that elapsed; `demo.py` may use more).
- T2ST/S2ST use the vocoder post-warmup (weight prep after same-session warmups; at mel ≥ 1024, S2ST also runs untimed `vocode_units` on the timed device using throwaway unit ids).
- At mel **≥ 1024**, **S2TT / S2ST / ASR** warmups run on a throwaway mesh; the timed session prewarms the speech encoder. T2ST does not use split warmups.

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

Four-chip QuietBox (`MeshShape(1, 4)`, batch-1, TP=4). Tables from `scripts/outputs/perf_sweep.txt`.

### Sequence length: 32

| Task | TTFT | Encoder | Prefill | decode t/s/u | ms/tok (steady) | E2E | Output |
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

Single-chip P150 (`MeshShape(1, 1)`). Tables from `scripts/outputs/perf_sweep.txt`.

### Sequence length: 32

| Task | TTFT      | Encoder   | Prefill  | decode t/s/u | ms/tok (steady) | E2E        | Output                     |
| ---- | --------- | --------- | -------- | ------------ | --------------- | ---------- | -------------------------- |
| T2TT | 868.6 ms  | 25.0 ms   | 64.7 ms  | 80.07        | 12.5            | 1195.7 ms  | 27 tok                     |
| T2ST | 822.0 ms  | 47.7 ms   | 89.6 ms  | 71.91        | 13.9            | 4552.9 ms  | 103360 smp (RTF **0.70×**) |
| S2TT | 885.3 ms  | 151.3 ms  | 66.9 ms  | 14.55        | 68.7            | 1023.4 ms  | 3 tok (32 mel)             |
| S2ST | 2219.6 ms | 1467.9 ms | 95.6 ms  | 21.39        | 46.7            | 5548.9 ms  | 24640 smp (RTF **3.60×**) |
| ASR  | 894.4 ms  | 154.7 ms  | 70.4 ms  | 44.34        | 22.6            | 1099.0 ms  | 10 tok (32 mel)            |


| Task | T2U     | Vocoder | RTF   |
| ---- | ------- | ------- | ----- |
| T2ST | 1735 ms | 1567 ms | 0.70× |
| S2ST | 1979 ms | 1092 ms | 3.60× |

### Sequence length: 64

| Task | TTFT      | Encoder   | Prefill  | decode t/s/u | ms/tok (steady) | E2E        | Output                     |
| ---- | --------- | --------- | -------- | ------------ | --------------- | ---------- | -------------------------- |
| T2TT | 795.9 ms  | 65.9 ms   | 70.7 ms  | 96.03        | 10.4            | 1352.2 ms  | 54 tok                     |
| T2ST | 995.7 ms  | 85.5 ms   | 103.4 ms | 90.09        | 11.1            | 5155.9 ms  | 196800 smp (RTF **0.42×**) |
| S2TT | 859.0 ms  | 143.2 ms  | 69.0 ms  | 30.92        | 32.3            | 1021.7 ms  | 6 tok (64 mel)             |
| S2ST | 2316.3 ms | 1496.6 ms | 77.8 ms  | 23.04        | 43.4            | 4968.5 ms  | 26240 smp (RTF **3.03×**) |
| ASR  | 858.5 ms  | 149.9 ms  | 71.8 ms  | 25.65        | 39.0            | 1015.4 ms  | 5 tok (64 mel)             |


| Task | T2U     | Vocoder | RTF   |
| ---- | ------- | ------- | ----- |
| T2ST | 1984 ms | 1493 ms | 0.42× |
| S2ST | 1844 ms | 630 ms  | 3.03× |

### Sequence length: 128

| Task | TTFT      | Encoder   | Prefill  | decode t/s/u | ms/tok (steady) | E2E        | Output                     |
| ---- | --------- | --------- | -------- | ------------ | --------------- | ---------- | -------------------------- |
| T2TT | 849.5 ms  | 111.3 ms  | 68.8 ms  | 109.25       | 9.2             | 1938.6 ms  | 119 tok                    |
| T2ST | 869.0 ms  | 103.0 ms  | 76.3 ms  | 107.98       | 9.3             | 4611.0 ms  | 406400 smp (RTF **0.18×**) |
| S2TT | 857.3 ms  | 143.5 ms  | 66.9 ms  | 52.25        | 19.1            | 1069.2 ms  | 12 tok (128 mel)           |
| S2ST | 2229.9 ms | 1422.4 ms | 73.2 ms  | 51.47        | 19.4            | 5038.7 ms  | 47360 smp (RTF **1.70×**) |
| ASR  | 872.4 ms  | 153.1 ms  | 68.3 ms  | 48.12        | 20.8            | 1081.5 ms  | 11 tok (128 mel)           |


| Task | T2U     | Vocoder | RTF   |
| ---- | ------- | ------- | ----- |
| T2ST | 1917 ms | 715 ms  | 0.18× |
| S2ST | 1918 ms | 652 ms  | 1.70× |

### Sequence length: 256

| Task | TTFT      | Encoder   | Prefill  | decode t/s/u | ms/tok (steady) | E2E        | Output                     |
| ---- | --------- | --------- | -------- | ------------ | --------------- | ---------- | -------------------------- |
| T2TT | 962.9 ms  | 191.2 ms  | 71.8 ms  | 114.38       | 8.7             | 3207.4 ms  | 256 tok                    |
| T2ST | 1007.2 ms | 186.0 ms  | 73.3 ms  | 113.67       | 8.8             | 5994.0 ms  | 858240 smp (RTF **0.11×**) |
| S2TT | 887.2 ms  | 151.4 ms  | 71.8 ms  | 74.44        | 13.4            | 1226.2 ms  | 26 tok (256 mel)           |
| S2ST | 2172.0 ms | 1416.6 ms | 72.5 ms  | 79.53        | 12.6            | 4926.2 ms  | 91200 smp (RTF **0.86×**) |
| ASR  | 950.0 ms  | 150.0 ms  | 72.1 ms  | 68.61        | 14.6            | 1244.3 ms  | 21 tok (256 mel)           |


| Task | T2U     | Vocoder | RTF   |
| ---- | ------- | ------- | ----- |
| T2ST | 1939 ms | 774 ms  | 0.11× |
| S2ST | 1668 ms | 712 ms  | 0.86× |

### Sequence length: 512

| Task | TTFT      | Encoder   | Prefill  | decode t/s/u | ms/tok (steady) | E2E        | Output                     |
| ---- | --------- | --------- | -------- | ------------ | --------------- | ---------- | -------------------------- |
| T2TT | 1206.8 ms | 360.5 ms  | 72.1 ms  | 109.82       | 9.1             | 3548.6 ms  | 256 tok                    |
| T2ST | 1183.2 ms | 357.4 ms  | 72.1 ms  | 111.82       | 8.9             | 6211.3 ms  | 803200 smp (RTF **0.12×**) |
| S2TT | 848.9 ms  | 96.4 ms   | 71.6 ms  | 90.23        | 11.1            | 1318.4 ms  | 43 tok (512 mel)           |
| S2ST | 2144.0 ms | 1368.2 ms | 76.1 ms  | 84.70        | 11.8            | 4938.3 ms  | 135680 smp (RTF **0.58×**) |
| ASR  | 841.4 ms  | 96.0 ms   | 72.6 ms  | 85.06        | 11.8            | 1232.4 ms  | 34 tok (512 mel)           |


| Task | T2U     | Vocoder | RTF   |
| ---- | ------- | ------- | ----- |
| T2ST | 1920 ms | 797 ms  | 0.12× |
| S2ST | 1660 ms | 687 ms  | 0.58× |

### Sequence length: 1024

| Task | TTFT      | Encoder   | Prefill  | decode t/s/u | ms/tok (steady) | E2E        | Output                     |
| ---- | --------- | --------- | -------- | ------------ | --------------- | ---------- | -------------------------- |
| T2TT | 1512.3 ms | 657.8 ms  | 71.7 ms  | 106.52       | 9.4             | 3926.4 ms  | 256 tok                    |
| T2ST | 1557.5 ms | 684.3 ms  | 78.2 ms  | 105.27       | 9.5             | 6688.6 ms  | 772480 smp (RTF **0.14×**) |
| S2TT | 1047.3 ms | 206.8 ms  | 107.1 ms | 94.36        | 10.6            | 1859.6 ms  | 77 tok (1024 mel)          |
| S2ST | 1039.3 ms | 205.7 ms  | 104.8 ms | 96.58        | 10.4            | 5670.9 ms  | 264000 smp (RTF **0.34×**) |
| ASR  | 1094.7 ms | 205.8 ms  | 109.7 ms | 92.69        | 10.8            | 1832.8 ms  | 69 tok (1024 mel)          |


| Task | T2U     | Vocoder | RTF   |
| ---- | ------- | ------- | ----- |
| T2ST | 1925 ms | 748 ms  | 0.14× |
| S2ST | 2069 ms | 1789 ms | 0.34× |

### Sequence length: 2048

| Task | TTFT      | Encoder   | Prefill  | decode t/s/u | ms/tok (steady) | E2E        | Output                     |
| ---- | --------- | --------- | -------- | ------------ | --------------- | ---------- | -------------------------- |
| T2TT | 2402.7 ms | 1352.3 ms | 74.0 ms  | 95.59        | 10.5            | 5092.3 ms  | 256 tok                    |
| T2ST | 2457.1 ms | 1363.3 ms | 76.0 ms  | 96.25        | 10.4            | 7851.3 ms  | 767360 smp (RTF **0.16×**) |
| S2TT | 2749.4 ms | 1863.5 ms | 115.2 ms | 99.54        | 10.0            | 3864.1 ms  | 111 tok (2048 mel)         |
| S2ST | 2637.4 ms | 1809.5 ms | 108.1 ms | 105.14       | 9.5             | 8238.8 ms  | 425600 smp (RTF **0.31×**) |
| ASR  | 2803.8 ms | 1911.5 ms | 107.2 ms | 103.18       | 9.7             | 4141.9 ms  | 138 tok (2048 mel)         |


| Task | T2U     | Vocoder | RTF   |
| ---- | ------- | ------- | ----- |
| T2ST | 1953 ms | 758 ms  | 0.16× |
| S2ST | 2067 ms | 2224 ms | 0.31× |

### Sequence length: 4096

| Task | TTFT      | Encoder   | Prefill  | decode t/s/u | ms/tok (steady) | E2E         | Output                     |
| ---- | --------- | --------- | -------- | ------------ | --------------- | ----------- | -------------------------- |
| T2TT | 4174.2 ms | 2813.3 ms | 79.7 ms  | 80.74        | 12.4            | 7346.9 ms   | 256 tok                    |
| T2ST | 4088.1 ms | 2683.8 ms | 72.4 ms  | 81.36        | 12.3            | 9855.1 ms   | 664000 smp (RTF **0.24×**) |
| S2TT | 2288.7 ms | 1402.9 ms | 107.7 ms | 103.94       | 9.6             | 4135.7 ms   | 191 tok (4096 mel)         |
| S2ST | 2205.1 ms | 1393.5 ms | 105.8 ms | 105.30       | 9.5             | 8918.4 ms   | 608320 smp (RTF **0.23×**) |
| ASR  | 2387.4 ms | 1447.9 ms | 115.0 ms | 103.90       | 9.6             | 4213.1 ms   | 189 tok (4096 mel)         |


| Task | T2U     | Vocoder | RTF   |
| ---- | ------- | ------- | ----- |
| T2ST | 1867 ms | 734 ms  | 0.24× |
| S2ST | 2104 ms | 2633 ms | 0.23× |


---


Task notes:

- **T2TT** — text encoder + traced text-decoder loop.
- **T2ST** — text path + T2U + vocoder; vocoder dominates E2E; use **RTF** for speech QoS.
- **S2TT / ASR** — speech encoder dominates TTFT; **decode t/s/u** isolates the text-decoder steady rate.
- **S2ST** — speech encoder + decoder + T2U + vocoder.

**Cold start:** not reported as a separate metric. Warmups (text=1, speech=2) run before the timed iter. A new unit-length / `t_audio` bucket can still pay a one-time Metal compile on the timed T2ST/S2ST path. Short-audio S2ST can show RTF &gt; 1 even with a steady vocoder because the WAV is only ~1–3 s.
