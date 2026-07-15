# Seamless M4T v2 Large — performance

Phase-separated timings from `[scripts/demo_perf_sweep.py](scripts/demo_perf_sweep.py)` with `generate(return_timings=True)`. Inputs:

- **Text** — *A Tale of Two Cities* (`models/tt_transformers/tests/tale-of-two-cities.txt.bz2`)
- **Audio** — concatenated LibriSpeech-dummy utterances (`scripts/outputs/long_speech_input_librispeech.wav`; preamble fallback if the dataset is unavailable)

Sequence lengths double **32 → 4096** (text = source tokens; speech = mel frames). Host pre/post-processing (token decode, WAV I/O) is excluded. Each task opens its own mesh device.

Sweep timing vs `demo.py`:

- Untimed warmups match the demo (text=1, speech=2).
- **2 timed iters** for T2ST and for S2ST when mel **< 1024** (report min); **1 timed iter** for other tasks and for S2ST at mel ≥ 1024.
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

Single-chip P150 (`MeshShape(1, 1)`). Tables from `scripts/outputs/perf_sweep.txt`.

**Note:** T2ST vocoder is steady across lengths (~1.4–3.4 s). S2ST still shows cold JIT on new unit/`t_audio` buckets — **256** (vocoder ~62 s) and **4096** (vocoder ~66 s, T2U ~11 s). Treat those points as compile outliers; warm S2ST vocoder is ~1.5–2.5 s (e.g. 1024 / 2048).

### Sequence length: 32

| Task | TTFT      | Encoder   | Prefill  | decode t/s/u | ms/tok (steady) | E2E        | Output                     |
| ---- | --------- | --------- | -------- | ------------ | --------------- | ---------- | -------------------------- |
| T2TT | 847.6 ms  | 29.1 ms   | 75.9 ms  | 78.6         | 12.7            | 1180.9 ms  | 27 tok                     |
| T2ST | 841.2 ms  | 54.6 ms   | 100.7 ms | 67.3         | 14.8            | 4578.9 ms  | 103360 smp (RTF **0.71×**) |
| S2TT | 952.4 ms  | 172.3 ms  | 82.5 ms  | 12.4         | 80.9            | 1114.9 ms  | 3 tok (32 mel)             |
| S2ST | 2409.8 ms | 1558.4 ms | 113.1 ms | 18.3         | 54.6            | 6623.4 ms  | 20800 smp (RTF **5.09×**)  |
| ASR  | 945.0 ms  | 166.0 ms  | 79.2 ms  | 43.3         | 23.1            | 1154.4 ms  | 10 tok (32 mel)            |


| Task | T2U     | Vocoder | RTF   |
| ---- | ------- | ------- | ----- |
| T2ST | 1866 ms | 1391 ms | 0.71× |
| S2ST | 2295 ms | 1571 ms | 5.09× |

### Sequence length: 64

| Task | TTFT      | Encoder   | Prefill  | decode t/s/u | ms/tok (steady) | E2E        | Output                     |
| ---- | --------- | --------- | -------- | ------------ | --------------- | ---------- | -------------------------- |
| T2TT | 853.6 ms  | 71.2 ms   | 80.5 ms  | 96.3         | 10.4            | 1407.6 ms  | 54 tok                     |
| T2ST | 901.9 ms  | 98.0 ms   | 109.3 ms | 87.5         | 11.4            | 5281.1 ms  | 196800 smp (RTF **0.43×**) |
| S2TT | 914.8 ms  | 166.0 ms  | 77.3 ms  | 28.5         | 35.1            | 1091.3 ms  | 6 tok (64 mel)             |
| S2ST | 2291.8 ms | 1525.7 ms | 106.1 ms | 19.4         | 51.4            | 6360.8 ms  | 26240 smp (RTF **3.88×**)  |
| ASR  | 968.7 ms  | 214.0 ms  | 95.5 ms  | 21.1         | 47.4            | 1159.5 ms  | 5 tok (64 mel)             |


| Task | T2U     | Vocoder | RTF   |
| ---- | ------- | ------- | ----- |
| T2ST | 2152 ms | 1528 ms | 0.43× |
| S2ST | 2185 ms | 1571 ms | 3.88× |

### Sequence length: 128

| Task | TTFT      | Encoder   | Prefill  | decode t/s/u | ms/tok (steady) | E2E        | Output                     |
| ---- | --------- | --------- | -------- | ------------ | --------------- | ---------- | -------------------------- |
| T2TT | 910.2 ms  | 115.8 ms  | 83.0 ms  | 108.0        | 9.3             | 2012.0 ms  | 119 tok                    |
| T2ST | 964.4 ms  | 154.1 ms  | 109.6 ms | 104.4        | 9.6             | 6604.3 ms  | 406400 smp (RTF **0.26×**) |
| S2TT | 922.8 ms  | 169.5 ms  | 79.6 ms  | 50.2         | 19.9            | 1143.6 ms  | 12 tok (128 mel)           |
| S2ST | 2472.3 ms | 1600.1 ms | 113.5 ms | 40.6         | 24.6            | 6734.6 ms  | 47360 smp (RTF **2.28×**)  |
| ASR  | 936.0 ms  | 170.3 ms  | 79.1 ms  | 45.6         | 21.9            | 1157.2 ms  | 11 tok (128 mel)           |


| Task | T2U     | Vocoder | RTF   |
| ---- | ------- | ------- | ----- |
| T2ST | 2305 ms | 2050 ms | 0.26× |
| S2ST | 2256 ms | 1537 ms | 2.28× |

### Sequence length: 256

| Task | TTFT      | Encoder   | Prefill  | decode t/s/u | ms/tok (steady) | E2E         | Output                     |
| ---- | --------- | --------- | -------- | ------------ | --------------- | ----------- | -------------------------- |
| T2TT | 1023.9 ms | 227.9 ms  | 83.5 ms  | 112.5        | 8.9             | 3311.4 ms   | 256 tok                    |
| T2ST | 1209.0 ms | 295.4 ms  | 111.0 ms | 112.2        | 8.9             | 9666.2 ms   | 853440 smp (RTF **0.18×**) |
| S2TT | 925.7 ms  | 163.4 ms  | 81.2 ms  | 74.6         | 13.4            | 1263.7 ms   | 26 tok (256 mel)           |
| S2ST | 2512.7 ms | 1594.0 ms | 131.3 ms | 63.6         | 15.7            | 15096.7 ms  | 91200 smp (RTF **2.65×**)  |
| ASR  | 944.7 ms  | 199.1 ms  | 96.1 ms  | 69.4         | 14.4            | 1234.9 ms   | 21 tok (256 mel)           |


| Task | T2U     | Vocoder | RTF   |
| ---- | ------- | ------- | ----- |
| T2ST | 2677 ms | 3343 ms | 0.18× |
| S2ST | 4078 ms | 5404 ms | 2.65× |

### Sequence length: 512

| Task | TTFT      | Encoder   | Prefill  | decode t/s/u | ms/tok (steady) | E2E         | Output                     |
| ---- | --------- | --------- | -------- | ------------ | --------------- | ----------- | -------------------------- |
| T2TT | 1254.8 ms | 437.8 ms  | 89.2 ms  | 110.6        | 9.0             | 3575.0 ms   | 256 tok                    |
| T2ST | 1554.2 ms | 632.9 ms  | 121.7 ms | 108.4        | 9.2             | 10009.9 ms  | 802880 smp (RTF **0.20×**) |
| S2TT | 842.7 ms  | 115.6 ms  | 86.0 ms  | 87.4         | 11.4            | 1327.9 ms   | 43 tok (512 mel)           |
| S2ST | 2294.2 ms | 1511.5 ms | 124.2 ms | 72.2         | 13.8            | 7651.1 ms   | 135680 smp (RTF **0.90×**) |
| ASR  | 847.9 ms  | 115.1 ms  | 86.2 ms  | 81.4         | 12.3            | 1257.1 ms   | 34 tok (512 mel)           |


| Task | T2U     | Vocoder | RTF   |
| ---- | ------- | ------- | ----- |
| T2ST | 2638 ms | 3296 ms | 0.20× |
| S2ST | 2756 ms | 1944 ms | 0.90× |

### Sequence length: 1024

| Task | TTFT      | Encoder   | Prefill  | decode t/s/u | ms/tok (steady) | E2E         | Output                     |
| ---- | --------- | --------- | -------- | ------------ | --------------- | ----------- | -------------------------- |
| T2TT | 1738.1 ms | 819.2 ms  | 85.4 ms  | 106.1        | 9.4             | 4160.3 ms   | 256 tok                    |
| T2ST | 2029.4 ms | 969.6 ms  | 123.6 ms | 103.5        | 9.7             | 10504.7 ms  | 772800 smp (RTF **0.22×**) |
| S2TT | 1046.8 ms | 206.6 ms  | 118.0 ms | 94.2         | 10.6            | 1860.6 ms   | 77 tok (1024 mel)          |
| S2ST | 1072.1 ms | 206.9 ms  | 119.5 ms | 93.2         | 10.7            | 6243.7 ms   | 263360 smp (RTF **0.38×**) |
| ASR  | 1073.6 ms | 206.3 ms  | 117.9 ms | 91.3         | 11.0            | 1822.2 ms   | 69 tok (1024 mel)          |


| Task | T2U     | Vocoder | RTF   |
| ---- | ------- | ------- | ----- |
| T2ST | 2560 ms | 3270 ms | 0.22× |
| S2ST | 2366 ms | 1936 ms | 0.38× |

### Sequence length: 2048

| Task | TTFT      | Encoder   | Prefill  | decode t/s/u | ms/tok (steady) | E2E         | Output                     |
| ---- | --------- | --------- | -------- | ------------ | --------------- | ----------- | -------------------------- |
| T2TT | 2694.2 ms | 1600.4 ms | 84.6 ms  | 96.1         | 10.4            | 5366.4 ms   | 256 tok                    |
| T2ST | 2940.6 ms | 1792.4 ms | 114.3 ms | 93.5         | 10.7            | 11334.4 ms  | 767360 smp (RTF **0.24×**) |
| S2TT | 2928.8 ms | 1936.5 ms | 156.1 ms | 100.7        | 9.9             | 4037.6 ms   | 112 tok (2048 mel)         |
| S2ST | 2932.0 ms | 2054.3 ms | 118.5 ms | 102.5        | 9.8             | 9256.4 ms   | 425600 smp (RTF **0.35×**) |
| ASR  | 2853.0 ms | 1990.5 ms | 124.9 ms | 102.7        | 9.7             | 4194.5 ms   | 138 tok (2048 mel)         |


| Task | T2U     | Vocoder | RTF   |
| ---- | ------- | ------- | ----- |
| T2ST | 2531 ms | 2957 ms | 0.24× |
| S2ST | 2448 ms | 2427 ms | 0.35× |

### Sequence length: 4096

| Task | TTFT      | Encoder   | Prefill  | decode t/s/u | ms/tok (steady) | E2E         | Output                     |
| ---- | --------- | --------- | -------- | ------------ | --------------- | ----------- | -------------------------- |
| T2TT | 4643.5 ms | 3406.3 ms | 85.5 ms  | 81.0         | 12.3            | 7810.4 ms   | 256 tok                    |
| T2ST | 4969.3 ms | 3617.5 ms | 116.8 ms | 79.5         | 12.6            | 13808.3 ms  | 663680 smp (RTF **0.33×**) |
| S2TT | 3387.7 ms | 2379.5 ms | 156.1 ms | 104.6        | 9.6             | 5352.7 ms   | 205 tok (4096 mel)         |
| S2ST | 3314.6 ms | 2378.6 ms | 139.6 ms | 100.5        | 9.9             | 86669.8 ms  | 552640 smp (RTF **2.51×**) † |
| ASR  | 3313.4 ms | 2378.7 ms | 118.1 ms | 103.0        | 9.7             | 5046.0 ms   | 178 tok (4096 mel)         |


| Task | T2U      | Vocoder  | RTF   |
| ---- | -------- | -------- | ----- |
| T2ST | 2609 ms  | 2823 ms  | 0.33× |
| S2ST | 10680 ms | 66326 ms | 2.51× † |


---


Task notes:

- **T2TT** — text encoder + traced text-decoder loop.
- **T2ST** — text path + T2U + vocoder; vocoder dominates E2E; use **RTF** for speech QoS.
- **S2TT / ASR** — speech encoder dominates TTFT; **decode t/s/u** isolates the text-decoder steady rate.
- **S2ST** — speech encoder + decoder + T2U + vocoder.

**Cold start:** a new unit-length / `t_audio` bucket can still pay a one-time Metal compile on the timed S2ST path (~60 s vocoder on P150 in this sweep at mel 256 and 4096). Warm steady vocoder is ~1.4–3.4 s (T2ST) and ~1.5–2.5 s (S2ST). Speech warmups + vocoder post-warmup / split-path `vocode_units` JIT keep most reported points near steady state; BH QB cold outliers are typically smaller (~15–20 s) when the on-disk cache is empty.
