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

Single-chip P150 (`MeshShape(1, 1)`). Tables from `scripts/outputs/perf_sweep.txt`.

**Note:** Several S2ST vocoder times (~35–64 s) are cold JIT / first-shape compile outliers; compare against S2ST@64 (~2.0 s vocoder) for steady-state. T2ST vocoder is closer to steady across lengths.

### Sequence length: 32


| Task | TTFT      | Encoder   | Prefill  | decode t/s/u | ms/tok (steady) | E2E        | Output                     |
| ---- | --------- | --------- | -------- | ------------ | --------------- | ---------- | -------------------------- |
| T2TT | 764.8 ms  | 28.6 ms   | 82.6 ms  | 77.7         | 12.9            | 1102.4 ms  | 27 tok                     |
| T2ST | 905.3 ms  | 59.6 ms   | 101.4 ms | 64.1         | 15.6            | 6299.3 ms  | 103040 smp (RTF **0.98×**) |
| S2TT | 2577.7 ms | 1689.3 ms | 124.3 ms | 17.8         | 56.0            | 2803.1 ms  | 5 tok (32 mel)             |
| S2ST | 2789.1 ms | 1844.5 ms | 141.2 ms | 16.7         | 60.0            | 77606.6 ms | 24640 smp (RTF **50.39×**) |
| ASR  | 2517.5 ms | 1675.0 ms | 115.8 ms | 32.7         | 30.6            | 2763.5 ms  | 9 tok (32 mel)             |



| Task | T2U     | Vocoder  | RTF    |
| ---- | ------- | -------- | ------ |
| T2ST | 2444 ms | 2322 ms  | 0.98×  |
| S2ST | 7490 ms | 63944 ms | 50.39× |


### Sequence length: 64


| Task | TTFT      | Encoder   | Prefill  | decode t/s/u | ms/tok (steady) | E2E       | Output                     |
| ---- | --------- | --------- | -------- | ------------ | --------------- | --------- | -------------------------- |
| T2TT | 903.3 ms  | 76.6 ms   | 81.9 ms  | 97.1         | 10.3            | 1452.4 ms | 54 tok                     |
| T2ST | 935.1 ms  | 104.1 ms  | 113.2 ms | 88.2         | 11.3            | 8262.9 ms | 196800 smp (RTF **0.67×**) |
| S2TT | 2444.4 ms | 1593.7 ms | 107.8 ms | 23.2         | 43.1            | 2661.0 ms | 6 tok (64 mel)             |
| S2ST | 2760.1 ms | 1773.8 ms | 126.1 ms | 15.1         | 66.1            | 7517.6 ms | 26240 smp (RTF **4.58×**)  |
| ASR  | 2511.2 ms | 1685.9 ms | 112.3 ms | 18.1         | 55.2            | 2733.1 ms | 5 tok (64 mel)             |



| Task | T2U     | Vocoder | RTF   |
| ---- | ------- | ------- | ----- |
| T2ST | 3950 ms | 2564 ms | 0.67× |
| S2ST | 2359 ms | 2026 ms | 4.58× |


### Sequence length: 128


| Task | TTFT      | Encoder   | Prefill  | decode t/s/u | ms/tok (steady) | E2E        | Output                     |
| ---- | --------- | --------- | -------- | ------------ | --------------- | ---------- | -------------------------- |
| T2TT | 978.6 ms  | 122.6 ms  | 82.8 ms  | 110.6        | 9.0             | 2054.1 ms  | 119 tok                    |
| T2ST | 1031.4 ms | 154.4 ms  | 114.8 ms | 105.2        | 9.5             | 10269.8 ms | 403520 smp (RTF **0.41×**) |
| S2TT | 2700.7 ms | 1782.6 ms | 125.6 ms | 39.0         | 25.7            | 2984.7 ms  | 12 tok (128 mel)           |
| S2ST | 2819.7 ms | 1918.5 ms | 121.4 ms | 39.3         | 25.5            | 65422.5 ms | 47360 smp (RTF **22.10×**) |
| ASR  | 2667.5 ms | 1755.7 ms | 122.6 ms | 35.0         | 28.6            | 2955.1 ms  | 11 tok (128 mel)           |



| Task | T2U     | Vocoder  | RTF    |
| ---- | ------- | -------- | ------ |
| T2ST | 4171 ms | 3649 ms  | 0.41×  |
| S2ST | 2329 ms | 59873 ms | 22.10× |


### Sequence length: 256


| Task | TTFT      | Encoder   | Prefill  | decode t/s/u | ms/tok (steady) | E2E        | Output                     |
| ---- | --------- | --------- | -------- | ------------ | --------------- | ---------- | -------------------------- |
| T2TT | 1033.8 ms | 234.2 ms  | 80.3 ms  | 116.3        | 8.6             | 3243.7 ms  | 256 tok                    |
| T2ST | 1311.4 ms | 313.2 ms  | 119.4 ms | 111.5        | 9.0             | 14861.5 ms | 853760 smp (RTF **0.28×**) |
| S2TT | 2670.3 ms | 1756.5 ms | 125.4 ms | 63.6         | 15.7            | 3066.7 ms  | 26 tok (256 mel)           |
| S2ST | 2835.1 ms | 1894.9 ms | 126.4 ms | 60.0         | 16.7            | 66676.6 ms | 86720 smp (RTF **12.30×**) |
| ASR  | 2761.0 ms | 1845.2 ms | 131.9 ms | 53.6         | 18.7            | 3136.4 ms  | 21 tok (256 mel)           |



| Task | T2U     | Vocoder  | RTF    |
| ---- | ------- | -------- | ------ |
| T2ST | 4605 ms | 6157 ms  | 0.28×  |
| S2ST | 8463 ms | 51817 ms | 12.30× |


### Sequence length: 512


| Task | TTFT      | Encoder   | Prefill  | decode t/s/u | ms/tok (steady) | E2E        | Output                     |
| ---- | --------- | --------- | -------- | ------------ | --------------- | ---------- | -------------------------- |
| T2TT | 1396.3 ms | 448.9 ms  | 84.5 ms  | 111.6        | 9.0             | 3700.2 ms  | 256 tok                    |
| T2ST | 1589.6 ms | 566.4 ms  | 125.1 ms | 107.8        | 9.3             | 14236.1 ms | 797440 smp (RTF **0.29×**) |
| S2TT | 2486.8 ms | 1604.8 ms | 122.0 ms | 79.9         | 12.5            | 3028.9 ms  | 44 tok (512 mel)           |
| S2ST | 2635.7 ms | 1755.0 ms | 126.2 ms | 74.4         | 13.4            | 66172.5 ms | 128320 smp (RTF **8.25×**) |
| ASR  | 2661.0 ms | 1731.1 ms | 132.2 ms | 70.2         | 14.2            | 3148.5 ms  | 35 tok (512 mel)           |



| Task | T2U     | Vocoder  | RTF   |
| ---- | ------- | -------- | ----- |
| T2ST | 4484 ms | 5261 ms  | 0.29× |
| S2ST | 8154 ms | 51725 ms | 8.25× |


### Sequence length: 1024


| Task | TTFT       | Encoder    | Prefill  | decode t/s/u | ms/tok (steady) | E2E        | Output                     |
| ---- | ---------- | ---------- | -------- | ------------ | --------------- | ---------- | -------------------------- |
| T2TT | 2078.8 ms  | 1077.7 ms  | 102.7 ms | 104.5        | 9.6             | 4542.8 ms  | 256 tok                    |
| T2ST | 2345.5 ms  | 1209.9 ms  | 139.1 ms | 102.9        | 9.7             | 15673.8 ms | 771840 smp (RTF **0.32×**) |
| S2TT | 12945.5 ms | 11998.4 ms | 148.5 ms | 95.2         | 10.5            | 13790.9 ms | 81 tok (1024 mel)          |
| S2ST | 13376.4 ms | 12424.4 ms | 154.9 ms | 112.3        | 8.9             | 62135.5 ms | 600960 smp (RTF **1.65×**) |
| ASR  | 13364.7 ms | 12329.2 ms | 205.2 ms | 90.4         | 11.1            | 14410.2 ms | 95 tok (1024 mel)          |



| Task | T2U     | Vocoder  | RTF   |
| ---- | ------- | -------- | ----- |
| T2ST | 4674 ms | 5679 ms  | 0.32× |
| S2ST | 6204 ms | 35583 ms | 1.65× |


### Sequence length: 2048


| Task | TTFT       | Encoder    | Prefill  | decode t/s/u | ms/tok (steady) | E2E        | Output                     |
| ---- | ---------- | ---------- | -------- | ------------ | --------------- | ---------- | -------------------------- |
| T2TT | 2935.0 ms  | 1801.4 ms  | 90.8 ms  | 96.2         | 10.4            | 5604.3 ms  | 256 tok                    |
| T2ST | 3170.7 ms  | 1986.4 ms  | 128.8 ms | 93.1         | 10.7            | 15961.7 ms | 766720 smp (RTF **0.33×**) |
| S2TT | 23155.8 ms | 22136.8 ms | 142.1 ms | 97.6         | 10.2            | 24344.5 ms | 116 tok (2048 mel)         |
| S2ST | 22987.9 ms | 21961.8 ms | 129.5 ms | 100.1        | 10.0            | 73607.1 ms | 430080 smp (RTF **2.74×**) |
| ASR  | 23083.4 ms | 22093.8 ms | 120.1 ms | 98.4         | 10.2            | 24466.3 ms | 136 tok (2048 mel)         |



| Task | T2U     | Vocoder  | RTF   |
| ---- | ------- | -------- | ----- |
| T2ST | 4464 ms | 5166 ms  | 0.33× |
| S2ST | 6738 ms | 38030 ms | 2.74× |


### Sequence length: 4096


| Task | TTFT      | Encoder   | Prefill  | decode t/s/u | ms/tok (steady) | E2E        | Output                     |
| ---- | --------- | --------- | -------- | ------------ | --------------- | ---------- | -------------------------- |
| T2TT | 4915.9 ms | 3622.7 ms | 89.6 ms  | 81.2         | 12.3            | 8073.9 ms  | 256 tok                    |
| T2ST | 5283.8 ms | 3857.9 ms | 124.6 ms | 79.8         | 12.5            | 18313.5 ms | 663040 smp (RTF **0.44×**) |
| S2TT | 7624.5 ms | 6654.4 ms | 121.8 ms | 90.4         | 11.1            | 8459.4 ms  | 76 tok (4096 mel)          |
| S2ST | 7598.6 ms | 6641.8 ms | 124.6 ms | 87.2         | 11.5            | 74236.9 ms | 244800 smp (RTF **4.85×**) |
| ASR  | 7691.7 ms | 6691.7 ms | 120.7 ms | 87.8         | 11.4            | 8518.0 ms  | 73 tok (4096 mel)          |



| Task | T2U     | Vocoder  | RTF   |
| ---- | ------- | -------- | ----- |
| T2ST | 4549 ms | 4866 ms  | 0.44× |
| S2ST | 9622 ms | 52984 ms | 4.85× |


---

Task notes:

- **T2TT** — text encoder + traced text-decoder loop.
- **T2ST** — text path + T2U + vocoder; vocoder dominates E2E; use **RTF** for speech QoS.
- **S2TT / ASR** — speech encoder dominates TTFT; **decode t/s/u** isolates the text-decoder steady rate.
- **S2ST** — speech encoder + decoder + T2U + vocoder.

**Cold start:** the first timed speech-synthesis call in a **brand-new process** with a cold on-disk JIT cache can still pay a one-time vocoder compile outlier (~15–20 s on BH QB; often larger on P150 when a new unit-length bucket is hit). Subsequent timed iters in the same session and later demo invocations (warm disk cache) report steady vocoder times (~1.1–1.6 s on BH QB; ~2–6 s on P150 when warm). Speech warmups + vocoder prewarm (in `demo.py`) keep the **reported** timed iter near steady state.
