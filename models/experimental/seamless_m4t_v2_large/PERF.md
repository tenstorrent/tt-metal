# Seamless M4T v2 Large — performance

## Demo wall timings (Blackhole BH QB, 2CQ + decode trace)

Phase-separated timings from `[scripts/demo_perf_sweep.py](scripts/demo_perf_sweep.py)` with `generate(return_timings=True)` on a four-chip Blackhole QB host (`MeshShape(1, 4)`). Inputs:

- **Text** — *A Tale of Two Cities* (`models/tt_transformers/tests/tale-of-two-cities.txt.bz2`)
- **Audio** — concatenated LibriSpeech-dummy utterances (`scripts/outputs/long_speech_input_librispeech.wav`; preamble fallback if the dataset is unavailable)

Sequence lengths double **32 → 4096** (text = source tokens; speech = mel frames). Host pre/post-processing (token decode, WAV I/O) is excluded. Each task opens its own mesh device.

Sweep timing vs `demo.py`:

- Untimed warmups match the demo (text=1, speech=2).
- **1 timed iter** per task (demo uses 2 timed iters for T2ST/S2ST).
- No vocoder `post_warmup_fn` in the sweep (demo prewarms vocoder convs after speech warmups).
- At mel **≥ 1792**, **S2TT / S2ST / ASR** warmups run on a throwaway mesh device; T2ST does not.

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

Tables below are from a representative BH QB run. Re-run the sweep locally to refresh numbers; a partial `scripts/outputs/perf_sweep.txt` in-tree may not match these tables.

Reproduce:

```bash
python models/experimental/seamless_m4t_v2_large/scripts/demo_perf_sweep.py
```

### Sequence length: 32


| Task | TTFT      | Encoder  | Prefill  | decode t/s/u | ms/tok (steady) | E2E       | Output                     |
| ---- | --------- | -------- | -------- | ------------ | --------------- | --------- | -------------------------- |
| T2TT | 151.8 ms  | 23.6 ms  | 48.2 ms  | 107.1        | 9.3             | 447.6 ms  | 32 tok                     |
| T2ST | 215.4 ms  | 35.1 ms  | 59.5 ms  | 100.3        | 10.0            | 2564.0 ms | 100480 smp (RTF **0.41×**) |
| S2TT | 869.1 ms  | 709.7 ms | 89.3 ms  | 7.9          | 127.1           | 997.2 ms  | 2 tok (32 mel)             |
| S2ST | 976.2 ms  | 778.6 ms | 100.5 ms | 36.9         | 27.1            | 3101.0 ms | 11840 smp (RTF **4.19×**)  |
| ASR  | 1080.4 ms | 864.5 ms | 69.8 ms  | 18.4         | 54.5            | 1245.3 ms | 4 tok (32 mel)             |



| Task | T2U    | Vocoder | RTF   |
| ---- | ------ | ------- | ----- |
| T2ST | 872 ms | 983 ms  | 0.41× |
| S2ST | 938 ms | 987 ms  | 4.19× |


### Sequence length: 64


| Task | TTFT      | Encoder  | Prefill | decode t/s/u | ms/tok (steady) | E2E       | Output                     |
| ---- | --------- | -------- | ------- | ------------ | --------------- | --------- | -------------------------- |
| T2TT | 140.3 ms  | 25.1 ms  | 48.3 ms | 130.8        | 7.6             | 1074.3 ms | 120 tok                    |
| T2ST | 232.6 ms  | 63.7 ms  | 66.4 ms | 125.5        | 8.0             | 5452.0 ms | 315520 smp (RTF **0.28×**) |
| S2TT | 972.6 ms  | 774.9 ms | 95.8 ms | 18.2         | 55.0            | 1083.9 ms | 3 tok (64 mel)             |
| S2ST | 1065.6 ms | 859.6 ms | 68.1 ms | 24.2         | 41.3            | 3355.4 ms | 24960 smp (RTF **2.15×**)  |
| ASR  | 1149.8 ms | 918.0 ms | 71.5 ms | 17.3         | 57.7            | 1324.5 ms | 4 tok (64 mel)             |



| Task | T2U     | Vocoder | RTF   |
| ---- | ------- | ------- | ----- |
| T2ST | 1636 ms | 2262 ms | 0.28× |
| S2ST | 971 ms  | 1081 ms | 2.15× |


### Sequence length: 128


| Task | TTFT      | Encoder  | Prefill  | decode t/s/u | ms/tok (steady) | E2E       | Output                     |
| ---- | --------- | -------- | -------- | ------------ | --------------- | --------- | -------------------------- |
| T2TT | 177.0 ms  | 25.9 ms  | 48.5 ms  | 124.3        | 8.0             | 1084.3 ms | 111 tok                    |
| T2ST | 244.1 ms  | 69.6 ms  | 98.0 ms  | 90.8         | 11.0            | 5331.9 ms | 348160 smp (RTF **0.25×**) |
| S2TT | 1007.0 ms | 796.0 ms | 105.5 ms | 42.1         | 23.8            | 1199.3 ms | 9 tok (128 mel)            |
| S2ST | 1086.7 ms | 866.3 ms | 68.7 ms  | 44.5         | 22.5            | 3623.3 ms | 37120 smp (RTF **1.56×**)  |
| ASR  | 1189.1 ms | 944.1 ms | 72.1 ms  | 41.3         | 24.2            | 1410.0 ms | 10 tok (128 mel)           |



| Task | T2U     | Vocoder | RTF   |
| ---- | ------- | ------- | ----- |
| T2ST | 1683 ms | 1761 ms | 0.25× |
| S2ST | 1024 ms | 1221 ms | 1.56× |


### Sequence length: 256


| Task | TTFT      | Encoder   | Prefill | decode t/s/u | ms/tok (steady) | E2E       | Output                     |
| ---- | --------- | --------- | ------- | ------------ | --------------- | --------- | -------------------------- |
| T2TT | 149.3 ms  | 27.6 ms   | 48.8 ms | 136.0        | 7.4             | 1823.4 ms | 223 tok                    |
| T2ST | 292.3 ms  | 75.8 ms   | 68.1 ms | 130.5        | 7.7             | 6953.4 ms | 563520 smp (RTF **0.20×**) |
| S2TT | 1109.9 ms | 888.5 ms  | 72.7 ms | 58.7         | 17.0            | 1351.7 ms | 15 tok (256 mel)           |
| S2ST | 1221.9 ms | 992.1 ms  | 70.2 ms | 64.5         | 15.5            | 4567.1 ms | 83520 smp (RTF **0.87×**)  |
| ASR  | 1357.2 ms | 1085.9 ms | 76.1 ms | 46.2         | 21.7            | 1707.9 ms | 17 tok (256 mel)           |



| Task | T2U     | Vocoder | RTF   |
| ---- | ------- | ------- | ----- |
| T2ST | 1762 ms | 2559 ms | 0.20× |
| S2ST | 1133 ms | 1777 ms | 0.87× |


### Sequence length: 512


| Task | TTFT      | Encoder  | Prefill  | decode t/s/u | ms/tok (steady) | E2E       | Output                     |
| ---- | --------- | -------- | -------- | ------------ | --------------- | --------- | -------------------------- |
| T2TT | 151.3 ms  | 28.1 ms  | 50.1 ms  | 132.3        | 7.6             | 2125.5 ms | 256 tok                    |
| T2ST | 303.1 ms  | 78.0 ms  | 107.4 ms | 128.8        | 7.8             | 7208.7 ms | 541120 smp (RTF **0.21×**) |
| S2TT | 1026.6 ms | 805.8 ms | 73.9 ms  | 81.7         | 12.2            | 1363.2 ms | 28 tok (512 mel)           |
| S2ST | 1164.3 ms | 927.8 ms | 70.8 ms  | 83.0         | 12.1            | 4667.1 ms | 156160 smp (RTF **0.48×**) |
| ASR  | 1288.5 ms | 933.9 ms | 77.0 ms  | 65.5         | 15.3            | 1737.4 ms | 30 tok (512 mel)           |



| Task | T2U     | Vocoder | RTF   |
| ---- | ------- | ------- | ----- |
| T2ST | 1759 ms | 2537 ms | 0.21× |
| S2ST | 1131 ms | 1819 ms | 0.48× |


### Sequence length: 1024


| Task | TTFT      | Encoder   | Prefill  | decode t/s/u | ms/tok (steady) | E2E       | Output                     |
| ---- | --------- | --------- | -------- | ------------ | --------------- | --------- | -------------------------- |
| T2TT | 153.7 ms  | 27.6 ms   | 50.1 ms  | 123.6        | 8.1             | 2265.3 ms | 256 tok                    |
| T2ST | 342.1 ms  | 96.3 ms   | 81.2 ms  | 121.2        | 8.2             | 9035.7 ms | 799360 smp (RTF **0.18×**) |
| S2TT | 1165.1 ms | 917.1 ms  | 81.5 ms  | 95.6         | 10.5            | 1719.8 ms | 53 tok (1024 mel)          |
| S2ST | 1519.5 ms | 1257.7 ms | 78.0 ms  | 102.2        | 9.8             | 8403.0 ms | 286400 smp (RTF **0.47×**) |
| ASR  | 1868.7 ms | 1359.5 ms | 171.8 ms | 79.1         | 12.6            | 2626.4 ms | 60 tok (1024 mel)          |



| Task | T2U     | Vocoder | RTF   |
| ---- | ------- | ------- | ----- |
| T2ST | 1811 ms | 3860 ms | 0.18× |
| S2ST | 2235 ms | 3674 ms | 0.47× |


### Sequence length: 2048


| Task | TTFT      | Encoder   | Prefill  | decode t/s/u | ms/tok (steady) | E2E        | Output                     |
| ---- | --------- | --------- | -------- | ------------ | --------------- | ---------- | ------------------------- |
| T2TT | 179.4 ms  | 38.3 ms   | 50.6 ms  | 111.1        | 9.0             | 2521.8 ms  | 256 tok                    |
| T2ST | 318.4 ms  | 90.8 ms   | 71.8 ms  | 108.8        | 9.2             | 8106.2 ms  | 644800 smp (RTF **0.20×**) |
| S2TT | 8244.7 ms | 8022.0 ms | 74.4 ms  | 106.9        | 9.4             | 8975.2 ms  | 77 tok (2048 mel)          |
| S2ST | 8649.5 ms | 8394.1 ms | 73.1 ms  | 106.9        | 9.4             | 14272.6 ms | 300480 smp (RTF **0.76×**) |
| ASR  | 8908.3 ms | 8540.1 ms | 151.8 ms | 95.5         | 10.5            | 9773.6 ms  | 82 tok (2048 mel)          |



| Task | T2U     | Vocoder | RTF   |
| ---- | ------- | ------- | ----- |
| T2ST | 1781 ms | 2941 ms | 0.20× |
| S2ST | 1992 ms | 2695 ms | 0.76× |


### Sequence length: 4096


| Task | TTFT      | Encoder   | Prefill  | decode t/s/u | ms/tok (steady) | E2E       | Output                     |
| ---- | --------- | --------- | -------- | ------------ | --------------- | --------- | -------------------------- |
| T2TT | 315.9 ms  | 149.4 ms  | 51.0 ms  | 91.1         | 11.0            | 3164.1 ms | 256 tok                    |
| T2ST | 463.4 ms  | 160.9 ms  | 111.7 ms | 90.4         | 11.1            | 8761.7 ms | 647040 smp (RTF **0.22×**) |
| S2TT | 2223.4 ms | 1995.0 ms | 73.7 ms  | 115.7        | 8.6             | 3271.3 ms | 119 tok (4096 mel)         |
| S2ST | 2199.5 ms | 1966.2 ms | 71.4 ms  | 95.1         | 10.5            | 7158.0 ms | 226240 smp (RTF **0.51×**) |
| ASR  | 2523.2 ms | 2241.1 ms | 75.9 ms  | 104.0        | 9.6             | 3514.6 ms | 102 tok (4096 mel)         |



| Task | T2U     | Vocoder | RTF   |
| ---- | ------- | ------- | ----- |
| T2ST | 1796 ms | 2959 ms | 0.22× |
| S2ST | 2025 ms | 2183 ms | 0.51× |


Task notes:

- **T2TT** — text encoder + traced text-decoder loop.
- **T2ST** — text path + T2U + vocoder; vocoder dominates E2E; use **RTF** for speech QoS.
- **S2TT / ASR** — speech encoder dominates TTFT; **decode t/s/u** isolates the text-decoder steady rate.
- **S2ST** — speech encoder + decoder + T2U + vocoder.

**Cold start:** the first timed speech-synthesis call in a **brand-new process** with a cold on-disk JIT cache can still pay a one-time vocoder compile outlier (~15–20 s). Subsequent timed iters in the same session and later demo invocations (warm disk cache) report steady vocoder times (~1.1–1.6 s on BH QB). Speech warmups + vocoder prewarm (in `demo.py`) keep the **reported** timed iter near steady state.
