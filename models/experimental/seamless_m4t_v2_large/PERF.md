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

Tables below are from the BH QB run logged in `scripts/outputs/perf_sweep.txt`.

Reproduce:

```bash
python models/experimental/seamless_m4t_v2_large/scripts/demo_perf_sweep.py
```

### Sequence length: 32


| Task | TTFT      | Encoder  | Prefill | decode t/s/u | ms/tok (steady) | E2E       | Output                     |
| ---- | --------- | -------- | ------- | ------------ | --------------- | --------- | -------------------------- |
| T2TT | 134.8 ms  | 22.7 ms  | 40.7 ms | 100.7        | 9.9             | 399.8 ms  | 27 tok                     |
| T2ST | 166.6 ms  | 36.7 ms  | 49.9 ms | 88.4         | 11.3            | 2494.9 ms | 103040 smp (RTF **0.39×**) |
| S2TT | 847.7 ms  | 691.4 ms | 84.8 ms | 20.8         | 48.1            | 993.6 ms  | 4 tok (32 mel)             |
| S2ST | 918.0 ms  | 758.4 ms | 86.8 ms | 29.3         | 34.1            | 3033.3 ms | 12480 smp (RTF **3.89×**)  |
| ASR  | 1011.4 ms | 802.2 ms | 60.2 ms | 46.0         | 21.8            | 1210.1 ms | 10 tok (32 mel)            |



| Task | T2U    | Vocoder | RTF   |
| ---- | ------ | ------- | ----- |
| T2ST | 852 ms | 1042 ms | 0.39× |
| S2ST | 890 ms | 997 ms  | 3.89× |


### Sequence length: 64


| Task | TTFT      | Encoder  | Prefill | decode t/s/u | ms/tok (steady) | E2E       | Output                     |
| ---- | --------- | -------- | ------- | ------------ | --------------- | --------- | -------------------------- |
| T2TT | 136.3 ms  | 22.6 ms  | 45.7 ms | 115.1        | 8.7             | 607.2 ms  | 54 tok                     |
| T2ST | 221.3 ms  | 59.7 ms  | 64.0 ms | 102.8        | 9.7             | 3734.1 ms | 196800 smp (RTF **0.30×**) |
| S2TT | 868.2 ms  | 714.0 ms | 84.3 ms | 31.9         | 31.4            | 1027.0 ms | 6 tok (64 mel)             |
| S2ST | 963.7 ms  | 773.1 ms | 90.5 ms | 31.3         | 31.9            | 3135.8 ms | 26240 smp (RTF **1.91×**)  |
| ASR  | 1071.2 ms | 851.9 ms | 60.8 ms | 24.1         | 41.5            | 1238.8 ms | 5 tok (64 mel)             |



| Task | T2U     | Vocoder | RTF   |
| ---- | ------- | ------- | ----- |
| T2ST | 1648 ms | 1231 ms | 0.30× |
| S2ST | 914 ms  | 1069 ms | 1.91× |


### Sequence length: 128


| Task | TTFT      | Encoder  | Prefill | decode t/s/u | ms/tok (steady) | E2E       | Output                     |
| ---- | --------- | -------- | ------- | ------------ | --------------- | --------- | -------------------------- |
| T2TT | 140.0 ms  | 25.3 ms  | 45.4 ms | 128.2        | 7.8             | 1083.1 ms | 119 tok                    |
| T2ST | 224.0 ms  | 37.7 ms  | 58.7 ms | 124.0        | 8.1             | 4728.4 ms | 406080 smp (RTF **0.19×**) |
| S2TT | 939.6 ms  | 742.5 ms | 91.4 ms | 64.7         | 15.4            | 1112.5 ms | 12 tok (128 mel)           |
| S2ST | 1024.6 ms | 825.3 ms | 59.0 ms | 66.6         | 15.0            | 3418.8 ms | 47360 smp (RTF **1.16×**)  |
| ASR  | 1107.7 ms | 890.4 ms | 60.2 ms | 47.1         | 21.2            | 1323.0 ms | 11 tok (128 mel)           |



| Task | T2U     | Vocoder | RTF   |
| ---- | ------- | ------- | ----- |
| T2ST | 1612 ms | 1748 ms | 0.19× |
| S2ST | 959 ms  | 1181 ms | 1.16× |


### Sequence length: 256


| Task | TTFT      | Encoder   | Prefill  | decode t/s/u | ms/tok (steady) | E2E       | Output                     |
| ---- | --------- | --------- | -------- | ------------ | --------------- | --------- | -------------------------- |
| T2TT | 139.5 ms  | 25.4 ms   | 43.9 ms  | 133.6        | 7.5             | 1926.7 ms | 234 tok                    |
| T2ST | 296.3 ms  | 78.6 ms   | 104.3 ms | 125.8        | 8.0             | 6840.3 ms | 775680 smp (RTF **0.14×**) |
| S2TT | 1040.9 ms | 833.0 ms  | 62.1 ms  | 82.2         | 12.2            | 1375.8 ms | 28 tok (256 mel)           |
| S2ST | 1147.6 ms | 923.1 ms  | 63.7 ms  | 79.6         | 12.6            | 4456.4 ms | 91520 smp (RTF **0.78×**)  |
| ASR  | 1313.7 ms | 1018.4 ms | 71.8 ms  | 65.6         | 15.3            | 1623.7 ms | 21 tok (256 mel)           |



| Task | T2U     | Vocoder | RTF   |
| ---- | ------- | ------- | ----- |
| T2ST | 1686 ms | 2703 ms | 0.14× |
| S2ST | 1072 ms | 1813 ms | 0.78× |


### Sequence length: 512


| Task | TTFT      | Encoder  | Prefill | decode t/s/u | ms/tok (steady) | E2E       | Output                     |
| ---- | --------- | -------- | ------- | ------------ | --------------- | --------- | -------------------------- |
| T2TT | 142.5 ms  | 25.4 ms  | 44.7 ms | 128.2        | 7.8             | 2178.9 ms | 256 tok                    |
| T2ST | 288.6 ms  | 80.7 ms  | 63.0 ms | 124.9        | 8.0             | 7089.6 ms | 741120 smp (RTF **0.15×**) |
| S2TT | 996.0 ms  | 785.6 ms | 62.6 ms | 95.2         | 10.5            | 1446.2 ms | 43 tok (512 mel)           |
| S2ST | 1074.2 ms | 853.4 ms | 62.7 ms | 89.8         | 11.1            | 4580.2 ms | 145920 smp (RTF **0.50×**) |
| ASR  | 1183.9 ms | 918.4 ms | 66.8 ms | 79.9         | 12.5            | 1604.7 ms | 34 tok (512 mel)           |



| Task | T2U     | Vocoder | RTF   |
| ---- | ------- | ------- | ----- |
| T2ST | 1698 ms | 2787 ms | 0.15× |
| S2ST | 1093 ms | 1892 ms | 0.50× |


### Sequence length: 1024


| Task | TTFT      | Encoder   | Prefill  | decode t/s/u | ms/tok (steady) | E2E       | Output                     |
| ---- | --------- | --------- | -------- | ------------ | --------------- | --------- | -------------------------- |
| T2TT | 154.8 ms  | 28.1 ms   | 48.2 ms  | 121.5        | 8.2             | 2301.0 ms | 256 tok                    |
| T2ST | 302.8 ms  | 81.6 ms   | 101.2 ms | 115.9        | 8.6             | 7319.5 ms | 773120 smp (RTF **0.15×**) |
| S2TT | 1080.9 ms | 840.6 ms  | 62.1 ms  | 106.0        | 9.4             | 1726.2 ms | 68 tok (1024 mel)          |
| S2ST | 1243.1 ms | 1011.8 ms | 67.9 ms  | 106.1        | 9.4             | 7019.7 ms | 263680 smp (RTF **0.43×**) |
| ASR  | 1569.7 ms | 1261.5 ms | 91.3 ms  | 93.9         | 10.7            | 2307.8 ms | 69 tok (1024 mel)          |



| Task | T2U     | Vocoder | RTF   |
| ---- | ------- | ------- | ----- |
| T2ST | 1729 ms | 2785 ms | 0.15× |
| S2ST | 1951 ms | 3034 ms | 0.43× |


### Sequence length: 2048


| Task | TTFT      | Encoder   | Prefill | decode t/s/u | ms/tok (steady) | E2E        | Output                     |
| ---- | --------- | --------- | ------- | ------------ | --------------- | ---------- | -------------------------- |
| T2TT | 173.4 ms  | 41.4 ms   | 45.4 ms | 108.1        | 9.3             | 2580.3 ms  | 256 tok                    |
| T2ST | 316.7 ms  | 95.1 ms   | 61.8 ms | 105.5        | 9.5             | 7474.5 ms  | 766080 smp (RTF **0.16×**) |
| S2TT | 8055.3 ms | 7835.9 ms | 63.9 ms | 116.1        | 8.6             | 9094.8 ms  | 119 tok (2048 mel)         |
| S2ST | 8731.9 ms | 8500.5 ms | 63.1 ms | 117.5        | 8.5             | 15110.0 ms | 423040 smp (RTF **0.57×**) |
| ASR  | 8880.3 ms | 8601.6 ms | 68.5 ms | 107.3        | 9.3             | 10184.0 ms | 138 tok (2048 mel)         |



| Task | T2U     | Vocoder | RTF   |
| ---- | ------- | ------- | ----- |
| T2ST | 1703 ms | 2749 ms | 0.16× |
| S2ST | 1964 ms | 3146 ms | 0.57× |


### Sequence length: 4096


| Task | TTFT      | Encoder   | Prefill | decode t/s/u | ms/tok (steady) | E2E        | Output                     |
| ---- | --------- | --------- | ------- | ------------ | --------------- | ---------- | -------------------------- |
| T2TT | 253.5 ms  | 89.4 ms   | 54.8 ms | 89.3         | 11.2            | 3156.8 ms  | 256 tok                    |
| T2ST | 361.9 ms  | 108.1 ms  | 72.0 ms | 88.2         | 11.3            | 7731.0 ms  | 663040 smp (RTF **0.19×**) |
| S2TT | 2216.6 ms | 2009.7 ms | 63.5 ms | 106.1        | 9.4             | 3016.0 ms  | 84 tok (4096 mel)          |
| S2ST | 2248.9 ms | 2027.5 ms | 64.1 ms | 103.9        | 9.6             | 11533.9 ms | 250560 smp (RTF **0.74×**) |
| ASR  | 2604.6 ms | 2331.5 ms | 66.7 ms | 90.8         | 11.0            | 3400.7 ms  | 72 tok (4096 mel)          |



| Task | T2U     | Vocoder | RTF   |
| ---- | ------- | ------- | ----- |
| T2ST | 1716 ms | 2496 ms | 0.19× |
| S2ST | 2442 ms | 4497 ms | 0.74× |


Task notes:

- **T2TT** — text encoder + traced text-decoder loop.
- **T2ST** — text path + T2U + vocoder; vocoder dominates E2E; use **RTF** for speech QoS.
- **S2TT / ASR** — speech encoder dominates TTFT; **decode t/s/u** isolates the text-decoder steady rate.
- **S2ST** — speech encoder + decoder + T2U + vocoder.

**Cold start:** the first timed speech-synthesis call in a **brand-new process** with a cold on-disk JIT cache can still pay a one-time vocoder compile outlier (~15–20 s). Subsequent timed iters in the same session and later demo invocations (warm disk cache) report steady vocoder times (~1.1–1.6 s on BH QB). Speech warmups + vocoder prewarm (in `demo.py`) keep the **reported** timed iter near steady state.
