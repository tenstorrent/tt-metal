# CosyVoice2 performance (Wormhole N150)

Non-streaming (Stage 1) and streaming (Stage 3's targets). Every figure here comes from a run recorded in
[`docs/VALIDATION.md`](docs/VALIDATION.md), which has the per-utterance tables.

## Environment

- Device: Wormhole N150 (one chip). Host: AMD EPYC 7352 (24 cores, 96 threads).
- Start-up: KMD 2.3.0 and firmware 19.11.0, 2026-09-28. Stage 1 requests and streaming: a second N150 with KMD
  2.9.0 and firmware 19.11.0.0, re-run on 2026-09-30 on the masked HiFT (docs/VALIDATION.md, "Stage 1 and streaming
  re-run on the masked HiFT"). That board reproduced 09-28's Stage 1 figures on 09-29.
- tt-metal merge-base `1f29f312fa`.
- Configuration: `CosyVoice2Config.reported()`. It is bucketed, has an fp32-logit LLM head and the LLM decode trace,
  and uses HiFT fp32, in 512-frame chunks for a mel of 512 frames or more.

## Commands

```bash
# the demo: warms every bucket, then synthesizes the corpus's six LibriSpeech targets
python models/experimental/cosyvoice2/demo/demo.py --inputs <scripts/prepare_inputs.py dir> --out <dir>

# streaming: both warm-ups, then the same six targets
python models/experimental/cosyvoice2/demo/demo.py --inputs <scripts/prepare_inputs.py dir> --out <dir> --stream

# the Stage 1 RTF gate, then Stage 3's two figures, each in its own process
COSYVOICE2_INPUTS=<dir> pytest "models/experimental/cosyvoice2/tests/perf/test_pipeline_perf.py::test_device_nonstreaming_rtf_distinct_utterances"
COSYVOICE2_INPUTS=<dir> pytest "models/experimental/cosyvoice2/tests/perf/test_pipeline_perf.py::test_device_streaming_first_audio_and_rtf_distinct_utterances"
```

## Start-up

At start-up the pipeline runs every flow and HiFT bucket once (`warmup_buckets()`), so no request compiles:
8 LLM prefill lengths and a decode, 17 flow buckets, and 2 HiFT buckets, which chunked HiFT reuses. Streaming adds
its own set (`warmup_streaming()`): the streaming flow at the 17 buckets, and HiFT's streaming calls.

Measured on 2026-09-30, on the masked HiFT, in two fresh processes against one new kernel-cache directory
(`TT_METAL_CACHE`). The first process started empty; the second was identical.

| start | `warmup_buckets()` | then `warmup_streaming()` | kernels compiled |
|---|---|---|---|
| cold: empty kernel cache | **31.4 min** (1,885 s), after a 15 s build | 13.0 min (779 s) | 9,910 + 2,766 |
| warm: kernels on disk | **3.1 min** (188 s), after a 12 s build | 2.5 min (150 s) | 0 |

09-28's measurement of the buckets, on chunked HiFT before the masking, was 30.5 and 3.2 min. Before chunked HiFT
it was 76 and 9.6 min.

- **The warm start, split:**
  - conv safety checks 23.5 s, plus 11.5 s for the streaming set;
  - conv weight preparation 9.2 s, plus 3.9 s;
  - by stage: LLM 2.4 s, flow 165 s, HiFT 21 s; then the streaming flow 124 s and streaming HiFT 26 s.
- **The cold start is kernel compilation and first-sight checks.** The checks compile the reference convs' kernels
  too (452 s and 281 s cold). Over the warm start, cold adds 1,697 s for the buckets and 629 s for the streaming
  set.
- **Any change** to the code, the configuration, the checkpoint or the warm-up sequence costs one cold start. Some
  conv kernels carry DRAM addresses in their compile-time arguments, so a cached binary is reused only when a
  process allocates exactly as the one that compiled it.
- **Cold first request, no warm-up** (`demo.py --warmup none`, one utterance, 121-127105-0003 at 8.52 s, fresh
  process, 2026-09-30):

  | kernel cache | wall | RTF | kernels compiled |
  |---|---|---|---|
  | empty | 563.4 s | **66.1** | 2,929 |
  | holding every binary it needs (the day's earlier runs) | 17.4 s | **2.04** | 0 |

  - With the kernels on disk, what remains is each geometry's first run. HiFT's first-sight conv checks alone take
    10.7 s.
  - A filled cache is not one figure: it holds whatever earlier processes compiled. 09-28's RTF 32.5 came from a
    cache that still lacked 706 of this request's binaries.
  - This is the cost the warm-up moves to start-up. Warmed, the same utterance takes 3.76 s (RTF 0.441).

## Requests after start-up (Stage 1)

Six distinct LibriSpeech test-clean utterances of 3.0–13.9 s, each synthesized once:

| | RTF |
|---|---|
| per utterance | 0.441–0.654 |
| aggregate | **0.483** |

- The perf test, in its own process: worst 0.675, aggregate 0.481.
- LLM decode is 43–58 % of each request, at 89–98 tokens/s.
- The CFM (10 Euler steps) takes 0.69–1.36 s.
- HiFT takes 0.21–0.38 s in one pass, a padded call masked to compute upstream's call at the real length. That is
  0.07–0.10 s more than silence padding took (0.14–0.28 s) before 2026-09-30.
- The two long utterances run as two 512-frame chunks: 0.71–0.73 s.

## Streaming (Stage 3)

`demo.py --stream`, twice in fresh processes, on the same six utterances, each run after both warm-ups (the masked
HiFT, 2026-09-30):

| | run 1 | run 2 | target |
|---|---|---|---|
| time to first audio | 1.353–1.502 s | 1.313–1.432 s | < 0.5 s |
| streaming RTF, per utterance | 0.813–1.121 | 0.790–1.103 | worst < 0.4 |
| streaming RTF, aggregate | 0.851 | 0.836 | |

09-29's two runs, before the final call was masked, fall in the same spread: 1.336–1.479 s; aggregate RTF
0.843–0.853. The streaming perf test enforces both figures, each inside its recorded band (its first run: worst
first audio 1,469 ms, worst RTF 1.110).

- **Start-up** adds `warmup_streaming()` after `warmup_buckets()`: 149 s with the kernels on disk (the bucket
  warm-up took 184 s in the same runs). A streaming request is refused without it.
- **The first chunk:**
  - 0.37–0.47 s of text and LLM until its tokens are in;
  - then its flow, 0.82–0.91 s, of which the CFM's 10 Euler steps take 0.68–0.74 s;
  - then HiFT, 0.12–0.13 s.

  Even a free flow would leave first audio at 0.49–0.60 s.
- **RTF:** every chunk reruns the flow over the prompt and the whole prefix, as upstream does, and the final chunk runs
  it non-streaming. So a chunk's flow costs at least 0.82 s (bucket 256, 512 mel frames), and 1.01–2.71 s at the
  larger buckets.
  - The one utterance above 1.0 is the 3.8 s 121-127105-0015. Its tokens make three chunks (32 + 50 + 13), and the
    13-token final chunk pays a full non-streaming flow, 1.01 s, for 0.52 s of audio.
  - The 3.0 s utterance fits in two chunks (25 + 50): 0.905–0.957.
  - The flows alone take 0.47–0.74 of every utterance's duration (`docs/VALIDATION.md`, "Streaming RTF above 1.0").
- **One CFM Euler step** at the first chunk's size (512 mel frames, batch 2; `docs/VALIDATION.md`, "One CFM Euler
  step, profiled"):
  - 64.7 ms eager, of which the host spends 62.6 ms enqueueing the estimator's 1,158 ops; 49.2 ms traced;
  - on the device, 47.8 ms of kernel time, 18.5 ms of it merging attention heads (a transpose and a reshape in each
    of the 56 transformer blocks).
- **Fewer Euler steps** (`docs/VALIDATION.md`, "The Euler step sweep"): each step costs the first chunk about 70 ms.
  - At 8 steps: first audio 1.21–1.40 s, worst streaming RTF 0.98–1.03.
  - At 5 steps: first audio 0.98–1.14 s, worst streaming RTF 0.82–0.84, Stage 1 worst 0.51–0.54.
  - WER and SIM do not move, but the audio does.
  - The reported configuration keeps upstream's 10 steps.
