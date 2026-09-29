# CosyVoice2 performance (Wormhole N150)

Non-streaming (Stage 1) and streaming (Stage 3's targets). Every figure here comes from a run recorded in
[`docs/VALIDATION.md`](docs/VALIDATION.md), which has the per-utterance tables.

## Environment

- Device: Wormhole N150 (one chip). Host: AMD EPYC 7352 (24 cores, 96 threads).
- Stage 1: KMD 2.3.0 and firmware 19.11.0, 2026-09-28. Streaming: a second N150 with KMD 2.9.0 and firmware
  19.11.0.0, 2026-09-29; that board reproduced the Stage 1 figures (docs/VALIDATION.md, "Re-verified on a second
  N150").
- tt-metal merge-base `1f29f312fa`.
- Configuration: `CosyVoice2Config.reported()`. It is bucketed, has an fp32-logit LLM head and the LLM decode trace,
  and uses HiFT fp32, in 512-frame chunks for a mel of 512 frames or more.

## Commands

```bash
# the demo: warms every bucket, then synthesizes the corpus's six LibriSpeech targets
python models/experimental/cosyvoice2/demo/demo.py --inputs <scripts/prepare_inputs.py dir> --out <dir>

# streaming: both warm-ups, then the same six targets
python models/experimental/cosyvoice2/demo/demo.py --inputs <scripts/prepare_inputs.py dir> --out <dir> --stream

# the Stage 1 RTF gate
COSYVOICE2_INPUTS=<dir> pytest models/experimental/cosyvoice2/tests/perf/test_pipeline_perf.py
```

## Start-up

At start-up the pipeline runs every flow and HiFT bucket once (`warmup_buckets()`), so no request compiles:
8 LLM prefill lengths and a decode, 17 flow buckets, and 2 HiFT buckets, which chunked HiFT reuses.

| start | warm-up | kernels compiled | before chunked HiFT (12 HiFT buckets) |
|---|---|---|---|
| cold: empty kernel cache | **30.5 min** (1,831 s), after a 16 s build | 9,959 | 76 min (4,561 s), 19,068 kernels |
| warm: kernels on disk | **3.2 min** (195 s), after a 13 s build | 0 | 9.6 min (577 s) |

- **The warm start, split:**
  - conv safety checks: 21.5 s (11 %);
  - conv weight preparation: 7.9 s (4 %);
  - each geometry's first run: 165 s (85 %).

  By stage: LLM 2.4 s, flow 157 s (81 %), HiFT 20 s. The Stage 1 demo and perf test, later warm starts on the same
  cache, warmed in 179 s and 176 s.
- **The cold start is mostly kernel compilation:** the 1,636 s it adds is 89 % of it.
- **Any change** to the code, the configuration, the checkpoint or the warm-up sequence costs one cold start. Some
  conv kernels carry DRAM addresses in their compile-time arguments, so a cached binary is reused only when a
  process allocates exactly as the one that compiled it.
- **Cold first request, no warm-up** (`demo.py --warmup none`, one utterance, fresh process; measured before chunked
  HiFT): 277.2 s for 8.52 s of audio, **RTF 32.5**, with 706 kernels compiled. That is the cost the warm-up moves to
  start-up.

## Requests after start-up (Stage 1)

Six distinct LibriSpeech test-clean utterances of 3.0–13.9 s, each synthesized once:

| | RTF |
|---|---|
| per utterance | 0.433–0.628 |
| aggregate | **0.479** |

- The perf test, in its own process: worst 0.621, aggregate 0.490.
- LLM decode is 44–60 % of each request, at 90–97 tokens/s.
- The CFM (10 Euler steps) takes 0.69–1.38 s.
- HiFT takes 0.15–0.28 s in one pass, and 0.70–0.71 s for the two long utterances, which run as two 512-frame
  chunks.

## Streaming (Stage 3)

`demo.py --stream`, twice in fresh processes, on the same six utterances, each run after both warm-ups:

| | run 1 | run 2 | target |
|---|---|---|---|
| time to first audio | 1.365–1.455 s | 1.336–1.479 s | < 0.5 s |
| streaming RTF, per utterance | 0.806–1.057 | 0.787–1.122 | worst < 0.4 |
| streaming RTF, aggregate | 0.853 | 0.843 | |

- **Start-up** adds `warmup_streaming()` after `warmup_buckets()`: 149–150 s with the kernels on disk (the bucket
  warm-up took 186 s in the same runs). A streaming request is refused without it.
- **The first chunk:**
  - 0.37–0.47 s of text and LLM until its tokens are in;
  - then its flow, 0.81–0.92 s, of which the CFM's 10 Euler steps take 0.67–0.73 s;
  - then HiFT, 0.12 s.

  Even a free flow would leave first audio at 0.51–0.59 s.
- **RTF:** every chunk reruns the flow over the whole prefix, as upstream does, and the final chunk runs it
  non-streaming. The two short utterances (3.0 and 3.8 s) are the worst: they carry the ~1.4 s first chunk over the
  least audio.
