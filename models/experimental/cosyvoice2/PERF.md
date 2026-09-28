# CosyVoice2 performance (Wormhole N150)

Non-streaming only; streaming (Stages 2 and 3) is not built. Every figure here comes from a run recorded in
[`docs/VALIDATION.md`](docs/VALIDATION.md), which has the per-utterance tables.

## Environment

- Device: Wormhole N150 (one chip). Host: AMD EPYC 7352 (24 cores, 96 threads).
- KMD 2.3.0, firmware 19.11.0, tt-metal merge-base `1f29f312fa`.
- Configuration: `CosyVoice2Config.reported()`. It is bucketed, has an fp32-logit LLM head and the LLM decode trace,
  and uses HiFT fp32.
- Date: 2026-09-28.

## Commands

```bash
# the demo: warms every bucket, then synthesizes the corpus's six LibriSpeech targets
python models/experimental/cosyvoice2/demo/demo.py --inputs <scripts/prepare_inputs.py dir> --out <dir>

# the Stage 1 RTF gate
COSYVOICE2_INPUTS=<dir> pytest models/experimental/cosyvoice2/tests/perf/test_pipeline_perf.py
```

## Start-up

At start-up the pipeline runs every flow and HiFT bucket once (`warmup_buckets()`), so no request compiles.

| start | warm-up | kernels compiled |
|---|---|---|
| cold: empty kernel cache | **76 min** (4,561 s), after a 17 s build | 19,068 |
| warm: kernels on disk | **9.6 min** (577 s), after a 13 s build | 0 |

- **The warm start, split:**
  - conv safety checks: 182 s (32 %);
  - conv weight preparation: 23 s (4 %);
  - each geometry's first run: 372 s (64 %).

  By stage: LLM 2.5 s, flow 96 s, HiFT 443 s (77 %). The Stage 1 demo and perf test, later warm starts on the same
  cache, warmed in 542 s and 534 s.
- **The cold start is mostly kernel compilation:** the 3,984 s it adds is 87 % of it.
- **Any change** to the code, the configuration, the checkpoint or the warm-up sequence costs one cold start. Some
  conv kernels carry DRAM addresses in their compile-time arguments, so a cached binary is reused only when a
  process allocates exactly as the one that compiled it.
- **Cold first request, no warm-up** (`demo.py --warmup none`, one utterance, fresh process): 277.2 s for 8.52 s of
  audio, **RTF 32.5**, with 706 kernels compiled. That is the cost the warm-up moves to start-up.

## Requests after start-up (Stage 1)

Six distinct LibriSpeech test-clean utterances of 3.0–13.9 s, each synthesized once:

| | RTF |
|---|---|
| per utterance | 0.428–0.633 |
| aggregate | **0.481** |

- LLM decode is 44–60 % of each request, at 90–98 tokens/s.
- The CFM (10 Euler steps) takes 0.70–1.38 s, and HiFT 0.15–0.47 s.
