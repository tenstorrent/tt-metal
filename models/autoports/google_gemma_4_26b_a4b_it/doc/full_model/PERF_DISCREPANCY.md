# Two full-model decode rates differ by 10x and the difference is not explained

Operator note added after stage 06, before stage 07.

`doc/full_model/README.md` reports two token-out decode rates side by side with
no account of why they differ:

| workload | TTFT | decode |
|---|---:|---:|
| 128 prompt / 128 generated (headline) | 320.84 ms | **23.76 t/s/u** |
| 64-token short-prompt diagnostic | 215.45 ms | **2.50 t/s/u** |

`work_log.md` gives the raw figures for the second: "Optimized 64-token
token-out measurement: TTFT 215.445 ms, 63-token traced decode 25.1519 s,
2.5048 t/s/u". Both land in `performance.json`
(`decode_t_per_s_per_user: 2.504780189404154`).

So 63 tokens took **25.15 s**, about **399 ms/token**, against roughly
**42 ms/token** for the headline run. A *shorter* prompt being ~10x slower per
generated token is not explicable by prompt length, and nothing in the stage
documents a cause.

Candidate explanations, none confirmed here:

- the 64-token run used a cold or untraced path while the headline run is
  trace-verified over 128 replays;
- it included per-token host readback or a sync the headline path avoids;
- it captured trace-capture or warm-up cost inside the measured window.

Why it matters: `performance.json` carries both numbers, so any consumer
selecting the wrong field understates this model by 10x. Stage 07 owns
before/after perf accounting for the full-model path and is the right place to
either reproduce and explain the 2.50 figure or withdraw it. Until then treat
23.76 t/s/u as the model's decode rate and the 64-token row as unexplained.

## Related, for stage 07

The selected sampler is `Sampling1D` native **force-argmax** (README "Selected:
`Sampling1D` native force-argmax"), chosen on measured latency against
alternatives. The stage-07 goal requires that if force-argmax, a full-vocab
all-gather, a generic `TopKDeviceOperation`, or another sampler op *dominates*
token-out decode, the LM-head/sampling contract is fixed before the stage
completes. Falcon and Qwen both ended on split sampling rather than
force-argmax. Stage 07 should show the profile share rather than inherit the
choice unexamined.
