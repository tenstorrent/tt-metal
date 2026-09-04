# Kimi-K3 pipeline-prefill traces

Gantt charts of one 55k-token producer pass (11 chunks of 5120) on a 2-rank, 72-layer
(36+36) pipeline across two Blackhole Galaxies. x is wall-clock seconds since the pass's
first chunk; each bar is one chunk on one rank.

Produced from a runner log with:

    python -m models.demos.deepseek_v3_d_p.scripts.slice_pipeline_run <run.log> --chunks 11 -o one.log
    python -m models.demos.deepseek_v3_d_p.scripts.plot_pipeline_trace one.log -o out.png

The slice step is needed because the runner is a persistent server: with
`PREFILL_SEND_SHUTDOWN=0` the chunk index never resets, so one log holds every pass ever
pushed at it and plotting the whole thing compresses the compute into slivers between
minutes of idle.

## before_handoff_fix

Rank 1 running with an EMPTY AttnRes sealed set — it believed it was the start of the
model, so its 36 layers read against nothing inherited. ~1.2 s/chunk, ~4280 tok/s. The
model is wrong here; the number is an upper bound, not a baseline.

## after_handoff_fix

Rank 1 inheriting the sealed set across the rank boundary and doing its real share of the
work: 36 read sites against a 4-to-6-deep sealed set. ~5 s/chunk, ~1050 tok/s. Untraced;
tracing measured 1.81x on the same model.

Both show the same pipeline shape — rank 1 one chunk behind rank 0 (the fill bubble),
both ranks busy through the middle, rank 1 draining one chunk after rank 0.
