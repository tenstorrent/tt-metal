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

## pipeline_4rank_93L_rebased

All 93 layers as 24/24/24/21 across four Blackhole Galaxies, on the branch rebased onto main. One
55k-token pass, 11 chunks of 5120, untraced, `PREFILL_SYNC_PER_CHUNK=1` so each bar is measured
compute rather than push rate.

Per-rank mean, against the same run before the rebase:

| rank | layers | before | after |
|---|---|---|---|
| 0 | 0-23 | 955 ms | 947 ms |
| 1 | 24-47 | 921 ms | 956 ms |
| 2 | 48-71 | 892 ms | 962 ms |
| 3 | 72-92 | 827 ms | 822 ms |

Rank 3 is fastest because it holds 21 layers, not 24. The shape is the expected one: a four-deep
fill bubble, every rank busy through the middle, and a staggered drain.

## The KDA inverse, after #55626

Kimi-K3 carried a workaround commit that pinned the pre-#54937 KDA inverse, because
`invert_block_ps4` was uncorrelated on K3's real gate magnitudes (#55420). `#55626` fixed that in
main with `invert_block_nested`, so the workaround is dropped rather than carried, and what follows
is the evidence that main's inverse is at least as good on K3 as the workaround was.

`N` is the negated strictly-lower `Akk`, so `T_inv = (I-N)^-1`, and `N` carries `exp(G_i - G_j)`
where `G` is the per-chunk cumsum of a gate saturated at its -5.0 lower bound: about -150 over 32
rows. Score `T_inv` on its strictly-lower part, not the whole tile -- whole-tensor PCC is dominated
by the identity diagonal and read 0.99508 while the off-diagonals were wrong enough to take the
recurrence from 1.0 to 0.0014.

Measured on one Blackhole Galaxy, real Kimi-K3 weights:

| inverse | strictly-lower PCC, layers 0 and 1 |
|---|---|
| `invert_doubling` (retired in #54937) | 0.01186, 0.02302 |
| `invert_block_ps4` (the #55420 bug) | 8/18 layers above 0.999, worst 0.7508 |
| `invert_horner` (the dropped workaround) | 18/18 above 0.999, worst 0.99982 |
| `invert_block_nested` (main, #55626) | **0.99999, 1.00000** |

`test_kda_single_device_matches_reference` passes on both layers; layer 1 read 0.69629 under
`invert_doubling`. The bisection instruments that found #55420 -- `test_kda_tinv_precision.py`,
`test_kda_prepare_vs_scan.py`, `test_kda_decay_magnitude.py`, `test_kda_stage_bisect.py` and
`test_layer0_stages.py` -- were deleted once these numbers were taken; their durable content is the
op-level coverage under `tests/ttnn/nightly/unit_tests/operations/experimental/kda/`.

## How many concurrent users fit

Two allocations scale with `num_users`, and only one of them scales with context:

| | what | per slot per chip, one rank | scales with context |
|---|---|---|---|
| KV | `num_users * mla_layers` user-major slots of `max_seq_len` rows, 576 wide, bfloat8_b, SP-sharded | 24.7 MiB at 56320 | yes |
| KDA | one carry per (slot, KDA layer): recurrent `[1, heads/TP, 128, 128]` FLOAT32 + a bf16 convolution history | 27.9 MiB | **no** |

A rank of the 93-layer split holds 6 MLA slabs and 18 KDA carries, so at 56320 a slot costs
52.6 MiB/chip and the KDA half of that does not shrink when the request does.

Measured by allocating against a 16.8 GiB/chip ballast standing in for one rank's weights, bisected
to the first refusal from `bank_manager.cpp`:

| context | total/slot | predicted | measured | users x context |
|---|---|---|---|---|
| 5120 | 30.1 MiB | 443 | 448 | 2.3M |
| 56320 | 52.6 MiB | 254 | 272 | 15.3M |
| 262144 | 142.7 MiB | 93 | 96 | 25.2M |

Within 1 to 7% across a 51x context range; the gap is allocator reserve, 1.1 to 1.9 GiB/chip.

`users x context` is therefore NOT the invariant it would be for a pure-KV model -- it grows 11x over
that range, because the fixed KDA carry dominates at short context. And there is a ceiling around 480
slots per rank that no amount of context shortening lifts, all of it the FLOAT32 recurrent state.
Halving it to bf16 is the obvious lever if concurrency ever becomes the binding constraint.

`scripts/`-adjacent reproduction lives in the run logs rather than the tree: the sweep allocates the
same two structures the adapter does and needs no weights, so it answers in minutes rather than the
hour a full bring-up costs.
