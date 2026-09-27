# Common sampler selection

Selected `models/common/sampling/generator.py` (`SamplingGenerator` with
`TTSampling`). This owns seed/penalty state, explicit sampling parameter reset,
mode-keyed internal sampling traces, and persistent `tt_out_tok` feedback. It
supports TP4 vocabulary shards and top-k/top-p. The wrapper owns device seed
state and leaves the common host seed manager inactive, allowing seeded
sampling to keep its internal trace. Explicit seeds are reproducible; omitted
seeds receive per-request/per-slot entropy at setup. Seed advancement is
captured in the model trace; seeded repeatability, seed advancement and sampled feedback pass the reduced
real-terminal checks. Default greedy is k=1, p=0, temperature=1 after
parameter normalization, with 32 physical local candidates per shard.

Rejected `models/common/modules/sampling/sampling_1d.py`: declarative 1D topology
fits TP4 and accepts `tt_out_tok`, but caller must own trace wrapping and penalties.
Current `_sample_topk` calls `_topk(x_bf16, active_batch)` while both bound strategy
methods accept only `x_bf16`; they also reference missing `active_batch` and
`_local_indices`. Selection is based on the stronger existing state/trace
contract, not a measured performance claim. No custom sampling implementation.

The sampler consumes 32 logical rows, not merely tile-padded B rows. Normal
split greedy and sampled known-winner tests pass B1/B3/B32 after terminal
padding (`sampler_padded.json`, `sampler_selected_batches.json`). The unpadded
B3 minimal repro fails at binary addition of3-row indices and32-row offsets;
`sampler_raw_batch3.json`, `AUTODEBUG_sampler_batch.md`.

Force-argmax is tested only in the standalone comparison probe; the delivered
generator disables it. Semantically greedy
B1 normal split sampling measures508.87us versus3302.49us force-argmax (100
warmed trace replays, host-wall duration, tracker off). Normal split greedy is
selected. These are isolated sampler timings, not full-model device times.
The alternative also returned a wrong global index on one changed-input B3
row; source investigation is recorded separately. B32 normal greedy and sampled
paths pass every row on all four ranks. The reduced terminal profile confirms sampling does not dominate:
512.37us sampler trace within a3078.76us complete two-layer window
(`profile_terminal/summary.json`). No sampled-mode timing substitutes for greedy.

Both common implementations use `LogProbsCalculator`, which returns no result
on TP4. The generator explicitly rejects optional logprob requests before any
request-state mutation. Supporting this optional API would require a separate
numerical/trace qualification of the common calculator on four devices.
Greedy, top-k/top-p, request seeds and penalties remain device paths.
