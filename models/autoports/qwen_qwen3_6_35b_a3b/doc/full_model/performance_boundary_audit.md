# Performance boundary source audit

CPU/source audit only; no new hardware result. Audited
`tests/run_full_model_performance.py`, `tests/profile_full_model_components.py`,
`tests/collect_full_model_components.sh` and native stage5 processing patterns.

The queued `token_out` loop is valid device-resident autoregressive production:
both model and common sampler replay, the sampled token buffer is the next
model input, positions advance on device, and the loop forbids host tensor
computation/conversions. All refresh/readback/recapture counters must remain0.
Synchronizing at boundaries yields queued throughput, but discarding intermediate
outputs means it does not measure caller-visible token delivery.

The runner now adds `token_out_observed` in unprofiled runs. Every forward remains
under `device_only`; the caller reads exactly one token per step outside that
guard but inside the elapsed interval. No read token is supplied to the next
forward. The interval includes Python loop/guard overhead. The128input/128output
workload has one prefill prediction followed by127 readbacks and127 model/sampler
replays,0 input/page/position refreshes, unchanged program cache/traces, final
next position255 and final-token equality with the queued run
are asserted. First-token equality after reset is also checked. The exact token
read layout is warmed by `seed_request(captured=False)` before capture; position
readback is likewise warmed. This is the primary all40 decode headline metric.
The queued value remains separately reported. Both exclude first-token time;
TTFT includes prefill, device head/sampler handoff, position initialization,
traced sampling and first token delivery, after preparation/warmup/reset.

The instrumented runner retains three unchanged signpost pairs. Native timing
conversion preserves raw device CSV/trace data and operation metadata; only
optional host compile/write children are omitted by the inherited compaction.
The analyzer validates every component trace ID, four ranks,16 replay sessions,
and complete repeated operation signatures. Kernel sums are per-rank work, not
elapsed time. Same-rank model/sampler sums within token-out provide useful
attribution; independent fixed-token model and fixed-logit sampler loops do not
form an additive autoregressive decomposition. Op dtypes/fidelity are copied
from actual rows, with missing metadata left missing.

The24.99350229ms isolated-layer budget is a prior-workload estimate and excludes
embedding/final norm/LM-head/sampling/orchestration. It cannot prove the full
model has no regression or provide an exact overhead subtraction. The reduced
profile has real layers0/3 and real terminal/sampler contracts, suitable for
identifying terminal operations; its sampler share alone cannot determine the
all40 bottleneck. Final all40 component timings and actual profiling must be
read together before accepting that sampler overhead is small. If the all40
sampler control dominates token-out, investigate the real terminal graph before
stage completion.

Validation performed here: Python syntax, CPU fixture metadata and AST checks
that observed forwards use resident inputs and no readback, while the one token
read stays inside the measured loop and outside the forward guard. Hardware
acceptance, final numbers and source-bound artifact comparison remain pending.
