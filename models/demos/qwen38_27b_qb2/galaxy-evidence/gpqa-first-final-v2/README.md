# Corrected full GPQA and separate OpenBench result, October 9, 2026

Corrected GPQA completed **176/198 (88.89%)**, no truncations, in 50m49s.
All 198 questions are included. Configuration: BFP8/HiFi2 decoder, native
recurrence, BFP8 KV, FP32 recurrent state, eight TP4 replicas, concurrency 128,
temperature 1, top-p .95, top-k 20, seed 42, output budget 65,536 tokens.
The user accepted GPQA. The immutable receipt still reports `passed: false`
against the original .892 threshold (177/198); it was not rewritten.

`gpqa-first-v2/qualification/evaluation/gpqa/` contains the original completed
summary and progress. Measured client output was 419.48 aggregate tokens/s and
13.01 mean decode tokens/s/user; this is the variable-length evaluation workload,
not a saturated fixed-length throughput sweep. Raw private questions and answers
remain on the test host.

The separate pinned OpenBench run completed 190 questions and errored on eight:
170/190 scored correct (89.47%), or **170/198 (85.86%)** with the errors counted
incorrect. Its summary explicitly says `complete: false`. The upstream scorer
excludes request errors from its accuracy denominator; do not present 89.47% as
a clean 198-question result. No completed response was truncated.

All eight errors were HTTP read timeouts. The installed OpenAI-compatible
provider retained its SDK's 600-second HTTP timeout and two automatic retries;
Inspect's `timeout=7000, max_retries=0` applies to a separate layer. Error durations
were 1,802-1,806 seconds, consistent with three timed-out requests. The original
protocol's assertion of no retries was therefore not enforced at the SDK layer.
`openbench-overnight-v1/error-audit.json` preserves IDs, timings and error types.

The runner now sets the SDK timeout explicitly to 7,000 seconds and disables
SDK retries, while retaining the same Inspect policy. A delayed-response HTTP
regression against the pinned installed evaluator proves a short timeout fails
after exactly one POST, then a longer timeout succeeds and the upstream scorer
uses the final answer. Receipt: `openbench-timeout-control-v4/transport-proof/`.
Earlier probe v2/v3 attempts failed on incorrect timeout-inspection assumptions;
the SDK overrides the httpx transport's timeout when building each request.
Those were diagnostic errors, not model inference failures.

The new run will evaluate all 198 questions afresh against the actual container
after its API/tool smoke, with identical upstream prompt/scorer and temperature
.5, one epoch, and a 65,536-token output cap. It has separate output and will not
merge or select the better answer from the two runs. This is not exact OpenRouter
parity: upstream uses ten epochs, and OpenRouter's internal overrides are unknown.

The TTIS branch `anatarajan/qwen38-galaxy-release-20261009` contains the persistent
container supervisor and its launch receipts. The existing performance queue is
unchanged; the container job waits for its exact invocation and clean terminal
receipt, then acquires the hardware lock. Status snapshots here are historical.
