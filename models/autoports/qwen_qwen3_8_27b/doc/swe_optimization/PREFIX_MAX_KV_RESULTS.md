# B32 prefix reuse after reclaiming DRAM

The matched serving comparison passed all212 requests in each arm. All input
identities, salts and eligible-prefix counts match; all212 greedy token outputs
are identical across arms. Native context262144, B32/A32, checkpoint512, the
checkpoint/tokenizer revision, precision and one-output generation policy are
fixed. Each arm uses a fresh vLLM0.30 server and the same immutable source image.
The JSON companion records exact source/image, bytes and CI run IDs.

| Arm | Physical KV IDs | KV bytes/device | Eligible-prefix reuse | Client wall | Observed preemptions |
|---|---:|---:|---:|---:|---:|
| Original allocation |1368|23,646,044,160|39.61%|3255.67s|25|
| Maximum without weight sharing |1379|23,836,753,920|40.18%|3228.81s|24|
| Maximum with weight sharing |1470|25,409,617,920|47.52%|2950.16s|17|

Reclaimed weights provide91 additional global IDs and1,572,864,000 physical KV
bytes/device (1.465GiB) beyond the maximum control. The separate1368→1379 gain
uses190,709,760 bytes of existing headroom. Native maxima require the same512MiB
normal-DRAM event-peak/8MiB TRACE/fragmentation guards, closed three-cycle stress
passes and adjacent-ID guard failures. Serving attests these exact pool sizes.

Relative to the maximum control, shared weights improve eligible-prefix reuse
by7.34 percentage points and reduce observed client wall time by8.63%. These
are single-run synthetic mechanism results across the CI runner pool; they do
not establish repeated performance, SWE scores or agent end-to-end speedup.
The previously measured3.67% standalone decode cost remains relevant.

## Why reuse is still below90%

Capacity remains restrictive:32 private32K histories need2048 attention IDs
alone, exceeding the1470-ID pool before recurrent states/reservations. The
shared pool physically allocates6,550,978,560 attention bytes and18,858,639,360
recurrent-snapshot bytes/device; the current global-ID layout reserves both
backings even though groups compete for IDs. Seventeen preemptions remain.

Checkpoint coverage is another measured limitation. Every16K warm request
reuses12288 tokens versus15872 eligible (77.42%). Recovered worker events show
512-token physical chunks but no state save at the first seven boundaries of
a4096-token submission; the eighth boundary saves. The adapter saves only
boundaries with allocated recurrent IDs; its spec currently disables extra
prefill checkpoint blocks. Thus512-token execution is not512-token reusable
state coverage. Exact attribution of every miss to attention eviction versus
missing/evicted recurrent checkpoints still requires scheduler group ownership
and checkpoint-allocation evidence. Installed-runtime source readback is
queued in CI40913533837 before changing the integration.

The47.52% metric excludes compulsory misses: denominator3,112,960 eligible
prefix tokens, with5,310,464 total input tokens. Cold/churn/fresh-control requests
are intentionally private; even perfect retention yields only58.62% cached
input over all requests. A90% target must use eligible-prefix retention here.

The first shared-arm attempt40869225573 failed before model startup because a
prior diagnostic directory prevented checkout cleanup. The repaired attempt
40882793766 passed. Worker JSONL is recovered through the normal tt_triage
artifact; the separate exact-two-file audit exporter could not export because
the completions-only probe did not create its chat-only HTTP audit file. Client
responses, worker events, physical attestations and sampled metrics are retained
locally/CI; no raw logs, graphs, CSV or Tracy data are committed.
