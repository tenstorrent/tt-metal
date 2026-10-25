# Stage 11 run notes

First timestamped setup command observed 2026-10-25 14:22:26 UTC; full setup elapsed time was not measured; see setup.json. Stage10 left no live server; its last B1 configuration and prior results are retained under optimized_vllm. Prior startup duration is unknown. The first launch will count inside benchmark_stage run.

Four configured batches give a 7200-second total client budget. All launches, compilation, warmups, verification, inference, phase collection and reporting count.

GPQA uses the frozen common 10/198 subset; exact mirror population hash verified. See protocol.json for all settings. SWE-bench Verified and Terminal-Bench 2 require goal12 agent tools and are not covered by lm-eval. No accuracy acceptance threshold is imposed.

Phase accounting is host wall time, not device time. No live device profiling or added synchronization. Detailed estimated work and limitations are in tests/benchmark_work.py.

Setup commit: d84c5effaba (local only). During the timed first selection, repaired bridge-network transport with tests/benchmark_proxy.py. Host loopback sends through this transparent relay; measured HTTP latency includes it. All64-layer B32 server ready at14:40:46UTC; collector live prefill/decode check passed. Startup log reports nominal1350MHz request settling1331MHz; roofline uses documented theoretical1350MHz peak, not a measured clock-derived peak.

Attempt01 was interrupted after independent AutoDebug identified prefill/decode transition misattribution in the collector. Its runner measured939.963857 seconds; five responses stopped normally before cancellation. No scored accuracy or performance row is accepted from that incomplete attempt. Raw responses, original manifest/configuration, timing stream and failed report remain in attempt-01-incomplete/. The owned server was stopped gracefully; no reset was needed. The original stage deadline remains2026-10-25T16:36:33Z (derived from initial runner configuration timestamp plus7200s); retry budget is reduced to remaining time, including repair/verification/reporting.

Repair adds actual drain, input-build, state-apply, admission, request-finish and scheduler-consumption spans. The collector preserves output-ready boundaries, carves explicit old-phase host spans, and separates request-free turnaround. Deferred terminal bookkeeping that was not executed during generation is recorded without invented duration. Synthetic tests cover both transition directions, delayed consumption, idle gaps and once-only completion; live readiness must validate the repaired partition before accuracy.
