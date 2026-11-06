# Terminal smoke runtime decomposition

All47 calls have ordered API request timings and token usage (100% coverage). Task/trajectory/job usage reconciles exactly:1,248,134 input tokens and70,620 output tokens. Suite makespan2295.204564s (0.637557h). Device cost for this analysis0; release savings unknown. This effort ranks below validity repairs and does not re-audit scores.

| Task | Calls | Agent seconds | API seconds | Other seconds | API share | Peak prompt / completion tokens |
|---|---:|---:|---:|---:|---:|---:|
| Scheduler |5|436.563550|433.239068|3.324482|99.24%|10737 /4669|
| Mailman |42|2182.950288|1872.590740|310.359548|85.78%|53645 /3447|

API seconds are client wall time around await chat.chat in pinned Harbor e76c762f, including server queuing/prefill/decode/network/client work. They are not device generation time. Other seconds combine tool execution, waits, parsing, terminal/control IO and other host work. Requested command waits total2.7s scheduler and304.0s Mailman, close to residuals; those configured waits are not independently measured tool timings. Per-call timing/usage and episode residuals are in calls.json, without raw messages.

The critical path reconciles exactly:8.842307s before Mailman agent +2182.950288s Mailman agent +103.411969s afterward =2295.204564s suite. Summed agent time2619.513838s and summed API time2305.829807s include concurrent work; neither is suite wall/device time. Agent-window start/end and all step timestamp differences also reconcile within1microsecond. Verifier times are2.602471s scheduler and92.659297s Mailman; setup/cleanup explain the remaining trial time.

Four recovered invalid-JSON responses consume403.807461 summed request-seconds and12,692 output tokens: scheduler step4=121.182376s; Mailman steps5/18/37=282.625085s combined. Their non-API episode residual is0.209786s total; this is an upper aggregate of inter-episode host work around those responses, not measured parser CPU. Eliminating malformed responses could save work, but release-hour savings and ensuing trajectory/context changes are unknown. No fix or new run was attempted.

Peak prompt+completion usage is14982 scheduler and54247 Mailman. These are observed client token counts, not full advertised-context tests. Per-request cap32768 and context262144 exceed observed values. No raw serialized chat requests or token IDs are present: parsed analysis/plan, reasoning, commands and observations cannot reconstruct exact working-state prompts. Zero reported cache tokens do not prove zero server prefix reuse.

Coverage exceeds95% for call timings/usage;95% fine-grained generation/tool/parser attribution is unavailable. Missing: device durations, TTFT/inter-token times, per-request start/end timestamps, actual per-command start/end, separate parser CPU, and finish reasons. This prevents device TPS/roofline claims or exact concurrent request-wall-union measurement. Source semantic checks and all reconciliation details are in result.json. Existing42190891568 artifacts only; no source, central docs, launches or device changes.
