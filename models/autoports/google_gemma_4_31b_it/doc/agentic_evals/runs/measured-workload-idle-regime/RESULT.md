# Idle-host workload measurement result

Passed all five diagnostic requests, planned_clients2 on supported B8. Cold inputs4096/31755/107585 each produced one token; anchor128-input/2048-output and overlapping107585-input/one-output both completed. Anchor remained live through the entire overlap window; missing_reasons empty. No optimization or runtime change was made, and no quality, real-agent capacity, score or release savings claim follows.

| Cold input tokens | Idle TTFT ms | Prior HF-contended retry TTFT ms | Idle/prior time ratio | Idle input tokens/s |
| --- | --- | --- | --- | --- |
| 4096 | 782.831 | 2228.286 | 0.3513 | 5232.291 |
| 31755 | 8284.344 | 16951.805 | 0.4887 | 3833.134 |
| 107585 | 48065.080 | 72683.447 | 0.6613 | 2238.319 |

These are HTTP host-wall first-token times, including scheduler/transport work, not device latency or rooflines. APC was reset while idle and initial prefixes were unique.

Long-prefill overlap window: 49.259438s from submit to first token, versus prior 78.910179s; time ratio 0.624247. The ongoing anchor delivered28 tokens during each window, now 0.568419tokens/s versus 0.354834. Anchor lifetime from submit to completion was 105.017666s. Client wall elapsed166.855231s; final request completion162.169970s after the client's internal start. Client exit0.

Configuration comparison: complete server_command arrays match exactly between saved summaries. Autoport registration events and physical-cache allocation events match exactly: this checkout's generator_vllm:Gemma4ForCausalLM, all60 layers, devices0–3/mesh1×4, same head_bfp4 precision policy. Context262144, block32, B8, prefix caching, native gemma4 parsers, asynchronous scheduling, model/tokenizer revision842da3794eaa0b77d5f08bae87a17459d91ff475 and top_k32 served defaults match. Workload JSON and client controls match; corrected request-timeout600 socket/total-deadline behavior was host-regression checked. Historical per-run source-byte hashes were not captured, so matching module identity/configuration is not an independent byte-for-byte source attestation. No TT source commits were found in execution checkout history between prior startup and this run. Phase-export output paths differ by effort.

Host regime: HF PID93575 absent before client;16-CPU affinity0–15 and unlimited cgroup CPU quota. Load averages before5.08/7.64/12.15 and after4.26/5.86/10.61. Ordinary coordinator commands and short pre-commit hooks overlapped; no HF reference or stress workers. Thus 'idle' means absence of the deliberate HF/stress load, not an isolated or motionless host. Fresh UUID prompt content, one sequential unrandomized measurement per regime, cache/startup state and ordinary host work limit comparisons. Observed timing ratios are descriptive, not a causal optimization claim.

Queue actual 0.1090498825device hours (392.579577s), versus0.20h/720s guess and0.35h/1260s box; actual/forecast 0.545249. Prior queue0.2758033131h included failed readiness/client attempts and retry, so its ratio 0.395390 is accounting only, not comparable serving speed. New queue includes startup through idle owner closure. Running/waiting metrics both0 before API SIGINT; wrapper exit1 is intentional idle shutdown, client and queue exit0, owners empty afterward.

Evidence: result.json; client-exit.json; client-command.json; host-before.json; idle-shutdown.json; server/vllm_run_summary.json and server.log; ../../device_queue/local/measured-workload-idle-regime/job.json; ../measured-workload-prefill/result-retry.json and server summary/log. No further device work or central-document edits were performed for this result.
