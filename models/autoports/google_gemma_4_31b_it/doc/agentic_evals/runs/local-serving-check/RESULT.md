# Local serving confirmation result

The full60 autoport passed both synthetic HTTP checks. `client_manifest.json` verifies all 14 published source hashes and registration of `models.autoports.google_gemma_4_31b_it.tt.generator_vllm:Gemma4ForCausalLM`, layers 0–59, on devices 0–3 with mesh 1×4. No numerical source change was made. The local packaging/runtime question is solved; CI image execution, real agent capacity and valid suite scores remain open.

`server/vllm_run_summary.json` preserves the exact launch: B8, context 262144, block32, prefix caching, async scheduling, HF/tokenizer revision 842da3794eaa0b77d5f08bae87a17459d91ff475, served top_k32 override. Startup log confirms effective max_num_batched_tokens 2048 and vLLM 0.31.0. The local runner provides readiness_check; the absent-readiness fallback is covered by prior host packaging tests, not this live run.

| Measurement | Observed result | Evidence |
|---|---:|---|
| Growing client process wall time | 25.161 s | client_manifest.json |
| Growing HTTP workload time | 20.607 s | growing.json |
| Load-cost client process wall time | 51.308 s | client_manifest.json |
| Combined client process wall time | 76.469 s (0.02124 h) | client_manifest.json |
| C2 growing requests | 10/10 completed, 8 generated tokens each | growing.json |
| Final context per conversation | 32768 tokens | growing.json |
| Prefix hit fraction of attainable ceiling | 0.994669 | growing.json |
| Late hit fraction (last four requests) | 0.996534 | growing.json |
| Prefilled tokens / final context sum | 66048 / 65536 = 1.0078125 | growing.json |
| Preemptions | 0 | growing.json |
| Post-warmup trace captures | 0 | growing.json |
| Cold 4096-token prompt, one output token | TTFT 757.864 ms; 5404.666 input tokens/s | load_costs.json |
| Cold 32768-token prompt, one output token | TTFT 8452.578 ms; 3876.687 input tokens/s | load_costs.json |
| Seven decoders during one 32768-token prefill | 63 delivered tokens over 8.816666 s; 7.145558 aggregate tokens/s | load_costs.json |

The growing context sequence was 4096, 8192, 16384, 24576, 32768, with two clients interleaved on the B8 server. Each measured reuse round exceeded 0.98. Decode/sample/buffered-sample traces each had one startup capture and zero new captures during this workload; the existing trace lifetime check passed. All eight overlap clients completed, including seven 1024-token anchor outputs. Every anchor remained live throughout the long prompt's submit-to-first-token window. That window includes scheduling delay and is not device prefill time.

Across the combined client interval, sampling logs contain 61 host-compatible intermediate-prefill steps, 11 device-prefill steps and 1066 device-decode steps. This is the accepted stock-plugin mixed chunked sampling limit; no plugin runtime change or scheduler workaround was applied. Growing alone accounts for 30 host-prefill, six device-prefill and 43 device-decode steps. No measured client reported an error or missing reason.

The prediction was 0.2 device hours with a 0.5-hour box. Client measurements alone used 0.02124 hours; startup, coordinator idle intervals and device close must be charged from the coordinator's queue log, so this report does not label client wall time as total device usage. There is no predicted speed saving: this was validity confirmation. The client-side confirmation succeeded within the box; device queue closure remains coordinator-owned.

These are synthetic deterministic raw-continuation requests, not chat quality, scored tasks, or measured real-agent working sets. The 4096/32768 lengths came from existing workload estimates. The load-cost artifact retains the original target_concurrency32 estimate; actual measured concurrency was one for cold prefill and eight for overlap. It does not establish C32 behavior. Host contention at largest validated batch32 was outside this effort. Neither cold-prefill row is the required 4096/128/B1 headline measurement; no device timing, profiling or roofline measurement was performed.

`idle_shutdown.json` records completed clients and both running/waiting gauges at zero before the coordinator requested graceful shutdown of its owned API process. It records the shutdown request, not proof that devices have closed; queue closure and owner recheck belong to the coordinator. No additional device work was launched by this effort.
