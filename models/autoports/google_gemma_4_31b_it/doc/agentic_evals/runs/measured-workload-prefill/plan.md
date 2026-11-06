# Prefill at observed workload lengths

Coordinator goal: measure cold prefill at4096,31755,107585 tokens and interference on one ongoing decoder while107585 tokens prefill, using the full60 stock B8 server. The typical31755 and observed working-set107585 values come from150 actual SWE response-usage rows;11 attempts have missing usage and the task scores are quarantined. These token lengths are observed workload sizing inputs, not general capacity bounds. Diagnostic synthetic prompts do not execute tools or score tasks.

Use the existing check_serving_load_costs.py unchanged with a private workload JSON containing these lengths. Its prefill stage resets prefix cache while idle and creates unique prefixes for each cold measurement. Its overlap stage with --batch2 creates one decode anchor plus one prefill request; --batch is client concurrency here, while the separately launched server remains max_num_seqs8. Set overlap output2048 to keep the anchor alive through expected long prefill. Check all_anchors_live_through_window; a completed anchor before long prefill makes overlap incomplete rather than a valid throughput estimate. Request timeout600 accommodates long prefill under potential host contention.

Initial estimate0.15device hours (540s) including startup; device box0.3h(1080s). Client prediction300s: cold4096/31755 approximately1/10seconds,107585 perhaps60seconds based on earlier86016 prefill35.4s with nonlinear attention, then one2048-token anchor plus overlapping long prefill approximately140seconds, plus startup/client overhead margin. These are guesses under the new context lengths and host regime. Bounded client600s within the overallbox; reassess remaining box after server startup.

Regime: local four-chip1×4, serialized TT hardware. HF CPU reference may run concurrently. Record whether it actually overlaps and its PID/CPU activity; this makes host-wall timing a contended-host measurement, not directly comparable to idle-host earlier runs or CI performance. Do not stop another effort's process or imply a causal speed regression. No profiler/watcher or configuration tuning.

Coordinator launches existing tests/run_serving.sh --stages serve --max-num-seqs8 --enable-prefix-caching --output-dir <effort>/server, retaining currently selected native reasoning/tool parsers and autoexport settings exactly as its other current servers. Preserve pinned revision, full60, context262144, chunk2048 and top_k32 served default. Record exact command/runtime identity; the diagnostic's greedy requests do not alter release defaults.

Client (replace ENGINE_PID; all paths from checkout):

```bash
timeout 600 python_env/bin/python models/autoports/google_gemma_4_31b_it/tests/check_serving_load_costs.py --engine-pid ENGINE_PID --server-log models/autoports/google_gemma_4_31b_it/doc/agentic_evals/runs/measured-workload-prefill/server/server.log --output models/autoports/google_gemma_4_31b_it/doc/agentic_evals/runs/measured-workload-prefill/result.json --workload models/autoports/google_gemma_4_31b_it/doc/agentic_evals/runs/measured-workload-prefill/workload.json --batch 2 --stages prefill,overlap --overlap-decode-tokens 2048 --request-timeout 600
```

Confirmation: three cold prefill lengths complete, both overlap requests complete, anchor remains alive, source/runtime unchanged, client missing_reasons empty. Report TTFT/input rate and other delivered output tokens/s during submit-to-first-token interval, including scheduler wait. No device-time or exact phase attribution claim. Client alone does not capture stream-gap percentiles; earlier diagnostic already established that mechanism, so do not duplicate it. Release impact remains conditional on measured real task overlap/generation share; no speculative hours saved.

Coordinator owns launch and lifecycle. Check owners before the server starts, verify health before the client, retain partial JSON/logs on failure, and do not manually terminate outstanding device work. After success require running/waiting metrics bothzero before graceful shutdown; finish queue only after closure/owner check. Final evidence written once after results. This preparation launches nothing.
