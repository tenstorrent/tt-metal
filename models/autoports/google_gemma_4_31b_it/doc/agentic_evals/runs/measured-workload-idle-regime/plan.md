# Measured workload under idle host regime

Coordinator goal: repeat the prior measured-workload-prefill five-request protocol after the HF CPU reference ends. Use full60 stock B8, planned_clients2: cold prefills4096/31755/107585, one2048-output-token anchor and one107585-token overlapping prefill. Same workload JSON, script, sampling and output controls; script-generated UUID prefixes intentionally differ between runs to bypass APC. This is an exact protocol/length repeat, not byte-identical synthetic prompts. No scored evaluation, task execution or runtime change.

Prediction:0.20device hours (720s), guess, versus prior measured0.275803313h including startup/failures/retry. Box0.35h (1260s), inclusive startup. Healthy prior retry ~4min supports a client600s bound after readiness; allow up to600s startup then reassess remaining time. Do not start a full600s client when fewer than600s remain. Coordinator owns queue, hardware, launch, health, and graceful shutdown. The launcher has a broader7200s wrapper; the coordinator must enforce this effort's1260s queue budget. No automatic hard kill of active device requests.

Server command from repository root (coordinator preserves the current phase-export environment and records it):

```bash
bash models/autoports/google_gemma_4_31b_it/tests/run_serving.sh --stages serve --max-num-seqs 8 --enable-prefix-caching --output-dir models/autoports/google_gemma_4_31b_it/doc/agentic_evals/runs/measured-workload-idle-regime/server
```

Launcher preserves model/tokenizer revision842da3794eaa0b77d5f08bae87a17459d91ff475, full262144 context, native gemma4 reasoning/tool parsers, automatic tool choice, stock plugin and top_k32 served default. Confirm actual registered autoport and full60 load from server log; compare runtime source hashes/config with prior server before making comparison claims. No profiler, CPU stress workers or unrelated clients. Confirm HF CPU process93575 is finished rather than assuming idle; record current CPU affinity/quota, load, active CPU consumers and competing workloads at client boundaries. Ordinary background OS activity remains a limitation.

Client command after health and concrete engine PID are known:

```bash
timeout 600 python_env/bin/python models/autoports/google_gemma_4_31b_it/tests/check_serving_load_costs.py --engine-pid ENGINE_PID --server-log models/autoports/google_gemma_4_31b_it/doc/agentic_evals/runs/measured-workload-idle-regime/server/server.log --output models/autoports/google_gemma_4_31b_it/doc/agentic_evals/runs/measured-workload-idle-regime/result.json --workload models/autoports/google_gemma_4_31b_it/doc/agentic_evals/runs/measured-workload-idle-regime/workload.json --batch 2 --stages prefill,overlap --overlap-decode-tokens 2048 --request-timeout 600
```

Timeout fix verified in source: urlopen(timeout=args.request_timeout); total elapsed deadline check remains. Prior failure was a60s socket cap; do not restore it. Client resets APC while idle between stages and uses unique prompt prefixes. Cold output controls and overlap anchors are unchanged stock client defaults. --batch2 controls client count, not server B8.

Confirmation: all five requests complete; three exact cold lengths; all_anchors_live_through_window true; empty missing_reasons; same serving identity/config; idle-host regime documented. Report host-wall TTFT/input rates and delivered anchor tokens per overlap window, never device time or roofline. Compare successful prior retry only for request timings; prior queue total includes failed attempts and cannot establish throughput gain. Single sequential unrandomized samples, fresh UUID prefixes, startup/cache differences and host background activity prevent a causal speed claim. No release hours saved claimed.

After measurement record result write time for device charge, check metrics running=waiting=0, request graceful API SIGINT and allow idle owner closure. Preserve partial results on failures and report rather than silently retrying outside box. Final evidence is written once when coordinator supplies results; this preparation performs no device access.
