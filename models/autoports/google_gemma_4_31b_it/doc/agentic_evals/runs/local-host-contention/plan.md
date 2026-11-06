# Largest-batch host contention measurement

Coordinator goal: measure largest validated batch32 decode delivery and engine process CPU per decode step, first idle and then under one busy process per available host CPU. No numerical changes, scored tasks or performance optimization. Coordinator owns the occupied local device and launches this only after the current B1 job closes and ownership checks permit it.

Confirmation: existing check_serving_load_costs.py succeeds with contention-only, 32 complete requests per wave, a valid decode-only CPU/log window containing active_rows32, and cleanup of all owned stress workers. Compare aggregate delivered output tokens/s and engine_host_cpu_ms_per_decode_step. The latter is summed engine process CPU across its threads, not exclusive host-wall overhead; no device timing or profiling.

Host inspection in host_admission.json found affinity CPUs0–15, os.cpu_count16, effective cpuset0–15 and cpu.max `max 100000` (no CPU quota). The existing script creates exactly16 unpinned busy workers inheriting this affinity. This is the requested one-process-per-available-CPU load. Do not change affinity or quota. Run no other local benchmark/client concurrently. Recheck the host facts at actual launch; if changed, record the new regime and use the client's actual affinity-based count.

Predicted client time:60seconds (0.0167h), based on earlier stage10 B32 idle8.272s and busy12.489s decode windows plus setup/cache resets and Python/tokenizer startup. This is a conservative estimate, not a new measurement. Device-hour box:0.25h after server startup, with `timeout 900` around the client. Existing serve launcher is bounded7200s; coordinator must keep its stage queue bound and manage graceful idle shutdown. Never terminate a server with outstanding device work manually. No added hardware launch by this effort.

Server command from checkout (after B1 close):

```bash
bash models/autoports/google_gemma_4_31b_it/tests/run_serving.sh --stages serve --max-num-seqs 32 --enable-prefix-caching --output-dir models/autoports/google_gemma_4_31b_it/doc/agentic_evals/runs/local-host-contention/server
```

The existing launcher preserves context262144, pinned HF/tokenizer revision, async scheduling, stock plugin runtime and top_k32 served default. Confirm effective chunk2048 in startup log and exact autoport/full60 registration; record hashes and actual launch in coordinator queue manifest. Do not inherit profiler/watcher or reduced-layer settings.

Client command (substitute the server's actual EngineCore PID):

```bash
timeout 900 python_env/bin/python models/autoports/google_gemma_4_31b_it/tests/check_serving_load_costs.py --url http://localhost:8000 --engine-pid ENGINE_PID --server-log models/autoports/google_gemma_4_31b_it/doc/agentic_evals/runs/local-host-contention/server/server.log --output models/autoports/google_gemma_4_31b_it/doc/agentic_evals/runs/local-host-contention/contention.json --batch 32 --stages contention --decode-tokens 256 --request-timeout 120
```

Requests use128 prompt tokens,256 output tokens and deterministic top_k1 stress. No headline4096/128/B1 performance claim. Record client exit/timestamps and source identity. The client terminates/reaps its own16 stress workers in finally; if timeout fires verify workers and outstanding HTTP/device work before any graceful server stop. A failed clean-window check is evidence for diagnosis, not permission to weaken it. After success require zero running/waiting requests before coordinator graceful shutdown, then owner recheck and queue closure. Write final effort evidence once after the run.
