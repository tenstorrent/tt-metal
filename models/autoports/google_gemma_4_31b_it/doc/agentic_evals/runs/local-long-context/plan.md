# Two-client long-history cache check

Coordinator goal: extend synthetic growing histories beyond the earlier32768-token check toward the long histories possible under GPQA's release output cap126976. Measure cache validity and pool admission for two smoke clients on supported batch8. This is neither GPQA execution nor a quality check nor evidence that real agents fit the pool.

Confirmation: the existing check_agentic_serving_traffic module completes all five rounds for both clients, records hit fractions and late hits against its documented attainable ceiling, prefilled/final token ratio, preemptions and captures. Expect its existing checks to pass without weakening them. Quantify and report any cache/pool shortfall; preserve partial results and diagnose failures before further hardware work. No numerical source or serving-policy change is proposed.

Device budget:0.5hours including startup. Initial prediction:0.15hours (540seconds), a guess extrapolated from the measured cold32768 prompt8.453seconds plus startup and long-attention scaling uncertainty. Two final131072-token histories entail262144 total resident history tokens; this is a workload demand, not a pool-capacity proof. Both clients generate only8 tokens per turn. Longer decoded replies can change eviction, cache occupancy and correctness; those remain for actual trajectories.

Coordinator owns and serializes device work, with ownership checks after prior jobs close. This effort does not start a server, open devices or reset hardware. Use existing launcher at B8, advertised context262144, prefix caching, async scheduling, pinned HF/tokenizer revision and served top_k32 default. Verify actual full60 autoport registration, source hashes and chunk2048 from startup evidence. No profiler/watcher or reduced-layer setting.

Server command:

```bash
bash models/autoports/google_gemma_4_31b_it/tests/run_serving.sh --stages serve --max-num-seqs 8 --enable-prefix-caching --output-dir models/autoports/google_gemma_4_31b_it/doc/agentic_evals/runs/local-long-context/server
```

Client command after warmup:

```bash
timeout 1200 python_env/bin/python -m models.autoports.google_gemma_4_31b_it.tests.check_agentic_serving_traffic --concurrency 2 --contexts 4096,32768,65536,98304,131072 --output-tokens 8 --server-log models/autoports/google_gemma_4_31b_it/doc/agentic_evals/runs/local-long-context/server/server.log --output models/autoports/google_gemma_4_31b_it/doc/agentic_evals/runs/local-long-context/growing.json
```

Record planned_clients2, batch8, the bound and startup time before queue launch. The coordinator must fit startup/client/shutdown into the1800second effort box; if startup consumes600seconds, re-estimate the remaining client allowance before beginning it. The unchanged launcher has its own7200second safety timeout; it is not authorization to exceed the effort box. Do not manually terminate a server with outstanding device work. On client failure preserve logs and inspect active requests, returning control to the coordinator. On success confirm zero running/waiting requests, request graceful shutdown and close the queue only after device owners clear. Final evidence is written once when the run ends.

The client sends deterministic raw token continuations and reports HTTP wall time, not device performance. No score, headline4096/128/B1 number, output-cap adequacy claim or supported-context reduction follows from this test. Context131072 is a test point inside the advertised262144, not a new model limit.
