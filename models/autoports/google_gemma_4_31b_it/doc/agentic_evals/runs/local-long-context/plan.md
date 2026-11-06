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

## Coordinator amendment: advertised-context envelope

After the first phase completes, reuse the same idle B8 server for a second independent two-client history sequence:131072,196608,262136 prompt tokens, eight output tokens per request. The final prompt+output equals262144, preserving the advertised context. This phase uses fresh unique prefixes and does not reset the cache; the prior phase can remain as evictable cached state. Therefore this is also an admission/eviction observation, not a clean-cache performance comparison. Record the two phases separately.

The existing client supports this exact endpoint: it requires each next context to exceed the previous context plus output_tokens, which all three points satisfy. It submits explicit token IDs, validates exact prompt/output usage and vocabulary bounds, and does not round context to a tile boundary. Final262136 is intentionally not block32-aligned; the attainable prefix-hit ceiling accounts for the final unprocessed generated token and block alignment. The same existing trace/prefix checks apply unchanged. At final completion two logical histories demand524288 tokens; actual physical allocation depends on full/sliding cache groups and eviction and must be read from evidence, not inferred solely from that logical sum.

```bash
timeout 600 python_env/bin/python -m models.autoports.google_gemma_4_31b_it.tests.check_agentic_serving_traffic --concurrency 2 --contexts 131072,196608,262136 --output-tokens 8 --server-log models/autoports/google_gemma_4_31b_it/doc/agentic_evals/runs/local-long-context/server/server.log --output models/autoports/google_gemma_4_31b_it/doc/agentic_evals/runs/local-long-context/advertised_context.json
```

Prediction300seconds is a guess based on first-phase observed long-prefill progress; longer full-attention work can grow nonlinearly, so monitor and retain the original1800second total box. Coordinator reports the server age about360seconds and first-phase active runtime about100seconds at amendment. Re-evaluate remaining box when first phase ends before launching. The600second client bound does not extend the original effort deadline. A timeout or pool/preemption shortfall is retained and diagnosed, not hidden by lowering advertised context. Never send manual SIGTERM while device requests remain outstanding.

No scored quality claim follows from completing eight greedy output tokens at the endpoint. GPQA can produce much longer free-running replies; request+output accounting, not task accuracy, is the purpose of this phase.
