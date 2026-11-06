# Local serving packaging confirmation

Coordinator goal: confirm the published autoport source loads all 60 layers through the stock local vLLM runtime, then measure smoke-shaped two-client growing traffic at supported batch 8. Confirm exact registration, successful replies, prefix reuse, and no trace capture after warmup. This is synthetic capacity evidence, not a valid agentic suite score or a claim about real agent capacity.

Device budget: 0.5 device hours, initial estimate 0.2 hours including startup (guess informed by earlier stage startup). Coordinator owns the sole local device queue and launches every command. This effort only prepares HTTP client orchestration; it never opens hardware, resets, starts a server, or terminates one.

1. Coordinator checks owners and launches tests/run_serving.sh with serve-only, max_num_seqs 8, prefix caching, original context 262144, and max_num_batched_tokens 2048. Preserve revision, async scheduling and top_k32 defaults. No profiler, watcher or modified plugin runtime.
2. Before client traffic, read the server's autoport_registration event and require exact autoport class, module file and layer indices 0..59. Record the source hashes and compare them to setup/source_publication.json; publication doc-free fallback has separate host-test evidence because the local runner installs readiness_check.
3. Submit two interleaved synthetic growing conversations at 4096, 8192, 16384, 24576, 32768 prompt tokens, eight generated tokens each, through the existing check_agentic_serving_traffic module.
4. Use existing check_serving_load_costs.py at batch 8 for cold 4096/32768 prefill and seven overlapping decode requests while the long request prefills. These synthetic workload lengths remain estimates until real trajectories exist. Skip contention here: the full skill's contention measurement belongs at largest validated batch 32.
5. Save client commands, timings, exit statuses, sampling counts and registration into this run directory. On client failure retain evidence and return to the coordinator for diagnosis, without shutting down a possibly busy server. On success coordinator verifies idle metrics before graceful shutdown. If idle is unproven, no manual SIGTERM; bounded server timeout owns termination.

Acceptance: successful registration/full60, both clients complete every growing round, the existing cache/trace checks pass, and prefill/overlap client succeeds. Record shortfalls without calling this a release preflight or real-capacity measurement. No numerical source change is proposed, so no scored pair is needed.
