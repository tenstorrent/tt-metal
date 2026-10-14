# Bring-up LOG (append-only; see models/common/bringup_kit/README.md)

## E261014-0001-a72
- date: 2026-10-14T00:01:01
- commit: a69ec795094
- branch: ssinghal/olmo3-galaxy-main
- host: UF-EV-A9-GWH02
- track: decode
- status: verified
- claim: Baseline traced decode, 64 layers, batch 32, ISL~128 (batch-32 cfg), greedy: 38.31 ms/token = 26.10 tok/s/user (835 tok/s), TTFT 1326 ms; gate-b32 (32 distinct prompts, 256 tokens): 38.19 ms = 26.19 tok/s/user. Single unpaired runs, warm steady-state average from the demo.
- evidence: /home/ubuntu/ssinghal/olmo3-logs/a69ec795094_base1_batch-32.log
- paths: models/demos/olmo_galaxy, ttnn/cpp/ttnn/operations/transformer/sdpa_decode
- data: {"decode_ms": 38.31, "tok_s_user": 26.10, "batch": 32, "layers": 64, "ttft_ms": 1326}

## E261014-0001-2a4
- date: 2026-10-14T00:01:01
- commit: a69ec795094
- branch: ssinghal/olmo3-galaxy-main
- host: UF-EV-A9-GWH02
- track: correctness
- status: verified
- claim: Traced batch-32 decode is NOT deterministic run-to-run on one host and build: two identical gate-b32 runs (separate processes) diverge for 32/32 users, first divergence at generated token 6..229; output stays coherent. Determinism gate FAILS at baseline.
- evidence: /home/ubuntu/ssinghal/olmo3-logs/a69ec795094_base2_gate-b32.json
- paths: models/demos/olmo_galaxy, models/common/sampling

## E261014-0001-a54
- date: 2026-10-14T00:01:02
- commit: 256d9b8daed
- branch: ssinghal/olmo3-galaxy-main
- host: UF-EV-A9-GWH02
- track: correctness
- status: inferred
- claim: After 256d9b8 (sampler no longer frees persistent SAMPLING_VALUES/INDICES; measured on the uncommitted tree that became 256d9b8): untraced decode (OLMO_DECODE_TRACE=0) runs (556 ms/token eager); traced gate-b32 still nondeterministic (2/32 users identical across 2 runs); eager vs traced diverge from token 2. Remaining nondeterminism source not yet found.
- evidence: /home/ubuntu/ssinghal/olmo3-logs/a69ec795094-dirty_fix2_gate-b32.json
- paths: models/demos/olmo_galaxy, models/common/sampling

## E261014-0001-8b4
- date: 2026-10-14T00:01:02
- commit: 256d9b8daed
- branch: ssinghal/olmo3-galaxy-main
- host: UF-EV-A9-GWH02
- track: handoff
- status: inferred
- claim: Handoff: branch ssinghal/olmo3-galaxy-main @256d9b8 (local, not pushed). Goal: optimize batch-32 decode, no precision lowering (human decision 2026-10-13). Blocker before perf work: decode determinism gate fails (see entries above). Next: bisect nondeterminism (eager-vs-eager twice; force-argmax vs top-k sampling path; per-layer batch-identity probes across two runs), then Tracy op table via models/demos/olmo_galaxy/tests/test_olmo_decode_perf.py. Tools: ~/ssinghal/olmo3/run_demo.sh <tag> <cfg>, scripts/compare_decode_tokens.py.
- evidence: -
- paths: models/demos/olmo_galaxy

## E261014-0357-b67
- date: 2026-10-14T03:57:48
- commit: d78f786419e
- branch: ssinghal/olmo3-galaxy-main
- host: UF-EV-A9-GWH02
- track: correctness
- status: inferred
- claim: Interpretation (not proven): marginal hardware/ring-matmul behaviour on chip 14 (slot 22) cores (6,6)/(6,7) or their DRAM banks; effect is ~1 ulp on one column so output quality is unaffected but sampled continuations diverge near ties. Next: run same repro on a different Galaxy; try non-ring (DRAM-sharded) w1/w3 matmul; ask HW team if chip 14 is known flaky. Bitwise run-to-run gate cannot pass on this host until then; use per-step logit tolerance + GSM8K/AIME accuracy instead.
- evidence: -
- paths: models/demos/olmo_galaxy

## E261014-0357-a56
- date: 2026-10-14T03:57:57
- commit: b0c4a50e45f
- branch: ssinghal/olmo3-galaxy-main
- host: UF-EV-A9-GWH02
- track: correctness
- status: verified
- claim: Run-to-run decode nondeterminism is ulp-level and localized to ONE chip: replaying one decode step (same token/pos, host argmax) gives different logits in 74/2000 traced replays (3.7%, 64 layers), 9/3000 (3 layers), 1/3000 (1 layer). With per-stage probes (1 layer, eager, 11000 replays): 17/17 first-differing stages are mlp.w1_out_dram/w3_out_dram on mesh slot 22 (row5,col2 = chip 14) only; each hit = ONE output column, all 32 users, ~1 bf16 ulp (1e-3), in ring cores 0/1 (logical (6,6),(6,7); cols 94 x7, 110, 166, 206, 222). Re-running the w1 ring matmul alone on identical inputs reproduces it (only slot 22, ~1/400 reruns). Plain bf8 DRAM matmul on all 32 chips: 0 flips/2000. Weight readback from slot 22: 0 flips/300. Not the prefetcher (use_prefetcher=False), not CCL, not sampling, not KV/prefill (identical). Swapping ring cores breaks the model.
- evidence: /home/ubuntu/ssinghal/olmo3-logs/a4d7a853746-dirty_rep_trace_gate-b32.log
- paths: models/demos/olmo_galaxy, ttnn/cpp/ttnn/operations/transformer/sdpa_decode
- data: {"replay_hit_rate_64l": 0.037, "device_slot": 22, "chip_id": 14}
