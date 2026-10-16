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

## E261014-0455-120
- date: 2026-10-14T04:55:51
- commit: a37fc8e1287
- branch: ssinghal/olmo3-galaxy-main
- host: UF-EV-A9-GWH02
- track: serving
- status: inferred
- claim: Serving through tt-inference-server (run.py --local-server, vLLM 0.30.0 + vllm-tt-plugin b16250b, spec olmo3_32b_galaxy) on the Galaxy, vllm bench serve random 128/128, 32 concurrent, greedy: TPOT 42.4 ms (23.6 tok/s/user), TTFT 1508 ms, 593 tok/s output, 96/96 ok. Demo batch-32 at a69ec79: 38.3 ms (26.1 tok/s/user). Model-card sampling (T=0.6, top_p=0.95) with server default top_k=32: 42.8 ms (device sampling); with top_k unset and no default: 127 ms (host sampling). 2K/4K prompts x32 concurrent: 0/64 failures (TPOT 51 ms; P99 ITL 55 ms at 4K). Needle retrieval OK at 28.5K and 57K prompt tokens; 5/5 factual checks across 32 concurrent users. Measured on the tree that became a37fc8e (uncommitted at the time); single unpaired runs.
- evidence: /home/ubuntu/ssinghal/olmo3-logs/bench_ttis1.log
- paths: models/demos/olmo_galaxy
- data: {"tpot_ms": 42.4, "demo_tpot_ms": 38.3, "ttft_ms": 1508, "concurrency": 32}

## E261014-0548-7c3
- date: 2026-10-14T05:48:19
- commit: 3512fcd0626
- branch: ssinghal/olmo3-galaxy-main
- host: UF-EV-A9-GWH02
- track: serving
- status: inferred
- claim: All sampling on device (first token via the generator's prefill sampling branch; decode traces keyed by sampling variant, all captured at warmup). Found and fixed: the single baked decode trace silently ignored frequency/presence/repetition penalties (hello-fraction 1.00 with penalties=2.0 on a repetitive prompt; now 0.02-0.03). Through tt-inference-server at 32 concurrent, 128/128: greedy TPOT 40.7 ms (was 42.4), TTFT 1336 ms (was 1508); model-card sampling (T0.6 top_p0.95, default top_k32) 42.6 ms; 2K 48.8 ms; 4K P99 ITL 53 ms; mixed-variant stress 96/96 ok; seeds repeatable and distinct. Demo batch-32 unchanged 38.3 ms. Measured on the tree that became 3512fcd (uncommitted at the time); single unpaired runs.
- evidence: /home/ubuntu/ssinghal/olmo3-logs/bench_x.log
- paths: models/demos/olmo_galaxy
- data: {"tpot_greedy_ms": 40.7, "tpot_sampled_ms": 42.6, "ttft_ms": 1336, "demo_tpot_ms": 38.3}

## 2026-10-14 long-generation / eval check (NOT resolved)
- 32 users x 24000 tokens, sampled, through tt-inference-server: all complete (756K tokens, 194 tok/s aggregate). KV pool (~166K tokens) limits
  concurrent long generations to ~11 at 24K each; the rest queue.
- `--workflow evals --limit-samples-mode smoke-test` (r1_aime24, 3 problems, T0.6/top_p0.95/top_k32): 1/3 correct. One sample (23K tokens) correct,
  one (8K) wrong, one degenerated into a "Wait,\n\n" loop from ~24.5K tokens (no </think>). Model-card AIME24 is 80.6, so this is not yet evidence either way.
- Replaying that 24.6K-token prefix on a fresh server: greedy top-1 at the end of prefill matches an HF CPU bf16 reference (" ("), but
  sampled output (T0.6) is garbage from the first token, and repeated sampled requests on short/medium prompts (2K-16K) degrade request by request.
  => suspect the device-sampling path (prefill-sampling branch / per-slot params / seeds), not attention. Needs a controlled sequence test
  (scratchpad seq.py: sampled x6, greedy, sampled, top_k=0 host reference).
- Server hit "device timeout in fetch queue wait" in prefill of a 24.6K prompt after ~1h40m uptime; a fresh server handled it.
- Host state at end: chip 16 reads 0xffffffff over PCIe; `tt-smi -glx_reset` and `-r` fail. Needs host reboot / power cycle.

## 2026-10-15 accuracy: device-sampled text had occasional repeated words (FIXED), long prompts corrupt (OPEN)
Symptom: tt-inference-server `r1_aime24` smoke test 1/3; device-sampled completions contained doubled words ("a a brief",
"with with history"), 0.5-2.6 per 1000 words, against 0 for host-sampled (`top_k=0`) and 0 in the demo. For every one of
these the host reference gave the repeated token ~0 probability, so they were not legitimate tail samples.
Ruled out (each measured): sampler probabilities (flat-context test vs the host's logprobs, TV 0.003, 0 tokens outside the
nucleus); stale input tokens/positions (per-step device-buffer checks, 0 mismatches); k/top-p/temperature tensors on device
(0 mismatches in 500+ steps); host/device reload mismatches; semaphore reset after prefill sampling; number of decode traces
(1 vs 6: same rate); warmup order; forced input reload after prefill. Demo with the same sampling params and on-device first
token sampling: 0 doubled words in 24.7K words, so it is not the model or the sampler kernel.
Bisect of the on-device-first-token block in `Generator.prefill_forward_text` (OLMO_PS_SKIP, doubled words per 1000 words):
  skip everything 0.00 | switch_mode only 2.68 | seed handling only 0.50 | params only 0.00 | mode+seed 4.5-7.6.
Causes: (1) the block called `switch_mode("decode")` while the prefill tail was still in flight and left the model in decode
mode; ending prefill in decode mode corrupts later decode steps (the host-sampled path then glitched too: 10.7/1000 words
after device-sampled requests had run). (2) `seed_manager.reset_seed` + `get_new_values(empty_slots)` at every request join
re-initialises the unseeded device PRNG for all lanes.
Fix: synchronize before the switch; switch back to prefill mode after sampling (so the next decode re-enters decode mode itself);
re-seed at prefill only when a request carries an explicit seed. Result on the fixed server: 0 doubled words in 42.5K words
(device) and 42.9K words (host); explicit seeds repeat / differ correctly, greedy repeatable, penalties still work (hello
fraction 1.00 -> 0.09), 96/96 mixed-config stress ok, 64 seeded concurrent requests 0 glitches. Scripts: `debug/serving/`.
OPEN: prompts over 4096 tokens use the eager (untraced) prefill. Under vLLM the first request at a padded length is correct
and later ones return garbage from the first token (also with the older generator, so not caused by the sampling work);
`PREFILL PT` logging shows the first call had an ascending page table `[1,2,3,...]` and the second a descending one
`[128,127,126,...]` (vLLM's free-block order). The demo repeats an 8K prefill 3x with a fixed table and is correct each time.
Unresolved: page-table order vs stale cached program (a test with `mesh_device.clear_program_cache()` before each eager
prefill was queued but the host lost chip 16 first). Each device hang/engine crash leaves chip 16 unrecoverable by
`tt-smi -glx_reset`; it needed a power cycle.

## 2026-10-15 (later) long prompts (> 4096 tokens): narrowed, NOT fixed
Under vLLM, prefill longer than 4096 tokens (padded 8192/16384/...) is unreliable. Findings:
- Eager path (default for > 4096): first request at a padded length is correct, later ones are garbage from the first token
  or hang the device. Reproduced offline in the demo: repeating `long-8k-b1` is fine, but allocating a DRAM spacer tensor
  between batches (`OLMO_PERTURB_ALLOC=1`) makes the second 8K prefill hang in an eager `ttnn.all_gather`.
  Not the program cache (`OLMO_NO_PROGRAM_CACHE=1` still hangs; clearing it under live traces hangs decode).
  Not the KV cache contents. The page-table order differs between good and bad calls under vLLM but also changes with
  anything else; it is not proven to matter.
- Tracing 8K / 16K prefill (`OLMO_TRACE_SEQLENS=8192,16384`, trace region 640 MB) fixes the demo repeat + spacer test and four
  repeated 8.2K server requests, but is not a complete fix: after bursts of concurrent traffic only 8/20 (8K) and 7/20 (16K)
  greedy requests matched the majority output, and one traced replay hung the device. 32K / 64K traced do not fit in DRAM
  next to the KV cache (OOM allocating the per-length persistent buffers).
- Prompts up to 4096 tokens (traced) were reliable in every test (2K x10 identical, 3.9K x10, mixed stress, 4K x32).
- Capping prompts via `max_num_batched_tokens` does not work: the plugin bumps it back to max_model_len (no chunked prefill).
- (The per-layer sync experiment was run later and hung the device; see 2026-10-16 below. Removed.)
Opt-in switches left in the code: OLMO_TRACE_SEQLENS, OLMO_LOG_PT, demo OLMO_PERTURB_ALLOC /
OLMO_NO_KV_RESET / OLMO_NO_PROGRAM_CACHE / OLMO_TRACE_REGION / OLMO_REPEAT_BATCHES. Scripts: debug/serving/.

## 2026-10-16 demo regression, accuracy localisation, and the per-layer sync option (REMOVED)
- Demo regression on the final code (HEAD 7891fb822e2 + env-gated debug), no resets, all pass: batch-32 38.33 ms/token (26.1 tok/s/user),
  batch-1 26.5, gate-b32 38.2, long-4k-b32 22.0, 8K 24.4, 16K 23.8, aime24-p1 25.3, 32K 22.5, 64K 20.5 tok/s/user (eager long prefill with the
  program cache on); gate-b32 with on-device first-token sampling 39.1 ms, 24.7K words, 0 doubled words.
- No perf change vs the earlier baseline (38.3 ms/token): this session was correctness only; the decode optimisation pass has not started.
- One aime24-p1 run (seed 42) degraded to junk tokens from ~4.3K generated tokens. Localisation: scoring its tokens with the HF CPU model
  put the collapse at position ~4096 (the sliding-window length). BUT: teacher-forced decode over 3600..4600 (true coherent text) matches HF
  across the boundary (logit PCC ~0.99, top-1 agreement 93-97%); the paged SDPA decode op with window 4096 passes against torch at OLMo's
  settings (16 configs incl. single active user); greedy / unseeded / seeds 7 and 123 / a re-run of seed 42 are all clean to 6200 tokens, and the
  re-run of seed 42 diverges from the first run at token 237 (run-to-run nondeterminism). So the junk run was a one-off fault, not a
  reproducible window or sampler bug. Left open; consistent with the machine's other flakiness (chip 14 bit flips, chips falling off PCIe).
- `OLMO_EAGER_PREFILL_LAYER_SYNC=1` (sync the device between layers of an eager prefill) HUNG THE DEVICE: both server starts that used it
  (21:46 and 00:24) hit "device timeout in fetch queue wait" within seconds of the first requests and lost all 32 chips (tt-triage then
  reports PcieHangError on every device). Servers without it were fine. The option is removed; do not re-add it.

## 2026-10-16 tt-inference-server evals (ci-nightly) on the fixed code: AIME24 6/15, collapsed outputs; server hung in MATH-500
- r1_aime24 (15 of 30 problems, T0.6 / top_p 0.95 / top_k 32 default, max 32768 tokens, up to 15-32 concurrent): 6/15 = 40% vs 80.6 published.
  Of the 9 wrong: 7 are collapsed generations (random multi-script junk or "Thus, Thus," loops, no answer, 85-128K chars ~ the token cap),
  2 are wrong arithmetic. Of the 8 outputs that closed </think>, 6 are correct. The 38-minute stage finished without any device error.
- MATH-500 stage then hit "Timeout detected" at 01:55:26 (device hang) at 92.7% KV usage with 18 running / 13 waiting; generation stayed at 0 tok/s
  until the stalled server/eval were killed. No experiment options were set. Devices need a power cycle.
- Hypothesis (untested): sustained heavy concurrent decode trips di/dt limits on Wormhole -> corrupted compute (junk after thousands of tokens) and
  eventually a hang; single-user demo runs to 6.2K tokens are clean. TT_MM_THROTTLE_PERF (matmul throttle) is the planned test: demo batch-32 speed
  with it, then the same AIME24 eval with it set globally. Eager prefill > 4096 now runs with TT_MM_THROTTLE_PERF=3 (see generator.py).
