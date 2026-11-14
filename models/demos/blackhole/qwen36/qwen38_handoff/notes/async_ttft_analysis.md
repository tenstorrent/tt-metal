# C5 (async ON) vs C5off TTFT regression under concurrency

Read-only analysis. Files: S=/home/ttuser/atupe/qwen38_work/serve. Helper: ttft_analysis/waves.py, output waves.txt.
INFERRED = not directly proven by a log line or code line.

## Summary
The extra TTFT is not in arrival->prefill or in prefill duration (both identical). It is a PUBLICATION delay: with
async scheduling the engine does not publish a prefill step's first token until the NEXT step's execute_model() call has
returned (batch queue depth 2). TT executes prefill synchronously inside execute_model, and the first decode step after
a prefill carries a ~320 ms transition submit. So:
- the 7 requests of the "7-user" prefill wait +~320 ms (the D1 submit),
- the 1 request prefilled alone first waits +~860 ms (the whole 7-user prefill that follows it).
Model: dTTFT_mean = (1/8)*860 + (7/8)*320 = 387 ms (measured 395). R4: (1/32)*3730 + (31/32)*324 = 431 ms (measured 461).
p50 delta: R3 319 ms (1240-921), R4 322 ms (3963-3641). Both equal the ~320 ms D1 submit.

## Q1 Timeline (R3, c8, n32). Logs have only prefill start/finish lines (ms), no request-added/arrival or per-step lines.
R3 = 4 waves of 8 requests (lockstep, all finish together, engine idle between waves: "EngineCore waiting for work" at
C5 server.log:2131). Each wave is two prefill steps: P1 (1 user) then P7 (7 users, queued while P1 ran).
Warmup waves (2 requests) precede R3 and are excluded.

| wave start (P1) | log | P1 prefill ms | P7 prefill ms | P7 finished -> first decode-side log (ms) | wave period s |
|---|---|---|---|---|---|
| C5 30.119 | C5/server.log:2133-2136 | 109 | 860 | 318 (sampling-trace precompile, line 2138) | 7.06 |
| C5 37.179 | | 112 | 801 | n/a, no decode-side log (trace cached) | 7.00 |
| C5 44.181 | | 107 | 803 | n/a | 7.01 |
| C5 51.188 | | 107 | 806 | n/a | |
| C5off 38.116 | C5off/server.log:2124-2130 | 107 | 848 | 324 (conv-format sync #7) | 7.25 |
| C5off 45.366 | | 109 | 801 | 322 | 7.24 |
| C5off 52.607 | | 107 | 804 | 323 | 7.25 |
| C5off 59.859 | | 108 | 805 | 323 | |

R4 (c32): C5 P1 ~110 ms, P31 3734/3518 ms, then 324 ms to first decode log (C5 server.log:2064 area, "GDN conv-format sync #5" 28:14.012);
C5off P31 3660/3517 ms, 331 ms.

Per-request chain (per wave):
- arrival -> prefill start: ~0 in both (engine idle; P1 starts immediately; the other 7 arrive during P1 and form P7).
  Per-request arrival is not logged, so this is INFERRED from the idle-engine log line and from prefill batches being identical.
- prefill duration: identical in both logs (P1 ~108 ms, P7 ~800-860 ms; N counts match).
- prefill end -> first token emitted:
  - C5off (sync): token is published right after the step (TTFT p50 921 ms ~= P1 107 + P7 ~800, i.e. ~0 after prefill end).
  - C5 (async): published only after D1 (the decode step after the prefill) has been submitted: +~320 ms for P7 users,
    +P7 duration for the P1 user. TTFT p50 1240 = 921 + 319.
- The extra ~300 ms per request is in "prefill end -> first token emitted", not before it.
c1 (R1/R2/R5): width 1, post-prefill gap is 5-6 ms in both logs (waves.txt, "1 -> conv-format 6 ms"), hence TTFT within +-2.5%.

## Q2 Code path
- TTScheduler is prefill-first: scheduler.py:555-575 (_has_pending_prefill), :641-674 (if pending prefill -> prefill step,
  except decode_interleave after 2 consecutive prefill steps). A new arrival is therefore scheduled as a prefill on the very
  next schedule() call, even with decode outstanding; there is no policy that keeps issuing decode while placeholders are
  outstanding, and none that waits for a drain before choosing prefill. (SCHEDULING.md "Local scheduler policy".)
  throttle_prefills is accepted and ignored (scheduler.py:~610 NOTE).
- Drain happens in the runner, not the scheduler: model_runner.py:2480-2483 -> async_decode.py:862-890 must_drain_pending_async_steps
  returns True when not steady_decode_candidate (a prefill step) -> wait_for_all_pending_async_steps. Cost is at most one
  outstanding decode step (~45 ms), paid only when a request arrives while decode is running. In R3 waves the engine is idle
  at arrival, so this is not part of the R3 gap (INFERRED to matter in mixed/streaming load).
- Prefill is synchronous: model_runner.py:2500-2506 (non-decode path runs _forward_with_model_input in execute_model, sampling in
  sample_tokens); uniproc executor runs it inline (uniproc_executor.py collective_rpc non_block=True runs run_method inline).
  Docs: SCHEDULING.md "Even when a call crosses an async-looking executor boundary, TT prefill still behaves like a synchronous step".
- The first token is emitted late: vllm core.py:617-693 step_with_batch_queue. After enqueueing step N, if len(batch_queue) <
  batch_queue_size (2) and scheduler.has_requests() it returns None (core.py:674-679) and next iteration schedules and executes
  step N+1 BEFORE popping/consuming N (core.py:688-703, update_from_output). Docstring core.py:624-626: "fulfilling the batch queue
  has a higher priority than getting model outputs". AsyncScheduler adds output placeholders so N+1 can be scheduled
  (async_scheduler.py:_update_after_schedule). So yes: P's first token waits for one later step's submit to finish.
  The plugin doc says the same for decode tokens: SCHEDULING.md "a step's sampled token is published once the next step has
  been submitted ... that call runs a whole chunk synchronously, so that token is published a chunk later".
- After P1 the next step is P7 (waiting non-empty) -> P1's token waits for all of P7 (~860 ms). After P7 the next step is D1
  (decode_interleave also forces a decode after 2 prefill steps, scheduler.py:641) -> P7's tokens wait for D1's execute_model.

## Q3 Is it a fixed cost per arrival?
It is a fixed time per prefill event, not a count of decode steps: ~320 ms = D1 submit, constant for N=3, 7, 31 (317-331 ms in
both logs) and 5 ms for width 1. At TPOT 45.6 ms that is ~7 decode-step equivalents (INFERRED equivalence only; it is host/device
transition work, not decode steps). The P1 user additionally pays the full following P7 prefill (~860 ms ~ 19 steps).
Check vs R3: predicted mean +387 ms vs measured +395; p50 +319 vs +319; p99 +376 vs +376 (1340-964).
What the 320 ms consists of is INFERRED: it exists identically in C5off between prefill end and the first decode-side log, and
is absent at width 1. The decode-forward preamble (qwen36_vllm.py:450-469) does, in order, _remap_gdn_slots(slot_remap) (eager
gather = 32 slice + concat + copy per buffer, over 36 GDN layers: gdn/tp.py:1620-1645) and prepare_gdn_decode_width (conv-format
sync, ~82 ms per width class change, logged at model.py:2795 and visible twice per wave in C5off, e.g. C5off server.log:2129-2130).
Slot remap is requested by the plugin whenever row->slot differs (model_runner.py:1166-1220). No log proves a non-identity remap
per wave in C5; the 320 ms there is deduced from the TTFT delta. First wave in C5 also pays sampling-trace capture (line 2138).

## Q4 Knobs (ranked by expected impact, TPOT gain preserved)
1. Publish a prefill step's tokens before submitting the next step (engine/plugin level). Removes both the +320 ms and the
   +860 ms terms: TTFT -> ~C5off levels (R3 mean ~830, p50 ~920; R4 ~3600). TPOT unaffected (decode steps still overlap).
   How: a vLLM override (the plugin already carries vllm overrides, docs/vllm-overrides.txt): in step_with_batch_queue
   (core.py:674) do not return early when the step just enqueued was a prefill step; fall through to the pop at :688.
   Plugin-only variant (INFERRED, untested): TTScheduler.has_requests() returns False once, right after scheduling a prefill step,
   so the check at core.py:674 fails and the engine pops/consumes the (already complete) prefill future first.
2. Make the D1 transition submit cheap (model side, qwen36_vllm.py / model.py / gdn/tp.py). Expected: P7 users lose up to ~320 ms
   (roughly -25% of R3 TTFT); also shortens C5off. Options: (a) one batched gather per layer, or move only the slots in
   _pending_state_slot_moves (model_runner.py:1204-1213) instead of rebuilding all 32 slices (gdn/tp.py:1637-1642); (b) run the
   remap/prepare_gdn_decode_width inside the prefill call, after the last write_slot (qwen36_vllm.py:252-293), so it is part of
   the already-sync prefill and the first decode submit is a plain replay; (c) make the plugin keep slot==row so remap is identity
   (_alloc_prefill_state_slots, model_runner.py:1125-1160); (d) skip the conv-format sync when moving between width classes
   by prefilling directly in the format of the decode width (prepare_gdn_decode_width, model.py:2773-2796).
   Measure first: time _remap_gdn_slots and prepare_gdn_decode_width separately; there is no log for it today.
3. Run with async scheduling off but resident decode on: `--no-async-scheduling`. Expected: TTFT back to ~C5off (the
   publication lag disappears), TPOT between C5off and C5 (loses the host/device overlap; resident decode itself is kept) -
   split is UNMEASURED. Platform only disables async when the model lacks supports_async_decode (platform.py:2166-2176); with it
   declared, the plugin still sends the four reload commands under sync scheduling (async_decode.py:1236-1244, 1250-1259) and
   steady_decode_base_enabled just returns False (async_decode.py:503) - resident token feedback with sync scheduling is INFERRED to work.
   Caveat: C5off is not "resident + sync": it is QWEN36_SERVE_DEVICE_DECODE=0 (legacy decode; the platform warning at C5off/server.log:82
   shows supports_async_decode False; qwen36_vllm.py:58-77), so the C5 vs C5off TPOT/TTFT gap mixes two changes.
4. decode_interleave_* (scheduler.py:641, SCHEDULING.md "Decode interleave"): no help. P1 is followed by P7 and P7 by D1 regardless;
   decode_interleave_enabled=false only removes the forced decode after 2 prefill steps, and the next step is still D1.
5. Batching P1+P7 into one P8 step (e.g. a short admission window) would not reduce the mean by itself: total prefill is the same,
   but P1's +860 ms term would become +320 ms (mean -~70 ms). Minor.

Flag syntax: vLLM 0.26 accepts `--no-async-scheduling`. arg_utils.py:1486 registers `--async-scheduling` with
scheduler_kwargs; bool-typed fields get argparse.BooleanOptionalAction (arg_utils.py:340-347) which defines both
`--async-scheduling` and `--no-async-scheduling`. `--async-scheduling false` is NOT valid (BooleanOptionalAction takes no value).
With tt-inference-server, vllm-override bool False emits nothing (run_vllm_api_server.py:1074-1084 _append_vllm_arg), so the bare
token `--no-async-scheduling` must reach the vllm serve command (INFERRED, per C5prep/c5_plan.md).

## Q5 Inherent or TT-specific?
Both; the mechanism is upstream, the magnitude is TT-specific.
- Upstream, by design: core.py:624-626 "fulfilling the batch queue has a higher priority than getting model outputs"; core.py:708-710
  NOTE: handling deferred tasks later "slightly favors TTFT over TPOT/throughput" (acknowledges the trade). With the queue depth 2,
  any step's output is consumed only after the next step is submitted.
- On GPU the next execute_model(non_block) returns in about a millisecond, so the lag is invisible. On TT (a) prefill runs synchronously
  inside execute_model, so the previous prefill's token waits a whole prefill (the P1 user, +860 ms, R4 +3.7 s); (b) the first decode
  submit after a prefill does a transition (reload, remap, width sync) of ~320 ms in our model, so the prefill's token waits for it.
  Plugin docs already describe (a) (SCHEDULING.md decode-interleave section). (b) is our model's cost (qwen36_vllm.py:450-469, gdn/tp.py:1620).
- Neither upstream docs nor comments in this install quantify a TTFT regression; the quotes above are the only statements found.
