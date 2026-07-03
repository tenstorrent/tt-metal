---
name: kimi-prefill-trace
description: "ttnn trace capture of the Kimi/DeepSeek prefill transformer.forward() — how, the one blocker + fix, measured trace memory"
metadata:
  node_type: memory
  type: project
  originSessionId: d8256768-f775-435c-b45d-d00da80ed2c8
---

Added ttnn-trace support to the chunked-prefill `transformer.forward()` to collapse op2op
(host-dispatch) gaps. Branch context: ppoppovic/pipeline_e2e_time (work cherry-picked onto
ppopovic/op2op_gap_tests off main).

**How it's wired:**
- `test_prefill_transformer_chunked.py::run_chunked_transformer_no_pcc` builds the transformer with
  `kv_only_last_layer=True` (the EXISTING mechanism that strips final RMSNorm + LM head + sampling, so
  `forward()` returns after the layers with NO host readback — the only way the forward is device-only
  and trace-capturable). Do NOT invent a new cutoff; kv_only_last_layer is it. New `use_trace` param +
  `use_trace=[False,True]` parametrize (ids `notrace`/`trace`). Trace path: persistent chunk-0 input →
  warm/compile → begin/end_trace_capture → measure → execute_trace replay loop → release_trace. PINNED
  to chunk 0 (multi-chunk deferred).
- `tt_deepseek_prefill_pipeline.py`: `use_trace` config flag (asserts kv_only_last_layer), `_capture_trace()`
  in compile(), execute_trace in prefill(), release()/__del__.
- `device_params` needs `trace_region_size` > 0 (set 256MB; conftest.py:587 keeps it if present).
- Trace memory readout: `ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE)` →
  `total_bytes_allocated_per_bank * num_banks`.

**THE ONE TRACE BLOCKER (and fix):** MoE swaps a sub-device manager INSIDE forward
(`tt_moe.py:502/527` load/clear, gated by `overlap_shared_expert_with_dispatch`, to overlap shared
expert with dispatch). That resets worker state → `begin_trace_capture` aborts with
"Cannot reset worker state during trace capture" / "Reads are not supported during trace capture".
FIRST fix (SUPERSEDED — do not rely on this): threaded `overlap_shared_expert_with_dispatch` (default
True) through TtPrefillTransformer→TtPrefillBlock(`_build_moe`)→TtMoe and passed `not use_trace` to
DISABLE overlap under trace. This was replaced by the SEGMENTED-TRACE approach below (overlap now stays
ON; controller handles the swap). The plumbing param still exists (default True) but is no longer set to
disable. A full forward-path scan confirmed the sub-device swap is the ONLY trace-hostile op (all other
to_torch/sync/ReadDeviceProfiler are gated off by kv_only / return_intermediates=False / debug-off /
DEVICE_FP32 gate). Note: user also commented out the sync+on_layer_complete in MLA `_forward_kv_only`
(mla.py:1019-1020) — leave commented.

**Measured (8x4 BH, Kimi K2.6, 1 chunk = 5120 tok, warm), 2026-06-24:**
- L10: trace = **7.06 MB**; replay **0.170 s/iter** vs notrace ~0.21-0.23 s/iter (~20% faster).
- L61: trace = **49.25 MB**; replay **1.273 s/iter**. (~0.8 MB/layer; 256MB region is ample.)
  (These numbers are the overlap-OFF single-trace variant.)

**SEGMENTED TRACE w/ shared-expert/dispatch OVERLAP ON (the real goal):** the MoE swaps a 2-sub-device
manager around shared-expert∥dispatch (`tt_moe.py:502/527`), which can't be inside one trace. New
`utils/sub_device_trace.py::SubDeviceTraceController` runs forward ONCE during capture and CHOPS it into
trace segments at each load/clear: end_trace_capture → real host load/clear → begin_trace_capture. MoE
routes its load/clear through `self._trace_controller` (set via
transformer/block `.set_trace_controller()`; no forward() signature changes). Replay walks recorded
program `[trace, load, trace, clear, trace, ...]`. Boundary tensors persist as forward locals across the
splits (trace replay reproduces them at same addresses). Validated bit-exact in a single-chip PoC
(`scratchpad/poc_segmented_trace.py`) BEFORE the real refactor. Segments = 1 + 2*(num MoE-overlap layers).
- L10: **17 segments, 7.31 MB**, replay 0.172 s/iter (vs overlap-off 7.06MB/0.170s — ~same).
- L61: **119 segments, 51.14 MB**, replay 1.288 s/iter (vs 49.25MB/1.273s — ~same).
- So enabling overlap under trace adds negligible mem/time but keeps the optimization. Overlap is now
  ON in BOTH no_trace and with_trace (test passes `overlap_shared_expert_with_dispatch=True`).
GOTCHA: a 2-sub-device manager makes a FULL-GRID op fail ("Programs must be executed on a single
sub-device") — the overlap is fine because shared/dispatch are core-confined to one sub-device each.
GOTCHA: parametrize ids are now `no_trace`/`with_trace` (NOT notrace/trace); perf driver selectors use
`no_trace`.

**TRACE CORRECTNESS TEST:** `test_kimi_prefill_transformer_chunked_trace_kv_pcc` (params L10/L61) captures
the segmented trace on chunk 0, replays once, then PCCs the device KV cache vs golden kv_post_transform
(KV-only — kv_only_last_layer has no decoder-output/logits). `verify_kv_cache_pcc=True` arg to
run_chunked_transformer_no_pcc; uses EXACT golden tokens (no vocab modulo); `_record_kv_cache_pcc` gained
`assert_threshold`. KV_CACHE_PCC_THRESHOLD=0.96. L10 result (2026-06-24): min KV PCC=0.9939 (all layers
>0.99, nope 0.994-0.9999 / pe ~0.9997) -> PASS, trace replay writes correct KV. Needs
TT_KIMI_PREFILL_HOST_REF_CACHE=/mnt/models/kimi-prefill-cache/golden.

**FIXED teardown segfault:** `close_mesh_device` segfaulted at pytest teardown (after PASSED) on
overlap-on runs because the MoE created a sub-device manager per layer (59 at L61) via
create_sub_device_manager and NEVER removed them → dangling managers crashed close. Fix:
`TtMoe.release_sub_device_manager()` (remove_sub_device_manager, idempotent) + block/transformer
`release_sub_device_managers()` (transformer clears loaded first); called at end of the no_pcc worker and
in pipeline.release(). L61 with_trace now exits clean (0 segfaults, 0.31s teardown, 1 passed). (The
realtime-profiler "Skipped zones" warnings are harmless and were NOT the cause.)

**FILES TOUCHED (all in models/demos/deepseek_v3_d_p/):**
- `utils/sub_device_trace.py` — NEW, SubDeviceTraceController (begin_capture/end_capture/sub_device_load/
  sub_device_clear/replay/release/trace_bytes/num_segments).
- `tt/moe/tt_moe.py` — `_trace_controller` attr; load/clear route through it; `set_trace_controller()`;
  `release_sub_device_manager()`.
- `tt/tt_prefill_block.py` — `_build_moe` takes `overlap_shared_expert_with_dispatch`; block stores it;
  `set_trace_controller()` + `release_sub_device_managers()` passthroughs.
- `tt/tt_prefill_transformer.py` — `overlap_shared_expert_with_dispatch` ctor param; `set_trace_controller()`
  + `release_sub_device_managers()` (walks layers).
- `tt/tt_deepseek_prefill_pipeline.py` — `use_trace` config; `_capture_trace()` uses controller; prefill()
  replays; release() removes managers.
- `tests/test_prefill_transformer_chunked.py` — no_pcc worker `use_trace`/`verify_kv_cache_pcc` params,
  segmented capture/replay, KV PCC; `test_kimi_prefill_transformer_chunked_trace_kv_pcc` (L10/L61);
  `_record_kv_cache_pcc(assert_threshold=)`; KV_CACHE_PCC_THRESHOLD=0.96.
- `tests/perf/test_prefill_chunked_perf.py` — selectors use `no_trace` (driver hardcoded L5/no_trace;
  doesn't do 61 or trace — change _SELECT_DEVICE/_SELECT_E2E to retarget).

**RUN CMDS** (env: KIMI_K2_6_HF_MODEL=/data/nbabin/Kimi-K2_6-dequantized,
TT_KIMI_PREFILL_TTNN_CACHE=/mnt/models/Kimi-K2_6-Cache/Kimi-K2_6-Cache-prefill, TT_DS_PREFILL_TTNN_CACHE
same; KV-PCC also needs TT_KIMI_PREFILL_HOST_REF_CACHE=/mnt/models/kimi-prefill-cache/golden +
TT_DS_PREFILL_HOST_REF_CACHE same). PYTHONPATH=. python_env/bin/python -m pytest <...> -s :
- no_pcc perf (A/B): `...::test_kimi_prefill_transformer_chunked_no_pcc -k "L61 and ten_iters"` (both
  no_trace+with_trace). Isolate: add `and with_trace` (does NOT match no_trace) or `and no_trace`.
- KV-cache PCC: `...::test_kimi_prefill_transformer_chunked_trace_kv_pcc -k "L61"` (or L10).
- device-vs-e2e loss driver: `tests/perf/test_prefill_chunked_perf.py` (L5/no_trace only).
GOTCHA: ids are `no_trace`/`with_trace` (substring `trace` matches BOTH; `with_trace`/`no_trace` are
unambiguous). GOTCHA: a trace TT_FATAL leaves the pytest proc HUNG holding `CHIP_IN_USE_0_PCIe` — `pkill
-9 -f chunked_no_pcc` (robust mutex recovers) before re-running.

**STATE / TODO:** All working + verified on 8x4 BH. Deferred: multi-chunk tracing (pinned chunk 0 — needs
copy_host_to_device into persistent input + per-actual_start traces; LayerAck on_layer_complete migration
callback incompatible w/ trace — host sync). Possible opt: merge each layer's post-seg with next layer's
pre-seg → ~N+1 segments instead of 2N+1 (overlap added ~0 replay speedup at current 2N+1; segmentation host
overhead offsets overlap gain). 2cq not done (single CQ).

Related: [[kimi-h2d-service-dispatch-tax]] (op2op tax this trace eliminates), [[kimi-perlayer-op2op-harness]],
[[kimi-prefill-env-vars]].
