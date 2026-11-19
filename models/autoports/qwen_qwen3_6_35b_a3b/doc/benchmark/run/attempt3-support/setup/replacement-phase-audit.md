# Replacement-request phase ordering: source audit

No replacement-specific overlap defect is apparent in the selected source path.
Pending decode outputs are finalized before replacement prefill begins.

Paths below are relative to the tt-metal checkout unless prefixed with `../`.

1. New requests, resumed requests and cached prefill continuations make steady
   decode ineligible:
   `../vllm-tt-plugin/src/vllm_tt_plugin/async_decode.py:520-535,589-598`.
2. Before building the next input, the runner calls
   `wait_for_all_pending_async_steps()` when steady decode is ineligible:
   `../vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py:3210-3220`.
3. The drain calls each pending step's `ensure_finalized()`. Its finalization
   lock also waits for a concurrent finalizer:
   `../vllm-tt-plugin/src/vllm_tt_plugin/async_decode.py:226-244,935-945`.
4. Finalization synchronizes the existing read events and invokes the adapter's
   host-output hook:
   `../vllm-tt-plugin/src/vllm_tt_plugin/async_decode.py:1653-1677`.
   That hook records `decode_done` before returning:
   `models/autoports/qwen_qwen3_6_35b_a3b/tt/generator_vllm.py:342-350`.
5. Only afterward can replacement prefill enter `prefill_start`, at
   `models/autoports/qwen_qwen3_6_35b_a3b/tt/generator_vllm.py:184`.
   Prefill completion is recorded after its synchronous host read, at lines
   210-214 of that file.

Decode intervals may overlap other decode intervals; the collector unions
those intervals. They should close before replacement-prefill intervals begin.
The collector's cross-phase overlap rejection should remain enabled; this source
audit does not justify changing it. See
`models/autoports/qwen_qwen3_6_35b_a3b/tools/collect_roofline.py:108-139`.

The collector assigns intervening host gaps to the next phase and records the
HTTP envelope's leading/trailing time separately. These are existing timing
conventions, not evidence of missing intervals caused by replacement requests.

This conclusion is based on source inspection only. Actual event ordering and
complete request-lifetime/decode-position reconciliation still await the idle
collection after all measured HTTP requests complete. No live phase flush,
tests, ML/device imports, source edits or hardware operations were performed.
