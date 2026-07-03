# AutoFix Report

## Starting Evidence

- vLLM traced greedy decode could not use the `Sampling1D` force-argmax path because trace capture hit `TT_FATAL: Writes are not supported during trace capture`.
- Routing traced greedy through the trace-safe TP4 sampler avoided the trace-capture fatal, but qualitative output initially degenerated.
- Direct HF no-thinking greedy controls and direct full-model host-argmax checks produced clean outputs, so the issue was isolated to TP4 on-device sampling.

## Hypotheses Tested

- `Sampling1D` force-argmax can be used in traced token feedback.
  - Refuted. Its `ttnn.argmax(..., output_tensor=...)` path is not trace-capture safe for this adapter.
  - Kept fix: vLLM traced decode sets `force_argmax=False` and uses the TP4 greedy sampler for trace-safe token feedback.

- Generic top-k/greedy fallback can replace the TP4 greedy sampler.
  - Refuted for the serving path. The generic path is not the full-model generator's canonical split-sampling path and did not satisfy the no-fallback vLLM stage contract.

- TP4 pair reduction was reading gathered local-winner rows with the wrong stride.
  - Verified. The EOS probe showed local winners contained shard 3 EOS token `151645`, while global TP4 sampling selected shard 0 token `25521`.
  - Kept fix: `qwen_argmax_pair_reduce_kernel.cpp` now uses the gathered pairs tensor's aligned page size for row stride.
  - Verification: the same EOS probe reduced to EOS token `151645`; final vLLM qualitative outputs are coherent and the degenerate-output checker reports no findings.

## Final Status

- Fixed the traced force-argmax crash by keeping vLLM traced greedy decode on the TP4 trace-safe sampler.
- Fixed TP4 greedy degeneration by correcting gathered-pair row addressing in the pair reducer.
- Added an idle trace-release control in the TT vLLM runner. The final non-aligned rerun still logs TT-Metal's active-trace allocation warning once after the final decode signpost, but `readiness_vllm/trace_warning_audit.log` proves no decode or trace execution follows it.
- Stage review rereview returned `clean-pass`.

## Verification

- `pytest -q models/autoports/qwen_qwen3_4b/tests/test_full_model_contract.py --tb=short`
  - `20 passed`, `2 warnings`
- `run_vllm_server --stages serve,sampling,qualitative,benchmark --sampling-profile full`
  - Sampling: `72 passed`, `1 skipped`
  - Qualitative: coherent final outputs, no degenerate-output findings
  - Primary single-user benchmark: 128/128/1, TPOT-derived `75.4 t/s/u`
  - CI serving burst: 100/100/32, secondary TPOT-derived `30.2 t/s/u`
- Non-aligned prompt check:
  - `random_input_len=37`, `random_output_len=8`, completed `1/1`
- Cleanup:
  - Wrapper-launched vLLM servers terminated cleanly.
  - Final process audit found no live vLLM/EngineCore workers holding devices.
