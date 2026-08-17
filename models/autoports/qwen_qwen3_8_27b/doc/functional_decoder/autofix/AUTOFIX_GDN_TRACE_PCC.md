# AutoFix: GDN traced-decode PCC

## Starting Evidence

- Starting report: `AUTOFIX_GDN_STATE.md`.
- Original command: `pytest -q models/autoports/qwen_qwen3_8_27b/tests/test_functional_decoder.py -k real_weights_deltanet -s`.
- Failure after the verified DRAM-state placement fix: traced replay PCC `0.9740499002269837`, below `0.995`.

## Precision and state-address ledger

- TT input/state dtype: BF16.
- GDN projection policy: inherited BFP8 weights, LoFi math, FP32 destination accumulation; unchanged by the fix.
- Persistent recurrent and fused-convolution state: BF16 tile tensors in DRAM.
- Stable DRAM addresses across in-place warmup, capture, and replay: recurrent `58784384`, fused convolution `57660032`.
- The autoport wrapper did not replace either state buffer in in-place mode.

## Hypothesis experiments

### Trace or wrapper replaces persistent state buffers

- Experiment: record recurrent/fused-convolution buffer addresses before and after in-place warmup, capture, and replay; measure each executed phase against its matching sequential Torch step.
- Command/log: original command above; `/tmp/autofix_gdn_trace_phase.log`.
- Result: addresses were stable. Eager PCC `0.9461134`, warmup `0.9657135`, replay `0.9740499`. Trace capture produced no valid output because capture records commands without executing them.
- Verdict: refuted. The error existed before trace replay and was not an address-lifecycle bug.

### BFP8 projection precision causes the failure

- Experiment: replace only the GDN projection and mega-fused projection tensors with BF16 at setup time.
- Log: `/tmp/autofix_gdn_trace_bf16_gdn.log`.
- Result: eager `0.9494786`, warmup `0.9662813`, replay `0.9757122`; still below the bar.
- Verdict: refuted. The experiment was reverted; the inherited precision policy remains intact.

### Torch oracle omits prefill convolution history

- Evidence: `causal_conv1d_forward` returns a new convolution state after prefill only when a state tensor is supplied. The test passed `None` on its first call, so its subsequent decode reference used zero/absent history while TTNN correctly retained the last three prefill projection tokens.
- Prediction: initialize zero Q/K/V convolution-state buffers before Torch prefill; prefill output remains equivalent, and sequential decode PCC rises above `0.995`.
- Fix: initialize real-shape zero convolution states in `_deltanet_reference` when no cache is supplied. This is an oracle correction, not a runtime fallback or model precision change.
- Verification command/log: original command above; `/tmp/autofix_gdn_trace_cache_fix.log`.
- Result: eager PCC `0.9988066013`, in-place warmup PCC `0.9995071971`, traced replay PCC `0.9993015776`; test passed. The replay matched Torch reference step three because trace capture itself does not execute commands.
- Verdict: verified and fixed.

## Final Status

- Fixed. The real-weight GDN traced-decode test passes the `0.995` acceptance bar.
- Verified DRAM placement is preserved, and state addresses remain stable through warmup/capture/replay.
- Device-facing commands were serialized. The pytest fixture closed device 0 after every run. `tt-smi` was unavailable in this shell (`No such file or directory`).
