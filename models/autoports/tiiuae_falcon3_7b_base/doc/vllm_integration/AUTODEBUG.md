# AutoDebug: Falcon3 vLLM sampling smoke failure
## Scope and evidence

This is an inspection-only diagnosis of the current Falcon3 adapter, the shared
vLLM TT plugin, `readiness_vllm/server.log`, and
`readiness_vllm/sampling_tests.log`. No hardware was run and no implementation
file was edited.

The required `.agents/scripts/autodebug.sh --agent codex ...` runner did launch
the requested fresh `gpt-5.5`/`xhigh` session. That isolated session and each of
its three explorers could not execute repository reads because its command
sandbox failed before execution with a missing `bwrap` launcher. The stalled
run was stopped, and the findings below were then derived in a fresh manual
read-only pass. Runtime consequences beyond the existing logs remain to be
confirmed on hardware.

## Headline findings

### 1. The logged greedy `top_k=131072` crash is real, but current edits already intercept it

The input batch normalizes any vLLM `top_k <= 0` (including greedy
`temperature=0, top_k=0`) to `vocab_size`, which is 131072 here
(`../vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/input_batch.py:337-340`).
The captured server run then fails in `Falcon3Generator.set_sampling_params`
because 131072 is outside its device sampler's supported `[1, 32]` range
(`readiness_vllm/server.log:129,172-175`). This fully explains why all four
sampling tests immediately received dead/empty responses; their individual
failures are downstream fallout, not four independent reproductions
(`readiness_vllm/sampling_tests.log`).

However, the current adapter now converts every zero-temperature slot to
`top_k=1, top_p=0` before calling the generator
(`tt/generator_vllm.py:92-98`), and the generator also normalizes the original
`temperature=0, top_k=0` representation (`tt/generator.py:447-463`). Thus the
specific logged exception is stale with respect to the current working tree.
The next server run should verify this fix; another code change for this exact
exception is not currently justified.

### 2. The next deterministic bug is the missing explicit host-sampling compatibility path

The shared plugin deliberately sets `perform_device_sampling=False` for
host-only sampling features (min-p, bad words, logit bias, allowed token IDs,
min tokens, custom logits processors, structured output) and for logprobs on a
four-device rank
(`../vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py:2504-2550`).
When false, prefill and decode omit the model `sampling_params` keyword
(`model_runner.py:2570-2608` and
`async_decode.py:560-604`). `sampling_params=None` is therefore a mode signal,
not a request for default greedy device sampling.

The Falcon3 adapter currently interprets `None` as greedy parameters and
unconditionally invokes the canonical device sampler in both prefill and
decode (`tt/generator_vllm.py:89-91,119-127,149-172`). Consequently the plugin's
host branch receives token IDs where it requires full logits. In merged-lane
extraction the contract is explicit: device mode consumes sampled tokens, while
host mode passes logits to `host_sampler`; prefill logits are expected as
`[scheduled, 1, vocab]` (or `[scheduled, vocab]`) and decode logits as
`[capacity, vocab]`
(`../vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/input_batch.py:1290-1367`).

This is guaranteed to break the requested min-p/logprobs compatibility after
the greedy crash is removed. It also violates the intended boundary by silently
using a device token-out path after the runner selected host sampling.

The minimal explicit compatibility contract is:

- `sampling_params is not None`: keep the existing on-device canonical split
  sampling and traced token-feedback path used for serving performance.
- `sampling_params is None`: return full host logits without sampling. Prefill
  should delegate to `Falcon3Generator.prefill_forward(...,
  sampling_mode="host")`, which already returns CPU logits shaped
  `[active_batch, 1, vocab]` (`tt/generator.py:255-341`). Decode should delegate
  to `Falcon3Generator.decode_forward(..., sampling_mode="host",
  enable_trace=False)`, which returns CPU logits shaped
  `[active_batch, vocab]` (`tt/generator.py:719-790`). This mode is optional and
  explicit; it must not replace the measured device/traced path.

### 3. Host decode currently has two additional boundary failures

First, the adapter rejects `enable_trace=False` and forces device trace mode
(`tt/generator_vllm.py:145-146,170-171`). The low-level generator intentionally
supports trace replay only with device sampling; host decode requires concrete
tokens/current positions/page table and eager execution
(`tt/generator.py:744-790`). The adapter must ignore/override the runner's trace
request only in the explicit host-compat branch and preserve all concrete host
inputs there. It must not apply its steady-state stale-token logic to host mode.

Second, async decode always calls the adapter's `read_decode_output` when that
method exists (`../vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/async_decode.py:629-635`).
The current method calls `tt_out.cpu(blocking=False)` unconditionally
(`tt/generator_vllm.py:175-179`), but host compatibility returns a PyTorch
tensor and `torch.Tensor.cpu()` does not accept TTNN's `blocking` keyword. For
an already-host tensor, the method should return the tensor unchanged with an
empty event list. The TTNN token-output branch should retain its current
nonblocking copy/event behavior.

`process_decode_output_host` also explicitly raises for `is_tokens=False`
(`tt/generator_vllm.py:181-186`). Normally already-host PyTorch logits bypass
that callback (`async_decode.py:670-680`), so correcting
`read_decode_output` is sufficient for the intended host result. If a future
implementation returns TTNN logits, this callback would also need a real
full-logits gather; the existing low-level host mode avoids that ambiguity by
already gathering to PyTorch.

## Other likely issues to verify after host compatibility

1. `reset_batch` and `slot_remap` are only forwarded by the plugin when device
   sampling is active (`async_decode.py:584-604`). This is acceptable for an
   eager host branch only if that branch never uses persistent traced slot
   state. The adapter must base its steady/stale-input behavior on explicit
   device mode, not merely its default `reset_batch=True` value.

2. The device sampler supports `top_k <= 32`, but unrestricted stochastic vLLM
   requests are normalized to `top_k=vocab_size`. After greedy works, a request
   such as `temperature=1, top_k=0` will still reach 131072 and be rejected.
   This should not be silently converted to top-32 because that changes the
   distribution. Either shared policy must route unrestricted sampling to the
   explicit host compatibility path, or the server's final supported sampling
   profile must reject it clearly. The performance profile must remain within
   the supported on-device contract.

3. Device logprobs on TP4 are intentionally disabled by shared policy, so the
   all-vocabulary logprobs test necessarily exercises full-logits host mode.
   The test may be rejected/skipped by the configured max-logprobs cap, but if
   admitted it cannot be satisfied by the current top-32 device-logprobs helper.

4. `server.log` also records a chat-template warmup failure because this base
   tokenizer has no chat template. It does not prevent the completion server
   from becoming healthy, but chat qualitative/logprobs requests need an
   explicit valid template or must use the completions endpoint.

## Recommended next checks (hardware required)

1. Add narrow adapter tests that mock the low-level generator and prove
   `sampling_params=None` returns logits with the exact prefill/decode shapes,
   does not call `set_sampling_params`, and never enters steady token feedback.
2. Re-run the first greedy completion to prove the stale 131072 crash is gone.
3. Run min-p and a supported logprobs request to force host compatibility, then
   return to greedy/device decode and prove trace capture and async steady decode
   remain active.
4. Test unrestricted stochastic `top_k=0` explicitly and choose a truthful
   host-route or rejection policy rather than a distribution-changing fallback.
5. Re-run stale-token/current-position/page-table adapter tests on the device
   path to ensure the compatibility branch did not weaken the canonical traced
   path.
