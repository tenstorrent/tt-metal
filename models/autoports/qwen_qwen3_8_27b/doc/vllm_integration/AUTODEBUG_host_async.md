# AutoDebug: explicit host compatibility under async decode

Source-only diagnosis, 2026-09-14. This report precedes the focused test and
implementation repair. No hardware or server execution by this investigator.

## Observed failure

The parent's `min_p` serving test fails in
`doc/vllm_integration/host_async_failure.log`, timestamp 00:55:21:

```text
async_decode.py:636: runner.model.read_decode_output(tt_out, async_read=True)
generator_vllm.py:222: ttnn.get_device_tensors(tt_out)[0].cpu(...)
TypeError: get_device_tensors(): incompatible function arguments
Invoked with types: torch.Tensor
```

## Source findings

1. The shared runner selects host sampling for `min_p`. In explicit
   compatibility mode, the Qwen adapter calls its existing generator with
   `host_sampling=True`. `QwenGenerator.decode_forward` materializes CPU logits
   through `_host_logits`; the adapter returns their `[batch,1,vocab]` view.
2. `TTAsyncDecodeController.submit_decode` calls a model's `read_decode_output`
   whenever the method exists, including when `decode_forward` already returned
   a Torch tensor. It expects `(output, list_of_read_events)` for async reads.
3. The Qwen adapter's read method currently assumes a TTNN tensor and always
   calls `get_device_tensors`. This matches the exact failing type boundary.
4. `finalize_decode` waits over returned events, then recognizes Torch output
   and bypasses model-side conversion. An empty event list is valid for logits
   whose host transfer has already completed.
5. Direct callers of `process_decode_output_host(..., is_tokens=False)` also
   currently fail even when given completed Torch logits in explicit
   compatibility mode. Its existing TTNN path is exclusively for token IDs.

## Hypothesis and predicted experiment

Hypothesis: add an explicitly gated Torch passthrough to the adapter's two
output methods. A host tensor needs no further TTNN transfer or event. Preserve
the existing TTNN token branch verbatim, including the one-replica 32-UINT32
read, counter, nonblocking submission, and event ordering.

The focused host experiment should reproduce the pre-fix Torch-to-TTNN type
error by driving the actual plugin controller's submit/finalize methods around
a stubbed model forward returning CPU logits. After the repair, the same
tensor must survive submission and finalization with an empty event list and
zero TTNN read/conversion/event calls. Compatibility disabled must reject host
output. Native token tests must retain one replica read, the blocking flag,
and event-before-conversion ordering.

Only `read_decode_output` and `process_decode_output_host` need an implementation
change. No shared-plugin change, new sampler, trace change, or host fallback in
the normal device path is justified by this failure. Server output quality and
real device async behavior remain for the parent's hardware validation.
