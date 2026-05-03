# Lowering Spec: Trace Decode Runtime

Source: local `models/demos/gemma4/demo/text_demo.py` trace path and TTNN trace APIs.

## Trace Signature

| Field | Value |
| --- | --- |
| Batch | `1` |
| Decode sequence shape | one token per replay |
| Hidden input | `[1, 1, 1, 2816]` BF16 host-created embedding copied into persistent device buffer |
| Position input | `[1, 32]` UINT32 padded RoPE lookup index |
| Cache position | `[1]` INT32 cache update index |
| Page table | `[1, max_seq_len / page_block_size]` INT32, persistent |
| KV cache | Persistent per-layer paged K/V tensors |
| Output | Token id if on-device sampling is enabled, otherwise logits |

## Runtime Pseudocode

```python
# compile iteration
inputs_d = to_device(make_decode_inputs(token, pos))
output = decode_forward(inputs_d)

# capture
trace_inputs = to_device(make_decode_inputs(next_token, pos + 1))
trace_id = begin_trace_capture(mesh, cq_id=0)
trace_output = decode_forward(trace_inputs)
end_trace_capture(mesh, trace_id, cq_id=0)

# replay loop
for token, pos in decode_steps:
    host_inputs = make_decode_inputs(token, pos)
    copy_host_to_device_tensor(host_inputs["embeds"], trace_inputs["embeds"])
    copy_host_to_device_tensor(host_inputs["position"], trace_inputs["position"])
    copy_host_to_device_tensor(host_inputs["position_int32"], trace_inputs["position_int32"])
    execute_trace(mesh, trace_id, cq_id=0, blocking=False)
    token = extract_token(trace_output)
```

## Trace Risks

Trace capture must not allocate tensors whose addresses change between replay iterations. Current high-risk areas are router dense scatter, sparse expert intermediates, CCL temporaries, and SDPA program buffers. If trace capture fails with a trace-region allocator error, the next concrete unblocker is a nonzero Gemma4 trace-region size plus removal of deallocations inside the captured decode region.

## Acceptance Evidence Required

1. Untraced decode and traced decode produce matching next tokens for the same prompt state.
2. Trace replay runs for long decode length, at least 128 new tokens for the current demo signature.
3. The accepted timing excludes compile iteration and reports TTFT and steady decode tokens/sec/user.
