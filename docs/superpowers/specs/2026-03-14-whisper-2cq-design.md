# Whisper 2CQ (Two Command Queue) Trace Implementation

## Overview

Add dual command queue (2CQ) support to the Whisper model's traced decoder execution. This replaces the existing single command queue trace path, overlapping host-to-device data transfers (Queue 1) with computation (Queue 0) to improve decode throughput.

## Goals

- Overlap decoder input transfers with trace execution within the decode loop
- Support all hardware configurations: n150, n300, T3K, Galaxy, Blackhole (p100, p150)
- Apply to conditional generation and streaming paths; skip audio classification

## Non-Goals

- Overlapping during the prefill pass (one-time cost, not worth the complexity)
- Changing the encoder path
- Modifying the functional model code (`ttnn_optimized_functional_whisper.py`)
- Batch-to-batch encoder feature overlap (deferred — requires caller-level pipelining outside the generator; can be added later once within-loop 2CQ is validated)

## Key Architectural Decision: Preprocessing and the Q1 Boundary

`preprocess_decoder_inputs` (in `ttnn_optimized_functional_whisper.py:784`) currently:
1. Transfers token IDs to device via `ttnn.from_torch(..., device=device)` on the default queue
2. Runs `ttnn.embedding(...)` on device
3. Adds position embeddings on device (indexed by host Python integer: `embed_positions.weight[decode_pos : decode_pos + 1]`)
4. Returns `decoder_hidden_states` already on device (DRAM)

Since the embedding table lives on device, the embedding lookup must run on device. This means we cannot do a pure host→device Q1 transfer of pre-embedded hidden states like sentence_bert does with its raw inputs.

**Constraint: Position embedding uses host-integer indexing.** The current code slices the position embedding weight table with a Python integer (`decode_pos`). Traces cannot accept new Python arguments per iteration — they replay fixed operations. Including position add inside the trace would require changing the indexing to use a device tensor (e.g., `ttnn.embedding(current_decode_pos, embed_positions_weight)`), which means modifying `ttnn_optimized_functional_whisper.py`.

**Chosen approach: Trace captures `embedding + decoder`, position add stays outside.**

1. **Host-prep phase:** Create the token ID tensor on host. Store as host staging tensor.
2. **Q1 transfer:** Copy the token ID (uint32, tiny — 1 token per batch item) to DRAM staging buffer on Q1.
3. **Q0 pre-trace:** Embedding lookup on Q0 produces `inputs_embeds`. Position embedding is sliced using host-known position and added: `decoder_hidden_states = inputs_embeds + positions`. This happens outside the trace since position indexing requires the host-known decode step.
4. **Q0 trace execution:** `to_memory_config` moves `decoder_hidden_states` to L1 trace input, then `decoder` forward pass executes within the trace.

The trace boundary stays at `decoder` (same as today). The 2CQ benefit: Q1 transfers the next token ID while Q0 finishes the previous trace execution. Embedding + position add run on Q0 after Q1's transfer completes but before the trace, adding minimal latency (single-token operations).

**Why this works for autoregressive decoding:** While Q0 executes the trace for token N, the host prepares token N+1's token ID. When `_execute_decoder_trace` is called for N+1, Q1 starts the small host→device copy while Q0 may still be finishing N's trace tail. The event protocol ensures Q0 doesn't start embedding + trace until Q1's transfer completes.

**Attention mask:** The traced decode path uses `decoder_attention_mask=None` (KV-cache mode doesn't use attention masks) and `create_attention_mask=False`. No attention mask tensor needs to be transferred via Q1.

### Decode Position: Stays Device-Managed

The current code uses `ttnn.plus_one(self.current_decode_pos_per_size[trace_key])` to increment the decode position in-place on device. This is a single scalar increment — no benefit to host-managing it via Q1. `plus_one` continues to run on Q0 after trace execution (or could be included in the trace itself). No `decode_pos_host` staging buffer needed.

## Design

### Device Configuration

All Whisper `device_params` add `num_command_queues: 2`:

```python
# Before
{"l1_small_size": WHISPER_L1_SMALL_SIZE, "trace_region_size": WHISPER_TRACE_REGION_SIZE}

# After
{"l1_small_size": WHISPER_L1_SMALL_SIZE, "trace_region_size": WHISPER_TRACE_REGION_SIZE, "num_command_queues": 2}
```

Starting values for `l1_small_size` (1024) and `trace_region_size` (100000000) remain unchanged, adjusted based on testing. The trace boundary is unchanged (decoder only), so trace region usage should be similar to today.

**Multi-device mesh semantics:** `ttnn.record_event(mesh_device, cq_id)` and `ttnn.wait_for_event(cq_id, event)` operate transparently on mesh devices (same as `ttnn.begin_trace_capture(mesh_device, cq_id=0)` which Whisper already uses). No per-device event management needed.

### New Instance Variables in WhisperGenerator.__init__

- `self.op_event` — tracks completion of computation on Queue 0
- `self.write_event` — tracks completion of host-to-device transfers on Queue 1
- `self.token_id_host[batch_size]` — per-batch-size host staging tensor for token IDs (uint32, shape [batch_size, 1])
- `self.token_id_device[batch_size]` — per-batch-size DRAM staging tensor for token IDs on device (Q1 copy target)

Existing pre-allocated tensors (encoder_hidden_states, current_decode_pos, KV caches, cross-attention caches) are unchanged.

Events are initialized to `None` and created during `_capture_decoder_trace` (Phase 1).

### Trace Capture: _capture_decoder_trace (2CQ Version)

Replaces the existing method. The trace boundary remains `decoder` only (same as today). Embedding + position add run on Q0 before the trace. The 2CQ pattern adds event-synchronized Q1 transfers.

The function being traced is the same `traced_decoder_fn(trace_key, hidden_states)` that calls `ttnn_optimized_functional_whisper.decoder(...)` with the pre-allocated encoder_hidden_states, KV cache, and cross-attention cache.

Follows sentence_bert's three-phase pattern:

**Phase 1 — JIT run:**
1. `record_event(mesh_device, 0)` → initial `op_event`
2. `wait_for_event(1, op_event)` → Q1 waits for Q0
3. `copy_host_to_device_tensor(token_id_host, token_id_device, 1)` → transfer token IDs on Q1
4. `record_event(mesh_device, 1)` → `write_event`
5. `wait_for_event(0, write_event)` → Q0 waits for Q1 transfer
6. Embedding lookup + position add on Q0 (produces decoder_hidden_states)
7. `to_memory_config(decoder_hidden_states, L1_MEMORY_CONFIG)` → move to L1
8. `record_event(mesh_device, 0)` → update `op_event`
9. Run `traced_decoder_fn` on Q0 (non-traced) for JIT compilation
10. Deallocate output

**Phase 2 — Optimized run:**
Same event-synchronized sequence as Phase 1. Runs the post-JIT optimized path. Output is deallocated before Phase 3.

**Phase 3 — Trace capture:**
Same input transfer + embedding + position add sequence, then:
1. `to_memory_config(decoder_hidden_states, L1_MEMORY_CONFIG)` → creates trace input in L1
2. Record buffer address of trace input
3. `begin_trace_capture(mesh_device, cq_id=0)`
4. Run `traced_decoder_fn` (decoder only, captured in trace)
5. Allocate tensor with same spec (reclaims buffer address for next iteration's trace input)
6. `end_trace_capture(mesh_device, trace_id, cq_id=0)`
7. `synchronize_device(mesh_device)`
8. Assert buffer address matches (validates trace correctness)

**Prerequisite:** Cross-attention cache must already be populated before trace capture (same as current behavior — first decoder iteration runs un-traced to populate it).

### Trace Execution: _execute_decoder_trace (2CQ Version)

Replaces the existing method. Accepts the host token ID tensor and the host-known decode position. Each call in the decode loop:

1. `wait_for_event(1, op_event)` — Q1 waits for previous computation
2. `copy_host_to_device_tensor(token_id_host, token_id_device, 1)` — transfer token ID on Q1
3. `record_event(mesh_device, 1)` — `write_event`
4. `wait_for_event(0, write_event)` — Q0 waits for transfer
5. Embedding lookup: `ttnn.embedding(token_id_device, embed_tokens_weight)` on Q0
6. Position add: `inputs_embeds + embed_positions_weight[decode_pos:decode_pos+1]` on Q0 (host-integer indexed, outside trace)
7. `to_memory_config(decoder_hidden_states, L1_MEMORY_CONFIG, output_tensor=trace_input)` — copy to trace input address on Q0
8. `record_event(mesh_device, 0)` — update `op_event`
9. `execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)` — non-blocking execution
10. `synchronize_device(mesh_device)` — wait for completion before reading output

**Note on `synchronize_device`:** This per-iteration sync is required because `to_torch` reads the output tensor. The explicit sync may be redundant if `to_torch` already implies a device sync — this should be tested during implementation and removed if so.

**Note on `reshard` vs `to_memory_config`:** Sentence_bert uses `reshard` because its inputs change memory layout (DRAM sharded → L1 sharded). For Whisper, `to_memory_config` with `output_tensor` (same as current code) is sufficient since we're copying decoder_hidden_states to the pre-allocated L1 trace input tensor at a known buffer address.

Key change from current: `blocking=False` (was `blocking=True`), with explicit sync after.

### Generation Loop Changes

In `_generate_with_temperature`:

**First iteration (i == generation_start):** Unchanged — runs un-traced decoder to populate cross-attention cache, then triggers 2CQ trace capture.

**Subsequent iterations (traced):**
```
1. Prepare host token ID tensor from previous iteration's sampled token
2. _execute_decoder_trace(token_id_host, decode_pos=i)
   # 2CQ: Q1 transfers token ID, Q0 runs embedding+pos_add+trace(decoder)
3. ttnn.to_torch(output)              # waits for trace completion
4. Sample next token (host-side)
5. ttnn.plus_one(current_decode_pos)  # device-side position increment
```

The call to `preprocess_decoder_inputs` is replaced on the traced path — embedding + position add now happen inside `_execute_decoder_trace` (before the trace, on Q0), and the decoder runs within the trace. `preprocess_decoder_inputs` remains for the un-traced first iteration and prefill pass.

**Prefill pass:** Unchanged — runs with non-overlapped path using `preprocess_decoder_inputs` directly.

**Streaming mode:** Works as-is — per-token yield happens after `to_torch` and sampling.

**`use_trace` parameter:** Removed from the public API. Internally, trace is always used when KV cache is present (`kv_cache_per_batch_size[trace_key]` is truthy). The non-KV-cache path (which recomputes the full sequence each step) continues to work without trace, as today.

### Cleanup: _release_decoder_trace Updates

The existing cleanup method releases trace IDs and nulls out trace tensors. With 2CQ, additionally:
- `self.op_event` and `self.write_event` are set to `None`
- `self.token_id_host` and `self.token_id_device` per-batch-size dicts are cleared
- No explicit event deallocation API is needed (events are lightweight device resources released with the device)

## Files Changed

| File | Changes |
|------|---------|
| `tt/whisper_generator.py` | Replace `_capture_decoder_trace` with 2CQ version (trace boundary: decoder, same as today), replace `_execute_decoder_trace` with 2CQ version (handles Q1 transfer + embedding + position add + trace execution), add event/staging instance variables to `__init__`, update decode loop in `_generate_with_temperature` to use new `_execute_decoder_trace` instead of `preprocess_decoder_inputs` on traced path, remove `use_trace` parameter from API, update `_release_decoder_trace` to clean up events and staging tensors |
| `tt/ttnn_optimized_functional_whisper.py` | No changes — `preprocess_decoder_inputs` remains for un-traced/prefill use. The embedding + position add logic is duplicated in `_execute_decoder_trace` (simple: `ttnn.embedding` + one addition). |
| `demo/demo.py` | Add `num_command_queues: 2` to all `device_params`, remove `use_trace` arguments from demo function calls |
| `tests/test_whisper_modules.py` | Add `num_command_queues: 2` to trace-related test `device_params` |

## Files Not Changed

- Encoder path — runs once per audio sample, not in the hot loop
- Audio classification path — excluded from scope
- KV cache allocation — pre-allocated tensors keep stable addresses
- Cross-attention cache population logic — first iteration remains un-traced

## Risks

- **Autoregressive serial dependency:** Unlike encoder-only models (sentence_bert, ViT), each decode iteration depends on the previous token. The overlap window is limited to: Q1 transfers token N+1's input while Q0 finishes token N's trace tail. The primary throughput gain is from eliminating Q0 idle time during host→device transfers, not from full pipeline parallelism.
- **Minor code duplication:** Embedding + position add logic is duplicated between `preprocess_decoder_inputs` (for un-traced/prefill) and `_execute_decoder_trace` (for traced 2CQ path). This is acceptable — the logic is trivial (two lines) and avoids modifying the functional module.

## Testing Strategy

- **Correctness:** Existing demos and tests validate transcription output quality
- **Performance:** Compare `avg_decode_throughput` (already logged) before and after
- **Memory:** Monitor L1/trace region usage, adjust `WHISPER_L1_SMALL_SIZE` and `WHISPER_TRACE_REGION_SIZE` if needed
- **Trace correctness:** Buffer address assertions in Phase 3 catch trace capture errors

## Reference

- Sentence_bert 2CQ implementation: `models/demos/sentence_bert/runner/performant_runner.py`
- ViT 2CQ implementation: `models/demos/vision/classification/vit/common/tests/vit_performant_imagenet.py`
- Current Whisper trace: `models/demos/audio/whisper/tt/whisper_generator.py`
