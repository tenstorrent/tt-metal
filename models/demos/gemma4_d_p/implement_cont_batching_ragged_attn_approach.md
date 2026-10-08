# Implement packed multi-request prefill with split/concat attention

Implement a "ragged attention" prefill path for Gemma4-31B-it in
`models/demos/gemma4_d_p`, targeting Blackhole CP8/TP4. Reduce padding work by
packing requests together for tokenwise computation, while reusing existing
batch-1 ring-attention and KV-cache-update operations per request.

## Execution flow

```text
Packed tokens + packed per-request absolute positions
    -> shared embedding, projections, normalization, RoPE, global KV packing
    -> split Q/K/V (or Q and global packed KV) by request
        -> request-specific KV-cache update
        -> request-specific ring attention
    -> remove attention padding and concatenate valid outputs
    -> shared output projection, residuals, MLP
```

Repeat the split/attention/concat flow at every attention layer. Where attention
requires padding, add it around the individual attention calls and remove padded
outputs before returning to shared tokenwise computation.

## Requirements

- **RoPE:** Pack positions in exactly the same order as tokens. Each token uses
  its request's absolute position, not its offset in the concatenated tensor.
  For example, A continuing at 4096 and B starting at 0 need positions
  `[4096, 4097, ..., 0, 1, ...]`. Preserve the existing sliding/global RoPE
  differences and global packed-KV transforms. RoPE may run before splitting
  when the position mapping is correct.
- **KV-cache updates:** Split before cache writes. Invoke the existing update
  operation separately for each request, using its own KV slot and prefix offset,
  then invoke its attention operation. Sliding layers write separate K/V;
  global layers write packed KV. Track actual valid lengths separately from
  padded attention extents and wire valid-length clamping where needed.
- **CP layout and cache geometry:** Verify token distribution and inverse output
  mapping across CP ranks and TP row collectives. Handle unequal useful-token
  counts across ranks. Shorter query fragments must not silently reinterpret
  historical KV: cache placement currently depends on chunk geometry. Verify
  alignment, padding, and any redistribution against the implementation.
- **Trace replay:** Use stable per-request metadata buffers. Support changing
  request slots and prefix lengths across replays; explicitly define packing
  buckets, trace variants, and fallbacks for changing segment sizes or occupancy.
  Inactive lanes must not write live caches. Preserve correct buffer/semaphore
  ownership and ordering across separate attention calls.
- **Serving integration:** Preserve request identity, per-request chunk ordering,
  valid output ownership, migration address mapping, and layer-completion
  acknowledgments. Handle requests finishing and slots being refilled at batch
  boundaries. Keep the current single-request path available for comparison.

Read the code to verify constraints before implementing. Relevant starting points:
[attention](tt/attention/__init__.py), [ring attention/cache wrappers](tt/attention/ring_prefill.py),
[RoPE staging](tt/model.py), [metadata](tt/prefill_metadata.py),
[runtime](tt/runners/runtime.py), and [migration mapping](tt/runners/kv_chunk_table.py).

## Validation and performance

Compare each request's valid outputs **and written KV** with independent
single-request execution. Cover unequal lengths, different prefix positions,
partial final chunks, alignment boundaries, slot reuse, changing batch
composition, repeated trace replay, and migration correctness. Changing one
request must not affect another request's results or cache.

Measure useful tokens/sec and end-to-end request latency, including packing,
redistribution, slicing/padding/concatenation, staging, and migration costs. Include
mixed-length workloads that expose padding savings and long full-chunk workloads
that expose overhead. Report supported shapes, remaining padding, limitations,
and any checks that could not be run.

Run the canonical baseline from the repository root:

```bash
export HF_MODEL=google/gemma-4-31B-it
export HF_HOME=/mnt/models/huggingface
export TT_CACHE_PATH=/mnt/models/huggingface/tt_cache/gemma4_d_p/google--gemma-4-31B-it
export HF_HUB_OFFLINE=1

pytest 'models/demos/gemma4_d_p/demo/text_demo_prefill.py::test_prefill_long_context_traced[blackhole-ctx_256k-chunk8192-text-8x4]' -sv
```

This baseline contains only full chunks; it does not establish ragged-path
correctness or padding savings. Deliver the implementation, targeted correctness
coverage, and measured comparison.
