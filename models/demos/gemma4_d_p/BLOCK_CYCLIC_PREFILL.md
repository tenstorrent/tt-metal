# Block-cyclic prefill

## Request contract

IS/dgen/tt-llm-engine owns scheduling, resident KV history, padding, and token
packing. Gemma consumes a fixed-size chunk in CP-rank order plus
`(slot_id, actual_start, actual_end)`. Real tokens occupy `[actual_start, actual_end)`;
the preceding prefix must already be cached. Callers discard padded outputs.

Start must be 32-token aligned. End can be unaligned:
`0 <= start < end <= min(start + chunk_size, max_seq_len)`.

For chunk size C, CP degree P, and local size L=C/P, cache row i on rank r holds
position `(i // L)*C + r*L + i%L`. Pack each rank's new Q rows in increasing
absolute-position order, including padding to L rows.

Example: C=8192, P=8, start=7008, end=9000. Rank 6 receives 7008–7167 followed
by padding at 14336–15199; rank 7 receives 7168–8191; rank 0 receives
8192–9215, with 9000 onward padded. RoPE uses these same absolute positions.

## Native operations

- `update_padded_kv_cache`: `kv_actual_global=start` selects each rank's write
  offset; `valid_global=end` stops writes at ceil32(end), including final-tile pads.
- RoPE: Gemma gathers its tables using staged block-cyclic position indices.
- Global and sliding ring SDPA: `kv_actual_isl_tensor=start` maps local Q rows
  to absolute positions. The metadata path derives the padded extent as
  `min(start + C, cache_capacity)`. Causal masking prevents padding after end
  from affecting real query rows.
- Sliding SDPA handles both Q segments natively, including a compute block
  crossing the wrap. It reads local KV and exchanges predecessor tails in one
  invocation. Aligned ranks send one tail; a split destination receives two.
  Gemma reserves two halo slots so the trace can replay arbitrary offsets.
  The upstream op also accepts one slot for traces that never wrap Q.
  Halo size is `ceil((window - 1) / k_chunk_size) * k_chunk_size` per slot.
- Scalar SDPA uses `kv_actual_isl=start`, `logical_n=end`; both are tile-aligned.
  Partial groups and runtime cache reuse are supported. Circular KV retains
  its aligned-group scalar contract; circular metadata remains unsupported.

No Gemma-specific Q gathers, output restoration, mode selection, or extra
attention calls are needed. The physical KV layout is unchanged.

## Calling and tracing

Eager: `model(hidden_states, user_id=slot, actual_start=start, actual_end=end)`.
The hidden states must already have the request's CP row order.

Capture one trace after warmup with `model._prefill_metadata_external = True`.
Before each replay, copy tokens into the fixed input buffer and call:

```python
model.prefill_metadata.update(slot_idx=slot, actual_start=start, actual_end=end)
```

The same trace handles aligned, rotated, partial, and rewound requests.
Metadata tensor addresses remain stable. Input and metadata staging precede
trace timing. The Gemma runner stages the same metadata and permits rewinds
within each slot's populated prefix; input tokens must already use CP row order.

## Validation

- C++ work-plan tests enumerate required causal K chunks and halo addresses,
  including mid-block wraps, partial/cache-end Q, and circular-cache regression.
- Shared SDPA tests cover Gemma and GPT-OSS shapes, Q blocks 64/128, scalar
  cache reuse, determinism, mixed sliding/global metadata traces, and circular KV.
- Gemma operator tests replay one trace across slots, offsets, and partial ends
  on 8x4 and 4x8. They check attention against PyTorch, absolute RoPE values,
  written KV, unchanged prefix/future tiles, and the other user's cache.
- `tests/test_block_cyclic_golden.py` has a packing utility and one 256K test.
  It tokenizes the Gutenberg text, verifies token IDs, and replays 54 requests
  with random tile-aligned starts, unaligned ends, and rewinds. It compares all
  final-layer KV heads and positions against the GPU trace (PCC >=0.98).

```sh
HF_MODEL=google/gemma-4-31B-it \
HF_HOME=/mnt/models/huggingface \
TT_CACHE_PATH=/mnt/models/huggingface/tt_cache/gemma4_d_p/google--gemma-4-31B-it \
HF_HUB_OFFLINE=1 \
python_env/bin/python -m pytest models/demos/gemma4_d_p/tests/test_block_cyclic_golden.py -sv
```

Validated on 2026-09-25 after rebasing onto main `e2e1771d624`, using a local
`./build_metal.sh -ce` build. All 72 selected host unit cases, four device replay
variants, the 256K golden test, and the canonical 256K traced test passed.
The branch changes only `models/demos/gemma4_d_p`; native SWA comes from main.

Blackhole 8x4 synchronized trace timings, including main's activation/SDPA tuning:

| Run | First replay | Second replay | Final replay | Total replay |
| --- | ---: | ---: | ---: | ---: |
| Block-cyclic, 54 requests with rewinds | 197.5 ms | 206.3 ms | 496.4 ms | 18.413 s |
| Canonical, 32 aligned chunks | 197.7 ms | 206.6 ms | 505.8 ms | 11.1 s |

Block-cyclic final-layer KV PCC is 0.982167 against the GPU golden (threshold 0.98).
Staging, warmup, and capture are excluded from replay timing. Serving integration
is not exercised by these direct tests; runner metadata staging is unit-tested.
