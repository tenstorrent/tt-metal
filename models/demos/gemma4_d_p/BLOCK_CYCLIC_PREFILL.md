# Block-cyclic prefill

## Request contract

IS/dgen/tt-llm-engine owns scheduling, cached history, and token packing. Gemma
consumes a fixed-size chunk in CP-rank order plus `(slot_id, actual_start,
actual_end)`. Real tokens occupy `[actual_start, actual_end)`; preceding KV must
already exist. Starts must be multiples of 32; ends may be unaligned. Require
`start < end <= min(start + chunk_size, max_seq_len)`. Outputs preserve input
row order; the caller discards padded rows.

For chunk size C, CP degree P, and L=C/P, cache row i on rank r represents
`(i // L)*C + r*L + i%L`. The request fills each rank from its first owned token
at or after start. This includes a wrap within the boundary rank; a global roll
is insufficient. `tt/prefill_metadata.py::chunk_positions` implements the mapping.

Example: C=8192, P=8, `[7008,9000)`. Rank 0 receives 8192–9215 (9000 onward
is padding), rank 6 receives 7008–7167 followed by 14336–15199 (padding), and
rank 7 receives 7168–8191.

## Native operator path

- **KV writes:** `update_padded_kv_cache` receives `kv_actual_global=start` and
  `valid_global=end`. It writes through ceil32(end); the final partial tile may
  contain padding. Existing history and later tiles remain untouched.
- **RoPE:** Gemma gathers resident tables using block-cyclic position indices.
  Out-of-capacity padded positions use index zero; their outputs are discarded.
- **Global and sliding SDPA:** both receive `slot_id` and
  `kv_actual_isl_tensor=start`. Native SDPA maps Q rows to absolute positions.
  The tensor API derives the query bound from start+C, clamped to cache capacity;
  real queries cannot attend to padded keys because attention is causal.
- **Sliding halo:** a wrapped rank can need two predecessor tails. Allocate
  `2 * max(ceil((window - 1) / k_chunk) * k_chunk, 32)` rows in each halo buffer.
  For smaller chunks whose local Q slab is below the 1024-token window, native
  SDPA uses multi-hop halos with one slot. These starts must be CP-slab-aligned;
  wrapped Q with multi-hop halos is not supported upstream. Keep the existing
  per-layer buffer keys to avoid reuse between consecutive SWA layers.

Each sliding layer makes one native SDPA call. There is no Gemma Q/output
reordering, group metadata, or selection between multiple traces. Native mapping
and halo contracts live in the SDPA kernel's `chunked_q_mapping.hpp` and
`sliding_window_work_plan.hpp`.

## Staging and replay

Eager: `model(hidden_states, user_id=slot, actual_start=start, actual_end=end)`.
Hidden states must already follow the request's block-cyclic row order.

For tracing, set `model._prefill_metadata_external = True`, warm up, and capture
one graph with fixed input buffers. Before every replay, copy the packed tokens
and call `model.prefill_metadata.update(slot_idx=slot, actual_start=start,
actual_end=end)`. This updates scalar metadata and RoPE indices in place.
The same trace handles aligned, rotated, partial, and rewound requests.

The Gemma runtime uses this same staging API, accepts tile-aligned rewinds, and
rejects gaps beyond its populated prefix. Token packing remains upstream's job.

## Tests

- `tests/unit/test_block_cyclic_prefill.py`: independent ownership mapping and
  invalid-bound checks; device replay for global/sliding attention on 8x4/4x8.
  One trace changes slots, starts, and ends. Checks RoPE, attention against
  PyTorch (PCC >=0.995), cache values (>=0.999), stable metadata addresses, and
  exact preservation of untouched cache tiles and the other user slot.
- `tests/unit/test_prefill_runner.py`: runtime staging, all user slots, rewinds,
  unaligned ends, and rejection of gaps.
- `tests/test_block_cyclic_golden.py`: 256K Gutenberg text, 54 seeded requests
  with offsets and rewinds, final-layer KV against GPU traces (PCC >=0.98).
- `tests/test_block_cyclic_aligned.py`: same randomized requests against a
  fully aligned TT run, clearing all KV between runs (PCC >=0.9999).

Both full-model tests use 1D ring fabric and one trace. Replay timings exclude
input/metadata staging, warmup, and capture. Direct model tests do not exercise
producer/runner transport.

```sh
export HF_MODEL=google/gemma-4-31B-it HF_HOME=/mnt/models/huggingface \
  TT_CACHE_PATH=/mnt/models/huggingface/tt_cache/gemma4_d_p/google--gemma-4-31B-it \
  HF_HUB_OFFLINE=1
python_env/bin/python -m pytest models/demos/gemma4_d_p/tests/unit/test_block_cyclic_prefill.py -sv
python_env/bin/python -m pytest models/demos/gemma4_d_p/tests/test_block_cyclic_aligned.py -sv
python_env/bin/python -m pytest models/demos/gemma4_d_p/tests/test_block_cyclic_golden.py -sv
```
