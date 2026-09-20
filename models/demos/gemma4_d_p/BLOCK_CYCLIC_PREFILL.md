# Block-cyclic prefill

## Request contract

IS/dgen/tt-llm-engine owns scheduling, resident KV history, padding, and token
packing. Gemma consumes one fixed-size chunk in CP-rank order and the request
`(slot_id, actual_start, actual_end)`. Real tokens occupy the absolute interval
`[actual_start, actual_end)`; the prefix before it must already be cached.
The output preserves the input row order; upstream discards padded rows.

For chunk size C, CP degree P, and L=C/P, cache row i on rank r represents
`(i // L)*C + r*L + i%L`. With `g=start//C`, `b=(start//L)%P`, and `o=start%L`,
the local write begins at `(g+1)*L` for r<b, `g*L+o` for r=b, and `g*L` for r>b.
Input row j must contain the token belonging to that cache row plus j. This
includes a wrap within the boundary rank; a simple global roll is insufficient.

Example: C=8192, P=8, start=7008, end=9000. Rank 0 receives positions 8192–9215
(9000 onward is padding). Rank 6 receives 7008–7167 followed by 14336–15199
(padding); rank 7 receives 7168–8191. Upstream must supply this layout.

**Current operator limit:** start must be a multiple of 32, not necessarily C.
End can be unaligned; require `start < end <= min(start+C, max_seq_len)`.
Supporting starts such as 7000 requires sub-tile cache writes and corresponding
RoPE/SDPA changes; do not silently round a request down.

## Existing operator behavior

- `update_padded_kv_cache`: uses `kv_actual_global=start` for the per-rank write
  offset and `valid_global=end` to stop at ceil32(end). The final partial tile
  still contains pad values. Consumers must honor end; this is not zero-padding
  for bytewise migration comparisons.
- RoPE: DeepSeek's `rotary_embedding_indexed` uses the same local write offset.
  Gemma already gathers replicated RoPE tables through device position indices;
  supplying the correct rotated positions preserves its packed channel layouts.
- Global ring SDPA: `slot_id` and `kv_actual_isl_tensor=start` enable rotated Q
  mapping on device. This metadata API derives its bound from start+C, clamped
  to cache capacity; it has no compatible end tensor. Causality prevents padded
  keys (positions >=end) from affecting real queries (positions <end).
- Sliding ring SDPA: its work plan and halo exchange require aligned, complete
  groups. Forwarding a rotated start is insufficient. Gemma will gather Q into
  two aligned groups, call existing SWA for each, then gather back to request
  order. Dummy Q rows do not interact with real Q rows. The second group is
  clamped to the last cache group for end-of-cache padding. This costs two SWA
  calls; a native rotated halo kernel is a future performance improvement.

Relevant implementations: `tt/attention/ring_prefill.py`, `tt/model.py`,
`tt/prefill_metadata.py`; shared operator contracts live under
`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/` and
`ttnn/cpp/ttnn/operations/transformer/sdpa/`. DeepSeek's reference position mapping
is `models/demos/deepseek_v3_d_p/tt/mla/utils.py::rotated_chip_positions`.

## Changes

1. Extend Gemma's metadata object with validated request bounds, rotated RoPE
   indices, and sliding gather indices/group starts. Keep device addresses stable.
   Stage all buffers before trace capture/replay, never inside the captured graph.
2. Accept actual_start/actual_end at the model boundary, retaining aligned demo
   compatibility. Feed staged RoPE indices through the existing gather/packing
   path, pass end to both cache writers, and use the SWA adaptation above.
3. Update demo staging to use the same request metadata. No scheduler, producer,
   shared kernel, DeepSeek, or cache-address-table changes: physical KV layout
   does not change. This checkout has no registered Gemma serving adapter; its
   eventual adapter must call the metadata staging API before each replay.

## Tests

- Host mapping/validation: aligned, rank-boundary, mid-rank, cross-group, partial
  tails, cache-end padding, invalid bounds, and both Galaxy mesh orientations.
  Compare mapping with independently enumerated global cache ownership.
- Synthetic device tests: packed global and separate sliding cache writes plus
  attention against PyTorch, with resident prefixes and two user slots. Check
  real output rows, retained prefixes, and untouched cache tiles beyond end.
- Replay one trace with changed starts/ends/slots to catch captured host scalars;
  compare rotated RoPE lookup with absolute-position lookup. No model weights
  are needed. Run existing Gemma host tests and normal pre-commit hooks.

## Upstream check (2026-09-20)

Checked merged GPT-OSS PRs [#51438](https://github.com/tenstorrent/tt-metal/pull/51438),
[#53153](https://github.com/tenstorrent/tt-metal/pull/53153), and
[#54935](https://github.com/tenstorrent/tt-metal/pull/54935), plus the active runner
PR [#56519](https://github.com/tenstorrent/tt-metal/pull/56519). They do not add
rotated sliding groups. Upstream main's sliding work-plan blob matches this
checkout (`a88eab0176c0e36efeb063000468e57a656a98a7`). Circular-cache trace support
is separately tracked in open issue [#56115](https://github.com/tenstorrent/tt-metal/issues/56115).
Dynamic-length PR [#55115](https://github.com/tenstorrent/tt-metal/pull/55115)
does not support combining tensor logical_n with KV-pad rotation.

The SWA workaround is isolated in `tt/attention/sliding_chunk.py`. Replace its
call from `sliding_ring_prefill_attention` when native rotated SWA is available,
then remove its metadata staging hook. No changes to RoPE or cache layout needed.

## Usage and validation

Eager: `model(hidden_states, user_id=slot, actual_start=start, actual_end=end)`.
The hidden states must already have the service's block-cyclic CP row order.
For tracing, set `model._prefill_metadata_external = True`; before capture/replay,
call `model.prefill_metadata.update(slot_idx=slot, actual_start=start, actual_end=end)`
and copy the new input into its existing device buffer. Updating only the three
scalar buffers is insufficient: RoPE and the SWA adapter also need staging.

Validated on Blackhole Galaxy (2026-09-20), without model weights:

```sh
python_env/bin/python -m pytest models/demos/gemma4_d_p/tests/unit -k 'not device' -q
python_env/bin/python -m pytest models/demos/gemma4_d_p/tests/unit/test_block_cyclic_prefill.py -k device -sv
```

61 host tests passed. All four device cases passed (global/sliding × 8x4/4x8),
using one trace per case across `(start,end)` = `(0,1056)`, `(1056,9000)`,
`(7008,9000)`, `(8192,12001)`, `(15392,16381)` and two user slots. Tests check
absolute RoPE lookup, attention PCC >=0.995, cache PCC >=0.999, stable metadata
addresses, and exact preservation of prior KV, later tiles, and the other slot.

`tests/test_block_cyclic_golden.py` contains just a packing utility and one
unparameterized 256K test. It tokenizes the Gutenberg input, checks the token IDs
against the GPU trace, and runs 54 requests with seed 42: 53 starts off the 8K
boundary, 53 rewinds larger than one tile, and 51 unaligned ends. All requests
replay one trace using a fixed token buffer. Tokens and metadata are uploaded and
synchronized before the timer; warmup and capture are logged separately. It compares
all heads and all positions in the final decoder layer's packed KV cache against
`/mnt/models/huggingface/gpu_traces/gemma4_d_p/gutenberg-135`, requiring PCC >=0.98.

```sh
HF_MODEL=google/gemma-4-31B-it \
HF_HOME=/mnt/models/huggingface \
TT_CACHE_PATH=/mnt/models/huggingface/tt_cache/gemma4_d_p/google--gemma-4-31B-it \
HF_HUB_OFFLINE=1 \
python_env/bin/python -m pytest models/demos/gemma4_d_p/tests/test_block_cyclic_golden.py -sv
```

The traced standalone test passed in 137.58 seconds, with final-layer PCC 0.983890.
Its first two replays took 392.546 ms and 397.450 ms on this run. The canonical
256K/8192/8x4 traced test also passed on this branch, with initial replays of
393.1 ms and 405.1 ms. Both paths currently use the two-pass SWA adapter, including
aligned requests; these timings include that additional device work.
The BFP8 model's native single-SWA aligned baseline is also PCC 0.983890 against this BF16
GPU trace, so this end-to-end test uses 0.98. A stricter exploratory check found
per-head sliding V scores of 0.987504 (layer 24) and 0.981308 (layer 25) for both
aligned and rotated runs; a per-head 0.99 requirement rejects the aligned model
too. The focused operator tests retain the stricter thresholds above.
Serving integration is not exercised by these direct model/operator tests.
