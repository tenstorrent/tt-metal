# Gemma4 rotated prefill chunks

## Objective and contract

Add end-to-end support for a contiguous absolute token range delivered in a
position-derived, per-SP-rank layout. This is not an arbitrary permutation.
Existing prefix KV must already be resident. Default geometry: chunk size 8192,
SP8, TP4; every request carries a full padded chunk (1024 rows per SP rank).

Wire metadata is three little-endian uint32 words:
`(slot_id, actual_start, actual_end)`, with an exclusive end. Start is the existing
prefix length (`kv_actual_global` / `kv_actual_isl`); end is `valid_global`;
new real-token count is `end - start`.

Require a 32-aligned start, a valid slot, `0 < end - start <= chunk_size`, and
cache bounds. To resume at an unaligned position, align down and replay up to
31 preceding tokens; never align up and leave a hole.

## Layout

Reuse `rotated_chip_positions` in
`models/demos/deepseek_v3_d_p/tt/mla/utils.py` as the reference mapping.
With local chunk size C, rank r owns absolute blocks
`[n * chunk_size + r * C, n * chunk_size + (r + 1) * C)`.
Each incoming rank-local slab contains its next C owned positions in chronological
order, starting at its cache frontier. Positions >= actual_end are padding.
Padding is a local suffix, not necessarily a global suffix after concatenation.

Example: continue at 7000 through 9000 by sending `[6976, 9000)`:
rank 0 has 808 real rows for `[8192,9000)`; rank 6 has 192 for `[6976,7168)`;
rank 7 has 1024 for `[7168,8192)`; all other rows are padding.

## Required changes

1. **Producer and input staging:** use the shared mapping to pack tokens and pad
   rows. Update `models/demos/common/prefill/runners/prefill_producer.py`, which
   currently slices sequential tokens and only reshapes them over SP. Inspect
   the receiving adapter/runtime on the working branch and validate the contract.
2. **Gemma4 RoPE:** stage absolute position IDs in the same layout as the tokens.
   In `models/demos/gemma4_d_p/tt/model.py`, RoPE uses gathered cos/sin with
   `rotary_embedding_llama`; it does not automatically consume start metadata.
   Update demo/runtime staging, currently sequential `arange`, accordingly.
3. **Trace metadata:** update persistent slot/start/end device tensors before
   each replay; preserve their addresses. Reference
   `deepseek_v3_d_p/tt/tt_prefill_runtime.py::_metadata_from_msg`.
4. **Cache writes and migration:** in
   `gemma4_d_p/tt/attention/ring_prefill.py`, pass end as `valid_global` to both
   sliding and packed-global KV writes. Clear migration padding before layer
   acknowledgements; verify zeroing supports both cache formats.
5. **Attention and outputs:** verify rotated causal and sliding-window masking.
   Any final-token/output selection must use the same position mapping.
   Preserve aligned-chunk behavior; avoid unrelated changes.

## Kernel support to reuse

- `update_padded_kv_cache`: derives local write offsets from start; optional end
  clamps writes to tiles containing real tokens. The last partial tile still
  needs padding cleanup for migration.
- `rotary_embedding_indexed`: derives RoPE offsets from start with block-cyclic
  tables. Useful as a reference; Gemma4 currently uses explicit position gathers.
- `ring_joint_scaled_dot_product_attention`: reads slot/start tensors and derives
  query positions. Its traced extent is start + padded chunk size, not actual_end;
  real queries exclude later padding through causality. Sliding attention expects
  a full padded query group.
- `zero_padded_kv_cache`: uses end to clear the migration-aligned pad window.

Start with Python integration; modify kernels only if tests expose missing
support. Follow repository AGENTS.md verification requirements.

## Tests and acceptance

- **Host mapping:** exact coverage without duplicates, local suffix padding,
  aligned-case equivalence, invalid metadata rejection.
- **Ops:** verify KV placement and preserved prefix; compare RoPE and both global
  and sliding attention against absolute-position references. Existing tests live
  under `deepseek_v3_d_p/tests/op_unit_tests/` for indexed RoPE, padded KV writes,
  and pad zeroing.
- **Gemma4 numerical integration:** compare aligned prefill against rotated
  continuation for identical tokens, checking valid outputs and KV caches.
  Begin with layers covering both attention types, then run all 60 layers.
- Cover `[6976,9000)`, partial final tiles, repeated continuations, interleaved
  slots, and requests ending at cache capacity. Replay one captured trace while
  changing slots, starts, and ends to detect stale metadata.
- Require numerical agreement within established model tolerances, correct
  migration padding/ack ordering, and no aligned-path regression. Synthetic
  completion or finite outputs alone are insufficient.

Initial findings came from source inspection on `main`; no rotated Gemma4 device
test or external IS/dgen payload capture was performed. Recheck the working
branch and confirm the actual sender's layout before claiming interoperability.
