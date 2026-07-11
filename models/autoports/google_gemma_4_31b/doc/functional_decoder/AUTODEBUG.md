# AutoDebug: functional-decoder decode evidence

## Executive verdict

The stage-review failure is primarily real: the current exact-context test is an
evidence-test bug, not proof of a bad final-position implementation. It prefills
position 262143 and then replays the same hidden state at position 262143; its
decode PCC is TT prefill versus TT decode, so it neither creates a 262144th token
nor supplies an HF decode oracle.

Static inspection did not find an implementation defect specific to the requested
`history_length=262143, decode_position=262143` case. The final position flows from
the stable device buffers into both RoPE and cache/SDPA, and the bounded sliding
path wraps it to slot 1023 as intended. That conclusion still needs the changed-
buffer trace experiment below.

Inspection did find a separate, concrete implementation bug in non-aligned bounded
sliding prefill: padding is rounded up and written through the circular cache, so
padding after the logical end can overwrite still-live history slots. The exact
262143-history case has only one padded token and that token lands in the expired
slot, so this bug does not invalidate that exact case; it does affect lengths such
as 1025 and 262113 and should not be hidden under the evidence-only finding.

No TT hardware was used. No implementation or test file was modified.

## Headline finding 1 (confirmed, evidence-test bug): the long decode is a replayed prefill token and has no HF decode oracle

Evidence:

- `tests/test_functional_decoder.py:478-486` (line numbers in the reviewed tree)
  calls `prefill_forward` with all `seq_len` hidden states. Position
  `seq_len - 1` is therefore already in the cache.
- `tests/test_functional_decoder.py:500-507` constructs `decode_token` from
  `hidden[:, -1:, :]`; `:508-525` supplies the same absolute position
  `seq_len - 1`.
- `:535-536` compares the TT prefill output for that token with the TT replay
  output. Neither operand is the HF final-position decode result.
- The full-attention HF control at `:489-493` covers only the first 2049
  prefill outputs. The sliding control covers the last prefill output, not a new
  decode update.
- In the traced batch-32 test, the host token and both position tensors created at
  `:359-383` are copied unchanged at `:407-415`. This proves stable-address replay
  and determinism, but not that replay consumes changed contents.

Causal scope: these mistakes explain the stage-review evidence gap completely.
They do not, by themselves, establish a decoder implementation bug. A no-op final
cache update can pass because the same K/V is already present, and a trace that
bakes position 32 can pass because every replay asks for position 32.

## Headline finding 2 (confirmed, implementation bug): padded sliding prefill can clobber live circular-cache entries

The earliest divergence is in `tt/functional_decoder.py:408-422` and `:150-165`:

1. A non-aligned logical sequence is padded upward (normally to 32 tokens).
2. For bounded sliding attention, `fill_len` is rounded upward to an effective
   cache block at `:155-159`.
3. `paged_fill_cache` is passed that padded K/V plus
   `cache_position_modulo=1024` at `:160-165`.
4. The paged-fill contract explicitly writes each input position to its wrapped
   slot. This is documented and tested in
   `tests/ttnn/unit_tests/operations/sdpa/test_bounded_sliding_kv_cache.py:279-338`
   and in
   `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/fill_cache/paged_fill_cache_device_operation_types.hpp:19-25`.

Concrete instances (the decoder input is padded to 32-token tiles; its paged
cache uses 64-token blocks):

- Logical length 1025 is padded to 1056 and all 1056 K/V rows are passed to the
  cache fill (`fill_len` computes 1088, but because that is not less than the
  1056-row tensor, the code keeps the whole 1056-row tensor). Padding positions
  1025..1055 overwrite circular slots 1..31. After the real token at position
  1025 is decoded into slot 1, slots 2..31 are still padding, although they are
  part of the live 1024-token attention window.
- Logical length 262113 has remainder 993 modulo 1024 and is rounded to 262144.
  Its 31 padded positions overwrite slots 993..1023. Most of those slots hold
  live history for the next decode.
- Logical length 262143 has remainder 1023 and is rounded by only one token. The
  padding overwrites slot 1023, which is the expired slot immediately before a
  genuine decode at absolute position 262143; that decode overwrites slot 1023
  with the real current K/V. Thus the requested exact-limit experiment is not
  explained by this bug.

The current boundary tests compare prefill outputs, whose attention uses local
Q/K/V rather than the subsequently consumed circular cache. The current 262113
decode comparison is TT versus TT and replays an already-prefilled token, so it is
not a reliable refutation. This defect needs a cache-read-after-prefill decode
test, ideally with an attention-sensitive negative control.

Smallest likely intervention: make bounded sliding prefill write exactly the
logical valid K/V set without allowing padded lanes to become later circular
writes. The exact mechanism needs a focused implementation experiment because
the paged-fill op is block/tile oriented; plausible choices are a valid-length
aware paged-fill contract, an aligned bulk fill plus exact tail updates, or
restoring wrapped live entries after the padded fill. Merely changing the Python
slice to another rounded length does not fix the ownership problem.

## HF and cache semantics relevant to the oracle

- Gemma4 attention computes Q/K/V and RoPE, then calls
  `past_key_values.update` before attention
  (`transformers/models/gemma4/modeling_gemma4.py:1230-1287`).
- A config-less `DynamicCache()` creates ordinary growing `DynamicLayer` entries
  (`transformers/cache_utils.py:1321-1327`). It does **not** acquire Gemma4's
  sliding cache semantics. This is harmless in the existing position-32 tests
  because the history is shorter than the 1024 window, but it is not an acceptable
  long sliding oracle.
- In installed Transformers 5.10.2, Gemma4 has no operative `cache_position`
  argument. The model derives absent `position_ids` from cache sequence length
  (`modeling_gemma4.py:1683-1686`), RoPE consumes those IDs, and cache update is
  append-only. An absolute position cannot be used to jump an HF `DynamicCache`
  over missing history; a reduced sliding oracle must set both its retained K/V
  and its `cumulative_length` consistently.
- `DynamicCache(config=hf_config)` maps sliding layers to
  `DynamicSlidingWindowLayer` (`cache_utils.py:867-877, 1279-1300`). That layer
  keeps the last `window - 1` states, tracks the absolute cumulative length, and
  returns those states plus the current update (`:190-257`). Therefore a decode at
  262143 after exactly 262143 prior tokens attends positions 261120..262143.
- Full-attention `DynamicLayer` concatenates all history (`:109-157`). Feeding
  262143 history tokens through the HF layer is not memory-feasible because it
  also performs quadratic prefill attention; repeatedly updating it in chunks is
  additionally a repeated-`torch.cat` design.
- The HF eager decode expands four full-attention KV heads to 32 query heads
  (`modeling_gemma4.py:809-850`). At 262143 tokens, compact BF16 K+V are about
  2.0 GiB, while materialized repeated K+V are about 16.0 GiB. The actual decode
  logits are only `32 x 262144`, about 32 MiB in FP32. The impossible object is
  the history-prefill `262143 x 262143` attention, not the one-query decode score
  vector.

## Ranked oracle and test designs

### 1. Recommended acceptance design: periodic real-weight history plus a one-query HF oracle

Use a deterministic bank of 32 or 64 distinct BF16 hidden vectors and repeat it
for exactly 262143 positions. Use a separately seeded, nonidentical final token at
absolute position 262143. This preserves real model weights, real absolute RoPE,
all 262143 logical history positions, and a meaningful nonconstant V history,
while avoiding 262143 dense K/V projections on CPU.

For TT:

1. Build the repeated history, prefill it with `valid_seq_len=262143`, and assert
   the host page table is not identity (the current one-block roll is nonidentity
   for both 16 sliding pages and 4096 full pages). Emit history length, decode
   position, period, page-table checksum, and source hash in the log.
2. Allocate the trace token and both position buffers once. Capture the complete
   decoder. Copy the distinct final token and 262143 into those allocations and
   replay. Compare that replay, not the capture enqueue result, to the oracle.
3. Spot-check at least two expected physical cache blocks (including the rotated
   block containing the final update) so a page table that is merely nonidentity
   in host construction is distinguished from correct physical placement.

For the full HF reference:

1. Use the official HF layer's input norm, projections, per-head norms, RoPE,
   output projection, post-attention norm, MLP, post-feedforward norm, residuals,
   and layer scalar.
2. Project only the small pattern bank and the final token. V is periodic and K
   becomes position-dependent only through RoPE. For this target's global layer,
   reproduce HF's exact `attention_k_eq_v=True` order: K and V share the raw
   `k_proj` result, but K then receives K norm plus RoPE while V independently
   receives the scale-free V norm. Use scale 1.0 and map each group of eight query
   heads to one of four KV heads.
3. Reshape the one-token Q as grouped-query attention and generate rotated K in
   moderate position chunks. Store only the 32-by-262144 logits, apply one global
   FP32 softmax (then BF16, matching HF), and aggregate the probabilities by
   pattern to multiply the corresponding V vectors. This avoids both a full
   prefill attention matrix and expanded 32-head K/V.
4. Pass the resulting attention vector through the official one-token layer tail.

For sliding attention, either use the same grouped one-query oracle over absolute
positions 261120..262143, or seed a config-aware HF sliding cache with exactly the
last 1023 projected states and `cumulative_length=262143`, then call the official
HF layer for the final token. The latter is only about 16 MiB of BF16 K+V.

Require decoder-output PCC >= 0.995, but also add a sensitivity gate: construct a
wrong-position/no-final-update HF control and show that the chosen inputs make it
materially worse than the correct oracle. Decoder residuals can otherwise make a
cache-addressing error retain deceptively high PCC. An attention-output comparison
or a correct-versus-negative error margin is stronger than decoder PCC alone.

Resource risk: the repeated host hidden tensor is about 2.625 GiB, which the
existing long test already allocates. The periodic oracle adds tens of MiB rather
than full history K/V. The main cost remains the TT full prefill. Generate/repeat
the pattern deterministically and release unused long output slices promptly.
If periodic input is considered too synthetic for acceptance, the same grouped
one-query calculation can stream arbitrary history chunks with online FP32
log-sum-exp state. That remains O(sequence length) in memory but is much more
expensive because it performs dense K/V projection for all 262143 tokens; validate
its reduction against stock HF at short length before using PCC from it.

### 2. Strong diagnostic, weaker acceptance oracle: read back and unpage TT K/V

Read selected/all TT physical cache pages, invert the page table and modulo map,
and run a PyTorch grouped one-query attention plus the HF layer tail. This is very
useful for localizing projection versus addressing versus SDPA and can prove the
physical page layout. It is not a sufficient standalone HF-vs-TT acceptance
oracle because TT-produced K/V appear on both sides. Full K+V readback is about
2 GiB; selected-page checks are cheap.

### 3. Rejected as the primary design: HF DynamicCache prefill in chunks

Calling the HF layer on 262143 history tokens materializes quadratic attention.
Chunking the calls still computes growing history attention and ordinary
`DynamicLayer.update` repeatedly concatenates its tensors. Preallocating a full
HF cache avoids the repeated concatenation but not eager KV head expansion and
still performs trillions of dense projection operations for random history.

## Changed stable-buffer trace matrix

Use sequential positions so cache state remains semantically valid; changing a
position backward after an unrelated replay is not a clean oracle.

1. **Page transition, both layer kinds:** prefill positions 0..126. Capture once.
   Replay token A at 127, compare with HF; copy token B and both position tensors
   with 128 into the same allocations, replay again, and compare with the
   sequential HF cache. This crosses block 1 offset 63 to the next physical page.
2. **Sliding modulo wrap:** prefill positions 0..1022. Capture once. Replay token A
   at 1023, then update the same token/position allocations to token B at 1024 and
   replay. Compare both to a config-aware HF sliding cache. This crosses circular
   slot 1023 to slot 0 and should use the rolled page table.
3. **Non-boundary/random legal position:** choose a logged seed and sample a
   position that is neither a tile/page/window boundary (for example in
   `[129, 1000]`). Prefill through its predecessor, replay at that position, then
   update to the next sequential position and a distinct token and replay again.
   The oracle must advance in the same order. Assert the copied host buffers
   differ byte-for-byte from their prior values before replay.

The static implementation supports this design: the 2-D RoPE path embeds
`position_idx` on device (`models/demos/gemma4/tt/attention/decode.py:86-117`),
`position_idx_cache` drives paged update and paged SDPA (`:119-128, 203-220,
252-270`), and `FunctionalDecoder.decode_forward` forwards both device tensors
without converting them to Python (`tt/functional_decoder.py:445-472`). No
implementation change is indicated unless a changed-buffer replay fails.

## Focused experiments in execution order

1. Add a cheap attention-sensitive regression for the padding bug: sliding
   prefill lengths 1025, 1057, and 262113 followed by a distinct next-token decode;
   compare to a config-aware HF sliding oracle and read back the wrapped tail
   slots. This should be run before trusting a near-context decode.
2. Add the page-transition and modulo-wrap stable-buffer trace tests above. They
   are short and will adjudicate whether any trace-input implementation change is
   needed.
3. Validate the periodic one-query oracle against ordinary HF layer decode at a
   small length (for both layer types) before using it at 262143 history. Require
   near-identity between the reduced oracle and official HF at the small control.
4. Run the exact 262143-history, distinct-token, position-262143 trace replay with
   the periodic oracle, nonidentity page checks, sensitivity negative control,
   and direct HF-vs-TT PCC for both layer kinds.

## Claims revisited against code

- **“The existing long decode is not genuine final-position decode evidence”:**
  confirmed by following the exact hidden-state slice, position, cache state, and
  PCC operands. Headline retained.
- **“Trace may ignore mutable positions”:** not proven as an implementation bug.
  Both positions reach their intended device consumers; the current test simply
  never changes them. Kept as a test gap, not a code defect.
- **“A config-less DynamicCache is wrong for long sliding reference”:** confirmed
  by cache construction and layer mapping. It does not invalidate short position
  32 results. Kept as an oracle-design constraint.
- **“Sliding padding corrupts all non-aligned lengths”:** narrowed. Corruption
  occurs when rounded padding wraps onto slots still needed by the following
  decode. Exact 262143 history is a safe special case because its sole padded slot
  is the next/expired slot. Headline retained with that boundary.
- **“Full long decode requires a full HF prefill matrix”:** false. Only a one-query
  score vector is mathematically required if history K/V are synthesized without
  running history attention. Rejected from the resource assumptions.

## Scope

Inspected the stage-review report, functional decoder test and implementation,
Gemma4 TT attention/decode/cache helpers, bounded-cache unit/operation contracts,
and the installed Transformers Gemma4 attention and DynamicCache implementations.
All report artifacts are confined to
`models/autoports/google_gemma_4_31b/doc/functional_decoder/`.
