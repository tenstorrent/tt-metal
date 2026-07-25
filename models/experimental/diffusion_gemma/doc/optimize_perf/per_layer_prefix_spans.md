# Per-layer prefix spans for denoise (#51080)

Roadmap item 3. Give the 25 sliding layers a bounded prefix span instead of the full `p_max`,
and correct the sliding-window reference at the same time.

## Status 2026-07-24

The item split cleanly into a fidelity half and a perf half, and they are **better landed
separately** — the gating is completely different and bundling them would make the decision
delta unattributable.

**Landed: the fidelity fix** (`DG_DENOISE_SLIDING_WINDOW`, default OFF).

* Corrected the reference: the `abs(q_abs - k_abs) <= sliding_window` staircase is gone from
  `reference/attention_mask.py` (both the reveal-mask builder and `build_canvas_denoise_mask`),
  and `_sliding_layer_needs_denoise_mask`'s threshold is now `prompt_len > sliding_window - 1`
  (it previously included `canvas_len`, which only made sense under the staircase).
* Per-layer-TYPE reveal masks in the adapter, mirroring the existing canvas-RoPE buffer
  discipline: one buffer per layer type, **all of the same `[1,1,C,p_max+C]` shape**, so
  enabling the window changes mask CONTENT only and every captured trace stays shape-valid.
  `_reveal_mask_provider(layer_idx)` dispatches on layer type.
* Read span is still the full `p_max`. That is deliberate — it makes the fidelity fix need
  **zero** new buffer machinery and zero shape changes.

**Root cause of the wrong rule, found while doing this.** `tests/test_real_transformers_parity.py`
was validating the *decoder's denoise* mask against `transformers.masking_utils.create_masks_for_generate`
— the **generic** mask builder (DiffusionGemma's same-named override lives on
`DiffusionGemmaEncoderModel` and merely delegates to it). That function describes the
**encoder's** prompt self-attention, where the sliding window genuinely *is* a per-(q,k)
distance predicate, so its mask is a staircase. The denoise pass is the **decoder**, which uses
`create_diffusion_decoder_attention_mask` and applies no staircase. Validating the decoder
against the encoder reference is how the staircase got in. That test now asserts the
encoder-side staircase *as encoder behaviour* and points at the decoder pinning test.

**Not landed: the perf half** (bounded 1024-row sliding read + block-resident buffers). Design
below stands unchanged. Note it becomes much easier to gate *after* the fidelity fix: once the
out-of-window keys are already masked to `NEG`, not reading them at all changes only the flash
K-chunk accumulation order, so it is a near-bit-exact perf change rather than a decision change.

### Device results (full 30L, traced, `serving_smoke --upfront`)

Harness: `doc/optimize_perf/verify_denoise_sliding_window.sh`. Committed prefix at block *k* is
`P = cache_len + 256(k-1)`, so a single run walks both regimes.

At the real window (`W = 1024`), 5 blocks — `P = 32/288/544/800/1056`:

| regime | blocks | result |
|---|---|---|
| `P <= W-1` (unbound) | 1–4 | **bit-identical** — required, and confirmed |
| `P >= W` (bound) | 5 | identical committed tokens |

Block 5 evicts exactly `P-(W-1) = 33` of 1312 attended columns (**2.5%**) and flipped **zero** of
its 256 argmax decisions. That is decision *stability*, not an inert implementation — and it is
why the gate in that script is deliberately **one-sided**. Its first version asserted "bound
blocks must differ" and reported a false FAIL; asserting that conflates *the mask changed* with
*the decisions changed*. The mask change is verified directly and host-side instead.

To prove the plumbing end-to-end, re-run with `DG_DENOISE_SLIDING_WINDOW_OVERRIDE=128`, which
evicts ~88% of committed keys:

| regime | blocks | result |
|---|---|---|
| `P <= 127` (unbound) | 1 | bit-identical |
| `P >= 128` (bound) | 2–5 | **block 3 differs** → enforcement is live on device |

Only 1 of 4 bound blocks moved even at 88% eviction, which says the committed output is highly
insensitive here — expected for the degenerate/repetitive text produced under
`--disable-eos-stop`. Read it as "the mechanism is live and the decision impact is small", not as
"the window barely matters"; a quality-representative measurement needs the fp32 HF
decision-agreement run, which is the actual acceptance gate for flipping the default.

This is a **fidelity fix that is also cheaper**, not a performance optimization that happens to
change decisions. See #51080 for the bug statement.

## Verified HF reference semantics

Pinned by `tests/test_hf_sliding_window_reference.py` (CPU-only, 4 tests, no checkpoint):

1. A sliding layer's cache retains exactly **`sliding_window - 1` = 1023** committed tokens —
   `DynamicSlidingWindowLayer.update` keeps `full_key_states[:, :, -sliding_window + 1:, :]`
   (`transformers/cache_utils.py:245-247`).
2. For the ordinary **unpadded** `DynamicCache` path (what DG serving is),
   `create_diffusion_decoder_attention_mask` returns `{"full_attention": None,
   "sliding_attention": None}` — there is **no sliding mask at all**. The window is purely a
   cache-truncation effect.
3. When a padding mask *is* materialized, the sliding mask is expanded from a 1-D per-key
   vector (`decoder_attention_mask[:, None, None, :].expand(...)`), so it has **no query-index
   dependence**. Every canvas row sees the same key set.

> **Therefore: HF sliding-layer denoise visibility = the 1023 most recent committed tokens,
> ALL-ATTEND, plus the full canvas. There is no `|q_abs - k_abs| <= W` staircase.**

The repo's own reference asserted the staircase and must be corrected as part of this work:

| Site | Problem |
|---|---|
| `reference/attention_mask.py:11-17` | docstring states the `abs(q_idx - kv_idx) <= sliding_window` rule |
| `reference/attention_mask.py:78-81` | `enforce_sliding_window` implements that staircase |
| `reference/attention_mask.py:158-163` | same rule in `build_canvas_denoise_mask` |
| `tt/denoise_forward.py:153-154` | `_sliding_layer_needs_denoise_mask` threshold derives from the staircase; correct threshold is simply `P >= 1024` |

Enabling `enforce_sliding_window=True` as it stands would hide up to 255 keys per canvas row
that HF *does* attend — trading one fidelity bug for another.

## Geometry

Canvas queries sit at absolute positions `q_abs = P + i`, `i in [0, 256)`, where `P` is the
committed prefix length. `P` is always 32-aligned (prefill pads `cache_len`, then each commit
advances by `canvas_len = 256`), so a tile-aligned window offset is always available.

A tile-aligned span cannot be 1023, so the design reads **1024** rows and masks the one extra:

```
span = 1024                       (tile-aligned; HF retains 1023)
lo   = max(0, P - span)           (32-aligned because P is)
prefix column r  <->  absolute position  lo + r,   r in [0, span)
```

**Mask truth table for a sliding layer** — prefix column `r` is attended iff both hold:

| predicate | meaning |
|---|---|
| `lo + r < P` | the position is actually committed (the existing reveal predicate) |
| `lo + r >= P - 1023` | HF still retains it in the sliding cache |

Canvas columns are always attended (`max abs(i - j) = 255 < 1024`).

Two regimes fall out, and both are desirable:

* **`P <= 1023`** (window not yet binding): `lo = 0`, so column `r` ↔ position `r`, and the
  retention predicate is vacuous. The mask reduces to `r < P` — **exactly today's reveal
  predicate**. Behaviour is unchanged, which is why the gate below can demand bitwise identity
  in this regime.
* **`P >= 1024`** (steady state): `lo = P - 1024`, so column `r` ↔ `P - 1024 + r`. The commit
  predicate is always true and the retention predicate becomes `r >= 1`. The mask is therefore
  **"drop column 0", independent of `P`** — it stops changing after the first block.

`tests/test_hf_sliding_window_reference.py::test_tt_window_span_formula_matches_hf_key_set`
checks this arithmetic against HF's key set for `P in {1, W-1, W, W+1, 2W, 5W}`.

## Why the read must move outside the trace

`lo` slides with `P`, and a slice offset is baked into a captured trace. So the sliding-layer
prefix read cannot stay where it is (inside the traced per-layer loop at
`tt/denoise_forward.py:604`). It becomes per-layer **block-resident buffers**:

* allocate persistent `[1, kvh_l, 1024, hd_l]` K/V buffers per sliding layer **before**
  `begin_trace_capture`, in `traced_denoise._prepare_fixed_reveal`;
* refresh their *contents* once per block **outside** any trace, via `ttnn.copy` from a
  `ttnn.slice(cache, lo, lo + 1024)`;
* the reader returns the buffers themselves (`owns_result` False — the item-2 borrow contract
  already added for the full-span case).

The refresh hooks already exist and already have exactly this "contents change, baked addresses
stay stable" contract: `advance_prefix_after_commit` (`:1480-1497`, already refreshes the
reveal mask per block) and `rebind_prompt` (`:1451-1478`, per request).

Structural checks done: `read_prompt_kv_cache_by_layer` already takes `layer_idx` (`:794-802`),
and `_denoise_layer_forward` passes `prompt_source` / `attn_mask` per layer into
`denoise_attention`, which derives `k_seq_len` and the SDPA program config from the actual
tensors — so per-layer differing spans and mask shapes need no new plumbing. Prefix K keeps the
RoPE baked in at cache-write time, so a window sub-range needs no re-rotation (the
`canvas_rope_provider` note at `:495-502` states the invariant).

The 5 full-attention layers keep `p_max` and today's in-trace read, so this change does **not**
remove the `p_max` cost coupling — it reduces the SDPA key rows from `30·(p_max+256)` to
`25·1280 + 5·(p_max+256)`.

## Gate

Two tiers, because the change is bit-exact in one regime and decision-changing in the other.
Must not be bundled with any other change.

1. **`P <= 1023`** — the window never binds, so require **bitwise** identical
   `committed_sha256` against the current golden.
2. **`P >= 1024`** — decision-changing. Measure decision agreement against **fp32 HF
   DiffusionGemma**, not against today's TT output, since today's TT output is the defect being
   corrected. Agreement is expected to *improve*.

Risk to close explicitly: a stale block-resident buffer would silently replay old prompt KV.
Carry a fail-loud `(cache_generation, prompt_len)` stamp on the reader and assert it before
every block.

## Inherited prerequisite: the buffer-aliasing trap (found on device by item 2)

The block-resident buffers here are handed to the layer forward **borrowed** (the reader reports
`owns_result` False), exactly like item 2's full-span read. Item 2 shipped only after a
device-only failure exposed a second, non-obvious owner:

```python
prefix_k_concat = ttnn.to_memory_config(prefix_k, canvas_k.memory_config())
...
if prefix_k_concat is not prefix_k:      # object identity — NOT sufficient
    prefix_k_concat.deallocate(True)     # frees an ALIAS of prefix_k's buffer
```

`ttnn.to_memory_config` returns a **fresh Tensor object that aliases the input buffer** when no
conversion is needed — device-observed as `distinct_buffer=False, same_object=False`. The
identity check therefore deallocated the borrowed cache, and the next op died with
`TT_FATAL: Input Tensor is not allocated`. Guarding only the obvious free site
(`denoise_hidden_forward`'s per-layer `finally`) was **not** enough.

Fixed by `_is_distinct_buffer` in `tt/diffusion_attention.py`, which compares
`buffer_address()` and refuses to free when ownership cannot be proven (a leaked conversion is
recoverable; freeing the model KV cache is not). Regression-tested CPU-only in
`tests/test_prefix_buffer_ownership.py`.

**Consequence for this work:** any new borrowed tensor must be audited for *buffer* owners, not
just for explicit `deallocate` calls on the object itself. Grep the consumer path for
`to_memory_config`, `reshard`, `clone`, and `slice` on the borrowed tensor, and assert
allocation state across a block boundary in the device test.
