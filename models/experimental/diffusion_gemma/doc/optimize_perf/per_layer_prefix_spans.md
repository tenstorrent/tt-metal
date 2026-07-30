# Per-layer prefix spans for denoise (#51080 item 3)

Status: current — the fidelity half (`DG_DENOISE_SLIDING_WINDOW`, **default ON** since 2026-07-27,
`=0` opts out) and the perf half (bounded 1024-row sliding read + per-layer block-resident buffers,
landed 2026-07-29, **unconditional**) are both shipped. The canvas-tail workspace absorbed here is
provenance-only: it was measured, rejected and removed from the tree.
Owns: the sliding-window retention rule and its HF reference, the bounded-read geometry and gate,
the ttnn buffer-aliasing trap (`to_memory_config` returns a fresh Tensor aliasing the input
buffer), and the canvas-tail workspace record.
See also: [refuted list](../REFUTED.md), [optimize_perf hub](README.md).

Over the 100-line cap: three refutations, two open contradictions and a verified HF-semantics
reference share this file.

## What shipped

The item split into a fidelity half and a perf half, deliberately landed separately so the decision
delta stays attributable.

**Fidelity.** Live in `tt/denoise_forward.py::denoise_sliding_window_enabled`. The
`abs(q_abs - k_abs) <= sliding_window` staircase is gone from four sites —
`reference/attention_mask.py:11-17` (docstring), `:78-81` (`enforce_sliding_window`), `:158-163`
(`build_canvas_denoise_mask`), and `tt/denoise_forward.py:153-154`
(`_sliding_layer_needs_denoise_mask`, whose correct threshold is simply `P >= 1024`).
`tests/test_real_transformers_parity.py`, which imported the staircase in the first place, now
asserts it *as encoder behaviour* and points at the decoder pinning test. Per-layer-TYPE reveal
masks live in the adapter, all of the same `[1,1,C,p_max+C]` shape, so enabling the window changes
mask CONTENT only and every captured trace stays shape-valid.

**Perf.** The bounded read has **no flag of its own** — it follows `DG_DENOISE_SLIDING_WINDOW`
because, with out-of-window keys already masked to NEG, not reading them changes only the flash
K-chunk accumulation order, and an all-NEG chunk moves neither the running max nor the running sum.
The result is **byte-identical**: equal sha256 on 10/10 completions in two independent pairings plus
an earlier `committed_sha256` match at p_max 4096. There was no decision to gate. It must
nonetheless stay gated on **retention**: without the retention mask a bounded read would silently
CHANGE visibility instead of implementing it, and
`DenoiseLogitsAdapter.prepare_reveal_mask_buffers` raises on a span without enforced retention.

Measured **−18.9% s/block at p_max 16384**, SDPA key rows per step falling **499200 → 115200**. DRAM
cost is net **+42.5 MiB per chip resident** (50 MiB of window buffers against a 7.5 MiB smaller
sliding mask — reveal masks are keyed by layer TYPE, so there is one, not 25). Key-row formula: per
step, `30·(p_max+256)` → `25·1280 + 5·(p_max+256)`. The 5 full-attention layers keep `p_max` and
their in-trace read, so this does **not** remove the `p_max` cost coupling. [README](README.md)
additionally records 130560 → 53760 (2.43x) at p_max 4096.

## Verified HF reference semantics

Pinned CPU-only, no checkpoint, by `tests/test_hf_sliding_window_reference.py` (4 tests):

1. A sliding layer's cache retains exactly `sliding_window - 1` = **1023** committed tokens —
   `DynamicSlidingWindowLayer.update` keeps `full_key_states[:, :, -sliding_window + 1:, :]`
   (`transformers/cache_utils.py:245-247`).
2. On the ordinary unpadded `DynamicCache` path — which is what DG serving is —
   `create_diffusion_decoder_attention_mask` returns `{"full_attention": None, "sliding_attention":
   None}`: there is **no sliding mask at all** and the window is purely a cache-truncation effect.
3. When a padding mask IS materialized, the sliding mask is expanded from a 1-D per-key vector
   (`decoder_attention_mask[:, None, None, :].expand(...)`), so it has **no query-index dependence**
   and every canvas row sees the same key set.

> **Therefore: HF sliding-layer denoise visibility = the 1023 most recent committed tokens,
> ALL-ATTEND, plus the full canvas. There is no `|q_abs - k_abs| <= W` staircase.**

Root cause of the wrong rule, and the reason it survived so long: the repo was validating the
DECODER's denoise mask against the ENCODER's reference — see [refuted list](../REFUTED.md).

## Geometry

Canvas queries sit at `q_abs = P + i`, `i in [0,256)`, where `P` is the committed prefix length. `P`
is always 32-aligned (prefill pads `cache_len`, each commit advances by `canvas_len = 256`), so a
tile-aligned window offset always exists. A tile-aligned span cannot be 1023, so the design reads
`span = 1024` rows with `lo = max(0, P - span)` and masks the extra column: prefix column `r` is
attended iff `lo + r < P` (committed) AND `lo + r >= P - 1023` (HF still retains it). Canvas columns
are always attended since `max |i - j| = 255 < 1024`.

Two regimes: at `P <= 1023` the retention predicate is vacuous and the mask reduces to `r < P`,
exactly today's reveal predicate (which is why a bitwise gate is legitimate there); at `P >= 1024`
the mask becomes "drop column 0" independent of `P` and stops changing after the first block. Pinned
by `tests/test_hf_sliding_window_reference.py::test_tt_window_span_formula_matches_hf_key_set` for
`P in {1, W-1, W, W+1, 2W, 5W}`.

`lo` slides with `P` and a slice offset is baked into a captured trace, so the sliding-layer prefix
read moves out of the trace: persistent `[1, kvh_l, 1024, hd_l]` K/V buffers allocated before
`begin_trace_capture` in `traced_denoise._prepare_fixed_reveal`, refreshed once per block OUTSIDE
any trace via `ttnn.copy` from `ttnn.slice(cache, lo, lo + 1024)`, returned **borrowed**
(`owns_result` False). Prefix K keeps its RoPE baked in at cache-write time, so a window sub-range
needs no re-rotation.

## Device gate and its traps

Reproduction — env: see [plan](../../plan.md).

```bash
models/experimental/diffusion_gemma/doc/optimize_perf/verify_denoise_sliding_window.sh
```

Full 30L traced, `serving_smoke --upfront`. The committed prefix at block *k* is
`P = cache_len + 256(k-1)`, so one 5-block run walks both regimes at `P = 32/288/544/800/1056`.

At the real window `W = 1024`: blocks 1–4 (`P <= W-1`) **bit-identical** as required; block 5
(`P >= W`) committed identical tokens — it evicts exactly `P-(W-1) = 33` of 1312 attended columns
(2.5%) and flipped **zero** of its 256 argmax decisions. Plumbing proof: re-run with
`DG_DENOISE_SLIDING_WINDOW_OVERRIDE=128`, which evicts ~88% of committed keys — block 1 stays
bit-identical and block 3 DIFFERS, proving enforcement is live on device.

> **INTERPRETATION TRAP.** Only 1 of 4 bound blocks moved even at 88% eviction, and that run used
> `--disable-eos-stop` degenerate/repetitive text. Read it as "the mechanism is live and the decision
> impact is small", **not** as "the window barely matters".

The script's gate is deliberately **one-sided**; asserting the opposite produced a false FAIL — see
[refuted list](../REFUTED.md).

> **OPEN CONTRADICTION (unexplained):** what gated the default flip. This file's account is that the
> acceptance gate is a tier-2 decision-agreement run against **fp32 HF DiffusionGemma** (never
> against today's TT output, which is the defect being corrected), still owed.
> [README](README.md) says the flip happened on the GPQA-Diamond decision-agreement run recorded in
> [device_gumbel_restored.md](../decision_fidelity/device_gumbel_restored.md) §10. Both accounts are
> in the tree and **not explained**.

Risk to close explicitly: a stale block-resident buffer would silently replay old prompt KV — carry
a fail-loud `(cache_generation, prompt_len)` stamp on the reader and assert it before every block.

## The buffer-aliasing trap (canonical home)

`ttnn.to_memory_config` returns a **fresh Tensor object that ALIASES the input buffer** when no
conversion is needed — device-observed as `distinct_buffer=False, same_object=False`. An `is not`
identity check therefore deallocated the borrowed KV cache and the next op died with
`TT_FATAL: Input Tensor is not allocated`. Guarding only the obvious free site
(`denoise_hidden_forward`'s per-layer `finally`) was NOT enough.

Fixed by `_is_distinct_buffer` in `tt/diffusion_attention.py`, which compares `buffer_address()` and
refuses to free when ownership cannot be proven (a leaked conversion is recoverable; freeing the
model KV cache is not). Regression-tested CPU-only in `tests/test_prefix_buffer_ownership.py`.
**Consequence:** audit every new borrowed tensor for BUFFER owners, not just for explicit
`deallocate` calls on the object — grep the consumer path for `to_memory_config`, `reshard`, `clone`
and `slice`. The sibling full-span-`ttnn.slice` aliasing trap is in
[commit batching](commit_batching.md).

## Absorbed: the canvas-tail workspace (item 4) — measured REJECTION

The idea was a DG-owned per-layer scratch tensor `[1, nkv_l, read_span_l + C, hd_l]` allocated once
pre-capture, whose tail receives the canvas K/V via `ttnn.fill_cache(update_idx=read_span_l)` inside
the trace, so denoise SDPA reads one contiguous tensor and the per-step prefix concat disappears.
The prefix copy would not vanish, it would move from per step to per block.

Measured (full 30L traced, `p_max=4096`, 6 blocks): run 1 control 2.716 s vs candidate 2.749 s
(**+1.2% slower**), run 2 2.788 s vs 2.833 s (**+1.6% slower**), with `committed_sha256` identical
in both runs for every one of 6 blocks. The ~1.4% gap exceeds the ~1% within-arm block spread and is
consistent in direction across two independent runs, so it reads as a real small regression — most
likely the chunked per-block fill plus the extra DRAM footprint's effect on allocator locality.

**Why it lost:** the bounded sliding read above had already taken the win, shrinking the per-step
concat to `span + C = 1280` rows on 25 of 30 layers *before* this change ran — ~3.4x smaller than
what the design priced — and what remained is roughly cancelled by the new per-block prefix refresh
(30 layers × 2 tensors × span rows written in core-bounded chunks).

> **GENERALISABLE TRAP:** a prefix-cost model that has not been re-measured end-to-end AFTER the
> previous optimisation landed will over-predict. Same lesson as the q-chunk sweep.

> **OPEN CONTRADICTION (unexplained):** the same source gave two DRAM costs for the identical
> configuration — the design section computed ~2 KiB per row per layer giving about **44 MB** at
> `p_max=4096` for all 30 layers (and ~1.3 GB at 128K), while the recommendation section said
> **~106 MB** of persistent scratch at `p_max=4096`. Both readings are recorded; neither was
> reconciled, and the discrepancy is **not explained**.

Superseded estimate, one line: item 2 (borrowing the cache instead of cloning it) removed 2 of the 4
prefix passes and measured ~5.5% off the steady block (2.323 s vs 2.460 s, full 30L traced,
`p_max=2048`); item 4 was priced at another ~5% and delivered a small loss instead.

The one configuration where re-testing would still be justified: very large `p_max` with the bounded
sliding read enabled, where the 5 full-attention layers still concat `p_max + C` and the 2.43x span
reduction does not help them. The device-verified `fill_cache`-in-trace primitive that survives this
rejection is recorded in [commit batching](commit_batching.md); the flag and its verify script are
gone from the tree.

Canonical denoise mask geometry (all-attend vs HF bidirectional sliding visibility; never pass
`sliding_window_size` on the denoise path): see [plan](../../plan.md).
