# Canvas-tail workspace for denoise K/V (#51080 roadmap item 4)

Remove the last per-step prefix copy. Today `denoise_attention` builds its K/V as

```python
prefix_k_concat = ttnn.to_memory_config(prefix_k, canvas_k.memory_config())
tt_k = ttnn.concat([prefix_k_concat, canvas_k], dim=2)     # copies the WHOLE p_max prefix
```

so every step, every layer re-materialises `[1, nkv, p_max + C, hd]`. Item 2 removed the *read*
copy (the `ttnn.clone` of the cache); this removes the *concat* copy, the other half.

**Idea.** Size the KV cache as `p_max + canvas_len` in the seq dim and write the canvas K/V into
the TAIL with `ttnn.fill_cache(cache, canvas_kv, 0, update_idx=p_max)`. SDPA then reads one
already-contiguous tensor and no prefix bytes move at all.

The reveal mask needs no change: it is already `[1, 1, C, p_max + C]`, exactly the layout of the
workspace (prefix columns then canvas columns). The canvas lives at a fixed scratch offset
`p_max` while its *content* carries the correct absolute-position RoPE from
`canvas_rope_provider`, so position semantics are unaffected.

## Status 2026-07-24: primitive VERIFIED on device, wiring NOT landed

The core assumption was device-tested in isolation first, because its failure mode is silent
wrong output rather than a crash: a captured trace bakes buffer addresses, so a traced
`fill_cache` into a persistent tensor followed by a traced read of that same tensor must have
every replay observe *its own* write, not a stale tail.

`tests/test_device_fill_cache_in_trace.py` — **3 passed** on P150x4:

| check | result |
|---|---|
| eager `fill_cache(update_idx=p_max)` writes only the tail, prefix byte-identical | pass |
| traced fill + read, replayed twice with different canvas contents | each replay sees **its own** write; prefix intact; replays differ; cache `buffer_address()` stable |
| traced fill + **SDPA** over the same tensor | pcc > 0.99 vs torch reference on both replays; outputs differ across replays |

So the workspace idea is sound: a trace may write a scratch tail and read it back in the same
capture, and refreshing only the *contents* of a pre-capture input buffer is enough to drive it.

**One caveat the run surfaced.** Metal warned `Allocating device buffers is unsafe due to the
existence of an active trace. These buffers may be corrupted once a trace is executed.` — the
test allocates its output inside the capture (`ttnn.clone(cache)`). It passed, but the real
wiring must pre-allocate every output buffer BEFORE `begin_trace_capture`, which is the same
session-8 rule the up-front controller already follows (`_warm_persistent_outputs` allocates the
persistent outputs first). Do not copy the test's allocation pattern into the model.

## Revised design: a DG-owned workspace, NOT a resized KV cache

The original sketch (resize the model KV cache to `served + canvas_len` and write the canvas into
its tail) has two avoidable problems, and item 3's bounded-span work removed the third:

1. it reinterprets `--max-seq-len` and moves `_validate_next_block_capacity` / the QB2
   memory-budget assertions;
2. it writes into the model's KV cache during a pass `kv_phase` declares `DENOISE_READONLY`;
3. ~~it should follow item 3's bounded span~~ — item 3's perf half has landed, so the layer split
   is now known: the 25 sliding layers read `span + C`, the 5 full layers read `p_max + C`.

Both remaining problems dissolve by allocating a **separate DG-owned workspace tensor per layer**
instead of resizing the model cache:

```
workspace[l] = [1, nkv_l, read_span_l + C, hd_l]      # allocated ONCE, pre-capture

per BLOCK  (outside any trace):  copy cache[lo : lo + read_span_l] -> workspace[l][0 : read_span_l]
per STEP   (inside the trace):   ttnn.fill_cache(workspace[l], canvas_kv, 0, update_idx=read_span_l)
                                 SDPA reads workspace[l] directly — no concat, no prefix copy
```

* The model KV cache is never written, so `DENOISE_READONLY` holds — the scratch is DG's own
  tensor. No carve-out needed.
* The cache sizing contract is untouched; `--max-seq-len` keeps its meaning.
* The prefix copy does not disappear, it moves from **per step to per block** — 48x fewer at the
  released step count, and it rides the refresh hook item 3 already added
  (`_refresh_prefix_windows`).
* The reveal mask needs no change: `[1, 1, C, read_span + C]` is already exactly the workspace
  layout, and `k_seq_len` is unchanged, so the SDPA program config is unchanged too.

Extra DRAM is the workspace itself. Full layers have 1 local KV head at `head_dim=512` (1 KiB per
row per tensor), sliding layers 2 heads at 256 (also 1 KiB), so it is ~2 KiB per row per layer:
about **44 MB** at `p_max=4096` for all 30 layers, ~1.3 GB at 128K. Acceptable at the shipped
span; at very long context it is a real trade of DRAM for bandwidth and should be measured
against the 256K envelope before use there.

## Measured outcome 2026-07-24: bit-exact, but NO benefit on top of item 3 — keep it OFF

Landed behind `DG_DENOISE_CANVAS_TAIL` (default OFF) and device-gated with
`doc/optimize_perf/verify_canvas_tail.sh` (full 30L traced, `p_max=4096`, 6 blocks, both arms with
`DG_DENOISE_SLIDING_WINDOW=1 DG_DENOISE_SLIDING_SPAN=1` so the workspace is the only variable):

| run | control (per-step concat) | candidate (workspace) | delta |
|---|---|---|---|
| 1 | 2.716 s | 2.749 s | **+1.2% slower** |
| 2 | 2.788 s | 2.833 s | **+1.6% slower** |

`committed_sha256` **identical** in both runs, every one of 6 blocks — so the workspace is
correct: it hands SDPA byte-identical K/V, at an unchanged shape, hence even the flash
accumulation order is preserved.

**Why the win did not materialise: item 3 already took it.** The payoff was estimated by analogy
to item 2, which removed two whole-*cache* copies (4352 rows/layer/step at `p_max=4096`). But
item 3's bounded span shrank the concat to `span + C = 1280` rows on 25 of 30 layers *before* this
change ran, so the remaining per-step concat is ~3.4x smaller than the one that was priced. What
is left is roughly cancelled by the new per-block prefix refresh, which must write
`30 layers x 2 tensors x span rows` in core-bounded chunks.

So the correct reading is not "the workspace is bad" but "**the concat stopped being the
bottleneck once the span was bounded**". Consistent direction across two independent runs, and the
gap (~1.4%) is larger than the ~1% within-arm block spread, so it is probably a real small
regression rather than noise — most likely the chunked per-block fill plus the extra DRAM
footprint's effect on allocator locality.

**Recommendation: leave `DG_DENOISE_CANVAS_TAIL` off.** It costs ~106 MB of persistent scratch at
`p_max=4096` and buys nothing measurable. It is worth re-testing in exactly one situation: a
configuration where the full-attention layers dominate (very large `p_max` with item 3 enabled,
where those 5 layers still concat `p_max + C`), since that is the case this design actually
targets and the one the 2.43x span reduction does not help.

This is the same lesson as the q-chunk sweep in `qchunk_sweep_20260724.md`: a prefix-cost model
that has not been measured end-to-end after the previous optimisation landed will over-predict.

## Original expected payoff (superseded by the measurement above)

Item 2 removed 2 of the 4 prefix passes and measured **~5.5%** off the steady block
(2.323 s vs 2.460 s, full 30L traced, `p_max=2048`). The concat is the remaining 2 passes, so
this is plausibly another ~5% at that span — materially more than the ~1% estimated during
design review, which had undercounted by assuming a single pass.

That estimate should be re-derived at the shipped `p_max=4096` before committing effort: the
term scales with `p_max`, and the q-chunk sweep (`qchunk_sweep_20260724.md`) is a caution
against trusting a prefix-cost model that has not been measured end to end.

## Prerequisite for whoever picks this up

Audit every consumer of the workspace tensor for **buffer** owners, not just for explicit
`deallocate` calls on the object. Item 2 shipped only after a device-only failure showed that
`ttnn.to_memory_config` returns a fresh Tensor object *aliasing* the input buffer, so an
`is not` identity check freed the model KV cache
(`TT_FATAL: Input Tensor is not allocated`). See `_is_distinct_buffer` in
`tt/diffusion_attention.py` and `tests/test_prefix_buffer_ownership.py`.
