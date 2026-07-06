# APC / frozen prompt-prefix KV reuse — work log (#47466)

Agent B. Branch `dg-vllm-apc`. Device: shared QB2 (4× Blackhole, `(1,4)` mesh) —
every device run wrapped in `flock /tmp/dg-mesh.lock timeout 900`.

## Goal

Reuse the already-computed prompt-prefix K/V across `generate` calls when two
requests share a prompt prefix, instead of re-encoding it. Prototype at the DG
serving layer behind `DG_PREFIX_CACHE` (default OFF). Do NOT wire vLLM's block pool
(that is #47488). Acceptance bar: bit-exact committed argmax with the cache ON vs
OFF for two requests sharing a long prefix, plus a logged prefill-time saving.

## What I changed

| File | Change |
|---|---|
| `tt/prefix_cache.py` (new) | `PrefixKVCache` — host-side registry of the aligned prompt tokens the contiguous cache currently holds; `plan()` decides bit-exact reuse; `prefix_cache_enabled()` reads `DG_PREFIX_CACHE`. No device state. |
| `tt/serving.py` | `BlockDiffusionServingSession.__init__` gains `prefix_cache=`; `prefill()` skips `prefill_prompt_tokens` when the aligned prompt is a full prefix of the resident cache; exposes `prefill_reused` / `prefill_time_s`. |
| `tt/generator_vllm.py` | adapter holds one shared `PrefixKVCache`, passed to every session; `model_capabilities["supports_prefix_caching"]` stays **False** (vLLM block pool not wired — #47488). |
| `demo/prefix_cache_smoke.py` (new) | reduced-surface device driver: ON-vs-OFF bit-exact committed-argmax check + prefill-saving, for exact-match and aligned-proper-prefix cases. |
| `tests/test_prefix_cache.py` (new) | 9 CPU unit tests pinning the reuse decision (which cases are bit-exact-safe vs the #47488 partial-prefix case). |
| `doc/vllm_integration/prefix_cache/README.md` (new) | design note: RoPE / three-phase-KV / sliding-window correctness argument + the #47488 boundary. |

## Correctness argument (see README for the full version)

The frozen prompt K/V is written to `tt_model.tt_kv_cache[layer]` at absolute
positions `[0:cache_len]` by causal prefill and read read-only by denoise. Because
(1) RoPE is absolute (position `i`'s rotation depends only on `i`), (2) causal
prefill makes position `i`'s K/V a pure function of `tokens[0:i]` regardless of the
total prefill length, and (3) both sliding-window and full-attention layers attend
to a subset of `[0:i]`, a new request whose *aligned* prompt is a byte-identical
leading span of the resident cache's aligned prompt has bit-identical K/V for that
span — so its prefill can be skipped. Alignment constraint: the reuse span must
match over the whole `cache_len` including 32-tile pad, so only exact-full-match
and aligned-proper-prefix qualify (enforced by `plan`).

## Where #47488 is required (documented, stopped cleanly)

The productionly-valuable **shared-prefix + differing/extending suffix** case
cannot be bit-exact at the DG serving layer: the suffix must cross-attend to the
cached prefix during prefill = chunked/prefix prefill, which the Gemma4 backbone
does not support (`models/demos/gemma4/tt/model.py:1291,1298,1385` — `del
start_pos`, "Gemma4 doesn't chunk-prefill") and which I may not add by editing the
shared backbone. A DG-local commit-decode "suffix prefill" is functionally correct
but not bit-exact to the batched-prefill matmul in bf16 (and #48291 shows small
drift can flip a diffusion decision), so it fails the bar. General APC belongs in
the #47488 paged-cache ownership path. The prototype logs
`partial-prefix miss: matched N ... → full prefill (needs chunked prefill / #47488)`
and counts `partial_prefix_misses` so the reuse a paged path could capture is
visible.

## Evidence

### CPU unit tests (device-free) — PASS

`python -m pytest models/experimental/diffusion_gemma/tests/test_prefix_cache.py -q`
→ **9 passed**. Covers: empty resident, exact match, aligned proper prefix,
non-aligned proper prefix (rejected), extending suffix (partial-prefix miss),
longer-than-resident (rejected), record re-anchoring, stats/time tracking, and
input validation.

### Device bit-exact check (under flock)

Command (see README for the flow):
```
flock /tmp/dg-mesh.lock timeout 900 \
  env TT_METAL_HOME=$PWD TT_METAL_RUNTIME_ROOT=$PWD ARCH_NAME=blackhole PYTHONPATH=$PWD \
      DG_CKPT=/home/zni/dg_models/diffusiongemma-26B-A4B-it TT_LOGGER_LEVEL=ERROR \
  python -m models.experimental.diffusion_gemma.demo.prefix_cache_smoke \
    --mesh P150x4 --num-layers <N> --max-seq-len 1024 \
    --num-blocks 1 --max-denoising-steps 2 --gumbel-mode argmax --case both \
    --local-files-only --metrics-json doc/vllm_integration/prefix_cache/prefix_cache_smoke.json
```
Status: **PENDING device run** (results + `DG_PREFIX_CACHE_SMOKE_*` marker filled
in after the flock run below).

## SHAs

- (pending) prototype + CPU tests + design note.

## OPEN QUESTIONS / BLOCKED

- General shared-prefix+suffix APC is BLOCKED on #47488 (paged-cache ownership) /
  chunked prefill — documented above; not attempted (would need a forbidden gemma4
  edit or a non-bit-exact commit-decode suffix prefill).
