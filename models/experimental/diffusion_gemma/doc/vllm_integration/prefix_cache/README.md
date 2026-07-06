# DiffusionGemma — APC / frozen prompt-prefix KV reuse (#47466)

Agent B. Prototype behind `DG_PREFIX_CACHE` (default **OFF**). Branch `dg-vllm-apc`.

## TL;DR

Today `supports_prefix_caching=False` (`tt/generator_vllm.py`) and DiffusionGemma
uses a **model-owned contiguous** KV cache (`tt_model.tt_kv_cache`, one active
sequence). This adds a DG-serving-layer prefix-KV cache, keyed by the *aligned
prompt token-id sequence*, that **skips re-prefilling a prompt whose aligned
token span is a full prefix of the K/V already resident in the contiguous cache**.

- **What is bit-exact and shipped (behind the flag):** reuse when the incoming
  request's aligned prompt is byte-identical to a *leading span* of the resident
  cache's aligned prompt (`new_aligned == resident_aligned[:new_cache_len]`). This
  covers the two provable cases — **exact full-prompt match** and **aligned proper
  prefix** (new prompt ⊑ resident prompt). Reuse skips the entire prefill forward.
- **What fundamentally needs #47488 (documented, not shipped):** reuse when the
  new prompt *extends* the shared prefix with a differing suffix (the common
  system-prompt-shared case). That requires the suffix tokens to cross-attend to
  the cached prefix during prefill = **chunked / prefix prefill**, which the
  Gemma4 backbone does not support (`prepare_inputs_prefill` does `del start_pos`;
  "Gemma4 doesn't chunk-prefill", `models/demos/gemma4/tt/model.py:1291,1298`) and
  which we may **not** add by editing the shared backbone. The prototype detects
  this case, logs the matched-prefix length that *could* be saved, and falls back
  to a full prefill. See **§ Where #47488 is required**.

## Why prefix reuse is correct here (the three-part argument)

The frozen prompt K/V lives in the model-owned contiguous cache
`tt_model.tt_kv_cache[layer] = (K, V)` of shape `[B, heads, max_seq, head_dim]`,
written by `prefill_prompt_tokens` at absolute positions `[0:cache_len]`. The
denoise phase reads `[0:cache_len]` from it (`read_prompt_kv_cache_slice`) as the
frozen prefix the canvas cross-attends to. Reusing that prefix K/V is correct
iff the reused span is *bit-identical* to what a fresh prefill of the new prompt
would have written. Three properties guarantee that for the shipped cases:

1. **RoPE offsets are absolute and prefix-anchored.** Prefill applies RoPE at
   absolute positions `0..cache_len-1`; position `i`'s rotation depends only on
   `i`. A shared prefix token sits at the same absolute position `i` in both
   requests, so its rotated K is identical. (This is the invariant the whole
   diffusion RoPE bookkeeping relies on — `_get_rope_mats(seq_len=...)` slices the
   same absolute-position cos/sin table; `denoise_forward` even exploits it for
   the cross-block-reusable canvas RoPE.)

2. **Causal prefill ⇒ position `i`'s K/V depends only on tokens `[0:i]`.** Prefill
   attention is causal. Token `i`'s hidden state (hence its K/V projection)
   attends only to `[0:i]`; positions `> i` and the trailing padded columns are
   masked to `-inf` → 0 after softmax and never contribute. So the K/V written at
   position `i` is a pure function of `tokens[0:i]` (plus absolute position `i`),
   **independent of the total prefill length**. Therefore prefilling a long prompt
   `A` and then reading back its leading span `[0:L]` yields exactly the K/V a
   standalone prefill of `A[:L]` would produce — *provided* `A[:L]` is the literal
   token span (no alignment-pad mismatch; see §Alignment). This is the standard
   APC correctness argument, and it is *stronger* here because it is the same
   physical cache, not a copy.

3. **Sliding-window + full-attention layers both preserve it.** Gemma-4
   interleaves sliding-window (`θ=1e4`, window 1024) and full-attention (`θ=1e6`)
   layers. For a *full* layer, token `i` attends causally to `[0:i]` — covered by
   (2). For a *sliding* layer, token `i` attends to `[max(0, i-window):i]`, a
   subset of `[0:i]` — still a pure function of a prefix of `tokens[0:i]`, so a
   shared prefix still yields identical K/V at position `i`. No layer type breaks
   the prefix invariant. (The denoise-side sliding overlay is a separate concern;
   it operates on the *canvas* queries, not on how the prefix K/V was written.)

### The three-phase KV machine, and why reuse is safe against it

- **Prefill (causal, write)** — writes prompt K/V into `[0:cache_len]`. This is the
  span we cache/reuse.
- **Denoise (bidirectional, read-only)** — the canvas recomputes its *own* K/V every
  step and cross-attends to the frozen `[0:cache_len]` prefix. It **never writes**
  the frozen region, so reuse cannot be corrupted by denoise.
- **Commit (causal, append)** — appends the committed 256-token canvas K/V at
  `[cache_len : cache_len + 256*N]`, i.e. strictly *after* the prompt region.

Consequence for the resident tracking: a session's own commits only touch
positions `≥ cache_len`, so the prompt prefix `[0:cache_len]` it reused stays
valid throughout its generation. The prefix cache therefore records, after every
prefill, exactly the *current* request's `(aligned_tokens, prompt_len, cache_len)`
as the resident state — this is precisely the span of the contiguous cache that
is guaranteed intact once that request starts committing (`[cache_len:]` is about
to be overwritten by its own decode, so it is never claimed as reusable).

### Alignment (the one non-obvious constraint)

Prefill pads the prompt to a 32-tile multiple (`_pad_prompt_tokens_for_prefill`)
and the denoise read covers the *aligned* `cache_len`, **including** the pad
positions. So the reuse span must match the resident cache byte-for-byte over the
*entire* `new_cache_len` (real tokens **and** any pad). The shipped rule enforces
exactly this: `resident_aligned[:new_cache_len] == new_aligned`. Practical effect:

- **exact full match** (`new_aligned == resident_aligned`) always qualifies;
- an **aligned proper prefix** qualifies when the new prompt length is a 32-multiple
  (its pad is empty, so its aligned tokens equal the resident's leading real
  tokens);
- a non-32-aligned proper prefix does *not* qualify, because its zero-pad would
  claim positions that hold the resident's real-token K/V — correctly rejected.

## How the prototype works

`tt/prefix_cache.py :: PrefixKVCache` — a tiny host-side registry (no device
state of its own; the K/V lives in the model cache):

- `plan(aligned_tokens, prompt_len, cache_len) -> ReusePlan` — computes the
  longest-common-prefix length vs the resident record and whether a full bit-exact
  reuse is possible (`reuse=True` iff `cache_len <= resident_cache_len` and
  `resident_aligned[:cache_len] == aligned_tokens`).
- `record(aligned_tokens, prompt_len, cache_len)` — set the resident state after a
  prefill (real or reused). Always the *current* request's own prompt (see above).
- stats: `hits`, `misses`, `partial_prefix_misses`, `tokens_reused`,
  `prefill_time_saved_s`.

`tt/serving.py :: BlockDiffusionServingSession` — `__init__` gains
`prefix_cache=None`; `prefill()`:

1. builds the aligned token tuple (same `_pad_prompt_tokens_for_prefill` the real
   prefill uses);
2. if a `PrefixKVCache` is attached **and** `DG_PREFIX_CACHE` is on, calls `plan`;
3. **reuse** → skip `prefill_prompt_tokens` entirely, set
   `prompt_len/cache_len/next_pos` from the plan, build the denoise logits adapter
   (which lazily reads the already-resident `[0:cache_len]` cache), record the
   saved prefill wall-time;
4. **miss / partial-prefix** → normal `prefill_prompt_tokens`, then `record`.

`prefill()` also sets `self.prefill_reused: bool` and `self.prefill_time_s` so the
serving driver can log the saving. The flag defaults OFF; with the flag off (or no
`PrefixKVCache` passed) the path is byte-identical to the pre-existing prefill.

## Where #47488 is required (documented, stopped cleanly)

The productionly-valuable case — **shared prefix + differing/extending suffix**
(same system prompt, different user turn) — cannot be made bit-exact at the DG
serving layer without one of:

- **chunked / prefix prefill in the backbone** so the suffix tokens can
  cross-attend to the cached prefix K/V during prefill. Gemma-4 prefill does not
  support a nonzero start position or reading prior cache
  (`models/demos/gemma4/tt/model.py:1291,1298,1385`), and editing `models/demos/gemma4/`
  is forbidden. A DG-local commit-decode "suffix prefill" (token-by-token via the
  decode path) is *functionally* correct but **not bit-exact** to the batched
  prefill matmul geometry in bf16 — and #48291 shows small-probability drift can
  flip a diffusion accept/renoise decision — so it fails the bit-exact bar and is
  not shipped.
- **vLLM paged-cache ownership (#47488)** so prefixes live in a block pool with
  per-request block tables and vLLM's own prefix-cache bookkeeping, decoupled from
  the single contiguous model cache. This is the real home for general APC and is
  explicitly out of scope for this prototype (the task says do **not** wire vLLM's
  block pool).

The prototype surfaces this precisely: on a shared-prefix-with-suffix request it
logs `partial-prefix miss: matched N aligned tokens, suffix differs → full prefill
(needs chunked prefill / #47488)` and counts it in `partial_prefix_misses` /
tracks the `N` tokens that a paged path could have saved.

## Device correctness check

`demo/prefix_cache_smoke.py` (reduced-surface, model-owned cache, no vLLM) runs,
on the free QB2 mesh under `flock`, an **ON-vs-OFF bit-exact** comparison for a
request `B` that shares a long prefix with a warm request `A`:

1. **OFF**: fresh session, cache disabled → full prefill of `B` → committed argmax
   `O_off`, prefill wall-time `t_off`.
2. **warm**: real prefill of the long prompt `A` → resident := `A`.
3. **ON**: session sharing the `PrefixKVCache` → `B`'s aligned prompt is a full
   prefix of resident `A` → **reuse** (prefill skipped) → committed argmax `O_on`,
   prefill wall-time `t_on ≈ 0`.
4. **assert** `O_on == O_off` bit-for-bit (identical committed 256-token block),
   and report the prefill saving `t_off - t_on`.

Both `B` runs use the same session seed, so the seeded canvas-init/noise sequences
are identical; any output difference would come *only* from the prompt K/V, which
reuse keeps bit-identical. Cases exercised: **exact full match** (`B == A`) and
**aligned proper prefix** (`B = A` truncated at a 32-token boundary).

Run outcome + metrics: see `work_log.md` (`DG_PREFIX_CACHE_SMOKE_*` markers) and
`prefix_cache_smoke.json`.

## Status

- [x] Design note (this file) with the RoPE / three-phase-KV / sliding-window
      correctness argument and the #47488 boundary.
- [x] Prototype behind `DG_PREFIX_CACHE` (default OFF): `tt/prefix_cache.py`,
      `tt/serving.py` wiring, `tt/generator_vllm.py` opt-in wiring.
- [ ] Device bit-exact check under `flock` (see `work_log.md` for the live run).
