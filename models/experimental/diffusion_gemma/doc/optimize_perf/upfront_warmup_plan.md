# Reusable up-front denoise-trace capture — implementation plan

Status: **PLAN ONLY (no code changed).** Target: make the DiffusionGemma serving denoise
trace behave like other tt-metal models (gemma4 `warmup_gemma4_model_prefill`) — captured
**once at model startup** and **reused across every request**, instead of captured inline on
block 0 of each request and thrown away at request teardown.

Verification environment is available: 4× Blackhole `p300c`, venv
`/home/zni/venvs/tt-diffusion-gemma`, checkpoint
`~/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it`.

---

## Key finding (why this is narrower than "re-home the controller")

Reading the code, the per-prompt *in-place refresh* machinery **already exists** and is already
exercised within a single request:

- After each 256-token commit, `denoise_and_commit_block` calls
  `advance_prefix_after_commit(next_pos)` ([denoise_forward.py:1362](../../tt/denoise_forward.py))
  which advances `adapter.prompt_len` / `adapter.q_rope_offset`, advances the reader
  (`MutablePrefixKVReader.set_prompt_len`, [denoise_forward.py:825](../../tt/denoise_forward.py)),
  and refreshes the reveal mask (`update_reveal_mask_buffer`) — all **outside** any trace.
- The controller's reveal-mask branch already replays block N over block 0's trace with **no
  recapture** ([traced_denoise.py:1079-1110](../../tt/traced_denoise.py), early-halt; mirrored in
  the single-step and multistep controllers). Every replay re-refreshes canvas RoPE + reveal mask
  from the adapter's fields.

So the trace is **already shape-invariant and reused across blocks** under
`DG_DENOISE_REVEAL_MASK`. The ONLY thing forcing per-request recapture is lifetime:

- The adapter (`DenoiseLogitsAdapter`) **owns the trace-baked persistent buffers** — canvas RoPE
  `_canvas_rope_bufs`, reveal mask `_reveal_mask_buf`, self-cond `signal_buf`, and the sharded-
  terminal constants `_vocab_offsets`/`_embedding_weight_sharded` — and it is **rebuilt every
  prefill** ([serving.py:321](../../tt/serving.py)). A rebuild reallocates those buffers at NEW
  addresses, so the block-0 trace (which baked the old addresses) is invalid → the controller,
  cached on the adapter, is discarded with it and re-captures on the next request's block 0.

**Therefore Tier B = keep the adapter + controller alive for the model lifetime, rebind them to
each new prompt in place, and move the one-time capture to startup with a mock prompt.** It reuses
the existing refresh machinery; it does not need a new trace mechanism.

## The one behavior being accepted

Up-front capture fixes the reveal-mask prefix read span `p_max` at startup, and the denoise reads
that **full `p_max` span every step** (O(`p_max`) per denoise step, independent of the true prompt
length). Consequences:

- `p_max` must be **bounded** to the max served context via `DG_DENOISE_REVEAL_PMAX` (tile-aligned).
  Servable `prompt + generated` is capped at `p_max`. This is exactly gemma4's "max bucket".
- The plan **fails loud** if `DG_UPFRONT_CAPTURE` is set while `DG_DENOISE_REVEAL_MASK` is off, or
  if the resolved `p_max` is unbounded/huge (e.g. the full 262144 KV allocation) with no explicit
  `DG_DENOISE_REVEAL_PMAX` — never silently fall back to per-request capture or to a 256K/step read.

---

## Design

Gated behind a new `DG_UPFRONT_CAPTURE` flag (default OFF; today's per-request path is byte-for-byte
unchanged when off). When ON it REQUIRES `DG_DENOISE_REVEAL_MASK=1`, a traced serving path
(`DG_VLLM_TRACE=1` / a `DG_DENOISE_*` traced variant), a sized `DG_TRACE_REGION_SIZE`, and a bounded
`DG_DENOISE_REVEAL_PMAX`.

1. **Startup capture (wrapper `warmup_model_prefill`).** Build one session, run a MOCK
   `prefill(mock_tokens)` + one `decode_block()`. That path already prepares the reveal mask / canvas
   RoPE / sharded terminal and captures the controller trace at the fixed `p_max` (the controller's
   `_capture` → `_prepare_reveal_if_enabled`). Then detach the adapter from the throwaway session and
   store it on the wrapper as `self._persistent_adapter`; discard the session shell WITHOUT releasing
   the adapter. The mock's committed KV lands in the `[mock_len:p_max]` tail, which every real request
   masks out or overwrites.
2. **Per-request reuse (wrapper `prefill_forward`).** For each real request, create the per-request
   session as today but hand it the persistent adapter. `session.prefill(real_tokens)` writes the real
   prompt KV into `[0:cache_len]` and **rebinds** the persistent adapter (`rebind_prompt(cache_len)`)
   instead of rebuilding it. `session.decode_block()` replays the startup-captured trace (reveal-mask
   reuse branch) — **no capture**.
3. **Lifetime (release sites).** The persistent adapter + controller are NOT released at request
   teardown; only the per-request session's reference is detached. They are released once at model
   shutdown (or freed by mesh close).

## Correctness argument

Every trace-baked address stays fixed across requests:

- controller buffers (`canvas_buf`, `committed_buf`, `noise_bufs`, `consts`, gumbel state) — controller
  persists → stable;
- adapter buffers (`_canvas_rope_bufs`, `_reveal_mask_buf`, `signal_buf`, sharded-terminal consts) —
  adapter persists (not rebuilt) → stable;
- prefix-KV read source — the model-owned `tt_kv_cache`, read at the fixed `p_max` span → stable
  address, and its CONTENT is refreshed by each request's prefill.

Per-request content flows in outside the trace: prefill overwrites cache `[0:cache_len]`; `rebind_prompt`
resets `prompt_len`/`q_rope_offset`/reader-floor and refreshes the reveal mask to reveal `[0:cache_len]`
and hide the stale `[cache_len:p_max]` tail; the replay re-refreshes canvas RoPE. So the fixed-`p_max`
read sees exactly this request's committed prefix. This is the sanctioned "paged/fixed-shape prefix
input" reuse, NOT an unsound prompt-only-trace replay.

---

## Exact edit surface (all DiffusionGemma-local; no shared `gemma4` edits)

**`tt/traced_denoise.py`** — add `upfront_capture_enabled()` next to `reveal_mask_enabled()`
([:115](../../tt/traced_denoise.py)); docstring states the reveal-mask requirement + `p_max` bound.
(No change to the controllers themselves — the reveal-mask reuse branch already handles a changed
`prompt_len` on every replay.)

**`tt/denoise_forward.py`**
- `MutablePrefixKVReader.reset_prompt_len(prompt_len)` (after `set_prompt_len`,
  [:825](../../tt/denoise_forward.py)) — request-boundary reset that allows ANY tile-aligned
  `prompt_len <= read_span` (shrink OK), unlike the grow-only `set_prompt_len`.
- `DenoiseLogitsAdapter.rebind_prompt(prompt_len)` (near `advance_prefix_after_commit`,
  [:1362](../../tt/denoise_forward.py)) — reset `prompt_len`/`q_rope_offset`, call
  `reader.reset_prompt_len`, refresh reveal mask + canvas RoPE in place. Raises if not on the
  reveal-mask path (a rebind on a prefix-baked trace would silently read a stale prompt).

**`tt/serving.py`**
- `__init__`: add `self._persistent_adapter = None` ([:206](../../tt/serving.py)).
- `attach_persistent_adapter(adapter)` — inject the wrapper's persistent adapter before `prefill`.
- `prefill` ([:321](../../tt/serving.py)): if a persistent adapter is attached, set
  `self._logits_fn = self._persistent_adapter` and `self._logits_fn.rebind_prompt(cache_len)`
  instead of `self._logits_fn_builder(...)`.
- `reset` ([:399](../../tt/serving.py)): if `self._logits_fn is self._persistent_adapter`, do NOT
  release the controller/adapter — only detach the reference + reset scalar state.

**`tt/generator_vllm.py`**
- `__init__` ([:158](../../tt/generator_vllm.py)): add `self._persistent_adapter = None`; resolve
  `self._upfront = upfront_capture_enabled()` (with reveal-mask + trace guards). Import
  `upfront_capture_enabled`, `reveal_mask_enabled`.
- `warmup_model_prefill` ([:378](../../tt/generator_vllm.py), currently a no-op): when `_upfront`,
  validate guards (reveal-mask on, `DG_DENOISE_REVEAL_PMAX` bounded, trace on), run the mock
  prefill+decode capture, extract + store `self._persistent_adapter`, detach from the throwaway
  session. Fail loud on any guard miss or capture failure (a poisoned trace region needs `tt-smi -r`).
- `prefill_forward` ([:~465](../../tt/generator_vllm.py)): when `_persistent_adapter` is set,
  `session.attach_persistent_adapter(self._persistent_adapter)` before `session.prefill`.
- `release_request` ([:613](../../tt/generator_vllm.py)) → `session.reset()`: relies on the session
  guard above to keep the persistent adapter alive.
- Add `release_persistent_capture()` for model shutdown (best-effort; mesh close also frees it).

## Verification (device-gated, `DG_RUN_DEVICE=1`, `DG_CKPT`)

New `tests/test_upfront_capture.py`:
1. **Reuse across prompts (the tracing-skill-mandated stale-input test).** Warmup-capture, then serve
   prompt A (len LA), prompt B (len LB ≠ LA), then A again. Assert `_persistent_adapter`'s controller
   `stats()["capture_events"] == 1` throughout (no recapture), and that A vs B committed outputs
   DIFFER (proves the new prompt's KV/mask flow in — not a stale replay).
2. **Bit-exact vs per-request reveal-mask reuse.** For a single fixed prompt+seed, the up-front-captured
   committed sha256 == the existing (non-up-front) reveal-mask committed sha256 == eager. This is the
   decision-fidelity gate — up-front capture must not move a single committed decision.
3. **Multi-request smoke (symptom-table probe).** Serve the same prompt twice with a different prompt
   between; assert non-garbled / coherent and no recapture. Guards the "garbled serving while standalone
   is clean = stale cross-request state" failure mode.

Evidence dir: `doc/optimize_perf/` (README + work_log + the three artifacts). Do NOT collect Tracy/
profiler from a live server (tt-enable-tracing rule) — use direct-session/block-harness evidence.

## Risks & caveats

- **reveal-mask is default-OFF** and per memory bit-exact on full 30L 26B `serving_smoke`; re-run that
  bit-exactness check after the lifetime change. Do NOT flip the reveal-mask default in this work.
- **Release lifecycle is the sharpest edge.** The session `reset` guard is the single point that keeps
  the persistent adapter alive; a missed guard → use-after-free of released traces (garbage or crash),
  an over-eager retain on the default (non-up-front) path → leaked traces. Must be strictly flag-scoped.
- **Trace-region memory / device poisoning.** Persistent capture holds the trace region for the whole
  run; size `DG_TRACE_REGION_SIZE`. A startup-capture overflow poisons the device (`tt-smi -r`) — fail
  loud at startup, never mid-serving.
- **O(`p_max`) per step + context cap** — see "the one behavior being accepted".
- **Single active sequence.** DG serving is single-sequence (`prefill_forward` rejects `num_reqs>1`);
  one persistent adapter is compatible. Concurrent batched serving is #47488/#47557, out of scope.
- **Mock-capture cache pollution** is benign: the mock's committed KV tail is masked/overwritten by
  every real request.

## Sequencing

1. Land the primitives (`upfront_capture_enabled`, `reset_prompt_len`, `rebind_prompt`) + the session
   attach/rebind/guard behind the flag (default off) → run the existing reveal-mask `serving_smoke`
   bit-exactness check to prove the default path is unchanged.
2. Wire the wrapper startup capture + persistent-adapter reuse.
3. Add and run `test_upfront_capture.py` on device; capture the three artifacts.
4. (Only then) consider whether reveal-mask should become a supported serving default — separate call.
