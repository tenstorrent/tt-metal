# Qwen3.6-27B decode-drift — Fix Design

Companion to `qwen36_decode_drift_investigation.md`. Root cause: the GDN **chunk (prefill)** and
**recurrent (decode)** kernels diverge, growing with recurrence length; the divergence propagates and
compounds across the 64-layer hybrid stack → long-generation reasoning drift.

**Leverage point:** the attention SDPA-decode op is clean (0.9998); its per-layer decode divergence
comes from an already-diverged *input* hidden state fed from upstream GDN layers. So **stopping the GDN
state from drifting also stops most of the attention-side compounding.** Fixing GDN is the high-leverage
move.

---

## 0. Decisive branch: chunk-vs-torch (do this first — cheap)

We know `recurrent ≈ torch` (0.9998). We have **not** measured `chunk ≈ torch`. This one number
splits the whole design:

- **If chunk-vs-torch is also ~0.999** → both kernels are individually accurate; the divergence is a
  benign fp32 reformulation gap and the *only* issue is that decode uses a different path than the one
  the network reasons under. → Fix = **path consistency** (Option A/B below).
- **If chunk-vs-torch is low (~0.97)** → the **chunk (prefill) kernel is under-precise** and is the real
  culprit; HF (which also does chunk-prefill + recurrent-decode and gets ~90%) simply has a tighter
  kernel. → Fix = **chunk-kernel precision** (Option C), a smaller and more upstream-mergeable change.

Add a torch-reference arm to `test_gdn_chunk_vs_rec.py` (replicate qkvz/ab projections + FIR conv +
gated-delta recurrence in fp32 torch on the same `x`, compare BOTH kernels to it). ~1 device run.
Also worth: the GPU/llama.cpp reference (deferred option B) confirms "model+recurrence is fine at 90%",
independently pointing at TT kernel precision.

---

## Option A — Periodic state re-sync (pragmatic serving fix)

Every `N` decode steps, re-run the **chunk** kernel over the accumulated tokens (prompt + generated so
far) and overwrite the recurrent GDN state with the chunk result. Bounds the chunk-vs-recurrent
divergence to `N` steps' worth instead of letting it grow unboundedly.

- **Effort:** medium. Reuse `forward_prefill(capture_state=True)` over a growing window; wire a
  step counter in `decode_tp_batched` / the vLLM decode contract; handle the KV/conv state refresh.
- **Cost:** one chunk-prefill every `N` steps. At `N=32` and prefill ≈ a few ms/token amortized, this
  is a modest throughput hit, tunable. Sliding-window (last `W` tokens) caps cost for long contexts.
- **Risk:** medium — must keep the refresh trace-safe (chunk-prefill inside/around the decode trace),
  and per-slot for B>1 continuous batching.
- **Expected gain:** directly bounds the dominant divergence; should recover most of the GSM8K gap if
  the recurrence is the compounding source (as measured). Accuracy/speed knob via `N`.
- **Best when:** chunk-vs-torch is good (path-consistency problem, not kernel precision).

## Option B — Decode via chunk kernel (correctness ceiling, slow)

Run every decode step through the chunk kernel over the full sequence (= `test_prefill_gen`, which
reasons correctly). This is Option A with `N=1`.

- **Effort:** low (already validated as correct offline).
- **Cost:** O(T) per step — prohibitive for real serving; useful as the **accuracy oracle** and as the
  `N→1` limit to bound how much Option A can recover.
- **Use:** benchmark target, not a shipping path.

## Option C — Chunk-kernel precision alignment (proper upstream fix)

If §0 shows the chunk kernel is the under-precise one, fix it in the upstream C++ op
(`ttnn/cpp/.../transformer/gated_delta_attn/`): higher-precision accumulation / accumulation order that
matches the sequential recurrence, or fp32 dest-acc where it currently rounds.

- **Effort:** high (C++ kernel + rebuild + PCC re-validation).
- **Risk:** medium; upstream-shared op, affects prefill everywhere.
- **Expected gain:** fixes the root without a per-step runtime cost; most upstream-mergeable.
- **Best when:** chunk-vs-torch is low.

## Option D — Do nothing to numerics; ship with flat-norm + document (status quo)

Keep the current flat-norm mitigation (already the best of the tested norm schemes), document the
long-CoT limitation, and rely on sampling to avoid loops for chat-style use. The fused-decode PRs are
accuracy-neutral and unaffected.

- **Use:** if the target workloads are short-form (chat, recall, short answers) where the drift does
  not bite, and the multi-step-reasoning gap is acceptable for now.

---

## Results (executed 2026-07-12)

- **§0 chunk-vs-torch:** `recurrent_vs_torch = 0.99998` (flat over all positions),
  `chunk_vs_torch = 0.988`. → The **chunk (prefill) kernel is the under-precise path**; the recurrent
  (decode) kernel is near-exact. Root cause reframed: prefill builds context in a slightly-off
  "chunk-space" (0.988); prefill-gen reasons correctly because it stays entirely in chunk-space; real
  decode continues that chunk-space context with the accurate recurrent kernel → **space mismatch**.
- **Option C cheap probe (matmul HiFi4):** the chunk gap is **not** matmul fidelity. Inputs to the C++
  `gated_delta_attn_seq` op are already fp32; raising the python wrapper matmuls to HiFi4 → no change;
  changing the compute-kernel fidelity in `gated_delta_attn_program_factory.cpp:170` (HiFi2→HiFi4) and
  rebuilding → `chunk_vs_torch` 0.98884 → **0.98887 (no change)**. The 0.988 gap is **structural** in
  the chunk-parallel formulation (the `L_inv` matrix-inverse / chunk decomposition, or bf16 CB
  intermediates inside the C++ kernel), **not a one-line fidelity knob**.

→ **Option C is NOT the cheap upstream fix originally hoped for.** It needs deep C++ kernel work
(CB data formats or the chunk-decomposition numerics) with uncertain payoff.

## Option A — VALIDATED (2026-07-12)

`test_decode_resync.py`: every `N` generated tokens, re-prefill the whole sequence-so-far
(`prefill_seed_tp` rebuilds attention KV cache + GDN state in chunk-space, `finalize_seed`); recurrent
decode between re-syncs. Janet problem (correct = $18):

| mode | reaches $18 | behavior |
|---|---|---|
| pure decode (N=∞) | ✗ | confabulates numbers (16→14→12), drifts |
| re-sync N=8 | ✗ | numbers now correct (16/3/4/$2) but loops, no answer in budget |
| **re-sync N=4** | **✓** | correct reasoning, reaches $18 |

→ Confirmed at the generation level (not just the prefill-gen oracle): keeping decode in chunk-space
restores reasoning. The sensitivity is steep — the recurrent state diverges enough to loop within ~8
tokens, so `N=4` was needed here. Cost is correspondingly high (re-prefill ≈ every 4 tokens, O(L) each).

**Multi-problem eval (`test_resync_eval.py`, 3 GSM8K, greedy):** the single-problem win did NOT
generalize cleanly. With max_new=200: pure 1/3, N4_full 1/3 (recovers Janet=18), N4_win64 1/3 (loses
Janet). Each config solves a *different* single problem → variance, not systematic gain. Throughput
(eager decode baseline ~5 tok/s): N4_full ≈ 3.0, N4_win64 ≈ 3.5 tok/s (re-sync ~halves eager decode;
window slightly cheaper but no accuracy gain). A ceiling run (N=1 oracle, N=2) at max_new=120 returned
0/3 but was **confounded by truncation** — Janet at N=1 showed correct on-track numbers (16,3,4,…) cut
off before reaching 18. **Caveats:** the eval is noisy — fragile last-3-numbers answer extraction, no
EOS (models ramble to the token cap), and token caps that truncate the ~150–200-token CoT these
problems need. So Option A's multi-problem benefit is **neither proven nor refuted** here; only the
qualitative Janet recovery (confabulation → correct reasoning under N=4) is solid.

**Rigorous eval + confound (test_gsm8k_resync.py, 0-shot, 12 GSM8K):** first pass (thinking mode,
480-tok budget) gave pure 0/12, N4_full 1/12, N4_win64 1/12 — but ALL 36 generations hit the token cap
with **no EOS**: Qwen3.6 is a *thinking* model, its `<think>` chain doesn't finish in 480 tokens, so
every generation was truncated mid-think and the answer extraction was noise. That confound, not
re-sync, drove the low scores. **Non-thinking (concise) re-run (`enable_thinking=False`, 256-tok):**
**pure 0/12 (0%) vs N4_full 3/12 (25%)** — real hits (#0 18=18 terminated at 220 tok, #3 540=540 at
182 tok, #4 20=20). So once termination is clean, **re-sync N=4 gives a real, measurable improvement
(0→25%)**. Absolute accuracy is modest (concise 0-shot is hard; N=4 ≈ ½ decode throughput), but the
direction is clear and positive → Option A is validated; proceeding to vLLM wiring (d).

## Revised recommendation

1. **Option A (periodic re-sync) — validated and works.** Productionization: (a) a `generate_tp_resync(prompt,
   max_new, resync_every)` model method; (b) measure throughput at N=4/8 vs pure decode; (c) reduce cost
   with a sliding-window re-sync (re-prefill only the last W tokens, carry prior state) instead of full
   re-prefill; (d) wire into the vLLM decode contract. Tunable `N` trades accuracy vs throughput.
2. **Alternative:** compute prefill's GDN path with the *recurrent* kernel (accurate, `0.99998`) so
   prefill-space = decode-space = true-space. Correct but `O(T)` sequential prefill — measure the
   throughput hit; may be acceptable for short prompts.
3. **Option D (status quo + document):** flat-norm is the best tested mitigation; ship short-form,
   document the long-CoT limitation. The fused-decode PRs are unaffected.
4. **Deep Option C — investigated 2026-07-12, premise does not hold.** The hoped-for fix ("raise the
   chunk kernel's bf16 intermediates to fp32") is a no-op: the C++ `gated_delta_attn` kernel's
   circular buffers are **already all fp32** (`df_f32`), matmul fidelity HiFi4 has **no effect**
   (0.98884→0.98887), and `ttnn.exp` already runs in **accurate mode**. The 0.988 gap is not a config
   knob — it is the **SFPU transcendental precision (exp / reciprocal) accumulated through the
   ill-conditioned within-chunk `L_inv` (matrix inverse)**. The recurrent path stays 0.99998 because it
   applies exp once per step on a small value; the chunk path evaluates `exp(cumsum(g))` over a wide
   range plus a 128×128 exp matrix plus a reciprocal-based inverse. Truly fixing it needs a
   **numerical reformulation of the chunk math** (better-conditioned inverse, higher-precision
   transcendental emulation) — research-level kernel work, uncertain payoff. **Not recommended as a
   near-term fix.**
5. Keep flat-norm; do **not** adopt the branch's sharp-norm or SDPA config (both worse in main).

## Productionization status (2026-07-12)

- **(a) `model.generate_tp_resync(prompt, max_new, resync_every, eos_id, window)`** — implemented in
  `model.py` (demo/eager path). `resync_every<=0` = pure decode; `window=None` = full re-prefill;
  `window=W` = re-prefill prompt + last W generated tokens (compacted positions).
- **(b/c/e)** — measured via `test_resync_eval.py` (multi-problem GSM8K × {pure, N=4 full, N=4
  window=64}; correctness + tok/s). See results section (appended after the run).
- **(d) vLLM serving integration — STARTED & mechanism validated (2026-07-12):** implemented a
  `decode_forward` hook in `qwen36_vllm.py` (delegate style: `prefill_forward` stashes the prompt
  tokens + page_table; `decode_forward` every N steps calls `model.prefill_traced_chunked(seq[:pos],
  page_table, actual_len=pos)` to refresh paged KV + GDN state into chunk-space, then delegates the
  step to `super().decode_forward` so the decode output contract is untouched). Env
  `TT_QWEN_DECODE_RESYNC_N` (0=off, default), B=1 + host sampling. Offline vLLM run (N=4): the hook
  **fires 49× with no crash** and the output is coherent initially — proving a re-prefill CAN be
  injected into the parked-trace decode path without breaking it (the main risk). **Open:** output
  degrades later ("…3 for 33 3"+whitespace). GDN reset is in-place (trace-safe, confirmed), so the
  leading suspect is that re-syncing at a growing sequence length crosses an **un-warmed prefill
  bucket** and JIT-compiles mid-serving, clobbering the parked decode trace. Next: pre-warm the bucket
  lengths re-sync will hit (or bound re-prefill length), and diff the paged re-prefill state against
  the demo path's full re-prefill (which reaches $18 at N=4). Hook is default-off so serving is
  unaffected.
- **(d) original spec (retained for reference):**
  Serving decode is `Qwen36ForCausalLM.decode_forward` → `Generator.decode_forward` (parked trace
  replay); prefill is model-owned (`prefill_traced_chunked(token_ids, page_table, actual_len)` writes
  the request's paged KV blocks, `reset_state_inplace` preserves GDN buffer addresses so the parked
  decode trace stays valid). Integration:
  1. In `decode_forward`, keep a per-slot **decoded-token history** and **step counter** (the current
     token is already in the decode inputs; append it each call).
  2. Every `N` steps for a slot, instead of replaying the decode trace, call `prefill_traced_chunked`
     over that slot's `[prompt + generated]` (or `[prompt + last W]`) into its existing `page_table`
     blocks, then `reset_state_inplace` + reseed the GDN state; return the resulting logits.
  3. Because TP prefill uses the pre-warmed masked-bucket program set (warmed before the decode trace
     parks — the compile-clobbers-trace fix is already in place), the mid-decode re-prefill does not
     recompile into the parked trace. Verify this holds under continuous batching (B>1).
  Expose `N`/`window` via `--additional-config '{"tt":{"decode_resync_every":4,"decode_resync_window":64}}'`.

  **Building block already exists:** `model.prefill_traced_chunked(token_ids, page_table, actual_len)`
  rebuilds a slot's paged KV cache + GDN state and returns the next-token logits — exactly a re-sync.
  The only new work is the *hook* that calls it every N decode steps with the slot's accumulated tokens.

  **Why this is a live-server integration, not a drop-in (verified against `Generator.decode_forward`,
  2026-07-12):** decode is not a plain "return logits" call — it is tightly coupled to (i) on-device
  sampling (the device token buffer is authoritative; host tokens lag under async scheduling), (ii) the
  parked decode trace and its fixed input/output buffers, (iii) `slot_remap` + continuous batching, and
  (iv) data-parallel chunking. Substituting a prefill mid-replay must reconcile the device token/pos
  buffers and the parked trace, so it needs iteration against a running server (≈210 s standup) and
  interacts with the deferred B>1 path. **Recommendation:** wire it as a focused follow-on with the
  server up — start B=1 (no slot_remap/DP: the reconciliation collapses), validate via the existing
  `bench/gsm8k_eval.py` OpenAI harness (which handles thinking-mode termination + extraction correctly,
  unlike the demo-path probe), then extend to B>1. Accuracy/throughput knob via `N` (and `window`, though
  `window` only helps when the prompt is short — see the concise-eval note).

## (d) vLLM re-sync — clean A/B outcome (2026-07-12)

- `switch_mode` ruled out (Qwen's `switch_mode` is a no-op — no prefetcher).
- Env caveat: setting `os.environ` inside the driver script does NOT reach the vLLM engine
  subprocess; pass `TT_QWEN_DECODE_RESYNC_N` via docker `-e` / bash-level env.
- Clean A/B (offline, same Janet-concise prompt, max_tokens 140, resync fired 34× for N=4):
  - **N=0 (pure serving):** coherent ~30 tokens → catastrophic multilingual-garbage collapse.
  - **N=4 (resync):** coherent ~30 tokens → degrades to whitespace but **avoids the collapse**.
- So re-sync **helps** in serving (prevents the worst collapse) but does **not** reach the demo path's
  quality (which hits $18 at N=4). Decisive: **serving pure decode is far more degraded than demo pure
  decode** (demo = coherent English number-confabulation; serving = multilingual garbage), i.e. the
  serving decode path carries an *additional* degradation beyond the chunk-vs-recurrent drift (paged
  KV / traced decode / fused kernel in the serving path) that re-sync alone cannot fix.

**(2) delivered:** hook implemented (default-off, non-breaking) + mechanism validated (fires, no crash)
+ shown to help (prevents collapse). **Full serving parity requires a separate investigation of the
serving-decode-specific degradation** — a new, broader task, not a re-sync tuning issue.

## Serving-specific degradation localized to PAGED KV (2026-07-12)

Follow-up on "serving pure decode is far worse than demo pure decode" (multilingual-garbage collapse
vs coherent English number-confabulation). Isolated by elimination on the Janet-concise prompt:

| variant | result |
|---|---|
| serving pure, fused GDN **ON** | coherent ~30 tok → multilingual garbage collapse |
| serving pure, fused GDN **OFF** | same collapse (byte-identical) → **fused kernel exonerated** |
| serving pure, `enforce_eager=True` (no decode trace) | same collapse (byte-identical) → **traced decode exonerated** |
| demo pure (internal concat KV caches, eager) | coherent English (confabulates numbers, no collapse) |

The ONLY remaining difference between the coherent demo decode and the collapsing serving decode is the
**paged KV cache** path: `paged_scaled_dot_product_attention_decode` + `paged_update_cache` + external
`page_table` (serving) vs the internal per-head concat caches `k_caches/v_caches` (demo). GDN, RoPE,
norms are identical between the two. → **The serving-specific severe degradation is a bug in the paged
KV decode path**, collapsing around pos ~104 (prompt 74 + ~30 tokens; block_size 64 → block 1). This is
a distinct, more severe problem than the (gradual) chunk-vs-recurrent drift.

**Next:** teacher-forced per-position PCC of paged-decode vs internal-cache-decode logits to pinpoint
the paged breakage (page_table indexing / block-boundary handling / paged SDPA-decode), then fix. This
is the highest-value serving fix — it likely dominates the observed serving accuracy loss.

## BREAKTHROUGH: serving collapse is a paged SDPA-decode multi-chunk bug (2026-07-12)

`test_paged_vs_internal.py` — paged-KV decode vs internal-cache decode (oracle), teacher-forced, full
model, per-step logits PCC:

- Divergence starts EXACTLY at cached length **pos=128** (PCC 0.99→0.51), **independent of block_size**
  (32 and 64 give byte-identical drops) → not block-index, an absolute-128 boundary.
- RoPE + cur_pos ruled out (`prepare_decode_inputs_host` uses `freqs=outer(pos,inv_freq)` with `pos`
  directly, no 128 limit; matches the coherent oracle).
- Varying the paged SDPA-decode `k_chunk_size` MOVES the collapse: k_chunk=0/auto(→128) collapses at
  128; k_chunk=32 collapses at ~96 (worse, min 0.19); **k_chunk=512 (single-chunk, ≥ cached length) →
  NO collapse, all steps PCC ≥ 0.99.** → the bug is the **paged `scaled_dot_product_attention_decode`
  MULTI-CHUNK KV gather** (the non-paged op with the same config is fine).

**End-to-end serving validation:** pure decode (N=0, no re-sync), `QWEN_PAGED_KCHUNK=512`: the
multilingual-garbage collapse is GONE — coherent English for 140 tokens, tracking the prompt numbers
(vs ~30-token collapse with the default auto k_chunk). **So the dominant serving accuracy problem was
the paged multi-chunk bug, and single-chunk k_chunk fixes it — no re-sync needed.**

**Constraint:** k_chunk=2048 crashes (SDPA CBs = 4.67 MB > 1.5 MB L1, `program.cpp:1554`). Single-chunk
is only viable up to k_chunk ≈ 512 (context ≤ 512). Beyond that the multi-chunk bug returns. So the
config fix covers short/medium contexts; **the proper long-context fix is upstream: correct the ttnn
paged SDPA-decode multi-chunk gather.**

### Two distinct decode problems (final)
1. **Paged SDPA-decode multi-chunk bug** — dominant serving problem (catastrophic multilingual
   collapse). Fixed for context ≤512 via `QWEN_PAGED_KCHUNK=512` (env, `attention/tp.py` paged branch);
   long-context needs the upstream ttnn kernel fix.
2. **GDN chunk-vs-recurrent drift** — milder number-confabulation. Mitigated by Option A re-sync
   (GSM8K 0→25% demo path). Present in both demo and serving; secondary to (1) in serving.

## Option C final assessment — SFPU-precision-limited, not a bounded fix (2026-07-13)

Investigated the chunk kernel's 0.988-vs-torch to exhaustion:
- matmul fidelity: HiFi4 (python wrapper AND C++ program-factory rebuild) → no change (0.98884→0.98887).
- circular buffers: already all fp32.
- `ttnn.exp`: already accurate mode (not fast/approx).
- per-position error: ~uniform ~1% (0.991→0.988), not an accumulating trend.

⇒ the 0.988 is the **inherent precision of the SFPU transcendentals** the chunk formula uses heavily
(`exp(cumsum(g))`, the 128×128 `exp` decay matrix, the reciprocal-based `L_inv`). The recurrent path
is 0.99998 because it applies `exp` once per step on a small value; the chunk path evaluates far more
transcendentals over a wider dynamic range. Option C's fix target is therefore either a
**higher-precision on-device exp/reciprocal (custom LLK)** or a **reformulation of the chunk math** —
both research-level kernel work, not reachable via config/fidelity knobs. And since the network
tolerates the 0.988 (prefill-gen reasons correctly), the payoff of chunk-accuracy for decode
consistency is uncertain.

**Conclusion:** Option C is not a bounded, deployable fix. The practical accuracy lever remains
**Option A (re-sync)** — keep decode in the chunk-space the network is calibrated to, tunable by N
(N=1 = prefill-gen = correct but O(T)/token; larger N = cheaper, less accurate). "Recurrent prefill"
was refuted (test_all_recurrent: feeding the prompt through the recurrent path produces off-topic
hallucination — the network is chunk-space-calibrated, so the accurate recurrent path is the
*inconsistent* one).
