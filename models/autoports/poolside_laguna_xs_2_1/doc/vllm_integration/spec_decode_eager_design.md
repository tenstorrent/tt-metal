# Eager ngram spec-decode in the served (vLLM) path — design + honest ship verdict

**Status:** DESIGN ONLY (off-device). Scope: wire the *eager* decode-verify ngram spec-decode
(`SpeculativeDecoder(verify_mode="decode", traced=False)`) into the served vLLM generator behind an
env flag, default OFF. The *traced* verify path (the only one fast enough to net the full ~2.5×) is
OUT OF SCOPE — it crashes at `warmup_verify_decode` trace capture (b2_verdict.md; the CCL two-trace
coexistence hazard).

**Headline verdict (read first):** the "1.53×" for eager decode-verify is measured **eager-spec vs an
eager baseline** (`plain_greedy_via_verify`, `traced=False`) on **P300x2**, NOT against the production
**traced** `decode_forward` the server actually runs. There is **no on-device measurement of eager
decode-verify beating production traced decode**, and the structural evidence says it very likely does
**not** (see §4). Recommendation: land the flag **OFF by default, batch-1 only, as a
correctness-preserved experimental opt-in and validation vehicle** — not as a promised per-user
speedup.

---

## 1. Where the eager loop plugs into the served path, and the flag

### The core obstacle (why this is not a small change)

The generator-level `SpeculativeDecoder.generate()` (`tt/spec_decode.py:246`) is a **self-contained
multi-token loop**: it owns the running `history`, proposes K drafts, runs ONE verify that **writes KV
for positions `P-1 … P-1+K`**, and **commits `m+1` tokens per iteration**. The standalone driver
(`spec_decode_driver.py`) works because it pre-allocates the request a **full-width page table**
(`pt = [range(nblocks)]`, driver line 217-218) and drives prefill/verify directly, entirely **outside
the vLLM scheduler**.

The served vLLM V1 path is the opposite shape (`async_decode.py:623`,
`runner.model.decode_forward(...)`): **one `decode_forward` call per scheduler step, one token out per
request, position advanced by exactly 1**, and **KV blocks allocated incrementally** by vLLM's block
manager (a new block roughly every `block_size` tokens, allocated for the *next* token only). A
generator-level verify that writes KV for `P … P+K` writes into blocks vLLM **has not allocated for
this request yet** — those page-table rows are OOB or belong to *other* requests → corruption. This is
the fundamental conflict; it is the reason the working win does not "just drop in."

There are therefore two candidate insertion points. **Point A** is what this doc specifies (matches the
working code, batch-1 lane); **Point B** is the architecturally-correct route and is a larger effort.

### Point A (recommended for the flag): intercept inside the generator adapter, batch-1 lane

- **File:line:** `tt/generator_vllm.py:1010` — `LagunaForCausalLM.decode_forward`, in the
  `sampling_params is not None` device-sampling branch. Add, right after `B = tokens.shape[0]`
  (line 1035), a guard:

  ```
  if _SPEC_DECODE_ENABLED and B == 1 and <spec-state present for this request>:
      return self._spec_decode_step(...)   # runs propose + verify_greedy_decode, returns tokens
  ```

- **What `_spec_decode_step` reuses (all already on the adapter):**
  - `NgramProposer.propose(history, K)` — `tt/spec_decode.py:92`.
  - `verify_greedy_decode(tokens, positions, ..., traced=False)` — `tt/generator_vllm.py:865`, which
    for `traced=False` calls `verify_forward_decode` (`:788`) and returns host-argmax ids.
  - `SpeculativeDecoder._accept_greedy` — `tt/spec_decode.py:195`.
  - The per-request `history` (see below) so the proposer has full context.

- **The KV-allocation fix that makes Point A safe (mandatory):** the served request must be given a
  **pre-allocated full-width page table for its whole advertised length**, exactly like the standalone
  driver, so verify's ahead-writes at `P … P+K` always land in blocks owned by *this* request. At
  batch-1 single-user serving this is affordable (one request holds the KV pool; per
  `laguna-decode-headroom` ~25 GB/device is free and the KV pool is policy-capped at 1.5× context).
  Concretely this means the flag also forces: **`--max-num-seqs 1`**, and the plugin/block-manager to
  hand this request a static, contiguous, full-length block table (not incremental). Without this,
  Point A is **unsafe** — do not ship it writing into vLLM's incrementally-allocated table.

- **Returning `m+1` tokens through a 1-token/step interface:** two sub-options —
  1. **Buffer-and-drain (least invasive):** `_spec_decode_step` runs ONE spec iteration, KV for the
     committed positions is written by the verify, returns `committed[0]` to vLLM and stashes
     `committed[1:]` in a per-request queue; subsequent `decode_forward` calls for that request drain
     the queue **without running the model** (KV already present, position math already consistent
     because vLLM advances by 1 each drained step). Downside: vLLM still bills one scheduler step per
     token, so the throughput accounting is only correct if the scheduler's per-step wall-clock
     collapses on drained steps — needs a plugin-side fast-path so drained steps don't re-enter the
     model. This couples to `async_decode.py` / `lane_scheduler.py`.
  2. **Multi-token output (correct but deeper):** emit `scheduled_spec_decode_tokens`-style multi-token
     acceptance through the plugin so vLLM's block manager grows the sequence by `m+1` — that is Point B.

- **Per-request `history`:** the proposer needs the full running context. The plugin already threads
  `prompt_tokens` / `output_tokens` into decode kwargs (`async_decode.py:598-601`); the adapter can
  reconstruct `history = prompt_tokens + output_tokens` per request. If those are not populated on the
  greedy device-sampling path, the flag's setup must request them (small plugin change).

### Point B (architecturally correct, larger — the "dead scaffold"): vLLM-native ngram spec-decode

The plugin already has the *start* of vLLM's native ngram path, currently inert:
- `platform.py:487` — spec-config gate: asserts `method == "ngram"` (only ngram allowed).
- `model_runner.py:143-149` — builds vLLM's own `NgramProposer` when `speculative_config.method ==
  "ngram"`; `num_spec_tokens` read; **`propose_draft_token_ids` (step 4) not wired**.
- `model_runner.py:2103` `take_draft_token_ids` + `worker.py:331` delegate — return `None` today, so
  "spec-on behaves exactly as non-spec decode" (model_runner.py:142).
- `lane_scheduler.py` — already carries `scheduled_spec_decode_tokens` (`:138,169,213,545`) and
  `update_draft_token_ids*` (`:611-622`) plumbing.

This route lets vLLM's block manager allocate the K draft slots (no ahead-write hazard) and verify K+1
in one forward — the *right* home for a real spec-decode. But it needs the verify wired end-to-end
(steps 4-7) AND a verify forward fast enough to win, which today is only the *traced* decode-verify
(blocked). **Does the scaffold interfere with Point A?** No. It is only constructed when a
`speculative_config` is present (`model_runner.py:146`); with no `--speculative-config` on the server
it stays `None` and `take_draft_token_ids` returns `None`. Point A's env flag is independent of it, so
the two do not collide. If Point A ships, leave the scaffold as-is (or revert per `laguna-spec-decode`,
which calls it "unused/dead").

### The flag

- **`TT_LAGUNA_SPEC_DECODE`** (default `"0"` / off), read once in `LagunaForCausalLM.__init__`
  (`tt/generator_vllm.py:107`) alongside the other `TT_LAGUNA_*` reads (mirror `_PREFIX_CACHE_ENABLED`
  at `:89`). Store `self._spec_decode_enabled`.
- Companion tunables (all optional, defaulted from the host replay, §2):
  `TT_LAGUNA_SPEC_K` (draft_len), `TT_LAGUNA_SPEC_MIN_N`, `TT_LAGUNA_SPEC_MAX_N`.
- **Hard preconditions the flag must self-enforce** (fail closed / log-and-disable if unmet):
  `B == 1`, greedy request (`temperature <= 0`), full-width static page table present. If any fails,
  fall through to the normal traced `decode_forward` path.

---

## 2. Config (from the host replay) + the batch-1 constraint

Host replay on 7 real SWE-bench tool-calling trajectories (679 turns / 176,790 tokens; proposer
byte-validated over 209,916 checks — `spec_decode_accept/results.md`, `scripts/spec_accept_replay.py`):

- **Best: `min_n=1, max_n=10, K=16` → mean committed tokens/verify = 2.504** (mean accepted drafts
  m = 1.509). `min_n=1, max_n=8, K=16` is essentially tied (2.501). 23/36 configs clear the 2.0 bar.
- **Defaults for the flag:** `TT_LAGUNA_SPEC_MIN_N=1`, `TT_LAGUNA_SPEC_MAX_N=10`, `TT_LAGUNA_SPEC_K=16`.
  Note `min_n=1` (unigram fallback) is what lets the proposer fire on almost every step; `NgramProposer`
  (`tt/spec_decode.py:86`) defaults `max_n=3` — override it.
- **Caveat on K=16 for eager:** the eager verify runs **K+1 = 17 candidates in the decode batch dim**
  per iteration with **per-op host dispatch** (`verify_forward_decode`, `:788` — no trace). Host
  dispatch cost scales with the op count, which is fixed per forward, but the readback and sampler
  cost scale with K+1. A smaller K (e.g. 8) lowers per-iteration cost at some accept loss; the perf
  probe (§3) should sweep K ∈ {8, 16} eagerly since the replay's 2.5 accept assumed the *ideal*
  verify-≈-one-step economics that eager does not have (§4).

**Batch-1 constraint (hard):** native tool-calling and concurrent decode are unstable on this stack —
under `--workers >1` the batched decode corrupts tokens (`/testbedbed`, `separableable`), 0 patches
(`laguna-toolcalling`, `laguna-batched-decode-corruption`). The `SpeculativeDecoder` is batch-1 by
construction (`tt/spec_decode.py:4,109`; verify replicates one user's block row across candidates,
`verify_forward_decode:815-816`). So spec-decode is a **single-user, batch-1-only** opt-in and must
refuse `B>1`. This aligns with how agents are already run (batch-1, `--workers 1`).

---

## 3. Correctness plan (on device, when the mesh frees up)

The module's invariant: committed tokens are always the target verify's greedy result, so **greedy
spec-decode is token-identical to plain greedy** (`tt/spec_decode.py:27-34`). Validate in this order:

1. **Standalone accuracy (already the proof vehicle):**
   ```
   spec_decode_driver.py --mode accuracy --verify-mode decode \
       --isl-acc 512 --osl-acc 48 --draft-len 16 --ngram-max-n 10 --prompt-mode code
   ```
   Assert `[RESULT] ACCURACY PASS` (0 mismatch vs `plain_greedy_via_verify`, driver `run_accuracy`).
   This is the eager decode-verify path (`--traced` omitted) — exactly what the flag runs.

2. **Latency reconfirm (~1.5×, the eager-vs-eager number):**
   ```
   spec_decode_driver.py --mode latency --verify-mode decode --isl 4096 \
       --osl 128 --draft-len 16 --ngram-max-n 10 --prompt-mode code --baseline
   ```
   `--baseline` prints `spec speedup=…x` vs `plain_greedy_via_verify`. Expect ~1.3–1.5× on repetitive
   input; **this ratio is against the eager baseline, not the server** (see §4).

3. **Served token-identity gate (the ship gate):** with `TT_LAGUNA_SPEC_DECODE=1` at batch-1, decode a
   fixed prompt on the server and assert the output stream is **token-identical to the same prompt
   decoded with `TT_LAGUNA_SPEC_DECODE=0`**. IMPORTANT subtlety (§4): the served baseline uses the
   **on-device sampler** (top-k=1), while eager verify uses **host argmax** (`verify_forward_decode` →
   `torch.argmax`, `:886`). These differ only on bf16 near-ties, but that means spec-on may diverge
   token-for-token from spec-off on a near-tie. Two acceptable resolutions: (a) accept "equally-valid
   greedy trajectory, may differ at near-ties" and gate on *distributional*/text equivalence, or
   (b) make the eager verify use the **on-device sampler** (`verify_sampler_eager`, `:903`, currently
   diagnostic) so spec-on matches the server's greedy exactly. Prefer (b) for a clean identity gate.

4. **Unit safety net (no device):** `tests/test_spec_decode.py` already proves spec ≡ greedy against a
   host stub for K=1,2,4,8 — extend with K=16 and the `min_n=1/max_n=10` config so the flag's params
   are covered off-device.

---

## 4. Risks and the honest ship/no-ship verdict

### What the 1.53× is actually measured against

`plain_greedy_via_verify` (`tt/spec_decode.py:299`) drives ONE **eager** (`traced=False`) batched-decode
verify per token — same numerics family as spec, for a clean apples-to-apples accept isolation. The
1.53× (P300x2, 4k, best-case repetitive accept 4.00/4: 31.6 vs 20.6 t/s/u) is **eager-spec ÷
eager-baseline**. The **production server does NOT use the eager path** — it uses **traced**
`decode_forward` (`:1093` `ttnn.execute_trace`), whose floor is ~33 ms/tok (~30 t/s/u short ctx,
`context_contract.json`) and ~77 ms/tok at 32k (`b2_verdict.md`). The eager verify forward carries
**per-op host dispatch that trace elimination removes**; the eager baseline itself was only 20.6 t/s/u
(48.5 ms/tok) at 4k — i.e. the eager path is already ~1.5× slower than the traced decode it would have
to beat.

### Does eager spec-decode beat production traced decode? Almost certainly NOT (net), and it has never
### been shown to on device.

- **No such measurement exists.** Every "win" number for eager decode-verify is vs the eager baseline,
  on P300x2. The server runs on P150x4 with traced decode. `b2_verdict.md` is the actual on-device
  study and its verdict is **"Do NOT ship spec-decode on the current stack"**: the working paths
  (prefill-verify; and, transitively, eager decode-verify) are structurally slower per verify than a
  traced decode step, and the one verify fast enough to win (**traced** decode-verify) **crashes at
  `warmup_verify_decode`**.
- **The break-even math is unfavorable for eager.** Spec wins iff
  `(mean committed tokens/iter) > (eager verify cost in units of a traced decode step)`. Prefill-verify
  is ~2.4× a traced decode step at 32k → nets 0.93× even at accept 2.2 (`b2_verdict.md:24-27`). Eager
  decode-verify is cheaper than prefill-verify per KV-read but adds K+1-candidate host dispatch and is
  still an **eager** forward (slower than the **traced** step it competes with). To net a win it needs
  sustained committed-tokens/iter comfortably above its per-verify cost multiple — plausibly >2.5 at
  K=16 — which the replay's 2.504 only barely reaches *under the ideal "verify ≈ one step" assumption
  that eager violates*.
- **The replay accept may not transfer to the served trajectory.** The 2.504 was measured with **host
  argmax** greedy (the true-greedy chain). The server's **on-device sampler** trajectory diverges from
  host-argmax on the first bf16 near-tie, after which an ngram proposer that "learned" the argmax chain
  mis-drafts (documented for the synthetic prompt: sampler accept collapsed to 0.07 at 4k while eager
  argmax showed 4.00/4 — `laguna-spec-decode`, "the argmax trajectory the hardware does NOT serve").
  Real coding input is less degenerate, but the served accept is **unmeasured** and is the real driver.
- **Workload sensitivity.** Even if agentic/repetitive decode clears break-even, generic
  (non-repetitive) decoding will fall below it and **regress** vs traced decode. So a global default is
  wrong; per-user opt-in only.

### Other risks
- **KV ahead-write safety** (§1): unsafe unless the request holds a static full-width page table;
  ship-blocking if not enforced.
- **Multi-token return vs 1-token/step scheduler** (§1): the buffer-and-drain shim must not re-enter
  the model on drained steps, or the accounting/throughput is wrong.
- **Host-argmax vs on-device-sampler identity** (§3.3): resolve before any "token-identical to
  production" claim.
- **Interaction with prefix caching / chunked prefill:** spec-decode assumes the request's KV is fully
  resident and its page table static; validate it composes with `TT_LAGUNA_PREFIX_CACHE` and that
  chunked prefill (disabled in the plugin, `platform.py:481`) is not required.

### Verdict

**NO-GO as a shipped speedup. GO only as an OFF-by-default, batch-1, experimental opt-in +
validation vehicle.** The eager path is correctness-proven (token-identical to greedy) and the flag is
cheap to land, but the 1.53× is against a slower eager baseline, not the server's traced decode, and
the on-device study (`b2_verdict.md`) concludes the current stack does **not** net-win. The real
production win requires either the **traced decode-verify path made stable** (fix the CCL two-trace
coexistence crash at `warmup_verify_decode`) or the **vLLM-native ngram route** (Point B) with a
fast enough verify — both larger, separate efforts needing explicit go-ahead. Landing
`TT_LAGUNA_SPEC_DECODE` now buys a clean, correctness-gated harness to measure served eager accept and
to A/B once the traced path is unblocked; it should not be advertised as a per-user speedup until an
on-device eager-vs-**traced**-decode measurement shows a net win on the target workload.
