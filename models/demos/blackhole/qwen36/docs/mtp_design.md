# Qwen3.8-27B MTP / speculative decode design

Target checkpoint: Qwen/Qwen3.8-27B (`model_type: qwen3_5`, hidden 5120, 64 layers
= 48 GDN + 16 full-attn, vocab 248320, `mtp_num_hidden_layers=1`,
`mtp_use_dedicated_embeddings=false`).

## MTP head structure

From the HF safetensors index and vLLM's `qwen3_5_mtp.py` (transformers ignores
`mtp.*` — vLLM is the reference implementation), the checkpoint holds 15 `mtp.*`
tensors:

```
mtp.fc.weight                       [5120, 10240]
mtp.pre_fc_norm_embedding.weight    [5120]
mtp.pre_fc_norm_hidden.weight       [5120]
mtp.norm.weight                     [5120]
mtp.layers.0.{input_layernorm,post_attention_layernorm}.weight
mtp.layers.0.self_attn.{q_proj,k_proj,v_proj,o_proj,q_norm,k_norm}.weight
mtp.layers.0.mlp.{gate_proj,up_proj,down_proj}.weight
```

The single MTP hidden layer is a **full-attention** decoder layer
(`Qwen3_5DecoderLayer(layer_type="full_attention")` in vLLM) with the target's
attention geometry (24 gated Q heads, 4 KV heads, head_dim 256, partial RoPE) —
**not** a GDN block. `mtp_use_dedicated_embeddings=false` ⇒ the head shares the
target's `embed_tokens` and `lm_head`.

Forward (one drafter step, vLLM `Qwen3_5MultiTokenPredictor.forward`):

```
x = fc(cat[rmsnorm_e(embed(tok)), rmsnorm_h(target_hidden)])   # [.., 2H] -> [.., H]
x = decoder_layer(x, position)                                  # full-attn, own KV
h_out = rmsnorm(x); logits = lm_head(h_out)
```

Token/position convention (vLLM `llm_base_proposer.py`): input ids are the target
ids **shifted by one**, positions unchanged — the pair `(target_hidden[i],
token[i+1])` sits at drafter position `i` (drafter KV slot `i`) and predicts
`token[i+2]`. Chained drafting feeds the drafter's own post-norm hidden as the
next step's `target_hidden` with `position += 1`. `target_hidden` is the target's
**post-final-norm** hidden.

TT module (`tt/mtp.py`, single device): `Qwen36GatedAttention` +
`Qwen36MLP` + framework RMSNorms, shared `model.embd` / `model.lm_head_weight`,
a dedicated paged KV pair (same shape as one target attn layer) with an identity
page table. All drafter forwards are T=1 paged decode steps (Branch B) — a
1-layer step is cheap, so per-token drafting avoids prefill-shape alignment
entirely. Weights load straight from safetensors (`load_qwen36_mtp_state_dict`)
because the target's HF-based loader (`Qwen3_5ForCausalLM`) drops `mtp.*` via
`_keys_to_ignore_on_load_unexpected`.

## Draft → verify → accept loop (`tt/spec_decode.py`, batch=1, greedy)

The hard constraint vs a pure-KV model: GDN recurrent state is a running
accumulation — a batch-dim verify (gemma4 style) cannot advance it per candidate,
and its per-token update is not rollback-able.

**Verify = one masked-bucket chunk forward.** The existing masked fixed-bucket
prefill path (`_forward_prefill_chunk_masked`, bucket 128) already gives exact
multi-token target semantics from carried state: GDN chunk kernel with
`valid_len` masking, paged KV fill + flexible chunked SDPA at
`chunk_start_idx_tensor`. Per-row logits/hiddens come from a one-hot row-select
matmul + final norm + lm_head over ≤32 rows (fixed programs; row indices are
data).

**GDN state strategy: verify-never-commits + block-aligned deferred commit.**
`paged_fill_cache` writes a chunk's K/V from the start of its first block, so
every chunk forward must start block-aligned (block 64). We therefore keep two
positions:

- `a` — block-aligned *state anchor*: GDN state (all 48 layers) reflects tokens
  `0..a-1` exactly; the target KV for those tokens is committed.
- `c` — position of the last committed token (`c - a < 64 + K`).

Each iteration:
1. Drafter: run pending `(token, hidden, pos)` pairs (the tokens committed last
   iteration, with their true target hiddens from the last verify), then chain
   `K` draft steps on-model.
2. Snapshot GDN state (handle swap — the chunk path *reassigns* state tensors,
   so a snapshot is just the current handles; verified non-mutating:
   `chunk_gated_delta_rule_seq_adapter` and `_causal_conv1d_fir` only read the
   incoming state).
3. Verify chunk `[t_a .. t_c, d_1 .. d_K]` at `chunk_start=a`,
   `valid_len=(c-a)+1+K`; extract logits+hidden rows `(c-a) .. (c-a)+K`.
4. Greedy accept: longest matching prefix `m`, commit `d_1..d_m` plus the
   target's correction/bonus token.
5. **Always restore** the snapshot (deallocate the polluted handles). When
   `c+1-a > 64`, run one *commit chunk* over committed tokens only
   (`valid_len = 64*floor((c-a)/64)`) and advance `a`. The commit always
   leaves at least one committed token uncommitted: the accept row for draft 1
   is the verify row after processing `t_c`, so the anchor must never catch up
   to `c+1`. Amortized target cost: ~`1 + (m+1)/64` chunk forwards per `m+1`
   committed tokens.

KV rollback is implicit (batch=1): rejected-draft K/V rows and the re-processed
prefix are overwritten by the next chunk, which always starts at `a <= c'` and
covers every polluted position before anything attends past them.

Prefill: spec-owned segmented masked-bucket prefill (2048-token segments) with
`valid_len` truncated to `a0 = 64*floor((T-1)/64)`; the prompt tail `t_{a0}..
t_{T-1}` rides in the first (draft-less) verify chunk, which also samples the
first token. Per-segment post-norm hiddens are captured to host to seed the
drafter KV over the prompt (window-capped via `TT_SPEC_SEED_WINDOW`, default
2048 — earlier drafter slots stay zero, an accepted accuracy trade).

## Traced loop (TP, default; TT_SPEC_TRACE=0 for eager)

Two traces, captured lazily and replayed every iteration:

- **Verify trace** — the full masked-bucket-128 chunk (all 64 layers) plus the
  in-trace row-select + final norm + LM head. Everything that varies is DATA in
  persistent replicated buffers refreshed by `copy_host_to_device_tensor`:
  tokens, the RoPE window for `chunk_start..+128`, a FIXED 2-block chunk page
  table, the device `chunk_start`, the 32-row select one-hot, and the GDN
  `valid_len` masks. The masks are the key enabler: the eager masked path
  builds them host-side per call (`from_torch` — illegal under capture), so
  `valid_masks` plumbing (`layer.forward` → `TPGatedDeltaNet.forward_prefill` →
  `_causal_conv1d_fir(valid_sel=...)` / fused-chunk adapter) feeds them as
  caller-owned device tensors instead. GDN state carry is already in-place
  (`_stable_state`), so replays mutate fixed addresses; the graph is
  SELF-RESTORING — it begins by copying the standing anchor snapshot back into
  the state buffers, so the verify-never-commits contract costs zero eager
  dispatches per iteration (the ~2x48 eager `ttnn.copy` round per iteration was
  the dominant fixed cost in the v2 measurement). The snapshot refreshes only
  at commits. Greedy scoring is also in-graph: per-vocab-shard argmax/max (no
  vocab all-gather, no 32xvocab readback), host-combined across shards.
- **Drafter window traces** — an iteration's drafter legs sit at CONSECUTIVE
  positions, so pos/cos/sin for the whole run upload ONCE per iteration into
  window buffers and one trace per window index bakes a static slice of them.
  A leg is one tiny token upload + one replay; the chained hidden stays on
  device, the greedy pick happens on device (pad-to-32 multicore argmax + a
  grid max per vocab shard), and only drafts read back a [2]-value score per
  device — catch-up legs read nothing. No CCL in these traces (the head is
  fully replicated), which sidesteps the distinct-CCL-trace interleaving
  deadlock gemma4 hit.

Capture ordering is load-bearing (#48536: a compile after a trace is parked can
clobber it): prefill segments + eager drafter seeding + the eager first verify
(+ explicit warms for the 32-row extraction, the 2-block fill, and the
score graph) compile every program FIRST; the drafter window traces capture at
iteration 1 (ALL compile passes, then all captures — per-index slice offsets
may compile distinct programs), the verify trace right after — its compile run
compiles nothing new. Commit chunks stay eager (amortized ~1/64 of iterations;
all-warm programs); in traced mode a commit eagerly restores the anchor first
and refreshes the standing snapshot after. TT_SPEC_TIMING=1 logs per-phase
attribution (draft/verify/commit, per-leg draft cost).

Adaptive draft length (TT_SPEC_ADAPTIVE_K=1): K_t = clamp(round(EMA(accepts))+1,
1, K). Purely data-driven — chunk `valid_len`, the masks, and the accept-row
count all ride the same buffers, so it works traced and eager.

## Decode-shaped sequential verify — deferred (fit correction)

The 41337 headline (113.7 ms/token) charged the chunk verify for costs that a
clean K-sweep re-attributed: with in-window capture pollution removed, the fit
is ~26 ms per draft leg and ~123 ms iteration-fixed, of which the verify chunk
itself is only ~30-45 ms — the rest was per-iteration eager dispatch (the
~2x48-layer GDN snapshot/restore copy storm) and per-leg host I/O (per-leg
cos/sin uploads + a full 8-shard logits readback per draft). v3 removes those
on the host side (self-restoring verify trace, window drafter, on-device
scores); the chunk verify STAYS.

If a future measurement shows the remaining verify floor (~30-45 ms) gating,
the decode-shaped verify remains the designed next step: gemma4 batch-alias
attention (W=K+1 pseudo-users, replicated page-table row) + a GDN W-step
no-commit recurrence with per-row state stashes (composition first via the
non-in-place `recurrent_gated_delta_rule_decode_ttnn`; then a `seq_rows`
compile-time mode on the fused `recurrence` generic_op from gdn-decode-fused —
one core per v-head loops the W rows in-kernel, state resident in L1, per-row
stash, no writeback). Commit becomes a host-selected
`ttnn.copy(state_stash[m] -> rec_state)`, which would retire the block-anchor
machinery entirely.

## TP (P150x4 / P150x8), B=1 — built

The spec loop runs on a TP mesh at B=1: the verify/commit chunks go through the
TP masked-bucket path (`_forward_prefill_chunk_masked_tp`), row extraction uses
a replicated one-hot + DistributedNorm + the vocab-sharded LM head's gather,
and the GDN snapshot is COPY-based (`rec_state`/`conv_carry` clones) because
the TP chunk path carries state in place (`_stable_state`). The drafter head
executes fully REPLICATED — weights, KV, and inputs on every device, no CCL; a
1-layer step is small enough that redundant compute beats sharding. It keeps
its own replicated embed_tokens copy (the target's table is hidden-sharded)
and reuses the target's vocab-sharded LM head with host shard-concat.

## Batch > 1 (goal: concurrency 8) — BUILT, eager + traced (TT_SPEC_BATCHED=1)

`tt/spec_decode_batched.py::Qwen36BatchedSpeculativeDecoder` — per-user
drafting, ONE grouped chunk verify per iteration (the c8 fixed-cost
amortization), fully desynced per-user accept/commit:

- **Batched verify**: rides the silicon-validated grouped machinery
  (`prefill_paged_grouped` / `test_gdn_fused_batch`, Bg=8 at bucket 128) with
  two changes: GDN runs `forward_prefill_batched(carry=True)` over the batched
  per-user ANCHOR state rows (`rec_state[B,...]` + `_batched_conv_carry`
  [B,K-1,D]), and each user's per-user attention pass gets its OWN device
  chunk_start, rope window, and page-table rows — the desync is pure data.
  Per-user row extraction splits hidden rows (up to 64 for the first verify's
  prompt tail) from score rows (exactly 32 — the multicore-argmax contract)
  with independent anchors.
- **Verify never commits**: one batched snapshot (2 clones per GDN layer — the
  SAME tensor count as B=1) restored after every verify.
- **Commits desync per user**: when `commit_advance` fires for user u, a B=1
  masked chunk runs on the persistent prefill scratch seeded from snapshot row
  u, and the result is row-written (`TPGatedDeltaNet._write_index`) into the
  live batched state AND the snapshot.
- **Prefill**: per-user segmented masked chunks on the B=1 scratch (per-user
  drafter-KV block ranges via `Qwen36MTPHead(users=B)`), stitched into the
  batched anchor buffers host-side once.
- Host-emulation coverage: `test_batched_loop_desync` (per-user accept rates
  differ -> anchors/accepts/commits desync while every user emits exactly its
  own target sequence), uniform-prompt identity, first-verify pending arming.

**Traced batched loop** (default; TT_SPEC_TRACE=0 for eager, which is also the
adaptive-K mode — the traced drafter is static-K):

- **Batched drafter**: one B-row T=1 trace per window index over an END-ALIGNED
  per-user schedule (every user's last pending lands at leg width-K; shorter
  catch-ups left-pad by replaying their first pending — an idempotent KV
  rewrite). Per iteration: one pos/rope window upload, one tiny token upload
  per leg, per-user greedy picks as rows of ONE 32-row on-device argmax (the
  [B,1,Vs] -> [1,1,B,Vs] row merge goes through ROW_MAJOR — a padded-TILE
  reshape is a host-fallback read, illegal under capture), two tiny readbacks
  per DRAFT leg for all 8 users. Kills the measured 510 ms eager draft phase.
- **Traced grouped verify**: the full B x bucket-128 graph (batched GDN with
  valid_masks as stacked [B,...] device tensors + carry_inplace — the handle
  swap would deallocate a buffer the captured graph reads — plus per-user
  attention over in-graph slices of stacked cos/sin/csi/chunk-page-table
  buffers), self-restoring from the batched anchor snapshot, per-user
  hidden-row + score outputs. ~10 host uploads per iteration total.
- **Capture ordering** (all compiles strictly before any capture): eager first
  verify -> drafter window compile passes -> verify-graph compile pass ->
  drafter captures -> verify capture. Iteration 1's oversized prompt-tail
  window falls back to eager legs (programs warm from seeding).

Banked clue for the u=1 traced lane (job 49490): after the capture-read fix,
an IDENTICAL deterministic CB/L1 clash remains (program 360 vs L1 buffer
1517312) — one deterministic allocation still lands under a parked trace;
suspects are the eager commit-chunk L1 activations between replays and the
iteration-1 eager fallback legs. Diff the allocation set between capture-warm
and replay paths when rotating back; the batched loop shares the commit
pattern, so a batched repro would confirm it.

## v1 limitations (explicit)

- Single device or TP mesh at B=1; greedy sampling; eager (no trace). Batched
  (c8) spec — including spec-on-one-slot-while-others-decode — is not built.
- Drafter prompt seeding costs one T=1 step per seeded position (window-capped).
- vLLM serving integration is out of scope; the hook is `TT_SPEC_DECODE=1` in
  the text demo (reachable from every test id — spec ignores `use_trace`; the
  `spec_128` / `spec_2k` / `paged_128` ids are the probe entry points, and
  `QWEN36_PROMPT_FILE` overrides the prompt distribution).
