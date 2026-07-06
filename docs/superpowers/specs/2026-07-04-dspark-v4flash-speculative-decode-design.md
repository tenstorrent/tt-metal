# DSpark Speculative Decode for ttnn DeepSeek-V4-Flash — Design

Date: 2026-07-04
Status: Approved-by-default (owner delegated approach + verification decisions to recommended defaults)
Scope target: `models/experimental/deepseek_v4_flash/`

## 1. Goal

Add **DSpark block-parallel speculative decoding** to the ttnn `DeepSeekV4Model` so
decode produces multiple tokens per target forward, with output that is lossless at
`temperature=0` (greedy) versus the plain autoregressive target. Success is measured
by (a) draft-logit PCC vs. the reference implementation and (b) mean accepted block
length + end-to-end tokens/sec on the demo prompt.

## 2. Background — what DSpark is

DSpark is a draft-then-verify speculative decoder. The `DeepSeek-V4-Flash-DSpark`
checkpoint bundles the draft model under the `mtp.*` namespace as **3 full V4 decoder
stages** (`mtp.0/1/2`), each an MLA sliding-window attention (`compress_ratio == 0`,
no compressor/indexer) + 256-expert MoE + hyperconnections. Special pieces:

- `mtp.0`: `main_proj` + `main_norm` — projects the concatenated target hidden states
  from `dspark_target_layer_ids = [40, 41, 42]` down to `hidden_size`.
- `mtp.2`: final `norm`, `hc_head`, `markov_head` (`markov_rank=256`), `confidence_head`.

Relevant config (`config.json` / `inference/config.json`):
`dspark_block_size=5`, `dspark_noise_token_id=128799`,
`dspark_target_layer_ids=[40,41,42]`, `dspark_markov_rank=256`, `hc_mult=4`,
`sliding_window=128`, `hidden_size=4096`, `vocab_size=129280`.

The authoritative implementation is the checkpoint's `inference/model.py`
(`DSparkBlock`, `DSparkAttention`, `DSparkMarkovHead`, `DSparkConfidenceHead`,
`Transformer.forward_spec`). The HF `modular_deepseek_v4.py` deliberately ignores
`mtp.*` (`_keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]`), and the bundled
`generate.py` is plain autoregressive — **the accept/verify loop is not written
anywhere** and is the core deliverable here.

### Per-step algorithm

1. **Propose** (draft, on `dspark_device`): embed `[last_token, MASK*(bs-1)]`
   (`bs = block_size`), replicate across `hc_mult` streams, project the target
   `main_hidden` via `main_proj`/`main_norm`, run the 3 mtp stages. `DSparkAttention`
   attends over the target layer's **sliding-window KV cache** plus the block's own KV
   (`get_dspark_topk_idxs`). Then `hc_head` → `norm` → `head` gives block base logits;
   the Markov head autoregressively adds a token-conditioned bias and greedily/temp
   samples the `bs` draft tokens; the confidence head scores each position.
2. **Verify** (target): one forward over `[last_token, d1..d_bs]` at consecutive
   positions `pos .. pos+bs`, yielding `[1, bs+1, vocab]` logits and a fresh
   `main_hidden` for the accepted positions.
3. **Accept**: longest matching prefix. Greedy (T=0): accept `d_i` while
   `d_i == argmax(target_logits[i-1])`; the first mismatch is replaced by the target
   argmax; commit `accepted+1` tokens. Advance the sliding/compressor caches to the
   committed length; update the draft context (`main_hidden`, last token).

## 3. Current codebase state (already scaffolded)

- `DeepSeekV4Model.dspark_device`: with `layers_per_device=7` the 43-layer stack ends
  on submesh 6, and submesh 7 is reserved for DSpark (falls back to last submesh).
- `decode_main_hidden(token_id, pos, rope)` / `_decode_impl(..., collect_targets)`:
  taps `main_hidden = concat_dim=-1( mean(streams, dim=hc) for layer in target_ids )`
  = `[B, 1, len(target)*hidden]`. **Eager path only** (`decode_traced` does not tap it).
- `DeepseekV4WeightLoader` reads `mtp.*` tensors by raw checkpoint name (verified);
  fp8 `.scale` companions resolve for `mtp.*.weight` too.
- Existing reusable ttnn blocks: `DeepSeekV4DecoderLayer`, attention, MoE
  (`DeepSeekV4PreloadedExperts`, `DeepSeekV4HashRouter`), `DeepSeekV4HyperHead`,
  `DeepSeekV4RMSNorm`, `Linear`, `make_rope_table`, `_LayerKVCache`.

## 4. Chosen approach

**Approach A — eager, correctness-first.** Build all three units on the existing
*eager* `decode` path. Validate draft logits (PCC) and accepted-length parity against
the reference `inference/model.py` executed on host. Reasons: `S>1` verify is
straightforward eager (no fixed-shape trace to fight); the draft module and verify are
structured to be reusable by a later traced fast-path (Approach B, out of scope here).

**Verification semantics:** greedy / lossless at `temperature=0` first
(accept while draft token == target argmax). Rejection sampling (draft_probs vs
target_probs, `sample_residual`) is a follow-up for `temperature > 0`.

Rejected alternatives: (B) trace integration up front — high risk before correctness;
(C) host torch draft — never exercises the draft on device, slow host readback per step.

## 5. Components / units

### 5.1 `tt/dspark.py` — DSpark draft module
- **`DSparkAttention`**: MLA sliding attention with `compress_ratio == 0`. Reads the
  paired target layer's sliding-window KV cache (window=128) and appends the draft
  block's own KV; RoPE at positions `pos+1 .. pos+bs`. No compressor/indexer.
- **`DSparkStage`** (mtp block): reuse hyperconnection pre/post + attn + MoE from the
  existing decoder layer; stage 0 adds `main_proj`/`main_norm`; stage 2 adds
  `norm`/`hc_head`/`markov_head`/`confidence_head`.
- **`DSparkMarkovHead`**: `markov_w1` (embedding, rank 256) + `markov_w2`
  (rank→vocab); autoregressive per-position logit bias + sampling loop over `bs`.
- **`DSparkConfidenceHead`**: `Linear(hidden + markov_rank, 1)` → per-position score.
- **`DSparkDraft.propose(main_hidden, last_token, pos) -> (draft_ids[bs],
  draft_probs[bs,vocab] | None, confidence[bs])`**.
- Interface/deps: depends on the target's per-layer sliding KV caches for
  `target_layer_ids` (read-only) and the shared `embed`/`head` weights; owns the
  `mtp.*` weights on `dspark_device`.

### 5.2 Target-side multi-token verify (in `tt/model.py`)
- **`decode_verify(tokens: list[int], start_pos: int) -> (logits[1,k,vocab],
  main_hidden)`**: feed `k` tokens at consecutive positions in one eager pass against
  the in-place caches, taps `main_hidden` for the target layers. Generalizes today's
  strictly-`S=1` `_decode_impl`. Cache advance must commit only accepted positions.
- Requires the per-layer sliding-window / compressor cache writes and the additive
  masks to handle `S = k > 1` at consecutive absolute positions (today's decode masks
  are single-row). This is the largest genuinely-new capability.

### 5.3 Speculative loop + test
- **`decode_dspark(last_token, pos) -> (committed_ids, new_pos)`** on the model,
  orchestrating propose → verify → accept → commit → context-update.
- **Test** `tests/test_dspark_*.py`: (1) draft-logit PCC vs. reference host run for a
  fixed prefix; (2) end-to-end greedy generation equals plain autoregressive greedy
  output; report mean accepted length and tokens/sec.

## 6. Data flow

```
target decode (eager) --main_hidden[40,41,42]--> DSparkDraft.propose
                                                       |
                                        draft_ids[bs], draft_probs, confidence
                                                       v
target decode_verify([last, d1..d_bs]) --logits[bs+1]--> accept prefix (greedy)
                                                       |
                        committed tokens + resampled correction; caches advanced
                                                       v
                                    loop with new last_token / pos / main_hidden
```

## 7. Error handling / edge cases
- Empty/`0`-length proposal (confidence truncation) → fall back to committing the
  single target token (plain decode step).
- Block crossing sliding-window / compressor boundaries: verify positions are
  contiguous; ensure per-position window indices and compressor window emission match
  the multi-token case (the compressor emits on `(pos+1) % ratio == 0`).
- EOS inside an accepted block → stop at the EOS position.
- Capped stacks (`DEEPSEEK_V4_DECODE_LAYERS < 41`) exclude some `target_layer_ids`;
  `main_hidden` tap returns `None` → DSpark disabled with a clear log (demo asserts
  full stack when DSpark is on).

## 8. Testing strategy
- Host golden: run reference `inference/model.py` `forward`+`forward_spec` on the same
  prefix; compare draft block logits (PCC ≥ target used elsewhere in this model, e.g.
  ~0.98) and sampled draft ids.
- Parity: greedy `decode_dspark` output ids == greedy plain `decode` output ids.
- Metrics: mean accepted length (expect >1), tokens/sec vs. baseline eager decode.

## 9. Out of scope (follow-ups)
- Approach B: folding draft+verify into the captured per-submesh traces (fixed
  `block_size` padded verify trace + `main_hidden` socket-copy to the dspark submesh).
- Rejection sampling for `temperature > 0`.
- Batch size > 1.
