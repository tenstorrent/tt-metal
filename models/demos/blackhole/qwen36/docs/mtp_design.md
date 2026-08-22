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

## Trace compatibility

v1 runs eager. The verify/commit chunks use only masked-bucket programs (a
bounded, warmup-compilable set — the same property that made the serving path
trace-safe), and the drafter is a fixed-shape T=1 step, so a later fused
per-iteration trace (gemma4 `_capture_fused_trace` style: persistent inputs,
on-device argmax/re-embed, one `execute_trace` per iteration) is the natural
next step. Drafting inside the existing decode trace is not attempted: the
decode trace is a single-token program, while spec decode replaces single-token
decode entirely.

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

## Batch > 1 (goal: concurrency 8) — design only, NOT built

- Verify: per-user B=1 chunk forwards against the user's page-table row and GDN
  slot (mirroring `prefill_paged_slots`), or a batched GDN chunk kernel at B=8
  once available. The per-slot dense GDN state means the snapshot/commit scheme
  applies per slot unchanged.
- Drafter: the MTP step is a standard B-wide paged decode — batch the pending +
  chained steps across users directly.
- Scheduling: users have different `m` per iteration; commit points diverge, so
  per-user `(a, c)` bookkeeping (already per-slot in this design) carries over.

## v1 limitations (explicit)

- Single device or TP mesh at B=1; greedy sampling; eager (no trace). Batched
  (c8) spec — including spec-on-one-slot-while-others-decode — is not built.
- Drafter prompt seeding costs one T=1 step per seeded position (window-capped).
- vLLM serving integration is out of scope; the hook is `TT_SPEC_DECODE=1` in
  the text demo (reachable from every test id — spec ignores `use_trace`; the
  `spec_128` / `spec_2k` / `paged_128` ids are the probe entry points, and
  `QWEN36_PROMPT_FILE` overrides the prompt distribution).
