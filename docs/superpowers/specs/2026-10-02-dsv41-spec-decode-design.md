# DeepSeek-V4.1-Flash on BH Galaxy: speculative decoding with the checkpoint's DSpark (mtp.*) layers

Status: M0 design (2026-10-02). Code: `models/demos/blackhole/deepseek_v41_flash/{tt/spec_*.py,tests/test_spec_*.py,reference/ref_spec_*.py}`.
Numbers marked (measured) come from the memory file / earlier runs; (est) are estimates to be replaced by M1/M2 measurements.

## 0. Key finding: the "MTP" of this checkpoint is DSpark, a BLOCK drafter, not a DeepSeek-V3 style 1-token MTP

`config.json`: `num_nextn_predict_layers=3`, `dspark_block_size=5`, `dspark_noise_token_id=128799`, `dspark_target_layer_ids=[37,38,39]`,
`dspark_markov_rank=256`, `dspark_n_routed_experts=128`, `dspark_n_activated_experts=3`. `inference/model.py` implements only the forward pass
(`Transformer.forward_spec`, class `DSparkBlock`, `DSparkAttention`, `DSparkMarkovHead`, `DSparkConfidenceHead`); `generate.py` does NOT call it
("the speculative-decoding loop itself is out of scope for this repo"). So accept/reject is ours to define; the model.py `__main__` shows the call pattern.

The 3 `mtp.N` modules are three STAGES of ONE drafter that are always run together (a pipeline, not 3 alternative 1-token heads):
one draft pass = 3 stacked blocks over a block of 5 positions per user, producing 5 draft tokens.
So "k = 1,2,3 MTP layers" does not exist as such. The tunable is how many of the 5 drafted tokens we verify (k = 1..5); the draft cost and DRAM are fixed.

### 0.1 Exact semantics (from model.py)

Backbone step (`Transformer.forward`, input token x_p at position p) returns `output_ids` (token t_{p+1} sampled from the head) and
`main_hidden` [b, s, 3*5120] = concat over target layers i in (37, 38, 39) of `h.mean(dim=2)` where h is the 4-stream mHC state at the INPUT of layer i
(after Engram, before the block: "the MTP head reads the attention input of its target layers"). Mean over the 4 streams (plain mean, not pre_mix).

`forward_spec(input_ids = t_{p+1}, main_hidden, start_pos = p)`:
1. stage 0 `forward_embed`: `main_x = main_norm(main_proj(main_hidden))` ([b,1,5120]; `main_proj` 15360->5120 fp8 Linear, `main_norm` RMSNorm).
   Draft block ids = `[t_{p+1}, noise, noise, noise, noise]` (noise id 128799); `x = embed(ids)` with the BACKBONE embedding (tied), repeated to 4 mHC streams;
   `pre_mix` = one-hot identity (stream 0). Block position i sits at sequence position p+1+i.
2. For each of the 3 stages `Block.forward` (normal mHC block: hc mixes on attn and ffn, attn_norm, attention, ffn_norm, MoE) with `DSparkAttention` instead of `Attention`:
   * `compress_ratio == 0` (window only; no compressor, no indexer; the compress_ratios list has 0 for layers 40,41,42).
   * `main_kv = kv_norm(wkv_s(main_x))`, RoPE at position p (own stage weights wkv_s, stage's own `kv_norm`), fp8-quantised; written into the stage's
     window ring slot `p % 128` (`window_kv_cache`). At start_pos == 0 (prefill) `main_x` of ALL prompt positions seeds the ring (`attn(x,0,main_x)` only seeds, the block output is dropped).
   * the block's own q/kv come from the block hidden: `q = wq_b(q_norm(wq_a(x)))`, RoPE at positions p+1..p+5; `kv = kv_norm(wkv(x))` RoPE'd.
   * keys = [ring (all slots valid, i.e. positions <= p within the last 128) ; the 5 block kv]; each of the 5 queries attends ALL 5 block keys
     (NON-causal inside the block) + the ring, with the stage's attn_sink. `topk_idxs = [arange(min(128,p+1)), 128+arange(5)]`.
   * same grouped low-rank output projection (wo_a 8 groups, wo_b) as the backbone; same shapes (64 heads, head_dim 512, 1 KV head, q_lora 1280, o_lora 1024).
   * FFN: MoE with **128 routed experts, top-3** (`get_moe_config`), sqrtsoftplus, selection bias, norm_topk, route_scale 1.5, swiglu clamp 10, + 1 shared expert (inter 2304, same as backbone).
   * mHC (`hc_attn_*`, `hc_ffn_*`) per stage exactly like the backbone (hc_mult 4, 20 sinkhorn iterations); `pre_mix` is carried from stage to stage.
3. `forward_head` (last stage's params): `x = hc_pre(x, pre_mix)`; `logits = head(norm_2(x))` for all 5 positions with the BACKBONE LM head weight (tied, `convert.py`) but the stage-2 own `norm`
   (`mtp.2.norm.weight`). Then for i in 0..4 sequentially: `bias_i, markov_embed_i = markov_head(output_ids[i])` where the markov head is a bigram correction:
   `markov_embed = Embedding_{129280x256}(token)` and `bias = markov_embed @ Head_{256x129280}^T`; `logits[:, i] += bias_i`; `output_ids[i+1] = sample(logits[:, i])`
   (greedy for temperature 0). output_ids[0] = t_{p+1} (the already known token), output_ids[1..5] = d_1..d_5 are drafts for positions p+2..p+6.
   `confidence = proj_{5376->1}(cat[x (pre-norm hc_pre output), markov_embed])` per position: a learned acceptance score (fp32 linear). Not needed for greedy-exact verification; usable to choose k per user/round.

Consequences for us:
* Draft and draft-time state depend only on (t_{p+1}, main_hidden at p) and the stage window rings of main_kv for positions <= p. Drafts do not depend on earlier drafts.
* The stage rings need `main_x` (hence `main_proj`, `wkv_s`, RoPE, write) for EVERY accepted position, not only at the frontier: with several tokens per round we compute them for all (1+k) block positions in the verify step (cheap, see 3).
* Engram is not used by the drafter (no engram on layers 40-42). The drafter needs no compressor/indexer/compressed cache.
* Draft quality only affects speed; the output is always the backbone's greedy stream (verification is exact), so the drafter can use cheaper numerics (bfp4 experts, LoFi).

### 0.2 Checkpoint tensors (model.safetensors.index.json, shards 44-46; sizes are on-disk bytes)

Per stage N in {0,1,2} (prefix `mtp.N.`), identical shapes:

| tensor | shape | dtype | MB |
|---|---|---|---|
| attn.wq_a / .scale | [1280,5120] / [40,160] | F8_E4M3 / F8_E8M0 | 6.6 |
| attn.q_norm | [1280] | BF16 | ~0 |
| attn.wq_b / .scale | [32768,1280] / [1024,40] | F8_E4M3 / E8M0 | 41.9 |
| attn.wkv / .scale | [512,5120] / [16,160] | F8 / E8M0 | 2.6 |
| attn.kv_norm, attn.attn_sink | [512] BF16, [64] F32 | | ~0 |
| attn.wo_a / .scale | [8192,4096] / [256,128] | F8 / E8M0 | 33.6 (stored fp8; the reference module holds it bf16, dequantise like backbone) |
| attn.wo_b / .scale | [5120,8192] / [160,256] | F8 / E8M0 | 41.9 |
| attn_norm, ffn_norm | [5120] | BF16 | ~0 |
| ffn.gate.weight / bias / bias_vl | [128,5120] BF16 / [128] F32 / [128] F32 | | 1.3 |
| ffn.experts.{0..127}.w1,w3 / scales | [2304,2560] I8 (packed fp4) / [2304,160] E8M0 | | 5.9+0.37 each |
| ffn.experts.{0..127}.w2 / scale | [5120,1152] I8 / [5120,72] E8M0 | | 5.9+0.37 |
| ffn.shared_experts.w1,w3,w2 + scales | [2304,5120]/[5120,2304] | F8 / E8M0 | 35 (3x11.8) |
| hc_attn_fn, hc_ffn_fn | [24,20480] | F32 | 2.0 each |
| hc_*_base [24], hc_*_scale [3] | | F32 | ~0 |

Stage-specific:
* stage 0: `main_proj.weight [5120,15360] F8 (78.6 MB)`, `main_proj.scale [160,480] E8M0`, `main_norm.weight [5120] BF16`.
* stage 2: `norm.weight [5120] BF16`, `markov_head.embed.weight [129280,256] BF16 (66.2 MB)`, `markov_head.head.weight [129280,256] BF16 (66.2 MB)`, `confidence_head.proj.weight [1,5376] BF16`.
* No embed/head tensors under mtp (tied to `embed.weight` / `head.weight`).

Totals on disk: stage 0 2.65 GB, stage 1 2.57 GB, stage 2 2.71 GB (routed experts 2.44 GB each). Note: 128 experts per stage, not 384 (the index listing of 384 per pattern is the sum over 3 stages).

### 0.3 DRAM budget on device (per chip, 4x8 = 32 chips)

Routed experts: 128 / 32 = 4 experts per chip (GPT-OSS uses the same 128-expert moe_compute layout; the backbone uses 12 per chip).
Per expert 3 x 2304 x 5120 = 35.4 M params -> bfp8 (1.0625 B) 37.6 MB, bfp4 (0.5625) 19.9 MB. Per chip per stage:

| item | bfp8 | bfp4 experts |
|---|---|---|
| routed experts (4/chip) | 150 MB | 80 MB |
| shared expert (replicated, bfp8) | 37 MB | 37 MB |
| attention (wqkv replicated ~10; wq_b, wo_a, wo_b TP-sharded ~16) | ~26 MB | ~26 MB |
| gate, mHC fns (fp32 2x2 MB, replicated), norms | ~6 MB | ~6 MB |
| **per stage** | **~220 MB** | **~150 MB** |

Plus: `main_proj` (bfp8, column-sharded over 8: 10 MB, or replicated 84 MB), markov embed (bf16 replicated, needed as a row-major gather table: 66 MB; or sharded + gather), markov head (vocab-sharded bf16 8 MB/chip), confidence head ~0.
**Whole drafter: 3 x 220 + 10 + 66 + 8 = ~745 MB/chip (bfp8) or ~530 MB/chip (bfp4 experts) = 11-12% of the ~6.5 GB free.**

Cost per additional MTP stage: ~220 MB bfp8 (150 MB bfp4 experts) + 1 window ring. All 3 stages are required for the trained drafter (dropping stages changes the function; a 1-stage drafter would need re-training), so the budget is the fixed ~0.75 GB above, and there is no per-"k" DRAM at all except activations.
Cost per draft depth (verifying k tokens): no weights; per user one more KV position per layer per step and per-token activations:
* backbone ring/window caches get 1+k writes per step; ring size must grow by k (section 2.3); with the pager/linear layout nothing extra.
* drafter window rings: 3 stages x (128+8 slots) x 512 x 2 B = 418 KB per user per chip (replicated over columns), e.g. 1.7 MB/chip at 4 users/row, 13 MB at 32 users/row.
* verify activations: [T_tok, 4, 5120] fp32 streams = 80 KB per token (x ~3 live buffers), logits [T_tok, 16160] fp32 = 65 KB per token per chip. At 64 tokens/row (16 users x k=3) ~ 15 MB. Negligible vs weights.
The KV/paged-capacity agent's design doc (docs/superpowers/specs/2026-10-02-dsv41-kv-paged-capacity-design.md) did not exist when this was written: the section-2 KV layout below must be reconciled with it and its page-size/page-table numbers substituted.

## 1. Expected speed

Measured (batch = tokens per step, host ~5 ms included): 16 tok 65.0 ms, 32 tok 73.7, 64 tok 95.8 (slope ~0.54-0.69 ms/token). Interpolated: 48 tok ~84.5, 80 tok ~107, 96 tok ~119 (est).
Draft pass device time (est, to be measured in M2): 3 stages x ~1.2-1.5 ms at 20 tokens/row (mHC/router/moe_compute/shared/attention) ~4 ms, + main_proj on the 1+k block tokens ~0.3 ms, head on 5 positions
(~0.3 ms), 5 sequential markov steps (embedding gather + 256x16160 matmul + global argmax with 2 allgathers ~0.15 ms each) ~1 ms, accept/select ~0.5 ms: **~6-8 ms, I use 8 ms** (the 2 ms in the task text looks optimistic).
Round time = verify(1+k) + draft + accept. Host loop: ONE round trip per round (the trace ends with accept + draft; host reads emitted tokens + next drafts, builds Engram rows for the next block,
uploads), same structure as today so the ~5 ms host is already in the 65/73.7/95.8 numbers (rows for 1+k tokens/user cost a bit more).

tok/s/user = (1 + A_k) / round, A_k = E[accepted drafts] = sum_j prod_{i<=j} p_i (geometric alpha shown):

| k (n=1+k tok/user) | round ms | a=0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | alpha needed for 20 tok/s |
|---|---|---|---|---|---|---|---|---|---|
| 0 (plain) | 65 | 15.4 | | | | | | | - |
| 1 (32 tok) | 81.7 | 15.9 | 17.1 | 18.4 | 19.6 | 20.8 | 22.0 | 23.3 | 0.63 |
| 2 (48 tok) | 92.5 | 15.0 | 16.9 | 18.9 | 21.2 | 23.7 | 26.4 | 29.3 | 0.55 |
| 3 (64 tok) | 103.8 | 13.7 | 15.6 | 18.1 | 21.0 | 24.4 | 28.4 | 33.1 | 0.57 |
| 4 (80 tok) | 115 | 12.4 | 14.3 | 16.8 | 20.0 | 24.1 | 29.2 | 35.6 | 0.60 |
| 5 (96 tok) | 127 | 11.2 | 13.1 | 15.5 | 18.8 | 23.2 | 29.0 | 36.9 | 0.63 |

k = 2..3 is the sweet spot. 20 tok/s/user needs per-position conditional acceptance ~0.55-0.6 (k=1 needs 0.63 because the draft cost is amortised over fewer tokens).
(With the optimistic 2 ms draft the k=1 requirement would be ~0.52, as in the task text.) Larger batch per row is cheaper per token but this table is for batch 16 users.

## 2. What changes for verifying (1+k) tokens per user in one step

Principle: **treat every (user, j) token as a "virtual user" row** (rows = tokens, exactly what the 32/64-token measurements ran), but let the (1+k) rows of one user
share ONE KV cache through a page table. mHC / router / MoE / shared expert / head / sampling are per-token and need no change (T = 8..24 tokens per device row; mHC/router/MoE verified to T=32 per the task statement; T=20 for the 5-position drafter block is new, chunk 16+4 if a kernel limit shows).

### 2.1 Attention causal inside the block

Today (`tt/attention.py`): cache `[T,1,L,512]` is one row per token-user; `paged_update_cache(cache, kv, update_idxs=pos)` writes 1 row per user; SDPA decode with `cur_pos_tensor` (window layers, linear cache + `sliding_window_size`)
or `is_causal=False` + per-token additive mask `[T,1,NH,L]` (compressed layers: ring and compressed slots in one tensor).

Design (new classes in `tt/spec_attention.py`, subclass `DSV41Attention` / `DSV41CompressedAttention`, same math, same projections):
* **Paged cache** `[num_pages, 1, page, 512]` + `page_table [T_tok, pages_per_user]` int32 row-major, one row per virtual user; the (1+k) rows of one user hold the SAME page ids. Both ops take it:
  `ttnn.experimental.paged_update_cache(cache, kv_rows, update_idxs_tensor=pos_tok, page_table=pt)` (each row writes its own K==V at position p+j; all distinct slots, one call) and
  `scaled_dot_product_attention_decode(..., page_table_tensor=pt, cur_pos_tensor=pos_tok, sliding_window_size=128, attention_sink=...)` for window layers.
  Causality is automatic: token j has `cur_pos = p+j`, so it sees positions <= p+j, which are all written BEFORE the SDPA call (write all (1+k) rows first, then one SDPA over all rows).
  Compressed layers: the existing non-causal + additive mask path with a per-token mask row (already the case: `st["mask"]` is per token, built from that token's position) and `k_chunk_size`; the C++ validation allows paged + non-causal (needs k_chunk_size).
  The paged SDPA at head_dim 512 + sink + sliding window must be probed on BH in M1 (GPT-OSS uses paged+sink+sliding at head_dim 64).
* Cost: SDPA reads the (<=128+comp) window per virtual user instead of per user: (1+k)x cache reads, which is the same as the measured 32/64-token runs.
* Fallback if paged SDPA at d=512 does not work: pack the (1+k) tokens' q heads into one tile of the user (8 heads x (1+k) rows <= 32 -> k <= 3 if the kv-in-q-tile trick moves out) with a per-row mask. Documented, not planned.
* Step state (`tt/step_state.py` analogue in `tt/spec_state.py`): everything (RoPE C/S/nS rows, ring slot, comp slot, mask, Cg/Sg) is a table gather by position; for the block the positions are `p_u + j` (pos_tok = base[:,None] + arange(n)), so the same `DSV41StepState.build` works on a token position vector of length T_tok. Positions are device-side int32; no change except the table size (positions up to p+k).

### 2.2 Ring / compressed cache writes for several positions

* Window layers (linear paged cache + sliding window): positions p..p+k are plain writes; no wrap issue.
* Ring layout (compressed layers keep `[ring 128 | compressed]`): ring slot = pos % 128. A rejected speculative write at position q overwrites the entry of position q-128, which the NEXT round's first query (at position p+m+1, window [p+m-126 .. p+m+1]) may still need when m < k-2. **The ring must be 128 + k slots (use 128+32 = 160 slots for 32-multiples; slot = pos % 160, window enforced by the mask),** or use the paged linear layout with `cache_position_modulo`/sliding window. This is the one place where rollback is NOT free.
* Ratio-1 layers (20-39): latent = rms_norm(x @ wkv) per token, written at compressed slot `pos` (per token); trivially batched.
* Compressed layers read-only (non kv-source): write the owner's `last_lat` rows, per token.

### 2.3 The ratio-2 compressor with several tokens in one step

State today (step-independent mode): `prev_cs` = previous token's [kv|score] fp32 per user; each step pools (prev, current) = softmax over 2 slots, writes the pooled latent at slot WINDOW + pos//2, mask hides it until the group is complete (odd pos).
Block version, all (1+k) tokens at once:
* `cs_j = x_j @ [wkv|wgate]` for all block rows (one matmul on T_tok rows).
* `prev_j = cs_{j-1}` for j >= 1 and `prev_0 = prev_cs[u]`: a row shift inside each user's group of n rows: build with a 0/1 shift matrix matmul on `[U, n, 1024]` or slice+concat on the user axis (device-side).
* pooled_j = pool(prev_j, cs_j); latent_j = rope(rms_norm(pooled_j)); only tokens at ODD positions complete a group. Tokens at even positions would write a junk latent to the slot of the group in progress, which then collides with the
  odd token's real write to the SAME slot inside one `paged_update_cache` call (undefined order). Fix: `comp_idx` table sends even-position writes to a spare trash slot (WINDOW + max_comp - 1, always masked) instead of the group slot.
* A group boundary inside the block is therefore handled by position parity only (token j at pos p+j sees complete groups < (p+j+1)//2; the latent completed by the odd token j' < j is already written because all writes precede SDPA).
* Rollback: `cs_all [U, n, 1024]` is kept; at the start of the next round `prev_cs[u] = cs_all[u, m_u]` (m = accepted count, the last token of the block that was accepted as an INPUT is block index m; the new round's first token is the bonus token at position p+m+1, whose predecessor is block token m). A one-hot select on device, inside the trace.
* A compressed latent written for a rejected position q (odd q) sits in slot q//2 >= comp_len of the new frontier: hidden by the mask, and overwritten (write-before-read in the same step) when the group completes for real. No cleanup.

### 2.4 Engram for each verified position

The Engram hash is a function of the token ids of the last 4 POSITIONS (`NgramHashState.cache[b, pos]`, written at `start_pos..start_pos+L` on every call): so the block tokens' hashes come from one call
`hashes(block_tokens [B, n], start_pos=p)` with the (host-known) draft tokens; rejected entries are simply overwritten by the next call at the new frontier (**no explicit rollback of the host hash state; the cache is position-indexed**).
Rows for the block: B*n*24 rows per Engram layer from RAM (0.25 ms/layer for 16 tokens -> ~1 ms for 48 tokens), uploaded packed (`set_packed_inputs`, row-major, one buffer).
The device Engram forward is per token (rows [T_tok,1,1,Kin]); no change. This is why the host must see the drafts of round r+1 before verify r+1: the trace ends with draft, the host reads (emitted tokens, m, drafts) and uploads rows+tokens.

### 2.5 RoPE / mask step state for several positions per user

`positions_tok = base_u + j` computed on device from `base_u` (int32 [U]) and `n`: `pos_tok [T_tok]`. All tables (`DSV41StepState`) gather by that vector exactly as they do for `pos` today. With device-side accept the next base is `base + m + 1` (a device tensor, no host involvement).

### 2.6 Draft-side state (main_kv rings of the 3 stages)

Per round and per block token: `main_x = main_norm(main_proj(hidden_tok))` (hidden_tok = concat of the 3 target-layer INPUT stream means, tapped inside the verify step), then per stage `kv_s = rope(kv_norm(wkv_s(main_x)))` written to the stage ring at position p+j (virtual-user rows, same page-table machinery, ring 128+8).
The 5-position draft block attends [ring (mask: valid positions <= frontier) ; its own 5 kv held in 8 scratch slots of the same cache tensor] non-causally = the compressed-layer composite path (ring + extra region + mask), 5 virtual rows per user, `is_causal=False`.
Prefill must also seed these rings (main_x of all prompt positions): the device prefill agent has to tap layers 37-39 inputs (or reuse hidden of the last 128 positions only).

## 3. Accept / reject and rollback

Greedy verification of block `[t, d_1..d_k]` (t = last verified/bonus token, position q): backbone logits per position give `a_j = argmax(logits_j)`, j=0..k. `m` = number of leading j with `d_{j+1} == a_j`. Emitted: `a_0 .. a_m` (m+1 tokens, a_m is the new bonus token t'); new frontier position q+m+1. Output is identical to plain greedy decode (up to the numerical difference between batch shapes, see 5).

Device state to restore or select after the round (all inside the trace, per user, one-hot select by m):
1. Backbone window/ring caches and compressed caches: nothing (entries beyond q+m are overwritten later; ring has the 128+k margin; pages are never freed per token, the positions are just rewritten).
2. Ratio-2 `prev_cs`: select `cs_all[:, m]` (section 2.3). (Only the kv-source layers 2, 8, 14 own a ratio-2 compressor and a `prev_cs`; the other ratio-2 layers read `last_lat`.)
3. Drafter main_kv rings: nothing (same overwrite argument).
4. Next draft input: token t' = a_m and `hidden_tok[:, m]` (3x5120): a one-hot select over the n block positions.
5. Host: Engram hash cache is position-indexed (overwritten); output token list gets a_0..a_m.

Where: **device**: argmax per token via `sample_global` on [T_tok] rows, compare with the shifted drafts, m = sum of the cumulative-AND (cumprod) of equality (a handful of small ops on [U, n]), the one-hot selects, then the draft pass; the trace ends with draft tokens `[U,k]`, `a_0..a_k`, `m`.
Host reads one small tensor per round (emitted tokens, m, next drafts: 2.3 ms like today) and uses it for Engram rows + output bookkeeping + EOS; it never has to feed `m` back because the device already applied it. Host-side decision would need a second round trip (read, decide, upload prev_cs/hidden select) = +5-8 ms per round: rejected.
Users accept different numbers: all selects/positions are per user; nothing needs lockstep except the common step count (users that run ahead simply have more tokens emitted that round).
Paged cache with per-user advancing positions: page table static per user; frontier positions per user are a device int32 vector.

## 4. Acceptance-rate measurement

Cheapest meaningful method: run the CPU reference (resident, all 40 layers + the 3 DSpark stages) on public GSM8K questions, generate GREEDILY with the backbone and draft with `forward_spec` after every step. Because drafts depend only on the stream prefix, the exact accepted-prefix statistics of
greedy spec decoding (any k <= 5) follow offline from (stream, drafts); no rollback needed in the measurement. Script: `reference/ref_spec_accept.py` (new file), output `/mnt/tt-data/ssinghal/dsv4-spec-accept/results.pt` (stream, drafts, confidences, main_hidden for prefill and every step: reusable as M2 reference input), log `/mnt/tt-data/ssinghal/dsv4-logs/spec_agent_accept_gsm8k.log`.
Setup: 8 GSM8K test questions of identical templated length (80 tokens, chat mode, `</think>` prefix = direct answer) from /mnt/tt-data/ssinghal/datasets/gsm8k_test.jsonl (public openai/grade-school-math), 64 greedy new tokens each, unquantised activations (FAKE_QUANT off, same as the device reference), temperature 0.
Speed patches only (fp4 LUT dequant, memoised dequant), math unchanged. Started 2026-10-02 ~23:00 on .44 CPU (nice 10, 32 threads), expected 1.5-2.5 h.
Caveats: only 8 prompts x 64 tokens = 512 start positions (+-3% standard error on a rate), math answers only (high-acceptance domain per the user), CPU unquantised reference, not the bf8/bfp8 device numerics (the device logits PCC is 0.85-0.9 vs reference so its greedy stream differs; acceptance on the device stream is measured in M3), checkpoint trained with fp8 activations.
RESULT (finished 2026-10-03 01:16; 8 GSM8K prompts x 64 greedy tokens, 472 start positions with full 5-token lookahead, CPU reference unquantised): per-position conditional match d1..d5 = **0.905 / 0.796 / 0.693 / 0.598 / 0.489** (match = draft j equals the backbone greedy token, NOT conditioned on earlier drafts matching);
mean accepted drafts per round (leading-prefix) A_k = **0.90 / 1.67 / 2.30 / 2.81 / 3.22** for k = 1..5, i.e. tokens/round 1.90 / 2.67 / 3.30 / 3.81 / 4.22. Raw data: /mnt/tt-data/ssinghal/dsv4-spec-accept/results_final.pt.

## 5. Verification-numerics caveat

Verifying on the device gives logits that depend slightly on the batch shape (bfp8 kernels, routing flips at near-ties), so "tokens identical to plain greedy" in M3 can only hold bit-exactly if the plain decode is run with the same shape; the test will compare with plain greedy at the SAME users-per-row and report first-divergence statistics if bit-exactness fails.
Theoretically exact speculative decoding is preserved by always emitting the verifier's own argmaxes (we do).

## 6. Milestones and open risks

* M1 `tt/spec_attention.py` (+ `tt/spec_state.py`, `tt/spec_decoder_core.py` as needed): paged cache probe at d=512 first (go/no-go), then 2-token blocks per user for window layers 0-1 and compressed layers 2+ against `dsv4-chain-m` (6 steps teacher-forced, 16 users), time at 32 tokens.
* M2 `tt/mtp.py`: 3 stages, 128-expert moe_compute (4/chip), per-stage expert cache (tt/moe_weights.py pattern, `weight_cache_dir`), draft block attention, markov head, PCC vs the CPU DSpark reference (saved hidden from the measurement above).
* M3 `tt/spec_decoder.py` loop, identical-output test, tok/s/user and acceptance on device.
* Risks: paged SDPA at head_dim 512; mHC kernels with T=20 tokens (5 draft positions x 4 users); router.py `_forward_exact_fast` hard-codes 384 experts / top-6 (my own subclass for 128/top-3); `moe_compute` with 4 experts/chip on BH; ring margin; the host round trip per round (~5-8 ms) is the largest non-device cost; DRAM numbers wait for the KV-capacity doc.

## 7. Target 50 tok/s/user (batch 16): analysis (2026-10-03)

Tokens per round needed = 0.05 s x round time. Acceptance so far (GSM8K, CPU reference, 20 of 64 steps, 8 users, partial, per-position conditional match 0.88/0.76/0.65/0.58/0.46;
mean accepted drafts A_k = 0.90 / 1.66 / 2.28 / 2.78 / 3.14 for k = 1..5, i.e. tokens/round E = 1+A_k = 1.90 / 2.66 / 3.28 / 3.78 / 4.14).
Round times below are the interpolated step times (65 / 73.7 / 84.5 / 95.8 / 107 / 119 ms for 16..96 rows incl. ~5 ms host) + 8 ms draft estimate; they are replaced by measured values in M1-M3.

| k | rows/user | round (est) | tokens/round | tok/s/user now | round needed for 50 | tokens needed at this round |
|---|---|---|---|---|---|---|
| 1 | 2 | 81.7 | 1.90 | 23.3 | 38 | 4.1 |
| 2 | 3 | 92.5 | 2.66 | 28.8 | 53 | 4.6 |
| 3 | 4 | 103.8 | 3.28 | 31.6 | 66 | 5.2 |
| 4 | 5 | 115 | 3.78 | 32.9 | 76 | 5.8 |
| 5 | 6 | 127 | 4.14 | 32.6 | 83 | 6.4 |

Measured acceptance alone gives ~33 tok/s/user at k=4-5 with today's costs. 50 needs E/round = 0.05 per ms, e.g. k=4: round <= 76 ms (now 115), k=5: <= 83 ms (now 127). The base step is 65 ms, so
the verify rows (+~10.5 ms per extra row index = per 16 rows) and the draft must be nearly free: **round - base step <= ~15 ms at k=4-5**, currently 50-60 ms.
Scenarios (A_k as measured; extra-row cost = ms per extra row index of 16 tokens; host 5 ms taken off the critical path by device-side accept + one-round pipelining):

| scenario | k=2 | k=3 | k=4 | k=5 |
|---|---|---|---|---|
| host off path, draft 8, 10 ms/extra row | 30.2 | 33.5 | 35.0 | 35.1 |
| + draft 4 ms (fused stages, bfp4 drafter), 6 ms/extra row | 35.0 | 40.0 | 43.0 | 44.0 |
| + 3 ms/extra row, base 55 ms | 38.0 | 44.9 | 49.7 | 52.4 |

Where the time must come from (priority order):
1. **Extra-row cost (~10 ms per 16 rows now, drives everything).** It is MoE expert weight traffic: the 1+k rows of one user route to different experts, so the busiest device streams more experts (moe_compute ~83 us per active expert on the busiest device, 40 layers).
   Levers: (a) route the rows of a user with a preference for the experts already loaded for its row 0 (a routing-sharing trick changes outputs, not allowed for exactness; instead only exploit natural overlap),
   (b) dispatch the (1+k) x 16 rows so each device processes a token-block once per expert (already the case; check that expert-tile loading is amortised over tokens: cost per extra row index should fall as rows per expert grow, the measured 8.7/10.8/11.3 ms steps suggest it is not yet), (c) bfp4 experts for the verify path is NOT allowed (accuracy), but LoFi/packing changes in moe_compute are a kernel task for the MoE owner.
   Attention/mHC/router/shared scale ~linearly in rows (~0.1-0.2 ms per layer per 16 rows ~ 4-8 ms per 16 rows): fused mHC and the T-chunked matmuls matter here too.
2. **Host round trip ~5 ms**: device-side accept/reject (section 3) + the next round's Engram rows requires the host to see the new tokens. Pipelining k-step: run the draft for round r+1 on device, upload rows for the *speculated* continuation (all drafts accepted) and re-gather only on mismatch... host-free Engram (hash kernel exists, row gather was dropped) is the clean fix; until then ~5 ms stay.
3. **Draft 8 ms (est)**: 3 stages x (mHC + attention + 128-expert MoE with 20 rows/row-of-mesh). Fuse stage chains, bfp4 experts (acceptance impact is measurable, drafter errors cost only acceptance), 4 experts/chip means the MoE is ~0.2 ms/stage, shared expert/attn/mHC dominate. Target ~4 ms. Also the 5 sequential markov steps (~1 ms): precompute `markov_embed @ head^T` rows lazily: only the rows of sampled tokens are needed, a gather of 256-wide rows then one 256x16160 matmul per step is already minimal; keep.
4. **Adaptive k / draft length from the confidence head**: the confidence score per draft position (sigmoid of the fp32 linear) predicts acceptance. Policy: verify only the draft prefix whose cumulative confidence stays above a threshold; in a fixed-shape trace, shorter blocks cannot shrink the row count, so use 2-3 traced variants (k = 1/3/5, chosen per round for the whole batch by the mean confidence; per-row masking saves nothing) or pad low-confidence rows with copies. With measured per-position acceptance 0.88->0.46 the marginal gain of d5 is 0.46 tokens for ~10 ms: break-even for 50 tok/s needs (marginal tokens)/(marginal ms) >= 0.05, i.e. d5 pays only if its row costs < 9 ms (currently 10-11): **k=5 and k=4 are within noise of each other today; with cheaper rows k=5 wins.** Confidence is useful mostly to pick k per round once the row cost is low.
5. Larger batch is a separate lever (rows amortise expert loads) but the target is stated for batch 16.

Conclusion: 50 tok/s/user is reachable only if (extra-row cost <= ~3 ms per 16 rows) AND (draft <= ~4 ms) AND (host off the critical path) AND k >= 4. The first item is the long pole and is a MoE-kernel/dispatch problem, to be quantified with the measured k=1..5 round times in M1/M3. Acceptance is not the bottleneck (A_5 = 3.1 already).


## 8. Implementation notes (M1/M2, 2026-10-03; code in the private overlay /mnt/tt-data/ssinghal/wt/h44s, deliverable diff changes.diff)

* Paged SDPA decode at d=512 with sink + 128-window + per-row cur_pos works on BH: PCC 0.99998 vs torch (tests/test_spec_probe.py, page 64/256, shuffled pages, n = 2/4/6 rows sharing one user's pages).
* **`paged_update_cache` loses writes when several rows of one call hit the same 32-row tile** (RMW of whole tiles; adjacent positions of one block). Fix: one call per block index j with the other rows' index set to -1 (skip). Costs n extra tiny launches per layer; indices are computed once per round per layer kind (SpecStepState).
* Ratio-2 even positions must not write a latent: the planned "trash slot" (last slot of the page) corrupted reads (a write at row 319 of a 320-slot page changed attention outputs; cause not isolated), writing with idx -1 (skip) is exact.
* Window layers and every compressed layer type (ratio-2 owner, ratio-2 reader, ratio-1 owner, ratio-1 reader) match the original single-token attention at PCC >= 0.99997 for n = 2 and n = 6 (tests/test_spec_attn_vs_orig.py); full layers vs the reference chain (layers 0-3, k = 1 and 5, 3 consecutive blocks): PCC >= 0.9995 for every block index, equal to the single-token path.
