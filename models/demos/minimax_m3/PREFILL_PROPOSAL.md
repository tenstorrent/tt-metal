# MiniMax-M3 Prefill — Bring-Up Proposal

> **Goal of this document:** Define the architecture and work breakdown for a
> *functional* (correctness-first, not optimized) **MiniMax-M3** prefill pipeline
> on a single Blackhole Galaxy, by **transforming the already-validated MiniMax-M2.7
> codebase in place** (same folder, renamed to `minimax_m3` — NOT a fork).
>
> **This is a LIVING doc** — kept in sync with the code as the source of truth for
> the team and for handing context to an agent (especially when we move onto the
> Galaxy box). Rule: it never claims something works that isn't validated; status is
> either *validated (with PCC + conditions)* or *scaffold (owner + blocker)*.
>
> **Pivot note (2026-06-18):** We are switching the target from M2.7 to **M3**. M3 is
> a major rearchitecture (sparse attention, 1M context, hybrid dense/MoE, doubled
> dims, native multimodal). We keep the validated M2.7 pipeline as the **foundation**
> and convert it in place. Multimodality is **out of scope for now** (text backbone first).

---

## 0. Status (living) — last updated 2026-06-18

### 0.1 What we inherit (validated MiniMax-M2.7 base — the foundation we convert)
Everything below was validated on a real Blackhole Galaxy (32 chips) for **M2.7** and
is the starting point we transform. **It is NOT M3-validated.**
- attention / router / experts / full decoder layer — PCC vs HF @ TP=1 (random weights).
- **TP=8 multi-device** correct vs HF (attn 0.9991, experts 0.9989) @ mesh `(1,8)`.
- **Real M2.7 weights → first token VERIFIED vs HF** (argmax + top-5 match; logit PCC 0.953).
- **EP=32 end-to-end** ( `(4,8)` DP-attention + shared EP MoE ), on-device EP bridge (traceable).
- First traced full-model prefill perf ~9.3K tok/s (device-bound, untuned).
- Code/docs MiniMax-only; Galaxy run recipe + env rules captured in §10.

### 0.2 The M3 switch — status (Phase 0 in progress, 2026-06-18)
M3 architecture understood from the HF config (`MiniMaxAI/MiniMax-M3`, `text_config`) + MSA
repo/paper (§2–§3). Big finding (§4): **the team is already building the DSA sparse-attention
op pipeline** (`topk_large_indices` merged, `indexer_score` in PR, `sparse_sdpa` MLA-variant in
progress) — DSA == MSA's NSA family, so M3 attention is largely **integration, not a new kernel**.

**Phase 0 progress:**
- ✅ **Folder renamed** `models/demos/minimax_m2` → `minimax_m3` (+ config dir `MiniMax-M2`→`MiniMax-M3`);
  import/path refs + our program-config class names swept; `py_compile` green.
- ✅ **`config.json` re-dimmed** to M3 (`configs/MiniMax-M3/config.json`): hidden 6144, 64/4 heads,
  60 layers, 1M pos, bf16 (no quant), + new fields (`moe_layer_freq`, `sparse_attention_config`,
  `n_shared_experts`, `dense_intermediate_size`, `swiglu_alpha/limit`, `qk_norm_type=per_head`,
  `use_gemma_norm`, `routed_scaling_factor`) and the dense-vs-MSA layer schedule.
- ⏳ **`ModelArgs` + consumers** — NOT done. Key gotcha: `ModelArgs` reads dims from `self.hf_config`
  (loaded HF AutoConfig); **M3's HF config is NESTED** (`text_config.*`) under the VL wrapper, so the
  real-weights path must read `hf_config.text_config` (the dummy/dims path uses our flat config.json).
  Plus: consume the layer schedule, shared expert, sparse_attention_config, swiglu clamp, gemma norm,
  bf16 load + strip `language_model.` prefix.
- ⏳ **Test-rewiring** — HF-ref class strings (`MiniMaxM2DecoderLayer`→M3 `…Sparse…`), add a
  `minimax_m3` mesh-config to the deepseek hub, doc/README prose. Deferred (need semantic decisions).

### 0.3 ⚠️ GAPS — what is NOT done (read before assuming "it works")
- **Nothing M3 yet** — no re-dim, no rename, no M3 config wired.
- **KV cache:** current M2.7 code runs **full non-cached SDPA on fresh Q/K/V** (the
  `kv_cache`/`page_table` args are dead). Target is **chunked KV** (DeepSeek-style, §5),
  NOT vLLM paged attention. Chunked KV is not wired here yet.
- **MSA:** not wired. Depends on the DSA ops (§4) + a **GQA variant of `sparse_sdpa`** (we own).
- **MoE deltas** (128 experts / top-4 / +1 shared / first-3-dense / routed_scaling) not done.
- **Parallelism** for M3 (TP=4 + SP) not decided/implemented (§6).
- **Runner / pipeline / scheduler / KV migration** — still scaffold (inherited from M2.7).
- **Multimodal** — out of scope for now.

---

## 1. MiniMax-M3 at a glance

From `MiniMaxAI/MiniMax-M3` HF config (`text_config`). Full multimodal class is
`MiniMaxM3SparseForConditionalGeneration`; the text backbone is `MiniMaxM3SparseForCausalLM`.

- ~**428B** params / ~**23B** activated (M2.7: ~229B / ~10B).
- **1M context** (`max_position_embeddings = 1,048,576`).
- **Native multimodal** (text + image + video) — **we skip this; text backbone only.**
- **Weights ship bf16** (VERIFIED: safetensors index `total_size` = **869 GB**, every shard-header
  tensor `BF16`; no `quantization_config`, unlike M2.7's fp8). Loading path: download bf16 →
  **quantize on-device to bf8/bfp4** (no fp8-dequant step M2.7 needed). The teammate's "~420GB bf8"
  is the on-device target, not the source.
- **Weight names** carry a `language_model.` prefix (multimodal wrapper) — strip `language_model.model.`
  for the text backbone. Confirmed in checkpoint: `layers.N.mlp.{gate,up,down}_proj` (dense MLP, layers
  0–2, gate/up `[12288,6144]`), `self_attn.{q,k}_norm [128]` (per-head QK-norm), `self_attn.{q,k,v,o}_proj`.

---

## 2. M3 vs M2.7 — the deltas (what changes in our code)

### 2.1 Dims (text backbone)
| | M2.7 (have) | M3 (target) |
|---|---|---|
| hidden_size | 3072 | **6144** (2×) |
| num_hidden_layers | 62 | 60 |
| num_attention_heads | 48 | **64** |
| num_key_value_heads | 8 | **4** |
| head_dim / rotary_dim (partial) | 128 / 64 | 128 / 64 (same) |
| rope_theta / vocab | 5e6 / 200064 | 5e6 / 200064 (same) |
| max_position | 196608 | **1,048,576** |

### 2.2 The five architectural deltas (each touches code)
1. **MSA replaces GQA attention** (layers 3–59). Block-sparse: lightweight indexer scores
   128-token blocks, top-16 selected, attention restricted to them (+ init/local blocks).
   **Layers 0–2 stay full attention.** See §3. *(biggest item; mostly integration — §4)*
2. **Hybrid dense/MoE + shared expert.** `moe_layer_freq=[0,0,0,1,…]` → **layers 0–2 are
   dense MLP** (`dense_intermediate_size=12288`); layers 3+ are MoE with **128 experts /
   top-4** (M2.7: 256/top-8) **+ 1 always-on shared expert** (M2.7 had none),
   `routed_scaling_factor=2.0`. See §7.
3. **Clamped SwiGLU** (`hidden_act="swigluoai"`, `swiglu_alpha=1.702`, `swiglu_limit=7.0`) —
   gpt-oss-style clamp (M2.7 used plain SiLU SwiGLU).
4. **Norms:** QK-norm `per_layer`→**`per_head`**; RMSNorm→**Gemma `(1+w)`** (`use_gemma_norm=true`).
5. **Vision tower** (CLIP-like + 3D-RoPE video + projector). **Out of scope for now.**

### 2.3 What stays the same (reuse directly)
- Embedding, partial-RoPE geometry (rotary_dim 64), sigmoid + correction-bias routing
  **logic** (counts differ), EP dispatch/combine machinery, the serving/migration scaffold,
  the Galaxy run recipe (§10).

---

## 3. MSA — MiniMax Sparse Attention (the new attention block)

> Source of truth: HF `text_config.sparse_attention_config` + the MSA repo
> (`github.com/MiniMax-AI/MSA`, NVIDIA SM100 kernel) + paper `arXiv:2606.13392`.
> The MSA kernel is **NVIDIA-only** — there is no TT version; the *algorithm* is what
> we port, and it's the standard NSA/DSA "indexer → top-k → sparse attention" family.

### 3.1 Config (per-layer; layers 0–2 are dense, 3–59 sparse)
`sparse_block_size=128`, `sparse_topk_blocks=16`, `sparse_num_index_heads=4`,
`sparse_index_dim=128`, `sparse_score_type="max"`, `sparse_init_block=0` (sink),
`sparse_local_block=1` (recent block), `sparse_attention_freq=[0,0,0,1,…]`.

**VERIFIED projection shapes** (real M3 checkpoint, layer 3 `self_attn`, bf16):
- main attn: `q_proj 6144→8192` (64h×128), `k_proj 6144→512` (4h×128), `v_proj 6144→512`
  (4h×128), `o_proj 8192→6144`; `q_norm`/`k_norm` `[128]` (per-head). GQA group = 64/4 = 16.
  K and V both head_dim 128 — **plain GQA, no latent split**.
- indexer (MQA-style): `index_q_proj 6144→512` (**4** heads×128), `index_k_proj 6144→128`
  (**1** head×128), `index_q_norm`/`index_k_norm` `[128]`. Matches `indexer_score`'s
  `q[B,4,Sq,128] / k[B,1,T,128]` signature exactly.

### 3.2 Block scheme (one sparse layer's attention)
```
 h [S, 6144]
   ├─ Wq ─► Q [S, 64, 128] ─┐
   ├─ Wk ─► K [S,  4, 128] ─┤ per_head QK-norm ─► partial RoPE (dims 0..63)
   ├─ Wv ─► V [S,  4, 128] ─┘
   │
   │   INDEXER ("lightning indexer", MQA-style): iQ [S,4,128] (index_q_proj→512),
   │                                              iK [S,1,128] (index_k_proj→128) + own qk-norms
   │     STEP 1  per-key score   s[s,t] = Σ_h relu(iQ·iKᵀ)·w          (indexer_score op — §4)
   │     STEP 1b block-pool      block_score[s,b] = max over 128 keys in b   ← M3-specific glue
   │     STEP 2  top-k blocks    keep top-16 blocks (+ always block 0 + local)  (topk op — §4)
   │     STEP 2b expand          chosen blocks → key-index list (into the KV buffer)
   │
   ▼   STEP 3  SPARSE SDPA       softmax(Q·Kᵀ/√d)·V  over ONLY the gathered keys
   │           ~16×128 = 2048 keys/query regardless of S  → O(S·2048), not O(S²)
   │           (sparse_sdpa op — §4; we need a GQA variant)
   ▼
   concat heads ─► Wo ─► reduce-scatter ─► out [S, 6144]
```
DSA selects per-**key** (top-N keys); MSA selects per-**block** (top-16 of 128-token blocks,
score = max over the block) → STEP 1b/2b (block-pool + expand) is the M3-specific glue around
the reusable ops. Indexer wiring now **verified** from the checkpoint: separate `index_{q,k}_proj`
(4 q-heads / 1 k-head, MQA-style) each with their own `[128]` qk-norm. Still to confirm vs modeling
code: the `w` gating term in STEP 1, whether RoPE applies to the index heads, and where block-pool lives.

---

## 4. Reuse map — the team's DSA sparse-attention ops

The org is converging DeepSeek-V3.2 / GLM-5.1 / V4 / M3 onto the same three Blackhole ops.
They map 1:1 onto §3's steps.

| Op | = MSA step | Status / owner | M3 reuse |
|---|---|---|---|
| `ttnn.experimental.topk_large_indices` | ② top-k select | **MERGED** (#46833, pavlejosipovic) | **as-is** (K≤2048; M3 top-16) |
| `ttnn.experimental.indexer_score` | ① block/key scoring | **open PR** #47223 (skrsticTT); GLM=8-head, DS-V3.2=16/64-head | M3 = **4-head/dim-128** config + **block-max-pool** glue (§3.2) |
| `sparse_sdpa` | ③ sparse attention | **WIP branch** `pjosipovic/sparse_mla_prefill_ref`; **pavle is ADDING GQA support** (confirmed) | **He owns the kernel**; we provide M3 GQA **shapes + a torch test case** (golden). Our Phase 3c shrinks from "build a variant" to "spec + golden + integrate". |

**Consume (reuse):** `topk_large_indices`, `indexer_score`, EP dispatch/combine, full GQA
SDPA (layers 0–2), DeepSeek chunked-KV substrate (§5), the `deepseek_v32` demo + its torch
reference `reference_cpu/sparse_sdpa_prefill.py` as a golden template.
**Own (build):** M3 re-dim/config, dense+MoE deltas, the **block-max-pool adapter** (§3.2),
the **GQA `sparse_sdpa` variant**, full-model assembly.
**Coordinate:** skrsticTT (indexer + ring/MLA SDPA area), pavlejosipovic (topk + sparse_sdpa).
**ANSWERED (pavle):** GQA is required and he'll add it to `sparse_sdpa` — *"give me the shapes + a
torch test case."* So Phase 3c is now: we deliver the verified GQA shapes (§3.1) + a torch golden,
he builds the kernel. Open Q for him: index format — key-level `[1,S,2048]` (like his MLA op) or
block-level `[1,S,16]`?

**M3 GQA `sparse_sdpa` op tensors** (per chip, mirroring his MLA op; deltas vs MLA = GQA grouping +
plain head_dim 128, no latent split): `Q [1,64,S,128]` bf16 · `K/V [1,4,T,128]` bf16 · `Indices
[1,S,TOPK]` uint32 (per-query, shared across q heads, `0xFFFFFFFF`=masked) · `Out [1,64,S,128]` bf16
· `TOPK=16×128=2048`, `scale=128**-0.5`, S/T/TOPK parametric.

---

## 5. KV cache — chunked (DeepSeek reference), NOT paged

> **Reference: `models/demos/deepseek_v3_d_p/tt/mla/mla.py`** (`_chunked_attn`,
> `update_padded_kv_cache`, `_chunked_kv_buf`). This is our model for M3 KV.

- **NOT vLLM paged attention** (no page tables / `max_num_blocks`). DeepSeek uses a
  **persistent per-chip chunked KV buffer**: each chunk is written via
  `ttnn.experimental.deepseek_prefill.update_padded_kv_cache` (per-chip write offset derived
  on-device from `kv_actual_isl` = cumulative valid tokens so far), and attention reads the
  accumulated cache (`ring_mla` for MLA). SP-sharded across rows.
- **Current M2.7 code does NEITHER** — full non-cached SDPA on the call's fresh Q/K/V; the
  `kv_cache`/`page_table` args are dead. (Earlier drafts of this doc wrongly said "paged" — fixed.)
- **For M3 + MSA:** the chunked KV buffer holds all K/V; **`sparse_sdpa` gathers the selected
  key rows from it by index** (the index list from §3 STEP 2b). So M3 KV = DeepSeek chunked-KV
  substrate + index-gather on top. Block-size for MSA selection = 128 (`sparse_block_size`).

---

## 6. Parallelism on the Galaxy (direction — TP=4 + SP, SP over DP)

> Galaxy = 4×8 = 32 Blackhole chips. Direction below; exact layout still to confirm w/ team.

### 6.1 Axes
- **TP=4** (was 8 for M2.7). Forced by **`num_key_value_heads=4`** — TP=8 can't shard 4 KV
  heads cleanly; TP=4 → 1 KV head/chip. Also matches how DS shards its 64-head indexer (TP=4).
- **SP (sequence parallel)** — needed because M3 targets **1M context**; a single prompt's
  sequence is sharded across chips (DeepSeek-style chunked prefill + SP). This is the
  long-context path M2.7 never finished.
- **EP for experts** — 128 experts spread across chips; exact EP mapping rides on the SP layout.

### 6.2 TP and SP apply to BOTH dense and MSA layers
The dense (0–2) vs MSA (3–59) split is **not** a parallelism difference — it's an attention-
*compute* difference (full O(S²) vs sparse O(S·2048)). Both use the same TP=4 head-shard and the
same SP sequence-shard. Evidence: skrstic's `indexer_score` op already shards the 64-head DS
indexer across TP=4, so MSA's indexer is TP-sharded just like dense attention. (Earlier framing
of "TP+SP for dense, SP-only for MSA" was wrong.)

### 6.3 The chip budget: SP over DP
`DP × TP × SP = 32`, and with `TP=4` → `DP × SP = 8`. DP and SP **trade off**:
```
   Config A (long context):  DP=1, TP=4, SP=8   → one prompt, 1M ctx sharded 8 ways   ← M3's headline
   Config B (throughput):    DP=4, TP=4, SP=2   → 4 prompts, ctx sharded only 2 ways  (~500K KV/chip/prompt)
```
**Recommendation: spend chips on SP, not DP.** For a 1M-context model SP is the scarce resource;
DP=4 (inherited from gpt-oss-120b) steals exactly the chips long context needs. **Multi-user =
replicate galaxies** (one TP=4×SP=8 replica per galaxy, add galaxies for more users — "galaxy-level
DP"), **NOT** PP and **NOT** DP-on-one-box. PP would only help if M3 didn't fit on one galaxy — it
does, so PP buys nothing on capacity.

### 6.4 The attention runs SP, body stays TP — via a CCL bracket (team's design, confirmed)
The indexer score is a **sum over heads** (`Σ_h relu(iQ·iKᵀ)`), and TP shards heads — so under TP
you'd need a cross-head all-reduce just to score blocks. The team's plan (pavle): **TP doesn't make
sense for the sparse path** (even on GLM). Instead:
1. **CCL converts TP→SP across the 4 chips** for *both* Q tensors (indexer Q + main Q).
2. **indexer → topk → sparse_sdpa run fully SP** (each chip owns query tokens with ALL heads → the
   head-sum is local; clean per-query block selection).
3. **CCL converts SP→TP back** for the rest of the layer.

The **model body stays TP=4** purely to minimize churn for the rest of the system; *long-term TP
probably goes away* (full SP). So we don't design the cross-SP selection ourselves — the op pipeline
(indexer/topk/sparse_sdpa) consumes the SP layout, and the CCL bracket is the integration seam.

### 6.5 Open decision (blocking Phase 0 configs)
The attention-internal layout is settled (TP→SP CCL bracket, §6.4). Still open: the body's precise
TP=4 × SP × EP physical mapping on 4×8 (which axis is SP vs EP). Reference DeepSeek `(8,4)` SP8/TP4.

---

## 6b. Serving & batching (what prefill owns vs not)

- **Prefill is a stateless KV factory.** `prefill(token_ids, actual_isl, slot_id, dst_slot)`
  prefills one fresh prompt, writes KV, migrates to decode (`dst_slot`). It does **not** cache
  conversations. **Prefix/KV reuse across chat turns is a server + decode concern** — decode holds
  the KV and continues; the server decides what prompt/slot to send. (The `kv_actual_isl` "cumulative
  valid tokens" primitive is the future hook for multi-turn append, but unused for reuse today.)
- **Prefill batch size ≠ decode batch size.** Expert-sharing (the case for batch>1) is a big win in
  *decode* (1 token/user → batching fills skinny expert matmuls) but **weak in prefill**: a single
  prompt's chunk already carries thousands of tokens (5K-token chunk, 128 exp/top-4 → ~160 tokens/
  expert before any batching). **The knob that fills experts in prefill is chunk size, not batch.**
  Static DP=4 also forces the server to batch ragged-length requests (head-of-line blocking, padding
  waste); **continuous/dynamic batching** is the better lever. → don't inherit DP=4 reflexively.
- **Open experiment (Phase-2 follow-up):** measure prefill expert MFU at **batch=1 vs batch=4** for a
  realistic chunk size. If batch=1 + a ~5K chunk already hits good MFU, the DP=4 question is moot.

---

## 7. MoE for M3 (layers 3–59)

- **Counts:** 128 experts, **top-4** routing, **+1 always-on shared expert**
  (`n_shared_experts=1`, `shared_intermediate_size=3072`), `routed_scaling_factor=2.0`.
  Per-expert `intermediate_size=3072`. (M2.7: 256/top-8, no shared expert.)
- **Reuse from M2.7:** routing **logic** is identical — `tt/topk.py` already does sigmoid →
  +correction-bias (selection only) → top-k → gather unbiased → normalize. Feed it
  `num_experts=128, top_k=4` from config. EP dispatch/combine (`TtMiniMaxMoE`) carries over.
- **New work:** (1) **shared-expert path** (always-on, à la DeepSeek — M2 had none),
  (2) `routed_scaling_factor` multiply on the routed output, (3) **hybrid dense/MoE per-layer
  dispatch** (layers 0–2 dense MLP `dense_intermediate_size=12288`; 3+ MoE), (4) clamped
  SwiGLU (`alpha/limit`) in both dense and expert FFN.

---

## 8. The plan (functional first; multimodal skipped)

**Guiding principle:** build the dense+MoE backbone first (fast, low-risk, all reuse from
M2.7); run the MSA attention track in parallel as **integration** of the DSA ops (§4), not a
from-scratch kernel.

**Decision to make first (gates everything): the TP=4 × SP × EP Galaxy layout (§6).**

- **Phase 0 — Re-dim + rename (in place, NO fork).** Transform `models/demos/minimax_m2` →
  `minimax_m3`: rename folder + all identifiers, wire M3 `text_config` (6144 / 64-4 / 60
  layers), bf16 weight path (no fp8 dequant), drop `minimax_m3_config.py` into the DeepSeek
  reference hub next to `minimax_m2_7_config.py`. Resolve §6 layout. Goal: model instantiates.
- **Phase 1 — Dense layers (0–2), PCC vs HF.** No sparse needed (existing full GQA SDPA).
  Validates per-head QK-norm, Gemma norm, clamped SwiGLU, dense MLP, new dims.
  **Split tests: a dense-layer suite (full attn) vs an MSA-layer suite (sparse).**
- **Phase 2 — MoE block (layers 3+ MLP), full-attention placeholder.** Hybrid dispatch +
  shared expert + 128/top-4 + routed_scaling. Reuse EP. PCC vs HF with full attention.
- **Phase 3 — MSA (parallel track). FIRST GOAL: a composite, UNOPTIMIZED but FUNCTIONAL path
  from already-merged ops to UNBLOCK TESTING.** Then optimize.
  - 3a. consume `topk_large_indices` (merged) — top-16 select.
  - 3b. drive `indexer_score` (#47223) at M3 4-head/dim-128 + add block-max-pool glue.
  - 3c. **GQA `sparse_sdpa`** — pavle builds the kernel; **we deliver the verified GQA shapes
    (§3.1/§4) + a torch test case (golden)**, then integrate. (No longer our kernel item.)
  - 3d. wire indexer→pool→topk→sparse_sdpa on the chunked-KV substrate, inside the TP→SP CCL
    bracket (§6.4); layers 0–2 stay full attention.
  - **Leave `# M3-TODO:` comments at each seam** describing the intended final shape + deps,
    and keep §3/§4/§5 of this doc extended as we learn.
- **Phase 4 — Full 60-layer integration + Galaxy bring-up.** Assemble, re-tune TP/EP/SP, real-
  weights first-token vs HF (same playbook that worked for M2.7). **Comment heavily, here and
  in this doc, so context transfers cleanly on the BH Galaxy box.**
- **Phase 5 — Multimodal. SKIPPED for now.**

```
TP/SP/EP decision ─► Phase 0 ─► Phase 1 (dense) ─► Phase 2 (MoE) ─► Phase 4 ─► Galaxy
                                     │
                     Phase 3 (MSA) ──┘  parallel; gated on #47223 + GQA sparse_sdpa
```

---

## 9. Open decisions (close with the team)
1. **TP=4 × SP × EP layout on 4×8** — blocking Phase 0 (§6). *Recommendation: SP over DP
   (Config A: DP=1/TP=4/SP=8), multi-user via galaxy replication, not DP=4 (§6.3/§6b).* Confirm
   the program's priority (1M-context latency vs 4-user throughput) and validate w/ the batch=1-vs-4
   MFU experiment (§6b).
2. ~~Is a GQA `sparse_sdpa` planned?~~ **CLOSED** — pavle is adding GQA; we owe shapes (§3.1, done)
   + a torch golden. Remaining sub-question for him: index format (key-level `[1,S,2048]` vs block-level).
3. Does `indexer_score`'s `chunk_start_idx` causality compose with our block-pool, or should
   pooling live inside the indexer op? (skrsticTT)
4. Exact MSA indexer wiring (shared vs separate iK, RoPE on index heads) — confirm vs modeling code.
5. Chunked-KV chunk size + SP shard order for 1M context (reference DeepSeek `_chunked_attn`).

---

## 10. Running on a Blackhole Galaxy + env (inherited from M2.7 — update names for M3)

> These rules were learned bringing up M2.7 on the real box; they carry over. Update the
> model dir/name and dims for M3; the **mesh/fabric rules are unchanged**.

### 10.1 Environment
```bash
cd /data/vmelnykov/tt-metal
export TT_METAL_HOME=/data/vmelnykov/tt-metal
export PYTHONPATH=$TT_METAL_HOME
source python_env/bin/activate              # ttnn is built here; /usr/bin/python3 has NO ttnn
# venv has no pip — install with: uv pip install --python $TT_METAL_HOME/python_env/bin/python <pkg>
# M3 modeling code may need a newer transformers than M2.7's 4.57.1 — check the M3 config's
# transformers_version (4.52.4 in the released config) and the trust_remote_code modeling files.
```

### 10.2 Mesh / fabric / MGD — the rules that bite
This box is a **plain 8×4 MESH** (no torus / wrap-around). Two hard rules (else hang/crash):
1. **Mesh-open shape must equal the MGD `device_topology dims`.** Stock `single_bh_galaxy_*`
   MGDs are `[8,4]`. We added `single_bh_galaxy_4x8` (`[4,8]`) and `single_bh_galaxy_1x8`
   (line of 8) under `tt_metal/fabric/mesh_graph_descriptors/`.
2. **Collectives need `FABRIC_1D` + `ttnn.Topology.Linear`** (NOT Ring/torus). The model's
   `CCLManager` defaults to `Ring`; pass `topology=Linear` on this box.
```bash
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_1x8_mesh_graph_descriptor.textproto
```
> For M3, TP=4 changes the mesh mapping vs M2.7's TP=8 — revisit which MGD/shape to open
> (likely a `(8,4)` SP8/TP4-style layout like DeepSeek). This is open (§6).

### 10.3 Verification scripts (inherited; under `tests/`, will be re-pointed at M3)
`galaxy_mesh_smoke.py`, `galaxy_layer_smoke.py`, `galaxy_model_smoke.py`,
`galaxy_first_token.py` (real weights → first token), `hf_reference_oracle.py` (HF CPU golden),
`tests/perf/test_model_perf.py` (traced perf). PCC unit tests need only the modeling code +
`HF_MODEL`; real-weights runs need the bf16 checkpoint (M3 ships bf16 — no dequantize).

> **Perf-test gotcha (carries over):** run `test_model_fwd` directly (not the Tracy wrapper)
> for the full model — Tracy's per-op CSV overflows at 60 layers × 128 experts. Bump
> `trace_region_size` (~1 GB). Kill leftover tracy processes before re-running (they hold
> `CHIP_IN_USE_*` and the next run times out).

---

## 11. Reference implementations
| What to copy | Source |
|---|---|
| Chunked KV cache + chunked prefill | `deepseek_v3_d_p/tt/mla/mla.py` (`_chunked_attn`, `update_padded_kv_cache`) |
| Sparse-attention op pipeline (DSA) | `topk_large_indices` (#46833, merged) · `indexer_score` (#47223) · `sparse_sdpa` (`pjosipovic/sparse_mla_prefill_ref`) |
| Sparse-attention torch golden | `models/demos/deepseek_v32/reference_cpu/sparse_sdpa_prefill.py` |
| Sibling model demo structure | `models/demos/deepseek_v32/` |
| EP dispatch/combine | `ttnn.experimental.deepseek_prefill.{dispatch, routed_expert_ffn, combine}` + our `TtMiniMaxMoE` |
| Runner + per-layer migration pattern | `deepseek_v3_d_p/tt/runners/` + `tt_deepseek_prefill_pipeline.py` |
| Reference config hub (drop `minimax_m3_config.py`) | `deepseek_v3_d_p/reference/` |
