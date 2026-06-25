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

## 0. Status (living) — last updated 2026-06-25

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
- ✅ **Layer schedule + per-layer arch deltas consumed** (Phase 1/2 below).
- ⏳ **`ModelArgs` model-level config plumbing** — still the Phase 0 tail. Key gotcha: `ModelArgs`
  reads dims from `self.hf_config` (loaded HF AutoConfig); **M3's HF config is NESTED** (`text_config.*`)
  under the VL wrapper, so the real-weights path must read `hf_config.text_config` (the dummy/dims path
  uses our flat config.json). Plus `load_state_dict` real path: bf16 load + strip `language_model.`
  prefix + **skip `mtp`/`nextn`** keys (the checkpoint carries 7 MTP modules). Needed for full-model
  assembly + real weights, NOT for the per-module PCC track (which runs on random weights).

### 0.2b Phase 1 (dense path) + Phase 2 (functional MoE) — VALIDATED 2026-06-19
All on a real Blackhole card, **TP=1, random weights**, each vs a **self-authored torch reference**
anchored to the upstream `transformers` `minimax_m3_vl` arch (NEW finding §0.4). PCC 0.97–0.99.
Each module has a `tests/unit/test_*_vs_ref.py`. **Conditions:** TP=1 single-card only (TP>1/CCL
paths unvalidated); random weights (not real); composite/non-fused expert FFN (no fused kernel).
- ✅ Gemma `(1+w)` RMSNorm (`tt/rms_norm.py`).
- ✅ Clamped `swigluoai` SwiGLU (`tt/experts/operations.py` + the routed-expert inline path).
- ✅ Per-head QK-norm (`tt/attention/{operations,prefill,weights}.py`); swizzle/qk-norm interaction
  confirmed correct end-to-end by the full-attention test.
- ✅ Dense MLP, layers 0-2 (`tt/dense_mlp.py`).
- ✅ Full GQA attention block (QKV→split→per-head QK-norm→partial RoPE→GQA causal SDPA→o-proj).
- ✅ Dense decoder layer + **hybrid dense/MoE schedule** (`tt/layer.py`, branch on `moe_layer_freq`).
- ✅ Functional MoE block: `routed_scaling_factor` (after normalize) + always-on **shared expert**
  (reuses `DenseMLP`) + clamped-swigluoai routed experts. **Non-fused FFN** (reused single-device AND
  EP=32; the fused MoE kernel's baked-in activation ≠ M3's, so a clamped-swigluoai fused kernel is a
  later perf task — copy+modify).
- ✅ MoE (sparse) decoder layer.

### 0.2c Full model + MULTI-CARD — VALIDATED 2026-06-21 (functional bring-up COMPLETE)
The dev box is a **32-chip Blackhole Galaxy** (`get_num_devices()==32`), so multi-card was validated
locally (not deferred). All vs self-authored torch refs, random weights, full-GQA placeholder (S<2048).
- ✅ **Full model assembly, single-card** (#9, `test_model_vs_ref.py`): embedding → hybrid layer stack
  → final gemma norm → lm_head → logits, PCC 0.983 + last-token argmax match.
- ✅ **TP=4 multi-card** (#10) on `(8,4)` [stock single_bh_galaxy MGD; tp=cols=4, rows replicate]:
  attention o_proj reduce-scatter/all-gather (0.9993) + dense-MLP down all-reduce (0.9998). TP=4 is
  M3's tensor-parallel factor (forced by 4 KV heads; TP=8 impossible).
- ✅ **EP=32 MoE** (#11, `test_ep_moe_vs_ref.py`) on `(8,4)`: 128 experts/top-4 → 4/device, via
  `CompositeRoutedExpert` (clamped swigluoai = ttnn matmuls + apply_swiglu, reusing deepseek
  dispatch/combine/extract/insert). PCC 0.9831. (Fused-kernel rewrite = later perf, see §7.)
- ✅ **Functional MULTI-CARD WHOLE MODEL** (#12, `test_model_ep_vs_ref.py`) on `(8,4)` = **TP=4 +
  EP=32 + DP=8** (8 prompts, one/row): real prefill I/O (prepare_inputs_prefill → ttnn_prefill_forward
  use_ep_moe → per-row TP-sharded logits gather), all 8 prompts PCC ~0.98.
- Run commands: see `TESTING.md`.

### 0.2d Real weights + SP building blocks — VALIDATED 2026-06-23
- ✅ **Step A done** (ModelArgs real-weights plumbing: `text_config` unwrap + bf16 safetensors, strip
  `language_model.`, drop multimodal; no mtp/nextn in the ckpt). 869 GB bf16 downloaded; weight cache
  built (bf8 attn/dense + bf4 experts).
- ✅ **REAL-WEIGHTS GROUND-TRUTH GOLDEN** (`tests/golden_hf_first_token.py`): the real
  `MiniMaxM3SparseForConditionalGeneration` (HF `minimax_m3_vl`, CPU offload) vs our TTNN bf4 galaxy
  run — **first-token argmax 8/8 MATCH** (every prompt → `<mm:think>`) + **oracle 3/3 exact**
  (`<mm:think>The user`), plus coherent 6-token generation (`tests/galaxy_generate_m3.py`). Chain
  closes: TTNN ==(device PCC)== our refs ==(oracle PCC 1.0)== arch **AND** TTNN bf4 ==(8/8 golden)== real M3.
- ✅ **SP=8 building blocks DE-RISKED standalone** (NOT yet integrated into the model attention):
  - `tests/unit/test_ring_joint_sp_vs_ref.py` — SP attention op: `ring_joint` GQA causal, SP=8×TP=4
    on `(8,4)`, **PCC 0.99998** (grouped V, no inflation). Depends on the kernel fix below.
  - `tests/unit/test_kv_cache_gqa_sp_vs_ref.py` — GQA chunked-KV cache (TP heads + SP block-cyclic
    seq + DRAM NdShard) write/readback, **PCC 0.99997**. `update_padded_kv_cache` is reusable for GQA
    (per-chip 1 head under TP=4) — **no new write op needed**.
- ⚠️ **`ring_joint` grouped-V kernel fix is LOCAL on this branch** (relaxes the tensor-V `NQH==NVH`
  assert to `NQH%NVH==0`; the reader already broadcasts V like K). Branch `ring-joint-gqa-v` is prepped
  for a PR — **pending code-owner review**. The SP op test depends on it.

> **BASELINE vs REARCHITECTURE (read this).** The validated path is **functional M3 on real weights**
> but with **M2-shaped serving mechanics**: full-GQA-everywhere, **single-shot non-cached** SDPA, TP=4
> (+ DP across rows). This is **exact for S ≤ ~2048** (one prompt fits a chip) — the regime the golden
> runs in. The **SP=8 + chunked-KV + MSA** rearchitecture (1M context) is the remaining work; its two
> hardest pieces (the SP attn op + the GQA chunked cache) are de-risked above. **UPDATE 2026-06-25 (§0.2e):
> SP=8 IS now wired into `tt/attention/prefill.py` and validated whole-model (dense + MSA + EP MoE);
> single-chunk done, multi-chunk KV-cache (§5.1) is next.**

### 0.2e SP=8 INTEGRATED into the model + MSA-under-SP + cleanup — 2026-06-25
The §0.2d building blocks are now **wired into the whole-model prefill** (the rearchitecture the BASELINE
note called "next"). Deployment config **TP=4 × SP=8 × EP=32, DP eliminated**. All validated at REAL
shapes/parallelization on `(8,4)` vs composed torch refs (reduced config, random weights):
- ✅ **Dense SP forward** (`ring_joint` no-cache + per-row RoPE + SP-sharded residual): PCC 0.9966,
  per-row uniform. `tt/attention/dense_sp.py` + `prefill.py` gate on `config.sequence_parallel`.
- ✅ **MSA under SP — unblocked.** Added optional per-device `chunk_offset` to `indexer_score_msa`
  (`fb808efb7db`) so SP-sharded queries score/select blocks with correct causality at their true global
  positions; `msa_sp_attention_sharded` keeps Q sharded + AllGathers K/V/index_k across SP. Validated
  chunked at SP=8×TP=4 (`test_msa_sp_chunked_vs_ref`, incl. cross-chunk selection into cached context).
  *`chunk_offset` is a stopgap → switch to the DeepSeek mesh-coordinate approach (functionally equiv,
  cf. #47939) once upstream `indexer_msa` carries per-device coords; that also drops the shared-op overlap.*
- ✅ **MoE under SP** (EP=32 `TtMiniMaxMoE` unified kernel over the seq-sharded residual): PCC 0.976,
  row-uniform. The EP MoE is token-parallel → consumes the SP-sharded stream with NO change (DP-prompts
  vs SP-seq-shards are transparent).
- ✅ **Full input path** (`prepare_inputs_prefill` SP-shard + per-row RoPE re-shard → logits): PCC 0.977,
  **last-token argmax matches** the ref. (`tt/model.py`, `test_model_sp_vs_ref.py` {dense, moe, tokens}.)
- ✅ **Repo cleanup** — `tt/` slimmed to mirror `deepseek_v3_d_p` (chunked-prefill main pass only). MoE
  collapsed to a single EP backend; deleted the EP=1 `experts/`, non-EP throughput, and decode paths
  (~38→26 files); tests slimmed to the main pass (42, all green at real shapes).
- ⚠️ **BLOCKER — real-weights 5k SP run:** intermittent `SIGBUS "non-existent physical address"` in
  `SiliconTlbWindow::write32` → `write_shard_to_device` (`PinnedMemory`) during **expert-weight upload**
  (model build, before any forward). Crash layer varies run-to-run (0/5/16+); bare-device + small-CCL
  smokes are 100% reliable; host RAM fine (489 GB free); not corrupted weights (a no-model repro
  `tests/ep_upload_repro.py` with random weights reproduces it); not ep_seq (cache is ep_seq-independent
  — load path, not convert). Looks like an **intermittent large bf4 sharded-write issue over 32 chips** —
  escalating to infra/UMD with the minimal repro; NOT M3 logic. (Reduced-config model tests are
  unaffected, so functional work proceeds in parallel.)

### 0.4 ⓘ HF oracle — it EXISTS upstream (correction)
Earlier drafts said "no HF modeling code." **Correction:** native `transformers` **main** has
`minimax_m3_vl` (full modeling; `MiniMaxM3VLRMSNorm`/attention/MLP), and **vLLM** has
`models/minimax_m3`. It's only absent from (a) the checkpoint's `trust_remote_code` files and (b) our
pinned `transformers` (rebuilt env has 4.53.0). Decision: keep self-authored torch refs for leaf ops;
bring a **branch-local** transformers (git main) at full-model assembly to PCC against the real impl
(the fused-kernel-free path). We confirmed M3's gemma-norm / per-head-qk-norm-before-RoPE / clamped
swigluoai against this source.

### 0.3 ⚠️ GAPS — what is NOT done (read before assuming "it works")
- ✅ **DONE since this list was written (see §0.2d):** ModelArgs real-weights plumbing (Step A);
  869 GB download; **real-weights → first-token vs the HF `minimax_m3_vl` golden (8/8)**. (The oracle
  ran via CPU offload of the real checkpoint, not a branch-local transformers — simpler and direct.)
- **Per-module PCC track (§0.2b/§0.2c) is vs SELF-AUTHORED refs + random weights** at reduced configs
  (2 layers, small experts/vocab). The full 60L×128E runs only on real weights (the golden path).
- ✅ **DONE 2026-06-25 (see §0.2e) — supersedes the stale "NOT integrated" bullets below:** SP=8×TP=4×EP=32
  WIRED into the whole-model prefill + validated at real shapes (dense/MoE/full-token-path); MSA runs
  under SP (single-chunk, indexer `chunk_offset`); repo cleaned to mirror `deepseek_v3_d_p`.
- **Full-GQA placeholder** is exact for prompts ≤ ~2K tokens; >2K uses MSA — now available under SP.
- **MSA sparse attention — INTEGRATED (single-chunk).** `sparse_sdpa_msa` + indexer/topk wired into
  `tt/attention/msa.py`; validated chunked at SP=8×TP=4 (`test_msa_sp_chunked_vs_ref`). Remaining: the
  multi-chunk cache lifecycle (§5.1) and the `chunk_offset`→mesh-coords swap (§0.2e).
- **KV cache / SP — SINGLE-CHUNK INTEGRATED; MULTI-CHUNK is the next gap.** SP=8 + the SP attention forwards
  are wired into `prefill.py` and validated whole-model (§0.2e). What's NOT done: the **multi-chunk
  chunked-KV cache lifecycle** (per-layer SP-sharded cache, chunk loop, cache-read) — design decided in
  §5.1, not yet built. Chunked KV is DeepSeek-style (§5), NOT vLLM paged.
- ⚠️ **Real-weights full-60L SP run BLOCKED** on an intermittent expert-upload `SIGBUS` (§0.2e) — infra/UMD
  escalation with `tests/ep_upload_repro.py`. Reduced-config model tests are green, so functional work
  (multi-chunk) proceeds in parallel.
- **Perf (tok/s) — not yet measured.** Pairs with the chunk loop (mirror DS's `[prefill timing]`).
- **Fused MoE expert kernel** for clamped swigluoai — perf task (copy+modify; activation ≠ M2/DS).
  Functional EP uses the composite per-expert loop (`CompositeRoutedExpert`), correct but slower.
- **Runner / pipeline / scheduler / KV migration** — still scaffold (`runners/prefill_runner.py`); the
  multi-chunk loop (§5.1) is what fills it in.
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
   │   INDEXER ("lightning indexer"): ONE index q-head PER GQA GROUP (4 groups) + one
   │                       SHARED index k-head. iQ [S,4,128] (index_q_proj→512), iK [S,1,128]
   │                       (index_k_proj→128) + own qk-norms.
   │     STEP 1  per-(group,key) score  s[g,s,t] = relu(iQ_g·iKᵀ)·w   PER GROUP g — NO sum
   │                       across groups (this is the M3/MSA delta vs DeepSeek-DSA's Σ_h single
   │                       list). (indexer_score op — §4)
   │     STEP 1b block-pool      block_score[g,s,b] = max over 128 keys in b   ← M3-specific glue
   │     STEP 2  top-k blocks    keep top-16 blocks PER GROUP (+ block 0 + local)  (topk op — §4)
   │     STEP 2b expand          chosen blocks → per-group key-index list [4,S,topk] into the KV buffer
   │
   ▼   STEP 3  SPARSE SDPA       softmax(Q·Kᵀ/√d)·V over ONLY each group's gathered keys (GQA)
   │           ~16×128 = 2048 keys/query/group regardless of S  → O(S·2048), not O(S²)
   │           (sparse_sdpa op — §4; GQA variant: 4 KV heads, per-group index list)
   ▼
   concat heads ─► Wo ─► reduce-scatter ─► out [S, 6144]
```
DSA selects per-**key** (top-N keys); MSA selects per-**block** (top-16 of 128-token blocks,
score = max over the block) → STEP 1b/2b (block-pool + expand) is the M3-specific glue around
the reusable ops. **Selection is PER GQA GROUP, not one shared list** — VERIFIED (2026-06-18):
HF config `sparse_num_index_heads=4 == num_key_value_heads=4` (one index head/group); MSA paper
(arXiv:2606.13392) "independently selects a Top-k subset **for each GQA group**"; MSA reference
(`MiniMax-AI/MSA` `api.py`) `sparse_topk_select → [total_qo_len, num_qo_heads, topk]` and
`kv_block_indexes [.., num_kv_heads|num_qo_heads, ..]` (per-head axis). So the earlier `Σ_h`
single-list formula (DeepSeek-DSA convention) was WRONG for M3; index list is **[4, S, topk]**.
Index branch order = `index_q/k_proj → per-head norm (index_q/k_norm) → RoPE`, then `indexer_score_msa`
(takes pre-roped q/k). Confidence: **norm HIGH** (index_q_norm/index_k_norm ship in the checkpoint, so
they're applied); **rope MEDIUM** — from a WebFetch *summary* of transformers-main `MiniMaxM3VLIndexer`
(`apply_rotary_pos_emb(idx_q, idx_k, cos[..,:head_dim], sin[..,:head_dim])`), matches DeepSeek-DSA but
NOT verified against the actual source. **TODO(REVISIT)** before trusting end-to-end output: confirm
rope is on both index_q AND index_k, the rotary width (partial-64 like main vs full-128), and ordering.
Also still to confirm: the `w` gating term. (Mirrored as a TODO in tt/attention/msa.py index_branch_forward.)

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

**M3 GQA `sparse_sdpa` op tensors** (per chip; deltas vs MLA = GQA grouping + plain head_dim 128,
no latent split, **per-GQA-group index lists**): `Q [1,64,S,128]` bf16 · `K [1,4,T,128]` + `V
[1,4,T,128]` bf16 (disjoint, NOT V⊂K) · `Indices [1,4,S,TOPK]` uint32 (**one list per GQA group**;
`0xFFFFFFFF`=masked tail) · `Out [1,64,S,128]` bf16 · `TOPK=16×128=2048`, `scale=128**-0.5`,
S/T parametric. **Torch golden + selfcheck: `reference/sparse_gqa_prefill.py`** (verified gather ==
dense-mask on real head geometry). Open for the kernel owner: K/V as two tensors vs packed
`[1,4,T,256]`; index head-axis per-group `[1,4,..]` vs per-q-head `[1,64,..]`.

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

### 5.1 Multi-chunk under SP — design decided (2026-06-25 discussion)
Single-chunk SP is validated (§0.2e, `cached_len=0`). Multi-chunk (process a long prompt chunk-by-chunk,
each attending the accumulated cache) is the next milestone — mirror DeepSeek's runner
(`runners/prefill_runner.py`): `prefill()` = ONE chunk, the loop advances `kv_actual = c*chunk_size`;
chunk-aligned offsets degenerate the block-cyclic layout to a per-chip reshape (`is_balanced=False`);
first chunk uses the no-cache path (`ring_joint` needs `Q.seq < K.seq`).

The two layer types need DIFFERENT cache *access*, but the **persistent cache is SP-sharded for both**:
- **Dense — native.** `ring_joint` reads the SP-sharded chunked-KV cache AND gathers across SP
  *internally* (online softmax over the ring) — never materializes the full context. No blocker;
  `dense_sp_attention` (cache-read) validated (`test_ring_joint_cache_read_sp_vs_ref`, PCC 0.99994).
- **MSA — sharded cache + transient AllGather.** `sparse_sdpa_msa(q,k,v,indices,…,cache_batch_idx)` is a
  **pure full-context kernel**: the `cache_batch_idx` selects a per-slot cache buffer but it is NOT an
  online/streaming read — it needs the full-length K/V resident to gather the top-k blocks. So the
  wrapper writes each chunk's K/V **and index_k** into the SP-sharded cache, then **AllGathers to full
  per device only during the forward** (freed after), runs `indexer(chunk_start_idx=cached_len)` +
  `sparse_sdpa_msa`. MSA must cache **index_k** too (extra vs DS's MLA).
- **DECISION (capacity-first):** persistent cache stays **SP-sharded** (1/SP per device) so per-user
  footprint is small → more users cached → good KV-cache hit rate. The MSA AllGather is **transient**
  (forward-only, deallocated after) so it costs prefill comm but **not** persistent capacity. *Rejected*
  a replicated full-context MSA cache: it stores the context ×SP per user (every device holds the whole
  thing) and kills serving capacity — even though it's only 1 KV head/device at TP=4 and would avoid the
  per-chunk re-gather.
- **Open / future optimization (PROFILE FIRST):** the MSA AllGather re-gathers the growing context each
  chunk → O(ISL²/chunk) prefill comm; *might* be a bottleneck at long ISL, might not. The endgame, only
  if profiling shows it hurts: a **sparse block-gather across SP** (add fabric into `sparse_sdpa_msa` to
  fetch ONLY the top-k selected blocks from the sharded cache) → per-chunk comm O(topk×block_size), flat
  in ISL, keeping the sharded persistent cache. Needs kernel/CCL work + the sparse-SDPA/indexer authors.
- **Validate:** 2-chunk prefill == single-shot forward of the full sequence (chunked==contiguous golden)
  at reduced config (E=32; unaffected by the §0.2e SIGBUS). Then perf (tok/s) off the chunk loop.

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

### 6.6 SP bring-up — implementation plan (grounded in `deepseek_v3_d_p` reuse)
> Investigated 2026-06-23. The strategy above (§6.1–6.5) is the *direction*; this is the concrete
> code path, staged so dense-layer SP lands first (the perf target) with no MSA/EP dependency.

**Current state.** SP is *plumbed but not consumed*: `config.py` has `ModeConfig.sp`, `MeshConfig`
auto-defaults prefill to `sp=rows, ep=1` and sets `sp_axis = ep_axis` (rows) — but no layer in `tt/`
actually shards the sequence (grep: `sp` appears only in comments). Attention prefill
(`tt/attention/prefill.py`) runs full non-cached SDPA on the call's own Q/K/V; the sequence is whole
per chip (replicated under TP, one-prompt-per-row under the DP=8 stopgap).

**Key insight — SP and EP alternate (they share the rows axis).** `sp_axis == ep_axis`, so they
cannot both shard the rows of the *same* tensor. Resolution (DeepSeek's): **attention + norms +
dense-MLP run in SP** (tokens sharded on rows); **inside MoE, dispatch/combine all-to-all re-lays
SP-sharded tokens onto EP-sharded experts**, then combine returns to SP layout. Consequence: **dense
layers (0–2) get SP with zero MoE/EP entanglement** — the clean first step.

#### Stage SP-1 — dense layers (self-contained; the perf target)
1. **Input + embedding → seq-shard on rows.** `ShardTensor2dMesh(dims=(sp_axis=0, None))`; embedding
   replicated on rows, TP-sharded on cols. **Drop-in reuse:** `deepseek_v3_d_p/tt/tt_parallel_embedding.py`
   + `tt/runners/runner_utils.py:~246` (`prepare_prefill_input_tensor`).
2. **Dense MLP + RMSNorm → no change.** Token-wise ops; each chip processes its `S/8` tokens. The TP
   collectives (gate/up col-parallel, down all-reduce) are on **cols**, orthogonal to SP on rows. Works
   as-is once the input is seq-sharded.
3. **Attention → the only real work.** Q/K/V become seq-sharded; each query needs all keys.
   **Mechanism: `ttnn.transformer.ring_joint_scaled_dot_product_attention`** (investigated 2026-06-23) —
   a fused CCL+SDPA op that **overlaps the ring all-gather of K/V with the SDPA matmul** (semaphore
   handshake, CCL workers on a non-overlapping sub-device column; `ring_fusion.cpp`). It is **GQA-native**
   (`nkh<nqh`), **causal** (`is_causal=True` — the op handles the per-shard causal offset internally, so
   NO hand-built offset mask), and uses **online-softmax** across the ring. It takes M3's shapes directly:
   per chip (after TP) `Q[1,16,S/8,128]`, `K/V[1,1,S/8,128]`, `joint_*=None`, `cluster_axis=sp_axis(rows)`,
   persistent K/V buffers `[1,1,S,128]`, ring semaphores. **The op is already proven for GQA+causal+SP**
   by `tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py` (separate `nhq/nhk/nhv`, `is_causal`,
   `is_balanced`) and used in production at `deepseek_v3_d_p/tt/mla/mla.py:857` — so our work is M3
   integration + CCL plumbing, NOT re-proving the kernel. (Supersédes the earlier all-gather-KV idea: same
   KV memory, but comm overlapped with compute and causality handled by the op.) `ring_distributed_sdpa`
   is a simpler-API fallback (no explicit semaphores) if the ring_joint plumbing misbehaves.

#### Stage SP-2 — MSA layers (later; kernel-gated)
SP-shard the **chunked-KV buffer across rows** (§5 already states this); `sparse_sdpa` (pavle's GQA
variant) gathers selected blocks **cross-row** by index. Plus the **MoE SP↔EP dispatch bridge**.
Gated on: the `sparse_sdpa` GQA kernel (§4) + SP-aware dispatch/combine. This is also where §6.4's
**TP→SP CCL bracket** lives (the head-sum indexer needs all heads local).

#### Memory note — full-S buffer vs chunked
`ring_joint`'s persistent K/V buffer is full-S (`[1,1,S,128]` per chip), so it overlaps comm but does
NOT shrink KV memory by itself — fine for the dense-layer perf measurement at moderate-to-large S. For
**true 1M**, layer **chunked prefill** on top (chunk Q/S, drive `logical_n`/`kv_actual_isl` — deepseek's
machinery) so each chip holds only a chunk at a time. (`ring_mla` is the MLA/latent-KV cousin — NOT a fit
for M3's plain GQA; `exp_ring_joint` rejects GQA. `ring_joint` is the one.)

#### Reuse scorecard
| Need | Reuse | Effort |
|---|---|---|
| Seq-shard input | `ShardTensor2dMesh(dims=(0,None))` + parallel embedding (`deepseek_v3_d_p`) | drop-in |
| Dense MLP/norm in SP | nothing — already token-wise | none |
| Ring attn (GQA+causal+SP) | `ttnn.transformer.ring_joint_scaled_dot_product_attention` (proven: `test_ring_joint_sdpa.py`; called at `mla.py:857`) | small — call + CCL plumbing |
| Ring CCL semaphores | `CCLManager.ring_attention_ccl_semaphore_handles` (added 2026-06-23, mirrors `tt_ccl`) | done |
| 1M memory | chunked prefill on top of ring_joint (`logical_n`/`kv_actual_isl`) | medium, later |
| MSA in SP | chunked-KV SP-shard + `sparse_sdpa` cross-row gather | kernel-gated |

**Next concrete step:** (a) M3 `ring_joint` SP helper + standalone PCC test (M3 dims, our CCLManager,
`(8,4)`) vs torch GQA-causal golden; (b) seq-shard inputs + wire the helper into `attention/prefill.py`
behind `mesh_config.sp>1`; (c) `test_dense_sp_*` perf harness on layers 0–2 at growing S, measure vs TP-only.

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
- ✅ **Phase 1 — Dense layers (0–2). DONE 2026-06-19** (TP=1, random weights, vs self-authored ref;
  §0.2b). Validated per-head QK-norm, Gemma norm, clamped SwiGLU, dense MLP, full GQA attention
  block, dense decoder layer + hybrid schedule, new dims. PCC 0.97–0.99.
- ✅ **Phase 2 — MoE block (layers 3+), full-attention placeholder. DONE 2026-06-19** (§0.2b):
  hybrid dispatch + shared expert + 128/top-4 + routed_scaling + clamped-swigluoai routed experts,
  on the non-fused FFN (EP-reusable). MoE decoder layer PCC-validated. (EP=32 multi-card, real
  weights, and the fused kernel remain.)
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
- **Phase 4 — Full integration + Galaxy bring-up.** ✅ *Functional multi-card DONE 2026-06-21*
  (§0.2c): full model on (8,4) = TP=4 + EP=32 + DP=8, PCC vs self-authored ref. **Remaining:**
  `ModelArgs` config plumbing (step A) → 869 GB download → real-weights first-token vs the
  `minimax_m3_vl` oracle (same playbook that worked for M2.7) → full 60-layer depth.
- **Phase 5 — Multimodal. SKIPPED for now.**

```
TP/SP/EP decision ─► Phase 0 ─► Phase 1 (dense) ─► Phase 2 (MoE) ─► Phase 4 ─► Galaxy
                                     │
                     Phase 3 (MSA) ──┘  parallel; gated on #47223 + GQA sparse_sdpa
```

---

## 9. Open decisions (close with the team)
1. ~~TP=4 × SP × EP layout on 4×8?~~ **RESOLVED 2026-06-25 — Config A: TP=4 / SP=8 / EP=32, DP
   eliminated** (§0.2e), validated whole-model at real shapes. Multi-user via galaxy replication.
   (Still worth the batch=1-vs-4 MFU experiment in §6b once perf numbers land.)
2. ~~Is a GQA `sparse_sdpa` planned?~~ **CLOSED** — GQA is being added; shapes (§3.1) + torch golden
   (`reference/sparse_gqa_prefill.py`) delivered. Remaining for the kernel owner: K/V packing (two
   tensors vs packed `[1,4,T,256]`) and the index head-axis the op consumes (per-group `[1,4,..]` vs
   per-q-head `[1,64,..]`).
3. ~~Does `indexer_score`'s `chunk_start_idx` causality compose with our block-pool?~~ **RESOLVED
   2026-06-25** — yes; per-device `chunk_offset` extends it for SP-sharded queries, pooling/sink+local
   stay in the wrapper (`tt/attention/msa.py`), MSA chunked validated at SP=8×TP=4 (§0.2e). To migrate to
   the DeepSeek mesh-coordinate variant once upstream `indexer_msa` carries per-device coords.
4. ~~shared vs per-group selection?~~ **RESOLVED — per GQA group** (4 lists; §3.2, verified vs config +
   paper + MSA reference). Still open: the `w` gating term in STEP 1, and whether RoPE applies to the
   index heads — confirm vs modeling code.
5. Chunked-KV chunk size + SP shard order for 1M context — **design decided §5.1** (SP shard order
   contiguous / `is_balanced=False`; persistent cache SP-sharded; dense=ring_joint, MSA=sharded+transient-AG).
   Chunk size itself still TBD by profiling (§5.1).

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
