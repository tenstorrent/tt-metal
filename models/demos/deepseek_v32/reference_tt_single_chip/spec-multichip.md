# Implement multichip compute for MLA layer with DSA from DeepSeek v3.2

## 1. Goal

Add **multichip** support for the DeepSeek-V3.2 attention stack — **MLA** (Multi-Head Latent Attention) and the
nested **Indexer** (the "lightning indexer" of DeepSeek Sparse Attention, *DSA*). It should run on **multiple Tenstorrent devices**.
The TT layer must be **functionally equivalent** to the CPU reference: same forward math, same prefill/decode caching contract, validated numerically via PCC.
This is a **functional** port. Performance (speed) is explicitly out of scope.
Implementation should follow existing DeepSeek-v3 TT reference.

Starting point: the **single-device** v3.2 port is complete and validated (`reference_tt_single_chip/{utils,mla,indexer,test_model}.py`,
spec `reference_tt_single_chip/spec.md`, all 12 TT + 10 reference tests pass). Multichip builds on top of it — the single-device path
must keep working (it is the `world_size == 1` regression guard).

## 2. References

- **CPU reference (source of truth)** — `models/demos/deepseek_v32/reference_cpu/`
  - `model.py` — `ModelArgs`, `IndexerCPU`, `MLACPU`, building blocks (`Linear`, `LayerNorm`, `RMSNorm`).
  - `weights.py` — `initialize_weights` (single entry point: random or pretrained layer load + HF fp8 dequant).
  - `utils.py` — CPU kernel equivalents (`act_quant_cpu`, `fp8_index_cpu`, `rotate_activation_cpu`) + RoPE (`precompute_freqs_cis`, `apply_rotary_emb`).
  - `test_model.py` — equivalence / determinism / pretrained harness (the contract our TT tests mirror).
  - `MLA_LAYER.md`, `README.md` — full architecture, shapes, dataflow diagrams, and the prefill/decode/cache invariants.
    **In particular `MLA_LAYER.md` §1/§2 Appendix C already document the intended parallelism** (🟩 replicated front-end,
    🟦 TP per-head, 🟥 `all_reduce` after `wo`). **Read these; this spec does not restate the math.**
- **Single-device v3.2 port (the thing we extend)** — `models/demos/deepseek_v32/reference_tt_single_chip/{spec.md,mla.py,indexer.py,utils.py}`.
- **TT MLA from DeepSeek v3 (multichip pattern donor, do not modify)** — `models/demos/deepseek_v3_d_p/tt/mla/`
  - The **only** in-tree example of a *multichip* MLA. Used as a reference for conventions: 2D mesh + `ShardTensor2dMesh`
    weight mappers (`mla.py::_convert_and_cache_weights`), `sp_axis`/`tp_axis`, CCL via `reference_tt_single_chip/tt_ccl.py::get_tt_ccl`
    (`reduce_scatter_minimal_async`, `all_gather_async`, `fast_reduce_nc`), ring attention
    (`ring_joint_scaled_dot_product_attention`), and SP-aware cache read-back (`kv_cache_to_host`).
  - Do not change this code. Explore how to reuse it. **But see §3 — v3.2 cannot adopt v3's attention path wholesale.**

## Critical

- Prioritize closing major decisions before implementation. I must review them **before the implementation**.
- Follow best engineering practices.
- Whenever possible, reuse existing solution. Do not reinvent the wheel.
- If ttnn op doesn't exist - fallback to CPU.
- Clearly document major decisions, learning points, workarounds at the end of this file.
- We are starting from QuietBox - 4 blackhole chips.

---

## 3. The central architectural decision: v3 ring-SDPA vs v3.2 manual attention **[RESOLVED — pending your review]**

This is the collision that has to be resolved before any other decision, because every downstream choice
(mesh shape, sharding, CCL, the fate of the host fallbacks) hangs off it.

### 3.1 What v3 does for multichip *(verified against `mla.py` source, not the docs)*

The v3 donor (`deepseek_v3_d_p/tt/mla/mla.py`) parallelizes MLA in **two dimensions at once**, and — important
correction to earlier drafts — its **activations are sharded too**: `forward` expects
`hidden_states [1, 1, S/sp_factor, hidden/tp_factor]` (mla.py:646-647). The residual stream lives hidden-sharded
on the TP axis and sequence-sharded on the SP axis; it is *not* replicated anywhere.

- **TP over the 128 heads + hidden dim** — `q_b_proj`/`wkv_b1`/`wkv_b2` are column-parallel by head (`mapper_tp1`),
  each device owns `H / tp_factor` heads. The **stems** (`q_a_proj`, `kv_a_proj_with_mqa`, `o_proj`) are
  **input-sharded** (`mapper_tp0` shards the *input* dim) because the activation arrives hidden-sharded: each
  device computes a **partial** stem output, combined with `reduce_scatter`+`all_gather` (q path, dim 3) or
  `all_gather`+`fast_reduce_nc` (kv path, dim 1). After `o_proj` only a `reduce_scatter` runs — output stays
  hidden-sharded for the next layer. So v3's TP collectives exist **because the residual is hidden-sharded**,
  not because head-sharding itself needs comm.
- **SP over the sequence/KV axis** — the KV-cache is sequence-sharded, attention is
  **`ttnn.transformer.ring_joint_scaled_dot_product_attention`** (or `ring_mla` for chunked prefill): fused ring-CCL
  SDPA streaming KV shards around the SP ring.

So v3's attention core is **one fused SDPA op** that internally owns causality, scaling, softmax, *and* the
cross-device sequence communication.

**v3 pieces that do NOT transfer to v3.2** (the "what's not suitable" list):

| v3 piece | Why unusable for v3.2 |
|---|---|
| `ring_joint_scaled_dot_product_attention` / `ring_mla` | no `attn_mask` hook → can't inject DSA index mask (§3.2) |
| Hidden-sharded residual + stem RS/AG collectives | only needed if activations are sharded; v3.2 replicates them (§3.4) |
| Sequence-sharded KV cache + `kv_cache_to_host` composer, `update_padded_kv_cache`, zigzag padding | SP-only machinery; v3.2 keeps host-side replicated cache |
| `wkv_b1`/`wkv_b2` decode-absorb weights | decode dropped (§3.5) |
| Indexed/rotated RoPE (`rotary_embedding_indexed`) | chunked-prefill-only; v3.2 uses single-shot `rotary_embedding_llama` |
| Tuned matmul/SDPA program configs (`mla_config.py`) | perf machinery, out of scope |
| No Indexer at all in v3 | DSA is new — no donor pattern exists for it |

What **does** transfer: `ShardTensor2dMesh` weight mappers (per-head column shard = v3's `mapper_tp1`), the
`sp_axis`/`tp_axis` 2D-mesh convention, CCL semaphore management (see D4), `num_links=2` on Blackhole, and the
weight-cache (`as_tensor(cache_file_name=…)`) discipline already adopted single-device.

### 3.2 What v3.2 single-device does — and why it can't use that op

The v3.2 single-device port (`reference_tt_single_chip/mla.py::_prefill_attn` / `_decode_attn`) computes attention **manually**:
`matmul(q, kᵀ) → ×scale → +causal_mask → +index_mask → softmax → matmul(·, v)`, mapping 1:1 onto the reference
einsums; the index mask is a plain `ttnn.add` between the two matmuls.

The one thing that blocks adopting v3's attention path is **DSA's additive index mask** (checked against the op source,
not assumed). The Indexer produces a `{0, -inf}` mask `[B|1, 1, S, T]` over the **key axis**, added to the scores
*before softmax* on top of the causal mask. v3's SP attention op,
`ring_joint_scaled_dot_product_attention`, exposes **no `attn_mask` parameter** — masking is causal-only
(`is_causal` / `is_cross` / `is_chunked`; `ring_joint_sdpa_device_operation.cpp`). There is no hook to inject an
arbitrary `{0,-inf}` per-(query,key) mask, so v3.2's manual core
(`matmul(q,kᵀ) → ×scale → +causal_mask → +index_mask → softmax → matmul(·,v)`) cannot be replaced by it.

### 3.3 The collision

The conflict is narrow but decisive — **DSA's index mask vs the ring SDPA op**:

| | v3 multichip (`ring_joint` SDPA) | v3.2 functional requirement |
|---|---|---|
| Additive `{0,-inf}` index mask (DSA) | **no `attn_mask` hook** (causal-only) | **mandatory** (the reason DSA exists) |
| Sequence comm | inside the SDPA ring (SP) | none — sequence is local |

You **cannot** inject DSA's index mask into v3's `ring_joint_scaled_dot_product_attention` — that op has no additive-mask
parameter. Adopting v3's SP path therefore means either giving up the index mask (defeats the port's purpose) or
re-implementing ring attention by hand with an injected, SP-sharded index mask — a large, kernel-adjacent effort that is
squarely *performance* territory.

### 3.4 Resolution

**Parallelize v3.2 over heads only (TP), keep the sequence (and KV-cache) replicated, and keep the existing
manual attention untouched.** SP / ring attention is a **documented follow-up**, not built here.

Rationale:
- This is a **functional** port; perf (and therefore the long-sequence motivation for SP) is explicitly out of scope.
- TP over heads is the parallelism `MLA_LAYER.md` already documents as the primary axis (🟦 per-head, 🟥 `all_reduce`
  after `wo`). 128 heads / 4 chips = 32 heads/chip — clean, no padding.
- With the sequence replicated, the attention core stays **device-local**: the existing `_prefill_attn`
  (manual matmul + `+index_mask` + softmax) runs unchanged on each device's head shard. **No ring attention, no SDPA,
  the index mask keeps working as a plain `ttnn.add`.** This is the single biggest risk-remover.
- It removes the hardest distributed problems entirely (sharded KV-cache read-back, distributed top-k over a
  sequence-sharded key axis, ring attention with an injected mask) — all of which only arise under SP.

Consequence (with **replicated activations**, D1b): the stems (`wq_a`, `q_norm`, `wkv_a`, `kv_norm`) run replicated
with **no comm at all** — exactly the GPU reference layout (`MLA_LAYER.md` C.1). `wq_b`/`wkv_b` are column-parallel
(no comm; output is local heads). The **single data-path collective in the whole layer** is the all-reduce after
the row-parallel `wo` (RS+AG pair along the TP axis). With the indexer replicated (D2) there is no other CCL.
That is the whole multichip surface — strictly smaller than v3's (which also pays stem RS/AG because its residual
stream is hidden-sharded, §3.1).

> **Decision to confirm (D1 below):** approve TP-only over heads with replicated sequence/cache, SP deferred. If long-sequence
> SP is a hard requirement for this milestone, say so — it reopens §3.3 and roughly triples the scope.

### 3.5 Scope: prefill only — **drop the decode path** **[RESOLVED]**

The multichip port implements **prefill only**. The decode path (MQA with `wkv_b` absorption against the cache,
`reference_tt_single_chip/mla.py::_decode_attn`) is **removed** from the multichip scope.

Rationale:
- The v3 multichip donor (`deepseek_v3_d_p/tt/mla/mla.py`) is **prefill-only** — a single causal `forward`
  (`ring_joint_scaled_dot_product_attention`, `is_causal=True`, `fill_cache_for_user_`); it has **no decode forward**.
  So "follow the v3 reference" gives a donor for prefill and *none* for decode under multichip.
- Decode is a functional duplicate of math already exercised by prefill + the cache-correctness invariant; adding it
  buys no new multichip coverage but pulls in the decode-only host fallbacks (F4 single-token cache indexing, the
  `wkv_b1`/`wkv_b2` absorption broadcasts W2/W3) and their sharding edge cases.

Consequences:
- `ttMLA.forward` runs the prefill (MHA) branch only; the `causal_mask is None` (decode) branch and `_decode_attn`
  are not part of the multichip path.
- The `wkv_b1`/`wkv_b2` (decode-absorb) weights drop out of the multichip weight set (§D3); only `wkv_b` (prefill
  K/V materialize) is needed.
- The single-device port keeps both paths — this cut applies to the **multichip** path only, and the prefill-vs-decode
  equivalence / determinism tests still run there (`tp_factor==1`).

---

## 3.6 End-state scaffold: fused indexer + topk + ring sparse attention

The ultimate goal is full v3-style SP×TP with three ops created (fused or new): **`indexer_score`**, device
**`topk`**, and **`ring_sparse_attention`** (ring SDPA that honors the DSA selection). The functional TP-only port
(§3.4) is **phase 0** of this scaffold, not the end state. If these ops exist, distribution converges to v3's
layout exactly:

### 3.6.1 End-state distribution (mesh `sp×tp`, residual `[1, 1, S/sp, dim/tp]` as in v3)

| Block | Distribution | Comm |
|---|---|---|
| Stems `wq_a`/`wkv_a` + norms | input-sharded on TP (= v3 `mapper_tp0`) | RS+AG / AG+reduce on TP axis (v3 mla.py:691-712, 769-782) |
| `wq_b`, `wkv_b` heads | TP per-head, `H/tp` local heads | none |
| MLA kvpe cache | **SP-sharded** seq, replicated on TP (v3 pattern) | written by `update_padded_kv_cache` / `fill_cache_for_user_` |
| Indexer `wq_b` | **replicated heads** on TP, queries SP-local → `[B, 64, S/sp, 128]` | none |
| Indexer `wk`/`k_norm`/`weights_proj` | input-sharded on TP (x is hidden-sharded) | one small AG+reduce on TP (output ≤128 wide) |
| Indexer K cache | **SP-sharded** `[slots·layers, 1, T/sp, 128]` bf8 — same slot-major scheme as v3 kvpe cache | same write path as kvpe |
| `indexer_score` | per device: local queries × **local keys** → `[B, S/sp, T/sp]` | partial — full-T rows need SP comm |
| `topk(2048)` | needs full-T score rows | AG `[B, S/sp, T]` (simple) or local-topk + global merge (scales) |
| `topk_indices` | SP-local `[B, S/sp, 2048]` (global key ids), **replicated on TP** (all head shards mask identical keys) | TP broadcast-free (computed identically per shard) |
| `ring_sparse_attention` | TP heads × SP ring over latent kvpe, **selection-aware** | inside the op |
| `o_proj` | row-parallel + RS, residual stays hidden-sharded | RS on TP |

Indexer cache: **yes, a replica of the v3 kvpe pattern** — same SP-sharded slot-major allocation, same tile-aligned
chunk writes, same `kv_actual_isl` offset math, head_dim 128 instead of 576, single shared head (no fp8 scale; bf8_b
block exponent stands in functionally).

Indexer compute: **replicate heads on TP, distribute keys on SP.** Reference replicates at TP=8 (A.8) and the
64-head score is cheap; the axis that must scale (long-context T) is sequence — distributing keys gives free
score sharding; what it costs is full-T row assembly before top-k (AG or topk-merge), which becomes the single
indexer-specific collective.

### 3.6.2 Missing pieces (exists / partial / new)

| Piece | Status |
|---|---|
| `ttnn.topk` | **exists** (last-dim, bf16); k=2048 + (B·S/sp) rows untested — validate, else chunked merge |
| `indexer_score` fused (matmul+relu+w·Σheads) | **new** (composable today: matmul/relu/mul/sum — P0 already does this) |
| `ring_sparse_attention` | **new** — siblings exist (`ring_mla` for ring+latent, `chunked sdpa` plumbing); needs per-query 2048-key gather/skip + causal-within-selection |
| Non-interleaved RoPE | **no op needed** — dot-product invariant under shared dim permutation: permute the rope half (pair `i` with `i+32` → adjacent) in `wq_b`/`wk` output cols **and** `k_norm` weight/bias offline, then the interleaved op computes the identical score; validate vs F1 |
| Indexer K-cache write | exists (kvpe write path, head_dim 128) |
| `{0,-inf}` mask materialization | only needed while attention is dense (P0/P1); drops when topk_indices feed the op |
| CCL | existing RS/AG suffice |

### 3.6.3 Phasing

- **P0 (this spec):** TP-only, host topk/mask, dense masked attention. Numerics truth.
- **P1:** still TP-only — device topk + RoPE permutation trick (retire F1–F3); `indexer_score` fusion when profitable.
- **P2:** SP — sharded caches, score AG, `ring_sparse_attention`. Each phase regression-checks against P0.

---

## 4. Open decisions — review before implementation

Status legend: **[proposed]** = I have a recommendation, need your sign-off · **[open]** = needs your input.

### D1 — Parallelism strategy & mesh shape **[proposed]**
- **Recommendation:** TP-only across the 128 heads, sequence + KV-cache replicated; manual attention kept (per §3.4).
  Mesh = **`1×4`** TP on the QuietBox (4 Blackhole chips). Make `tp_factor` derive from the mesh shape so
  `world_size == 1` (the existing single-device path) is just `tp_factor == 1` and remains a regression guard.
- **Open sub-question:** keep the layout 2D (`sp_axis`/`tp_axis` like v3, with `sp_factor == 1` for now) so SP can be
  added later without reshaping the weight mappers? **Recommended yes** — costs nothing now, keeps the SP follow-up cheap.
- **D1b — activation layout [proposed]:** v3 keeps the residual stream **hidden-sharded** (`[1,1,S/sp,dim/tp]`, §3.1),
  which is why it needs RS/AG around the stems. **Recommendation: replicate activations** (`[B,1,S,dim]` on every
  device, the GPU-reference layout). Stems then run replicated with no comm; the only collective is after `wo`.
  This also keeps the layer's external interface identical to the single-device port (drop-in for the tests), at the
  cost of `dim`-wide replicated input — fine, this layer is tested standalone. If/when integrated into a v3-style
  pipeline (hidden-sharded residual), add the v3 stem AG/RS at the boundary — documented follow-up alongside SP.
- **Alternative (only if SP is mandatory):** `2×2` SP×TP + ring attention — reopens §3.3, large scope.

### D2 — Indexer under TP: **replicate it, don't shard it** **[proposed]**
Earlier draft proposed TP-sharding the 64 index heads + an `all_reduce` of the per-head score sum before top-k.
**Withdrawn.** The GPU reference itself runs the Indexer **fully replicated** on every rank precisely because
`topk_indices` must be bit-identical across the head shards (`MLA_LAYER.md` A.8 — replication is the reference's
deliberate design, with a broadcast+assert guard). Replicating follows the reference *and* the donor philosophy:
- Zero indexer CCL; no partial-sum trap. Every device computes the full `[B,S,T]` score, runs the same host top-k,
  builds the same mask — identical selection by determinism of identical inputs (the assert in the reference checks
  exactly this, single-process determinism makes it trivially true).
- The host fallbacks F1/F2/F3 use `ttnn.to_torch(ConcatMeshToTensor(dim=0))`, which **only works for replicated
  tensors** — sharding the index heads would force shard-aware composers through every fallback (the real #1
  correctness trap, dodged entirely).
- The indexer is small (`wq_b` 1536→8192, `wk` 7168→128, `weights_proj` 7168→64); replicated compute is what the
  reference deployment accepts at TP=8 (A.7/A.8).
- Cost: the `{0,-inf}` mask upload is per-device replicated; mask broadcasts over the local head shard at attention
  time exactly as on single device.
- Note the TT indexer scores only the current chunk (no key cache) — fine, prefill-only (`S == T`).

**Decide:** approve replicated indexer for P0/P1. Replication on the TP axis stays the end-state answer too;
what changes in P2 is the **SP key axis** — the indexer K cache and score shard by sequence (§3.6.1), never by head.

### D3 — Weight-sharding table **[proposed]**
Make explicit (mirrors v3's `mapper_tp0`/`mapper_tp1`); `reference_tt_single_chip/utils.py::convert_mla_weights`/`convert_indexer_weights`
currently replicate everything and need mesh-mapper args:

| Weight | Layout | Axis |
|---|---|---|
| `wq_a`, `q_norm`, `wkv_a`, `kv_norm` | replicated | — |
| `wq_b` | column-parallel | shard `H` (output heads) |
| `wkv_b` (prefill K/V materialize) | column-parallel | shard `H` |
| ~~`wkv_b1`/`wkv_b2` (decode absorb)~~ | — | dropped — decode out of scope (§3.5) |
| `wo` | row-parallel | shard input `H*v` (= shard by head); **RS+AG (all-reduce) on output** |
| Indexer (all weights: `wq_b`, `wk`, `k_norm`, `weights_proj`) | **replicated** (per D2) | — |

- Mapper plumbing: extend `reference_tt_single_chip/utils.py::convert_mla_weights` with mesh-mapper args (mirror v3 `_convert_and_cache_weights`
  `mapper_tp1` for `wq_b`/`wkv_b`; `wo` shards its **input** dim). `convert_indexer_weights` stays replicate-only.
- Cache files become mesh-shape-specific — include `tp_factor` in `cache_file_name` (or separate cache dirs per mesh)
  so 1-chip and 4-chip runs don't poison each other.
- Resolved: `index_n_heads = 64` (`ModelArgs`), divisible by 4 — but moot under D2 (replicated).

### D4 — CCL infrastructure: reuse vs. the "nothing outside `deepseek_v32/`" constraint **[open]**
`spec.md` decision #4 was a **hard constraint**: edit nothing outside `deepseek_v32/`. The needed CCL surface is now
tiny — under D1b+D2 it is exactly **one all-reduce (RS+AG along the TP axis) after `wo` per layer**. Options:
- **(a) Import `models/common/modules/tt_ccl.py` read-only.** The v3 file is itself a fork of this common module
  (its only delta is ring-attention core-grid carve-out, which we don't need without ring SDPA). Importing the
  *common* module gives the same semaphore management without depending on `deepseek_v3_d_p` at all. *Recommended.*
- **(b) Import `deepseek_v3_d_p/tt/tt_ccl.py`** — works, but ties us to a peer demo's fork for no benefit.
- **(c) Host-fallback all-reduce** (gather output shards to host, sum, re-upload) — zero CCL infra, consistent with
  the "no ttnn op → CPU fallback" rule; useful as a bring-up stepping stone and as the numerical truth to validate
  the device CCL against, but ~`dim`-sized transfer per layer.
- **Decide:** (a) as target, with (c) as bring-up scaffold? Both are read-only imports — same waiver needed:
  the "nothing outside `deepseek_v32/`" rule relaxed to *read-only imports*.

### D5 — Multichip validation strategy **[proposed]**
- CPU reference is `world_size == 1`, so PCC is still against a single-device truth. After the `wo` all-reduce the
  layer output is **replicated** — read one replica for PCC, and assert all replicas match (the cheap analogue of the
  reference's broadcast+assert).
- **Recommendation:** parametrize the existing 12 tests over `tp_factor ∈ {1, 4}`; `tp_factor==1` is the regression
  guard, `tp_factor==4` the new path. `skip`/`xfail` the 4-chip cases when fewer than 4 devices are visible.
  Add one explicit "tp_factor==4 output == tp_factor==1 output (within PCC)" assertion, plus the replica-equality check.
- **Decide:** target hardware for CI — real 4-chip QuietBox only, or also a smaller mesh (e.g. emulated) for the
  `tp_factor==2` middle case?

### D6 — Fate of the existing host fallbacks under TP **[proposed]**
Confirm each single-device fallback survives TP-only (most do, because sequence/cache stay replicated):
- **F1 non-interleaved RoPE (indexer):** indexer replicated (D2) → read-back/re-upload stays replicated → **unchanged**.
- **F2/F3 top-k + mask:** indexer replicated (D2) → full score on every device, host top-k identical → **unchanged**.
- **F4 host KV-cache:** the latent is computed replicated; cache write/read stays replicated → **unchanged**.
  (Becomes SP-aware only under the SP follow-up.)
- **General rule that makes the above hold:** every `ttnn.to_torch(ConcatMeshToTensor(dim=0))` host fallback assumes
  a replicated tensor. After this port the **only** sharded activations are q/k/v heads inside attention and the
  pre-`wo` partials — none of which a fallback touches. Assert replication where fallbacks read.
- **W1 head-merge transpose+reshape:** operates on `H_local = H/tp_factor` — read head count from the mesh, not the
  config. (At 32 local heads `nlp_concat_heads` may no longer overflow L1; optional cleanup, not required.)
- **W2–W3 decode `repeat`:** decode-only → **N/A** (decode dropped, §3.5).

---

## 5. What to implement (once D1–D6 are signed off)

Extend `ttMLA` / `ttIndexer` (not rewrite): add `tp_axis`/`sp_axis` (sp_factor=1), mesh-mapper-aware weight conversion
(D3: shard `wq_b`/`wkv_b`/`wo` by head, replicate the rest incl. the whole indexer), one RS+AG all-reduce after `wo`
(D4), and read `H_local` from the mesh everywhere head count is used. **Prefill (MHA) path only** (§3.5): `forward` runs the `causal_mask is not None`
branch; `_decode_attn` and the decode-absorb weights are not part of the multichip path. The prefill attention core,
caches, RoPE, and top-k logic otherwise stay as in the single-device port.
A full sharded-shape walkthrough + op/CCL mapping table will be filled in here after sign-off (mirroring `spec.md` §4/§6).

## 6. Tests

Per D5: the existing `reference_tt_single_chip/test_model.py` suite parametrized over `tp_factor`, plus the reassembly assertion. PCC
thresholds carry over from `spec.md` §7 (bf16 ≥ 0.99 op-path; determinism ≥ 0.9999). The **prefill-only** multichip path
(§3.5) runs the prefill-PCC, indexer-PCC, determinism, and pretrained tests at `tp_factor==4`; the prefill-vs-decode
equivalence test is `tp_factor==1`-only (decode is single-device, ≥ 0.999).

## 7. Out of scope

- **Decode (MQA-absorbed) path** — multichip is **prefill only** (§3.5); the v3 donor is prefill-only too. Decode stays
  in the single-device port (`tp_factor==1`).
- Performance / speed optimization, tuned program configs.
- **SP / ring attention and the new ops** (`indexer_score`, device topk, `ring_sparse_attention`) — these are the
  **end state**, scaffolded in §3.6 (P1/P2); P0 structures for them (2D mesh, sp_factor=1, slot-major caches) but
  builds none of them.
- Implementing ttnn ops that don't exist (fall back to CPU + document instead) — P0 only.
- True fp8 storage / Hadamard / sparse-skip attention (functional sim only; already dropped single-device).
- Editing `deepseek_v3_d_p/` or anything outside `deepseek_v32/` (CCL **reuse** is read-only import — pending D4).

## 8. Issues & learning points

*(Filled during bring-up, mirroring `spec.md` §11 — one row per multichip-specific fallback / workaround / deviation.)*
