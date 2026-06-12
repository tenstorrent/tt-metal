# MiniMax-M2 Prefill — Functional Bring-Up Proposal

> **Goal of this document:** Define the architecture and work breakdown for a
> *functional* (correctness-first, not optimized) MiniMax-M2 prefill pipeline
> on a single Blackhole Galaxy. This is a starting point for team discussion.
>
> **This is a LIVING doc** — it is kept in sync with the code as the source of
> truth for the team (and for handing context to an agent). Rule: it never claims
> something works that isn't validated; status is either *validated (with PCC +
> conditions)* or *scaffold (owner + blocker)*.

---

## 0. Status (living) — last updated 2026-06-12

**Model (validated, single Wormhole, TP=1/EP=1/SP=1, random weights, seq=128):**
attention block 0.9991 · router (decomposed) · experts 0.9990 · **full decoder layer
0.99993** vs HF. The M2 architecture deltas (partial RoPE, distributed QK-norm,
sigmoid+bias router, SiLU-SwiGLU experts, no biases/sinks/sliding-window) are done.

**NEW — validated on a real Blackhole Galaxy (32 chips), 2026-06-12 (see §14 for how-to-run):**
- **TP=8 collectives correct vs HF** — attention PCC **0.9991**, experts PCC **0.9989** at
  mesh `(1,8)`/TP=8 (match the TP=1 baselines). First time the reduce-scatter / all-gather /
  expert TP all-reduce paths have run multi-device.
- **Full 62-layer model assembles + runs** end-to-end at TP=8 (random weights smoke).
- **Real M2.7 weights → first token VERIFIED vs HF** — prompt "The capital of France is" →
  first token `'The'`, **argmax == HF, top-5 identical**, full-vocab logit PCC **0.953**
  (the gap from 0.999 is bfp4 expert quantization over 62 layers; exact argmax+top-5 ⇒
  correct). Ground truth saved to `/data/vmelnykov/minimax_m2_ref/`.
- **The whole 230B model FITS on 8 chips** at TP=8/EP=1/bfp4 (~16 GB/chip).
- **EP=32 VERIFIED END-TO-END** — `(4,8)` DP-attention (4 prompts, one/row) + EP=32 shared MoE
  (256 experts spread 8/chip) + real weights → prompt 0 first token `'The'` == HF oracle (MATCH).
  EP MoE block also validated standalone (PCC 0.9803). EP gives ~3.3 GB/chip experts (vs 14 at EP=1).
- **3 bugs fixed** in never-before-run paths: experts `Topology.Ring`→`ccl_manager.topology`
  (plain-MESH Galaxy needs FABRIC_1D + Linear, no torus); stale `SamplingGenerator(
  enable_internal_trace=)` kwarg; `load_state_dict` missing `trust_remote_code=True`.

**Done:** code + docs are MiniMax-only; program-config classes + identifiers all MiniMax-named.

**In progress — Phase 2:** EP (expert-parallel) MoE, SP=4 + chunked/paged attention, serving.
- *EP — SCOPED, reusable (was "Tier 3 undefined"):* the dispatch/combine machinery already
  exists as compiled ops `ttnn.experimental.deepseek_prefill.{dispatch, routed_expert_ffn,
  combine, ...}` and is generic. MiniMax MoE shape == DeepSeek V3 (256/top-8/sigmoid+bias)
  but **no expert groups** (plain top-8) and **no shared expert**. Target on `(4,8)`: TP=8 +
  experts spread 8/chip across 32 (= "EP=32") → ~3.3 GB/chip. Plan: wire into
  `tt/experts_throughput/`. See §15.
- *Tier 1:* **chunked/paged GQA attention rewire** (replaces dead write-only KV fill) + SP=4
  input sharding — needed for full `(4,8)` long-context prefill.
- *Tier 2 (cross-team):* per-layer KV migration, P/D disaggregation.

### ⚠️ GAPS — what is NOT done yet (read this before assuming "it works")
The token-correct runs above are **single-shot, short-seq, non-paged, prefill-only**. Real
serving needs the following, **none of which exist/are validated yet**:
- **Attention is the non-chunked path** (full-sequence SDPA on fresh Q/K/V). The **chunked/paged
  GQA attention** (`chunked_scaled_dot_product_attention` + `chunk_start_idx` + page-table slicing)
  is **NOT wired** → no long context, no chunked prefill.
- **No real KV cache in use.** All token runs use `kv_cache=None`; the paged KV layout exists but
  **paged-KV write→read-back is NOT validated** (the chunked SDPA above would exercise it).
- **No prefill runner / pipeline / scheduler.** `prefill_runner.py`, `MiniMaxPrefillPipeline.prefill`,
  `_prepare_input_tensor` are still `NotImplementedError`. No standalone loop, no SHM, no request loop.
- **No KV migration** (prefill→decode disaggregation) — Tier-2, cross-team; endpoint is a NoOp stub.
- **SP=4 (sequence parallel)** not done — needed for config (A) long-context single-prompt prefill.
- **Decode is out of scope** (runs in tt-blaze) → this repo produces the **first token only**, not text.
- **EP integration is TEST-grade:** the DP-row↔EP-dgs bridge in `mlp.py` is a **host round-trip**
  (~2 hops/layer) — correctness-only; **production needs an on-device bridge**.
- **bfp4 accuracy:** full-model logit PCC ~0.95 (argmax/top-5 correct). Fine for first-token;
  long generation accuracy under bfp4 unmeasured.

**Still not validated (one-liner):** chunked/paged attention, real paged-KV read-back, SP=4,
runner/pipeline/scheduler, KV migration, long context, decode, on-device EP bridge, bfp4 long-gen.

---

## 1. Model Architecture at a Glance

From `configs/MiniMax-M2/config.json`:

| Parameter | Value |
|---|---|
| `num_hidden_layers` | 62 |
| `hidden_size` | 3072 |
| `num_attention_heads` | 48 |
| `num_key_value_heads` | 8 |
| `head_dim` | 128 (partial rotary: 64) |
| `num_local_experts` | 256 |
| `num_experts_per_tok` | 8 (top-8 sigmoid routing) |
| `intermediate_size` (per expert) | 1536 |
| `mlp_intermediate_size` (shared) | 8192 |
| `max_position_embeddings` | 196 608 (~192K) |
| `vocab_size` | 200 064 |
| `use_qk_norm` | True (full-width RMSNorm on Q, K) |
| `scoring_func` | sigmoid + `e_score_correction_bias` |
| `sliding_window` | None (full causal attention) |

**Notable differences from a standard transformer:**
- Partial RoPE: only first 64 of 128 head dims are rotated.
- QK-norm: RMSNorm on Q and K before head split (unusual, matters for accuracy in decode).
- Sigmoid router with a correction bias applied for selection only (not for the returned weights).
- 256 experts, top-8 routing — **essentially identical MoE shape to DeepSeek V3** (also 256 experts / top-8, sigmoid + correction-bias routing). This is a strong reason the DeepSeek prefill framework fits.

---

## 2. Target System — One Blackhole Galaxy

A Galaxy is a 4×8 mesh of 32 Blackhole chips connected via Ethernet fabric.

```
  col:   0     1     2     3     4     5     6     7
       ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
row 0: │ BH  │ BH  │ BH  │ BH  │ BH  │ BH  │ BH  │ BH  │  ← SP group 0
       ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
row 1: │ BH  │ BH  │ BH  │ BH  │ BH  │ BH  │ BH  │ BH  │  ← SP group 1
       ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
row 2: │ BH  │ BH  │ BH  │ BH  │ BH  │ BH  │ BH  │ BH  │  ← SP group 2
       ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
row 3: │ BH  │ BH  │ BH  │ BH  │ BH  │ BH  │ BH  │ BH  │  ← SP group 3
       └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
         ◄──────────────── TP = 8 ──────────────────────►
         ◄── SP = 4 (rows) ───────────────────────────────►
```

**Parallelism for prefill (from `config.py`):**

| Axis | Mode | Value | What it means |
|---|---|---|---|
| TP | Both | 8 | Weight matrices column-sharded across 8 chips in a row |
| SP | Prefill | 4 | Sequence sharded across 4 rows; each row processes `seq_len / 4` tokens |
| EP | Decode | 4 | Expert parallel in decode; each row owns 64 of 256 experts |
| EP | Prefill | 1 | Expert parallel disabled in prefill (all rows see all experts) |

With SP=4 and 5 120-token chunks, each row processes **1 280 tokens per chunk**.

---

## 3. Where We Are Today

### What already works (in `models/demos/minimax_m2/tt/`)

> **Validation caveat:** everything below is **PCC-validated only at TP=1 / EP=1 /
> SP=1 on a single Wormhole card with random weights, seq_len=128** (attention block
> 0.9991, router decomposed, experts 0.9990, full decoder layer 0.99993 vs HF). The
> TP/EP/SP **all-reduce / all-gather paths and paged KV read-back are NOT yet
> exercised** — they need the multi-card Blackhole box. "Works" = code exists +
> single-card math validated, not multi-device validated.

```
✅ Attention prefill (MATH validated @ TP=1)
     QKV projection → distributed QK-norm → partial RoPE (dim 64)
     → NON-chunked causal SDPA on the fresh Q/K/V → output proj + reduce-scatter
  ⚠️  KV-cache fill is currently WRITE-ONLY / DEAD: paged_fill_cache (or fill_cache)
      runs, but SDPA reads the fresh tensors, not the cache. No chunked SDPA is wired.
      OWNER: attention (rewire — see §6/§9); this dead path is removed in that work.

✅ Experts prefill
     sigmoid router with e_score_correction_bias
     → sparse_matmul gate+up → SwiGLU → sparse_matmul down
     → EP/TP/SP allreduce chain

✅ KV cache
     Paged layout: [max_num_blocks, num_kv_heads, block_size, head_dim] bfp8
     TP-sharded, optional users_row_sharded

✅ Mode-aware MeshConfig
     Separate prefill (SP=4, EP=1, TP=8) and decode (EP=4, SP=1, TP=8) configs

✅ Full decoder layer + end-to-end PCC test vs HF (Step 5 done)
```

### Hard blockers (will crash at runtime)

```
❌ batch_size > 1 in experts  →  NotImplementedError  (experts/prefill.py)
                                  blocks DP=4 mode entirely

ℹ️ Decode is NOT in this repo  →  this is a PREFILL-only component; decode runs
                                  separately (tt-blaze, like all new models). The
                                  decode forward path + dispatch were removed. (A few
                                  decode Generator I/O helpers remain in model.py as a
                                  noted follow-up; harmless dead code.)

❌ No prefill runner / server connection
                              →  prefill_runner.py is scaffold (stubs);
                                  no standalone loop, SHM, or request loop yet

❌ No chunked prefill at generator level
                              →  max_prefill_chunk_size = 128K (the whole sequence)
                                  no interleaving with decode

❌ No KV migration
                              →  KV cache never leaves the prefill chips
```

### Known performance TODOs (not blockers for functional bringup)

```
⚠️  slice_write instead of concat    experts/prefill.py:197   realloc every chunk
⚠️  reduce_scatter at MLP output     layer.py:146             allreduce wastes BW
⚠️  No prefill TTNN trace            model_config.py:160      Python overhead per fwd
⚠️  No warmup seq_lens               model_config.py:174      Issue #32818
⚠️  disable_batched_prefill = True   model_config.py:117
```

---

## 4. Target Architecture — Full System

```
                         HOST CPU
   ┌──────────────────────────────────────────────────────────────┐
   │  C++ Inference Server                                        │
   │  - receives HTTP/gRPC from users                            │
   │  - manages request queue and KV slot pool                   │
   │  - allocates dst_slot for each new request                  │
   └────────────┬──────────────────────────────────┬─────────────┘
                │ SHM write                         │ SHM read
                │ {task_id, token_ids, dst_slot}    │ {task_id, first_token}
                ▼                                   ▲
   ┌─────────────────────────┐         ┌────────────────────────────┐
   │  prefill_runner.py      │         │  decode_runner.py          │
   │  PREFILL_SUBCTX_ID = 1  │         │  DECODE_SUBCTX_ID  = 0     │
   │                         │         │                            │
   │  Galaxy 4×8  SP=4 TP=8  │         │  Galaxy 4×8  EP=4 TP=8     │
   │  32 BH chips            │         │  32 BH chips               │
   └─────────────────────────┘         └────────────────────────────┘
                │  KV migration per-layer over fabric  ▲
                └──────────────────────────────────────┘
                   ttnn disaggregation API (migration team)
                   granularity: 32-token chunks (DRAM bank aligned)
```

> **Note:** The KV migration transport layer (the fabric send/recv, address
> tables, sub-context setup) is owned by the migration team. We integrate with
> it via a `BoundMigrationEndpoint` interface identical to DeepSeek V3 D/P.

---

## 5. Chunked Prefill Flow (chunk size = 5 120 tokens)

With SP=4, each row sees `5120 / 4 = 1280` tokens per chunk.
For a 128K prompt that is `128K / 5120 = 25` chunks.

```
Full 128K sequence:
[chunk 0: tok 0..5119][chunk 1: tok 5120..10239] ... [chunk 24: tok 122880..128K]

SCHEDULER VIEW (generator level):
  Step 1:  run prefill chunk 0  (5 120 tokens, 62 layers)
           → migrate KV layer 0..61 to decode chips
  Step 2:  run decode step(s) for active users      ← no decode stall
  Step 3:  run prefill chunk 1
  Step 4:  run decode step(s)
  ...
  Step 49: run prefill chunk 24
  Step 50: decode starts for new user (KV fully available)

Per chunk, each layer:
  ┌──────────────────────────────────────────────────────────────────────┐
  │  Transformer Layer k  (chunk c, tokens c*5120 .. (c+1)*5120 - 1)    │
  │                                                                      │
  │  hidden [SP=4, 1, 1280, 3072]   ← 1 280 tokens per row              │
  │         │                                                            │
  │    RMSNorm                                                           │
  │         │                                                            │
  │    Attention prefill                                                 │
  │      QKV proj → dist. QK-norm → partial RoPE                        │
  │      chunk_start_idx = c * 5120   ← tells SDPA where we are         │
  │      chunk_page_table = page_table[:, c*80 : (c+1)*80]              │
  │                          (80 blocks of 64 = 5120 tokens)            │
  │      SDPA: Q[1280, D] × K_past[c*5120 + 1280, D]ᵀ                  │
  │            worst case: [1280, 128K] — fits, no [N,N] problem        │
  │      paged_fill_cache → writes this chunk's K,V                     │
  │         │                                                            │
  │    MoE experts                                                       │
  │      sigmoid router → top-8 → sparse matmul                        │
  │      SP allreduce → residual add                                     │
  │         │                                                            │
  │    output hidden [SP=4, 1, 1280, 3072]                              │
  └────────────────────────────┬─────────────────────────────────────────┘
                               │  after all 62 layers
                               ▼
                   on_layer_complete(layer_idx)
                     synchronize_device()
                     migrate_layer(layer_idx, pos_start=0, pos_end=end_pos,
                                   src_slot=slot_id, dst_slot=dst_slot)
                     wait(uuid)
```

---

## 6. KV Cache Layout

> **Verified status (2026-06-10):**
> - **Chunked SDPA exists and supports M2's GQA.** `ttnn.transformer.chunked_scaled_dot_product_attention`
>   validates `nqh >= nkv && nqh % nkv == 0` (48/8 ✅) and a GQA case (nh=8/nkv=1, head_dim 128)
>   is in the nightly test. So GQA is NOT a blocker.
> - **Chunked SDPA requires PAGED KV** (reads K/V from the cache via page_table; no non-paged variant).
> - **We do NOT have working paged/chunked attention yet.** Current prefill fills the cache but the
>   SDPA call is the *non-chunked* op on fresh Q/K/V — the fill is dead. The rewire (attention owner):
>   `paged_fill_cache(chunk) → chunked_scaled_dot_product_attention(q, k_cache, v_cache, page_table, chunk_start_idx)`.
> - **MLA vs GQA — important:** DeepSeek's attention is **MLA** (`chunked_flash_mla_prefill`, latent KV).
>   **M2 is standard GQA**, so M2 uses the standard `chunked_scaled_dot_product_attention` — reference
>   the **llama3_70b_galaxy** GQA path, NOT DeepSeek's MLA kernel. Follow DeepSeek for the serving/
>   migration/chunking *pattern* only; the attention *op* comes from the GQA chunked-SDPA users.


```
Prefill KV cache (lives on prefill Galaxy):

  k_cache: [max_num_blocks, num_kv_heads // TP, block_size, head_dim]
  v_cache: [same shape]

  With TP=8:  num_kv_heads // 8 = 8 // 8 = 1  → each chip holds 1 KV head
  block_size: TBD (DeepSeek uses 64; good starting point)
  dtype: bfloat8_b

  With SP=4 and users_row_sharded:
    Each row holds KV for seq_len / 4 token positions.
    Row 0: tokens 0..32K (for a 128K context)
    Row 1: tokens 32K..64K
    Row 2: tokens 64K..96K
    Row 3: tokens 96K..128K

  Migration granularity: 32 tokens = 1 DRAM bank chunk
    chunk_size_bytes = 32 × 1 (kv_head/chip) × head_dim × bfp8_bytes
                     ≈ 32 × 128 × ~1.06  (bfp8 overhead)
    (smaller than DeepSeek's 19 584 bytes because head_dim=128 vs 576)

Decode KV cache (lives on decode Galaxy):
  Same shape, pre-allocated.
  dst_slot selects which user's block range to write into.
  Migration fills it layer by layer as prefill progresses.
```

---

## 7. Per-Layer Migration — Detailed Flow

This follows the DeepSeek V3 D/P pattern exactly.
Reference: `models/demos/deepseek_v3_d_p/tt/tt_deepseek_prefill_pipeline.py`

```
pipeline.prefill(token_ids, slot_id, actual_isl, dst_slot):
│
├─ _prepare_input_tensor()
│    shard token_ids across SP=4 rows
│    [128K] → [4, 1, 32K] on device
│
├─ _build_migration_callback(slot_id, actual_isl, dst_slot)
│    returns on_layer_complete() closure
│
└─ model.forward(tt_tokens, kv_cache, on_layer_complete=callback)
      │
      for layer_idx in range(62):
      │
      │   ┌─── transformer layer ───────────────────────────────┐
      │   │  attention → fills kv_cache[layer_idx]              │
      │   │  moe experts                                        │
      │   └─────────────────────────────────────────────────────┘
      │
      └── on_layer_complete(layer_idx):          ← fires after each layer
              ttnn.synchronize_device()           wait all 32 chips done
              uuid = endpoint.migrate_layer(
                  layer_idx,
                  pos_start = 0,
                  pos_end   = ceil(actual_isl / 128) * 128,
                  src_slot  = slot_id,            which prefill KV slot
                  dst_slot  = dst_slot,           which decode KV slot
              )
              endpoint.wait(uuid)                 wait for fabric ack
                                                  (conservative: ensures
                                                   no overlap issues)

After all 62 layers:
  lm_head → argmax → first_token
  return first_token to request loop
  write first_token to SHM → C++ server → user
```

---

## 8. prefill_runner.py Design for MiniMax-M2

Reference: `deepseek_v3_d_p/tt/runners/prefill_runner.py`.

> *Open idea (not a decision):* the prefill scheduler/runner/migration loop looks
> largely model-agnostic, so it might be shareable across models rather than forked
> per model — worth exploring with the team. If we go that way, we'd note where it
> isn't general yet (e.g. assumes MLA / DeepSeek dims). For now this is just a
> reference; our `tt/runners/` is placeholder scaffold.

Shape (mirrors the reference):

```python
# Key env vars (mirrors DeepSeek pattern):
PREFILL_SP = 4
PREFILL_TP = 8
PREFILL_NUM_LAYERS = 62
PREFILL_MAX_SEQ_LEN = 128 * 1024       # or pass per-request actual_isl
PREFILL_CHUNK_SIZE = 5120              # new vs DeepSeek (no chunking there)
PREFILL_ENABLE_MIGRATION = 0 / 1

def main():
    mesh_device = open_mesh_device(shape=(SP, TP))   # 4×8

    pipeline = MiniMaxPrefillPipeline(
        mesh_device, hf_config, config
    )
    pipeline.compile()                # warmup forward

    if enable_migration:
        endpoint = setup_prefill_migration(...)   # migration team's API
        pipeline.setup_migration(endpoint, DECODE_EP_ID)

    if standalone:
        run_standalone_loop(pipeline)  # JSON input, no SHM
    else:
        run_request_loop(pipeline)     # SHM from C++ server
```

**Two modes:**

| Mode | When | How |
|---|---|---|
| Standalone | Testing / benchmarking | JSON file with token_ids, no C++ server needed |
| SHM | Production | C++ server writes requests via shared memory |

---

## 9. Chunked Prefill — What Needs to Change in MiniMax-M2

Currently `max_prefill_chunk_size = 128K` (no chunking). To support 5K chunks:

```
model_config.py:
  self.max_prefill_chunk_size = 5120    # was 128K

generator / prefill_runner (new):
  for chunk_start in range(0, actual_isl, CHUNK_SIZE):
      chunk_end = min(chunk_start + CHUNK_SIZE, actual_isl)
      chunk_tokens = token_ids[:, chunk_start:chunk_end]
      chunk_page_table = page_table[:, chunk_start//block_size : chunk_end//block_size]

      model.ttnn_prefill_forward(
          chunk_tokens,
          chunk_start_idx=chunk_start,    ← SDPA needs this for causal mask
          page_table=chunk_page_table,
          kv_cache=kv_cache,
      )
      on_layer_complete(...)              ← migrate each layer after chunk done?
                                            or after all chunks of a layer?
                                            (open question, see §11)
```

**Key question for the attention kernel:** Does `chunked_scaled_dot_product_attention`
(from `tt_transformers`) accept `chunk_start_idx` as a parameter today for MiniMax-M2's
attention shape? This needs to be verified — it's what other models (llama3_70b_galaxy)
use but MiniMax-M2 currently calls the non-chunked SDPA.

---

## 10. Work Breakdown — Phase 1 (Functional, No Optimization)

Priority order — each item unblocks the next.

### P0 — Correctness fixes (must have before anything else)

| # | Task | File | Notes |
|---|---|---|---|
| 1 | **Decode QK-norm** | `tt/attention/decode.py` | Without this, long-generation quality is wrong. Reference: prefill.py distributed RMSNorm path |
| 2 | **batch_size > 1 in experts** | `tt/experts/prefill.py:243`, `decode.py:50` | Needed for DP=4 multi-user mode. Understand if sparse_matmul kernel limitation or just not implemented |

### P1 — Chunked prefill wiring

| # | Task | File | Notes |
|---|---|---|---|
| 3 | **Set chunk_size=5120** | `tt/model_config.py` | One-line change. Validate SDPA with chunk_start_idx |
| 4 | **Wire chunk_start_idx into SDPA** | `tt/attention/prefill.py` | Need chunked_sdpa or equivalent with correct causal mask offset |
| 5 | **chunk_page_table slicing** | generator / prefill_runner | Slice page_table per chunk like llama3_70b_galaxy does |
| 6 | **Chunk loop in prefill_runner** | `prefill_runner.py` (new) | Iterate chunks, call model forward per chunk |

### P2 — Server connection and migration integration

| # | Task | File | Notes |
|---|---|---|---|
| 7 | **prefill_runner.py** (new) | `tt/runners/prefill_runner.py` | Model on deepseek_v3_d_p. Two modes: standalone + SHM |
| 8 | **MiniMaxPrefillPipeline** (new) | `tt/tt_minimax_prefill_pipeline.py` | Wraps model, handles _prepare_input_tensor, _build_migration_callback |
| 9 | **Migration callback** | inside pipeline | Call migration_team's endpoint.migrate_layer() after each layer |
| 10 | **migration_setup.py** | `tt/runners/migration_setup.py` | Adapt from deepseek_v3_d_p; compute chunk_size_bytes for head_dim=128 |

### P3 — Performance (after functional is confirmed)

| # | Task | Notes |
|---|---|---|
| 11 | slice_write in experts/prefill.py:197 | Eliminate realloc in chunked loop |
| 12 | reduce_scatter at MLP output (layer.py:146) | Keep SP across MLP residual |
| 13 | TTNN trace for prefill | Issue #32056 |
| 14 | Warmup seq_lens | Issue #32818 |
| 15 | Throughput experts (all-to-all) | For high-batch decode |

---

## 11. Open Questions

These need answers before or during P1/P2 work:

**1. Migration granularity: per-layer-per-chunk or per-layer-after-all-chunks?**
```
Option A: migrate after each chunk × each layer
  → 25 chunks × 62 layers = 1550 migration events per request
  → decode has partial KV sooner (layer 0 chunk 0 arrives fast)
  → complex ordering guarantees needed

Option B: migrate after all chunks of a layer complete (per-layer only)
  → 62 migration events per request
  → simpler; decode can't start until prefill finishes all 25 chunks of layer 0
  → recommended for functional bring-up
```

**2. Does chunked_scaled_dot_product_attention support MiniMax-M2's shapes? — RESOLVED ✅**
- `ttnn.transformer.chunked_scaled_dot_product_attention(q, k, v, page_table, chunk_start_idx[/_tensor], ...)`
  exists (ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa.hpp:29). Scalar or device-tensor chunk_start_idx.
- GQA / head_dim agnostic (NQH/NKH are runtime args); llama3_70b_galaxy uses it with GQA + head_dim 128.
- Partial RoPE + QK-norm happen BEFORE SDPA, so the kernel sees plain Q/K — no kernel change needed.
- **Constraint:** chunked SDPA **requires paged KV** (reads K/V from the cache via page_table; no non-paged
  chunked variant). So chunked prefill ⇒ paged mode. Bonus: this exercises the KV-cache READ path
  (currently unvalidated). `chunk_start_idx` is the ABSOLUTE token position and must be a multiple of
  q_chunk_size (scalar variant). Integration is "call the op + slice page_table per chunk", not new kernel work.

**3. SP=4 + chunked prefill — does each row see seq/4 per chunk, or the full chunk?**
- With SP=4 and chunk_size=5120: each row sees 1280 tokens per chunk.
- The `_prepare_input_tensor` SP sharding must happen after chunking, not before.
- Confirm the reshape order in the new pipeline.

**4. What does migration team's API look like exactly?**
- Is it identical to `BoundMigrationEndpoint.migrate_layer()` in DeepSeek?
- Does it support MiniMax-M2's chunk_size_bytes (smaller than DeepSeek's 19 584)?
- When can we get a stub/mock of the endpoint for testing without full disaggregation?

**5. Is `block_size=64` the right paged attention block size?**
- DeepSeek uses 64. For MiniMax-M2, 32-token migration chunks means block_size should be a multiple of 32.
- block_size=64 fits naturally. Verify with memory team.

**6. expert batch_size > 1 — kernel limitation or just not implemented?**
- If the sparse_matmul kernel doesn't support batch>1, DP=4 requires a different experts path.
- If it's just a code gap, it's a straightforward fix in experts/prefill.py.

---

## 12. Reference Implementations

| What to copy | Source |
|---|---|
| prefill_runner.py structure | `models/demos/deepseek_v3_d_p/tt/runners/prefill_runner.py` |
| Pipeline + migration callback | `models/demos/deepseek_v3_d_p/tt/tt_deepseek_prefill_pipeline.py` |
| migration_setup.py | `models/demos/deepseek_v3_d_p/tt/runners/migration_setup.py` |
| KV cache address table | `models/demos/deepseek_v3_d_p/utils/kv_cache_utils.py` |
| Chunked prefill generator loop | `models/tt_transformers/tt/generator.py:1096-1172` |
| chunk_page_table slicing | `models/demos/llama3_70b_galaxy/` generator |

---

## 13. Summary

```
PHASE 1 TARGET: functional prefill, single user, no throughput optimization

INPUT:   token_ids (up to 128K), dst_slot from C++ server via SHM
OUTPUT:  first_token back via SHM, KV cache migrated to decode chips

KEY DESIGN DECISIONS:
  - chunk_size = 5 120 tokens  (25 chunks for 128K)
  - SP = 4  (each row: 1 280 tokens per chunk)
  - TP = 8  (weights sharded across row)
  - Migration: per-layer after all chunks complete  (Option B, simplest)
  - Migration granularity: 32 tokens / chunk  (DRAM bank aligned)
  - runner architecture: standalone mode first, then SHM loop
  - reference: deepseek_v3_d_p for runner + migration pattern

WHAT MIGRATION TEAM OWNS:
  - fabric transport (send/recv, sub-context setup)
  - KvChunkAddressTable + BoundMigrationEndpoint API
  - setup_prefill_migration() setup function

WHAT WE OWN:
  - MiniMaxPrefillPipeline (wraps existing model, adds chunking + migration callback)
  - prefill_runner.py (server connection, request loop)
  - Decode QK-norm fix
  - chunk_start_idx wiring into SDPA
  - chunk_page_table slicing
```

---

## 14. Running on a Blackhole Galaxy + verification scripts (2026-06-12)

### 14.1 Environment

```bash
cd /data/vmelnykov/tt-metal
export TT_METAL_HOME=/data/vmelnykov/tt-metal
export PYTHONPATH=$TT_METAL_HOME
source python_env/bin/activate              # ttnn is built here; /usr/bin/python3 has NO ttnn
# venv has no pip — install with: uv pip install --python $TT_METAL_HOME/python_env/bin/python <pkg>
# requires transformers==4.57.1 (M2.7 modeling code needs GenericForQuestionAnswering)
```

### 14.2 Mesh / fabric / MGD — the rules that bite

This box is a **plain 8×4 MESH** (no torus / wrap-around links). Two hard rules (else hang/crash):
1. **Mesh-open shape must equal the MGD `device_topology dims`.** Stock `single_bh_galaxy_*`
   MGDs are `[8,4]`. We added `single_bh_galaxy_4x8` (`[4,8]`, transpose — for TP=8 on cols)
   and `single_bh_galaxy_1x8` (line of 8 — pure TP=8) under `tt_metal/fabric/mesh_graph_descriptors/`.
2. **Collectives need `FABRIC_1D` + `ttnn.Topology.Linear`.** `FABRIC_1D_RING` / torus MGDs do
   NOT fit this box. The model's `CCLManager` defaults to `Ring`; pass `topology=Linear` here.

```bash
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_1x8_mesh_graph_descriptor.textproto
```

Parallelism that actually maps to this hardware: DeepSeek uses `(8,4)`=SP8/TP4; we run
TP=8 via the transposed `(4,8)` (cols=TP=8, rows=SP=4), or `(1,8)` for pure TP=8.

### 14.3 Verification scripts (all under `tests/`)

| Script | What it does | Mesh | Needs |
|---|---|---|---|
| `galaxy_mesh_smoke.py` | mesh opens + 1 collective round-trips (sanity that fabric works) | any | — |
| `galaxy_layer_smoke.py` | one `DecoderLayer` fwd, random wts (TP collective path executes) | `(1,8)` | — |
| `galaxy_model_smoke.py` | full 62-layer assembly fwd, random wts (no OOM/hang, valid first token) | `(1,8)` | — |
| `galaxy_first_token.py` | **real weights** → prefill → first token (`--dump-logits` to save) | `(1,8)` | `HF_MODEL` |
| `hf_reference_oracle.py` | HF CPU forward → saves argmax/top-5/logits ground truth | — (CPU) | `HF_MODEL` |
| `tests/unit/test_attention_vs_hf.py -k 1x8` | attention PCC vs HF at TP=8 | `(1,8)` | `HF_MODEL` (modeling code only) |
| `tests/unit/test_experts_vs_hf.py -k 1x8` | experts PCC vs HF at TP=8 | `(1,8)` | `HF_MODEL` (modeling code only) |
| `test_ep_moe_vs_ref.py` | **EP MoE block** vs torch (256 exp 8/chip) — PASS 0.9803 | `(8,4)`/`(4,8)` | — |
| `galaxy_ep_forward_smoke.py` | full model DP-attn + EP=32 fwd, random wts (runs/finite) | `(4,8)` | — |
| `galaxy_first_token_ep.py` | **real weights, EP=32, 4 prompts → first tokens**; verifies prompt0 vs oracle (PASS: 758 'The') | `(4,8)` | `HF_MODEL` |

### 14.3.1 Local-only vs pushable
The `tests/unit/*.py` files **are committed** (they need only `HF_MODEL` = the small modeling
code / a checkpoint). The standalone `galaxy_*.py` / `test_ep_moe_vs_ref.py` driver scripts are
**committed too**, but they depend on artifacts that **cannot be pushed to origin** and stay LOCAL:
- the **dequantized bf16 checkpoint** (`/data/vmelnykov/MiniMax-M2`, ~460 GB),
- the **HF oracle** ground truth (`/data/vmelnykov/minimax_m2_ref/`),
- the **per-config TTNN weight caches** (`…/tensor_cache_*`).
So a teammate reproduces by: download `MiniMaxAI/MiniMax-M2.7` → `dequantize_hf_checkpoint.py`
→ set `HF_MODEL` → run. The scripts + MGDs are in-repo; the weights are not.

PCC tests need only the M2.7 **modeling code** (`*.py`+config, a few KB — they build random-weight
references). The first-token run needs the full **dequantized bf16** checkpoint at `HF_MODEL`
(download `MiniMaxAI/MiniMax-M2.7` fp8 — public, no token — then `dequantize_hf_checkpoint.py`;
output dir must be named `MiniMax-M2` for the `model_config.py` assert).

### 14.4 Regression against ground truth
`/data/vmelnykov/minimax_m2_ref/` holds `ref_results.json` + HF/device logit `.npy`. After any
change, re-run `galaxy_first_token.py --dump-logits …` and compare argmax + logit PCC.

---

## 15. EP (expert parallel) — Phase 2 plan (scoped 2026-06-12)

EP is **no longer "undefined"** — the ops are compiled and generic, and the placement is verified.

**Layout (DeepSeek scheme, our TP=8).** `(4,8)`: `num_dispatch_groups = cols = 8`,
`dispatch_group_size = rows = 4` → `256 // 8 // 4 = 8` **whole experts/chip** across all 32 chips
→ ~3.3 GB/chip bfp4 (vs EP=1's 14 GB — EP=1 replicates all experts across SP rows). Verified via
`deepseek_v3_d_p` `ExpertMapping.create_dispatch_table`.

**Reuse (do):** `ttnn.experimental.deepseek_prefill.{dispatch, routed_expert_ffn, combine}` +
`TtDispatchModule`/`TtCombineModule` + `ExpertMapping`. **Do NOT reuse** DeepSeek attention (MLA ≠
our GQA), `moe_grouped_topk` (M2 has no expert groups → plain top-8), or shared-expert (M2 has none).

**Build order:**
- **E2** wire MiniMax router top-8 `(indices, weights)` → `dispatch` → `routed_expert_ffn` →
  `combine` → reduce, in `tt/experts_throughput/`; load 8 experts/chip expert-centric.
- **E3** PCC-validate the EP MoE block in isolation (like `test_experts_vs_hf`, EP layout) —
  doesn't need SP attention, so can start on `(1,8)` (dispatch correctness, 32 experts/chip).
- **E4** integrate into layer/model on `(4,8)`; full-model logit PCC vs the saved oracle. Couples
  with the SP=4 + chunked-attention Tier-1 work.

---

## 16. Parallelism strategy on 32 chips (proposal — for team review, 2026-06-12)

How to lay MiniMax-M2 across the BH Galaxy's 32 chips. **Target architecture (confirmed):**
**TP for attention / dense / lm_head; EP for the MoE experts.**

### 16.1 Fixed: TP=8 across columns
Every weight matrix (attention q/k/v/o, lm_head, embed) is sharded ÷8 across the 8 **columns**.
One row of 8 chips = one complete TP-sharded copy of the dense weights. `num_kv_heads=8` → 1 KV
head/chip (clean). Validated: attention PCC 0.9991 at TP=8.

### 16.2 The 4 rows — three ways to use them (DP / SP / EP)

| Axis on rows | Rows hold | Dense weights | Experts | Goal |
|---|---|---|---|---|
| **DP** | 4 different prompts | dup ×4 | (see EP below) | throughput |
| **SP** | 1 prompt, seq÷4 | dup ×4 | (see EP below) | long context / latency |
| **EP** | — (orthogonal) | — | experts spread, not duplicated | expert-memory efficiency |

The dense (attention) weights are duplicated across the 4 rows in **both** DP and SP (each row needs
full attention weights for its tokens). EP is about the **experts**: spread vs duplicated.

### 16.3 Governing rules (the arithmetic)
- `(distinct expert copies) = DP`, and `DP × (chips per expert copy) = 32`. So **EP=32 (one copy
  over all 32 chips) ⟹ DP=1**; **DP=4 ⟹ EP=8 per replica (experts duplicated ×4)**.
- **EP's `dispatch_group_size` = the mesh ROW axis = the SP/DP axis.** The dispatch routes tokens
  *among the rows* to expert-owning chips, so it needs rows>1 with tokens distributed across them
  (`dgs=1` breaks the routing op). EP and SP/DP-attention share the row axis by design.

### 16.4 Two target configs

**(A) SP=4 + EP=32** — one prompt, long context, min expert memory.
Rows = seq-shards (seq÷4); experts spread 8/chip (no dup). **~3.3 GB/chip** experts (bfp4).
Needs the SP-attention path (chunked/paged, Tier-1).

**(B) DP-attention(=4) + shared EP=32** — 4 prompts at once, no SP needed.
Each row = one prompt doing **full** attention (TP=8 only); one **shared** EP=32 expert pool;
dispatch gathers all 4 prompts' tokens → experts → scatters back. Experts **not** duplicated
(~3.3 GB/chip). This is the SOTA "DP-attention + shared-EP" serving pattern, and it's reachable
**without** writing SP attention (uses the model's `users_row_sharded` path). Being brought up now.

> Note: (B)'s "DP=4" is *attention* data-parallel with a **shared** expert pool — NOT 4 independent
> replicas. The shared EP keeps experts at 32-way (no ×4 duplication); the dispatch juggles all
> streams' tokens. Independent `DP=4` replicas would instead force EP=8 (experts ×4, ~14 GB/chip).

### 16.5 Token flow through dispatch/combine (same for SP+EP and DP+EP)
```
(4×8): TP=8 cols. EP: dispatch_group_size=4 ROWS, num_dispatch_groups=8 COLS. 8 whole experts/chip.
Column c owns experts [32c..32c+31] (8 per row). Rows = SP seq-shards OR DP prompts (identical mechanics).

token T on row R1, router picked experts {5, 70, 200}:
 ① after attention: T's emb TP-sharded across C0..C7 on R1
 ② all-gather emb across COLS  → every chip on R1 has T's FULL emb (T replicated across cols)
 ③ DISPATCH (all-to-all within each column, across the 4 rows):
       C0 owns e0-31  → T→chip owning e5  in C0
       C2 owns e64-95 → T→chip owning e70 in C2
       C6 owns e192.. → T→chip owning e200 in C6
 ④ ROUTED_EXPERT_FFN (local, WHOLE expert weights, dense matmul on routed tokens)
 ⑤ COMBINE (all-to-all back) → results return to T's origin row R1
 ⑥ REDUCE (weighted-sum over top-k by router weights + reduce-scatter across COLS)
       out(T) = Σ w_e·e(T) → lands TP-sharded on R1 for the next layer
```
Experts are **stateless about which stream a token came from** — that's why the rows can be SP
seq-shards or DP prompts with the *same* dispatch/combine machinery.

### 16.6 Memory (experts dominate: 224B of 229B params; bfp4 ≈ 0.5 B/param)
| Config | experts/chip | bfp4 GB/chip | notes |
|---|---|---|---|
| EP=1, TP=8 (current functional) | all 256 (÷8 TP), dup ×SP rows | ~14 | wasteful; fits |
| **EP=32** (configs A & B) | **8 whole** | **~3.3** | target |
| DP=4 independent replicas (EP=8) | 32 whole | ~14 | throughput, experts ×4 |

### 16.7 Status
TP=8 attention/experts validated vs HF (§0). EP MoE block (`TtMiniMaxMoE`) validated PCC 0.9803
on `(8,4)`. Config **(B)** (DP-attention + EP) being brought up via `users_row_sharded` + `use_ep_moe`
— the reachable full-forward EP path. Config **(A)** (SP=4 + EP) awaits the chunked/SP-attention
Tier-1 work. Decode (EP=4 across rows) lives in tt-blaze.

---

## 17. Cleanup / tech-debt follow-up (2026-06-12)

**Principle (important):** "dead code" here must be judged against the **full serving call
graph**, not in-repo references. Many methods show **0 internal refs** yet are **live API
surface** — the callers are the `tt_transformers` **Generator**, the prefill **pipeline /
runner / scheduler**, and **migration** (mostly `NotImplementedError` scaffold *now*, so they
don't show as refs yet but *will* call this surface), plus **vLLM** duck-typing the interface.
A plain `grep` over `models/demos/minimax_m2/` will mislabel interface methods as dead.

### Bucket A — KEEP (interface / load-bearing; do NOT delete)
- `prepare_inputs_prefill` / `prepare_inputs_decode` / `process_output_prefill` /
  `process_output_decode` / `prepare_prefill_inputs_trace` / `process_logits_after_prefill_trace`
  — the **Generator interface** (`models/tt_transformers/tt/generator.py` calls all of these by
  name). Needed by the serving layer.
- `tt/experts_throughput/decode.py` — **prefill** imports `_apply_swiglu` from it (load-bearing
  for prefill despite the name).
- `tt/experts_throughput/fused_decode.py` — imported by `experts_throughput/__init__.py` and
  called in `ThroughputExperts.forward`'s decode branch.

### Bucket B — CONSOLIDATION CANDIDATE (deliberate refactor, separate PR, team sign-off)
- Retire the **older EP scaffold `ThroughputExperts`** (the `use_throughput_experts` path) now
  that **`TtMiniMaxMoE`** (`use_ep_moe`, validated PCC 0.9803) covers EP. That would remove
  `decode.py` + `fused_decode.py` + the `use_throughput_experts` branch — but requires
  re-pointing the MLP, relocating `_apply_swiglu`, and dropping the decode forwards. Not a quick
  delete; a conscious consolidation.

### Bucket C — SAFE-DELETE (trivially unreachable; verify + run unit tests after)
- The unused `sinks` tensor in `tests/test_factory._generate_dummy_state_dict` (M2 has no sinks).
- (sweep for other leftover debug/experimental bits during the cleanup pass.)

### Decode-specific helpers note
`prepare_inputs_decode` / `process_output_decode` / `_increment_decode_positions_device` are
*vestigial for this prefill-only repo* (decode runs in tt-blaze), but are part of the Generator
interface vLLM/serving may duck-type. **Leave them with a `# kept for Generator/serving interface`
note rather than delete** — removing risks the serving integration.
