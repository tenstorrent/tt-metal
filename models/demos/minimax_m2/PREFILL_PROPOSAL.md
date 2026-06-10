# MiniMax-M2 Prefill — Functional Bring-Up Proposal

> **Goal of this document:** Define the architecture and work breakdown for a
> *functional* (correctness-first, not optimized) MiniMax-M2 prefill pipeline
> on a single Blackhole Galaxy. This is a starting point for team discussion.

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
✅ Attention prefill
     QKV projection → distributed QK-norm → partial RoPE (dim 64)
     → paged_fill_cache (per user) → causal SDPA → output proj + reduce-scatter

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
❌ batch_size > 1 in experts  →  NotImplementedError  (experts/prefill.py:243)
                                  blocks DP=4 mode entirely

⚠️ Decode QK-norm UNVALIDATED →  the qk-norm call IS wired into decode.py, but the
                                  decode path uses a sharded-L1 xqkv layout the
                                  apply_qk_norm slicing was not written/tested for.
                                  Decode is entirely unrun — validate before trusting
                                  long-generation accuracy. (attention/decode.py)

❌ No prefill runner / server connection
                              →  text_demo.py is a standalone script,
                                  no SHM or request loop

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

Modelled directly on `deepseek_v3_d_p/tt/runners/prefill_runner.py`:

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

**2. Does chunked_scaled_dot_product_attention support MiniMax-M2's shapes?**
- MiniMax-M2 uses partial RoPE (dim 64), QK-norm, full causal (no sliding window).
- Verify `chunk_start_idx` parameter works with the kernel as-is, or if adaptation needed.

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
