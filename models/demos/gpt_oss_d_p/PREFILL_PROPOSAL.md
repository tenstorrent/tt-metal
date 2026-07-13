# GPT-OSS Prefill — Functional Bring-Up Proposal

> **Goal of this document:** Define the architecture and work breakdown for a
> *functional* (correctness-first, not optimized) GPT-OSS prefill pipeline
> running as a prefill-only component in a disaggregated prefill/decode setup.
>
> **This is a LIVING doc** — it is kept in sync with the code as the source of
> truth for the team (and for handing context to an agent). Rule: it never claims
> something works that isn't validated; status is either *validated (with PCC +
> conditions)* or *scaffold (owner + blocker)*.

---

## 0. Status (living) — last updated 2026-06-15

**Model (validated — existing tests):**
Prefill and decode paths functional. Existing accuracy tests in `tests/accuracy/` and
unit tests in `tests/unit/` pass. The architecture deltas vs a plain transformer
(sliding-window attention on alternating layers, TopK MoE with 128 experts / top-4,
attention sinks, paged attention) are all implemented and tested.

**P0 + P1 + P2 DONE:**

- ✅ `on_layer_complete` hook wired into `tt/model.py` (`_forward_layers_and_head` +
  `ttnn_prefill_forward`). Default=None → zero impact on existing paths.
- ✅ `tt/runners/migration_setup.py` — `MigrationEndpoint` protocol + `NoOpMigrationEndpoint`.
- ✅ `tt/runners/prefill_runner.py` — `build_pipeline()` + `run_standalone_loop()` implemented.
  `run_request_loop()` remains `NotImplementedError` (Tier 2 / serving team).
- ✅ `tt/tt_gpt_oss_prefill_pipeline.py` — `GptOssPrefillPipeline` fully implemented:
  - `_prepare_input_tensor()`: SP-shards token IDs across mesh rows, embeds on device.
  - `_extract_first_token()`: SP-aware output extraction (targets the SP row holding the last real token).
  - `prefill()`: pads, builds callback, calls `ttnn_prefill_forward`, extracts first token.
  - `compile()`: warmup forward pass for JIT.
- ✅ `tt/runners/standalone_input.json` — sample input for standalone bring-up.

**Remaining (P3 — migration team):**

- **`setup_prefill_migration()` full impl** — fabric transport + sub-context setup
- **KV chunk address table** for separate k+v per layer (head_dim=64, bfp8, ≈2176 bytes)

### ⚠️ GAPS — what is NOT done yet

- **No KV migration.** `NoOpMigrationEndpoint` used; real endpoint blocked on migration team.
- **Decode is out of scope** (runs in tt-blaze) → this component produces the **first token only**.
- **`run_request_loop()`** — Tier 2, serving team, blocked on SHM protocol + C++ server.
- **`_extract_first_token` not validated on Galaxy** — SP-aware slicing is correct per analysis
  but needs a hardware run to confirm the `get_device_tensors` ordering assumption (row-major:
  `device_tensors[sp_row * tp + col]`). Verify with a known-token end-to-end test.

---

## 1. Model Architecture at a Glance

From `configs/gpt-oss-120b/config.json`:

| Parameter | Value |
|---|---|
| `num_hidden_layers` | 36 |
| `hidden_size` | 2880 |
| `num_attention_heads` | 64 |
| `num_key_value_heads` | 8 |
| `head_dim` | 64 |
| `num_local_experts` | 128 |
| `num_experts_per_tok` | 4 (top-4 routing) |
| `intermediate_size` | 2880 |
| `max_position_embeddings` | 131 072 (~128K) |
| `vocab_size` | 201 088 |
| `sliding_window` | 128 (alternating layers) |

**Notable differences from MiniMax-M2 / DeepSeek:**
- **Standard GQA** (64 Q heads / 8 KV heads, no QK-norm, no partial RoPE).
- **Sliding-window attention** on alternating layers (window=128). Full-attention
  layers and sliding-window layers alternate; both are handled by the same
  `prefill_forward` via the `sliding_window` config field.
- **128 experts / top-4** — smaller MoE than DeepSeek (256/top-8) but same
  architecture (dense matmul routing via `TopKRouter`, no sigmoid correction bias).
- **Separate `k_cache` + `v_cache` per layer** — NOT the single `kvpe_cache`
  used by DeepSeek V3. Migration must handle two tensors per layer.
- **Attention sinks** — pre-divided by scale factor, stored in `weights.sinks`.

---

## 2. Target System — One Blackhole Galaxy

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
         ◄── SP = 4 (rows) ──────────────────────────────►
```

**Parallelism for prefill (from `config.py` `MeshConfig`):**

| Axis | Mode | Value | What it means |
|---|---|---|---|
| TP | Both | 8 | Weight matrices column-sharded across 8 chips in a row |
| SP | Prefill | 4 | Sequence sharded across 4 rows; each row processes `seq_len / 4` tokens |
| EP | Decode | 4 | Expert parallel in decode; each row owns 32 of 128 experts |
| EP | Prefill | 1 | Experts replicated in prefill (all rows see all 128 experts) |

With SP=4, `num_kv_heads=8`, `TP=8`: each chip holds 1 KV head (`8 // 8 = 1`). Clean.

---

## 3. What Already Works

```
✅ Attention prefill (existing tests)
     QKV projection → RoPE → causal SDPA (sliding-window or full)
     → attention sinks → output proj + reduce-scatter / allreduce

✅ KV cache
     Paged and non-paged variants
     Shape: [batch_or_blocks, num_kv_heads // TP, seq_len_or_block, head_dim]
     dtype: bfloat8_b
     Initialized per layer in tt/attention/kv_cache.py

✅ MoE prefill (existing tests)
     TopKRouter → top-4 selection → experts (sparse matmul)
     → EP/TP/SP allreduce chain

✅ Mode-aware MeshConfig
     Separate prefill (SP=4, EP=1, TP=8) and decode (EP=4, SP=1, TP=8) configs

✅ Full model forward + lm_head + sampling
     ttnn_prefill_forward() and _forward_layers_and_head() both exist
     Paged attention, sliding-window, per-layer page tables all supported
```

**Hard gap for disaggregated serving:**
```
❌ No on_layer_complete hook in model.py
   → ttnn_prefill_forward cannot fire per-layer callbacks
   → blocks the migration seam

❌ No runners/ or pipeline
   → no GptOssPrefillPipeline, no prefill_runner.py, no migration_setup.py
   → decode never receives KV from prefill
```

---

## 4. Target Architecture — Disaggregated System

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
   │  prefill_runner.py      │         │  decode_runner.py (blaze)  │
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

> **Note:** The KV migration transport layer (fabric send/recv, address tables,
> sub-context setup) is owned by the migration team. We integrate via a
> `MigrationEndpoint` interface (see §7). Until it lands, `NoOpMigrationEndpoint`
> makes the pipeline runnable in standalone mode.

---

## 5. Per-Layer KV Migration Flow

Unlike DeepSeek V3 D/P which has a single contiguous `kvpe_cache`, GPT-OSS has
**separate `k_cache` and `v_cache` per layer**. The migration callback fires after
each layer's KV write is complete; the migration endpoint must handle both tensors.

```
pipeline.prefill(token_ids, slot_id, actual_isl, dst_slot):
│
├─ _prepare_input_tensor()     [Tier 1]
│    shard token_ids across SP=4 rows
│
├─ _build_migration_callback(slot_id, actual_isl, dst_slot)
│    returns on_layer_complete() closure
│
└─ model.ttnn_prefill_forward(tt_tokens, kv_cache, on_layer_complete=callback)
      │
      for layer_idx in range(36):
      │
      │   ┌─── transformer layer ───────────────────────────────┐
      │   │  attention → fills kv_cache[layer_idx] (k + v)      │
      │   │  moe experts                                        │
      │   └─────────────────────────────────────────────────────┘
      │
      └── on_layer_complete(layer_idx):
              ttnn.synchronize_device()
              uuid = endpoint.migrate_layer(
                  layer_idx,
                  pos_start = 0,
                  pos_end   = ceil(actual_isl / 32) * 32,  # 32-token DRAM-bank granularity
                  src_slot  = slot_id,
                  dst_slot  = dst_slot,
              )
              endpoint.wait(uuid)     # conservative: wait on every layer

After all 36 layers:
  lm_head → argmax → first_token
  return first_token to request loop
```

**KV layout note for migration team:**
- Shape (per layer, per device): `[batch, num_kv_heads // TP, seq_len // SP, head_dim]`
  = `[1, 1, seq_len // 4, 64]` at SP=4, TP=8
- `chunk_size_bytes` = `32 × 1 (kv_head/chip) × 64 (head_dim) × ~1.06 (bfp8 overhead)`
  ≈ **2 176 bytes** — smaller than DeepSeek's 19 584 (DeepSeek: head_dim=576 kvpe)
- Two addresses per layer (k_cache and v_cache): the address table must encode both

---

## 6. Work Breakdown

### P0 — Migration hook in model.py (unblocks everything)

One small change, two locations. Reference: `minimax_m2/tt/model.py`.

| # | File | Change |
|---|---|---|
| 1 | `tt/model.py:_forward_layers_and_head()` | Add `on_layer_complete=None` param; call `on_layer_complete(i)` after each layer |
| 2 | `tt/model.py:ttnn_prefill_forward()` | Add `on_layer_complete=None` param; thread it into `_forward_layers_and_head` |

Total: ~6 lines. Zero risk to existing paths (default=None = current behavior).

### P1 — New files: runners + pipeline

All three are ports from `minimax_m2`, adapted for GPT-OSS config.

| # | File (new) | Source | Notes |
|---|---|---|---|
| 3 | `tt/runners/__init__.py` | — | Empty |
| 4 | `tt/runners/migration_setup.py` | `minimax_m2/tt/runners/migration_setup.py` | `MigrationEndpoint` protocol + `NoOpMigrationEndpoint`; `setup_prefill_migration` stays `NotImplementedError` (migration team) |
| 5 | `tt/runners/prefill_runner.py` | `minimax_m2/tt/runners/prefill_runner.py` | Scaffold; adapt env var defaults: `PREFILL_NUM_LAYERS=36`, `PREFILL_MAX_SEQ_LEN=128*SP`, etc. |
| 6 | `tt/tt_gpt_oss_prefill_pipeline.py` | `minimax_m2/tt/tt_minimax_prefill_pipeline.py` | `GptOssPrefillPipeline`; `prefill()` and `_prepare_input_tensor()` start as `NotImplementedError` scaffold |

### P2 — Fill in the scaffolds (Tier 1 runner work)

| # | Task | Notes |
|---|---|---|
| 7 | `GptOssPrefillPipeline._prepare_input_tensor()` | Shard token_ids per SP axis; SP sharding happens AFTER chunking |
| 8 | `GptOssPrefillPipeline.prefill()` | Chunk loop + `model.ttnn_prefill_forward(on_layer_complete=...)` |
| 9 | `prefill_runner.build_pipeline()` | Open mesh, `create_tt_model`, wrap in pipeline, `pipeline.compile()` |
| 10 | `prefill_runner.run_standalone_loop()` | JSON token_ids in, `pipeline.prefill()`, first_token out |
| 11 | `prefill_runner.run_request_loop()` | SHM request loop (serving team, Tier 2) |

### P3 — Real migration (cross-team)

| # | Task | Owner |
|---|---|---|
| 12 | `setup_prefill_migration()` full impl | Migration team |
| 13 | KV chunk address table for k_cache + v_cache layout | Migration team (needs BH DRAM layout for gpt_oss head_dim=64) |
| 14 | Sub-context setup, fabric transport | Migration team |

---

## 7. Env Vars (mirror DeepSeek / MiniMax)

| Var | Default | Notes |
|---|---|---|
| `PREFILL_SP` | `4` | SP axis (rows) |
| `PREFILL_TP` | `8` | TP axis (cols) |
| `PREFILL_NUM_LAYERS` | `36` | gpt_oss-120B |
| `PREFILL_MAX_SEQ_LEN` | `128 * PREFILL_SP` | sequence length budget |
| `PREFILL_ENABLE_MIGRATION` | `0` | `1` → real endpoint; `0` → NoOp |
| `PREFILL_STANDALONE` | `0` | `1` → JSON input mode, no SHM |
| `DEEPSEEK_V3_HF_MODEL` | — | Path to HF config (reused var) |

---

## 8. Open Questions

**1. Migration granularity: per-layer or per-layer-per-chunk?**
- Per-layer (Option B, 36 migration events/request) is recommended for initial bring-up.
- Per-chunk would allow decode to start earlier but complicates ordering.

**2. Does the migration team's endpoint support separate k/v addresses per layer?**
- DeepSeek uses a single `kvpe_cache` buffer. GPT-OSS has two (k+v).
- Need to confirm if `migrate_layer()` accepts two source addresses or if we need
  to migrate k and v as separate calls.

**3. Sliding-window layers and migration.**
- Alternating layers have `sliding_window=128`. Only the last 128 token positions
  in those KV caches are meaningful. Should migration skip old positions for
  sliding-window layers, or send the full buffer?
- Recommendation: send full buffer initially (simpler); optimize later.

**4. Chunk size.**
- DeepSeek uses full sequence (no chunking). MiniMax-M2 targets 5120-token chunks.
- GPT-OSS 120B has 36 layers (shorter stack). A simple starting point is no chunking
  until the per-layer migration rate proves acceptable, then tune.

**5. KV cache block size.**
- Current `init_kv_cache` defaults to the paged attention config block size.
- Migration chunk granularity is 32 tokens. `block_size` should be a multiple of 32.
- Confirm block_size=64 with memory team (matches DeepSeek default).

---

## 9. Reference Implementations

| What to copy | Source |
|---|---|
| `on_layer_complete` hook | `minimax_m2/tt/model.py` lines 351, 401 |
| `MigrationEndpoint` + `NoOpMigrationEndpoint` | `minimax_m2/tt/runners/migration_setup.py` |
| `prefill_runner.py` structure | `minimax_m2/tt/runners/prefill_runner.py` |
| `GptOssPrefillPipeline` structure | `minimax_m2/tt/tt_minimax_prefill_pipeline.py` |
| Full migration impl (when ready) | `deepseek_v3_d_p/tt/runners/migration_setup.py` |
| KV chunk address table | `deepseek_v3_d_p/utils/kv_cache_utils.py` |
| BoundMigrationEndpoint | `deepseek_v3_d_p/tt/tt_deepseek_prefill_pipeline.py` |

---

## 10. What Migration Team Owns

- Fabric transport (send/recv, sub-context setup, NOC address encoding)
- `KvChunkAddressTable` for GPT-OSS's layout (k + v separate, head_dim=64, bfp8)
  - `chunk_size_bytes` ≈ 2176 bytes (vs DeepSeek's 19584)
  - Two-tensor-per-layer addressing
- `setup_prefill_migration()` — currently `NotImplementedError` in `migration_setup.py`
- `BoundMigrationEndpoint` → real impl of `migrate_layer()` + `wait()`

---

## 11. Verification

After P0 (model.py hook):

```bash
# Regression: existing prefill tests must still pass
pytest models/demos/gpt_oss/tests/ -x

# Smoke: on_layer_complete fires for each layer
python -c "
import ttnn
# ... open mesh, load model ...
fired = []
model.ttnn_prefill_forward(
    x, kv_cache=kv_cache,
    on_layer_complete=lambda i: fired.append(i)
)
assert fired == list(range(36)), fired
"
```

After P1 (pipeline + runners):

```bash
# Import smoke (no device needed)
python -c "from models.demos.gpt_oss_d_p.tt.tt_gpt_oss_prefill_pipeline import GptOssPrefillPipeline"
python -c "from models.demos.gpt_oss_d_p.tt.runners.migration_setup import NoOpMigrationEndpoint; e = NoOpMigrationEndpoint(); assert e.migrate_layer(0,0,128,0,0) == 0"

# NoOp migration callback smoke
pipeline = GptOssPrefillPipeline(mesh, hf_config, model)
cb = pipeline._build_migration_callback(slot_id=0, actual_isl=512, dst_slot=0)
cb(0)  # should not error
```
