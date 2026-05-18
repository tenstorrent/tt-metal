# [SUPERSEDED 2026-05-17] V2-PAD: Move head sharding to 8-way ring

**This plan was based on a false premise and is no longer applicable.**

The "match llama70b" framing assumed llama70b splits attention heads on
`cluster_axis=0` (the 8-way ring), which would have required padding qwen3.6's
4 KV heads to 8 for divisibility. Verification on 2026-05-17 showed:
- llama70b uses `mesh_shape=(8,4)` (hardcoded in `llama_ccl.py:1255`, `lm_head.py:114`)
- llama70b's `llama_rs_create_heads(cluster_axis=1)` keeps heads on the 4-way ring
- llama70b's WO `line_all_reduce(cluster_axis=0)` is on the 8-way ring
- V2 already matches this exactly (post V2-CCL)

The remaining full-attn-vs-DeltaNet per-call RS gap is NOT cluster_axis — it's
that V2-CCL-followup left the width-sharded L1 persistent-buffer WO path behind
env flags (`QWEN36_FULLATTN_LAR` / `QWEN36_FULLATTN_WO_TUNED`). See V2-WO-PROMOTE
task for the actual fix (~50 lines, not ~600).

Original (now-irrelevant) plan follows:

---

## Why (original — no longer applies)

v2 currently splits QKV heads on `cluster_axis=1` (cols, 4-way ring):
- 24 Q / 4 cols = 6 Q per chip
- 4 KV / 4 cols = 1 KV per chip

This is forced by the **4 KV heads** — they don't divide by 8 rows. WO `line_all_reduce`
already lives on `cluster_axis=0` (8-way) after V2-CCL, but everything upstream of
WO (QKV create_heads, SDPA, AG-concat) is on the 4-way ring.

To move the upstream ops onto the 8-way ring, KV must be padded 4 → 8 by
**replicating each KV head at load time**:
```
[k0, k1, k2, k3]  →  [k0, k0, k1, k1, k2, k2, k3, k3]   # GQA grouping preserved
```
After padding:
- 24 Q / 8 rows = 3 Q per chip
- 8 KV / 8 rows = 1 KV per chip  ← divides cleanly

GQA grouping stays correct: original 24 Q / 4 KV → 6 Q per KV group. With each KV
replicated twice, the 6 Q heads in each original group still attend to a copy of
the original KV head — math is identical.

## Mesh-shape convention (v2)

- `cluster_shape = (8, 4)` → rows=8, cols=4
- `cluster_axis=0` = rows (8-way ring)
- `cluster_axis=1` = cols (4-way ring)

(Olmo uses `cluster_shape=(4,8)`, the transpose — its `cluster_axis=0` is the
4-way ring. Do NOT copy olmo's `cluster_axis` literals; copy the *intent*.)

## Files to change

### 1. `tt/qwen36_model_config.py`

| line(s) | change |
|---|---|
| `self.n_kv_heads = 4` | `self.n_kv_heads_unpadded = 4` + `self.n_kv_heads = 8` (padded) |
| `self.n_local_heads = n_q_heads // cluster_shape[1]` | `// cluster_shape[0]` (rows). 24/8=3 |
| add | `self.n_local_kv_heads = self.n_kv_heads // cluster_shape[0]` (8/8=1) |
| add | `self.n_local_heads_padded = max(self.n_local_heads, 8)` if tile-pad needed (3*256=768 already tile-aligned, so no extra Q pad needed; keep field for parity with olmo) |
| `assert self.n_kv_heads % self.cluster_shape[1] == 0` | `assert self.n_kv_heads % self.cluster_shape[0] == 0` |
| `min_kv_prefill_shard_seqlen` formula | swap `cluster_shape[1]` → `cluster_shape[0]` |
| QKV size in program configs | recompute with padded KV: `head_dim * (n_q + 2*n_kv_padded) = 256 * (24 + 16) = 10240` (was `256*(24+8)=8192`) |
| Per-chip QKV width | `(n_q + 2*n_kv_padded) * head_dim / rows = (24+16)*256/8 = 1280` (was `(24+8)*256/4 = 2048`) |
| WO input width per chip | `n_q * head_dim / rows = 24*256/8 = 768` (was `24*256/4 = 1536`) |

Audit every other `cluster_shape[1]` reference in this file — most should flip to
`cluster_shape[0]` for the head-split path. Norm / LM-head / MLP configs may stay
on cluster_shape[1].

### 2. `tt/load_checkpoints.py` (qwen36 branch)

In the qwen36 weight-ingestion path:

```python
# K projection: [hidden, n_kv_unpadded * head_dim] = [5120, 1024]
# Pad to:       [hidden, n_kv_padded   * head_dim] = [5120, 2048]
# By replicating each head twice (preserves GQA grouping for 6 Q / KV).
k_w = state_dict["...k_proj.weight"]  # [n_kv * head_dim, hidden]
k_w = k_w.view(n_kv_unpadded, head_dim, hidden)
k_w_padded = k_w.repeat_interleave(2, dim=0)  # [8, head_dim, hidden]
k_w_padded = k_w_padded.reshape(n_kv_padded * head_dim, hidden)
# Same for v_proj and k_norm.weight (length 4 → 8 via repeat_interleave)
```

For QKV fused weight: concat Q (24 heads, native) + K_padded (8 heads, replicated)
+ V_padded (8 heads, replicated). Final QKV out-dim: `(24+8+8)*head_dim = 10240`
(was `(24+4+4)*256 = 8192`).

**ShardTensor2dMesh dims change.** The QKV weight is sharded 2D. Current dims
might be `(None, -1)` or `(0, None)` for cluster_axis=1 split. New dims must put
the head split on cluster_axis=0 (rows). Mirror llama70b's pattern:
```python
mesh_mapper=ttnn.ShardTensor2dMesh(
    mesh_device, dims=(3, 2),  # head-dim split on axis 1, hidden on axis 0 — verify against llama70b
    mesh_shape=cluster_shape  # (8, 4)
)
```
Read llama70b/tt/llama_attention.py:169-181 for the exact dims tuple and confirm
this puts heads on rows (axis=0) and hidden on cols (axis=1).

### 3. `tt/llama_attention.py` (is_qwen36 branch)

| line | change |
|---|---|
| 911 | `llama_rs_create_heads(... cluster_axis=1)` → `cluster_axis=0` |
| 1034 | `all_gather_concat(... cluster_axis=1)` → re-evaluate; if WO input wants heads on rows, this should also be cluster_axis=0 |
| 1120 | `line_all_reduce(... cluster_axis=1, buffer_key="QKV")` — check if QKV residual is now sharded differently; may stay on cluster_axis=1 (input residual is hidden-dim sharded on cols, unchanged) OR flip to 0 |
| 1294 | `ATTN_REPLICATE` — only used in some chunked-SDPA path; re-derive |
| 1338, 1348 | `ring_all_gather(cluster_axis=1)` — same |
| QK-norm | now operates on 3 Q heads (was 6) and 1 KV head (was 1, unchanged) |
| SDPA program config | n_local_kv=1 (unchanged), n_local_q=3 (was 6) — adjust q_chunk_size if needed |
| Drop env gates | `QWEN36_FULLATTN_WO_TUNED`, `QWEN36_FULLATTN_WO_SHARDED`, `QWEN36_CCL_NUM_LINKS_FA` — these were V2-CCL A/B levers, no longer needed |

### 4. `tt/llama_ccl.py`

Recompute every persistent buffer that's keyed by QKV / WO_AG / WO width:

| key | old shard width per chip | new shard width per chip |
|---|---|---|
| `QKV` (after AR on cluster_axis=1) | 2048 | 1280 (per-chip QKV after head sharding on rows) |
| `QKV_BF16` | 2048 | 1280 |
| `WO_AG` | (per-chip WO input width) | 768 (was 1536) |
| `WO_AG_BF16` | | 768 |
| `WO` | | 768 |
| `FF1`, `FF3` | unchanged | unchanged (MLP path) |

### 5. KV cache + paged_update_cache + paged_SDPA

- Cache shape: `[max_blocks, 1, block_size, head_dim]` per chip (was `[max_blocks, 1, block_size, head_dim]` — unchanged per-chip; total across mesh now 8 KV heads vs 4)
- `paged_update_cache`: writes 1 KV head per chip (unchanged op signature)
- `paged_scaled_dot_product_attention_decode`: per-chip Q is now 3 heads (was 6); per-chip KV is 1 (unchanged). GQA inside SDPA expands KV 1 → 3 (was 1 → 6).
- Verify SDPA program config: `q_chunk_size`, `k_chunk_size`, `compute_with_storage_grid_size` — may need re-tuning for 3 heads instead of 6.

## Test sequence (PCC > 0.99 at every gate)

1. **Smoke load**: instantiate `TtTransformer` with 1 full-attn layer + all weights loaded. Verify per-chip DRAM footprint matches new sharding (no replication regressions).
2. **Full-attn block PCC**: `tests/test_decoder_layer.py -k full_attention` (a layer where `linear_attention_pattern[i] == "full_attention"`). PCC > 0.99 both prefill + decode vs HF reference.
3. **Paged attention PCC**: `tests/test_paged_attention.py` — all sub-tests. Confirms cache layout change is correct.
4. **4L full-model PCC**: prefill AND decode > 0.99 vs HF.
5. **64L full-model PCC** (V2-PAD-10, user-explicit): real prompt ("The capital of France is"), 1-step decode. Both pre-LM-head hidden state AND post-LM-head logits PCC > 0.99 vs HF. **This is the gate that was missing.**
6. **Trace parity** (V2-PAD-11): eager-vs-traced PCC > 0.9999.
7. **Tracy 1L decode**: confirm WO RS unchanged (already on cluster_axis=0); check QKV create_heads dropped from 4-way ring per-call latency to 8-way.

## Expected perf win

The 4-way → 8-way switch happens on:
- QKV create_heads (line 911) — RS cost halves
- AG-concat between SDPA and WO (line 1034) — AG cost halves
- QKV input AR (line 1120) if it flips — AR cost halves (less certain)

Aggregate: ~2-4 ms saved per full-attn layer in decode. 16 full-attn layers ×
trace replay: ~30-60 ms/step saving → ~3-5 tok/s/user added on the current
baseline. WO RS unchanged (already on 8-way).

## Risk

- KV replication breaks GQA if `repeat_interleave(2, dim=0)` doesn't match the
  HF GQA grouping. **Mitigation**: write a stand-alone PyTorch test that takes
  the un-replicated K/V + un-padded Q, runs HF GQA attention, then takes the
  REPLICATED K/V + same Q, runs attention with n_kv=8 grouping, and verifies
  bit-identical output. Run this BEFORE any device work.
- Paged cache stride doubles in the head dim → page_table layout must remain
  per-block (block_size stays 64; just 8 KV heads instead of 4 inside each block).
- The 763-line is_qwen36 branch in llama_attention.py will grow ~50 lines.
