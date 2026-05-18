# V2-2D: full 2D-sharding refactor + KV pad to match llama70b's 8-way ring

## Goal (option C — full refactor in one atomic change)

Move qwen3.6 full-attention from its current **1D col-only sharding (4-way ring
WO reduce)** to llama70b's **2D-sharding (8-way ring WO reduce)**. This requires
KV head padding from 4 → 8 so the head split divides cleanly across the 8 mesh
rows. After the refactor:

- Per-chip: 3 Q heads + 1 K head + 1 V head (was 6 Q + 1 K + 1 V)
- WO reduce_scatter on `cluster_axis=0` (8-way ring) using width-sharded L1
  persistent buffer (matches DeltaNet pattern + llama70b's WO RS)
- Math identical to v1 via GQA-preserving K/V replication

## Why this is the actual "match llama70b"

| dimension | v2 today (1D col-sharded) | llama70b (2D-sharded) | v2 after refactor |
|---|---|---|---|
| QKV weight ShardTensor2dMesh | `dims=(None, 3)` | `dims=(2, 3)` for 2D | `dims=(2, 3)` |
| WO weight ShardTensor2dMesh | `dims=(None, 2)` | `dims=(2, 3)` | `dims=(2, 3)` |
| Per-chip Q heads | 6 | 8 (64/8) | 3 (24/8) |
| Per-chip KV heads | 1 (4/4) | 1 (8/8) | 1 (8 padded / 8) |
| WO reduce cluster_axis | 1 (4-way ring) | 0 (8-way ring) | 0 (8-way ring) |
| WO reduce per-call cost | ~181 µs | ~48 µs (DeltaNet matches) | target ~48 µs |

The previous V2-CCL "mirror llama70b WO RS" applied the cluster_axis=0 flip to
the *non-qwen36* code path in `llama_attention.py` (line 1054), which the qwen36
forward never enters. The actual qwen36 forward (`_forward_decode_qwen36` at
1608, `_forward_prefill_qwen36` at 1454) used `cluster_axis=1` for WO. This
refactor fixes that.

## GQA-preserving KV replication

Original mapping: `q_i` attends to `k_{i // 6}`, `v_{i // 6}` (group size 6
because 24 Q / 4 KV).

After replicating each KV head twice with `repeat_interleave(2, dim=0)`:
`[k0, k1, k2, k3] → [k0, k0, k1, k1, k2, k2, k3, k3]`. Group size becomes 3
(24 Q / 8 padded KV). For any Q index `q_i`:
- Original KV index: `q_i // 6`
- New KV index: `q_i // 3`
- Replication structure: `k_padded[2j] == k_padded[2j+1] == k_original[j]`
- So `k_padded[q_i // 3] == k_original[(q_i // 3) // 2] == k_original[q_i // 6]` ✓

Math is bit-identical. **Verify this in CPU before any device work (V2-2D-1).**

## Files & precise changes

### File 1 — `tt/qwen36_model_config.py`

```python
# Was:
self.n_kv_heads = 4
self.n_local_heads = self.n_q_heads // self.cluster_shape[1]   # 24 / 4 = 6
# ... cluster_shape[1] in min_kv_prefill_shard_seqlen, assert, etc.

# Becomes:
self.n_kv_heads_unpadded = 4
self.n_kv_heads = 8                                              # padded for divisibility
self.n_local_heads = self.n_q_heads // self.cluster_shape[0]    # 24 / 8 = 3
self.n_local_kv_heads = self.n_kv_heads // self.cluster_shape[0]  # 8 / 8 = 1
assert self.n_kv_heads % self.cluster_shape[0] == 0
# min_kv_prefill_shard_seqlen formula axis flip 1 → 0
# QKV/WO program configs recomputed:
#   QKV per-chip width: (n_q + 2*n_kv) * hd / rows = (24 + 16) * 256 / 8 = 1280
#   WO  per-chip width: n_q * hd / rows = 24 * 256 / 8 = 768
```

Audit every `cluster_shape[1]` reference in this file. Norm/MLP/LM-head configs
that don't depend on head sharding stay on `[1]`.

### File 2 — `tt/llama_attention.py` weight loader (~lines 360-450)

```python
# K replication (preserves GQA grouping):
wk_native = state_dict[k_proj_str]                       # [4*hd, H]
wk_padded = wk_native.view(n_kv_unpadded, hd, H) \
                    .repeat_interleave(2, dim=0) \
                    .reshape(n_kv_padded * hd, H)        # [8*hd, H]
# Same for wv_native and k_norm.weight (length 4 → 8)

# WQKVG cat now uses padded K/V:
col_blocks.append(torch.cat([
    wq_native[qs:qe],
    wgate_native[qs:qe],
    wk_padded[ks_padded:ke_padded],
    wv_padded[ks_padded:ke_padded],
], dim=0))

# WQKVG ShardTensor2dMesh: 2D shard now
mesh_mapper=ttnn.ShardTensor2dMesh(
    self.mesh_device, dims=(2, 3),                       # was (None, 3)
    mesh_shape=configuration.cluster_shape
)

# WO ShardTensor2dMesh: 2D shard now (matches llama70b llama_attention.py:202)
mesh_mapper=ttnn.ShardTensor2dMesh(
    self.mesh_device, dims=(2, 3),                       # was (None, 2)
    mesh_shape=configuration.cluster_shape
)
```

Rename `n_q_per_col`, `n_kv_per_col`, `q_dim_per_col`, etc. → `_per_chip`. Old
values were 6/1/1536; new are 3/1/768.

### File 3 — `tt/llama_attention.py` qwen36 forward (~1454-2017)

`_forward_prefill_qwen36`:
- QK-norm operates on per-chip head counts: 3 Q heads, 1 KV head
- KV cache fill: `paged_fill_cache` on 1 KV head per chip (unchanged signature)
- GQA expand: `gqa_pc = n_q_pc // n_kv_pc = 3 // 1 = 3` (was 6)
- SDPA: same call, GQA in tensor shapes
- **WO output reduce**: replace
  ```python
  gathered = ttnn.all_gather(dense_partial, dim=0, cluster_axis=1, ...)
  dense_out = ttnn.experimental.fast_reduce_nc(gathered, dims=[0], ...)
  ```
  with
  ```python
  dense_out = self.tt_ccl.line_all_reduce(
      dense_partial, cluster_axis=0,
      num_links=self.model_config["GALAXY_NUM_LINKS"],
      memory_config=sharded_memcfg_axis0,
      use_optimal_ccl_for_llama=True,
      use_qwen36_residual_buffer=True,
  )
  ```

`_forward_decode_qwen36`:
- Same `_per_chip` rename + GQA group=3
- `paged_scaled_dot_product_attention_decode` accepts the new shapes natively
- **Delete entire env-gated block (lines 1873-2001)** and replace with single path:
  ```python
  sharded_memcfg = self.tt_ccl.qwen36_residual_output_memcfgs[0]   # axis-0 now
  dense_partial = ttnn.linear(gated, self.wo, dtype=ttnn.bfloat16,
                              memory_config=sharded_memcfg,
                              compute_kernel_config=self.compute_kernel_config_hifi4)
  gated.deallocate(True)
  dense_out_sharded = self.tt_ccl.line_all_reduce(
      dense_partial, cluster_axis=0,
      num_links=self.model_config["GALAXY_NUM_LINKS"],
      memory_config=sharded_memcfg,
      use_optimal_ccl_for_llama=True,
      use_qwen36_residual_buffer=True,
  )
  dense_partial.deallocate(True)
  dense_out_full = ttnn.to_memory_config(dense_out_sharded, ttnn.DRAM_MEMORY_CONFIG)
  dense_out_sharded.deallocate(True)
  ```
  Drop `QWEN36_FULLATTN_LAR`, `QWEN36_FULLATTN_WO_TUNED`,
  `QWEN36_FULLATTN_WO_SHARDED`, `QWEN36_CCL_NUM_LINKS_FA` env flags. They were
  A/B levers for the slower 4-way ring path; the new default IS the optimal path.

### File 4 — `tt/llama_ccl.py`

`_build_qwen36_residual_buffers` (or equivalent): persistent buffers indexed by
cluster_axis. The current code likely fills both indices [0] and [1]; ensure the
[0]-axis buffer has the per-chip width matching the new 2D-sharded WO output
(per-chip 768 width).

```python
# Old per-chip widths (cluster_axis=1, 4-way ring):
#   QKV: 6*256 + 1*256 + 1*256 = 2048
#   WO:  6*256 = 1536
# New per-chip widths (cluster_axis=0, 8-way ring):
#   QKV: 3*256 + 1*256 + 1*256 = 1280
#   WO:  3*256 = 768
```

Audit all qwen36 buffer key entries — both bf16 and bf8b dual-keys — and
recompute shard dims for the 2D pattern.

### File 5 — KV cache (in `llama_attention.py` init around layer_past setup)

```python
# Was: [max_batch, n_kv_unpadded // cols=4, ...] = [B, 1, S, hd] per chip
# Now: [max_batch, n_kv_padded   // rows=8, ...] = [B, 1, S, hd] per chip
#   (same per-chip footprint; mesh total now 8 distinct KV slots due to padding)
```

paged_update_cache / paged_fill_cache / paged_SDPA: call signatures unchanged
(they take cache + 1-head-per-chip writes). Validate via `test_paged_attention.py`.

### File 6 — `tests/test_full_model.py` (V2-2D-9)

Add 64L decode-mode case:
```python
@pytest.mark.parametrize("n_layers", [64])
def test_full_model_64layer_decode_real_prompt(...):
    # Real prompt: "The capital of France is"
    prompt = "The capital of France is"
    # Prefill phase: get HF reference KV cache + last-token logits
    ref_kv, ref_hidden, ref_logits = hf_model.prefill(prompt)
    # Decode 1 token, capture pre-LM-head hidden state AND post-LM-head logits
    ttnn_hidden, ttnn_logits = ttnn_model.decode_one_step(prompt, capture_hidden=True)
    assert pcc(ttnn_hidden, ref_hidden) > 0.99
    assert pcc(ttnn_logits, ref_logits) > 0.99
```

This gate has historically been skipped; it's the user-explicit hardening
required regardless of the refactor.

## Validation sequence (each gate must pass before next)

1. **V2-2D-1** — CPU GQA math sanity check (pre-flight, 5 min)
2. **V2-2D-2..6** — file edits (no device runs yet)
3. **V2-2D-7** — full-attention block PCC test + paged_attention PCC test
4. **V2-2D-8** — 4L full-model PCC (prefill + decode)
5. **V2-2D-9** — 64L PCC, real prompt, pre-LM and post-LM heads
6. **V2-2D-10** — trace parity (eager-vs-traced PCC > 0.9999) + tracy
   confirming WO RS dropped to ~48 µs

If any PCC gate fails, **stop** and debug before proceeding. Do not env-gate
the new path "until we figure it out" — that's how V2-CCL-followup ended up with
the optimal code behind a flag for 2 weeks.

## Risks

1. **KV cache stride changes** — 8 KV heads instead of 4 in the mesh-total cache.
   Paged blocks reshape. **Mitigation**: V2-2D-7 paged_attention test catches this.
2. **SDPA program config sized for 6 Q heads** — current decode SDPA uses
   `compute_with_storage_grid_size=(1,1)`, which is single-core. With 3 Q per
   chip instead of 6, the grid size doesn't change but `q_chunk_size`/`k_chunk_size`
   may benefit from re-tuning. **Mitigation**: V2-2D-8 4L PCC + V2-2D-10 tracy
   compare.
3. **2D-sharded weight tensor doesn't fit** — per-chip WQKVG is now (24+16)*256 /
   (8 rows × 4 cols) = 10240/32 = 320 columns per chip × 5120 hidden = ~1.6 MB
   per chip per layer (was 14336/4 = 3584 columns × 5120 = ~17 MB per chip per
   layer). DRAM usage DROPS; no fit issue.
4. **DeltaNet path untouched** — DeltaNet already uses cluster_axis=0 width-
   sharded path. This refactor only touches the full-attention path. 16 full-
   attn layers / 64 total; DeltaNet layers continue working.

## Expected gain

- WO RS per-call: 181 µs → ~48 µs (3.8× — matches DeltaNet)
- Decode step: 16 full-attn layers × (181 - 48) µs ≈ 2.1 ms saved per step (eager)
- With trace replay: full per-call gap recovered, ~2-4 ms / step → +3-5 tok/s/user

## Out of scope (explicitly NOT in this refactor)

- BH 130-core grid migration (V2-grid, separate task)
- lm_head MinimalMatmul tuning (V2-lm-head, separate task)
- Persistent DeltaNet state through trace (V2-generator, separate task)
- Any further tt-lang kernel work
