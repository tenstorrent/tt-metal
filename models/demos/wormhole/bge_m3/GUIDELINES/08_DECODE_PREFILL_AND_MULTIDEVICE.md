# 08 · Decode vs Prefill, KV-Cache, and Multi-Device

Encoder models (BGE-M3, ViT, Swin) run one forward over a full sequence. **Generative
LLMs** (Llama, DeepSeek) run two fundamentally different phases — **prefill** and
**decode** — with opposite bottlenecks. This file covers that split, the KV cache, GQA,
and multi-device fracturing, distilled from the TT-Transformers / Llama-70B / DeepSeek-V3
implementations.

---

## 1. The two phases — opposite bottlenecks

| | Prefill | Decode |
|---|---|---|
| Input shape | `(1, 1, seq_len, dim)`, `bsz=1` | `(1, 1, bsz, dim)`, `seq_len=1` |
| Parallelized over | sequence length | **batch** (one token/user) |
| Compute pattern | large matmuls, **compute-bound** | tiny activations, **DRAM-bandwidth-bound** (weights dominate) |
| Activation location | **DRAM** (seq can be 128k — won't fit L1) | **L1 width-sharded** (small, keep close to compute) |
| Matmul variant | **Matmul 2D** (interleaved DRAM in/out) | **DRAM-sharded matmul** (weights sharded across banks) |
| Trace + 2CQ | not used (compute-bound, host hidden) | **essential** (host-bound, tiny ops) |
| `q/k_chunk` in SDPA | 512 default; `exp_approx_mode=False` for long seq | flash-**decode** (whole Q, chunked K/V) |

**The single most important LLM insight:** prefill and decode need *different program
configs, memory configs, and matmul variants for the same weight.* Build both paths.

---

## 2. Matmul variant by regime — the decision

| Variant | Parallelizes | Activation / output | Weights | Use for |
|---|---|---|---|---|
| **Matmul 2D** (`MatmulMultiCoreReuseMultiCast`) | M and N | DRAM interleaved | DRAM interleaved or sharded | **all prefill matmuls** (M,N ≥ 256, compute-bound) |
| **DRAM-sharded** (`...MultiCastDRAMSharded`) | N | L1 width-sharded | **DRAM width-sharded across banks** | **all decode matmuls** (DRAM-bandwidth-bound) |
| **Matmul 1D** (`...MultiCast1D`) | N only | L1 width-sharded | DRAM interleaved | decode where DRAM-sharded doesn't apply |

- **Prefill = compute-bound** → 2D mcast, maximize `in0_block_w` and subblock, full grid.
- **Decode = DRAM-bound** → DRAM-sharded matmul reads weights at ~240 GB/s vs ~190 GB/s
  interleaved (Wormhole). Activation + output width-sharded in L1.

```python
# Decode DRAM-sharded (weights pre-sharded across 12 banks):
weights_memcfg = create_dram_sharded_mem_config(k=K, n=N)
pc = dram_matmul_config(m=M, k=K, n=N, num_cores=core_grid.num_cores)
out = ttnn.linear(act, weights, program_config=pc,
                  memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                  compute_kernel_config=compute)
```
**Caution:** DRAM-sharded matmul has **no padding** support — the core grid must evenly
divide both activation and output. (This is also why it OOMs for large-batch *prefill*:
the activation width-shard exceeds L1 — see 01 §6.)

---

## 3. The KV cache

Decode reads the whole history from a cache instead of recomputing it.

| Op | Phase | Purpose |
|---|---|---|
| `ttnn.fill_cache(K_cache, K, batch_id)` | prefill | populate cache for one batch |
| `ttnn.experimental.paged_fill_cache(K_cache, K, page_table, batch_idx)` | prefill | paged variant |
| `ttnn.experimental.paged_update_cache(keys, K, update_idxs=cur_pos, page_table=...)` | decode | append one token's K/V |

- **Paged KV cache** (page_table) is how continuous batching / vLLM integration works —
  non-contiguous physical pages mapped to logical positions.
- **Current position as a tensor, not a list.** `cur_pos_tensor` lives in device memory and
  can be updated with `ttnn.add` — this makes it **trace-compatible** (traced vars must be
  statically known at compile time; a Python list can't change between traced iterations).

---

## 4. Decode vs prefill attention ops — they are different ops

| Step | Prefill op | Decode op |
|---|---|---|
| head split | `nlp_create_qkv_heads` | `nlp_create_qkv_heads_decode` (height-sharded over batch) |
| SDPA | `scaled_dot_product_attention` (flash, `is_causal=True`) | `scaled_dot_product_attention_decode` / `paged_..._decode` (flash-decode) |
| concat | `nlp_concat_heads` | `nlp_concat_heads_decode` |

**Flash-decode** processes the entire (small) Q against chunked K/V, parallelized over
batch then kv-head. When `heads·batch < cores`, multiple cores cooperate on one head —
that's the point of flash-decode. `max_cores_per_head_batch=16` caps it (beyond 16 cores
per head the NoC bandwidth between cores bottlenecks).

`exp_approx_mode`: **True** for small `seqlen/chunk_size`, **False** for long sequences
(>16k) where the approximation error accumulates through chunk accumulation and tanks PCC.

---

## 5. Grouped-Query Attention (GQA)

Modern LLMs use fewer KV heads than Q heads (`n_kv_heads < n_q_heads`). The head-split ops
take `num_kv_heads` separately:

```python
Q, K, V = ttnn.experimental.nlp_create_qkv_heads(
    xqkv_fused, num_heads=n_q_heads, num_kv_heads=n_kv_heads, transpose_k_heads=False)
```

The fused QKV weight width is `(n_q_heads + 2·n_kv_heads)·head_dim`, not `3·n_q_heads·head_dim`.
SDPA handles the Q-group → KV-head broadcast internally. This shrinks the KV cache and the
K/V projections proportionally — a real memory + bandwidth win at decode.

---

## 6. RoPE — fused rotary position embeddings

LLMs apply rotary embeddings to Q and K before attention. Use the fused op, not a manual
rotate:

```python
q = ttnn.experimental.rotary_embedding_llama(q_pre, cos, sin, trans_mat, is_decode_mode=False)
k = ttnn.experimental.rotary_embedding_llama(k_pre, cos, sin, trans_mat, is_decode_mode=False)
```

| | Prefill (`is_decode_mode=False`) | Decode (`is_decode_mode=True`) |
|---|---|---|
| Input | `[1, n_heads, seq, head_dim]` interleaved L1 | `[1, batch, n_heads, head_dim]` height-sharded L1 |
| cos/sin | computed once at init for the seq length | **updated per token** via `RotarySetup` (positions change) |
| trans matrix | `[1,1,TH,TW]` interleaved | `[1,1,TH·batch,TW]` height-sharded |

Decode RoPE needs fresh cos/sin each iteration because each user's position advances —
generate them on-device via `RotarySetup.get_rot_mats(position_ids)`, don't push from host.

---

## 7. Multi-device weight fracturing

For models too big for one chip, weights are **fractured** (sharded) across devices:

| System | Scheme |
|---|---|
| n300 / T3000 (1D mesh) | 1D column-fractured weights |
| TG / Galaxy (2D mesh) | 2D fractured weights |

- **MLP**: w1/w3 fractured on the output (up-projection) dim, w2 on the input
  (down-projection) dim, so the cross-device reduction lands at the right place.
- After a fractured matmul you need a **CCL** (collective) op to recombine:
  - `all_gather` — concatenate shards across devices (e.g. after column-fractured proj).
  - `reduce_scatter` — sum partial results and re-shard (e.g. after row-fractured proj).
  - `all_reduce` — sum and replicate.
- **Fuse CCL with matmul** where possible: `ttnn.experimental.all_gather_matmul` overlaps
  the all-gather with the matmul — a real latency win on multi-device.

**CCL cost rules:**
- CCLs are expensive — minimize them. Use **bf8b** inputs to CCLs instead of bf16 to halve
  the bytes moved.
- For 2D weight sharding, verify the **reduction dimension** is set correctly — a wrong
  `cluster_axis` silently produces garbage (a documented accuracy-debug check).

---

## 8. Host-device communication minimization (decode is host-sensitive)

Decode runs hundreds of tiny ops; host overhead dominates without care:
- **Embeddings on-device** — token IDs are smaller than embeddings to push.
- **Return only the last token from prefill**, not all positions.
- **Sample on-device** (`ttnn.argmax`) — don't round-trip logits to host.
- **Generate masks / rotation matrices on-device** or reuse across iterations.
- **Never tilize/untilize on host.** `to_torch` untilizes on host by default; instead
  `ttnn.untilize(x, use_multicore=True)` on device, then `to_torch`.

These matter only in the host-bound (decode/small-batch) regime — see 06 §8 and 01 §8.

---

## 9. The DRAM weight-layout trick for norms

Norm weights (γ, β) pushed in TILE layout need padding to TILE_HEIGHT, wasting DRAM
bandwidth. Instead wrap them into TILE_WIDTH sticks in ROW_MAJOR — no padding, done once
at init:

```python
gamma = gamma.view(1, 1, embedding_dim // TILE_WIDTH, TILE_WIDTH)
ttnn_gamma_rm = ttnn.as_tensor(gamma, layout=ttnn.ROW_MAJOR_LAYOUT,
                               dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
```

Zero runtime overhead, smaller DRAM footprint, faster weight reads each forward.

---

## 10. Quick reference

| Question | Answer |
|---|---|
| Prefill matmul | Matmul 2D, DRAM interleaved, compute-bound, full grid |
| Decode matmul | DRAM-sharded, L1 width-sharded act/out, weights across banks |
| Decode activation | L1 width-sharded (smallest core grid keeping tile size) |
| Prefill activation | DRAM interleaved (won't fit L1) |
| KV cache update | `paged_update_cache` (decode), `paged_fill_cache` (prefill) |
| Current position | tensor (trace-compatible), not a Python list |
| Head split | `nlp_create_qkv_heads` (prefill) vs `..._decode` (decode) — different ops |
| SDPA | `scaled_dot_product_attention` vs `..._decode` (flash-decode) |
| GQA | pass `num_kv_heads`; QKV width = `(n_q+2·n_kv)·head_dim` |
| RoPE | `rotary_embedding_llama`, `is_decode_mode` flag; decode regenerates cos/sin |
| `exp_approx_mode` | True for short seq, False for >16k |
| Multi-device weights | 1D fracture (n300/T3K), 2D fracture (TG); recombine with CCL |
| CCL dtype | bf8b not bf16 (halve bytes) |
| CCL + matmul | fuse via `all_gather_matmul` |
| Norm weights | TILE_WIDTH sticks, ROW_MAJOR, no padding |
| On-device | embeddings, sampling, untilize, mask/RoPE generation |
