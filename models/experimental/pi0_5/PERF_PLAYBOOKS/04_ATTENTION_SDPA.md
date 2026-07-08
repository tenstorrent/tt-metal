# 04 · Attention — SDPA vs Manual, Chunking, Softmax

The attention core is `scores = Q·Kᵀ → scale → softmax → context = P·V`. There are two
ways to run it on TT hardware: a **fused SDPA kernel** (flash-attention style) or **manual
BMMs + softmax**. This file covers both, plus the chunk sizing, score dtype, and softmax
tuning that mattered across BGE-M3, ViT, and Swin-L.

---

## 1. SDPA vs manual attention — when to use which

| | Manual (BMM + softmax) | SDPA (fused) |
|---|---|---|
| Ops | 4+ (`Q·Kᵀ`, scale, softmax, `P·V`) | 1 fused kernel |
| Memory | stores full `[S, S]` score matrix | flash-attention chunking, O(S) |
| Best for | short seq where `S×S` fits L1 (ViT 224, BGE-M3 S512) | long seq where `S×S` would OOM (high-res ViT, large batch) |
| Precision | per-op configurable | one compute config for the whole kernel |

**Decision rule:**
- Short sequence, score matrix fits L1 → **manual height-sharded BMMs** can be faster
  (ViT 224 uses manual; full control over each BMM's sharding).
- Long sequence or large batch where `[B, heads, S, S]` blows L1 → **SDPA** (mandatory;
  flash chunking is the only way it fits).
- Swin-L windowed attention: manual per-window BMM + softmax (SDPA's flash chunking
  dropped PCC to 0.81 on the small 64-token windows — rejected).

---

## 2. Manual attention — height-sharded BMMs (ViT 224)

Each `(batch, head)` pair is an independent matmul; height-shard over `B·heads·S`:

```python
# Q·Kᵀ  →  [B, heads, S, S]
scores = ttnn.matmul(q, k_t,
    memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG, dtype=ttnn.bfloat8_b,
    program_config=ttnn.MatmulMultiCoreReuseProgramConfig(   # BMM "reuse", no mcast
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=head_size_t, out_subblock_h=1, out_subblock_w=seqL_t,
        per_core_M=(heads·seqL_t)//gx, per_core_N=seqL_t))

# scale + softmax (ViT: no mask, so separate ops)
scores = ttnn.mul_(scores, 1.0 / head_size**0.5)
probs  = ttnn.softmax_in_place(scores, program_config=softmax_sharded_config)

# P·V  →  [B, heads, S, head_dim]
context = ttnn.matmul(probs, v,
    memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG, dtype=ttnn.bfloat8_b,
    program_config=ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=seqL_t, out_subblock_h=1, out_subblock_w=head_size_t,
        per_core_M=(heads·seqL_t)//gx, per_core_N=head_size_t))
```

Key points:
- Use `MatmulMultiCoreReuseProgramConfig` (BMM reuse, **no** mcast) — both inputs and
  output height-sharded, one core per head.
- Keep Q/K/V/scores/probs in `L1_HEIGHT_SHARDED` so no reshards between the BMMs and softmax.
- **scale + softmax fusion**: `ttnn.transformer.attention_softmax_` fuses them but
  *requires a mask*. ViT has no mask → uses separate `mul_` + `softmax_in_place`. Masked
  models (causal LLM, padded encoder) should use the fused `attention_softmax_`.

### Softmax program config (sharded)
```python
ttnn.SoftmaxShardedMultiCoreProgramConfig(
    compute_with_storage_grid_size=(gx, gy),
    subblock_w=seqL_t, block_h=(heads·seqL_t)//gx, block_w=seqL_t,
)
```

---

## 3. SDPA — the fused flash kernel

```python
context = ttnn.transformer.scaled_dot_product_attention(
    q, k, v,
    attn_mask=mask,              # additive, DRAM-only (hard-asserted); None if unpadded
    is_causal=False,             # True for decoder
    program_config=ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        q_chunk_size=...,        # KEY KNOB
        k_chunk_size=...,        # KEY KNOB
        exp_approx_mode=False,   # exact softmax (see §6)
        # max_cores_per_head_batch=16,  # flat across 2..16 in practice
    ),
    compute_kernel_config=sdpa_compute,
    memory_config=...,
)
```

### Chunk sizing — the single most important SDPA knob
Smaller chunks = more chunk-pair iterations, smaller CBs. Larger chunks = fewer
iterations, bigger CBs. Empirical winners:

| Workload | q_chunk, k_chunk | Note |
|---|---|---|
| BGE-M3 batch 1, S512 | 128, 128 | baseline |
| BGE-M3 batch 32, S512 | **256, 512** | doubling chunks amortizes dispatch at large batch (−3.5 ms vs 128,256) |
| ViT high-res, S1024–4096 | 256, 256 | symmetric, balances memory vs launch overhead |

**Heuristic:** start `q=k=128`. If SDPA < 20% of device time, stop. If > 30%, try doubling
both chunks (large batch / long seq usually wants larger). If > 50%, you're likely
bandwidth-bound on Q/K/V — check for an unnecessary bf16 cast (§5).

---

## 3b. Flash-decode — SDPA for autoregressive generation

Decoder LLMs in decode mode (`seq_len=1`, parallel over batch) use a *different* SDPA op:

```python
attn = ttnn.transformer.scaled_dot_product_attention_decode(
    Q, K, V, cur_pos_tensor=cur_pos, is_causal=True)      # or paged_..._decode with page_table
```

- **Flash-decode** processes the whole (tiny) Q against chunked K/V — `q_chunk_size` is
  unused; only `k_chunk_size` matters. Parallelized over batch, then kv-head.
- When `heads*batch < cores`, multiple cores cooperate on one head (that's the point).
  `max_cores_per_head_batch=16` caps it — beyond 16 cores/head the inter-core NoC
  bandwidth bottlenecks.
- **`is_causal=True`** removes the need for an `attn_mask` (lower-triangular implied) —
  saves mask bandwidth. Use it for standard causal decode/prefill. Only pass `attn_mask`
  for non-causal cases (e.g. cross-attention in VLMs).
- **Current position as a tensor** (`cur_pos_tensor`), not a list — required for tracing
  (see 08 section 3).

Prefill uses the regular `scaled_dot_product_attention` with `is_causal=True` (no mask
needed for causal). See 08 section 4 for the full prefill/decode op table.

---

## 3c. `exp_approx_mode` depends on seqlen/chunk ratio

The earlier "exact is faster on BH" finding holds for *short* encoder sequences. For LLMs:
- **Short `seqlen/chunk_size`**: `exp_approx_mode=True` is fine and can be faster.
- **Long sequences (>16k)**: `exp_approx_mode=False` — the approximation error accumulates
  through flash chunk accumulation and tanks PCC. Llama uses `q=k_chunk=512`,
  `exp_approx_mode=False` for long context.

Rule: the longer the sequence (more chunks to accumulate over), the more you need exact exp.

---

## 4. DRAM staging for SDPA (long sequence)

For long sequences SDPA's internal flash buffers need L1 room. Stage Q/K/V in **DRAM**
before the call so L1 is free for the working chunks:

```python
q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
k = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)
v = ttnn.to_memory_config(v, ttnn.DRAM_MEMORY_CONFIG)
context = ttnn.transformer.scaled_dot_product_attention(q, k, v, ...)
```

ViT high-res: at 768² (S=2304, 16 heads) Q+K+V is ~14 MB — can't stay in L1 with SDPA
working memory. DRAM staging lets SDPA stream from full NoC bandwidth and use larger
(256) chunks. **Pair `nlp_create_qkv_heads(memory_config=DRAM)` with DRAM-staged SDPA.**

---

## 5. Score dtype — keep Q/K/V native (the −13.7 ms lesson)

The largest single batch-32 win in BGE-M3 was **removing** a Q/K/V → bf16 cast before
SDPA. Stock attention inserted `3 tensors × 24 layers = 72` typecasts/forward.

```python
def _attention_score_dtype(seq, batch):
    # Keep the model's native dtype (bf8b); do NOT cast to bf16
    return None
```

Result: typecast bucket 11.1 → 1.6 ms, SDPA itself faster on bf8b inputs (35.2 → 21.5 ms),
total **−27 ms wall**. **Audit your attention forward for any `ttnn.typecast` before SDPA
and remove it** unless PCC demands otherwise. Keep the mask in model dtype too.

---

## 6. Exact vs approximate softmax

`exp_approx_mode=True` uses a polynomial-LUT exp. Counter-intuitively, on Blackhole at
S=512 **exact softmax (`False`) is faster** than approximate for both BGE-M3 batch sizes
(+0.2–0.3 ms when approx is on). ViT high-res also uses `exp_approx_mode=False` for score
precision. **Default to False; only enable approx if your specific shape proves it wins.**

---

## 7. SDPA compute kernel

```python
sdpa_compute = ttnn.init_device_compute_kernel_config(
    device.arch(),
    math_fidelity=ttnn.MathFidelity.LoFi,    # bf8b inputs; HiFi2/HiFi4 for bf16/high-precision
    math_approx_mode=False,                   # exact softmax
    fp32_dest_acc_en=True,                    # softmax SUM needs fp32 precision
    packer_l1_acc=True,
)
```

- `fp32_dest_acc_en=True` for SDPA — the softmax sum loses precision in fp16 (PCC drops to
  ~0.938 at False). **This is the opposite of the matmul rule** — softmax accumulation,
  like normalization, needs fp32.
- bf8b inputs tolerate LoFi (PCC holds). bf16 or precision-critical paths (ViT high-res
  uses HiFi4) want higher fidelity.

---

## 8. The attention mask

- **Mask must be in DRAM** — `sdpa_device_operation.cpp` hard-asserts it. Moving it to L1
  just inserts a reshard that ships it back. Don't.
- **`attn_mask=None` fast-path**: if your inputs are unpadded, pass `None`. SDPA's
  compile-time `use_provided_mask=false` strips the mask read + add from the kernel
  entirely (BGE-M3: −237 µs wall at batch 1). The entire mask cost is *internal to SDPA*
  (the `add_block_inplace(scores, mask)` step), not external ops.
- **Mask dtype**: bf8b is enough; bf4b gave no further win (mask is not bandwidth-bound).

---

## 9. Windowed SDPA — rejected for short windows

`ttnn.transformer.windowed_scaled_dot_product_attention` mandates Q/K/V in DRAM. For
L1-resident short attention (BGE-M3 B1) that forces reshards (+158 µs) that outweigh the
SDPA saving (−110 µs) → net +48 µs. Swin-L windowed attention also rejected SDPA (flash
chunking dropped PCC to 0.81 on 64-token windows). **Windowed SDPA is for long-seq VLMs
with DRAM-resident Q/K/V, not short-window encoders.**

---

## 10. What didn't work (with data)

| Attempt | Result |
|---|---|
| SDPA for Swin-L 64-token windows | PCC 0.81 (flash chunk error on small windows) |
| `softmax_in_place` in Swin-L (bf8b TILE) | regressed — breaks pipelining for downstream `attn·V` |
| Q/K/V → bf16 before SDPA | +13.7 ms B32 |
| `exp_approx_mode=True` at S512 | +0.2–0.3 ms (exact is faster on BH) |
| `fp32_dest_acc_en=False` for SDPA | PCC 0.938 (fail) |
| mask in L1 | hard-asserted to DRAM; no-op + extra dispatch |
| bf4b mask | no win (not bandwidth-bound) |
| windowed SDPA (L1-resident Q/K/V) | +48 µs (forced reshards) |
| `max_cores_per_head_batch` sweep | flat 2..16 |

---

## 11. Quick reference

| Question | Answer |
|---|---|
| SDPA or manual? | manual (height-sharded BMM) for short seq fitting L1; SDPA for long seq / large batch |
| q_chunk, k_chunk | 128/128 start; 256/512 large batch; 256/256 high-res |
| Q/K/V dtype | native (bf8b); never cast to bf16 before SDPA |
| `exp_approx_mode` | False (exact is faster on BH) |
| `fp32_dest_acc_en` (SDPA) | **True** (softmax sum precision) — opposite of matmul |
| Math fidelity | LoFi (bf8b), HiFi2/HiFi4 (bf16/precision-critical) |
| Mask location | DRAM (hard-asserted) |
| Unpadded inputs | pass `attn_mask=None` → strips mask read/add (−240 µs) |
| Long-seq Q/K/V | DRAM-stage before SDPA; pair with `nlp_create_qkv_heads(DRAM)` |
| scale+softmax fusion | `attention_softmax_` (needs mask) else `mul_`+`softmax_in_place` |
