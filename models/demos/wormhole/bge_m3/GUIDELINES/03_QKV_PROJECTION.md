# 03 · QKV Projection and Head Split

The attention front-end is `Norm → QKV matmul → split into heads → SDPA`. This file
covers the QKV projection matmul tuning and the head-split strategies, combining BGE-M3
(BERT QKV, custom head-split kernels), ViT (fused QKV + transformer split helper), and
Swin-L (windowed QKV).

---

## 1. Always fuse QKV into one matmul

Compute Q, K, V as a **single** `[*, H] × [H, 3H]` matmul, not three separate ones.
Splitting into three matmuls adds two ops and underutilizes the grid. All three campaigns
fuse QKV:

```python
qkv = ttnn.linear(
    hidden,                       # [B, S, H]  (block-sharded or DRAM, per regime)
    qkv_weight,                   # [H, 3H]
    bias=qkv_bias,
    memory_config=...,            # match the downstream head-split consumer
    dtype=ttnn.bfloat8_b,
    program_config=qkv_program_config,
    compute_kernel_config=qkv_compute,
)
```

BGE-M3 tested "split QKV into Q, K, V separately" → regressed (extra matmuls). Don't.

---

## 2. Program config — the two layouts

### 2D-mcast block-sharded (ViT, BGE-M3 batch 32)
For prefill (large `M = B·S`), block-shard the activation and use 2D multicast:

```python
ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(gx, gy),
    in0_block_w=K_tiles // gx,      # inner-dim slice per core
    out_subblock_h=1,
    out_subblock_w=...,             # ≤ 8/out_subblock_h with fp32_dest=False
    per_core_M=M_tiles // gy,       # = seqL_t in ViT
    per_core_N=3 · (H_tiles // gx), # 3× because QKV is 3H wide
    transpose_mcast=False,          # ROW_MAJOR when grid_x ≥ grid_y
    fused_activation=None,
)
```

ViT 224-seq example (grid 12×10, H=768): `in0_block_w=2`, `out_subblock_w=2`,
`per_core_M=7`, `per_core_N=6`. BGE-M3 B32 (grid 11×10, H=1024): `in0_block_w=8`,
`out_subblock_w=3`, `per_core_M=52`, `per_core_N=9`.

**ROW_MAJOR vs COLUMN_MAJOR (`transpose_mcast`):**
| Condition | Choice |
|---|---|
| M ≫ N (tall) or grid_x ≥ grid_y | ROW_MAJOR (`transpose_mcast=False`) |
| N ≫ M (wide) or grid_y > grid_x | COL_MAJOR (`transpose_mcast=True`) |

### Auto-routing (BGE-M3 batch 1)
At small batch, `ttnn.linear` with `program_config=None` (auto) often matches or beats a
hand-tuned config because the default routing already uses the full grid. **Try auto
first at small batch**; only hand-tune if Tracy shows the QKV matmul is a real bucket.

---

## 3. Compute kernel — LoFi unlocks the matmul

```python
qkv_compute = ttnn.init_device_compute_kernel_config(
    device.arch(),
    math_fidelity=ttnn.MathFidelity.LoFi,    # bf8b → LoFi
    math_approx_mode=False,
    fp32_dest_acc_en=False,                   # unlocks subblock cap to h·w ≤ 8
    packer_l1_acc=True,
)
```

BGE-M3 B32 QKV: HiFi4→HiFi2 saved ~8.5 ms, HiFi2→LoFi another ~2.2 ms across 24 layers.
At batch 1 (bf16 path) HiFi2 was kept (LoFi gave no measurable win there).

`fp32_dest_acc_en=False` is what makes wider subblocks legal — re-sweep subblock after
flipping it (see 01_FOUNDATIONS §3).

---

## 4. The K-dimension determines `in0_block_w`

`in0_block_w` must divide `K_tiles`. For QKV, `K = H`:
- H=768 → K_tiles=24 → `in0_block_w ∈ {1,2,3,4,6,8,12,24}`, ViT uses 2.
- H=1024 → K_tiles=32 → `in0_block_w ∈ {1,2,4,8,16,32}`, BGE-M3 uses 8.

Larger `in0_block_w` = fewer K-chunk iterations but bigger in0 CB. Sweep within the
divisors, prefer the largest that fits L1.

---

## 5. Head split — three strategies, pick by regime

After QKV you must reshape `[B, S, 3H] → 3 × [B, heads, S, head_dim]`. Options:

### (a) `split_query_key_value_and_split_heads` (ViT 224, BGE-M3)
The transformer helper does split + head reshape + (optional) K-transpose in one op:
```python
q, k_t, v = ttnn.transformer.split_query_key_value_and_split_heads(
    qkv, num_heads=H, transpose_key=True,
    memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,  # matches the attention BMMs
)
```
- Output is **height-sharded** for the manual attention BMMs.
- `transpose_key=True` pre-transposes K for `Q·Kᵀ`.

### (b) `nlp_create_qkv_heads` (ViT high-res, SDPA path)
For the SDPA path, output Q/K/V **directly to DRAM** and let SDPA transpose K internally:
```python
q, k, v = ttnn.experimental.nlp_create_qkv_heads(
    qkv, num_heads=H, num_kv_heads=H,
    transpose_k_heads=False,            # SDPA transposes K itself
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```
Supports GQA via `num_kv_heads`. DRAM output frees L1 for SDPA's flash buffers (see 04).

### (c) Custom `generic_op` head-split (BGE-M3 — for under-utilized stock op)
Stock `nlp_create_qkv_heads` assigns **one work unit per (batch, seq_tile)** pair. At
small batch that leaves most cores idle (BGE-M3 B1: 16 of 110 cores). A custom
`generic_op` with a `head_groups` axis distributes `B × seq_tiles × head_groups` work
units across the grid:

```
work_unit → (block = wu / head_groups, group = wu % head_groups)
block → (s_tile, batch);  head_start = group · (num_heads / head_groups)
```

**`head_groups` sweet spot (BGE-M3):**
| Op | B1 winner | B32 winner | Why |
|---|---|---|---|
| QKV create-heads | 16 | 4 | larger groups saturate grid; tiny groups → dispatch overhead dominates |
| concat-heads | **4** | 4 | at groups=16 each unit is only 2 tiles → launch overhead wins |

Tracy: stock create_heads 15 µs/call → 4 µs/call (groups tuned); concat 7.1 → 2.5 µs.

**Decision:** use (a) for L1-resident manual attention, (b) for the SDPA/DRAM path,
(c) only if profiling shows the stock head op is grid-starved at your batch size.

---

## 5b. RoPE — fused rotary embeddings (LLMs)

Decoder LLMs apply rotary position embeddings to Q and K *after* the head split, *before*
attention. Use the fused op, never a manual slice+rotate+concat:

```python
q = ttnn.experimental.rotary_embedding_llama(q_pre, cos, sin, trans_mat, is_decode_mode=False)
k = ttnn.experimental.rotary_embedding_llama(k_pre, cos, sin, trans_mat, is_decode_mode=False)
```

- Prefill (`is_decode_mode=False`): cos/sin computed once at init for the seq length;
  inputs interleaved L1.
- Decode (`is_decode_mode=True`): cos/sin must be regenerated each token (positions
  advance) via `RotarySetup` on-device; inputs height-sharded over batch.

Generate cos/sin on-device — don't push rotation matrices from host each iteration. Full
detail in 08 section 6.

---

## 5c. Decode head-split is a different op

The head-split op differs by phase (they are genuinely different kernels):

| Phase | Op | Output |
|---|---|---|
| Prefill | `nlp_create_qkv_heads` | `[1, n_heads, seq, head_dim]` |
| Decode | `nlp_create_qkv_heads_decode` | height-sharded over **batch** on `bsz` cores |
| Concat (prefill) | `nlp_concat_heads` | — |
| Concat (decode) | `nlp_concat_heads_decode` | — |

Decode parallelizes over batch (one token/user), so its head-split height-shards across
the batch dimension. See 08 section 4.

---

## 6. Resharding between QKV and attention (ViT pattern)

ViT runs the fused QKV on the **fixed full grid** (10×12) but the attention BMMs on a
**batch-dependent grid**. It reshards QKV between them:

```python
qkv = ttnn.reshard(qkv, block_sharded_config_variable_cores)  # full grid → batch grid
q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(qkv, ...)
```

This is deliberate: the projection wants max grid utilization; the per-head BMMs want a
grid shaped by `(batch × heads)`. The reshard is part of the "attention contract," not
waste. For high-res, ViT also DRAM-stages Q/K/V (see 04 §DRAM staging).

---

## 7. Memory-layout cheat sheet

| Regime | QKV output | Head-split output |
|---|---|---|
| L1-resident, manual attention (ViT 224, BGE-M3 B1) | block-sharded L1 | height-sharded L1 |
| Large batch, manual attention (BGE-M3 B32) | DRAM bf8b | DRAM, then per-head |
| SDPA path (ViT high-res) | block-sharded L1 → interleaved L1 | DRAM (frees L1 for SDPA) |

---

## 8. `minimal_matmul` for QKV — usually NOT a win

`ttnn.experimental.minimal_matmul` (streaming-K) is designed for very tall activations
(M in the thousands–tens-of-thousands, DiT-scale). For QKV at typical encoder shapes it
was tested and **lost**:
- BGE-M3 QKV: minimal_matmul ~510 µs vs ~424 µs for tuned `ttnn.linear`.
- Swin-L QKV (M=2560, K=768, N=2304): full 2520-config minimal_matmul sweep best 109 µs vs
  tuned `ttnn.linear` 88 µs — a **25% loss**.

**Use `minimal_matmul` only when 2D-mcast `ttnn.linear` crashes the L1 CB budget** (MLP at
large batch — see 05), not as a default QKV op.

---

## 9. What didn't work (with data)

| Attempt | Result |
|---|---|
| Split QKV into 3 separate matmuls | regressed (extra ops) |
| Q/K/V cast to bf16 before SDPA | +13.7 ms B32 — keep native bf8b |
| QKV DRAM-sharded weights (prefill) | OOM: activation > 8 banks × bank size |
| QKV `minimal_matmul` (encoder shapes) | slower than tuned `ttnn.linear` |
| QKV `out_subblock=2×3 out_block_h=26` (standalone winner) | in-model CB clash |
| `head_groups=16` for batch-32 concat | regressed vs `head_groups=4` |
| Sealed-wrapper custom head op integration | L1 clash (persistent L1 outputs) |

---

## 10. Quick reference

| Question | Answer |
|---|---|
| Fuse QKV? | Always — one `[H, 3H]` matmul |
| Program config (prefill) | 2D-mcast block-sharded; ROW_MAJOR if grid_x ≥ grid_y |
| Program config (small batch) | try `None` (auto) first |
| Compute | LoFi + fp32_dest=False (bf8b); HiFi2 (bf16) |
| `in0_block_w` | largest divisor of K_tiles that fits L1 |
| Head split (L1 manual attn) | `split_query_key_value_and_split_heads` → height-sharded |
| Head split (SDPA) | `nlp_create_qkv_heads` → DRAM, `transpose_k_heads=False` |
| Head split (grid-starved small batch) | custom `generic_op`, tune `head_groups` |
| `minimal_matmul` for QKV | no — only when `ttnn.linear` clashes L1 |
| Q/K/V dtype into attention | native (bf8b), never cast to bf16 |
