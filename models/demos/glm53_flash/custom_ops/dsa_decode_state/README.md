# `dsa_decode_state` — fused DSA decode state maintenance (model-local `ttnn.generic_op`)

One program for the state side of a GLM-5.3-Flash DSA decode step, from the decode stem matmul output
(`[q_a | kv_a | indexer k | pool gate | ...]`): the q_a RMSNorm (returned as `q_resid`), the latent append, the
indexer ring update and the pooled key of the current pool — all four caches read-modify-written **in place**.
Kernels: `kernels/{reader,writer,compute}_dsa_decode_state.cpp`; launcher `op.py::DsaDecodeState`; torch reference
`DsaDecodeState.golden`; test `tests/custom_ops/test_dsa_decode_state.py`. Used by `tt/dsa.py` (`GLM53_DSA_FUSED=1`).

## Contract (interleaved DRAM/L1; D = 128; Lq, L multiples of 128; pool size P <= 32)

| tensor | shape | dtype / layout | role |
|---|---|---|---|
| `stem` | `[1, 1, B, Ws]` | bf16 TILE | decode stem matmul output; column blocks at the constructor's tile-aligned offsets |
| `latent_cache` | `[n_blocks, 1, 32, L]` | bf16 TILE | paged latent cache, **in place**: row `a % 32` of page `latent_page_table[b, a // 32]` |
| `latent_page_table` | `[B, blocks_per_seq]` | int32 ROW_MAJOR | the latent cache's page table (any unique mapping) |
| `positions` | `[B]` | int32 ROW_MAJOR | position `a` of each user's token (ring slot `a % P`, pool `a // P`) |
| `ring_k`, `ring_g` | `[>= B, 1, 32, D]` | bf16 TILE | per-user rings (block = user), **in place** at slot `a % P` |
| `pool_cache` | `[B * pool_blocks, 1, 32, D]` | bf16 TILE | pooled keys (identity page table), **in place** at row `(a//P) % 32` of block `b*pool_blocks + (a//P)//32` |
| `qa_norm_w` / `kva_norm_w` / `k_norm_w` / `k_norm_b` | `[1,1,1,Lq]` / `[1,1,1,L]` / `[1,1,1,D]` / `[1,1,1,D]` | fp32 TILE | norm weights (row 0) |
| `ape` | `[1, 1, 32, D]` | fp32 TILE | rows < P: the in-pool position embedding; rows >= P: `-1e9` (masks the unused ring rows) |
| `selectors` | `[33*32, 32]` | bf16 TILE | `kda_decode.op.selector_tiles()` |
| `q_resid` | `[1, 1, B, Lq]` | bf16 TILE | **pre-allocated output**: `rmsnorm(stem[:, qa]) * w` |

Math per user: `latent = rmsnorm(kv_a) * w_kva`; `k = layernorm(k_raw) * w_k + b_k`; `g = raw gate`;
`pooled = sum_j softmax_j(g_ring[j] + ape[j]) k_ring[j]` over the P ring rows (the softmax is shifted by the first
in-pool row: shift-invariant, exact to fp32 rounding; in-pool logit spreads beyond ~80 would overflow).

Constructor: `DsaDecodeState(q_lora_rank, latent_dim, index_dim, pool_size, pool_blocks, qa_offset, kva_offset,
k_offset, g_offset, eps_rms=1e-5, eps_ln=1e-6)`; call with the tensors above in that order, `q_resid` last.

## Program

* Cores `0 .. Bt-1` (one per 32-user tile row) run role A: the q_a block RMSNorm (sum of squares in 4-tile chunks
  through a running-sum CB, `* rsqrt(mean + eps) * w`), packed bf16 tiles. Cores `Bt .. Bt+B-1` run role B, one
  per user: the reader reads the index vectors and the page-table entry (one 64 B aligned chunk), pushes the three
  selector tiles of the target rows (`a % 32`, `a % P`, `(a//P) % 32`) and the user's stem blocks, constants and
  cache tiles; compute normalises the whole kv_a / k blocks, lifts row `b % 32` (selector matmul), builds `E_s^T`
  and the keep masks `1 - e_s`, updates each cache tile as `old * keep + E_s^T row`, computes the pooled key from the
  updated fp32 ring (column sums / row sums by ones-tile matmuls) and places it; the writer writes the page, ring
  and pool tiles back and the role-A q_resid tiles. Needs `B + ceil(B/32)` cores.
* Compile-time args — compute: `Lq_t, L_t, D_t, eps_rms, eps_ln, 1/Lq, 1/L, 1/D`; reader: `Lq_t, L_t, D_t, Ws_t,
  qa_t, kva_t, k_t, g_t, P, pool_blocks, Bt` + `TensorAccessorArgs` of the 13 input tensors in call order;
  writer: `Lq_t, L_t, D_t` + accessors of q_resid, latent_cache, ring_k, ring_g, pool_cache.
* Runtime args per core — reader: `role, idx, <13 buffer addresses>, lpt_page_bytes, pos_page_bytes`; writer:
  `role, idx, q_resid, latent_cache, ring_k, ring_g, pool_cache addresses`; compute: `role, idx`. Addresses are
  baked: under trace the io tensors must be persistent or allocated inside the trace.
* Mesh: SPMD; the caches and the stem are the device's local copies.

Measured (chip 3, layer-3 shapes): 48 us at B=4, 75 us at B=32 (eager).
