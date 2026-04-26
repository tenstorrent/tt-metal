# Bark Small L1 Memory Budget

## Per-Core L1 Budget (Wormhole N300)

- Total L1 per core: 1.5 MB
- Reserved for firmware/stack: ~256 KB
- Available for tensors: ~1.25 MB

## Per-Stage Architecture

| Parameter | Semantic | Coarse | Fine |
| :--- | :--- | :--- | :--- |
| Hidden size | 768 | 768 | 768 |
| Num heads | 12 | 12 | 12 |
| Head dim | 64 | 64 | 64 |
| Num layers | 12 | 12 | 12 |
| Attention | Causal | Causal | Non-causal |
| Output vocab | 10,048 | 10,048 | 1,024 |
| KV cache | Yes (DRAM) | Yes (DRAM) | No |

## Memory Config per Operation

| Operation | Weights | Input | Output | Notes |
| :--- | :--- | :--- | :--- | :--- |
| Embedding | DRAM | L1 | L1 | Small output, CPU-side lookup |
| LayerNorm | DRAM (small) | L1 | L1 | In-place friendly |
| QKV Linear | DRAM | L1 | L1 | Stream weights from DRAM |
| SDPA (prefill) | — | L1 | L1 | Chunked, 128-tile chunks |
| SDPA (decode) | — | DRAM | DRAM | Matmul-based, seq < 32 |
| MLP Up+GELU | DRAM | L1 | L1 | GELU_NEW decomposed on-device |
| MLP Down | DRAM | L1 | L1 | |
| LM Head | DRAM | L1 | DRAM | 768→vocab, large output |

## KV Cache Budget

KV cache is stored in DRAM to prevent L1 circular buffer overflow during long autoregressive sequences.

| Stage | Max seq | Per-layer KV size | Total (12 layers) |
| :--- | :--- | :--- | :--- |
| Semantic | 768 tokens | 2 × 768 × 768 × 2B = 2.25 MB | ~27 MB |
| Coarse | 1024 tokens | 2 × 1024 × 768 × 2B = 3 MB | ~36 MB |
| Fine | N/A | No KV cache (non-causal) | 0 |

## L1 Residency Strategy

1. **Weights always in DRAM** — too large for L1 across 12 layers
2. **LayerNorm params** — small enough for L1 but stored alongside weights in DRAM for simplicity
3. **Activations during compute** — streamed through L1
4. **KV cache** — DRAM only, grows linearly with sequence length
5. **Embedding tables** — DRAM, accessed via CPU-side `nn.Embedding`

## Known Constraints

- TTNN SDPA requires `chunk_size >= 32`, so decode mode (seq=1) falls back to explicit matmul
- QKV split done on host after projection due to tile layout constraints in `split_query_key_value_and_split_heads`
- Fine model codebook embeddings extracted on host as uint32 for dtype compatibility
- Transposed key tensor in decode path is deallocated immediately after matmul to free L1
