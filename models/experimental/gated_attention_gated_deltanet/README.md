<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gated Attention & Gated DeltaNet — TTNN Implementation

This module contains pure-functional PyTorch reference implementations and their TTNN (Tenstorrent) counterparts for two attention mechanisms used in the **Qwen3-Next** architecture:

- **Gated Attention** — Standard scaled dot-product attention with a query-dependent sigmoid gate applied to the output. The Q projection is 2x wide; the second half serves as the gate signal.
- **Gated DeltaNet** — A linear-attention layer built on the *delta rule* with gated exponential decay. It maintains a fixed-size recurrent state (`[B, H, K, V]`) instead of a growing KV cache, enabling O(1) memory per token at decode time.

Both layers originate from the Qwen3-Next model, which alternates Gated Attention layers (for long-range SDPA) with Gated DeltaNet layers (for efficient linear recurrence).

## Directory Structure

```
gated_attention_gated_deltanet/
├── torch_functional/                # Pure-torch functional references
│   ├── __init__.py
│   ├── gated_attention.py           # Gated Attention (SDPA + sigmoid gate)
│   ├── gated_deltanet.py            # Gated DeltaNet full layer
│   └── delta_rule_ops.py            # Core delta-rule algorithms (recurrent & chunked)
│
├── tt/                              # TTNN implementations for TT hardware
│   ├── __init__.py
│   ├── ttnn_gated_attention.py      # Gated Attention using TTNN ops
│   ├── ttnn_gated_deltanet.py       # Gated DeltaNet layer using TTNN ops
│   └── ttnn_delta_rule_ops.py       # Core delta-rule algorithms using TTNN ops
│
├── tests/                           # Validation and benchmarking
│   ├── __init__.py
│   ├── test_gated_attention.py      # Torch-only tests for Gated Attention
│   ├── test_gated_deltanet.py       # Torch-only tests for Gated DeltaNet
│   ├── test_ttnn_validation.py      # TTNN vs torch PCC validation + benchmarks
│   └── sweep_seq_len.py             # Sequence-length sweep: latency & PCC tables
│
└── README.md
```

## Environment Setup

All TTNN tests require a Tenstorrent Wormhole device. Set up the environment from the `tt-metal` root:

```bash
cd /path/to/tt-metal
source python_env/bin/activate
export ARCH_NAME=wormhole_b0
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
```

All test commands below assume this environment is active and are run from the `tt-metal` root directory.

## Test Commands

### Torch-Only Tests (no hardware required)

These validate the PyTorch reference implementations on CPU.

```bash
# Run all Gated Attention tests
python models/experimental/gated_attention_gated_deltanet/tests/test_gated_attention.py

# Run all Gated DeltaNet tests (delta-rule ops + full layer)
python models/experimental/gated_attention_gated_deltanet/tests/test_gated_deltanet.py
```

### TTNN Validation Tests (requires Tenstorrent hardware)

These compare TTNN outputs against torch golden references using PCC.

```bash
# Validate both Gated Attention and Gated DeltaNet
python models/experimental/gated_attention_gated_deltanet/tests/test_ttnn_validation.py

# Validate only Gated Attention
python models/experimental/gated_attention_gated_deltanet/tests/test_ttnn_validation.py --module attention

# Validate only Gated DeltaNet in recurrent mode (default T=16)
python models/experimental/gated_attention_gated_deltanet/tests/test_ttnn_validation.py --module deltanet

# Validate Gated DeltaNet with a specific sequence length
python models/experimental/gated_attention_gated_deltanet/tests/test_ttnn_validation.py --module deltanet --seq-len 32

# Validate Gated DeltaNet in chunked mode (for prefill)
python models/experimental/gated_attention_gated_deltanet/tests/test_ttnn_validation.py --module deltanet --mode chunk --seq-len 128 --chunk-size 64
```

### TTNN Benchmarks (requires Tenstorrent hardware)

Add `--bench` to run timing benchmarks after validation. Configurable warmup and iteration count.

```bash
# Validate + benchmark both modules
python models/experimental/gated_attention_gated_deltanet/tests/test_ttnn_validation.py --bench

# Benchmark Gated DeltaNet recurrent mode with custom iterations
python models/experimental/gated_attention_gated_deltanet/tests/test_ttnn_validation.py --module deltanet --bench --warmup 5 --iterations 20

# Benchmark Gated DeltaNet chunked mode at T=256
python models/experimental/gated_attention_gated_deltanet/tests/test_ttnn_validation.py --module deltanet --mode chunk --seq-len 256 --bench
```

### Sequence-Length Sweep (requires Tenstorrent hardware)

Produces a full latency/PCC table across sequence lengths from T=1 to T=32768.

```bash
# Sweep both modules (attention up to T=4096, deltanet up to T=32768)
python models/experimental/gated_attention_gated_deltanet/tests/sweep_seq_len.py

# Sweep only Gated Attention up to T=16384
python models/experimental/gated_attention_gated_deltanet/tests/sweep_seq_len.py --module attention --max-t 16384

# Sweep only Gated DeltaNet up to T=32768
python models/experimental/gated_attention_gated_deltanet/tests/sweep_seq_len.py --module deltanet --max-t 32768
```

## Performance Results

Results from Wormhole N300 (DP=1), comparing torch CPU vs TTNN.

**Column key:** T = sequence length, B = batch size, PCC = Pearson Correlation Coefficient (1.0 = perfect match).

### Gated Attention (H=8, H_kv=2, D=64)

- **H** — query heads, **H_kv** — key/value heads (GQA ratio = H/H_kv = 4), **D** — head dimension (hidden_size = H × D = 512)


| T     | B   | Torch (ms) | TTNN (ms) | Speedup | PCC    |
| ----- | --- | ---------- | --------- | ------- | ------ |
| 1     | 2   | 0.62       | 1.89      | 0.33x   | 0.9999 |
| 2     | 2   | 0.83       | 1.94      | 0.43x   | 0.9998 |
| 4     | 2   | 0.84       | 1.88      | 0.45x   | 0.9999 |
| 8     | 2   | 0.98       | 1.91      | 0.52x   | 0.9998 |
| 16    | 2   | 1.45       | 1.88      | 0.77x   | 0.9998 |
| 32    | 2   | 1.98       | 1.95      | 1.02x   | 0.9998 |
| 64    | 2   | 3.98       | 1.88      | 2.11x   | 0.9997 |
| 128   | 2   | 5.07       | 2.20      | 2.31x   | 0.9997 |
| 256   | 2   | 10.68      | 2.01      | 5.30x   | 0.9996 |
| 512   | 2   | 22.68      | 3.57      | 6.36x   | 0.9996 |
| 1024  | 2   | 55.70      | 6.92      | 8.05x   | 0.9996 |
| 2048  | 2   | 168.45     | 13.02     | 12.94x  | 0.9995 |
| 4096  | 2   | 562.96     | 26.58     | 21.18x  | 0.9994 |
| 8192  | 1   | 966.79     | 29.83     | 32.41x  | 0.9993 |
| 16384 | 1   | 3693.90    | 71.05     | 51.99x  | 0.9993 |
| 32768 | 1   | 22744.50   | 178.42    | 127.48x | 0.9991 |


TTNN uses `ttnn.transformer.scaled_dot_product_attention` (fused FlashAttention-2 kernel) with `SDPAProgramConfig` and `WormholeComputeKernelConfig(HiFi4, fp32_dest_acc_en=True)`. GQA is handled natively by the kernel. B=1 for T >= 8192 to avoid host OOM on the torch reference.

### Gated DeltaNet — Chunked Mode (B=2, H=4, K=128, V=256, chunk_size=64)

- **H** — attention heads, **K** — key/query head dim, **V** — value head dim, **chunk_size** — tokens processed in parallel per chunk
- Recurrent state per head is a K×V = 128×256 matrix (fixed size, independent of T)


| T     | Torch (ms) | TTNN (ms) | Speedup | PCC    |
| ----- | ---------- | --------- | ------- | ------ |
| 1     | 1.77       | 48.84     | 0.04x   | 0.9995 |
| 2     | 2.45       | 46.83     | 0.05x   | 0.9997 |
| 4     | 3.42       | 47.99     | 0.07x   | 0.9995 |
| 8     | 5.41       | 50.02     | 0.11x   | 0.9995 |
| 16    | 9.21       | 49.98     | 0.18x   | 0.9996 |
| 32    | 17.58      | 50.45     | 0.35x   | 0.9994 |
| 64    | 31.71      | 48.39     | 0.66x   | 0.9995 |
| 128   | 22.70      | 49.78     | 0.46x   | 0.9995 |
| 256   | 38.35      | 53.35     | 0.72x   | 0.9995 |
| 512   | 75.38      | 60.05     | 1.26x   | 0.9995 |
| 1024  | 151.23     | 40.18     | 3.76x   | 0.9998 |
| 2048  | 310.70     | 70.61     | 4.40x   | 0.9998 |
| 4096  | 686.50     | 134.79    | 5.09x   | 0.9998 |
| 8192  | 1614.37    | 303.10    | 5.33x   | 0.9998 |
| 16384 | 3299.19    | 676.65    | 4.88x   | 0.9998 |
| 32768 | 6931.67    | 1414.78   | 4.90x   | 0.9998 |


DeltaNet's O(T) complexity (via chunked processing) vs Gated Attention's O(T²) complexity.

## Op Mapping

Dispatch counts below are **device dispatches** (kernel launches or data-movement ops on TT hardware).
`ttnn.reshape` and `ttnn.unsqueeze` are metadata-only (0 dispatches) and shown for completeness.

### Gated Attention — Op Mapping (Validated, PCC ≥ 0.9992)


| PyTorch Op                                              | TTNN Op                                                       | Dispatches | Notes                                                                                                                                                                                                                                |
| ------------------------------------------------------- | ------------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `F.linear` (Q/K/V/O projections)                        | `ttnn.linear`                                                 | 4          | Q projection is 2× wide (query + gate)                                                                                                                                                                                               |
| `qg.view` + `torch.chunk` (split Q, gate)               | `ttnn.reshape` + `ttnn.chunk`                                 | 1          | reshape is metadata; chunk is 1 dispatch                                                                                                                                                                                             |
| `x.pow(2).mean().rsqrt()*(1+w)` (zero-centered RMSNorm) | `ttnn.multiply`×3 + `ttnn.mean` + `ttnn.add`×2 + `ttnn.rsqrt` | 14         | 7 ops × 2 (Q norm, K norm). Zero-centered variant: `x * rsqrt(mean(x²)+ε) * (1+weight)`                                                                                                                                              |
| `.transpose(1,2)` (Q/K/V, output)                       | `ttnn.transpose`                                              | 4          | Q,K,V to head-first `[B,H,T,D]`; output back to `[B,T,H*D]`                                                                                                                                                                          |
| `cos.unsqueeze` / `sin.unsqueeze`                       | `ttnn.unsqueeze`                                              | 0          | Metadata only (broadcast dims)                                                                                                                                                                                                       |
| `torch.cat((-x2, x1))` (RoPE `rotate_half`)             | `ttnn.neg` + `ttnn.concat`                                    | 4          | 2 ops × 2 (once per Q, once per K)                                                                                                                                                                                                   |
| `q*cos + rotate_half(q)*sin` (RoPE apply)               | `ttnn.multiply`×4 + `ttnn.add`×2                              | 6          | For both Q and K                                                                                                                                                                                                                     |
| `F.scaled_dot_product_attention`                        | `ttnn.transformer.scaled_dot_product_attention`               | 1          | Fused FlashAttention-2 kernel. Handles GQA (H_q ≠ H_kv) and causal masking internally. No `repeat_kv` needed. Uses `SDPAProgramConfig(q_chunk=64/256, k_chunk=64/256)` + `WormholeComputeKernelConfig(HiFi4, fp32_dest_acc_en=True)` |
| `torch.sigmoid` (output gate)                           | `ttnn.sigmoid`                                                | 1          | Direct 1:1                                                                                                                                                                                                                           |
| `output * sigmoid(gate)`                                | `ttnn.multiply`                                               | 1          | Direct 1:1                                                                                                                                                                                                                           |
| `.view` / `.reshape` (multi-head ×5)                    | `ttnn.reshape`                                                | 0          | Metadata only (no device dispatch)                                                                                                                                                                                                   |
| Precomputed cos/sin embeddings                          | Precomputed tensors                                           | 0          | Passed in, no `ttnn.cos`/`ttnn.sin` at runtime                                                                                                                                                                                       |
| **Total**                                               |                                                               | **~36**    | **Fused SDPA with native GQA + causal masking**                                                                                                                                                                                      |


### DeltaNet Recurrent — Op Mapping (Validated, PCC ≥ 0.9980)

Total dispatches: **~64 + 8T** (T = sequence length). For decode (T=1): **~72 dispatches**.


| PyTorch Op                                             | TTNN Op                                                                   | Dispatches   | Notes                                                                                                                                                                                                 |
| ------------------------------------------------------ | ------------------------------------------------------------------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **— Layer Ops —**                                      |                                                                           |              |                                                                                                                                                                                                       |
| `F.linear` (Q/K/V projections)                         | `ttnn.linear`                                                             | 3            | Direct 1:1                                                                                                                                                                                            |
| `F.conv1d` + `F.silu` (causal conv1d ×3)               | `ttnn.conv1d` (fused SiLU)                                                | 18           | 6 ops/conv (native, T ≤ 512): to_layout → zeros → concat → conv1d+silu → sharded_to_interleaved → to_layout. SiLU fused via `Conv1dConfig(activation=UnaryWithParam(SILU))`. FIR fallback for T > 512 |
| `.reshape` (multi-head Q/K/V)                          | `ttnn.reshape`                                                            | 0            | Metadata only                                                                                                                                                                                         |
| `repeat_interleave` (GVA Q/K expand)                   | `ttnn.repeat_interleave`                                                  | 0–2          | Only when `num_v_heads > num_heads`                                                                                                                                                                   |
| `F.linear` + `sigmoid` (beta)                          | `ttnn.linear` + `ttnn.sigmoid`                                            | 2            | Write strength                                                                                                                                                                                        |
| `F.linear(a)` + `a + dt_bias` + `F.softplus`           | `ttnn.linear` + `ttnn.add` + `ttnn.softplus`                              | 3            | dt computation                                                                                                                                                                                        |
| `A_log.exp()` → `-A` → `-A * sp` (decay g)             | `ttnn.exp` + `ttnn.neg` + `ttnn.multiply`                                 | 3            | `g = -exp(A_log) * softplus(a + dt_bias)`                                                                                                                                                             |
| `F.linear` (gate projection)                           | `ttnn.linear`                                                             | 1            | For output gating                                                                                                                                                                                     |
| `x.pow(2).mean().rsqrt()*w*silu(gate)` (gated RMSNorm) | `ttnn.multiply`×4 + `ttnn.mean` + `ttnn.add` + `ttnn.rsqrt` + `ttnn.silu` | 8            | multiply(x,x) + mean + add(ε) + rsqrt + multiply(x,inv) + multiply(norm,w) + silu(gate) + multiply(norm,gate)                                                                                         |
| `F.linear` (output projection)                         | `ttnn.linear`                                                             | 1            |                                                                                                                                                                                                       |
| **Layer subtotal**                                     |                                                                           | **~39**      | **Native conv1d path (T ≤ 512), fused SiLU**                                                                                                                                                          |
| **— Recurrent Delta Rule Core —**                      |                                                                           |              |                                                                                                                                                                                                       |
| *Setup (one-time):*                                    |                                                                           |              |                                                                                                                                                                                                       |
| `x * rsqrt(Σx²+ε)` (L2 norm Q, K)                      | `ttnn.multiply`×2 + `ttnn.sum` + `ttnn.add` + `ttnn.rsqrt`                | 10           | 5 ops × 2 (Q and K)                                                                                                                                                                                   |
| `q * scale`                                            | `ttnn.multiply`                                                           | 1            | `scale = K^{-0.5}`                                                                                                                                                                                    |
| `.transpose(1,2)` (Q/K/V/β/g)                          | `ttnn.transpose`                                                          | 5            | To head-first [B,H,T,D]                                                                                                                                                                               |
| `.to(float32)` (Q/K/V/β/g)                             | `ttnn.typecast`                                                           | 5            | Float32 for recurrent precision                                                                                                                                                                       |
| `torch.zeros` (state [B,H,K,V])                        | `ttnn.zeros`                                                              | 1            |                                                                                                                                                                                                       |
| *Per timestep (×T):*                                   |                                                                           |              |                                                                                                                                                                                                       |
| `g.exp()` (decay factor)                               | `ttnn.exp`                                                                | 1            |                                                                                                                                                                                                       |
| `h * exp(g)` (decay state)                             | `ttnn.multiply`                                                           | 1            |                                                                                                                                                                                                       |
| `(h * k).sum(-2)` (read from state)                    | `ttnn.matmul`                                                             | 1            | k_row @ h — contracts over K dim                                                                                                                                                                      |
| `v - v_read` (compute delta)                           | `ttnn.subtract`                                                           | 1            |                                                                                                                                                                                                       |
| `delta * beta` (scale by write strength)               | `ttnn.multiply`                                                           | 1            |                                                                                                                                                                                                       |
| `outer(k, delta)` + `h += outer` (write state)         | `ttnn.matmul` + `ttnn.add`                                                | 2            | k_col @ d_row outer product                                                                                                                                                                           |
| `einsum('bhd,bhdm->bhm', q, h)` (query state)          | `ttnn.matmul`                                                             | 1            | q_row @ h                                                                                                                                                                                             |
| **Per-step subtotal**                                  |                                                                           | **8**        | **1 exp + 2 mul + 3 mm + 1 sub + 1 add**                                                                                                                                                              |
| *Teardown (one-time):*                                 |                                                                           |              |                                                                                                                                                                                                       |
| `torch.stack` / `cat` (collect outputs)                | `ttnn.concat`                                                             | 1            | Along T dim                                                                                                                                                                                           |
| `.transpose(1,2)`                                      | `ttnn.transpose`                                                          | 1            | Back to [B,T,H,V]                                                                                                                                                                                     |
| `.to(bfloat16)`                                        | `ttnn.typecast`                                                           | 1            |                                                                                                                                                                                                       |
| **Delta rule subtotal**                                |                                                                           | **25 + 8T**  | **Setup=22, teardown=3**                                                                                                                                                                              |
| **Total (Layer + Core)**                               |                                                                           | **~64 + 8T** | **T=1 → ~72, T=16 → ~192**                                                                                                                                                                            |


### DeltaNet Chunked — Op Mapping (Validated, PCC ≥ 0.9995)

Total dispatches: **~98 + 24C** (C = T / chunk_size). For T=1024, cs=64: **~482 dispatches**.

Layer ops are identical to the recurrent table above (~39 dispatches). The chunked delta rule core replaces the recurrent core:


| PyTorch Op                                             | TTNN Op                                                                                            | Dispatches    | Notes                                                                |
| ------------------------------------------------------ | -------------------------------------------------------------------------------------------------- | ------------- | -------------------------------------------------------------------- |
| **— Layer Ops (same as recurrent) —**                  |                                                                                                    | **~39**       | **See recurrent table above**                                        |
| **— Chunked Delta Rule Core —**                        |                                                                                                    |               |                                                                      |
| *Setup (one-time):*                                    |                                                                                                    |               |                                                                      |
| `x * rsqrt(Σx²+ε)` (L2 norm Q, K)                      | `ttnn.multiply`×2 + `ttnn.sum` + `ttnn.add` + `ttnn.rsqrt`                                         | 10            | 5 ops × 2                                                            |
| `.transpose` + `.to(float32)` (Q/K/V/β/g)              | `ttnn.transpose` + `ttnn.typecast`                                                                 | 10            | 5 each                                                               |
| `q * scale`                                            | `ttnn.multiply`                                                                                    | 1             |                                                                      |
| `F.pad` (zero-pad to chunk boundary)                   | `ttnn.zeros` + `ttnn.concat`                                                                       | 0–10          | 5 × (zeros+concat) when T%cs ≠ 0                                     |
| `v * β[...,None]`, `k * β[...,None]`                   | `ttnn.multiply`                                                                                    | 2             | Weighted values and keys                                             |
| *Masks + Woodbury (one-time):*                         |                                                                                                    |               |                                                                      |
| `g.cumsum(dim=-1)`                                     | `ttnn.matmul` (with triu ones)                                                                     | 1             | Cumsum via matmul — no native `ttnn.cumsum`                          |
| `decay.exp()`                                          | `ttnn.exp`                                                                                         | 1             |                                                                      |
| `(g_i − g_j).exp().tril()` (L_mask)                    | `ttnn.subtract` + `ttnn.exp` + `ttnn.multiply`×2                                                   | 4             | Intra-chunk causal decay mask                                        |
| `−(k_β @ k^T) * L * strict_tril` (interaction M)       | `ttnn.transpose` + `ttnn.matmul` + `ttnn.neg` + `ttnn.multiply`×2                                  | 5             | Strictly lower-triangular                                            |
| `(I − M)^{-1}` (Woodbury / Neumann series)             | `ttnn.add`×(S+1) + `ttnn.matmul`×(2S+1)                                                            | 17            | Repeated squaring: R += R@P, P = P@P. S = ⌈log₂(cs)⌉−1 = 5 for cs=64 |
| `attn @ v_β` (corrected values)                        | `ttnn.matmul`                                                                                      | 1             |                                                                      |
| `attn @ (k_β * decay_exp)` (cumdecay keys)             | `ttnn.multiply` + `ttnn.matmul`                                                                    | 2             |                                                                      |
| `g_c.sum(dim=-1)` (per-chunk total decay)              | `ttnn.sum`                                                                                         | 1             |                                                                      |
| `torch.zeros` (state [BH,K,V])                         | `ttnn.zeros`                                                                                       | 1             |                                                                      |
| *Per chunk (×C, where C = T / chunk_size):*            |                                                                                                    |               |                                                                      |
| Index slice + re-tilize (q/k/v_cor/k_cum/L_mask)       | `ttnn.to_layout`                                                                                   | 5             | Slicing from 4D may lose TILE_LAYOUT                                 |
| `(q @ k^T) * L_mask * tril` (intra-chunk attn)         | `ttnn.transpose` + `ttnn.matmul` + `ttnn.multiply`×2                                               | 4             | Lower-triangular causal                                              |
| `k_cumdecay @ S` (cross-chunk state read)              | `ttnn.matmul`                                                                                      | 1             |                                                                      |
| `v_corrected − v_prime`                                | `ttnn.subtract`                                                                                    | 1             |                                                                      |
| `(q * exp(decay)) @ S` (cross-chunk query)             | `ttnn.exp` + `ttnn.multiply` + `ttnn.matmul`                                                       | 3             |                                                                      |
| `intra_attn @ v_new + o_inter` (combine)               | `ttnn.matmul` + `ttnn.add`                                                                         | 2             |                                                                      |
| `S = S*exp(g_last) + k_decay^T @ v_new` (state update) | `ttnn.exp`×2 + `ttnn.multiply`×2 + `ttnn.subtract` + `ttnn.transpose` + `ttnn.matmul` + `ttnn.add` | 8             | Decay + rotate keys + rank-1 update                                  |
| **Per-chunk subtotal**                                 |                                                                                                    | **24**        | **5 mm + 5 mul + 3 exp + 2 sub + 2 add + 5 to_layout + 2 tr**        |
| *Teardown (one-time):*                                 |                                                                                                    |               |                                                                      |
| `cat` / `.transpose` / `.to(bf16)`                     | `ttnn.concat` + `ttnn.transpose` + `ttnn.typecast`                                                 | 3             |                                                                      |
| **Chunked delta rule subtotal**                        |                                                                                                    | **~59 + 24C** | **Setup+masks ≈ 56, teardown = 3**                                   |
| **Total (Layer + Core)**                               |                                                                                                    | **~98 + 24C** | **C=1 → ~122, C=16 → ~482, C=512 → ~12,386**                         |
