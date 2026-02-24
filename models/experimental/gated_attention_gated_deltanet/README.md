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
| 1     | 2   | 0.68       | 2.09      | 0.32x   | 0.9999 |
| 2     | 2   | 0.77       | 2.11      | 0.37x   | 0.9999 |
| 4     | 2   | 0.88       | 2.10      | 0.42x   | 0.9998 |
| 8     | 2   | 1.02       | 2.10      | 0.48x   | 0.9998 |
| 16    | 2   | 1.33       | 2.12      | 0.63x   | 0.9997 |
| 32    | 2   | 2.07       | 2.14      | 0.97x   | 0.9997 |
| 64    | 2   | 3.51       | 2.13      | 1.65x   | 0.9997 |
| 128   | 2   | 6.77       | 2.21      | 3.06x   | 0.9996 |
| 256   | 2   | 11.87      | 2.31      | 5.14x   | 0.9995 |
| 512   | 2   | 23.75      | 4.31      | 5.50x   | 0.9994 |
| 1024  | 2   | 153.40     | 9.23      | 16.6x   | 0.9994 |
| 2048  | 2   | 544.31     | 21.90     | 24.9x   | 0.9992 |
| 4096  | 2   | 1980.27    | 60.63     | 32.7x   | 0.9989 |
| 8192  | 1   | 3783.42    | 104.03    | 36.4x   | 0.9989 |
| 16384 | 1   | 16974.90   | 363.56    | 46.7x   | 0.9988 |


B=1 for T >= 8192 to avoid host OOM on the O(T²) attention matrix. T=32768 OOMs on the torch CPU reference.

### Gated DeltaNet — Chunked Mode (B=2, H=4, K=128, V=256, chunk_size=64)

- **H** — attention heads, **K** — key/query head dim, **V** — value head dim, **chunk_size** — tokens processed in parallel per chunk
- Recurrent state per head is a K×V = 128×256 matrix (fixed size, independent of T)


| T     | Torch (ms) | TTNN (ms) | Speedup | PCC    |
| ----- | ---------- | --------- | ------- | ------ |
| 1     | 2.58       | 46.73     | 0.06x   | 0.9992 |
| 2     | 2.81       | 48.39     | 0.06x   | 0.9997 |
| 4     | 4.00       | 48.99     | 0.08x   | 0.9996 |
| 8     | 6.22       | 47.69     | 0.13x   | 0.9994 |
| 16    | 9.45       | 48.98     | 0.19x   | 0.9993 |
| 32    | 17.91      | 46.06     | 0.39x   | 0.9995 |
| 64    | 31.86      | 47.98     | 0.66x   | 0.9995 |
| 128   | 25.89      | 51.70     | 0.50x   | 0.9995 |
| 256   | 44.79      | 53.05     | 0.84x   | 0.9995 |
| 512   | 87.96      | 56.29     | 1.56x   | 0.9995 |
| 1024  | 157.32     | 39.20     | 4.01x   | 0.9998 |
| 2048  | 319.11     | 67.26     | 4.74x   | 0.9998 |
| 4096  | 776.16     | 127.63    | 6.08x   | 0.9998 |
| 8192  | 1832.81    | 303.88    | 6.03x   | 0.9998 |
| 16384 | 3587.04    | 688.15    | 5.21x   | 0.9998 |
| 32768 | 7074.67    | 1459.16   | 4.85x   | 0.9998 |


DeltaNet's O(T) complexity (via chunked processing)
Gated Attention's O(T²) complexity
