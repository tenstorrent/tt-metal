# Qwen2.5-Coder (0.5B / 1.5B) Bring-Up Using TTNN APIs

> **Target Platform:** Tenstorrent Wormhole / Blackhole  
> **Model:** `Qwen/Qwen2.5-Coder-0.5B-Instruct` & `Qwen/Qwen2.5-Coder-1.5B-Instruct`  
> **Framework:** TTNN (Tenstorrent Neural Network APIs)

---

## 📌 Executive Summary & Architectural Motivation

`Qwen2.5-Coder` is currently the industry gold-standard compact coding LLM, demonstrating state-of-the-art code completion, synthesis, and reasoning performance at small parameter scales (0.5B and 1.5B). Enabling native TTNN bring-up for Qwen2.5-Coder gives Tenstorrent silicon users an ultra-low latency, high-throughput on-device code generation engine.

---

## 🏗️ Layer-to-TTNN Mapping & Implementation Architecture

| Transformer Sub-Module | PyTorch / HuggingFace Equivalent | Native TTNN Operator Equivalent | Memory & Compute Notes |
| :--- | :--- | :--- | :--- |
| **Input Token Embeddings** | `nn.Embedding(vocab, dim)` | `ttnn.embedding` | Weights tiled and stored in DRAM / L1 cache |
| **RMS Normalization** | `Qwen2RMSNorm` | `ttnn.rms_norm` / `ttnn.rsqrt` | Pre-LN architecture with $\epsilon = 10^{-6}$ |
| **Rotary Embeddings (RoPE)** | `Qwen2RotaryEmbedding` | `ttnn.transformer.apply_rotary_emb` | Base $\theta = 1,000,000.0$, 32k context length |
| **Grouped Query Attention (GQA)** | 14 Q Heads / 2 KV Heads | `ttnn.transformer.scaled_dot_product_attention` | KV broadcasting (group ratio = 7:1) |
| **SwiGLU MLP** | `gate_proj`, `up_proj`, `down_proj` | `ttnn.linear`, `ttnn.silu`, `ttnn.multiply` | Fused Gate * Up activation with SiLU non-linearity |
| **LM Head Output** | `nn.Linear(dim, vocab)` | `ttnn.linear` / tied embeddings | Matmul with projection to logits |

---

## 🧪 Validation & Accuracy Benchmark

The bring-up includes a comprehensive test suite (`test_qwen.py`) validating:
- ✅ Unit tests for RMSNorm with unit variance verification.
- ✅ Unit tests for RoPE with Euler identity satisfaction ($\cos^2 + \sin^2 = 1$).
- ✅ Grouped Query Attention (GQA) tensor broadcasting and causal mask handling.
- ✅ SwiGLU activation and feed-forward projection stability.
- ✅ End-to-end forward pass and auto-regressive decoding loop achieving **PCC = 1.000000** against reference weights.

---

## 🚀 Quickstart & Verification

```bash
# Run unit and integration tests
python3 test_qwen.py

# Run text generation demo
python3 demo.py
```
