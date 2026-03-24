# Qwen3.5-9B on Blackhole P150 — Bringup Status

**Date:** 2026-03-21
**Device:** Blackhole P150 (single device)
**Model:** Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2
**Scope:** Text-only (no vision encoder)
**Weights:** `/localdev/atupe/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2` (safetensors, 18.1GB bf16)

---

## Architecture

Qwen3.5-9B is a hybrid linear-attention + softmax-attention transformer:

- **32 layers** in a repeating `[DeltaNet, DeltaNet, DeltaNet, FullAttention]` x 8 pattern
- **24 Gated DeltaNet layers** (linear attention with delta rule, causal Conv1D, exponential gating)
- **8 Gated Full Attention layers** (softmax GQA: 16 Q heads, 4 KV heads, head_dim=256, output gate)
- **SwiGLU MLP** (hidden=4096, intermediate=12288)
- **RoPE** with partial_rotary_factor=0.25 (64 of 256 dims), theta=10M
- **Vocab:** 248,320 tokens

---

## What Is Implemented

### Model Components (8 files, 723 lines)

| File | Lines | Description |
|---|---|---|
| `tt/model_config.py` | 83 | `Qwen35ModelArgs` — loads config.json, all model/device params |
| `tt/weight_mapping.py` | 103 | `remap_qwen35_state_dict()` — HF→internal key remapping, QKV/conv splits |
| `tt/qwen35_rope.py` | 73 | `Qwen35RoPESetup` — partial RoPE (64/256 dims, theta=10M) |
| `tt/qwen35_gated_attention.py` | 66 | `Qwen35GatedAttention` — wraps experimental SDPA+gate TTNN op |
| `tt/qwen35_gated_deltanet.py` | 166 | `Qwen35GatedDeltaNet` — wraps experimental DeltaNet TTNN op with state mgmt |
| `tt/qwen35_mlp.py` | 34 | `Qwen35MLP` — SwiGLU feed-forward |
| `tt/qwen35_decoder.py` | 68 | `Qwen35TransformerBlock` — hybrid layer dispatch + manual RMSNorm |
| `tt/qwen35_model.py` | 130 | `Qwen35Model` — full model with prefill/decode/state management |

### Tests (5 files)

| File | Tests | Status |
|---|---|---|
| `tests/test_weight_mapping.py` | 25 tests | All PASS (CPU-only) |
| `tests/test_model_config.py` | 10 tests | All PASS (CPU-only) |
| `tests/test_single_layer.py` | 3 tests | All PASS (on P150) |
| `tests/test_component_pcc.py` | Component PCC | On P150 |
| `tests/test_chunked_pcc.py` | Chunked vs recurrent PCC | On P150 |

### Demo (1 file)

| File | Description |
|---|---|
| `demo/text_demo.py` | E2E pytest test (prefill_128, prefill_2k, prefill_4k, prefill_8k) |

### Dependencies (read-only, not modified)

| File | Description |
|---|---|
| `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_attention.py` | Gated Attention TTNN op (from sdawle branch) |
| `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_deltanet.py` | Gated DeltaNet TTNN op (from sdawle branch) |
| `models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_ops.py` | Delta rule core algorithms |

### Design Documents

| File | Description |
|---|---|
| `docs/superpowers/specs/2026-03-21-qwen35-9b-blackhole-bringup-design.md` | Design spec |
| `docs/superpowers/plans/2026-03-21-qwen35-9b-blackhole-bringup.md` | Implementation plan |

---

## Current Performance

Measured on Blackhole P150, batch_size=1, max_seq_len=2048.

| Metric | Current | Previous | Baseline |
|---|---|---|---|
| **Model load time** | ~190s | ~150s | ~145s |
| **TTFT (prefill, 5 tokens)** | ~12s | ~18s | ~19s |
| **Decode latency (avg, steady-state)** | ~165ms/token | ~287ms/token | ~16.7s/token |
| **Decode throughput** | 6.1 tok/s | 3.5 tok/s | 0.06 tok/s |
| **First decode** | ~468ms | ~1.7s | ~17.6s |
| **Min decode** | ~164ms | ~290ms | ~15.2s |
| **Output quality** | Coherent text | Coherent text | No output (garbage) |

**Key improvements (2026-03-22, optimization pass):**
- **1.7x decode speedup** (287ms → 165ms/token, 3.5 → 6.1 tok/s)
- **1.5x TTFT improvement** (18s → 12s)
- **3.6x first decode speedup** (1.7s → 468ms)
- **Enabled program cache** — caches compiled TTNN kernels across calls
- **Eliminated CPU-device round-trips** — conv states kept on device (was 72 ttnn.to_torch() per step)
- **Fast T=1 conv path** — specialized decode conv1d avoiding FIR loop overhead
- **Device-cached RoPE** — pre-computed full table on device, slice for decode
- **Fused QKV projection** — single matmul instead of 3 for DeltaNet layers
- **bfloat8_b projection weights** — halved memory bandwidth for DeltaNet + attention projections
- **Tensor deallocation** — explicit ttnn.deallocate() for intermediates (MLP, decoder, attention)
- **Fused RMSNorm** — ttnn.rms_norm in DeltaNet output normalization
- **Precomputed conv weight taps and bias** — device-side tensors avoid per-call CPU conversion

**Key improvements (2026-03-22, bringup):**
- **58x decode speedup** (16.7s → 287ms/token)
- **Fixed zero-centered RMSNorm** — root cause of garbage output
- **Added KV cache** for 8 gated attention layers
- **Added conv state management** for 24 DeltaNet layers
- **Replaced manual 6-op RMSNorm** with fused `ttnn.rms_norm`
- **Fixed chunk mode** for short sequences (uses float32 instead of bfloat16)
- **Added chat template** formatting with `<think>` reasoning prefix

---

## Bugs Fixed

| Bug | Root Cause | Fix |
|---|---|---|
| No coherent output | Layer norms used `weight * x_normed` instead of `(1 + weight) * x_normed`. Zero-centered weights (~0.03) squashed signal 30x per layer. | Changed to zero-centered RMSNorm, pre-offset weights by +1. |
| Decode produces only whitespace | Missing KV cache for 8 gated attention layers. During decode (T=1), attention only saw current token. | Implemented KV cache with proper `is_causal=False` for decode. |
| Decode produces only spaces | Missing conv state for 24 DeltaNet layers. Conv1d used zeros instead of history during decode. | Implemented conv state management (save last 3 tokens). |
| PCC degrades across layers during prefill | DeltaNet fell back to recurrent mode (bfloat16) when T < chunk_size. | Always use chunk mode (float32) for prefill with T > 1. |
| Chunk mode output trimming crash | `o[:, :T]` sliced wrong dimension after concat. | Reshape to merge chunk dims before slicing. |

---

## What Remains (Optimization Roadmap)

### P0 — Correctness

| Item | Status | Impact |
|---|---|---|
| Conv state management for DeltaNet decode | **DONE** | Conv1d now uses proper history during decode. |
| KV cache for Gated Attention layers | **DONE** | Full KV cache with concat during decode. |
| Zero-centered RMSNorm | **DONE** | `(1 + weight) * x_normed` for all layer norms. |
| PCC validation against torch reference | **DONE** | Component PCC > 0.999 for all modules. |

### P1 — Performance (further improvements possible)

| Item | Expected Impact | Notes |
|---|---|---|
| **Trace capture/replay** | 2-5x decode speedup | `ttnn.begin_trace_capture()` / `ttnn.end_trace_capture()` — would bring decode to ~60-150ms |
| **Precompute RoPE on device** | Eliminate host round-trip | Currently computed on CPU and transferred each step |
| **Weight caching** | 5-10x faster load | Cache converted ttnn weights to disk |
| **Paged attention** | Enable longer sequences | Replace concat-based KV cache with paged version |
| **Op fusion / program config tuning** | 2-5x op speedup | Blackhole-specific grid sizes, matmul tiling |
| **bfloat8_b activations** | 2x memory reduction | Currently using bfloat16 for activations |

### P2 — Features

| Item | Notes |
|---|---|
| Vision encoder integration | Enable multimodal (image+text) inference |
| Multi-token prediction (MTP) | Model has MTP weights — speculative decoding |
| Larger batch sizes | Currently batch=1, scale to 4+ |
| Longer context | Currently max_seq_len=2048, model supports 262K |
| vLLM integration | Follow Qwen3 Embedding 8B pattern in `generator_vllm.py` |

---

## File Tree

```
models/demos/blackhole/qwen3_5_9b/
├── STATUS.md                              ← this file
├── tt/
│   ├── model_config.py                    Config loading
│   ├── weight_mapping.py                  HF→internal weight remapping
│   ├── qwen35_rope.py                     Partial RoPE
│   ├── qwen35_gated_attention.py          Full attention wrapper
│   ├── qwen35_gated_deltanet.py           Linear attention wrapper + state
│   ├── qwen35_mlp.py                      SwiGLU MLP
│   ├── qwen35_decoder.py                  Hybrid layer dispatch
│   └── qwen35_model.py                    Full model assembly
├── tests/
│   ├── test_weight_mapping.py             25 tests — weight remapping (CPU)
│   ├── test_model_config.py               10 tests — config loading (CPU)
│   ├── test_single_layer.py               3 tests — per-layer (P150)
│   ├── test_component_pcc.py              Component PCC validation (P150)
│   └── test_chunked_pcc.py                Chunked vs recurrent PCC (P150)
└── demo/
    ├── text_demo.py                       E2E pytest test (the canonical e2e test)
    └── sample_prompts/                    JSON prompt files for various seq lengths
```

---

## How to Run

```bash
# Activate the tt-metal Python environment
source /localdev/atupe/tt-metal/python_env/bin/activate

# Run CPU-only tests (no device needed)
pytest models/demos/blackhole/qwen3_5_9b/tests/test_weight_mapping.py -v --noconftest
pytest models/demos/blackhole/qwen3_5_9b/tests/test_model_config.py -v --noconftest

# Run device tests (requires P150)
pytest models/demos/blackhole/qwen3_5_9b/tests/test_single_layer.py -v -s --noconftest --timeout=300

# Run E2E demo test (all configurations)
pytest models/demos/blackhole/qwen3_5_9b/demo/text_demo.py -v -s
# Run a specific configuration
pytest models/demos/blackhole/qwen3_5_9b/demo/text_demo.py -v -s -k "prefill_128"

# Reset device if tests hang
tt-smi -r
```
