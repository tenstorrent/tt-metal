# FIBO on TTNN — Model Documentation

## Overview

This document covers the Tenstorrent tt_dit implementation of Bria's **FIBO** text-to-image model. FIBO is a flow-matching MMDiT (Flux-shaped) model conditioned on an LLM text encoder. The implementation is decomposed into four sub-projects, built in data-flow order. See the design spec for full detail:
[`docs/superpowers/specs/2026-07-06-fibo-smollm3-encoder-design.md`](../../../docs/superpowers/specs/2026-07-06-fibo-smollm3-encoder-design.md)

## 4-Sub-Project Decomposition

| # | Sub-project | Status | Strategy |
|---|---|---|---|
| **1** | **SmolLM3 text encoder** | **Done** | New `encoders/smollm3/`; decoder layer from Qwen25VL, all-hidden-states shell from Gemma |
| 2 | BriaFibo transformer | TODO | New `transformer_bria_fibo.py` from Flux1 + per-layer "concat-halves" text injection |
| 3 | Wan VAE + solver wiring | TODO | Reuse `vae_wan2_1.py` (T=1 decode) + `EulerSolver` + dynamic-shift scheduler |
| 4 | Pipeline + Blackhole bringup | TODO | New `pipelines/bria_fibo/`; CFG batched=2; 2×2 mesh (`cfg=(1,0) sp=(2,0) tp=(2,1)`) |

---

## Sub-project 1: SmolLM3 Text Encoder

### Architecture

SmolLM3-3B serves as the text encoder for FIBO. Key configuration:

| Parameter | Value |
|---|---|
| Layers | 36 |
| Hidden size | 2048 |
| Attention heads / KV heads | 16 / 4 (GQA) |
| Intermediate size | 11008 |
| Activation | SiLU |
| Norm | RMSNorm, eps=1e-6 |
| RoPE theta | 5,000,000 |
| Max position embeddings | 65,536 |
| NoPE (no positional embedding) | every 4th layer (0-indexed 3, 7, 11, ..., 35); 9 NoPE layers total |
| Vocab size | 128,256 |
| Attention bias | False |

### NoPE Layers

`no_rope_layers[i] = int((i + 1) % 4 != 0)` — value `1` = apply RoPE, value `0` = NoPE (no positional embedding applied). The 9 NoPE layers are at indices 3, 7, 11, 15, 19, 23, 27, 31, 35.

### HF-Exact Output Contract

The encoder replicates `SmolLM3ForCausalLM(..., output_hidden_states=True)` exactly:

- **All hidden states**: length `num_hidden_layers + 1 = 37`. `hidden_states[0]` is the embedding output (input to layer 0); `hidden_states[i]` is the output of layer `i-1`; `hidden_states[-1]` is the final RMSNorm output.
- **`prompt_embeds`**: `torch.cat([hidden_states[-1], hidden_states[-2]], dim=-1)` → shape `[B, T, 4096]` (2 × 2048). This is the primary FIBO conditioning signal consumed by the transformer.
- **All 37 states** are also returned for per-block injection into the BriaFibo transformer (sub-project 2).

### Files

```
models/tt_dit/
  encoders/smollm3/
    __init__.py
    config.py              # SmolLM3Config + from_hf_config()
    model_smollm3.py       # SmolLM3TextEncoder, SmolLM3DecoderLayer, SmolLM3Attention, SmolLM3Mlp
  tests/encoders/smollm3/
    test_smollm3.py        # Unit tests + full-mesh (2×2) validation
```

### Running the Tests

```bash
# All SmolLM3 encoder tests (needs FIBO weights + login, see below)
HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
  python_env/bin/python -m pytest models/tt_dit/tests/encoders/smollm3/test_smollm3.py -v

# Full-mesh (2×2 Blackhole) validation only
HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
  python_env/bin/python -m pytest "models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_smollm3_encoder_full_mesh" -v
```

**Prerequisites:**

- **Weights**: `briaai/FIBO` is a gated HuggingFace repo. Accept the license on the HF model page, then authenticate: `huggingface-cli login`. The test will `pytest.skip` with a clear message if weights are unavailable.
- **`FIBO_PATH`** (optional): Override the HF model path, e.g. `FIBO_PATH=/data/models/FIBO pytest ...`. Defaults to `briaai/FIBO`.
- **`N_LAYERS`** (optional): Truncate the encoder to `N` layers for the `test_smollm3_encoder_all_layers` and `test_smollm3_encode_contract` tests (default: 6). Use `N_LAYERS=36` for a full-depth truncated run. The `test_smollm3_encoder_full_mesh` test always runs all 36 layers.
- **Devices**: The full-mesh test requires **4 Blackhole devices** (2×2 mesh) with `FABRIC_1D` fabric config. Single-device tests (`tp=1`) use any single Blackhole.

### Measured PCCs (real `briaai/FIBO` weights)

| Test | PCC |
|---|---|
| MLP (single layer) | 99.998% |
| Attention, RoPE path | ~99.99% |
| Attention, NoPE path | ~99.99% |
| Decoder layer (single) | 99.9996% |
| Encoder all-layers (6L, seq=128, tp=1) — per-layer min | 99.957% |
| encode() contract (6L, seq=128, tp=1) | 99.98% |
| **Full 36L mesh (seq=128, tp=2 on 2×2)** | **99.9362%** |
| **Full 36L mesh (seq=2048, tp=2 on 2×2)** | **99.9597%** |

All tests pass PCC ≥ 0.99. No bf16 depth drift observed even at seq=2048 over all 36 layers. CCL (`all_gather_async`) on the mesh axis was exercised without hangs.
