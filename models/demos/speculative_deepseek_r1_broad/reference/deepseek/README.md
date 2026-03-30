# DeepSeek-R1 reference helpers

This subfolder mirrors the helper structure used by `deepseek_v3/reference/deepseek/`
for the R1-0528 model.

## Contents

- `model.py` — Standalone DeepSeek-R1 Transformer implementation with MLA attention,
  MoE routing, YaRN RoPE, and tensor-parallel support.  Uses `ModelArgs` dataclass
  for configuration (matches the V3 architecture).
- `kernel.py` — FP8 quantization/dequantization kernels (Triton-based, with graceful
  fallback when Triton is not available).
- `rope_helpers.py` — Standalone RoPE frequency precomputation and application
  functions (no nn.Module dependency).
- `configs/config_r1_0528.json` — Architecture hyperparameters for the full
  DeepSeek-R1-0528 (671B) model in `ModelArgs` format.

## Architecture

DeepSeek-R1-0528 shares the DeepSeek-V3 architecture:

- **671B parameters** (37B active per token)
- **61 transformer layers** (3 dense + 58 MoE)
- **128 attention heads** with Multi-Head Latent Attention (MLA)
  - Q LoRA rank: 1536, KV LoRA rank: 512
  - QK nope head dim: 128, QK rope head dim: 64
  - V head dim: 128
- **256 routed experts** (8 activated per token) + 1 shared expert
- **8 expert groups** with top-4 group selection
- **YaRN RoPE** scaling (factor=40, 163K context)
- **FP8 quantized** weights with block-wise scaling
