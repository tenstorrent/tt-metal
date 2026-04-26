# Kimi K2.5 (kimi_k25)

## Platforms
    Galaxy (WH) — TG (32 chips), DUAL (64 chips), QUAD (128 chips)

## Introduction

This demo targets the [moonshotai/Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5) model — a 384-expert Mixture-of-Experts (MoE) language model with Multi-head Latent Attention (MLA).

Kimi K2.5's text backbone is architecturally near-identical to DeepSeek V3. This implementation reuses the `models/demos/deepseek_v3` runtime (MLA, MoE dispatch, CCL, RoPE, embeddings) and adds:

1. **`config_adapter.py`** — maps Kimi K2.5 HF config (`kimi_k2`/`kimi_k25` model type) to the DSV3-compatible parameter struct
2. **INT4 group-32 weight loader** *(in progress)* — dequantizes expert weights from compressed-tensors W4A16 format
3. **Flat top-k gate** *(validation pending)* — verifies `moe_gate.py` handles `n_group=1` (384 experts, no grouping)

## Key Architecture Differences from DeepSeek V3

| Parameter | Kimi K2.5 | DeepSeek V3 |
|---|---|---|
| Routed experts | **384** | 256 |
| Attention heads | **64** | 128 |
| n_group / topk_group | **1 / 1** (flat) | 8 / 4 (grouped) |
| first_k_dense_replace | **1** | 3 |
| rope_theta | **50,000** | 10,000 |
| rms_norm_eps | **1e-5** | 1e-6 |
| vocab_size | **163,840** | ~129,280 |
| routed_scaling_factor | **2.827** | 2.5 |
| Weight quantization | **INT4 group-32** (experts only) | FP8 |
| MTP layers | **0** | 1 |

## Hardware Requirements

- **Minimum**: 1× Galaxy 6U (TG, 32 Wormhole chips, 384 GB DRAM)
- Weights: ~220 GB (85 GB BF16 attention + ~120 GB INT4 experts + ~15 GB misc)
- KV cache (256K context, 1 seq): ~530 MB/device on TG

| Topology | Chips | DRAM | Experts/device |
|---|---|---|---|
| TG (1× Galaxy 6U) | 32 | 384 GB | 12 |
| DUAL (2× Galaxy) | 64 | 768 GB | 6 |
| QUAD (4× Galaxy) | 128 | 1.5 TB | 3 |

## Prerequisites

- `tenstorrent/tt-metal` cloned and built: see [INSTALLING.md](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- `pip install transformers>=4.57.1 safetensors compressed-tensors`
- HF weights downloaded: `moonshotai/Kimi-K2.5` (~600 GB raw, ~220 GB post-dequant)

## Environment Variables

```bash
export KIMI_K25_HF_MODEL=/path/to/kimi-k2-5-weights   # local weight directory
export KIMI_K25_CACHE=/localdev/$USER/kimi-k25-cache   # TTNN weight cache
export MESH_DEVICE=TG                                   # or DUAL / QUAD
```

## Quick Start (config validation — no weights needed)

```bash
# Validate config adapter with hardcoded fixture (offline, no download)
python models/demos/kimi_k25/utils/config_adapter.py --fixture

# Validate config adapter against HF config.json (network required)
python models/demos/kimi_k25/utils/config_adapter.py \
  --model-path moonshotai/Kimi-K2.5

# Validate against local weights
python models/demos/kimi_k25/utils/config_adapter.py \
  --model-path $KIMI_K25_HF_MODEL
```

Expected output:
```
[OK] Config loaded and validated successfully
KimiK25Config(
  layers=61, heads=64, hidden=7168
  experts=384 (top-8, n_group=1, first_dense=1)
  ...
)
[PASS] KimiK25Config smoke test complete
```

## Running Tests

*(Tests are in progress — this section will be updated as milestones are completed)*

```bash
# Single-layer MLA + MoE accuracy test (random weights)
pytest models/demos/kimi_k25/tests/test_mla.py
pytest models/demos/kimi_k25/tests/test_moe.py

# Full model smoke test (TG, random weights)
MESH_DEVICE=TG pytest models/demos/kimi_k25/tests/test_model.py
```

## Running the Demo

*(Coming in M5/M6 milestone)*

```bash
./models/demos/deepseek_v3/scripts/launch_multihost_galaxy.py 2x -- \
  python models/demos/kimi_k25/demo/demo.py \
    --model-path $KIMI_K25_HF_MODEL \
    --cache-dir $KIMI_K25_CACHE \
    "Hello, Kimi!"
```

## Milestone Status

| Milestone | Status |
|---|---|
| M1: Scaffold + config_adapter | 🟡 In progress |
| M2: INT4 weight loader | ⬜ Not started |
| M3: MoE gate (flat routing) | ⬜ Not started |
| M4: Single-layer accuracy | ⬜ Not started |
| M5: Full model on Galaxy | ⬜ Not started |
| M6: CI + Demo | ⬜ Not started |

## References

- [Kimi K2.5 paper](https://arxiv.org/abs/2602.02276)
- [HuggingFace model](https://huggingface.co/moonshotai/Kimi-K2.5)
- [DeepSeek V3 tt-metal demo](../deepseek_v3/)
- Research doc: `/workspace/group/research/kimi-k2-5-galaxy-port.md`
