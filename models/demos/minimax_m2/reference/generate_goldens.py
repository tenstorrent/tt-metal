# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Generate golden output tensors for MiniMax-M2.5 reference blocks.

Goldens are saved to models/demos/minimax_m2/reference/golden/ and used
for TTNN PCC verification (target PCC > 0.99).

Usage (from tt-metal root):
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate
    python models/demos/minimax_m2/reference/generate_goldens.py

When real weights are available:
    python models/demos/minimax_m2/reference/generate_goldens.py \
        --model-path /path/to/MiniMax-M2.5
"""

import argparse
import os
from pathlib import Path

import torch

from models.demos.minimax_m2.reference.functional import (
    MiniMaxM2Config,
    attention_forward,
    build_rope_cache,
    decoder_layer_forward,
    make_random_state_dict,
    model_forward,
    moe_forward,
    rmsnorm_forward,
)

GOLDEN_DIR = Path(__file__).parent / "golden"
GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

# Use a small config for fast golden generation (full-size when real weights available)
SMALL_CFG = MiniMaxM2Config(
    hidden_size=256,
    head_dim=64,
    num_attention_heads=4,
    num_key_value_heads=2,
    num_hidden_layers=2,
    intermediate_size=128,
    num_local_experts=4,
    num_experts_per_tok=2,
    rotary_dim=32,
    rope_theta=5_000_000.0,
    vocab_size=512,
    use_qk_norm=True,
    use_routing_bias=True,
)

SEED = 42
BATCH = 1
SEQ = 32
DTYPE = torch.float32


def generate_rmsnorm_golden(cfg: MiniMaxM2Config):
    torch.manual_seed(SEED)
    x = torch.randn(BATCH, SEQ, cfg.hidden_size, dtype=DTYPE)
    weight = torch.ones(cfg.hidden_size, dtype=DTYPE)
    output = rmsnorm_forward(x, weight, cfg.rms_norm_eps)
    path = GOLDEN_DIR / "rmsnorm_golden.pt"
    torch.save({"input": x, "weight": weight, "output": output, "eps": cfg.rms_norm_eps}, path)
    print(f"  Saved {path}  shape={output.shape}")


def generate_attention_golden(cfg: MiniMaxM2Config, sd: dict):
    torch.manual_seed(SEED)
    x = torch.randn(BATCH, SEQ, cfg.hidden_size, dtype=DTYPE)
    cos, sin = build_rope_cache(SEQ, cfg.rotary_dim, cfg.rope_theta, dtype=DTYPE)

    layer_sd = {k.removeprefix("model.layers.0.self_attn."): v for k, v in sd.items() if "layers.0.self_attn" in k}
    output = attention_forward(x, layer_sd, cos, sin, cfg)

    path = GOLDEN_DIR / "attention_golden.pt"
    torch.save(
        {
            "input": x,
            "cos": cos,
            "sin": sin,
            "weights": layer_sd,
            "output": output,
            "config": {
                "num_q": cfg.num_attention_heads,
                "num_kv": cfg.num_key_value_heads,
                "head_dim": cfg.head_dim,
                "rotary_dim": cfg.rotary_dim,
            },
        },
        path,
    )
    print(f"  Saved {path}  shape={output.shape}")


def generate_moe_golden(cfg: MiniMaxM2Config, sd: dict):
    torch.manual_seed(SEED)
    x = torch.randn(BATCH, SEQ, cfg.hidden_size, dtype=DTYPE)
    moe_sd = {
        k.removeprefix("model.layers.0.block_sparse_moe."): v for k, v in sd.items() if "layers.0.block_sparse_moe" in k
    }
    output = moe_forward(x, moe_sd, cfg)

    path = GOLDEN_DIR / "moe_golden.pt"
    torch.save(
        {
            "input": x,
            "weights": moe_sd,
            "output": output,
            "config": {"num_experts": cfg.num_local_experts, "top_k": cfg.num_experts_per_tok},
        },
        path,
    )
    print(f"  Saved {path}  shape={output.shape}")


def generate_decoder_layer_golden(cfg: MiniMaxM2Config, sd: dict):
    torch.manual_seed(SEED)
    x = torch.randn(BATCH, SEQ, cfg.hidden_size, dtype=DTYPE)
    cos, sin = build_rope_cache(SEQ, cfg.rotary_dim, cfg.rope_theta, dtype=DTYPE)
    layer_sd = {k.removeprefix("model.layers.0."): v for k, v in sd.items() if k.startswith("model.layers.0.")}
    output = decoder_layer_forward(x, layer_sd, cos, sin, cfg)

    path = GOLDEN_DIR / "decoder_layer_golden.pt"
    torch.save({"input": x, "cos": cos, "sin": sin, "output": output}, path)
    print(f"  Saved {path}  shape={output.shape}")


def generate_full_model_golden(cfg: MiniMaxM2Config, sd: dict):
    torch.manual_seed(SEED)
    input_ids = torch.randint(0, cfg.vocab_size, (BATCH, SEQ))
    logits = model_forward(input_ids, sd, cfg)

    path = GOLDEN_DIR / "full_model_golden.pt"
    torch.save(
        {
            "input_ids": input_ids,
            "logits": logits,
            "config": {"vocab_size": cfg.vocab_size, "hidden_size": cfg.hidden_size},
        },
        path,
    )
    print(f"  Saved {path}  logits shape={logits.shape}")


def dequantize_fp8_weight(weight: torch.Tensor, scale_inv: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """Dequantize a float8_e4m3fn weight using per-block scale_inv.

    Args:
        weight:    [out_features, in_features] float8_e4m3fn
        scale_inv: [out_features/128, in_features/128] float32
        block_size: quantization block size (128)

    Returns:
        [out_features, in_features] float32
    """
    out_f, in_f = weight.shape
    w = weight.float()
    # Tile scale_inv to match weight shape, then crop
    s = scale_inv.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    s = s[:out_f, :in_f]
    return w * s


def load_and_dequant(raw_sd: dict) -> dict:
    """Convert an fp8 safetensors state dict to float32 by dequantizing all fp8 weights.

    For each key K ending in '.weight' that has a paired 'K_scale_inv', applies
    dequantization. All other tensors are cast to float32.
    """
    sd = {}
    skip = set()
    for k, v in raw_sd.items():
        if k.endswith(".weight_scale_inv"):
            skip.add(k)

    for k, v in raw_sd.items():
        if k in skip:
            continue
        scale_key = k + "_scale_inv"
        if scale_key in raw_sd and v.dtype == torch.float8_e4m3fn:
            sd[k] = dequantize_fp8_weight(v, raw_sd[scale_key])
        else:
            sd[k] = v.float() if v.dtype not in (torch.float32,) else v
    return sd


def generate_from_real_weights(model_path: str):
    """Generate goldens from real MiniMax-M2.5 checkpoint.
    Only loads the shards containing layer 0 + embeddings (~3.6GB).
    """
    import json

    from safetensors.torch import load_file

    print(f"\nLoading config from {model_path} ...")
    with open(os.path.join(model_path, "config.json")) as f:
        cfg_dict = json.load(f)

    cfg = MiniMaxM2Config(
        hidden_size=cfg_dict["hidden_size"],
        head_dim=cfg_dict["head_dim"],
        num_attention_heads=cfg_dict["num_attention_heads"],
        num_key_value_heads=cfg_dict["num_key_value_heads"],
        num_hidden_layers=1,  # only layer 0 for block-level goldens
        intermediate_size=cfg_dict["intermediate_size"],
        num_local_experts=cfg_dict["num_local_experts"],
        num_experts_per_tok=cfg_dict["num_experts_per_tok"],
        rotary_dim=cfg_dict["rotary_dim"],
        rope_theta=cfg_dict["rope_theta"],
        rms_norm_eps=cfg_dict["rms_norm_eps"],
        vocab_size=cfg_dict["vocab_size"],
        use_qk_norm=cfg_dict.get("use_qk_norm", True),
        use_routing_bias=cfg_dict.get("use_routing_bias", True),
    )

    # Find shards needed for layer 0 + embeddings using index
    with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    needed_shards = set()
    for k, shard in weight_map.items():
        if "layers.0." in k or "embed_tokens" in k or k == "model.norm.weight" or "lm_head" in k:
            needed_shards.add(shard)
    needed_shards = sorted(needed_shards)
    print(f"Loading {len(needed_shards)} shards for layer 0 + embeddings: {needed_shards}")

    raw_sd = {}
    for shard in needed_shards:
        print(f"  Loading {shard} ...")
        raw_sd.update(load_file(os.path.join(model_path, shard)))

    print("Dequantizing fp8 weights to float32 ...")
    sd = load_and_dequant(raw_sd)

    print("\nGenerating goldens from REAL weights (layer 0):")
    generate_rmsnorm_golden(cfg)
    generate_attention_golden(cfg, sd)
    generate_moe_golden(cfg, sd)
    generate_decoder_layer_golden(cfg, sd)

    # Full model forward uses only layer 0 (cfg.num_hidden_layers=1)
    print("Running full model forward with layer 0 only ...")
    generate_full_model_golden(cfg, sd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to MiniMax-M2.5 HuggingFace checkpoint. " "If not provided, uses random weights.",
    )
    args = parser.parse_args()

    if args.model_path:
        generate_from_real_weights(args.model_path)
        return

    print("No --model-path provided. Generating goldens with RANDOM weights (structural verification only).")
    print("Replace with real weights once checkpoint is available.\n")

    cfg = SMALL_CFG
    sd = make_random_state_dict(cfg, num_layers=cfg.num_hidden_layers, seed=SEED)

    print("Generating goldens:")
    generate_rmsnorm_golden(cfg)
    generate_attention_golden(cfg, sd)
    generate_moe_golden(cfg, sd)
    generate_decoder_layer_golden(cfg, sd)
    generate_full_model_golden(cfg, sd)

    print(f"\nAll goldens saved to {GOLDEN_DIR}/")
    print("NOTE: These are random-weight goldens. Re-run with --model-path for real goldens.")


if __name__ == "__main__":
    main()
