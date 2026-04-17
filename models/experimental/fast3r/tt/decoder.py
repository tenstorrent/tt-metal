"""tt-nn Fast3R decoder: Linear (decoder_embed) + 24 blocks + LayerNorm."""
from __future__ import annotations

from typing import Dict

import torch
import ttnn
from safetensors import safe_open

from models.experimental.fast3r.reference.model import Fast3RConfig
from models.experimental.fast3r.tt.attention import TtAttention
from models.experimental.fast3r.tt.block import TtLayerNorm, TtNormMlp
from models.experimental.fast3r.tt.mlp import to_device_bias, to_device_weight


class TtDecoderBlock:
    """Pre-norm transformer block without RoPE (decoder path)."""

    def __init__(self, device, cfg: Fast3RConfig, sd: Dict[str, torch.Tensor]):
        self.norm1 = TtLayerNorm(device, sd["norm1.weight"], sd["norm1.bias"])
        self.attn = TtAttention(
            device, cfg.num_heads,
            sd["attn.qkv.weight"], sd["attn.qkv.bias"],
            sd["attn.proj.weight"], sd["attn.proj.bias"],
        )
        self.norm_mlp = TtNormMlp(
            device,
            sd["norm2.weight"], sd["norm2.bias"],
            sd["mlp.fc1.weight"], sd["mlp.fc1.bias"],
            sd["mlp.fc2.weight"], sd["mlp.fc2.bias"],
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.add(x, self.attn(self.norm1(x)), memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.add(x, self.norm_mlp(x), memory_config=ttnn.L1_MEMORY_CONFIG)
        return x


class TtDecoder:
    def __init__(self, device, cfg: Fast3RConfig, weights_path: str):
        self.cfg = cfg
        self.device = device
        with safe_open(weights_path, framework="pt") as f:
            self.embed_w = to_device_weight(device, f.get_tensor("decoder.decoder_embed.weight"))
            self.embed_b = to_device_bias(device, f.get_tensor("decoder.decoder_embed.bias"))
            norm_w = f.get_tensor("decoder.dec_norm.weight")
            norm_b = f.get_tensor("decoder.dec_norm.bias")
            self.blocks = [
                TtDecoderBlock(
                    device, cfg,
                    {k.split(f".{i}.", 1)[1]: f.get_tensor(k)
                     for k in f.keys() if k.startswith(f"decoder.dec_blocks.{i}.")},
                )
                for i in range(cfg.dec_depth)
            ]
        self.norm = TtLayerNorm(device, norm_w, norm_b)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.linear(x, self.embed_w, bias=self.embed_b)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)
