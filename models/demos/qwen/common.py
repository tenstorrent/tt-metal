"""
Common utilities, memory configurations, and weight loaders for Qwen2.5 TTNN bring-up.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class Qwen2_5Config:
    vocab_size: int = 151936
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = True
    dtype: str = "bfloat16"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Qwen2_5Config":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

def comp_pcc(expr_out: torch.Tensor, golden_out: torch.Tensor, pcc_threshold: float = 0.99) -> tuple[bool, float]:
    """
    Standard Tenstorrent PCC (Pearson Correlation Coefficient) verification function.
    """
    x = expr_out.detach().flatten().float()
    y = golden_out.detach().flatten().float()
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    denom = torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8
    pcc = (torch.sum(vx * vy) / denom).item()
    passed = pcc >= pcc_threshold
    return passed, pcc

def comp_allclose(expr_out: torch.Tensor, golden_out: torch.Tensor, atol: float = 1e-2, rtol: float = 1e-2) -> tuple[bool, str]:
    """
    Standard Tenstorrent AllClose verification helper.
    """
    passed = torch.allclose(expr_out.float(), golden_out.float(), atol=atol, rtol=rtol)
    max_diff = (expr_out.float() - golden_out.float()).abs().max().item()
    return passed, f"Max diff: {max_diff:.6f}"
