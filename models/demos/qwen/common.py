"""
Common configurations, presets, and memory layouts for Qwen3 & Qwen2.5 TTNN bring-up.
"""

import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class QwenConfig:
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
    is_reasoning_model: bool = False   # For QwQ & Qwen3 dual-thinking mode

    @classmethod
    def qwen3_27b(cls) -> "QwenConfig":
        """Configuration for Qwen3.8-27B flagship model."""
        return cls(
            vocab_size=152064,
            hidden_size=5120,
            intermediate_size=27648,
            num_hidden_layers=64,
            num_attention_heads=40,
            num_key_value_heads=8,
            max_position_embeddings=131072,
            is_reasoning_model=True
        )

    @classmethod
    def qwq_32b(cls) -> "QwenConfig":
        """Configuration for QwQ-32B deep reasoning model."""
        return cls(
            vocab_size=152064,
            hidden_size=5120,
            intermediate_size=27648,
            num_hidden_layers=64,
            num_attention_heads=40,
            num_key_value_heads=8,
            max_position_embeddings=131072,
            is_reasoning_model=True
        )

    @classmethod
    def qwen2_5_coder_0_5b(cls) -> "QwenConfig":
        """Configuration for lightweight edge Qwen2.5-Coder-0.5B."""
        return cls(
            vocab_size=151936,
            hidden_size=896,
            intermediate_size=4864,
            num_hidden_layers=24,
            num_attention_heads=14,
            num_key_value_heads=2
        )

def comp_pcc(expr_out: torch.Tensor, golden_out: torch.Tensor, pcc_threshold: float = 0.99) -> tuple[bool, float]:
    """Tenstorrent PCC (Pearson Correlation Coefficient) verification function."""
    x = expr_out.detach().flatten().float()
    y = golden_out.detach().flatten().float()
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    denom = torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8
    pcc = (torch.sum(vx * vy) / denom).item()
    passed = pcc >= pcc_threshold
    return passed, pcc

def comp_allclose(expr_out: torch.Tensor, golden_out: torch.Tensor, atol: float = 1e-2, rtol: float = 1e-2) -> tuple[bool, str]:
    """Tenstorrent AllClose verification helper."""
    passed = torch.allclose(expr_out.float(), golden_out.float(), atol=atol, rtol=rtol)
    max_diff = (expr_out.float() - golden_out.float()).abs().max().item()
    return passed, f"Max diff: {max_diff:.6f}"

# Backward compatibility alias
Qwen2_5Config = QwenConfig
