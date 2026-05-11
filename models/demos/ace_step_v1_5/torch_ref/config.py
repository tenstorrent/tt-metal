from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AttentionImpl = Literal["explicit", "sdpa"]


@dataclass(frozen=True)
class AceConfig:
    d_model: int = 512
    n_heads: int = 8
    d_head: int | None = None  # if None, inferred as d_model // n_heads
    d_ff: int = 2048
    cond_dim: int = 512
    eps: float = 1e-5
    """explicit: manual QKᵀV + mask; sdpa: F.scaled_dot_product_attention (fused)."""
    attention_impl: AttentionImpl = "explicit"

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads: {self.d_model} % {self.n_heads} != 0")
        d_head = self.d_head if self.d_head is not None else self.d_model // self.n_heads
        if d_head * self.n_heads != self.d_model:
            raise ValueError("d_head * n_heads must equal d_model")
