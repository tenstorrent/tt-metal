from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AttentionImplTTNN = Literal["explicit", "sdpa"]


@dataclass(frozen=True)
class AceConfigTTNN:
    d_model: int = 512
    n_heads: int = 8
    d_head: int | None = None
    d_ff: int = 2048
    cond_dim: int = 512
    eps: float = 1e-5
    attention_impl: AttentionImplTTNN = "explicit"

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads: {self.d_model} % {self.n_heads} != 0")
        d_head = self.d_head if self.d_head is not None else self.d_model // self.n_heads
        if d_head * self.n_heads != self.d_model:
            raise ValueError("d_head * n_heads must equal d_model")
