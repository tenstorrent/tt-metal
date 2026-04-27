"""Gemma4 model configuration. Hard-coded for `google/gemma-4-31B-it`."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Gemma4Config:
    """Static model dimensions captured from the source HF config.

    These values are baked into the codegen blob (as reshape constants
    and consteval scalar values); the dataclass is purely for readability
    and to avoid magic numbers in the new `gemma4/` classes.
    """

    hidden_size: int = 5376
    intermediate_size: int = 21504
    num_layers: int = 60
    num_attention_heads: int = 32
    num_kv_heads: int = 16
    head_dim_sliding: int = 256
    head_dim_full: int = 512
    rms_eps: float = 1e-6
    softcap: float = 30.0
    sliding_window: int = 1024
    layer_types: List[str] = field(
        default_factory=lambda: (
            ["sliding"] * 5
            + ["full"]
            + ["sliding"] * 5
            + ["full"]
            + ["sliding"] * 5
            + ["full"]
            + ["sliding"] * 5
            + ["full"]
            + ["sliding"] * 5
            + ["full"]
            + ["sliding"] * 5
            + ["full"]
            + ["sliding"] * 5
            + ["full"]
            + ["sliding"] * 5
            + ["full"]
            + ["sliding"] * 5
            + ["full"]
            + ["sliding"] * 5
            + ["full"]
        )
    )
