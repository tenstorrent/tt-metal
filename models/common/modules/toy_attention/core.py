from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import ttnn


@dataclass
class AttentionSpec:
    """
    Model-level attention specification.

    This captures the architectural shape of the attention module
    (hidden size, number of heads, etc.) and is intentionally
    free of device- or implementation-specific details.
    """

    hidden_size: int
    num_heads: int
    head_dim: int
    use_rotary_embeddings: bool = False
    use_sliding_window: bool = False
    window_size: Optional[int] = None
    dropout: float = 0.0

    @property
    def total_dim(self) -> int:
        return self.num_heads * self.head_dim


@dataclass
class OpConfig:
    """
    Per-op TTNN configuration. Different TTNN ops often need distinct
    kernel/program/memory choices, so we capture them independently.
    """

    memory_config: Optional[Any] = None
    program_config: Optional[Any] = None
    compute_kernel_config: Optional[Any] = None


@dataclass
class AttentionConfig:
    """
    Device-facing configuration knobs for attention.

    Each TTNN op is configurable independently; callers may override
    any subset while using module-provided defaults for the rest.
    """

    qkv: OpConfig = field(default_factory=OpConfig)
    q: OpConfig = field(default_factory=OpConfig)
    k: OpConfig = field(default_factory=OpConfig)
    v: OpConfig = field(default_factory=OpConfig)
    attn_scores: OpConfig = field(default_factory=OpConfig)
    attn_output: OpConfig = field(default_factory=OpConfig)
    out_proj: OpConfig = field(default_factory=OpConfig)
    shard_layout: Optional[Any] = None


class AttentionCore(Protocol):
    """
    Minimal interface for TTTv2 attention cores.

    Implementations encapsulate a specific attention algorithm
    (e.g., standard MHA, flash attention, sliding-window attention)
    and consume TTNN tensors only.
    """

    spec: AttentionSpec
    config: AttentionConfig

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        **kwargs: Any,
    ) -> ttnn.Tensor:
        """
        Apply attention to `hidden_states` with an optional `attention_mask`.

        Implementations may support additional keyword-only arguments
        (e.g., KV cache handles, rotary embeddings, prefill/decode mode),
        but must keep the core contract TTNN-only and model-agnostic.
        """
        ...


class BaseAttentionCore:
    """
    Simple base implementation of `AttentionCore`.

    This is a convenient starting point for concrete implementations
    and can be used as a placeholder while wiring higher-level code.
    """

    def __init__(self, spec: AttentionSpec, config: Optional[AttentionConfig] = None):
        self.spec = spec
        self.config = config or AttentionConfig()

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        **kwargs: Any,
    ) -> ttnn.Tensor:
        raise NotImplementedError("BaseAttentionCore.forward must be implemented by subclasses.")


def build_attention_core(
    spec: AttentionSpec,
    config: Optional[AttentionConfig] = None,
) -> AttentionCore:
    """
    Factory for constructing a default attention core.

    For now this returns a `BaseAttentionCore` stub so callers can
    depend on a stable constructor shape; future work can swap in
    a concrete TTNN implementation (e.g., standard MHA or flash attention)
    without changing call sites.
    """
    return BaseAttentionCore(spec=spec, config=config)
