# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Python models package for ttml.

This package provides Python implementations of models using ttml operations.
C++ types are imported from _ttml.models and must be available before Python
submodules that depend on them.
"""

# Import C++ types first (needed by Python submodules)
# Note: _ttml is a top-level module, not a subpackage of ttml
import _ttml
import sys
from enum import Enum

# --- C++ enums and classes from _ttml.models ---
from _ttml.models import (
    BaseTransformer,
    KvCache,
    KvCacheConfig,
    RunnerType,
    WeightTyingType,
    memory_efficient_runner,
)


# --- Python-side config enums (siblings of the C++ enums above) ---
class EmbeddingPlacement(Enum):
    """Placement of the token-embedding table across the tensor-parallel axis.

    A model-config discriminator (used like :class:`WeightTyingType`) that selects
    the embedding module the TP path builds — replicate the table, or shard it on
    the vocab or feature dimension. Only meaningful under tensor parallelism; the
    non-TP path always uses a plain replicated :class:`ttml.modules.Embedding`.

    Attributes:
        Replicated: Do not shard the table — every device holds the full copy (a
            plain :class:`ttml.modules.Embedding`). The lookup
            is fully local with no collective, and the replicated gradients already
            match across TP ranks (``synchronize_gradients`` reduces only the DDP
            axis). Weight tying is unavailable, since the
            tied weight would be the vocab-sharded LM head.
        VocabParallel: Shard the table on the vocabulary dimension, mirroring a
            vocab-parallel LM head. Layout-compatible with weight tying.
        FeatureParallel: Shard the table on the feature (hidden) dimension.
            Cheaper than :attr:`VocabParallel` (an all-gather instead of an
            all-reduce) with no id-masking logic, but its layout is incompatible
            with a vocab-parallel LM head, so weight tying is unavailable.
    """

    Replicated = "replicated"
    VocabParallel = "vocab_parallel"
    FeatureParallel = "feature_parallel"

    @classmethod
    def from_string(cls, s: str) -> "EmbeddingPlacement":
        """Parse a config string, accepting the canonical names or the short
        aliases (``"vocab"`` / ``"feature"`` / ``"none"`` / ``"disabled"`` /
        ``"off"``), case-insensitively."""
        aliases = {
            "replicated": cls.Replicated,
            "none": cls.Replicated,
            "disabled": cls.Replicated,
            "off": cls.Replicated,
            "vocab": cls.VocabParallel,
            "vocab_parallel": cls.VocabParallel,
            "feature": cls.FeatureParallel,
            "feature_parallel": cls.FeatureParallel,
        }
        try:
            return aliases[s.strip().lower()]
        except KeyError:
            raise ValueError(
                f"Unknown embedding_placement {s!r}; expected one of: {', '.join(sorted(aliases))}."
            ) from None


# --- C++ submodules without Python package counterparts ---
# Alias _ttml.models.* nanobind submodules so attribute access like
# ttml.models.gpt2.create_gpt2_model(...) works.
gpt2 = _ttml.models.gpt2
sys.modules[f"{__name__}.gpt2"] = gpt2

mlp = _ttml.models.mlp
sys.modules[f"{__name__}.mlp"] = mlp

distributed = _ttml.models.distributed
sys.modules[f"{__name__}.distributed"] = distributed

# --- Python implementations ---
from .linear_regression import LinearRegression, create_linear_regression_model
from .nanogpt import NanoGPT, NanoGPTConfig, create_nanogpt
from .llama import Llama, LlamaConfig
from .deepseek import DeepSeek, DeepSeekConfig
from .qwen3 import Qwen3, Qwen3Config

__all__ = [
    # C++ enums / classes
    "BaseTransformer",
    "KvCache",
    "KvCacheConfig",
    "RunnerType",
    "WeightTyingType",
    "memory_efficient_runner",
    # C++ submodules
    "distributed",
    "gpt2",
    "mlp",
    # Python implementations
    "DeepSeek",
    "DeepSeekConfig",
    "EmbeddingPlacement",
    "Llama",
    "LlamaConfig",
    "LinearRegression",
    "NanoGPT",
    "NanoGPTConfig",
    "Qwen3",
    "Qwen3Config",
    "create_linear_regression_model",
    "create_nanogpt",
]
