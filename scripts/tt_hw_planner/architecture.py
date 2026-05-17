# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Architecture-aware memory models.

The base class `MemoryModel` defines the memory contract every transformer-
family architecture must implement.  Subclasses override the parts that
differ.  Detection happens in `select_model()` based on config fields probed
from the HuggingFace `config.json`.

  - DenseTransformerModel  : standard GQA decoder (Llama, Qwen, Mistral)
  - MLATransformerModel    : Multi-head Latent Attention (DeepSeek-V2/V3/V4)
                             — KV is compressed to a latent vector, 10-20x smaller
  - SlidingWindowModel     : sliding-window-only attention (early Mistral,
                             window-only Phi-3) — KV grows up to window, not seq
  - SSMModel               : state-space models (Mamba, Mamba-2, RWKV) —
                             fixed-size state, no per-token KV growth
  - MoEModel               : Mixture-of-Experts dense outer + sparse experts;
                             only `experts_per_token` are active per token,
                             but all expert weights must be resident.

References:
  - Standard KV formula:        Vaswani 2017, "Attention is All You Need"
  - Activation coefficient ~12: Korthikanti 2022, "Reducing Activation
                                  Recomputation in Large Transformer Models"
  - MLA compressed KV:          DeepSeek-V2 paper (Liu et al., 2024)
  - Mamba state:                Gu & Dao 2023, "Mamba: Linear-Time Sequence
                                  Modeling with Selective State Spaces"

All sizes are returned in *bytes* (callers convert to GB).  The model never
divides by parallelism — that happens in parallelism.py.  Each method takes
the raw model-level memory before sharding.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional


# bytes per element for dtypes we report
DTYPE_BYTES = {
    "bf16": 2.0,
    "fp16": 2.0,
    "fp32": 4.0,
    "bfp8_b": 1.0625,  # 8 bits + shared exponent per 16-elem block (1 bit/elem)
    "bfp4_b": 0.5625,  # 4 bits + shared exponent per 16-elem block (1 bit/elem)
    "fp8": 1.0,
}


@dataclass
class ArchitectureSpec:
    """The narrow set of config fields every memory model needs."""

    family: str  # "dense" | "mla" | "sliding_window" | "ssm" | "moe"
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int = 0
    max_position_embeddings: int = 0

    # MLA-only
    kv_lora_rank: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None

    # Sliding-window-only
    sliding_window: Optional[int] = None
    global_attention_layers: int = 0  # full-attention layers (0 = pure SW)

    # SSM-only
    state_size: Optional[int] = None
    conv_kernel: Optional[int] = None

    # MoE-only
    num_experts: Optional[int] = None
    experts_per_token: Optional[int] = None
    moe_intermediate_size: Optional[int] = None


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class MemoryModel(ABC):
    """Memory model for a transformer-family architecture."""

    def __init__(self, arch: ArchitectureSpec, total_params: int, weight_bytes_on_disk: int):
        self.arch = arch
        self.total_params = total_params
        self.weight_bytes_on_disk = weight_bytes_on_disk

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    def family(self) -> str:
        ...

    def weights_bytes(self, dtype: str) -> int:
        """
        Bytes occupied by model weights at the requested target dtype.

        We scale the on-disk footprint by `target_per_param / source_per_param`:
        this correctly handles all four directions:
          - bf16 on disk, target bf16   → same size
          - bf16 on disk, target bfp8_b → ~half
          - fp8  on disk, target bf16   → ~2x  (dequantize at load time)
          - fp8  on disk, target bfp8_b → ~same (tt-metal stores as bfp8_b natively)
        """
        target_per_param = DTYPE_BYTES.get(dtype)
        if target_per_param is None:
            raise ValueError(f"unknown dtype: {dtype}")
        source_per_param = (self.weight_bytes_on_disk / max(self.total_params, 1)) or target_per_param
        return int(self.weight_bytes_on_disk * (target_per_param / source_per_param))

    @abstractmethod
    def kv_cache_bytes(self, batch: int, seq: int, kv_dtype_bytes: float = 2.0) -> int:
        ...

    def activation_bytes(self, batch: int, seq: int, dtype: str = "bf16") -> int:
        """
        Peak activation memory for a single layer's forward pass.

        Default uses the Korthikanti 2022 coefficient (~12 hidden-state-
        equivalents in flight per token).  Subclasses override when the
        architecture's activation pattern differs materially (SSMs use a
        recurrent activation, MLA needs a Q-decompression buffer, etc.).
        """
        bytes_per_elem = DTYPE_BYTES.get(dtype, 2.0)
        return int(batch * seq * self.arch.hidden_size * 12 * bytes_per_elem)


# ---------------------------------------------------------------------------
# Dense transformer (Llama/Qwen/Mistral GQA)
# ---------------------------------------------------------------------------


class DenseTransformerModel(MemoryModel):
    """Standard transformer with Grouped-Query Attention."""

    def family(self) -> str:
        return "dense"

    def kv_cache_bytes(self, batch: int, seq: int, kv_dtype_bytes: float = 2.0) -> int:
        # 2 (K and V) * batch * seq * kv_heads * head_dim * layers
        return int(
            2 * batch * seq * self.arch.num_key_value_heads * self.arch.head_dim * self.arch.num_layers * kv_dtype_bytes
        )


# ---------------------------------------------------------------------------
# MLA (DeepSeek-V2/V3/V4)
# ---------------------------------------------------------------------------


class MLATransformerModel(MemoryModel):
    """
    Multi-head Latent Attention.

    The full K, V tensors are reconstructed on the fly from a compressed
    latent of dim `kv_lora_rank`.  The cache stores only the latent + the
    RoPE-rotated portion of K.

    Per-token cached state:  kv_lora_rank + qk_rope_head_dim
    (vs. dense:              num_kv_heads * head_dim)

    For DeepSeek-V3 this is ~512+64 = 576 elems vs. 8*128 = 1024 elems for
    a comparable GQA layer — roughly 1.8x smaller KV cache.  For larger
    head counts the savings are dramatically larger.
    """

    def family(self) -> str:
        return "mla"

    def kv_cache_bytes(self, batch: int, seq: int, kv_dtype_bytes: float = 2.0) -> int:
        lora = self.arch.kv_lora_rank or self.arch.head_dim
        rope = self.arch.qk_rope_head_dim or 0
        per_token = lora + rope
        return int(batch * seq * per_token * self.arch.num_layers * kv_dtype_bytes)


# ---------------------------------------------------------------------------
# Sliding-window-only attention (early Mistral, parts of Phi-3)
# ---------------------------------------------------------------------------


class SlidingWindowModel(MemoryModel):
    """
    Sliding-window attention — KV cache per layer is capped at `sliding_window`
    tokens, NOT total sequence length.

    Note: many recent models (e.g. Llama 4-class) use INTERLEAVED sliding
    windows (some layers global, others windowed).  If `global_attention_layers
    > 0` we model that: `global` layers see full seq, `local` layers see window.
    """

    def family(self) -> str:
        return "sliding_window"

    def kv_cache_bytes(self, batch: int, seq: int, kv_dtype_bytes: float = 2.0) -> int:
        window = self.arch.sliding_window or seq
        per_token_bytes = 2 * self.arch.num_key_value_heads * self.arch.head_dim * kv_dtype_bytes

        n_global = self.arch.global_attention_layers
        n_local = self.arch.num_layers - n_global

        local_seq = min(seq, window)
        global_kv = batch * seq * per_token_bytes * n_global
        local_kv = batch * local_seq * per_token_bytes * n_local
        return int(global_kv + local_kv)


# ---------------------------------------------------------------------------
# State-Space Models (Mamba, RWKV, Falcon-Mamba)
# ---------------------------------------------------------------------------


class SSMModel(MemoryModel):
    """
    State-space models have no per-token KV cache.  They carry a fixed-size
    recurrent state of dim ~ (hidden * state_size) per layer per batch.
    The state size is constant in `seq` — that's the whole point of SSMs.
    """

    def family(self) -> str:
        return "ssm"

    def kv_cache_bytes(self, batch: int, seq: int, kv_dtype_bytes: float = 2.0) -> int:
        state = self.arch.state_size or 16  # Mamba default
        per_layer_state = self.arch.hidden_size * state
        # The conv-kernel state adds (conv_kernel - 1) * hidden per layer.
        if self.arch.conv_kernel:
            per_layer_state += (self.arch.conv_kernel - 1) * self.arch.hidden_size
        return int(batch * per_layer_state * self.arch.num_layers * kv_dtype_bytes)


# ---------------------------------------------------------------------------
# MoE (Mixtral, Qwen3-Coder-A3B, DeepSeek-V2/V3/V4)
# ---------------------------------------------------------------------------


class MoEModel(MemoryModel):
    """
    MoE inherits the parent dense memory model for KV/activations but
    flags that weights cannot be sharded the same way.  Concretely:
      - All N experts must be *resident* (every token may route there);
        expert parallelism EP shards experts across chips orthogonally to
        TP, but until parallelism.py implements EP search, we conservatively
        keep `weights_bytes` at the full footprint.
      - Activation memory is dominated by the routed forward path, NOT N
        experts in parallel — so the parent activation formula is correct.
    """

    def __init__(self, *args, base: MemoryModel, **kwargs):
        super().__init__(*args, **kwargs)
        self.base = base

    def family(self) -> str:
        return f"moe+{self.base.family()}"

    def kv_cache_bytes(self, batch: int, seq: int, kv_dtype_bytes: float = 2.0) -> int:
        return self.base.kv_cache_bytes(batch, seq, kv_dtype_bytes)

    def activation_bytes(self, batch: int, seq: int, dtype: str = "bf16") -> int:
        return self.base.activation_bytes(batch, seq, dtype)


# ---------------------------------------------------------------------------
# Detection / dispatch
# ---------------------------------------------------------------------------


def select_model(arch: ArchitectureSpec, total_params: int, weight_bytes_on_disk: int) -> MemoryModel:
    """Pick the right MemoryModel subclass based on architecture fields."""

    # MLA → DeepSeek-family
    if arch.kv_lora_rank is not None and arch.kv_lora_rank > 0:
        base: MemoryModel = MLATransformerModel(arch, total_params, weight_bytes_on_disk)
    # SSM → Mamba / RWKV
    elif arch.state_size is not None or arch.family == "ssm":
        base = SSMModel(arch, total_params, weight_bytes_on_disk)
    # Sliding-window
    elif arch.sliding_window is not None and arch.sliding_window > 0:
        base = SlidingWindowModel(arch, total_params, weight_bytes_on_disk)
    else:
        base = DenseTransformerModel(arch, total_params, weight_bytes_on_disk)

    # MoE wraps whichever base we picked
    if arch.num_experts and arch.num_experts > 1:
        return MoEModel(arch, total_params, weight_bytes_on_disk, base=base)
    return base


# ---------------------------------------------------------------------------
# Architecture detection from HF config
# ---------------------------------------------------------------------------

# SSM-class model_types we recognise.
SSM_MODEL_TYPES = {"mamba", "mamba2", "rwkv", "rwkv4", "rwkv5", "rwkv6", "falcon_mamba"}


def detect_architecture(cfg: dict) -> str:
    """Return one of: dense | mla | sliding_window | ssm | moe | unknown."""
    if not cfg:
        return "unknown"
    model_type = (cfg.get("model_type") or "").lower()

    if model_type in SSM_MODEL_TYPES:
        return "ssm"
    if cfg.get("kv_lora_rank"):  # DeepSeek MLA marker
        return "mla"
    if cfg.get("num_local_experts") or cfg.get("n_routed_experts"):
        return "moe"
    if cfg.get("sliding_window") and cfg.get("sliding_window") > 0:
        return "sliding_window"
    return "dense"


def build_arch_spec(cfg: dict, family: str) -> ArchitectureSpec:
    """Pull the relevant fields out of a HF config dict into ArchitectureSpec."""
    H = cfg.get("hidden_size") or cfg.get("d_model") or 0
    Q = cfg.get("num_attention_heads") or cfg.get("n_head") or 1
    KV = cfg.get("num_key_value_heads", Q)
    head_dim = cfg.get("head_dim") or (H // Q if Q else 0)
    L = cfg.get("num_hidden_layers") or cfg.get("num_layers") or cfg.get("n_layer") or 0

    # Detect interleaved global/local attention if the config exposes it.
    n_global = cfg.get("num_global_layers") or 0
    if "attention_layers" in cfg and isinstance(cfg["attention_layers"], list):
        # Some configs list per-layer attention type explicitly.
        n_global = sum(1 for t in cfg["attention_layers"] if t in ("global", "full"))

    return ArchitectureSpec(
        family=family,
        num_layers=L,
        hidden_size=H,
        num_attention_heads=Q,
        num_key_value_heads=KV,
        head_dim=head_dim,
        vocab_size=cfg.get("vocab_size", 0),
        max_position_embeddings=cfg.get("max_position_embeddings", 0),
        kv_lora_rank=cfg.get("kv_lora_rank"),
        qk_rope_head_dim=cfg.get("qk_rope_head_dim"),
        sliding_window=cfg.get("sliding_window"),
        global_attention_layers=n_global,
        state_size=cfg.get("state_size") or cfg.get("d_state"),
        conv_kernel=cfg.get("conv_kernel") or cfg.get("d_conv"),
        num_experts=(cfg.get("num_local_experts") or cfg.get("n_routed_experts") or cfg.get("num_experts")),
        # `top_k` ALSO names HF's sampling top-k; only read it when an
        # MoE-specific marker is present (otherwise we read sampling top_k
        # as "experts per token" for dense models).
        experts_per_token=(
            cfg.get("num_experts_per_tok")
            or (
                cfg.get("top_k")
                if (cfg.get("num_local_experts") or cfg.get("n_routed_experts") or cfg.get("num_experts"))
                else None
            )
        ),
        moe_intermediate_size=cfg.get("moe_intermediate_size"),
    )
