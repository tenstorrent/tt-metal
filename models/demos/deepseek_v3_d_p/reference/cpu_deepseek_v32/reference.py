# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Public API for the DeepSeek-V3.2 / GLM-5.1 sparse-MLA CPU reference.

Shaped after the device model ``ttMLA``: construct with ``(config, weights)`` plus ``seq_len``, then
``forward(hidden_states)``. The HF-attribute config is the only configuration input; the upstream
``ModelArgs`` is derived from it internally and never leaves the package. A single canonical weights
dict (HF/checkpoint naming — the same one ``ttMLA`` consumes) initializes both backends; this module
remaps it to the upstream CPU parameter names on load.

See ``API_SPEC.md`` in this directory for the full design.
"""

from __future__ import annotations

from os import PathLike
from typing import Protocol, runtime_checkable

import torch

from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32.model import MLACPU, ModelArgs
from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32.utils import precompute_freqs_cis
from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32.weights import (
    CANONICAL_TO_CPU,
    DEFAULT_REPO,
    load_attention_state_dict,
    resolve_layer_shards,
)

# Canonical weights: keyed by HF/checkpoint names — the exact dict ttMLA consumes.
Weights = dict[str, torch.Tensor]


@runtime_checkable
class SparseMLAConfig(Protocol):
    """The HF-attribute config contract this package reads (deepseek_v32_hf_config / glm_hf_config)."""

    hidden_size: int
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    rope_theta: float
    rope_scaling: dict  # factor, mscale, beta_fast, beta_slow, original_max_position_embeddings
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    index_rope_interleave: bool


# Canonical (HF/checkpoint) <-> CPU-port name maps. The correspondence is owned by weights.py
# (CANONICAL_TO_CPU); here we just take it and its inverse so device weights and the CPU truth agree.
_CANONICAL_TO_CPU = CANONICAL_TO_CPU
_CPU_TO_CANONICAL: dict[str, str] = {cpu: canon for canon, cpu in _CANONICAL_TO_CPU.items()}

# Canonical key set, for callers that want to validate against the ttnn contract.
CANONICAL_WEIGHT_NAMES: tuple[str, ...] = tuple(_CANONICAL_TO_CPU)


def _model_args_from_hf(config: SparseMLAConfig, seq_len: int) -> ModelArgs:
    """Derive the upstream ModelArgs from the HF config (the single source of dims).

    YaRN is gated inside MLACPU by ``max_seq_len > original_seq_len``; we drive that gate from
    ``rope_scaling.factor`` (not from ``config.max_seq_len``, which tests mutate to the device rope-table
    length). ``seq_len`` sizes the RoPE table + the internal KV/PE cache. Fields the config doesn't
    carry (``scale_fmt``) keep the ModelArgs defaults.
    """
    rs = config.rope_scaling
    original = int(rs["original_max_position_embeddings"])
    factor = float(rs["factor"])
    if factor > 1.0:  # YaRN active (V3.2): gate must be True
        original_seq_len = original
        max_seq_len = max(int(seq_len), original + 1)
    else:  # no YaRN (GLM): gate must be False (max == original)
        original_seq_len = max_seq_len = max(int(seq_len), original)
    return ModelArgs(
        max_batch_size=1,
        max_seq_len=max_seq_len,
        original_seq_len=original_seq_len,
        dim=config.hidden_size,
        n_heads=config.num_attention_heads,
        q_lora_rank=config.q_lora_rank,
        kv_lora_rank=config.kv_lora_rank,
        qk_nope_head_dim=config.qk_nope_head_dim,
        qk_rope_head_dim=config.qk_rope_head_dim,
        v_head_dim=config.v_head_dim,
        rope_theta=float(config.rope_theta),
        rope_factor=factor,
        beta_fast=int(rs["beta_fast"]),
        beta_slow=int(rs["beta_slow"]),
        mscale=float(rs["mscale"]),
        index_n_heads=config.index_n_heads,
        index_head_dim=config.index_head_dim,
        index_topk=config.index_topk,
        index_rope_interleave=config.index_rope_interleave,
    )


def random_mla_weights(config: SparseMLAConfig, *, seed: int = 42) -> Weights:
    """Config-driven random MLA + indexer weights in canonical naming (no model, no caches).

    Dtypes match the upstream modules: bf16 Linear weights, fp32 norms (γ) / k_norm bias (β) /
    ``weights_proj``. Cheap enough for the perf path (50k seq) since nothing is allocated but the
    weights themselves.
    """
    g = torch.Generator().manual_seed(seed)
    std = float(getattr(config, "initializer_range", 0.02))
    h = config.hidden_size
    n_heads = config.num_attention_heads
    qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    idx_heads = config.index_n_heads
    idx_dim = config.index_head_dim

    def lin(out_f: int, in_f: int) -> torch.Tensor:
        return (torch.randn(out_f, in_f, generator=g) * std).to(torch.bfloat16)

    return {
        "q_a_proj.weight": lin(config.q_lora_rank, h),
        "q_a_layernorm.weight": torch.ones(config.q_lora_rank, dtype=torch.float32),
        "q_b_proj.weight": lin(n_heads * qk_head_dim, config.q_lora_rank),
        "kv_a_proj_with_mqa.weight": lin(config.kv_lora_rank + config.qk_rope_head_dim, h),
        "kv_a_layernorm.weight": torch.ones(config.kv_lora_rank, dtype=torch.float32),
        "kv_b_proj.weight": lin(n_heads * (config.qk_nope_head_dim + config.v_head_dim), config.kv_lora_rank),
        "o_proj.weight": lin(h, n_heads * config.v_head_dim),
        "indexer.wq_b.weight": lin(idx_heads * idx_dim, config.q_lora_rank),
        "indexer.wk.weight": lin(idx_dim, h),
        "indexer.k_norm.weight": torch.ones(idx_dim, dtype=torch.float32),
        "indexer.k_norm_bias.weight": torch.zeros(idx_dim, dtype=torch.float32),
        "indexer.weights_proj.weight": (torch.randn(idx_heads, h, generator=g) * std).to(torch.float32),
    }


def pretrained_mla_weights(
    config: SparseMLAConfig,
    *,
    layer: int,
    repo: str = DEFAULT_REPO,
    checkpoint_path: str | PathLike[str] | list[str] | None = None,
) -> Weights:
    """Load layer-``layer`` pretrained MLA + indexer weights as the canonical dict (fp8 dequantized).

    ``checkpoint_path`` (a local shard path or list) takes precedence over HF resolution from ``repo``.
    ``config`` is accepted for symmetry / future validation; shapes come from the checkpoint.
    """
    shards = checkpoint_path if checkpoint_path is not None else resolve_layer_shards(layer, repo)
    cpu_sd = load_attention_state_dict(shards, layer)  # upstream CPU names, dequantized
    return {_CPU_TO_CANONICAL[cpu]: t for cpu, t in cpu_sd.items()}


class SparseMLAReference:
    """CPU truth for one DeepSeek-V3.2 / GLM-5.1 sparse-MLA layer. Built like ttMLA: ``(config, weights)``."""

    def __init__(
        self,
        config: SparseMLAConfig,
        weights: Weights,
        *,
        seq_len: int,
        simulate_fp8: bool = False,
    ) -> None:
        self._seq_len = int(seq_len)
        self._args = _model_args_from_hf(config, seq_len)
        # simulate_fp8=False (default) is the bf16 functional-parity path the ttnn port matches.
        self._mla = MLACPU(self._args, simulate_fp8=simulate_fp8).eval()
        self._mla.indexer.use_fp8_path = simulate_fp8

        missing_canon = [c for c in _CANONICAL_TO_CPU if c not in weights]
        if missing_canon:
            raise KeyError(f"weights missing canonical keys: {missing_canon}")
        cpu_weights = {_CANONICAL_TO_CPU[c]: t for c, t in weights.items() if c in _CANONICAL_TO_CPU}
        self._mla.load_state_dict(cpu_weights, strict=True)

        self._freqs = precompute_freqs_cis(self._args)
        self._filled = 0  # highest end position written into the caches (across chunked calls)
        self._last_topk: torch.Tensor | None = None
        self._last_logits: torch.Tensor | None = None
        # Capture the indexer's (topk, logits) from inside the single forward pass — no second run.
        self._mla.indexer.register_forward_hook(self._capture_indexer)

    def _capture_indexer(self, module, inputs, output) -> None:  # noqa: ANN001 (torch hook signature)
        topk, logits = output
        self._last_topk, self._last_logits = topk, logits

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        actual_start: int = 0,
        actual_end: int | None = None,
    ) -> torch.Tensor:
        """Single forward pass; returns the MLA output [B, S, dim].

        Mirrors ``ttMLA.forward``: ``actual_start``/``actual_end`` are the cache write offsets. RoPE and
        the causal mask are built internally. The indexer runs inside this pass — its logits/topk and the
        KV/index caches are exposed via the properties below. Chunked prefill = call this in a loop with
        increasing ``actual_start`` (the same pattern as the device chunked test).
        """
        x = hidden_states.to(torch.bfloat16)
        seqlen = x.shape[-2]
        end = int(actual_end) if actual_end is not None else actual_start + seqlen
        assert end <= self._seq_len, f"end {end} exceeds seq_len {self._seq_len}"
        freqs = self._freqs[actual_start:end]
        # Causal mask [seqlen, end]; triu offset by actual_start keeps chunk causality (== single-shot at 0).
        mask = torch.full((seqlen, end), float("-inf")).triu_(actual_start + 1)
        with torch.no_grad():
            out = self._mla.forward(x, actual_start, freqs, mask)
        self._filled = max(self._filled, end)
        return out

    @property
    def indexer_logits(self) -> torch.Tensor:
        """Indexer index-scores from the last forward [1, S, end] (causal region matches the reference)."""
        assert self._last_logits is not None, "call forward() first"
        return self._last_logits

    @property
    def indexer_topk(self) -> torch.Tensor:
        """Indexer top-k selected indices from the last forward [1, S, k]."""
        assert self._last_topk is not None, "call forward() first"
        return self._last_topk

    @property
    def kvpe_cache(self) -> torch.Tensor:
        """Latent-kv ++ k_pe in device layout [1, 1, S, kv_lora_rank + qk_rope_head_dim]."""
        kv = self._mla.kv_cache[:1, : self._filled]
        pe = self._mla.pe_cache[:1, : self._filled]
        return torch.cat([kv, pe], dim=-1).unsqueeze(1)

    @property
    def index_cache(self) -> torch.Tensor:
        """Indexer key cache [1, S, index_head_dim]."""
        return self._mla.indexer.k_cache[:1, : self._filled]
