# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Build missing TTNN weight-cache files from the HF checkpoint, only for the categories the operator selects.

The production prefill runner is cache-only: it hands the model an empty ``state_dict`` and the model reads
``*.tensorbin`` files. A site whose cache predates a weight change (e.g. the centred MoE gate bias) or never had
a piece (final norm / LM head, the MTP tree) used to fail at start-up, and the only remedy was to rsync the
whole cache from another site.

For each rank the runner checks every category of its layer slice, and

* nothing missing -> load from cache, exactly as before;
* something missing -> read just those tensors from the checkpoint (``PREFILL_HF_MODEL``) and let the model write
  the cache while it builds. Nothing is ever built from ``torch.empty`` placeholders.

Categories (rank-local unless noted):

    embedding        token-embedding table                  (first rank, or last rank running MTP)
    norms            attn_norm / ffn_norm of every layer
    mla              MLA + lightning-indexer projections of every layer
    dense_ffn        FFN of the dense layers
    moe_gate         router weight + (centred) correction bias of every MoE layer
    shared_experts   shared expert of every MoE layer
    routed_experts   routed experts of every MoE layer     (by far the largest: 256 fp8 experts per layer)
    lm_head          final norm + LM head                   (last rank running MTP)
    mtp              the MTP predictor tree (its own cache root; last rank running MTP)
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

import torch
from loguru import logger

EMBEDDING = "embedding"
NORMS = "norms"
MLA = "mla"
DENSE_FFN = "dense_ffn"
MOE_GATE = "moe_gate"
SHARED_EXPERTS = "shared_experts"
ROUTED_EXPERTS = "routed_experts"
LM_HEAD = "lm_head"
MTP = "mtp"

CATEGORIES = (EMBEDDING, NORMS, MLA, DENSE_FFN, MOE_GATE, SHARED_EXPERTS, ROUTED_EXPERTS, LM_HEAD, MTP)
_PROJ = ("gate_proj", "up_proj", "down_proj")


# ---------------------------------------------------------------------------------------------------------------
# What is missing from the cache
# ---------------------------------------------------------------------------------------------------------------


def _mla_cache_complete(cache_path: Path, prefix: str, layer_idx: int, model_cfg, hf_config, tp_factor: int) -> bool:
    """Whether the MLA (+ indexer) cache of one layer is complete, as the model itself writes and reads it.

    ``ttMLA.check_cache_complete`` is not enough here: it always wants ``mla.wkv_b1``, but a sparse layer at TP > 1
    loads wkv_b1 replicated and a runner-built cache holds only ``mla_repl.wkv_b1`` (test-built caches hold both);
    and it checks no indexer files, which a ``full``-indexer layer needs (a ``shared`` one owns none).
    """
    from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
    from models.demos.deepseek_v3_d_p.tt.mla.indexer import TtIndexer, indexer_layer_is_reused, resolve_has_indexer
    from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import pattern_exists

    # Sparse (DSA) attention is a property of the model; a `shared` layer is still sparse (it replicates wkv_b1) but
    # owns no indexer weights, so only the indexer files depend on the layer's own type.
    sparse = hf_config is not None and resolve_has_indexer(hf_config)
    owns_indexer = sparse and not indexer_layer_is_reused(hf_config, layer_idx)
    replicated = sparse and ttMLA.replicates_wkv_b1(hf_config, tp_factor, sparse)
    for name in ttMLA.weight_names(bool(getattr(model_cfg, "USE_OUTPUT_GATE", False))):
        stem = f"{prefix}.mla_repl.{name}" if replicated and name == "wkv_b1" else f"{prefix}.mla.{name}"
        if not pattern_exists(f"{stem}*.tensorbin", "MLA"):
            return False
    return not owns_indexer or TtIndexer.check_cache_complete(cache_path, f"{prefix}.mla")


def missing_trunk_categories(
    cache_path: Path,
    *,
    num_layers: int,
    first_layer_idx: int,
    experts_per_chip: int,
    model_cfg: type,
    routed_expert_weights_dtype,
    is_first_rank: bool,
    is_last_rank: bool,
    mtp_levels: int,
    kv_only_last_layer: bool,
    hf_config=None,
    tp_factor: int = 1,
) -> dict[str, list[int]]:
    """``{category: [global layer idx, ...]}`` for every trunk cache piece this rank is missing.

    ``embedding`` / ``lm_head`` carry ``[-1]``. Uses the same per-component checks as
    ``TtPrefillTransformer.check_cache_complete``, one category at a time.
    """
    from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import TtMoEGatePrefill, gate_bias_centered
    from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
    from models.demos.deepseek_v3_d_p.tt.moe.tt_shared_expert import TtSharedExpert
    from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm
    from models.demos.deepseek_v3_d_p.tt.tt_ffn import TtFfn
    from models.demos.deepseek_v3_d_p.tt.tt_lm_head import TtLMHead
    from models.demos.deepseek_v3_d_p.tt.tt_parallel_embedding import TtParallelEmbedding
    from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import rank_loads_embedding
    from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker, pattern_exists

    missing: dict[str, list[int]] = {}

    def miss(category: str, layer: int) -> None:
        missing.setdefault(category, []).append(layer)

    cache_path = Path(cache_path)
    if cache_path.is_dir():
        init_checker(cache_path)
    # A cache dir that does not exist yet is simply "everything missing"; the per-file checks below then report
    # False without needing the fast checker.
    exists = cache_path.is_dir()

    if rank_loads_embedding(is_first_rank, is_last_rank, mtp_levels):
        if not (exists and TtParallelEmbedding.check_cache_complete(cache_path)):
            miss(EMBEDDING, -1)

    first_k_dense = model_cfg.NUM_DENSE_LAYERS
    last_layer_idx = first_layer_idx + num_layers - 1
    for layer_idx in range(first_layer_idx, first_layer_idx + num_layers):
        prefix = f"layer_{layer_idx}"
        if kv_only_last_layer and is_last_rank and layer_idx == last_layer_idx:
            # Built kv-only: the model reads attn_norm and the KV projections of this layer and nothing else, so
            # that is all a runner-built cache ever holds for it (no Q/indexer/output projections, FFN or MoE).
            if not (exists and TtDistributedRmsNorm.check_cache_complete(cache_path, f"{prefix}.attn_norm")):
                miss(NORMS, layer_idx)
            if not (
                exists
                and all(
                    pattern_exists(f"{prefix}.mla.{name}*.tensorbin", "MLA")
                    for name in ("kv_a_layernorm", "kv_a_proj_with_mqa")
                )
            ):
                miss(MLA, layer_idx)
            continue
        if not (
            exists
            and TtDistributedRmsNorm.check_cache_complete(cache_path, f"{prefix}.attn_norm")
            and TtDistributedRmsNorm.check_cache_complete(cache_path, f"{prefix}.ffn_norm")
        ):
            miss(NORMS, layer_idx)
        if not (exists and _mla_cache_complete(cache_path, prefix, layer_idx, model_cfg, hf_config, tp_factor)):
            miss(MLA, layer_idx)
        if layer_idx < first_k_dense:
            if not (exists and TtFfn.check_cache_complete(cache_path, f"{prefix}.ffn")):
                miss(DENSE_FFN, layer_idx)
            continue
        if not (
            exists
            and TtMoEGatePrefill.check_cache_complete(
                cache_path, f"{prefix}.gate", bias_centered=gate_bias_centered(model_cfg)
            )
        ):
            miss(MOE_GATE, layer_idx)
        if not (exists and TtSharedExpert.check_cache_complete(cache_path, f"{prefix}.shared_expert")):
            miss(SHARED_EXPERTS, layer_idx)
        if not (
            exists
            and TtRoutedExpert.check_cache_complete(
                cache_path, f"{prefix}.routed_expert", experts_per_chip, routed_expert_weights_dtype
            )
        ):
            miss(ROUTED_EXPERTS, layer_idx)

    if mtp_levels and is_last_rank and not kv_only_last_layer:
        if not (
            exists
            and TtDistributedRmsNorm.check_cache_complete(cache_path, "norm")
            and TtLMHead.check_cache_complete(cache_path)
        ):
            miss(LM_HEAD, -1)
    return missing


def format_missing(missing: dict[str, list[int]]) -> str:
    parts = []
    for category, layers in missing.items():
        real = [i for i in layers if i >= 0]
        parts.append(f"{category}" + (f" (layers {real[0]}..{real[-1]}, {len(real)} total)" if real else ""))
    return ", ".join(parts)


# ---------------------------------------------------------------------------------------------------------------
# Reading the checkpoint
# ---------------------------------------------------------------------------------------------------------------


def checkpoint_available(path: str | Path) -> bool:
    """Whether ``path`` holds a sharded safetensors checkpoint to convert from. A config-only directory does not."""
    return bool(path) and (Path(path) / "model.safetensors.index.json").is_file()


class _Checkpoint:
    """Tensor-level reader over a sharded HF safetensors checkpoint. Opens only the shards a request touches."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        index = self.path / "model.safetensors.index.json"
        if not index.is_file():
            raise FileNotFoundError(
                f"{index} not found: rebuilding the missing weight cache needs the FP8 checkpoint "
                "(set PREFILL_HF_MODEL to the directory holding model.safetensors.index.json)"
            )
        self.weight_map: dict[str, str] = json.loads(index.read_text())["weight_map"]

    def get(self, key: str) -> torch.Tensor:
        from safetensors import safe_open

        shard = self.weight_map.get(key)
        if shard is None:
            raise KeyError(f"{key} is not in the checkpoint at {self.path}")
        with safe_open(str(self.path / shard), framework="pt") as f:
            return f.get_tensor(key)

    def bf16(self, key: str) -> torch.Tensor:
        """The tensor as bf16, undoing the per-block FP8 scale when the checkpoint stored it quantised."""
        from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32.weights import _dequant_fp8

        t = self.get(key)
        if t.dtype == torch.float8_e4m3fn:
            t = _dequant_fp8(t, self.get(f"{key}_scale_inv"))
        return t.to(torch.bfloat16)

    def shards_with(self, prefix: str) -> list[str]:
        return [str(self.path / s) for s in sorted({v for k, v in self.weight_map.items() if k.startswith(prefix)})]


def _layer_state_dict(ckpt: _Checkpoint, config, model_cfg, layer_idx: int, categories: set[str]) -> dict:
    """One decoder layer's ``TtPrefillBlock`` state dict, holding only ``categories``."""
    prefix = f"model.layers.{layer_idx}."
    sd: dict = {}
    if NORMS in categories:
        sd["attn_norm_weight"] = ckpt.get(f"{prefix}input_layernorm.weight").to(torch.bfloat16)
        sd["ffn_norm_weight"] = ckpt.get(f"{prefix}post_attention_layernorm.weight").to(torch.bfloat16)
    if MLA in categories:
        from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32.reference import pretrained_mla_weights

        shards = ckpt.shards_with(f"{prefix}self_attn.")
        assert shards, f"no self_attn tensors for layer {layer_idx} in {ckpt.path}"
        sd["mla_weights"] = pretrained_mla_weights(config, layer=layer_idx, checkpoint_path=shards)
    if DENSE_FFN in categories:
        sd["ffn_weights"] = {p: ckpt.bf16(f"{prefix}mlp.{p}.weight") for p in _PROJ}
    if MOE_GATE in categories:
        sd["gate_weights"] = {
            "weight": ckpt.get(f"{prefix}mlp.gate.weight").to(torch.bfloat16),
            "e_score_correction_bias": ckpt.get(f"{prefix}mlp.gate.e_score_correction_bias").float(),
        }
    if SHARED_EXPERTS in categories:
        sd["shared_expert_weights"] = {p: ckpt.bf16(f"{prefix}mlp.shared_experts.{p}.weight") for p in _PROJ}
    if ROUTED_EXPERTS in categories:
        sd["routed_expert_weights"] = [
            {p: ckpt.bf16(f"{prefix}mlp.experts.{e}.{p}.weight") for p in _PROJ}
            for e in range(model_cfg.NUM_ROUTED_EXPERTS)
        ]
    return sd


class _LazyLayers(Sequence):
    """``state_dict["layers"]`` that reads a layer from the checkpoint when the model first indexes it.

    The model builds layers one after another, so only one layer's tensors (up to ~20 GB with routed experts) are
    ever resident: the previous layer is dropped when the next is requested.
    """

    def __init__(self, ckpt, config, model_cfg, first_layer_idx: int, num_layers: int, wanted: dict[int, set[str]]):
        self._args = (ckpt, config, model_cfg)
        self._first = first_layer_idx
        self._n = num_layers
        self._wanted = wanted
        self._held: tuple[int, dict] | None = None

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, local_idx: int) -> dict:
        if not 0 <= local_idx < self._n:
            raise IndexError(local_idx)
        if self._held is not None and self._held[0] == local_idx:
            return self._held[1]
        self._held = None  # free the previous layer before reading this one
        layer_idx = self._first + local_idx
        cats = self._wanted.get(layer_idx, set())
        if cats:
            logger.warning(
                f"Weight-cache rebuild: reading layer {layer_idx} ({local_idx + 1}/{self._n} on this rank) from the "
                f"checkpoint for categories {sorted(cats)}"
            )
        sd = _layer_state_dict(*self._args, layer_idx, cats)
        self._held = (local_idx, sd)
        return sd


def load_trunk_state_dict(
    checkpoint_path: str | Path,
    config,
    model_cfg: type,
    *,
    missing: dict[str, list[int]],
    first_layer_idx: int,
    num_layers: int,
) -> dict:
    """The partial ``TtPrefillTransformer`` state dict covering exactly what ``missing`` lists."""
    ckpt = _Checkpoint(checkpoint_path)
    wanted: dict[int, set[str]] = {}
    for category, layers in missing.items():
        for layer in layers:
            if layer >= 0:
                wanted.setdefault(layer, set()).add(category)

    sd: dict = {}
    if EMBEDDING in missing:
        sd["embed_weight"] = ckpt.get("model.embed_tokens.weight").to(torch.bfloat16).float()
    if LM_HEAD in missing:
        sd["norm_weight"] = ckpt.bf16("model.norm.weight")
        sd["lm_head_weight"] = ckpt.bf16("lm_head.weight")
    if wanted:
        sd["layers"] = _LazyLayers(ckpt, config, model_cfg, first_layer_idx, num_layers, wanted)
    return sd


def load_mtp_state_dict(checkpoint_path: str | Path, config, model_cfg: type, mtp_layer_idx: int) -> dict:
    """``{"mtp": {...}, "layer": {...}}`` for ``TtMTPPredictor``: the fused projection tensors and the MTP layer's
    own decoder block (MLA + indexer, both norms, the router, shared expert and all routed experts)."""
    from models.demos.deepseek_v3_d_p.tt.mtp_prefill.utils import load_mtp_state_dict as load_mtp_tensors

    mtp = {k: v.to(torch.bfloat16) for k, v in load_mtp_tensors(str(checkpoint_path), layer_idx=mtp_layer_idx).items()}
    ckpt = _Checkpoint(checkpoint_path)
    layer = _layer_state_dict(
        ckpt, config, model_cfg, mtp_layer_idx, {NORMS, MLA, MOE_GATE, SHARED_EXPERTS, ROUTED_EXPERTS}
    )
    return {"mtp": mtp, "layer": layer}
