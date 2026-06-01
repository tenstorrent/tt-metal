"""Lazy safetensors weight loader for DeepSeek-V4-Flash.

The checkpoint ships as 46 sharded .safetensors files plus a
``model.safetensors.index.json`` describing which shard owns each tensor. This
module exposes a small loader that:

* resolves the snapshot directory (supports the HuggingFace cache layout),
* parses the index once and caches `safe_open` handles per shard,
* yields individual tensors on demand (no shard is materialised in RAM until
  a tensor from it is requested),
* maps HF-style parameter names (e.g. ``model.embed_tokens.weight``) to the
  native names used inside the checkpoint (e.g. ``embed.weight``).

The HF -> checkpoint name map is derived from ``modular_deepseek_v4.py`` and
the tensor listing produced by ``list_deepseek_tensors.py``.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import torch
from safetensors import safe_open


INDEX_FILENAME = "model.safetensors.index.json"


def resolve_snapshot_dir(model_dir: Path) -> Path:
    """Return the directory that contains the .safetensors shards.

    Accepts either a flat directory or the HuggingFace cache layout
    (``<repo>/snapshots/<hash>/``).
    """
    model_dir = Path(model_dir)
    if (model_dir / INDEX_FILENAME).is_file() or any(model_dir.glob("*.safetensors")):
        return model_dir

    snapshots = model_dir / "snapshots"
    if snapshots.is_dir():
        for candidate in sorted(p for p in snapshots.iterdir() if p.is_dir()):
            if (candidate / INDEX_FILENAME).is_file() or any(candidate.glob("*.safetensors")):
                return candidate

    raise FileNotFoundError(f"Could not find safetensors shards (or {INDEX_FILENAME}) under {model_dir}")


# (regex pattern, replacement) tuples applied in order to translate an
# HF-style parameter name into the native checkpoint name. The replacement
# uses ``\1`` etc. backrefs for the captured layer index.
_HF_TO_CKPT_RULES: tuple[tuple[str, str], ...] = (
    # Top-level
    (r"^model\.", ""),
    (r"^embed_tokens\.weight$", "embed.weight"),
    (r"^lm_head\.weight$", "head.weight"),
    (r"^norm\.weight$", "norm.weight"),
    (r"^hc_head\.hc_fn$", "hc_head_fn"),
    (r"^hc_head\.hc_base$", "hc_head_base"),
    (r"^hc_head\.hc_scale$", "hc_head_scale"),
    # Per-decoder-layer norms
    (r"^layers\.(\d+)\.input_layernorm\.weight$", r"layers.\1.attn_norm.weight"),
    (r"^layers\.(\d+)\.post_attention_layernorm\.weight$", r"layers.\1.ffn_norm.weight"),
    # Hyper-connections (attn site / ffn site)
    (r"^layers\.(\d+)\.attn_hc\.fn$", r"layers.\1.hc_attn_fn"),
    (r"^layers\.(\d+)\.attn_hc\.base$", r"layers.\1.hc_attn_base"),
    (r"^layers\.(\d+)\.attn_hc\.scale$", r"layers.\1.hc_attn_scale"),
    (r"^layers\.(\d+)\.ffn_hc\.fn$", r"layers.\1.hc_ffn_fn"),
    (r"^layers\.(\d+)\.ffn_hc\.base$", r"layers.\1.hc_ffn_base"),
    (r"^layers\.(\d+)\.ffn_hc\.scale$", r"layers.\1.hc_ffn_scale"),
    # Self-attention block
    (r"^layers\.(\d+)\.self_attn\.q_a_proj\.", r"layers.\1.attn.wq_a."),
    (r"^layers\.(\d+)\.self_attn\.q_b_proj\.", r"layers.\1.attn.wq_b."),
    (r"^layers\.(\d+)\.self_attn\.q_a_norm\.weight$", r"layers.\1.attn.q_norm.weight"),
    (r"^layers\.(\d+)\.self_attn\.kv_proj\.", r"layers.\1.attn.wkv."),
    (r"^layers\.(\d+)\.self_attn\.kv_norm\.weight$", r"layers.\1.attn.kv_norm.weight"),
    (r"^layers\.(\d+)\.self_attn\.o_a_proj\.", r"layers.\1.attn.wo_a."),
    (r"^layers\.(\d+)\.self_attn\.o_b_proj\.", r"layers.\1.attn.wo_b."),
    (r"^layers\.(\d+)\.self_attn\.sinks$", r"layers.\1.attn.attn_sink"),
    # CSA / HCA compressor (lives under self_attn.compressor in the HF model)
    (r"^layers\.(\d+)\.self_attn\.compressor\.kv_proj\.", r"layers.\1.attn.compressor.wkv."),
    (r"^layers\.(\d+)\.self_attn\.compressor\.gate_proj\.", r"layers.\1.attn.compressor.wgate."),
    (r"^layers\.(\d+)\.self_attn\.compressor\.position_bias$", r"layers.\1.attn.compressor.ape"),
    (r"^layers\.(\d+)\.self_attn\.compressor\.kv_norm\.weight$", r"layers.\1.attn.compressor.norm.weight"),
    # Lightning Indexer (CSA only)
    (r"^layers\.(\d+)\.self_attn\.compressor\.indexer\.kv_proj\.", r"layers.\1.attn.indexer.compressor.wkv."),
    (r"^layers\.(\d+)\.self_attn\.compressor\.indexer\.gate_proj\.", r"layers.\1.attn.indexer.compressor.wgate."),
    (r"^layers\.(\d+)\.self_attn\.compressor\.indexer\.position_bias$", r"layers.\1.attn.indexer.compressor.ape"),
    (
        r"^layers\.(\d+)\.self_attn\.compressor\.indexer\.kv_norm\.weight$",
        r"layers.\1.attn.indexer.compressor.norm.weight",
    ),
    (r"^layers\.(\d+)\.self_attn\.compressor\.indexer\.q_b_proj\.", r"layers.\1.attn.indexer.wq_b."),
    (r"^layers\.(\d+)\.self_attn\.compressor\.indexer\.weights_proj\.", r"layers.\1.attn.indexer.weights_proj."),
    # MoE: router / experts / shared experts (HF `MixtralExperts` style)
    (r"^layers\.(\d+)\.mlp\.gate\.weight$", r"layers.\1.ffn.gate.weight"),
    (r"^layers\.(\d+)\.mlp\.gate\.e_score_correction_bias$", r"layers.\1.ffn.gate.bias"),
    (r"^layers\.(\d+)\.mlp\.gate\.tid2eid$", r"layers.\1.ffn.gate.tid2eid"),
    (r"^layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.", r"layers.\1.ffn.experts.\2.w1."),
    (r"^layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.", r"layers.\1.ffn.experts.\2.w2."),
    (r"^layers\.(\d+)\.mlp\.experts\.(\d+)\.up_proj\.", r"layers.\1.ffn.experts.\2.w3."),
    (r"^layers\.(\d+)\.mlp\.shared_experts\.gate_proj\.", r"layers.\1.ffn.shared_experts.w1."),
    (r"^layers\.(\d+)\.mlp\.shared_experts\.down_proj\.", r"layers.\1.ffn.shared_experts.w2."),
    (r"^layers\.(\d+)\.mlp\.shared_experts\.up_proj\.", r"layers.\1.ffn.shared_experts.w3."),
)
_HF_TO_CKPT_COMPILED: tuple[tuple[re.Pattern, str], ...] = tuple(
    (re.compile(pat), repl) for pat, repl in _HF_TO_CKPT_RULES
)


def hf_to_checkpoint_name(name: str) -> str:
    """Translate an HF-style parameter name to its checkpoint name.

    Unknown names are returned unchanged so callers can pass raw checkpoint
    names through the loader transparently.
    """
    for pattern, replacement in _HF_TO_CKPT_COMPILED:
        new_name, n = pattern.subn(replacement, name)
        if n:
            return new_name
    return name


class DeepseekV4WeightLoader:
    """Index-driven, lazy reader over the V4-Flash safetensors shards."""

    def __init__(self, model_dir: str | Path):
        self.snapshot_dir = resolve_snapshot_dir(Path(model_dir))
        index_path = self.snapshot_dir / INDEX_FILENAME
        if index_path.is_file():
            with index_path.open("r") as fh:
                index = json.load(fh)
            weight_map = index["weight_map"]
            self._name_to_shard: dict[str, Path] = {
                name: self.snapshot_dir / shard for name, shard in weight_map.items()
            }
        else:
            # Fall back to scanning all shards (slower one-time cost) when no index ships.
            self._name_to_shard = {}
            for shard in sorted(self.snapshot_dir.glob("*.safetensors")):
                with safe_open(str(shard), framework="pt") as f:
                    for name in f.keys():
                        self._name_to_shard[name] = shard

        if not self._name_to_shard:
            raise FileNotFoundError(f"No tensors discovered under {self.snapshot_dir}")

    # ------------------------------------------------------------------ #
    # Discovery
    # ------------------------------------------------------------------ #
    def keys(self) -> Iterable[str]:
        """All tensor names that appear in the checkpoint."""
        return self._name_to_shard.keys()

    def has(self, name: str, *, translate: bool = True) -> bool:
        ckpt_name = hf_to_checkpoint_name(name) if translate else name
        return ckpt_name in self._name_to_shard

    def shard_of(self, name: str, *, translate: bool = True) -> Path:
        ckpt_name = hf_to_checkpoint_name(name) if translate else name
        try:
            return self._name_to_shard[ckpt_name]
        except KeyError as exc:
            raise KeyError(f"Tensor '{name}' (-> '{ckpt_name}') not found in checkpoint") from exc

    # ------------------------------------------------------------------ #
    # Shard handle cache
    # ------------------------------------------------------------------ #
    @lru_cache(maxsize=None)
    def _open_shard(self, shard_path_str: str):
        """Cache a ``safe_open`` handle per shard.

        The cache key is the string path (lru_cache requires hashable args).
        Handles stay open for the lifetime of the loader, which is fine for
        the memory-mapped safetensors format.
        """
        return safe_open(shard_path_str, framework="pt")

    # ------------------------------------------------------------------ #
    # Tensor access
    # ------------------------------------------------------------------ #
    def get_tensor(self, name: str, *, translate: bool = True) -> torch.Tensor:
        """Return the named tensor as a ``torch.Tensor``.

        Only the requested tensor is read from disk; the rest of the shard
        stays memory-mapped.
        """
        shard = self.shard_of(name, translate=translate)
        handle = self._open_shard(str(shard))
        ckpt_name = hf_to_checkpoint_name(name) if translate else name
        print("Loading tensor: ", ckpt_name)
        return handle.get_tensor(ckpt_name)

    def get_scale(self, name: str, *, translate: bool = True) -> Optional[torch.Tensor]:
        """Companion fp8 scale tensor for ``<name>``, or ``None`` if absent.

        V4-Flash ships its MoE / projection weights as e4m3 fp8 with a
        per-block ue8m0 scale stored under the same prefix with ``.scale``
        instead of ``.weight``. Non-quantized tensors (embed, norms, biases,
        position_bias, etc.) have no companion scale and this returns
        ``None`` for them.
        """
        ckpt_name = hf_to_checkpoint_name(name) if translate else name
        if not ckpt_name.endswith(".weight"):
            return None
        scale_name = ckpt_name[: -len(".weight")] + ".scale"
        if scale_name not in self._name_to_shard:
            return None
        return self.get_tensor(scale_name, translate=False)

    def get_meta(self, name: str, *, translate: bool = True) -> tuple[str, tuple[int, ...]]:
        """Return ``(dtype_str, shape)`` for ``name`` without loading data."""
        shard = self.shard_of(name, translate=translate)
        handle = self._open_shard(str(shard))
        ckpt_name = hf_to_checkpoint_name(name) if translate else name
        sl = handle.get_slice(ckpt_name)
        return sl.get_dtype(), tuple(sl.get_shape())


__all__ = [
    "DeepseekV4WeightLoader",
    "hf_to_checkpoint_name",
    "resolve_snapshot_dir",
]
