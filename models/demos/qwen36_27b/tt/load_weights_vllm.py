# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Streaming weight loading for Qwen3.6-27B (HF arch ``Qwen3_5ForConditionalGeneration``).

This is the vLLM-path loader: it streams safetensors shards lazily (one weight at
a time) so the full ~27B checkpoint never has to live in host RAM at once.

HF key layout (this checkpoint is the multimodal wrapper; the TEXT backbone is
under ``model.language_model.``):
  * model.language_model.embed_tokens.weight   -> canonical model.embed_tokens.weight
  * model.language_model.norm.weight           -> canonical model.norm.weight
  * lm_head.weight                             -> lm_head.weight (top level)
  * model.language_model.layers.{i}.linear_attn.{in_proj_qkv,in_proj_z,in_proj_b,
        in_proj_a,out_proj}.weight, .{A_log,dt_bias}, .conv1d.weight, .norm.weight
  * model.language_model.layers.{i}.self_attn.{q,k,v,o}_proj.weight + q_norm/k_norm
  * model.language_model.layers.{i}.mlp.{gate_proj,up_proj,down_proj}.weight (DENSE)
  * model.language_model.layers.{i}.{input_layernorm,post_attention_layernorm}.weight

NOTE: unlike the HF Qwen3-Next checkpoints (which fuse in_proj_qkvz / in_proj_ba),
this checkpoint stores the DeltaNet input projections ALREADY SPLIT
(in_proj_qkv / in_proj_z / in_proj_b / in_proj_a), so no row-reordering /
_fused_split is required.

The ``model.visual.*`` (vision tower) weights are ignored — this port is the
text backbone only.
"""

import json
from pathlib import Path

import torch

from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig


def _normalize_key(key: str) -> str:
    """Stored HF key -> canonical model key (strip the language_model nesting)."""
    if key.startswith("model.language_model."):
        return "model." + key[len("model.language_model.") :]
    return key


def _denormalize_candidates(key: str):
    """Canonical key -> candidate stored keys in the HF checkpoint."""
    cands = [key]
    if key.startswith("model.") and not key.startswith("model.language_model."):
        cands.append("model.language_model." + key[len("model.") :])
    return cands


class StreamingStateDict:
    """Lazy, dict-like view over a sharded safetensors checkpoint.

    Drop-in for a plain ``state_dict`` (supports ``[]``, ``in``, ``get``,
    ``keys``). Keys use the model's canonical names; the ``language_model.``
    nesting in the checkpoint is resolved transparently. Vision-tower weights
    (``model.visual.*``) are excluded from ``keys()``.
    """

    def __init__(self, model_path: str, config: Qwen36ModelConfig | None = None):
        from safetensors import safe_open

        self._safe_open = safe_open
        self.config = config
        self.model_path = Path(model_path)
        shard_files = sorted(self.model_path.glob("*.safetensors"))
        if not shard_files:
            raise FileNotFoundError(f"No safetensors files found in {self.model_path}")

        # Map stored-key -> shard file. Prefer the index.json when present.
        self._key_to_file: dict[str, Path] = {}
        index = self.model_path / "model.safetensors.index.json"
        if index.exists():
            weight_map = json.loads(index.read_text())["weight_map"]
            for k, fname in weight_map.items():
                self._key_to_file[k] = self.model_path / fname
        else:
            for f in shard_files:
                with safe_open(str(f), framework="pt") as handle:
                    for k in handle.keys():
                        self._key_to_file[k] = f

        # canonical-name -> stored-name (text backbone only; skip vision tower)
        self._canon: dict[str, str] = {}
        for stored in self._key_to_file:
            if ".visual." in stored or stored.startswith("model.visual."):
                continue
            self._canon[_normalize_key(stored)] = stored

    def _resolve(self, key: str):
        for cand in _denormalize_candidates(key):
            if cand in self._key_to_file:
                return cand
        if key in self._canon:
            return self._canon[key]
        return None

    def _read(self, stored: str) -> torch.Tensor:
        with self._safe_open(str(self._key_to_file[stored]), framework="pt") as handle:
            return handle.get_tensor(stored)

    def __contains__(self, key: str) -> bool:
        return self._resolve(key) is not None

    def __getitem__(self, key: str) -> torch.Tensor:
        stored = self._resolve(key)
        if stored is not None:
            return self._read(stored)
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        return self._canon.keys()


def load_streaming_state_dict(
    config: Qwen36ModelConfig, model_path: str | None = None
) -> StreamingStateDict:
    """Open a checkpoint as a lazy StreamingStateDict (downloads if needed)."""
    if model_path is None:
        from huggingface_hub import snapshot_download

        model_name = getattr(config, "model_name", "Qwen/Qwen3.6-27B")
        model_path = snapshot_download(model_name, allow_patterns=["*.safetensors", "*.json"])
    return StreamingStateDict(model_path, config=config)
