# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Weight conversion utilities for MaskFormer Swin-Base.

This module will implement a bridge between HuggingFace checkpoints
(`facebook/maskformer-swin-base-coco`) and TT-NN layout requirements.
Responsibilities include:

* Downloading / caching the reference PyTorch weights using ``huggingface_hub``
  or local checkpoints.
* Translating parameter names to TT-NN module expectations.
* Converting PyTorch tensors into TT-compatible formats/layouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch

from .ttnn_compat import ttnn, get_default_dtype

DEFAULT_TT_DTYPE = get_default_dtype()
try:
    from huggingface_hub import snapshot_download
except ModuleNotFoundError:  # pragma: no cover - optional dependency for now.
    snapshot_download = None

try:
    from safetensors.torch import load_file as load_safetensor
except ModuleNotFoundError:  # pragma: no cover - safetensors is optional.
    load_safetensor = None


@dataclass
class WeightConversionConfig:
    """Options controlling how reference weights are fetched and converted."""

    pretrained_model_name: str = "facebook/maskformer-swin-base-coco"
    cache_dir: Optional[Path] = None
    dtype: object = DEFAULT_TT_DTYPE


@dataclass
class ReferenceWeights:
    """Container bundling the downloaded state dict and associated metadata."""

    state_dict: Dict[str, object]
    config: Dict[str, object]
    checkpoint_path: Path
    checkpoint_dir: Path


def download_reference_weights(config: WeightConversionConfig) -> ReferenceWeights:
    """
    Retrieve the HuggingFace checkpoint, returning the raw PyTorch state dict plus config.

    Returns
    -------
    ReferenceWeights
        State dict, JSON config payload, and the resolved checkpoint path.
    """

    candidate = Path(config.pretrained_model_name)
    if candidate.exists():
        checkpoint_path = _resolve_checkpoint_path(candidate)
    else:
        if snapshot_download is None:
            raise ModuleNotFoundError(
                "huggingface_hub is required to download MaskFormer weights. "
                "Install via `pip install huggingface_hub`."
            )
        checkpoint_path = _download_checkpoint_from_hub(config)

    state_dict = _load_checkpoint_state_dict(checkpoint_path)
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unexpected checkpoint structure at {checkpoint_path}")

    config_payload = _load_model_config(checkpoint_path)
    checkpoint_dir = checkpoint_path.parent
    return ReferenceWeights(
        state_dict=state_dict,
        config=config_payload,
        checkpoint_path=checkpoint_path,
        checkpoint_dir=checkpoint_dir,
    )


def convert_state_dict_to_tt(
    state_dict: Dict[str, object],
    config: WeightConversionConfig,
) -> Dict[str, object]:
    """
    Convert HuggingFace tensors into TT-NN friendly layouts.

    This function should encapsulate:
    * Layout conversions (e.g. linear vs conv weights)
    * Parameter splitting / merging for QKV projections
    * Any precision casts required by TT kernels
    """

    converter = _WeightConverter(config)
    return converter.convert(state_dict)


def extract_backbone_state(tt_state_dict: Dict[str, object]) -> Dict[str, object]:
    """Return a HuggingFace-compatible Swin backbone state dict."""

    prefix = "model.pixel_level_module.encoder."
    extracted: Dict[str, object] = {}
    for name, tensor in tt_state_dict.items():
        if name.startswith(prefix):
            extracted[name[len(prefix) :]] = tensor
    return extracted


def extract_pixel_decoder_state(tt_state_dict: Dict[str, object]) -> Dict[str, object]:
    """Return a HuggingFace-compatible pixel decoder state dict."""

    prefix = "model.pixel_level_module.decoder."
    return {name[len(prefix) :]: tensor for name, tensor in tt_state_dict.items() if name.startswith(prefix)}


def extract_transformer_state(tt_state_dict: Dict[str, object]) -> Dict[str, object]:
    """Return a HuggingFace-compatible transformer module state dict."""

    prefix = "model.transformer_module."
    return {name[len(prefix) :]: tensor for name, tensor in tt_state_dict.items() if name.startswith(prefix)}


def extract_heads_state(tt_state_dict: Dict[str, object]) -> Dict[str, object]:
    """Return class predictor and mask embedder state dict."""

    state: Dict[str, object] = {}
    for name, tensor in tt_state_dict.items():
        if name.startswith("model.transformer_module."):
            continue
        if name.startswith("model.class_predictor"):
            state[name[len("model.") :]] = tensor
        elif name.startswith("model.mask_embedder"):
            state[name[len("model.") :]] = tensor
        elif name.startswith("class_predictor"):
            state[name] = tensor
        elif name.startswith("mask_embedder"):
            state[name] = tensor
    return state


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TT_TO_TORCH_DTYPE: Dict[object, torch.dtype] = {}
if hasattr(ttnn, "bfloat16"):
    _TT_TO_TORCH_DTYPE[ttnn.bfloat16] = torch.bfloat16
if hasattr(ttnn, "float16"):
    _TT_TO_TORCH_DTYPE[ttnn.float16] = torch.float16
if hasattr(ttnn, "float32"):
    _TT_TO_TORCH_DTYPE[ttnn.float32] = torch.float32


class _WeightConverter:
    """Helper class responsible for orchestrating per-module conversions."""

    SECTION_PREFIXES = {
        "backbone": ("model.pixel_level_module.encoder.model.",),
        "pixel_decoder": (
            "model.pixel_level_module.decoder.",
            "model.pixel_level_module.encoder.hidden_states_norms.",
        ),
        "transformer_decoder": ("model.transformer_module.",),
        "heads": ("class_predictor", "mask_embedder", "criterion."),
    }

    def __init__(self, config: WeightConversionConfig) -> None:
        self.config = config
        self.torch_dtype = _TT_TO_TORCH_DTYPE.get(config.dtype, torch.float32)

    def convert(self, state_dict: Dict[str, object]) -> Dict[str, object]:
        partitions = self._partition_state_dict(state_dict)
        converted: Dict[str, object] = {}
        converted.update(self._convert_backbone(partitions["backbone"]))
        converted.update(self._convert_pixel_decoder(partitions["pixel_decoder"]))
        converted.update(self._convert_transformer_decoder(partitions["transformer_decoder"]))
        converted.update(self._convert_heads(partitions["heads"]))
        return converted

    def _partition_state_dict(
        self,
        state_dict: Dict[str, object],
        *,
        strict: bool = True,
    ) -> Dict[str, Dict[str, object]]:
        partitions = {section: {} for section in self.SECTION_PREFIXES}
        leftovers: Dict[str, object] = {}

        for name, tensor in state_dict.items():
            matched = False
            for section, prefixes in self.SECTION_PREFIXES.items():
                if name.startswith(prefixes):
                    partitions[section][name] = tensor
                    matched = True
                    break
            if not matched:
                leftovers[name] = tensor

        if leftovers and strict:
            sample = ", ".join(list(leftovers.keys())[:5])
            raise NotImplementedError("Encountered parameters without a conversion rule. " f"Sample keys: {sample}")

        return partitions

    # ------------------------------------------------------------------
    # Per-module conversion stubs (to be filled in with TT-NN specifics)
    # ------------------------------------------------------------------

    def _convert_backbone(self, section: Dict[str, object]) -> Dict[str, object]:
        if not section:
            raise RuntimeError("Backbone weights missing from state dict.")
        return {name: self._cast_tensor(tensor) for name, tensor in section.items()}

    def _convert_pixel_decoder(self, section: Dict[str, object]) -> Dict[str, object]:
        if not section:
            return {}
        return {name: self._cast_tensor(tensor) for name, tensor in section.items()}

    def _convert_transformer_decoder(self, section: Dict[str, object]) -> Dict[str, object]:
        if not section:
            return {}
        return {name: self._cast_tensor(tensor) for name, tensor in section.items()}

    def _convert_heads(self, section: Dict[str, object]) -> Dict[str, object]:
        if not section:
            return {}
        return {name: self._cast_tensor(tensor) for name, tensor in section.items()}

    def _cast_tensor(self, tensor: object) -> object:
        if isinstance(tensor, torch.Tensor):
            if tensor.is_floating_point():
                tensor = tensor.to(self.torch_dtype)
        return tensor


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _download_checkpoint_from_hub(config: WeightConversionConfig) -> Path:
    """Download a HuggingFace snapshot and return the primary checkpoint path."""

    if snapshot_download is None:  # pragma: no cover - guard for optional dependency.
        raise ModuleNotFoundError(
            "huggingface_hub is required to download MaskFormer weights. " "Install via `pip install huggingface_hub`."
        )

    repo_id = config.pretrained_model_name
    cache_dir = str(config.cache_dir) if config.cache_dir else None

    snapshot_path = Path(
        snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            allow_patterns=["*.bin", "*.pt", "*.safetensors", "config.json"],
            local_dir=None,
            local_dir_use_symlinks=False,
        )
    )
    return _resolve_checkpoint_path(snapshot_path)


def _resolve_checkpoint_path(path: Path) -> Path:
    """Identify the best checkpoint file within ``path``."""

    if path.is_file():
        return path

    patterns = ("*.safetensors", "pytorch_model.bin", "*.bin", "*.pt")
    for pattern in patterns:
        matches = sorted(path.rglob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Could not locate a checkpoint file under {path}. "
        "Expected one of: *.safetensors, pytorch_model.bin, *.bin, *.pt"
    )


def _load_checkpoint_state_dict(checkpoint_path: Path) -> Dict[str, object]:
    """Load a checkpoint file into CPU memory."""

    suffix = checkpoint_path.suffix
    if suffix == ".safetensors":
        if load_safetensor is None:
            raise ModuleNotFoundError(
                "safetensors is required to load .safetensors files. " "Install via `pip install safetensors`."
            )
        state_dict = load_safetensor(str(checkpoint_path))
    else:
        state_dict = torch.load(str(checkpoint_path), map_location="cpu")

    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    return state_dict


def _load_model_config(checkpoint_path: Path) -> Dict[str, object]:
    """Load the HuggingFace config JSON next to the checkpoint."""

    config_dir = checkpoint_path.parent if checkpoint_path.is_file() else checkpoint_path
    for config_name in ("config.json", "maskformer_swin_config.json"):
        config_path = config_dir / config_name
        if config_path.exists():
            break
    else:
        raise FileNotFoundError(f"Could not find config JSON next to {checkpoint_path}")

    import json

    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected config content in {config_path}")
    return payload


def resolve_cli_dtype(name: str) -> object:
    key = name.replace("-", "").replace("_", "").lower()
    mapping = {
        "auto": DEFAULT_TT_DTYPE,
        "bf16": getattr(ttnn, "bfloat16", None),
        "bfloat16": getattr(ttnn, "bfloat16", None),
        "fp16": getattr(ttnn, "float16", None),
        "float16": getattr(ttnn, "float16", None),
        "fp32": getattr(ttnn, "float32", None),
        "float32": getattr(ttnn, "float32", None),
    }
    dtype = mapping.get(key, DEFAULT_TT_DTYPE)
    if dtype is None:
        print(f"[weights] Warning: dtype '{name}' not available; proceeding with torch.float32.", flush=True)
    return dtype
