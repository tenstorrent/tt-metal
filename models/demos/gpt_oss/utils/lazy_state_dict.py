# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Lazy, read-only view over a GPT-OSS safetensors checkpoint.

``AutoModelForCausalLM.from_pretrained`` dequantizes every MXFP4 expert tensor to bf16 up front,
so a cold GPT-OSS-120B load holds the whole model (~234 GB) on the host before the first layer
reaches the device (#60193). This mapping exposes the same keys and the same tensors as
``model.state_dict()`` after that load, but reads (and dequantizes) a tensor only when it is
accessed. Nothing is cached: once the caller has built a layer and dropped its sub-dict, the
host memory for that layer is released.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from pathlib import Path

import torch
from safetensors import safe_open
from transformers.integrations.mxfp4 import convert_moe_packed_tensors

from models.tt_transformers.tt.load_checkpoints import convert_hf_qkv_to_meta_format

_MXFP4_SUFFIXES = ("_blocks", "_scales")


def resolve_checkpoint_dir(weights_path: str | os.PathLike) -> Path:
    """Return the local directory that holds the safetensors files for ``weights_path``
    (a local directory or a Hugging Face hub id, which is resolved through the local HF cache)."""
    path = Path(weights_path)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(weights_path), allow_patterns=["*.json", "*.safetensors"]))


class GptOssLazyStateDict(Mapping[str, torch.Tensor]):
    """Mapping with the keys of the HF ``GptOssForCausalLM.state_dict()``.

    - ``*.mlp.experts.{gate_up_proj,down_proj}`` are dequantized from their ``_blocks`` / ``_scales``
      pair with the same HF function ``from_pretrained`` uses, so values are bit-identical.
    - With ``convert_to_meta_format=True`` the q/k projections are permuted exactly like
      ``convert_hf_qkv_to_meta_format`` does on an eager state dict.
    - Every other float tensor is returned as bf16 (``from_pretrained(torch_dtype=torch.bfloat16)``).
    """

    def __init__(self, weights_path: str | os.PathLike, head_dim: int, convert_to_meta_format: bool = True):
        self._dir = resolve_checkpoint_dir(weights_path)
        self._head_dim = head_dim
        self._convert_to_meta_format = convert_to_meta_format
        self._handles: dict[str, object] = {}

        index_path = self._dir / "model.safetensors.index.json"
        if index_path.is_file():
            weight_map = json.loads(index_path.read_text())["weight_map"]
        else:
            single = self._dir / "model.safetensors"
            with safe_open(str(single), framework="pt") as f:
                weight_map = {k: single.name for k in f.keys()}

        # logical key -> shard file. An MXFP4 pair folds into one logical key; HF stores both halves
        # of a pair in the same shard.
        self._key_to_file: dict[str, str] = {}
        for key, filename in weight_map.items():
            for suffix in _MXFP4_SUFFIXES:
                if key.endswith(suffix):
                    key = key[: -len(suffix)]
                    break
            self._key_to_file[key] = filename
        self._mxfp4_keys = {k[: -len("_blocks")] for k in weight_map if k.endswith("_blocks")}

    def _handle(self, filename: str):
        handle = self._handles.get(filename)
        if handle is None:
            handle = safe_open(str(self._dir / filename), framework="pt")
            self._handles[filename] = handle
        return handle

    def __getitem__(self, key: str) -> torch.Tensor:
        filename = self._key_to_file[key]
        handle = self._handle(filename)
        if key in self._mxfp4_keys:
            blocks = handle.get_tensor(f"{key}_blocks")
            scales = handle.get_tensor(f"{key}_scales")
            return convert_moe_packed_tensors(blocks, scales, dtype=torch.bfloat16)

        tensor = handle.get_tensor(key)
        if tensor.is_floating_point() and tensor.dtype != torch.bfloat16:
            tensor = tensor.to(torch.bfloat16)
        if self._convert_to_meta_format:
            tensor = convert_hf_qkv_to_meta_format({key: tensor}, self._head_dim)[key]
        return tensor

    def __contains__(self, key: object) -> bool:
        return key in self._key_to_file

    def __iter__(self) -> Iterator[str]:
        return iter(self._key_to_file)

    def __len__(self) -> int:
        return len(self._key_to_file)
