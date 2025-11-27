# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open


class LazyStateDict(Mapping[str, torch.Tensor]):
    """
    Mapping-like view over HuggingFace safetensors weights that loads tensors lazily on access.

    Keys are exposed relative to the configured base prefix. Values are cached after first access.
    """

    def __init__(
        self,
        model_path: Path,
        base_prefix: str = "",
        *,
        _full_to_file: Optional[dict[str, str]] = None,
        _cache: Optional[dict[str, torch.Tensor]] = None,
        _num_layers: Optional[int] = None,
    ):
        self._model_path = Path(model_path)
        self._base_prefix = base_prefix
        if _full_to_file is None:
            index_path = self._model_path / "model.safetensors.index.json"
            weight_map = json.load(index_path.open("r"))["weight_map"]
            self._full_to_file = dict(weight_map)
        else:
            self._full_to_file = _full_to_file
        self._cache: dict[str, torch.Tensor] = {} if _cache is None else _cache
        self._num_layers: Optional[int] = _num_layers

    def view_with_prefix(self, prefix: str, num_layers: int | None = None) -> "LazyStateDict":
        """
        Return a new LazyStateDict view narrowed to keys under base_prefix + prefix.
        Values cache is shared between views.
        """
        combined_prefix = self._base_prefix + prefix
        return LazyStateDict(
            self._model_path,
            combined_prefix,
            _full_to_file=self._full_to_file,
            _cache=self._cache,
            _num_layers=num_layers,
        )

    def _full_key(self, key: str) -> str:
        return self._base_prefix + key

    def _passes_layer_filter(self, full_key: str) -> bool:
        if self._num_layers is None:
            return True
        if full_key.startswith("model.layers."):
            suffix = full_key.removeprefix("model.layers.")
            digits = []
            for ch in suffix:
                if ch.isdigit():
                    digits.append(ch)
                else:
                    break
            if digits:
                try:
                    layer_idx = int("".join(digits))
                    return layer_idx < self._num_layers
                except ValueError:
                    return True
        return True

    def __getitem__(self, key: str) -> torch.Tensor:
        full_key = self._full_key(key)
        if full_key in self._cache:
            return self._cache[full_key]
        if full_key not in self._full_to_file or not self._passes_layer_filter(full_key):
            raise KeyError(key)
        file_name = self._full_to_file[full_key]
        file_path = self._model_path / file_name
        with safe_open(file_path, framework="pt", device="cpu") as f:
            tensor = f.get_tensor(full_key)
        self._cache[full_key] = tensor
        return tensor

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        full_key = self._full_key(key)
        return full_key in self._full_to_file and self._passes_layer_filter(full_key)

    def __iter__(self) -> Iterator[str]:
        base = self._base_prefix
        for full_key in self._full_to_file.keys():
            if not full_key.startswith(base):
                continue
            if not self._passes_layer_filter(full_key):
                continue
            yield full_key[len(base) :]

    def __len__(self) -> int:
        # Iterate once to count to avoid computing and storing an intermediate list
        return sum(1 for _ in self.__iter__())
