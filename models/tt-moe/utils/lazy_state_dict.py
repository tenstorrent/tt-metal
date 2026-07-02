# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
#
# SPDX-License-Identifier: Apache-2.0

"""
LazyStateDict for loading HuggingFace model weights lazily.
Adapted from models/demos/deepseek_v3/utils/lazy_state_dict.py
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from time import perf_counter
from typing import Optional

import torch
from loguru import logger
from safetensors import safe_open


class LazyStateDict(Mapping[str, torch.Tensor]):
    """
    Mapping-like view over HuggingFace safetensors that loads tensors lazily on access.

    Invariants and behavior:
    - Keys exposed by the mapping are trimmed by the configured base prefix
    - Tensors are loaded lazily on first access and cached
    - The mapping is read-only and assumes files do not change during process lifetime
    """

    def __init__(
        self,
        model_path: Path,
        base_prefix: str = "",
        *,
        _full_to_file: Optional[dict[str, str]] = None,
        _cache: Optional[dict[str, torch.Tensor]] = None,
        _num_layers: Optional[int] = None,
        _file_handles: Optional[dict[str, object]] = None,
    ):
        self._model_path = Path(model_path)
        self._base_prefix = base_prefix

        if _full_to_file is None:
            t0 = perf_counter()
            index_path = self._model_path / "model.safetensors.index.json"
            if not index_path.is_file():
                raise ValueError(f"Unable to find index file at {index_path}. Is the model path correct?")
            try:
                with index_path.open("r", encoding="utf-8") as f:
                    index_obj = json.load(f)
            except Exception as e:
                raise ValueError(f"Failed to parse index JSON at {index_path}: {e}") from e
            self._full_to_file = dict(index_obj["weight_map"])
            elapsed_ms = (perf_counter() - t0) * 1e3

            num_keys = len(self._full_to_file)
            num_files = len(set(self._full_to_file.values()))
            logger.info(
                f"LazyStateDict initialized from {index_path} ({num_keys} keys across {num_files} files) in {elapsed_ms:.1f} ms"
            )
        else:
            self._full_to_file = _full_to_file

        self._num_layers: Optional[int] = _num_layers
        self._file_handles: dict[str, object] = {} if _file_handles is None else _file_handles
        self._cache: dict[str, torch.Tensor] = {} if _cache is None else _cache

    def _get_handle(self, filename: str):
        """Return a cached safe_open handle for a shard filename, opening it on first use."""
        handle = self._file_handles.get(filename)
        if handle is not None:
            return handle
        filepath = self._model_path / filename
        handle = safe_open(filepath, framework="pt", device="cpu")
        self._file_handles[filename] = handle
        return handle

    def clear_cache(self) -> None:
        """Clear cached tensors while keeping file handles open for reuse."""
        if self._cache:
            self._cache.clear()

    def close(self) -> None:
        """Close all cached shard handles and clear cached tensors."""
        if self._file_handles:
            for filename, handle in list(self._file_handles.items()):
                try:
                    close_fn = getattr(handle, "close", None)
                    if callable(close_fn):
                        close_fn()
                except Exception as e:
                    logger.error(f"Failed to close handle for '{filename}': {e}")
            self._file_handles.clear()
        if self._cache:
            self._cache.clear()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def view_with_prefix(self, prefix: str, num_layers: Optional[int] = None) -> "LazyStateDict":
        """Return a new LazyStateDict view narrowed to keys under base_prefix + prefix."""
        combined_prefix = self._base_prefix + prefix
        child_num_layers = self._num_layers if num_layers is None else num_layers

        return LazyStateDict(
            self._model_path,
            combined_prefix,
            _full_to_file=self._full_to_file,
            _cache=self._cache,
            _num_layers=child_num_layers,
            _file_handles=self._file_handles,
        )

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __iter__(self) -> Iterator[str]:
        for full_key in self._full_to_file:
            if not full_key.startswith(self._base_prefix):
                continue

            if self._num_layers is not None:
                import re

                match = re.match(r"model\.layers\.(\d+)\.", full_key)
                if match:
                    layer_idx = int(match.group(1))
                    if layer_idx >= self._num_layers:
                        continue

            relative_key = full_key[len(self._base_prefix) :]
            yield relative_key

    def __getitem__(self, relative_key: str) -> torch.Tensor:
        full_key = self._base_prefix + relative_key

        cached = self._cache.get(full_key)
        if cached is not None:
            return cached

        filename = self._full_to_file.get(full_key)
        if filename is None:
            raise KeyError(f"Key '{relative_key}' not found (full key: '{full_key}')")

        handle = self._get_handle(filename)
        tensor = handle.get_tensor(full_key)

        self._cache[full_key] = tensor
        return tensor
