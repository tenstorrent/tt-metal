# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
#
# SPDX-License-Identifier: Apache-2.0

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
    - Keys exposed by the mapping are trimmed by the configured base prefix; internally we keep
      the full HuggingFace key.
    - Tensors are loaded lazily on first access and cached; the cache is shared across views and
      keyed by the full parameter name to avoid collisions across prefixes while allowing reuse
      of identical parameters accessed through different views.
    - The mapping is read-only and assumes the index file and weight files do not change during
      the process lifetime.
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

        # If _full_to_file is provided then we are a now a view ofthe original LazyStateDict.
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
            if num_keys <= 0:
                raise ValueError(f"No keys found in index file at {index_path}")
            if num_files <= 0:
                raise ValueError(f"No files found in index file at {index_path}")
            logger.info(
                f"LazyStateDict initialized from {index_path} ({num_keys} keys across {num_files} files) in {elapsed_ms:.1f} ms"
            )
        else:
            self._full_to_file = _full_to_file
        self._cache: dict[str, torch.Tensor] = {} if _cache is None else _cache
        self._num_layers: Optional[int] = _num_layers
        # Track total cached host bytes for visibility
        self._bytes_cached: int = 0
        self._tensor_bytes: dict[str, int] = {}

    def view_with_prefix(self, prefix: str, num_layers: Optional[int] = None) -> "LazyStateDict":
        """
        Return a new LazyStateDict view narrowed to keys under base_prefix + prefix.

        - Keys yielded by the view are trimmed by the combined prefix (i.e., callers
          see keys relative to base_prefix + prefix).
        - If num_layers is provided, the view only exposes keys whose layer index
          parsed from \"model.layers.<idx>.…\" is strictly less than num_layers.
        - The underlying cache is shared across views and is keyed by the full key
          (the original HuggingFace parameter name), preventing collisions across
          different prefixes while allowing reuse for the same full key.
        """
        combined_prefix = self._base_prefix + prefix

        # Inherit parent's layer filter if not explicitly overridden
        child_num_layers = self._num_layers if num_layers is None else num_layers
        logger.debug(
            f"LazyStateDict view created: base_prefix='{self._base_prefix}', add_prefix='{prefix}', combined='{combined_prefix}', num_layers={child_num_layers}"
        )

        return LazyStateDict(
            self._model_path,
            combined_prefix,
            _full_to_file=self._full_to_file,
            _cache=self._cache,
            _num_layers=child_num_layers,
        )

    def _full_key(self, key: str) -> str:
        return self._base_prefix + key

    def _passes_layer_filter(self, full_key: str) -> bool:
        if self._num_layers is None:
            return True
        prefix = "model.layers."
        if not full_key.startswith(prefix):
            return True
        layer_part = full_key[len(prefix) :].split(".", 1)[0]
        return (not layer_part.isdigit()) or (int(layer_part) < self._num_layers)

    def __getitem__(self, key: str) -> torch.Tensor:
        """
        Get the tensor for a key relative to our current base prefix.

        Raises:
            KeyError: if the full key does not exist in the index or is filtered out by num_layers.
            FileNotFoundError: if the index points to a weight file that does not exist on disk.
        """
        full_key = self._full_key(key)
        if full_key in self._cache:
            return self._cache[full_key]
        if full_key not in self._full_to_file or not self._passes_layer_filter(full_key):
            raise KeyError(key)
        filename = self._full_to_file[full_key]
        filepath = self._model_path / filename
        if not filepath.exists():
            raise FileNotFoundError(
                f"Attempted to load weight {full_key} from file {filepath} but the file does not exist."
            )
        t0 = perf_counter()
        logger.debug(f"Opening safetensors file '{filepath}' to load key '{full_key}'")
        with safe_open(filepath, framework="pt", device="cpu") as f:
            tensor = f.get_tensor(full_key).clone()
        load_ms = (perf_counter() - t0) * 1e3
        try:
            logger.debug(
                f"Loaded '{full_key}' from '{filename}' in {load_ms:.1f} ms (shape={tuple(tensor.shape)}, dtype={tensor.dtype})"
            )
        except Exception:
            logger.debug(f"Loaded '{full_key}' from '{filename}' in {load_ms:.1f} ms")
        self._cache[full_key] = tensor
        # Cache accounting and debug log
        try:
            size_bytes = tensor.element_size() * tensor.numel()
            # Avoid double counting if key somehow reinserted
            if full_key not in self._tensor_bytes:
                self._tensor_bytes[full_key] = size_bytes
                self._bytes_cached += size_bytes
            logger.debug(
                f"LazyStateDict cached {size_bytes/1e6:.2f} MB (key={full_key}); "
                f"total={self._bytes_cached/1e6:.2f} MB; tensors={len(self._cache)}"
            )
        except Exception:
            pass
        return tensor

    def __contains__(self, key: object) -> bool:
        """
        Return True if the key is present in the current view
        """
        if not isinstance(key, str):
            raise TypeError(f"Key must be a string but got {type(key)}")
        full_key = self._full_key(key)
        return full_key in self._full_to_file and self._passes_layer_filter(full_key)

    def __iter__(self) -> Iterator[str]:
        """
        Iterate keys present under the current view
        """
        base = self._base_prefix
        for full_key in self._full_to_file.keys():
            if not full_key.startswith(base):
                continue
            if not self._passes_layer_filter(full_key):
                continue
            yield full_key[len(base) :]

    def __len__(self) -> int:
        """
        Number of keys visible in this view after filters. Computed by iterating the
        filtered keys at call time.
        """
        return sum(1 for _ in self.__iter__())

    # Introspection
    def cached_bytes(self) -> int:
        return self._bytes_cached

    def cache_info(self) -> dict:
        return {
            "num_tensors": len(self._cache),
            "bytes": self._bytes_cached,
            "mb": self._bytes_cached / (1024 * 1024),
        }
