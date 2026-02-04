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
        _file_handles: Optional[dict[str, object]] = None,
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

        self._num_layers: Optional[int] = _num_layers

        # Cache of open safetensors handles keyed by filename to avoid repeated mmaps
        self._file_handles: dict[str, object] = {} if _file_handles is None else _file_handles
        self._cache: dict[str, torch.Tensor] = {} if _cache is None else _cache

    def _get_handle(self, filename: str):
        """
        Return a cached safe_open handle for a shard filename, opening it on first use.
        """
        handle = self._file_handles.get(filename)
        if handle is not None:
            return handle
        filepath = self._model_path / filename
        handle = safe_open(filepath, framework="pt", device="cpu")
        self._file_handles[filename] = handle
        return handle

    def clear_cache(self) -> None:
        """
        Clear cached tensors while keeping file handles open for reuse.
        This allows memory to be freed while preserving the performance benefits
        of keeping mmap'd file handles open across test cases.
        """
        if self._cache:
            self._cache.clear()

    def close(self) -> None:
        """
        Close all cached shard handles and clear cached tensors and accounting.
        After calling this, previously returned tensors may become invalid if they
        referenced the underlying mmap. Further accesses will lazily reopen shards.
        """
        if self._file_handles:
            for filename, handle in list(self._file_handles.items()):
                try:
                    close_fn = getattr(handle, "close", None)
                    if callable(close_fn):
                        close_fn()
                except Exception as e:
                    logger.error(f"Failed to close handle for '{filename}': {e}")
            self._file_handles.clear()
        # Clear cached tensors and accounting to avoid stale references to mmaps
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
            _file_handles=self._file_handles,
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
            raise KeyError(f"Attempted to load weight {full_key} from file {filepath} but the file does not exist.")

        # Reuse shard-level handle to avoid creating a new mmap per tensor
        f = self._get_handle(filename)
        tensor = f.get_tensor(full_key)
        self._cache[full_key] = tensor

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
