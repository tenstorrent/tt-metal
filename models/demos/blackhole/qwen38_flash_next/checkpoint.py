# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Lazy, header-validated access to the exact Qwen3.8-Flash-Next checkpoint."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from safetensors import safe_open

from models.demos.blackhole.qwen38_flash_next.config import Qwen38Config

PINNED_CHECKPOINT_REVISION = "f5d08274bafd880402bd16f5e3e6c514136ec06c"
INDEX_SHA256 = "99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de"
CHECKPOINT_FILE_MANIFEST_SHA256 = "13c88f393ffbe4f5e9733d8e48a59f21c77e69a5f02b76bb073835d9a1ca0ea9"
CHECKPOINT_TENSOR_MANIFEST_SHA256 = "ebf4de2015233c20e21ff1e7e388e871208158ed350fea42964dbd52ca69359b"
EXPECTED_SHARDS = 131
EXPECTED_TENSORS = 1658
EXPECTED_TENSOR_BYTES = 359_999_963_128
EXPECTED_FILE_BYTES = 360_000_192_888
EXPECTED_INT64_ELEMENTS = 35
# The vision tower (model.visual.*): 333 BF16 tensors, all in the first shard, read only when the tower is requested.
VISION_TENSOR_PREFIX = "model.visual."
VISION_TENSOR_COUNT = 333
VISION_TENSOR_BYTES = 897_862_112
VISION_SHARD = "model-00001-of-00131.safetensors"

_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def _file_identity(status: os.stat_result) -> tuple[int, ...]:
    # Same nine fields as the component gate's CheckpointFileIdentity.
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_nlink,
        status.st_uid,
        status.st_gid,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


@dataclass(frozen=True)
class TensorMetadata:
    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]

    @property
    def elements(self) -> int:
        return math.prod(self.shape)

    @property
    def data_bytes(self) -> int:
        return self.data_offsets[1] - self.data_offsets[0]


@dataclass(frozen=True)
class CheckpointSummary:
    shard_count: int
    tensor_count: int
    tensor_bytes: int
    file_bytes: int
    int64_elements: int


class Qwen38Checkpoint:
    """Read tensors on demand and reject any manifest/header inconsistency."""

    def __init__(
        self,
        root: str | Path,
        *,
        file_guard: Callable[[Path, str], Path | None] | None = None,
    ):
        self.root = Path(root).resolve()
        self._file_guard = file_guard
        config_path = self.root / "config.json"
        stable_config_path = self._guard(config_path, "before-config-read")
        try:
            self.config = Qwen38Config.from_checkpoint(self.root, config_file=stable_config_path)
        finally:
            self._guard(config_path, "after-config-read")
        index_path = self.root / "model.safetensors.index.json"
        stable_index_path = self._guard(index_path, "before-index-read")
        try:
            index_bytes = stable_index_path.read_bytes()
        finally:
            self._guard(index_path, "after-index-read")
        digest = hashlib.sha256(index_bytes).hexdigest()
        if digest != INDEX_SHA256:
            raise ValueError(f"safetensors index SHA-256 must be {INDEX_SHA256}, got {digest}")
        index = json.loads(index_bytes)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or len(weight_map) != EXPECTED_TENSORS:
            raise ValueError(f"weight_map must contain exactly {EXPECTED_TENSORS} tensors")
        if any(not isinstance(name, str) or not isinstance(shard, str) for name, shard in weight_map.items()):
            raise ValueError("weight_map names and shard paths must be strings")
        if any(Path(shard).name != shard or not shard.endswith(".safetensors") for shard in weight_map.values()):
            raise ValueError("weight_map contains a non-local or non-safetensors shard path")
        self.weight_map: dict[str, str] = weight_map
        self.shards = tuple(sorted(set(weight_map.values())))
        if len(self.shards) != EXPECTED_SHARDS:
            raise ValueError(f"weight_map must contain exactly {EXPECTED_SHARDS} shards")

    def _guard(self, path: Path, phase: str) -> Path:
        if self._file_guard is None:
            return path
        stable_path = self._file_guard(path, phase)
        return path if stable_path is None else Path(stable_path)

    @lru_cache(maxsize=None)
    def _header(self, shard: str) -> tuple[int, dict[str, Any]]:
        if shard not in self.shards:
            raise KeyError(f"unknown checkpoint shard: {shard}")
        path = self.root / shard
        stable_path = self._guard(path, f"before-header-read:{shard}")
        try:
            with stable_path.open("rb") as stream:
                length_raw = stream.read(8)
                if len(length_raw) != 8:
                    raise ValueError(f"truncated safetensors length in {shard}")
                (header_length,) = struct.unpack("<Q", length_raw)
                if header_length < 2 or header_length > 16 * 1024 * 1024:
                    raise ValueError(f"invalid safetensors header length {header_length} in {shard}")
                header_raw = stream.read(header_length)
        finally:
            self._guard(path, f"after-header-read:{shard}")
        if len(header_raw) != header_length:
            raise ValueError(f"truncated safetensors header in {shard}")
        try:
            header = json.loads(header_raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid safetensors JSON header in {shard}") from error
        if not isinstance(header, dict):
            raise ValueError(f"safetensors header root is not an object in {shard}")
        return header_length, header

    def metadata(self, name: str) -> TensorMetadata:
        try:
            shard = self.weight_map[name]
        except KeyError as error:
            raise KeyError(f"tensor is absent from the pinned checkpoint: {name}") from error
        _, header = self._header(shard)
        entry = header.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"index maps {name} to {shard}, but its header does not")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if dtype not in _DTYPE_BYTES:
            raise ValueError(f"unsupported dtype {dtype!r} for {name}")
        if not isinstance(shape, list) or any(not isinstance(value, int) or value < 0 for value in shape):
            raise ValueError(f"invalid shape for {name}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(value, int) or value < 0 for value in offsets)
            or offsets[1] < offsets[0]
        ):
            raise ValueError(f"invalid data offsets for {name}")
        metadata = TensorMetadata(name, shard, dtype, tuple(shape), (offsets[0], offsets[1]))
        expected_bytes = metadata.elements * _DTYPE_BYTES[dtype]
        if metadata.data_bytes != expected_bytes:
            raise ValueError(
                f"tensor byte extent disagrees with dtype/shape for {name}: {metadata.data_bytes} != {expected_bytes}"
            )
        return metadata

    def names_with_prefix(self, prefix: str) -> tuple[str, ...]:
        return tuple(sorted(name for name in self.weight_map if name.startswith(prefix)))

    def vision_tensor_names(self) -> tuple[str, ...]:
        """The vision load domain: every ``model.visual.*`` tensor, proven to be the 333 BF16 tensors of shard 1."""

        names = self.names_with_prefix(VISION_TENSOR_PREFIX)
        if len(names) != VISION_TENSOR_COUNT:
            raise ValueError(f"expected {VISION_TENSOR_COUNT} vision tensors, got {len(names)}")
        total = 0
        for name in names:
            metadata = self.metadata(name)
            if metadata.shard != VISION_SHARD or metadata.dtype != "BF16":
                raise ValueError(
                    f"{name}: vision tensors are BF16 in {VISION_SHARD}, got {metadata.dtype} in {metadata.shard}"
                )
            total += metadata.data_bytes
        if total != VISION_TENSOR_BYTES:
            raise ValueError(f"vision tensor bytes {total} differ from the pinned {VISION_TENSOR_BYTES}")
        return names

    def vision_state_dict(self) -> dict[str, torch.Tensor]:
        """Host BF16 ``{name without the model.visual. prefix: tensor}`` of the whole tower from one open of shard 1.

        Nothing of the tower is read unless this is called: the text-only paths keep the domain omitted."""

        names = self.vision_tensor_names()
        path = self.root / VISION_SHARD
        stable_path = self._guard(path, "before-vision-read")
        tensors: dict[str, torch.Tensor] = {}
        try:
            with safe_open(stable_path, framework="pt", device="cpu") as handle:
                for name in names:
                    tensor = handle.get_tensor(name)
                    if tuple(tensor.shape) != self.metadata(name).shape or tensor.dtype != torch.bfloat16:
                        raise ValueError(f"loaded shape or dtype disagrees with header for {name}")
                    tensors[name[len(VISION_TENSOR_PREFIX) :]] = tensor
        finally:
            self._guard(path, "after-vision-read")
        return tensors

    def tensor(self, name: str) -> torch.Tensor:
        metadata = self.metadata(name)
        path = self.root / metadata.shard
        stable_path = self._guard(path, f"before-tensor-read:{name}")
        try:
            with safe_open(stable_path, framework="pt", device="cpu") as handle:
                tensor = handle.get_tensor(name)
        finally:
            self._guard(path, f"after-tensor-read:{name}")
        if tuple(tensor.shape) != metadata.shape:
            raise ValueError(f"loaded shape disagrees with header for {name}")
        return tensor

    def tensor_slice(self, name: str, selection) -> torch.Tensor:
        metadata = self.metadata(name)
        path = self.root / metadata.shard
        stable_path = self._guard(path, f"before-tensor-slice:{name}")
        try:
            with safe_open(stable_path, framework="pt", device="cpu") as handle:
                tensor = handle.get_slice(name)[selection]
        finally:
            self._guard(path, f"after-tensor-slice:{name}")
        return tensor

    def tensor_rows(self, name: str, indices: torch.Tensor) -> torch.Tensor:
        """Read sparse rows directly from one safetensors payload with ``pread``.

        This is the synchronous correctness path for the 51.2B-parameter PLE
        table.  It reads only requested rows and never maps or materializes a
        complete 800-MB table part.  Async/coalesced prefetch is a later
        optimization over this value/order contract.
        """

        metadata = self.metadata(name)
        if metadata.dtype != "BF16" or len(metadata.shape) < 2:
            raise ValueError(f"sparse row reads require a rank>=2 BF16 tensor, got {metadata.dtype} {metadata.shape}")
        if indices.dtype != torch.long:
            raise ValueError(f"row indices must be torch.long, got {indices.dtype}")
        if indices.device.type != "cpu":
            raise ValueError("row indices must reside on CPU")
        flat_indices = indices.contiguous().view(-1)
        if flat_indices.numel() and (int(flat_indices.min()) < 0 or int(flat_indices.max()) >= metadata.shape[0]):
            raise IndexError(f"row index is outside [0, {metadata.shape[0]}) for {name}")
        row_elements = math.prod(metadata.shape[1:])
        row_bytes = row_elements * _DTYPE_BYTES[metadata.dtype]
        header_length, _ = self._header(metadata.shard)
        tensor_base = 8 + header_length + metadata.data_offsets[0]
        output = torch.empty((flat_indices.numel(), row_elements), dtype=torch.bfloat16)
        path = self.root / metadata.shard
        stable_path = self._guard(path, f"before-tensor-rows:{name}")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        # A component-gate guard returns /proc/self/fd/N intentionally: that
        # kernel-owned link duplicates the already admitted descriptor.  For
        # ordinary paths, retain O_NOFOLLOW so the standalone loader cannot
        # traverse a mutable symlink.
        if stable_path.parent != Path("/proc/self/fd"):
            flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor: int | None = None
        try:
            descriptor = os.open(stable_path, flags)
            cache: dict[int, torch.Tensor] = {}
            for output_index, row_index in enumerate(flat_indices.tolist()):
                row = cache.get(row_index)
                if row is None:
                    payload = os.pread(descriptor, row_bytes, tensor_base + row_index * row_bytes)
                    if len(payload) != row_bytes:
                        raise ValueError(f"truncated sparse row {row_index} for {name}")
                    row = torch.frombuffer(bytearray(payload), dtype=torch.bfloat16).clone()
                    cache[row_index] = row
                output[output_index].copy_(row)
        finally:
            try:
                if descriptor is not None:
                    os.close(descriptor)
            finally:
                self._guard(path, f"after-tensor-rows:{name}")
        return output.reshape(*indices.shape, *metadata.shape[1:])

    def validate_all_headers(self) -> CheckpointSummary:
        indexed_names = set(self.weight_map)
        header_names: set[str] = set()
        tensor_bytes = 0
        file_bytes = 0
        int64_elements = 0

        for shard in self.shards:
            path = self.root / shard
            stable_path = self._guard(path, f"before-header-validation:{shard}")
            try:
                header_length, header = self._header(shard)
                data_end = 0
                for name, entry in header.items():
                    if name == "__metadata__":
                        continue
                    if name in header_names:
                        raise ValueError(f"tensor appears in multiple shard headers: {name}")
                    if self.weight_map.get(name) != shard:
                        raise ValueError(f"header/index shard mismatch for {name}")
                    header_names.add(name)
                    metadata = self.metadata(name)
                    tensor_bytes += metadata.data_bytes
                    data_end = max(data_end, metadata.data_offsets[1])
                    if metadata.dtype == "I64":
                        int64_elements += metadata.elements
                expected_file_size = 8 + header_length + data_end
                actual_file_size = stable_path.stat().st_size
            finally:
                self._guard(path, f"after-header-validation:{shard}")
            if actual_file_size != expected_file_size:
                raise ValueError(
                    f"shard size disagrees with header for {shard}: {actual_file_size} != {expected_file_size}"
                )
            file_bytes += actual_file_size

        missing = indexed_names - header_names
        extra = header_names - indexed_names
        if missing or extra:
            raise ValueError(f"header/index tensor mismatch: missing={sorted(missing)[:4]}, extra={sorted(extra)[:4]}")
        summary = CheckpointSummary(len(self.shards), len(header_names), tensor_bytes, file_bytes, int64_elements)
        expected = CheckpointSummary(
            EXPECTED_SHARDS,
            EXPECTED_TENSORS,
            EXPECTED_TENSOR_BYTES,
            EXPECTED_FILE_BYTES,
            EXPECTED_INT64_ELEMENTS,
        )
        if summary != expected:
            raise ValueError(f"checkpoint totals disagree with pinned release: {summary} != {expected}")
        return summary


class Qwen38CheckpointRowReader:
    """Persistent-descriptor sparse row reads with the ``tensor_rows`` value/order contract.

    The file guard brackets exactly one open per part.  Bytes read through an
    open descriptor can only change if the inode itself changes, so every later
    batch re-proves the identity captured at that open with ``fstat`` before
    and after its ``pread`` calls instead of re-resolving the path.  The
    identity is the shard FILE's (``shard``): a caller reading several parts
    of one file in a batch proves it once per touched file around the batch
    (:meth:`prove_identity`) and reads the parts with :meth:`advise_proven` /
    :meth:`read_rows_proven`; :meth:`advise` and :meth:`read_rows` prove it
    themselves.
    """

    def __init__(self, checkpoint: Qwen38Checkpoint, name: str) -> None:
        metadata = checkpoint.metadata(name)
        if metadata.dtype != "BF16" or len(metadata.shape) < 2:
            raise ValueError(f"sparse row reads require a rank>=2 BF16 tensor, got {metadata.dtype} {metadata.shape}")
        header_length, _ = checkpoint._header(metadata.shard)
        self.name = name
        self.shard = metadata.shard
        self.rows = metadata.shape[0]
        self.row_shape = metadata.shape[1:]
        self.row_bytes = math.prod(self.row_shape) * _DTYPE_BYTES[metadata.dtype]
        self._tensor_base = 8 + header_length + metadata.data_offsets[0]
        path = checkpoint.root / metadata.shard
        stable_path = checkpoint._guard(path, f"before-tensor-rows-open:{name}")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if stable_path.parent != Path("/proc/self/fd"):
            flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor: int | None = None
        try:
            try:
                descriptor = os.open(stable_path, flags)
                self._identity = _file_identity(os.fstat(descriptor))
            finally:
                checkpoint._guard(path, f"after-tensor-rows-open:{name}")
        except BaseException:
            if descriptor is not None:
                os.close(descriptor)
            raise
        self._descriptor: int | None = descriptor

    def _open_descriptor(self) -> int:
        if self._descriptor is None:
            raise RuntimeError(f"sparse row reader for {self.name} is closed")
        return self._descriptor

    def _checked_descriptor(self, when: str) -> int:
        descriptor = self._open_descriptor()
        if _file_identity(os.fstat(descriptor)) != self._identity:
            raise RuntimeError(f"checkpoint part identity drifted {when} sparse row read of {self.name}")
        return descriptor

    def prove_identity(self, when: str) -> None:
        """One ``fstat`` re-proof of the admitted inode (the identity of the whole shard file): a batch over several
        parts of one file proves it before and after the batch through any one of their readers."""

        self._checked_descriptor(when)

    def _offset(self, row_index: int) -> int:
        if not 0 <= row_index < self.rows:
            raise IndexError(f"row index {row_index} is outside [0, {self.rows}) for {self.name}")
        return self._tensor_base + row_index * self.row_bytes

    def advise(self, row_indices: Sequence[int]) -> None:
        """Start the page-ins for a batch so the following preads overlap one disk latency."""

        if getattr(os, "posix_fadvise", None) is None:
            return
        self._checked_descriptor("before advising")
        self.advise_proven(row_indices)

    def advise_proven(self, row_indices: Sequence[int]) -> None:
        """:meth:`advise` for a caller that proved the file's identity itself: one WILLNEED per distinct row."""

        fadvise = getattr(os, "posix_fadvise", None)
        if fadvise is None:
            return
        descriptor = self._open_descriptor()
        for row_index in set(row_indices):
            fadvise(descriptor, self._offset(row_index), self.row_bytes, os.POSIX_FADV_WILLNEED)

    def read_rows(self, row_indices: Sequence[int]) -> list[bytes]:
        """Return each row's raw BF16 bytes in request order, identity-checked around the batch."""

        self._checked_descriptor("before")
        payloads = self.read_rows_proven(row_indices)
        self._checked_descriptor("after")
        return payloads

    def read_rows_proven(self, row_indices: Sequence[int]) -> list[bytes]:
        """:meth:`read_rows` for a caller that proved the file's identity around its batch: one ``pread`` per
        distinct row, no ``fstat``."""

        descriptor = self._open_descriptor()
        payloads: dict[int, bytes] = {}
        for row_index in row_indices:
            if row_index not in payloads:
                payload = os.pread(descriptor, self.row_bytes, self._offset(row_index))
                if len(payload) != self.row_bytes:
                    raise ValueError(f"truncated sparse row {row_index} for {self.name}")
                payloads[row_index] = payload
        return [payloads[row_index] for row_index in row_indices]

    def read(self, indices: torch.Tensor) -> torch.Tensor:
        """Same result as ``Qwen38Checkpoint.tensor_rows`` for this tensor."""

        if indices.dtype != torch.long:
            raise ValueError(f"row indices must be torch.long, got {indices.dtype}")
        if indices.device.type != "cpu":
            raise ValueError("row indices must reside on CPU")
        flat_indices = indices.reshape(-1).tolist()
        if not flat_indices:
            return torch.empty((*indices.shape, *self.row_shape), dtype=torch.bfloat16)
        self.advise(flat_indices)
        payload = bytearray().join(self.read_rows(flat_indices))
        return torch.frombuffer(payload, dtype=torch.bfloat16).reshape(*indices.shape, *self.row_shape)

    def close(self) -> None:
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None
