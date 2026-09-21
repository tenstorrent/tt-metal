"""Read the recorded Torch ZIP tensor archives as bytes, using only the standard library.

This is deliberately limited to CPU Byte/BFloat16/Long storage and contiguous
non-gradient tensors in the supplied evidence format. It does not execute Torch
or arbitrary pickle globals; unsupported layouts/types fail explicitly.
"""
import collections
from dataclasses import dataclass
import gzip
import io
import lzma
import math
import pickle
from pathlib import Path
import struct
import zipfile


@dataclass(frozen=True)
class StorageKind:
    name: str
    itemsize: int


@dataclass(frozen=True)
class Storage:
    kind: StorageKind
    data: bytes


@dataclass(frozen=True)
class TensorBytes:
    storage: Storage
    offset: int
    shape: tuple
    stride: tuple

    def packed(self):
        expected_stride = 1
        for size, stride in reversed(list(zip(self.shape, self.stride))):
            if size > 1 and stride != expected_stride:
                raise ValueError("Unsupported non-contiguous tensor")
            expected_stride *= size
        itemsize = self.storage.kind.itemsize
        start, end = self.offset * itemsize, (self.offset + math.prod(self.shape)) * itemsize
        if not 0 <= start <= end <= len(self.storage.data):
            raise ValueError("Tensor exceeds recorded storage")
        return self.storage.data[start:end]

    def byte_values(self):
        if self.storage.kind.name != "ByteStorage":
            raise ValueError("Expected uint8 byte tensor")
        return self.packed()

    def int64_rows(self):
        if self.storage.kind.name != "LongStorage" or len(self.shape) != 2:
            raise ValueError("Expected a 2D int64 tensor")
        values = [x[0] for x in struct.iter_unpack("<q", self.packed())]
        width = self.shape[1]
        return [values[i : i + width] for i in range(0, len(values), width)]


def rebuild(storage, offset, size, stride, requires_grad, backward_hooks, metadata=None):
    if requires_grad or backward_hooks or metadata:
        raise ValueError("Unsupported tensor metadata")
    if not isinstance(storage, Storage) or offset < 0 or len(size) != len(stride):
        raise ValueError("Invalid tensor descriptor")
    if not all(isinstance(x, int) and x >= 0 for x in (*size, *stride)):
        raise ValueError("Invalid tensor dimensions")
    return TensorBytes(storage, offset, tuple(size), tuple(stride))


class ArchiveUnpickler(pickle.Unpickler):
    def __init__(self, stream, archive, prefix):
        super().__init__(stream)
        self.archive, self.prefix, self.storages = archive, prefix, {}

    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return rebuild
        kinds = {"ByteStorage": 1, "BFloat16Storage": 2, "LongStorage": 8}
        if module == "torch" and name in kinds:
            return StorageKind(name, kinds[name])
        raise pickle.UnpicklingError(f"Unsupported pickle global: {module}.{name}")

    def persistent_load(self, identifier):
        if not isinstance(identifier, tuple) or len(identifier) != 5 or identifier[0] != "storage":
            raise pickle.UnpicklingError("Unsupported persistent reference")
        _, kind, key, location, count = identifier
        if not isinstance(kind, StorageKind) or location != "cpu" or not str(key).isdigit() or count < 0:
            raise pickle.UnpicklingError("Unsupported tensor storage")
        cache_key = (key, kind)
        if cache_key not in self.storages:
            data = self.archive.read(self.prefix + "data/" + str(key))
            if len(data) != count * kind.itemsize:
                raise ValueError("Storage size mismatch")
            self.storages[cache_key] = Storage(kind, data)
        if len(self.storages[cache_key].data) != count * kind.itemsize:
            raise ValueError("Conflicting storage sizes")
        return self.storages[cache_key]


def load_archive(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else lzma.open if path.suffix == ".xz" else open
    with opener(path, "rb") as stream:
        payload = stream.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        candidates = [n for n in names if n.endswith("/data.pkl")]
        if len(candidates) != 1 or len(names) != len(set(names)):
            raise ValueError("Unexpected Torch archive layout")
        prefix = candidates[0][: -len("data.pkl")]
        if archive.read(prefix + "byteorder") != b"little":
            raise ValueError("Only the recorded little-endian format is supported")
        result = ArchiveUnpickler(io.BytesIO(archive.read(candidates[0])), archive, prefix).load()
        if not isinstance(result, dict):
            raise ValueError("Expected a capture dictionary")
        return result
