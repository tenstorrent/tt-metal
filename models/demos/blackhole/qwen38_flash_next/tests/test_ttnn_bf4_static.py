# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
import os
import re
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from functools import reduce
from operator import getitem
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

import models.demos.blackhole.qwen38_flash_next.ttnn.bf4 as bf4_module
import models.demos.blackhole.qwen38_flash_next.ttnn.contracts as contracts_module
from models.demos.blackhole.qwen38_flash_next.checkpoint import (
    CHECKPOINT_FILE_MANIFEST_SHA256,
    CHECKPOINT_TENSOR_MANIFEST_SHA256,
    PINNED_CHECKPOINT_REVISION,
)
from models.demos.blackhole.qwen38_flash_next.config import CONFIG_SHA256
from models.demos.blackhole.qwen38_flash_next.ttnn.bf4 import (
    BF4Artifact,
    BF4CacheIdentity,
    BF4CleanupError,
    BF4LayerRecord,
    Qwen38BF4Cache,
    Qwen38BF4Streamer,
    packed_bf4_bytes_per_device,
    packed_bf4_model_bytes_per_device,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import BACKBONE_LAYERS, Qwen38TTNNBuilder
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import (
    Qwen38MeshContract,
    TensorPlacement,
    replicate_tensor_2d_mesh_mapper,
)

RING7_WORKERS = ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0))
RING7_ORDER = (6, 5, 4, 3, 2, 1, 0)  # RING7_WORKERS as the packing orders it: banks by worker (y, x) descending
EXPERT_RANGES = ((0, 128), (128, 256), (256, 384), (384, 512))
MEMORY_CONFIGS = SimpleNamespace(w0_w1=object(), w2=object())  # what get_weight_mem_configs returns per slot
_TEST_TENSORBIN_HEADER_BYTES = 64
CONVERTER_SOURCES = (
    ("models/demos/blackhole/qwen38_flash_next/ttnn/bf4.py", "1" * 64),
    ("tt_metal/impl/data_format/bfloat4.cpp", "2" * 64),
)


def _identity() -> BF4CacheIdentity:
    return BF4CacheIdentity(
        checkpoint_revision=PINNED_CHECKPOINT_REVISION,
        checkpoint_config_sha256=CONFIG_SHA256,
        checkpoint_file_manifest_sha256=CHECKPOINT_FILE_MANIFEST_SHA256,
        checkpoint_hash_manifest_sha256=CHECKPOINT_TENSOR_MANIFEST_SHA256,
        converter_sources=CONVERTER_SOURCES,
        mesh_shape=(1, 4),
        physical_ids=(0, 1, 2, 3),
        ring_size=7,
        dram_bank_ring_order=RING7_ORDER,
    )


def _mesh_shapes(identity: BF4CacheIdentity) -> dict[str, tuple[int, ...]]:
    """The coordinate-local shard shapes a mesh tensor presents (dim 2 = one device's experts)."""

    return {
        name: (*shape[:2], identity.experts_per_device, *shape[3:])
        for name, shape in bf4_module.canonical_packed_shapes(ring_size=identity.ring_size).items()
    }


def _fast_sparse_digest_fd(descriptor: int) -> str:
    """Test-only digest that samples the materialized edges of sparse fixtures."""

    metadata = os.fstat(descriptor)
    window = min(4096, metadata.st_size)
    digest = hashlib.sha256(str(metadata.st_size).encode())
    digest.update(os.pread(descriptor, window, 0))
    if metadata.st_size > window:
        digest.update(os.pread(descriptor, window, metadata.st_size - window))
    return digest.hexdigest()


def _fast_sparse_digest(path: Path) -> str:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        return _fast_sparse_digest_fd(descriptor)
    finally:
        os.close(descriptor)


def _write_sparse_tensorbin(path: Path, *, payload_bytes: int, marker: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_size = bf4_module.TENSORBIN_HEADER_PREFIX_BYTES + _TEST_TENSORBIN_HEADER_BYTES + payload_bytes
    with path.open("wb") as stream:
        stream.write(_TEST_TENSORBIN_HEADER_BYTES.to_bytes(8, byteorder="little", signed=False))
        stream.write(bytes(_TEST_TENSORBIN_HEADER_BYTES))
        stream.write(marker)
        stream.seek(file_size - 1)
        stream.write(b"\0")


def _copy_sparse_tensorbin(source: Path, destination: Path) -> None:
    size = source.stat().st_size
    window = min(4096, size)
    with source.open("rb") as input_stream, destination.open("wb") as output_stream:
        prefix = input_stream.read(window)
        input_stream.seek(size - window)
        suffix = input_stream.read(window)
        output_stream.write(prefix)
        output_stream.seek(size - window)
        output_stream.write(suffix)
        output_stream.truncate(size)


def _write_layer(
    cache: Qwen38BF4Cache, layer_index: int, *, namespace: str = "backbone"
) -> tuple[BF4LayerRecord, dict[str, Path]]:
    paths = {name: bf4_module._tensorbin_path(cache._base(namespace, layer_index, name)) for name in ("w0_w1", "w2")}
    logical_shapes = bf4_module.canonical_packed_shapes(ring_size=cache.identity.ring_size)
    artifacts = {}
    for name, path in paths.items():
        payload = f"packed:{namespace}:{layer_index}:{name}".encode()
        _write_sparse_tensorbin(
            path, payload_bytes=bf4_module._packed_payload_bytes(logical_shapes[name]), marker=payload
        )
        artifacts[name] = BF4Artifact(
            name=name,
            relative_path=str(path.relative_to(cache.root)),
            sha256=_fast_sparse_digest(path),
            bytes=path.stat().st_size,
            logical_shape=logical_shapes[name],
        )
    record = BF4LayerRecord(
        namespace=namespace,
        layer_index=layer_index,
        expert_ranges=EXPERT_RANGES,
        ring_size=cache.identity.ring_size,
        **artifacts,
    )
    cache._record(record)
    return record, paths


def _artifact_name(path) -> str:
    """The slot a native load reads; the path must be the canonical regular tensorbin."""
    path = Path(path)
    assert path.is_absolute() and path == path.resolve()
    assert path.suffix == ".tensorbin" and path.is_file() and not path.is_symlink()
    return "w0_w1" if path.name.startswith("w0_w1_") else "w2"


class _TrackingMeshContract:
    def __init__(self, physical_ids: tuple[int, int, int, int], mesh: object) -> None:
        self.physical_ids = physical_ids
        self.mesh = mesh
        self.mesh_validations = 0
        self.tensor_validations = 0

    def validate_mesh(self, mesh: object) -> None:
        if mesh is not self.mesh:
            raise RuntimeError("unexpected mesh object")
        self.mesh_validations += 1

    def validate_tensor(self, _tensor, *, placement, shard_dim) -> None:
        if placement is not TensorPlacement.EXPERT_SHARDED or shard_dim != 2:
            raise RuntimeError("unexpected BF4 tensor placement")
        self.tensor_validations += 1


class _FakeMeshDevice:
    shape = (1, 4)

    @staticmethod
    def get_device_ids():
        return (0, 1, 2, 3)

    @staticmethod
    def get_device_id(coordinate):
        row, column = tuple(coordinate)
        assert row == 0
        return column


class _FakeLocalTensor:
    def __init__(self, mesh, physical_id, address):
        self._device = mesh
        self._coordinate = (0, physical_id)
        self._address = address

    def device(self):
        return self._device

    def device_coords(self):
        return (self._coordinate,)

    def buffer_address(self):
        return self._address


class _FakeTopology:
    @staticmethod
    def distribution_shape():
        return (1, 4)

    @staticmethod
    def mesh_coords():
        return ((0, 0), (0, 1), (0, 2), (0, 3))


class _FakeBackingTensor:
    _next_backing = 1
    _mesh = _FakeMeshDevice()

    def __init__(self, *, backing=None):
        self.backing = type(self)._next_backing if backing is None else backing
        if backing is None:
            type(self)._next_backing += 1
        self.locals = tuple(
            _FakeLocalTensor(self._mesh, physical_id, self.backing * 0x10000 + physical_id * 0x1000)
            for physical_id in range(4)
        )

    def device(self):
        return self._mesh

    @staticmethod
    def tensor_topology():
        return _FakeTopology()


class _FakeBF4Tensor(_FakeBackingTensor):
    def __init__(self, memory_config, shape) -> None:
        super().__init__()
        self._memory_config = memory_config
        self.shape = shape
        self.dtype = bf4_module.ttnn.bfloat4_b

    def memory_config(self):
        return self._memory_config


class _NamedTensor(_FakeBackingTensor):
    def __init__(self, name, backing):
        super().__init__(backing=backing)
        self.name = name

    def __str__(self):
        return self.name


@pytest.fixture(autouse=True)
def _fast_sparse_hashes():
    # Sparse fixtures preserve the exact multi-GiB file-size/header contract without reading holes.  Production
    # continues to use full SHA-256.
    with mock.patch.object(bf4_module, "_sha256_fd", side_effect=_fast_sparse_digest_fd), mock.patch.object(
        contracts_module.ttnn, "get_device_tensors", side_effect=lambda tensor: tensor.locals
    ):
        yield


@pytest.fixture
def layer0(tmp_path):
    """A cache holding backbone layer 0 behind a tracking contract; ``load_tensor`` loads a slot as a fake tensor of
    ``shapes[slot]`` (the coordinate-local shard shape) and appends it to ``loaded``."""
    identity, mesh = _identity(), object()
    contract = _TrackingMeshContract(identity.physical_ids, mesh)
    cache = Qwen38BF4Cache(tmp_path, identity, contract)
    record, paths = _write_layer(cache, 0)
    layer = SimpleNamespace(identity=identity, mesh=mesh, contract=contract, cache=cache, record=record, paths=paths)
    layer.shapes, layer.loaded, layer.retained = _mesh_shapes(identity), [], []

    def load_tensor(path, *, device):
        assert device is mesh
        name = _artifact_name(path)
        layer.loaded.append(_FakeBF4Tensor(getattr(MEMORY_CONFIGS, name), layer.shapes[name]))
        return layer.loaded[-1]

    layer.load_tensor = load_tensor
    return layer


@dataclass(frozen=True)
class _RetainedFD:
    """A verification FD a load retained and the (device, inode) it was opened on.  A closed FD's number goes to the
    process's next open, so the FD is open only while its number still refers to that file."""

    descriptor: int
    file: tuple[int, int]

    @property
    def open(self) -> bool:
        try:
            status = os.fstat(self.descriptor)
        except OSError:
            return False
        return (status.st_dev, status.st_ino) == self.file


@contextmanager
def _patched_load(layer, load_tensor=None, *, deallocate_error=None):
    """The native side of a cache load on fakes (the live ring, the slot memory configs, ``load_tensor``); the
    model-owned verification FDs a load retains are observed, lifetime unchanged, into ``layer.retained``.  Yields the
    ``ttnn.deallocate`` mock."""
    original = bf4_module._verified_artifact_fd

    @contextmanager
    def record(path, artifact, *, expected_signature=None):
        with original(path, artifact, expected_signature=expected_signature) as verified:
            if expected_signature is not None:
                status = os.fstat(verified.descriptor)
                layer.retained.append(_RetainedFD(verified.descriptor, (status.st_dev, status.st_ino)))
            yield verified

    with (
        mock.patch.object(bf4_module, "_verified_artifact_fd", side_effect=record),
        mock.patch.object(bf4_module, "qualify_live_bf4_ring", return_value=RING7_ORDER),
        mock.patch.object(bf4_module.ttnn, "load_tensor", side_effect=load_tensor or layer.load_tensor),
        mock.patch.object(bf4_module.ttnn.experimental, "get_weight_mem_configs", return_value=MEMORY_CONFIGS),
        mock.patch.object(bf4_module.ttnn, "deallocate", side_effect=deallocate_error) as deallocate,
    ):
        yield deallocate


def _second_load_fails(layer, message: str):
    def load_tensor(path, *, device):
        tensor = layer.load_tensor(path, device=device)
        if len(layer.loaded) == 2:
            raise RuntimeError(message)
        return tensor

    return load_tensor


def test_replication_mapper_is_explicit_two_axis_topology():
    mesh = object()
    mapper = object()
    with (
        mock.patch.object(contracts_module.ttnn, "PlacementReplicate", side_effect=["row-replicate", "tp-replicate"]),
        mock.patch.object(contracts_module.ttnn, "MeshShape", return_value="mesh-1x4") as mesh_shape,
        mock.patch.object(contracts_module.ttnn, "MeshMapperConfig", return_value="explicit-config") as config,
        mock.patch.object(contracts_module.ttnn, "create_mesh_mapper", return_value=mapper) as create,
    ):
        assert replicate_tensor_2d_mesh_mapper(mesh) is mapper
    mesh_shape.assert_called_once_with(1, 4)
    config.assert_called_once_with(["row-replicate", "tp-replicate"], "mesh-1x4")
    create.assert_called_once_with(mesh, "explicit-config")


def test_exact_packer_storage_not_raw_bf4_storage():
    # moe_compute's compact owned-column layout: the 8-bank ring stores the 2560/640 expert without padding
    # (3,200 gate/up + 1,600 W2 tiles, 20-tile transactions, a half-width last W2 iteration); the 7-bank ring
    # keeps 14-tile transactions (K padded to 7-tile blocks) and W2's grouped form.
    assert packed_bf4_bytes_per_device(ring_size=7) == (260_112_384, 130_056_192)
    assert packed_bf4_bytes_per_device(ring_size=8) == (235_929_600, 117_964_800)
    assert packed_bf4_model_bytes_per_device(ring_size=7) == 19_118_260_224
    assert packed_bf4_model_bytes_per_device(ring_size=8) == 17_340_825_600
    assert bf4_module.canonical_packed_shapes(ring_size=7) == {
        "w0_w1": (7, 1, 512, 18, 224, 128),
        "w2": (7, 1, 512, 3, 672, 128),
    }
    assert bf4_module.canonical_packed_shapes(ring_size=8) == {
        "w0_w1": (8, 1, 512, 10, 320, 128),
        "w2": (8, 1, 512, 5, 320, 128),
    }
    for ring_size in (7, 8):
        shapes = bf4_module.canonical_packed_shapes(ring_size=ring_size)
        payload = tuple(bf4_module._packed_payload_bytes(shapes[name]) // 4 for name in ("w0_w1", "w2"))
        assert payload == packed_bf4_bytes_per_device(ring_size=ring_size)
    # the cache identity pins the packed layout version
    assert bf4_module.LAYOUT_VERSION == asdict(_identity())["layout_version"] == 2


def test_tensorbin_header_and_exact_payload_contract_rejects_size_aliases(expect_error, tmp_path):
    def validate(
        name: str,
        *,
        header_size: int,
        payload_bytes: int,
        prefix_bytes: int = 8,
        materialized_header_bytes: int | None = None,
    ):
        path = tmp_path / f"{name}.tensorbin"
        with path.open("wb") as stream:
            stream.write(header_size.to_bytes(8, byteorder="little", signed=False)[:prefix_bytes])
            stream.write(
                bytes(min(header_size, 128) if materialized_header_bytes is None else materialized_header_bytes)
            )
            stream.write(bytes(payload_bytes))
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        try:
            return bf4_module._validate_tensorbin_payload_fd(
                descriptor, signature=bf4_module._artifact_fd_signature(descriptor), expected_payload_bytes=32
            )
        finally:
            os.close(descriptor)

    assert validate("valid", header_size=16, payload_bytes=32) == 16
    for name, sizes, match in (
        ("tiny", {"header_size": 0, "payload_bytes": 0, "prefix_bytes": 4}, "truncated before"),
        (
            "truncated",
            {"header_size": 64, "payload_bytes": 0, "materialized_header_bytes": 32},
            "header size/alignment",
        ),
        ("misaligned", {"header_size": 10, "payload_bytes": 32}, "header size/alignment"),
        (
            "oversized-header",
            {"header_size": bf4_module.MAX_TENSORBIN_HEADER_BYTES + 8, "payload_bytes": 0},
            "header size/alignment",
        ),
        ("short-payload", {"header_size": 16, "payload_bytes": 31}, "payload byte count"),
        ("oversized-payload", {"header_size": 16, "payload_bytes": 33}, "payload byte count"),
    ):
        with expect_error(RuntimeError, match):
            validate(name, **sizes)


def test_self_hashed_tiny_tensorbin_cannot_claim_a_canonical_global_shape(expect_error, tmp_path):
    identity = _identity()
    cache = Qwen38BF4Cache(tmp_path, identity, Qwen38MeshContract(identity.physical_ids))
    _, paths = _write_layer(cache, 0)
    tiny = paths["w0_w1"]
    with tiny.open("wb") as stream:
        stream.write((16).to_bytes(8, byteorder="little", signed=False))
        stream.write(bytes(16))
        stream.write(b"tiny-payload")
    document = json.loads(cache.manifest_path.read_text(encoding="utf-8"))
    artifact = document["layers"]["backbone:0"]["w0_w1"]
    artifact["bytes"] = tiny.stat().st_size
    artifact["sha256"] = _fast_sparse_digest(tiny)
    bf4_module._atomic_json(cache.manifest_path, document)
    with expect_error(RuntimeError, "payload byte count"):
        cache.verify_layer("backbone", 0)


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"ring_size": 7.0}, "exact integer"),
        ({"format_version": True}, "exact integer"),
        ({"mesh_shape": ("1", 4)}, "exact integer"),
        ({"dram_bank_ring_order": (6.0, *RING7_ORDER[1:])}, "exact integer"),
        *(({"layout_version": stale}, "encoding identity|exact integers") for stale in (1, 3, True, 2.0)),
        *(
            ({"dram_bank_ring_order": order}, "every live DRAM bank once")
            for order in ((0,) * 7, RING7_ORDER[:6], (*RING7_ORDER[:6], 7))
        ),
        # the complete file and tensor manifests are distinct pins
        ({"checkpoint_file_manifest_sha256": "0" * 64}, "checkpoint_file_manifest_sha256"),
        ({"checkpoint_hash_manifest_sha256": "0" * 64}, "checkpoint_hash_manifest_sha256"),
        *(
            ({"converter_sources": malformed}, "converter sources")
            for malformed in (
                (),
                tuple(reversed(CONVERTER_SOURCES)),
                (CONVERTER_SOURCES[0], CONVERTER_SOURCES[0]),
                (("models/demos/blackhole/qwen38_flash_next/ttnn/bf4.py", "X" * 64),),
                (("models/demos/blackhole/qwen38_flash_next/ttnn/bf4.py", "1" * 40),),
                [tuple(CONVERTER_SOURCES[0])],
            )
        ),
    ],
)
def test_cache_identity_rejects_aliases_stale_layouts_partial_rings_and_malformed_pins(expect_error, changes, match):
    with expect_error(ValueError, match):
        replace(_identity(), **changes)


def test_shape_normalization_rejects_bool_float_and_string_aliases(expect_error):
    canonical = bf4_module.canonical_packed_shapes(ring_size=7)["w0_w1"]
    for shape in (
        (canonical[0], True, *canonical[2:]),
        (*canonical[:2], 512.0, *canonical[3:]),
        (*canonical[:-1], "128"),
    ):
        with expect_error(RuntimeError, "shape contains"):
            bf4_module._native_integer_shape(shape, label="BF4 conversion")


def test_canonical_shapes_are_the_layout_packers_output():
    # The cache derives its slot shapes from the geometry helpers of ttnn.experimental.moe_compute_utils; the
    # packers themselves (one zero expert, L = E = 1) must lay the bytes out in exactly those shapes.
    import torch

    for ring_size in (7, 8):
        w01_map, w2_map = bf4_module.ring_shard_maps(2560, 640, ring_size)
        gate = torch.zeros(1, 1, 2560, 640, dtype=torch.bfloat16)
        down = torch.zeros(1, 1, 640, 2560, dtype=torch.bfloat16)
        packed = {
            "w0_w1": bf4_module.prepare_w0_w1_tensor_for_moe_compute(gate, gate, 1, 1, 2560, 640, w01_map),
            "w2": bf4_module.prepare_w2_tensor_for_moe_compute(down, 1, 1, 640, 2560, w2_map, w01_map),
        }
        shapes = bf4_module.canonical_packed_shapes(ring_size=ring_size)
        for name, tensor in packed.items():
            assert tuple(tensor.shape) == (*shapes[name][:2], 1, *shapes[name][3:]), (ring_size, name)
        w01_bytes, w2_bytes = packed_bf4_bytes_per_device(ring_size=ring_size, experts_per_device=1)
        assert w01_bytes == packed["w0_w1"].numel() // 1024 * 576
        assert w2_bytes == packed["w2"].numel() // 1024 * 576


def test_manifest_round_trip_preserves_tuple_contract_and_hashes(expect_error, tmp_path):
    identity = _identity()
    contract = Qwen38MeshContract(identity.physical_ids)
    cache = Qwen38BF4Cache(tmp_path, identity, contract)
    record, paths = _write_layer(cache, 0)
    with mock.patch.object(bf4_module, "_sha256_fd", wraps=bf4_module._sha256_fd) as sha256:
        assert cache.verify_layer("backbone", 0) == record
        assert sha256.call_count == 2
        assert cache.verify_layer("backbone", 0) == record
        assert sha256.call_count == 2, "unchanged artifacts must not be re-hashed per decode"

        assert Qwen38BF4Cache(tmp_path, identity, contract).verify_layer("backbone", 0) == record
        assert sha256.call_count == 4, "a fresh cache object must establish its own verification"

        original_stat = paths["w0_w1"].stat()
        payload_offset = bf4_module.TENSORBIN_HEADER_PREFIX_BYTES + _TEST_TENSORBIN_HEADER_BYTES
        with paths["w0_w1"].open("r+b") as stream:
            stream.seek(payload_offset)
            original_byte = stream.read(1)
            stream.seek(payload_offset)
            stream.write(bytes((original_byte[0] ^ 1,)))
        # Restoring mtime does not defeat the inode/ctime session guard.
        os.utime(paths["w0_w1"], ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        with expect_error(RuntimeError, "hash validation"):
            cache.verify_layer("backbone", 0)
        assert sha256.call_count == 5

        untrusted_fresh_cache = Qwen38BF4Cache(tmp_path, identity, contract)
        with expect_error(RuntimeError, "hash validation"):
            untrusted_fresh_cache.verify_layer("backbone", 0)
        assert sha256.call_count == 6, "fresh cache state must not inherit another object's trust"


def test_complete_backbone_hashes_once_then_streams_by_verified_identity(expect_error, layer0):
    cache, mesh, contract = layer0.cache, layer0.mesh, layer0.contract
    for layer_index in range(1, BACKBONE_LAYERS):
        _write_layer(cache, layer_index)
    builder = object.__new__(Qwen38TTNNBuilder)
    builder.bf4_cache = cache
    builder.placement = SimpleNamespace(expert_ranges=EXPERT_RANGES)
    builder.identity = SimpleNamespace(ring_size=layer0.identity.ring_size)
    streamer = Qwen38BF4Streamer(cache, mesh)
    builder.expert_streamer = streamer
    assert builder.bf4_cache is builder.expert_streamer.cache

    with (
        _patched_load(layer0) as deallocate,
        mock.patch.object(cache, "_read_manifest", wraps=cache._read_manifest) as read_manifest,
        mock.patch.object(bf4_module, "_sha256_fd", wraps=bf4_module._sha256_fd) as sha256,
    ):
        records = builder.require_complete_backbone_bf4()
        assert tuple(record.layer_index for record in records) == tuple(range(BACKBONE_LAYERS))
        assert sha256.call_count == 2 * BACKBONE_LAYERS

        for traversal in range(2):
            for layer_index in range(BACKBONE_LAYERS):
                with streamer.layer(layer_index) as tensors:
                    assert streamer._active == ("backbone", layer_index) and len(tensors) == 2
                    if traversal == layer_index == 0:
                        with expect_error(RuntimeError, "already owns active layer"), streamer.layer(1):
                            pass
            assert streamer._active is None

        assert (
            sha256.call_count == 2 * BACKBONE_LAYERS
        ), "decode traversal must bind fresh FDs to the session hash without rehashing the model"
        assert read_manifest.call_count == 3 * BACKBONE_LAYERS
        assert bf4_module.qualify_live_bf4_ring.call_count == 2 * BACKBONE_LAYERS
        assert bf4_module.ttnn.load_tensor.call_count == deallocate.call_count == 4 * BACKBONE_LAYERS
        assert contract.mesh_validations == 1 + 2 * BACKBONE_LAYERS
        assert contract.tensor_validations == len(layer0.retained) == 4 * BACKBONE_LAYERS
        assert not any(fd.open for fd in layer0.retained)


def test_bf4_load_retains_verified_fds_through_topology_validation(layer0):
    original_validate = layer0.contract.validate_tensor

    def validate_while_retained(tensor, *, placement, shard_dim):
        assert len(layer0.retained) == 2 and all(fd.open for fd in layer0.retained)
        return original_validate(tensor, placement=placement, shard_dim=shard_dim)

    with _patched_load(layer0) as deallocate, mock.patch.object(
        layer0.contract, "validate_tensor", side_effect=validate_while_retained
    ):
        tensors = layer0.cache.load_layer(layer0.mesh, layer_index=0)
    assert len(tensors) == 2
    assert not any(fd.open for fd in layer0.retained)
    deallocate.assert_not_called()


@pytest.mark.parametrize(
    "name, shape, match",
    [
        ("w2", (7, 1, 128, 3, 672, 160), "loaded shape"),  # a tile wider than the slot's shard
        ("w0_w1", (7, True, 512, 18, 224, 128), "shape contains"),
        ("w0_w1", (7, 1, 512.0, 18, 224, 128), "shape contains"),
        ("w0_w1", (7, 1, 512, 18, 224, "128"), "shape contains"),
    ],
    ids=["wrong shape", "bool alias", "float alias", "string alias"],
)
def test_bf4_load_rejects_a_wrong_or_aliased_loaded_shape_and_releases_both_tensors(
    expect_error, layer0, name, shape, match
):
    layer0.shapes[name] = shape
    with _patched_load(layer0) as deallocate, expect_error(RuntimeError, match):
        layer0.cache.load_layer(layer0.mesh, layer_index=0)
    assert [call.args[0] for call in deallocate.call_args_list] == layer0.loaded


def test_bf4_load_rejects_path_replacement_after_native_reads_and_closes_fds(expect_error, layer0):
    artifact = layer0.paths["w0_w1"]

    def replace_after_both_reads(path, *, device):
        tensor = layer0.load_tensor(path, device=device)
        if len(layer0.loaded) == 2:
            replacement = artifact.with_name("replacement.tensorbin")
            _copy_sparse_tensorbin(artifact, replacement)
            os.replace(replacement, artifact)
        return tensor

    with _patched_load(layer0, replace_after_both_reads) as deallocate, expect_error(
        RuntimeError, "identity changed while its descriptor was retained"
    ):
        layer0.cache.load_layer(layer0.mesh, layer_index=0)
    assert len(layer0.loaded) == 2
    assert [call.args[0] for call in deallocate.call_args_list] == layer0.loaded
    assert not any(fd.open for fd in layer0.retained)


def test_bf4_load_rechecks_in_place_mutation_after_topology_validation(expect_error, layer0):
    contract = layer0.contract
    original_validate = contract.validate_tensor

    def mutate_during_topology(tensor, *, placement, shard_dim):
        assert all(fd.open for fd in layer0.retained)
        if contract.tensor_validations == 0:
            with layer0.paths["w2"].open("ab") as stream:
                stream.write(b"mutation")
        return original_validate(tensor, placement=placement, shard_dim=shard_dim)

    with (
        _patched_load(layer0) as deallocate,
        mock.patch.object(contract, "validate_tensor", side_effect=mutate_during_topology),
        expect_error(RuntimeError, "changed while its descriptor was retained"),
    ):
        layer0.cache.load_layer(layer0.mesh, layer_index=0)
    assert contract.tensor_validations == 2
    assert [call.args[0] for call in deallocate.call_args_list] == layer0.loaded
    assert not any(fd.open for fd in layer0.retained)


def test_bf4_load_failure_closes_both_fds_and_releases_first_tensor(expect_error, layer0):
    with _patched_load(layer0, _second_load_fails(layer0, "synthetic second load failure")) as deallocate, expect_error(
        RuntimeError, "synthetic second load failure"
    ):
        layer0.cache.load_layer(layer0.mesh, layer_index=0)
    deallocate.assert_called_once_with(layer0.loaded[0])
    assert len(layer0.retained) == 2
    assert not any(fd.open for fd in layer0.retained)


def test_bf4_cache_load_aggregates_load_and_cleanup_failures(layer0):
    cache = layer0.cache
    assert cache.verify_layer("backbone", 0) == layer0.record
    with (
        _patched_load(
            layer0,
            _second_load_fails(layer0, "synthetic cache load failure"),
            deallocate_error=RuntimeError("synthetic cache cleanup failure"),
        ) as deallocate,
        pytest.raises(BF4CleanupError, match="cache load failed") as raised,  # allow-pytest.raises: .value
    ):
        cache._load_verified_tensors(
            layer0.mesh,
            record=layer0.record,
            w01_path=layer0.paths["w0_w1"],
            w2_path=layer0.paths["w2"],
            memory_configs=MEMORY_CONFIGS,
        )
    error, first = raised.value, layer0.loaded[0]
    assert "synthetic cache load failure" in str(error.primary_error)
    assert len(error.cleanup_errors) == 1
    assert error.unreleased_tensors == (first,)
    assert len(error.tensor_cleanup_outcomes) == 2
    first_outcome, second_outcome = error.tensor_cleanup_outcomes
    assert first_outcome.release_attempted and not first_outcome.released and first_outcome.tensor is first
    assert second_outcome.tensor is None
    deallocate.assert_called_once_with(first)
    assert not any(fd.open for fd in layer0.retained)


def test_bf4_verification_rejects_symlink_artifact_without_loading(expect_error, layer0):
    artifact = layer0.paths["w0_w1"]
    target = artifact.with_name("target.tensorbin")
    _copy_sparse_tensorbin(artifact, target)
    artifact.unlink()
    artifact.symlink_to(target)
    with _patched_load(layer0), expect_error(RuntimeError, "unavailable or invalid"):
        layer0.cache.load_layer(layer0.mesh, layer_index=0)
    assert layer0.loaded == []


def test_manifest_mutation_cannot_redirect_verified_or_loaded_layer(expect_error, layer0):
    cache = layer0.cache
    cache.verify_layer("backbone", 0)
    baseline = json.loads(cache.manifest_path.read_text(encoding="utf-8"))
    key = "backbone:0"
    slot = "backbone/layer-01/w0_w1_dtype_BFLOAT4_B_layout_TILE.tensorbin"
    mutations = (  # (the refusal, the manifest path, the value written there)
        ("record identity", ("layers", key, "layer_index"), 1),
        ("expert ownership", ("layers", key, "expert_ranges"), [[128, 256], [0, 128], [256, 384], [384, 512]]),
        ("requested slot", ("layers", key, "w0_w1", "relative_path"), slot),
        ("canonical slot shape", ("layers", key, "w0_w1", "logical_shape"), [7, 1, 512, 18, 224, 160]),
        ("scalar schema", ("layers", key, "w2", "bytes"), True),
        ("manifest schema", ("layers", key, "w2", "unbound"), "field"),
        ("identity digest", ("identity_key",), "0" * 64),
        ("manifest identity", ("identity", "ring_size"), 7.0),
        ("manifest identity", ("identity", "physical_ids", 0), False),
        ("manifest format", ("format_version",), True),
        ("manifest format", ("format_version",), 1.0),
    )
    with _patched_load(layer0):
        for expected, (*parents, leaf), value in mutations:
            document = copy.deepcopy(baseline)
            reduce(getitem, parents, document)[leaf] = value
            bf4_module._atomic_json(cache.manifest_path, document)
            with expect_error(RuntimeError, expected):
                cache.load_layer(layer0.mesh, layer_index=0)
    assert layer0.loaded == []


def test_cache_rejects_layer_aliases_and_non_target_slots_before_manifest_access(expect_error, tmp_path):
    identity = _identity()
    cache = Qwen38BF4Cache(tmp_path, identity, Qwen38MeshContract(identity.physical_ids))
    with mock.patch.object(cache, "_read_manifest", side_effect=AssertionError("manifest was read")):
        for invalid in (True, 0.0, "0"):
            with expect_error(TypeError, "exact integer"):
                cache.verify_layer("backbone", invalid)
        with expect_error(ValueError, r"\[0,48\)"):
            cache.verify_layer("backbone", BACKBONE_LAYERS)
        with expect_error(ValueError, r"\[0,1\)"):
            cache.verify_layer("mtp", 1)


def test_layer_host_packing_fills_four_canonical_ranges_without_cat():
    shapes = bf4_module.canonical_packed_shapes(ring_size=7)
    allocations = []
    copies = []
    loaded = []

    class Prepared:
        def __init__(self, name, device_index, shape):
            self.name = name
            self.device_index = device_index
            self.shape = shape
            self.dtype = bf4_module.torch.bfloat16

    class DestinationSlice:
        def __init__(self, name, start, length):
            self.name = name
            self.start = start
            self.length = length

        def copy_(self, source):
            copies.append((self.name, self.start, self.length, source.name, source.device_index))
            return self

    class Destination:
        def __init__(self, shape, dtype):
            self.name = "w0_w1" if not allocations else "w2"
            self.shape = shape
            self.dtype = dtype
            allocations.append((self.name, shape, dtype))

        def narrow(self, dimension, start, length):
            if dimension != 2:
                raise AssertionError(f"unexpected expert dimension {dimension}")
            return DestinationSlice(self.name, start, length)

    class Weights:
        expert_ranges = EXPERT_RANGES

        @staticmethod
        def routed_device_shard(device_index):
            loaded.append(device_index)
            return SimpleNamespace(
                expert_range=EXPERT_RANGES[device_index], gate_up=f"gate-up-{device_index}", down=f"down-{device_index}"
            )

    def split(gate_up, split_size, *, dim):
        device_index = int(gate_up.rsplit("-", 1)[1])
        assert (split_size, dim) == (640, -1)
        return f"gate-{device_index}", f"up-{device_index}"

    def prepare_w01(gate, up, layers, experts, hidden, intermediate, _shard_map):
        device_index = int(gate.rsplit("-", 1)[1])
        assert up == f"up-{device_index}"
        assert (layers, experts, hidden, intermediate) == (1, 128, 2560, 640)
        return Prepared("w0_w1", device_index, shapes["w0_w1"][:2] + (128,) + shapes["w0_w1"][3:])

    def prepare_w2(down, layers, experts, intermediate, hidden, _w2_map, _w01_map):
        device_index = int(down.rsplit("-", 1)[1])
        assert (layers, experts, intermediate, hidden) == (1, 128, 640, 2560)
        return Prepared("w2", device_index, shapes["w2"][:2] + (128,) + shapes["w2"][3:])

    with (
        mock.patch.object(bf4_module.torch, "empty", side_effect=Destination) as empty,
        mock.patch.object(bf4_module.torch, "split", side_effect=split),
        mock.patch.object(bf4_module, "prepare_w0_w1_tensor_for_moe_compute", side_effect=prepare_w01),
        mock.patch.object(bf4_module, "prepare_w2_tensor_for_moe_compute", side_effect=prepare_w2),
        mock.patch.object(bf4_module.torch, "cat", side_effect=AssertionError("whole-layer cat is forbidden")),
    ):
        w01, w2 = bf4_module._prepare_routed_layer_host_tensors(Weights(), ring_size=7)

    assert w01.shape == shapes["w0_w1"] and w2.shape == shapes["w2"]
    assert loaded == [0, 1, 2, 3]
    assert copies == [
        item
        for device_index, start in enumerate((0, 128, 256, 384))
        for item in (("w0_w1", start, 128, "w0_w1", device_index), ("w2", start, 128, "w2", device_index))
    ]
    assert empty.call_count == 2
    assert [shape for _, shape, _ in allocations] == [shapes["w0_w1"], shapes["w2"]]


def _convert_layer0(root, *, uploaded_shapes=None, as_tensor=None, deallocate_error=None):
    """One fresh layer-0 conversion on fakes.  By default ``as_tensor`` writes the staged tensorbins (the digests
    sample the sparse fixtures) and uploads tensors of ``uploaded_shapes`` (the shape a mesh tensor presents)."""
    identity, mesh = _identity(), SimpleNamespace(shape=(1, 4))
    cache = Qwen38BF4Cache(root, identity, _TrackingMeshContract(identity.physical_ids, mesh))
    logical_shapes = bf4_module.canonical_packed_shapes(ring_size=identity.ring_size)
    weights = SimpleNamespace(expert_ranges=EXPERT_RANGES)
    uploaded = []

    def stage(host, *, dtype, layout, device, memory_config, mesh_mapper, cache_file_name):
        name = cache_file_name.name
        assert device is mesh and memory_config is getattr(MEMORY_CONFIGS, name)
        _write_sparse_tensorbin(
            bf4_module._tensorbin_path(cache_file_name),
            payload_bytes=bf4_module._packed_payload_bytes(logical_shapes[name]),
            marker=f"packed:{name}".encode(),
        )
        uploaded.append(_FakeBF4Tensor(memory_config, uploaded_shapes[name]))
        return uploaded[-1]

    with (
        mock.patch.object(bf4_module, "qualify_live_bf4_ring", return_value=RING7_ORDER),
        mock.patch.object(bf4_module, "Qwen38MoEWeights", return_value=weights),
        mock.patch.object(bf4_module.ttnn, "ShardTensor2dMesh", return_value=object()),
        mock.patch.object(bf4_module.ttnn.experimental, "get_weight_mem_configs", return_value=MEMORY_CONFIGS),
        mock.patch.object(
            bf4_module, "_prepare_routed_layer_host_tensors", return_value=(object(), object())
        ) as prepare,
        mock.patch.object(bf4_module.ttnn, "as_tensor", side_effect=as_tensor or stage),
        mock.patch.object(bf4_module, "_sha256", side_effect=_fast_sparse_digest),
        mock.patch.object(bf4_module.ttnn, "deallocate", side_effect=deallocate_error) as deallocate,
    ):
        try:
            result = cache._convert_and_upload_locked(
                object(), SimpleNamespace(expert_ranges=EXPERT_RANGES), mesh, layer_index=0
            )
        except RuntimeError as error:
            result = error
    prepare.assert_called_once_with(weights, ring_size=identity.ring_size)
    return SimpleNamespace(cache=cache, result=result, uploaded=uploaded, deallocate=deallocate)


def test_fresh_conversion_aggregates_second_upload_and_first_tensor_cleanup_failures(tmp_path):
    first_tensor = _FakeBF4Tensor(MEMORY_CONFIGS.w0_w1, bf4_module.canonical_packed_shapes(ring_size=7)["w0_w1"])
    run = _convert_layer0(
        tmp_path,
        as_tensor=[first_tensor, RuntimeError("synthetic second upload failure")],
        deallocate_error=RuntimeError("synthetic conversion cleanup failure"),
    )
    error = run.result
    assert isinstance(error, BF4CleanupError) and "owned tensor cleanup was incomplete" in str(error)
    assert "synthetic second upload failure" in str(error.primary_error)
    assert len(error.cleanup_errors) == 1
    assert "synthetic conversion cleanup failure" in str(error.cleanup_errors[0])
    run.deallocate.assert_called_once_with(first_tensor)


def test_fresh_conversion_records_the_slot_shape_and_accepts_coordinate_local_shards(tmp_path):
    identity = _identity()
    logical_shapes = bf4_module.canonical_packed_shapes(ring_size=identity.ring_size)
    mesh_shapes = _mesh_shapes(identity)
    assert mesh_shapes["w0_w1"][2] * 4 == logical_shapes["w0_w1"][2]
    run = _convert_layer0(tmp_path, uploaded_shapes=mesh_shapes)
    assert list(run.result) == run.uploaded
    run.deallocate.assert_not_called()
    record = run.cache.verify_layer("backbone", 0)
    assert record is not None
    # The manifest records the slot's global shape; the uploaded tensors present one coordinate's shard.
    assert record.w0_w1.logical_shape == logical_shapes["w0_w1"] and record.w2.logical_shape == logical_shapes["w2"]
    assert [tuple(tensor.shape) for tensor in run.uploaded] == [mesh_shapes["w0_w1"], mesh_shapes["w2"]]
    published = sorted(str(path.relative_to(run.cache.root)) for path in run.cache.root.rglob("*") if path.is_file())
    assert published == sorted([record.w0_w1.relative_path, record.w2.relative_path, "manifest.json"])


def test_fresh_conversion_refuses_a_global_shaped_upload_and_publishes_nothing(tmp_path):
    run = _convert_layer0(tmp_path, uploaded_shapes=bf4_module.canonical_packed_shapes(ring_size=7))
    assert re.search("BF4 conversion w0_w1 loaded shape .* coordinate-local slot shape", str(run.result))
    assert [call.args[0] for call in run.deallocate.call_args_list] == run.uploaded
    # No orphan tensorbin, no manifest: the refused layer is absent, not unadoptable.
    assert [path for path in run.cache.root.rglob("*") if path.is_file()] == []
    assert run.cache.verify_layer("backbone", 0) is None


def _layer7_streamer():
    """A streamer over a cache whose backbone layer 7 loads two named tensors."""
    tensors = (_NamedTensor("w01", 1), _NamedTensor("w2", 2))

    def load_layer(_mesh, *, layer_index, namespace):
        assert (namespace, layer_index) == ("backbone", 7)
        return tensors

    cache = SimpleNamespace(mesh_contract=SimpleNamespace(validate_mesh=lambda _mesh: None), load_layer=load_layer)
    return Qwen38BF4Streamer(cache, object()), tensors


def test_streamer_attempts_both_releases_and_clears_active_slot(expect_error):
    streamer, (w01, w2) = _layer7_streamer()
    released = []

    def release(tensor):
        released.append(tensor)
        if tensor is w01:
            raise RuntimeError("first release failed")

    with mock.patch.object(bf4_module.ttnn, "deallocate", side_effect=release):
        with expect_error(RuntimeError, "first release failed"), streamer.layer(7):
            pass
    assert released == [w01, w2]
    assert streamer._active is None


def test_streamer_aggregates_body_and_all_cleanup_failures():
    streamer, (w01, w2) = _layer7_streamer()
    released = []

    def release(tensor):
        released.append(tensor)
        raise RuntimeError(f"release failed for {tensor}")

    with (
        mock.patch.object(bf4_module.ttnn, "deallocate", side_effect=release),
        pytest.raises(BF4CleanupError, match="streamer body failed") as raised,  # allow-pytest.raises: .value
    ):
        with streamer.layer(7):
            raise ValueError("synthetic body failure")
    assert "synthetic body failure" in str(raised.value.primary_error)
    assert len(raised.value.cleanup_errors) == 2
    assert released == [w01, w2]
    assert streamer._active is None


# The cache pins the converter's sources instead of the tt-metal revision, adopts the old slots, checks bytes.


@pytest.mark.skipif(
    not all(path.is_relative_to(bf4_module.REPO_ROOT) for path in bf4_module.CONVERTER_SOURCES),
    reason="the converter identity reads the sources of a ttnn built from this checkout, not of an installed wheel",
)
def test_converter_source_identity_pins_this_module_the_layout_packer_and_the_bfp4_packer():
    identity = bf4_module.bf4_converter_source_identity()
    assert [path for path, _ in identity] == [
        "models/demos/blackhole/qwen38_flash_next/ttnn/bf4.py",
        "tt_metal/impl/data_format/bfloat4.cpp",
        "tt_metal/impl/data_format/blockfloat_common.cpp",
        "tt_metal/impl/data_format/blockfloat_common.hpp",
        "ttnn/ttnn/_experimental/moe_compute_utils.py",
    ]
    for relative, digest in identity:
        assert digest == hashlib.sha256((bf4_module.REPO_ROOT / relative).read_bytes()).hexdigest()
    accepted = replace(_identity(), converter_sources=identity)
    assert "tt_metal_revision" not in asdict(accepted)
    assert accepted.key != _identity().key


def test_legacy_slots_keyed_by_the_revision_or_the_worker_coordinates_are_adopted_in_place(tmp_path):
    identity = _identity()
    contract = Qwen38MeshContract(identity.physical_ids)

    def demote(document: dict, legacy_identity: dict) -> str:
        """Rewrite the manifest as the legacy format its identity names; the slot's new directory name."""

        key = hashlib.sha256(json.dumps(legacy_identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        document.update(format_version=legacy_identity["format_version"], identity=legacy_identity, identity_key=key)
        return key

    root, empty = tmp_path / "root", tmp_path / "empty"
    root.mkdir()
    empty.mkdir()
    cache = Qwen38BF4Cache(root, identity, contract)
    record0, paths0 = _write_layer(cache, 0)
    record5, _ = _write_layer(cache, 5)
    document = json.loads(cache.manifest_path.read_text(encoding="utf-8"))
    current = document["identity"]
    common = {key: value for key, value in current.items() if key != "dram_bank_ring_order"}
    # Format 2 stored the first die's coordinates and this module's digest of the time.  This slot's box had a
    # differently harvested die first (worker column 1, not 0): other coordinates, the same derived ring order.
    format2 = dict(common)
    format2["dram_bank_worker_order"] = [[x, 1] for x in range(7)]
    format2["converter_sources"] = [
        [path, "f" * 64 if path.endswith("/bf4.py") else digest] for path, digest in current["converter_sources"]
    ]
    format2["format_version"] = 2
    assert bf4_module.dram_bank_ring_order(tuple(map(tuple, format2["dram_bank_worker_order"]))) == RING7_ORDER
    format2_key = demote(document, format2)
    bf4_module._atomic_json(cache.manifest_path, document)
    os.rename(cache.root, root / format2_key)
    # Decoys, every one newer than the slot so it would win if it were adoptable: another ring size, coordinates
    # that derive to another ring order (banks 0 and 1 swap workers), another BFP4 packer digest, a format the
    # code does not know, an unreadable manifest and a stray file.
    names = ("ring-8", "other-order", "other-packer", "format-9", "no-layout-version", "layout-1")
    decoys = {f"decoy-{name}": copy.deepcopy(format2) for name in names}
    decoys["decoy-ring-8"]["ring_size"] = 8
    decoys["decoy-other-order"]["dram_bank_worker_order"] = [[1, 1], [0, 1], *([x, 1] for x in range(2, 7))]
    decoys["decoy-other-packer"]["converter_sources"] = [
        [path, "e" * 64 if path.endswith("bfloat4.cpp") else digest] for path, digest in format2["converter_sources"]
    ]
    decoys["decoy-format-9"]["format_version"] = 9
    # A slot converted before the packed layout joined the identity, and one that names the stride layout:
    # their bytes are the per-core stride layout, never adoptable into a layout-2 slot.
    del decoys["decoy-no-layout-version"]["layout_version"]
    decoys["decoy-layout-1"]["layout_version"] = 1
    for name, legacy_identity in decoys.items():
        decoy = copy.deepcopy(document)
        demote(decoy, legacy_identity)
        decoy["updated_utc"] = "9999-01-01T00:00:00Z"
        bf4_module._atomic_json(root / name / "manifest.json", decoy)
    (root / "decoy-broken").mkdir()
    (root / "decoy-broken" / "manifest.json").write_text("{", encoding="utf-8")
    (root / "stray.txt").write_text("x", encoding="utf-8")

    adopted = Qwen38BF4Cache(root, identity, contract)

    assert adopted.root == root / identity.key
    assert adopted.manifest_path.exists()
    assert not (root / format2_key).exists()
    for name in (*decoys, "decoy-broken"):
        assert (root / name / "manifest.json").exists(), name
    adopted_document = json.loads(adopted.manifest_path.read_text(encoding="utf-8"))
    assert adopted_document["format_version"] == 3
    assert adopted_document["identity_key"] == identity.key
    assert adopted_document["identity"] == bf4_module._json_normalized(asdict(identity))
    assert sorted(adopted_document["layers"]) == ["backbone:0", "backbone:5"]
    assert adopted_document["created_utc"] == document["created_utc"]
    assert adopted.verify_layer("backbone", 0) == record0
    assert adopted.verify_layer("backbone", 5) == record5
    w01_path = adopted._validate_record(record0, namespace="backbone", layer_index=0)[0]
    assert w01_path == adopted.root / paths0["w0_w1"].relative_to(cache.root)
    # Format 1 was keyed by the tt-metal revision and had no converter sources.
    format1 = {key: value for key, value in common.items() if key != "converter_sources"}
    format1["dram_bank_worker_order"] = [list(coordinate) for coordinate in RING7_WORKERS]
    format1["tt_metal_revision"] = "9fb1403a93" + "0" * 30
    format1["format_version"] = 1
    format1_key = demote(adopted_document, format1)
    bf4_module._atomic_json(adopted.manifest_path, adopted_document)
    os.rename(adopted.root, root / format1_key)
    adopted = Qwen38BF4Cache(root, identity, contract)
    assert not (root / format1_key).exists()
    assert json.loads(adopted.manifest_path.read_text(encoding="utf-8"))["format_version"] == 3
    assert adopted.verify_layer("backbone", 5) == record5
    # An interrupted adoption (format 3, our key, another directory name) is only renamed.
    os.rename(adopted.root, root / "interrupted")
    renamed = Qwen38BF4Cache(root, identity, contract)
    assert renamed.manifest_path.exists()
    assert not (root / "interrupted").exists()
    assert renamed.verify_layer("backbone", 5) == record5
    # Nothing to adopt: a fresh root stays empty until a layer is converted.
    assert not Qwen38BF4Cache(empty, identity, contract).manifest_path.exists()
    assert os.listdir(empty) == []


def test_expert_byte_ranges_walk_the_tiled_shards_one_run_per_ring_bank():
    shapes = bf4_module.canonical_packed_shapes(ring_size=7)
    physical_ids = (0, 1, 2, 3)
    assert bf4_module.ADMISSION_EXPERT == 511
    for name, blocks, rows in (("w0_w1", 18, 224), ("w2", 3, 672)):
        run = blocks * (rows // 32) * 4 * 576
        payload = bf4_module._packed_payload_bytes(shapes[name])
        shard = payload // 4
        first = bf4_module._expert_byte_ranges(shapes[name], 0, physical_ids)
        last = bf4_module._expert_byte_ranges(shapes[name], 511, physical_ids)
        middle = bf4_module._expert_byte_ranges(shapes[name], 200, physical_ids)
        assert first == tuple((ring_bank * 128 * run, run) for ring_bank in range(7)), name
        assert last[0] == (3 * shard + 127 * run, run) and last[-1][0] + run == payload, name
        assert middle == tuple((shard + (ring_bank * 128 + 72) * run, run) for ring_bank in range(7)), name


def test_admission_compares_one_expert_with_a_fresh_packing_and_names_the_differing_bank(expect_error, tmp_path):
    identity = _identity()
    shapes = bf4_module.canonical_packed_shapes(ring_size=7)
    cache = Qwen38BF4Cache(tmp_path, identity, Qwen38MeshContract(identity.physical_ids))
    _, paths = _write_layer(cache, 3)
    fresh = {}
    for name in ("w0_w1", "w2"):
        ranges = bf4_module._expert_byte_ranges(shapes[name], 200, identity.physical_ids)
        fresh[name] = bytes(hashlib.sha256(name.encode()).digest() * (ranges[0][1] * 7 // 32))
        with paths[name].open("r+b") as stream:
            for ring_bank, (offset, size) in enumerate(ranges):
                stream.seek(bf4_module.TENSORBIN_HEADER_PREFIX_BYTES + _TEST_TENSORBIN_HEADER_BYTES + offset)
                stream.write(fresh[name][ring_bank * size : (ring_bank + 1) * size])
    checkpoint, placement = object(), object()
    with mock.patch.object(bf4_module, "_fresh_expert_bf4_bytes", return_value=fresh) as packer:
        admitted = cache.admit_converted_bytes(checkpoint, placement, layer_index=3, expert=200)
    assert packer.call_args.kwargs | {"scratch": None} == {
        "namespace": "backbone",
        "layer_index": 3,
        "expert": 200,
        "ring_size": 7,
        "scratch": None,
    }
    assert packer.call_args.args == (checkpoint, placement)
    assert str(packer.call_args.kwargs["scratch"]).startswith(str(cache.root / ".admit."))
    assert sorted(os.listdir(cache.root)) == ["backbone", "manifest.json"]  # the scratch is gone
    assert sorted(os.listdir(cache.root / "backbone" / "layer-03")) == sorted(path.name for path in paths.values())
    assert {key: admitted[key] for key in ("namespace", "layer_index", "expert", "compared_bytes")} == {
        "namespace": "backbone",
        "layer_index": 3,
        "expert": 200,
        "compared_bytes": sum(map(len, fresh.values())),
    }
    assert admitted["seconds"] >= 0.0

    corrupt = dict(fresh)
    corrupt["w2"] = bytearray(fresh["w2"])
    corrupt["w2"][4 * (len(fresh["w2"]) // 7) + 17] ^= 0x01
    corrupt["w2"] = bytes(corrupt["w2"])
    with mock.patch.object(bf4_module, "_fresh_expert_bf4_bytes", return_value=corrupt):
        with expect_error(RuntimeError, r"backbone layer 3 expert 200 \(w2, ring bank 4\) differs .* at byte 17 of "):
            cache.admit_converted_bytes(checkpoint, placement, layer_index=3, expert=200)
    short = dict(fresh)
    short["w0_w1"] = fresh["w0_w1"][:-1]
    with mock.patch.object(bf4_module, "_fresh_expert_bf4_bytes", return_value=short):
        with expect_error(RuntimeError, "fresh w0_w1 packing of expert 200"):
            cache.admit_converted_bytes(checkpoint, placement, layer_index=3, expert=200)
    with expect_error(RuntimeError, "holds no backbone layer 9 to admit"):
        cache.admit_converted_bytes(checkpoint, placement, layer_index=9)
    with expect_error(ValueError, "expert must be in"):
        cache.admit_converted_bytes(checkpoint, placement, layer_index=3, expert=512)


class _FakeCore:
    __slots__ = ("x", "y")

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y


class _RingMeshDevice(_FakeMeshDevice):
    """A four-die Blackhole mesh whose per-die DRAM bank-to-worker assignment is scripted; die 0 is the reference."""

    def __init__(self, per_die_signatures):
        self._per_die = tuple(per_die_signatures)

    @staticmethod
    def arch():
        return bf4_module.ttnn.Arch.BLACKHOLE

    def assignment(self, coordinate=(0, 0)):
        _, column = tuple(coordinate)
        return [_FakeCore(x, y) for x, y in self._per_die[column]]

    def get_optimal_dram_bank_to_logical_worker_assignment(self, noc, coordinate):
        # Native returns a bank-keyed map whose iteration order is not the bank order.
        return dict(reversed(list(enumerate(self.assignment(coordinate)))))


# Observed on a QuietBox 2 (2x p300c): three dies serve their banks from worker column 6, the fourth from column 5.
QB2_COLUMN_6 = ((0, 9), (0, 0), (0, 7), (0, 3), (6, 9), (6, 1), (6, 6), (6, 4))
QB2_COLUMN_5 = ((0, 9), (0, 0), (0, 7), (0, 3), (5, 9), (5, 1), (5, 6), (5, 4))
QB2_RING_ORDER = (4, 0, 2, 6, 7, 3, 5, 1)


def test_ring_order_is_the_bank_order_the_packing_produces():
    """The guard's sort against ``get_weight_core_shard_maps`` on the same assignment: one sort, pinned here."""

    scrambled = ((3, 2), (0, 0), (5, 1), (1, 7), (6, 2), (2, 9), (4, 4), (0, 3))
    for signature in (QB2_COLUMN_6, QB2_COLUMN_5, RING7_WORKERS, scrambled):
        mesh = _RingMeshDevice([signature] * 4)
        with mock.patch.object(
            bf4_module.ttnn.device,
            "get_optimal_dram_bank_to_logical_worker_assignment",
            side_effect=lambda device, noc: mesh.assignment(),
        ):
            _, _, placement_grid = bf4_module.moe_compute_utils.get_weight_core_shard_maps(mesh, 2560, 640)
        packed_bank_order = tuple(core_range.start.x for core_range in placement_grid.ranges())
        assert len(packed_bank_order) == len(signature)
        assert bf4_module.dram_bank_ring_order(signature) == packed_bank_order
    assert bf4_module.dram_bank_ring_order(QB2_COLUMN_6) == QB2_RING_ORDER
    assert bf4_module.dram_bank_ring_order(RING7_WORKERS) == RING7_ORDER


@pytest.mark.parametrize(
    "dies",
    [
        [QB2_COLUMN_6] * 4,
        # The QuietBox 2 case: one die's workers a column away from its banks, the same bank ids in the same order.
        [QB2_COLUMN_6, QB2_COLUMN_5, QB2_COLUMN_6, QB2_COLUMN_6],
        # The odd die first: the same result, so the cache identity is the same.
        [QB2_COLUMN_5, QB2_COLUMN_6, QB2_COLUMN_6, QB2_COLUMN_6],
    ],
    ids=["uniform", "mixed harvesting", "odd die first"],
)
def test_one_ring_order_is_accepted(dies):
    assert bf4_module.qualify_live_bf4_ring(_RingMeshDevice(dies)) == QB2_RING_ORDER


@pytest.mark.parametrize(
    "dies, match",
    [
        # Two banks swap workers on one die: the same coordinates as a set, another ring order.
        (
            [QB2_COLUMN_6, (QB2_COLUMN_6[1], QB2_COLUMN_6[0], *QB2_COLUMN_6[2:]), QB2_COLUMN_6, QB2_COLUMN_6],
            r"mixed Blackhole DRAM ring orders.*\(0, 9\).*\(4, 0, 2, 6, 7, 3, 5, 1\)",
        ),
        ([QB2_COLUMN_6, QB2_COLUMN_6[:7], QB2_COLUMN_6, QB2_COLUMN_6], "mixed Blackhole DRAM ring sizes.*8.*7"),
        ([QB2_COLUMN_6[:5]] * 4, "unsupported DRAM worker order"),
    ],
    ids=["mixed ring order", "mixed ring size", "unsupported ring size"],
)
def test_mixed_or_unsupported_rings_are_rejected(expect_error, dies, match):
    with expect_error(RuntimeError, match):
        bf4_module.qualify_live_bf4_ring(_RingMeshDevice(dies))
