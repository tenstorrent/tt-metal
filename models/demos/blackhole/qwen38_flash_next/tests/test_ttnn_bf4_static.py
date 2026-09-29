# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
import os
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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
    cache: Qwen38BF4Cache,
    layer_index: int,
    *,
    namespace: str = "backbone",
) -> tuple[BF4LayerRecord, dict[str, Path]]:
    paths = {name: bf4_module._tensorbin_path(cache._base(namespace, layer_index, name)) for name in ("w0_w1", "w2")}
    logical_shapes = bf4_module.canonical_packed_shapes(ring_size=cache.identity.ring_size)
    artifacts = {}
    for name, path in paths.items():
        payload = f"packed:{namespace}:{layer_index}:{name}".encode()
        _write_sparse_tensorbin(
            path,
            payload_bytes=bf4_module._packed_payload_bytes(logical_shapes[name]),
            marker=payload,
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


class TTNNBF4StaticTest(unittest.TestCase):
    def setUp(self):
        # Sparse fixtures preserve the exact multi-GiB file-size/header contract
        # without reading holes.  Production continues to use full SHA-256.
        patcher = mock.patch.object(bf4_module, "_sha256_fd", side_effect=_fast_sparse_digest_fd)
        patcher.start()
        self.addCleanup(patcher.stop)
        backing_api = mock.patch.object(
            contracts_module.ttnn,
            "get_device_tensors",
            side_effect=lambda tensor: tensor.locals,
        )
        backing_api.start()
        self.addCleanup(backing_api.stop)

    def test_replication_mapper_is_explicit_two_axis_topology(self):
        mesh = object()
        mapper = object()
        with (
            mock.patch.object(
                contracts_module.ttnn,
                "PlacementReplicate",
                side_effect=["row-replicate", "tp-replicate"],
            ),
            mock.patch.object(contracts_module.ttnn, "MeshShape", return_value="mesh-1x4") as mesh_shape,
            mock.patch.object(contracts_module.ttnn, "MeshMapperConfig", return_value="explicit-config") as config,
            mock.patch.object(contracts_module.ttnn, "create_mesh_mapper", return_value=mapper) as create,
        ):
            self.assertIs(replicate_tensor_2d_mesh_mapper(mesh), mapper)
        mesh_shape.assert_called_once_with(1, 4)
        config.assert_called_once_with(["row-replicate", "tp-replicate"], "mesh-1x4")
        create.assert_called_once_with(mesh, "explicit-config")

    def test_qwen38_production_sources_do_not_use_legacy_replication_mapper(self):
        model_root = Path(__file__).resolve().parents[1]
        production_sources = (*(model_root / "ttnn").glob("*.py"), *(model_root / "tools").glob("*.py"))
        offenders = [path for path in production_sources if "ReplicateTensorToMesh" in path.read_text()]
        self.assertEqual(offenders, [])

    def test_exact_packer_storage_not_raw_bf4_storage(self):
        # moe_compute's compact owned-column layout: the 8-bank ring stores the 2560/640 expert without padding
        # (3,200 gate/up + 1,600 W2 tiles, 20-tile transactions, a half-width last W2 iteration); the 7-bank ring
        # keeps 14-tile transactions (K padded to 7-tile blocks) and W2's grouped form.
        self.assertEqual(packed_bf4_bytes_per_device(ring_size=7), (260_112_384, 130_056_192))
        self.assertEqual(packed_bf4_bytes_per_device(ring_size=8), (235_929_600, 117_964_800))
        self.assertEqual(packed_bf4_model_bytes_per_device(ring_size=7), 19_118_260_224)
        self.assertEqual(packed_bf4_model_bytes_per_device(ring_size=8), 17_340_825_600)
        self.assertEqual(
            bf4_module.canonical_packed_shapes(ring_size=7),
            {
                "w0_w1": (7, 1, 512, 18, 224, 128),
                "w2": (7, 1, 512, 3, 672, 128),
            },
        )
        self.assertEqual(
            bf4_module.canonical_packed_shapes(ring_size=8),
            {
                "w0_w1": (8, 1, 512, 10, 320, 128),
                "w2": (8, 1, 512, 5, 320, 128),
            },
        )
        for ring_size in (7, 8):
            shapes = bf4_module.canonical_packed_shapes(ring_size=ring_size)
            expected = packed_bf4_bytes_per_device(ring_size=ring_size)
            self.assertEqual(
                tuple(bf4_module._packed_payload_bytes(shapes[name]) // 4 for name in ("w0_w1", "w2")),
                expected,
            )

    def test_tensorbin_header_and_exact_payload_contract_rejects_size_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def validate(
                name: str,
                *,
                header_size: int,
                payload_bytes: int,
                expected_payload: int,
                prefix_bytes: int = 8,
                materialized_header_bytes: int | None = None,
            ):
                path = root / f"{name}.tensorbin"
                with path.open("wb") as stream:
                    stream.write(header_size.to_bytes(8, byteorder="little", signed=False)[:prefix_bytes])
                    remaining_header = (
                        min(header_size, 128) if materialized_header_bytes is None else materialized_header_bytes
                    )
                    stream.write(bytes(remaining_header))
                    stream.write(bytes(payload_bytes))
                descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
                try:
                    return bf4_module._validate_tensorbin_payload_fd(
                        descriptor,
                        signature=bf4_module._artifact_fd_signature(descriptor),
                        expected_payload_bytes=expected_payload,
                    )
                finally:
                    os.close(descriptor)

            self.assertEqual(validate("valid", header_size=16, payload_bytes=32, expected_payload=32), 16)
            with self.assertRaisesRegex(RuntimeError, "truncated before"):
                validate("tiny", header_size=0, payload_bytes=0, expected_payload=32, prefix_bytes=4)
            with self.assertRaisesRegex(RuntimeError, "header size/alignment"):
                validate(
                    "truncated",
                    header_size=64,
                    payload_bytes=0,
                    expected_payload=32,
                    materialized_header_bytes=32,
                )
            with self.assertRaisesRegex(RuntimeError, "header size/alignment"):
                validate("misaligned", header_size=10, payload_bytes=32, expected_payload=32)
            with self.assertRaisesRegex(RuntimeError, "header size/alignment"):
                validate(
                    "oversized-header",
                    header_size=bf4_module.MAX_TENSORBIN_HEADER_BYTES + 8,
                    payload_bytes=0,
                    expected_payload=32,
                )
            with self.assertRaisesRegex(RuntimeError, "payload byte count"):
                validate("short-payload", header_size=16, payload_bytes=31, expected_payload=32)
            with self.assertRaisesRegex(RuntimeError, "payload byte count"):
                validate("oversized-payload", header_size=16, payload_bytes=33, expected_payload=32)

    def test_self_hashed_tiny_tensorbin_cannot_claim_a_canonical_global_shape(self):
        identity = _identity()
        contract = Qwen38MeshContract(identity.physical_ids)
        with tempfile.TemporaryDirectory() as directory:
            cache = Qwen38BF4Cache(directory, identity, contract)
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
            with self.assertRaisesRegex(RuntimeError, "payload byte count"):
                cache.verify_layer("backbone", 0)

    def test_cache_identity_and_shape_normalization_reject_bool_float_and_string_aliases(self):
        identity = _identity()
        mutations = (
            {"ring_size": 7.0},
            {"format_version": True},
            {"mesh_shape": ("1", 4)},
            {"dram_bank_ring_order": (6.0, *RING7_ORDER[1:])},
        )
        for changes in mutations:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, "exact integer|exact integer tuples"):
                    replace(identity, **changes)

        canonical = bf4_module.canonical_packed_shapes(ring_size=7)["w0_w1"]
        aliases = (
            (canonical[0], True, *canonical[2:]),
            (*canonical[:2], 512.0, *canonical[3:]),
            (*canonical[:-1], "128"),
        )
        for shape in aliases:
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(RuntimeError, "shape contains"):
                    bf4_module._native_integer_shape(shape, label="BF4 conversion")

    def test_cache_identity_pins_the_packed_layout_version(self):
        self.assertEqual(bf4_module.LAYOUT_VERSION, 2)
        self.assertEqual(asdict(_identity())["layout_version"], 2)
        for stale in (1, 3, True, 2.0):
            with self.subTest(layout_version=stale):
                with self.assertRaisesRegex(ValueError, "encoding identity|exact integers"):
                    replace(_identity(), layout_version=stale)

    def test_canonical_shapes_are_the_layout_packers_output(self):
        # The cache derives its slot shapes from the geometry helpers of ttnn.experimental.moe_compute_utils; the
        # packers themselves (one zero expert, L = E = 1) must lay the bytes out in exactly those shapes.
        import torch

        for ring_size in (7, 8):
            with self.subTest(ring_size=ring_size):
                w01_map, w2_map = bf4_module.ring_shard_maps(2560, 640, ring_size)
                gate = torch.zeros(1, 1, 2560, 640, dtype=torch.bfloat16)
                down = torch.zeros(1, 1, 640, 2560, dtype=torch.bfloat16)
                packed = {
                    "w0_w1": bf4_module.prepare_w0_w1_tensor_for_moe_compute(gate, gate, 1, 1, 2560, 640, w01_map),
                    "w2": bf4_module.prepare_w2_tensor_for_moe_compute(down, 1, 1, 640, 2560, w2_map, w01_map),
                }
                shapes = bf4_module.canonical_packed_shapes(ring_size=ring_size)
                for name, tensor in packed.items():
                    self.assertEqual(tuple(tensor.shape), (*shapes[name][:2], 1, *shapes[name][3:]), name)
                w01_bytes, w2_bytes = packed_bf4_bytes_per_device(ring_size=ring_size, experts_per_device=1)
                self.assertEqual(w01_bytes, packed["w0_w1"].numel() // 1024 * 576)
                self.assertEqual(w2_bytes, packed["w2"].numel() // 1024 * 576)

    def test_cache_identity_requires_every_live_bank_once_in_ring_order(self):
        for ring_order in ((0,) * 7, RING7_ORDER[:6], (*RING7_ORDER[:6], 7)):
            with self.subTest(ring_order=ring_order):
                with self.assertRaisesRegex(ValueError, "every live DRAM bank once"):
                    replace(_identity(), dram_bank_ring_order=ring_order)

    def test_cache_identity_distinguishes_complete_file_and_tensor_manifests(self):
        identity = _identity()
        with self.assertRaisesRegex(ValueError, "checkpoint_file_manifest_sha256"):
            replace(identity, checkpoint_file_manifest_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "checkpoint_hash_manifest_sha256"):
            replace(identity, checkpoint_hash_manifest_sha256="0" * 64)

    def test_manifest_round_trip_preserves_tuple_contract_and_hashes(self):
        identity = _identity()
        contract = Qwen38MeshContract(identity.physical_ids)
        with tempfile.TemporaryDirectory() as directory:
            cache = Qwen38BF4Cache(directory, identity, contract)
            record, paths = _write_layer(cache, 0)
            real_sha256_fd = bf4_module._sha256_fd
            with mock.patch.object(bf4_module, "_sha256_fd", wraps=real_sha256_fd) as sha256:
                self.assertEqual(cache.verify_layer("backbone", 0), record)
                self.assertEqual(sha256.call_count, 2)
                self.assertEqual(cache.verify_layer("backbone", 0), record)
                self.assertEqual(sha256.call_count, 2, "unchanged artifacts must not be re-hashed per decode")

                fresh_cache = Qwen38BF4Cache(directory, identity, contract)
                self.assertEqual(fresh_cache.verify_layer("backbone", 0), record)
                self.assertEqual(sha256.call_count, 4, "a fresh cache object must establish its own verification")

                original_stat = paths["w0_w1"].stat()
                payload_offset = bf4_module.TENSORBIN_HEADER_PREFIX_BYTES + _TEST_TENSORBIN_HEADER_BYTES
                with paths["w0_w1"].open("r+b") as stream:
                    stream.seek(payload_offset)
                    original_byte = stream.read(1)
                    stream.seek(payload_offset)
                    stream.write(bytes((original_byte[0] ^ 1,)))
                # Restoring mtime does not defeat the inode/ctime session guard.
                os.utime(paths["w0_w1"], ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
                with self.assertRaisesRegex(RuntimeError, "hash validation"):
                    cache.verify_layer("backbone", 0)
                self.assertEqual(sha256.call_count, 5)

                untrusted_fresh_cache = Qwen38BF4Cache(directory, identity, contract)
                with self.assertRaisesRegex(RuntimeError, "hash validation"):
                    untrusted_fresh_cache.verify_layer("backbone", 0)
                self.assertEqual(sha256.call_count, 6, "fresh cache state must not inherit another object's trust")

    def test_complete_backbone_hashes_once_then_streams_by_verified_identity(self):
        identity = _identity()
        mesh = object()
        contract = _TrackingMeshContract(identity.physical_ids, mesh)
        with tempfile.TemporaryDirectory() as directory:
            cache = Qwen38BF4Cache(directory, identity, contract)
            for layer_index in range(BACKBONE_LAYERS):
                _write_layer(cache, layer_index)

            builder = object.__new__(Qwen38TTNNBuilder)
            builder.bf4_cache = cache
            builder.placement = SimpleNamespace(expert_ranges=EXPERT_RANGES)
            builder.identity = SimpleNamespace(ring_size=identity.ring_size)
            streamer = Qwen38BF4Streamer(cache, mesh)
            builder.expert_streamer = streamer
            self.assertIs(builder.bf4_cache, builder.expert_streamer.cache)

            memory_configs = SimpleNamespace(w0_w1=object(), w2=object())
            mesh_shapes = _mesh_shapes(identity)
            retained_paths = []

            def load_tensor(path, *, device):
                self.assertIs(device, mesh)
                self.assertRegex(str(path), r"\A/proc/self/fd/[1-9][0-9]*\Z")
                retained_paths.append(Path(path))
                name = Path(os.readlink(path)).name
                artifact_name = "w0_w1" if name.startswith("w0_w1_") else "w2"
                memory_config = getattr(memory_configs, artifact_name)
                return _FakeBF4Tensor(memory_config, mesh_shapes[artifact_name])

            real_sha256_fd = bf4_module._sha256_fd
            with (
                mock.patch.object(cache, "_read_manifest", wraps=cache._read_manifest) as read_manifest,
                mock.patch.object(bf4_module, "_sha256_fd", wraps=real_sha256_fd) as sha256,
                mock.patch.object(
                    bf4_module,
                    "qualify_live_bf4_ring",
                    return_value=RING7_ORDER,
                ) as qualify_ring,
                mock.patch.object(bf4_module.ttnn, "load_tensor", side_effect=load_tensor) as load,
                mock.patch.object(
                    bf4_module.ttnn.experimental,
                    "get_weight_mem_configs",
                    return_value=memory_configs,
                ),
                mock.patch.object(bf4_module.ttnn, "deallocate") as deallocate,
            ):
                records = builder.require_complete_backbone_bf4()
                self.assertEqual(len(records), BACKBONE_LAYERS)
                self.assertEqual(tuple(record.layer_index for record in records), tuple(range(BACKBONE_LAYERS)))
                self.assertEqual(sha256.call_count, 2 * BACKBONE_LAYERS)

                for traversal in range(2):
                    for layer_index in range(BACKBONE_LAYERS):
                        with streamer.layer(layer_index) as tensors:
                            self.assertEqual(streamer._active, ("backbone", layer_index))
                            self.assertEqual(len(tensors), 2)
                            if traversal == layer_index == 0:
                                with self.assertRaisesRegex(RuntimeError, "already owns active layer"):
                                    with streamer.layer(1):
                                        pass
                    self.assertIsNone(streamer._active)

                self.assertEqual(
                    sha256.call_count,
                    2 * BACKBONE_LAYERS,
                    "decode traversal must bind fresh FDs to the session hash without rehashing the model",
                )
                self.assertEqual(read_manifest.call_count, 3 * BACKBONE_LAYERS)
                self.assertEqual(qualify_ring.call_count, 2 * BACKBONE_LAYERS)
                self.assertEqual(load.call_count, 4 * BACKBONE_LAYERS)
                self.assertEqual(deallocate.call_count, 4 * BACKBONE_LAYERS)
                self.assertEqual(contract.mesh_validations, 1 + 2 * BACKBONE_LAYERS)
                self.assertEqual(contract.tensor_validations, 4 * BACKBONE_LAYERS)
                self.assertEqual(len(retained_paths), 4 * BACKBONE_LAYERS)
                self.assertTrue(all(not path.exists() for path in retained_paths))

    def test_load_tensor_wrapper_owns_canonical_proc_fd_through_native_read(self):
        core_module = bf4_module.ttnn.operations.core
        sentinel = object()
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "weight.tensorbin"
            artifact.write_bytes(b"valid-flatbuffer-placeholder")
            descriptor = os.open(artifact, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            caller_path = Path(f"/proc/self/fd/{descriptor}")
            native_paths = []

            def native_load(path, device):
                self.assertIsNone(device)
                self.assertRegex(path, r"\A/proc/self/fd/[1-9][0-9]*\Z")
                self.assertNotEqual(path, str(caller_path))
                self.assertEqual(os.readlink(path), str(artifact))
                os.fstat(int(Path(path).name))
                native_paths.append(Path(path))
                return sentinel

            try:
                with (
                    mock.patch.object(
                        core_module.ttnn._ttnn.tensor,
                        "load_tensor_flatbuffer",
                        side_effect=native_load,
                    ) as native,
                    mock.patch.object(core_module.ttnn, "deallocate") as deallocate,
                ):
                    self.assertIs(core_module.load_tensor(caller_path), sentinel)
                    self.assertTrue(caller_path.exists(), "the wrapper must not close its caller's descriptor")
                    self.assertEqual(len(native_paths), 1)
                    self.assertFalse(native_paths[0].exists(), "the wrapper-owned duplicate must close on success")

                    self.assertIs(core_module.load_tensor(artifact), sentinel)
                    self.assertRegex(native.call_args_list[-1].args[0], r"\A/proc/self/fd/[1-9][0-9]*\Z")
                    self.assertEqual(len(native_paths), 2)
                    self.assertFalse(native_paths[1].exists(), "the ordinary-path descriptor must close on success")
                    deallocate.assert_not_called()
            finally:
                os.close(descriptor)
            self.assertFalse(caller_path.exists())

    def test_load_tensor_wrapper_rejects_descriptor_aliases_and_invalid_targets(self):
        core_module = bf4_module.ttnn.operations.core
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "weight.tensorbin"
            artifact.write_bytes(b"payload")
            descriptor = os.open(artifact, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            proc_path = Path(f"/proc/self/fd/{descriptor}")
            alias = root / "alias.tensorbin"
            alias.symlink_to(proc_path)
            inner_alias = root / "inner.tensorbin"
            inner_alias.symlink_to(proc_path)
            outer_alias = root / "outer.tensorbin"
            outer_alias.symlink_to(inner_alias)
            pid_alias = root / "pid-alias.tensorbin"
            pid_alias.symlink_to(f"/proc/{os.getpid()}/fd/{descriptor}")
            plain = root / "weight.bin"
            plain.write_bytes(b"payload")
            plain_descriptor = os.open(plain, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            directory_descriptor = os.open(root, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            closed_descriptor = os.dup(descriptor)
            os.close(closed_descriptor)
            try:
                with mock.patch.object(core_module.ttnn._ttnn.tensor, "load_tensor_flatbuffer") as native:
                    invalid_spellings = (
                        f"/proc/self/fd/{descriptor}.tensorbin",
                        f"/proc/self/fd/0{descriptor}",
                        f"/proc/self/fd/{descriptor}/../{descriptor}",
                        f"/proc/self/fd/{descriptor}/",
                        f"/proc/self/fd/./{descriptor}",
                        f"/proc/self/fd//{descriptor}",
                        "/proc/self/fd/0",
                    )
                    for invalid in invalid_spellings:
                        with self.subTest(invalid=invalid):
                            with self.assertRaisesRegex(RuntimeError, "canonical /proc/self/fd"):
                                core_module.load_tensor(invalid)
                    with self.assertRaisesRegex(RuntimeError, "must not alias"):
                        core_module.load_tensor(alias)
                    with self.assertRaisesRegex(RuntimeError, "must not alias"):
                        core_module.load_tensor(outer_alias)
                    with self.assertRaisesRegex(RuntimeError, "must not alias"):
                        core_module.load_tensor(pid_alias)
                    with self.assertRaisesRegex(RuntimeError, r"absolute \.tensorbin"):
                        core_module.load_tensor(f"/proc/self/fd/{plain_descriptor}")
                    with self.assertRaisesRegex(RuntimeError, "regular file"):
                        core_module.load_tensor(f"/proc/self/fd/{directory_descriptor}")
                    with self.assertRaisesRegex(RuntimeError, "does not name an open descriptor"):
                        core_module.load_tensor(f"/proc/self/fd/{closed_descriptor}")
                    native.assert_not_called()
            finally:
                os.close(directory_descriptor)
                os.close(plain_descriptor)
                os.close(descriptor)

    def test_load_tensor_wrapper_retains_ordinary_symlink_compatibility_and_bounds_loops(self):
        core_module = bf4_module.ttnn.operations.core
        sentinel = object()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "weight.tensorbin"
            artifact.write_bytes(b"payload")
            ordinary_alias = root / "ordinary.tensorbin"
            ordinary_alias.symlink_to(artifact)
            loop_a = root / "loop-a.tensorbin"
            loop_b = root / "loop-b.tensorbin"
            loop_a.symlink_to(loop_b)
            loop_b.symlink_to(loop_a)

            def native_load(path, device):
                self.assertIsNone(device)
                self.assertRegex(path, r"\A/proc/self/fd/[1-9][0-9]*\Z")
                self.assertEqual(os.readlink(path), str(artifact))
                return sentinel

            with mock.patch.object(
                core_module.ttnn._ttnn.tensor,
                "load_tensor_flatbuffer",
                side_effect=native_load,
            ) as native:
                self.assertIs(core_module.load_tensor(ordinary_alias), sentinel)
                with self.assertRaisesRegex(RuntimeError, "hop bound"):
                    core_module.load_tensor(loop_a)
                native.assert_called_once()

    def test_load_tensor_wrapper_binds_ordinary_path_to_owned_fd_across_replacement(self):
        core_module = bf4_module.ttnn.operations.core
        sentinel = object()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "weight.tensorbin"
            artifact.write_bytes(b"original")
            replacement = root / "replacement.tensorbin"
            replacement.write_bytes(b"replacement")
            stable_paths = []

            def replace_during_native(path, device):
                self.assertIsNone(device)
                stable_paths.append(Path(path))
                self.assertEqual(os.readlink(path), str(artifact))
                os.replace(replacement, artifact)
                return sentinel

            with (
                mock.patch.object(
                    core_module.ttnn._ttnn.tensor,
                    "load_tensor_flatbuffer",
                    side_effect=replace_during_native,
                ),
                mock.patch.object(core_module.ttnn, "deallocate") as deallocate,
            ):
                with self.assertRaisesRegex(RuntimeError, "changed during tensor load"):
                    core_module.load_tensor(artifact)
                deallocate.assert_called_once_with(sentinel)
            self.assertEqual(len(stable_paths), 1)
            self.assertFalse(stable_paths[0].exists())

    def test_load_tensor_wrapper_detects_native_read_mutation_and_cleans_up(self):
        core_module = bf4_module.ttnn.operations.core
        sentinel = object()
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "weight.tensorbin"
            artifact.write_bytes(b"payload")
            descriptor = os.open(artifact, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            proc_path = Path(f"/proc/self/fd/{descriptor}")
            native_paths = []

            def mutate_during_native(path, _device):
                native_paths.append(Path(path))
                artifact.write_bytes(b"payload-mutated")
                return sentinel

            try:
                with (
                    mock.patch.object(
                        core_module.ttnn._ttnn.tensor,
                        "load_tensor_flatbuffer",
                        side_effect=mutate_during_native,
                    ),
                    mock.patch.object(core_module.ttnn, "deallocate") as deallocate,
                ):
                    with self.assertRaisesRegex(RuntimeError, "changed during tensor load"):
                        core_module.load_tensor(proc_path)
                    deallocate.assert_called_once_with(sentinel)
                self.assertEqual(len(native_paths), 1)
                self.assertFalse(native_paths[0].exists(), "the duplicate must close on validation failure")
                self.assertTrue(proc_path.exists(), "the caller retains ownership after wrapper failure")
            finally:
                os.close(descriptor)

    def test_load_tensor_wrapper_reports_returned_tensor_cleanup_failure(self):
        core_module = bf4_module.ttnn.operations.core
        sentinel = object()
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "weight.tensorbin"
            artifact.write_bytes(b"payload")
            descriptor = os.open(artifact, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            proc_path = Path(f"/proc/self/fd/{descriptor}")
            native_paths = []

            def mutate_during_native(path, _device):
                native_paths.append(Path(path))
                artifact.write_bytes(b"payload-mutated")
                return sentinel

            try:
                with (
                    mock.patch.object(
                        core_module.ttnn._ttnn.tensor,
                        "load_tensor_flatbuffer",
                        side_effect=mutate_during_native,
                    ),
                    mock.patch.object(
                        core_module.ttnn,
                        "deallocate",
                        side_effect=RuntimeError("synthetic core cleanup failure"),
                    ),
                ):
                    with self.assertRaisesRegex(RuntimeError, "returned-tensor cleanup also failed") as raised:
                        core_module.load_tensor(proc_path)
                self.assertRegex(str(raised.exception.primary_error), "changed during tensor load")
                self.assertRegex(str(raised.exception.cleanup_error), "synthetic core cleanup failure")
                self.assertFalse(native_paths[0].exists(), "the wrapper duplicate must close despite cleanup failure")
                self.assertTrue(proc_path.exists(), "the caller descriptor must remain caller-owned")
            finally:
                os.close(descriptor)

    def test_bf4_load_retains_verified_fds_through_topology_validation(self):
        identity = _identity()
        mesh = object()
        contract = _TrackingMeshContract(identity.physical_ids, mesh)
        memory_configs = SimpleNamespace(w0_w1=object(), w2=object())
        mesh_shapes = _mesh_shapes(identity)
        retained_paths = []
        with tempfile.TemporaryDirectory() as directory:
            cache = Qwen38BF4Cache(directory, identity, contract)
            _write_layer(cache, 0)

            def load_tensor(path, *, device):
                self.assertIs(device, mesh)
                self.assertRegex(str(path), r"\A/proc/self/fd/[1-9][0-9]*\Z")
                retained_paths.append(Path(path))
                target = Path(os.readlink(path)).name
                artifact_name = "w0_w1" if target.startswith("w0_w1_") else "w2"
                return _FakeBF4Tensor(getattr(memory_configs, artifact_name), mesh_shapes[artifact_name])

            original_validate = contract.validate_tensor

            def validate_while_retained(tensor, *, placement, shard_dim):
                self.assertEqual(len(retained_paths), 2)
                self.assertTrue(all(path.exists() for path in retained_paths))
                return original_validate(tensor, placement=placement, shard_dim=shard_dim)

            with (
                mock.patch.object(bf4_module, "qualify_live_bf4_ring", return_value=RING7_ORDER),
                mock.patch.object(bf4_module.ttnn, "load_tensor", side_effect=load_tensor),
                mock.patch.object(
                    bf4_module.ttnn.experimental,
                    "get_weight_mem_configs",
                    return_value=memory_configs,
                ),
                mock.patch.object(contract, "validate_tensor", side_effect=validate_while_retained),
                mock.patch.object(bf4_module.ttnn, "deallocate") as deallocate,
            ):
                tensors = cache.load_layer(mesh, layer_index=0)
            self.assertEqual(len(tensors), 2)
            self.assertTrue(all(not path.exists() for path in retained_paths))
            deallocate.assert_not_called()

    def test_bf4_load_rejects_wrong_loaded_shape_and_releases_both_tensors(self):
        identity = _identity()
        mesh = object()
        contract = _TrackingMeshContract(identity.physical_ids, mesh)
        memory_configs = SimpleNamespace(w0_w1=object(), w2=object())
        mesh_shapes = _mesh_shapes(identity)
        loaded = []
        with tempfile.TemporaryDirectory() as directory:
            cache = Qwen38BF4Cache(directory, identity, contract)
            _write_layer(cache, 0)

            def load_tensor(path, *, device):
                self.assertIs(device, mesh)
                target = Path(os.readlink(path)).name
                artifact_name = "w0_w1" if target.startswith("w0_w1_") else "w2"
                shape = mesh_shapes[artifact_name]
                if artifact_name == "w2":
                    shape = (*shape[:-1], shape[-1] + 32)
                tensor = _FakeBF4Tensor(getattr(memory_configs, artifact_name), shape)
                loaded.append(tensor)
                return tensor

            with (
                mock.patch.object(bf4_module, "qualify_live_bf4_ring", return_value=RING7_ORDER),
                mock.patch.object(bf4_module.ttnn, "load_tensor", side_effect=load_tensor),
                mock.patch.object(
                    bf4_module.ttnn.experimental,
                    "get_weight_mem_configs",
                    return_value=memory_configs,
                ),
                mock.patch.object(bf4_module.ttnn, "deallocate") as deallocate,
            ):
                with self.assertRaisesRegex(RuntimeError, "loaded shape"):
                    cache.load_layer(mesh, layer_index=0)
            self.assertEqual([call.args[0] for call in deallocate.call_args_list], loaded)

    def test_bf4_load_rejects_bool_float_and_string_shape_aliases_and_releases_both_tensors(self):
        identity = _identity()
        mesh = object()
        memory_configs = SimpleNamespace(w0_w1=object(), w2=object())
        mesh_shapes = _mesh_shapes(identity)
        aliases = (
            (7, True, 512, 18, 224, 128),
            (7, 1, 512.0, 18, 224, 128),
            (7, 1, 512, 18, 224, "128"),
        )
        for invalid_shape in aliases:
            with self.subTest(invalid_shape=invalid_shape), tempfile.TemporaryDirectory() as directory:
                contract = _TrackingMeshContract(identity.physical_ids, mesh)
                cache = Qwen38BF4Cache(directory, identity, contract)
                _write_layer(cache, 0)
                loaded = []

                def load_tensor(path, *, device):
                    self.assertIs(device, mesh)
                    target = Path(os.readlink(path)).name
                    artifact_name = "w0_w1" if target.startswith("w0_w1_") else "w2"
                    shape = invalid_shape if artifact_name == "w0_w1" else mesh_shapes[artifact_name]
                    tensor = _FakeBF4Tensor(getattr(memory_configs, artifact_name), shape)
                    loaded.append(tensor)
                    return tensor

                with (
                    mock.patch.object(bf4_module, "qualify_live_bf4_ring", return_value=RING7_ORDER),
                    mock.patch.object(bf4_module.ttnn, "load_tensor", side_effect=load_tensor),
                    mock.patch.object(
                        bf4_module.ttnn.experimental,
                        "get_weight_mem_configs",
                        return_value=memory_configs,
                    ),
                    mock.patch.object(bf4_module.ttnn, "deallocate") as deallocate,
                ):
                    with self.assertRaisesRegex(RuntimeError, "shape contains"):
                        cache.load_layer(mesh, layer_index=0)
                self.assertEqual([call.args[0] for call in deallocate.call_args_list], loaded)

    def test_bf4_load_rejects_path_replacement_after_native_reads_and_closes_fds(self):
        identity = _identity()
        mesh = object()
        contract = _TrackingMeshContract(identity.physical_ids, mesh)
        memory_configs = SimpleNamespace(w0_w1=object(), w2=object())
        mesh_shapes = _mesh_shapes(identity)
        retained_paths = []
        loaded_tensors = []
        with tempfile.TemporaryDirectory() as directory:
            cache = Qwen38BF4Cache(directory, identity, contract)
            _, paths = _write_layer(cache, 0)

            def replace_after_both_reads(path, *, device):
                self.assertIs(device, mesh)
                retained_paths.append(Path(path))
                target = os.readlink(path)
                artifact_name = "w0_w1" if Path(target).name.startswith("w0_w1_") else "w2"
                tensor = _FakeBF4Tensor(getattr(memory_configs, artifact_name), mesh_shapes[artifact_name])
                loaded_tensors.append(tensor)
                if len(loaded_tensors) == 2:
                    replacement = paths["w0_w1"].with_name("replacement.tensorbin")
                    _copy_sparse_tensorbin(paths["w0_w1"], replacement)
                    os.replace(replacement, paths["w0_w1"])
                return tensor

            with (
                mock.patch.object(bf4_module, "qualify_live_bf4_ring", return_value=RING7_ORDER),
                mock.patch.object(bf4_module.ttnn, "load_tensor", side_effect=replace_after_both_reads),
                mock.patch.object(
                    bf4_module.ttnn.experimental,
                    "get_weight_mem_configs",
                    return_value=memory_configs,
                ),
                mock.patch.object(bf4_module.ttnn, "deallocate") as deallocate,
            ):
                with self.assertRaisesRegex(RuntimeError, "identity changed while its descriptor was retained"):
                    cache.load_layer(mesh, layer_index=0)
            self.assertEqual(len(loaded_tensors), 2)
            self.assertEqual([call.args[0] for call in deallocate.call_args_list], loaded_tensors)
            self.assertTrue(all(not path.exists() for path in retained_paths))

    def test_bf4_load_rechecks_in_place_mutation_after_topology_validation(self):
        identity = _identity()
        mesh = object()
        contract = _TrackingMeshContract(identity.physical_ids, mesh)
        memory_configs = SimpleNamespace(w0_w1=object(), w2=object())
        mesh_shapes = _mesh_shapes(identity)
        retained_paths = []
        loaded_tensors = []
        with tempfile.TemporaryDirectory() as directory:
            cache = Qwen38BF4Cache(directory, identity, contract)
            _, paths = _write_layer(cache, 0)

            def load_tensor(path, *, device):
                self.assertIs(device, mesh)
                retained_paths.append(Path(path))
                target = Path(os.readlink(path)).name
                artifact_name = "w0_w1" if target.startswith("w0_w1_") else "w2"
                tensor = _FakeBF4Tensor(getattr(memory_configs, artifact_name), mesh_shapes[artifact_name])
                loaded_tensors.append(tensor)
                return tensor

            original_validate = contract.validate_tensor

            def mutate_during_topology(tensor, *, placement, shard_dim):
                self.assertTrue(all(path.exists() for path in retained_paths))
                if contract.tensor_validations == 0:
                    with paths["w2"].open("ab") as stream:
                        stream.write(b"mutation")
                return original_validate(tensor, placement=placement, shard_dim=shard_dim)

            with (
                mock.patch.object(bf4_module, "qualify_live_bf4_ring", return_value=RING7_ORDER),
                mock.patch.object(bf4_module.ttnn, "load_tensor", side_effect=load_tensor),
                mock.patch.object(
                    bf4_module.ttnn.experimental,
                    "get_weight_mem_configs",
                    return_value=memory_configs,
                ),
                mock.patch.object(contract, "validate_tensor", side_effect=mutate_during_topology),
                mock.patch.object(bf4_module.ttnn, "deallocate") as deallocate,
            ):
                with self.assertRaisesRegex(RuntimeError, "changed while its descriptor was retained"):
                    cache.load_layer(mesh, layer_index=0)
            self.assertEqual(contract.tensor_validations, 2)
            self.assertEqual([call.args[0] for call in deallocate.call_args_list], loaded_tensors)
            self.assertTrue(all(not path.exists() for path in retained_paths))

    def test_bf4_load_failure_closes_both_fds_and_releases_first_tensor(self):
        identity = _identity()
        mesh = object()
        contract = _TrackingMeshContract(identity.physical_ids, mesh)
        memory_configs = SimpleNamespace(w0_w1=object(), w2=object())
        mesh_shapes = _mesh_shapes(identity)
        retained_paths = []
        first_tensor = _FakeBF4Tensor(memory_configs.w0_w1, mesh_shapes["w0_w1"])
        with tempfile.TemporaryDirectory() as directory:
            cache = Qwen38BF4Cache(directory, identity, contract)
            _write_layer(cache, 0)

            def fail_second_load(path, *, device):
                self.assertIs(device, mesh)
                retained_paths.append(Path(path))
                if len(retained_paths) == 1:
                    return first_tensor
                raise RuntimeError("synthetic second load failure")

            with (
                mock.patch.object(bf4_module, "qualify_live_bf4_ring", return_value=RING7_ORDER),
                mock.patch.object(bf4_module.ttnn, "load_tensor", side_effect=fail_second_load),
                mock.patch.object(
                    bf4_module.ttnn.experimental,
                    "get_weight_mem_configs",
                    return_value=memory_configs,
                ),
                mock.patch.object(bf4_module.ttnn, "deallocate") as deallocate,
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic second load failure"):
                    cache.load_layer(mesh, layer_index=0)
            deallocate.assert_called_once_with(first_tensor)
            self.assertEqual(len(retained_paths), 2)
            self.assertTrue(all(not path.exists() for path in retained_paths))

    def test_bf4_cache_load_aggregates_load_and_cleanup_failures(self):
        identity = _identity()
        mesh = object()
        contract = _TrackingMeshContract(identity.physical_ids, mesh)
        memory_configs = SimpleNamespace(w0_w1=object(), w2=object())
        mesh_shapes = _mesh_shapes(identity)
        first_tensor = _FakeBF4Tensor(memory_configs.w0_w1, mesh_shapes["w0_w1"])
        retained_paths = []
        with tempfile.TemporaryDirectory() as directory:
            cache = Qwen38BF4Cache(directory, identity, contract)
            record, paths = _write_layer(cache, 0)
            self.assertEqual(cache.verify_layer("backbone", 0), record)

            def fail_second_load(path, *, device):
                self.assertIs(device, mesh)
                retained_paths.append(Path(path))
                if len(retained_paths) == 1:
                    return first_tensor
                raise RuntimeError("synthetic cache load failure")

            with (
                mock.patch.object(bf4_module.ttnn, "load_tensor", side_effect=fail_second_load),
                mock.patch.object(
                    bf4_module.ttnn,
                    "deallocate",
                    side_effect=RuntimeError("synthetic cache cleanup failure"),
                ) as deallocate,
            ):
                with self.assertRaisesRegex(bf4_module.BF4CleanupError, "cache load failed") as raised:
                    cache._load_verified_tensors(
                        mesh,
                        record=record,
                        w01_path=paths["w0_w1"],
                        w2_path=paths["w2"],
                        memory_configs=memory_configs,
                    )
            self.assertRegex(str(raised.exception.primary_error), "synthetic cache load failure")
            self.assertEqual(len(raised.exception.cleanup_errors), 1)
            self.assertEqual(raised.exception.unreleased_tensors, (first_tensor,))
            self.assertEqual(len(raised.exception.tensor_cleanup_outcomes), 2)
            self.assertTrue(raised.exception.tensor_cleanup_outcomes[0].release_attempted)
            self.assertFalse(raised.exception.tensor_cleanup_outcomes[0].released)
            self.assertIs(raised.exception.tensor_cleanup_outcomes[0].tensor, first_tensor)
            self.assertIsNone(raised.exception.tensor_cleanup_outcomes[1].tensor)
            deallocate.assert_called_once_with(first_tensor)
            self.assertTrue(all(not path.exists() for path in retained_paths))

    def test_bf4_verification_rejects_symlink_artifact_without_loading(self):
        identity = _identity()
        mesh = object()
        contract = _TrackingMeshContract(identity.physical_ids, mesh)
        with tempfile.TemporaryDirectory() as directory:
            cache = Qwen38BF4Cache(directory, identity, contract)
            _, paths = _write_layer(cache, 0)
            artifact = paths["w0_w1"]
            target = artifact.with_name("target.tensorbin")
            _copy_sparse_tensorbin(artifact, target)
            artifact.unlink()
            artifact.symlink_to(target)
            with (
                mock.patch.object(bf4_module, "qualify_live_bf4_ring", return_value=RING7_ORDER),
                mock.patch.object(bf4_module.ttnn, "load_tensor") as load,
            ):
                with self.assertRaisesRegex(RuntimeError, "unavailable or invalid"):
                    cache.load_layer(mesh, layer_index=0)
                load.assert_not_called()

    def test_manifest_mutation_cannot_redirect_verified_or_loaded_layer(self):
        identity = _identity()
        mesh = object()
        contract = _TrackingMeshContract(identity.physical_ids, mesh)
        with tempfile.TemporaryDirectory() as directory:
            cache = Qwen38BF4Cache(directory, identity, contract)
            _write_layer(cache, 0)
            cache.verify_layer("backbone", 0)
            baseline = json.loads(cache.manifest_path.read_text(encoding="utf-8"))
            key = "backbone:0"

            mutations = (
                (
                    "record identity",
                    lambda document: document["layers"][key].__setitem__("layer_index", 1),
                ),
                (
                    "expert ownership",
                    lambda document: document["layers"][key].__setitem__(
                        "expert_ranges", [[128, 256], [0, 128], [256, 384], [384, 512]]
                    ),
                ),
                (
                    "requested slot",
                    lambda document: document["layers"][key]["w0_w1"].__setitem__(
                        "relative_path",
                        "backbone/layer-01/w0_w1_dtype_BFLOAT4_B_layout_TILE.tensorbin",
                    ),
                ),
                (
                    "canonical slot shape",
                    lambda document: document["layers"][key]["w0_w1"].__setitem__(
                        "logical_shape", [7, 1, 512, 18, 224, 160]
                    ),
                ),
                (
                    "scalar schema",
                    lambda document: document["layers"][key]["w2"].__setitem__("bytes", True),
                ),
                (
                    "manifest schema",
                    lambda document: document["layers"][key]["w2"].__setitem__("unbound", "field"),
                ),
                (
                    "identity digest",
                    lambda document: document.__setitem__("identity_key", "0" * 64),
                ),
                (
                    "manifest identity",
                    lambda document: document["identity"].__setitem__("ring_size", 7.0),
                ),
                (
                    "manifest identity",
                    lambda document: document["identity"]["physical_ids"].__setitem__(0, False),
                ),
                (
                    "manifest format",
                    lambda document: document.__setitem__("format_version", True),
                ),
                (
                    "manifest format",
                    lambda document: document.__setitem__("format_version", 1.0),
                ),
            )
            with (
                mock.patch.object(bf4_module, "qualify_live_bf4_ring", return_value=RING7_ORDER),
                mock.patch.object(bf4_module.ttnn, "load_tensor") as load,
            ):
                for expected, mutate in mutations:
                    with self.subTest(expected=expected):
                        document = copy.deepcopy(baseline)
                        mutate(document)
                        bf4_module._atomic_json(cache.manifest_path, document)
                        with self.assertRaisesRegex(RuntimeError, expected):
                            cache.load_layer(mesh, layer_index=0)
                        load.assert_not_called()

    def test_cache_rejects_layer_aliases_and_non_target_slots_before_manifest_access(self):
        identity = _identity()
        contract = Qwen38MeshContract(identity.physical_ids)
        with tempfile.TemporaryDirectory() as directory:
            cache = Qwen38BF4Cache(directory, identity, contract)
            with mock.patch.object(cache, "_read_manifest", side_effect=AssertionError("manifest was read")):
                for invalid in (True, 0.0, "0"):
                    with self.subTest(invalid=invalid):
                        with self.assertRaisesRegex(TypeError, "exact integer"):
                            cache.verify_layer("backbone", invalid)
                with self.assertRaisesRegex(ValueError, r"\[0,48\)"):
                    cache.verify_layer("backbone", BACKBONE_LAYERS)
                with self.assertRaisesRegex(ValueError, r"\[0,1\)"):
                    cache.verify_layer("mtp", 1)

    def test_layer_host_packing_fills_four_canonical_ranges_without_cat(self):
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
                    expert_range=EXPERT_RANGES[device_index],
                    gate_up=f"gate-up-{device_index}",
                    down=f"down-{device_index}",
                )

        def split(gate_up, split_size, *, dim):
            device_index = int(gate_up.rsplit("-", 1)[1])
            self.assertEqual((split_size, dim), (640, -1))
            return f"gate-{device_index}", f"up-{device_index}"

        def prepare_w01(gate, up, layers, experts, hidden, intermediate, _shard_map):
            device_index = int(gate.rsplit("-", 1)[1])
            self.assertEqual(up, f"up-{device_index}")
            self.assertEqual((layers, experts, hidden, intermediate), (1, 128, 2560, 640))
            local_shape = shapes["w0_w1"][:2] + (128,) + shapes["w0_w1"][3:]
            return Prepared("w0_w1", device_index, local_shape)

        def prepare_w2(down, layers, experts, intermediate, hidden, _w2_map, _w01_map):
            device_index = int(down.rsplit("-", 1)[1])
            self.assertEqual((layers, experts, intermediate, hidden), (1, 128, 640, 2560))
            local_shape = shapes["w2"][:2] + (128,) + shapes["w2"][3:]
            return Prepared("w2", device_index, local_shape)

        with (
            mock.patch.object(bf4_module.torch, "empty", side_effect=Destination) as empty,
            mock.patch.object(bf4_module.torch, "split", side_effect=split),
            mock.patch.object(bf4_module, "prepare_w0_w1_tensor_for_moe_compute", side_effect=prepare_w01),
            mock.patch.object(bf4_module, "prepare_w2_tensor_for_moe_compute", side_effect=prepare_w2),
            mock.patch.object(bf4_module.torch, "cat", side_effect=AssertionError("whole-layer cat is forbidden")),
        ):
            w01, w2 = bf4_module._prepare_routed_layer_host_tensors(Weights(), ring_size=7)

        self.assertEqual(w01.shape, shapes["w0_w1"])
        self.assertEqual(w2.shape, shapes["w2"])
        self.assertEqual(loaded, [0, 1, 2, 3])
        self.assertEqual(
            copies,
            [
                item
                for device_index, start in enumerate((0, 128, 256, 384))
                for item in (
                    ("w0_w1", start, 128, "w0_w1", device_index),
                    ("w2", start, 128, "w2", device_index),
                )
            ],
        )
        self.assertEqual(empty.call_count, 2)
        self.assertEqual([shape for _, shape, _ in allocations], [shapes["w0_w1"], shapes["w2"]])

    def test_fresh_conversion_aggregates_second_upload_and_first_tensor_cleanup_failures(self):
        identity = _identity()
        mesh = SimpleNamespace(shape=(1, 4))
        contract = _TrackingMeshContract(identity.physical_ids, mesh)
        placement = SimpleNamespace(expert_ranges=EXPERT_RANGES)
        memory_configs = SimpleNamespace(w0_w1=object(), w2=object())
        logical_shapes = bf4_module.canonical_packed_shapes(ring_size=identity.ring_size)
        first_tensor = _FakeBF4Tensor(memory_configs.w0_w1, logical_shapes["w0_w1"])

        class HostTensor:
            pass

        weights = SimpleNamespace(
            expert_ranges=EXPERT_RANGES,
        )
        with tempfile.TemporaryDirectory() as directory:
            cache = Qwen38BF4Cache(directory, identity, contract)
            with (
                mock.patch.object(cache, "verify_layer", return_value=None),
                mock.patch.object(bf4_module, "qualify_live_bf4_ring", return_value=RING7_ORDER),
                mock.patch.object(bf4_module, "Qwen38MoEWeights", return_value=weights),
                mock.patch.object(bf4_module.ttnn, "ShardTensor2dMesh", return_value=object()),
                mock.patch.object(
                    bf4_module.ttnn.experimental,
                    "get_weight_mem_configs",
                    return_value=memory_configs,
                ),
                mock.patch.object(
                    bf4_module,
                    "_prepare_routed_layer_host_tensors",
                    return_value=(HostTensor(), HostTensor()),
                ) as prepare,
                mock.patch.object(
                    bf4_module.ttnn,
                    "as_tensor",
                    side_effect=[first_tensor, RuntimeError("synthetic second upload failure")],
                ),
                mock.patch.object(
                    bf4_module.ttnn,
                    "deallocate",
                    side_effect=RuntimeError("synthetic conversion cleanup failure"),
                ) as deallocate,
            ):
                with self.assertRaisesRegex(
                    bf4_module.BF4CleanupError, "owned tensor cleanup was incomplete"
                ) as raised:
                    cache._convert_and_upload_locked(
                        object(),
                        placement,
                        mesh,
                        layer_index=0,
                    )
            self.assertRegex(str(raised.exception.primary_error), "synthetic second upload failure")
            self.assertEqual(len(raised.exception.cleanup_errors), 1)
            self.assertRegex(str(raised.exception.cleanup_errors[0]), "synthetic conversion cleanup failure")
            deallocate.assert_called_once_with(first_tensor)
            prepare.assert_called_once_with(weights, ring_size=identity.ring_size)

    def _fresh_conversion(self, cache: Qwen38BF4Cache, mesh, *, uploaded_shapes: dict[str, tuple[int, ...]]):
        """One layer conversion with an ``as_tensor`` that writes the staged tensorbins and returns tensors of
        ``uploaded_shapes`` (the shape a mesh tensor presents); the digests sample the sparse fixtures."""

        identity = cache.identity
        memory_configs = SimpleNamespace(w0_w1=object(), w2=object())
        logical_shapes = bf4_module.canonical_packed_shapes(ring_size=identity.ring_size)
        uploaded = []

        def as_tensor(host, *, dtype, layout, device, memory_config, mesh_mapper, cache_file_name):
            name = cache_file_name.name
            self.assertIs(device, mesh)
            self.assertIs(memory_config, getattr(memory_configs, name))
            _write_sparse_tensorbin(
                bf4_module._tensorbin_path(cache_file_name),
                payload_bytes=bf4_module._packed_payload_bytes(logical_shapes[name]),
                marker=f"packed:{name}".encode(),
            )
            uploaded.append(_FakeBF4Tensor(memory_config, uploaded_shapes[name]))
            return uploaded[-1]

        with (
            mock.patch.object(bf4_module, "qualify_live_bf4_ring", return_value=RING7_ORDER),
            mock.patch.object(
                bf4_module, "Qwen38MoEWeights", return_value=SimpleNamespace(expert_ranges=EXPERT_RANGES)
            ),
            mock.patch.object(bf4_module.ttnn, "ShardTensor2dMesh", return_value=object()),
            mock.patch.object(bf4_module.ttnn.experimental, "get_weight_mem_configs", return_value=memory_configs),
            mock.patch.object(bf4_module, "_prepare_routed_layer_host_tensors", return_value=(object(), object())),
            mock.patch.object(bf4_module.ttnn, "as_tensor", side_effect=as_tensor),
            mock.patch.object(bf4_module, "_sha256", side_effect=_fast_sparse_digest),
            mock.patch.object(bf4_module, "_sha256_fd", side_effect=_fast_sparse_digest_fd),
            mock.patch.object(bf4_module.ttnn, "deallocate") as deallocate,
        ):
            try:
                tensors = cache._convert_and_upload_locked(
                    object(), SimpleNamespace(expert_ranges=EXPERT_RANGES), mesh, layer_index=0
                )
            except RuntimeError as error:
                return uploaded, deallocate, error
        self.assertEqual(list(tensors), uploaded)
        return uploaded, deallocate, None

    def test_fresh_conversion_records_the_slot_shape_and_accepts_coordinate_local_shards(self):
        identity = _identity()
        mesh = SimpleNamespace(shape=(1, 4))
        logical_shapes = bf4_module.canonical_packed_shapes(ring_size=identity.ring_size)
        mesh_shapes = _mesh_shapes(identity)
        self.assertEqual(mesh_shapes["w0_w1"][2] * 4, logical_shapes["w0_w1"][2])
        with tempfile.TemporaryDirectory() as directory:
            cache = Qwen38BF4Cache(directory, identity, _TrackingMeshContract(identity.physical_ids, mesh))
            uploaded, deallocate, error = self._fresh_conversion(cache, mesh, uploaded_shapes=mesh_shapes)
            self.assertIsNone(error)
            deallocate.assert_not_called()
            with mock.patch.object(bf4_module, "_sha256_fd", side_effect=_fast_sparse_digest_fd):
                record = cache.verify_layer("backbone", 0)
            self.assertIsNotNone(record)
            # The manifest records the slot's global shape; the uploaded tensors present one coordinate's shard.
            self.assertEqual(record.w0_w1.logical_shape, logical_shapes["w0_w1"])
            self.assertEqual(record.w2.logical_shape, logical_shapes["w2"])
            self.assertEqual([tuple(tensor.shape) for tensor in uploaded], [mesh_shapes["w0_w1"], mesh_shapes["w2"]])
            published = sorted(str(path.relative_to(cache.root)) for path in cache.root.rglob("*") if path.is_file())
            self.assertEqual(published, sorted([record.w0_w1.relative_path, record.w2.relative_path, "manifest.json"]))

    def test_fresh_conversion_refuses_a_global_shaped_upload_and_publishes_nothing(self):
        identity = _identity()
        mesh = SimpleNamespace(shape=(1, 4))
        logical_shapes = bf4_module.canonical_packed_shapes(ring_size=identity.ring_size)
        with tempfile.TemporaryDirectory() as directory:
            cache = Qwen38BF4Cache(directory, identity, _TrackingMeshContract(identity.physical_ids, mesh))
            uploaded, deallocate, error = self._fresh_conversion(cache, mesh, uploaded_shapes=logical_shapes)
            self.assertRegex(str(error), "BF4 conversion w0_w1 loaded shape .* coordinate-local slot shape")
            self.assertEqual([call.args[0] for call in deallocate.call_args_list], uploaded)
            # No orphan tensorbin, no manifest: the refused layer is absent, not unadoptable.
            self.assertEqual([path for path in cache.root.rglob("*") if path.is_file()], [])
            self.assertIsNone(cache.verify_layer("backbone", 0))

    def test_streamer_attempts_both_releases_and_clears_active_slot(self):
        class Tensor(_FakeBackingTensor):
            def __init__(self, name, identity):
                super().__init__(backing=identity)
                self.name = name

            def __str__(self):
                return self.name

        w01, w2 = Tensor("w01", 1), Tensor("w2", 2)

        class Contract:
            @staticmethod
            def validate_mesh(_mesh):
                return None

        class Cache:
            mesh_contract = Contract()

            @staticmethod
            def load_layer(_mesh, *, layer_index, namespace):
                self.assertEqual((namespace, layer_index), ("backbone", 7))
                return (w01, w2)

        streamer = Qwen38BF4Streamer(Cache(), object())
        released = []

        def release(tensor):
            released.append(tensor)
            if tensor is w01:
                raise RuntimeError("first release failed")

        with mock.patch.object(bf4_module.ttnn, "deallocate", side_effect=release):
            with self.assertRaisesRegex(RuntimeError, "first release failed"):
                with streamer.layer(7):
                    pass
        self.assertEqual(released, [w01, w2])
        self.assertIsNone(streamer._active)

    def test_streamer_aggregates_body_and_all_cleanup_failures(self):
        class Tensor(_FakeBackingTensor):
            def __init__(self, name, identity):
                super().__init__(backing=identity)
                self.name = name

            def __str__(self):
                return self.name

        w01, w2 = Tensor("w01", 1), Tensor("w2", 2)

        class Contract:
            @staticmethod
            def validate_mesh(_mesh):
                return None

        class Cache:
            mesh_contract = Contract()

            @staticmethod
            def load_layer(_mesh, *, layer_index, namespace):
                self.assertEqual((namespace, layer_index), ("backbone", 7))
                return (w01, w2)

        streamer = Qwen38BF4Streamer(Cache(), object())
        released = []

        def release(tensor):
            released.append(tensor)
            raise RuntimeError(f"release failed for {tensor}")

        with mock.patch.object(bf4_module.ttnn, "deallocate", side_effect=release):
            with self.assertRaisesRegex(bf4_module.BF4CleanupError, "streamer body failed") as raised:
                with streamer.layer(7):
                    raise ValueError("synthetic body failure")
        self.assertRegex(str(raised.exception.primary_error), "synthetic body failure")
        self.assertEqual(len(raised.exception.cleanup_errors), 2)
        self.assertEqual(released, [w01, w2])
        self.assertIsNone(streamer._active)


class TTNNBF4ConverterIdentityTest(unittest.TestCase):
    """The cache pins the converter's sources instead of the tt-metal revision, adopts the old slots, checks bytes."""

    def setUp(self):
        patcher = mock.patch.object(bf4_module, "_sha256_fd", side_effect=_fast_sparse_digest_fd)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_converter_source_identity_pins_this_module_the_layout_packer_and_the_bfp4_packer(self):
        identity = bf4_module.bf4_converter_source_identity()
        self.assertEqual(
            [path for path, _ in identity],
            [
                "models/demos/blackhole/qwen38_flash_next/ttnn/bf4.py",
                "tt_metal/impl/data_format/bfloat4.cpp",
                "tt_metal/impl/data_format/blockfloat_common.cpp",
                "tt_metal/impl/data_format/blockfloat_common.hpp",
                "ttnn/ttnn/_experimental/moe_compute_utils.py",
            ],
        )
        for relative, digest in identity:
            self.assertEqual(digest, hashlib.sha256((bf4_module.REPO_ROOT / relative).read_bytes()).hexdigest())
        accepted = replace(_identity(), converter_sources=identity)
        self.assertNotIn("tt_metal_revision", asdict(accepted))
        self.assertNotEqual(accepted.key, _identity().key)
        for malformed in (
            (),
            tuple(reversed(CONVERTER_SOURCES)),
            (CONVERTER_SOURCES[0], CONVERTER_SOURCES[0]),
            (("models/demos/blackhole/qwen38_flash_next/ttnn/bf4.py", "X" * 64),),
            (("models/demos/blackhole/qwen38_flash_next/ttnn/bf4.py", "1" * 40),),
            [tuple(CONVERTER_SOURCES[0])],
        ):
            with self.subTest(malformed=malformed):
                with self.assertRaisesRegex(ValueError, "converter sources"):
                    replace(_identity(), converter_sources=malformed)

    def test_legacy_slots_keyed_by_the_revision_or_the_worker_coordinates_are_adopted_in_place(self):
        identity = _identity()
        contract = Qwen38MeshContract(identity.physical_ids)

        def slot_key(legacy_identity: dict) -> str:
            return hashlib.sha256(
                json.dumps(legacy_identity, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()

        def demote(document: dict, legacy_identity: dict) -> str:
            """Rewrite the manifest as the legacy format its identity names; the slot's new directory name."""

            key = slot_key(legacy_identity)
            document.update(
                format_version=legacy_identity["format_version"], identity=legacy_identity, identity_key=key
            )
            return key

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
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
                [path, "f" * 64 if path.endswith("/bf4.py") else digest]
                for path, digest in current["converter_sources"]
            ]
            format2["format_version"] = 2
            self.assertEqual(
                bf4_module.dram_bank_ring_order(tuple(map(tuple, format2["dram_bank_worker_order"]))), RING7_ORDER
            )
            format2_key = demote(document, format2)
            bf4_module._atomic_json(cache.manifest_path, document)
            os.rename(cache.root, root / format2_key)
            # Decoys, every one newer than the slot so it would win if it were adoptable: another ring size, coordinates
            # that derive to another ring order (banks 0 and 1 swap workers), another BFP4 packer digest, a format the
            # code does not know, an unreadable manifest and a stray file.
            decoys = {}
            decoys["decoy-ring-8"] = copy.deepcopy(format2)
            decoys["decoy-ring-8"]["ring_size"] = 8
            decoys["decoy-other-order"] = copy.deepcopy(format2)
            decoys["decoy-other-order"]["dram_bank_worker_order"] = [[1, 1], [0, 1], *([x, 1] for x in range(2, 7))]
            decoys["decoy-other-packer"] = copy.deepcopy(format2)
            decoys["decoy-other-packer"]["converter_sources"] = [
                [path, "e" * 64 if path.endswith("bfloat4.cpp") else digest]
                for path, digest in format2["converter_sources"]
            ]
            decoys["decoy-format-9"] = copy.deepcopy(format2)
            decoys["decoy-format-9"]["format_version"] = 9
            # A slot converted before the packed layout joined the identity, and one that names the stride layout:
            # their bytes are the per-core stride layout, never adoptable into a layout-2 slot.
            decoys["decoy-no-layout-version"] = copy.deepcopy(format2)
            del decoys["decoy-no-layout-version"]["layout_version"]
            decoys["decoy-layout-1"] = copy.deepcopy(format2)
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

            self.assertEqual(adopted.root, root / identity.key)
            self.assertTrue(adopted.manifest_path.exists())
            self.assertFalse((root / format2_key).exists())
            for name in (*decoys, "decoy-broken"):
                self.assertTrue((root / name / "manifest.json").exists(), name)
            adopted_document = json.loads(adopted.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(adopted_document["format_version"], 3)
            self.assertEqual(adopted_document["identity_key"], identity.key)
            self.assertEqual(adopted_document["identity"], bf4_module._json_normalized(asdict(identity)))
            self.assertEqual(sorted(adopted_document["layers"]), ["backbone:0", "backbone:5"])
            self.assertEqual(adopted_document["created_utc"], document["created_utc"])
            self.assertEqual(adopted.verify_layer("backbone", 0), record0)
            self.assertEqual(adopted.verify_layer("backbone", 5), record5)
            self.assertEqual(
                adopted._validate_record(record0, namespace="backbone", layer_index=0)[0],
                adopted.root / paths0["w0_w1"].relative_to(cache.root),
            )
            # Format 1 was keyed by the tt-metal revision and had no converter sources.
            format1 = {key: value for key, value in common.items() if key != "converter_sources"}
            format1["dram_bank_worker_order"] = [list(coordinate) for coordinate in RING7_WORKERS]
            format1["tt_metal_revision"] = "9fb1403a93" + "0" * 30
            format1["format_version"] = 1
            format1_key = demote(adopted_document, format1)
            bf4_module._atomic_json(adopted.manifest_path, adopted_document)
            os.rename(adopted.root, root / format1_key)
            adopted = Qwen38BF4Cache(root, identity, contract)
            self.assertFalse((root / format1_key).exists())
            self.assertEqual(json.loads(adopted.manifest_path.read_text(encoding="utf-8"))["format_version"], 3)
            self.assertEqual(adopted.verify_layer("backbone", 5), record5)
            # An interrupted adoption (format 3, our key, another directory name) is only renamed.
            os.rename(adopted.root, root / "interrupted")
            renamed = Qwen38BF4Cache(root, identity, contract)
            self.assertTrue(renamed.manifest_path.exists())
            self.assertFalse((root / "interrupted").exists())
            self.assertEqual(renamed.verify_layer("backbone", 5), record5)
            # Nothing to adopt: a fresh root stays empty until a layer is converted.
            with tempfile.TemporaryDirectory() as empty:
                self.assertFalse(Qwen38BF4Cache(empty, identity, contract).manifest_path.exists())
                self.assertEqual(os.listdir(empty), [])

    def test_expert_byte_ranges_walk_the_tiled_shards_one_run_per_ring_bank(self):
        shapes = bf4_module.canonical_packed_shapes(ring_size=7)
        physical_ids = (0, 1, 2, 3)
        for name, blocks, rows in (("w0_w1", 18, 224), ("w2", 3, 672)):
            with self.subTest(name=name):
                run = blocks * (rows // 32) * 4 * 576
                payload = bf4_module._packed_payload_bytes(shapes[name])
                shard = payload // 4
                first = bf4_module._expert_byte_ranges(shapes[name], 0, physical_ids)
                last = bf4_module._expert_byte_ranges(shapes[name], 511, physical_ids)
                middle = bf4_module._expert_byte_ranges(shapes[name], 200, physical_ids)
                self.assertEqual(first, tuple((ring_bank * 128 * run, run) for ring_bank in range(7)))
                self.assertEqual(last[0], (3 * shard + 127 * run, run))
                self.assertEqual(last[-1][0] + run, payload)
                self.assertEqual(middle, tuple((shard + (ring_bank * 128 + 72) * run, run) for ring_bank in range(7)))
                self.assertEqual(bf4_module.ADMISSION_EXPERT, 511)

    def test_admission_compares_one_expert_with_a_fresh_packing_and_names_the_differing_bank(self):
        identity = _identity()
        contract = Qwen38MeshContract(identity.physical_ids)
        shapes = bf4_module.canonical_packed_shapes(ring_size=7)
        with tempfile.TemporaryDirectory() as directory:
            cache = Qwen38BF4Cache(directory, identity, contract)
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
            self.assertEqual(
                packer.call_args.kwargs | {"scratch": None},
                {"namespace": "backbone", "layer_index": 3, "expert": 200, "ring_size": 7, "scratch": None},
            )
            self.assertEqual(packer.call_args.args, (checkpoint, placement))
            self.assertTrue(str(packer.call_args.kwargs["scratch"]).startswith(str(cache.root / ".admit.")))
            self.assertEqual(sorted(os.listdir(cache.root)), ["backbone", "manifest.json"])  # the scratch is gone
            self.assertEqual(
                sorted(os.listdir(cache.root / "backbone" / "layer-03")), sorted(path.name for path in paths.values())
            )
            self.assertEqual(
                {key: admitted[key] for key in ("namespace", "layer_index", "expert", "compared_bytes")},
                {
                    "namespace": "backbone",
                    "layer_index": 3,
                    "expert": 200,
                    "compared_bytes": sum(map(len, fresh.values())),
                },
            )
            self.assertGreaterEqual(admitted["seconds"], 0.0)

            corrupt = dict(fresh)
            corrupt["w2"] = bytearray(fresh["w2"])
            size = len(fresh["w2"]) // 7
            corrupt["w2"][4 * size + 17] ^= 0x01
            corrupt["w2"] = bytes(corrupt["w2"])
            with mock.patch.object(bf4_module, "_fresh_expert_bf4_bytes", return_value=corrupt):
                with self.assertRaisesRegex(
                    RuntimeError, r"backbone layer 3 expert 200 \(w2, ring bank 4\) differs .* at byte 17 of "
                ):
                    cache.admit_converted_bytes(checkpoint, placement, layer_index=3, expert=200)
            short = dict(fresh)
            short["w0_w1"] = fresh["w0_w1"][:-1]
            with mock.patch.object(bf4_module, "_fresh_expert_bf4_bytes", return_value=short):
                with self.assertRaisesRegex(RuntimeError, "fresh w0_w1 packing of expert 200"):
                    cache.admit_converted_bytes(checkpoint, placement, layer_index=3, expert=200)
            with self.assertRaisesRegex(RuntimeError, "holds no backbone layer 9 to admit"):
                cache.admit_converted_bytes(checkpoint, placement, layer_index=9)
            with self.assertRaisesRegex(ValueError, "expert must be in"):
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


# Observed on a QuietBox 2 (2x p300c): three dies serve their banks from worker column 6, the fourth from column 5.
QB2_COLUMN_6 = ((0, 9), (0, 0), (0, 7), (0, 3), (6, 9), (6, 1), (6, 6), (6, 4))
QB2_COLUMN_5 = ((0, 9), (0, 0), (0, 7), (0, 3), (5, 9), (5, 1), (5, 6), (5, 4))
QB2_RING_ORDER = (4, 0, 2, 6, 7, 3, 5, 1)


class BF4LiveRingQualificationTest(unittest.TestCase):
    def _qualify(self, per_die_signatures):
        mesh = _RingMeshDevice(per_die_signatures)
        with mock.patch.object(
            bf4_module.ttnn.device,
            "get_optimal_dram_bank_to_logical_worker_assignment_at_mesh_coordinate",
            side_effect=lambda device, noc, coordinate: mesh.assignment(coordinate),
        ):
            return bf4_module.qualify_live_bf4_ring(mesh)

    def test_ring_order_is_the_bank_order_the_packing_produces(self):
        """The guard's sort against ``get_weight_core_shard_maps`` on the same assignment: one sort, pinned here."""

        scrambled = ((3, 2), (0, 0), (5, 1), (1, 7), (6, 2), (2, 9), (4, 4), (0, 3))
        for signature in (QB2_COLUMN_6, QB2_COLUMN_5, RING7_WORKERS, scrambled):
            with self.subTest(signature=signature):
                mesh = _RingMeshDevice([signature] * 4)
                with mock.patch.object(
                    bf4_module.ttnn.device,
                    "get_optimal_dram_bank_to_logical_worker_assignment",
                    side_effect=lambda device, noc: mesh.assignment(),
                ):
                    _, _, placement_grid = bf4_module.moe_compute_utils.get_weight_core_shard_maps(mesh, 2560, 640)
                packed_bank_order = tuple(core_range.start.x for core_range in placement_grid.ranges())
                self.assertEqual(len(packed_bank_order), len(signature))
                self.assertEqual(bf4_module.dram_bank_ring_order(signature), packed_bank_order)
        self.assertEqual(bf4_module.dram_bank_ring_order(QB2_COLUMN_6), QB2_RING_ORDER)
        self.assertEqual(bf4_module.dram_bank_ring_order(RING7_WORKERS), RING7_ORDER)

    def test_uniform_ring_is_accepted(self):
        self.assertEqual(self._qualify([QB2_COLUMN_6] * 4), QB2_RING_ORDER)

    def test_mixed_harvesting_with_one_ring_order_is_accepted(self):
        """The QuietBox 2 case: one die's workers a column away from its banks, the same bank ids in the same order."""

        self.assertEqual(self._qualify([QB2_COLUMN_6, QB2_COLUMN_5, QB2_COLUMN_6, QB2_COLUMN_6]), QB2_RING_ORDER)
        # The odd die first: the same result, so the cache identity is the same.
        self.assertEqual(self._qualify([QB2_COLUMN_5, QB2_COLUMN_6, QB2_COLUMN_6, QB2_COLUMN_6]), QB2_RING_ORDER)

    def test_mixed_ring_order_is_rejected_with_the_coordinates(self):
        # Two banks swap workers on one die: the same coordinates as a set, another ring order.
        reordered = (QB2_COLUMN_6[1], QB2_COLUMN_6[0], *QB2_COLUMN_6[2:])
        with self.assertRaisesRegex(
            RuntimeError, "mixed Blackhole DRAM ring orders.*\\(0, 9\\).*\\(4, 0, 2, 6, 7, 3, 5, 1\\)"
        ):
            self._qualify([QB2_COLUMN_6, reordered, QB2_COLUMN_6, QB2_COLUMN_6])

    def test_mixed_ring_size_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "mixed Blackhole DRAM ring sizes.*8.*7"):
            self._qualify([QB2_COLUMN_6, QB2_COLUMN_6[:7], QB2_COLUMN_6, QB2_COLUMN_6])

    def test_unsupported_ring_size_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "unsupported DRAM worker order"):
            self._qualify([QB2_COLUMN_6[:5]] * 4)


if __name__ == "__main__":
    unittest.main()
