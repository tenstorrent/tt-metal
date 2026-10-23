# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device contracts for the exact TTNN Qwen3.8 MTP input mixer."""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path
from types import SimpleNamespace

import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.checkpoint import (
    CHECKPOINT_FILE_MANIFEST_SHA256,
    CHECKPOINT_TENSOR_MANIFEST_SHA256,
    INDEX_SHA256,
    PINNED_CHECKPOINT_REVISION,
)
from models.demos.blackhole.qwen38_flash_next.config import CONFIG_SHA256
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import (
    Qwen38TTNNMTPComponents,
    validate_builder_constructor_contract,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import TensorPlacement
from models.demos.blackhole.qwen38_flash_next.ttnn.mtp import (
    EMBEDDING_GLOBAL_SHAPE,
    EMBEDDING_LOCAL_SHAPE,
    HIDDEN_NORM_SCALE_GLOBAL_SHAPE,
    HIDDEN_NORM_SCALE_LOCAL_SHAPE,
    HIDDEN_SIZE,
    MTP_INPUT_TENSOR_SPECS,
    PROJECTION_LOCAL_SHAPE,
    RESIDUAL_BRANCHES,
    RESIDUAL_GLOBAL_SHAPE,
    RESIDUAL_LOCAL_SHAPE,
    RESIDUAL_WIDTH,
    Qwen38TTNNMTPInput,
    Qwen38TTNNMTPInputCleanupError,
    Qwen38TTNNMTPInputWeights,
    _cache_directory,
    _prepare_host_weights,
    _validate_checkpoint_contract,
    validate_mtp_input_static_contract,
)


class _FakeTensor:
    _next_id = 1

    def __init__(self, shape, dtype, placement, *, shard_dim=None, name="tensor", tensor_id=None) -> None:
        self.shape = tuple(shape)
        self.dtype = dtype
        self.layout = ttnn.TILE_LAYOUT
        self.placement = placement
        self.shard_dim = shard_dim
        self.name = name
        self.deallocated = False
        if tensor_id is None:
            self._id = _FakeTensor._next_id
            _FakeTensor._next_id += 1
        else:
            self._id = tensor_id

    def tensor_id(self) -> int:
        return self._id


class _FakeMeshContract:
    physical_ids = (0, 1, 2, 3)

    def validate_tensor(self, tensor, *, placement, shard_dim=None, **_) -> None:
        if tensor.placement is not placement:
            raise RuntimeError(f"{tensor.name} placement {tensor.placement} != {placement}")
        if shard_dim is not None and tensor.shard_dim != shard_dim:
            raise RuntimeError(f"{tensor.name} shard dim {tensor.shard_dim} != {shard_dim}")


def _exact_config():
    return SimpleNamespace(
        config_sha256=CONFIG_SHA256,
        hidden_size=HIDDEN_SIZE,
        residual_branches=RESIDUAL_BRANCHES,
        residual_width=RESIDUAL_WIDTH,
        rms_norm_eps=1.0e-6,
        mtp_layers=1,
        mtp_uses_shared_embeddings=True,
    )


def _exact_metadata():
    return {name: SimpleNamespace(dtype=dtype, shape=shape) for name, (dtype, shape) in MTP_INPUT_TENSOR_SPECS.items()}


def _fake_weights() -> Qwen38TTNNMTPInputWeights:
    hidden = TensorPlacement.HIDDEN_SHARDED
    return Qwen38TTNNMTPInputWeights(
        embedding_norm_scale=_FakeTensor(
            EMBEDDING_LOCAL_SHAPE,
            ttnn.float32,
            hidden,
            shard_dim=3,
            name="embedding_norm_scale",
        ),
        hidden_norm_scale=_FakeTensor(
            HIDDEN_NORM_SCALE_LOCAL_SHAPE,
            ttnn.float32,
            hidden,
            shard_dim=3,
            name="hidden_norm_scale",
        ),
        fc_embedding=_FakeTensor(
            PROJECTION_LOCAL_SHAPE,
            ttnn.bfloat16,
            hidden,
            shard_dim=3,
            name="fc_embedding",
        ),
        fc_hidden=_FakeTensor(
            PROJECTION_LOCAL_SHAPE,
            ttnn.bfloat16,
            hidden,
            shard_dim=3,
            name="fc_hidden",
        ),
        epsilon=1.0e-6,
        cache_directory=Path("/tmp/static-mtp-input-cache"),
    )


def test_exact_mtp_input_tensor_names_geometry_and_builder_api() -> None:
    validate_mtp_input_static_contract()
    validate_builder_constructor_contract()
    assert dict(MTP_INPUT_TENSOR_SPECS) == {
        "mtp.pre_fc_norm_embedding.weight": ("BF16", (2560,)),
        "mtp.pre_fc_norm_hidden.weight": ("BF16", (10240,)),
        "mtp.fc_embedding.weight": ("BF16", (2560, 2560)),
        "mtp.fc_hidden.weight": ("BF16", (2560, 2560)),
    }
    assert tuple(field.name for field in dataclasses.fields(Qwen38TTNNMTPComponents)) == (
        "identity",
        "input_mixer",
        "decoder_layer",
        "final_mixer",
    )
    mixer_parameters = tuple(inspect.signature(Qwen38TTNNMTPInput.__call__).parameters)
    assert mixer_parameters == ("self", "input_embedding", "hidden_residual")


def test_checkpoint_metadata_is_fully_preflighted_before_loading(expect_error) -> None:
    config = _exact_config()
    metadata = _exact_metadata()
    checkpoint = SimpleNamespace(config=config, metadata=lambda name: metadata[name])
    placement = SimpleNamespace(
        config=config,
        mesh_shape=(1, 4),
        physical_ids=(0, 1, 2, 3),
        hidden_ranges=((0, 640), (640, 1280), (1280, 1920), (1920, 2560)),
    )
    _validate_checkpoint_contract(checkpoint, placement, _FakeMeshContract())

    metadata["mtp.pre_fc_norm_hidden.weight"] = SimpleNamespace(dtype="BF16", shape=(2560,))
    with expect_error(ValueError, "mtp.pre_fc_norm_hidden.weight must be BF16"):
        _validate_checkpoint_contract(checkpoint, placement, _FakeMeshContract())


def test_host_preparation_keeps_branch_major_norm_and_transposes_projections() -> None:
    tensors = {
        "mtp.pre_fc_norm_embedding.weight": torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16),
        "mtp.pre_fc_norm_hidden.weight": torch.zeros(RESIDUAL_WIDTH, dtype=torch.bfloat16),
        "mtp.fc_embedding.weight": torch.zeros(HIDDEN_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16),
        "mtp.fc_hidden.weight": torch.zeros(HIDDEN_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16),
    }
    tensors["mtp.pre_fc_norm_embedding.weight"][19] = 2.0
    tensors["mtp.pre_fc_norm_hidden.weight"][2 * HIDDEN_SIZE + 23] = 3.0
    tensors["mtp.fc_embedding.weight"][17, 29] = 4.0
    tensors["mtp.fc_hidden.weight"][31, 37] = 5.0

    prepared = _prepare_host_weights(tensors)

    assert prepared["embedding_norm_scale"].shape == EMBEDDING_GLOBAL_SHAPE
    assert prepared["hidden_norm_scale"].shape == HIDDEN_NORM_SCALE_GLOBAL_SHAPE
    assert prepared["embedding_norm_scale"].dtype == prepared["hidden_norm_scale"].dtype == torch.float32
    assert float(prepared["embedding_norm_scale"][0, 0, 0, 19]) == 3.0
    assert float(prepared["hidden_norm_scale"][0, 0, 2, 23]) == 4.0
    assert float(prepared["fc_embedding"][0, 0, 29, 17]) == 4.0
    assert float(prepared["fc_hidden"][0, 0, 37, 31]) == 5.0


def test_cache_path_binds_complete_checkpoint_runtime_and_physical_identity(tmp_path) -> None:
    checkpoint = SimpleNamespace(config=SimpleNamespace(config_sha256=CONFIG_SHA256))
    cache = _cache_directory(
        tmp_path.resolve(),
        checkpoint,
        _FakeMeshContract(),
        tt_metal_sha="1" * 40,
        ttnn_runtime_sha256="2" * 64,
    )
    rendered = str(cache)
    for required in (
        f"revision-{PINNED_CHECKPOINT_REVISION}",
        f"index-{INDEX_SHA256}",
        f"files-{CHECKPOINT_FILE_MANIFEST_SHA256}",
        f"tensors-{CHECKPOINT_TENSOR_MANIFEST_SHA256}",
        f"config-{CONFIG_SHA256}",
        f"tt-metal-{'1' * 40}",
        f"ttnn-runtime-{'2' * 64}",
        "mesh-1x4-physical-0-1-2-3",
    ):
        assert required in rendered


def test_mixer_operation_graph_is_distributed_norm_then_column_parallel_projection(monkeypatch) -> None:
    contract = _FakeMeshContract()
    weights = _fake_weights()
    mixer = Qwen38TTNNMTPInput.__new__(Qwen38TTNNMTPInput)
    mixer.mesh_contract = contract
    mixer.weights = weights
    mixer.collective_topology = object()
    mixer.compute_config = object()
    mixer.projection_compute_config = mixer.compute_config  # bf16: the same HiFi4 config

    hidden = TensorPlacement.HIDDEN_SHARDED
    replicated = TensorPlacement.REPLICATED
    embedding = _FakeTensor(EMBEDDING_LOCAL_SHAPE, ttnn.bfloat16, hidden, shard_dim=3, name="embedding")
    residual = _FakeTensor(RESIDUAL_LOCAL_SHAPE, ttnn.bfloat16, hidden, shard_dim=3, name="residual")
    all_gathers: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    rms_pre_inputs: list[tuple[int, ...]] = []
    rms_post_inputs: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    linears: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    released: list[int] = []

    def rms_norm_pre(tensor, **_):
        rms_pre_inputs.append(tensor.shape)
        return _FakeTensor((*tensor.shape[:-1], 32), ttnn.float32, hidden, shard_dim=3, name="stats")

    def reshape(tensor, shape):
        # TTNN reshape is a metadata view; it must not mutate the caller's
        # branch-major tensor or create a second allocation owner.
        return _FakeTensor(
            shape,
            tensor.dtype,
            tensor.placement,
            shard_dim=tensor.shard_dim,
            name=f"{tensor.name}.reshape",
            tensor_id=tensor.tensor_id(),
        )

    def all_gather(tensor, **_):
        output_shape = (*tensor.shape[:-1], tensor.shape[-1] * 4)
        all_gathers.append((tensor.shape, output_shape))
        return _FakeTensor(output_shape, tensor.dtype, replicated, name="gathered")

    def rms_norm_post(tensor, gathered, *, weight, **_):
        assert gathered.shape[-1] == 128
        rms_post_inputs.append((tensor.shape, weight.shape))
        return _FakeTensor(tensor.shape, ttnn.bfloat16, hidden, shard_dim=3, name="normalized")

    def linear(tensor, weight, **_):
        linears.append((tensor.shape, weight.shape))
        return _FakeTensor((*tensor.shape[:-1], weight.shape[-1]), ttnn.bfloat16, hidden, shard_dim=3, name="linear")

    def repeat(tensor, repeats, **_):
        return _FakeTensor(
            tuple(size * repeat for size, repeat in zip(tensor.shape, repeats)),
            tensor.dtype,
            hidden,
            shard_dim=3,
            name="broadcast",
        )

    def add(left, right, **_):
        assert left.shape == right.shape == RESIDUAL_LOCAL_SHAPE
        return _FakeTensor(left.shape, ttnn.bfloat16, hidden, shard_dim=3, name="output")

    def deallocate(tensor):
        assert not tensor.deallocated
        tensor.deallocated = True
        released.append(tensor.tensor_id())

    monkeypatch.setattr(ttnn, "rms_norm_pre_all_gather", rms_norm_pre)
    monkeypatch.setattr(ttnn, "reshape", reshape)
    monkeypatch.setattr(ttnn, "all_gather", all_gather)
    monkeypatch.setattr(ttnn, "rms_norm_post_all_gather", rms_norm_post)
    monkeypatch.setattr(ttnn, "linear", linear)
    monkeypatch.setattr(ttnn, "repeat", repeat)
    monkeypatch.setattr(ttnn, "add", add)
    monkeypatch.setattr(ttnn, "deallocate", deallocate)

    output = mixer(embedding, residual)

    assert output.shape == RESIDUAL_LOCAL_SHAPE
    assert output.placement is TensorPlacement.HIDDEN_SHARDED
    assert not embedding.deallocated and not residual.deallocated and not output.deallocated
    assert embedding.shape == EMBEDDING_LOCAL_SHAPE
    assert residual.shape == RESIDUAL_LOCAL_SHAPE
    flattened_hidden_local_shape = (1, 1, 1, RESIDUAL_BRANCHES * RESIDUAL_LOCAL_SHAPE[-1])
    assert rms_pre_inputs == [EMBEDDING_LOCAL_SHAPE, flattened_hidden_local_shape]
    assert rms_post_inputs == [
        (EMBEDDING_LOCAL_SHAPE, EMBEDDING_LOCAL_SHAPE),
        (
            flattened_hidden_local_shape,
            flattened_hidden_local_shape,
        ),
    ]
    assert all_gathers == [
        ((1, 1, 1, 32), (1, 1, 1, 128)),
        # pre_fc_norm_hidden is one global 4H=10,240-wide norm, not four
        # independent H-wide norms.
        ((1, 1, 1, 32), (1, 1, 1, 128)),
        (EMBEDDING_LOCAL_SHAPE, EMBEDDING_GLOBAL_SHAPE),
        (RESIDUAL_LOCAL_SHAPE, RESIDUAL_GLOBAL_SHAPE),
    ]
    assert linears == [
        (EMBEDDING_GLOBAL_SHAPE, PROJECTION_LOCAL_SHAPE),
        (RESIDUAL_GLOBAL_SHAPE, PROJECTION_LOCAL_SHAPE),
    ]
    assert len(released) == len(set(released)) == 11


def test_weight_cleanup_attempts_every_tensor_and_retries_only_failed_owner(monkeypatch, expect_error) -> None:
    weights = _fake_weights()
    original = [weights.embedding_norm_scale, weights.hidden_norm_scale, weights.fc_embedding, weights.fc_hidden]
    failed_id = original[1].tensor_id()
    attempts: list[int] = []
    fail_once = {failed_id}

    def deallocate(tensor):
        attempts.append(tensor.tensor_id())
        if tensor.tensor_id() in fail_once:
            fail_once.remove(tensor.tensor_id())
            raise RuntimeError("injected release failure")

    monkeypatch.setattr(ttnn, "deallocate", deallocate)

    with expect_error(Qwen38TTNNMTPInputCleanupError, "cleanup failed for 1 resource"):
        weights.deallocate()
    assert weights.embedding_norm_scale is None
    assert weights.hidden_norm_scale is original[1]
    assert weights.fc_embedding is None
    assert weights.fc_hidden is None
    assert not weights.released

    weights.deallocate()
    assert weights.released
    assert attempts.count(failed_id) == 2
    assert all(
        attempts.count(tensor.tensor_id()) == (2 if tensor.tensor_id() == failed_id else 1) for tensor in original
    )
    with expect_error(RuntimeError, "already deallocated"):
        weights.deallocate()
