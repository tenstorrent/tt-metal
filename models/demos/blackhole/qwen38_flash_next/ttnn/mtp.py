# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Exact TP4 Qwen3.8-Flash-Next MTP input mixer.

The released checkpoint does not concatenate a token embedding and a normal
hidden state.  Its recurrent hidden input is the four-branch gated-residual
state.  The two inputs are normalized independently, projected by two distinct
``2560 x 2560`` matrices, and the projected embedding is broadcast into every
residual branch::

    e = fc_embedding(zero_centered_rms_norm(embedding))
    h = fc_hidden(zero_centered_rms_norm(residual.reshape(4, H)))
    output = h + e[:, None, :, :]

Both inputs and the result are hidden-sharded over the exact ``1x4`` mesh.  A
projection weight is column-parallel (its output dimension is sharded); the
normalized activation is therefore explicitly all-gathered before the linear.
No projection or residual branch is silently replicated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence

import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.checkpoint import (
    CHECKPOINT_FILE_MANIFEST_SHA256,
    CHECKPOINT_TENSOR_MANIFEST_SHA256,
    INDEX_SHA256,
    PINNED_CHECKPOINT_REVISION,
    Qwen38Checkpoint,
)
from models.demos.blackhole.qwen38_flash_next.config import CONFIG_SHA256, Qwen38Placement
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import (
    CHUNK_ROW_COUNTS,
    MESH_SHAPE,
    Qwen38MeshContract,
    TensorPlacement,
    same_buffer,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.decode_matmul import dense_dtype_tag, dense_math_fidelity_name

TP_AXIS = 1
TP_SIZE = 4
HIDDEN_SIZE = 2560
LOCAL_HIDDEN_SIZE = HIDDEN_SIZE // TP_SIZE
RESIDUAL_BRANCHES = 4
RESIDUAL_WIDTH = RESIDUAL_BRANCHES * HIDDEN_SIZE
RMS_NORM_EPS = 1.0e-6
RMS_STATS_WIDTH_PER_DEVICE = 32
RMS_STATS_GLOBAL_WIDTH = RMS_STATS_WIDTH_PER_DEVICE * TP_SIZE
MTP_INPUT_CACHE_FORMAT_VERSION = 1

EMBEDDING_LOCAL_SHAPE = (1, 1, 1, LOCAL_HIDDEN_SIZE)
RESIDUAL_LOCAL_SHAPE = (1, RESIDUAL_BRANCHES, 1, LOCAL_HIDDEN_SIZE)
EMBEDDING_GLOBAL_SHAPE = (1, 1, 1, HIDDEN_SIZE)
RESIDUAL_GLOBAL_SHAPE = (1, RESIDUAL_BRANCHES, 1, HIDDEN_SIZE)
PROJECTION_LOCAL_SHAPE = (1, 1, HIDDEN_SIZE, LOCAL_HIDDEN_SIZE)
# The stored hidden RMS scale keeps its qualified [1,1,4,H] row layout (and
# weight cache); it is only ever consumed through the flattened 10,240-wide
# reshape, which is layout-invariant.
HIDDEN_NORM_SCALE_LOCAL_SHAPE = (1, 1, RESIDUAL_BRANCHES, LOCAL_HIDDEN_SIZE)
HIDDEN_NORM_SCALE_GLOBAL_SHAPE = (1, 1, RESIDUAL_BRANCHES, HIDDEN_SIZE)

MTP_INPUT_TENSOR_SPECS: Mapping[str, tuple[str, tuple[int, ...]]] = {
    "mtp.pre_fc_norm_embedding.weight": ("BF16", (HIDDEN_SIZE,)),
    "mtp.pre_fc_norm_hidden.weight": ("BF16", (RESIDUAL_WIDTH,)),
    "mtp.fc_embedding.weight": ("BF16", (HIDDEN_SIZE, HIDDEN_SIZE)),
    "mtp.fc_hidden.weight": ("BF16", (HIDDEN_SIZE, HIDDEN_SIZE)),
}


def _shape(tensor) -> tuple[int, ...]:
    return tuple(int(value) for value in tensor.shape)


def _require_lower_hex(value: str, length: int, *, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase {length}-hex, got {value!r}")


def _tensor_key(tensor) -> tuple[str, int]:
    tensor_id = getattr(tensor, "tensor_id", None)
    if callable(tensor_id):
        tensor_id = tensor_id()
    return ("ttnn", int(tensor_id)) if tensor_id is not None else ("python", id(tensor))


class Qwen38TTNNMTPInputCleanupError(RuntimeError):
    """One or more independent mixer-owned tensor releases failed."""

    def __init__(self, label: str, errors: Sequence[BaseException], *, primary: BaseException | None = None) -> None:
        self.label = label
        self.errors = tuple(errors)
        self.primary = primary
        detail = "; ".join(f"{type(error).__name__}: {error}" for error in errors)
        super().__init__(f"{label} cleanup failed for {len(errors)} resource(s): {detail}")


def _release_slots(
    slots: list[Any | None],
    *,
    label: str,
    primary: BaseException | None = None,
) -> None:
    """Release every unique tensor and preserve failed slots for one retry."""

    groups: dict[tuple[str, int], list[int]] = {}
    for index, tensor in enumerate(slots):
        if tensor is not None:
            groups.setdefault(_tensor_key(tensor), []).append(index)
    errors: list[BaseException] = []
    for indices in groups.values():
        tensor = slots[indices[0]]
        try:
            ttnn.deallocate(tensor)
        except BaseException as error:
            errors.append(error)
        else:
            for index in indices:
                slots[index] = None
    if errors:
        raise Qwen38TTNNMTPInputCleanupError(label, errors, primary=primary) from primary


def _cleanup_after_failure(slots: list[Any | None], *, label: str, primary: BaseException) -> NoReturn:
    try:
        _release_slots(slots, label=label, primary=primary)
    except BaseException as cleanup_error:
        raise cleanup_error
    raise primary


def _cache_directory(
    root: str | Path,
    checkpoint: Qwen38Checkpoint,
    mesh_contract: Qwen38MeshContract,
    *,
    tt_metal_sha: str,
    ttnn_runtime_sha256: str,
) -> Path:
    root = Path(root)
    if not root.is_absolute():
        raise ValueError(f"MTP input cache root must be absolute, got {root}")
    root = root.resolve()
    if root == Path(root.anchor):
        raise ValueError("MTP input cache root cannot be a filesystem root")
    _require_lower_hex(tt_metal_sha, 40, label="tt_metal_sha")
    _require_lower_hex(ttnn_runtime_sha256, 64, label="ttnn_runtime_sha256")
    physical = "-".join(str(value) for value in mesh_contract.physical_ids)
    path = (
        root
        / "mtp-input"
        / f"format-{MTP_INPUT_CACHE_FORMAT_VERSION}"
        / f"revision-{PINNED_CHECKPOINT_REVISION}"
        / f"index-{INDEX_SHA256}"
        / f"files-{CHECKPOINT_FILE_MANIFEST_SHA256}"
        / f"tensors-{CHECKPOINT_TENSOR_MANIFEST_SHA256}"
        / f"config-{checkpoint.config.config_sha256}"
        / f"tt-metal-{tt_metal_sha}"
        / f"ttnn-runtime-{ttnn_runtime_sha256}"
        / f"mesh-1x4-physical-{physical}"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_checkpoint_contract(
    checkpoint: Qwen38Checkpoint,
    placement: Qwen38Placement,
    mesh_contract: Qwen38MeshContract,
) -> None:
    config = checkpoint.config
    if placement.config != config:
        raise ValueError("MTP input checkpoint and placement configurations differ")
    if tuple(placement.mesh_shape) != MESH_SHAPE:
        raise ValueError(f"MTP input placement must use mesh {MESH_SHAPE}, got {placement.mesh_shape}")
    if tuple(placement.physical_ids) != mesh_contract.physical_ids:
        raise ValueError(
            f"MTP input placement physical order {placement.physical_ids} differs from admitted "
            f"{mesh_contract.physical_ids}"
        )
    exact = {
        "config_sha256": CONFIG_SHA256,
        "hidden_size": HIDDEN_SIZE,
        "residual_branches": RESIDUAL_BRANCHES,
        "residual_width": RESIDUAL_WIDTH,
        "rms_norm_eps": RMS_NORM_EPS,
        "mtp_layers": 1,
        "mtp_uses_shared_embeddings": True,
    }
    for name, expected in exact.items():
        actual = getattr(config, name)
        if actual != expected:
            raise ValueError(f"pinned MTP input requires {name}={expected!r}, got {actual!r}")
    expected_hidden_ranges = tuple(
        (device_index * LOCAL_HIDDEN_SIZE, (device_index + 1) * LOCAL_HIDDEN_SIZE) for device_index in range(TP_SIZE)
    )
    if tuple(placement.hidden_ranges) != expected_hidden_ranges:
        raise ValueError(f"MTP input hidden placement must be contiguous TP4, got {placement.hidden_ranges}")
    for tensor_name, (expected_dtype, expected_shape) in MTP_INPUT_TENSOR_SPECS.items():
        metadata = checkpoint.metadata(tensor_name)
        if metadata.dtype != expected_dtype or metadata.shape != expected_shape:
            raise ValueError(
                f"MTP input tensor {tensor_name} must be {expected_dtype} {expected_shape}, "
                f"got {metadata.dtype} {metadata.shape}"
            )


def _prepare_host_weights(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Transpose the two checkpoint ``[out,in]`` matrices for TTNN linear."""

    loaded: dict[str, torch.Tensor] = {}
    for tensor_name, (_, expected_shape) in MTP_INPUT_TENSOR_SPECS.items():
        try:
            value = tensors[tensor_name]
        except KeyError as error:
            raise KeyError(f"missing exact MTP input tensor {tensor_name}") from error
        if value.dtype != torch.bfloat16 or tuple(value.shape) != expected_shape:
            raise ValueError(
                f"loaded MTP input tensor {tensor_name} must be BF16 {expected_shape}, "
                f"got {value.dtype} {tuple(value.shape)}"
            )
        loaded[tensor_name] = value

    prepared = {
        # Qwen4Exp GemmaRMSNorm stores delta-gamma, hence the FP32 unit offset.
        "embedding_norm_scale": (1.0 + loaded["mtp.pre_fc_norm_embedding.weight"].float())
        .reshape(EMBEDDING_GLOBAL_SHAPE)
        .contiguous(),
        "hidden_norm_scale": (1.0 + loaded["mtp.pre_fc_norm_hidden.weight"].float())
        .reshape(HIDDEN_NORM_SCALE_GLOBAL_SHAPE)
        .contiguous(),
        "fc_embedding": loaded["mtp.fc_embedding.weight"]
        .transpose(0, 1)
        .reshape(1, 1, HIDDEN_SIZE, HIDDEN_SIZE)
        .contiguous(),
        "fc_hidden": loaded["mtp.fc_hidden.weight"]
        .transpose(0, 1)
        .reshape(1, 1, HIDDEN_SIZE, HIDDEN_SIZE)
        .contiguous(),
    }
    expected = {
        "embedding_norm_scale": (EMBEDDING_GLOBAL_SHAPE, torch.float32),
        "hidden_norm_scale": (HIDDEN_NORM_SCALE_GLOBAL_SHAPE, torch.float32),
        "fc_embedding": ((1, 1, HIDDEN_SIZE, HIDDEN_SIZE), torch.bfloat16),
        "fc_hidden": ((1, 1, HIDDEN_SIZE, HIDDEN_SIZE), torch.bfloat16),
    }
    for name, value in prepared.items():
        expected_shape, expected_dtype = expected[name]
        if tuple(value.shape) != expected_shape or value.dtype != expected_dtype:
            raise RuntimeError(
                f"prepared MTP input {name} must be {expected_dtype} {expected_shape}, "
                f"got {value.dtype} {tuple(value.shape)}"
            )
    return prepared


def validate_mtp_input_static_contract() -> None:
    """No-device proof of the released MTP input geometry and tensor names."""

    expected_specs = {
        "mtp.pre_fc_norm_embedding.weight": ("BF16", (2560,)),
        "mtp.pre_fc_norm_hidden.weight": ("BF16", (10240,)),
        "mtp.fc_embedding.weight": ("BF16", (2560, 2560)),
        "mtp.fc_hidden.weight": ("BF16", (2560, 2560)),
    }
    if dict(MTP_INPUT_TENSOR_SPECS) != expected_specs:
        raise RuntimeError("MTP input checkpoint tensor contract drifted")
    if (
        TP_SIZE,
        HIDDEN_SIZE,
        LOCAL_HIDDEN_SIZE,
        RESIDUAL_BRANCHES,
        RESIDUAL_WIDTH,
        RMS_NORM_EPS,
        EMBEDDING_LOCAL_SHAPE,
        RESIDUAL_LOCAL_SHAPE,
        HIDDEN_NORM_SCALE_LOCAL_SHAPE,
        PROJECTION_LOCAL_SHAPE,
    ) != (
        4,
        2560,
        640,
        4,
        10240,
        1.0e-6,
        (1, 1, 1, 640),
        (1, 4, 1, 640),
        (1, 1, 4, 640),
        (1, 1, 2560, 640),
    ):
        raise RuntimeError("MTP input TP4 geometry drifted from the pinned target")


@dataclass
class Qwen38TTNNMTPInputWeights:
    """Resident output-hidden-sharded MTP input weights owned by one mixer."""

    embedding_norm_scale: Any | None
    hidden_norm_scale: Any | None
    fc_embedding: Any | None
    fc_hidden: Any | None
    epsilon: float
    cache_directory: Path
    projection_dtype: Any = ttnn.bfloat16  # fc_embedding / fc_hidden dtype (QWEN38_DENSE_WEIGHT_DTYPE)
    _released: bool = field(default=False, init=False, repr=False)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Qwen38Checkpoint,
        placement: Qwen38Placement,
        mesh_device,
        mesh_contract: Qwen38MeshContract,
        cache_root: str | Path,
        *,
        tt_metal_sha: str,
        ttnn_runtime_sha256: str,
        projection_dtype=None,
    ) -> "Qwen38TTNNMTPInputWeights":
        mesh_contract.validate_mesh(mesh_device)
        if projection_dtype is None:
            projection_dtype = ttnn.bfloat16  # the production path
        projection_tag = dense_dtype_tag(projection_dtype)
        _validate_checkpoint_contract(checkpoint, placement, mesh_contract)
        prepared = _prepare_host_weights({name: checkpoint.tensor(name) for name in MTP_INPUT_TENSOR_SPECS})
        cache = _cache_directory(
            cache_root,
            checkpoint,
            mesh_contract,
            tt_metal_sha=tt_metal_sha,
            ttnn_runtime_sha256=ttnn_runtime_sha256,
        )
        mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 3))
        uploads: list[Any | None] = []

        def upload(value: torch.Tensor, name: str, dtype):
            tensor = ttnn.as_tensor(
                value.contiguous(),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
                cache_file_name=cache / name,
            )
            uploads.append(tensor)
            return tensor

        try:
            result = cls(
                embedding_norm_scale=upload(
                    prepared["embedding_norm_scale"], "embedding-norm-scale.fp32", ttnn.float32
                ),
                hidden_norm_scale=upload(prepared["hidden_norm_scale"], "hidden-norm-scale.fp32", ttnn.float32),
                fc_embedding=upload(prepared["fc_embedding"], f"fc-embedding.{projection_tag}", projection_dtype),
                fc_hidden=upload(prepared["fc_hidden"], f"fc-hidden.{projection_tag}", projection_dtype),
                epsilon=RMS_NORM_EPS,
                cache_directory=cache,
                projection_dtype=projection_dtype,
            )
            result.validate(mesh_contract)
        except BaseException as error:
            _cleanup_after_failure(uploads, label="partially uploaded MTP input weights", primary=error)
        return result

    @property
    def released(self) -> bool:
        return self._released

    def validate(self, mesh_contract: Qwen38MeshContract) -> None:
        if self._released:
            raise RuntimeError("MTP input weights were already deallocated")
        expected = {
            "embedding_norm_scale": (EMBEDDING_LOCAL_SHAPE, ttnn.float32),
            "hidden_norm_scale": (HIDDEN_NORM_SCALE_LOCAL_SHAPE, ttnn.float32),
            "fc_embedding": (PROJECTION_LOCAL_SHAPE, self.projection_dtype),
            "fc_hidden": (PROJECTION_LOCAL_SHAPE, self.projection_dtype),
        }
        for name, (shape, dtype) in expected.items():
            tensor = getattr(self, name)
            if tensor is None:
                raise RuntimeError(f"MTP input {name} is not resident")
            if _shape(tensor) != shape or tensor.dtype != dtype or tensor.layout != ttnn.TILE_LAYOUT:
                raise RuntimeError(
                    f"MTP input {name} must be TILE {dtype} {shape}, "
                    f"got {tensor.layout} {tensor.dtype} {_shape(tensor)}"
                )
            mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        if float(self.epsilon) != RMS_NORM_EPS:
            raise RuntimeError(f"MTP input RMS epsilon must be {RMS_NORM_EPS}, got {self.epsilon}")

    def deallocate(self) -> None:
        if self._released:
            raise RuntimeError("MTP input weights are already deallocated")
        slots = [self.embedding_norm_scale, self.hidden_norm_scale, self.fc_embedding, self.fc_hidden]
        try:
            _release_slots(slots, label="MTP input weights")
        finally:
            self.embedding_norm_scale, self.hidden_norm_scale, self.fc_embedding, self.fc_hidden = slots
            self._released = all(tensor is None for tensor in slots)


class Qwen38TTNNMTPInput:
    """One exact global-B1 TP4 MTP embedding/residual input fusion.

    The mixer has no recurrent state of its own.  A four-step controller feeds
    the target residual on the first call and each preceding MTP layer residual
    on the next call.  This owner retains the resident weights and releases
    them through :meth:`deallocate`.
    """

    def __init__(
        self,
        mesh_device,
        mesh_contract: Qwen38MeshContract,
        weights: Qwen38TTNNMTPInputWeights,
        *,
        collective_topology=None,
    ) -> None:
        mesh_contract.validate_mesh(mesh_device)
        weights.validate(mesh_contract)
        self.mesh_device = mesh_device
        self.mesh_contract = mesh_contract
        self.weights = weights
        self.collective_topology = collective_topology or ttnn.Topology.Linear
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # The fc linears run the fidelity of their weight format (decode_matmul: HiFi4 for bf16, HiFi2 for bf8, LoFi
        # for bf4); the norms keep compute_config.
        self.projection_compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, dense_math_fidelity_name(weights.projection_dtype)),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def deallocate(self) -> None:
        """Release the four mixer-owned resident tensors."""

        self.weights.deallocate()

    def _validate_input(self, tensor, *, label: str, shape: tuple[int, ...]) -> None:
        if _shape(tensor) != shape or tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.TILE_LAYOUT:
            raise ValueError(
                f"MTP {label} must be TILE BFLOAT16 with local shape {shape}, "
                f"got {tensor.layout} {tensor.dtype} {_shape(tensor)}"
            )
        self.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)

    def _normalize(
        self,
        tensor,
        weight,
        *,
        label: str,
        local_shape: tuple[int, ...],
        flatten_residual_branches: bool = False,
    ):
        # The embedding norm is H-wide.  In contrast, Qwen4Exp constructs
        # ``pre_fc_norm_hidden`` with width ``hc_count * hidden_size`` and
        # applies it *before* viewing the result as [4,H].  Preserve that
        # 10,240-wide contract by flattening each coordinate's four H/4
        # slices to one 2,560-wide local shard.  The TP all-gather of RMS
        # statistics then covers all four branches on all four devices.  A
        # per-branch RMS here is shape-compatible but numerically wrong.
        operation_tensor = tensor
        operation_weight = weight
        operation_shape = local_shape
        if flatten_residual_branches:
            if local_shape != RESIDUAL_LOCAL_SHAPE:
                raise ValueError("only the four-branch hidden residual may use the flattened MTP RMS path")
            operation_shape = (1, 1, 1, RESIDUAL_BRANCHES * LOCAL_HIDDEN_SIZE)
            operation_tensor = ttnn.reshape(tensor, operation_shape)
            operation_weight = ttnn.reshape(weight, operation_shape)
            self.mesh_contract.validate_tensor(
                operation_tensor,
                placement=TensorPlacement.HIDDEN_SHARDED,
                shard_dim=3,
            )
            self.mesh_contract.validate_tensor(
                operation_weight,
                placement=TensorPlacement.HIDDEN_SHARDED,
                shard_dim=3,
            )
            if _shape(operation_tensor) != operation_shape or _shape(operation_weight) != operation_shape:
                raise RuntimeError(
                    "flattened MTP hidden RMS inputs must both have local shape "
                    f"{operation_shape}, got activation={_shape(operation_tensor)} weight={_shape(operation_weight)}"
                )

        stats = None
        gathered = None
        normalized = None
        try:
            stats = ttnn.rms_norm_pre_all_gather(
                operation_tensor,
                dtype=ttnn.float32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_config,
            )
            stats_shape = (*operation_shape[:-1], RMS_STATS_WIDTH_PER_DEVICE)
            stats = ttnn.reshape(stats, stats_shape)
            self.mesh_contract.validate_tensor(stats, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
            if _shape(stats) != stats_shape:
                raise RuntimeError(
                    f"MTP {label} local RMS statistics have shape {_shape(stats)}, expected {stats_shape}"
                )
            gathered = ttnn.all_gather(
                stats,
                dim=3,
                cluster_axis=TP_AXIS,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            gathered_shape = (*operation_shape[:-1], RMS_STATS_GLOBAL_WIDTH)
            self.mesh_contract.validate_tensor(gathered, placement=TensorPlacement.REPLICATED)
            if _shape(gathered) != gathered_shape:
                raise RuntimeError(
                    f"MTP {label} gathered RMS statistics have shape {_shape(gathered)}, expected {gathered_shape}"
                )
            normalized = ttnn.rms_norm_post_all_gather(
                operation_tensor,
                gathered,
                epsilon=self.weights.epsilon,
                weight=operation_weight,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_config,
                dtype=ttnn.bfloat16,
            )
            if flatten_residual_branches:
                normalized = ttnn.reshape(normalized, local_shape)
            self.mesh_contract.validate_tensor(
                normalized,
                placement=TensorPlacement.HIDDEN_SHARDED,
                shard_dim=3,
            )
            if _shape(normalized) != local_shape or normalized.dtype != ttnn.bfloat16:
                raise RuntimeError(
                    f"MTP {label} normalization must produce BF16 {local_shape}, "
                    f"got {normalized.dtype} {_shape(normalized)}"
                )
        except BaseException as error:
            _cleanup_after_failure(
                [stats, gathered, normalized],
                label=f"failed MTP {label} normalization",
                primary=error,
            )
        cleanup = [stats, gathered]
        try:
            _release_slots(cleanup, label=f"MTP {label} normalization temporaries")
        except BaseException as error:
            _cleanup_after_failure([normalized], label=f"MTP {label} normalization result", primary=error)
        return normalized

    def _all_gather(self, tensor, *, label: str, global_shape: tuple[int, ...]):
        gathered = None
        try:
            gathered = ttnn.all_gather(
                tensor,
                dim=3,
                cluster_axis=TP_AXIS,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.mesh_contract.validate_tensor(gathered, placement=TensorPlacement.REPLICATED)
            if _shape(gathered) != global_shape:
                raise RuntimeError(f"MTP {label} all-gather produced {_shape(gathered)}, expected {global_shape}")
        except BaseException as error:
            _cleanup_after_failure([gathered], label=f"invalid MTP {label} all-gather", primary=error)
        return gathered

    def __call__(self, input_embedding, hidden_residual):
        if self.weights.released:
            raise RuntimeError("cannot run MTP input fusion after its weights were deallocated")
        self._validate_input(input_embedding, label="embedding", shape=EMBEDDING_LOCAL_SHAPE)
        self._validate_input(hidden_residual, label="hidden residual", shape=RESIDUAL_LOCAL_SHAPE)

        temporaries: list[Any | None] = []
        output = None
        try:
            normalized_embedding = self._normalize(
                input_embedding,
                self.weights.embedding_norm_scale,
                label="embedding",
                local_shape=EMBEDDING_LOCAL_SHAPE,
            )
            temporaries.append(normalized_embedding)
            normalized_hidden = self._normalize(
                hidden_residual,
                self.weights.hidden_norm_scale,
                label="hidden residual",
                local_shape=RESIDUAL_LOCAL_SHAPE,
                flatten_residual_branches=True,
            )
            temporaries.append(normalized_hidden)

            full_embedding = self._all_gather(
                normalized_embedding,
                label="embedding",
                global_shape=EMBEDDING_GLOBAL_SHAPE,
            )
            temporaries.append(full_embedding)
            full_hidden = self._all_gather(
                normalized_hidden,
                label="hidden residual",
                global_shape=RESIDUAL_GLOBAL_SHAPE,
            )
            temporaries.append(full_hidden)

            projected_embedding = ttnn.linear(
                full_embedding,
                self.weights.fc_embedding,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.projection_compute_config,
            )
            temporaries.append(projected_embedding)
            projected_hidden = ttnn.linear(
                full_hidden,
                self.weights.fc_hidden,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.projection_compute_config,
            )
            temporaries.append(projected_hidden)
            self._validate_input(projected_embedding, label="projected embedding", shape=EMBEDDING_LOCAL_SHAPE)
            self._validate_input(projected_hidden, label="projected hidden", shape=RESIDUAL_LOCAL_SHAPE)

            broadcast_embedding = ttnn.repeat(
                projected_embedding,
                (1, RESIDUAL_BRANCHES, 1, 1),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            temporaries.append(broadcast_embedding)
            self._validate_input(
                broadcast_embedding,
                label="broadcast projected embedding",
                shape=RESIDUAL_LOCAL_SHAPE,
            )
            output = ttnn.add(projected_hidden, broadcast_embedding, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            self._validate_input(output, label="fused output", shape=RESIDUAL_LOCAL_SHAPE)
            if _tensor_key(output) in {_tensor_key(tensor) for tensor in temporaries if tensor is not None}:
                raise RuntimeError("MTP fusion output unexpectedly aliases a temporary tensor")
        except BaseException as error:
            if output is not None:
                temporaries.append(output)
            _cleanup_after_failure(temporaries, label="failed MTP input fusion", primary=error)

        try:
            _release_slots(temporaries, label="MTP input fusion temporaries")
        except BaseException as error:
            _cleanup_after_failure([output], label="MTP input fusion output", primary=error)
        return output

    def rows(self, input_embedding_rows, hidden_residual_rows):
        """:meth:`__call__` for a chunk of token rows: 32 (the MTP alignment rows of an MTP v2 verify pass, the MTP
        rows of a 32-row prefill chunk) or 128 (the MTP rows of a 128-row prefill chunk).

        ``input_embedding_rows`` ``[1,1,rows,640]`` and the branch-major ``hidden_residual_rows`` ``[1,4,rows,640]``
        give the fused ``[1,4,rows,640]`` rows; the row count is the inputs' (one of :data:`CHUNK_ROW_COUNTS`).  The
        hidden norm keeps its 10,240-wide per-token contract: the rows go token-major and flat (``[1,1,rows,2560]``
        local, the GR rows walk), are normalized with the same flattened scale, and go back to branch-major for the
        per-branch projection.  Row j depends only on row j: the norms are per row, and at 128 rows the two
        projections run once per 32-row tile of the gathered rows (:meth:`_project_rows`), so every row's result is
        the 32-row form's bitwise.
        """

        if self.weights.released:
            raise RuntimeError("cannot run MTP input fusion after its weights were deallocated")
        rows = _shape(input_embedding_rows)[2]
        if rows not in CHUNK_ROW_COUNTS:
            raise ValueError(
                f"MTP input rows take {CHUNK_ROW_COUNTS} token rows, got embedding rows {_shape(input_embedding_rows)}"
            )
        embedding_shape = (1, 1, rows, LOCAL_HIDDEN_SIZE)
        residual_shape = (1, RESIDUAL_BRANCHES, rows, LOCAL_HIDDEN_SIZE)
        flat_shape = (1, 1, rows, RESIDUAL_BRANCHES * LOCAL_HIDDEN_SIZE)
        self._validate_input(input_embedding_rows, label="embedding rows", shape=embedding_shape)
        self._validate_input(hidden_residual_rows, label="hidden residual rows", shape=residual_shape)
        dram = ttnn.DRAM_MEMORY_CONFIG

        normalized_embedding = self._normalize(
            input_embedding_rows, self.weights.embedding_norm_scale, label="embedding rows", local_shape=embedding_shape
        )
        token_rows = ttnn.permute(hidden_residual_rows, (0, 2, 1, 3), memory_config=dram)
        flat_rows = ttnn.reshape(token_rows, flat_shape)
        if not same_buffer(flat_rows, token_rows):  # a copy today (the last dim changes); never free a view's source
            ttnn.deallocate(token_rows)
        self.mesh_contract.validate_tensor(flat_rows, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        # The flattened scale broadcasts over the token rows (one [1,1,1,2560] row).
        normalized_flat = self._normalize(
            flat_rows,
            ttnn.reshape(self.weights.hidden_norm_scale, (1, 1, 1, RESIDUAL_BRANCHES * LOCAL_HIDDEN_SIZE)),
            label="hidden residual rows",
            local_shape=flat_shape,
        )
        ttnn.deallocate(flat_rows)
        normalized_tokens = ttnn.reshape(normalized_flat, (1, rows, RESIDUAL_BRANCHES, LOCAL_HIDDEN_SIZE))
        if not same_buffer(normalized_tokens, normalized_flat):  # a copy today (the last dim changes)
            ttnn.deallocate(normalized_flat)
        normalized_hidden = ttnn.permute(normalized_tokens, (0, 2, 1, 3), memory_config=dram)
        ttnn.deallocate(normalized_tokens)
        self._validate_input(normalized_hidden, label="normalized hidden rows", shape=residual_shape)

        full_embedding = self._all_gather(
            normalized_embedding, label="embedding rows", global_shape=(1, 1, rows, HIDDEN_SIZE)
        )
        full_hidden = self._all_gather(
            normalized_hidden, label="hidden residual rows", global_shape=(1, RESIDUAL_BRANCHES, rows, HIDDEN_SIZE)
        )
        ttnn.deallocate(normalized_embedding)
        ttnn.deallocate(normalized_hidden)
        projected_embedding = self._project_rows(full_embedding, self.weights.fc_embedding, rows=rows)
        projected_hidden = self._project_rows(full_hidden, self.weights.fc_hidden, rows=rows)
        ttnn.deallocate(full_embedding)
        ttnn.deallocate(full_hidden)
        self._validate_input(projected_embedding, label="projected embedding rows", shape=embedding_shape)
        self._validate_input(projected_hidden, label="projected hidden rows", shape=residual_shape)
        broadcast_embedding = ttnn.repeat(projected_embedding, (1, RESIDUAL_BRANCHES, 1, 1), memory_config=dram)
        ttnn.deallocate(projected_embedding)
        self._validate_input(broadcast_embedding, label="broadcast projected embedding rows", shape=residual_shape)
        output = ttnn.add(projected_hidden, broadcast_embedding, memory_config=dram)
        ttnn.deallocate(projected_hidden)
        ttnn.deallocate(broadcast_embedding)
        self._validate_input(output, label="fused output rows", shape=residual_shape)
        return output

    def _project_rows(self, gathered, weight, *, rows: int):
        """``ttnn.linear`` of the gathered rows ``[1,B,rows,2560]`` by ``weight`` in the 32-row form's program: one
        call at 32 rows; at 128 rows one call per 32-row tile (the tiles sliced along the rows, projected with the
        32-row call's shapes, the outputs concatenated).  A 128-row ``ttnn.linear`` would pick its own matmul program,
        and another K block order rounds the sums differently; per tile every row keeps the 32-row program and its
        result."""

        dram = ttnn.DRAM_MEMORY_CONFIG

        def project(tile):
            return ttnn.linear(
                tile,
                weight,
                memory_config=dram,
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.projection_compute_config,
            )

        if rows == ttnn.TILE_SIZE:
            return project(gathered)
        shape = _shape(gathered)
        projected = []
        for tile in range(rows // ttnn.TILE_SIZE):
            rows_tile = ttnn.slice(
                gathered,
                (0, 0, tile * ttnn.TILE_SIZE, 0),
                (shape[0], shape[1], (tile + 1) * ttnn.TILE_SIZE, shape[3]),
                memory_config=dram,
            )
            projected.append(project(rows_tile))
            ttnn.deallocate(rows_tile)
        output = ttnn.concat(projected, dim=2, memory_config=dram)
        for tile in projected:
            ttnn.deallocate(tile)
        return output


validate_mtp_input_static_contract()


__all__ = [
    "MTP_INPUT_TENSOR_SPECS",
    "Qwen38TTNNMTPInput",
    "Qwen38TTNNMTPInputCleanupError",
    "Qwen38TTNNMTPInputWeights",
    "validate_mtp_input_static_contract",
]
