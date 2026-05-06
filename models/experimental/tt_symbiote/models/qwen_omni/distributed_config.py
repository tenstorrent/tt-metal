# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Mesh/tensor distribution metadata (mappers, composers, logical shapes).

:class:`DistributedTensorConfig` / :class:`DistributedConfig` match legacy ``run_config`` defaults (replicated
fallback via ``create_mesh_composer``). Qwen3-Omni uses ``ConcatMeshToTensor(dim=0)`` plus optional dim-0 slice via
:func:`qwen_omni_maybe_slice_replicated_mesh_compose` (wired by :func:`ensure_qwen_omni_normalrun_to_torch_slice`
when :class:`QwenOmniDeviceInit` boots; :meth:`NormalRun.to_torch` in ``run_config`` stays vanilla).

Note: :class:`QwenOmniDeviceInit` lives in ``utils/device_management`` to avoid an import cycle with
``run_config`` (device management imports :class:`DispatchManager` from ``run_config``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch
import ttnn
from models.tt_transformers.tt.ccl import TT_CCL

logger = logging.getLogger(__name__)


def _tensor_shardable_for_default_mesh_config(mesh_device, tensor) -> bool:
    """True when ``tensor`` dims divide cleanly along mesh batch/channel axes (default 2-D shard layout)."""
    if tensor is None:
        return True
    if len(tensor.shape) < 2:
        return False
    ms = mesh_device.shape
    return tensor.shape[-1] % ms[-1] == 0 and tensor.shape[0] % ms[0] == 0


@dataclass
class CCLManagerConfig:
    """Configuration for CCLManager."""

    mesh_device: Any
    num_links: Optional[int] = None
    topology: Optional[Any] = None

    def __post_init__(self):
        if self.num_links is None:
            self.num_links = 1
        if self.topology is None:
            self.topology = ttnn.Topology.Linear


@dataclass
class DistributedTensorConfig:
    """Configuration for distributed tensor operations."""

    mesh_mapper: Any
    mesh_composer: Any
    logical_shape_fn: Optional[Any] = None

    def get_logical_shape(self, sharded_shape):
        if self.logical_shape_fn is not None:
            return self.logical_shape_fn(sharded_shape)
        return sharded_shape


def distributed_config_col_sharded_last_dim(mesh_device) -> DistributedTensorConfig:
    """Build metadata for last-dim column-sharded activations on ``mesh_device``."""

    def logical_shape_for_col_sharded(sharded_shape):
        shape_list = list(sharded_shape)
        n = int(mesh_device.get_num_devices())
        shape_list[-1] = int(shape_list[-1]) * int(n)
        return tuple(shape_list)

    return DistributedTensorConfig(
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1),
        logical_shape_fn=logical_shape_for_col_sharded,
    )


def logical_shape_for_batch_channel_sharding(mesh_shape):
    """Logical shape for 2-D mesh sharding on batch (dim 0) and last dim — legacy ``run_config`` helper."""

    def _logical_shape(shape):
        shape = list(shape)
        logical_shape = [shape[0] * mesh_shape[0]] + shape[1:-1] + [shape[-1] * mesh_shape[1]]
        return tuple(logical_shape)

    return _logical_shape


@dataclass
class DistributedConfig:
    """Configuration for distributed operations (legacy ``run_config`` semantics)."""

    mesh_device: Any
    tensor_config: Optional[DistributedTensorConfig] = None
    ccl_manager: Optional[Any] = None

    def __post_init__(self):
        if self.tensor_config is None and self.mesh_device.get_num_devices() > 1:
            self.tensor_config = DistributedTensorConfig(
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, self.mesh_device.shape, (0, -1)),
                mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, self.mesh_device.shape, (0, -1)),
                logical_shape_fn=logical_shape_for_batch_channel_sharding(self.mesh_device.shape),
            )
        if self.ccl_manager is None and self.mesh_device.get_num_devices() > 1:
            self.ccl_manager = TT_CCL(self.mesh_device)

    def get_tensor_config_for_tensor(self, module_name, tensor):
        if tensor is not None:
            if (
                len(tensor.shape) < 2
                or tensor.shape[-1] % self.mesh_device.shape[-1] != 0
                or tensor.shape[0] % self.mesh_device.shape[0] != 0
            ):
                print(
                    f"Could not determine tensor config for {module_name} with shape {tensor.shape}. Assuming replication to all devices. Override set_output_tensors_config_impl in the module to set the correct config for this tensor."
                )
                return DistributedTensorConfig(
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                    mesh_composer=ttnn.create_mesh_composer(
                        self.mesh_device,
                        ttnn.MeshComposerConfig([0, len(tensor.shape)]),
                    ),
                )
        return self.tensor_config


def qwen_omni_maybe_slice_replicated_mesh_compose(cfg, ttnn_tensor, result: torch.Tensor) -> torch.Tensor:
    """After mesh ``to_torch``, slice stacked replicas on dim 0 when ``cfg`` opts in (Qwen3-Omni).

    Used only from the Qwen patch installed by :func:`ensure_qwen_omni_normalrun_to_torch_slice` (not from
    ``run_config`` directly).
    """
    if getattr(cfg, "replicate_compose_slice_dim0_to_leading", False) and result.dim() >= 1:
        lead = int(ttnn_tensor.shape[0])
        if result.shape[0] > lead:
            return result[:lead].contiguous()
    return result


_QWEN_OMNI_NORMALRUN_TO_TORCH_PATCHED = False


def ensure_qwen_omni_normalrun_to_torch_slice() -> None:
    """Idempotently patch :meth:`NormalRun.to_torch` to apply Omni dim-0 slice after ``ttnn.to_torch`` when configured.

    Call from :meth:`QwenOmniDeviceInit.init_state_impl` so generic ``run_config`` stays unchanged; only Omni
    device init activates the patched staticmethod.
    """
    global _QWEN_OMNI_NORMALRUN_TO_TORCH_PATCHED
    if _QWEN_OMNI_NORMALRUN_TO_TORCH_PATCHED:
        return
    from models.experimental.tt_symbiote.core.run_config import NormalRun

    def to_torch(self):
        """Convert to PyTorch tensor."""
        if self.elem is not None and self.elem.device.type != "meta" and self.ttnn_tensor is None:
            return self.elem

        def _to_torch(self_inner):
            is_mesh_device = self_inner.ttnn_distributed_tensor_config is not None
            if is_mesh_device:
                result = ttnn.to_torch(
                    self_inner.ttnn_tensor,
                    mesh_composer=self_inner.ttnn_distributed_tensor_config.mesh_composer,
                ).to(self_inner.device, self_inner.dtype)
                result = qwen_omni_maybe_slice_replicated_mesh_compose(
                    self_inner.ttnn_distributed_tensor_config, self_inner.ttnn_tensor, result
                )
            else:
                result = ttnn.to_torch(self_inner.ttnn_tensor).to(self_inner.device, self_inner.dtype)
            return result

        result = self.elem
        if self.ttnn_tensor is not None and self.elem is None:
            result = _to_torch(self)
        assert result is not None, "Both ttnn_tensor and elem are None. This should not happen."
        if result.device.type == "meta" and self.ttnn_tensor is not None:
            result = _to_torch(self)
        self.elem = result if self.elem is None else self.elem
        return self.elem

    NormalRun.to_torch = staticmethod(to_torch)
    _QWEN_OMNI_NORMALRUN_TO_TORCH_PATCHED = True


# --- Qwen3-Omni-MoE (concat dim 0 + optional slice via qwen_omni_maybe_slice_replicated_mesh_compose) ---


@dataclass
class QwenOmniReplicatedMeshTensorConfig(DistributedTensorConfig):
    """Replicated readback via ``ConcatMeshToTensor(dim=0)``; sets ``replicate_compose_slice_dim0_to_leading``."""

    replicate_compose_slice_dim0_to_leading: bool = True


def qwen_omni_replicated_concat_dim0_tensor_config(mesh_device) -> Optional[QwenOmniReplicatedMeshTensorConfig]:
    """Replicate + concat on dim 0, then slice to one logical batch row when Omni ``to_torch`` patch is active."""
    if mesh_device is None or mesh_device.get_num_devices() <= 1:
        return None
    return QwenOmniReplicatedMeshTensorConfig(
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )


@dataclass
class QwenOmniDistributedConfig(DistributedConfig):
    """Qwen3-Omni: ambiguous replicated tensors use ``ConcatMeshToTensor(dim=0)`` + dim-0 slice on readback."""

    def get_tensor_config_for_tensor(self, module_name, tensor):
        if tensor is not None and not _tensor_shardable_for_default_mesh_config(self.mesh_device, tensor):
            logger.warning(
                "Could not determine tensor config for %s with shape %s. Assuming replication to all devices. "
                "Override set_output_tensors_config_impl in the module to set the correct config for this tensor.",
                module_name,
                getattr(tensor, "shape", tensor),
            )
            return QwenOmniReplicatedMeshTensorConfig(
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0),
            )
        return self.tensor_config


__all__ = [
    "CCLManagerConfig",
    "DistributedConfig",
    "DistributedTensorConfig",
    "QwenOmniDistributedConfig",
    "QwenOmniReplicatedMeshTensorConfig",
    "distributed_config_col_sharded_last_dim",
    "ensure_qwen_omni_normalrun_to_torch_slice",
    "logical_shape_for_batch_channel_sharding",
    "qwen_omni_maybe_slice_replicated_mesh_compose",
    "qwen_omni_replicated_concat_dim0_tensor_config",
]
