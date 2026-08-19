# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-safe contract tests for the Wormhole Galaxy RMSNorm2D module."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import ttnn
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.rmsnorm import rmsnorm_2d
from models.common.modules.rmsnorm.rmsnorm_2d import (
    RMSNorm2D,
    RMSNorm2DConfig,
    RMSNorm2DGeometry,
    RMSNorm2DResidualPolicy,
    _prefill_stats_shape,
    _resolve_2d_config,
)


def _mesh(shape=(8, 4), *, devices=32, arch=ttnn.device.Arch.WORMHOLE_B0):
    mesh = MagicMock(spec=ttnn.MeshDevice)
    mesh.shape = shape
    mesh.get_num_devices.return_value = devices
    mesh.arch.return_value = arch
    return mesh


def _weight(dim: int, mesh=None):
    return LazyWeight(source=torch.ones(dim), device=mesh)


def _context(mesh, mode):
    resources = SimpleNamespace(
        key=SimpleNamespace(operation="all_gather", cluster_axis=1, geometry="norm-stats", sequence_key=None),
        cluster_axis=1,
        topology=ttnn.Topology.Linear,
        num_links=2,
        persistent_output_buffers=(f"{mode}-stats-buffer",),
        intermediate_output_buffers=(),
    )
    return SimpleNamespace(
        mesh_device=mesh,
        mode=mode,
        worker_sub_device_id=f"{mode}-subdevice",
        resources=lambda name, *_selector: resources,
        next_semaphore_handles=lambda name, *_selector: (f"{mode}-{name}-semaphore",),
        next_barrier_semaphore_handle=lambda name, *_selector: f"{mode}-{name}-barrier",
    )


def _ccl(mesh):
    contexts = {mode: _context(mesh, mode) for mode in ("decode", "prefill")}
    return SimpleNamespace(mesh_device=mesh, context=lambda mode: contexts[mode])


@pytest.mark.parametrize("dim", [8192, 128], ids=["llama-final-norm", "qwen-qk-norm"])
def test_resolves_representative_norm_geometries(dim):
    mesh = _mesh()
    resolved = _resolve_2d_config(RMSNorm2DConfig(weight=_weight(dim, mesh), mesh_device=mesh, tt_ccl=_ccl(mesh)))

    assert resolved.cluster_shape == (8, 4)
    assert resolved.weight.source.shape == (1, 1, dim // 32, 32)
    assert resolved.decode_residual_memcfg is not None
    assert resolved.prefill_residual_memcfg is not None
    assert resolved.decode_output_memcfg is not None
    assert resolved.prefill_output_memcfg is not None
    assert resolved.is_resolved()
    expected_geometry = RMSNorm2DGeometry.HEAD_LOCAL if dim == 128 else RMSNorm2DGeometry.DISTRIBUTED
    assert resolved.geometry is expected_geometry


@pytest.mark.parametrize(
    "shape,devices,arch,error",
    [
        ((4, 8), 32, ttnn.device.Arch.WORMHOLE_B0, "WH Galaxy mesh"),
        ((8, 4), 31, ttnn.device.Arch.WORMHOLE_B0, "exactly 32 devices"),
        ((8, 4), 32, ttnn.device.Arch.BLACKHOLE, "requires Wormhole"),
    ],
)
def test_resolution_fails_closed_on_non_wh_galaxy(shape, devices, arch, error):
    mesh = _mesh(shape, devices=devices, arch=arch)
    with pytest.raises(AssertionError, match=error):
        _resolve_2d_config(RMSNorm2DConfig(weight=_weight(8192, mesh), mesh_device=mesh, tt_ccl=_ccl(mesh)))


def test_resolution_rejects_resources_from_another_mesh():
    mesh = _mesh()
    other = _mesh()
    context = SimpleNamespace(mesh_device=other)

    with pytest.raises(AssertionError, match="prefetch context"):
        _resolve_2d_config(
            RMSNorm2DConfig(
                weight=_weight(8192, mesh),
                mesh_device=mesh,
                tt_ccl=_ccl(mesh),
                decode_prefetch_context=context,
            )
        )


def test_distributed_resolution_fails_closed_without_ccl_resources():
    mesh = _mesh()
    with pytest.raises(TypeError, match="context"):
        _resolve_2d_config(
            RMSNorm2DConfig(
                weight=_weight(8192, mesh),
                mesh_device=mesh,
                tt_ccl=SimpleNamespace(mesh_device=mesh),
            )
        )


def test_fused_decode_returns_normalized_output_and_residual_sum(monkeypatch):
    mesh = _mesh()
    normalized = object()
    x = SimpleNamespace(shape=(1, 1, 32, 2048))
    residual = object()
    module = object.__new__(RMSNorm2D)
    module.config = _resolve_2d_config(
        RMSNorm2DConfig(
            weight=_weight(8192, mesh),
            mesh_device=mesh,
            tt_ccl=_ccl(mesh),
            residual_policy=RMSNorm2DResidualPolicy.FUSED_DECODE,
        )
    )
    module.weight = "weight"
    module.load_device_weights = lambda: None
    monkeypatch.setattr(rmsnorm_2d, "_load_input_device_tensor_2d", lambda value, *_args, **_kwargs: value)
    fused = MagicMock(return_value=normalized)
    monkeypatch.setattr(ttnn, "fused_rms_minimal", fused)

    result = module.decode_forward(x, residual=residual)

    assert result == (normalized, residual)
    assert fused.call_args.args[4] == "decode-all_gather-semaphore"
    assert fused.call_args.kwargs["residual_input_tensor"] is residual
    assert fused.call_args.kwargs["stats"] == "decode-stats-buffer"


def test_qwen_128_wide_norm_is_head_local_and_skips_collectives(monkeypatch):
    mesh = _mesh()
    resolved = _resolve_2d_config(RMSNorm2DConfig(weight=_weight(128, mesh), mesh_device=mesh))
    module = object.__new__(RMSNorm2D)
    module.config = resolved
    module.weight = "weight"
    module.load_device_weights = lambda: None
    monkeypatch.setattr(rmsnorm_2d, "_load_input_device_tensor_2d", lambda value, *_args, **_kwargs: value)
    local_norm = MagicMock(return_value="normalized")
    gather = MagicMock()
    monkeypatch.setattr(ttnn, "rms_norm", local_norm)
    monkeypatch.setattr(ttnn.experimental, "all_gather_async", gather)

    assert module.decode_forward("q") == "normalized"
    assert resolved.geometry is RMSNorm2DGeometry.HEAD_LOCAL
    local_norm.assert_called_once()
    gather.assert_not_called()


def test_chunked_prefill_stats_shape_preserves_all_token_axes():
    assert _prefill_stats_shape((1, 4, 1024, 2048)) == (1, 4, 1024, 32)
    assert _prefill_stats_shape((3, 1, 128, 2048)) == (3, 1, 128, 32)


def test_distributed_decode_consumes_resolved_gather_resources(monkeypatch):
    mesh = _mesh()
    resolved = _resolve_2d_config(RMSNorm2DConfig(weight=_weight(8192, mesh), mesh_device=mesh, tt_ccl=_ccl(mesh)))
    module = object.__new__(RMSNorm2D)
    module.config = resolved
    module.weight = "weight"
    source = MagicMock()
    distributed = MagicMock()
    stats = MagicMock()
    gathered = MagicMock()
    output = MagicMock()
    monkeypatch.setattr(ttnn, "to_memory_config", MagicMock(side_effect=[distributed, output]))
    monkeypatch.setattr(ttnn, "rms_norm_pre_all_gather", MagicMock(return_value=stats))
    gather = MagicMock(return_value=gathered)
    monkeypatch.setattr(ttnn.experimental, "all_gather_async", gather)
    monkeypatch.setattr(ttnn, "rms_norm_post_all_gather", MagicMock(return_value=output))

    assert module._decode_distributed(source) is output

    kwargs = gather.call_args.kwargs
    assert gather.call_args.args[1:5] == (3, 1, mesh, ttnn.Topology.Linear)
    assert kwargs["persistent_output_tensor"] == "decode-stats-buffer"
    assert kwargs["barrier_semaphore"] is None
    assert kwargs["subdevice_id"] == "decode-subdevice"
    stats.deallocate.assert_called_once_with(True)
    gathered.deallocate.assert_not_called()


def test_distributed_prefill_uses_resolved_standard_gather(monkeypatch):
    mesh = _mesh()
    resolved = _resolve_2d_config(RMSNorm2DConfig(weight=_weight(8192, mesh), mesh_device=mesh, tt_ccl=_ccl(mesh)))
    module = object.__new__(RMSNorm2D)
    module.config = resolved
    module.weight = "weight"
    module.load_device_weights = lambda: None
    source = SimpleNamespace(shape=(1, 1, 128, 2048))
    stats = MagicMock()
    reshaped_stats = MagicMock()
    gathered = MagicMock()
    output = MagicMock()
    monkeypatch.setattr(rmsnorm_2d, "_load_input_device_tensor_2d", lambda value, *_args, **_kwargs: value)
    monkeypatch.setattr(ttnn, "rms_norm_pre_all_gather", MagicMock(return_value=stats))
    monkeypatch.setattr(ttnn, "reshape", MagicMock(return_value=reshaped_stats))
    gather = MagicMock(return_value=gathered)
    monkeypatch.setattr(ttnn, "all_gather", gather)
    async_gather = MagicMock()
    monkeypatch.setattr(ttnn.experimental, "all_gather_async", async_gather)
    post_gather = MagicMock(return_value=output)
    monkeypatch.setattr(ttnn, "rms_norm_post_all_gather", post_gather)
    to_memory_config = MagicMock()
    monkeypatch.setattr(ttnn, "to_memory_config", to_memory_config)

    assert module.prefill_forward(source) is output

    assert gather.call_args.args == (reshaped_stats, 3)
    assert gather.call_args.kwargs == {
        "cluster_axis": 1,
        "topology": ttnn.Topology.Linear,
        "num_links": 2,
        "memory_config": ttnn.DRAM_MEMORY_CONFIG,
        "subdevice_id": "prefill-subdevice",
    }
    async_gather.assert_not_called()
    reshaped_stats.deallocate.assert_not_called()
    assert post_gather.call_args.kwargs["memory_config"] == resolved.prefill_output_memcfg
    to_memory_config.assert_not_called()


def test_from_model_args_is_not_part_of_contract():
    assert not hasattr(RMSNorm2D, "from_model_args")


def test_mode_bound_ccl_contexts_are_resolved_at_construction():
    mesh = _mesh()
    decode_context = _context(mesh, "decode")
    prefill_context = _context(mesh, "prefill")
    collaborator = SimpleNamespace(
        mesh_device=mesh,
        context=lambda mode: {"decode": decode_context, "prefill": prefill_context}[mode],
    )

    resolved = _resolve_2d_config(RMSNorm2DConfig(weight=_weight(8192, mesh), mesh_device=mesh, tt_ccl=collaborator))

    assert resolved.decode_ccl_context is decode_context
    assert resolved.prefill_ccl_context is prefill_context


def test_simple_constructor_accepts_explicit_galaxy_ccl():
    mesh = _mesh()
    module = RMSNorm2D(_weight(8192, mesh), tt_ccl=_ccl(mesh), mesh_device=mesh)

    assert module.config.geometry is RMSNorm2DGeometry.DISTRIBUTED


def test_forward_rejects_unknown_mode():
    module = object.__new__(RMSNorm2D)
    with pytest.raises(ValueError, match="mode must be"):
        module.forward(object(), mode="train")
