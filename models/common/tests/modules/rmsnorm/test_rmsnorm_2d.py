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


def _stats_buffer(core=(2, 0)):
    """Persistent stats buffer stand-in, width-sharded on a single core."""
    memory_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, 32, 128),
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*core), ttnn.CoreCoord(*core))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        use_height_and_width_as_shard_shape=True,
    )
    return SimpleNamespace(memory_config=lambda: memory_config)


def _context(mesh, mode, stats_buffer=None):
    resources = SimpleNamespace(
        key=SimpleNamespace(operation="all_gather", cluster_axis=1, geometry="norm-stats", sequence_key=None),
        cluster_axis=1,
        topology=ttnn.Topology.Linear,
        num_links=2,
        persistent_output_buffers=(stats_buffer if stats_buffer is not None else _stats_buffer(),),
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


def _ccl(mesh, decode_stats_buffer=None):
    contexts = {
        mode: _context(mesh, mode, decode_stats_buffer if mode == "decode" else None) for mode in ("decode", "prefill")
    }
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
def test_resolution_fails_closed_on_non_wh_galaxy(shape, devices, arch, error, expect_error):
    mesh = _mesh(shape, devices=devices, arch=arch)
    with expect_error(AssertionError, error):
        _resolve_2d_config(RMSNorm2DConfig(weight=_weight(8192, mesh), mesh_device=mesh, tt_ccl=_ccl(mesh)))


def test_resolution_rejects_resources_from_another_mesh(expect_error):
    mesh = _mesh()
    other = _mesh()
    context = SimpleNamespace(mesh_device=other)

    with expect_error(AssertionError, "prefetch context"):
        _resolve_2d_config(
            RMSNorm2DConfig(
                weight=_weight(8192, mesh),
                mesh_device=mesh,
                tt_ccl=_ccl(mesh),
                decode_prefetch_context=context,
            )
        )


def test_distributed_resolution_fails_closed_without_ccl_resources(expect_error):
    mesh = _mesh()
    with expect_error(TypeError, "context"):
        _resolve_2d_config(
            RMSNorm2DConfig(
                weight=_weight(8192, mesh),
                mesh_device=mesh,
                tt_ccl=SimpleNamespace(mesh_device=mesh),
            )
        )


def _fused_decode_module(mesh, ccl):
    module = object.__new__(RMSNorm2D)
    module.config = _resolve_2d_config(
        RMSNorm2DConfig(
            weight=_weight(8192, mesh),
            mesh_device=mesh,
            tt_ccl=ccl,
            residual_policy=RMSNorm2DResidualPolicy.FUSED_DECODE,
        )
    )
    module.weight = "weight"
    module.load_device_weights = lambda: None
    return module


def test_fused_decode_returns_normalized_output_and_residual_sum(monkeypatch):
    mesh = _mesh()
    normalized = object()
    x = SimpleNamespace(shape=(1, 1, 32, 2048))
    residual = object()
    stats_buffer = _stats_buffer()
    module = _fused_decode_module(mesh, _ccl(mesh, stats_buffer))
    monkeypatch.setattr(rmsnorm_2d, "_load_input_device_tensor_2d", lambda value, *_args, **_kwargs: value)
    fused = MagicMock(return_value=normalized)
    monkeypatch.setattr(ttnn, "fused_rms_minimal", fused)

    result = module.decode_forward(x, residual=residual)

    assert result == (normalized, residual)
    assert fused.call_args.args[4] == "decode-all_gather-semaphore"
    assert fused.call_args.kwargs["residual_input_tensor"] is residual
    assert fused.call_args.kwargs["stats"] is stats_buffer


def test_fused_decode_rejects_stats_buffer_off_the_norm_sender_core(monkeypatch, expect_error):
    """The fused stats circular buffer is bound to the first norm core's L1 address."""
    mesh = _mesh()
    module = _fused_decode_module(mesh, _ccl(mesh, _stats_buffer(core=(1, 0))))
    monkeypatch.setattr(rmsnorm_2d, "_load_input_device_tensor_2d", lambda value, *_args, **_kwargs: value)
    fused = MagicMock()
    monkeypatch.setattr(ttnn, "fused_rms_minimal", fused)

    with expect_error(ValueError, "first core of the norm input shard grid"):
        module.decode_forward(SimpleNamespace(shape=(1, 1, 32, 2048)), residual=object())
    fused.assert_not_called()


def test_head_local_decode_stays_interleaved():
    """A 128-wide head norm takes any row count, so it must not pin a 32-row L1 shard."""
    mesh = _mesh()
    resolved = _resolve_2d_config(RMSNorm2DConfig(weight=_weight(128, mesh), mesh_device=mesh))

    assert resolved.geometry is RMSNorm2DGeometry.HEAD_LOCAL
    assert resolved.decode_input_memcfg == ttnn.DRAM_MEMORY_CONFIG
    assert resolved.decode_residual_memcfg == ttnn.DRAM_MEMORY_CONFIG
    assert resolved.decode_output_memcfg == ttnn.DRAM_MEMORY_CONFIG


def test_distributed_decode_stats_share_the_norm_sender_core():
    mesh = _mesh()
    resolved = _resolve_2d_config(RMSNorm2DConfig(weight=_weight(8192, mesh), mesh_device=mesh, tt_ccl=_ccl(mesh)))

    stats_grid = resolved.decode_stats_memcfg.shard_spec.grid
    input_grid = resolved.decode_input_memcfg.shard_spec.grid
    assert stats_grid.num_cores() == 1
    assert stats_grid.bounding_box().start == input_grid.bounding_box().start


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
    stats_buffer = _stats_buffer()
    resolved = _resolve_2d_config(
        RMSNorm2DConfig(weight=_weight(8192, mesh), mesh_device=mesh, tt_ccl=_ccl(mesh, stats_buffer))
    )
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
    assert kwargs["persistent_output_tensor"] is stats_buffer
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


def test_forward_rejects_unknown_mode(expect_error):
    module = object.__new__(RMSNorm2D)
    with expect_error(ValueError, "mode must be"):
        module.forward(object(), mode="train")


def test_distributed_decode_does_not_deallocate_a_tensor_it_returns(monkeypatch):
    """The decode norm must not free its own output when no copy was made.

    ``ttnn.to_memory_config`` returns *the same* tt_metal tensor when the
    requested config already matches, and nanobind hands that back as a fresh
    Python wrapper - so a plain ``if placed is not source: source.deallocate()``
    cannot tell "no copy was made" from "a copy was made", and frees the buffer
    it is about to return. The next op then fails with "Tensor is not allocated".

    Both of ``_decode_distributed``'s placements are in that position: the norm's
    ``decode_input_memcfg`` is the residual placement the previous layer already
    produced, and ``rms_norm_post_all_gather`` already returns its output in
    ``decode_output_memcfg``. So on the real Galaxy decode path the short-circuit
    fires *every time*, for both of them.

    This went unnoticed for a whole milestone because the sibling test above
    mocks ``to_memory_config`` with ``side_effect=[distributed, output]``, which
    always hands back a distinct object - exactly the case where the identity
    test happens to be right. Here the tensors already carry the requested
    configs, which is what the hardware does.
    """

    mesh = _mesh()
    stats_buffer = _stats_buffer()
    resolved = _resolve_2d_config(
        RMSNorm2DConfig(weight=_weight(8192, mesh), mesh_device=mesh, tt_ccl=_ccl(mesh, stats_buffer))
    )
    module = object.__new__(RMSNorm2D)
    module.config = resolved
    module.weight = "weight"

    def placed(memory_config):
        tensor = MagicMock()
        tensor.memory_config = MagicMock(return_value=memory_config)
        return tensor

    source = placed(resolved.decode_input_memcfg)
    output = placed(resolved.decode_output_memcfg)
    stats = MagicMock()

    relocate = MagicMock(side_effect=AssertionError("to_memory_config called for a placement already satisfied"))
    monkeypatch.setattr(rmsnorm_2d, "_load_input_device_tensor_2d", lambda value, *_a, **_k: value)
    monkeypatch.setattr(ttnn, "to_memory_config", relocate)
    monkeypatch.setattr(ttnn, "rms_norm_pre_all_gather", MagicMock(return_value=stats))
    monkeypatch.setattr(ttnn.experimental, "all_gather_async", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(ttnn, "rms_norm_post_all_gather", MagicMock(return_value=output))

    assert module._decode_distributed(source) is output

    # Neither placement was re-issued, so neither identity test could misfire.
    relocate.assert_not_called()
    # And nothing that is still in use was released.
    output.deallocate.assert_not_called()
    source.deallocate.assert_not_called()


def _head_local_module(monkeypatch, mesh, *, compute_cores=None, output_memcfg=None):
    """Return a `RMSNorm2D` with a resolved head-local config and a stub weight."""

    monkeypatch.setattr(rmsnorm_2d, "resolve_lazy_weight", lambda weight, **_: weight)
    config = _resolve_2d_config(
        RMSNorm2DConfig(
            weight=_weight(128, mesh),
            mesh_device=mesh,
            cluster_shape=(8, 4),
            geometry=RMSNorm2DGeometry.HEAD_LOCAL,
            decode_output_memcfg=output_memcfg,
            decode_compute_cores=compute_cores,
        )
    )
    module = object.__new__(RMSNorm2D)
    module.config = config
    module.weight = object()
    module.load_device_weights = lambda: None
    return module


def _fake_tensor(memory_config, padded_shape=(1, 8, 32, 128)):
    tensor = MagicMock(spec=ttnn.Tensor)
    tensor.memory_config.return_value = memory_config
    tensor.padded_shape = padded_shape
    return tensor


def _norm_cores():
    """One core wide, eight tall, as the Galaxy Qwen model asks for."""

    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 7))})


def _memcfg(layout):
    memory_config = MagicMock(spec=ttnn.MemoryConfig)
    memory_config.memory_layout = layout
    memory_config.is_sharded.return_value = layout != ttnn.TensorMemoryLayout.INTERLEAVED
    return memory_config


def test_head_local_decode_without_a_compute_placement_is_one_rms_norm(monkeypatch):
    """The default behaviour is unchanged: one op, on the input as given.

    That is the right thing on an unpartitioned device, and it is what every
    caller that has already placed its input somewhere the op accepts wants.
    """

    mesh = _mesh()
    interleaved = _memcfg(ttnn.TensorMemoryLayout.INTERLEAVED)
    module = _head_local_module(monkeypatch, mesh, output_memcfg=interleaved)
    calls = []
    monkeypatch.setattr(rmsnorm_2d.ttnn, "rms_norm", lambda x, **kwargs: calls.append((x, kwargs)) or "normed")
    monkeypatch.setattr(
        rmsnorm_2d.ttnn,
        "sharded_to_interleaved",
        lambda *args, **kwargs: pytest.fail("no relocation without a compute placement"),
    )

    source = _fake_tensor(interleaved)
    assert module.decode_forward(source) == "normed"
    assert len(calls) == 1
    assert calls[0][0] is source
    assert calls[0][1]["memory_config"] is interleaved


def test_head_local_decode_runs_the_kernel_in_the_compute_placement(monkeypatch):
    """Height-sharded in, height-sharded out, the kernel block-sharded between.

    Measured on WH Galaxy `(8, 4)` as D-B26. Both obvious spellings abort:

    * interleaved input -> `LayerNormDefaultProgramConfig`, whose rows spread
      over the whole compute grid ->
      ``TT_FATAL: Kernel group cores do not match sub device cores``;
    * the created heads' own placement -> ``Height sharded inputs are not
      supported`` (`layernorm_device_operation.cpp:166`).

    So the kernel runs in a third, block-sharded placement, and the relocations
    at both ends use `sharded_to_interleaved` / `interleaved_to_sharded` rather
    than `to_memory_config`, which between two shard specs resolves to `reshard`
    and would abort exactly like the first case.
    """

    mesh = _mesh()
    heads = _memcfg(ttnn.TensorMemoryLayout.HEIGHT_SHARDED)
    compute = _memcfg(ttnn.TensorMemoryLayout.BLOCK_SHARDED)
    module = _head_local_module(monkeypatch, mesh, compute_cores=_norm_cores(), output_memcfg=heads)

    ops: list[str] = []
    staged_in, placed_in = _fake_tensor(_memcfg(ttnn.TensorMemoryLayout.INTERLEAVED)), _fake_tensor(compute)
    staged_out, placed_out = _fake_tensor(_memcfg(ttnn.TensorMemoryLayout.INTERLEAVED)), _fake_tensor(heads)
    normalized = _fake_tensor(compute)
    # 8 users x one tile of padded heads = 256 rows over 8 cores: one tile row
    # each. Sizing this from the full physical batch instead is what produced
    #   TT_FATAL: Shard layout requires 2x1 = 2 shards but shard grid has 8 cores
    derived = rmsnorm_2d._head_local_compute_memory_config(_fake_tensor(heads), _norm_cores())
    assert derived.shard_spec.shape == [32, 128]
    assert derived.memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED
    to_interleaved = iter((staged_in, staged_out))
    to_sharded = iter((placed_in, placed_out))
    monkeypatch.setattr(
        rmsnorm_2d.ttnn,
        "sharded_to_interleaved",
        lambda *args, **kwargs: ops.append("sharded_to_interleaved") or next(to_interleaved),
    )
    monkeypatch.setattr(
        rmsnorm_2d.ttnn,
        "interleaved_to_sharded",
        lambda *args, **kwargs: ops.append("interleaved_to_sharded") or next(to_sharded),
    )
    monkeypatch.setattr(
        rmsnorm_2d.ttnn, "to_memory_config", lambda *args, **kwargs: pytest.fail("to_memory_config is not partition-safe")
    )
    norm_calls = []
    monkeypatch.setattr(
        rmsnorm_2d.ttnn, "rms_norm", lambda x, **kwargs: norm_calls.append((x, kwargs)) or normalized
    )

    source = _fake_tensor(heads)
    assert module.decode_forward(source) is placed_out
    assert ops == [
        "sharded_to_interleaved",
        "interleaved_to_sharded",
        "sharded_to_interleaved",
        "interleaved_to_sharded",
    ]
    assert norm_calls[0][0] is placed_in
    # The kernel's own placement is derived from the tensor, not configured.
    assert norm_calls[0][1]["memory_config"] == derived
    # The caller owns its input; only the module's own intermediates are freed.
    source.deallocate.assert_not_called()
    for intermediate in (staged_in, placed_in, staged_out, normalized):
        intermediate.deallocate.assert_called_once_with(True)


def test_head_local_decode_refuses_a_residual_it_cannot_place(monkeypatch, expect_error):
    """`ttnn.rms_norm` demands the residual carry the input's shard spec exactly.

    Relocating it in step with `x` is possible but no caller needs it - a
    per-head Q/K norm has no residual - and guessing is worse than saying so.
    """

    mesh = _mesh()
    heads = _memcfg(ttnn.TensorMemoryLayout.HEIGHT_SHARDED)
    module = _head_local_module(monkeypatch, mesh, compute_cores=_norm_cores(), output_memcfg=heads)

    with expect_error(ValueError, "does not support a residual"):
        module.decode_forward(_fake_tensor(heads), residual=_fake_tensor(heads))


def test_head_local_compute_grid_must_be_a_one_wide_rectangle():
    """Both are the sharded layernorm's own requirements, not a style choice."""

    tensor = _fake_tensor(_memcfg(ttnn.TensorMemoryLayout.HEIGHT_SHARDED))
    wide = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(2, 3))})
    with pytest.raises(ValueError, match="one core wide"):  # allow-pytest.raises: module-level helper, no fixture
        rmsnorm_2d._head_local_compute_memory_config(tensor, wide)
    # 256 rows over 3 cores is not a whole number of tiles per core.
    ragged = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 2))})
    with pytest.raises(ValueError, match="do not divide"):  # allow-pytest.raises: module-level helper, no fixture
        rmsnorm_2d._head_local_compute_memory_config(tensor, ragged)
