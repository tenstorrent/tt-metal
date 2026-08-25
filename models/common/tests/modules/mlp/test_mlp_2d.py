# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-safe contract tests for the Wormhole Galaxy MLP2D module."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import ttnn
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.mlp import mlp_2d
from models.common.modules.mlp.mlp_2d import (
    MLP2D,
    MLP2DConfig,
    _prefetch_kwargs,
    _resolve_mlp2d_config,
    _select_collective_resources,
)


class _ShapeOnlyTensor:
    def __init__(self, shape):
        self.shape = shape


def _mesh(shape=(8, 4), *, devices=32, arch=ttnn.device.Arch.WORMHOLE_B0):
    mesh = MagicMock(spec=ttnn.MeshDevice)
    mesh.shape = shape
    mesh.get_num_devices.return_value = devices
    mesh.arch.return_value = arch
    return mesh


def _weights(dim, hidden_dim, mesh):
    return (
        LazyWeight(source=_ShapeOnlyTensor((dim, hidden_dim)), device=mesh),
        LazyWeight(source=_ShapeOnlyTensor((hidden_dim, dim)), device=mesh),
        LazyWeight(source=_ShapeOnlyTensor((dim, hidden_dim)), device=mesh),
    )


def _collective_resources(name):
    axis = 0 if name == "all_reduce" else 1
    return SimpleNamespace(
        key=SimpleNamespace(operation=name, cluster_axis=axis, geometry=f"{name}-geometry", sequence_key=None),
        cluster_axis=axis,
        topology=ttnn.Topology.Linear,
        num_links=2,
        persistent_output_buffers=(f"{name}-output",),
        intermediate_output_buffers=((f"{name}-intermediate",) if name == "reduce_scatter" else ()),
    )


def _context(mesh, mode):
    resources = {name: _collective_resources(name) for name in ("reduce_scatter", "all_gather", "all_reduce")}
    return SimpleNamespace(
        mesh_device=mesh,
        mode=mode,
        worker_sub_device_id=f"{mode}-subdevice",
        resources=lambda name, *_selector: resources[name],
        next_semaphore_handles=lambda name, *_selector: f"{mode}-{name}-semaphore",
        next_semaphore_window=lambda name, *_selector, count: [
            f"{mode}-{name}-semaphore-{index}" for index in range(count)
        ],
        next_barrier_semaphore_handle=lambda name, *_selector: f"{mode}-{name}-barrier",
    )


def _ccl(mesh):
    contexts = {mode: _context(mesh, mode) for mode in ("decode", "prefill")}
    return SimpleNamespace(mesh_device=mesh, context=lambda mode: contexts[mode])


def _config(dim=8192, hidden_dim=28672, **overrides):
    mesh = overrides.pop("mesh_device", _mesh())
    w1, w2, w3 = _weights(dim, hidden_dim, mesh)
    values = dict(
        w1=w1,
        w2=w2,
        w3=w3,
        mesh_device=mesh,
        tt_ccl=_ccl(mesh),
        topology=ttnn.Topology.Linear,
        dim=dim,
        hidden_dim=hidden_dim,
    )
    values.update(overrides)
    return MLP2DConfig(**values)


@pytest.fixture(autouse=True)
def _avoid_device_materialization(monkeypatch):
    monkeypatch.setattr(mlp_2d, "resolve_lazy_weight", lambda weight, **_kwargs: weight)


@pytest.mark.parametrize(
    "dim,hidden_dim",
    [(8192, 28672), (5120, 25600)],
    ids=["llama-3.3-70b", "qwen3-32b"],
)
def test_resolves_representative_model_geometry(dim, hidden_dim):
    resolved = _resolve_mlp2d_config(_config(dim, hidden_dim))

    assert resolved.dim == dim
    assert resolved.hidden_dim == hidden_dim
    assert resolved.decode_input_memcfg is not None
    assert resolved.prefill_input_memcfg is not None
    assert resolved.decode_w1_w3_output_memcfg is not None
    assert resolved.prefill_w1_w3_output_memcfg is not None
    assert callable(resolved.prefill_w1_w3_prg_config)
    assert callable(resolved.prefill_w2_prg_config)
    assert resolved.is_resolved()


@pytest.mark.parametrize(
    "shape,devices,arch,error",
    [
        ((4, 8), 32, ttnn.device.Arch.WORMHOLE_B0, "WH Galaxy mesh"),
        ((8, 4), 8, ttnn.device.Arch.WORMHOLE_B0, "exactly 32 devices"),
        ((8, 4), 32, ttnn.device.Arch.BLACKHOLE, "requires Wormhole"),
    ],
)
def test_resolution_fails_closed_on_non_wh_galaxy(shape, devices, arch, error):
    mesh = _mesh(shape, devices=devices, arch=arch)
    with pytest.raises(AssertionError, match=error):
        _resolve_mlp2d_config(_config(mesh_device=mesh))


def test_resolution_validates_projection_shapes():
    config = _config()
    config = replace(config, w2=LazyWeight(source=_ShapeOnlyTensor((8192, 28672)), device=config.mesh_device))
    with pytest.raises(AssertionError, match="w2 must have shape"):
        _resolve_mlp2d_config(config)


def test_resolution_rejects_foreign_ccl_and_prefetch_resources():
    mesh = _mesh()
    other_mesh = _mesh()

    with pytest.raises(AssertionError, match="CCL collaborator"):
        _resolve_mlp2d_config(_config(mesh_device=mesh, tt_ccl=SimpleNamespace(mesh_device=other_mesh)))

    with pytest.raises(AssertionError, match="prefetch context"):
        _resolve_mlp2d_config(
            _config(mesh_device=mesh, decode_prefetch_context=SimpleNamespace(mesh_device=other_mesh))
        )


def test_mode_specific_tunings_resolve_independently():
    decode_kernel = object()
    prefill_kernel = object()
    resolved = _resolve_mlp2d_config(
        _config(
            decode_activation_dtype=ttnn.bfloat16,
            prefill_activation_dtype=ttnn.bfloat8_b,
            decode_ccl_dtype=ttnn.bfloat16,
            prefill_ccl_dtype=ttnn.bfloat8_b,
            decode_ff1_3_compute_kernel_cfg=decode_kernel,
            prefill_ff1_3_compute_kernel_cfg=prefill_kernel,
        )
    )

    assert resolved.decode_activation_dtype == ttnn.bfloat16
    assert resolved.prefill_activation_dtype == ttnn.bfloat8_b
    assert resolved.decode_ccl_dtype == ttnn.bfloat16
    assert resolved.prefill_ccl_dtype == ttnn.bfloat8_b
    assert resolved.decode_ff1_3_compute_kernel_cfg is decode_kernel
    assert resolved.prefill_ff1_3_compute_kernel_cfg is prefill_kernel


def test_prefill_program_configs_are_sequence_keyed():
    w1_w3_factory = MagicMock(side_effect=lambda seq_len: f"w1-w3-{seq_len}")
    w2_factory = MagicMock(side_effect=lambda seq_len: f"w2-{seq_len}")
    resolved = _resolve_mlp2d_config(_config(prefill_w1_w3_prg_config=w1_w3_factory, prefill_w2_prg_config=w2_factory))

    assert resolved.prefill_w1_w3_prg_config(128) == "w1-w3-128"
    assert resolved.prefill_w1_w3_prg_config(2048) == "w1-w3-2048"
    assert resolved.prefill_w2_prg_config(128) == "w2-128"
    assert resolved.prefill_w2_prg_config(2048) == "w2-2048"


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"collective_resource_selector": "selector"}, "collective_resource_selector must be callable"),
        ({"mlp_activation_type": "silu"}, "mlp_activation_type must be a ttnn.UnaryOpType"),
        ({"prefill_w1_w3_prg_config": "program"}, "prefill_w1_w3_prg_config must be callable"),
        ({"prefill_w2_prg_config": "program"}, "prefill_w2_prg_config must be callable"),
    ],
)
def test_resolution_rejects_non_static_strategy_values(overrides, error):
    with pytest.raises(TypeError, match=error):
        _resolve_mlp2d_config(_config(**overrides))


def test_resolution_rejects_non_positive_prefill_cutoff():
    with pytest.raises(ValueError, match="prefill_len_cutoff must be positive"):
        _resolve_mlp2d_config(_config(prefill_len_cutoff=0))


@pytest.mark.parametrize("mode", ["decode", "prefill"])
@pytest.mark.parametrize(
    "collective,cluster_axis",
    [("reduce_scatter", 1), ("all_gather", 1), ("all_reduce", 0)],
)
def test_collective_selector_receives_mode_context_and_runtime_tensor(mode, collective, cluster_axis):
    mesh = _mesh()
    selector = MagicMock(return_value=_collective_resources(collective))
    resolved = _resolve_mlp2d_config(
        _config(mesh_device=mesh, tt_ccl=_ccl(mesh), collective_resource_selector=selector)
    )
    tensor = object()

    resources = _select_collective_resources(
        resolved,
        mode=mode,
        collective=collective,
        cluster_axis=cluster_axis,
        tensor=tensor,
    )

    assert resources is selector.return_value
    selector.assert_called_once_with(
        resolved.decode_ccl_context if mode == "decode" else resolved.prefill_ccl_context,
        collective,
        cluster_axis,
        tensor,
        None,
    )


def test_prefetch_context_maps_to_ttnn_collaborator_arguments():
    context = SimpleNamespace(global_cb="global-cb", worker_sub_device_id="worker-subdevice")
    assert _prefetch_kwargs(context) == {"global_cb": "global-cb", "sub_device_id": "worker-subdevice"}
    assert _prefetch_kwargs(None) == {}


def test_from_model_args_is_not_part_of_contract():
    assert not hasattr(MLP2D, "from_model_args")


def test_mode_bound_ccl_contexts_are_resolved_at_construction():
    mesh = _mesh()
    decode_context = _context(mesh, "decode")
    prefill_context = _context(mesh, "prefill")
    collaborator = SimpleNamespace(
        mesh_device=mesh,
        context=lambda mode: {"decode": decode_context, "prefill": prefill_context}[mode],
    )

    resolved = _resolve_mlp2d_config(_config(mesh_device=mesh, tt_ccl=collaborator))

    assert resolved.decode_ccl_context is decode_context
    assert resolved.prefill_ccl_context is prefill_context


def test_resolution_fails_closed_when_collective_resources_are_incomplete():
    mesh = _mesh()
    context = _context(mesh, "decode")
    context.resources = lambda name, *_selector: SimpleNamespace(
        key=SimpleNamespace(
            operation=name,
            cluster_axis=0 if name == "all_reduce" else 1,
            geometry=f"{name}-geometry",
            sequence_key=None,
        ),
        cluster_axis=0 if name == "all_reduce" else 1,
        topology=ttnn.Topology.Linear,
        num_links=1,
        persistent_output_buffers=(),
        intermediate_output_buffers=(),
    )
    collaborator = _ccl(mesh)
    collaborator.context = lambda mode: context if mode == "decode" else _context(mesh, mode)

    with pytest.raises(ValueError, match="requires persistent"):
        _resolve_mlp2d_config(_config(mesh_device=mesh, tt_ccl=collaborator))


def test_collectives_consume_resolved_buffers_and_subdevice(monkeypatch):
    mesh = _mesh()
    resolved = _resolve_mlp2d_config(_config(mesh_device=mesh, tt_ccl=_ccl(mesh)))
    module = object.__new__(MLP2D)
    module.config = resolved
    tensor = object()
    reduce_scatter = MagicMock(return_value="scattered")
    all_gather = MagicMock(return_value="gathered")
    monkeypatch.setattr(ttnn.experimental, "reduce_scatter_minimal_async", reduce_scatter)
    monkeypatch.setattr(ttnn.experimental, "all_gather_async", all_gather)

    assert module._reduce_scatter_axis1(tensor, "memory", "decode") == "scattered"
    assert module._all_gather_axis1(tensor, "memory", "decode") == "gathered"

    rs_kwargs = reduce_scatter.call_args.kwargs
    assert rs_kwargs["persistent_output_buffers"] == ["reduce_scatter-intermediate", "reduce_scatter-output"]
    assert rs_kwargs["subdevice_id"] == "decode-subdevice"
    ag_kwargs = all_gather.call_args.kwargs
    assert ag_kwargs["persistent_output_tensor"] == "all_gather-output"
    assert all_gather.call_args.args[1] == 3
    assert ag_kwargs["subdevice_id"] == "decode-subdevice"
    assert ag_kwargs["mesh_device"] is mesh
    assert ag_kwargs["multi_device_global_semaphore"] == [
        "decode-all_gather-semaphore-0",
        "decode-all_gather-semaphore-1",
    ]
    assert ag_kwargs["barrier_semaphore"] is None
    assert ag_kwargs["use_optimal_ccl_for_llama"] is True


def test_all_reduce_consumes_resolved_buffer_and_subdevice(monkeypatch):
    mesh = _mesh()
    resolved = _resolve_mlp2d_config(_config(mesh_device=mesh, tt_ccl=_ccl(mesh)))
    module = object.__new__(MLP2D)
    module.config = resolved
    tensor = SimpleNamespace(shape=(1, 1, 32, 128), dtype=resolved.decode_ccl_dtype)
    tensor.memory_config = lambda: "input-memory"
    reduced = SimpleNamespace(shape=tensor.shape)
    all_reduce = MagicMock(return_value=reduced)
    to_memory_config = MagicMock(side_effect=lambda value, *_args, **_kwargs: value)
    monkeypatch.setattr(ttnn.experimental, "all_reduce_async", all_reduce)
    monkeypatch.setattr(ttnn, "reshape", lambda value, _shape: value)
    monkeypatch.setattr(ttnn, "to_memory_config", to_memory_config)

    assert (
        module._all_reduce_tg(
            tensor,
            cluster_axis=0,
            dim=3,
            sharded=True,
            memory_config="output-memory",
            ccl_dtype=resolved.decode_ccl_dtype,
        )
        is reduced
    )

    assert all_reduce.call_args.args[1] == "all_reduce-output"
    assert all_reduce.call_args.kwargs["subdevice_id"] == "decode-subdevice"
    assert all_reduce.call_args.kwargs["use_optimal_ccl_for_llama"] is True
    to_memory_config.assert_not_called()


def test_prefill_all_reduce_uses_axis0_reduce_scatter_then_all_gather(monkeypatch):
    mesh = _mesh()
    resolved = _resolve_mlp2d_config(_config(mesh_device=mesh, tt_ccl=_ccl(mesh)))
    module = object.__new__(MLP2D)
    module.config = resolved
    tensor = SimpleNamespace(shape=(1, 1, 128, 2048), dtype=resolved.prefill_ccl_dtype)
    scattered = SimpleNamespace(shape=(1, 1, 128, 256))
    gathered = SimpleNamespace(shape=tensor.shape)
    module._reduce_scatter = MagicMock(return_value=scattered)
    module._all_gather = MagicMock(return_value=gathered)
    all_reduce = MagicMock()
    monkeypatch.setattr(ttnn.experimental, "all_reduce_async", all_reduce)
    monkeypatch.setattr(ttnn, "reshape", lambda value, _shape: value)
    monkeypatch.setattr(ttnn, "to_memory_config", lambda value, *_args, **_kwargs: value)

    result = module._all_reduce_tg(
        tensor,
        cluster_axis=0,
        dim=3,
        sharded=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ccl_dtype=resolved.prefill_ccl_dtype,
        mode="prefill",
    )

    assert result is gathered
    module._reduce_scatter.assert_called_once_with(
        tensor,
        ttnn.DRAM_MEMORY_CONFIG,
        "prefill",
        cluster_axis=0,
        sequence_key="final",
        persistent=False,
    )
    module._all_gather.assert_called_once_with(
        scattered,
        ttnn.DRAM_MEMORY_CONFIG,
        "prefill",
        cluster_axis=0,
        sequence_key="final",
        persistent=False,
    )
    all_reduce.assert_not_called()


def test_decode_fused_matmul_uses_static_resource_key_and_preserves_output_order(monkeypatch):
    class InputWithoutHostShape:
        @property
        def shape(self):
            raise AssertionError("decode resource selection must not inspect the stalled input tensor")

    input_tensor = InputWithoutHostShape()
    context = SimpleNamespace(
        worker_sub_device_id="decode-subdevice",
        next_semaphore_handles=MagicMock(return_value=("decode-semaphore",)),
    )
    resources = SimpleNamespace(
        key=SimpleNamespace(
            operation="reduce_scatter",
            cluster_axis=1,
            geometry="decode-geometry",
            sequence_key=None,
        ),
        cluster_axis=1,
        topology=ttnn.Topology.Ring,
        num_links=4,
        persistent_output_buffers=("persistent-output",),
        intermediate_output_buffers=("intermediate-output",),
    )
    selector = MagicMock(return_value=resources)
    module = object.__new__(MLP2D)
    module.config = SimpleNamespace(
        max_batch_size=32,
        hidden_dim=28672,
        collective_resource_selector=selector,
        decode_ccl_context=context,
        mesh_device="mesh",
        ff1_out_reduce_scatter_memcfg="rs-memory",
        decode_w1_w3_output_memcfg="matmul-memory",
        decode_ff1_3_compute_kernel_cfg="compute-kernel",
        decode_activation_dtype="activation-dtype",
        decode_w1_w3_prg_config="program-config",
        decode_prefetch_context=SimpleNamespace(global_cb="global-cb"),
    )
    module.w1, module.w3 = "w1", "w3"
    first_projection, w3_projection, w1_reduced = object(), object(), object()
    fused_matmul = MagicMock(return_value=(first_projection, w3_projection, w1_reduced))
    deallocate = MagicMock()
    monkeypatch.setattr(ttnn.experimental, "llama_rs_matmul", fused_matmul)
    monkeypatch.setattr(ttnn, "deallocate", deallocate)

    assert module._double_matmul_reduce_scatter_axis1(input_tensor) == (w1_reduced, w3_projection)

    selector.assert_called_once_with(
        context,
        "reduce_scatter",
        1,
        (1, 1, 32, 28672 // 8),
        None,
    )
    context.next_semaphore_handles.assert_called_once_with("reduce_scatter", 1, "decode-geometry", None)
    assert fused_matmul.call_args.args == (
        input_tensor,
        "w1",
        "intermediate-output",
        3,
        "decode-semaphore",
        1,
        "mesh",
        4,
        "decode-subdevice",
    )
    assert fused_matmul.call_args.kwargs["second_weight_tensor"] == "w3"
    assert fused_matmul.call_args.kwargs["global_cb"] == "global-cb"
    deallocate.assert_called_once_with(first_projection)


def test_decode_w3_uses_llama_reduce_scatter_padded_geometry(monkeypatch):
    context = SimpleNamespace(
        worker_sub_device_id="decode-subdevice",
        next_semaphore_handles=MagicMock(return_value=("decode-semaphore",)),
    )
    resources = SimpleNamespace(
        key=SimpleNamespace(
            operation="reduce_scatter",
            cluster_axis=1,
            geometry="decode-geometry",
            sequence_key=None,
        ),
        cluster_axis=1,
        topology=ttnn.Topology.Ring,
        num_links=4,
        persistent_output_buffers=("persistent-output",),
        intermediate_output_buffers=("intermediate-output",),
    )
    selector = MagicMock(return_value=resources)
    module = object.__new__(MLP2D)
    module.config = SimpleNamespace(
        max_batch_size=32,
        hidden_dim=28672,
        collective_resource_selector=selector,
        decode_ccl_context=context,
        mesh_device="mesh",
        ff1_out_reduce_scatter_memcfg="rs-memory",
    )
    reduce_scatter = MagicMock(return_value="w3-reduced")
    monkeypatch.setattr(ttnn.experimental, "llama_reduce_scatter", reduce_scatter)

    assert module._llama_reduce_scatter_axis1("w3-projection") == "w3-reduced"
    selector.assert_called_once_with(context, "reduce_scatter", 1, (1, 1, 32, 28672 // 8), None)
    assert reduce_scatter.call_args.args == (
        "w3-projection",
        "intermediate-output",
        3,
        "decode-semaphore",
        "decode-subdevice",
    )
    assert reduce_scatter.call_args.kwargs == {
        "cluster_axis": 1,
        "mesh_device": "mesh",
        "num_links": 4,
        "memory_config": "rs-memory",
        "topology": ttnn.Topology.Ring,
    }


def test_simple_constructor_requires_explicit_ccl_injection():
    mesh = _mesh()
    weights = _weights(8192, 28672, mesh)
    module = MLP2D(*weights, tt_ccl=_ccl(mesh), mesh_device=mesh)

    assert module.config.tt_ccl is not None


@pytest.mark.parametrize("activation", [ttnn.UnaryOpType.SILU, ttnn.UnaryOpType.GELU])
def test_decode_uses_configured_activation_and_releases_only_module_owned_transients(monkeypatch, activation):
    def tensor(name, shape=(1, 1, 32, 128)):
        value = SimpleNamespace(name=name, shape=shape)
        value.memory_config = lambda: f"{name}-memory"
        return value

    caller_input = tensor("caller")
    w1_scattered = tensor("w1-persistent")
    w3_projection, w3_scattered = tensor("w3-projection"), tensor("w3-persistent")
    gated, gathered = tensor("gated"), tensor("gathered-persistent")
    w2_output, final = tensor("w2-output"), tensor("final")
    module = object.__new__(MLP2D)
    module.config = SimpleNamespace(
        decode_prefetch_context=None,
        decode_activation_dtype="activation-dtype",
        decode_ff1_3_compute_kernel_cfg="ff1-kernel",
        decode_w1_w3_prg_config="ff1-program",
        decode_w1_w3_output_memcfg="ff1-memory",
        ff1_out_reduce_scatter_memcfg="rs-memory",
        mlp_activation_type=activation,
        decode_mul_dtype="mul-dtype",
        decode_ff2_compute_kernel_cfg="ff2-kernel",
        decode_ccl_dtype="ccl-dtype",
        decode_w2_prg_config="ff2-program",
        decode_w2_input_memcfg="ff2-input-memory",
        decode_w2_output_memcfg="ff2-memory",
        ff2_out_reduce_scatter_memcfg="all-reduce-memory",
        sharded_attn_input_memcfg=None,
    )
    module.w1, module.w2, module.w3 = "w1", "w2", "w3"
    module.load_device_weights = lambda *_args: None
    module._double_matmul_reduce_scatter_axis1 = MagicMock(return_value=(w1_scattered, w3_projection))
    module._llama_reduce_scatter_axis1 = MagicMock(return_value=w3_scattered)
    module._all_gather_axis1 = MagicMock(return_value=gathered)
    module._all_reduce_tg = MagicMock(return_value=final)
    monkeypatch.setattr(mlp_2d, "_load_input_device_tensor", lambda value, *_args, **_kwargs: value)
    monkeypatch.setattr(ttnn, "linear", MagicMock(return_value=w2_output))
    monkeypatch.setattr(ttnn, "mul", MagicMock(return_value=gated))
    monkeypatch.setattr(ttnn, "reshape", lambda value, _shape: value)
    deallocate = MagicMock()
    monkeypatch.setattr(ttnn, "deallocate", deallocate)

    assert module.decode_forward(caller_input) is final

    released = [call.args[0] for call in deallocate.call_args_list]
    assert caller_input not in released
    assert w1_scattered not in released
    assert w3_scattered not in released
    assert gathered not in released
    assert released == [w3_projection, gated, w2_output]
    assert ttnn.mul.call_args.kwargs["input_tensor_a_activations"] == [activation]
    module._llama_reduce_scatter_axis1.assert_called_once_with(w3_projection)
    assert module._all_gather_axis1.call_args.args[1] == "ff2-input-memory"
    assert ttnn.linear.call_args.args[0] is gathered


@pytest.mark.parametrize("activation", [ttnn.UnaryOpType.SILU, ttnn.UnaryOpType.GELU])
def test_prefill_uses_mode_specific_configs_and_configured_activation(monkeypatch, activation):
    def tensor(name, shape=(1, 1, 128, 128)):
        value = SimpleNamespace(name=name, shape=shape)
        value.memory_config = lambda: f"{name}-memory"
        return value

    caller_input = tensor("caller")
    w1_projection, w3_projection = tensor("w1-projection"), tensor("w3-projection")
    w1_scattered, w3_scattered = tensor("w1-persistent"), tensor("w3-persistent")
    gated, gathered = tensor("gated"), tensor("gathered-persistent")
    w2_output, final = tensor("w2-output"), tensor("final")
    w1_w3_factory = MagicMock(return_value="prefill-ff1-program")
    w2_factory = MagicMock(return_value="prefill-ff2-program")
    module = object.__new__(MLP2D)
    module.config = SimpleNamespace(
        prefill_prefetch_context=None,
        prefill_len_cutoff=1024,
        prefill_w1_w3_prg_config=w1_w3_factory,
        prefill_w2_prg_config=w2_factory,
        prefill_activation_dtype="prefill-activation-dtype",
        prefill_ff1_3_compute_kernel_cfg="prefill-ff1-kernel",
        prefill_w1_w3_output_memcfg="prefill-ff1-memory",
        mlp_activation_type=activation,
        prefill_mul_dtype="prefill-mul-dtype",
        prefill_ff2_compute_kernel_cfg="prefill-ff2-kernel",
        prefill_ccl_dtype="prefill-ccl-dtype",
        prefill_w2_output_memcfg="prefill-ff2-memory",
    )
    module.w1, module.w2, module.w3 = "w1", "w2", "w3"
    module.load_device_weights = lambda *_args: None
    module.prefill_w1, module.prefill_w2, module.prefill_w3 = "w1", "w2", "w3"
    module._reduce_scatter_axis1 = MagicMock(side_effect=[w1_scattered, w3_scattered])
    module._all_gather_axis1 = MagicMock(return_value=gathered)
    module._all_reduce_tg = MagicMock(return_value=final)
    monkeypatch.setattr(mlp_2d, "_load_input_device_tensor", lambda value, *_args, **_kwargs: value)
    linear = MagicMock(side_effect=[w1_projection, w3_projection, w2_output])
    monkeypatch.setattr(ttnn, "linear", linear)
    mul = MagicMock(return_value=gated)
    monkeypatch.setattr(ttnn, "mul", mul)
    monkeypatch.setattr(ttnn, "reshape", lambda value, _shape: value)
    deallocate = MagicMock()
    monkeypatch.setattr(ttnn, "deallocate", deallocate)

    assert module.prefill_forward(caller_input) is final

    w1_w3_factory.assert_called_once_with(128)
    w2_factory.assert_called_once_with(128)
    assert linear.call_args_list[0].kwargs["program_config"] == "prefill-ff1-program"
    assert linear.call_args_list[2].kwargs["program_config"] == "prefill-ff2-program"
    assert mul.call_args.kwargs["input_tensor_a_activations"] == [activation]
    assert module._reduce_scatter_axis1.call_args_list[0].args[2:] == ("prefill", "w1")
    assert module._reduce_scatter_axis1.call_args_list[1].args[2:] == ("prefill", "w3")
    assert module._all_gather_axis1.call_args.args[2:] == ("prefill", "gated")
    assert module._all_reduce_tg.call_args.kwargs["mode"] == "prefill"
    assert [call.args[0] for call in deallocate.call_args_list] == [w1_projection, w3_projection, gated, w2_output]


def test_forward_rejects_unknown_mode():
    module = object.__new__(MLP2D)
    with pytest.raises(ValueError, match="mode must be"):
        module.forward(object(), mode="train")
