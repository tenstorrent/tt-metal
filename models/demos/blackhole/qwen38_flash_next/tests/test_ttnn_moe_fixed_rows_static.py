# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device gates for the one/five-row TTNN MoE path."""

from __future__ import annotations

import ast
import inspect
import itertools
import textwrap
from pathlib import Path
from unittest import mock

import pytest
import torch

import models.demos.blackhole.qwen38_flash_next.ttnn.builder as builder_module
import models.demos.blackhole.qwen38_flash_next.ttnn.moe as moe_module
from models.demos.blackhole.qwen38_flash_next.ttnn import decode_matmul as decode_matmul_module
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import TensorPlacement
from models.demos.blackhole.qwen38_flash_next.ttnn.moe import (
    BLACKHOLE_MOE_NUMERIC_ISSUE,
    EXPERTS_PER_DEVICE,
    HIDDEN_SIZE,
    MOE_STAGE_FENCES,
    PREFILL_CHUNK_ROWS,
    ROUTED_EXPERTS,
    ROWS5_HARDWARE_PROVEN,
    ROWS32_HARDWARE_PROVEN,
    SUPPORTED_ROWS,
    TARGET_VERIFIER_ROWS,
    TOP_K,
    Qwen38TTNNMoE,
    Qwen38TTNNMoEResult,
    Qwen38TTNNMoERowContract,
    Qwen38TTNNMoESyncPolicy,
    Qwen38TTNNRouting,
    _expert_owner_mapping,
)

REPO_ROOT = Path(__file__).resolve().parents[5]


class _FakeTensor:
    _ids = itertools.count(1)

    def __init__(self, shape, *, dtype=None, layout=None, memory=None, tensor_id=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.layout = layout
        self._memory = memory
        self.tensor_id = next(self._ids) if tensor_id is None else tensor_id

    def memory_config(self):
        return self._memory

    def device(self):
        return "mesh"


def _bare_moe(rows: int) -> Qwen38TTNNMoE:
    instance = object.__new__(Qwen38TTNNMoE)
    instance.row_contract = Qwen38TTNNMoERowContract(rows)
    instance.rows = rows
    instance.synchronization_policy = Qwen38TTNNMoESyncPolicy.CORRECTNESS_FENCED
    instance.mesh_device = object()
    instance.mesh_contract = mock.Mock()
    instance.tt_ccl = object()
    instance.weights = object()
    instance.compute_config = object()
    instance.routing_l1_memory_config = object()
    instance.hidden_act_memory_config = object()
    instance.hidden_gather_memory_config = instance.hidden_act_memory_config
    instance.expert_mapping = _FakeTensor((4, ROUTED_EXPERTS), dtype=moe_module.ttnn.uint16)
    instance.local_combine_output = _FakeTensor(
        instance.row_contract.local_combine,
        dtype=moe_module.ttnn.bfloat16,
        layout=moe_module.ttnn.ROW_MAJOR_LAYOUT,
        memory=moe_module.ttnn.DRAM_MEMORY_CONFIG,
    )
    instance.local_output = False  # the production path: FullLocal (QWEN38_MOE_LOCAL_OUTPUT unset)
    instance._owned_buffers_released = False
    instance._owns_local_combine_output = True
    instance.output_height_shard_dim = 1
    instance.routed_tokens = rows
    instance.routed_calls = 1
    instance._poisoned_error = None
    instance._poisoned_device_owners = []
    return instance


def _attribute_call(tree: ast.AST, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == name
    ]


def test_rows_form_switch(expect_error, monkeypatch) -> None:
    """QWEN38_MOE_ROWS_FORM (unset = rings2): the one-tile rows forms' moe_compute launch form.  rings2 = the local output
    path with two rings (the default since 2026-09-27: per launch 42.0 + 10.87 n -> 36.1 + 7.39 n us over n distinct
    local experts on the 1x4 p150 line, the combine pages bitwise the FullLocal form); fulllocal = the fused local
    combine (the fallback); localoutput / replay / rings3 diagnostics.  Only one-tile rows instances read it; a ring
    count of two or more turns the op's zero fill off (the model's fill before every call keeps the unowned rows zero).
    """

    assert moe_module.MOE_ROWS_FORM_ENV == "QWEN38_MOE_ROWS_FORM"
    assert moe_module.MOE_ROWS_FORM_DEFAULT == "rings2"
    assert moe_module.MOE_ROWS_FORMS == {
        "fulllocal": (False, None),
        "localoutput": (True, None),
        "replay": (True, 1),
        "rings2": (True, 2),
        "rings3": (True, 3),
    }
    monkeypatch.delenv(moe_module.MOE_ROWS_FORM_ENV, raising=False)
    assert moe_module.moe_rows_form() == "rings2"
    for form in moe_module.MOE_ROWS_FORMS:
        monkeypatch.setenv(moe_module.MOE_ROWS_FORM_ENV, form)
        assert moe_module.moe_rows_form() == form
    monkeypatch.setenv(moe_module.MOE_ROWS_FORM_ENV, "rings4")
    with expect_error(ValueError):  # allow-pytest.raises: pure contract test
        moe_module.moe_rows_form()
    monkeypatch.delenv(moe_module.MOE_ROWS_FORM_ENV)
    # the instance: one-tile rows read the switch, every other instance is fulllocal there; the local output path
    # and the kwargs follow the table
    init = inspect.getsource(Qwen38TTNNMoE.__init__)
    assert 'self.rows_form = moe_rows_form() if self.row_contract.row_tiles == 1 else "fulllocal"' in init
    assert (
        "self.local_output = moe_local_output_enabled() or self.slab_one_call or MOE_ROWS_FORMS[self.rows_form][0]"
        in init
    )
    # every moe_compute launch of the class passes the form's kwargs: the plain path (_routed_partial), the served
    # routing-in-L1 path (_routed_local_sum, the decode and MTP verify rows) and the 128-row form's tiles
    for launch in (Qwen38TTNNMoE._routed_partial, Qwen38TTNNMoE._routed_local_sum, Qwen38TTNNMoE._routed_partial_tiles):
        source = inspect.getsource(launch)
        assert source.count("ttnn.experimental.moe_compute(") == 1, launch.__name__
        assert "local_combine=not self.local_output," in source, launch.__name__
        assert "zero_fill_non_owned_rows=self.zero_fill_non_owned_rows," in source, launch.__name__
        assert "prefill_rings=self.prefill_rings," in source, launch.__name__
    partial = inspect.getsource(Qwen38TTNNMoE._routed_partial)
    zero_fill = inspect.getsource(Qwen38TTNNMoE.zero_fill_non_owned_rows.fget)
    assert 'return (MOE_ROWS_FORMS[getattr(self, "rows_form", "fulllocal")][1] or 0) < 2' in zero_fill
    for rows, form, rings, local_output, zero_fill_expected in (
        (1, "rings2", 2, True, False),
        (5, "rings2", 2, True, False),
        (5, "fulllocal", None, False, True),
        (5, "localoutput", None, True, True),
        (5, "replay", 1, True, True),
        (32, "rings3", 3, True, False),
    ):
        instance = _bare_moe(rows)
        instance.rows_form = form
        instance.local_output = moe_module.MOE_ROWS_FORMS[form][0]
        assert instance.prefill_rings == rings
        assert instance.local_output is local_output
        assert instance.zero_fill_non_owned_rows is zero_fill_expected
    # the fill before every call stays for every one-tile rows form (the unowned rows of a rings form are zero by it)
    assert "if self.rows != LONG_PREFILL_CHUNK_ROWS and not self.slab_one_call:" in partial
    assert "ttnn.fill(self.local_combine_output, 0.0, output_tensor=self.local_combine_output)" in partial


def test_builder_owns_one_lazy_ccl_manager(monkeypatch) -> None:
    mesh_device = object()
    manager = mock.Mock()
    manager.get_num_links.return_value = 2
    manager.get_and_cycle_barrier_semaphore_handle.return_value = object()
    manager.get_and_cycle_ag_semaphore_handles.return_value = object()
    manager.get_and_cycle_rs_semaphore_handles.return_value = object()
    factory = mock.Mock(return_value=manager)
    monkeypatch.setattr(builder_module, "TT_CCL", factory)

    lazy_ccl = builder_module._Qwen38LazyTTCCL(mesh_device)
    factory.assert_not_called()

    assert lazy_ccl.get_num_links(0) == 2
    assert (
        lazy_ccl.get_and_cycle_barrier_semaphore_handle() is manager.get_and_cycle_barrier_semaphore_handle.return_value
    )
    assert lazy_ccl.get_and_cycle_ag_semaphore_handles(1) is manager.get_and_cycle_ag_semaphore_handles.return_value
    assert lazy_ccl.get_and_cycle_rs_semaphore_handles() is manager.get_and_cycle_rs_semaphore_handles.return_value
    factory.assert_called_once_with(mesh_device)
    manager.get_num_links.assert_called_once_with(0)
    manager.get_and_cycle_barrier_semaphore_handle.assert_called_once_with(None)
    manager.get_and_cycle_ag_semaphore_handles.assert_called_once_with(1)
    manager.get_and_cycle_rs_semaphore_handles.assert_called_once_with(None)

    constructor_source = inspect.getsource(builder_module.Qwen38TTNNBuilder.__init__)
    assert "self.tt_ccl = _Qwen38LazyTTCCL(mesh_device)" in constructor_source


def test_fixed_row_contract_is_exact_and_fail_closed(expect_error) -> None:
    # 1..32 rows (1 = decode, 5 = the MTP verifier, 32 = the prefill chunk, B = the batched lanes) and the 128-row chunk.
    assert (
        SUPPORTED_ROWS == (*range(1, 33), 128)
        and PREFILL_CHUNK_ROWS == 32
        and moe_module.LONG_PREFILL_CHUNK_ROWS == 128
    )
    ordinary = Qwen38TTNNMoERowContract(1)
    assert ordinary.hidden_sharded == (1, 1, 1, 640)
    assert ordinary.full_hidden == (1, 1, 1, HIDDEN_SIZE)
    assert ordinary.routing == (1, 1, 1, TOP_K)
    assert ordinary.moe_sparse_input == (1, 1, 1, HIDDEN_SIZE)
    assert ordinary.moe_routing == (1, 1, TOP_K)
    assert ordinary.routing_shard == (1, TOP_K)
    assert ordinary.local_combine == (TOP_K, 1, HIDDEN_SIZE)
    assert ordinary.fast_reduce_input == (TOP_K, 1, 1, HIDDEN_SIZE)
    assert ordinary.fast_reduce_scores == (1, 1, 1, TOP_K)
    assert ordinary.output_sharded == (1, 1, 1, 640)

    verifier = Qwen38TTNNMoERowContract(TARGET_VERIFIER_ROWS)
    assert verifier.hidden_sharded == (1, 1, 5, 640)
    assert verifier.full_hidden == (1, 1, 5, HIDDEN_SIZE)
    assert verifier.routing == (1, 1, 5, TOP_K)
    assert verifier.moe_sparse_input == (1, 5, HIDDEN_SIZE)
    assert verifier.moe_routing == (1, 5, TOP_K)
    assert verifier.routing_shard == (5, TOP_K)
    assert verifier.local_combine == (TOP_K, 5, HIDDEN_SIZE)
    assert verifier.fast_reduce_input == (TOP_K, 1, 5, HIDDEN_SIZE)
    assert verifier.fast_reduce_scores == (5, 1, 1, TOP_K)
    assert verifier.output_sharded == (1, 1, 5, 640)

    chunk = Qwen38TTNNMoERowContract(PREFILL_CHUNK_ROWS)
    assert chunk.hidden_sharded == (1, 1, 32, 640)
    assert chunk.full_hidden == (1, 1, 32, HIDDEN_SIZE)
    assert chunk.routing == (1, 1, 32, TOP_K)
    assert chunk.moe_sparse_input == (1, 32, HIDDEN_SIZE)
    assert chunk.moe_routing == (1, 32, TOP_K)
    assert chunk.routing_shard == (32, TOP_K)
    assert chunk.local_combine == (TOP_K, 32, HIDDEN_SIZE)
    assert chunk.fast_reduce_input == (TOP_K, 1, 32, HIDDEN_SIZE)
    assert chunk.fast_reduce_scores == (32, 1, 1, TOP_K)
    assert chunk.output_sharded == (1, 1, 32, 640)

    # Every lane count 1..32 is a row count (the batched decode); anything else is refused with the range.
    for rows in (2, 4, 6, 31):
        lanes = Qwen38TTNNMoERowContract(rows)
        assert lanes.moe_sparse_input == (1, rows, HIDDEN_SIZE) and lanes.fast_reduce_scores == (rows, 1, 1, TOP_K)
    for rows in (False, True, 0, 33, 64, 1.0, "5", "32"):
        with expect_error(ValueError, match=r"MoE rows must be exactly one of"):  # allow-pytest.raises: contract
            Qwen38TTNNMoERowContract(rows)


def test_row_contract_admits_an_explicit_override_for_one_instance_only(expect_error) -> None:
    """``admitted_rows`` widens the admission for the instance that names it (the MTP v2 runner's explicit MoE rows
    override, e.g. 6 rows for k = 5); the module's ``SUPPORTED_ROWS`` and proof flags do not move."""

    six = Qwen38TTNNMoERowContract(6, (6,))
    assert six.rows == 6 and six.hidden_sharded == (1, 1, 6, 640) and six.moe_sparse_input == (1, 6, HIDDEN_SIZE)
    assert six.moe_routing == (1, 6, TOP_K) and six.fast_reduce_scores == (6, 1, 1, TOP_K)
    assert Qwen38TTNNMoERowContract(5, (5, 6)).rows == 5
    assert Qwen38TTNNMoERowContract(1).admitted_rows == SUPPORTED_ROWS == (*range(1, 33), 128)
    for rows, admitted in ((33, SUPPORTED_ROWS), (7, (6,)), (0, (0,)), (129, (129,)), (6, (True, 6)), (6, [6])):
        with expect_error(ValueError):  # allow-pytest.raises: pure contract test
            Qwen38TTNNMoERowContract(rows, admitted)
    with expect_error(ValueError):  # allow-pytest.raises: an empty admission admits nothing
        Qwen38TTNNMoERowContract(1, ())
    parameter = inspect.signature(Qwen38TTNNMoE.__init__).parameters["admitted_rows"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY and parameter.default == SUPPORTED_ROWS
    assert moe_module.ROWS5_HARDWARE_PROVEN is True and moe_module.ROWS32_HARDWARE_PROVEN is True


def test_default_constructor_contract_preserves_the_one_row_api() -> None:
    parameter = inspect.signature(Qwen38TTNNMoE.__init__).parameters["rows"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == 1
    policy = inspect.signature(Qwen38TTNNMoE.__init__).parameters["synchronization_policy"]
    assert policy.kind is inspect.Parameter.KEYWORD_ONLY
    assert policy.default is Qwen38TTNNMoESyncPolicy.CORRECTNESS_FENCED


@pytest.mark.parametrize(
    ("expert_residency", "expected_policy"),
    (
        ("resident", Qwen38TTNNMoESyncPolicy.RESIDENT_ASYNC),
        ("streamed", Qwen38TTNNMoESyncPolicy.CORRECTNESS_FENCED),
    ),
)
def test_builder_selects_async_only_for_proven_resident_expert_ownership(
    monkeypatch,
    expert_residency: str,
    expected_policy: Qwen38TTNNMoESyncPolicy,
) -> None:
    builder = object.__new__(builder_module.Qwen38TTNNBuilder)
    builder.checkpoint = object()
    builder.placement = object()
    builder.mesh_device = object()
    builder.mesh_contract = object()
    builder.component_cache_root = object()
    builder.provenance = mock.Mock(tt_metal_sha="a" * 40)
    builder.tt_ccl = object()
    builder.collective_topology = object()
    builder.expert_residency = expert_residency
    builder.dense_weight_plan = decode_matmul_module.default_dense_weight_plan({})  # bf16: the production path
    weights = object()
    constructed = object()
    from_checkpoint = mock.Mock(return_value=weights)
    constructor = mock.Mock(return_value=constructed)
    monkeypatch.setattr(builder_module.Qwen38TTNNMoEWeights, "from_checkpoint", from_checkpoint)
    monkeypatch.setattr(builder_module, "Qwen38TTNNMoE", constructor)

    result = builder._build_moe(namespace="backbone", layer_index=7)

    assert result is constructed
    assert constructor.call_args.kwargs["synchronization_policy"] is expected_policy


def test_sync_policy_rejects_untyped_alias_before_any_device_access(expect_error) -> None:
    with expect_error(TypeError, match="exact Qwen38TTNNMoESyncPolicy"):  # allow-pytest.raises: pure contract
        Qwen38TTNNMoE(
            object(),
            object(),
            object(),
            tt_ccl=object(),
            synchronization_policy="resident-async",
        )


def test_five_row_clone_borrows_resident_weights_and_allocates_only_private_buffers(monkeypatch) -> None:
    # the fused MoE post program and shared expert (on by default) upload constants through the mesh at construction,
    # which a Mock mesh cannot do; the router tail's constants are patched below and its resolution is asserted
    monkeypatch.setenv(moe_module.fused.OFF_ENV, "moe_post,shared_expert")
    ttnn = moe_module.ttnn
    mesh_device = mock.Mock()
    mesh_device.arch.return_value = object()
    mesh_device.dram_grid_size.return_value = ttnn.CoreCoord(8, 1)  # decode matmul configs read the bank grid
    mesh_contract = mock.Mock()
    # the fused shared expert (on by default) checks the concatenated weight is loaded; the clone borrows it as is
    borrowed_weights = mock.Mock(shared_gate_up_scalar=object(), shared_dtype=ttnn.bfloat16)
    mapping_tensor = _FakeTensor((4, ROUTED_EXPERTS), dtype=ttnn.uint16)
    local_tensor = _FakeTensor(
        (TOP_K, TARGET_VERIFIER_ROWS, HIDDEN_SIZE),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory=ttnn.DRAM_MEMORY_CONFIG,
    )
    drain = ttnn.CoreCoord(0, 0)

    with (
        mock.patch.object(moe_module, "effective_matmul_ring_size", return_value=7),
        mock.patch.object(moe_module, "auto_output_width_shard_dim", return_value=1),
        mock.patch.object(ttnn, "init_device_compute_kernel_config", return_value=object()),
        mock.patch.object(ttnn.experimental, "get_moe_tilize_drain_core", return_value=drain) as get_drain,
        mock.patch.object(
            moe_module,
            "replicate_tensor_2d_mesh_mapper",
            return_value=object(),
        ) as replicate_mapper,
        mock.patch.object(ttnn, "from_torch", side_effect=(mapping_tensor, local_tensor)) as from_torch,
        mock.patch.object(moe_module.fused.router_tail, "router_tail_prepare", return_value=object()) as prepare,
    ):
        clone = Qwen38TTNNMoE(
            mesh_device,
            mesh_contract,
            borrowed_weights,
            tt_ccl=object(),
            rows=TARGET_VERIFIER_ROWS,
        )

    assert clone.weights is borrowed_weights
    assert clone.rows == TARGET_VERIFIER_ROWS
    # the default tail is the fused program (QWEN38_FUSED_OFF=router_tail keeps the chain); its constants upload once
    assert clone._route_tail is moe_module.fused.kernel("router_tail").fused
    prepare.assert_called_once_with(mesh_device)
    assert clone.expert_mapping is mapping_tensor
    assert clone.local_combine_output is local_tensor
    assert clone.routing_l1_memory_config.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    assert clone.routing_l1_memory_config.buffer_type == ttnn.BufferType.L1
    assert clone.routing_l1_memory_config.shard_spec.shape == [TARGET_VERIFIER_ROWS, TOP_K]
    assert from_torch.call_count == 2
    assert [call.kwargs["mesh_mapper"] for call in from_torch.call_args_list] == [
        replicate_mapper.return_value,
        replicate_mapper.return_value,
    ]
    assert replicate_mapper.call_args_list == [mock.call(mesh_device), mock.call(mesh_device)]
    assert tuple(from_torch.call_args_list[1].args[0].shape) == (TOP_K, TARGET_VERIFIER_ROWS, HIDDEN_SIZE)
    get_drain.assert_called_once_with(mesh_device, 1, 1, HIDDEN_SIZE)
    mesh_contract.validate_tensor.assert_has_calls(
        [
            mock.call(mapping_tensor, placement=TensorPlacement.REPLICATED),
            mock.call(local_tensor, placement=TensorPlacement.LOCAL_PARTIAL),
        ]
    )


def test_release_owned_buffers_is_retry_safe_and_never_releases_borrowed_weights(expect_error) -> None:
    module = _bare_moe(TARGET_VERIFIER_ROWS)
    borrowed_weights = module.weights
    mapping = module.expert_mapping
    local = module.local_combine_output
    released = []

    def fail_mapping_once(tensor):
        released.append(tensor)
        if tensor is mapping:
            raise RuntimeError("mapping release failed")

    with mock.patch.object(moe_module.ttnn, "deallocate", side_effect=fail_mapping_once):
        with expect_error(RuntimeError, match="expert_mapping"):  # allow-pytest.raises: mocked deallocation
            module.release_owned_buffers()

    assert released == [mapping, local]
    assert module.expert_mapping is mapping
    assert module.local_combine_output is None
    assert module.weights is borrowed_weights
    assert module._owned_buffers_released is False

    with mock.patch.object(moe_module.ttnn, "deallocate") as deallocate:
        module.release_owned_buffers()
        deallocate.assert_called_once_with(mapping)
        assert module.expert_mapping is None
        assert module.local_combine_output is None
        assert module.weights is borrowed_weights
        assert module._owned_buffers_released is True
        module.release_owned_buffers()
        deallocate.assert_called_once_with(mapping)

    with expect_error(RuntimeError, match="buffers are unavailable"):  # allow-pytest.raises: no device fixture
        module.forward(_FakeTensor((1, 1, 5, 640)), object(), object())


def test_expert_owner_metadata_encodes_four_disjoint_128_expert_shards() -> None:
    mapping = _expert_owner_mapping()
    assert mapping.dtype == torch.int32
    assert tuple(mapping.shape) == (4, ROUTED_EXPERTS)
    assert all(torch.equal(mapping[0], row) for row in mapping)
    assert tuple(torch.bincount(mapping[0], minlength=4).tolist()) == (EXPERTS_PER_DEVICE,) * 4
    for owner in range(4):
        selected = torch.where(mapping[0] == owner)[0]
        assert selected.tolist() == list(range(owner * EXPERTS_PER_DEVICE, (owner + 1) * EXPERTS_PER_DEVICE))


def test_routed_weights_reject_expert_replication(expect_error) -> None:
    module = _bare_moe(TARGET_VERIFIER_ROWS)
    contract = module.row_contract
    ttnn = moe_module.ttnn
    full_hidden = _FakeTensor(contract.full_hidden)
    routing = Qwen38TTNNRouting(
        _FakeTensor(contract.routing),
        _FakeTensor(contract.routing),
    )
    replicated_w0_w1 = _FakeTensor((7, 1, ROUTED_EXPERTS, 1), dtype=ttnn.bfloat4_b)
    sharded_w2 = _FakeTensor((7, 1, EXPERTS_PER_DEVICE, 1), dtype=ttnn.bfloat4_b)
    with expect_error(RuntimeError, match="exactly 128 local experts"):  # allow-pytest.raises: no device fixture
        module._routed_partial(full_hidden, routing, replicated_w0_w1, sharded_w2)


@pytest.mark.parametrize("rows", [1, TARGET_VERIFIER_ROWS, PREFILL_CHUNK_ROWS])
def test_routed_partial_uses_the_exact_row_shapes_and_weight_ownership(rows: int) -> None:
    module = _bare_moe(rows)
    contract = module.row_contract
    ttnn = moe_module.ttnn
    full_hidden = _FakeTensor(
        contract.full_hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory=ttnn.DRAM_MEMORY_CONFIG,
    )
    routing = Qwen38TTNNRouting(
        scores=_FakeTensor(
            contract.routing,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory=ttnn.DRAM_MEMORY_CONFIG,
        ),
        indices=_FakeTensor(
            contract.routing,
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory=ttnn.DRAM_MEMORY_CONFIG,
        ),
    )
    packed_w0_w1 = _FakeTensor((7, 1, EXPERTS_PER_DEVICE, 1), dtype=ttnn.bfloat4_b)
    packed_w2 = _FakeTensor((7, 1, EXPERTS_PER_DEVICE, 1), dtype=ttnn.bfloat4_b)
    reshape_calls = []
    moe_compute_call = {}
    fast_reduce_call = {}

    def reshape(tensor, shape):
        reshape_calls.append((tensor, tuple(shape)))
        return _FakeTensor(
            shape,
            dtype=tensor.dtype,
            layout=tensor.layout,
            memory=tensor.memory_config(),
        )

    def to_layout(tensor, layout, *, memory_config, **_kwargs):
        return _FakeTensor(
            tensor.shape,
            dtype=tensor.dtype,
            layout=layout,
            memory=memory_config,
        )

    def to_memory_config(tensor, memory_config):
        return _FakeTensor(
            tensor.shape,
            dtype=tensor.dtype,
            layout=tensor.layout,
            memory=memory_config,
        )

    def moe_compute(*args, **kwargs):
        moe_compute_call.update(args=args, kwargs=kwargs)
        scratch = [_FakeTensor((1,)) for _ in range(5)]
        return [*scratch, module.local_combine_output]

    def unsqueeze(tensor, *, dim):
        shape = list(tensor.shape)
        shape.insert(dim, 1)
        return _FakeTensor(
            shape,
            dtype=tensor.dtype,
            layout=tensor.layout,
            memory=tensor.memory_config(),
        )

    def fast_reduce(*args, **kwargs):
        fast_reduce_call.update(args=args, kwargs=kwargs)
        return [
            _FakeTensor(
                contract.full_hidden,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory=ttnn.DRAM_MEMORY_CONFIG,
            )
        ]

    with (
        mock.patch.object(ttnn, "reshape", side_effect=reshape),
        mock.patch.object(ttnn, "to_layout", side_effect=to_layout),
        mock.patch.object(ttnn, "to_memory_config", side_effect=to_memory_config),
        mock.patch.object(ttnn, "fill", side_effect=lambda _tensor, _value, *, output_tensor: output_tensor),
        mock.patch.object(ttnn, "unsqueeze", side_effect=unsqueeze),
        mock.patch.object(ttnn, "deallocate"),
        mock.patch.object(ttnn.experimental, "moe_compute", side_effect=moe_compute),
        mock.patch.object(
            ttnn.experimental,
            "deepseek_moe_fast_reduce_nc_fused",
            side_effect=fast_reduce,
        ),
    ):
        output = module._routed_partial(full_hidden, routing, packed_w0_w1, packed_w2)

    assert output.shape == contract.full_hidden
    sparse_input, indices_l1, scores_l1 = moe_compute_call["args"][:3]
    assert sparse_input.shape == contract.moe_sparse_input
    assert (indices_l1.shape, scores_l1.shape) == (contract.moe_routing, contract.moe_routing)
    assert indices_l1.layout == scores_l1.layout == ttnn.ROW_MAJOR_LAYOUT
    assert indices_l1.dtype == ttnn.uint16
    assert scores_l1.dtype == ttnn.bfloat16
    assert indices_l1.memory_config() is module.routing_l1_memory_config
    assert scores_l1.memory_config() is module.routing_l1_memory_config
    assert moe_compute_call["args"][4:6] == (packed_w0_w1, packed_w2)
    assert moe_compute_call["kwargs"]["optional_output_tensor"] is module.local_combine_output
    assert moe_compute_call["kwargs"]["output_height_shard_dim"] == 1
    assert moe_compute_call["kwargs"]["cluster_axis"] == 0
    assert moe_compute_call["kwargs"]["compute_only"] is False
    assert moe_compute_call["kwargs"]["local_combine"] is True
    assert moe_compute_call["kwargs"]["num_shared_experts_per_device"] == 0

    fast_input, fast_indices = fast_reduce_call["args"][:2]
    assert fast_input.shape == contract.fast_reduce_input
    assert fast_indices is routing.indices
    assert fast_reduce_call["kwargs"]["scores_tensor"].shape == contract.fast_reduce_scores
    assert fast_reduce_call["kwargs"]["num_shared_experts"] == 0
    assert routing.scores.shape == routing.indices.shape == contract.routing

    # The gathered hidden is the linears' L1 shard; it is never reshaped in place.
    assert all(tensor is not full_hidden for tensor, _shape in reshape_calls)
    if rows == 1:
        assert fast_reduce_call["kwargs"]["scores_tensor"] is routing.scores
    else:
        # The rows are untilized first (the one-row op) and reshaped ROW_MAJOR: no un-shard or relayout kernel.
        assert [
            (tensor.memory_config(), tensor.layout, shape)
            for tensor, shape in reshape_calls
            if shape[-1] == HIDDEN_SIZE
        ] == [(ttnn.DRAM_MEMORY_CONFIG, ttnn.ROW_MAJOR_LAYOUT, contract.moe_sparse_input)]
        assert "to_memory_config(full_hidden" not in inspect.getsource(Qwen38TTNNMoE._routed_partial)
        assert fast_reduce_call["kwargs"]["scores_tensor"] is not routing.scores

    expected_weight_validation = [
        mock.call(packed_w0_w1, placement=TensorPlacement.EXPERT_SHARDED, shard_dim=2),
        mock.call(packed_w2, placement=TensorPlacement.EXPERT_SHARDED, shard_dim=2),
    ]
    for call in expected_weight_validation:
        assert call in module.mesh_contract.validate_tensor.call_args_list


def test_five_row_forward_validates_reduce_scatter_shape_and_topology() -> None:
    module = _bare_moe(TARGET_VERIFIER_ROWS)
    contract = module.row_contract
    ttnn = moe_module.ttnn
    hidden_sharded = _FakeTensor(contract.hidden_sharded)
    full_hidden = _FakeTensor(contract.full_hidden, memory=module.hidden_act_memory_config)
    routing = Qwen38TTNNRouting(_FakeTensor(contract.routing), _FakeTensor(contract.routing))
    routed_partial = _FakeTensor(contract.full_hidden)
    shared_partial = _FakeTensor(contract.full_hidden)
    local_sum = _FakeTensor(contract.full_hidden)
    output = _FakeTensor(contract.output_sharded)
    module.collective_topology = object()
    module._all_gather_hidden = mock.Mock(return_value=full_hidden)
    module._route = mock.Mock(return_value=routing)
    module._routed_partial = mock.Mock(return_value=routed_partial)
    module._shared_partial = mock.Mock(return_value=shared_partial)
    module._synchronize_stage = mock.Mock()

    with (
        mock.patch.object(ttnn, "to_memory_config") as to_memory_config,
        mock.patch.object(ttnn, "add", return_value=local_sum),
        mock.patch.object(moe_module, "tt_all_reduce", return_value=output) as tt_all_reduce,
        mock.patch.object(ttnn, "deallocate"),
    ):
        result = module.forward(hidden_sharded, object(), object(), return_routing=True)

    assert isinstance(result, Qwen38TTNNMoEResult)
    assert result.hidden_sharded is output
    assert result.routing is routing
    # The five-row verifier hands the gathered width-sharded hidden to the
    # router and shared-expert linears exactly like the one-row decode path.
    to_memory_config.assert_not_called()
    module._route.assert_called_once_with(full_hidden, hidden_tiles=None, phase_observer=mock.ANY)
    module._shared_partial.assert_called_once_with(hidden_sharded, full_hidden, None)
    tt_all_reduce.assert_called_once_with(
        local_sum,
        module.mesh_device,
        module.tt_ccl,
        cluster_axis=0,
        dim=3,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=module.collective_topology,
    )
    module.mesh_contract.mark_collective_shard.assert_called_once_with(
        output,
        replicated_reference=full_hidden,
        shard_dim=3,
        expected_local_shape=(1, 1, 5, 640),
    )
    assert module._synchronize_stage.call_args_list == [mock.call(stage) for stage in MOE_STAGE_FENCES]


def test_python_path_preserves_normalized_scores_and_dynamic_shared_gate() -> None:
    route_tree = ast.parse(textwrap.dedent(inspect.getsource(Qwen38TTNNMoE._route)))
    # Replicated router: one local full-width linear, no logits all-gather.
    assert len(_attribute_call(route_tree, "linear")) == 1
    assert len(_attribute_call(route_tree, "all_gather")) == 0
    assert len(_attribute_call(route_tree, "softmax")) == 1
    assert len(_attribute_call(route_tree, "topk")) == 1
    assert len(_attribute_call(route_tree, "sum")) == 1
    assert len(_attribute_call(route_tree, "div")) == 1

    routed_tree = ast.parse(textwrap.dedent(inspect.getsource(Qwen38TTNNMoE._routed_partial)))
    local_combine_fills = _attribute_call(routed_tree, "fill")
    moe_compute = _attribute_call(routed_tree, "moe_compute")
    routed_stage_syncs = sorted(_attribute_call(routed_tree, "synchronize_device"), key=lambda call: call.lineno)
    fast_reduce = _attribute_call(routed_tree, "deepseek_moe_fast_reduce_nc_fused")
    fill_contracts = [
        node
        for node in ast.walk(routed_tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Attribute)
            and isinstance(child.value, ast.Name)
            and child.value.id == "zeroed"
            and child.attr == "tensor_id"
            for child in ast.walk(node.test)
        )
    ]
    output_contracts = [
        node
        for node in ast.walk(routed_tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "len"
            and len(child.args) == 1
            and isinstance(child.args[0], ast.Name)
            and child.args[0].id == "outputs"
            for child in ast.walk(node.test)
        )
    ]
    alias_deallocations = [
        node
        for node in ast.walk(routed_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_deallocate"
        and any(
            isinstance(arg, ast.Subscript) and isinstance(arg.value, ast.Name) and arg.value.id == "outputs"
            for arg in node.args
        )
    ]
    assert len(local_combine_fills) == len(moe_compute) == len(fill_contracts) == 1
    assert len(routed_stage_syncs) == 0
    assert len(output_contracts) == len(alias_deallocations) == 1
    assert len(fast_reduce) == 1
    assert (
        local_combine_fills[0].lineno
        < fill_contracts[0].lineno
        < moe_compute[0].lineno
        < output_contracts[0].lineno
        < alias_deallocations[0].lineno
        < fast_reduce[0].lineno
    )
    fast_keywords = {keyword.arg: keyword.value for keyword in fast_reduce[0].keywords}
    assert isinstance(fast_keywords["scores_tensor"], ast.Name)
    assert fast_keywords["scores_tensor"].id == "fast_reduce_scores"

    shared_tree = ast.parse(textwrap.dedent(inspect.getsource(Qwen38TTNNMoE._shared_partial)))
    hidden_reshards = [
        call
        for call in _attribute_call(shared_tree, "to_memory_config")
        if len(call.args) >= 1 and isinstance(call.args[0], ast.Name) and call.args[0].id == "full_hidden"
    ]
    assert len(hidden_reshards) == 0
    scalar_linears = [
        call
        for call in _attribute_call(shared_tree, "linear")
        if len(call.args) >= 2
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "hidden_tile"
        and isinstance(call.args[1], ast.Attribute)
        and call.args[1].attr == "shared_scalar_gate"
    ]
    assert len(scalar_linears) == 1
    assert len(_attribute_call(shared_tree, "all_reduce")) == 0
    scalar_sigmoids = _attribute_call(shared_tree, "sigmoid")
    assert len(scalar_sigmoids) == 1
    gated_muls = [
        call
        for call in _attribute_call(shared_tree, "mul")
        if len(call.args) >= 2
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "partial"
        and isinstance(call.args[1], ast.Name)
        and call.args[1].id == "scalar_gate"
    ]
    assert len(gated_muls) == 1
    assert scalar_sigmoids[0].lineno < gated_muls[0].lineno

    forward_tree = ast.parse(textwrap.dedent(inspect.getsource(Qwen38TTNNMoE.forward)))
    routes = _attribute_call(forward_tree, "_route")
    routed_partials = _attribute_call(forward_tree, "_routed_partial")
    stage_fences = sorted(_attribute_call(forward_tree, "_synchronize_stage"), key=lambda call: call.lineno)
    shared_partials = _attribute_call(forward_tree, "_shared_partial")
    branch_adds = _attribute_call(forward_tree, "add")
    assert len(routes) == len(routed_partials) == len(shared_partials) == len(branch_adds) == 1
    assert [call.args[0].value for call in stage_fences] == list(MOE_STAGE_FENCES)
    assert routes[0].lineno < shared_partials[0].lineno < routed_partials[0].lineno
    assert shared_partials[0].lineno < branch_adds[0].lineno
    assert routed_partials[0].lineno < branch_adds[0].lineno

    fence_tree = ast.parse(textwrap.dedent(inspect.getsource(Qwen38TTNNMoE._synchronize_stage)))
    assert len(_attribute_call(fence_tree, "synchronize_device")) == 1

    upload_tree = ast.parse(textwrap.dedent(inspect.getsource(moe_module.Qwen38TTNNMoEWeights.from_checkpoint)))
    cache_names = {node.value for node in ast.walk(upload_tree) if isinstance(node, ast.Constant)}
    assert {"router_replicated_dram_sharded", "shared_scalar_gate_replicated_dram_sharded"} <= cache_names
    # Neither N-sharded router cache key (interleaved or DRAM-sharded) may load again.
    assert not {"router", "router_dram_sharded"} & cache_names
    replicate_calls = [
        node
        for node in ast.walk(upload_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "replicate_tensor_2d_mesh_mapper"
    ]
    assert len(replicate_calls) == 1
    assert isinstance(replicate_calls[0].args[0], ast.Name)
    assert replicate_calls[0].args[0].id == "mesh_device"
    replicated_validations = [
        call
        for call in _attribute_call(upload_tree, "validate_tensor")
        if any(
            keyword.arg == "placement"
            and isinstance(keyword.value, ast.Attribute)
            and keyword.value.attr == "REPLICATED"
            for keyword in call.keywords
        )
    ]
    # router and shared_scalar_gate are the two replicated resident tensors.
    assert [call.args[0].id for call in replicated_validations] == ["router", "shared_scalar_gate"]


def test_source_numeric_admission_labels_remain_explicit() -> None:
    # These source-era labels are not current hardware qualification. Shared
    # replication, shape and dispatch behavior is covered by the direct MoE
    # operation suite and the real-checkpoint model numerical controls.
    assert ROWS5_HARDWARE_PROVEN is True and ROWS32_HARDWARE_PROVEN is True
    assert BLACKHOLE_MOE_NUMERIC_ISSUE.endswith("/50038")


def test_router_logits_widen_inside_the_sharded_to_interleaved_move() -> None:
    route = inspect.getsource(Qwen38TTNNMoE._route)
    assert (
        route.count(
            "logits_tiles.append(ttnn.to_memory_config(logits_ws, ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.float32))"
        )
        == 1
    )
    assert "ttnn.typecast(logits" not in route
    assert "logits_fp32" not in route
    assert route.index("logits_tiles.append(ttnn.to_memory_config(") < route.index("probabilities = ttnn.softmax(")
    route_tree = ast.parse(textwrap.dedent(route))
    softmaxes = _attribute_call(route_tree, "softmax")
    assert len(softmaxes) == 1
    assert ast.unparse(softmaxes[0].args[0]) == "logits"
    # The two remaining casts: normalized scores to BF16, indices to UINT16.
    casts = sorted(_attribute_call(route_tree, "typecast"), key=lambda call: (call.lineno, call.col_offset))
    assert [ast.unparse(call.args[1]) for call in casts] == ["ttnn.bfloat16", "ttnn.uint16"]

    # Runtime contract behind the fold: a sharded -> interleaved move with a
    # dtype dispatches to sharded_to_interleaved, which accepts a dtype change
    # for TILE input and converts with a datacopy compute kernel (exact for the
    # BF16 -> FP32 widening).
    dispatch = (REPO_ROOT / "ttnn/cpp/ttnn/operations/core/to_memory_config/to_memory_config_op.cpp").read_text()
    assert (
        "return ttnn::prim::sharded_to_interleaved(tensor, memory_config, dtype.value_or(tensor.dtype()), "
        "output_tensor);"
    ) in dispatch
    sharded_to_interleaved = REPO_ROOT / "ttnn/cpp/ttnn/operations/data_movement/sharded/sharded_to_interleaved/device"
    validate = (sharded_to_interleaved / "sharded_to_interleaved_device_operation.cpp").read_text()
    assert 'return {false, "If diff output type, tensor must be TILED"};' in validate
    factory = (sharded_to_interleaved / "sharded_to_interleaved_program_factory.cpp").read_text()
    assert "bool convert_df = input_data_format != output_data_format;" in factory
    assert '.source = "ttnn/cpp/ttnn/kernel/compute/eltwise_copy_metal2.cpp",' in factory


# Device ops each production ttnn call issues on the pinned runtime (Tracy
# lean-v2 capture, LAYER-OP-INVENTORY-20260902 ops 145-182): ttnn.topk pads k
# to one tile and slices both outputs (FillPad, TopK, Slice, Slice); ttnn.sum
# fills the implicit tile padding before its reduce; row-major reshape and
# unsqueeze are views.
_DEVICE_OPS_PER_CALL = {
    "all_gather": 1,
    "to_memory_config": 1,
    "linear": 1,
    "softmax": 1,
    "topk": 4,
    "sum": 2,
    "div": 1,
    "typecast": 1,
    "to_layout": 1,
    "silu": 1,
    "mul": 1,
    "sigmoid": 1,
    "fill": 1,
    "moe_compute": 1,
    "deepseek_moe_fast_reduce_nc_fused": 1,
    "add": 1,
    "tt_all_reduce": 1,
    "reshape": 0,
    "unsqueeze": 0,
}


def _production_ttnn_calls(function) -> list[str]:
    """ttnn calls on the one-row production path, in source order.

    Skips the multi-row reshapes (``if self.rows != 1:``), the long chunk's per-tile concats
    (``if self.row_contract.row_tiles != 1:``), the per-tile routed stream's routing tiles
    (``if self.routed_calls != 1:``), the long chunk's own arms (``if self.rows == LONG_PREFILL_CHUNK_ROWS:``)
    and deallocations.
    """

    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    skipped: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and ast.unparse(node.test) in (
            "self.rows != 1",
            "self.row_contract.row_tiles != 1",
            "self.routed_calls != 1",
            "self.rows == LONG_PREFILL_CHUNK_ROWS",
        ):
            skipped.update(id(child) for statement in node.body for child in ast.walk(statement))
    calls: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if id(node) in skipped or not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "tt_all_reduce":
            calls.append((node.lineno, node.col_offset, node.func.id))
        elif isinstance(node.func, ast.Attribute) and ast.unparse(node.func.value) in {"ttnn", "ttnn.experimental"}:
            calls.append((node.lineno, node.col_offset, node.func.attr))
    return [name for _line, _column, name in sorted(calls) if name != "deallocate"]


def test_one_row_moe_layer_device_op_sequence_and_count() -> None:
    sequences = {
        Qwen38TTNNMoE._all_gather_hidden: ["all_gather"],
        Qwen38TTNNMoE.forward: ["add", "tt_all_reduce"],
        Qwen38TTNNMoE._route: [
            "linear",
            "to_memory_config",
            "softmax",
            "topk",
            "sum",
            "div",
            "typecast",
            "to_layout",
            "to_layout",
            "typecast",
        ],
        Qwen38TTNNMoE._shared_partial: [
            "linear",
            "linear",
            "silu",
            "mul",
            "linear",
            "linear",
            "to_memory_config",
            "sigmoid",
            "mul",
        ],
        Qwen38TTNNMoE._routed_partial: [
            "to_layout",
            "reshape",
            "reshape",
            "to_memory_config",
            "to_memory_config",
            "fill",
            "moe_compute",
            "unsqueeze",
            "to_layout",
            "deepseek_moe_fast_reduce_nc_fused",
        ],
    }
    for function, expected in sequences.items():
        assert _production_ttnn_calls(function) == expected, function.__name__
    per_layer = sum(_DEVICE_OPS_PER_CALL[call] for calls in sequences.values() for call in calls)
    # 35 before the router logits cast was folded into the sharded -> interleaved
    # move; 34 before the hidden all-gather wrote the five-core activation shard.
    assert per_layer == 33
    assert per_layer * builder_module.BACKBONE_LAYERS == 1584
    gather = inspect.getsource(Qwen38TTNNMoE._all_gather_hidden)
    gather_call = gather.split("full_hidden = ttnn.all_gather(", 1)[1].split("\n        )", 1)[0]
    assert "memory_config=self.hidden_gather_memory_config" in gather_call
    assert "cluster_axis=EP_AXIS" in gather_call
