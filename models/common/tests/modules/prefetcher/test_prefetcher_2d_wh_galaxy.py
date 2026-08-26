# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Real-hardware qualification for Prefetcher2D and the Galaxy resource owner.

`test_prefetcher_2d.py` covers the whole lifecycle against a `FakeMesh`, and the
MLP2D/RMSNorm2D device suites exercise a real prefetcher incidentally - but every
one of them pins a single mode for its whole lifetime, so on silicon the
prefill<->decode transition, the rollback paths, and cleanup from an active mode
have never run at all. That is what this file closes.

The payload is the qualified MLP2D geometry, imported wholesale from
`models.common.tests.modules._mlp_2d_galaxy`, so the PCC numbers here are directly
comparable with the MLP2D evidence and a wrong core grid cannot creep in.

Two contracts are observed rather than assumed:

* ttnn exposes no getter for the loaded subdevice manager or the active stall
  group, so `_SubdeviceRecorder` shadows the four `ttnn.MeshDevice` lifecycle
  methods with forwarding wrappers (`ttnn.MeshDevice` is bound with nanobind's
  `dynamic_attr`, so an instance attribute shadows the class method). The device
  sees exactly the calls it would otherwise see; the test additionally sees which.
* PCC is asserted at every transition step. A mode that merely switched but left
  the ring, the global CB or the stall group inconsistent still produces a tensor;
  only correlating it against the HuggingFace reference proves the context is
  usable after the transition.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field, replace
from typing import Any

import pytest
import torch
from loguru import logger

import ttnn
from models.common.modules.mlp.mlp_2d import MLP2D, MLP2DConfig, _load_input_device_tensor
from models.common.modules.prefetcher import Prefetcher2D
from models.common.tests.modules._hf_reference import get_mlp_weights_from_ref_model
from models.common.tests.modules._mlp_2d_galaxy import (
    assert_mlp_pcc,
    decode_all_reduce_configs,
    decode_reduce_scatter_memcfg,
    decode_ring_config,
    lazy_activation,
    prefill_weight_lazies,
    reference_mlp,
    resources_config,
    weight_lazies,
)
from models.common.tests.modules._wh_galaxy_hardware import (
    compose_2d_sharded_tensor,
    deallocate_module_weights,
    deallocate_tensor,
    exact_tensor_resource,
    galaxy_prefetch_decode_mode_plan,
    galaxy_prefetcher_config,
    galaxy_prefetcher_sender_cores,
    require_galaxy_hardware_resources,
)

_DIM = 8192
_HIDDEN_DIM = 28672
_PREFILL_SEQUENCE = 128
_DEVICE_PARAMS = {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}

# decode -> prefill -> decode -> prefill -> decode, twice over, so the whole cycle
# repeats and the seam between the repeats is itself a decode -> decode; the tail
# adds the repeated prefill the plan lists separately.
_TRANSITION_MATRIX = ("decode", "prefill", "decode", "prefill", "decode") * 2 + ("prefill", "prefill")


# =============================================================================
# Payload - the qualified MLP2D geometry, built once per test
# =============================================================================


@dataclass
class _Payload:
    mesh_device: Any
    resources_config: Any
    decode_ring: dict
    reduce_scatter_memcfg: Any
    all_reduce_output_memcfg: Any
    lazies: tuple
    prefill_lazies: tuple
    decode_x: torch.Tensor
    decode_expected: torch.Tensor
    prefill_x: torch.Tensor
    prefill_expected: torch.Tensor
    activation_dtype: Any = ttnn.bfloat8_b

    @property
    def prefetch_weights(self) -> tuple[tuple[str, Any], ...]:
        lazy_w1, lazy_w2, lazy_w3 = self.lazies
        return (
            ("mlp.w1", lazy_w1.get_device_weight()),
            ("mlp.w3", lazy_w3.get_device_weight()),
            ("mlp.w2", lazy_w2.get_device_weight()),
        )

    def expected(self, mode: str) -> torch.Tensor:
        return self.decode_expected if mode == "decode" else self.prefill_expected

    def activation(self, mode: str) -> torch.Tensor:
        return self.decode_x if mode == "decode" else self.prefill_x

    def plan(self, mode: str):
        return self.resources_config.decode if mode == "decode" else self.resources_config.prefill


@torch.no_grad()
def _build_payload(mesh_device, *, seed: int) -> _Payload:
    torch.manual_seed(seed)
    reference = reference_mlp(_DIM, _HIDDEN_DIM)
    w1, w2, w3 = get_mlp_weights_from_ref_model(reference)
    decode_x = torch.randn(1, 1, 32, _DIM, dtype=torch.bfloat16)
    prefill_x = torch.randn(1, 1, _PREFILL_SEQUENCE, _DIM, dtype=torch.bfloat16)
    weight_dtype = ttnn.bfloat8_b
    ring = decode_ring_config(_DIM, _HIDDEN_DIM)
    reduce_scatter_memcfg = decode_reduce_scatter_memcfg()
    all_reduce_output_memcfg, all_reduce_buffer_memcfg = decode_all_reduce_configs(_DIM)
    return _Payload(
        mesh_device=mesh_device,
        resources_config=resources_config(
            mesh_device,
            _DIM,
            _HIDDEN_DIM,
            decode_w2_input_memcfg=ring["decode_w2_input_memcfg"],
            decode_reduce_scatter_memcfg=reduce_scatter_memcfg,
            decode_all_reduce_buffer_memcfg=all_reduce_buffer_memcfg,
        ),
        decode_ring=ring,
        reduce_scatter_memcfg=reduce_scatter_memcfg,
        all_reduce_output_memcfg=all_reduce_output_memcfg,
        lazies=weight_lazies(w1, w2, w3, mesh_device, weight_dtype),
        prefill_lazies=prefill_weight_lazies(w1, w2, w3, mesh_device, weight_dtype),
        decode_x=decode_x,
        decode_expected=reference(decode_x),
        prefill_x=prefill_x,
        prefill_expected=reference(prefill_x),
    )


def _make_resources(payload: _Payload):
    return require_galaxy_hardware_resources(
        payload.mesh_device,
        config=payload.resources_config,
        prefetch_weights=payload.prefetch_weights,
    )


def _make_module(payload: _Payload, resources) -> MLP2D:
    """One MLP2D wired for both modes - the decode ring plus the prefill weights.

    The MLP suite splits these across two tests because each pins one mode; the
    transition matrix needs a single module that can serve either.
    """
    lazy_w1, lazy_w2, lazy_w3 = payload.lazies
    prefill_w1, prefill_w2, prefill_w3 = payload.prefill_lazies
    return MLP2D.from_config(
        MLP2DConfig(
            w1=lazy_w1,
            w2=lazy_w2,
            w3=lazy_w3,
            prefill_w1=prefill_w1,
            prefill_w2=prefill_w2,
            prefill_w3=prefill_w3,
            mesh_device=payload.mesh_device,
            tt_ccl=resources.ccl,
            collective_resource_selector=exact_tensor_resource,
            w1_w3_memcfg=lazy_w1.memory_config,
            w2_memcfg=lazy_w2.memory_config,
            **payload.decode_ring,
            ff1_out_reduce_scatter_memcfg=payload.reduce_scatter_memcfg,
            ff2_out_reduce_scatter_memcfg=payload.all_reduce_output_memcfg,
            sharded_attn_input_memcfg=ttnn.DRAM_MEMORY_CONFIG,
            decode_prefetch_context=resources.prefetch_context("decode"),
            prefill_prefetch_context=resources.prefetch_context("prefill"),
            activation_dtype=payload.activation_dtype,
            ccl_dtype=payload.activation_dtype,
            mul_dtype=payload.activation_dtype,
        )
    )


# =============================================================================
# Subdevice lifecycle recorder
# =============================================================================


@dataclass(frozen=True)
class _MeshCall:
    name: str
    payload: Any
    result: Any


@dataclass
class _SubdeviceRecorder:
    """Observe the real subdevice-manager and stall-group calls made on the mesh.

    ttnn has no getter for the loaded manager or the active stall group, so the
    only way to assert the transition contract on silicon is at the call
    boundary. Each wrapper forwards to the captured bound method, so the device
    sees an unchanged call sequence.
    """

    mesh_device: Any
    events: list[_MeshCall] = field(default_factory=list)
    _originals: dict = field(default_factory=dict)

    _METHODS = (
        "create_sub_device_manager",
        "load_sub_device_manager",
        "set_sub_device_stall_group",
        "reset_sub_device_stall_group",
        "clear_loaded_sub_device_manager",
        "remove_sub_device_manager",
    )

    def install(self) -> "_SubdeviceRecorder":
        for name in self._METHODS:
            original = getattr(self.mesh_device, name)
            self._originals[name] = original
            setattr(self.mesh_device, name, self._wrap(name, original))
        return self

    def uninstall(self) -> None:
        for name, original in self._originals.items():
            try:
                delattr(self.mesh_device, name)
            except AttributeError:
                setattr(self.mesh_device, name, original)
        self._originals.clear()

    def clear(self) -> None:
        self.events.clear()

    def named(self, name: str) -> list[_MeshCall]:
        return [event for event in self.events if event.name == name]

    def _wrap(self, name: str, original):
        def wrapper(*args, **kwargs):
            result = original(*args, **kwargs)
            payload = args[0] if args else None
            self.events.append(_MeshCall(name, tuple(payload) if isinstance(payload, list) else payload, result))
            return result

        return wrapper


def _assert_activation(recorder: _SubdeviceRecorder, prefetcher: Prefetcher2D, payload: _Payload, mode: str) -> None:
    """Assert the device really moved to `mode`'s manager, stall group and producer."""

    loads = recorder.named("load_sub_device_manager")
    stalls = recorder.named("set_sub_device_stall_group")
    assert loads, f"activating {mode} loaded no subdevice manager"
    assert (
        loads[-1].payload is prefetcher.context(mode).sub_device_manager_id
    ), f"activating {mode} loaded the wrong subdevice manager"
    assert stalls, f"activating {mode} set no stall group"
    assert (
        stalls[-1].payload == payload.plan(mode).stall_group
    ), f"{mode} stall group is {stalls[-1].payload}, expected {payload.plan(mode).stall_group}"
    assert prefetcher.active_mode == mode
    # The DRAM prefetch producer only exists in decode; prefill must have stopped it.
    assert (prefetcher.prefetch_result is not None) == (mode == "decode")


# =============================================================================
# Invocation
# =============================================================================


def _invoke(
    module: MLP2D,
    resources,
    payload: _Payload,
    *,
    mode: str,
    case: str,
    recorder: _SubdeviceRecorder | None = None,
) -> float:
    mesh_device = payload.mesh_device
    config = module.config
    input_dtype = config.decode_activation_dtype if mode == "decode" else config.prefill_activation_dtype
    device_input = _load_input_device_tensor(
        lazy_activation(payload.activation(mode), mesh_device, input_dtype), config, mode
    )
    if recorder is not None:
        recorder.clear()
    resources.activate(mode)
    if recorder is not None:
        _assert_activation(recorder, resources.prefetcher, payload, mode)
    output = module(device_input, mode=mode)
    try:
        resources.synchronize(mode)
        actual = compose_2d_sharded_tensor(output, mesh_device)
        pcc = assert_mlp_pcc(payload.expected(mode), actual, case=case)
        logger.info(f"[gap2] {case}: PCC={pcc:.7f}")
        print(f"[gap2] {case}: PCC={pcc:.7f}")
        return pcc
    finally:
        deallocate_tensor(output)
        deallocate_tensor(device_input)


def _release(module: MLP2D | None) -> None:
    deallocate_module_weights(module, "w1", "w2", "w3", "prefill_w1", "prefill_w2", "prefill_w3")


def _release_payload(payload: _Payload) -> None:
    """Release the payload's device weights without going through a module.

    The leak-detector test has to drop its MLP2D reference mid-test, so it cannot
    release through `deallocate_module_weights`. Only already-materialized
    LazyWeights are touched; asking for the device weight here would allocate one
    just to free it.
    """
    seen: set[int] = set()
    for lazy in (*payload.lazies, *payload.prefill_lazies):
        tensor = getattr(lazy, "_value", None)
        if tensor is not None and id(tensor) not in seen:
            seen.add(id(tensor))
            deallocate_tensor(tensor)


# =============================================================================
# Test 1 - the sealed resources are real on device
# =============================================================================


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=True)
@torch.no_grad()
def test_prefetcher_2d_wh_galaxy_sealed_resources_are_real_on_device(mesh_device):
    """Sealing must publish the real device addresses of the registered weights.

    Nothing else proves this: `weight_addresses` is what the DRAM prefetch
    producer streams from, and on a mock every address is whatever the fake
    returned. Here they are compared against the buffers themselves and against
    the packed address tensor read back off the 12 sender cores.
    """
    payload = _build_payload(mesh_device, seed=0)
    resources = _make_resources(payload)
    try:
        prefetcher = resources.prefetcher
        registered = payload.prefetch_weights
        assert prefetcher.initialized and prefetcher.sealed
        assert prefetcher.mesh_device is mesh_device

        prefill_context = prefetcher.context("prefill")
        decode_context = prefetcher.context("decode")
        for context, mode in ((prefill_context, "prefill"), (decode_context, "decode")):
            assert context.mode == mode
            assert context.mesh_device is mesh_device
            assert context.sub_device_manager_id is not None
            assert context.worker_sub_device_id == payload.plan(mode).worker_sub_device_id
            assert context.stall_group == payload.plan(mode).stall_group
            assert context.weight_addresses is not None
            assert context.weight_address_metadata is not None

        # Only decode streams weights through the ring, so only decode owns the
        # global CB and the borrowed weight tuple. Prefill runs plain DRAM
        # linears, so `global_cb is None` there is the contract, not a gap.
        assert decode_context.global_cb is not None
        assert decode_context.weights == tuple(tensor for _, tensor in registered)
        assert prefill_context.global_cb is None
        assert prefill_context.weights == ()

        # The addresses are the whole point of sealing: every entry must be the
        # registered tensor's actual device buffer address, in registration order.
        addresses = dict(decode_context.weight_addresses)
        assert list(addresses) == [name for name, _ in registered]
        for name, tensor in registered:
            assert addresses[name] == tensor.buffer_address(), f"{name} sealed a stale address"
        print(f"[gap2] sealed weight addresses: {addresses}")

        metadata = decode_context.weight_address_metadata
        assert metadata is prefill_context.weight_address_metadata
        assert metadata.dtype == ttnn.uint32
        assert metadata.layout == ttnn.ROW_MAJOR_LAYOUT
        assert tuple(metadata.shape) == (prefetcher.config.address_repeat_count, len(registered))
        shard_spec = metadata.memory_config().shard_spec
        assert shard_spec.grid == galaxy_prefetcher_sender_cores()
        assert shard_spec.grid.num_cores() == prefetcher.config.address_repeat_count == 12
        assert tuple(shard_spec.shape) == (1, len(registered))

        # Read the packed address tensor back off the senders: every sender core on
        # every device must hold the same address vector the seal published.
        readback = ttnn.to_torch(metadata, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        rows = readback.reshape(-1, len(registered)).to(torch.int64)
        expected = torch.tensor([tensor.buffer_address() for _, tensor in registered], dtype=torch.int64)
        assert rows.shape[0] == 32 * prefetcher.config.address_repeat_count
        assert bool((rows == expected).all()), f"address tensor mismatch: {rows.unique(dim=0).tolist()}"

        # __post_init__ invariants, on the configuration that actually reached silicon.
        config = prefetcher.config
        assert config.expected_weight_count == len(registered) == len(prefetcher.borrowed_weights)
        assert 0 < config.address_repeat_count <= len(config.sender_receiver_mapping)
        assert config.expected_weight_count % config.prefetch_num_layers == 0
        assert prefetcher.resolved_global_cb_size == config.global_cb_size
    finally:
        resources.cleanup()
        _release(None)


# =============================================================================
# Test 2 - the mode-transition matrix
# =============================================================================


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=True)
@torch.no_grad()
def test_prefetcher_2d_wh_galaxy_mode_transition_matrix(mesh_device):
    """One sealed Prefetcher2D, one MLP2D, every transition the plan names.

    decode->prefill, prefill->decode, repeated decode and repeated prefill, each
    with a real MLP2D invocation and a PCC assertion, so a context that switched
    but is no longer usable fails here rather than silently degrading.
    """
    payload = _build_payload(mesh_device, seed=0)
    resources = _make_resources(payload)
    recorder = _SubdeviceRecorder(mesh_device).install()
    module = None
    try:
        module = _make_module(payload, resources)
        previous = None
        results = []
        for step, mode in enumerate(_TRANSITION_MATRIX):
            label = f"{previous or 'cold'}->{mode}"
            pcc = _invoke(
                module,
                resources,
                payload,
                mode=mode,
                case=f"step {step:02d} {label}",
                recorder=recorder,
            )
            results.append((step, label, pcc))
            previous = mode
        transitions = {label for _, label, _ in results}
        assert "decode->prefill" in transitions
        assert "prefill->decode" in transitions
        assert "decode->decode" in transitions
        assert "prefill->prefill" in transitions
        print("[gap2] transition matrix: " + "; ".join(f"{s:02d} {label} pcc={pcc:.7f}" for s, label, pcc in results))
    finally:
        recorder.uninstall()
        try:
            resources.cleanup()
        finally:
            _release(module)


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=True)
@torch.no_grad()
def test_prefetcher_2d_wh_galaxy_failed_transition_rolls_back_on_device(mesh_device, expect_error):
    """A failure *during* a transition must leave the previous mode usable.

    The host suite proves the rollback bookkeeping against a mock. Here the
    failure is injected into the same seam `Prefetcher2D.__init__` exposes
    (`dram_prefetch_start`), so the rollback path really re-loads the prefill
    subdevice manager and re-sets its stall group on silicon - and the proof that
    it worked is that the next prefill invocation still hits PCC.
    """
    payload = _build_payload(mesh_device, seed=0)
    resources = _make_resources(payload)
    recorder = _SubdeviceRecorder(mesh_device).install()
    module = None
    try:
        module = _make_module(payload, resources)
        prefetcher = resources.prefetcher
        _invoke(module, resources, payload, mode="prefill", case="pre-failure prefill", recorder=recorder)

        original_start = prefetcher._dram_prefetch_start

        def failing_start(context):
            raise RuntimeError("injected dram prefetch start failure")

        recorder.clear()
        prefetcher._dram_prefetch_start = failing_start
        try:
            with expect_error(RuntimeError, "injected dram prefetch start failure"):
                resources.activate("decode")
        finally:
            prefetcher._dram_prefetch_start = original_start

        # Rolled back to prefill: no producer, prefill manager reloaded, prefill stall group.
        assert prefetcher.active_mode == "prefill"
        assert prefetcher.prefetch_result is None
        loads = recorder.named("load_sub_device_manager")
        stalls = recorder.named("set_sub_device_stall_group")
        assert loads[-1].payload is prefetcher.context("prefill").sub_device_manager_id
        assert stalls[-1].payload == payload.plan("prefill").stall_group

        # The rolled-back mode is not merely recorded, it still computes.
        _invoke(module, resources, payload, mode="prefill", case="post-failure prefill", recorder=recorder)
        # And the transition that failed succeeds once the injected failure is gone.
        _invoke(module, resources, payload, mode="decode", case="post-failure decode", recorder=recorder)
    finally:
        recorder.uninstall()
        try:
            resources.cleanup()
        finally:
            _release(module)


# =============================================================================
# Test 3 - cleanup from each active mode, and the leak detector
# =============================================================================


@pytest.mark.parametrize("active_mode", ["decode", "prefill"])
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=True)
@torch.no_grad()
def test_prefetcher_2d_wh_galaxy_cleanup_from_active_mode_frees_the_mesh(mesh_device, active_mode, expect_error):
    """Clean up with a mode still active, then prove the mesh is genuinely free.

    Returning from `cleanup()` proves nothing. The real assertion is that a
    second Prefetcher2D can be built, sealed and driven to a passing PCC on the
    same mesh in the same process: a leaked subdevice manager, a stall group left
    set, or a global CB whose L1 survived would surface exactly there.
    """
    payload = _build_payload(mesh_device, seed=0)
    first = _make_resources(payload)
    module = second = second_module = None
    try:
        module = _make_module(payload, first)
        _invoke(module, first, payload, mode=active_mode, case=f"first prefetcher {active_mode}")

        prefetcher = first.prefetcher
        assert prefetcher.active_mode == active_mode
        weights = [tensor for _, tensor in payload.prefetch_weights]
        addresses = [tensor.buffer_address() for tensor in weights]
        metadata = prefetcher.context("decode").weight_address_metadata

        first.cleanup()
        first.cleanup()  # idempotent, from an active mode

        # Registered weights are borrowed, never owned: cleanup must not release them.
        for tensor, address in zip(weights, addresses):
            assert tensor.is_allocated(), "cleanup released a registered weight"
            assert tensor.buffer_address() == address, "cleanup moved a registered weight"
        # The address metadata is owned, so it must be gone.
        assert not metadata.is_allocated(), "cleanup leaked the packed address tensor"
        # The owner's own bookkeeping must be empty afterwards.
        assert prefetcher.owned_resources == ()
        assert prefetcher.prefetch_result is None
        with expect_error(RuntimeError, "cleaned up"):
            prefetcher.context("decode")

        # Drop every consumer handle before the leak detector runs. `cleanup()`
        # deallocates the packed address tensor explicitly but not the global
        # circular buffer: ttnn exposes no free for one, so its L1 is released by
        # RAII when the last `global_circular_buffer` handle dies - and every
        # module handed a `Prefetcher2DContext` holds one (MLP2D keeps it as
        # `decode_prefetch_context`). Without this drop the second owner's
        # `create_global_circular_buffer` fails on L1. See finding F1 in
        # tttv2_milestone_a_gap2_evidence/REPORT.md; isolated in probe 02, where
        # holding nothing but the global CB object reproduces it and holding only
        # the address metadata does not.
        first_prefetcher_id = id(prefetcher)
        module = None
        del prefetcher
        gc.collect()

        # Leak detector: a second full owner on the same mesh, in the same process.
        second = _make_resources(payload)
        assert id(second.prefetcher) != first_prefetcher_id
        second_module = _make_module(payload, second)
        _invoke(second_module, second, payload, mode=active_mode, case=f"second prefetcher {active_mode}")
        _invoke(
            second_module,
            second,
            payload,
            mode="prefill" if active_mode == "decode" else "decode",
            case=f"second prefetcher {active_mode} -> other mode",
        )
    finally:
        try:
            if second is not None:
                second.cleanup()
        finally:
            _release_payload(payload)


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=True)
@torch.no_grad()
def test_prefetcher_2d_wh_galaxy_context_manager_cleanup_leaves_mesh_reusable(mesh_device, expect_error):
    """`with Prefetcher2D(...)` must remove its managers when the body raises.

    Cheap by construction: no kernel runs. The proof that the managers really
    went away is that a second prefetcher initializes and seals afterwards.
    """
    payload = _build_payload(mesh_device, seed=0)
    registered = payload.prefetch_weights
    config = galaxy_prefetcher_config(mesh_device, payload.resources_config, len(registered))
    recorder = _SubdeviceRecorder(mesh_device).install()
    try:
        prefetcher = Prefetcher2D(config)
        recorder.clear()
        with expect_error(RuntimeError, "body failure"):
            with prefetcher:
                assert prefetcher.initialized
                raise RuntimeError("body failure")
        created = recorder.named("create_sub_device_manager")
        removed = recorder.named("remove_sub_device_manager")
        assert len(created) == 2, "the context manager did not create both mode managers"
        assert {id(event.payload) for event in removed} == {id(event.result) for event in created}
        with expect_error(RuntimeError, "cleaned up"):
            prefetcher.initialize()

        second = Prefetcher2D(galaxy_prefetcher_config(mesh_device, payload.resources_config, len(registered)))
        try:
            second.initialize()
            for name, tensor in registered:
                second.register_weight(name, tensor)
            second.seal()
            assert second.sealed
            assert dict(second.context("decode").weight_addresses) == {
                name: tensor.buffer_address() for name, tensor in registered
            }
        finally:
            second.cleanup()
            second.cleanup()
    finally:
        recorder.uninstall()


# =============================================================================
# Test 4 - registration and sealing rejections on real tensors
# =============================================================================


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=True)
@torch.no_grad()
def test_prefetcher_2d_wh_galaxy_registration_and_sealing_rejections(mesh_device, expect_error):
    """The host rejection contracts, re-run where the tensors are real.

    Every case aborts before a kernel runs, so this stays cheap.

    Not covered here, deliberately: registering a tensor that belongs to a
    *different* mesh. The only second mesh available on a reserved 32-device
    Galaxy is a submesh of this one, and leaving one alive aborts the fixture
    teardown with `MeshDevice cq ID 0 is in use by parent mesh ID 0`; the
    rejection itself is mesh-agnostic and covered on host.
    """
    payload = _build_payload(mesh_device, seed=0)
    registered = payload.prefetch_weights
    prefetcher = Prefetcher2D(galaxy_prefetcher_config(mesh_device, payload.resources_config, len(registered)))
    try:
        prefetcher.initialize()
        first_name, first_tensor = registered[0]
        prefetcher.register_weight(first_name, first_tensor)

        with expect_error(ValueError, "already registered"):
            prefetcher.register_weight(first_name, registered[1][1])
        with expect_error(ValueError, "distinct device buffers"):
            prefetcher.register_weight("alias", first_tensor)
        with expect_error(TypeError, "materialized"):
            prefetcher.register_weight("host-side", object())
        with expect_error(ValueError, "cannot be empty"):
            prefetcher.register_weight("", registered[1][1])
        with expect_error(RuntimeError, "expected 3 registered weights"):
            prefetcher.seal()

        for name, tensor in registered[1:]:
            prefetcher.register_weight(name, tensor)
        with expect_error(ValueError, "exceeds the resolved configuration"):
            prefetcher.register_weight("one-too-many", registered[0][1])
        prefetcher.seal()
        with expect_error(RuntimeError, "sealed"):
            prefetcher.register_weight("after-seal", registered[0][1])
        with expect_error(ValueError, "unsupported prefetcher mode"):
            prefetcher.context("trace")

        # borrow_context must reject a subdevice policy that is not exactly the
        # sealed one. On device this matters more than on a mock: the resource
        # owner that borrows drives load_sub_device_manager afterwards, so a
        # tolerated mismatch would put real subdevice state out of step.
        decode_plan = payload.plan("decode")
        borrowed = prefetcher.borrow_context(
            "decode",
            sub_devices=decode_plan.sub_devices,
            worker_sub_device_id=decode_plan.worker_sub_device_id,
            stall_group=decode_plan.stall_group,
            local_l1_size=decode_plan.local_l1_size,
        )
        assert borrowed is prefetcher.context("decode")
        for mismatch in (
            {"sub_devices": decode_plan.sub_devices[:1]},
            {"worker_sub_device_id": ttnn.SubDeviceId(0)},
            {"stall_group": (ttnn.SubDeviceId(0),)},
            {"local_l1_size": decode_plan.local_l1_size + 1024},
        ):
            kwargs = {
                "sub_devices": decode_plan.sub_devices,
                "worker_sub_device_id": decode_plan.worker_sub_device_id,
                "stall_group": decode_plan.stall_group,
                "local_l1_size": decode_plan.local_l1_size,
                **mismatch,
            }
            with expect_error(ValueError, "do not match the Prefetcher2D subdevice policy"):
                prefetcher.borrow_context("decode", **kwargs)
    finally:
        prefetcher.cleanup()


# =============================================================================
# Test 5 - Attention2D alongside an active prefetch producer
# =============================================================================


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=True)
@torch.no_grad()
def test_prefetcher_2d_wh_galaxy_attention_decode_with_active_prefetch(mesh_device):
    """Attention2D decode on the production prefetch decode subdevice partition.

    Attention2D is the one 2D module that has never run alongside a prefetcher:
    it uses the CCL-only resource helper, so its worker subdevice is the whole
    compute grid. Production narrows it (`galaxy_prefetch_decode_mode_plan`:
    senders on x in {0, 4}, workers on x in {1, 2, 3} u {5, 6}) while a DRAM
    prefetch producer runs. This drives exactly that combination.

    The attention geometry is imported unchanged from the qualified attention
    suite and only the decode mode plan is swapped, so anything that fails here
    is the subdevice partition, not a transcription error.
    """
    from models.common.tests.modules.attention import test_attention_2d_wh_galaxy as attention_suite

    spec = attention_suite._MODEL_SPECS[0]
    torch.manual_seed(17)
    reference = attention_suite._reference_attention(spec)
    wqkv, wo, q_norm, k_norm, wqkv_bias = attention_suite.get_attention_weights_from_ref_model(
        reference.attention, num_devices=attention_suite._MESH_SHAPE[0]
    )
    assert wqkv_bias is None
    wqkv = wqkv[0, 0].to(torch.bfloat16).contiguous()
    wo = wo[0, 0].to(torch.bfloat16).contiguous()
    decode_ring = attention_suite._decode_ring_config(spec)
    decode_all_reduce = attention_suite._decode_all_reduce_config(spec, mesh_device, decode_ring["ring_cores"])
    dram_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(mesh_device.dram_grid_size().x - 1, 0))}
    )

    def dram_sharded(local_k: int, local_n: int) -> ttnn.MemoryConfig:
        import math

        padded_n = math.ceil(local_n / (32 * 12)) * 32 * 12
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram_grid, (local_k, padded_n // 12), ttnn.ShardOrientation.ROW_MAJOR),
        )

    wqkv_mapper = attention_suite._mesh_mapper(ttnn.PlacementShard(1), ttnn.PlacementShard(0))
    wo_mapper = attention_suite._mesh_mapper(ttnn.PlacementShard(0), ttnn.PlacementShard(1))
    lazy_wqkv = attention_suite._lazy_weight(
        wqkv,
        mesh_device,
        wqkv_mapper,
        memory_config=dram_sharded(
            spec.dim // attention_suite._MESH_SHAPE[1], spec.qkv_size // attention_suite._MESH_SHAPE[0]
        ),
    )
    lazy_wo = attention_suite._lazy_weight(
        wo,
        mesh_device,
        wo_mapper,
        memory_config=dram_sharded(
            spec.dim // attention_suite._MESH_SHAPE[0], spec.dim // attention_suite._MESH_SHAPE[1]
        ),
    )
    prefill_wqkv = attention_suite._lazy_weight(wqkv, mesh_device, wqkv_mapper)
    prefill_wo = attention_suite._lazy_weight(wo, mesh_device, wo_mapper)

    base_config = attention_suite._resources_config(mesh_device, spec, decode_all_reduce)
    prefetch_config = replace(
        base_config,
        decode=galaxy_prefetch_decode_mode_plan(base_config.decode.collectives),
    )
    resources = require_galaxy_hardware_resources(
        mesh_device,
        config=prefetch_config,
        prefetch_weights=(
            ("attn.wqkv", lazy_wqkv.get_device_weight()),
            ("attn.wo", lazy_wo.get_device_weight()),
        ),
    )
    module = binding = None
    try:
        module = attention_suite._make_module(
            spec,
            mesh_device,
            resources,
            lazy_wqkv,
            lazy_wo,
            prefill_wqkv,
            prefill_wo,
            decode_ring,
            decode_all_reduce,
            q_norm,
            k_norm,
        )
        binding = attention_suite._make_cache(module, mesh_device)
        module.bind_kv_cache(binding)
        reference.reset_cache_to_zeros(
            attention_suite._BATCH_SIZE, spec.n_kv_heads, attention_suite._DECODE_POSITIONS[0]
        )
        position = attention_suite._DECODE_POSITIONS[0]
        resources.activate("decode")
        assert resources.prefetcher.prefetch_result is not None, "no DRAM prefetch producer is running"
        x = torch.randn(1, 1, attention_suite._BATCH_SIZE, spec.dim, dtype=torch.bfloat16) * 0.05
        positions = torch.full((attention_suite._BATCH_SIZE,), position, dtype=torch.long)
        expected = attention_suite._reference_decode(reference, x, position)
        tt_x = attention_suite._to_device_input(x, mesh_device)
        tt_positions = ttnn.from_torch(
            positions[: module.config.users_per_column].to(torch.int32),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        output = None
        try:
            output = module.decode_forward(tt_x, None, attention_suite.DecodeMetadata(tt_positions))
            resources.synchronize("decode")
            attention_suite._assert_pcc(
                expected,
                attention_suite._compose_output(output, mesh_device, spec),
                "attention decode with active prefetch",
            )
        finally:
            if output is not None and not module.output_is_borrowed(output):
                deallocate_tensor(output)
            deallocate_tensor(tt_positions)
            deallocate_tensor(tt_x)
    finally:
        try:
            resources.cleanup()
            if module is not None:
                module._hardware_collectives.cleanup()
                module.close()
                deallocate_module_weights(module, "wqkv", "wo", "prefill_wqkv", "prefill_wo")
                for norm in (module._q_norm, module._k_norm):
                    deallocate_tensor(getattr(norm, "weight", None))
            if binding is not None:
                deallocate_tensor(binding.keys)
                deallocate_tensor(binding.values)
        finally:
            del reference, wqkv, wo, q_norm, k_norm
