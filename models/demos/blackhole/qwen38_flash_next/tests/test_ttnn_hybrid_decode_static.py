# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device ownership, cache, sampling, and handoff tests for hybrid decode."""

from __future__ import annotations

import inspect
import itertools
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

import models.demos.blackhole.qwen38_flash_next.ttnn.hybrid_decode as hybrid_module
from models.demos.blackhole.qwen38_flash_next.checkpoint import CHECKPOINT_FILE_MANIFEST_SHA256, INDEX_SHA256
from models.demos.blackhole.qwen38_flash_next.config import CONFIG_SHA256
from models.demos.blackhole.qwen38_flash_next.ttnn.bf4 import Qwen38BF4ResidentSet
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import (
    Qwen38BuildProvenance,
    Qwen38LiveBuildIdentity,
    Qwen38TTNNBuilder,
    Qwen38TTNNBuiltTarget,
    Qwen38TTNNMTPComponents,
    Qwen38TTNNTargetComponents,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import (
    PINNED_CHECKPOINT_REVISION,
    PINNED_TENSOR_MANIFEST_SHA256,
    Qwen38ShardedLogits,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.fixed_five import Qwen38TTNNFixedFiveTarget
from models.demos.blackhole.qwen38_flash_next.ttnn.hybrid_decode import (
    Qwen38HybridCacheMismatchError,
    Qwen38HybridCachePolicy,
    Qwen38HybridConstructionError,
    Qwen38HybridDecodePoisonedError,
    Qwen38HybridFactoryError,
    Qwen38HybridFallbackReason,
    Qwen38HybridMode,
    Qwen38HybridSessionStatus,
    Qwen38HybridStopReason,
    Qwen38HybridTimingPhase,
    Qwen38HybridTokenSource,
    Qwen38TTNNHybridDecodeSession,
    validate_hybrid_decode_static_contract,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.layer import BACKBONE_LAYERS, Qwen38TTNNDecoderLayerAux
from models.demos.blackhole.qwen38_flash_next.ttnn.model import (
    Qwen38TTNNTextModel,
    Qwen38TTNNTextModelOutput,
    Qwen38TTNNTextModelState,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.mtp_decode import (
    Qwen38MTPQSASelectionProof,
    Qwen38MTPSeed,
    Qwen38SpeculativeEmission,
    Qwen38SpeculativeHandoff,
    Qwen38SpeculativeRound,
    Qwen38SpeculativeStatus,
    Qwen38SpeculativeTokenSource,
    Qwen38TargetCommitMode,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.mtp_draft import Qwen38TTNNMTPDraftEngine
from models.demos.blackhole.qwen38_flash_next.ttnn.mtp_mechanics import execute_one_mtp_round
from models.demos.blackhole.qwen38_flash_next.ttnn.sampling import (
    Qwen38SampledTokens,
    Qwen38SamplingParameters,
    Qwen38SamplingTiming,
    Qwen38TTNNHostSampler,
)

PHYSICAL_IDS = (10, 11, 12, 13)
IDENTITY_KEY_LENGTH = 64
EOS = 999
_AUX = tuple(Qwen38TTNNDecoderLayerAux(None, None, False) for _ in range(BACKBONE_LAYERS))


class _Clock:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> int:
        self.value += 10
        return self.value


class _Synchronizer:
    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.calls = 0
        self.fail_on_call = fail_on_call

    def __call__(self) -> None:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError(f"injected synchronization failure {self.calls}")


class _FakeMesh:
    shape = (1, 4)

    def get_num_devices(self) -> int:
        return 4

    def get_device_ids(self) -> list[int]:
        return list(PHYSICAL_IDS)

    @staticmethod
    def get_device_id(coordinate) -> int:
        row, column = tuple(coordinate)
        assert row == 0
        return PHYSICAL_IDS[column]


class _BackingLocal:
    def __init__(self, mesh: _FakeMesh, coordinate: tuple[int, int], address: int) -> None:
        self._mesh = mesh
        self._coordinate = coordinate
        self._address = address

    def device(self) -> _FakeMesh:
        return self._mesh

    def device_coords(self):
        return (self._coordinate,)

    def buffer_address(self) -> int:
        return self._address


class _BackingTopology:
    @staticmethod
    def distribution_shape():
        return (1, 4)

    @staticmethod
    def mesh_coords():
        return ((0, 0), (0, 1), (0, 2), (0, 3))


class _FakeMeshContract:
    physical_ids = PHYSICAL_IDS

    def __init__(self) -> None:
        self.validation_calls = 0

    def validate_mesh(self, mesh: _FakeMesh) -> None:
        assert type(mesh) is _FakeMesh
        self.validation_calls += 1


class _Root:
    _ids = itertools.count(1)

    def __init__(self, token: int, position: int) -> None:
        self.token = token
        self.position = position
        self.identifier = next(self._ids)
        self.released = False
        self._mesh = _FakeMesh()
        self.locals = tuple(
            _BackingLocal(
                self._mesh,
                (0, column),
                self.identifier * 0x10000 + column * 0x1000,
            )
            for column in range(4)
        )

    def tensor_id(self) -> int:
        return self.identifier

    def is_allocated(self) -> bool:
        return not self.released

    def device(self) -> _FakeMesh:
        return self._mesh

    @staticmethod
    def tensor_topology():
        return _BackingTopology()


class _LogitTensor:
    _ids = itertools.count(10_000)

    def __init__(self) -> None:
        self.identifier = next(self._ids)
        self.allocated = True

    def tensor_id(self) -> int:
        return self.identifier

    def is_allocated(self) -> bool:
        return self.allocated


@dataclass(frozen=True)
class _MTPState:
    position: int


def _provenance() -> Qwen38BuildProvenance:
    return Qwen38BuildProvenance(
        checkpoint_revision=PINNED_CHECKPOINT_REVISION,
        checkpoint_index_sha256=INDEX_SHA256,
        checkpoint_config_sha256=CONFIG_SHA256,
        checkpoint_file_manifest_sha256=CHECKPOINT_FILE_MANIFEST_SHA256,
        checkpoint_hash_manifest_sha256=PINNED_TENSOR_MANIFEST_SHA256,
        tt_metal_sha="1" * 40,
        ttnn_runtime_sha256="2" * 64,
    )


def _proof(position: int, serial: int) -> Qwen38MTPQSASelectionProof:
    return Qwen38MTPQSASelectionProof(
        layer_index=0,
        epoch=serial,
        source_view_id=serial,
        result_view_id=serial,
        source_position=position,
        tail_start=position - position % 4,
        complete_token_count=0,
        complete_indices_key=None,
        valid_token_count=position + 1 - (position - position % 4),
    )


class _TargetBackend:
    def __init__(
        self,
        model: Qwen38TTNNTextModel,
        predictions: tuple[int, ...],
        events: list[tuple],
        *,
        emit_internal_phase_markers: bool,
    ) -> None:
        self.model = model
        self.predictions = deque(predictions)
        self.events = events
        self.emit_internal_phase_markers = emit_internal_phase_markers
        self.calls = 0
        self.reset_calls = 0
        self.release_calls = 0
        self.fail_reset = False
        self.fail_release = False

    def state(self, position: int) -> Qwen38TTNNTextModelState:
        return Qwen38TTNNTextModelState(position, (), self.model._state_owner)

    def allocate_state(self) -> Qwen38TTNNTextModelState:
        return self.state(0)

    def reset_state(self, state: Qwen38TTNNTextModelState) -> Qwen38TTNNTextModelState:
        assert state._owner is self.model._state_owner
        self.reset_calls += 1
        if self.fail_reset:
            raise RuntimeError("injected target reset failure")
        return self.state(0)

    def release_state(self, state: Qwen38TTNNTextModelState) -> None:
        assert state._owner is self.model._state_owner
        self.release_calls += 1
        if self.fail_release:
            raise RuntimeError("injected target release failure")

    def validate_state(self, state: Qwen38TTNNTextModelState) -> None:
        if type(state) is not Qwen38TTNNTextModelState or state._owner is not self.model._state_owner:
            raise ValueError("foreign target state")

    def forward_decode(self, token_id: int, state: Qwen38TTNNTextModelState, **kwargs) -> Qwen38TTNNTextModelOutput:
        self.calls += 1
        assert kwargs["return_logits"] is kwargs["resolve_greedy"] is False or not (
            kwargs["return_logits"] and kwargs["resolve_greedy"]
        )
        assert kwargs["retain_hidden"] is False
        assert kwargs["retain_hyper_residual"] is True
        assert kwargs["retain_input_state"] is False
        assert kwargs["return_routing"] is False
        phase_observer = kwargs["phase_observer"]
        if phase_observer is not None:
            for layer_index in range(BACKBONE_LAYERS):
                phase_observer(f"before-layer-{layer_index}")
                if self.emit_internal_phase_markers:
                    attention_stage = "gdn" if layer_index % 2 == 0 else "qsa"
                    for stage in (
                        "ple",
                        "attention-gr-read",
                        attention_stage,
                        "attention-gr-write",
                        "mlp-gr-read",
                        "expert-stream-acquire",
                        "moe-forward",
                        "expert-stream-release",
                        "mlp-gr-write",
                        "state-update",
                    ):
                        phase_observer(f"layer-{layer_index}-before-{stage}")
                        if stage == "moe-forward":
                            for moe_stage in (
                                "hidden-all-gather",
                                "router-logits",
                                "router-topk",
                                "shared-partial",
                                "routed-dispatch",
                                "moe-compute-launch",
                                "selective-reduce",
                                "partial-combine",
                                "output-reduce-scatter",
                                "output-release",
                            ):
                                phase_observer(f"layer-{layer_index}-before-{moe_stage}")
                                phase_observer(f"layer-{layer_index}-after-{moe_stage}")
                        phase_observer(f"layer-{layer_index}-after-{stage}")
                phase_observer(f"after-layer-{layer_index}")
            phase_observer("before-final-mixer")
            phase_observer("after-final-mixer")
            if kwargs["return_logits"] or kwargs["resolve_greedy"]:
                phase_observer("before-lm-head")
                phase_observer("after-lm-head")
        root = _Root(token_id, state.position)
        self.events.append(("target", token_id, state.position, root))
        greedy = None
        logits = None
        if kwargs["resolve_greedy"]:
            greedy = torch.tensor([[[self.predictions.popleft()]]], dtype=torch.int64)
        elif kwargs["return_logits"]:
            logits = Qwen38ShardedLogits(_LogitTensor(), (), (1, 1, 1, 248_320))
        return Qwen38TTNNTextModelOutput(
            input_token_id=token_id,
            position=state.position,
            hyper_residual_sharded=root,
            hidden_sharded=None,
            logits=logits,
            greedy_token=greedy,
            state=self.state(state.position + 1),
            layer_aux=_AUX,
        )


class _MTPBackend:
    def __init__(self, engine: Qwen38TTNNMTPDraftEngine, events: list[tuple]) -> None:
        self.engine = engine
        self.events = events
        self.owner: object | None = None
        self.serial = 100
        self.bootstrap_calls: list[tuple[tuple[int, _Root], ...]] = []
        self.advance_calls: list[tuple[tuple[int, _Root], ...]] = []
        self.released_seed_count = 0
        self.released_state_count = 0
        self.fail_bootstrap = False
        self.fail_advance = False

    def claim_runtime_owner(self, owner: object) -> None:
        if self.owner is not None:
            raise RuntimeError("MTP already owned")
        self.owner = owner

    def release_runtime_owner(self, owner: object) -> None:
        if self.owner is not owner:
            raise ValueError("wrong MTP owner")
        self.owner = None

    def transfer_runtime_owner(self, current: object, successor: object) -> None:
        if self.owner is not current:
            raise ValueError("wrong MTP transfer owner")
        self.owner = successor

    def allocate_state(self) -> _MTPState:
        return _MTPState(0)

    @staticmethod
    def state_position(state: _MTPState) -> int:
        if type(state) is not _MTPState:
            raise TypeError("foreign MTP state")
        return state.position

    def _consume(self, rows, base_position: int) -> tuple[tuple[tuple[int, _Root], ...], Qwen38MTPSeed]:
        consumed: list[tuple[int, _Root]] = []
        previous: _Root | None = None
        for token, root in rows:
            if previous is not None:
                assert previous.released, "engine requested a new row before consuming the prior root"
            assert type(root) is _Root and not root.released
            self.events.append(("mtp", token, root))
            root.released = True
            consumed.append((token, root))
            previous = root
        if not consumed:
            raise ValueError("fake MTP stream is empty")
        self.serial += 1
        position = base_position + len(consumed)
        current = consumed[-1][0]
        seed = Qwen38MTPSeed(
            position=position,
            current_token_id=current,
            first_draft_token_id=(current + 100) % 248_320,
            state=_MTPState(position),
            recurrent_residual=object(),
            qsa_selection=_proof(position - 1, self.serial),
            transaction=object(),
        )
        return tuple(consumed), seed

    def bootstrap_shifted_prefill_rows(self, rows, state: _MTPState) -> Qwen38MTPSeed:
        consumed, seed = self._consume(rows, state.position)
        self.bootstrap_calls.append(consumed)
        if self.fail_bootstrap:
            raise RuntimeError("injected bootstrap failure")
        return seed

    def advance_seed_rows(self, seed: Qwen38MTPSeed, rows) -> Qwen38MTPSeed:
        consumed, replacement = self._consume(rows, seed.position)
        self.advance_calls.append(consumed)
        if self.fail_advance:
            raise RuntimeError("injected advance failure")
        return replacement

    def release_seed(self, seed: Qwen38MTPSeed) -> None:
        assert type(seed) is Qwen38MTPSeed
        self.released_seed_count += 1

    def release_state(self, state: _MTPState) -> None:
        assert type(state) is _MTPState
        self.released_state_count += 1


class _SamplerBackend:
    def __init__(self, sampler: Qwen38TTNNHostSampler, predictions: tuple[int, ...], provenance_key: str) -> None:
        self.sampler = sampler
        self.predictions = deque(predictions)
        self.provenance_key = provenance_key
        self.calls: list[
            tuple[Qwen38SamplingParameters, tuple[int, ...], torch.Generator | None, torch.Tensor | None]
        ] = []
        self.closed = False

    def sample(
        self,
        logits,
        parameters,
        *,
        token_histories,
        source_identity_key,
        generator=None,
    ) -> Qwen38SampledTokens:
        del logits
        state = None if generator is None else generator.get_state().clone()
        if generator is not None:
            torch.rand((), generator=generator)
        self.calls.append((parameters, tuple(token_histories), generator, state))
        token = self.predictions.popleft()
        tensor = torch.tensor([[[token]]], dtype=torch.int64)
        return Qwen38SampledTokens(
            token_ids=tensor,
            eos_mask=tensor == EOS,
            parameters=parameters,
            timing=Qwen38SamplingTiming(1, 1, 3),
            source_identity_key=source_identity_key,
            source_provenance_key=self.provenance_key,
            rng_mode="per-call-seed" if generator is None else "request-generator",
        )

    def close(self) -> None:
        self.closed = True


@dataclass
class _Parts:
    built_target: Qwen38TTNNBuiltTarget
    mtp_components: Qwen38TTNNMTPComponents
    provenance: Qwen38BuildProvenance
    fixed_target: Qwen38TTNNFixedFiveTarget
    mtp_engine: Qwen38TTNNMTPDraftEngine
    sampler_owner: Qwen38TTNNHostSampler
    target: _TargetBackend
    mtp: _MTPBackend
    sampler_backend: _SamplerBackend
    events: list[tuple]


@dataclass
class _Fixture:
    session: Qwen38TTNNHybridDecodeSession
    parts: _Parts
    sync: _Synchronizer

    @property
    def target(self) -> _TargetBackend:
        return self.parts.target

    @property
    def mtp(self) -> _MTPBackend:
        return self.parts.mtp

    @property
    def sampler(self) -> _SamplerBackend:
        return self.parts.sampler_backend

    @property
    def events(self) -> list[tuple]:
        return self.parts.events


def _parts(
    monkeypatch,
    *,
    target_predictions: tuple[int, ...] = (),
    sampled_predictions: tuple[int, ...] = (),
    emit_internal_phase_markers: bool = False,
    resident_owner: Qwen38BF4ResidentSet | None = None,
    allocated_context: int | None = None,
) -> _Parts:
    provenance = _provenance()
    identity = Qwen38LiveBuildIdentity(
        provenance=provenance,
        mesh_shape=(1, 4),
        physical_ids=PHYSICAL_IDS,
        collective_topology="Ring",
        dram_bank_ring_order=(6, 5, 4, 3, 2, 1, 0),
        ring_size=7,
    )
    events: list[tuple] = []
    model = object.__new__(Qwen38TTNNTextModel)
    model._state_owner = object()
    model._runtime_owner = None
    model._poisoned_error = None
    model._active_snapshot = None
    model.mesh_device = _FakeMesh()
    model.mesh_contract = _FakeMeshContract()
    if allocated_context is not None:
        model.allocated_context = allocated_context
    model_io = SimpleNamespace(lm_head=object())
    model.model_io = model_io
    model.final_mixer = object()
    layers = tuple(SimpleNamespace() for _ in range(BACKBONE_LAYERS))
    model.layers = layers
    target_backend = _TargetBackend(
        model,
        target_predictions,
        events,
        emit_internal_phase_markers=emit_internal_phase_markers,
    )
    model.allocate_state = target_backend.allocate_state
    model.reset_state = target_backend.reset_state
    model.release_state = target_backend.release_state
    model._validate_state = target_backend.validate_state
    model.forward_decode = target_backend.forward_decode

    expert_streamer = object() if resident_owner is None else resident_owner
    components = Qwen38TTNNTargetComponents(
        identity=identity,
        bf4_cache=object(),
        io_cache=object(),
        expert_streamer=expert_streamer,
        model_io=model_io,
        layers=layers,
        final_mixer=model.final_mixer,
    )
    built = Qwen38TTNNBuiltTarget(model, components)
    mtp_components = Qwen38TTNNMTPComponents(
        identity,
        object(),
        SimpleNamespace(expert_streamer=expert_streamer),
        object(),
    )

    fixed = object.__new__(Qwen38TTNNFixedFiveTarget)
    fixed.model = model
    fixed.components = components
    fixed._identity_key = identity.key
    fixed._transaction_owner = object()
    fixed._active_transaction = None
    fixed._poisoned_error = None
    fixed._closed = False
    fixed.row5_moes = ()
    model._fixed_five_adapter = fixed

    mtp_engine = object.__new__(Qwen38TTNNMTPDraftEngine)
    mtp_engine._identity_key = identity.key
    mtp_engine.identity = identity
    mtp_engine._poisoned_error = None
    mtp_engine._bootstrap_phase_observer = None
    mtp_backend = _MTPBackend(mtp_engine, events)
    mtp_engine.claim_runtime_owner = mtp_backend.claim_runtime_owner
    mtp_engine.release_runtime_owner = mtp_backend.release_runtime_owner
    mtp_engine.transfer_runtime_owner = mtp_backend.transfer_runtime_owner
    mtp_engine.allocate_state = mtp_backend.allocate_state
    mtp_engine.state_position = mtp_backend.state_position
    mtp_engine.bootstrap_shifted_prefill_rows = mtp_backend.bootstrap_shifted_prefill_rows
    mtp_engine.advance_seed_rows = mtp_backend.advance_seed_rows
    mtp_engine.release_seed = mtp_backend.release_seed
    mtp_engine.release_state = mtp_backend.release_state

    sampler = object.__new__(Qwen38TTNNHostSampler)
    sampler.identity_key = identity.key
    sampler.live_identity = identity
    sampler.provenance = provenance
    sampler.lm_head = model_io.lm_head
    sampler_backend = _SamplerBackend(sampler, sampled_predictions, provenance.key)
    sampler.sample = sampler_backend.sample
    sampler.close = sampler_backend.close

    def deallocate(tensor) -> None:
        if isinstance(tensor, _Root):
            if tensor.released:
                raise AssertionError("double-free fake root")
            tensor.released = True
        else:
            if not tensor.allocated:
                raise AssertionError("double-free fake logits")
            tensor.allocated = False

    monkeypatch.setattr(hybrid_module.ttnn, "deallocate", deallocate)
    return _Parts(
        built,
        mtp_components,
        provenance,
        fixed,
        mtp_engine,
        sampler,
        target_backend,
        mtp_backend,
        sampler_backend,
        events,
    )


def _fixture(
    monkeypatch,
    *,
    target_predictions: tuple[int, ...] = (),
    sampled_predictions: tuple[int, ...] = (),
    emit_internal_phase_markers: bool = False,
    resident_owner: Qwen38BF4ResidentSet | None = None,
    owns_built_graph: bool = False,
    allocated_context: int | None = None,
) -> _Fixture:
    parts = _parts(
        monkeypatch,
        target_predictions=target_predictions,
        sampled_predictions=sampled_predictions,
        emit_internal_phase_markers=emit_internal_phase_markers,
        resident_owner=resident_owner,
        allocated_context=allocated_context,
    )
    sync = _Synchronizer()
    session = Qwen38TTNNHybridDecodeSession(
        parts.built_target,
        parts.mtp_components,
        expected_provenance=parts.provenance,
        expected_physical_ids=PHYSICAL_IDS,
        expected_identity_key=parts.built_target.components.identity.key,
        eos_token_ids=(EOS,),
        synchronize=sync,
        clock_ns=_Clock(),
        _test_collaborators=hybrid_module._Qwen38HybridTestCollaborators(
            parts.fixed_target,
            parts.mtp_engine,
            parts.sampler_owner,
        ),
        _owns_built_graph=owns_built_graph,
    )
    return _Fixture(session, parts, sync)


def _resident_owner(monkeypatch) -> tuple[Qwen38BF4ResidentSet, tuple[_Root, ...]]:
    class Contract:
        @staticmethod
        def validate_mesh(_mesh) -> None:
            return None

    class Cache:
        mesh_contract = Contract()

        @staticmethod
        def _validate_layer_request(namespace, layer_index) -> None:
            if (namespace, layer_index) not in {("backbone", 0), ("mtp", 0)}:
                raise ValueError("invalid synthetic resident key")

        @staticmethod
        def load_layer(_mesh, *, layer_index, namespace):
            position = 0 if namespace == "backbone" else 1
            return (_Root(layer_index, position), _Root(layer_index, position))

    monkeypatch.setattr(hybrid_module.ttnn, "get_device_tensors", lambda tensor: tensor.locals)
    owner = Qwen38BF4ResidentSet(Cache(), object())
    owner._preload((("backbone", 0), ("mtp", 0)))
    tensors = tuple(
        handle.tensor for key in owner.resident_layers for handle in owner._slots[key] if handle is not None
    )
    return owner, tensors


def test_static_capabilities_and_public_output_transfer_only() -> None:
    validate_hybrid_decode_static_contract()
    ordinary = Qwen38TTNNHybridDecodeSession.capabilities("ordinary")
    mtp = Qwen38TTNNHybridDecodeSession.capabilities("mtp")
    assert (ordinary.greedy, ordinary.official_thinking_sampling, ordinary.official_non_thinking_sampling) == (
        True,
        True,
        True,
    )
    assert not ordinary.speculative
    assert (mtp.greedy, mtp.speculative) == (True, True)
    assert not mtp.official_thinking_sampling and not mtp.official_non_thinking_sampling
    source = inspect.getsource(hybrid_module)
    assert "take_hyper_residual()" in source
    assert "take_logits()" in source
    assert "_owned_tensors" not in source
    constructor_parameters = inspect.signature(Qwen38TTNNHybridDecodeSession).parameters
    assert not {"fixed_target", "mtp_engine", "host_sampler"}.intersection(constructor_parameters)
    assert "_test_collaborators" in constructor_parameters
    assert "mesh_contract.validate_mesh(model.mesh_device)" in source


def test_borrowed_hybrid_session_does_not_consume_external_resident_owner(monkeypatch) -> None:
    owner, tensors = _resident_owner(monkeypatch)
    fixture = _fixture(monkeypatch, resident_owner=owner)

    fixture.session.close()
    fixture.session.close()

    assert not fixture.session.owns_built_graph
    assert fixture.session.borrowed_resident_expert_owner is owner
    assert not owner.closed
    assert all(not tensor.released for tensor in tensors)


def test_factory_owned_hybrid_clean_close_releases_shared_resident_owner_once(monkeypatch) -> None:
    owner, tensors = _resident_owner(monkeypatch)
    fixture = _fixture(monkeypatch, resident_owner=owner, owns_built_graph=True)

    fixture.session.close()
    fixture.session.close()

    assert fixture.session.owns_built_graph
    assert owner.closed
    assert all(tensor.released for tensor in tensors)
    assert "resident BF4 experts" not in fixture.session.unreleased_resources


def test_prompt_bootstrap_streams_one_root_at_a_time_and_keeps_ready_seed(monkeypatch) -> None:
    fixture = _fixture(monkeypatch, target_predictions=(20,))
    assert fixture.target.model.mesh_contract.validation_calls == 1
    step = fixture.session.begin((10, 11, 12), max_new_tokens=8, mode=Qwen38HybridMode.MTP)

    assert [(event[0], event[1]) for event in fixture.events] == [
        ("target", 10),
        ("mtp", 11),
        ("target", 11),
        ("mtp", 12),
        ("target", 12),
        ("mtp", 20),
    ]
    assert all(root.released for _, root in fixture.mtp.bootstrap_calls[0])
    assert fixture.session.consumed_token_ids == (10, 11, 12)
    assert fixture.session.transcript_token_ids == (10, 11, 12, 20)
    assert fixture.session.pending_token_id == 20
    assert fixture.session.state_position == 3
    assert step.tokens[0].cache_position == 3
    assert step.timing.phase is Qwen38HybridTimingPhase.TTFT
    assert (step.timing.target_rows, step.timing.mtp_extension_calls, step.timing.mtp_alignment_rows) == (3, 0, 3)
    assert step.timing.requested_mode is Qwen38HybridMode.MTP
    assert step.timing.executed_mode is Qwen38HybridMode.ORDINARY
    assert fixture.session._mtp_seed.first_draft_token_id == 120
    fixture.session.close()


def test_ready_seed_observer_brackets_target_mtp_sync_and_publication(monkeypatch) -> None:
    fixture = _fixture(monkeypatch, target_predictions=(20,))
    phases = []

    with fixture.session.observe_ready_seed_phases(phases.append):
        step = fixture.session.begin((10,), max_new_tokens=8, mode=Qwen38HybridMode.MTP)

    assert step.tokens[0].token_id == 20
    assert fixture.session._ready_seed_phase_observer is None
    assert fixture.parts.mtp_engine._bootstrap_phase_observer is None
    target_phases = [
        phase
        for layer_index in range(BACKBONE_LAYERS)
        for phase in (
            f"before-ready-seed-target-row-0-layer-{layer_index}",
            f"after-ready-seed-target-row-0-layer-{layer_index}",
        )
    ]
    assert phases == [
        "before-ready-seed-pre-synchronize",
        "after-ready-seed-pre-synchronize",
        "before-ready-seed-mtp-bootstrap",
        "before-ready-seed-target-row-0",
        *target_phases,
        "before-ready-seed-target-row-0-final-mixer",
        "after-ready-seed-target-row-0-final-mixer",
        "before-ready-seed-target-row-0-lm-head",
        "after-ready-seed-target-row-0-lm-head",
        "after-ready-seed-target-row-0",
        "before-ready-seed-yield-shifted-row-0",
        "after-ready-seed-yield-shifted-row-0",
        "after-ready-seed-mtp-bootstrap",
        "before-ready-seed-post-synchronize",
        "after-ready-seed-post-synchronize",
        "before-ready-seed-publish-pending-seed",
        "after-ready-seed-publish-pending-seed",
    ]
    fixture.session.close()


def test_ready_seed_observer_translates_real_model_internal_layer_phases_through_layer_one(monkeypatch) -> None:
    fixture = _fixture(monkeypatch, target_predictions=(20,), emit_internal_phase_markers=True)
    phases = []

    with fixture.session.observe_ready_seed_phases(phases.append):
        step = fixture.session.begin((10,), max_new_tokens=8, mode=Qwen38HybridMode.MTP)

    assert step.tokens[0].token_id == 20
    expected = [
        "before-ready-seed-target-row-0-layer-0-ple",
        "after-ready-seed-target-row-0-layer-0-ple",
        "before-ready-seed-target-row-0-layer-0-gdn",
        "after-ready-seed-target-row-0-layer-0-gdn",
        "before-ready-seed-target-row-0-layer-1-ple",
        "after-ready-seed-target-row-0-layer-1-ple",
        "before-ready-seed-target-row-0-layer-1-qsa",
        "after-ready-seed-target-row-0-layer-1-qsa",
        "before-ready-seed-target-row-0-layer-1-moe-compute-launch",
        "after-ready-seed-target-row-0-layer-1-moe-compute-launch",
    ]
    assert [phase for phase in phases if phase in expected] == expected
    fixture.session.close()


@pytest.mark.parametrize(
    "phase",
    (
        "layer-0-before-unknown-stage",
        "layer-0-during-ple",
        "layer-x-before-ple",
        "layer-00-before-ple",
        f"layer-{BACKBONE_LAYERS}-before-ple",
        "layer-0-before-ple-extra",
        "layer-0-before-",
    ),
)
def test_ready_seed_internal_layer_phase_translation_rejects_malformed_or_unknown_stages(
    expect_error, phase: str
) -> None:
    with expect_error(ValueError, match="invalid target decode phase"):
        hybrid_module._translate_ready_seed_target_phase(phase, row_index=0)


def test_ordinary_step_consumes_pending_and_authoritatively_advances_seed(monkeypatch) -> None:
    fixture = _fixture(monkeypatch, target_predictions=(20, 21))
    fixture.session.begin((10,), max_new_tokens=3)
    step = fixture.session.step()

    assert fixture.session.consumed_token_ids == (10, 20)
    assert fixture.session.pending_token_id == 21
    assert fixture.session.state_position == 2
    assert tuple(token for token, _ in fixture.mtp.advance_calls[0]) == (21,)
    assert step.tokens[0].source is Qwen38HybridTokenSource.ORDINARY_GREEDY
    assert (step.timing.target_rows, step.timing.mtp_extension_calls, step.timing.mtp_alignment_rows) == (1, 0, 1)
    assert step.timing.speculative_transaction_ns == 0
    fixture.session.close()


def test_mtp_output_budget_falls_back_without_constructing_speculation(monkeypatch) -> None:
    fixture = _fixture(monkeypatch, target_predictions=(20, 21))
    fixture.session.begin((10,), max_new_tokens=2, mode=Qwen38HybridMode.MTP)
    step = fixture.session.step()

    assert step.executed_mode is Qwen38HybridMode.ORDINARY
    assert step.fallback_reason is Qwen38HybridFallbackReason.OUTPUT_BUDGET
    assert fixture.session.fallback_count == 1
    assert step.tokens[-1].stop_reason is Qwen38HybridStopReason.MAX_NEW_TOKENS
    assert fixture.session.status is Qwen38HybridSessionStatus.FINISHED
    fixture.session.close()


def test_official_ordinary_sampling_reuses_request_generator_and_reset_reseeds(monkeypatch) -> None:
    fixture = _fixture(monkeypatch, sampled_predictions=(30, 31, 30))
    parameters = Qwen38SamplingParameters.official_thinking(seed=123)
    first = fixture.session.begin((10,), max_new_tokens=3, sampling=parameters)
    second = fixture.session.step()

    assert first.tokens[0].source is Qwen38HybridTokenSource.TARGET_SAMPLED
    assert second.tokens[0].source is Qwen38HybridTokenSource.ORDINARY_SAMPLED
    assert first.sampling_result is not None and first.sampling_result.timing.end_to_end_ns == 3
    assert second.sampling_result is not None and second.sampling_result.rng_mode == "request-generator"
    assert first.timing.sampling_ns > 0 and second.timing.sampling_ns > 0
    assert len(fixture.sampler.calls) == 2
    generator0 = fixture.sampler.calls[0][2]
    generator1 = fixture.sampler.calls[1][2]
    assert generator0 is generator1 is not None
    assert not torch.equal(fixture.sampler.calls[0][3], fixture.sampler.calls[1][3])
    assert fixture.sampler.calls[0][1] == (10,)
    assert fixture.sampler.calls[1][1] == (10, 30)

    first_state = fixture.sampler.calls[0][3]
    fixture.session.reset()
    third = fixture.session.begin((77,), max_new_tokens=1, sampling=parameters)
    assert third.tokens[0].token_id == 30
    assert fixture.sampler.calls[2][2] is not generator0
    assert torch.equal(fixture.sampler.calls[2][3], first_state)
    fixture.session.close()


def test_mtp_rejects_stochastic_sampling_before_any_target_row(monkeypatch, expect_error) -> None:
    fixture = _fixture(monkeypatch)
    parameters = Qwen38SamplingParameters.official_non_thinking(seed=7)
    with expect_error(ValueError, "MTP mode supports exact greedy"):
        fixture.session.begin((10,), max_new_tokens=8, mode="mtp", sampling=parameters)
    assert fixture.target.calls == 0
    assert fixture.session.status is Qwen38HybridSessionStatus.IDLE
    fixture.session.close()


def test_full_rendered_history_match_mismatch_and_explicit_reset(monkeypatch, expect_error) -> None:
    fixture = _fixture(monkeypatch, target_predictions=(20, 21, 22, 23))
    fixture.session.begin((10,), max_new_tokens=1)
    assert fixture.session.status is Qwen38HybridSessionStatus.FINISHED
    calls = fixture.target.calls
    with expect_error(Qwen38HybridCacheMismatchError, "consumed target-token prefix"):
        fixture.session.begin((99, 20), max_new_tokens=1)
    assert fixture.target.calls == calls

    continued = fixture.session.begin((10, 20, 30), max_new_tokens=1)
    assert fixture.session.consumed_token_ids == (10, 20, 30)
    assert continued.next_position == 3
    assert tuple(token for token, _ in fixture.mtp.advance_calls[-1]) == (30, 21)

    restarted = fixture.session.begin(
        (77, 78),
        max_new_tokens=1,
        cache_policy=Qwen38HybridCachePolicy.RESET,
    )
    assert fixture.session.reset_count == 1
    assert fixture.session.consumed_token_ids == (77, 78)
    assert restarted.base_position == 0 and restarted.next_position == 2
    assert fixture.target.reset_calls == 1
    fixture.session.close()


def test_post_take_validation_failure_releases_detached_root(monkeypatch, expect_error) -> None:
    fixture = _fixture(monkeypatch, target_predictions=(20,))
    original_take = Qwen38TTNNTextModelOutput.take_hyper_residual

    def corrupt_public_slot(output):
        root = original_take(output)
        output.hyper_residual_sharded = root
        return root

    monkeypatch.setattr(Qwen38TTNNTextModelOutput, "take_hyper_residual", corrupt_public_slot)
    with expect_error(Qwen38HybridDecodePoisonedError, "ownership transfer did not detach"):
        fixture.session.begin((10,), max_new_tokens=2)
    assert fixture.events[-1][3].released
    assert fixture.session.status is Qwen38HybridSessionStatus.POISONED


def test_output_cleanup_failure_still_releases_detached_root(monkeypatch, expect_error) -> None:
    fixture = _fixture(monkeypatch, target_predictions=(20,))
    original_take = Qwen38TTNNTextModelOutput.take_hyper_residual

    def leave_other_owned_slot(output):
        root = original_take(output)
        output.active = True
        return root

    def fail_output_cleanup(output):
        del output
        raise RuntimeError("injected model-output cleanup failure")

    monkeypatch.setattr(Qwen38TTNNTextModelOutput, "take_hyper_residual", leave_other_owned_slot)
    monkeypatch.setattr(Qwen38TTNNTextModelOutput, "release_tensors", fail_output_cleanup)
    with expect_error(Qwen38HybridDecodePoisonedError, "model-output residual tensors"):
        fixture.session.begin((10,), max_new_tokens=2)
    assert fixture.events[-1][3].released


def test_failure_between_target_return_and_yield_releases_untransferred_root(monkeypatch, expect_error) -> None:
    fixture = _fixture(monkeypatch, target_predictions=(20,))
    target_row = fixture.session._target_row

    def omit_final_prediction(*args, **kwargs):
        result = list(target_row(*args, **kwargs))
        result[2] = None
        return tuple(result)

    monkeypatch.setattr(fixture.session, "_target_row", omit_final_prediction)
    with expect_error(Qwen38HybridDecodePoisonedError, "final streamed target row"):
        fixture.session.begin((10,), max_new_tokens=2)
    assert fixture.events[-1][3].released
    assert not fixture.mtp.bootstrap_calls


def test_post_sync_failure_restores_rng_and_retains_returned_seed_ledger(monkeypatch, expect_error) -> None:
    fixture = _fixture(monkeypatch, sampled_predictions=(30,))
    fixture.sync.fail_on_call = fixture.sync.calls + 2
    parameters = Qwen38SamplingParameters.official_non_thinking(seed=71)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(parameters.seed)
    original_rng = generator.get_state().clone()

    with expect_error(Qwen38HybridDecodePoisonedError, "synchronization failure"):
        fixture.session.begin((10,), max_new_tokens=2, sampling=parameters, generator=generator)

    assert torch.equal(generator.get_state(), original_rng)
    assert fixture.session._pending_mtp_seed is not None
    assert fixture.session._mtp_seed is None and fixture.session._raw_mtp_state is None
    assert "pending MTP ReadySeed" in fixture.session.unreleased_resources
    assert fixture.mtp.bootstrap_calls[0][0][1].released


def test_constructor_sync_failure_unwinds_every_acquired_owner(monkeypatch, expect_error) -> None:
    parts = _parts(monkeypatch)
    sync = _Synchronizer(fail_on_call=1)
    with expect_error(Qwen38HybridConstructionError, "construction failed"):
        Qwen38TTNNHybridDecodeSession(
            parts.built_target,
            parts.mtp_components,
            expected_provenance=parts.provenance,
            expected_physical_ids=PHYSICAL_IDS,
            expected_identity_key=parts.built_target.components.identity.key,
            eos_token_ids=(EOS,),
            synchronize=sync,
            clock_ns=_Clock(),
            _test_collaborators=hybrid_module._Qwen38HybridTestCollaborators(
                parts.fixed_target,
                parts.mtp_engine,
                parts.sampler_owner,
            ),
        )

    assert parts.target.release_calls == 1
    assert parts.mtp.released_state_count == 1
    assert parts.mtp.owner is None
    assert parts.target.model._runtime_owner is None
    assert parts.sampler_backend.closed
    assert parts.fixed_target._closed


def test_owned_constructor_failure_retains_resident_weights_for_mesh_teardown(monkeypatch) -> None:
    owner, tensors = _resident_owner(monkeypatch)
    parts = _parts(monkeypatch, resident_owner=owner)
    caught = None
    try:
        Qwen38TTNNHybridDecodeSession(
            parts.built_target,
            parts.mtp_components,
            expected_provenance=parts.provenance,
            expected_physical_ids=PHYSICAL_IDS,
            expected_identity_key=parts.built_target.components.identity.key,
            eos_token_ids=(EOS,),
            synchronize=_Synchronizer(fail_on_call=1),
            clock_ns=_Clock(),
            _test_collaborators=hybrid_module._Qwen38HybridTestCollaborators(
                parts.fixed_target,
                parts.mtp_engine,
                parts.sampler_owner,
            ),
            _owns_built_graph=True,
        )
    except Qwen38HybridConstructionError as error:
        caught = error

    assert caught is not None
    assert caught.partial_session.requires_process_termination
    assert "resident BF4 experts" in caught.partial_session.unreleased_resources
    assert not owner.closed
    assert all(not tensor.released for tensor in tensors)


def test_owned_close_failure_skips_resident_release_and_poisoned_exit_does_not_retry(monkeypatch, expect_error) -> None:
    owner, tensors = _resident_owner(monkeypatch)
    fixture = _fixture(monkeypatch, resident_owner=owner, owns_built_graph=True)
    fixture.target.fail_release = True

    with expect_error(Qwen38HybridDecodePoisonedError, "close hybrid session"):
        fixture.session.close()
    assert fixture.session.requires_process_termination
    assert "resident BF4 experts" in fixture.session.unreleased_resources
    assert not owner.closed
    assert all(not tensor.released for tensor in tensors)

    assert fixture.session.__exit__(RuntimeError, RuntimeError("body failed"), None) is False
    assert all(not tensor.released for tensor in tensors)


def test_constructor_cleanup_failure_returns_inspectable_partial_session(monkeypatch) -> None:
    parts = _parts(monkeypatch)
    parts.target.fail_release = True
    sync = _Synchronizer(fail_on_call=1)
    caught = None
    try:
        Qwen38TTNNHybridDecodeSession(
            parts.built_target,
            parts.mtp_components,
            expected_provenance=parts.provenance,
            expected_physical_ids=PHYSICAL_IDS,
            expected_identity_key=parts.built_target.components.identity.key,
            eos_token_ids=(EOS,),
            synchronize=sync,
            clock_ns=_Clock(),
            _test_collaborators=hybrid_module._Qwen38HybridTestCollaborators(
                parts.fixed_target,
                parts.mtp_engine,
                parts.sampler_owner,
            ),
        )
    except Qwen38HybridConstructionError as error:
        caught = error
    assert caught is not None
    partial = caught.partial_session
    assert partial.requires_process_termination
    assert partial.poison_error is not None and partial.poison_error.partial_session is partial
    assert {"target state", "target runtime claim", "fixed-five private buffers"}.issubset(partial.unreleased_resources)
    assert caught.cleanup_failures and caught.unreleased_resources == partial.unreleased_resources


def test_from_builder_failure_retains_complete_target_for_process_teardown(monkeypatch) -> None:
    parts = _parts(monkeypatch)
    builder = object.__new__(Qwen38TTNNBuilder)
    builder.provenance = parts.provenance
    builder.identity = parts.built_target.components.identity
    builder.expert_streamer = object()
    builder._target_components_published = False
    builder._built_target = False
    builder._built_mtp = False
    builder._resident_build_failure = None
    builder.build_target = lambda: parts.built_target

    def fail_mtp_build():
        raise RuntimeError("injected MTP component build failure")

    builder.build_mtp_components = fail_mtp_build
    caught = None
    try:
        Qwen38TTNNHybridDecodeSession.from_builder(
            builder,
            expected_provenance=parts.provenance,
            expected_physical_ids=PHYSICAL_IDS,
            expected_identity_key=parts.built_target.components.identity.key,
            eos_token_ids=(EOS,),
            synchronize=lambda: None,
            clock_ns=_Clock(),
        )
    except Qwen38HybridFactoryError as error:
        caught = error
    assert caught is not None
    assert caught.builder is builder
    assert caught.built_target is parts.built_target
    assert caught.mtp_components is None


def test_reset_failure_releases_independent_mtp_state_and_poison_reports_ledger(monkeypatch, expect_error) -> None:
    fixture = _fixture(monkeypatch, target_predictions=(20,))
    fixture.session.begin((10,), max_new_tokens=1)
    fixture.target.fail_reset = True

    with expect_error(Qwen38HybridDecodePoisonedError, "target reset failure"):
        fixture.session.reset()

    assert fixture.mtp.released_seed_count == 1
    assert fixture.session._mtp_seed is None
    assert fixture.session.status is Qwen38HybridSessionStatus.POISONED
    assert "target state" in fixture.session.unreleased_resources
    assert fixture.session.cleanup_failures


def test_reset_retains_replacement_mtp_state_when_post_allocation_validation_fails(monkeypatch, expect_error) -> None:
    fixture = _fixture(monkeypatch, target_predictions=(20,))
    fixture.session.begin((10,), max_new_tokens=1)

    def wrong_replacement_position(state):
        assert type(state) is _MTPState
        return state.position + 1

    monkeypatch.setattr(fixture.parts.mtp_engine, "state_position", wrong_replacement_position)
    with expect_error(Qwen38HybridDecodePoisonedError, "replacement MTP state"):
        fixture.session.reset()

    assert fixture.session._raw_mtp_state is not None
    assert fixture.session._raw_mtp_state.position == 0
    assert "fresh MTP state" in fixture.session.unreleased_resources
    assert fixture.session.poison_error.partial_session is fixture.session


def test_close_failure_attempts_independent_resources_once_and_reports_ledger(monkeypatch, expect_error) -> None:
    fixture = _fixture(monkeypatch, target_predictions=(20,))
    fixture.session.begin((10,), max_new_tokens=1)
    fixture.target.fail_release = True

    with expect_error(Qwen38HybridDecodePoisonedError, "target release failure"):
        fixture.session.close()

    assert fixture.target.release_calls == 1
    assert fixture.mtp.released_seed_count == 1
    assert fixture.mtp.owner is None
    assert fixture.sampler.closed
    assert fixture.session.status is Qwen38HybridSessionStatus.POISONED
    assert {"target state", "target runtime claim", "fixed-five private buffers"}.issubset(
        fixture.session.unreleased_resources
    )
    assert fixture.session.cleanup_failures
    assert fixture.session.requires_process_termination
    assert fixture.session.poison_error.partial_session is fixture.session


class _FakeController:
    def __init__(
        self,
        target,
        mtp,
        *,
        target_state,
        mtp_seed,
        pending_token_id,
        eos_token_ids,
        diagnostic_exact_token_budget=False,
        previous_target_owner,
        previous_mtp_owner,
    ) -> None:
        del eos_token_ids
        self.target = target
        self.mtp = mtp
        self.target_state = target_state
        self.seed = mtp_seed
        self._pending = pending_token_id
        self._position = target_state.position
        self._owner = object()
        self._rounds: list[Qwen38SpeculativeRound] = []
        self.diagnostic_exact_token_budget = diagnostic_exact_token_budget
        self.status = Qwen38SpeculativeStatus.READY
        self.poison_error = None
        target.transfer_runtime_owner(previous_target_owner, self._owner)
        mtp.transfer_runtime_owner(previous_mtp_owner, self._owner)

    @property
    def position(self) -> int:
        return self._position

    @property
    def pending_token_id(self) -> int:
        return self._pending

    def step(self) -> Qwen38SpeculativeRound:
        base = self._position
        inputs = (self._pending, 101, 102, 103, 104)
        emissions = (
            Qwen38SpeculativeEmission(101, Qwen38SpeculativeTokenSource.ACCEPTED_DRAFT),
            Qwen38SpeculativeEmission(202, Qwen38SpeculativeTokenSource.TARGET_REPLACEMENT),
        )
        result = Qwen38SpeculativeRound(
            round_index=len(self._rounds),
            base_position=base,
            next_position=base + 2,
            input_token_ids=inputs,
            draft_token_ids=(101, 102, 103, 104),
            target_token_ids=(101, 202, 0, 0, 0),
            matched_draft_depth=1,
            accepted_draft_count=1,
            committed_input_count=2,
            emissions=emissions,
            pending_token_id=202,
            stop_reason=None,
            target_commit_mode=Qwen38TargetCommitMode.RESTORE_REPLAY,
            mtp_extension_count=3,
            mtp_alignment_count=2,
        )
        self._position += 2
        self._pending = 202
        self.target_state = Qwen38TTNNTextModelState(self._position, (), self.target.model._state_owner)
        self.seed = Qwen38MTPSeed(
            self._position,
            self._pending,
            302,
            _MTPState(self._position),
            object(),
            _proof(self._position - 1, 999),
            object(),
        )
        self._rounds.append(result)
        return result

    def handoff(self, *, next_target_owner, next_mtp_owner) -> Qwen38SpeculativeHandoff:
        self.target.transfer_runtime_owner(self._owner, next_target_owner)
        self.mtp.transfer_runtime_owner(self._owner, next_mtp_owner)
        self.status = Qwen38SpeculativeStatus.HANDED_OFF
        return Qwen38SpeculativeHandoff(
            identity_key=self.target.identity_key,
            target_state=self.target_state,
            mtp_seed=self.seed,
            position=self._position,
            pending_token_id=self._pending,
            finished=False,
            rounds=tuple(self._rounds),
        )


class _RejectingController(_FakeController):
    def step(self) -> Qwen38SpeculativeRound:
        base = self._position
        inputs = (self._pending, 101, 102, 103, 104)
        emissions = (Qwen38SpeculativeEmission(202, Qwen38SpeculativeTokenSource.TARGET_REPLACEMENT),)
        result = Qwen38SpeculativeRound(
            round_index=len(self._rounds),
            base_position=base,
            next_position=base + 1,
            input_token_ids=inputs,
            draft_token_ids=(101, 102, 103, 104),
            target_token_ids=(202, 0, 0, 0, 0),
            matched_draft_depth=0,
            accepted_draft_count=0,
            committed_input_count=1,
            emissions=emissions,
            pending_token_id=202,
            stop_reason=None,
            target_commit_mode=Qwen38TargetCommitMode.RESTORE_REPLAY,
            mtp_extension_count=3,
            mtp_alignment_count=1,
        )
        self._position += 1
        self._pending = 202
        self.target_state = Qwen38TTNNTextModelState(self._position, (), self.target.model._state_owner)
        self.seed = Qwen38MTPSeed(
            self._position,
            self._pending,
            302,
            _MTPState(self._position),
            object(),
            _proof(self._position - 1, 1000),
            object(),
        )
        self._rounds.append(result)
        return result


def test_speculative_round_audits_three_extensions_and_full_history_handoff(monkeypatch) -> None:
    fixture = _fixture(monkeypatch, target_predictions=(20, 203))
    monkeypatch.setattr(hybrid_module, "Qwen38SpeculativeDecodeController", _FakeController)
    fixture.session.begin((10,), max_new_tokens=10, mode="mtp")
    step = fixture.session.step()

    assert step.executed_mode is Qwen38HybridMode.MTP
    assert step.accepted_draft_count == 1
    assert step.timing.mtp_extension_calls == 3
    assert step.timing.mtp_alignment_rows == 2
    assert fixture.session.consumed_token_ids == (10, 20, 101)
    assert fixture.session.pending_token_id == 202
    assert fixture.session.state_position == 3

    fixture.session.set_mode("ordinary")
    ordinary = fixture.session.step()
    assert ordinary.executed_mode is Qwen38HybridMode.ORDINARY
    assert fixture.session.consumed_token_ids == (10, 20, 101, 202)
    fixture.session.close()


def test_nonqualifying_mtp_mechanics_executes_ordinary_then_draft_verify_accept(monkeypatch) -> None:
    fixture = _fixture(monkeypatch, target_predictions=(20,))
    monkeypatch.setattr(hybrid_module, "Qwen38SpeculativeDecodeController", _FakeController)

    bootstrap, speculative = execute_one_mtp_round(fixture.session, (10,))

    assert bootstrap.executed_mode is Qwen38HybridMode.ORDINARY
    assert [token.token_id for token in bootstrap.tokens] == [20]
    assert speculative.executed_mode is Qwen38HybridMode.MTP
    assert speculative.speculative_round is not None
    assert speculative.speculative_round.draft_token_ids == (101, 102, 103, 104)
    assert speculative.speculative_round.target_token_ids == (101, 202, 0, 0, 0)
    assert speculative.accepted_draft_count == 1
    assert speculative.committed_input_count == 2
    assert fixture.session.consumed_token_ids == (10, 20, 101)
    assert fixture.session.pending_token_id == 202
    assert fixture.session.state_position == 3
    fixture.session.close()


def test_nonqualifying_mtp_mechanics_rejects_first_draft_and_aligns_replacement(monkeypatch) -> None:
    fixture = _fixture(monkeypatch, target_predictions=(20,))
    monkeypatch.setattr(hybrid_module, "Qwen38SpeculativeDecodeController", _RejectingController)

    bootstrap, speculative = execute_one_mtp_round(fixture.session, (10,))

    assert bootstrap.executed_mode is Qwen38HybridMode.ORDINARY
    assert speculative.executed_mode is Qwen38HybridMode.MTP
    assert speculative.speculative_round is not None
    assert speculative.speculative_round.draft_token_ids == (101, 102, 103, 104)
    assert speculative.speculative_round.target_token_ids == (202, 0, 0, 0, 0)
    assert speculative.accepted_draft_count == 0
    assert speculative.committed_input_count == 1
    assert [token.token_id for token in speculative.tokens] == [202]
    assert fixture.session.consumed_token_ids == (10, 20)
    assert fixture.session.pending_token_id == 202
    assert fixture.session.state_position == 2
    fixture.session.close()


def test_stream_failure_after_target_mutation_poisoned_without_retry(monkeypatch, expect_error) -> None:
    fixture = _fixture(monkeypatch, target_predictions=(20,))
    fixture.mtp.fail_bootstrap = True
    with expect_error(Qwen38HybridDecodePoisonedError, "streamed prompt bootstrap"):
        fixture.session.begin((10,), max_new_tokens=2)
    assert fixture.session.status is Qwen38HybridSessionStatus.POISONED
    assert fixture.target.calls == 1
    with expect_error(Qwen38HybridDecodePoisonedError):
        fixture.session.step()


def test_eos_and_generate_termination(monkeypatch) -> None:
    fixture = _fixture(monkeypatch, target_predictions=(EOS,))
    tokens = tuple(fixture.session.generate((10,), max_new_tokens=9))
    assert [token.token_id for token in tokens] == [EOS]
    assert tokens[0].stop_reason is Qwen38HybridStopReason.EOS
    assert fixture.session.status is Qwen38HybridSessionStatus.FINISHED
    fixture.session.close()
    assert fixture.session.status is Qwen38HybridSessionStatus.CLOSED
    assert not fixture.session.requires_process_termination
    assert fixture.session.unreleased_resources == ()


def test_diagnostic_exact_budget_ignores_eos_until_max_new_tokens(monkeypatch) -> None:
    predictions = (EOS, *range(21, 52))
    fixture = _fixture(monkeypatch, target_predictions=predictions)
    tokens = tuple(
        fixture.session.generate(
            (10,),
            max_new_tokens=32,
            diagnostic_exact_token_budget=True,
        )
    )
    assert tuple(token.token_id for token in tokens) == predictions
    assert all(token.stop_reason is None for token in tokens[:-1])
    assert tokens[-1].stop_reason is Qwen38HybridStopReason.MAX_NEW_TOKENS
    assert fixture.session.status is Qwen38HybridSessionStatus.FINISHED
    fixture.session.close()


def test_diagnostic_exact_budget_does_not_bypass_native_context(monkeypatch) -> None:
    fixture = _fixture(monkeypatch, target_predictions=(EOS,), allocated_context=1)
    tokens = tuple(
        fixture.session.generate(
            (10,),
            max_new_tokens=32,
            diagnostic_exact_token_budget=True,
        )
    )
    assert [token.token_id for token in tokens] == [EOS]
    assert tokens[-1].stop_reason is Qwen38HybridStopReason.CONTEXT_LENGTH
    assert fixture.session.status is Qwen38HybridSessionStatus.FINISHED
    fixture.session.close()


def test_diagnostic_exact_budget_reaches_mtp_controller_after_eos_bootstrap(monkeypatch) -> None:
    fixture = _fixture(monkeypatch, target_predictions=(EOS,))
    monkeypatch.setattr(hybrid_module, "Qwen38SpeculativeDecodeController", _FakeController)
    first = fixture.session.begin(
        (10,),
        max_new_tokens=32,
        mode=Qwen38HybridMode.MTP,
        diagnostic_exact_token_budget=True,
    )
    assert first.tokens[0].token_id == EOS
    assert fixture.session.status is Qwen38HybridSessionStatus.ACTIVE
    speculative = fixture.session.step()
    assert speculative.executed_mode is Qwen38HybridMode.MTP
    assert fixture.session._controller.diagnostic_exact_token_budget is True
    fixture.session.close()


def test_diagnostic_exact_budget_rejects_truthy_non_boolean_before_device_mutation(expect_error, monkeypatch) -> None:
    fixture = _fixture(monkeypatch, target_predictions=(20,))
    with expect_error(TypeError, match="diagnostic_exact_token_budget must be boolean"):
        fixture.session.begin((10,), max_new_tokens=32, diagnostic_exact_token_budget=1)
    assert fixture.target.calls == 0
    assert fixture.session.status is Qwen38HybridSessionStatus.IDLE
    fixture.session.close()


@pytest.mark.parametrize("max_new_tokens", (31, 33, 512))
def test_diagnostic_exact_budget_rejects_any_non_32_budget_before_device_mutation(
    expect_error,
    monkeypatch,
    max_new_tokens: int,
) -> None:
    fixture = _fixture(monkeypatch, target_predictions=(20,))
    with expect_error(ValueError, match="requires max_new_tokens=32"):
        fixture.session.begin(
            (10,),
            max_new_tokens=max_new_tokens,
            diagnostic_exact_token_budget=True,
        )
    assert fixture.target.calls == 0
    assert fixture.session.status is Qwen38HybridSessionStatus.IDLE
    fixture.session.close()
