# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.ttnn.layer import (
    Qwen38TTNNDecoderLayerAux,
    Qwen38TTNNDecoderLayerResult,
    Qwen38TTNNDecoderLayerState,
    Qwen38TTNNLayerNamespace,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.moe import Qwen38TTNNRouting
from models.demos.blackhole.qwen38_flash_next.ttnn.mtp_draft import (
    BF4_LOAD_POLICY,
    Qwen38TTNNMTPDraftEngine,
    Qwen38TTNNMTPDraftPoisonedError,
    Qwen38TTNNMTPExtensionPositionOneObservation,
    Qwen38TTNNMTPExtensionPositionOneResult,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import Qwen38TTNNQSASelection

IDENTITY = "a" * 64


class _Tensor:
    _next = 1

    def __init__(self, label):
        self.label = label
        self._id = _Tensor._next
        _Tensor._next += 1

    def tensor_id(self):
        return self._id


@dataclass(frozen=True)
class _Attention:
    view_id: int
    next_position: int
    last_selection: Qwen38TTNNQSASelection | None = None


@dataclass
class _Snapshot:
    state: Qwen38TTNNDecoderLayerState
    active: bool = True


class _Embedding:
    def upload_tokens(self, tokens):
        return SimpleNamespace(tensor=_Tensor(f"token:{int(tokens.item())}"), token=int(tokens.item()))

    def __call__(self, uploaded):
        return _Tensor(f"embedding:{uploaded.token}")


class _PoisonedEmbedding(_Embedding):
    def __init__(self):
        self.poisoned = False
        self.poisoned_owner = _Tensor("poisoned-embedding-owner")
        self.uploaded = None

    @property
    def poisoned_device_owners(self):
        return (self.poisoned_owner,)

    def upload_tokens(self, tokens):
        self.uploaded = super().upload_tokens(tokens)
        return self.uploaded

    def __call__(self, uploaded):
        self.poisoned = True
        raise RuntimeError("synthetic asynchronous embedding failure")


class _InputMixer:
    def __init__(self):
        self.calls = []

    def _validate_input(self, tensor, *, label, shape):
        assert tensor is not None
        assert shape in ((1, 1, 1, 640), (1, 4, 1, 640))

    def __call__(self, embedding, residual):
        self.calls.append((embedding.label, residual.label))
        return _Tensor(f"mixed:{embedding.label}:{residual.label}")


class _RopeInputs:
    def __init__(self, position):
        self.position = position
        self.cos = _Tensor("cos")
        self.sin = _Tensor("sin")
        self.block_start_cos = None
        self.block_start_sin = None

    def deallocate(self):
        pass


class _Rope:
    def for_position(self, position):
        return _RopeInputs(position)


class _FinalMixer:
    def __call__(self, residual):
        return _Tensor(f"hidden:{residual.label}")


class _LMHead:
    def __init__(self, predictions):
        self.predictions = list(predictions)

    def __call__(self, hidden):
        return SimpleNamespace(tensor=_Tensor(f"logits:{hidden.label}"))

    def greedy_token(self, logits):
        return torch.tensor([[[self.predictions.pop(0)]]], dtype=torch.int64)


class _Layer:
    def __init__(self):
        self.view = 10
        self.calls = []
        self.fail_at = None
        self.released = []

    def allocate_state(self):
        return Qwen38TTNNDecoderLayerState(Qwen38TTNNLayerNamespace.MTP, 0, 0, _Attention(self.view, 0), None)

    def validate_state(self, state):
        assert state.namespace is Qwen38TTNNLayerNamespace.MTP and state.layer_index == 0
        assert state.attention.next_position == state.position

    def snapshot_state(self, state):
        return _Snapshot(state)

    def validate_restore_pair(self, current, snapshot):
        assert snapshot.active

    def validate_commit_pair(self, current, snapshot):
        assert snapshot.active

    def restore_state(self, current, snapshot):
        snapshot.active = False
        return snapshot.state

    def commit_state(self, current, snapshot):
        snapshot.active = False
        return current

    def release_state(self, state):
        self.released.append(state)

    def forward_decode(self, mixed, state, **kwargs):
        call_index = len(self.calls)
        if self.fail_at == call_index:
            raise RuntimeError(f"forward fault {call_index}")
        position = state.position
        reuse = kwargs["reuse_qsa_selection"]
        self.view += 1
        if reuse is None:
            selection = Qwen38TTNNQSASelection(
                layer_index=0,
                epoch=1,
                source_view_id=self.view,
                source_position=position,
                tail_start=0,
                complete_indices=None,
                complete_token_count=0,
                owns_complete_indices=False,
                sparse_indices=_Tensor("sparse"),
                valid_token_count=position + 1,
            )
        else:
            selection = replace(
                reuse,
                sparse_indices=_Tensor("sparse-reuse"),
                owns_complete_indices=False,
                valid_token_count=reuse.complete_token_count + position + 1 - reuse.tail_start,
            )
        attention = _Attention(self.view, position + 1, selection)
        next_state = Qwen38TTNNDecoderLayerState(Qwen38TTNNLayerNamespace.MTP, 0, position + 1, attention, None)
        self.calls.append(
            {
                "position": position,
                "reuse": reuse,
                "retain": kwargs["retain_input_state"],
                "mixed": mixed,
                "result_view": self.view,
                "returned_selection": selection,
                "return_routing": kwargs["return_routing"],
            }
        )
        routing = (
            Qwen38TTNNRouting(scores=_Tensor("routing-scores"), indices=_Tensor("routing-indices"))
            if kwargs["return_routing"]
            else None
        )
        return Qwen38TTNNDecoderLayerResult(
            residual_sharded=_Tensor(f"mtp-residual:{position}"),
            state=next_state,
            aux=Qwen38TTNNDecoderLayerAux(routing, selection, reuse is not None),
        )


def _engine(monkeypatch, predictions=range(100, 140)):
    released = []

    def deallocate(tensor):
        released.append(tensor)

    monkeypatch.setattr("models.demos.blackhole.qwen38_flash_next.ttnn.mtp_draft.ttnn.deallocate", deallocate)
    engine = object.__new__(Qwen38TTNNMTPDraftEngine)
    layer = _Layer()
    mixer = _InputMixer()
    engine._initialize_owner(
        identity_key=IDENTITY,
        embedding=_Embedding(),
        lm_head=_LMHead(predictions),
        input_mixer=mixer,
        decoder_layer=layer,
        final_mixer=_FinalMixer(),
        rope=_Rope(),
    )
    owner = object()
    engine.claim_runtime_owner(owner)
    return engine, layer, mixer, released, owner


def _roots(count):
    return tuple(_Tensor(f"root:{index}") for index in range(count))


def test_state_allocation_requires_full_lifetime_runtime_owner(monkeypatch, expect_error):
    engine, _, _, _, owner = _engine(monkeypatch)
    engine.release_runtime_owner(owner)
    with expect_error(RuntimeError, "no runtime owner"):
        engine.allocate_state()
    engine.claim_runtime_owner(owner)
    assert engine.allocate_state().position == 0


def test_async_embedding_failure_retains_uploaded_owner_and_poisons_mtp(monkeypatch, expect_error):
    engine, _, _, released, _ = _engine(monkeypatch)
    embedding = _PoisonedEmbedding()
    engine.embedding = embedding

    with expect_error(Qwen38TTNNMTPDraftPoisonedError, "MTP token embedding asynchronous chain"):
        engine._embed_token(17)

    assert released == []
    assert engine._poisoned_device_owners[0] is embedding.poisoned_owner
    assert engine._poisoned_device_owners[1] is embedding.uploaded.tensor
    assert engine.poisoned


def test_shifted_prompt_bootstrap_pairs_h_i_with_t_i_plus_1_and_pending(monkeypatch):
    engine, layer, mixer, released, _ = _engine(monkeypatch, predictions=(201, 202, 203))
    state = engine.allocate_state()
    roots = _roots(3)
    seed = engine.bootstrap_shifted_prefill((7, 8, 9), 10, roots, state)

    assert [call["position"] for call in layer.calls] == [0, 1, 2]
    assert mixer.calls == [
        ("embedding:8", "root:0"),
        ("embedding:9", "root:1"),
        ("embedding:10", "root:2"),
    ]
    assert seed.position == 3 and seed.current_token_id == 10
    assert seed.first_draft_token_id == 203
    assert seed.recurrent_residual.label == "mtp-residual:2"
    assert seed.qsa_selection.source_position == 2
    assert all(root in released for root in roots)


def test_bootstrap_rejects_empty_or_unshifted_history_without_mutating_state(monkeypatch, expect_error):
    engine, layer, _, released, _ = _engine(monkeypatch)
    state = engine.allocate_state()
    with expect_error(ValueError, "at least one consumed target token"):
        engine.bootstrap_shifted_prefill((), 10, (), state)
    with expect_error(ValueError, "one target root per consumed token"):
        engine.bootstrap_shifted_prefill((7, 8), 10, _roots(1), state)
    assert not layer.calls and not released and engine.raw_state_position(state) == 0


def test_streaming_bootstrap_consumes_each_root_before_requesting_the_next(monkeypatch):
    engine, _, mixer, released, _ = _engine(monkeypatch, predictions=(201, 202, 203))
    state = engine.allocate_state()
    roots = _roots(3)

    def rows():
        yield 8, roots[0]
        assert roots[0] in released
        yield 9, roots[1]
        assert roots[1] in released
        yield 10, roots[2]

    seed = engine.bootstrap_shifted_prefill_rows(rows(), state)
    assert seed.position == 3 and seed.current_token_id == 10 and seed.first_draft_token_id == 203
    assert mixer.calls == [
        ("embedding:8", "root:0"),
        ("embedding:9", "root:1"),
        ("embedding:10", "root:2"),
    ]
    assert all(root in released for root in roots)


def test_streaming_bootstrap_observer_reports_exact_nonsemantic_substages(monkeypatch):
    engine, _, _, _, _ = _engine(monkeypatch, predictions=(201,))
    state = engine.allocate_state()
    root = _roots(1)[0]
    phases = []

    with engine.observe_bootstrap_phases(phases.append):
        seed = engine.bootstrap_shifted_prefill_rows(iter(((10, root),)), state)

    assert seed.first_draft_token_id == 201
    assert engine._bootstrap_phase_observer is None
    assert phases == [
        "before-ready-seed-request-shifted-row-0",
        "after-ready-seed-request-shifted-row-0",
        "before-ready-seed-snapshot-state",
        "after-ready-seed-snapshot-state",
        "before-ready-seed-execute-mtp-row-0",
        "before-ready-seed-mtp-row-0-embedding",
        "after-ready-seed-mtp-row-0-embedding",
        "before-ready-seed-mtp-row-0-input-mixer",
        "after-ready-seed-mtp-row-0-input-mixer",
        "before-ready-seed-mtp-row-0-embedding-release",
        "after-ready-seed-mtp-row-0-embedding-release",
        "before-ready-seed-mtp-row-0-rope",
        "after-ready-seed-mtp-row-0-rope",
        "before-ready-seed-mtp-row-0-decoder-layer",
        "after-ready-seed-mtp-row-0-decoder-layer",
        "before-ready-seed-mtp-row-0-rope-release",
        "after-ready-seed-mtp-row-0-rope-release",
        "before-ready-seed-mtp-row-0-selection-proof",
        "after-ready-seed-mtp-row-0-selection-proof",
        "before-ready-seed-mtp-row-0-final-mixer",
        "after-ready-seed-mtp-row-0-final-mixer",
        "before-ready-seed-mtp-row-0-lm-head",
        "after-ready-seed-mtp-row-0-lm-head",
        "before-ready-seed-mtp-row-0-resolve-token",
        "after-ready-seed-mtp-row-0-resolve-token",
        "before-ready-seed-mtp-row-0-terminal-release",
        "after-ready-seed-mtp-row-0-terminal-release",
        "after-ready-seed-execute-mtp-row-0",
        "before-ready-seed-release-target-root-0",
        "after-ready-seed-release-target-root-0",
        "before-ready-seed-request-shifted-row-1",
        "after-ready-seed-shifted-rows-exhausted",
        "before-ready-seed-validate-commit-pair",
        "after-ready-seed-validate-commit-pair",
        "before-ready-seed-commit-state",
        "after-ready-seed-commit-state",
        "before-ready-seed-publish-committed-seed",
        "after-ready-seed-publish-committed-seed",
    ]


def test_bootstrap_observer_scope_is_restored_after_failure(expect_error, monkeypatch):
    engine, _, _, _, _ = _engine(monkeypatch)

    with expect_error(RuntimeError, match="observer fault"):
        with engine.observe_bootstrap_phases(lambda _phase: None):
            raise RuntimeError("observer fault")

    assert engine._bootstrap_phase_observer is None


def test_streaming_bootstrap_seed_publish_failure_is_terminal_after_commit(monkeypatch, expect_error):
    engine, _, _, released, _ = _engine(monkeypatch, predictions=(201,))
    state = engine.allocate_state()
    root = _roots(1)[0]

    def fail_seed_publish(**_kwargs):
        raise RuntimeError("seed publish fault")

    monkeypatch.setattr(engine, "_make_seed", fail_seed_publish)
    with expect_error(Qwen38TTNNMTPDraftPoisonedError, "seed publish fault"):
        engine.bootstrap_shifted_prefill_rows(iter(((10, root),)), state)
    assert engine.poisoned
    assert root in released


def test_four_drafts_are_seed_plus_exactly_three_recurrent_calls_with_seed_qsa_reuse(monkeypatch):
    engine, layer, _, _, _ = _engine(monkeypatch, predictions=(11, 12, 13, 14, 15, 16))
    state = engine.allocate_state()
    seed = engine.bootstrap_shifted_prefill((7,), 10, _roots(1), state)
    calls_before = len(layer.calls)

    batch = engine.draft_four(seed)

    assert batch.draft_token_ids == (11, 12, 13, 14)
    assert len(layer.calls) - calls_before == 3
    extension_calls = layer.calls[calls_before:]
    assert [call["position"] for call in extension_calls] == [1, 2, 3]
    assert [call["retain"] for call in extension_calls] == [True, False, False]
    assert all(call["reuse"] is not None for call in extension_calls)
    assert extension_calls[0]["reuse"] is seed.transaction.selection
    assert extension_calls[1]["reuse"] is extension_calls[0]["returned_selection"]
    assert extension_calls[2]["reuse"] is extension_calls[1]["returned_selection"]
    assert all(call["reuse"].source_view_id == seed.qsa_selection.source_view_id for call in extension_calls)
    assert all(call["return_routing"] is False for call in extension_calls)
    assert [step.precomputed for step in batch.steps] == [True, False, False, False]
    assert len({step.qsa_selection.result_view_id for step in batch.steps}) == 4
    assert engine.bf4_load_policy == BF4_LOAD_POLICY == "per-layer-call-serialized"


def test_position_one_diagnostic_observes_one_row_then_restores_ready_seed(monkeypatch):
    engine, layer, _, released, _ = _engine(monkeypatch, predictions=(11, 15))
    state = engine.allocate_state()
    seed = engine.bootstrap_shifted_prefill((7,), 10, _roots(1), state)
    calls_before = len(layer.calls)
    observed = []

    def observe(value):
        assert type(value) is Qwen38TTNNMTPExtensionPositionOneObservation
        observed.append(value)
        return {"readback": "complete"}

    result = engine.diagnose_extension_position_one(seed, observe)

    assert type(result) is Qwen38TTNNMTPExtensionPositionOneResult
    assert result.position == 1
    assert result.input_token_id == 11
    assert result.predicted_token_id == 15
    assert result.observer_result == {"readback": "complete"}
    assert len(observed) == 1
    assert observed[0].position == 1
    assert observed[0].input_token_id == 11
    assert observed[0].predicted_token_id == 15
    assert len(layer.calls) - calls_before == 1
    assert layer.calls[-1]["position"] == 1
    assert layer.calls[-1]["reuse"] is seed.transaction.selection
    assert layer.calls[-1]["retain"] is True
    assert layer.calls[-1]["return_routing"] is True
    assert observed[0].hidden_sharded in released
    assert observed[0].logits_sharded.tensor in released
    assert observed[0].routing.scores in released
    assert observed[0].routing.indices in released
    assert seed.recurrent_residual not in released
    assert engine.state_position(seed.state) == seed.position == 1
    assert engine._active_transaction is None


@pytest.mark.parametrize("count", range(1, 6))
def test_commit_alignment_restores_draft_then_pairs_emissions_with_each_target_root(monkeypatch, count):
    engine, layer, mixer, released, _ = _engine(monkeypatch, predictions=range(11, 80))
    state = engine.allocate_state()
    seed = engine.bootstrap_shifted_prefill((7,), 10, _roots(1), state)
    batch = engine.draft_four(seed)
    emissions = tuple(30 + index for index in range(count))
    roots = _roots(count)
    engine.preflight_alignment(batch, emissions, roots)

    commit = engine.commit_alignment(batch, emissions, roots)

    assert commit.position == 1 + count
    assert commit.aligned_token_ids == emissions
    assert commit.consumed_hyper_residuals == roots
    assert commit.consumed_seed is seed
    assert commit.seed.current_token_id == emissions[-1]
    assert commit.seed.first_draft_token_id == 14 + count
    assert [step.position for step in commit.alignment_steps] == list(range(1, 1 + count))
    assert [step.input_token_id for step in commit.alignment_steps] == list(emissions)
    assert all(step.qsa_selection.source_position == step.position for step in commit.alignment_steps)
    assert all(root in released for root in roots)
    assert seed.recurrent_residual in released
    aligned_pairs = mixer.calls[-count:]
    assert aligned_pairs == [(f"embedding:{token}", root.label) for token, root in zip(emissions, roots)]


def test_fixed_five_roots_and_old_seed_stay_live_until_replacement_seed_is_built(monkeypatch):
    engine, _, _, released, _ = _engine(monkeypatch, predictions=range(11, 40))
    state = engine.allocate_state()
    seed = engine.bootstrap_shifted_prefill((7,), 10, _roots(1), state)
    batch = engine.draft_four(seed)
    roots = _roots(3)
    original_make_seed = engine._make_seed
    observed = []

    def checked_make_seed(**kwargs):
        observed.append(True)
        assert seed.recurrent_residual not in released
        assert all(root not in released for root in roots)
        return original_make_seed(**kwargs)

    monkeypatch.setattr(engine, "_make_seed", checked_make_seed)
    commit = engine.commit_alignment(batch, (20, 21, 22), roots)
    assert observed == [True]
    assert commit.seed.position == 4
    assert seed.recurrent_residual in released and all(root in released for root in roots)


def test_abort_restores_seed_and_keeps_seed_residual_live(monkeypatch):
    engine, _, _, released, _ = _engine(monkeypatch, predictions=range(11, 30))
    state = engine.allocate_state()
    seed = engine.bootstrap_shifted_prefill((7,), 10, _roots(1), state)
    batch = engine.draft_four(seed)
    rollback = engine.abort(batch)
    assert rollback.position == seed.position
    assert seed.recurrent_residual not in released
    assert engine.state_position(seed.state) == seed.position


def test_ordinary_handoff_advance_uses_same_authoritative_alignment(monkeypatch):
    engine, layer, mixer, released, _ = _engine(monkeypatch, predictions=range(11, 40))
    state = engine.allocate_state()
    seed = engine.bootstrap_shifted_prefill((7,), 10, _roots(1), state)
    roots = _roots(2)
    next_seed = engine.advance_seed(seed, (20, 21), roots)
    assert next_seed.position == 3
    assert next_seed.current_token_id == 21
    assert [call["position"] for call in layer.calls[-2:]] == [1, 2]
    assert mixer.calls[-2:] == [("embedding:20", "root:0"), ("embedding:21", "root:1")]
    assert seed.recurrent_residual in released and all(root in released for root in roots)


def test_streaming_ordinary_alignment_releases_each_root_before_next(monkeypatch):
    engine, _, mixer, released, _ = _engine(monkeypatch, predictions=range(11, 40))
    state = engine.allocate_state()
    seed = engine.bootstrap_shifted_prefill((7,), 10, _roots(1), state)
    roots = _roots(3)

    def rows():
        yield 20, roots[0]
        assert roots[0] in released
        yield 21, roots[1]
        assert roots[1] in released
        yield 22, roots[2]

    next_seed = engine.advance_seed_rows(seed, rows())
    assert next_seed.position == 4 and next_seed.current_token_id == 22
    assert mixer.calls[-3:] == [
        ("embedding:20", "root:0"),
        ("embedding:21", "root:1"),
        ("embedding:22", "root:2"),
    ]
    assert all(root in released for root in roots)
    assert seed.recurrent_residual in released


def test_streaming_alignment_seed_publish_failure_poison_preserves_prior_seed(monkeypatch, expect_error):
    engine, _, _, released, _ = _engine(monkeypatch, predictions=range(11, 40))
    state = engine.allocate_state()
    seed = engine.bootstrap_shifted_prefill((7,), 10, _roots(1), state)
    root = _roots(1)[0]

    def fail_seed_publish(**_kwargs):
        raise RuntimeError("stream publish fault")

    monkeypatch.setattr(engine, "_make_seed", fail_seed_publish)
    with expect_error(Qwen38TTNNMTPDraftPoisonedError, "stream publish fault"):
        engine.advance_seed_rows(seed, iter(((20, root),)))
    assert engine.poisoned
    assert root in released
    assert seed.recurrent_residual not in released


def test_fixed_five_seed_publish_failure_poison_keeps_all_borrowed_inputs_live(monkeypatch, expect_error):
    engine, _, _, released, _ = _engine(monkeypatch, predictions=range(11, 40))
    state = engine.allocate_state()
    seed = engine.bootstrap_shifted_prefill((7,), 10, _roots(1), state)
    batch = engine.draft_four(seed)
    roots = _roots(3)

    def fail_seed_publish(**_kwargs):
        raise RuntimeError("fixed-five publish fault")

    monkeypatch.setattr(engine, "_make_seed", fail_seed_publish)
    with expect_error(Qwen38TTNNMTPDraftPoisonedError, "fixed-five publish fault"):
        engine.commit_alignment(batch, (20, 21, 22), roots)
    assert engine.poisoned
    assert all(root not in released for root in roots)
    assert seed.recurrent_residual not in released


def test_alignment_fault_poison_preserves_roots_and_seed_residual(monkeypatch, expect_error):
    engine, layer, _, released, _ = _engine(monkeypatch, predictions=range(11, 40))
    state = engine.allocate_state()
    seed = engine.bootstrap_shifted_prefill((7,), 10, _roots(1), state)
    batch = engine.draft_four(seed)
    roots = _roots(2)
    layer.fail_at = len(layer.calls) + 1
    with expect_error(Qwen38TTNNMTPDraftPoisonedError, "forward fault"):
        engine.commit_alignment(batch, (20, 21), roots)
    assert all(root not in released for root in roots)
    assert seed.recurrent_residual not in released
    assert engine.poisoned


def test_release_seed_is_single_owner_and_releases_state_and_terminal_residual(monkeypatch, expect_error):
    engine, layer, _, released, owner = _engine(monkeypatch, predictions=(11,))
    state = engine.allocate_state()
    seed = engine.bootstrap_shifted_prefill((7,), 10, _roots(1), state)
    engine.release_seed(seed)
    assert seed.recurrent_residual in released
    assert layer.released == [seed.state]
    with expect_error((ValueError, RuntimeError), "live|active|released"):
        engine.release_seed(seed)
    engine.release_runtime_owner(owner)
