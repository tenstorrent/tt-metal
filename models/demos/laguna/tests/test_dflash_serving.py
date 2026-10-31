# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-free lifecycle, rollback, and token-equivalence contracts for DFlash serving."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.laguna.tt.dflash_reference import (
    DFlashTargetAuxCapture,
    LagunaDFlashConfig,
    published_dflash_config,
)
from models.demos.laguna.tt.dflash_serving import DFlashServedController, DFlashServingEnvelope
from models.demos.laguna.tt.generator_vllm import LagunaForCausalLM
from models.demos.laguna.tt.model import LagunaModel
from models.demos.laguna.tt.model_spec import DFLASH_SPEC, MODEL_ID


def _published_config() -> LagunaDFlashConfig:
    """The selected checkpoint's published draft (``TT_LAGUNA_MODEL``; XS 5x2048, S 6x3072)."""

    config = published_dflash_config()
    assert config.target_model_id == MODEL_ID
    return config


@dataclass(frozen=True)
class _FakeHidden:
    """Shape-correct auxiliary tensor metadata without a num_aux * hidden-wide allocation."""

    shape: tuple[int, int, int]
    positions: tuple[int, ...]


def _capture(config: LagunaDFlashConfig, start: int, rows: int) -> DFlashTargetAuxCapture:
    positions = tuple(range(int(start), int(start) + int(rows)))
    return DFlashTargetAuxCapture(
        hidden_states=_FakeHidden((1, int(rows), config.num_aux_hidden_states * config.hidden_size), positions),
        start_position=int(start),
        row_count=int(rows),
    )


class _FakeCache:
    def __init__(self, core):
        self.core = core
        self.max_context_rows = core.config.sliding_window - 1
        self._request_id = None
        self._capture = None
        self.closed = False
        self.commits: list[tuple[int, ...]] = []

    def begin_request(self, request_id):
        if self.closed:
            raise RuntimeError("cache closed")
        if self._request_id is not None:
            raise RuntimeError("request already active")
        self._request_id = request_id

    def update_target_capture(self, capture, *, replace=False):
        capture.validate(self.core.config)
        if self._request_id is None:
            raise RuntimeError("no active request")
        incoming = tuple(capture.hidden_states.positions)
        if replace or self._capture is None:
            positions = incoming
        else:
            old = tuple(self._capture.hidden_states.positions)
            if incoming[0] != old[-1] + 1:
                raise ValueError("capture is not adjacent")
            positions = old + incoming
        positions = positions[-self.max_context_rows :]
        self._capture = _capture(self.core.config, positions[0], len(positions))
        self.commits.append(incoming)

    def target_capture(self):
        if self._request_id is None or self._capture is None:
            raise RuntimeError("no active target context")
        return self._capture

    def end_request(self, request_id=None):
        if self._request_id is None:
            raise RuntimeError("no active request")
        if request_id is not None and request_id != self._request_id:
            raise RuntimeError("wrong request")
        self._request_id = None
        self._capture = None

    def close(self):
        self._request_id = None
        self._capture = None
        self.closed = True


class _FakeCore:
    def __init__(self, proposal_builder):
        self.config = _published_config()
        self.proposal_builder = proposal_builder
        self.proposal_calls: list[int] = []

    def proposal_round(self, cache, *, target_model, bonus_token_id, enable_experimental=False):
        assert cache.core is self
        assert target_model is not None
        assert enable_experimental
        round_index = len(self.proposal_calls)
        self.proposal_calls.append(int(bonus_token_id))
        return SimpleNamespace(drafts=tuple(self.proposal_builder(int(bonus_token_id), round_index)))

    def capture_prefix(self, capture, row_count):
        capture.validate(self.config)
        row_count = int(row_count)
        if not 1 <= row_count <= capture.row_count:
            raise ValueError("invalid prefix length")
        return _capture(self.config, capture.start_position, row_count)


def _controller(proposal_builder, verify_greedy):
    core = _FakeCore(proposal_builder)
    cache = _FakeCache(core)
    controller = DFlashServedController(
        core=core,
        proposal_cache=cache,
        target_model=object(),
        verify_greedy=verify_greedy,
        draft_argmax=lambda proposal: proposal.drafts,
        envelope=DFlashServingEnvelope(enabled=True),
    )
    return controller, core, cache


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({}, "default-off"),
        ({"enabled": True, "batch_size": 2}, "exactly one request"),
        ({"enabled": True, "greedy": False}, "greedy-only"),
        ({"enabled": True, "prefix_caching": True}, "prefix caching"),
        ({"enabled": True, "cache_off": False}, "cache-off"),
    ],
)
def test_serving_envelope_fails_closed(override, match, expect_error):
    with expect_error(RuntimeError, match):
        DFlashServingEnvelope(**override).validate()


def test_full_accept_commits_input_rows_and_buffers_outputs_one_by_one():
    drafts = tuple(range(100, 115))
    verify_calls = []

    def verify(tokens, positions):
        verify_calls.append((tuple(tokens), tuple(positions)))
        return [*drafts, 999], _capture(_published_config(), positions[0], len(tokens))

    controller, core, cache = _controller(lambda bonus, round_index: drafts, verify)
    controller.begin_request("request-a", _capture(core.config, 0, 4))

    current, position = 42, 4
    outputs = []
    for _ in range(16):
        current = controller.serve_token(known_bonus=current, position=position)
        outputs.append(current)
        position += 1

    assert outputs == [*drafts, 999]
    assert len(core.proposal_calls) == 1
    assert verify_calls == [((42, *drafts), tuple(range(4, 20)))]
    assert cache.commits[-1] == tuple(range(4, 20))
    assert cache.target_capture().end_position == 19
    assert not controller.pending_tokens
    assert controller.rounds[0].accepted_drafts == 15


def test_rejection_discards_lookahead_aux_and_next_verify_overwrites_it():
    drafts = tuple(range(1, 16))
    verify_starts = []

    def verify(tokens, positions):
        verify_starts.append(int(positions[0]))
        if len(verify_starts) == 1:
            greedy = [1, 2, 99, *([777] * 13)]
        else:
            greedy = [555, *([888] * 15)]
        return greedy, _capture(_published_config(), positions[0], len(tokens))

    controller, core, cache = _controller(lambda bonus, round_index: drafts, verify)
    controller.begin_request("request-b", _capture(core.config, 0, 11))

    first = controller.serve_token(known_bonus=42, position=11)
    second = controller.serve_token(known_bonus=first, position=12)
    correction = controller.serve_token(known_bonus=second, position=13)
    assert [first, second, correction] == [1, 2, 99]
    assert cache.commits[-1] == (11, 12, 13)
    assert cache.target_capture().end_position == 13

    next_token = controller.serve_token(known_bonus=correction, position=14)
    assert next_token == 555
    # The first target call wrote speculative rows 14..26.  The next call starts
    # at authoritative position 14, so rejected target KV is overwritten in place.
    assert verify_starts == [11, 14]
    assert cache.commits[-1] == (14,)
    assert controller.rounds[0].accepted_drafts == 2
    assert controller.rounds[1].accepted_drafts == 0


def test_controller_stream_is_token_equivalent_to_plain_target_greedy():
    modulus = 997

    def oracle(token):
        return (int(token) * 17 + 3) % modulus

    def proposals(bonus, round_index):
        result = []
        token = bonus
        for _ in range(15):
            token = oracle(token)
            result.append(token)
        # Alternate long accepts with a deterministic rejection, exercising both
        # buffered delivery and rollback while preserving the target token stream.
        if round_index % 2:
            result[round_index % 15] = (result[round_index % 15] + 1) % modulus
        return result

    def verify(tokens, positions):
        return [oracle(token) for token in tokens], _capture(_published_config(), positions[0], len(tokens))

    controller, core, cache = _controller(proposals, verify)
    controller.begin_request("request-equivalence", _capture(core.config, 0, 8))
    current = 73
    position = 8
    outputs = []
    for _ in range(700):
        expected = oracle(current)
        current = controller.serve_token(known_bonus=current, position=position)
        outputs.append(current)
        assert current == expected
        position += 1
    while controller.pending_tokens:
        expected = oracle(current)
        current = controller.serve_token(known_bonus=current, position=position)
        outputs.append(current)
        assert current == expected
        position += 1

    assert outputs
    retained = cache.target_capture()
    assert retained.row_count == 511
    assert retained.end_position == position - 1
    assert retained.hidden_states.positions == tuple(range(position - 511, position))
    assert any(round_.accepted_drafts == 15 for round_ in controller.rounds)
    assert any(round_.accepted_drafts < 15 for round_ in controller.rounds)


def test_target_only_fallback_crosses_unallocated_block_tail_exactly():
    modulus = 997

    def oracle(token):
        return (int(token) * 17 + 3) % modulus

    # Force a zero-length accept whenever a full proposal is attempted.  This
    # drains the buffer immediately and exposes every 49..63 block-tail input to
    # the adapter's one-row fallback.
    def proposals(bonus, round_index):
        first = (oracle(bonus) + 1) % modulus
        return [first, *([0] * 14)]

    verify_calls = []

    def verify(tokens, positions):
        verify_calls.append((positions[0], len(tokens)))
        return [oracle(token) for token in tokens], _capture(_published_config(), positions[0], len(tokens))

    controller, core, cache = _controller(proposals, verify)
    controller.begin_request("block-tail", _capture(core.config, 0, 48))
    current = 73
    for position in range(48, 65):
        expected = oracle(current)
        if position % 64 > 48:
            current = controller.serve_target_token(known_bonus=current, position=position)
        else:
            current = controller.serve_token(known_bonus=current, position=position)
        assert current == expected

    assert verify_calls == [(48, 16), *[(position, 1) for position in range(49, 64)], (64, 16)]
    assert [round_.position for round_ in controller.rounds if round_.target_only] == list(range(49, 64))
    assert cache.target_capture().end_position == 64


def test_prefill_tail_lifecycle_discontinuity_and_close(expect_error):
    controller, core, cache = _controller(lambda bonus, round_index: range(15), lambda *args: None)
    controller.ingest_prefill_capture("request-c", _capture(core.config, 0, 511), new_request=True)
    controller.ingest_prefill_capture("request-c", _capture(core.config, 511, 10))
    retained = cache.target_capture()
    assert (retained.start_position, retained.end_position, retained.row_count) == (10, 520, 511)

    # A full tail from a later target chunk supersedes the older window.
    controller.ingest_prefill_capture("request-c", _capture(core.config, 700, 511))
    assert cache.target_capture().hidden_states.positions == tuple(range(700, 1211))
    with expect_error(ValueError, "not after retained end"):
        controller.ingest_prefill_capture("request-c", _capture(core.config, 600, 511))
    with expect_error(ValueError, "expected 1211"):
        controller.ingest_prefill_capture("request-c", _capture(core.config, 1300, 2))
    with expect_error(RuntimeError, "does not match active request"):
        controller.ingest_prefill_capture("request-other", _capture(core.config, 1211, 1))
    with expect_error(RuntimeError, "position discontinuity"):
        controller.serve_token(known_bonus=1, position=1300)

    controller.end_request("request-c")
    assert not controller.active
    with expect_error(RuntimeError, "no active prefilled request"):
        controller.serve_token(known_bonus=1, position=1211)
    controller.begin_request("request-d", _capture(core.config, 9, 2))
    controller.close()
    assert cache.closed
    controller.close()
    with expect_error(RuntimeError, "closed"):
        controller.begin_request("request-e", _capture(core.config, 0, 1))


# XS serves DFlash on p150x2 (D=2) and S on p150x4 (D=4); every other mesh fails closed.
_OTHER_DEVICE_COUNTS = tuple(d for d in (1, 2, 4) if d != DFLASH_SPEC.serving_device_count)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        *(({"device_count": d}, DFLASH_SPEC.serving_profile) for d in _OTHER_DEVICE_COUNTS),
        ({"max_batch_size": 2}, "max-num-seqs 1"),
        ({"prefix_enabled": True}, "PREFIX_CACHE=0"),
        ({"spec_mode": "1"}, "SPEC_DECODE"),
    ],
)
def test_vllm_dflash_envelope_rejects_unqualified_modes(override, match, expect_error):
    envelope = {
        "enabled": True,
        "device_count": DFLASH_SPEC.serving_device_count,
        "max_batch_size": 1,
        "prefix_enabled": False,
        "hybrid_enabled": False,
        "spec_mode": "",
    }
    envelope.update(override)
    with expect_error(RuntimeError, match):
        LagunaForCausalLM._validate_dflash_serving_envelope(**envelope)
    LagunaForCausalLM._validate_dflash_serving_envelope(**{**envelope, "enabled": False})


@pytest.mark.parametrize("hybrid_enabled", (False, True))
def test_vllm_dflash_envelope_accepts_the_selected_checkpoint_topology(hybrid_enabled):
    LagunaForCausalLM._validate_dflash_serving_envelope(
        enabled=True,
        device_count=DFLASH_SPEC.serving_device_count,
        max_batch_size=1,
        prefix_enabled=False,
        hybrid_enabled=hybrid_enabled,
        spec_mode="",
    )
    DFlashServingEnvelope(enabled=True, hybrid_kv=hybrid_enabled).validate()
    assert LagunaForCausalLM._DFLASH_DEVICE_COUNT == {"poolside/Laguna-XS-2.1": 2, "poolside/Laguna-S-2.1": 4}[MODEL_ID]


def test_vllm_dflash_initialization_requires_the_selected_full_target(monkeypatch, expect_error):
    """The adapter rejects a partial target stack before touching the draft checkpoint."""

    bridge = object.__new__(LagunaForCausalLM)
    bridge.model = SimpleNamespace(layers=[object()] * (DFLASH_SPEC.num_target_layers - 1))
    bridge.max_model_len = 1024
    with expect_error(RuntimeError, f"exact full {DFLASH_SPEC.num_target_layers}-layer target"):
        bridge._initialize_dflash_serving()

    # The draft RoPE horizon is the checkpoint's own (XS 262144, S 1048576).
    bridge.model = SimpleNamespace(layers=[object()] * DFLASH_SPEC.num_target_layers)
    bridge.max_model_len = DFLASH_SPEC.max_position_embeddings - 63
    with expect_error(RuntimeError, f"max_model_len \\+ 64 <= {DFLASH_SPEC.max_position_embeddings}"):
        bridge._initialize_dflash_serving()


def test_vllm_dflash_verify_is_contiguous_uniform_and_returns_aux_capture(expect_error):
    config = _published_config()

    class FakeGenerator:
        @staticmethod
        def _rep(value, dtype):
            return value.clone()

    class FakeModel:
        def __init__(self):
            self.decode_call = None

        @staticmethod
        def embed_decode(tokens):
            return tokens

        def decode_layers_with_dflash_aux(self, hidden, cur, ridx, pt, kv_cache, **kwargs):
            self.decode_call = (hidden, cur, ridx, pt, kv_cache, kwargs)
            rows = int(cur.numel())
            return torch.zeros((1, 1, rows, 4)), _capture(config, int(cur[0]), rows)

        @staticmethod
        def lm_head_shards_decode(hidden):
            rows = int(hidden.shape[-2])
            logits = torch.zeros((rows, 7))
            for row in range(rows):
                logits[row, row + 2] = 1
            return logits

        @staticmethod
        def logits_to_host(logits):
            return logits

    bridge = object.__new__(LagunaForCausalLM)
    bridge._DFLASH_SERVING_ENABLED = True
    bridge.vocab = 7
    bridge.max_model_len = 100
    bridge.gen = FakeGenerator()
    bridge.model = FakeModel()
    page_tables = []
    bridge._page_table_to_device = lambda value: page_tables.append(value.clone()) or value

    greedy, capture = bridge.verify_greedy_decode_with_dflash_aux(
        [1, 2, 3],
        [9, 10, 11],
        page_table=torch.tensor([[4, 5]], dtype=torch.int32),
        kv_cache=[object()],
    )
    assert greedy == [2, 3, 4]
    assert (capture.start_position, capture.row_count, capture.end_position) == (9, 3, 11)
    assert page_tables[0].tolist() == [[4, 5], [4, 5], [4, 5]]
    kwargs = bridge.model.decode_call[-1]
    assert kwargs["absolute_position"] == 9
    assert kwargs["sequential_kv_write"] is True
    assert kwargs["enable_experimental"] is True

    with expect_error(ValueError, "strictly contiguous"):
        bridge.verify_greedy_decode_with_dflash_aux([1, 2], [9, 11], page_table=[[4, 5]], kv_cache=[object()])

    # Hybrid KV: one block table per group (full + three sliding), each repeated to the verify rows, and
    # every logical layer receives its own group's table.
    from models.demos.laguna.tt.kv_grouping import (
        build_hybrid_kv_layout,
        laguna_hybrid_layer_kinds,
    )

    layout = build_hybrid_kv_layout(laguna_hybrid_layer_kinds(8))
    bridge._hybrid_kv_layout = lambda: layout
    page_tables.clear()
    per_layer = [torch.tensor([[10 + layer % 4, 20 + layer % 4]], dtype=torch.int32) for layer in range(8)]
    greedy, capture = bridge.verify_greedy_decode_with_dflash_aux(
        [1, 2],
        [30, 31],
        kv_cache=[object()] * 8,
        page_tables_per_layer=per_layer,
    )
    assert greedy == [2, 3]
    assert (capture.start_position, capture.row_count) == (30, 2)
    assert [t.tolist() for t in page_tables] == [[[10 + g, 20 + g]] * 2 for g in range(4)]
    layer_tables = bridge.model.decode_call[3]
    assert [t.tolist() for t in layer_tables] == [[[10 + layer % 4, 20 + layer % 4]] * 2 for layer in range(8)]


def test_vllm_dflash_output_buffer_and_runtime_guards(monkeypatch, expect_error):
    calls = []

    class FakeController:
        pending_tokens = ()
        active = False
        expected = (None, None)

        def expected_input(self):
            return self.expected

        @staticmethod
        def serve_token(**kwargs):
            calls.append(("proposal", kwargs))
            return 23

        @staticmethod
        def serve_target_token(**kwargs):
            calls.append(("target", kwargs))
            return 23

    bridge = object.__new__(LagunaForCausalLM)
    bridge._dflash_controller = FakeController()
    bridge._dflash_tok = object()
    bridge._dflash_core = SimpleNamespace(config=SimpleNamespace(block_size=16))
    bridge.max_model_len = 100
    bridge._spec_is_greedy = lambda params: float(params.temperature[0]) <= 0
    bridge._host_rank4_tok_batch = lambda token, batch: token
    bridge._read_tokens_host = lambda token, batch: torch.tensor([23], dtype=torch.int32)
    copied = []
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", lambda source, target: copied.append((source, target)))
    greedy = SimpleNamespace(temperature=torch.tensor([0.0]))

    result = bridge._dflash_serve(
        torch.tensor([[7]]),
        torch.tensor([20]),
        [[0]],
        [{"block_size": 64}],
        None,
        greedy,
        False,
    )
    assert result == [bridge._dflash_tok]
    assert calls[0][0] == "proposal"
    assert calls[0][1]["known_bonus"] == 7 and calls[0][1]["position"] == 20
    assert calls[0][1]["verify_kwargs"]["page_tables_per_layer"] is None
    assert int(copied[0][0].reshape(-1)[0]) == 23

    # Async steady decode: the host position/token can lag one step behind. A steady step (no batch reset)
    # follows the controller's expected input; a batch reset keeps the host values for the strict check.
    bridge._dflash_controller.active = True
    bridge._dflash_controller.expected = (21, 9)
    bridge._dflash_serve(
        torch.tensor([[7]]), torch.tensor([20]), [[0]], [{"block_size": 64}], None, greedy, False, reset_batch=False
    )
    assert calls[-1][1]["known_bonus"] == 9 and calls[-1][1]["position"] == 21
    bridge._dflash_serve(torch.tensor([[7]]), torch.tensor([20]), [[0]], [{"block_size": 64}], None, greedy, False)
    assert calls[-1][1]["known_bonus"] == 7 and calls[-1][1]["position"] == 20
    bridge._dflash_controller.active = False
    bridge._dflash_controller.expected = (None, None)

    for position, expected_path in ((48, "proposal"), (49, "target"), (63, "target"), (64, "proposal")):
        bridge._dflash_serve(
            torch.tensor([[7]]),
            torch.tensor([position]),
            [[0]],
            [{"block_size": 64}],
            None,
            greedy,
            False,
        )
        assert calls[-1][0] == expected_path

    # With the scheduler's 16-token KV look-ahead applied, every residue runs a full round.
    monkeypatch.setattr(LagunaForCausalLM, "_dflash_lookahead_tokens", staticmethod(lambda: 16))
    for position in (49, 63):
        bridge._dflash_serve(
            torch.tensor([[7]]), torch.tensor([position]), [[0]], [{"block_size": 64}], None, greedy, False
        )
        assert calls[-1][0] == "proposal"
    monkeypatch.setattr(LagunaForCausalLM, "_dflash_lookahead_tokens", staticmethod(lambda: 15))
    bridge._dflash_serve(torch.tensor([[7]]), torch.tensor([49]), [[0]], [{"block_size": 64}], None, greedy, False)
    assert calls[-1][0] == "target"
    monkeypatch.setattr(LagunaForCausalLM, "_dflash_lookahead_tokens", staticmethod(lambda: 0))

    # Buffered commits from a round that safely began at residue <=48 are
    # already verified and must drain even when the scheduler cursor is 49..63.
    bridge._dflash_controller.pending_tokens = (99,)
    bridge._dflash_serve(
        torch.tensor([[7]]),
        torch.tensor([49]),
        [[0]],
        [{"block_size": 64}],
        None,
        greedy,
        False,
    )
    assert calls[-1][0] == "proposal"
    bridge._dflash_controller.pending_tokens = ()

    # P+16==max_model_len is an exact fit; P+16>max_model_len is rejected.
    bridge._dflash_serve(
        torch.tensor([[7]]),
        torch.tensor([84]),
        [[0]],
        [{"block_size": 64}],
        None,
        greedy,
        False,
    )
    assert calls[-1][0] == "proposal"

    with expect_error(RuntimeError, "B=1"):
        bridge._dflash_serve(
            torch.tensor([[7], [8]]),
            torch.tensor([20, 20]),
            [[0], [0]],
            [{"block_size": 64}],
            None,
            greedy,
            False,
        )
    with expect_error(RuntimeError, "exact-greedy"):
        bridge._dflash_serve(
            torch.tensor([[7]]),
            torch.tensor([20]),
            [[0]],
            [{"block_size": 64}],
            None,
            SimpleNamespace(temperature=torch.tensor([1.0])),
            False,
        )
    hybrid_tables = [object()]
    bridge._dflash_serve(
        torch.tensor([[7]]),
        torch.tensor([20]),
        [[0]],
        [{"block_size": 64}],
        hybrid_tables,
        greedy,
        False,
    )
    assert calls[-1][1]["verify_kwargs"]["page_tables_per_layer"] is hybrid_tables
    with expect_error(RuntimeError, "exceed"):
        bridge._dflash_serve(
            torch.tensor([[7]]),
            torch.tensor([85]),
            [[0]],
            [{"block_size": 64}],
            None,
            greedy,
            False,
        )


def test_normal_target_forward_sources_remain_dflash_free():
    assert "dflash" not in inspect.getsource(LagunaModel.prefill_layers).lower()
    assert "dflash" not in inspect.getsource(LagunaModel.decode_layers).lower()


def test_speculative_modes_serve_only_plain_greedy_requests():
    """DFlash and n-gram reproduce plain greedy decoding only; host sampling (logprobs), temperature > 0 and
    penalties go to the eager fallback (they used to kill the engine or have their penalties ignored)."""

    bridge = object.__new__(LagunaForCausalLM)
    greedy = SimpleNamespace(temperature=torch.tensor([0.0]))
    assert bridge._spec_serves_exactly(greedy)
    assert not bridge._spec_serves_exactly(None)
    assert not bridge._spec_serves_exactly(SimpleNamespace(temperature=torch.tensor([0.7])))
    assert not bridge._spec_serves_exactly(
        SimpleNamespace(temperature=torch.tensor([0.0]), repetition_penalty=torch.tensor([1.2]))
    )
    assert not bridge._spec_serves_exactly(
        SimpleNamespace(temperature=torch.tensor([0.0]), frequency_penalty=torch.tensor([0.5]))
    )


def test_eager_fallback_decode_samples_on_host_into_the_warm_token_buffer(monkeypatch):
    bridge = object.__new__(LagunaForCausalLM)
    bridge.vocab = 8
    bridge.gen = SimpleNamespace(counters={})
    logits = torch.zeros(1, 8)
    logits[0, 5] = 10.0
    calls = []
    bridge._page_table_to_device = lambda table: ("pt", table)
    bridge._decode_host_sampling = lambda tokens, pos, pt, kv, read_from_device: (
        calls.append(read_from_device) or (logits if read_from_device else ["logit-shards"])
    )
    bridge._host_rank4_tok_batch = lambda token, batch: token
    copied = []
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", lambda source, target: copied.append((source, target)))
    tokens, pos = torch.tensor([[7]]), torch.tensor([20])
    buffer = object()

    # Host sampling (logprobs): logits go back to the plugin, nothing is sampled here.
    assert bridge._eager_decode_fallback(tokens, pos, [[0]], [], None, False, None, {}, False, buffer, True) == [
        "logit-shards"
    ]
    # Penalized greedy request: sampled on the host (argmax here) and written to the warm token buffer.
    params = SimpleNamespace(temperature=torch.tensor([0.0]), repetition_penalty=torch.tensor([1.0]))
    assert bridge._eager_decode_fallback(tokens, pos, [[0]], [], None, False, params, {}, False, buffer, True) == [buffer]
    assert int(copied[-1][0].reshape(-1)[0]) == 5 and copied[-1][1] is buffer
    # A steady (overlapped) step continues from the token it sampled, not the plugin's possibly stale input.
    seen = []
    bridge._decode_host_sampling = lambda tokens, pos, pt, kv, read_from_device: (
        seen.append((int(tokens.reshape(-1)[0]), int(pos.reshape(-1)[0]))) or logits
    )
    stale_tokens, stale_pos = torch.tensor([[7]]), torch.tensor([20])
    bridge._eager_decode_fallback(stale_tokens, stale_pos, [[0]], [], None, False, params, {}, False, buffer, False)
    assert seen[-1] == (5, 21)
    # A batch reset (new request) takes the plugin's input; synchronous readback returns the sampled ids directly.
    assert bridge._eager_decode_fallback(tokens, pos, [[0]], [], None, False, params, {}, True, buffer, True).tolist() == [5]
    assert seen[-1] == (7, 20)
    assert bridge.gen.counters["spec_mode_fallback_decode"] == 3
