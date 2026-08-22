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
from models.autoports.poolside_laguna_xs_2_1.tt.dflash_reference import DFlashTargetAuxCapture, LagunaDFlashConfig
from models.autoports.poolside_laguna_xs_2_1.tt.dflash_serving import DFlashServedController, DFlashServingEnvelope
from models.autoports.poolside_laguna_xs_2_1.tt.generator_vllm import LagunaForCausalLM
from models.autoports.poolside_laguna_xs_2_1.tt.model import LagunaModel


def _published_config() -> LagunaDFlashConfig:
    config = LagunaDFlashConfig(
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=5,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=100352,
        draft_vocab_size=100352,
        max_position_embeddings=262144,
        rms_norm_eps=1e-6,
        rope_theta=500_000.0,
        sliding_window=512,
        hidden_act="silu",
        attention_bias=False,
        gating="per-head",
        num_experts=0,
        architectures=("DFlashLagunaForCausalLM",),
        torch_dtype="bfloat16",
        layer_types=("sliding_attention",) * 5,
        aux_hidden_state_layer_ids=(2, 14, 26, 34, 40),
        target_layer_ids=(1, 13, 25, 33, 39),
        block_size=16,
        mask_token_id=12,
        causal=True,
    )
    config.validate()
    return config


@dataclass(frozen=True)
class _FakeHidden:
    """Shape-correct auxiliary tensor metadata without a 10,240-wide allocation."""

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
        ({"enabled": True, "hybrid_kv": True}, "hybrid KV"),
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


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"device_count": 1}, "p150x2"),
        ({"max_batch_size": 2}, "max-num-seqs 1"),
        ({"prefix_enabled": True}, "PREFIX_CACHE=0"),
        ({"hybrid_enabled": True}, "HYBRID_KV=0"),
        ({"spec_mode": "1"}, "SPEC_DECODE"),
    ],
)
def test_vllm_dflash_envelope_rejects_unqualified_modes(override, match, expect_error):
    envelope = {
        "enabled": True,
        "device_count": 2,
        "max_batch_size": 1,
        "prefix_enabled": False,
        "hybrid_enabled": False,
        "spec_mode": "",
    }
    envelope.update(override)
    with expect_error(RuntimeError, match):
        LagunaForCausalLM._validate_dflash_serving_envelope(**envelope)
    LagunaForCausalLM._validate_dflash_serving_envelope(**{**envelope, "enabled": False})


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
    with expect_error(RuntimeError, "hybrid"):
        bridge.verify_greedy_decode_with_dflash_aux(
            [1],
            [9],
            page_table=[[4, 5]],
            kv_cache=[object()],
            page_tables_per_layer=[[[4, 5]]],
        )


def test_vllm_dflash_output_buffer_and_runtime_guards(monkeypatch, expect_error):
    calls = []

    class FakeController:
        pending_tokens = ()

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
    with expect_error(RuntimeError, "hybrid"):
        bridge._dflash_serve(
            torch.tensor([[7]]),
            torch.tensor([20]),
            [[0]],
            [{"block_size": 64}],
            [object()],
            greedy,
            False,
        )
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
