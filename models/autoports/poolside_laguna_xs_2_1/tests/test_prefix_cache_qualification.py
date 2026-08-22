# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-free tests for the Laguna prefix-cache acceptance client."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from models.autoports.poolside_laguna_xs_2_1.tests import prefix_cache_qualification as Q


class _StreamingResponse:
    def __init__(self, chunks):
        self.lines = [f"data: {json.dumps(chunk)}\n\n".encode() for chunk in chunks]
        self.lines.append(b"data: [DONE]\n\n")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        return iter(self.lines)


def _stream_chunks(*, prompt=(10, 11), returned_prompt=None, cached_tokens=64):
    details = None if cached_tokens is None else {"cached_tokens": cached_tokens}
    usage = {
        "prompt_tokens": len(prompt),
        "completion_tokens": 2,
        "total_tokens": len(prompt) + 2,
    }
    if details is not None:
        usage["prompt_tokens_details"] = details
    return [
        {
            "id": "cmpl-test",
            "choices": [
                {
                    "index": 0,
                    "prompt_token_ids": list(prompt if returned_prompt is None else returned_prompt),
                    "token_ids": [31],
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "cmpl-test",
            "choices": [
                {
                    "index": 0,
                    "token_ids": [32],
                    "finish_reason": "length",
                }
            ],
        },
        {"id": "cmpl-test", "choices": [], "usage": usage},
    ]


def _completion(monkeypatch, *, chunks):
    client = Q.OpenAIClient("http://server", api_key=None, timeout=1)
    monkeypatch.setattr(client, "_open", lambda _request: _StreamingResponse(chunks))
    ticks = iter((10.0, 10.1, 10.3, 10.4))
    monkeypatch.setattr(Q.time, "perf_counter", lambda: next(ticks))
    payload = Q.build_completion_payload(
        model="model",
        prompt=[10, 11],
        output_len=2,
        cache_salt="salt",
        seed=1,
    )
    return client.completion(payload)


def _result(*, cached_tokens):
    return Q.CompletionResult(
        request_id="test",
        prompt_tokens=2,
        output_tokens=2,
        cached_tokens=cached_tokens,
        prompt_token_ids_sha256=Q.token_ids_sha256([10, 11]),
        token_ids=[31, 32],
        token_ids_sha256=Q.token_ids_sha256([31, 32]),
        finish_reason="length",
        ttft_ms=100.0,
        tpot_ms=10.0,
        e2e_ms=120.0,
    )


class _CacheOffClient:
    def __init__(self):
        self.health_calls = 0

    def get_text(self, path):
        if path == "/health":
            self.health_calls += 1
            return "healthy"
        assert path == "/metrics"
        return 'vllm:prefix_cache_queries_total{engine="0"} 0\n' 'vllm:prefix_cache_hits_total{engine="0"} 0\n'

    @staticmethod
    def get_json(path):
        assert path == "/v1/models"
        return {"data": [{"id": "model", "max_model_len": 131_072}]}

    @staticmethod
    def completion(payload):
        output_len = int(payload["max_tokens"])
        token_ids = list(range(100, 100 + output_len))
        prompt_ids = [int(token) for token in payload["prompt"]]
        return Q.CompletionResult(
            request_id="cache-off",
            prompt_tokens=len(prompt_ids),
            output_tokens=output_len,
            cached_tokens=None,
            prompt_token_ids_sha256=Q.token_ids_sha256(prompt_ids),
            token_ids=token_ids,
            token_ids_sha256=Q.token_ids_sha256(token_ids),
            finish_reason="length",
            ttft_ms=1_000.0,
            tpot_ms=10.0,
            e2e_ms=1_010.0,
        )


def test_exact_raw_tokens_are_reproducible_and_exclude_special_ids():
    first = Q.deterministic_token_ids(1_000, seed=17, vocab_size=128, excluded=(2, 9, 24))
    second = Q.deterministic_token_ids(1_000, seed=17, vocab_size=128, excluded=(2, 9, 24))

    assert first == second
    assert all(0 <= token < 128 for token in first)
    assert not {2, 9, 24}.intersection(first)
    assert Q.token_ids_sha256(first) == Q.token_ids_sha256(second)
    assert Q.token_ids_sha256(first) != Q.token_ids_sha256(first[:-1])


def test_partial_target_preserves_exact_prefix_and_changes_continuation():
    case = Q.CorrectnessCase("small", prefix_len=128, suffix_len=65, suffix_seed=99)
    base = Q.build_base_tokens(256, vocab_size=256, seed=7)

    target = Q.correctness_target(case, base, vocab_size=256)

    assert target[: case.prefix_len] == base[: case.prefix_len]
    assert len(target) == case.prompt_len
    assert target[case.prefix_len :] != base[case.prefix_len : case.prompt_len]


@pytest.mark.parametrize(
    ("prompt_len", "expected"),
    ((1, 0), (64, 0), (65, 64), (32_768, 32_704), (65_536, 65_472)),
)
def test_expected_full_hit_recomputes_final_block(prompt_len, expected):
    assert Q.expected_full_hit_tokens(prompt_len, 64) == expected


@pytest.mark.parametrize(
    ("raw_candidate_cached_tokens", "expected"),
    (
        (2_048, 0),
        (8_191, 0),
        (8_192, 8_192),
        (16_383, 8_192),
        (32_704, 24_576),
        (32_768, 32_768),
        (57_344, 57_344),
        (65_472, 57_344),
        (65_536, 65_536),
        (129_984, 122_880),
    ),
)
def test_canonical_8k_admission_floors_raw_candidate(raw_candidate_cached_tokens, expected):
    assert Q.expected_admitted_hit_tokens(raw_candidate_cached_tokens) == expected


def test_default_case_definitions_have_safe_exact_cached_token_counts():
    partial_expected = {
        "partial_2k": (2_048, 0),
        "partial_32k": (32_768, 32_768),
        "partial_65k": (65_536, 65_536),
        "partial_near_cap": (129_984, 122_880),
    }
    assert {
        case.name: (
            case.prefix_len,
            Q.expected_admitted_hit_tokens(case.prefix_len),
        )
        for case in Q.CORRECTNESS_CASES
    } == partial_expected

    full_expected = {
        "full_32k": (32_704, 24_576),
        "full_65k": (65_472, 57_344),
    }
    assert {
        case.name: (
            Q.expected_full_hit_tokens(case.prompt_len, Q.DEFAULT_BLOCK_SIZE),
            Q.expected_admitted_hit_tokens(Q.expected_full_hit_tokens(case.prompt_len, Q.DEFAULT_BLOCK_SIZE)),
        )
        for case in Q.PERFORMANCE_CASES
    } == full_expected


def test_poison_order_covers_oldest_hash_then_safe_32k_admission():
    plan = Q.poison_order_plan(output_len=128, block_size=64)

    assert [step.name for step in plan] == ["seed_2k", "target_32k_after_2k", "repeat_32k"]
    assert [step.prompt_len for step in plan] == [2_048, 32_768, 32_768]
    assert [step.output_len for step in plan] == [1, 128, 128]
    assert [step.raw_candidate_cached_tokens for step in plan] == [0, 2_048, 32_704]
    assert [step.expected_cached_tokens for step in plan] == [0, 0, 24_576]
    assert [step.compare_with_full_32k_oracle for step in plan] == [False, True, True]

    base = Q.build_base_tokens(Q.POISON_TARGET_LEN, vocab_size=256, seed=7)
    prompt_hashes = [Q.token_ids_sha256(base[: step.prompt_len]) for step in plan]
    assert prompt_hashes[0] != prompt_hashes[1]
    assert prompt_hashes[1] == prompt_hashes[2]


def test_decode_boundary_target_includes_generated_ids_beyond_8k():
    spec = Q.decode_boundary_spec()

    assert spec.admission_boundary == 8_192
    assert spec.seed_prompt_len == 8_180
    assert spec.seed_output_len == 16
    assert spec.target_appended_decode_tokens == 13
    assert spec.target_prompt_len == 8_193
    assert spec.potential_poisoned_hit_tokens == 8_192
    assert spec.expected_cached_tokens == 0
    assert spec.seed_prompt_len < spec.admission_boundary
    assert spec.seed_prompt_len + spec.seed_output_len > spec.admission_boundary

    seed_prompt = list(range(spec.seed_prompt_len))
    generated_ids = list(range(90_000, 90_000 + spec.seed_output_len))
    target = Q.build_decode_boundary_target(seed_prompt, generated_ids, spec)

    assert target[: spec.seed_prompt_len] == seed_prompt
    assert target[spec.seed_prompt_len :] == generated_ids[:13]
    assert len(target) == spec.target_prompt_len
    assert Q.token_ids_sha256(target) != Q.token_ids_sha256(seed_prompt)


def test_completion_payload_uses_exact_token_interface():
    payload = Q.build_completion_payload(
        model="model",
        prompt=(101, 202, 303),
        output_len=8,
        cache_salt="qualification-salt",
        seed=4,
    )

    assert payload["prompt"] == [101, 202, 303]
    assert payload["add_special_tokens"] is False
    assert payload["return_token_ids"] is True
    assert payload["cache_salt"] == "qualification-salt"
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["temperature"] == 0
    assert payload["ignore_eos"] is True


def test_cache_off_phase_accepts_missing_details_and_records_final_health(monkeypatch):
    monkeypatch.setattr(
        Q,
        "CORRECTNESS_CASES",
        (Q.CorrectnessCase("partial", prefix_len=64, suffix_len=1, suffix_seed=99),),
    )
    monkeypatch.setattr(
        Q,
        "PERFORMANCE_CASES",
        (Q.PerformanceCase("full", prompt_len=64, minimum_speedup=1.0),),
    )
    args = SimpleNamespace(
        base_url="http://server",
        model="model",
        block_size=64,
        cache_admission_granularity=8_192,
        vocab_size=256,
        seed=7,
        performance_output_len=2,
        correctness_output_len=2,
        repetitions=2,
        run_id="cache-off-test",
    )
    client = _CacheOffClient()

    artifact = Q.run_off(args, client)

    assert artifact["passed"] is True
    assert artifact["config"]["expected_prefix_cache_enabled"] is False
    assert artifact["config"]["cache_admission_granularity"] == 8_192
    assert artifact["config"]["cache_admission_policy"] == "canonical_floor_v1"
    assert artifact["correctness"]["partial"]["oracle"]["cached_tokens"] is None
    assert artifact["metrics"]["delta"]["prefix_cache_hits"] == 0
    assert artifact["server"]["health_body"] == "healthy"
    assert artifact["server"]["final_health_body"] == "healthy"
    assert client.health_calls == 2
    assert all(verdict["passed"] for verdict in artifact["verdicts"].values())


PROMETHEUS_FIXTURE = """
# HELP vllm:prefix_cache_queries_total Prefix cache queries in tokens.
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total{engine="0",model_name="laguna"} 32768
vllm:prefix_cache_queries_total{engine="1",model_name="laguna"} 64 1787328000
# HELP vllm:prefix_cache_hits_total Prefix cache hits in tokens.
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total{engine="0",model_name="laguna"} 32704
vllm:prefix_cache_hits_total{engine="1",model_name="laguna"} 0
"""


def test_prometheus_total_counter_fixture_sums_label_sets(expect_error):
    assert Q.parse_prometheus_counter(PROMETHEUS_FIXTURE, Q.PREFIX_CACHE_QUERIES_METRIC) == 32_832
    assert Q.parse_prometheus_counter(PROMETHEUS_FIXTURE, Q.PREFIX_CACHE_HITS_METRIC) == 32_704
    with expect_error(Q.QualificationError, "did not expose"):
        Q.parse_prometheus_counter(PROMETHEUS_FIXTURE, "vllm:prefix_cache_hits")


def test_prometheus_counter_rejects_invalid_sample(expect_error):
    with expect_error(Q.QualificationError, "invalid Prometheus sample"):
        Q.parse_prometheus_counter(
            'vllm:prefix_cache_hits_total{engine="0"} not-a-number',
            Q.PREFIX_CACHE_HITS_METRIC,
        )


@pytest.mark.parametrize("minimum_speedup", (3.0, 2.0))
def test_performance_gate_accepts_threshold_boundaries(minimum_speedup):
    oracle = {"median_ttft_ms": 1_000.0, "median_tpot_ms": 10.0}
    cold = {"median_ttft_ms": 1_050.0, "median_tpot_ms": 10.2}
    hit = {
        "median_ttft_ms": 1_000.0 / minimum_speedup,
        "median_tpot_ms": 10.2,
    }

    gate = Q.performance_gate(
        oracle=oracle,
        cold=cold,
        hit=hit,
        minimum_speedup=minimum_speedup,
    )

    assert gate["passed"] is True
    assert all(gate["checks"].values())


@pytest.mark.parametrize(
    ("cold_ttft", "cold_tpot", "hit_ttft", "hit_tpot", "failed_check"),
    (
        (1_000.0, 10.0, 334.0, 10.0, "hit_ttft_speedup"),
        (1_051.0, 10.0, 100.0, 10.0, "cold_ttft_regression"),
        (1_000.0, 10.21, 100.0, 10.0, "cold_tpot_regression"),
        (1_000.0, 10.0, 100.0, 10.21, "hit_tpot_regression"),
    ),
)
def test_performance_gate_rejects_each_regression(cold_ttft, cold_tpot, hit_ttft, hit_tpot, failed_check):
    gate = Q.performance_gate(
        oracle={"median_ttft_ms": 1_000.0, "median_tpot_ms": 10.0},
        cold={"median_ttft_ms": cold_ttft, "median_tpot_ms": cold_tpot},
        hit={"median_ttft_ms": hit_ttft, "median_tpot_ms": hit_tpot},
        minimum_speedup=3.0,
    )

    assert gate["passed"] is False
    assert gate["checks"][failed_check] is False


def test_performance_gate_requires_hit_to_improve_candidate_cold():
    gate = Q.performance_gate(
        oracle={"median_ttft_ms": 1_000.0, "median_tpot_ms": 10.0},
        cold={"median_ttft_ms": 1_000.0, "median_tpot_ms": 10.0},
        hit={"median_ttft_ms": 1_000.0, "median_tpot_ms": 10.0},
        minimum_speedup=1.0,
    )

    assert gate["checks"]["hit_ttft_speedup"] is True
    assert gate["checks"]["hit_ttft_improves_candidate_cold"] is False
    assert gate["passed"] is False


def test_streaming_completion_confirms_prompt_and_output_token_ids(monkeypatch):
    result = _completion(monkeypatch, chunks=_stream_chunks())

    assert result.request_id == "cmpl-test"
    assert result.cached_tokens == 64
    assert result.prompt_token_ids_sha256 == Q.token_ids_sha256([10, 11])
    assert result.token_ids == [31, 32]
    assert result.finish_reason == "length"
    assert result.ttft_ms == pytest.approx(100.0)
    assert result.tpot_ms == pytest.approx(200.0)
    assert result.e2e_ms == pytest.approx(400.0)


def test_cache_off_may_omit_details_but_cache_on_is_strict(monkeypatch, expect_error):
    result = _completion(monkeypatch, chunks=_stream_chunks(cached_tokens=None))

    assert result.cached_tokens is None
    Q._assert_result(
        result,
        expected_output_len=2,
        expected_cached_tokens=0,
        allow_missing_cached_tokens=True,
    )
    with expect_error(Q.QualificationError, "prompt_tokens_details"):
        Q._assert_result(result, expected_output_len=2, expected_cached_tokens=0)


def test_streaming_completion_rejects_changed_raw_prompt(monkeypatch, expect_error):
    with expect_error(Q.QualificationError, "exact raw prompt"):
        _completion(
            monkeypatch,
            chunks=_stream_chunks(returned_prompt=(10, 12)),
        )


def test_oracle_token_hash_is_verified(expect_error):
    tokens = [31, 32]
    oracle = {
        "correctness": {
            "partial": {
                "oracle": {
                    "token_ids": tokens,
                    "token_ids_sha256": Q.token_ids_sha256(tokens),
                }
            }
        }
    }

    assert Q._oracle_output(oracle, "correctness", "partial") == tokens
    oracle["correctness"]["partial"]["oracle"]["token_ids_sha256"] = "bad"
    with expect_error(Q.QualificationError, "token hash is invalid"):
        Q._oracle_output(oracle, "correctness", "partial")


def test_cached_token_assertion_rejects_wrong_count(expect_error):
    with expect_error(Q.QualificationError, "expected 64 cached tokens"):
        Q._assert_result(_result(cached_tokens=0), expected_output_len=2, expected_cached_tokens=64)
