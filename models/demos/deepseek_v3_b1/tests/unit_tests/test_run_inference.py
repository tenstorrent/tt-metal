# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from models.demos.deepseek_v3_b1.demo.model_pipeline import (
    DEFAULT_RELAXED_ACCEPT_DELTA,
    DEFAULT_RELAXED_ACCEPT_TOPN,
    ModelPipeline,
)
from models.demos.deepseek_v3_b1.model import CandidateToken, DecodeResult, TokenType

RUN_INFERENCE_TRACE_ENV = "DEEPSEEK_V3_B1_RUN_INFERENCE_TRACE"
DEFAULT_RUN_INFERENCE_TRACE = Path(__file__).parent / "reference_data" / "reference_trace_spec_decode_5_tokens.json"
SYNTHETIC_RUN_INFERENCE_TRACE_SUITE = (
    Path(__file__).parent / "reference_data" / "reference_trace_synthetic_mtp4_corner_cases.json"
)

TOKEN_TYPE_BY_NAME = {
    "PREFILL": TokenType.PREFILL,
    "BASE": TokenType.BASE,
    "SPEC": TokenType.SPEC,
}
TOKEN_TYPE_NAME_BY_VALUE = {value: name for name, value in TOKEN_TYPE_BY_NAME.items()}
EXCEPTION_BY_NAME = {
    "RuntimeError": RuntimeError,
    "ValueError": ValueError,
}


class _TraceBackedModel:
    def __init__(
        self,
        read_results: list[DecodeResult],
        *,
        capture_dynamic_fields: bool = False,
        expected_window_tokens: int = 0,
    ) -> None:
        self._read_results = list(read_results)
        self._capture_dynamic_fields = capture_dynamic_fields
        self._expected_window_tokens = int(expected_window_tokens)
        self.read_count = 0
        self.writes = []

    def read_result(self) -> DecodeResult:
        if not self._read_results:
            pytest.fail("run_inference attempted to read more device packets than the trace provides")
        self.read_count += 1
        return self._read_results.pop(0)

    def write_input(
        self,
        token_id: int,
        prefill_token_ids: int | list[int],
        request_id: int,
        position_id: int,
        token_type: int,
        lane_idx: int = 0,
        window_start_pos: int | None = None,
        num_window_tokens: int = 0,
        temperature: float = 1.0,
        top_k: int = 32,
        probability_mass_threshold: float = 1.0,
        top_p: float = 1.0,
    ) -> None:
        write = {
            "token_id": int(token_id),
            "type": TOKEN_TYPE_NAME_BY_VALUE[int(token_type)],
            "pos": int(position_id),
            "request_id": int(request_id),
            "prefill_id": int(prefill_token_ids[0] if isinstance(prefill_token_ids, list) else prefill_token_ids),
        }
        if self._capture_dynamic_fields:
            write.update(
                {
                    "lane_idx": int(lane_idx),
                    "window_start_pos": int(position_id - lane_idx if window_start_pos is None else window_start_pos),
                    "num_window_tokens": int(num_window_tokens or self._expected_window_tokens),
                }
            )
        self.writes.append(write)

    def assert_fully_consumed(self) -> None:
        assert not self._read_results, f"Trace has {len(self._read_results)} unread device packet(s)"


def _require_keys(value: dict, keys: tuple[str, ...], context: str) -> None:
    missing = [key for key in keys if key not in value]
    assert not missing, f"{context} missing required key(s): {missing}"


def _require_exact_keys(value: dict, keys: tuple[str, ...], context: str) -> None:
    _require_keys(value, keys, context)
    extra = sorted(set(value) - set(keys))
    assert not extra, f"{context} has unexpected key(s): {extra}"


def _require_exact_keys_with_optional(
    value: dict, required_keys: tuple[str, ...], optional_keys: tuple[str, ...], context: str
) -> None:
    _require_keys(value, required_keys, context)
    extra = sorted(set(value) - set(required_keys) - set(optional_keys))
    assert not extra, f"{context} has unexpected key(s): {extra}"


def _validate_token(token: dict, context: str) -> None:
    _require_exact_keys(token, ("token_id", "pos"), context)


def _validate_packet(packet: dict, context: str) -> None:
    _require_exact_keys(packet, ("request_id", "type", "token_0", "token_1"), context)
    assert packet["type"] in TOKEN_TYPE_BY_NAME, f"{context}.type must be PREFILL, BASE or SPEC, got {packet['type']!r}"
    _validate_token(packet["token_0"], f"{context}.token_0")
    _validate_token(packet["token_1"], f"{context}.token_1")


def _validate_dynamic_packet(packet: dict, context: str) -> None:
    _require_exact_keys(
        packet,
        (
            "request_id",
            "type",
            "lane_idx",
            "window_start_pos",
            "num_window_tokens",
            "tokens",
            "target_topn_tokens",
            "target_topn_probs",
        ),
        context,
    )
    assert packet["type"] in TOKEN_TYPE_BY_NAME, f"{context}.type must be PREFILL, BASE or SPEC, got {packet['type']!r}"
    assert isinstance(packet["tokens"], list), f"{context}.tokens must be a list"
    assert len(packet["tokens"]) == int(
        packet["num_window_tokens"]
    ), f"{context}.tokens length must match num_window_tokens"
    for token_idx, token in enumerate(packet["tokens"]):
        _validate_token(token, f"{context}.tokens[{token_idx}]")
    assert isinstance(packet["target_topn_tokens"], list), f"{context}.target_topn_tokens must be a list"
    assert isinstance(packet["target_topn_probs"], list), f"{context}.target_topn_probs must be a list"
    assert len(packet["target_topn_tokens"]) == len(
        packet["target_topn_probs"]
    ), f"{context}.target_topn_tokens and target_topn_probs must have the same length"


def _validate_expected_write(write: dict, context: str) -> None:
    _require_exact_keys(write, ("token_id", "type", "pos", "request_id", "prefill_id"), context)
    assert write["type"] in TOKEN_TYPE_BY_NAME, f"{context}.type must be PREFILL, BASE or SPEC, got {write['type']!r}"


def _validate_dynamic_expected_write(write: dict, context: str) -> None:
    _require_exact_keys(
        write,
        ("token_id", "type", "pos", "request_id", "prefill_id", "lane_idx", "window_start_pos", "num_window_tokens"),
        context,
    )
    assert write["type"] in TOKEN_TYPE_BY_NAME, f"{context}.type must be PREFILL, BASE or SPEC, got {write['type']!r}"


def _validate_metadata(metadata: dict, context: str) -> None:
    _require_exact_keys(metadata, ("num_accepts", "num_rejects"), context)
    assert isinstance(metadata["num_accepts"], int), f"{context}.num_accepts must be an integer"
    assert isinstance(metadata["num_rejects"], int), f"{context}.num_rejects must be an integer"


def _validate_expected_error(expected_error: dict, context: str) -> None:
    _require_exact_keys(expected_error, ("type", "match"), context)
    assert expected_error["type"] in EXCEPTION_BY_NAME, f"{context}.type must be one of {sorted(EXCEPTION_BY_NAME)}"
    assert isinstance(expected_error["match"], str), f"{context}.match must be a string"


def _normalize_expected_write(write: dict) -> dict:
    normalized = {
        "token_id": int(write["token_id"]),
        "type": write["type"],
        "pos": int(write["pos"]),
        "request_id": int(write["request_id"]),
        "prefill_id": int(write["prefill_id"]),
    }
    if "lane_idx" in write:
        normalized.update(
            {
                "lane_idx": int(write["lane_idx"]),
                "window_start_pos": int(write["window_start_pos"]),
                "num_window_tokens": int(write["num_window_tokens"]),
            }
        )
    return normalized


def _validate_trace(trace: dict, context: str) -> dict:
    _require_exact_keys_with_optional(
        trace,
        ("schema_version", "name", "params", "device_to_host"),
        ("expected_host", "metadata", "expected_error"),
        context,
    )
    has_expected_host = "expected_host" in trace
    has_expected_error = "expected_error" in trace
    assert (
        has_expected_host != has_expected_error
    ), f"{context} must define exactly one of expected_host or expected_error"
    assert ("metadata" in trace) == has_expected_host, f"{context}.metadata is required only for successful traces"
    assert trace["schema_version"] in (1, 2), f"{context} has unsupported schema_version={trace['schema_version']!r}"

    if trace["schema_version"] == 1:
        _require_exact_keys_with_optional(
            trace["params"],
            ("prompt_token_ids", "max_new_tokens", "eos_token_id"),
            ("relaxed_accept_topn", "relaxed_accept_delta"),
            f"{context}.params",
        )
    else:
        _require_exact_keys_with_optional(
            trace["params"],
            ("prompt_token_ids", "max_new_tokens", "eos_token_id", "num_speculative_tokens"),
            ("relaxed_accept_topn", "relaxed_accept_delta"),
            f"{context}.params",
        )
    _require_exact_keys(trace["device_to_host"], ("prefill_results", "read_results"), f"{context}.device_to_host")
    if has_expected_host:
        _require_exact_keys(
            trace["expected_host"], ("generated_tokens", "writes", "read_count"), f"{context}.expected_host"
        )
        _validate_metadata(trace["metadata"], f"{context}.metadata")
    else:
        _validate_expected_error(trace["expected_error"], f"{context}.expected_error")

    packet_validator = _validate_packet if trace["schema_version"] == 1 else _validate_dynamic_packet
    write_validator = _validate_expected_write if trace["schema_version"] == 1 else _validate_dynamic_expected_write
    for packet_idx, packet in enumerate(trace["device_to_host"]["prefill_results"]):
        packet_validator(packet, f"{context}.device_to_host.prefill_results[{packet_idx}]")
    for packet_idx, packet in enumerate(trace["device_to_host"]["read_results"]):
        packet_validator(packet, f"{context}.device_to_host.read_results[{packet_idx}]")
    if has_expected_host:
        for write_idx, write in enumerate(trace["expected_host"]["writes"]):
            write_validator(write, f"{context}.expected_host.writes[{write_idx}]")

    return trace


def _load_trace(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as trace_file:
        trace = json.load(trace_file)

    return _validate_trace(trace, str(path))


def _load_trace_suite(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as trace_file:
        suite = json.load(trace_file)

    _require_exact_keys(suite, ("suite_schema_version", "name", "traces"), str(path))
    assert suite["suite_schema_version"] == 1, f"{path} has unsupported suite_schema_version"
    assert isinstance(suite["traces"], list), f"{path}.traces must be a list"
    return [_validate_trace(trace, f"{path}.traces[{trace_idx}]") for trace_idx, trace in enumerate(suite["traces"])]


def _single_reference_trace_paths() -> list[Path]:
    reference_data_dir = DEFAULT_RUN_INFERENCE_TRACE.parent
    return sorted(
        trace_path
        for trace_path in reference_data_dir.glob("reference_trace_*.json")
        if trace_path != SYNTHETIC_RUN_INFERENCE_TRACE_SUITE
    )


def _packet_to_decode_result(packet: dict, *, force_prefill: bool = False) -> DecodeResult:
    token_type = TokenType.PREFILL if force_prefill else TOKEN_TYPE_BY_NAME[packet["type"]]
    if "tokens" in packet:
        tokens = packet["tokens"]
        candidate_tokens = [CandidateToken(int(token["token_id"]), int(token["pos"])) for token in tokens]
        return DecodeResult(
            token_type=token_type,
            tokens=candidate_tokens,
            request_id=int(packet["request_id"]),
            lane_idx=int(packet["lane_idx"]),
            position_id=int(packet["window_start_pos"]) + int(packet["lane_idx"]) - 1,
            p_top15_indices=[int(token) for token in packet["target_topn_tokens"]],
            p_top15_scores=[float(prob) for prob in packet["target_topn_probs"]],
        )

    token_0 = packet["token_0"]
    token_1 = packet["token_1"]
    lane_idx = 0 if token_type in (TokenType.PREFILL, TokenType.BASE) else 1
    token_0_pos = int(token_0["pos"])
    return DecodeResult(
        token_type=token_type,
        tokens=[
            CandidateToken(int(token_0["token_id"]), token_0_pos),
            CandidateToken(int(token_1["token_id"]), int(token_1["pos"])),
        ],
        request_id=int(packet["request_id"]),
        lane_idx=lane_idx,
        position_id=token_0_pos - 1,
    )


def _make_trace_backed_pipeline(trace: dict) -> tuple[ModelPipeline, _TraceBackedModel]:
    prefill_results = [
        _packet_to_decode_result(packet, force_prefill=True) for packet in trace["device_to_host"]["prefill_results"]
    ]
    read_results = [_packet_to_decode_result(packet) for packet in trace["device_to_host"]["read_results"]]

    num_speculative_tokens = int(trace["params"].get("num_speculative_tokens", 1))
    fake_model = _TraceBackedModel(
        read_results,
        capture_dynamic_fields=trace["schema_version"] == 2,
        expected_window_tokens=num_speculative_tokens + 1,
    )
    pipeline = ModelPipeline.__new__(ModelPipeline)
    pipeline.pipeline = SimpleNamespace(my_stage_idx=0)
    pipeline.model = fake_model
    pipeline.num_speculative_tokens = num_speculative_tokens
    pipeline.temperature = float(trace["params"].get("temperature", 0.6))
    pipeline.top_k = int(trace["params"].get("top_k", 32))
    pipeline.top_p = float(trace["params"].get("top_p", 1.0))
    pipeline._set_relaxed_acceptance_params(
        trace["params"].get("relaxed_accept_topn", DEFAULT_RELAXED_ACCEPT_TOPN),
        trace["params"].get("relaxed_accept_delta", DEFAULT_RELAXED_ACCEPT_DELTA),
    )

    def prefill_forward(prompt_token_ids: list[int]) -> list[DecodeResult]:
        assert prompt_token_ids == trace["params"]["prompt_token_ids"]
        return list(prefill_results)

    pipeline.prefill_forward = prefill_forward
    return pipeline, fake_model


def test_relaxed_acceptance_uses_configured_topn_and_delta():
    pipeline = ModelPipeline.__new__(ModelPipeline)
    result = DecodeResult(
        token_type=TokenType.BASE,
        tokens=[CandidateToken(101, 4)],
        p_top15_indices=[101, 202, 303],
        p_top15_scores=[0.9, 0.5, 0.35],
    )

    pipeline._set_relaxed_acceptance_params(relaxed_accept_topn=2, relaxed_accept_delta=0.6)
    assert not pipeline._relaxed_accepts_speculation(303, result)

    pipeline._set_relaxed_acceptance_params(relaxed_accept_topn=3, relaxed_accept_delta=0.6)
    assert pipeline._relaxed_accepts_speculation(303, result)

    pipeline._set_relaxed_acceptance_params(relaxed_accept_topn=3, relaxed_accept_delta=0.5)
    assert not pipeline._relaxed_accepts_speculation(303, result)


def _replay_trace(trace: dict) -> None:
    pipeline, fake_model = _make_trace_backed_pipeline(trace)
    emitted_tokens = []

    if "expected_error" in trace:
        expected_error = trace["expected_error"]
        with pytest.raises(EXCEPTION_BY_NAME[expected_error["type"]], match=expected_error["match"]):
            pipeline.run_inference(
                trace["params"]["prompt_token_ids"],
                trace["params"]["max_new_tokens"],
                on_token=emitted_tokens.append,
                eos_token_id=trace["params"]["eos_token_id"],
                return_generated_tokens=True,
            )
        return

    generated_tokens = pipeline.run_inference(
        trace["params"]["prompt_token_ids"],
        trace["params"]["max_new_tokens"],
        on_token=emitted_tokens.append,
        eos_token_id=trace["params"]["eos_token_id"],
        return_generated_tokens=True,
    )

    fake_model.assert_fully_consumed()
    assert generated_tokens == trace["expected_host"]["generated_tokens"]
    assert emitted_tokens == trace["expected_host"]["generated_tokens"]
    assert fake_model.writes == [_normalize_expected_write(write) for write in trace["expected_host"]["writes"]]
    assert fake_model.read_count == trace["expected_host"]["read_count"]
    assert pipeline.last_inference_stats["num_accepts"] == trace["metadata"]["num_accepts"]
    assert pipeline.last_inference_stats["num_rejects"] == trace["metadata"]["num_rejects"]


def _replay_trace_file(trace_path: Path) -> None:
    _replay_trace(_load_trace(trace_path))


def _synthetic_trace_cases() -> list[dict]:
    return _load_trace_suite(SYNTHETIC_RUN_INFERENCE_TRACE_SUITE)


@pytest.mark.parametrize("trace", _synthetic_trace_cases(), ids=lambda trace: trace["name"])
def test_run_inference_replays_synthetic_corner_case_trace(trace: dict):
    _replay_trace(trace)


@pytest.fixture
def run_inference_trace_path() -> Path:
    raw_path = os.getenv(RUN_INFERENCE_TRACE_ENV)
    if raw_path and raw_path.strip():
        trace_path = Path(raw_path.strip()).expanduser().resolve()
    else:
        trace_path = DEFAULT_RUN_INFERENCE_TRACE

    if not trace_path.is_file():
        pytest.fail(f"run_inference trace does not point to a file: {trace_path}")
    return trace_path


def test_run_inference_replays_spec_decode_trace(run_inference_trace_path: Path):
    _replay_trace_file(run_inference_trace_path)


@pytest.mark.parametrize("trace_path", _single_reference_trace_paths(), ids=lambda trace_path: trace_path.name)
def test_run_inference_replays_all_reference_traces(trace_path: Path):
    _replay_trace_file(trace_path)
