# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from models.demos.deepseek_v3_b1.demo import model_pipeline as model_pipeline_module
from models.demos.deepseek_v3_b1.demo.model_pipeline import ModelPipeline
from models.demos.deepseek_v3_b1.model import DecodeResult, TokenType

RUN_INFERENCE_TRACE_ENV = "DEEPSEEK_V3_B1_RUN_INFERENCE_TRACE"

TOKEN_TYPE_BY_NAME = {
    "BASE": TokenType.BASE,
    "SPEC": TokenType.SPEC,
}
TOKEN_TYPE_NAME_BY_VALUE = {value: name for name, value in TOKEN_TYPE_BY_NAME.items()}


class _TraceBackedModel:
    def __init__(self, read_results: list[DecodeResult]) -> None:
        self._read_results = list(read_results)
        self.read_count = 0
        self.writes = []

    def read_result(self) -> DecodeResult:
        if not self._read_results:
            pytest.fail("run_inference attempted to read more device packets than the trace provides")
        self.read_count += 1
        return self._read_results.pop(0)

    def write_input(
        self, token_id: int, prefill_token_id: int, user_id: int, position_id: int, token_type: int
    ) -> None:
        self.writes.append(
            {
                "token_id": int(token_id),
                "type": TOKEN_TYPE_NAME_BY_VALUE[int(token_type)],
                "pos": int(position_id),
                "user_id": int(user_id),
                "prefill_id": int(prefill_token_id),
            }
        )

    def assert_fully_consumed(self) -> None:
        assert not self._read_results, f"Trace has {len(self._read_results)} unread device packet(s)"


def _require_keys(value: dict, keys: tuple[str, ...], context: str) -> None:
    missing = [key for key in keys if key not in value]
    assert not missing, f"{context} missing required key(s): {missing}"


def _require_exact_keys(value: dict, keys: tuple[str, ...], context: str) -> None:
    _require_keys(value, keys, context)
    extra = sorted(set(value) - set(keys))
    assert not extra, f"{context} has unexpected key(s): {extra}"


def _validate_token(token: dict, context: str) -> None:
    _require_exact_keys(token, ("token_id", "pos"), context)


def _validate_packet(packet: dict, context: str) -> None:
    _require_exact_keys(packet, ("user_id", "type", "token_0", "token_1"), context)
    assert packet["type"] in TOKEN_TYPE_BY_NAME, f"{context}.type must be BASE or SPEC, got {packet['type']!r}"
    _validate_token(packet["token_0"], f"{context}.token_0")
    _validate_token(packet["token_1"], f"{context}.token_1")


def _validate_expected_write(write: dict, context: str) -> None:
    _require_exact_keys(write, ("token_id", "type", "pos", "user_id", "prefill_id"), context)
    assert write["type"] in TOKEN_TYPE_BY_NAME, f"{context}.type must be BASE or SPEC, got {write['type']!r}"


def _validate_metadata(metadata: dict, context: str) -> None:
    _require_exact_keys(metadata, ("num_accepts", "num_rejects"), context)
    assert isinstance(metadata["num_accepts"], int), f"{context}.num_accepts must be an integer"
    assert isinstance(metadata["num_rejects"], int), f"{context}.num_rejects must be an integer"


def _normalize_expected_write(write: dict) -> dict:
    return {
        "token_id": int(write["token_id"]),
        "type": write["type"],
        "pos": int(write["pos"]),
        "user_id": int(write["user_id"]),
        "prefill_id": int(write["prefill_id"]),
    }


def _load_trace(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as trace_file:
        trace = json.load(trace_file)

    _require_exact_keys(
        trace, ("schema_version", "name", "params", "device_to_host", "expected_host", "metadata"), str(path)
    )
    assert trace["schema_version"] == 1, f"{path} has unsupported schema_version={trace['schema_version']!r}"

    _require_exact_keys(trace["params"], ("prompt_token_ids", "max_new_tokens", "eos_token_id"), f"{path}.params")
    _require_exact_keys(trace["device_to_host"], ("prefill_results", "read_results"), f"{path}.device_to_host")
    _require_exact_keys(trace["expected_host"], ("generated_tokens", "writes", "read_count"), f"{path}.expected_host")
    _validate_metadata(trace["metadata"], f"{path}.metadata")

    for packet_idx, packet in enumerate(trace["device_to_host"]["prefill_results"]):
        _validate_packet(packet, f"{path}.device_to_host.prefill_results[{packet_idx}]")
    for packet_idx, packet in enumerate(trace["device_to_host"]["read_results"]):
        _validate_packet(packet, f"{path}.device_to_host.read_results[{packet_idx}]")
    for write_idx, write in enumerate(trace["expected_host"]["writes"]):
        _validate_expected_write(write, f"{path}.expected_host.writes[{write_idx}]")

    return trace


def _packet_to_decode_result(packet: dict) -> DecodeResult:
    token_0 = packet["token_0"]
    token_1 = packet["token_1"]
    return DecodeResult(
        token_0=int(token_0["token_id"]),
        token_type=TOKEN_TYPE_BY_NAME[packet["type"]],
        token_0_pos=int(token_0["pos"]),
        token_1=int(token_1["token_id"]),
        token_1_pos=int(token_1["pos"]),
        slot_id=int(packet["user_id"]),
    )


def _make_trace_backed_pipeline(trace: dict) -> tuple[ModelPipeline, _TraceBackedModel]:
    prefill_results = [_packet_to_decode_result(packet) for packet in trace["device_to_host"]["prefill_results"]]
    read_results = [_packet_to_decode_result(packet) for packet in trace["device_to_host"]["read_results"]]

    fake_model = _TraceBackedModel(read_results)
    pipeline = ModelPipeline.__new__(ModelPipeline)
    pipeline.pipeline = SimpleNamespace(my_stage_idx=0)
    pipeline.model = fake_model

    def prefill_forward(prompt_token_ids: list[int]) -> list[DecodeResult]:
        assert prompt_token_ids == trace["params"]["prompt_token_ids"]
        return list(prefill_results)

    pipeline.prefill_forward = prefill_forward
    return pipeline, fake_model


@pytest.fixture
def run_inference_trace_path() -> Path:
    raw_path = os.getenv(RUN_INFERENCE_TRACE_ENV)
    if raw_path is None or not raw_path.strip():
        pytest.skip(f"Set {RUN_INFERENCE_TRACE_ENV}=/path/to/reference_trace.json to run this host-side replay test")

    trace_path = Path(raw_path.strip()).expanduser().resolve()
    if not trace_path.is_file():
        pytest.fail(f"{RUN_INFERENCE_TRACE_ENV} does not point to a file: {trace_path}")
    return trace_path


def test_run_inference_replays_spec_decode_trace(run_inference_trace_path: Path, monkeypatch):
    trace_path = run_inference_trace_path
    trace = _load_trace(trace_path)
    pipeline, fake_model = _make_trace_backed_pipeline(trace)
    emitted_tokens = []

    monkeypatch.setattr(
        model_pipeline_module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(decode=lambda token_ids, **decode_kwargs: str(token_ids)),
    )

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
