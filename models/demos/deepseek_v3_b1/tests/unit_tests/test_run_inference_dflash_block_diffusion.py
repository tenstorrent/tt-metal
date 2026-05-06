# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Callable


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "dflash_block_diffusion"
RUN_INFERENCE_TRACE = FIXTURE_DIR / "run_inference_trace.json"


class TokenType:
    BASE = 0
    SPEC = 1


TOKEN_TYPE_BY_NAME = {
    "BASE": TokenType.BASE,
    "SPEC": TokenType.SPEC,
}
TOKEN_TYPE_NAME_BY_VALUE = {value: name for name, value in TOKEN_TYPE_BY_NAME.items()}


@dataclass(frozen=True)
class CandidateToken:
    token_id: int
    pos: int


@dataclass(frozen=True)
class DFlashVerificationResult:
    anchor_pos: int
    target_posterior: list[CandidateToken]


class _TraceBackedModel:
    def __init__(self, read_results: list[DFlashVerificationResult]) -> None:
        self._read_results = list(read_results)
        self.read_count = 0
        self.writes: list[dict[str, int | str]] = []

    def read_result(self) -> DFlashVerificationResult:
        if not self._read_results:
            raise AssertionError("DFlash run_inference attempted to read more device packets than the trace provides")
        self.read_count += 1
        return self._read_results.pop(0)

    def write_input(
        self,
        token_id: int,
        prefill_token_id: int,
        user_id: int,
        position_id: int,
        token_type: int,
        temperature: float = 0.0,
        top_k: int = 1,
        probability_mass_threshold: float = 1.0,
    ) -> None:
        del temperature, top_k, probability_mass_threshold
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


class _DFlashTraceBackedPipeline:
    """Temporary host-side seam until DeepSeek B1 exposes DFlash block proposals in ModelPipeline."""

    def __init__(self, trace: dict, model: _TraceBackedModel) -> None:
        self.pipeline = SimpleNamespace(my_stage_idx=0)
        self.model = model
        self._trace = trace
        self._draft_blocks = list(trace["host_draft_blocks"])
        self.last_inference_stats: dict[str, int | float] = {}

    def prefill_forward(self, prompt_token_ids: list[int]) -> list[CandidateToken]:
        assert prompt_token_ids == self._trace["params"]["prompt_token_ids"]
        return [_token_from_dict(packet["token"]) for packet in self._trace["device_to_host"]["prefill_results"]]

    def run_inference(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        on_token: Callable[[int], None] | None = None,
        eos_token_id: int | None = None,
        think_token_ids: list[int] | None = None,
        return_generated_tokens: bool = False,
    ) -> list[int] | None:
        del think_token_ids
        params = self._trace["params"]
        block_size = int(params["block_size"])
        max_length = len(prompt_token_ids) + max_new_tokens
        output_by_pos = {pos: token_id for pos, token_id in enumerate(prompt_token_ids)}

        prefill_tokens = self.prefill_forward(prompt_token_ids)
        assert len(prefill_tokens) == 1
        start = prefill_tokens[0].pos
        output_by_pos[start] = prefill_tokens[0].token_id

        generated_tokens: list[int] = []
        acceptance_lengths: list[int] = []
        num_accepts = 0
        num_rejects = 0

        for draft_block in self._draft_blocks:
            if len(generated_tokens) >= max_new_tokens:
                break

            block_token_ids = [int(token_id) for token_id in draft_block["token_ids"]]
            assert len(block_token_ids) == block_size
            assert int(draft_block["anchor_pos"]) == start
            assert output_by_pos[start] == block_token_ids[0]

            self._write_dflash_block(block_token_ids, anchor_pos=start)
            verified = self.model.read_result()
            assert verified.anchor_pos == start
            posterior_token_ids = [token.token_id for token in verified.target_posterior]

            matches = [
                draft_token_id == target_token_id
                for draft_token_id, target_token_id in zip(block_token_ids[1:], posterior_token_ids[:-1])
            ]
            accepted_after_anchor = _leading_true_count(matches)
            committed = accepted_after_anchor + 1
            acceptance_lengths.append(committed)
            num_accepts += accepted_after_anchor
            if accepted_after_anchor < block_size - 1:
                num_rejects += 1

            for offset, token_id in enumerate(block_token_ids[:committed]):
                output_pos = start + offset
                output_by_pos[output_pos] = token_id
                if output_pos >= len(prompt_token_ids) and len(generated_tokens) < max_new_tokens:
                    generated_tokens.append(token_id)
                    if on_token is not None:
                        on_token(token_id)
                    if eos_token_id is not None and token_id == eos_token_id:
                        break

            start += committed
            if start < max_length:
                output_by_pos[start] = posterior_token_ids[accepted_after_anchor]

        self.last_inference_stats = {
            "num_accepts": num_accepts,
            "num_rejects": num_rejects,
            "average_committed_tokens": sum(acceptance_lengths) / len(acceptance_lengths),
        }
        self.acceptance_lengths = acceptance_lengths
        self.output_token_ids = [output_by_pos[pos] for pos in range(max_length)]
        return generated_tokens if return_generated_tokens else None

    def _write_dflash_block(self, block_token_ids: list[int], *, anchor_pos: int) -> None:
        params = self._trace["params"]
        for offset, token_id in enumerate(block_token_ids):
            self.model.write_input(
                token_id,
                -1,
                user_id=0,
                position_id=anchor_pos + offset,
                token_type=TokenType.BASE if offset == 0 else TokenType.SPEC,
                temperature=float(params["temperature"]),
                top_k=int(params["top_k"]),
                probability_mass_threshold=float(params["top_p"]),
            )


def _leading_true_count(values: list[bool]) -> int:
    count = 0
    for value in values:
        if not value:
            break
        count += 1
    return count


def _require_keys(value: dict, keys: tuple[str, ...], context: str) -> None:
    missing = [key for key in keys if key not in value]
    assert not missing, f"{context} missing required key(s): {missing}"


def _token_from_dict(value: dict) -> CandidateToken:
    return CandidateToken(token_id=int(value["token_id"]), pos=int(value["pos"]))


def _packet_to_decode_result(packet: dict) -> DFlashVerificationResult:
    return DFlashVerificationResult(
        anchor_pos=int(packet["anchor_pos"]),
        target_posterior=[_token_from_dict(token) for token in packet["target_posterior"]],
    )


def _load_trace(path: Path = RUN_INFERENCE_TRACE) -> dict:
    with path.open("r", encoding="utf-8") as trace_file:
        trace = json.load(trace_file)
    return _validate_trace(trace, str(path))


def _validate_trace(trace: dict, context: str) -> dict:
    _require_keys(
        trace,
        (
            "schema_version",
            "name",
            "approach",
            "target_model",
            "draft_model",
            "implementation_status",
            "params",
            "device_to_host",
            "host_draft_blocks",
            "expected_host",
            "metadata",
        ),
        context,
    )
    assert trace["schema_version"] == 1
    assert trace["approach"] == "dflash_block_diffusion"
    assert trace["target_model"] == "moonshotai/Kimi-K2.5"
    assert trace["draft_model"] == "z-lab/Kimi-K2.5-DFlash"
    assert trace["implementation_status"]["production_hook"] == "missing"

    _require_keys(
        trace["params"],
        ("prompt_token_ids", "max_new_tokens", "eos_token_id", "block_size", "temperature", "top_k", "top_p"),
        f"{context}.params",
    )
    _require_keys(trace["device_to_host"], ("prefill_results", "read_results"), f"{context}.device_to_host")
    _require_keys(
        trace["expected_host"],
        ("generated_tokens", "output_token_ids", "acceptance_lengths", "read_count", "writes"),
        f"{context}.expected_host",
    )
    assert len(trace["host_draft_blocks"]) == len(trace["device_to_host"]["read_results"])
    return trace


def _normalize_expected_write(write: dict) -> dict[str, int | str]:
    return {
        "token_id": int(write["token_id"]),
        "type": write["type"],
        "pos": int(write["pos"]),
        "user_id": int(write["user_id"]),
        "prefill_id": int(write["prefill_id"]),
    }


def _make_trace_backed_pipeline(trace: dict) -> tuple[_DFlashTraceBackedPipeline, _TraceBackedModel]:
    read_results = [_packet_to_decode_result(packet) for packet in trace["device_to_host"]["read_results"]]
    fake_model = _TraceBackedModel(read_results)
    return _DFlashTraceBackedPipeline(trace, fake_model), fake_model


def test_run_inference_dflash_block_diffusion_replays_golden_trace() -> None:
    trace = _load_trace()
    pipeline, fake_model = _make_trace_backed_pipeline(trace)
    emitted_tokens: list[int] = []

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
    assert pipeline.output_token_ids == trace["expected_host"]["output_token_ids"]
    assert pipeline.acceptance_lengths == trace["expected_host"]["acceptance_lengths"]
    assert fake_model.writes == [_normalize_expected_write(write) for write in trace["expected_host"]["writes"]]
    assert fake_model.read_count == trace["expected_host"]["read_count"]
    assert pipeline.last_inference_stats["num_accepts"] == trace["metadata"]["num_accepts"]
    assert pipeline.last_inference_stats["num_rejects"] == trace["metadata"]["num_rejects"]
    assert math.isclose(
        pipeline.last_inference_stats["average_committed_tokens"],
        trace["metadata"]["average_committed_tokens"],
    )


def test_run_inference_dflash_block_diffusion_documents_missing_model_pipeline_hook() -> None:
    trace = _load_trace()

    assert trace["implementation_status"] == {
        "production_hook": "missing",
        "expected_hook": "ModelPipeline.run_inference with DFlash block proposals",
        "temporary_test_seam": "_DFlashTraceBackedPipeline",
    }
