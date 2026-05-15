# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from models.demos.blaze_spec_decode.dflash_moonshotai_kimi_k2_5.tests.unit_tests.dflash_golden_ops import (
    golden_accepted_after_anchor,
)
from models.demos.blaze_spec_decode.dflash_moonshotai_kimi_k2_5.tests.unit_tests.dflash_test_modes import (
    USE_GOLDEN_PARAMS,
    require_golden_or_not_implemented,
)
from models.demos.blaze_spec_decode.dflash_moonshotai_kimi_k2_5.tests.unit_tests.dflash_test_utils import (
    assert_decode_manager_outputs_match,
    expected_decode_manager_outputs,
    load_validated_dflash_reference,
)


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
    accepted_after_anchor: int
    committed_tokens: int


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
    """Temporary host-side seam until the Blaze spec-decode pipeline exposes DFlash block proposals."""

    def __init__(self, reference: dict, model: _TraceBackedModel) -> None:
        self.pipeline = SimpleNamespace(my_stage_idx=0)
        self.model = model
        self._reference = reference
        self._trace = reference["host_trace"]
        self._draft_blocks = list(self._trace["draft_blocks"])
        self.last_inference_stats: dict[str, int | float] = {}

    def prefill_forward(self, prompt_token_ids: list[int]) -> list[CandidateToken]:
        assert prompt_token_ids == self._reference["prompt"]["input_ids"][0].tolist()
        first_block = self._draft_blocks[0]
        return [CandidateToken(token_id=int(first_block["token_ids"][0]), pos=int(first_block["anchor_pos"]))]

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
        block_size = int(self._reference["config"]["runtime_block_size"])
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

            accepted_after_anchor = golden_accepted_after_anchor(block_token_ids, posterior_token_ids)
            committed = accepted_after_anchor + 1
            assert accepted_after_anchor == verified.accepted_after_anchor
            assert committed == verified.committed_tokens
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
        output_length = len(prompt_token_ids) + len(generated_tokens)
        self.output_token_ids = [output_by_pos[pos] for pos in range(output_length)]
        return generated_tokens if return_generated_tokens else None

    def _write_dflash_block(self, block_token_ids: list[int], *, anchor_pos: int) -> None:
        for offset, token_id in enumerate(block_token_ids):
            self.model.write_input(
                token_id,
                -1,
                user_id=0,
                position_id=anchor_pos + offset,
                token_type=TokenType.BASE if offset == 0 else TokenType.SPEC,
            )


def _packet_to_decode_result(packet: dict) -> DFlashVerificationResult:
    anchor_pos = int(packet["anchor_pos"])
    return DFlashVerificationResult(
        anchor_pos=anchor_pos,
        target_posterior=[
            CandidateToken(token_id=int(token_id), pos=anchor_pos + offset + 1)
            for offset, token_id in enumerate(packet["target_posterior_token_ids"])
        ],
        accepted_after_anchor=int(packet["accepted_after_anchor"]),
        committed_tokens=int(packet["committed_tokens"]),
    )


def _make_trace_backed_pipeline(reference: dict) -> tuple[_DFlashTraceBackedPipeline, _TraceBackedModel]:
    read_results = [
        _packet_to_decode_result(packet) for packet in reference["host_trace"]["target_verification_packets"]
    ]
    fake_model = _TraceBackedModel(read_results)
    return _DFlashTraceBackedPipeline(reference, fake_model), fake_model


def _run_golden_decode_manager_logic(reference: dict) -> dict[str, Any]:
    pipeline, fake_model = _make_trace_backed_pipeline(reference)
    emitted_tokens: list[int] = []
    params = reference["host_trace"].get("params", {})

    generated_tokens = pipeline.run_inference(
        reference["prompt"]["input_ids"][0].tolist(),
        int(reference["metadata"]["max_new_tokens"]),
        on_token=emitted_tokens.append,
        eos_token_id=params.get("eos_token_id"),
        return_generated_tokens=True,
    )

    fake_model.assert_fully_consumed()
    return {
        "generated_tokens": generated_tokens,
        "emitted_tokens": emitted_tokens,
        "output_token_ids": pipeline.output_token_ids,
        "acceptance_lengths": pipeline.acceptance_lengths,
        "writes": fake_model.writes,
        "read_count": fake_model.read_count,
        "stats": pipeline.last_inference_stats,
    }


def _run_device_decode_manager_logic(reference: dict) -> dict[str, Any]:
    del reference
    require_golden_or_not_implemented(False, "ModelPipeline.run_inference with DFlash block proposals")
    raise AssertionError("unreachable")


def _run_decode_manager_logic(reference: dict, *, use_golden: bool) -> dict[str, Any]:
    if use_golden:
        return _run_golden_decode_manager_logic(reference)
    return _run_device_decode_manager_logic(reference)


@pytest.mark.parametrize("use_golden", USE_GOLDEN_PARAMS)
def test_dflash_spec_decode_manager_logic_matches_golden_reference(use_golden: bool) -> None:
    # Host-loop contract: write draft blocks, consume verifier packets, and emit accepted tokens.
    reference = load_validated_dflash_reference()
    expected = expected_decode_manager_outputs(reference)

    actual = _run_decode_manager_logic(reference, use_golden=use_golden)

    assert_decode_manager_outputs_match(actual, expected)
