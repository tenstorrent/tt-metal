# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "dflash_block_diffusion"
MANIFEST_PATH = FIXTURE_DIR / "manifest.json"
TRACE_PATH = FIXTURE_DIR / "lossless_block_trace.json"


@dataclass(frozen=True)
class ReplayResult:
    output_token_ids: list[int]
    acceptance_lengths: list[int]
    host_writes: list[dict[str, int | str]]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _leading_true_count(values: list[bool]) -> int:
    count = 0
    for value in values:
        if not value:
            break
        count += 1
    return count


def _host_writes_for_block(block_token_ids: list[int], *, anchor_pos: int) -> list[dict[str, int | str]]:
    return [
        {
            "token_id": int(token_id),
            "token_type": "BASE" if offset == 0 else "SPEC",
            "position_id": int(anchor_pos + offset),
            "user_id": 0,
            "prefill_token_id": -1,
        }
        for offset, token_id in enumerate(block_token_ids)
    ]


def _replay_dflash_lossless_trace(trace: dict) -> ReplayResult:
    params = trace["params"]
    prompt_token_ids = [int(token_id) for token_id in params["prompt_token_ids"]]
    block_size = int(params["block_size"])
    max_new_tokens = int(params["max_new_tokens"])
    max_length = len(prompt_token_ids) + max_new_tokens

    output_by_pos = {pos: token_id for pos, token_id in enumerate(prompt_token_ids)}
    start = len(prompt_token_ids)
    output_by_pos[start] = int(trace["prefill"]["target_next_token_id"])

    acceptance_lengths: list[int] = []
    host_writes: list[dict[str, int | str]] = []

    for round_idx, round_record in enumerate(trace["rounds"]):
        assert int(round_record["anchor_pos"]) == start, f"round {round_idx} starts at unexpected position"

        block_token_ids = [int(token_id) for token_id in round_record["block_input_token_ids"]]
        target_posterior_token_ids = [int(token_id) for token_id in round_record["target_posterior_token_ids"]]
        assert len(block_token_ids) == block_size
        assert len(target_posterior_token_ids) == block_size
        assert output_by_pos[start] == block_token_ids[0]

        host_writes.extend(_host_writes_for_block(block_token_ids, anchor_pos=start))

        matches = [
            draft_token_id == target_token_id
            for draft_token_id, target_token_id in zip(block_token_ids[1:], target_posterior_token_ids[:-1])
        ]
        accepted_after_anchor = _leading_true_count(matches)
        committed = accepted_after_anchor + 1
        assert committed == int(round_record["expected_committed_tokens"])
        acceptance_lengths.append(committed)

        for offset, token_id in enumerate(block_token_ids[:committed]):
            output_by_pos[start + offset] = token_id

        next_anchor_pos = start + committed
        if next_anchor_pos < max_length:
            next_anchor_token_id = target_posterior_token_ids[accepted_after_anchor]
            assert next_anchor_token_id == int(round_record["expected_next_anchor_token_id"])
            output_by_pos[next_anchor_pos] = next_anchor_token_id

        start += committed
        if start >= max_length:
            break

    return ReplayResult(
        output_token_ids=[output_by_pos[pos] for pos in range(max_length)],
        acceptance_lengths=acceptance_lengths,
        host_writes=host_writes,
    )


def test_dflash_fixture_manifest_matches_golden_reference() -> None:
    manifest = _load_json(MANIFEST_PATH)

    assert manifest["approach"] == "dflash_block_diffusion"
    assert manifest["target_model"] == "moonshotai/Kimi-K2.5"
    assert manifest["draft_model"] == "z-lab/Kimi-K2.5-DFlash"
    assert manifest["golden_reference"]["manager"] == "golden_dflash.spec_decode.SpecDecodeManager"
    assert manifest["golden_reference"]["draft_runtime"] == "golden_dflash.spec_decode.TorchDFlashRuntime"
    assert manifest["acceptance_metric"] == "committed_tokens_per_verify_pass_including_anchor"
    assert {trace["path"] for trace in manifest["trace_files"]} == {TRACE_PATH.name}


def test_dflash_lossless_block_replay_matches_golden_fixture() -> None:
    trace = _load_json(TRACE_PATH)
    result = _replay_dflash_lossless_trace(trace)

    assert result.output_token_ids == trace["expected"]["output_token_ids"]
    assert result.acceptance_lengths == trace["expected"]["acceptance_lengths"]
    assert math.isclose(
        sum(result.acceptance_lengths) / len(result.acceptance_lengths),
        trace["expected"]["average_committed_tokens"],
    )
    assert sum(length - 1 for length in result.acceptance_lengths) == trace["expected"]["accepted_draft_tokens"]


def test_dflash_deepseek_b1_write_contract_preserves_block_positions() -> None:
    trace = _load_json(TRACE_PATH)
    result = _replay_dflash_lossless_trace(trace)

    assert result.host_writes == trace["expected"]["host_writes"]

    block_size = trace["params"]["block_size"]
    for block_start in range(0, len(result.host_writes), block_size):
        block_writes = result.host_writes[block_start : block_start + block_size]
        assert [write["token_type"] for write in block_writes] == ["BASE", "SPEC", "SPEC", "SPEC"]
        assert [write["position_id"] for write in block_writes] == list(
            range(block_writes[0]["position_id"], block_writes[0]["position_id"] + block_size)
        )
