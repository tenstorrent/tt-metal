# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
from pathlib import Path

from models.demos.blaze_spec_decode.dflash_block_diffusion_moonshotai_kimi_k2_5.tests.unit_tests.dflash_golden_ops import (
    golden_replay_dflash_lossless_trace,
)


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "dflash_block_diffusion"
MANIFEST_PATH = FIXTURE_DIR / "manifest.json"
TRACE_PATH = FIXTURE_DIR / "lossless_block_trace.json"
RUN_INFERENCE_TRACE_PATH = FIXTURE_DIR / "run_inference_trace.json"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_dflash_fixture_manifest_matches_golden_reference() -> None:
    manifest = _load_json(MANIFEST_PATH)

    assert manifest["approach"] == "dflash_block_diffusion"
    assert manifest["target_model"] == "moonshotai/Kimi-K2.5"
    assert manifest["draft_model"] == "z-lab/Kimi-K2.5-DFlash"
    assert manifest["golden_reference"]["manager"] == "golden_dflash.spec_decode.SpecDecodeManager"
    assert manifest["golden_reference"]["draft_runtime"] == "golden_dflash.spec_decode.TorchDFlashRuntime"
    assert manifest["acceptance_metric"] == "committed_tokens_per_verify_pass_including_anchor"
    assert {trace["path"] for trace in manifest["trace_files"]} == {TRACE_PATH.name, RUN_INFERENCE_TRACE_PATH.name}


def test_dflash_lossless_block_replay_matches_golden_fixture() -> None:
    trace = _load_json(TRACE_PATH)
    result = golden_replay_dflash_lossless_trace(trace)

    assert result.output_token_ids == trace["expected"]["output_token_ids"]
    assert result.acceptance_lengths == trace["expected"]["acceptance_lengths"]
    assert math.isclose(
        sum(result.acceptance_lengths) / len(result.acceptance_lengths),
        trace["expected"]["average_committed_tokens"],
    )
    assert sum(length - 1 for length in result.acceptance_lengths) == trace["expected"]["accepted_draft_tokens"]


def test_dflash_blaze_spec_decode_write_contract_preserves_block_positions() -> None:
    trace = _load_json(TRACE_PATH)
    result = golden_replay_dflash_lossless_trace(trace)

    assert result.host_writes == trace["expected"]["host_writes"]

    block_size = trace["params"]["block_size"]
    for block_start in range(0, len(result.host_writes), block_size):
        block_writes = result.host_writes[block_start : block_start + block_size]
        assert [write["token_type"] for write in block_writes] == ["BASE", "SPEC", "SPEC", "SPEC"]
        assert [write["position_id"] for write in block_writes] == list(
            range(block_writes[0]["position_id"], block_writes[0]["position_id"] + block_size)
        )
