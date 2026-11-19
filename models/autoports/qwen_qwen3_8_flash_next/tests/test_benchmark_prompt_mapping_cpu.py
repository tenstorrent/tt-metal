# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact client/runtime request binding without tokenizer or device imports."""

import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("prompt_mapping_roofline", TOOLS / "benchmark_roofline.py")
roofline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(roofline)


def fixture(tmp_path, prompts):
    rows = [
        dict(
            request_id=f"client-{i}",
            prompt_token_ids=tokens,
            input_tokens=len(tokens),
            output_tokens=128,
            prompt_token_ids_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        )
        for i, tokens in enumerate(prompts)
    ]
    artifact = tmp_path / "inputs.json"
    artifact.write_text(json.dumps({"requests": rows}))
    return artifact


def row(request, tokens, width, end):
    return dict(
        request=request,
        start=0,
        end=end,
        logical_prompt_tokens=len(tokens),
        prompt_sha256=hashlib.sha256(struct.pack(f"={width}i", *tokens[:width])).hexdigest(),
    )


def test_initial_hash_uses_submission_width_not_shorter_chunk_end(tmp_path):
    prompts = [[11, 12, 13, 14, 15], [21, 22, 23, 24, 25]]
    artifact = fixture(tmp_path, prompts)
    # Scheduler order differs from client order. Both buffers have width four,
    # despite request 9 consuming only the first two tokens on this submission.
    requests = [row(9, prompts[1], 4, 2), row(7, prompts[0], 4, 4)]
    events = [dict(phase="prefill", metadata={"requests": requests})]
    result = roofline.verify_client_prompts(events, requests, artifact)
    assert result["matched_requests"] == 2
    assert [r["client_request_id"] for r in requests] == ["client-1", "client-0"]
    assert [r["initial_buffer_tokens"] for r in requests] == [4, 4]


def test_wrong_runtime_tokens_fail_mapping(tmp_path):
    artifact = fixture(tmp_path, [[1, 2, 3]])
    requests = [row(1, [1, 99, 3], 3, 3)]
    events = [dict(phase="prefill", metadata={"requests": requests})]
    with pytest.raises(ValueError, match="unique unmatched"):  # allow-pytest.raises: host evidence validation
        roofline.verify_client_prompts(events, requests, artifact)


def test_ambiguous_initial_prefix_is_rejected(tmp_path):
    prompts = [[1, 2, 3], [1, 2, 4]]
    artifact = fixture(tmp_path, prompts)
    requests = [row(1, prompts[0], 2, 2), row(2, prompts[1], 2, 2)]
    events = [dict(phase="prefill", metadata={"requests": requests})]
    with pytest.raises(ValueError, match="unique unmatched"):  # allow-pytest.raises: host evidence validation
        roofline.verify_client_prompts(events, requests, artifact)


def test_unrecorded_buffer_padding_is_never_guessed(tmp_path):
    artifact = fixture(tmp_path, [[1, 2, 3]])
    requests = [dict(request=1, start=0, end=4, logical_prompt_tokens=3, prompt_sha256="unknown")]
    events = [dict(phase="prefill", metadata={"requests": requests})]
    with pytest.raises(ValueError, match="padding"):  # allow-pytest.raises: host evidence validation
        roofline.verify_client_prompts(events, requests, artifact)
