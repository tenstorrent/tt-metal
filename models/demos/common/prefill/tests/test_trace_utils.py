# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file


# Equal or longer saved prompts must skip the expensive reference forward; short/missing ones must generate.
@pytest.mark.parametrize("saved_length", [None, 4, 8, 16])
def test_ensure_trace_generates_only_for_missing_or_short_reference(tmp_path, saved_length):
    from models.demos.common.prefill.runners.trace_utils import ensure_trace

    saved, generated = tmp_path / "saved", tmp_path / "generated"
    if saved_length is not None:
        saved.mkdir()
        (saved / "metadata.json").write_text(json.dumps({"token_ids": list(range(saved_length))}))

    def generate():
        generated.mkdir()
        (generated / "metadata.json").write_text(json.dumps({"token_ids": list(range(8))}))
        return generated

    result = ensure_trace(saved, 8, generate)
    needs_generation = saved_length is None or saved_length < 8
    assert result == (generated if needs_generation else saved)
    assert generated.exists() == needs_generation


# Metadata alone is insufficient: a missing layer must trigger generation before a device run starts.
def test_ensure_trace_regenerates_incomplete_gqa_reference(tmp_path):
    from models.demos.common.prefill.runners.trace_utils import ensure_trace, validate_gqa_trace

    (tmp_path / "metadata.json").write_text(json.dumps({"token_ids": list(range(8))}))

    def validate(path, length):
        validate_gqa_trace(path, length, num_layers=2, num_kv_heads=1, head_dim=4)

    def generate():
        cache = tmp_path / "kv_cache"
        cache.mkdir()
        for layer in range(2):
            save_file(
                {f"{kind}_cache_layer_{layer}": torch.zeros(1, 1, 8, 4) for kind in ("key", "value")},
                cache / f"layer_{layer}.safetensors",
            )
        return tmp_path

    assert ensure_trace(tmp_path, 8, generate, validate=validate) == tmp_path
    assert (tmp_path / "kv_cache/layer_1.safetensors").is_file()


# The acceptance scenario must keep two allocated slots while allowing one or two producer slots.
@pytest.mark.parametrize("active_slots", [1, 2])
def test_llama_scenario_separates_producer_slots_from_allocated_slots(monkeypatch, tmp_path, active_slots):
    from models.demos.llama_3p1_8b_d_p.tests.utils import prefill_runner_scenario, validate_prefill_slot_traces

    monkeypatch.setenv("PREFILL_PRODUCER_NUM_USERS", str(active_slots))
    monkeypatch.setenv("PREFILL_MAX_SEQ_LEN", "2048")
    monkeypatch.delenv("PREFILL_NUM_USERS", raising=False)
    scenario = prefill_runner_scenario()
    assert scenario["users"] == 2
    assert scenario["expected_slots"] == active_slots
    assert scenario["producer"]["PREFILL_NUM_USERS"] == str(active_slots)
    assert scenario["producer"]["PREFILL_PRODUCER_MAX_REQUESTS"] == str(active_slots)
    (tmp_path / "metadata.json").write_text(json.dumps({"token_ids": list(range(4096))}))
    cache = tmp_path / "kv_cache"
    cache.mkdir()
    for layer in range(32):
        save_file(
            {
                f"{kind}_cache_layer_{layer}": torch.zeros(1, 8, 4096, 128, dtype=torch.bfloat16)
                for kind in ("key", "value")
            },
            cache / f"layer_{layer}.safetensors",
        )
    validate_prefill_slot_traces(str(tmp_path), scenario)
    # Invalid checkpoint/prompt paths prove the reuse path never attempts an HF forward.
    script = Path(__file__).parents[3] / "llama_3p1_8b_d_p/scripts/generate_prefill_trace.py"
    output = tmp_path / "selected"
    command = [
        sys.executable,
        str(script),
        "--checkpoint",
        str(tmp_path / "no-checkpoint"),
        "--prompt-file",
        str(tmp_path / "no-prompt"),
        "--output-dir",
        str(output),
        "--seq-len",
        "2048",
        "--num-slots",
        str(active_slots),
        "--reuse-trace-dirs",
        str(tmp_path),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(script.parents[4]))
    )
    assert result.returncode == 0, result.stderr
    assert json.loads((output / "trace_paths.json").read_text()) == [str(tmp_path)] * active_slots
