# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import json

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


# The acceptance scenario allocates the manifest's two slots unless PREFILL_NUM_USERS raises it, and the
# producer may fill any number of the allocated slots.
@pytest.mark.parametrize("allocated_slots, active_slots", [(None, 1), (None, 2), (4, 4), (16, 3)])
def test_llama_scenario_separates_producer_slots_from_allocated_slots(
    monkeypatch, tmp_path, allocated_slots, active_slots
):
    from models.demos.llama_3p1_8b_d_p.tests.utils import prefill_runner_scenario, validate_prefill_slot_traces

    monkeypatch.setenv("PREFILL_PRODUCER_NUM_USERS", str(active_slots))
    monkeypatch.setenv("PREFILL_MAX_SEQ_LEN", "2048")
    if allocated_slots is None:
        monkeypatch.delenv("PREFILL_NUM_USERS", raising=False)
    else:
        monkeypatch.setenv("PREFILL_NUM_USERS", str(allocated_slots))
    scenario = prefill_runner_scenario()
    assert scenario["users"] == (allocated_slots or 2)
    assert scenario["env"]["PREFILL_NUM_USERS"] == str(allocated_slots or 2)
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


@pytest.mark.parametrize(
    "allocated_slots, active_slots, message",
    [("2", "3", "exceeds PREFILL_NUM_USERS=2"), ("0", "1", "PREFILL_NUM_USERS"), ("four", "1", "PREFILL_NUM_USERS")],
)
def test_llama_scenario_rejects_unusable_slot_counts(monkeypatch, allocated_slots, active_slots, message):
    from models.demos.llama_3p1_8b_d_p.tests.utils import prefill_runner_scenario

    monkeypatch.setenv("PREFILL_NUM_USERS", allocated_slots)
    monkeypatch.setenv("PREFILL_PRODUCER_NUM_USERS", active_slots)
    monkeypatch.setenv("PREFILL_MAX_SEQ_LEN", "2048")
    with pytest.raises(ValueError, match=message):
        prefill_runner_scenario()
