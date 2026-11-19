# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only adapter boundaries; these do not establish serving correctness."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from models.autoports.qwen_qwen3_8_flash_next.tt.generator_vllm import QwenFlashNextForCausalLM
from models.autoports.qwen_qwen3_8_flash_next.tt.model import canonical_text_config


def stub_generator(batch):
    gen = Mock(max_batch_size=batch)
    gen.caches = [SimpleNamespace(position=0) for _ in range(batch)]
    gen.counters = {}
    return gen


def test_config_normalization_matches_standalone_consumed_fields():
    evidence = Path(__file__).resolve().parents[1] / "doc/vllm_integration"
    serving = json.loads((evidence / "serving_config.json").read_text())["text_config"]
    standalone = json.loads((evidence / "standalone_config.json").read_text())["text_config"]
    original = SimpleNamespace(**serving)
    normalized = canonical_text_config(original)
    attributes = json.loads((evidence / "config_attribute_audit.json").read_text())["source_attributes"]
    for name in attributes:
        if name in standalone or name in serving:
            assert getattr(normalized, name, None) == standalone.get(name), name
    assert original.layer_types.count("full_attention") == 12
    assert normalized.layer_types.count("indexed_attention") == 12
    assert normalized.rope_parameters == original.rope_parameters
    assert normalized.rope_parameters is not original.rope_parameters
    assert canonical_text_config(normalized).__dict__ == normalized.__dict__


def test_dense_attention_is_not_silently_reclassified(expect_error):
    with expect_error(ValueError, "complete QSA"):
        canonical_text_config(SimpleNamespace(layer_types=["full_attention"]))


def test_decode_preserves_reload_commands_and_cache_identity():
    gen = stub_generator(4)
    adapter = QwenFlashNextForCausalLM(gen)
    cache, sampling = object(), object()
    tokens = torch.tensor([[10], [20]])
    positions = torch.tensor([32, 64])
    table = torch.tensor([[2, 3, 4], [5, 6, 7]])
    remap = (1, 0, 3, 2)
    kwargs = dict(
        kv_cache=cache,
        page_table=table,
        sampling_params=sampling,
        reload_inputs=False,
        reload_page_table=True,
        reload_sampling_params=False,
        reset_sampling_state=False,
        slot_remap=remap,
        read_from_device=False,
    )
    output = adapter.decode_forward(tokens, positions, **kwargs)
    gen.decode_forward.assert_called_once_with(tokens, positions, sample_on_device=True, **kwargs)
    assert output is gen.decode_forward.return_value
    assert adapter.tt_supported_decode_batch_sizes == (1, 2, 4)


def test_explicit_host_output_does_not_select_device_sampling():
    gen = stub_generator(2)
    adapter = QwenFlashNextForCausalLM(gen)
    adapter.decode_forward(tokens=torch.tensor([[42]]), start_pos=torch.tensor([17]), kv_cache=object())
    assert gen.decode_forward.call_args.kwargs["sample_on_device"] is False


def test_text_prefill_returns_zero_mrope_delta_and_preserves_slots():
    gen = stub_generator(4)
    gen.prefill_forward.return_value = torch.tensor([[7], [9]])
    adapter = QwenFlashNextForCausalLM(gen)
    kwargs = dict(
        tokens=torch.zeros(2, 131, dtype=torch.int64), prompt_lens=(131, 129), empty_slots=(3, 1), start_pos=(64, 0)
    )
    output, delta = adapter.prefill_forward(**kwargs)
    gen.prefill_forward.assert_called_once_with(**kwargs)
    assert output is gen.prefill_forward.return_value
    assert delta.tolist() == [0, 0]


def test_all_intermediate_prefill_preserves_none_output():
    gen = stub_generator(2)
    gen.prefill_forward.return_value = None
    adapter = QwenFlashNextForCausalLM(gen)
    kwargs = dict(
        tokens=torch.zeros(2, 64, dtype=torch.int64), prompt_lens=(64, 64), prefill_sample_mask=[False, False]
    )
    output, delta = adapter.prefill_forward(**kwargs)
    assert output is None
    assert delta.tolist() == [0, 0]
    gen.prefill_forward.assert_called_once_with(**kwargs)


@pytest.mark.parametrize("kind", ["full_attention", "indexed_attention"])
def test_reduced_allocator_preserves_selected_scheduler_spec(kind):
    gen = Mock(max_batch_size=2)
    gen.model = SimpleNamespace(
        config=SimpleNamespace(layer_types=["linear_attention"] * 3 + [kind]),
        layer_indices=(0, 1, 3),
    )
    adapter = QwenFlashNextForCausalLM(gen)
    spec = ((8195, 1, 32, 256), torch.bfloat16, 0)
    pool = adapter.allocate_kv_cache_per_layer([spec])
    gen.allocate_kv_cache_per_layer.assert_called_once_with([spec])
    assert pool is gen.allocate_kv_cache_per_layer.return_value
