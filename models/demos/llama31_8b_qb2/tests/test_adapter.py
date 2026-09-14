# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host checks for scheduler placement without constructing a device model."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from models.demos.llama31_8b_qb2.tt.generator_vllm import LlamaForCausalLM


def test_prefill_maps_scheduler_slots_to_int32_device_inputs():
    generator = SimpleNamespace(
        mesh_device=None,
        max_batch_size=32,
        refresh_decode_inputs=Mock(),
        prefill_forward=Mock(),
    )
    adapter = LlamaForCausalLM(generator)
    adapter.prefill_forward(
        torch.tensor([[2, 3, 4], [5, 6, 0]]),
        torch.tensor([[7], [8]], dtype=torch.int32),
        kv_cache=(),
        prompt_lens=[3, 2],
        empty_slots=[5, 9],
    )
    _, positions = generator.refresh_decode_inputs.call_args.args
    table = generator.refresh_decode_inputs.call_args.kwargs["page_table"]
    assert positions.dtype == torch.int32
    assert positions[5].item() == 3 and positions[9].item() == 2
    assert (positions >= 0).sum().item() == 2
    assert table[5, 0].item() == 7 and table[9, 0].item() == 8
    assert generator.prefill_forward.call_args.kwargs["slots"] == [5, 9]


@pytest.mark.parametrize("reset_batch", [False, True])
def test_decode_rejects_legacy_input_contract_before_changing_state(reset_batch, expect_error):
    generator = Mock(mesh_device=None, max_batch_size=32)
    adapter = LlamaForCausalLM(generator)
    with expect_error(TypeError, "requires reload_inputs"):
        adapter.decode_forward(None, None, None, None, reset_batch=reset_batch)
    assert not generator.mock_calls
