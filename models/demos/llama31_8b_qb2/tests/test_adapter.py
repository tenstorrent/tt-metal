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
        adapter.decode_forward(
            None,
            None,
            None,
            None,
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=False,
            reset_sampling_state=False,
            reset_batch=reset_batch,
        )
    assert not generator.mock_calls


def test_decode_forwards_each_explicit_update_command_without_deriving_reload():
    output = object()
    generator = Mock(mesh_device=None, max_batch_size=32)
    generator.decode_forward.return_value = output
    adapter = LlamaForCausalLM(generator)

    assert (
        adapter.decode_forward(
            "tokens",
            "positions",
            "page-table",
            "cache",
            read_from_device=False,
            sampling_params=object(),
            reload_inputs=False,
            reload_page_table=True,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )
        is output
    )

    assert generator.decode_forward.call_args.kwargs["reload_inputs"] is False
    assert generator.decode_forward.call_args.kwargs["reload_page_table"] is True
    assert "reset_batch" not in generator.decode_forward.call_args.kwargs


def test_sampling_state_reset_does_not_reload_sampling_parameters():
    generator = Mock(mesh_device=None, max_batch_size=32)
    adapter = LlamaForCausalLM(generator)
    params = SimpleNamespace(top_k=[1], top_p=[1.0], temperature=[1.0], seed=[7])

    adapter.decode_forward(
        torch.tensor([[3]]),
        torch.tensor([11]),
        "page-table",
        "cache",
        read_from_device=False,
        sampling_params=params,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=True,
    )

    generator.reset_sampling_seed.assert_called_once()
    generator.set_sampling.assert_not_called()
