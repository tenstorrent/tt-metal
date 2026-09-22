# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host tests for Gemma4 input authority after decode slot permutations.

Native prefill marks old device slots, while CT reloads identify current
decode rows. Production dispatch must keep those coordinates separate when
it merges device feedback. TT execution is stubbed below the dispatcher.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.demos.gemma4.tests.unit.test_dflash_contract_adapter import _prefill, _tensor
from models.demos.gemma4.tests.unit.test_dflash_contract_adapter import adapter as _adapter_fixture
from models.demos.gemma4.tests.unit.test_dflash_contract_adapter import decoder_width_for as _decoder_width_for_fixture
from models.demos.gemma4.tests.unit.test_dflash_contract_adapter import model as _model_fixture
from models.demos.gemma4.tt.async_decode import merge_async_ahead_decode_tokens

adapter = _adapter_fixture
decoder_width_for = _decoder_width_for_fixture
model = _model_fixture


@pytest.fixture
def native_decode():
    source = Path(__file__).parents[2] / "tt" / "generator.py"
    module = ast.parse(source.read_text())
    mixin = next(
        item
        for item in module.body
        if isinstance(item, ast.ClassDef) and item.name == "ChunkedPrefillPageTableGuardMixin"
    )
    dispatch = next(item for item in mixin.body if isinstance(item, ast.FunctionDef) and item.name == "decode_forward")
    namespace = {
        "torch": torch,
        "Mode": SimpleNamespace(DECODE="DECODE"),
        "SamplingParams": object,
        "merge_async_ahead_decode_tokens": merge_async_ahead_decode_tokens,
        "ttnn": SimpleNamespace(get_device_tensors=lambda tensor: [tensor], to_torch=lambda tensor: tensor),
        "logger": SimpleNamespace(info=lambda *args: None),
    }
    exec(compile(ast.Module(body=[dispatch], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["decode_forward"]


def _generator(batch, device_tokens, device_positions, prefilled=(), data_parallel=1):
    chunks = list(
        zip(torch.chunk(_tensor(device_tokens), data_parallel), torch.chunk(_tensor(device_positions), data_parallel))
    )
    captured = []

    def replay(**kwargs):
        captured.append(kwargs)
        return object()

    instance = SimpleNamespace(
        mode="DECODE",
        data_parallel=data_parallel,
        model=[SimpleNamespace(switch_mode=lambda mode: None) for _ in range(data_parallel)],
        model_capabilities={"supports_async_decode": True},
        trace_inputs_decode={(True, batch // data_parallel): chunks},
        _slots_prefilled_since_decode=set(prefilled),
        _clear_sequential_batch_page_tables=lambda: None,
        _decode_forward_trace_text=replay,
        sample_decode_on_device=lambda output, **kwargs: output,
    )
    return instance, captured


def _submit(native_decode, instance, captured, tokens, positions, **kwargs):
    native_decode(
        instance,
        tokens=_tensor(tokens).reshape(-1, 1),
        start_pos=_tensor(positions),
        sampling_params=object(),
        reset_batch=True,
        enable_trace=True,
        read_from_device=False,
        **kwargs,
    )
    call = captured[-1]
    return torch.cat(call["tokens"]).reshape(-1).tolist(), torch.cat(call["current_pos"]).reshape(-1).tolist()


@pytest.mark.parametrize("device_ahead", [0, 1])
def test_prefilled_old_slot_follows_gather_without_suppressing_survivor(native_decode, device_ahead):
    instance, captured = _generator(2, [11, 99], [5 + device_ahead, 8 + device_ahead], prefilled={1})
    result = _submit(native_decode, instance, captured, [20, 10], [8, 5], slot_remap=_tensor([1, 0]))
    assert result == ([20, 11], [8, 5 + device_ahead])
    assert instance._slots_prefilled_since_decode == set()


def test_identity_layout_preserves_native_prefill_authority(native_decode):
    instance, captured = _generator(2, [11, 99], [6, 9], prefilled={1})
    assert _submit(native_decode, instance, captured, [10, 20], [5, 8]) == ([11, 20], [6, 8])


def test_native_prefill_authority_follows_slot_outside_smaller_decode_bucket(native_decode):
    instance, captured = _generator(1, [70, 80, 90, 99], [1, 2, 3, 9], prefilled={3})
    result = _submit(native_decode, instance, captured, [20], [8], slot_remap=_tensor([3, 0, 1, 2]))
    assert result == ([20], [8])


def test_current_host_rows_and_old_prefill_slots_are_independent_across_dp(native_decode):
    instance, captured = _generator(4, [11, 21, 31, 41], [6, 9, 12, 15], prefilled={3}, data_parallel=2)
    result = _submit(
        native_decode,
        instance,
        captured,
        [20, 10, 40, 30],
        [8, 5, 14, 11],
        slot_remap=_tensor([1, 0, 3, 2]),
        force_host_decode_rows={1},
    )
    assert result == ([21, 10, 40, 31], [9, 5, 14, 12])


def test_host_override_does_not_reactivate_inactive_rows(native_decode):
    instance, captured = _generator(2, [70, 11], [0, 6], prefilled={0})
    result = _submit(
        native_decode, instance, captured, [10, 0], [5, -1], slot_remap=_tensor([1, 0]), force_host_decode_rows={0}
    )
    assert result == ([10, 0], [5, -1])


def test_ct_reload_uses_current_row_after_old_slot_three_moves_to_zero(model, native_decode, monkeypatch):
    _prefill(model, keys=(10,), slots=[3])
    model._slots_prefilled_since_decode.clear()
    model._ct_force_reload = True
    native, captured = _generator(1, [70, 80, 90, 44], [0, 0, 0, 3])
    forwarded = []

    def decode(*args, **kwargs):
        forwarded.append(kwargs.copy())
        return native_decode(native, *args, **kwargs)

    monkeypatch.setattr(model, "_contract_target_decode", decode)
    model.decode_forward(
        tokens=_tensor([[3]]),
        start_pos=_tensor([2]),
        page_table=_tensor([[10, 11]]),
        kv_cache=model.kv_cache,
        sampling_params=object(),
        reset_batch=False,
        read_from_device=False,
        slot_remap=_tensor([3, 0, 1, 2]),
    )
    assert forwarded[-1]["force_host_decode_rows"] == {0}
    assert forwarded[-1]["reset_batch"] is True
    assert captured[-1]["tokens"][0].reshape(-1).tolist() == [3]
    assert captured[-1]["current_pos"][0].tolist() == [2]
    assert model._ct_requests[10].slot == 0
    assert model._slots_prefilled_since_decode == set()
    assert model._ct_force_reload is False
