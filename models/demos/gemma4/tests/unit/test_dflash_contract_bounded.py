# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host coverage for bounded target-cache admission in the dFlash contract.

Packed verification writes several target positions before attention reads the
live window. The production adapter must decline a proposal before those writes
can replace required history and must continue ordinary completion handling.
Lower target operations use recording stubs; device KV correctness is unverified.
"""

import pytest

from models.demos.gemma4.tests.unit.test_dflash_contract_adapter import (
    DeviceResult,
    _complete,
    _ordinary,
    _prefill,
    _tensor,
    _verify,
)
from models.demos.gemma4.tests.unit.test_dflash_contract_adapter import adapter as _adapter_fixture
from models.demos.gemma4.tests.unit.test_dflash_contract_adapter import decoder_width_for as _decoder_width_for_fixture
from models.demos.gemma4.tests.unit.test_dflash_contract_adapter import model as _model_fixture

adapter = _adapter_fixture
decoder_width_for = _decoder_width_for_fixture
model = _model_fixture


def _bounded_layer(model, index=0, ring=64, window=64):
    attention = model.model[0].layers[index].self_attn.config
    attention.cache_position_modulo = ring
    attention.sliding_window = window


@pytest.mark.parametrize("fresh_decoder", [False, True])
@pytest.mark.parametrize(
    "position, allowed", [(57, True), (58, True), (59, False), (63, False), (64, False), (128, False)]
)
def test_exact_ring_declines_before_first_wrapped_candidate(model, fresh_decoder, position, allowed):
    _bounded_layer(model)
    if fresh_decoder:
        model._spec_decoder = None
    assert model._contract_covers_position(object(), position) is allowed
    assert model.events == []


@pytest.mark.parametrize("fresh_decoder", [False, True])
@pytest.mark.parametrize("window, allowed", [(60, False), (59, True)])
def test_ring_requires_physical_width_minus_one_headroom(model, fresh_decoder, window, allowed):
    _bounded_layer(model, window=window)
    if fresh_decoder:
        model._spec_decoder = None
    assert model._contract_covers_position(object(), 80) is allowed
    assert model.events == []


@pytest.mark.parametrize("physical_width, allowed", [(6, True), (8, False)])
def test_ring_checks_actual_decoder_width(model, physical_width, allowed):
    _bounded_layer(model)
    model._spec_decoder.P_v = physical_width
    assert model._SPEC_N == 6
    assert model._contract_covers_position(object(), 57) is allowed


@pytest.mark.parametrize("unsafe_layer", [0, 1])
def test_every_bounded_layer_must_cover_the_proposal(model, unsafe_layer):
    _bounded_layer(model, index=1 - unsafe_layer, ring=128, window=64)
    _bounded_layer(model, index=unsafe_layer)
    assert model._contract_covers_position(object(), 59) is False


@pytest.mark.parametrize("fresh_decoder", [False, True])
def test_unbounded_layers_preserve_existing_coverage(model, fresh_decoder):
    if fresh_decoder:
        model._spec_decoder = None
    assert model._contract_covers_position(object(), 80) is True


@pytest.mark.parametrize("fresh_decoder", [False, True])
def test_unsafe_ring_declines_ordinary_completion_before_reconstruction(model, fresh_decoder):
    _bounded_layer(model)
    _prefill(model, prompts=[list(range(58))])
    _ordinary(model, [41], [58], [10], DeviceResult(_tensor([[42]])))
    if fresh_decoder:
        model._spec_decoder = None
    model.events.clear()
    proposal = _complete(model, [[42]], [[59]])
    assert proposal.num_valid.tolist() == [0]
    assert model._ct_requests[10].tokens[-2:] == [41, 42]
    assert model._ct_proposal is None
    assert model.events == []


def test_verified_completion_crosses_ring_limit_and_continues_ordinary_decode(model):
    _bounded_layer(model)
    _prefill(model, prompts=[list(range(57))])
    _ordinary(model, [41], [57], [10], DeviceResult(_tensor([[42]])))
    proposal = _complete(model, [[42]], [[58]])
    assert proposal.num_valid.tolist() == [5]
    owner = model._ct_requests[10]
    verified = _verify(model, blocks=[[42, 21, 22, 23, 24, 25]], positions=[list(range(58, 64))])
    model.events.clear()
    declined = _complete(model, [[21, 22, 99]], [[59, 60, 61]], hidden=verified.hidden)
    assert declined.num_valid.tolist() == [0]
    assert owner.tokens[-4:] == [42, 21, 22, 99]
    assert model._ct_proposal is None
    assert model.events == []

    ordinary = _ordinary(model, [99], [61], [10], DeviceResult(_tensor([[43]])))
    readback = model.read_decode_output(ordinary)
    assert model.process_decode_output_host(readback, is_tokens=True).tolist() == [[43]]
    assert any(event[0] == "decode" for event in model.events)
    model.events.clear()
    declined = _complete(model, [[43]], [[62]])
    assert declined.num_valid.tolist() == [0]
    assert owner.tokens[-5:] == [42, 21, 22, 99, 43]
    assert model.events == []
