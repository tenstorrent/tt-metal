# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import MethodType

import pytest
import torch

from models.demos.gemma4.tt.model import Gemma4Model


def _host_pli_model():
    model = Gemma4Model.__new__(Gemma4Model)
    model.hidden_size_per_layer_input = 2
    model.per_layer_input_weights = {"present": True}
    model._embed_weight_cpu = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    model.embed_scale = 0.5
    model.layers = [object(), object(), object()]
    calls = []

    def compute(self, token_ids, embeds):
        calls.append(tuple(token_ids.shape))
        # Include both inputs so duplicate and permuted tokens exercise the
        # candidate mapping, then round at the production boundary.
        base = token_ids.to(torch.float32).unsqueeze(-1) + embeds[..., :2] / 128
        return [(base + layer / 16).to(torch.bfloat16) for layer in range(len(self.layers))]

    model._compute_per_layer_inputs = MethodType(compute, model)
    return model, calls


@pytest.mark.parametrize("token_ids", [[1], [1, 3, 1, 7], [7, 1, 3, 1]])
def test_host_pli_batch_rows_equal_single_token_reference(token_ids):
    model, calls = _host_pli_model()
    actual = model.compute_host_pli_batch(token_ids)
    assert actual.shape == (len(model.layers), 1, len(token_ids), model.hidden_size_per_layer_input)

    for row, token_id in enumerate(token_ids):
        single = model.compute_host_pli(token_id)
        for layer in range(len(model.layers)):
            torch.testing.assert_close(actual[layer, :, row : row + 1], single[:, :, layer], rtol=0, atol=0)

    # The batch helper must never dispatch the shared computation with B > 1.
    assert all(shape == (1, 1) for shape in calls)


def test_host_pli_batch_rejects_empty_tokens(expect_error):
    model, _ = _host_pli_model()
    with expect_error(ValueError, "at least one token"):
        model.compute_host_pli_batch([])


def test_host_pli_batch_is_none_for_non_pli_target():
    model = Gemma4Model.__new__(Gemma4Model)
    model.hidden_size_per_layer_input = 0
    model.per_layer_input_weights = {}
    assert model.compute_host_pli_batch([1, 2]) is None


def test_verify_pli_requires_ids_or_explicit_inputs(expect_error):
    model, _ = _host_pli_model()
    with expect_error(ValueError, "requires token_ids_host"):
        model._verify_pli_device_tensors(None, None)


def test_verify_pli_validates_layer_count(expect_error):
    model, _ = _host_pli_model()
    with expect_error(ValueError, "3 layers"):
        model._verify_pli_device_tensors(None, [object(), object()])


def test_stacked_pli_validates_layer_count(expect_error):
    model, _ = _host_pli_model()
    with expect_error(ValueError, "3 layers"):
        model._validate_pli_stacked(torch.empty(2, 1, 4, 2))
