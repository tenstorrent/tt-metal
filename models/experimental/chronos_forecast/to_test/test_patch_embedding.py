# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Correctness checks for the parked host patch embedding. Speed vs P150 comes later."""

from __future__ import annotations

import pytest
import torch

from models.experimental.chronos_forecast.to_test.patch_embedding import (
    ResidualBlockWeights,
    embed_patched_inputs,
    residual_block,
)
from models.experimental.chronos_forecast.tt.model_preprocessing import Chronos2PatchedInputs


def _residual_weights(module, act_fn_name: str = "relu") -> ResidualBlockWeights:
    return ResidualBlockWeights(
        hidden_weight=module.hidden_layer.weight,
        hidden_bias=module.hidden_layer.bias,
        output_weight=module.output_layer.weight,
        output_bias=module.output_layer.bias,
        residual_weight=module.residual_layer.weight,
        residual_bias=module.residual_layer.bias,
        act_fn_name=act_fn_name,
    )


def _patched_inputs(
    patched_context: torch.Tensor,
    attention_mask: torch.Tensor,
    patched_future: torch.Tensor,
) -> Chronos2PatchedInputs:
    batch = patched_context.shape[0]
    return Chronos2PatchedInputs(
        patched_context=patched_context,
        attention_mask=attention_mask,
        patched_future=patched_future,
        loc_scale=(torch.zeros(batch, 1), torch.ones(batch, 1)),
        group_ids=torch.arange(batch),
        target_idx_ranges=[(0, batch)],
    )


def _reference_encode_embeds(model, patched_context, attention_mask, patched_future):
    """Embed sequence Chronos2Model.encode builds before self.encoder."""
    batch_size = patched_context.shape[0]
    input_embeds = model.input_patch_embedding(patched_context)
    mask = attention_mask
    if model.chronos_config.use_reg_token:
        reg_input_ids = torch.full((batch_size, 1), model.config.reg_token_id, device=input_embeds.device)
        reg_embeds = model.shared(reg_input_ids)
        input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
        mask = torch.cat([mask.to(model.dtype), torch.ones_like(reg_input_ids).to(model.dtype)], dim=-1)
    future_embeds = model.input_patch_embedding(patched_future)
    future_attention_mask = torch.ones(
        batch_size, patched_future.shape[1], dtype=model.dtype, device=input_embeds.device
    )
    input_embeds = torch.cat([input_embeds, future_embeds], dim=-2)
    mask = torch.cat([mask.to(dtype=future_attention_mask.dtype), future_attention_mask], dim=-1)
    return input_embeds, mask


def test_residual_block_matches_vendored_reference():
    from models.experimental.chronos_forecast.reference.chronos2.layers import ResidualBlock as RefResidualBlock

    torch.manual_seed(0)
    ref = RefResidualBlock(in_dim=48, h_dim=16, out_dim=8, act_fn_name="relu", dropout_p=0.1).eval()
    x = torch.randn(2, 4, 48)
    torch.testing.assert_close(residual_block(x, _residual_weights(ref)), ref(x), atol=0, rtol=0)


def test_residual_block_oracle_vs_amazon_submodule():
    from models.experimental.chronos_forecast.common.chronos_src import ensure_chronos_on_path

    ensure_chronos_on_path()
    from chronos.chronos2.layers import ResidualBlock as UpResidualBlock

    torch.manual_seed(0)
    up = UpResidualBlock(in_dim=48, h_dim=16, out_dim=8, act_fn_name="relu", dropout_p=0.1).eval()
    x = torch.randn(2, 4, 48)
    torch.testing.assert_close(residual_block(x, _residual_weights(up)), up(x), atol=0, rtol=0)


def test_residual_block_rejects_non_relu():
    weights = ResidualBlockWeights(
        hidden_weight=torch.zeros(2, 2),
        hidden_bias=torch.zeros(2),
        output_weight=torch.zeros(2, 2),
        output_bias=torch.zeros(2),
        residual_weight=torch.zeros(2, 2),
        residual_bias=torch.zeros(2),
        act_fn_name="gelu",
    )
    with pytest.raises(ValueError, match="relu"):
        residual_block(torch.zeros(1, 2), weights)


def test_embed_patched_halves_match_reference_model():
    from models.experimental.chronos_forecast.common.chronos_src import CHRONOS_SUBMODULE_ROOT
    from models.experimental.chronos_forecast.reference.chronos2.model import Chronos2Model as RefModel

    dummy = CHRONOS_SUBMODULE_ROOT / "test" / "dummy-chronos2-model"
    model = RefModel.from_pretrained(dummy).eval()
    torch.manual_seed(0)
    context = torch.randn(2, 32)
    patched_context, attention_mask, loc_scale = model._prepare_patched_context(context)
    future = torch.randn(2, 16)
    future[0, 3] = float("nan")
    patched_future, _ = model._prepare_patched_future(
        future_covariates=future,
        future_covariates_mask=None,
        loc_scale=loc_scale,
        num_output_patches=1,
        batch_size=2,
    )
    weights = _residual_weights(model.input_patch_embedding, act_fn_name=model.config.dense_act_fn)
    torch.testing.assert_close(
        residual_block(patched_context, weights),
        model.input_patch_embedding(patched_context),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        residual_block(patched_future, weights),
        model.input_patch_embedding(patched_future),
        atol=0,
        rtol=0,
    )

    patched = _patched_inputs(patched_context, attention_mask, patched_future)
    reg = model.shared.weight[model.config.reg_token_id] if model.chronos_config.use_reg_token else None
    embedded = embed_patched_inputs(patched, weights, reg_embedding=reg)
    n_context = patched_context.shape[1]
    n_reg = 1 if reg is not None else 0
    torch.testing.assert_close(
        embedded.inputs_embeds[:, :n_context],
        model.input_patch_embedding(patched_context),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        embedded.inputs_embeds[:, n_context + n_reg :],
        model.input_patch_embedding(patched_future),
        atol=0,
        rtol=0,
    )


def test_embed_patched_inputs_matches_reference_encode():
    from models.experimental.chronos_forecast.common.chronos_src import CHRONOS_SUBMODULE_ROOT
    from models.experimental.chronos_forecast.reference.chronos2.model import Chronos2Model as RefModel

    dummy = CHRONOS_SUBMODULE_ROOT / "test" / "dummy-chronos2-model"
    model = RefModel.from_pretrained(dummy).eval()
    torch.manual_seed(0)
    context = torch.randn(2, 32)
    patched_context, attention_mask, loc_scale = model._prepare_patched_context(context)
    future = torch.randn(2, 16)
    patched_future, _ = model._prepare_patched_future(
        future_covariates=future,
        future_covariates_mask=None,
        loc_scale=loc_scale,
        num_output_patches=1,
        batch_size=2,
    )
    weights = _residual_weights(model.input_patch_embedding, act_fn_name=model.config.dense_act_fn)
    reg = model.shared.weight[model.config.reg_token_id] if model.chronos_config.use_reg_token else None
    embedded = embed_patched_inputs(
        _patched_inputs(patched_context, attention_mask, patched_future),
        weights,
        reg_embedding=reg,
    )
    ref_embeds, ref_mask = _reference_encode_embeds(model, patched_context, attention_mask, patched_future)
    torch.testing.assert_close(embedded.inputs_embeds, ref_embeds, atol=0, rtol=0)
    torch.testing.assert_close(embedded.attention_mask, ref_mask, atol=0, rtol=0)


def test_embed_patched_inputs_inserts_reg_token():
    from models.experimental.chronos_forecast.reference.chronos2.layers import ResidualBlock as RefResidualBlock

    torch.manual_seed(0)
    block = RefResidualBlock(in_dim=48, h_dim=16, out_dim=8, act_fn_name="relu", dropout_p=0.1).eval()
    shared = torch.nn.Embedding(2, 8)
    reg_id = 1
    patched_context = torch.randn(2, 3, 48)
    patched_future = torch.randn(2, 1, 48)
    attention_mask = torch.tensor([[True, True, False], [True, False, True]])
    embedded = embed_patched_inputs(
        _patched_inputs(patched_context, attention_mask, patched_future),
        _residual_weights(block),
        reg_embedding=shared.weight[reg_id],
    )

    context_embeds = block(patched_context)
    future_embeds = block(patched_future)
    reg_embeds = shared(torch.full((2, 1), reg_id))
    expected = torch.cat([context_embeds, reg_embeds, future_embeds], dim=1)
    expected_mask = torch.cat(
        [
            attention_mask.to(dtype=context_embeds.dtype),
            torch.ones(2, 1, dtype=context_embeds.dtype),
            torch.ones(2, 1, dtype=context_embeds.dtype),
        ],
        dim=-1,
    )
    torch.testing.assert_close(embedded.inputs_embeds, expected, atol=0, rtol=0)
    torch.testing.assert_close(embedded.attention_mask, expected_mask, atol=0, rtol=0)
    assert embedded.inputs_embeds.shape == (2, 5, 8)
    assert torch.equal(embedded.group_ids, torch.arange(2))
