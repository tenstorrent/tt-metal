# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC tests for input construction and the embedding block."""

import pytest
import torch

from models.demos.time_series_transformer.reference.torch_reference import compute_metrics, compute_pcc
from models.demos.time_series_transformer.tt.config import get_ttnn_dtype
from models.demos.time_series_transformer.tt.embeddings import TimeSeriesEmbedding, sinusoidal_position_encoding
from models.demos.time_series_transformer.tt.inputs import create_network_inputs
from models.demos.time_series_transformer.tt.ops import to_device, to_torch
from models.demos.time_series_transformer.tt.weights import substate

PCC_THRESHOLD = 0.999


class TestNetworkInputs:
    """Host-side scaler, lags and covariate assembly against HF's create_network_inputs."""

    def test_transformer_inputs_match(self, config, goldens, hf_embedder_weights):
        result = create_network_inputs(
            config,
            past_values=goldens.inputs["past_values"],
            past_time_features=goldens.inputs["past_time_features"],
            past_observed_mask=goldens.inputs["past_observed_mask"],
            static_categorical_features=goldens.inputs.get("static_categorical_features"),
            static_real_features=goldens.inputs.get("static_real_features"),
            future_values=goldens.future_values,
            future_time_features=goldens.inputs["future_time_features"],
            embedder_weights=hf_embedder_weights,
        )

        assert result.transformer_inputs.shape == goldens.transformer_inputs.shape
        torch.testing.assert_close(result.transformer_inputs, goldens.transformer_inputs)
        torch.testing.assert_close(result.loc, goldens.loc)
        torch.testing.assert_close(result.scale, goldens.scale)
        torch.testing.assert_close(result.static_feat, goldens.static_feat)

    def test_feature_width_matches_config(self, config, goldens):
        assert goldens.transformer_inputs.shape[-1] == config.feature_size
        assert config.feature_size == config.input_size * len(config.lags_sequence) + config.num_covariate_features

    def test_encoder_only_inputs_have_context_length(self, config, goldens, hf_embedder_weights):
        """Without future values the sequence stops at context_length -- the generate path."""
        result = create_network_inputs(
            config,
            past_values=goldens.inputs["past_values"],
            past_time_features=goldens.inputs["past_time_features"],
            past_observed_mask=goldens.inputs["past_observed_mask"],
            static_categorical_features=goldens.inputs.get("static_categorical_features"),
            static_real_features=goldens.inputs.get("static_real_features"),
            embedder_weights=hf_embedder_weights,
        )
        assert result.transformer_inputs.shape[1] == config.context_length
        torch.testing.assert_close(result.transformer_inputs, goldens.transformer_inputs[:, : config.context_length])


class TestPositionalEmbedding:
    def test_matches_checkpoint_table(self, hf_state, config):
        """Our generated table should reproduce the checkpoint's stored buffer."""
        stored = hf_state["model.encoder.embed_positions.weight"].float()
        generated = sinusoidal_position_encoding(stored.shape[0], config.d_model)
        assert compute_pcc(stored, generated) > 0.9999


class TestTimeSeriesEmbedding:
    """value_embedding + positions + layernorm, i.e. the input to encoder/decoder layer 0."""

    @pytest.mark.parametrize("side", ["encoder", "decoder"])
    def test_embedding_pcc(self, device, config, goldens, hf_state, side):
        dtype = get_ttnn_dtype(config.dtype)
        embedding = TimeSeriesEmbedding(config, device=device, dtype=dtype)
        embedding.load_hf_state_dict(substate(hf_state, f"model.{side}"), strict=True)

        if side == "encoder":
            source = goldens.encoder_input
            expected = goldens.encoder_hidden[0]
            offset = 0
        else:
            source = goldens.decoder_input
            expected = goldens.decoder_hidden[0]
            # HF offsets decoder positions by context_length into the shared table.
            offset = config.context_length

        actual = to_torch(embedding(to_device(source, device=device, dtype=dtype), position_offset=offset))

        mse, mae, pcc = compute_metrics(expected, actual)
        assert pcc > PCC_THRESHOLD, f"{side} embedding PCC {pcc:.6f} (mse={mse:.3e}, mae={mae:.3e})"

    def test_value_projection_has_no_bias(self, device, config, hf_state):
        dtype = get_ttnn_dtype(config.dtype)
        embedding = TimeSeriesEmbedding(config, device=device, dtype=dtype)
        embedding.load_hf_state_dict(substate(hf_state, "model.encoder"), strict=True)
        assert embedding.value_embedding.weight_torch.shape == (config.d_model, config.feature_size)
        assert not any("value_projection.bias" in key for key in hf_state)
