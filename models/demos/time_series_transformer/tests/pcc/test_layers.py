# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC tests for the encoder and decoder stacks against HuggingFace hidden states."""

import pytest
import torch

from models.demos.time_series_transformer.reference.torch_reference import compute_metrics
from models.demos.time_series_transformer.tt.config import get_ttnn_dtype
from models.demos.time_series_transformer.tt.ops import to_device, to_torch
from models.demos.time_series_transformer.tt.transformer import Decoder, Encoder
from models.demos.time_series_transformer.tt.weights import substate

# Attention on this checkpoint tops out near 0.9998 because of device matmul precision, so a
# two-layer stack is held to 0.99 rather than 0.999.
STACK_PCC_THRESHOLD = 0.99
LAYER_PCC_THRESHOLD = 0.999


@pytest.fixture(scope="module")
def encoder(device, config, hf_state):
    dtype = get_ttnn_dtype(config.dtype)
    module = Encoder(config, device=device, dtype=dtype)
    result = module.load_hf_state_dict(substate(hf_state, "model.encoder"), strict=True)
    assert not result["missing_keys"]
    return module


@pytest.fixture(scope="module")
def decoder(device, config, hf_state):
    dtype = get_ttnn_dtype(config.dtype)
    module = Decoder(config, device=device, dtype=dtype)
    result = module.load_hf_state_dict(substate(hf_state, "model.decoder"), strict=True)
    assert not result["missing_keys"]
    return module


class TestEncoder:
    def test_last_hidden_state_pcc(self, device, config, goldens, encoder):
        dtype = get_ttnn_dtype(config.dtype)
        actual = to_torch(encoder(to_device(goldens.encoder_input, device=device, dtype=dtype)))

        mse, mae, pcc = compute_metrics(goldens.encoder_last_hidden_state, actual)
        assert pcc > STACK_PCC_THRESHOLD, f"encoder PCC {pcc:.6f} (mse={mse:.3e}, mae={mae:.3e})"

    def test_per_layer_hidden_states_pcc(self, device, config, goldens, encoder):
        dtype = get_ttnn_dtype(config.dtype)
        _, collected = encoder(to_device(goldens.encoder_input, device=device, dtype=dtype), output_hidden_states=True)

        assert len(collected) == len(goldens.encoder_hidden)
        for index, (expected, tt_hidden) in enumerate(zip(goldens.encoder_hidden, collected)):
            _, _, pcc = compute_metrics(expected, to_torch(tt_hidden))
            threshold = LAYER_PCC_THRESHOLD if index == 0 else STACK_PCC_THRESHOLD
            assert pcc > threshold, f"encoder hidden state {index} PCC {pcc:.6f}"

    def test_no_key_left_behind(self, hf_state, encoder):
        """Every encoder weight in the checkpoint must be consumed by some module."""
        result = encoder.load_hf_state_dict(substate(hf_state, "model.encoder"), strict=True)
        assert result["unexpected_keys"] == []


class TestDecoder:
    """The decoder is fed the reference encoder output so its error is measured alone."""

    def test_last_hidden_state_pcc(self, device, config, goldens, decoder):
        dtype = get_ttnn_dtype(config.dtype)
        actual = to_torch(
            decoder(
                to_device(goldens.decoder_input, device=device, dtype=dtype),
                to_device(goldens.encoder_last_hidden_state, device=device, dtype=dtype),
            )
        )

        mse, mae, pcc = compute_metrics(goldens.decoder_last_hidden_state, actual)
        assert pcc > STACK_PCC_THRESHOLD, f"decoder PCC {pcc:.6f} (mse={mse:.3e}, mae={mae:.3e})"

    def test_per_layer_hidden_states_pcc(self, device, config, goldens, decoder):
        dtype = get_ttnn_dtype(config.dtype)
        _, collected = decoder(
            to_device(goldens.decoder_input, device=device, dtype=dtype),
            to_device(goldens.encoder_last_hidden_state, device=device, dtype=dtype),
            output_hidden_states=True,
        )

        assert len(collected) == len(goldens.decoder_hidden)
        for index, (expected, tt_hidden) in enumerate(zip(goldens.decoder_hidden, collected)):
            _, _, pcc = compute_metrics(expected, to_torch(tt_hidden))
            threshold = LAYER_PCC_THRESHOLD if index == 0 else STACK_PCC_THRESHOLD
            assert pcc > threshold, f"decoder hidden state {index} PCC {pcc:.6f}"

    def test_causal_masking_hides_the_future(self, device, config, goldens, decoder):
        """Perturbing a later timestep must not change earlier decoder outputs."""
        dtype = get_ttnn_dtype(config.dtype)
        memory = to_device(goldens.encoder_last_hidden_state, device=device, dtype=dtype)

        original = goldens.decoder_input
        perturbed = original.clone()
        split = config.prediction_length // 2
        perturbed[:, split:, :] += 5.0

        base = to_torch(decoder(to_device(original, device=device, dtype=dtype), memory))
        after = to_torch(decoder(to_device(perturbed, device=device, dtype=dtype), memory))

        torch.testing.assert_close(base[:, :split, :], after[:, :split, :], atol=1e-3, rtol=1e-3)
        assert not torch.allclose(base[:, split:, :], after[:, split:, :])

    def test_cached_stepping_matches_full_pass(self, device, config, goldens, decoder):
        """One token at a time through the KV cache must reproduce the teacher-forced pass."""
        dtype = get_ttnn_dtype(config.dtype)
        memory = to_device(goldens.encoder_last_hidden_state, device=device, dtype=dtype)
        expected = to_torch(decoder(to_device(goldens.decoder_input, device=device, dtype=dtype), memory))

        caches = decoder.new_caches()
        outputs = []
        for step in range(config.prediction_length):
            token = goldens.decoder_input[:, step : step + 1, :]
            outputs.append(to_torch(decoder(to_device(token, device=device, dtype=dtype), memory, caches=caches)))

        actual = torch.cat(outputs, dim=1)
        assert caches[0].self_attention.length == config.prediction_length
        assert caches[0].cross_attention.length == config.context_length

        _, _, pcc = compute_metrics(expected, actual)
        assert pcc > 0.999, f"cached decode PCC {pcc:.6f}"

    def test_no_key_left_behind(self, hf_state, decoder):
        result = decoder.load_hf_state_dict(substate(hf_state, "model.decoder"), strict=True)
        assert result["unexpected_keys"] == []
