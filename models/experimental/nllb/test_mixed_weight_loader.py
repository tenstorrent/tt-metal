# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""Independent loader routing checks; tiny CPU payloads, no TT device."""

from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
import ttnn

if __package__:
    from . import backend
else:
    import backend


def configurations():
    small = dict(
        d_model=1024,
        vocab_size=256206,
        encoder_layers=12,
        decoder_layers=12,
        encoder_attention_heads=16,
        decoder_attention_heads=16,
        encoder_ffn_dim=4096,
        decoder_ffn_dim=4096,
        max_position_embeddings=1024,
    )
    anchor = dict(small, encoder_layers=24, decoder_layers=24, encoder_ffn_dim=8192, decoder_ffn_dim=8192)
    large = dict(anchor, d_model=2048)
    return small, anchor, large


@pytest.mark.parametrize("precision", ["bf16", "bfp8_b"])
def test_loader_precision_selection(precision):
    small, anchor, large = configurations()
    # Independently list keys, including neighboring layers, decoder and FFN.
    selected = {
        "model.encoder.layers.0.self_attn.q_proj.weight",
        "model.encoder.layers.0.self_attn.k_proj.weight",
        "model.encoder.layers.0.self_attn.v_proj.weight",
        "model.encoder.layers.0.self_attn.out_proj.weight",
    }
    selected |= {
        "model.encoder.layers.1.self_attn.q_proj.weight",
        "model.decoder.layers.0.self_attn.q_proj.weight",
        "model.decoder.layers.0.encoder_attn.k_proj.weight",
    }
    selected |= {"model.encoder.layers.0.fc1.weight", "model.encoder.layers.0.fc2.weight"}
    matrices = sorted(selected) + [
        "model.encoder.layers.1.self_attn.q_proj.weight",
        "model.decoder.layers.0.self_attn.q_proj.weight",
        "model.decoder.layers.0.encoder_attn.k_proj.weight",
        "model.encoder.layers.0.fc1.weight",
        "model.encoder.layers.0.fc2.weight",
    ]
    matrices = sorted(set(matrices))
    cases = [(small, True), (anchor, False), (large, False)]
    for key in small:
        altered = dict(small)
        altered[key] *= 2
        cases.append((altered, False))
    for config, is_small in cases:
        weights = {name: torch.tensor([[0.1234567, -0.2345678]]) for name in matrices}
        weights["model.shared.weight"] = torch.tensor([[0.3456789, -0.456789]])
        weights["model.encoder.layers.0.self_attn.q_proj.bias"] = torch.ones(2)
        weights["model.encoder.layer_norm.weight"] = torch.ones(2)
        uploads = []

        def upload(self, value, *, tiled=True, dtype=None):
            result = SimpleNamespace(source=value, dtype=dtype or ttnn.bfloat16, tiled=tiled)
            uploads.append(result)
            return result

        # Only payload shape validation/device transport are stubbed. Exercise
        # real factory, config validation, constructor routing and declarations.
        with (
            patch.object(backend, "load_checkpoint", return_value=weights),
            patch.object(backend, "validate_checkpoint"),
            patch.object(backend, "validate_checkpoint_config"),
            patch.object(backend.Backend, "upload", upload),
        ):
            device = SimpleNamespace(compute_with_storage_grid_size=lambda: ttnn.CoreCoord(1, 1))
            obj = backend.create_backend("fixture.bin", config, device, precision=precision)
        dominant = ttnn.bfloat8_b if precision == "bfp8_b" else ttnn.bfloat16
        for name in matrices:
            expected = ttnn.bfloat16 if is_small and name in selected else dominant
            assert obj.weights[name].dtype == expected, (config, precision, name)
            assert obj.weights[name].source is weights[name]  # direct checkpoint upload
        assert len(uploads) == len(weights) + 1  # shared embedding uploaded twice
        assert obj.lm_weight.dtype == dominant
        assert obj.embedding_weight.dtype == ttnn.bfloat16
        assert not obj.embedding_weight.tiled
        for name in weights:
            if weights[name].ndim == 1:
                assert obj.weights[name].dtype == ttnn.bfloat16
        policy = obj.precision_policy
        assert policy["mode"] == policy["weights"] == precision
        assert policy["activations"] == "bf16"
        assert ("600M only" in " ".join(policy["exceptions"])) == (is_small and precision == "bfp8_b")
        assert len(policy["exceptions"]) == (0 if precision == "bf16" else 3 + int(is_small))
