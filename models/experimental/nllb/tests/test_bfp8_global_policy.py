# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""Whole-checkpoint precision routing regression; tiny CPU payloads, no device."""
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
import ttnn
from models.experimental.nllb.tt import backend

POLICY = "B"


@pytest.mark.parametrize("precision", ["bf16", "bfp8_b"])
@pytest.mark.parametrize("small", [True, False])
def test_all_transformer_storage_and_lm_savings(precision, small):
    config = dict(
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
    if not small:
        config["encoder_layers"] = 24
    attention = []
    ffn = []
    for side in ("encoder", "decoder"):
        for layer in range(config[side + "_layers"]):
            prefix = f"model.{side}.layers.{layer}."
            attention.extend(
                prefix + name + "." + projection + ".weight"
                for name in (("self_attn",) if side == "encoder" else ("self_attn", "encoder_attn"))
                for projection in ("q_proj", "k_proj", "v_proj", "out_proj")
            )
            ffn.extend(prefix + name + ".weight" for name in ("fc1", "fc2"))
    weights = {name: torch.tensor([[0.1234567, -0.2345678]]) for name in attention + ffn}
    weights.update(
        {
            "model.shared.weight": torch.tensor([[0.3456789, -0.456789]]),
            "model.encoder.layer_norm.weight": torch.ones(2),
            "model.decoder.layers.0.fc1.bias": torch.ones(2),
        }
    )
    uploads = []

    def upload(self, value, *, tiled=True, dtype=None):
        item = SimpleNamespace(source=value, dtype=dtype or ttnn.bfloat16, tiled=tiled)
        uploads.append(item)
        return item

    with patch.object(backend, "load_checkpoint", return_value=weights), patch.object(
        backend, "validate_checkpoint"
    ), patch.object(backend, "validate_checkpoint_config"), patch.object(backend.Backend, "upload", upload):
        model = backend.create_backend(
            "tiny.bin",
            config,
            SimpleNamespace(compute_with_storage_grid_size=lambda: ttnn.CoreCoord(1, 1)),
            precision=precision,
        )
    dominant = ttnn.bfloat16 if precision == "bf16" else ttnn.bfloat8_b
    expected = set(attention + (ffn if POLICY == "B" else [])) if small and precision == "bfp8_b" else set()
    assert model.bf16_matrix_exceptions == expected
    assert len(expected) == (144 + (48 if POLICY == "B" else 0) if small and precision == "bfp8_b" else 0)
    for name in attention + ffn:
        assert model.weights[name].dtype == (ttnn.bfloat16 if name in expected else dominant)
        assert model.weights[name].source is weights[name]  # original checkpoint, never dequantized BFP8
        assert sum(x.source is weights[name] for x in uploads) == 1
    assert model.lm_weight.dtype == dominant and model.lm_weight.tiled
    assert model.embedding_weight.dtype == ttnn.bfloat16 and not model.embedding_weight.tiled
    assert len(uploads) == len(weights) + 1
    assert model.precision_policy["mode"] == model.precision_policy["weights"] == precision
    assert model.precision_policy["activations"] == "bf16"
    assert len(model.precision_policy["exceptions"]) == (0 if precision == "bf16" else 3 + int(small))
    if small and precision == "bfp8_b":
        assert model.lm_weight.dtype != model.weights[attention[0]].dtype  # not BF16 relabeling
        if POLICY == "A":
            assert all(model.weights[n].dtype == ttnn.bfloat8_b for n in ffn)
