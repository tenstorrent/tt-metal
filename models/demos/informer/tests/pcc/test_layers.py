# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC tests for Informer encoder and decoder layers."""

import pytest
import torch

import ttnn
from models.demos.informer.reference.torch_reference import (
    TorchDecoder,
    TorchDecoderLayer,
    TorchEncoder,
    TorchEncoderLayer,
    compute_metrics,
)
from models.demos.informer.reference.torch_reference import make_causal_mask as torch_make_causal_mask
from models.demos.informer.tt.config import TILE_SIZE, InformerConfig, get_ttnn_dtype
from models.demos.informer.tt.ops import make_causal_mask, to_torch
from models.demos.informer.tt.transformer import Decoder, DecoderLayer, Encoder, EncoderLayer, FeedForward, LayerNorm


def pad_length_to_tile(length: int) -> int:
    return ((length + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE


def _copy_linear_weights(tt_module, torch_module, *, weight_attr: str, bias_attr: str) -> None:
    torch_module.weight.data.copy_(getattr(tt_module, weight_attr))
    torch_module.bias.data.copy_(getattr(tt_module, bias_attr))


def _copy_attention_weights(tt_attn, torch_attn) -> None:
    _copy_linear_weights(tt_attn, torch_attn.q_proj, weight_attr="q_weight_torch", bias_attr="q_bias_torch")
    _copy_linear_weights(tt_attn, torch_attn.k_proj, weight_attr="k_weight_torch", bias_attr="k_bias_torch")
    _copy_linear_weights(tt_attn, torch_attn.v_proj, weight_attr="v_weight_torch", bias_attr="v_bias_torch")
    _copy_linear_weights(tt_attn, torch_attn.o_proj, weight_attr="o_weight_torch", bias_attr="o_bias_torch")


def _copy_ffn_weights(tt_ffn, torch_ffn) -> None:
    _copy_linear_weights(tt_ffn, torch_ffn.fc1, weight_attr="w1_torch", bias_attr="b1_torch")
    _copy_linear_weights(tt_ffn, torch_ffn.fc2, weight_attr="w2_torch", bias_attr="b2_torch")


def _copy_norm_weights(tt_norm, torch_norm) -> None:
    torch_norm.weight.data.copy_(tt_norm.weight_torch)
    torch_norm.bias.data.copy_(tt_norm.bias_torch)


def sync_encoder_layer_weights(tt_layer, torch_layer) -> None:
    _copy_attention_weights(tt_layer.attn, torch_layer.attn)
    _copy_ffn_weights(tt_layer.ffn, torch_layer.ffn)
    _copy_norm_weights(tt_layer.norm1, torch_layer.norm1)
    _copy_norm_weights(tt_layer.norm2, torch_layer.norm2)


def sync_decoder_layer_weights(tt_layer, torch_layer) -> None:
    _copy_attention_weights(tt_layer.self_attn, torch_layer.self_attn)
    _copy_attention_weights(tt_layer.cross_attn, torch_layer.cross_attn)
    _copy_ffn_weights(tt_layer.ffn, torch_layer.ffn)
    _copy_norm_weights(tt_layer.norm1, torch_layer.norm1)
    _copy_norm_weights(tt_layer.norm2, torch_layer.norm2)
    _copy_norm_weights(tt_layer.norm3, torch_layer.norm3)


class TestLayerNorm:
    """Test LayerNorm implementation."""

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("seq_len", [32, 64])
    @pytest.mark.parametrize("d_model", [64, 128])
    def test_layer_norm_pcc(self, device, batch_size, seq_len, d_model):
        """Test LayerNorm matches PyTorch reference."""
        rng = torch.Generator().manual_seed(42)
        dtype = ttnn.bfloat16

        # Create TTNN LayerNorm
        ttnn_ln = LayerNorm(d_model, rng, device=device, dtype=dtype)

        # Input
        x = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32)
        x_ttnn = ttnn.from_torch(x, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)

        # PyTorch reference
        expected = torch.nn.functional.layer_norm(
            x,
            (d_model,),
            weight=ttnn_ln.weight_torch,
            bias=ttnn_ln.bias_torch,
            eps=ttnn_ln.eps,
        )

        # TTNN forward
        actual = to_torch(ttnn_ln(x_ttnn))

        pcc = compute_metrics(expected, actual)[2]
        assert pcc > 0.99, f"LayerNorm PCC {pcc:.4f} < 0.99"


class TestFeedForward:
    """Test FeedForward (MLP) layer."""

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("seq_len", [32, 64])
    @pytest.mark.parametrize("d_model,d_ff", [(64, 256), (128, 512)])
    def test_feed_forward_pcc(self, device, batch_size, seq_len, d_model, d_ff):
        """Test FeedForward matches PyTorch reference."""
        rng = torch.Generator().manual_seed(42)
        dtype = ttnn.bfloat16

        # Create TTNN FFN
        ttnn_ffn = FeedForward(d_model, d_ff, 0.0, rng, device=device, dtype=dtype)

        # Input
        x = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32)
        x_ttnn = ttnn.from_torch(x, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)

        # PyTorch reference
        h = x @ ttnn_ffn.w1_torch.T + ttnn_ffn.b1_torch
        h = torch.nn.functional.gelu(h)
        expected = h @ ttnn_ffn.w2_torch.T + ttnn_ffn.b2_torch

        # TTNN forward
        actual = to_torch(ttnn_ffn(x_ttnn))

        pcc = compute_metrics(expected, actual)[2]
        assert pcc > 0.98, f"FeedForward PCC {pcc:.4f} < 0.98"


class TestEncoderLayer:
    """Test Encoder layer."""

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("seq_len", [32, 64])
    def test_encoder_layer_pcc(self, device, batch_size, seq_len):
        """Test EncoderLayer produces valid output."""
        config = InformerConfig(
            enc_in=7,
            d_model=64,
            n_heads=2,
            d_ff=256,
            e_layers=1,
            dtype="bfloat16",
            attention_type="prob",
        )
        rng = torch.Generator().manual_seed(42)
        dtype = get_ttnn_dtype(config.dtype)

        # Create encoder layer
        enc_layer = EncoderLayer(config, rng, device=device, dtype=dtype)

        # Input
        x = torch.randn(batch_size, seq_len, config.d_model, dtype=torch.float32)
        x_ttnn = ttnn.from_torch(x, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)

        # Forward pass
        out = enc_layer(x_ttnn, None, seq_len)
        out_torch = to_torch(out)
        out_ref_layer = TorchEncoderLayer(config)
        out_ref_layer.eval()
        sync_encoder_layer_weights(enc_layer, out_ref_layer)
        with torch.no_grad():
            out_ref = out_ref_layer(x, None, seq_len)

        assert out_torch.shape == (batch_size, seq_len, config.d_model)
        mse, mae, corr = compute_metrics(out_ref.float(), out_torch.float())
        assert mse < 1e-2, f"Encoder layer MSE {mse:.6f} too high"
        assert mae < 1e-1, f"Encoder layer MAE {mae:.6f} too high"
        assert corr > 0.95, f"Encoder layer PCC {corr:.4f} too low"


class TestDecoderLayer:
    """Test Decoder layer."""

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("dec_len,enc_len", [(32, 64), (48, 48)])
    def test_decoder_layer_pcc(self, device, batch_size, dec_len, enc_len):
        """Test DecoderLayer produces valid output."""
        config = InformerConfig(
            dec_in=7,
            d_model=64,
            n_heads=2,
            d_ff=256,
            d_layers=1,
            dtype="bfloat16",
        )
        rng = torch.Generator().manual_seed(42)
        dtype = get_ttnn_dtype(config.dtype)

        # Create decoder layer
        dec_layer = DecoderLayer(config, rng, device=device, dtype=dtype)

        # Inputs
        x = torch.randn(batch_size, dec_len, config.d_model, dtype=torch.float32)
        enc_out = torch.randn(batch_size, enc_len, config.d_model, dtype=torch.float32)
        x_ttnn = ttnn.from_torch(x, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        enc_ttnn = ttnn.from_torch(enc_out, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)

        # Causal mask
        causal_mask = make_causal_mask(
            dec_len,
            batch=batch_size,
            heads=config.n_heads,
            device=device,
            dtype=dtype,
            mask_value=config.attn_mask_value,
        )

        # Forward pass
        out = dec_layer(x_ttnn, enc_ttnn, causal_mask, None, enc_len)
        out_torch = to_torch(out)
        causal_mask_torch = torch_make_causal_mask(
            pad_length_to_tile(dec_len),
            config.attn_mask_value,
            device=x.device,
        )
        out_ref_layer = TorchDecoderLayer(config)
        out_ref_layer.eval()
        sync_decoder_layer_weights(dec_layer, out_ref_layer)
        with torch.no_grad():
            out_ref = out_ref_layer(x, enc_out, causal_mask_torch, None, enc_len)

        assert out_torch.shape == (batch_size, dec_len, config.d_model)
        mse, mae, corr = compute_metrics(out_ref.float(), out_torch.float())
        assert mse < 1e-2, f"Decoder layer MSE {mse:.6f} too high"
        assert mae < 1e-1, f"Decoder layer MAE {mae:.6f} too high"
        assert corr > 0.95, f"Decoder layer PCC {corr:.4f} too low"


class TestEncoder:
    """Test full Encoder with distilling."""

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_encoder_with_distilling(self, device, batch_size):
        """Test Encoder with distilling produces correct output shape."""
        config = InformerConfig(
            enc_in=7,
            seq_len=96,
            d_model=64,
            n_heads=2,
            d_ff=256,
            e_layers=2,
            dtype="bfloat16",
            attention_type="prob",
        )
        rng = torch.Generator().manual_seed(42)

        # Create encoder
        encoder = Encoder(config, rng, device=device)

        # Input
        x = torch.randn(batch_size, config.seq_len, config.d_model, dtype=torch.float32)
        x_ttnn = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        # Forward pass
        enc_out, valid_len = encoder(x_ttnn, None)
        enc_out_torch = to_torch(enc_out)
        encoder_ref = TorchEncoder(config)
        encoder_ref.eval()
        for tt_layer, torch_layer in zip(encoder.layers, encoder_ref.layers):
            sync_encoder_layer_weights(tt_layer, torch_layer)
        if encoder.distil_norm is not None and encoder_ref.distil_norm is not None:
            _copy_norm_weights(encoder.distil_norm, encoder_ref.distil_norm)
        with torch.no_grad():
            enc_ref, valid_len_ref = encoder_ref(x, None)

        assert enc_out_torch.shape[0] == batch_size
        assert enc_out_torch.shape[2] == config.d_model
        assert (
            valid_len < config.seq_len
        ), f"Valid length should reduce with distilling: {valid_len} >= {config.seq_len}"
        assert valid_len == valid_len_ref
        mse, mae, corr = compute_metrics(enc_ref.float(), enc_out_torch.float())
        assert mse < 2e-2, f"Encoder MSE {mse:.6f} too high"
        assert mae < 1e-1, f"Encoder MAE {mae:.6f} too high"
        assert corr > 0.90, f"Encoder PCC {corr:.4f} too low"


class TestDecoder:
    """Test full Decoder."""

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_decoder(self, device, batch_size):
        """Test Decoder produces correct output shape."""
        config = InformerConfig(
            dec_in=7,
            label_len=48,
            pred_len=24,
            d_model=64,
            n_heads=2,
            d_ff=256,
            d_layers=1,
            dtype="bfloat16",
        )
        rng = torch.Generator().manual_seed(42)

        # Create decoder
        decoder = Decoder(config, rng, device=device)

        # Inputs
        dec_len = config.label_len + config.pred_len
        enc_len = 64
        x = torch.randn(batch_size, dec_len, config.d_model, dtype=torch.float32)
        enc_out = torch.randn(batch_size, enc_len, config.d_model, dtype=torch.float32)
        x_ttnn = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        enc_ttnn = ttnn.from_torch(enc_out, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        # Causal mask
        causal_mask = make_causal_mask(
            dec_len,
            batch=batch_size,
            heads=config.n_heads,
            device=device,
            dtype=ttnn.bfloat16,
            mask_value=config.attn_mask_value,
        )

        # Forward pass
        dec_out = decoder(x_ttnn, enc_ttnn, causal_mask, None, enc_len)
        dec_out_torch = to_torch(dec_out)
        decoder_ref = TorchDecoder(config)
        decoder_ref.eval()
        for tt_layer, torch_layer in zip(decoder.layers, decoder_ref.layers):
            sync_decoder_layer_weights(tt_layer, torch_layer)
        causal_mask_torch = torch_make_causal_mask(
            pad_length_to_tile(dec_len),
            config.attn_mask_value,
            device=x.device,
        )
        with torch.no_grad():
            dec_ref = decoder_ref(x, enc_out, causal_mask_torch, None, enc_len)

        assert dec_out_torch.shape == (batch_size, dec_len, config.d_model)
        mse, mae, corr = compute_metrics(dec_ref.float(), dec_out_torch.float())
        assert mse < 2e-2, f"Decoder MSE {mse:.6f} too high"
        assert mae < 1e-1, f"Decoder MAE {mae:.6f} too high"
        assert corr > 0.90, f"Decoder PCC {corr:.4f} too low"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
