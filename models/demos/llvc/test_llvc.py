#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import sys
import os

# Add the models directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

try:
    import ttnn
    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False
    print("TTNN not available, skipping tests")
    sys.exit(0)

from models.demos.llvc.ttnn_llvc import TtDilatedCausalConvEncoder, TtCausalTransformerDecoder, TtMaskNet, TtLLVCModel


def test_encoder():
    """Test the Dilated Causal Convolution Encoder"""
    print("Testing TtDilatedCausalConvEncoder...")

    # Create a mock device
    device = None  # We'll use CPU/memory for testing

    encoder = TtDilatedCausalConvEncoder(
        channels=512,
        num_layers=10,
        device=device
    )

    # Test buffer initialization
    ctx_buf = encoder.init_ctx_buf(batch_size=1)
    print(f"Context buffer shape: {ctx_buf.shape}")

    # Test forward pass with dummy data
    x = ttnn.zeros((1, 512, 100), dtype=ttnn.bfloat16, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x_out, ctx_buf_out = encoder.forward(x, ctx_buf)
    print(f"Encoder output shape: {x_out.shape}")
    print("Encoder test passed!")


def test_decoder():
    """Test the Causal Transformer Decoder"""
    print("Testing TtCausalTransformerDecoder...")

    device = None

    decoder = TtCausalTransformerDecoder(
        model_dim=256,
        ctx_len=100,
        chunk_size=72,
        num_layers=2,
        nhead=8,
        use_pos_enc=True,
        ff_dim=512,
        dropout=0.0,
        device=device
    )

    # Test buffer initialization
    ctx_buf = decoder.init_ctx_buf(batch_size=1)
    print(f"Decoder context buffer shape: {ctx_buf.shape}")

    # Test forward pass with dummy data
    tgt = ttnn.zeros((1, 256, 72), dtype=ttnn.bfloat16, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    mem = ttnn.zeros((1, 256, 100), dtype=ttnn.bfloat16, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    tgt_out, ctx_buf_out = decoder.forward(tgt, mem, ctx_buf)
    print(f"Decoder output shape: {tgt_out.shape}")
    print("Decoder test passed!")


def test_mask_net():
    """Test the Mask Network"""
    print("Testing TtMaskNet...")

    device = None

    mask_net = TtMaskNet(
        enc_dim=512,
        num_enc_layers=10,
        dec_dim=256,
        dec_buf_len=100,
        dec_chunk_size=72,
        num_dec_layers=2,
        use_pos_enc=True,
        skip_connection=True,
        proj=True,
        decoder_dropout=0.0,
        device=device
    )

    # Test forward pass with dummy data
    x = ttnn.zeros((1, 512, 100), dtype=ttnn.bfloat16, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    l = ttnn.zeros((1, 512, 100), dtype=ttnn.bfloat16, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    enc_buf = mask_net.encoder.init_ctx_buf(batch_size=1)
    dec_buf = mask_net.decoder.init_ctx_buf(batch_size=1)

    m, enc_buf_out, dec_buf_out = mask_net.forward(x, l, enc_buf, dec_buf)
    print(f"Mask output shape: {m.shape}")
    print("MaskNet test passed!")


def test_llvc_model():
    """Test the full LLVC model"""
    print("Testing TtLLVCModel...")

    device = None

    model = TtLLVCModel(
        L=8,
        enc_dim=512,
        num_enc_layers=10,
        dec_dim=256,
        dec_buf_len=100,
        num_dec_layers=2,
        dec_chunk_size=72,
        out_buf_len=2,
        use_pos_enc=True,
        skip_connection=True,
        proj=True,
        lookahead=True,
        decoder_dropout=0.0,
        device=device
    )

    # Test buffer initialization
    enc_buf, dec_buf, out_buf = model.init_buffers(batch_size=1)
    print(f"Model buffers initialized: enc_buf={enc_buf.shape}, dec_buf={dec_buf.shape}, out_buf={out_buf.shape}")

    # Test forward pass with dummy data
    x = ttnn.zeros((1, 1, 200), dtype=ttnn.bfloat16, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    output = model.forward(x, enc_buf, dec_buf, out_buf, pad=True)
    print(f"Model output shape: {output.shape}")
    print("LLVC model test passed!")


def main():
    """Run all tests"""
    print("Running LLVC TTNN implementation tests...")
    print("=" * 50)

    try:
        test_encoder()
        print()

        test_decoder()
        print()

        test_mask_net()
        print()

        test_llvc_model()
        print()

        print("=" * 50)
        print("All tests passed! ✅")

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
