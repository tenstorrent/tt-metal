# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
PCC test for the DeepSeek-V4 HCA compressor (prefill).

Compares DeepseekV4HCACompressor (reference modeling_deepseek_v4.py, paper §2.3.2)
against the TTNN TtHCACompressor in stateless single-shot prefill mode. Checks the
compressed KV entries (PCC) and the causal block-bias mask (exact).
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HCACompressor
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.tt.mla.heavily_compressed_attention import TtHCACompressor
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "batch, seq_len",
    [
        (1, 128),
        (1, 256),
        (1, 512),
        (1, 1024),
        (1, 2048),
        (1, 4096),
        (1, 300),
        (1, 4095),
        (2, 512),
    ],
    ids=[
        "b1-seq128",
        "b1-seq256",
        "b1-seq512",
        "b1-seq1k",
        "b1-seq2k",
        "b1-seq4k",
        "b1-seq300-unaligned",
        "b1-seq4095-unaligned",
        "b2-seq512",
    ],
)
def test_hca_compressor(device, batch, seq_len):
    """
    Test TtHCACompressor PCC against DeepseekV4HCACompressor reference.

    Uses DeepSeek-V4-Flash dimensions:
        - hidden_size: 4096
        - head_dim: 512
        - compress_rate (HCA): 128
    """
    torch.manual_seed(42)

    config = DeepseekV4Config(
        hidden_size=DeepSeekV4FlashConfig.EMB_SIZE,
        head_dim=DeepSeekV4FlashConfig.HEAD_DIM,
        num_attention_heads=DeepSeekV4FlashConfig.NUM_ATTENTION_HEADS,
        num_hidden_layers=4,
        compress_rates=dict(DeepSeekV4FlashConfig.COMPRESS_RATES),
        compress_rope_theta=DeepSeekV4FlashConfig.COMPRESS_ROPE_THETA,
        rms_norm_eps=DeepSeekV4FlashConfig.RMS_NORM_EPS,
    )
    compress_rate = config.compress_rates["heavily_compressed_attention"]
    logger.debug(f"batch={batch}, seq_len={seq_len}, hidden_size={config.hidden_size}, head_dim={config.head_dim}")
    logger.debug(f"compress_rate={compress_rate}")

    # Create PyTorch reference with random weights (position_bias is torch.empty and
    # kv_norm.weight defaults to ones — randomise both so the comparison is meaningful).
    logger.debug("Creating DeepseekV4HCACompressor reference")
    ref = DeepseekV4HCACompressor(config).eval()
    with torch.no_grad():
        ref.position_bias.normal_(0.0, 0.02)
        ref.kv_norm.weight.uniform_(0.5, 1.5)

    hidden = torch.randn(batch, seq_len, config.hidden_size)
    q_residual = torch.zeros(batch, seq_len, config.q_lora_rank)  # unused by HCA
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)

    logger.debug("Running torch reference forward")
    with torch.no_grad():
        compressed_kv_ref, block_bias_ref = ref(hidden, q_residual, position_ids, past_key_values=None, layer_idx=0)
    logger.debug(f"Reference compressed_kv shape: {tuple(compressed_kv_ref.shape)}")

    logger.debug("Creating TtHCACompressor with same weights")
    tt_model = TtHCACompressor.from_reference(device, ref, config)

    tt_input = ttnn.from_torch(
        hidden.unsqueeze(1),  # [B, 1, S, hidden]
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    logger.debug(f"Created ttnn input: {tuple(tt_input.shape)}")

    logger.debug("Running ttnn forward")
    signpost("HCA_START")
    compressed_kv_tt, block_bias_tt = tt_model(tt_input, position_ids)
    signpost("HCA_END")
    compressed_kv_out = ttnn.to_torch(compressed_kv_tt)
    logger.debug(f"TTNN compressed_kv shape: {tuple(compressed_kv_out.shape)}")

    assert (
        compressed_kv_out.shape == compressed_kv_ref.shape
    ), f"shape mismatch: tt {tuple(compressed_kv_out.shape)} vs ref {tuple(compressed_kv_ref.shape)}"

    logger.debug("Comparing compressed_kv with PCC")
    pcc_passed, pcc_message = assert_with_pcc(
        compressed_kv_ref.to(torch.float32),
        compressed_kv_out.to(torch.float32),
        pcc=0.99,
    )
    logger.debug(f"compressed_kv PCC: {pcc_message}")
    assert pcc_passed, f"HCA compressor PCC test failed: {pcc_message}"

    logger.debug("Comparing block_bias (exact)")
    torch.testing.assert_close(block_bias_tt, block_bias_ref.to(torch.float32), rtol=0, atol=0)

    logger.debug("PCC test passed!")
