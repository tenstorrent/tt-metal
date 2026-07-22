# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
PCC tests for the DeepSeek-V4 Heavily Compressed Attention (HCA) components (prefill).

Each component of the HCA layer (reference modeling_deepseek_v4.py, paper §2.3.2) is
brought up and PCC-tested against its TTNN counterpart in stateless single-shot mode:
  - compressor  (DeepseekV4HCACompressor -> TtHCACompressor): compressed KV + block bias
  - query path  (DeepseekV4Attention L817-820 -> TtHCA._q_stem): q after norm + RoPE
  - KV path     (DeepseekV4Attention L822-823 -> TtHCA._kv_stem): sliding_kv after norm + RoPE

The stem tests exercise TtHCA methods in isolation (development scaffolding, not the
folder's class+forward idiom); remove them once the full TtHCA block forward is PCC-tested.
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4HCACompressor,
    apply_rotary_pos_emb,
)
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.tt.mla.heavily_compressed_attention import TtHCA, TtHCACompressor
from tests.ttnn.utils_for_testing import assert_with_pcc

_SHAPES = [
    (1, 128),
    (1, 256),
    (1, 512),
    (1, 1024),
    (1, 2048),
    (1, 4096),
    (1, 300),
    (1, 4095),
    (2, 512),
]
_SHAPE_IDS = [
    "b1-seq128",
    "b1-seq256",
    "b1-seq512",
    "b1-seq1k",
    "b1-seq2k",
    "b1-seq4k",
    "b1-seq300-unaligned",
    "b1-seq4095-unaligned",
    "b2-seq512",
]


def _flash_config(num_hidden_layers=4):
    return DeepseekV4Config(
        hidden_size=DeepSeekV4FlashConfig.EMB_SIZE,
        head_dim=DeepSeekV4FlashConfig.HEAD_DIM,
        num_attention_heads=DeepSeekV4FlashConfig.NUM_ATTENTION_HEADS,
        num_hidden_layers=num_hidden_layers,
        compress_rates=dict(DeepSeekV4FlashConfig.COMPRESS_RATES),
        compress_rope_theta=DeepSeekV4FlashConfig.COMPRESS_ROPE_THETA,
        rms_norm_eps=DeepSeekV4FlashConfig.RMS_NORM_EPS,
    )


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


@pytest.mark.parametrize("batch, seq_len", _SHAPES, ids=_SHAPE_IDS)
def test_hca_query_path(device, batch, seq_len):
    """
    Test TtHCA._q_stem PCC against the DeepseekV4Attention query path (lines 817-820).

    Compares q [B, num_heads, S, head_dim] after q_a_proj / q_a_norm / q_b_proj /
    q_b_norm (unweighted) / partial compress-RoPE.

    NOTE: development scaffolding — exercises a TtHCA method in isolation (not the
    folder's class+forward idiom). Remove once the full TtHCA block forward is PCC-tested.
    """
    torch.manual_seed(42)

    config = _flash_config()
    logger.debug(f"batch={batch}, seq_len={seq_len}, heads={config.num_attention_heads}, head_dim={config.head_dim}")

    ref = DeepseekV4Attention(config, layer_idx=0).eval()
    assert ref.compressor is not None, "layer_idx=0 must be a heavily_compressed_attention layer"
    with torch.no_grad():
        ref.q_a_norm.weight.uniform_(0.5, 1.5)

    hidden = torch.randn(batch, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)

    logger.debug("Running torch reference query path")
    with torch.no_grad():
        cos, sin = ref.compressor.rotary_emb(hidden, position_ids=position_ids, layer_type="compress")
        q_residual = ref.q_a_norm(ref.q_a_proj(hidden))
        q = ref.q_b_proj(q_residual).view(batch, seq_len, config.num_attention_heads, config.head_dim).transpose(1, 2)
        q = ref.q_b_norm(q)
        q_ref = apply_rotary_pos_emb(q, cos, sin)
    logger.debug(f"Reference q shape: {tuple(q_ref.shape)}")

    tt_model = TtHCA.from_reference(device, ref, config)
    tt_input = ttnn.from_torch(
        hidden.unsqueeze(1),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    logger.debug("Running ttnn forward")
    signpost("HCA_START")
    q_tt = tt_model._q_stem(tt_input, position_ids)
    signpost("HCA_END")
    q_out = ttnn.to_torch(q_tt)
    logger.debug(f"TTNN q shape: {tuple(q_out.shape)}")

    assert q_out.shape == q_ref.shape, f"shape mismatch: tt {tuple(q_out.shape)} vs ref {tuple(q_ref.shape)}"

    pcc_passed, pcc_message = assert_with_pcc(q_ref.to(torch.float32), q_out.to(torch.float32), pcc=0.99)
    logger.debug(f"query path PCC: {pcc_message}")
    assert pcc_passed, f"HCA query path PCC test failed: {pcc_message}"

    logger.debug("PCC test passed!")


@pytest.mark.parametrize("batch, seq_len", _SHAPES, ids=_SHAPE_IDS)
def test_hca_kv_path(device, batch, seq_len):
    """
    Test TtHCA._kv_stem PCC against the DeepseekV4Attention sliding KV path (lines 822-823).

    Compares the single-head sliding_kv [B, 1, S, head_dim] after kv_proj / kv_norm
    (weighted) / partial compress-RoPE, before the compressor concat (stateless, full S).

    NOTE: development scaffolding — exercises a TtHCA method in isolation (not the
    folder's class+forward idiom). Remove once the full TtHCA block forward is PCC-tested.
    """
    torch.manual_seed(42)

    config = _flash_config()
    logger.debug(f"batch={batch}, seq_len={seq_len}, head_dim={config.head_dim}")

    ref = DeepseekV4Attention(config, layer_idx=0).eval()
    assert ref.compressor is not None, "layer_idx=0 must be a heavily_compressed_attention layer"
    with torch.no_grad():
        ref.kv_norm.weight.uniform_(0.5, 1.5)

    hidden = torch.randn(batch, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)

    logger.debug("Running torch reference KV path")
    with torch.no_grad():
        cos, sin = ref.compressor.rotary_emb(hidden, position_ids=position_ids, layer_type="compress")
        kv = ref.kv_norm(ref.kv_proj(hidden)).view(batch, seq_len, -1, config.head_dim).transpose(1, 2)
        kv_ref = apply_rotary_pos_emb(kv, cos, sin)
    logger.debug(f"Reference sliding_kv shape: {tuple(kv_ref.shape)}")

    tt_model = TtHCA.from_reference(device, ref, config)
    tt_input = ttnn.from_torch(
        hidden.unsqueeze(1),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    logger.debug("Running ttnn forward")
    signpost("HCA_START")
    kv_tt = tt_model._kv_stem(tt_input, position_ids)
    signpost("HCA_END")
    kv_out = ttnn.to_torch(kv_tt)
    logger.debug(f"TTNN sliding_kv shape: {tuple(kv_out.shape)}")

    assert kv_out.shape == kv_ref.shape, f"shape mismatch: tt {tuple(kv_out.shape)} vs ref {tuple(kv_ref.shape)}"

    pcc_passed, pcc_message = assert_with_pcc(kv_ref.to(torch.float32), kv_out.to(torch.float32), pcc=0.99)
    logger.debug(f"KV path PCC: {pcc_message}")
    assert pcc_passed, f"HCA KV path PCC test failed: {pcc_message}"

    logger.debug("PCC test passed!")
