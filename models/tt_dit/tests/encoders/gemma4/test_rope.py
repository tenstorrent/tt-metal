# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Math parity: tt_dit ``Gemma4RotaryEmbedding`` vs the actual
``transformers.models.gemma4.modeling_gemma4.Gemma4TextRotaryEmbedding``.

HF returns full-dim cos/sin (shape ``[..., head_dim]`` from concat-style); our
module returns half-dim (shape ``[..., head_dim/2]``). The math is equivalent —
HF duplicates the half via ``torch.cat([freqs, freqs], dim=-1)`` — so we compare
our table against the first half of HF's.

    pytest models/tt_dit/tests/encoders/gemma4/test_rope.py -s
"""

import pytest
import torch
from loguru import logger

import ttnn

from ....encoders.gemma4.rope import Gemma4RotaryEmbedding
from ....utils.check import assert_quality
from ....utils.test import line_params

PCC_THRESHOLD = 0.9999
ALLCLOSE_ATOL = 1e-2
ALLCLOSE_RTOL = 1e-2


@pytest.mark.parametrize(
    ("mesh_device", "num_links", "device_params"),
    [pytest.param((1, 1), 1, line_params, id="single")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    ("layer_type", "seq_len"),
    [
        pytest.param("sliding_attention", 1024, id="sliding-1k"),
        pytest.param("sliding_attention", 4096, id="sliding-4k"),
        pytest.param("full_attention", 1024, id="full-1k"),
        pytest.param("full_attention", 4096, id="full-4k"),
    ],
)
def test_gemma4_rope(mesh_device: ttnn.MeshDevice, num_links: int, layer_type: str, seq_len: int) -> None:
    """Verify our RoPE output matches the actual HF Gemma4TextRotaryEmbedding."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextConfig, Gemma4TextRotaryEmbedding

    torch.manual_seed(0)

    # DiffusionGemma-26B-A4B-it text config.
    sliding_head_dim = 256
    sliding_theta = 10_000.0
    full_head_dim = 512
    full_theta = 1_000_000.0
    full_partial_rotary = 0.25
    max_position_embeddings = max(seq_len * 2, 8192)

    hf_config = Gemma4TextConfig(
        head_dim=sliding_head_dim,
        global_head_dim=full_head_dim,
        max_position_embeddings=max_position_embeddings,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": sliding_theta},
            "full_attention": {
                "rope_type": "proportional",
                "rope_theta": full_theta,
                "partial_rotary_factor": full_partial_rotary,
            },
        },
    )

    # HF reference. We need a dummy `x` for dtype/device extraction.
    hf_rope = Gemma4TextRotaryEmbedding(hf_config).eval()
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    dummy_x = torch.zeros(1, seq_len, hf_config.hidden_size)
    with torch.no_grad():
        hf_cos_full, hf_sin_full = hf_rope(dummy_x, position_ids, layer_type=layer_type)
    # hf_cos_full / hf_sin_full are shape (B, seq, head_dim) — concat-style full-dim.
    # Take the first half to compare with our half-dim table.
    expected_head_dim = sliding_head_dim if layer_type == "sliding_attention" else full_head_dim
    assert hf_cos_full.shape[-1] == expected_head_dim
    hf_cos = hf_cos_full[..., : expected_head_dim // 2].unsqueeze(1)  # (B, 1, seq, head_dim/2)
    hf_sin = hf_sin_full[..., : expected_head_dim // 2].unsqueeze(1)

    # Our module.
    rope = Gemma4RotaryEmbedding(
        max_position_embeddings=max_position_embeddings,
        sliding_head_dim=sliding_head_dim,
        sliding_rope_theta=sliding_theta,
        full_head_dim=full_head_dim,
        full_rope_theta=full_theta,
        full_partial_rotary_factor=full_partial_rotary,
        mesh_device=mesh_device,
    )
    tt_cos, tt_sin = rope.get_cos_sin(layer_type, position_ids)
    tt_cos_torch = ttnn.to_torch(tt_cos).float()
    tt_sin_torch = ttnn.to_torch(tt_sin).float()

    logger.info(f"layer_type={layer_type}, seq_len={seq_len}, head_dim/2={tt_cos_torch.shape[-1]}")

    assert tt_cos_torch.shape == hf_cos.shape, f"cos shape mismatch: {tt_cos_torch.shape} vs {hf_cos.shape}"
    assert tt_sin_torch.shape == hf_sin.shape

    assert_quality(hf_cos.float(), tt_cos_torch, pcc=PCC_THRESHOLD)
    assert_quality(hf_sin.float(), tt_sin_torch, pcc=PCC_THRESHOLD)

    cos_diff = (hf_cos.float() - tt_cos_torch).abs().max().item()
    sin_diff = (hf_sin.float() - tt_sin_torch).abs().max().item()
    logger.info(f"max abs cos diff: {cos_diff:.3e}, max abs sin diff: {sin_diff:.3e}")

    assert torch.allclose(hf_cos.float(), tt_cos_torch, atol=ALLCLOSE_ATOL, rtol=ALLCLOSE_RTOL)
    assert torch.allclose(hf_sin.float(), tt_sin_torch, atol=ALLCLOSE_ATOL, rtol=ALLCLOSE_RTOL)
