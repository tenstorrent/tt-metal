# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Vision-layer parity: TT ``Gemma4VisionEncoderLayer`` vs HF
``Gemma4VisionEncoderLayer``.

    pytest models/tt_dit/tests/encoders/gemma4/test_vision_layer.py -s
"""

import pytest
import torch
from loguru import logger

import ttnn

from ....encoders.gemma4.vision_layer import Gemma4VisionEncoderLayer
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor, local_device_to_torch
from ....utils.test import line_params, ring_params

PCC_THRESHOLD = 0.99
ALLCLOSE_ATOL = 5e-2
ALLCLOSE_RTOL = 5e-2


@pytest.mark.parametrize(
    ("mesh_device", "tp_axis", "num_links", "device_params", "topology"),
    [
        pytest.param((2, 4), 0, 1, line_params, ttnn.Topology.Linear, id="bh_qb2_tp2"),
        pytest.param((2, 4), 0, 1, ring_params, ttnn.Topology.Ring, id="wh_t3k_tp2"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("num_patches", [64])
def test_vision_layer(
    mesh_device: ttnn.MeshDevice, tp_axis: int, num_links: int, topology: ttnn.Topology, num_patches: int
) -> None:
    """TT vision encoder layer vs HF Gemma4VisionEncoderLayer."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionEncoderLayer as HFVisionLayer
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionRotaryEmbedding as HFVisionRotary

    torch.manual_seed(0)
    dtype = torch.float32

    hidden_size = 1152
    intermediate_size = 4304
    num_attention_heads = 16
    head_dim = 72
    head_dim_padded = 96
    position_embedding_size = 256
    rope_theta = 100.0
    eps = 1e-6
    B = 1

    tp_factor = tuple(mesh_device.shape)[tp_axis]

    hf_config = Gemma4VisionConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_attention_heads,
        head_dim=head_dim,
        rope_parameters={"rope_theta": rope_theta, "rope_type": "default"},
        position_embedding_size=position_embedding_size,
        patch_size=16,
        num_hidden_layers=1,
        rms_norm_eps=eps,
        hidden_activation="gelu_pytorch_tanh",
        attention_bias=False,
        attention_dropout=0.0,
    )

    hf_layer = HFVisionLayer(hf_config, layer_idx=0).to(dtype).eval()
    hf_rope = HFVisionRotary(hf_config).eval()

    hidden_states = torch.randn(B, num_patches, hidden_size, dtype=dtype)
    px = torch.randint(0, position_embedding_size, (B, num_patches), dtype=torch.long)
    py = torch.randint(0, position_embedding_size, (B, num_patches), dtype=torch.long)
    position_ids = torch.stack([px, py], dim=-1)

    with torch.no_grad():
        hf_cos_full, hf_sin_full = hf_rope(hidden_states, position_ids)
        torch_out = hf_layer(
            hidden_states=hidden_states,
            position_embeddings=(hf_cos_full, hf_sin_full),
            attention_mask=None,
            position_ids=position_ids,
        )

    # Reslice cos/sin into our half-dim padded layout (same logic as in test_vision_attention).
    unique_per_dim = head_dim // 4  # 18
    half = head_dim_padded // 2  # 48
    pad_len = half - 2 * unique_per_dim  # 12
    cos_half_x = hf_cos_full[..., :unique_per_dim]
    cos_half_y = hf_cos_full[..., 2 * unique_per_dim : 3 * unique_per_dim]
    sin_half_x = hf_sin_full[..., :unique_per_dim]
    sin_half_y = hf_sin_full[..., 2 * unique_per_dim : 3 * unique_per_dim]
    cos_pad = torch.ones(B, num_patches, pad_len)
    sin_pad = torch.zeros(B, num_patches, pad_len)
    cos = torch.cat([cos_half_x, cos_half_y, cos_pad], dim=-1).unsqueeze(1).to(torch.bfloat16)
    sin = torch.cat([sin_half_x, sin_half_y, sin_pad], dim=-1).unsqueeze(1).to(torch.bfloat16)
    tt_cos = ttnn.from_torch(cos, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_sin = ttnn.from_torch(sin, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=1 - tp_axis, factor=tuple(mesh_device.shape)[1 - tp_axis]),
        cfg_parallel=None,
    )

    tt_layer = Gemma4VisionEncoderLayer(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        head_dim=head_dim,
        head_dim_padded=head_dim_padded,
        rms_norm_eps=eps,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
    )
    tt_layer.load_state_dict(hf_layer.state_dict())

    tt_hidden = bf16_tensor(hidden_states.unsqueeze(0), device=mesh_device)
    tt_out = tt_layer(tt_hidden, tt_cos, tt_sin, attention_mask=None)
    tt_out_torch = local_device_to_torch(tt_out).squeeze(0)

    logger.info(f"torch_out: {torch_out.shape}, tt_out: {tt_out_torch.shape}")
    assert_quality(torch_out, tt_out_torch, pcc=PCC_THRESHOLD)

    abs_diff = (torch_out - tt_out_torch.to(dtype)).abs()
    logger.info(f"max abs diff: {abs_diff.max().item():.3e}")
    assert torch.allclose(torch_out, tt_out_torch.to(dtype), atol=ALLCLOSE_ATOL, rtol=ALLCLOSE_RTOL)
