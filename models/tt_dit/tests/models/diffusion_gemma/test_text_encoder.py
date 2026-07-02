# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end encoder parity: TT ``DiffusionGemmaEncoderTextModel`` vs the actual
HF ``DiffusionGemmaEncoderTextModel``.

Tiny config (256 hidden, 6 layers, 8 experts) so the test stays bounded.

    pytest models/tt_dit/tests/models/diffusion_gemma/test_text_encoder.py -s
"""

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.diffusion_gemma._state_utils import per_layer_moe_substates
from ....models.transformers.diffusion_gemma.text_encoder import DiffusionGemmaEncoderTextModel
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality
from ....utils.tensor import local_device_to_torch
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
@pytest.mark.parametrize("seq_len", [64])
def test_encoder_text_model(
    mesh_device: ttnn.MeshDevice, tp_axis: int, num_links: int, topology: ttnn.Topology, seq_len: int
) -> None:
    """TT encoder text model vs HF DiffusionGemmaEncoderTextModel."""
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        DiffusionGemmaEncoderTextModel as HFEncoderTextModel,
    )
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaTextConfig

    torch.manual_seed(0)
    dtype = torch.float32

    hf_config = DiffusionGemmaTextConfig(
        vocab_size=1024,
        hidden_size=256,
        intermediate_size=64,
        moe_intermediate_size=64,
        num_attention_heads=8,
        num_key_value_heads=4,
        num_global_key_value_heads=2,
        head_dim=32,
        global_head_dim=32,
        num_experts=8,
        top_k_experts=2,
        num_hidden_layers=6,
        sliding_window=64,
        rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh",
        attention_bias=False,
        attention_dropout=0.0,
        max_position_embeddings=2048,
        pad_token_id=0,
    )
    B = 1

    tp_factor = tuple(mesh_device.shape)[tp_axis]
    if hf_config.num_global_key_value_heads % tp_factor != 0:
        pytest.skip(f"num_global_kv_heads={hf_config.num_global_key_value_heads} != tp_factor={tp_factor}")

    # HF reference encoder.
    hf_encoder = HFEncoderTextModel(hf_config).to(dtype).eval()

    # Inputs.
    input_ids = torch.randint(low=1, high=hf_config.vocab_size, size=(B, seq_len), dtype=torch.long)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        hf_out = hf_encoder(
            input_ids=input_ids,
            attention_mask=None,  # HF auto-builds the sliding+full mask dict from the config.
            position_ids=position_ids,
            past_key_values=None,
        )
        hf_last = hf_out.last_hidden_state  # [B, S, hidden]

    # Re-build the exact same mask dict HF used (so TT consumes identical inputs).
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

    inputs_embeds_for_mask = hf_encoder.embed_tokens(input_ids)
    mask_kwargs = {
        "config": hf_config,
        "inputs_embeds": inputs_embeds_for_mask,
        "attention_mask": None,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    hf_masks = {
        "full_attention": create_causal_mask(**mask_kwargs),
        "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
    }

    # TT setup.
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=1 - tp_axis, factor=tuple(mesh_device.shape)[1 - tp_axis]),
        cfg_parallel=None,
    )

    hf_state = hf_encoder.state_dict()
    moe_state_dicts = per_layer_moe_substates(hf_state, num_layers=hf_config.num_hidden_layers)

    tt_encoder = DiffusionGemmaEncoderTextModel(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        intermediate_size=hf_config.intermediate_size,
        num_hidden_layers=hf_config.num_hidden_layers,
        layer_types=list(hf_config.layer_types),
        num_attention_heads=hf_config.num_attention_heads,
        num_key_value_heads=hf_config.num_key_value_heads,
        num_global_key_value_heads=hf_config.num_global_key_value_heads,
        head_dim=hf_config.head_dim,
        global_head_dim=hf_config.global_head_dim,
        sliding_window=hf_config.sliding_window,
        num_experts=hf_config.num_experts,
        top_k_experts=hf_config.top_k_experts,
        moe_intermediate_size=hf_config.moe_intermediate_size,
        rms_norm_eps=hf_config.rms_norm_eps,
        max_position_embeddings=hf_config.max_position_embeddings,
        sliding_rope_theta=hf_config.rope_parameters["sliding_attention"]["rope_theta"],
        full_rope_theta=hf_config.rope_parameters["full_attention"]["rope_theta"],
        full_partial_rotary_factor=hf_config.rope_parameters["full_attention"]["partial_rotary_factor"],
        pad_token_id=hf_config.pad_token_id,
        moe_state_dicts=moe_state_dicts,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        num_links=num_links,
        topology=topology,
    )
    tt_encoder.load_state_dict(hf_state)

    # Upload inputs / masks.
    tt_input_ids = ttnn.from_torch(input_ids, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_masks = {}
    for layer_type, m in hf_masks.items():
        if m is None:
            tt_masks[layer_type] = None
        else:
            tt_masks[layer_type] = ttnn.from_torch(
                m.to(torch.bfloat16), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
            )

    tt_h, _per_layer_kv = tt_encoder(tt_input_ids, position_ids, tt_masks)
    tt_h_torch = local_device_to_torch(tt_h).squeeze(0)

    logger.info(f"hf_last: {hf_last.shape}, tt_h: {tt_h_torch.shape}")
    assert_quality(hf_last, tt_h_torch, pcc=PCC_THRESHOLD)

    abs_diff = (hf_last - tt_h_torch.to(dtype)).abs()
    logger.info(f"max abs diff: {abs_diff.max().item():.3e}")
    assert torch.allclose(hf_last, tt_h_torch.to(dtype), atol=ALLCLOSE_ATOL, rtol=ALLCLOSE_RTOL)
