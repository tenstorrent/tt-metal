# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Stage-by-stage PCC diagnostic for the Wav2Vec2 audio encoder.

Runs the TT model through stages of the pipeline (feature extractor, projection,
pos-conv, encoder layers 0..11) and compares each stage against the HF golden
to localize where precision is lost.
"""

import pytest
import torch
from loguru import logger
from transformers import Wav2Vec2Model, Wav2Vec2Processor

import ttnn

from ....encoders.wav2vec2 import Wav2Vec2Config, Wav2Vec2Encoder
from ....parallel.config import EncoderParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality
from ....utils.conv3d import conv_pad_in_channels
from ....utils.tensor import from_torch, to_torch
from ....utils.test import line_params


@pytest.mark.parametrize(
    ("mesh_device", "tp_axis", "num_links", "device_params", "topology"),
    [
        pytest.param((4, 8), 0, 2, line_params, ttnn.Topology.Linear, id="bh_4x8_tp0"),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_wav2vec2_stage_pcc(
    mesh_device: ttnn.MeshDevice,
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
) -> None:
    model_id = "facebook/wav2vec2-base-960h"

    hf_model = Wav2Vec2Model.from_pretrained(model_id).eval()
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    config = Wav2Vec2Config.from_hf(hf_model.config)

    torch.manual_seed(0)
    waveform = torch.randn(16000)
    input_values = processor(waveform.numpy(), sampling_rate=16000, return_tensors="pt").input_values

    # ============================================================
    # Stage 1: feature extractor output (after the 7 conv layers).
    # ============================================================
    with torch.no_grad():
        golden_feats_BCT = hf_model.feature_extractor(input_values)  # [B, 512, T_out]
        golden_feats_BTC = golden_feats_BCT.transpose(1, 2).contiguous()
    logger.info(f"\n=== Stage 1: feature extractor ===")
    logger.info(f"Golden features: {tuple(golden_feats_BTC.shape)}")

    tp_factor = tuple(mesh_device.shape)[tp_axis]
    parallel_config = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    tt_model = Wav2Vec2Encoder(
        config=config,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
    )
    tt_model.load_torch_state_dict(hf_model.state_dict())

    # Manually run the feature extractor to compare its output.
    x = input_values.reshape(1, 16000, 1, 1, 1)
    x = conv_pad_in_channels(x)
    tt_audio = from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_feats_5d = tt_model.feature_extractor(tt_audio)  # [B, T_out, 1, 1, 512] tile

    tt_feats_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_feats_5d)[0]).reshape(1, -1, 512)
    logger.info(f"TT features: {tuple(tt_feats_torch.shape)}")
    assert_quality(tt_feats_torch.float(), golden_feats_BTC, pcc=0.99)

    # ============================================================
    # Stage 2: after feature_projection (on device).
    # ============================================================
    with torch.no_grad():
        golden_proj, _ = hf_model.feature_projection(golden_feats_BTC)
    logger.info(f"\n=== Stage 2: feature_projection (on device) ===")
    feats_3d_dev = ttnn.reshape(tt_feats_5d, (1, tt_feats_5d.shape[1], 512))
    proj_dev = tt_model.feature_projection(feats_3d_dev)
    proj_torch = to_torch(proj_dev).float().reshape(1, -1, 768)
    assert_quality(proj_torch, golden_proj, pcc=0.99)

    # ============================================================
    # Stage 3: pos_conv + initial LN (on device).
    # ============================================================
    with torch.no_grad():
        golden_pos = hf_model.encoder.pos_conv_embed(golden_proj)
        golden_pre_ln_in = golden_proj + golden_pos
        golden_pre_ln_out = hf_model.encoder.layer_norm(golden_pre_ln_in)
    logger.info(f"\n=== Stage 3a: pos_conv only (on device; bias-less golden for debug) ===")
    pos_dev = tt_model.encoder.pos_conv_embed(proj_dev)
    pos_torch = to_torch(pos_dev).float().reshape(1, -1, 768)
    # Recompute golden without bias to match the debug-mode TT path.
    with torch.no_grad():
        hf_pos = hf_model.encoder.pos_conv_embed
        # Strip bias temporarily
        orig_bias = hf_pos.conv.bias.detach().clone() if hf_pos.conv.bias is not None else None
        if hf_pos.conv.bias is not None:
            hf_pos.conv.bias.data.zero_()
        golden_pos = hf_pos(golden_proj)
        if orig_bias is not None:
            hf_pos.conv.bias.data.copy_(orig_bias)
    logger.info(f"pos_torch shape: {tuple(pos_torch.shape)}; golden_pos shape: {tuple(golden_pos.shape)}")
    logger.info(f"TT pos[0,0,:8]:     {pos_torch[0, 0, :8].tolist()}")
    logger.info(f"Golden pos[0,0,:8]: {golden_pos[0, 0, :8].tolist()}")
    logger.info(f"TT pos[0,5,:8]:     {pos_torch[0, 5, :8].tolist()}")
    logger.info(f"Golden pos[0,5,:8]: {golden_pos[0, 5, :8].tolist()}")
    logger.info(f"TT pos[0,0,48:56]:    {pos_torch[0, 0, 48:56].tolist()}")
    logger.info(f"Golden pos[0,0,48:56]:{golden_pos[0, 0, 48:56].tolist()}")
    assert_quality(pos_torch, golden_pos, pcc=0.95)

    logger.info(f"\n=== Stage 3b: pos_conv + initial LN (on device) ===")
    hidden_dev = ttnn.add(proj_dev, pos_dev)
    hidden_dev = tt_model.encoder.layer_norm(hidden_dev)
    tt_pre_ln_out = to_torch(hidden_dev).float().reshape(1, -1, 768)
    assert_quality(tt_pre_ln_out, golden_pre_ln_out, pcc=0.95)

    # ============================================================
    # Stage 4: each transformer layer (on device).
    # ============================================================
    logger.info(f"\n=== Stage 4: per-layer transformer outputs ===")
    hidden_torch = golden_pre_ln_out
    for i, tt_layer in enumerate(tt_model.encoder.layers):
        with torch.no_grad():
            hidden_torch = hf_model.encoder.layers[i](hidden_torch)[0]
        hidden_dev = tt_layer(hidden_dev)
        tt_layer_out = to_torch(hidden_dev).float()
        logger.info(f"layer {i}:")
        assert_quality(tt_layer_out, hidden_torch, pcc=0.95)  # loose bar; we log all PCCs
