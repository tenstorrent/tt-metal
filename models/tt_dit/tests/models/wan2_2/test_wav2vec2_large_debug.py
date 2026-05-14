# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Stage-by-stage PCC diagnostic for wav2vec2-large-xlsr-53 (the production
S2V audio encoder). Mirrors `test_wav2vec2_debug.py` but loads weights from the
bundled `Wan-AI/Wan2.2-S2V-14B` snapshot and exercises the pre-LN +
``feat_extract_norm="layer"`` code paths.
"""

from pathlib import Path

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


def _bundled_path() -> Path | None:
    base = Path.home() / ".cache/huggingface/hub/models--Wan-AI--Wan2.2-S2V-14B/snapshots"
    if not base.exists():
        return None
    snaps = list(base.iterdir())
    return (snaps[0] / "wav2vec2-large-xlsr-53-english") if snaps else None


_PATH = _bundled_path()


@pytest.mark.skipif(_PATH is None, reason="bundled wav2vec2-large not found")
@pytest.mark.parametrize(
    ("mesh_device", "tp_axis", "num_links", "device_params", "topology"),
    [
        pytest.param((4, 8), 0, 2, line_params, ttnn.Topology.Linear, id="bh_4x8_tp0"),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_wav2vec2_large_stage_pcc(
    mesh_device: ttnn.MeshDevice,
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
) -> None:
    hf_model = Wav2Vec2Model.from_pretrained(str(_PATH)).eval()
    processor = Wav2Vec2Processor.from_pretrained(str(_PATH))
    config = Wav2Vec2Config.from_hf(hf_model.config)
    assert config.do_stable_layer_norm
    assert config.feat_extract_norm == "layer"

    torch.manual_seed(0)
    waveform = torch.randn(16000)
    input_values = processor(waveform.numpy(), sampling_rate=16000, return_tensors="pt").input_values

    # -----------------------------------------------------------------
    # Stage 1: feature extractor (7 conv1d layers, "layer" norm mode).
    # -----------------------------------------------------------------
    with torch.no_grad():
        golden_feats_BCT = hf_model.feature_extractor(input_values)
        golden_feats_BTC = golden_feats_BCT.transpose(1, 2).contiguous()
    logger.info(f"\n=== Stage 1: feature extractor ===")
    logger.info(f"Golden features shape: {tuple(golden_feats_BTC.shape)}")

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
    tt_model.bind_cpu_modules(hf_model)

    x = input_values.reshape(1, 16000, 1, 1, 1)
    x = conv_pad_in_channels(x)
    # Use the feature extractor's dtype (fp32 for "layer" variant) — the
    # conv3d kernel requires input dtype to match weight dtype.
    tt_audio = from_torch(x, device=mesh_device, dtype=tt_model.feature_extractor.dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
    # Per-layer breakdown of the feature extractor.
    tt_layers = list(tt_model.feature_extractor.conv_layers)
    hf_layers = list(hf_model.feature_extractor.conv_layers)

    tt_h = tt_audio  # [B, T, 1, 1, 1]
    hf_h = input_values  # [B, T] for HF first conv (Conv1d wants [B, C, T])
    hf_h = hf_h.unsqueeze(1)  # [B, 1, T]
    for i, (tt_layer, hf_layer) in enumerate(zip(tt_layers, hf_layers)):
        # HF stage: feed [B, C_in, T], get [B, C_out, T'].
        with torch.no_grad():
            hf_h = hf_layer(hf_h)

        # TT stage: feed [B, T, 1, 1, C_in], get [B, T', 1, 1, C_out].
        tt_h = tt_layer(tt_h)
        # The next layer expects ROW_MAJOR input.
        tt_h_rm = ttnn.to_layout(tt_h, ttnn.ROW_MAJOR_LAYOUT)
        tt_out_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_h_rm)[0])
        # tt_out shape: [B, T', 1, 1, C_out]; HF shape: [B, C_out, T'].
        tt_out_BTC = tt_out_torch.reshape(1, tt_out_torch.shape[1], -1).float()
        hf_out_BTC = hf_h.transpose(1, 2).contiguous().float()
        logger.info(f"  conv layer {i}: TT {tuple(tt_out_BTC.shape)}, HF {tuple(hf_out_BTC.shape)}")
        assert_quality(tt_out_BTC, hf_out_BTC, pcc=0.90)
        # Feed forward as ROW_MAJOR to next conv.
        tt_h = tt_h_rm

    tt_feats_5d = ttnn.to_layout(tt_h, ttnn.TILE_LAYOUT)
    tt_feats_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_feats_5d)[0]).reshape(1, -1, 512)
    assert_quality(tt_feats_torch.float(), golden_feats_BTC, pcc=0.99)

    # -----------------------------------------------------------------
    # Stage 2: feature projection (on device).
    # -----------------------------------------------------------------
    with torch.no_grad():
        golden_proj, _ = hf_model.feature_projection(golden_feats_BTC)
    logger.info(f"\n=== Stage 2: feature_projection ===")
    feats_3d = ttnn.reshape(tt_feats_5d, (1, tt_feats_5d.shape[1], 512))
    # feature_projection has bf16 weights; cast the fp32 feature-extractor
    # output to bf16 to match (this is what `Wav2Vec2Encoder.forward` does).
    if feats_3d.dtype != ttnn.bfloat16:
        feats_3d = ttnn.typecast(feats_3d, ttnn.bfloat16)
    proj_dev = tt_model.feature_projection(feats_3d)
    proj_torch = to_torch(proj_dev).float().reshape(1, -1, 1024)
    assert_quality(proj_torch, golden_proj, pcc=0.99)

    # -----------------------------------------------------------------
    # Stage 3a: pos_conv only (CPU shadow).
    # -----------------------------------------------------------------
    with torch.no_grad():
        golden_pos = hf_model.encoder.pos_conv_embed(golden_proj)
    logger.info(f"\n=== Stage 3a: pos_conv (CPU) ===")
    with torch.no_grad():
        tt_pos = tt_model._cpu_pos_conv_embed(proj_torch)
    assert_quality(tt_pos, golden_pos, pcc=0.99)

    # -----------------------------------------------------------------
    # Stage 3b: pre-encoder summing (NO initial LN in stable mode).
    # -----------------------------------------------------------------
    logger.info(f"\n=== Stage 3b: input-to-encoder (proj + pos) ===")
    pre_stack_golden = golden_proj + golden_pos  # NO initial LN in stable
    pre_stack_tt = proj_torch + tt_pos
    assert_quality(pre_stack_tt, pre_stack_golden, pcc=0.99)

    # -----------------------------------------------------------------
    # Stage 4: each transformer layer (cumulative on device).
    # -----------------------------------------------------------------
    logger.info(f"\n=== Stage 4: per-layer transformer (pre-LN) ===")
    hidden_dev = from_torch(pre_stack_tt, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    hidden_torch = pre_stack_golden
    for i, tt_layer in enumerate(tt_model.encoder.layers):
        with torch.no_grad():
            hidden_torch = hf_model.encoder.layers[i](hidden_torch)[0]
        hidden_dev = tt_layer(hidden_dev)
        tt_layer_out = to_torch(hidden_dev).float()
        logger.info(f"  layer {i:02d}:")
        assert_quality(tt_layer_out, hidden_torch, pcc=0.90)  # loose; we log all PCCs

    # -----------------------------------------------------------------
    # Stage 5: final encoder layer_norm (on device, stable mode).
    # -----------------------------------------------------------------
    logger.info(f"\n=== Stage 5: final encoder.layer_norm ===")
    with torch.no_grad():
        golden_final = hf_model.encoder.layer_norm(hidden_torch)
    final_dev = tt_model.encoder.layer_norm(hidden_dev)
    final_torch = to_torch(final_dev).float()
    assert_quality(final_torch, golden_final, pcc=0.90)
