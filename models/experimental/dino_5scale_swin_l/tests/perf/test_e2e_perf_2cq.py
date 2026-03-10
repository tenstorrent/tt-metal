# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
E2E performance test for DINO-5scale Swin-L with 2CQ pipeline.
"""

import os
import pytest
import torch
import ttnn
from pathlib import Path

from loguru import logger
from models.common.utility_functions import profiler, run_for_wormhole_b0
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.experimental.dino_5scale_swin_l.common import (
    DINO_INPUT_H,
    DINO_INPUT_W,
    SWIN_L_EMBED_DIM,
    SWIN_L_DEPTHS,
    SWIN_L_NUM_HEADS,
    SWIN_L_WINDOW_SIZE,
    NUM_QUERIES,
    NUM_CLASSES,
    NUM_LEVELS,
    ENCODER_EMBED_DIMS,
    ENCODER_NUM_HEADS,
    ENCODER_NUM_POINTS,
    ENCODER_NUM_LAYERS,
    DECODER_NUM_LAYERS,
)
from models.perf.perf_utils import prep_perf_report
from models.tt_cnn.tt.pipeline import (
    PipelineConfig,
    create_pipeline_from_config,
    get_memory_config_for_persistent_dram_tensor,
)

PAD_CHANNELS = 16
NUM_MEASUREMENT_ITERS = 4


def _get_ckpt_path():
    base = Path(os.environ.get("TT_METAL_HOME", Path.cwd()))
    ckpt_dir = base / "models/experimental/dino_5scale_swin_l/checkpoints/dino_5scale_swin_l"
    for name in ("dino_5scale_swin_l.pth", "dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth"):
        path = ckpt_dir / name
        if path.is_file():
            return str(path)
    return str(ckpt_dir / "dino_5scale_swin_l.pth")


def _setup_sharded_input(device):
    inputs_mesh_mapper, _, _ = get_mesh_mappers(device)
    batch, height, width, channels = 1, DINO_INPUT_H, DINO_INPUT_W, 3
    hw = batch * height * width
    core_grid = ttnn.CoreGrid(x=8, y=4)
    num_cores = core_grid.num_cores
    rows_per_core = (hw + num_cores - 1) // num_cores
    height_dim_aligned = rows_per_core * num_cores
    pad_c = max(0, PAD_CHANNELS - channels)
    torch_input = torch.randn(1, height, width, channels, dtype=torch.bfloat16)
    if pad_c:
        torch_input = torch.nn.functional.pad(torch_input, (0, pad_c), value=0)
    flat = torch_input.reshape(1, hw, PAD_CHANNELS)
    padded = torch.nn.functional.pad(flat, (0, 0, 0, height_dim_aligned - hw), value=0)
    tt_inputs_host = ttnn.from_torch(
        padded.reshape(1, 1, height_dim_aligned, PAD_CHANNELS), dtype=ttnn.bfloat16, mesh_mapper=inputs_mesh_mapper
    )
    dram_config = get_memory_config_for_persistent_dram_tensor(
        tt_inputs_host.shape, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, device.dram_grid_size()
    )
    return tt_inputs_host, dram_config, dram_config, (batch, height, width, channels), hw


def _build_model_and_inputs(device):
    from models.experimental.swin_l.tt.model_preprocessing import load_backbone_weights, compute_attn_masks
    from models.experimental.dino_5scale_swin_l.tt.model_preprocessing import (
        load_neck_weights,
        load_encoder_weights,
        load_decoder_weights,
        _resolve_state_dict,
    )
    from models.experimental.dino_5scale_swin_l.tt.tt_dino import TtDINO

    ckpt_path = _get_ckpt_path()
    if not Path(ckpt_path).is_file():
        return None, None, None, None, None, None
    sd = _resolve_state_dict(ckpt_path)
    backbone_params = load_backbone_weights(
        sd,
        device,
        embed_dim=SWIN_L_EMBED_DIM,
        depths=tuple(SWIN_L_DEPTHS),
        num_heads=tuple(SWIN_L_NUM_HEADS),
        window_size=SWIN_L_WINDOW_SIZE,
    )
    neck_params = load_neck_weights(sd, device)
    encoder_params = load_encoder_weights(sd, device)
    decoder_params = load_decoder_weights(sd, device)
    attn_masks = compute_attn_masks(DINO_INPUT_H, DINO_INPUT_W, 4, SWIN_L_WINDOW_SIZE, device)
    del sd
    tt_model = TtDINO(
        encoder_params=encoder_params,
        decoder_params=decoder_params,
        device=device,
        backbone_params=backbone_params,
        neck_params=neck_params,
        attn_masks=attn_masks,
        num_queries=NUM_QUERIES,
        num_classes=NUM_CLASSES,
        num_levels=NUM_LEVELS,
        embed_dims=ENCODER_EMBED_DIMS,
        num_heads=ENCODER_NUM_HEADS,
        num_points=ENCODER_NUM_POINTS,
        encoder_num_layers=ENCODER_NUM_LAYERS,
        decoder_num_layers=DECODER_NUM_LAYERS,
        pe_temperature=20,
        embed_dim=SWIN_L_EMBED_DIM,
        depths=tuple(SWIN_L_DEPTHS),
        backbone_num_heads=tuple(SWIN_L_NUM_HEADS),
        window_size=SWIN_L_WINDOW_SIZE,
        in_channels=(192, 384, 768, 1536),
    )
    tt_inputs_host, dram_config, l1_config, input_dims, hw = _setup_sharded_input(device)
    return tt_model, tt_inputs_host, dram_config, l1_config, input_dims, hw


def _run_pipeline(device, tt_model, tt_inputs_host, dram_config, l1_config, input_dims, hw, num_iters):
    batch, height, width, channels = input_dims
    dram_cfg = ttnn.DRAM_MEMORY_CONFIG

    def model_wrapper(device_input):
        host = ttnn.to_torch(ttnn.from_device(device_input))
        host = host[:, :, :hw, :channels].reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        nchw = ttnn.from_torch(
            host.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=dram_cfg,
        )
        backbone_feats_tt = tt_model.backbone(nchw)
        neck_feats_tt = tt_model.neck(backbone_feats_tt)
        for bf in backbone_feats_tt:
            ttnn.deallocate(bf)
        pre_trans = tt_model.pre_transformer_tt(neck_feats_tt)
        for nf in neck_feats_tt:
            ttnn.deallocate(nf)
        feat_tt = pre_trans["feat_flatten"]
        feat_pos_tt = pre_trans["feat_pos"]
        memory_tt = tt_model.encoder(
            feat=feat_tt,
            feat_pos=feat_pos_tt,
            feat_mask=None,
            spatial_shapes=pre_trans["spatial_shapes"],
            level_start_index=pre_trans["level_start_index"],
            valid_ratios=pre_trans["valid_ratios"],
        )
        ttnn.deallocate(feat_tt)
        ttnn.deallocate(feat_pos_tt)
        pre_dec = tt_model.pre_decoder_ttnn(memory_tt, pre_trans["spatial_shapes"])
        hidden_states, references = tt_model.decoder(
            query=pre_dec["query"],
            value=memory_tt,
            key_padding_mask=None,
            self_attn_mask=None,
            reference_points=pre_dec["reference_points"],
            spatial_shapes=pre_trans["spatial_shapes"],
            level_start_index=pre_trans["level_start_index"],
            valid_ratios=pre_trans["valid_ratios"],
        )
        all_cls, all_coords = tt_model.forward_heads(hidden_states, references)
        cls_tt = ttnn.from_torch(
            all_cls.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=dram_cfg,
        )
        coords_tt = ttnn.from_torch(
            all_coords.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=dram_cfg,
        )
        return (cls_tt, coords_tt)

    pipeline = create_pipeline_from_config(
        config=PipelineConfig(
            use_trace=False,
            num_command_queues=2,
            all_transfers_on_separate_command_queue=False,
        ),
        model=model_wrapper,
        device=device,
        dram_input_memory_config=dram_config,
        l1_input_memory_config=l1_config,
    )
    ttnn.synchronize_device(device)
    profiler.start("compile")
    pipeline.compile(tt_inputs_host)
    profiler.end("compile")
    host_inputs = [tt_inputs_host] * num_iters
    pipeline.preallocate_output_tensors_on_host(num_iters)
    profiler.start("run_model_pipeline_2cqs")
    pipeline.enqueue(host_inputs).pop_all()
    profiler.end("run_model_pipeline_2cqs")
    pipeline.cleanup()


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32768, "trace_region_size": 50000000, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize("batch_size_per_device", (1,))
@pytest.mark.parametrize("expected_inference_throughput", (0.04,))
def test_dino_5scale_swin_l_e2e_perf_2cq_trace(device, batch_size_per_device, expected_inference_throughput):
    tt_model, tt_inputs_host, dram_config, l1_config, input_dims, hw = _build_model_and_inputs(device)
    if tt_model is None:
        pytest.skip("Checkpoint not found")
    profiler.clear()
    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices
    _run_pipeline(device, tt_model, tt_inputs_host, dram_config, l1_config, input_dims, hw, NUM_MEASUREMENT_ITERS)
    compile_time = profiler.get("compile")
    inference_time_avg = profiler.get("run_model_pipeline_2cqs") / NUM_MEASUREMENT_ITERS
    expected_inference_time = batch_size / expected_inference_throughput
    prep_perf_report(
        model_name=f"ttnn_dino_5scale_swin_l_trace_2cqs_batch_size{batch_size}",
        batch_size=batch_size,
        inference_and_compile_time=compile_time,
        inference_time=inference_time_avg,
        expected_compile_time=600,
        expected_inference_time=expected_inference_time,
        comments=f"{DINO_INPUT_H}x{DINO_INPUT_W}_batchsize{batch_size}",
        inference_time_cpu=0.0,
    )
    logger.info(
        f"DINO-5scale Swin-L {DINO_INPUT_H}x{DINO_INPUT_W} batch_size={batch_size} "
        f"compile={compile_time:.2f}s inference_avg={inference_time_avg:.4f}s FPS={batch_size/inference_time_avg:.2f}"
    )
