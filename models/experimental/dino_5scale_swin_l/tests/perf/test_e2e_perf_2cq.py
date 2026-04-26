# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
E2E performance test for DINO-5scale Swin-L with 2CQ pipeline.
"""

import os
from pathlib import Path

import pytest
import torch
import ttnn
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

_CHECKPOINT_CANDIDATES = (
    "dino_5scale_swin_l.pth",
    "dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth",
)
NUM_MEASUREMENT_ITERS = 4


def _get_ckpt_path():
    base = Path(os.environ.get("TT_METAL_HOME", Path.cwd()))
    ckpt_dir = base / "models/experimental/dino_5scale_swin_l/checkpoints/dino_5scale_swin_l"
    for name in _CHECKPOINT_CANDIDATES:
        path = ckpt_dir / name
        if path.is_file():
            return str(path)
    return str(ckpt_dir / "dino_5scale_swin_l.pth")


def _get_l1_input_memory_config(host_input):
    """Builds height-sharded L1 memory config for pipeline input (matches Swin test)."""
    height, width = host_input.shape[-2], host_input.shape[-1]
    core_grid = ttnn.CoreGrid(x=8, y=1)
    if height % core_grid.num_cores != 0:
        core_grid = ttnn.CoreGrid(x=4, y=1)
    return ttnn.create_sharded_memory_config(
        shape=(height // core_grid.num_cores, width),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _setup_sharded_input(device):
    """Setup input in Swin-compatible layout (1, 1, batch*3*H, padded_W) for reshape-only preprocessing."""
    inputs_mesh_mapper, _, _ = get_mesh_mappers(device)
    batch, height, width, channels = 1, DINO_INPUT_H, DINO_INPUT_W, 3
    padded_width = ((width + 31) // 32) * 32
    torch_input = torch.randn(batch, channels, height, padded_width, dtype=torch.bfloat16)
    tt_inputs_host = ttnn.from_torch(
        torch_input.reshape(1, 1, batch * channels * height, padded_width),
        dtype=ttnn.bfloat16,
        mesh_mapper=inputs_mesh_mapper,
    )
    dram_config = get_memory_config_for_persistent_dram_tensor(
        tt_inputs_host.shape, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, device.dram_grid_size()
    )
    l1_config = _get_l1_input_memory_config(tt_inputs_host)
    return tt_inputs_host, dram_config, l1_config, (batch, height, width, channels)


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
        return None, None, None, None, None
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
        trace_mode=False,
    )
    tt_inputs_host, dram_config, l1_config, input_dims = _setup_sharded_input(device)
    return tt_model, tt_inputs_host, dram_config, l1_config, input_dims


def _run_pipeline(device, tt_model, tt_inputs_host, dram_config, l1_config, input_dims, num_iters):
    batch, height, width, channels = input_dims
    dram_cfg = ttnn.DRAM_MEMORY_CONFIG
    padded_width = ((width + 31) // 32) * 32

    def model_wrapper(device_input):
        # Swin-compatible: reshape only (no slice) for trace capture.
        reshaped = ttnn.reshape(device_input, (batch, channels, height, padded_width))
        nchw = ttnn.to_memory_config(reshaped, dram_cfg)
        if reshaped is not nchw:
            ttnn.deallocate(reshaped)
        if padded_width != width:
            nchw_sliced = ttnn.slice(nchw, [0, 0, 0, 0], [batch, channels, height, width], memory_config=dram_cfg)
            ttnn.deallocate(nchw)
            nchw = nchw_sliced
        backbone_feats_tt = tt_model.backbone(nchw)
        neck_feats_tt = tt_model.neck(backbone_feats_tt)
        for bf in backbone_feats_tt:
            ttnn.deallocate(bf)
        pre_trans = tt_model.pre_transformer_tt(neck_feats_tt)
        for nf in neck_feats_tt:
            ttnn.deallocate(nf)
        feat_tt = pre_trans["feat_flatten"]
        feat_pos_tt = pre_trans["feat_pos"]
        enc_kw = dict(
            feat=feat_tt,
            feat_pos=feat_pos_tt,
            feat_mask=None,
            spatial_shapes=pre_trans["spatial_shapes"],
            level_start_index=pre_trans["level_start_index"],
            valid_ratios=pre_trans["valid_ratios"],
        )
        if "valid_ratios_tt" in pre_trans:
            enc_kw["valid_ratios_tt"] = pre_trans["valid_ratios_tt"]
        if "spatial_shapes_tt" in pre_trans:
            enc_kw["spatial_shapes_tt"] = pre_trans["spatial_shapes_tt"]
        memory_tt = tt_model.encoder(**enc_kw)
        ttnn.deallocate(feat_tt)
        ttnn.deallocate(feat_pos_tt)
        pre_dec = tt_model.pre_decoder_ttnn(memory_tt, pre_trans["spatial_shapes"])
        dec_kw = dict(
            query=pre_dec["query"],
            value=memory_tt,
            key_padding_mask=None,
            self_attn_mask=None,
            reference_points=pre_dec["reference_points"],
            spatial_shapes=pre_trans["spatial_shapes"],
            level_start_index=pre_trans["level_start_index"],
            valid_ratios=pre_trans["valid_ratios"],
        )
        if "valid_ratios_tt" in pre_trans:
            dec_kw["valid_ratios_tt"] = pre_trans["valid_ratios_tt"]
        if "spatial_shapes_tt" in pre_trans:
            dec_kw["spatial_shapes_tt"] = pre_trans["spatial_shapes_tt"]
        hidden_states, references = tt_model.decoder(**dec_kw)
        all_cls, all_coords = tt_model.forward_heads(hidden_states, references, return_device_tensors=True)
        cls_tt = ttnn.to_memory_config(all_cls, dram_cfg)
        coords_tt = ttnn.to_memory_config(all_coords, dram_cfg)
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
    profiler.start("run_model_pipeline_2cqs")
    pipeline.enqueue(host_inputs).pop_all()
    profiler.end("run_model_pipeline_2cqs")
    pipeline.cleanup()


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32768, "trace_region_size": 350000000, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize("batch_size_per_device", (1,))
@pytest.mark.parametrize("expected_inference_throughput", (0.04,))
def test_dino_5scale_swin_l_e2e_perf_2cq(device, batch_size_per_device, expected_inference_throughput):
    """Measures compile time and inference throughput for DINO-5scale Swin-L with 2CQ pipeline (non-traced)."""
    tt_model, tt_inputs_host, dram_config, l1_config, input_dims = _build_model_and_inputs(device)
    if tt_model is None:
        pytest.skip("Checkpoint not found")
    profiler.clear()
    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices
    _run_pipeline(device, tt_model, tt_inputs_host, dram_config, l1_config, input_dims, NUM_MEASUREMENT_ITERS)
    compile_time = profiler.get("compile")
    inference_time_avg = profiler.get("run_model_pipeline_2cqs") / NUM_MEASUREMENT_ITERS
    expected_inference_time = batch_size / expected_inference_throughput
    prep_perf_report(
        model_name=f"ttnn_dino_5scale_swin_l_2cqs_batch_size{batch_size}",
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
