# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
#
# SPDX-License-Identifier: Apache-2.0

import time
import pytest
import torch
import ttnn
import numpy as np
from loguru import logger

from models.perf.perf_utils import prep_perf_report
from models.tt_cnn.tt.pipeline import (
    PipelineConfig,
    create_pipeline_from_config,
)
from models.common.utility_functions import run_for_wormhole_b0
from models.experimental.BEVFormerV2.reference import bevformer_v2
from models.experimental.BEVFormerV2.tt.ttnn_bevformer_v2 import TtBevFormerV2
from models.experimental.BEVFormerV2.tt.model_preprocessing import (
    create_bevformerv2_model_parameters,
)
from models.experimental.BEVFormerV2.common import load_torch_model


def create_bevformerv2_pipeline_model(ttnn_model, torch_input, img_metas, dtype=ttnn.bfloat16):
    def run(dummy_input):
        ttnn_input = ttnn.from_torch(
            torch_input,
            device=ttnn_model.device,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        unwrapped_img_metas = img_metas
        while (
            isinstance(unwrapped_img_metas, list)
            and len(unwrapped_img_metas) > 0
            and isinstance(unwrapped_img_metas[0], list)
        ):
            unwrapped_img_metas = unwrapped_img_metas[0]

        if "can_bus" not in unwrapped_img_metas[0]:
            unwrapped_img_metas[0]["can_bus"] = [0.0] * 18

        img_feats = ttnn_model.extract_feat(img=ttnn_input, img_metas=unwrapped_img_metas)

        x = img_feats
        x0_old = x[0]
        x[0] = ttnn.to_layout(x[0], layout=ttnn.TILE_LAYOUT)
        if x0_old.is_allocated() and x0_old is not x[0]:
            ttnn.deallocate(x0_old)

        outs = ttnn_model.pts_bbox_head(x, unwrapped_img_metas, prev_bev=None)

        if ttnn_input.is_allocated():
            ttnn.deallocate(ttnn_input)
        for feat in img_feats:
            if feat.is_allocated():
                ttnn.deallocate(feat)

        bev_embed = outs["bev_embed"]
        all_cls_scores = outs["all_cls_scores"]
        all_bbox_preds = outs["all_bbox_preds"]

        if isinstance(bev_embed, torch.Tensor):
            bev_embed = ttnn.from_torch(bev_embed, device=ttnn_model.device, dtype=dtype)
        if isinstance(all_cls_scores, torch.Tensor):
            all_cls_scores = ttnn.from_torch(all_cls_scores, device=ttnn_model.device, dtype=dtype)
        if isinstance(all_bbox_preds, torch.Tensor):
            all_bbox_preds = ttnn.from_torch(all_bbox_preds, device=ttnn_model.device, dtype=dtype)

        if bev_embed.layout != ttnn.ROW_MAJOR_LAYOUT:
            bev_embed = ttnn.to_layout(bev_embed, ttnn.ROW_MAJOR_LAYOUT)
        if all_cls_scores.layout != ttnn.ROW_MAJOR_LAYOUT:
            all_cls_scores = ttnn.to_layout(all_cls_scores, ttnn.ROW_MAJOR_LAYOUT)
        if all_bbox_preds.layout != ttnn.ROW_MAJOR_LAYOUT:
            all_bbox_preds = ttnn.to_layout(all_bbox_preds, ttnn.ROW_MAJOR_LAYOUT)

        bev_embed = ttnn.to_memory_config(bev_embed, ttnn.DRAM_MEMORY_CONFIG)
        all_cls_scores = ttnn.to_memory_config(all_cls_scores, ttnn.DRAM_MEMORY_CONFIG)
        all_bbox_preds = ttnn.to_memory_config(all_bbox_preds, ttnn.DRAM_MEMORY_CONFIG)

        return (bev_embed, all_cls_scores, all_bbox_preds)

    return run


def create_img_metas():
    return [
        [
            [
                {
                    "filename": [
                        "./data/nuscenes/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg",
                        "./data/nuscenes/samples/CAM_FRONT_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_RIGHT__1533151603520482.jpg",
                        "./data/nuscenes/samples/CAM_FRONT_LEFT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_LEFT__1533151603504799.jpg",
                        "./data/nuscenes/samples/CAM_BACK/n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151603537558.jpg",
                        "./data/nuscenes/samples/CAM_BACK_LEFT/n008-2018-08-01-15-16-36-0400__CAM_BACK_LEFT__1533151603547405.jpg",
                        "./data/nuscenes/samples/CAM_BACK_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_BACK_RIGHT__1533151603528113.jpg",
                    ],
                    "ori_shape": [(360, 640, 3)] * 6,
                    "img_shape": [(256, 704, 3)] * 6,
                    "lidar2img": [
                        np.array(
                            [
                                [4.97195909e02, 3.36259809e02, 1.31050214e01, -1.41740456e02],
                                [-7.28050437e00, 2.14719425e02, -4.90215017e02, -2.57883151e02],
                                [-1.17025046e-02, 9.98471159e-01, 5.40221896e-02, -4.25203639e-01],
                                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                            ]
                        ),
                        np.array(
                            [
                                [5.45978616e02, -2.47705944e02, -1.61356657e01, -1.84657143e02],
                                [1.51784935e02, 1.28122911e02, -4.95917894e02, -2.77022512e02],
                                [8.43406855e-01, 5.36312055e-01, 3.21598489e-02, -6.10371854e-01],
                                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                            ]
                        ),
                        np.array(
                            [
                                [1.29479337e01, 6.01261709e02, 3.10492731e01, -1.20975154e02],
                                [-1.55728079e02, 1.28176621e02, -4.94981202e02, -2.71769902e02],
                                [-8.23415292e-01, 5.65940098e-01, 4.12196894e-02, -5.29677094e-01],
                                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                            ]
                        ),
                        np.array(
                            [
                                [-3.21592898e02, -3.40289545e02, -1.05750653e01, -3.48318395e02],
                                [-4.32931264e00, -1.78114385e02, -3.25958977e02, -2.83473696e02],
                                [-8.33350064e-03, -9.99200442e-01, -3.91028008e-02, -1.01645350e00],
                                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                            ]
                        ),
                        np.array(
                            [
                                [-4.74626444e02, 3.69304577e02, 2.13056637e01, -2.50136476e02],
                                [-1.85050206e02, -4.10162348e01, -5.00990867e02, -2.24731382e02],
                                [-9.47586752e-01, -3.19482867e-01, 3.16948959e-03, -4.32527296e-01],
                                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                            ]
                        ),
                        np.array(
                            [
                                [1.14075693e02, -5.87710608e02, -2.38253717e01, -1.09040128e02],
                                [1.77894417e02, -4.91302807e01, -5.00157067e02, -2.35298447e02],
                                [9.24052925e-01, -3.82246554e-01, -3.70989150e-03, -4.64645142e-01],
                                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                            ]
                        ),
                    ],
                    "pad_shape": [(256, 704, 3)] * 6,
                    "scale_factor": 1.0,
                    "flip": False,
                    "pcd_horizontal_flip": False,
                    "pcd_vertical_flip": False,
                    "sample_idx": "3e8750f331d7499e9b5123e9eb70f2e2",
                    "prev_idx": "",
                    "next_idx": "3950bd41f74548429c0f7700ff3d8269",
                    "pcd_scale_factor": 1.0,
                    "scene_token": "fcbccedd61424f1b85dcbf8f897f9754",
                    "can_bus": np.array(
                        [
                            6.50486842e02,
                            1.81754303e03,
                            0.00000000e00,
                            1.84843146e-01,
                            1.84843146e-01,
                            1.84843146e-01,
                            1.84843146e-01,
                            8.47522666e-01,
                            1.34135536e00,
                            9.58588434e00,
                            -9.57939215e-03,
                            6.51179999e-03,
                            3.75314295e-01,
                            3.77446848e00,
                            0.00000000e00,
                            0.00000000e00,
                            3.51370076e00,
                            2.01320224e02,
                        ]
                    ),
                }
            ]
        ]
    ]


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": 20 * 1024,
            "trace_region_size": 10000000,
            "num_command_queues": 1,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("num_iterations", [1])
@pytest.mark.parametrize(
    "batch_size, expected_compile_time, expected_throughput_fps",
    [(1, 100.0, 0.105)],  # Placeholder values - adjust based on actual performance
)
@pytest.mark.models_performance_bare_metal
def test_bevformerv2_e2e_performant_1cq_notrace(
    device,
    num_iterations,
    batch_size,
    expected_compile_time,
    expected_throughput_fps,
    reset_seeds,
    model_location_generator,
):
    torch.manual_seed(0)

    dtype = ttnn.bfloat16

    logger.info("Building BEVFormerV2 model...")
    torch_model = bevformer_v2.BEVFormerV2(
        use_grid_mask=True,
        img_backbone=dict(depth=50, in_channels=3, out_indices=(1, 2, 3), style="caffe"),
        img_neck=dict(in_channels=[512, 1024, 2048], out_channels=256, num_outs=5),
        pts_bbox_head=dict(bev_h=100, bev_w=100, num_query=900, num_classes=10, in_channels=256),
        video_test_mode=True,
    )

    torch_model = load_torch_model(torch_model=torch_model, model_location_generator=model_location_generator)

    encoder_layers = 6
    decoder_layers = 6
    logger.info(f"Using {encoder_layers} encoder layers and {decoder_layers} decoder layers")
    torch_model.pts_bbox_head.transformer.encoder.layers = torch.nn.ModuleList(
        list(torch_model.pts_bbox_head.transformer.encoder.layers)[:encoder_layers]
    )
    torch_model.pts_bbox_head.transformer.encoder.num_layers = encoder_layers
    torch_model.pts_bbox_head.transformer.decoder.layers = torch.nn.ModuleList(
        list(torch_model.pts_bbox_head.transformer.decoder.layers)[:decoder_layers]
    )
    torch_model.pts_bbox_head.transformer.decoder.num_layers = decoder_layers

    # Create input tensor
    input_shape = (batch_size, 6, 3, 256, 704)  # (B, N, C, H, W)
    sample_input = torch.randn(input_shape, dtype=torch.float32)

    # Create img_metas
    img_metas = create_img_metas()

    # Create model parameters
    parameter = create_bevformerv2_model_parameters(
        torch_model,
        [
            False,
            [sample_input],
            img_metas,
        ],
        device,
    )

    # Initialize TTNN model
    tt_model = TtBevFormerV2(
        device=device,
        params=parameter,
        use_grid_mask=False,
        img_backbone=dict(depth=50, in_channels=3, out_indices=(1, 2, 3), style="caffe"),
        img_neck=dict(in_channels=[512, 1024, 2048], out_channels=256, num_outs=5),
        pts_bbox_head=dict(
            bev_h=100,
            bev_w=100,
            num_query=900,
            num_classes=10,
            in_channels=256,
            encoder_num_layers=torch_model.pts_bbox_head.transformer.encoder.num_layers,
            decoder_num_layers=torch_model.pts_bbox_head.transformer.decoder.num_layers,
        ),
        video_test_mode=False,
    )

    ttnn.synchronize_device(device)

    logger.info("Creating pipeline model...")
    pipeline_model = create_bevformerv2_pipeline_model(tt_model, sample_input, img_metas, dtype=dtype)

    logger.info("Preparing dummy input tensor...")
    dummy_input = ttnn.from_torch(
        torch.zeros(1, 1, 1, 32),
        device=None,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    logger.info("Configuring pipeline (1CQ + No Trace)...")
    pipeline = create_pipeline_from_config(
        config=PipelineConfig(use_trace=False, num_command_queues=1, all_transfers_on_separate_command_queue=False),
        model=pipeline_model,
        device=device,
        dram_input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        l1_input_memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    input_tensors = [dummy_input] * num_iterations

    logger.info("Compiling pipeline (warmup)...")
    start = time.time()
    pipeline.compile(dummy_input)
    end = time.time()

    compile_time = end - start

    pipeline.preallocate_output_tensors_on_host(num_iterations)

    logger.info(f"Running {num_iterations} inference iterations...")
    start = time.time()
    pipeline.enqueue(input_tensors).pop_all()
    end = time.time()

    pipeline.cleanup()

    inference_time = (end - start) / num_iterations
    logger.info(f"Average model time={1000.0 * inference_time : .2f} ms")
    logger.info(f"Average model performance={num_iterations * batch_size / (end-start) : .2f} fps")

    total_num_samples = batch_size
    prep_perf_report(
        model_name="bevformerv2-notrace-1cq",
        batch_size=total_num_samples,
        inference_and_compile_time=compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=total_num_samples / expected_throughput_fps,
        comments=f"batch_{batch_size}",
    )

    logger.info("Performance test completed!")
