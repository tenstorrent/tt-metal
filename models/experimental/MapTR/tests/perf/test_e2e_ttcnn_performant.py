# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Performance test for MapTR using TT-CNN Pipeline API for optimized execution.
"""

import time
import traceback

import numpy as np
import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc, run_for_wormhole_b0
from models.experimental.MapTR.reference.maptr import MapTR
from models.experimental.MapTR.tt.ttnn_maptr import TtMapTR
from models.experimental.MapTR.tt.model_preprocessing import (
    create_maptr_model_parameters,
    load_maptr_weights,
)
from models.perf.perf_utils import prep_perf_report
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config


MAPTR_WEIGHTS_PATH = "models/experimental/MapTR/chkpt/maptr_tiny_r50_24e_bevformer.pth"


class ConfigDict(dict):
    def __getattr__(self, name):
        try:
            value = self[name]
            if isinstance(value, dict) and not isinstance(value, ConfigDict):
                value = ConfigDict(value)
                self[name] = value
            return value
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


def create_maptr_config():
    embed_dims = 256
    num_classes = 3
    num_vec = 50
    num_pts_per_vec = 20
    num_decoder_layers = 6
    pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
    bev_h, bev_w = 200, 100

    img_backbone_cfg = ConfigDict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=ConfigDict(type="BN", requires_grad=False),
        norm_eval=True,
        style="pytorch",
    )

    img_neck_cfg = ConfigDict(
        type="FPN",
        in_channels=[2048],
        out_channels=embed_dims,
        start_level=0,
        add_extra_convs="on_output",
        num_outs=1,
        relu_before_extra_convs=True,
    )

    transformer_cfg = ConfigDict(
        type="MapTRPerceptionTransformer",
        embed_dims=embed_dims,
        encoder=ConfigDict(
            type="BEVFormerEncoder",
            num_layers=1,
            pc_range=pc_range,
            num_points_in_pillar=4,
            return_intermediate=False,
            transformerlayer=ConfigDict(
                type="BEVFormerLayer",
                attn_cfgs=[
                    ConfigDict(type="TemporalSelfAttention", embed_dims=embed_dims, num_levels=1),
                    ConfigDict(
                        type="SpatialCrossAttention",
                        pc_range=pc_range,
                        deformable_attention=ConfigDict(
                            type="MSDeformableAttention3D", embed_dims=embed_dims, num_points=8, num_levels=1
                        ),
                        embed_dims=embed_dims,
                    ),
                ],
                feedforward_channels=512,
                ffn_dropout=0.1,
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            ),
        ),
        decoder=ConfigDict(
            type="MapTRDecoder",
            num_layers=num_decoder_layers,
            return_intermediate=True,
            transformerlayer=ConfigDict(
                type="DetrTransformerDecoderLayer",
                attn_cfgs=[
                    ConfigDict(type="MultiheadAttention", embed_dims=embed_dims, num_heads=8, dropout=0.1),
                    ConfigDict(type="CustomMSDeformableAttention", embed_dims=embed_dims, num_levels=1),
                ],
                feedforward_channels=512,
                ffn_dropout=0.1,
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            ),
        ),
    )

    bbox_coder_cfg = ConfigDict(
        type="MapTRNMSFreeCoder",
        pc_range=pc_range,
        post_center_range=[-20, -35, -20, -35, 20, 35, 20, 35],
        max_num=50,
        num_classes=num_classes,
    )

    pts_bbox_head_cfg = ConfigDict(
        type="MapTRHead",
        num_classes=num_classes,
        in_channels=embed_dims,
        embed_dims=embed_dims,
        num_query=num_vec * num_pts_per_vec,
        num_reg_fcs=2,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        code_size=2,
        code_weights=[1.0, 1.0, 1.0, 1.0],
        bev_h=bev_h,
        bev_w=bev_w,
        num_vec=num_vec,
        num_pts_per_vec=num_pts_per_vec,
        num_pts_per_gt_vec=num_pts_per_vec,
        query_embed_type="instance_pts",
        transform_method="minmax",
        gt_shift_pts_pattern="v0",
        dir_interval=1,
        transformer=transformer_cfg,
        bbox_coder=bbox_coder_cfg,
        loss_cls=ConfigDict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_bbox=ConfigDict(type="L1Loss", loss_weight=0.0),
        loss_iou=ConfigDict(type="GIoULoss", loss_weight=0.0),
        loss_pts=None,
        loss_dir=None,
        train_cfg=None,
        test_cfg=ConfigDict(max_per_img=50),
    )

    return ConfigDict(
        img_backbone=img_backbone_cfg,
        img_neck=img_neck_cfg,
        pts_bbox_head=pts_bbox_head_cfg,
        bev_h=bev_h,
        bev_w=bev_w,
        pc_range=pc_range,
        num_vec=num_vec,
        num_pts_per_vec=num_pts_per_vec,
        num_classes=num_classes,
        embed_dims=embed_dims,
    )


def create_input_dict(num_cams=6, img_h=384, img_w=640):
    input_dict = {
        "img_metas": [
            [
                {
                    "filename": [
                        "./data/nuscenes/samples/CAM_FRONT/sample.jpg",
                    ]
                    * num_cams,
                    "ori_shape": [(360, 640, 3)] * num_cams,
                    "img_shape": [(img_h, img_w, 3)] * num_cams,
                    "lidar2img": [
                        np.array(
                            [
                                [4.97195909e02, 3.36259809e02, 1.31050214e01, -1.41740456e02],
                                [-7.28050437e00, 2.14719425e02, -4.90215017e02, -2.57883151e02],
                                [-1.17025046e-02, 9.98471159e-01, 5.40221896e-02, -4.25203639e-01],
                                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                            ]
                        )
                        for _ in range(num_cams)
                    ],
                    "pad_shape": [(img_h, img_w, 3)] * num_cams,
                    "scale_factor": 1.0,
                    "flip": False,
                    "pcd_horizontal_flip": False,
                    "pcd_vertical_flip": False,
                    "sample_idx": "3e8750f331d7499e9b5123e9eb70f2e2",
                    "prev_idx": "",
                    "next_idx": "3950bd41f74548429c0f7700ff3d8269",
                    "pcd_scale_factor": 1.0,
                    "pts_filename": "data/pcd.bin",
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
        ],
    }
    return input_dict


def create_stateless_maptr_pipeline_model(tt_model_class, torch_model, tensor, device, config, input_dict):
    def run(input_tensor: ttnn.Tensor) -> tuple:
        fresh_parameters = create_maptr_model_parameters(
            torch_model,
            tensor,
            device,
        )

        fresh_model = tt_model_class(
            device=device,
            params=fresh_parameters,
            use_grid_mask=False,
            pts_voxel_layer=None,
            pts_voxel_encoder=None,
            pts_middle_encoder=None,
            pts_fusion_layer=None,
            img_backbone=True,
            pts_backbone=None,
            img_neck=True,
            pts_neck=None,
            pts_bbox_head=True,
            img_roi_head=None,
            img_rpn_head=None,
            train_cfg=None,
            test_cfg=None,
            pretrained=None,
            video_test_mode=False,
            bev_h=config.bev_h,
            bev_w=config.bev_w,
            pc_range=config.pc_range,
            num_vec=config.num_vec,
            num_pts_per_vec=config.num_pts_per_vec,
            num_classes=config.num_classes,
            embed_dims=config.embed_dims,
        )

        if input_tensor.memory_config().buffer_type == ttnn.BufferType.L1:
            input_tensor = ttnn.to_memory_config(input_tensor, ttnn.DRAM_MEMORY_CONFIG)

        ttnn_img_feats = fresh_model.extract_feat(input_tensor, input_dict["img_metas"])
        ttnn_head_outs = fresh_model.pts_bbox_head(
            ttnn_img_feats,
            lidar_feat=None,
            img_metas=input_dict["img_metas"][0],
            prev_bev=None,
        )

        output_keys = ["bev_embed", "all_cls_scores", "all_bbox_preds", "all_pts_preds"]
        output_tensors = []

        for key in output_keys:
            if key in ttnn_head_outs:
                ttnn_tensor = ttnn_head_outs[key]
                if not isinstance(ttnn_tensor, torch.Tensor):
                    if ttnn_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
                        ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.ROW_MAJOR_LAYOUT)
                    ttnn_tensor = ttnn.to_memory_config(ttnn_tensor, ttnn.DRAM_MEMORY_CONFIG)
                output_tensors.append(ttnn_tensor)

        return tuple(output_tensors)

    return run


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.models_performance_bare_metal
def test_maptr_e2e_performant(
    device,
    reset_seeds,
    model_location_generator,
):
    """
    Performance test for MapTR using TT-CNN Pipeline API with stateless wrapper.

    Uses a stateless wrapper that creates a fresh model instance for each pipeline
    call to avoid persistent tensor deallocation issues.
    """
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    config = create_maptr_config()

    logger.info("Creating PyTorch MapTR model...")
    torch_model = MapTR(
        use_grid_mask=False,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=config.img_backbone,
        pts_backbone=None,
        img_neck=config.img_neck,
        pts_neck=None,
        pts_bbox_head=config.pts_bbox_head,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        video_test_mode=False,
    )

    logger.info(f"Loading weights from {MAPTR_WEIGHTS_PATH}...")
    torch_model = load_maptr_weights(torch_model, MAPTR_WEIGHTS_PATH)
    torch_model.eval()

    input_dict = create_input_dict()
    tensor = torch.randn(1, 6, 3, 384, 640)

    logger.info("Creating TTNN model parameters...")
    parameters = create_maptr_model_parameters(
        torch_model,
        tensor,
        device,
    )

    logger.info("Creating TTNN MapTR model...")
    tt_model = TtMapTR(
        device=device,
        params=parameters,
        use_grid_mask=False,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=True,
        pts_backbone=None,
        img_neck=True,
        pts_neck=None,
        pts_bbox_head=True,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        video_test_mode=False,
        bev_h=config.bev_h,
        bev_w=config.bev_w,
        pc_range=config.pc_range,
        num_vec=config.num_vec,
        num_pts_per_vec=config.num_pts_per_vec,
        num_classes=config.num_classes,
        embed_dims=config.embed_dims,
    )

    logger.info("Extracting PyTorch head outputs...")
    with torch.no_grad():
        B, N, C, H, W = tensor.shape
        img_reshaped = tensor.reshape(B * N, C, H, W)
        torch_backbone_out = torch_model.img_backbone(img_reshaped)
        torch_fpn_out = torch_model.img_neck(torch_backbone_out)

        torch_fpn_reshaped = []
        for img_feat in torch_fpn_out:
            BN, C_feat, H_feat, W_feat = img_feat.size()
            img_feat_5d = img_feat.view(B, int(BN / B), C_feat, H_feat, W_feat)
            torch_fpn_reshaped.append(img_feat_5d)

        img_metas_for_head = input_dict["img_metas"][0]
        torch_head_outs = torch_model.pts_bbox_head(
            torch_fpn_reshaped,
            lidar_feat=None,
            img_metas=img_metas_for_head,
            prev_bev=None,
        )

    logger.info("Creating stateless pipeline model...")
    pipeline_model = create_stateless_maptr_pipeline_model(TtMapTR, torch_model, tensor, device, config, input_dict)

    logger.info("Preparing input tensor for pipeline...")
    # Use the same tensor as PyTorch reference for correct comparison
    pipeline_input = ttnn.from_torch(
        tensor,
        device=None,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    logger.info("Configuring pipeline...")
    pipeline = create_pipeline_from_config(
        config=PipelineConfig(
            use_trace=False,
            num_command_queues=1,
            all_transfers_on_separate_command_queue=False,
        ),
        model=pipeline_model,
        device=device,
        dram_input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        l1_input_memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    logger.info("Compiling pipeline (warmup)...")
    start = time.time()
    pipeline.compile(pipeline_input)
    ttnn.synchronize_device(device)
    end = time.time()
    compile_time = end - start
    logger.info(f"Compilation time: {compile_time:.2f}s")

    num_iterations = 1
    batch_size = 1
    input_tensors = [pipeline_input] * num_iterations

    pipeline.preallocate_output_tensors_on_host(num_iterations)

    logger.info(f"Running {num_iterations} inference iteration(s)...")
    start = time.time()
    outputs = pipeline.enqueue(input_tensors).pop_all()
    ttnn.synchronize_device(device)
    end = time.time()

    inference_time = (end - start) / num_iterations
    throughput_fps = num_iterations * batch_size / (end - start)

    # Convert pipeline outputs to expected format
    # outputs is a list where each element corresponds to one input
    # Each element is a tuple of (bev_embed, all_cls_scores, all_bbox_preds, all_pts_preds)
    ttnn_head_outs_torch = {}
    output_keys = ["bev_embed", "all_cls_scores", "all_bbox_preds", "all_pts_preds"]

    if len(outputs) > 0:
        output_tuple = outputs[0]
        for i, key in enumerate(output_keys):
            if i < len(output_tuple) and output_tuple[i] is not None:
                output_tensor = output_tuple[i]
                if not isinstance(output_tensor, torch.Tensor):
                    ttnn_head_outs_torch[key] = ttnn.to_torch(output_tensor).float()
                else:
                    ttnn_head_outs_torch[key] = output_tensor.float()

    # Cleanup pipeline
    pipeline.cleanup()

    logger.info("Final output PCC comparison:")
    raw_pred_pass = True
    threshold = 0.99

    try:
        outputs_to_check = ["all_cls_scores", "all_bbox_preds", "all_pts_preds", "bev_embed"]
        for key in outputs_to_check:
            if key in torch_head_outs and key in ttnn_head_outs_torch:
                ref = torch_head_outs[key].float()
                tt = ttnn_head_outs_torch[key]
                _, pcc = comp_pcc(ref, tt)
                status = "✓ PASS" if pcc >= threshold else "✗ FAIL"
                logger.info(f"  {key}: PCC={pcc:.6f} {status}")
                if pcc < threshold:
                    raw_pred_pass = False
            else:
                logger.warning(f"  {key}: Reference or TTNN output not available")

        assert raw_pred_pass, "PCC below threshold"
        logger.info("PCC check PASSED!")
    except Exception as e:
        logger.error(f"Error during PCC check: {e}")
        traceback.print_exc()
        raise

    logger.info(f"Average model time: {1000.0 * inference_time:.2f} ms")
    logger.info(f"Average model performance: {throughput_fps:.2f} fps")

    total_num_samples = batch_size
    prep_perf_report(
        model_name="maptr-e2e",
        batch_size=total_num_samples,
        inference_and_compile_time=compile_time,
        inference_time=inference_time,
        expected_compile_time=15.0,
        expected_inference_time=total_num_samples / 0.12,
        comments=f"img_384x640-iters_{num_iterations}",
    )

    logger.info("Performance test completed!")
