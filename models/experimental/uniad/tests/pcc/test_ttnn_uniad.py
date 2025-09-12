# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import torch
import pytest
from loguru import logger
import ttnn
from models.experimental.uniad.reference.uniad import UniAD
from models.experimental.uniad.tt.ttnn_uniad import TtUniAD
import numpy as np
from models.experimental.uniad.reference.utils import LiDARInstance3DBoxes
from models.experimental.uniad.tt.ttnn_utils import TtLiDARInstance3DBoxes
import copy

from models.experimental.uniad.tt.model_preprocessing_uniad import create_uniad_model_parameters_uniad
from models.experimental.uniad.common import load_torch_model


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_uniad(device, reset_seeds, model_location_generator):
    reference_model = UniAD(
        True,
        True,
        True,
        True,
        task_loss_weight={"track": 1.0, "map": 1.0, "motion": 1.0, "occ": 1.0, "planning": 1.0},
        **{
            "gt_iou_threshold": 0.3,
            "queue_length": 3,
            "use_grid_mask": True,
            "video_test_mode": True,
            "num_query": 900,
            "num_classes": 10,
            "vehicle_id_list": [0, 1, 2, 3, 4, 6, 7],
            "pc_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            "img_backbone": {
                "type": "ResNet",
                "depth": 101,
                "num_stages": 4,
                "out_indices": (1, 2, 3),
                "frozen_stages": 4,
                "norm_cfg": {"type": "BN2d", "requires_grad": False},
                "norm_eval": True,
                "style": "caffe",
                "dcn": {"type": "DCNv2", "deform_groups": 1, "fallback_on_stride": False},
                "stage_with_dcn": (False, False, True, True),
            },
            "img_neck": {
                "type": "FPN",
                "in_channels": [512, 1024, 2048],
                "out_channels": 256,
                "start_level": 0,
                "add_extra_convs": "on_output",
                "num_outs": 4,
                "relu_before_extra_convs": True,
            },
            "freeze_img_backbone": True,
            "freeze_img_neck": True,
            "freeze_bn": True,
            "freeze_bev_encoder": True,
            "score_thresh": 0.4,
            "filter_score_thresh": 0.35,
            "qim_args": {
                "qim_type": "QIMBase",
                "update_query_pos": True,
                "fp_ratio": 0.3,
                "random_drop": 0.1,
            },
            "mem_args": {"memory_bank_type": "MemoryBank", "memory_bank_score_thresh": 0.0, "memory_bank_len": 4},
            "loss_cfg": {
                "type": "ClipMatcher",
                "num_classes": 10,
                "weight_dict": None,
                "code_weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
                "assigner": {
                    "type": "HungarianAssigner3DTrack",
                    "cls_cost": {"type": "FocalLossCost", "weight": 2.0},
                    "reg_cost": {"type": "BBox3DL1Cost", "weight": 0.25},
                    "pc_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                },
                "loss_cls": {"type": "FocalLoss", "use_sigmoid": True, "gamma": 2.0, "alpha": 0.25, "loss_weight": 2.0},
                "loss_bbox": {"type": "L1Loss", "loss_weight": 0.25},
            },
            "pts_bbox_head": {
                "type": "BEVFormerTrackHead",
                "bev_h": 50,
                "bev_w": 50,
                "num_query": 900,
                "num_classes": 10,
                "in_channels": 256,
                "sync_cls_avg_factor": True,
                "with_box_refine": True,
                "as_two_stage": False,
                "past_steps": 4,
                "fut_steps": 4,
                "transformer": {
                    "type": "PerceptionTransformer",
                    "rotate_prev_bev": True,
                    "use_shift": True,
                    "use_can_bus": True,
                    "embed_dims": 256,
                    "encoder": {
                        "type": "BEVFormerEncoder",
                        "num_layers": 6,
                        "pc_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                        "num_points_in_pillar": 4,
                        "return_intermediate": False,
                        "transformerlayers": {
                            "type": "BEVFormerLayer",
                            "attn_cfgs": [
                                {"type": "TemporalSelfAttention", "embed_dims": 256, "num_levels": 1},
                                {
                                    "type": "SpatialCrossAttention",
                                    "pc_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                                    "deformable_attention": {
                                        "type": "MSDeformableAttention3D",
                                        "embed_dims": 256,
                                        "num_points": 8,
                                        "num_levels": 4,
                                    },
                                    "embed_dims": 256,
                                },
                            ],
                            "feedforward_channels": 512,
                            "operation_order": ("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
                        },
                    },
                    "decoder": {
                        "type": "DetectionTransformerDecoder",
                        "num_layers": 6,
                        "return_intermediate": True,
                        "transformerlayers": {
                            "type": "DetrTransformerDecoderLayer",
                            "attn_cfgs": [
                                {"type": "MultiheadAttention", "embed_dims": 256, "num_heads": 8},
                                {"type": "CustomMSDeformableAttention", "embed_dims": 256, "num_levels": 1},
                            ],
                            "feedforward_channels": 512,
                            "operation_order": ("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
                        },
                    },
                },
                "bbox_coder": {
                    "type": "NMSFreeCoder",
                    "post_center_range": [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                    "pc_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                    "max_num": 300,
                    "voxel_size": [0.2, 0.2, 8],
                    "num_classes": 10,
                },
                "positional_encoding": {
                    "type": "LearnedPositionalEncoding",
                    "num_feats": 128,
                    "row_num_embed": 50,
                    "col_num_embed": 50,
                },
                "loss_cls": {"type": "FocalLoss", "use_sigmoid": True, "gamma": 2.0, "alpha": 0.25, "loss_weight": 2.0},
                "loss_bbox": {"type": "L1Loss", "loss_weight": 0.25},
                "loss_iou": {"type": "GIoULoss", "loss_weight": 0.0},
            },
            "train_cfg": None,
            "pretrained": None,
            "test_cfg": None,
        },
    )

    # Load the modified checkpoint
    reference_model = load_torch_model(
        torch_model=reference_model, layer="", model_location_generator=model_location_generator
    )

    rescale = True

    img = []
    img.append(torch.randn(1, 6, 3, 640, 360))
    timestamp = [torch.tensor([1.5332e09], dtype=torch.float64)]
    l2g_r_mat = torch.tensor([[[-0.4814, -0.8765, -0.0070], [0.8749, -0.4801, -0.0631], [0.0519, -0.0365, 0.9980]]])
    l2g_t = torch.tensor([[601.0081, 1646.9954, 1.8233]])

    gt_lane_labels = [torch.tensor([[0, 2, 2, 3]])]
    gt_lane_bboxes = [torch.tensor([[[27, 49, 27, 49], [26, 0, 34, 49], [21, 0, 23, 49], [21, 0, 36, 49]]])]
    gt_lane_masks = [torch.zeros([1, 4, 50, 50])]

    gt_segmentation = [torch.zeros([1, 7, 50, 50])]

    gt_instance = [torch.zeros([1, 7, 50, 50])]

    gt_centerness = [torch.zeros([1, 7, 1, 50, 50], dtype=torch.float64)]

    gt_offset = [torch.full([1, 7, 2, 50, 50], 255.0, dtype=torch.float32)]
    gt_flow = [torch.full([1, 7, 2, 50, 50], 255.0, dtype=torch.float32)]
    gt_backward_flow = [torch.full([1, 7, 2, 50, 50], 255.0, dtype=torch.float32)]
    gt_occ_has_invalid_frame = [torch.tensor([True])]
    gt_occ_img_is_valid = [torch.Tensor([[False, False, True, True, True, True, True, True, True]])]
    sdc_planning = [
        torch.tensor(
            [
                [
                    [
                        [0.0758, 4.2526, 1.5528],
                        [0.2682, 8.4670, 1.5254],
                        [0.5579, 12.6398, 1.4962],
                        [0.9556, 16.8074, 1.4693],
                        [1.4746, 21.0605, 1.4454],
                        [2.1522, 25.3447, 1.4150],
                    ]
                ]
            ]
        )
    ]
    sdc_planning_mask = [torch.tensor([[[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]]])]

    command = torch.tensor(torch.tensor([0]))

    img_metas = [
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
                "ori_shape": [(900, 1600, 3)] * 6,
                "img_shape": [(640, 360, 3)] * 6,
                "lidar2img": [
                    np.array(
                        [
                            [1.24298977e03, 8.40649523e02, 3.27625534e01, -3.54351139e02],
                            [-1.82012609e01, 5.36798564e02, -1.22553754e03, -6.44707879e02],
                            [-1.17025046e-02, 9.98471159e-01, 5.40221896e-02, -4.25203639e-01],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                        ]
                    ),
                    np.array(
                        [
                            [1.36494654e03, -6.19264860e02, -4.03391641e01, -4.61642859e02],
                            [3.79462336e02, 3.20307276e02, -1.23979473e03, -6.92556280e02],
                            [8.43406855e-01, 5.36312055e-01, 3.21598489e-02, -6.10371854e-01],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                        ]
                    ),
                    np.array(
                        [
                            [3.23698342e01, 1.50315427e03, 7.76231827e01, -3.02437885e02],
                            [-3.89320197e02, 3.20441551e02, -1.23745300e03, -6.79424755e02],
                            [-8.23415292e-01, 5.65940098e-01, 4.12196894e-02, -5.29677094e-01],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                        ]
                    ),
                    np.array(
                        [
                            [-8.03982245e02, -8.50723862e02, -2.64376631e01, -8.70795988e02],
                            [-1.08232816e01, -4.45285963e02, -8.14897443e02, -7.08684241e02],
                            [-8.33350064e-03, -9.99200442e-01, -3.91028008e-02, -1.01645350e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                        ]
                    ),
                    np.array(
                        [
                            [-1.18656611e03, 9.23261441e02, 5.32641592e01, -6.25341190e02],
                            [-4.62625515e02, -1.02540587e02, -1.25247717e03, -5.61828455e02],
                            [-9.47586752e-01, -3.19482867e-01, 3.16948959e-03, -4.32527296e-01],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                        ]
                    ),
                    np.array(
                        [
                            [2.85189233e02, -1.46927652e03, -5.95634293e01, -2.72600319e02],
                            [4.44736043e02, -1.22825702e02, -1.25039267e03, -5.88246117e02],
                            [9.24052925e-01, -3.82246554e-01, -3.70989150e-03, -4.64645142e-01],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                        ]
                    ),
                ],
                "pad_shape": [(640, 360, 3)] * 6,
                "scale_factor": 1.0,
                "flip": False,
                "pcd_horizontal_flip": False,
                "pcd_vertical_flip": False,
                "box_type_3d": LiDARInstance3DBoxes,
                "img_norm_cfg": {
                    "mean": np.array([103.53, 116.28, 123.675], dtype=np.float32),
                    "std": np.array([1.0, 1.0, 1.0], dtype=np.float32),
                    "to_rgb": False,
                },
                "sample_idx": "3e8750f331d7499e9b5123e9eb70f2e2",
                "prev_idx": "",
                "next_idx": "3950bd41f74548429c0f7700ff3d8269",
                "pcd_scale_factor": 1.0,
                "pts_filename": "./data/nuscenes/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin",
                "scene_token": "fcbccedd61424f1b85dcbf8f897f9754",
                "can_bus": np.array(
                    [
                        6.00120214e02,
                        1.64749078e03,
                        0.00000000e00,
                        -9.68669702e-01,
                        -4.04339926e-03,
                        -7.66659427e-03,
                        2.48201296e-01,
                        -6.06941519e-01,
                        -7.63441180e-02,
                        9.87149385e00,
                        -2.10869126e-02,
                        -1.24397185e-02,
                        -2.30670013e-02,
                        8.56405970e00,
                        0.00000000e00,
                        0.00000000e00,
                        5.78155401e00,
                        3.31258644e02,
                    ]
                ),
            }
        ]
    ]

    ttnn_rescale = True
    ttnn_img = [ttnn.from_torch(img[0], device=device, dtype=ttnn.bfloat16)]
    ttnn_timestamp = [ttnn.from_torch(timestamp[0], device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)]
    ttnn_l2g_r_mat = [ttnn.from_torch(l2g_r_mat, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)]
    ttnn_l2g_t = [ttnn.from_torch(l2g_t, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)]

    ttnn_gt_lane_labels = [ttnn.from_torch(gt_lane_labels[0], device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.int32)]
    ttnn_gt_lane_bboxes = [ttnn.from_torch(gt_lane_bboxes[0], device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.int32)]
    ttnn_gt_lane_masks = [ttnn.from_torch(gt_lane_masks[0], device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.int32)]

    ttnn_gt_segmentation = [
        ttnn.from_torch(gt_segmentation[0], device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.int32)
    ]
    ttnn_gt_instance = [ttnn.from_torch(gt_instance[0], device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.int32)]
    ttnn_gt_centerness = [
        ttnn.from_torch(gt_centerness[0], device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    ]

    ttnn_gt_offset = [(ttnn.from_torch(gt_offset[0], device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16))]
    ttnn_gt_flow = [ttnn.from_torch(gt_flow[0], device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)]
    ttnn_gt_backward_flow = [
        ttnn.from_torch(gt_backward_flow[0], device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    ]

    ttnn_gt_occ_has_invalid_frame = [
        ttnn.from_torch(gt_occ_has_invalid_frame[0], device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.int32)
    ]
    ttnn_gt_occ_img_is_valid = [
        ttnn.from_torch(gt_occ_img_is_valid[0], device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.int32)
    ]
    ttnn_sdc_planning = [ttnn.from_torch(sdc_planning[0], device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)]
    ttnn_sdc_planning_mask = [
        ttnn.from_torch(sdc_planning_mask[0], device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    ]
    ttnn_command = [ttnn.from_torch(command, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.int32)]

    ttnn_img_metas = copy.deepcopy(img_metas)
    ttnn_img_metas[0][0]["box_type_3d"] = TtLiDARInstance3DBoxes

    parameters = create_uniad_model_parameters_uniad(reference_model, device)

    ttnn_model = TtUniAD(
        parameters,
        device,
        True,
        True,
        True,
        True,
        task_loss_weight={"track": 1.0, "map": 1.0, "motion": 1.0, "occ": 1.0, "planning": 1.0},
        **{
            "gt_iou_threshold": 0.3,
            "queue_length": 3,
            "use_grid_mask": True,
            "video_test_mode": True,
            "num_query": 900,
            "num_classes": 10,
            "vehicle_id_list": [0, 1, 2, 3, 4, 6, 7],
            "pc_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            "img_backbone": {
                "type": "ResNet",
                "depth": 101,
                "num_stages": 4,
                "out_indices": (1, 2, 3),
                "frozen_stages": 4,
                "norm_cfg": {"type": "BN2d", "requires_grad": False},
                "norm_eval": True,
                "style": "caffe",
                "dcn": {"type": "DCNv2", "deform_groups": 1, "fallback_on_stride": False},
                "stage_with_dcn": (False, False, True, True),
            },
            "img_neck": {
                "type": "FPN",
                "in_channels": [512, 1024, 2048],
                "out_channels": 256,
                "start_level": 0,
                "add_extra_convs": "on_output",
                "num_outs": 4,
                "relu_before_extra_convs": True,
            },
            "freeze_img_backbone": True,
            "freeze_img_neck": True,
            "freeze_bn": True,
            "freeze_bev_encoder": True,
            "score_thresh": 0.4,
            "filter_score_thresh": 0.35,
            "qim_args": {
                "qim_type": "QIMBase",
                "update_query_pos": True,
                "fp_ratio": 0.3,
                "random_drop": 0.1,
            },
            "mem_args": {"memory_bank_type": "MemoryBank", "memory_bank_score_thresh": 0.0, "memory_bank_len": 4},
            "loss_cfg": {
                "type": "ClipMatcher",
                "num_classes": 10,
                "weight_dict": None,
                "code_weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
                "assigner": {
                    "type": "HungarianAssigner3DTrack",
                    "cls_cost": {"type": "FocalLossCost", "weight": 2.0},
                    "reg_cost": {"type": "BBox3DL1Cost", "weight": 0.25},
                    "pc_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                },
                "loss_cls": {"type": "FocalLoss", "use_sigmoid": True, "gamma": 2.0, "alpha": 0.25, "loss_weight": 2.0},
                "loss_bbox": {"type": "L1Loss", "loss_weight": 0.25},
            },
            "pts_bbox_head": {
                "type": "BEVFormerTrackHead",
                "bev_h": 50,
                "bev_w": 50,
                "num_query": 900,
                "num_classes": 10,
                "in_channels": 256,
                "sync_cls_avg_factor": True,
                "with_box_refine": True,
                "as_two_stage": False,
                "past_steps": 4,
                "fut_steps": 4,
                "transformer": {
                    "type": "PerceptionTransformer",
                    "rotate_prev_bev": True,
                    "use_shift": True,
                    "use_can_bus": True,
                    "embed_dims": 256,
                    "encoder": {
                        "type": "BEVFormerEncoder",
                        "num_layers": 6,
                        "pc_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                        "num_points_in_pillar": 4,
                        "return_intermediate": False,
                        "transformerlayers": {
                            "type": "BEVFormerLayer",
                            "attn_cfgs": [
                                {"type": "TemporalSelfAttention", "embed_dims": 256, "num_levels": 1},
                                {
                                    "type": "SpatialCrossAttention",
                                    "pc_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                                    "deformable_attention": {
                                        "type": "MSDeformableAttention3D",
                                        "embed_dims": 256,
                                        "num_points": 8,
                                        "num_levels": 4,
                                    },
                                    "embed_dims": 256,
                                },
                            ],
                            "feedforward_channels": 512,
                            "operation_order": ("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
                        },
                    },
                    "decoder": {
                        "type": "DetectionTransformerDecoder",
                        "num_layers": 6,
                        "return_intermediate": True,
                        "transformerlayers": {
                            "type": "DetrTransformerDecoderLayer",
                            "attn_cfgs": [
                                {"type": "MultiheadAttention", "embed_dims": 256, "num_heads": 8},
                                {"type": "CustomMSDeformableAttention", "embed_dims": 256, "num_levels": 1},
                            ],
                            "feedforward_channels": 512,
                            "operation_order": ("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
                        },
                    },
                },
                "bbox_coder": {
                    "type": "NMSFreeCoder",
                    "post_center_range": [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                    "pc_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                    "max_num": 300,
                    "voxel_size": [0.2, 0.2, 8],
                    "num_classes": 10,
                },
                "positional_encoding": {
                    "type": "LearnedPositionalEncoding",
                    "num_feats": 128,
                    "row_num_embed": 50,
                    "col_num_embed": 50,
                },
                "loss_cls": {"type": "FocalLoss", "use_sigmoid": True, "gamma": 2.0, "alpha": 0.25, "loss_weight": 2.0},
                "loss_bbox": {"type": "L1Loss", "loss_weight": 0.25},
                "loss_iou": {"type": "GIoULoss", "loss_weight": 0.0},
            },
            "train_cfg": None,
            "pretrained": None,
            "test_cfg": None,
        },
    )

    reference_output = reference_model(
        return_loss=False,
        rescale=rescale,
        img_metas=img_metas,
        img=img,
        timestamp=timestamp,
        l2g_r_mat=l2g_r_mat,
        l2g_t=l2g_t,
        gt_lane_labels=gt_lane_labels,
        gt_lane_bboxes=gt_lane_bboxes,
        gt_lane_masks=gt_lane_masks,
        gt_segmentation=gt_segmentation,
        gt_instance=gt_instance,
        gt_centerness=gt_centerness,
        gt_offset=gt_offset,
        gt_flow=gt_flow,
        gt_backward_flow=gt_backward_flow,
        gt_occ_has_invalid_frame=gt_occ_has_invalid_frame,
        gt_occ_img_is_valid=gt_occ_img_is_valid,
        sdc_planning=sdc_planning,
        sdc_planning_mask=sdc_planning_mask,
        command=command,
    )

    ttnn_output = ttnn_model(
        return_loss=False,
        rescale=ttnn_rescale,
        img_metas=ttnn_img_metas,
        img=ttnn_img,
        timestamp=ttnn_timestamp,
        l2g_r_mat=ttnn_l2g_r_mat,
        l2g_t=ttnn_l2g_t,
        gt_lane_labels=ttnn_gt_lane_labels,
        gt_lane_bboxes=ttnn_gt_lane_bboxes,
        gt_lane_masks=ttnn_gt_lane_masks,
        gt_segmentation=ttnn_gt_segmentation,
        gt_instance=ttnn_gt_instance,
        gt_centerness=ttnn_gt_centerness,
        gt_offset=ttnn_gt_offset,
        gt_flow=ttnn_gt_flow,
        gt_backward_flow=ttnn_gt_backward_flow,
        gt_occ_has_invalid_frame=ttnn_gt_occ_has_invalid_frame,
        gt_occ_img_is_valid=ttnn_gt_occ_img_is_valid,
        sdc_planning=ttnn_sdc_planning,
        sdc_planning_mask=ttnn_sdc_planning_mask,
        command=ttnn_command,
    )

    logger.info(f"reference_output: {reference_output}")
    logger.info(f"ttnn_output: {ttnn_output}")
