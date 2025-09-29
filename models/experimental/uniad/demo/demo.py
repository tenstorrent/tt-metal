import os
import torch
from mmengine.config import Config
from mmengine.runner import Runner

# from mmdet3d.registry import DATASETS
from models.experimental.uniad.reference.uniad import UniAD
from models.experimental.uniad.common import load_torch_model
from mmengine.utils import ProgressBar
import torchvision.transforms as transforms
from loguru import logger


def test_demo(model_location_generator):
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
    reference_model = load_torch_model(
        torch_model=reference_model, layer="", model_location_generator=model_location_generator
    )
    # 1. Load config (use any NuScenes config available in configs/nuscenes)
    cfg = Config.fromfile("models/experimental/uniad/demo/config.py")

    dataloader = Runner.build_dataloader(cfg.val_dataloader)

    outputs = single_gpu_test(reference_model, dataloader)
    logger.info(outputs)


def single_gpu_test(model, data_loader, show=False, out_dir=None, show_score_thr=0.3):
    model.eval()

    bbox_results = []
    mask_results = []
    results = []
    dataset = data_loader.dataset
    prog_bar = ProgressBar(len(dataset))

    have_mask = False
    gpu_collect = False
    tmpdir = None
    num_occ = 0

    for test, data in enumerate(data_loader):
        transform = transforms.Compose(
            [
                transforms.Resize((640, 360)),
            ]
        )
        img_tensor = data["img"][0][0].data[0]

        resized_imgs = []
        for i in range(img_tensor.shape[0]):
            img = img_tensor[i]
            img_resized = transform(img)
            resized_imgs.append(img_resized)

        resized_imgs_tensor = torch.stack(resized_imgs)

        with torch.no_grad():
            img_metas = (data["img_metas"][0][0].data,)
            img = data["img"][0][0].data
            gt_lane_masks = data["gt_lane_masks"][0][0].unsqueeze(0)
            gt_lane_labels = data["gt_lane_labels"][0][0].unsqueeze(0)
            gt_segmentation = data["gt_segmentation"][0][0].unsqueeze(0)
            gt_instance = data["gt_instance"][0][0].unsqueeze(0)
            gt_centerness = data["gt_centerness"][0][0].unsqueeze(0)
            gt_offset = data["gt_offset"][0][0].unsqueeze(0)
            gt_occ_img_is_valid = torch.tensor(data["gt_occ_img_is_valid"][0][0]).unsqueeze(0)
            timestamp = torch.tensor(data["timestamp"][0][0])
            keys_to_delete = [
                "img_metas",
                "img",
                "gt_lane_masks",
                "gt_lane_labels",
                "gt_segmentation",
                "gt_instance",
                "gt_centerness",
                "gt_offset",
                "gt_occ_img_is_valid",
                "timestamp",
            ]

            for key in keys_to_delete:
                del data[key]

            result = model(
                return_loss=False,
                rescale=True,
                img_metas=[img_metas],
                img=[img],
                gt_lane_masks=[gt_lane_masks],
                gt_lane_labels=[gt_lane_labels],
                gt_segmentation=[gt_segmentation],
                gt_instance=[gt_instance],
                gt_centerness=[gt_centerness],
                gt_offset=[gt_offset],
                gt_occ_img_is_valid=[gt_occ_img_is_valid],
                timestamp=[timestamp],
                **data,
            )

            if os.environ.get("ENABLE_PLOT_MODE", None) is None:
                result[0].pop("occ", None)
                result[0].pop("planning", None)
            else:
                for k in ["seg_gt", "ins_seg_gt", "pred_ins_sigmoid", "seg_out", "ins_seg_out"]:
                    if k in result[0]["occ"]:
                        result[0]["occ"][k] = result[0]["occ"][k].detach().cpu()
                for k in [
                    "bbox",
                    "segm",
                    "labels",
                    "panoptic",
                    "drivable",
                    "score_list",
                    "lane",
                    "lane_score",
                    "stuff_score_list",
                ]:
                    if k in result[0]["pts_bbox"] and isinstance(result[0]["pts_bbox"][k], torch.Tensor):
                        result[0]["pts_bbox"][k] = result[0]["pts_bbox"][k].detach().cpu()

            # encode mask results
            if isinstance(result, dict):
                if "bbox_results" in result.keys():
                    bbox_result = result["bbox_results"]
                    batch_size = len(result["bbox_results"])
                    bbox_results.extend(bbox_result)
                if "mask_results" in result.keys() and result["mask_results"] is not None:
                    mask_result = custom_encode_mask_results(result["mask_results"])
                    mask_results.extend(mask_result)
                    have_mask = True
            else:
                batch_size = len(result)
                bbox_results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()

    if gpu_collect:
        bbox_results = collect_results_gpu(bbox_results, len(dataset))
        if have_mask:
            mask_results = collect_results_gpu(mask_results, len(dataset))
        else:
            mask_results = None
    else:
        bbox_results = collect_results_cpu(bbox_results, len(dataset), tmpdir)
        tmpdir = tmpdir + "_mask" if tmpdir is not None else None
        if have_mask:
            mask_results = collect_results_cpu(mask_results, len(dataset), tmpdir)
        else:
            mask_results = None

    if eval_planning:
        planning_results = planning_metrics.compute()
        planning_metrics.reset()

    ret_results = dict()
    ret_results["bbox_results"] = bbox_results
    if eval_occ:
        occ_results = {}
        for key, grid in EVALUATION_RANGES.items():
            panoptic_scores = panoptic_metrics[key].compute()
            for panoptic_key, value in panoptic_scores.items():
                occ_results[f"{panoptic_key}"] = occ_results.get(f"{panoptic_key}", []) + [100 * value[1].item()]
            panoptic_metrics[key].reset()

            iou_scores = iou_metrics[key].compute()
            occ_results["iou"] = occ_results.get("iou", []) + [100 * iou_scores[1].item()]
            iou_metrics[key].reset()

        occ_results["num_occ"] = num_occ  # count on one gpu
        occ_results["ratio_occ"] = num_occ / len(dataset)  # count on one gpu, but reflect the relative ratio
        ret_results["occ_results_computed"] = occ_results
    if eval_planning:
        ret_results["planning_results_computed"] = planning_results

    if mask_results is not None:
        ret_results["mask_results"] = mask_results
    logger.info(ret_results)
    return ret_results
