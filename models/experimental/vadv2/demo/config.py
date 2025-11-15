# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

_base_ = [
    # 'mmdet3d::_base_/datasets/nus-3d.py',
    "mmdet3d::_base_/default_runtime.py"
]
# custom_imports = dict(imports=['mmdet3d.datasets.transforms.loading'])


# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
voxel_size = [0.15, 0.15, 4]

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# For nuScenes we usually do 10-class detection
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]
num_classes = len(class_names)

# map has classes: divider, ped_crossing, boundary
map_classes = ["divider", "ped_crossing", "boundary"]
map_num_vec = 100
map_fixed_ptsnum_per_gt_line = 20  # now only support fixed_pts > 0
map_fixed_ptsnum_per_pred_line = 20
map_eval_use_same_gt_sample_num_flag = True
map_num_classes = len(map_classes)

input_modality = dict(use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=True)
# this means type='DETR3D' will be processed as 'mmdet3d.DETR3D'
default_scope = "mmdet3d"

_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 1
bev_h_ = 100
bev_w_ = 100
queue_length = 3  # each sequence contains `queue_length` frames.
total_epochs = 12


dataset_type = "mmdet3d.VADCustomNuScenesDataset"
data_root = "/home/ubuntu/sabira/tt-metal/models/experimental/vadv2/demo/data/nuscenes/"
file_client_args = dict(backend="disk")


test_transforms = [dict(type="mmdet3d.RandomResize3D", scale=(1600, 900), ratio_range=(1.0, 1.0), keep_ratio=True)]

file_client_args = dict(backend="disk")
test_pipeline = [
    dict(type="mmdet3d.CustomLoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="mmdet3d.CustomLoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(type="mmdet3d.LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    dict(type="mmdet3d.CustomObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="mmdet3d.CustomObjectNameFilter", classes=class_names),
    dict(type="mmdet3d.NormalizeMultiviewImage", **img_norm_cfg),
    dict(
        type="mmdet3d.MultiScaleFlipAug3D",
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type="mmdet3d.RandomScaleImageMultiViewImage", scales=[0.4]),
            dict(type="mmdet3d.PadMultiViewImage", size_divisor=32),
            dict(type="mmdet3d.CustomDefaultFormatBundle3D", class_names=class_names, with_label=False, with_ego=True),
            dict(
                type="mmdet3d.CustomCollect3D",
                keys=[
                    "points",
                    "gt_bboxes_3d",
                    "gt_labels_3d",
                    "img",
                    "fut_valid_flag",
                    "ego_his_trajs",
                    "ego_fut_trajs",
                    "ego_fut_masks",
                    "ego_fut_cmd",
                    "ego_lcf_feat",
                    "gt_attr_labels",
                ],
            ),
        ],
    ),
]


train_transforms = [dict(type="PhotoMetricDistortion3D")] + test_transforms

backend_args = None
metainfo = dict(classes=class_names)
data_prefix = dict(
    pts="",
    CAM_FRONT="samples/CAM_FRONT",
    CAM_FRONT_LEFT="samples/CAM_FRONT_LEFT",
    CAM_FRONT_RIGHT="samples/CAM_FRONT_RIGHT",
    CAM_BACK="samples/CAM_BACK",
    CAM_BACK_RIGHT="samples/CAM_BACK_RIGHT",
    CAM_BACK_LEFT="samples/CAM_BACK_LEFT",
)


val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    # sampler=dict(type='DistributedGroupSampler'),
    dataset=dict(
        type=dataset_type,
        data_root="/home/ubuntu/sabira/tt-metal/models/experimental/vadv2/demo/data/nuscenes/",
        pc_range=point_cloud_range,
        ann_file=data_root + "vad_nuscenes_infos_val.pkl",
        pipeline=test_pipeline,
        bev_size=(bev_h_, bev_w_),
        classes=class_names,
        modality=input_modality,
        map_classes=map_classes,
        map_ann_file=data_root + "nuscenes_map_anns_val.json",
        map_fixed_ptsnum_per_line=map_fixed_ptsnum_per_gt_line,
        map_eval_use_same_gt_sample_num_flag=map_eval_use_same_gt_sample_num_flag,
        use_pkl_result=True,
        custom_eval_version="vad_nusc_detection_cvpr_2019",
    ),
)


test_dataloader = val_dataloader
evaluation = dict(interval=total_epochs, pipeline=test_pipeline, metric="bbox", map_metric="chamfer")

# val_evaluator = dict(
#     type='NuScenesMetric',
#     data_root=data_root,
#     ann_file=data_root + 'nuscenes_infos_val.pkl',
#     metric='bbox',
#     backend_args=backend_args)
# test_evaluator = val_evaluator

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=2e-4, weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={"img_backbone": dict(lr_mult=0.1)}),
    clip_grad=dict(max_norm=35, norm_type=2),
)

# learning policy
param_scheduler = [
    dict(type="LinearLR", start_factor=1.0 / 3, by_epoch=False, begin=0, end=500),
    dict(type="CosineAnnealingLR", by_epoch=True, begin=0, end=24, T_max=24, eta_min_ratio=1e-3),
]

total_epochs = 24

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=total_epochs, val_interval=2)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=1, save_last=True))
load_from = "ckpts/fcos3d.pth"

# setuptools 65 downgrades to 58.
# In mmlab-node we use setuptools 61 but occurs NO errors
vis_backends = [dict(type="TensorboardVisBackend")]
visualizer = dict(type="Det3DLocalVisualizer", vis_backends=vis_backends, name="visualizer")
