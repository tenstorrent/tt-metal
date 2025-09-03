# _base_ = [
#     '../datasets/custom_nus-3d.py',
#     '../_base_/default_runtime.py'
# ]
# #
# plugin = True
# plugin_dir = 'projects/mmdet3d_plugin/'

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

dataset_type = "VADCustomNuScenesDataset"
data_root = "data/nuscenes/"
file_client_args = dict(backend="disk")

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "vad_nuscenes_infos_temporal_train.pkl",
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        queue_length=queue_length,
        map_classes=map_classes,
        map_fixed_ptsnum_per_line=map_fixed_ptsnum_per_gt_line,
        map_eval_use_same_gt_sample_num_flag=map_eval_use_same_gt_sample_num_flag,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d="LiDAR",
        custom_eval_version="vad_nusc_detection_cvpr_2019",
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        pc_range=point_cloud_range,
        ann_file=data_root + "vad_nuscenes_infos_temporal_val.pkl",
        pipeline=test_pipeline,
        bev_size=(bev_h_, bev_w_),
        classes=class_names,
        modality=input_modality,
        samples_per_gpu=1,
        map_classes=map_classes,
        map_ann_file=data_root + "nuscenes_map_anns_val.json",
        map_fixed_ptsnum_per_line=map_fixed_ptsnum_per_gt_line,
        map_eval_use_same_gt_sample_num_flag=map_eval_use_same_gt_sample_num_flag,
        use_pkl_result=True,
        custom_eval_version="vad_nusc_detection_cvpr_2019",
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        pc_range=point_cloud_range,
        ann_file=data_root + "vad_nuscenes_infos_temporal_val.pkl",
        pipeline=test_pipeline,
        bev_size=(bev_h_, bev_w_),
        classes=class_names,
        modality=input_modality,
        samples_per_gpu=1,
        map_classes=map_classes,
        map_ann_file=data_root + "nuscenes_map_anns_val.json",
        map_fixed_ptsnum_per_line=map_fixed_ptsnum_per_gt_line,
        map_eval_use_same_gt_sample_num_flag=map_eval_use_same_gt_sample_num_flag,
        use_pkl_result=True,
        custom_eval_version="vad_nusc_detection_cvpr_2019",
    ),
    shuffler_sampler=dict(type="DistributedGroupSampler"),
    nonshuffler_sampler=dict(type="DistributedSampler"),
)

optimizer = dict(
    type="AdamW",
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.1),
        }
    ),
    weight_decay=0.01,
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(policy="CosineAnnealing", warmup="linear", warmup_iters=500, warmup_ratio=1.0 / 3, min_lr_ratio=1e-3)

evaluation = dict(interval=total_epochs, pipeline=test_pipeline, metric="bbox", map_metric="chamfer")

runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)
load_from = "ckpts/VAD_tiny_stage_1.pth"
log_config = dict(interval=100, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")])
# fp16 = dict(loss_scale=512.)
# find_unused_parameters = True
checkpoint_config = dict(interval=1, max_keep_ckpts=total_epochs)


custom_hooks = [dict(type="CustomSetEpochInfoHook")]
