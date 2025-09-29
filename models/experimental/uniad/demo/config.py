_base_ = [
    "mmdet3d::_base_/default_runtime.py",
]


plugin = True

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
patch_size = [102.4, 102.4]
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

# For nuScenes we usually do 10-class detection
default_scope = "mmdet3d"
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
vehicle_id_list = [0, 1, 2, 3, 4, 6, 7]
group_id_list = [[0, 1, 2, 3, 4], [6, 7], [8], [5, 9]]
input_modality = dict(use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=True)

backend_args = None

_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 4
bev_h_ = 50
bev_w_ = 50
_feed_dim_ = _ffn_dim_
_dim_half_ = _pos_dim_
canvas_size = (bev_h_, bev_w_)
queue_length = 3  # each sequence contains `queue_length` frames.

### traj prediction args ###
predict_steps = 12
predict_modes = 6
fut_steps = 4
past_steps = 4
use_nonlinear_optimizer = True

## occflow setting
occ_n_future = 4
occ_n_future_plan = 6
occ_n_future_max = max([occ_n_future, occ_n_future_plan])

### planning ###
planning_steps = 6
use_col_optim = True
# there exists multiple interpretations of the planning metric, where it differs between uniad and stp3/vad
# uniad: computed at a particular time (e.g., L2 distance between the predicted and ground truth future trajectory at time 3.0s)
# stp3: computed as the average up to a particular time (e.g., average L2 distance between the predicted and ground truth future trajectory up to 3.0s)
planning_evaluation_strategy = "uniad"  # uniad or stp3

### Occ args ###
occflow_grid_conf = {
    "xbound": [-50.0, 50.0, 0.5],
    "ybound": [-50.0, 50.0, 0.5],
    "zbound": [-10.0, 10.0, 20.0],
}

# Other settings
train_gt_iou_threshold = 0.3

dataset_type = "mmdet3d.CustomNuScenesE2EDataset"
data_root = "models/experimental/uniad/demo/data/nuscenes/"
info_root = "models/experimental/uniad/demo/data/infos/"
file_client_args = dict(backend="disk")
ann_file_train = info_root + f"nuscenes_infos_temporal_train.pkl"
ann_file_val = info_root + f"nuscenes_infos_temporal_val.pkl"
ann_file_test = info_root + f"nuscenes_infos_temporal_val.pkl"

test_pipeline = [
    dict(
        type="mmdet3d.CustomLoadMultiViewImageFromFilesInCeph",
        to_float32=True,
        file_client_args=file_client_args,
        img_root="models/experimental/uniad/demo/data/nuscenes",
    ),
    dict(type="mmdet3d.CustomNormalizeMultiviewImage", **img_norm_cfg),
    dict(type="mmdet3d.CustomPadMultiViewImage", size_divisor=32),
    dict(
        type="mmdet3d.CustomLoadAnnotations3D_E2E",
        with_bbox_3d=False,
        with_label_3d=False,
        with_attr_label=False,
        with_future_anns=True,
        with_ins_inds_3d=False,
        ins_inds_add_1=True,  # ins_inds start from 1
    ),
    dict(
        type="mmdet3d.CustomGenerateOccFlowLabels",
        grid_conf=occflow_grid_conf,
        ignore_index=255,
        only_vehicle=True,
        filter_invisible=False,
    ),
    dict(
        type="mmdet3d.CustomMultiScaleFlipAug3D",
        img_scale=(640, 360),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type="mmdet3d.CustomDefaultFormatBundle3D", class_names=class_names, with_label=False),
            dict(
                type="mmdet3d.CustomCollect3D",
                keys=[
                    "img",
                    "timestamp",
                    "l2g_r_mat",
                    "l2g_t",
                    "gt_lane_labels",
                    "gt_lane_bboxes",
                    "gt_lane_masks",
                    "gt_segmentation",
                    "gt_instance",
                    "gt_centerness",
                    "gt_offset",
                    "gt_flow",
                    "gt_backward_flow",
                    "gt_occ_has_invalid_frame",
                    "gt_occ_img_is_valid",
                    # planning
                    "sdc_planning",
                    "sdc_planning_mask",
                    "command",
                ],
            ),
        ],
    ),
]


val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        file_client_args=file_client_args,
        data_root=data_root,
        ann_file=ann_file_val,
        pipeline=test_pipeline,
        patch_size=patch_size,
        canvas_size=canvas_size,
        modality=input_modality,
        test_mode=True,
        bev_size=(bev_h_, bev_w_),
        predict_steps=predict_steps,
        box_type_3d="LiDAR",
        use_nonlinear_optimizer=use_nonlinear_optimizer,
        past_steps=past_steps,
        fut_steps=fut_steps,
        classes=class_names,
        occ_n_future=occ_n_future_max,
        eval_mod=["det", "map", "track", "motion"],
    ),
)


val_evaluator = dict(
    type="NuScenesMetric",
    data_root=data_root,
    ann_file=data_root + "nuscenes_infos_val.pkl",
    metric="bbox",
    backend_args=backend_args,
)
test_evaluator = val_evaluator

vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(type="Det3DLocalVisualizer", vis_backends=vis_backends, name="visualizer")

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        file_client_args=file_client_args,
        data_root=data_root,
        ann_file=ann_file_train,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        patch_size=patch_size,
        canvas_size=canvas_size,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        predict_steps=predict_steps,
        past_steps=past_steps,
        fut_steps=fut_steps,
        use_nonlinear_optimizer=use_nonlinear_optimizer,
        occ_receptive_field=3,
        occ_n_future=occ_n_future_max,
        occ_filter_invalid_sample=False,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d="LiDAR",
    ),
    val=dict(
        type=dataset_type,
        file_client_args=file_client_args,
        data_root=data_root,
        ann_file=ann_file_val,
        pipeline=test_pipeline,
        patch_size=patch_size,
        canvas_size=canvas_size,
        bev_size=(bev_h_, bev_w_),
        predict_steps=predict_steps,
        past_steps=past_steps,
        fut_steps=fut_steps,
        use_nonlinear_optimizer=use_nonlinear_optimizer,
        classes=class_names,
        modality=input_modality,
        samples_per_gpu=1,
        eval_mod=["det", "map", "track", "motion"],
        occ_receptive_field=3,
        occ_n_future=occ_n_future_max,
        occ_filter_invalid_sample=False,
    ),
    test=dict(
        type=dataset_type,
        file_client_args=file_client_args,
        data_root=data_root,
        test_mode=True,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        patch_size=patch_size,
        canvas_size=canvas_size,
        bev_size=(bev_h_, bev_w_),
        predict_steps=predict_steps,
        past_steps=past_steps,
        fut_steps=fut_steps,
        occ_n_future=occ_n_future_max,
        use_nonlinear_optimizer=use_nonlinear_optimizer,
        classes=class_names,
        modality=input_modality,
        eval_mod=["det", "map", "track", "motion"],
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
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
total_epochs = 20
evaluation = dict(
    interval=4,
    pipeline=test_pipeline,
    planning_evaluation_strategy=planning_evaluation_strategy,
)
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)
log_config = dict(interval=10, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")])
checkpoint_config = dict(interval=1)
load_from = "ckpts/uniad_base_track_map.pth"

find_unused_parameters = True
