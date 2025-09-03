from models.experimental.vadv2.demo.utils.builder import build_dataset

test = dict(
    type="VADCustomNuScenesDataset",
    data_root="models/experimental/vadv2/data/nuscenes/",
    ann_file="models/experimental/vadv2/data/nuscenes/vad_nuscenes_infos_temporal_val.pkl",
    pipeline=[
        dict(type="LoadMultiViewImageFromFiles", to_float32=True),
        dict(
            type="LoadPointsFromFile", coord_type="LIDAR", load_dim=5, use_dim=5, file_client_args=dict(backend="disk")
        ),
        dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
        dict(type="CustomObjectRangeFilter", point_cloud_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]),
        dict(
            type="CustomObjectNameFilter",
            classes=[
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
            ],
        ),
        dict(type="NormalizeMultiviewImage", mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
        dict(
            type="MultiScaleFlipAug3D",
            img_scale=(1600, 900),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(type="RandomScaleImageMultiViewImage", scales=[0.4]),
                dict(type="PadMultiViewImage", size_divisor=32),
                dict(
                    type="CustomDefaultFormatBundle3D",
                    class_names=[
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
                    ],
                    with_label=False,
                    with_ego=True,
                ),
                dict(
                    type="CustomCollect3D",
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
    ],
    classes=[
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
    ],
    modality=dict(use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=True),
    test_mode=True,
    box_type_3d="LiDAR",
    pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
    bev_size=(100, 100),
    samples_per_gpu=1,
    map_classes=["divider", "ped_crossing", "boundary"],
    map_ann_file="models/experimental/vadv2/data/nuscenes/nuscenes_map_anns_val.json",
    map_fixed_ptsnum_per_line=20,
    map_eval_use_same_gt_sample_num_flag=True,
    use_pkl_result=True,
    custom_eval_version="vad_nusc_detection_cvpr_2019",
)


def test_demo(reset_seeds):
    dataset = build_dataset(test)

    print("----------------------------------------------", dataset)

    return dataset
