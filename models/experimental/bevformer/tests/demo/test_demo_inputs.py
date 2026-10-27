# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

from models.experimental.bevformer.config import builtin_config_path, load_builtin_config, load_model_config
from models.experimental.bevformer.config.convert_upstream import convert_upstream_config, variant_from_source
from models.experimental.bevformer.config.loader import _github_raw_url
from models.experimental.bevformer.demo.checkpoint import _mmdet3d_key
from models.experimental.bevformer.demo.dataset import SequenceManifest


def test_normalized_config(tmp_path, expect_error):
    tiny = load_builtin_config("tiny")
    assert tiny.raw_image_size == (800, 450)
    assert tiny.resnet_depth == 50
    assert tiny.bev_size == 50
    assert tiny.encoder_num_layers == 3

    path = tmp_path / "model.json"
    raw = json.loads(builtin_config_path("small").read_text())
    raw["variant"] = "unknown"
    path.write_text(json.dumps(raw))
    with expect_error(ValueError, "variant"):
        load_model_config(path)

    path.write_text("model = dict(type='BEVFormer')")
    with expect_error(ValueError, "expected a normalized BEVFormer JSON config"):
        load_model_config(path)

    with expect_error(ValueError, "SHA-256 mismatch"):
        load_model_config(builtin_config_path("tiny"), sha256="0" * 64)
    tiny_digest = hashlib.sha256(builtin_config_path("tiny").read_bytes()).hexdigest()
    assert load_model_config(builtin_config_path("tiny"), sha256=tiny_digest) == tiny


def test_github_config_url_normalization(expect_error):
    assert _github_raw_url("https://github.com/org/repo/blob/commit/config/model.json") == (
        "https://raw.githubusercontent.com/org/repo/commit/config/model.json"
    )
    with expect_error(ValueError, "one JSON file"):
        _github_raw_url("https://github.com/org/repo/tree/commit/config")
    assert variant_from_source("https://example.test/bevformer_small.py") == "small"


def test_safe_upstream_config_conversion():
    source = """
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
class_names = ['car', 'truck']
model = dict(
    img_backbone=dict(depth=50, out_indices=(3,), style='pytorch'),
    img_neck=dict(num_outs=1),
    pts_bbox_head=dict(
        bev_h=50, num_query=900,
        transformer=dict(
            embed_dims=256,
            encoder=dict(num_layers=3, num_points_in_pillar=4, transformerlayers=dict(
                feedforward_channels=512,
                attn_cfgs=[dict(type='TemporalSelfAttention'), dict(
                    type='SpatialCrossAttention', deformable_attention=dict(num_points=8))])),
            decoder=dict(num_layers=6, transformerlayers=dict(
                attn_cfgs=[dict(num_heads=8), dict(type='CustomMSDeformableAttention')]))),
        bbox_coder=dict(post_center_range=[-61.2, -61.2, -10, 61.2, 61.2, 10], max_num=300)))
"""
    config = convert_upstream_config(source, "tiny")
    assert config.bev_size == 50
    assert config.class_names == ("car", "truck")
    assert config.color_order == "RGB"


def test_mmdet3d_checkpoint_ownership_mapping():
    assert _mmdet3d_key("module.img_backbone.conv1.weight") == "img_backbone.conv1.weight"
    assert (
        _mmdet3d_key("pts_bbox_head.transformer.encoder.layers.0.norms.0.weight")
        == "transformer.encoder.layers.0.norms.0.weight"
    )
    assert (
        _mmdet3d_key("pts_bbox_head.transformer.decoder.layers.0.norms.0.weight")
        == "head.decoder.layers.0.norms.0.weight"
    )
    assert _mmdet3d_key("pts_bbox_head.bev_embedding.weight") == "bev_embedding.weight"


def test_manifest_selects_preparation_and_output_frames(tmp_path, expect_error):
    config = load_builtin_config("base")
    image_names = {camera: f"{camera}.jpg" for camera in config.camera_order}
    for name in image_names.values():
        (tmp_path / name).touch()
    identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    frames = []
    for index in range(4):
        frames.append(
            {
                "token": f"frame-{index}",
                "scene_token": "scene",
                "timestamp": 30 - index,
                "cameras": {
                    camera: {"image": image_names[camera], "lidar2img": identity} for camera in config.camera_order
                },
                "can_bus": [0] * 18,
            }
        )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"camera_order": list(config.camera_order), "frames": frames}))

    manifest = SequenceManifest.load(path, camera_order=config.camera_order, can_bus_dims=config.can_bus_dims)
    selected = manifest.sequence(num_output_frames=2, start_frame=1)
    assert [frame.timestamp for frame in selected] == [28, 29, 30]

    with expect_error(ValueError, "needs 5"):
        manifest.sequence(num_output_frames=4)

    wrong_order = list(config.camera_order)
    wrong_order[0], wrong_order[1] = wrong_order[1], wrong_order[0]
    with expect_error(ValueError, "camera_order"):
        SequenceManifest.load(path, camera_order=wrong_order, can_bus_dims=config.can_bus_dims)
