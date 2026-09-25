# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tracy harness for a single BEVFormer encoder layer.

Same shape as ``test_bevformer_encoder_perf``, one layer instead of six: a PCC
gate that doubles as the warmup, then signposted iterations so the report covers
already-compiled, already-dispatched programs. The reference points and the camera projection
are built once outside the measured region — the encoder does the same, so
what is measured here is the per-layer cost the encoder repeats.

Camera geometry comes from the dataset's fixed rig, not from random matrices.
``lidar2img`` decides ``bev_mask`` and therefore the spatial-cross-attention
rebatch length, which sizes every spatial-path tensor; drawing it from the RNG
would make the measured workload depend on the seed and on allocation order.

No trace capture. ``build_rebatch_plan`` still calls ``ttnn.to_torch`` on
``bev_mask`` and the host result decides ``max_len`` / tensor shapes for the
ops that follow; reads and writes both TT_FATAL inside a capture region.
Until that host round-trip is gone the signposted region carries host dispatch,
so read it as end-to-end device time, not as a traced-replay figure.
"""

import subprocess

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.experimental.bevformer.config.encoder_config import get_preset_config
from models.experimental.bevformer.tests.camera_rig import img_metas_for_dataset
from models.experimental.bevformer.reference.encoder import BEVFormerLayer
from models.experimental.bevformer.reference.point_sampling_3d_2d import (
    generate_reference_points,
    point_sampling_3d_to_2d,
)
from models.experimental.bevformer.tests.test_utils import check_with_pcc
from models.experimental.bevformer.tt.model_preprocessing import preprocess_bevformer_layer_parameters
from models.experimental.bevformer.tt.tt_encoder import TTBEVFormerLayer
from models.experimental.bevformer.tt.tt_point_sampling_3d_2d import point_sampling_3d_to_2d_ttnn
from models.experimental.bevformer.tt.tt_spatial_cross_attention import build_rebatch_plan

DEVICE_PERF_ITERS = 1


def _head_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


@torch.no_grad()
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("config_name", ["nuscenes_base"])
@pytest.mark.parametrize("bev_size", [(100, 100)])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("expected_pcc", [0.997])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32 * 1024}], indirect=True)
def test_bevformer_layer_perf(
    device,
    config_name,
    bev_size,
    batch_size,
    expected_pcc,
    reset_seeds,
    ensure_gc,
):
    logger.info(f"device-perf run of commit {_head_sha()}")

    config = get_preset_config(config_name)
    assert config is not None, f"Configuration '{config_name}' not found"

    dataset_config = config.dataset_config
    model_config = config.model_config
    encoder_kwargs = config.get_encoder_kwargs()

    bev_h, bev_w = bev_size
    num_queries = bev_h * bev_w
    embed_dims = model_config.embed_dims
    num_cams = dataset_config.num_cams
    num_levels = model_config.num_levels

    spatial_shapes = torch.tensor(dataset_config.spatial_shapes[:num_levels], dtype=torch.long)
    level_start_index = config.get_level_start_index()[:num_levels]
    bev_shape = torch.tensor([[bev_h, bev_w]], dtype=torch.long)

    bev_query = torch.randn(batch_size, num_queries, embed_dims, dtype=torch.float32)
    bev_pos = torch.randn(batch_size, num_queries, embed_dims, dtype=torch.float32)
    key_length = sum(h * w for h, w in spatial_shapes.tolist())
    camera_features = torch.randn(num_cams, key_length, batch_size, embed_dims, dtype=torch.float32)

    img_metas = img_metas_for_dataset(dataset_config, batch_size)
    lidar2img = torch.stack([torch.tensor(meta["lidar2img"], dtype=torch.float32) for meta in img_metas])

    reference_points_3d = generate_reference_points(
        bev_h=bev_h,
        bev_w=bev_w,
        z_cfg=encoder_kwargs["z_cfg"],
        batch_size=batch_size,
        dtype=torch.float32,
    )
    ref_points_cam, ref_bev_mask = point_sampling_3d_to_2d(
        reference_points=reference_points_3d,
        pc_range=encoder_kwargs["pc_range"],
        lidar2img=lidar2img,
        img_metas=img_metas,
    )
    tt_points_cam, tt_bev_mask = point_sampling_3d_to_2d_ttnn(
        reference_points=reference_points_3d,
        pc_range=encoder_kwargs["pc_range"],
        lidar2img=lidar2img,
        img_metas=img_metas,
        device=device,
    )
    bev_reference_points = ttnn.from_torch(
        reference_points_3d[:, :, 0, :2].unsqueeze(2),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    layer_kwargs = {
        key: encoder_kwargs[key]
        for key in (
            "embed_dims",
            "num_heads",
            "num_levels",
            "num_points",
            "num_cams",
            "feedforward_channels",
            "batch_first",
        )
        if key in encoder_kwargs
    }
    layer_kwargs["batch_first"] = True

    ref_model = BEVFormerLayer(**layer_kwargs)
    ref_model.eval()
    ref_output = ref_model(
        bev_query=bev_query,
        key=camera_features,
        value=camera_features,
        bev_pos=bev_pos,
        spatial_shapes=spatial_shapes,
        bev_shape=bev_shape,
        level_start_index=level_start_index,
        prev_bev=None,
        reference_points_3d=reference_points_3d,
        reference_points_cam=ref_points_cam,
        bev_mask=ref_bev_mask,
    )

    tt_model = TTBEVFormerLayer(
        device=device,
        params=preprocess_bevformer_layer_parameters(ref_model, device=device, dtype=ttnn.bfloat16),
        spatial_shapes=spatial_shapes,
        bev_shape=bev_shape,
        **layer_kwargs,
    )

    tt_bev_query = ttnn.from_torch(bev_query, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_key = ttnn.from_torch(camera_features, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_bev_pos = ttnn.from_torch(bev_pos, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    rebatch_plan = build_rebatch_plan(tt_points_cam, tt_bev_mask, embed_dims, device)

    def op_fn():
        return tt_model(
            bev_query=tt_bev_query,
            key=tt_key,
            value=tt_key,
            bev_pos=tt_bev_pos,
            prev_bev=None,
            reference_points_cam=tt_points_cam,
            bev_mask=tt_bev_mask,
            bev_reference_points=bev_reference_points,
            rebatch_plan=rebatch_plan,
        )

    # Doubles as the warmup: this call compiles the kernels and fills the program
    # cache, so the signposted iterations already run at steady state.
    tt_output = op_fn()
    tt_output_torch = ttnn.to_torch(tt_output, dtype=torch.float32)
    passed, message = check_with_pcc(ref_output, tt_output_torch, expected_pcc)
    assert passed, f"PCC check failed: {message}"
    logger.info(f"PCC gate: {message}")
    ttnn.deallocate(tt_output)

    ttnn.synchronize_device(device)
    outputs = []
    # Drains and resets the device profiler buffers so the signposted region starts
    # from empty; the PCC call's markers would otherwise eat into the same budget.
    ttnn.ReadDeviceProfiler(device)
    signpost("start")
    for _ in range(DEVICE_PERF_ITERS):
        outputs.append(op_fn())
        ttnn.synchronize_device(device)
    signpost("stop")

    for out in outputs:
        ttnn.deallocate(out)
