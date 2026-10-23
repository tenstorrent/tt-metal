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
from models.experimental.bevformer.model_config import FEEDFORWARD_CHANNELS
from models.experimental.bevformer.tests.common import PRESETS, img_metas_for_dataset
from models.experimental.bevformer.reference.encoder import BEVFormerLayer
from models.experimental.bevformer.reference.point_sampling_3d_2d import (
    bev_reference_points as reference_bev_points,
    generate_reference_points,
    point_sampling_3d_to_2d,
)
from models.experimental.bevformer.tests.test_utils import check_with_pcc
from models.experimental.bevformer.tt.model_preprocessing import create_bevformer_layer_parameters
from models.experimental.bevformer.tt.tt_encoder import TTBEVFormerLayer
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
@pytest.mark.parametrize(
    "bev_size",
    [
        pytest.param((100, 100), id="bev_100x100"),
        pytest.param((200, 200), id="bev_200x200"),
    ],
)
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

    config = PRESETS[config_name]

    bev_h, bev_w = bev_size
    num_queries = bev_h * bev_w
    embed_dims = config.embed_dims
    num_cams = config.num_cams
    num_levels = config.num_levels

    spatial_shapes = torch.tensor(config.spatial_shapes[:num_levels], dtype=torch.long)
    bev_shape = torch.tensor([[bev_h, bev_w]], dtype=torch.long)

    bev_query = torch.randn(batch_size, num_queries, embed_dims, dtype=torch.float32)
    bev_pos = torch.randn(batch_size, num_queries, embed_dims, dtype=torch.float32)
    key_length = sum(h * w for h, w in spatial_shapes.tolist())
    camera_features = torch.randn(num_cams, key_length, batch_size, embed_dims, dtype=torch.float32)

    img_metas = img_metas_for_dataset(config, batch_size)
    lidar2img = torch.stack([torch.tensor(meta["lidar2img"], dtype=torch.float32) for meta in img_metas])

    reference_points_3d = generate_reference_points(
        bev_h=bev_h,
        bev_w=bev_w,
        z_cfg=config.z_cfg,
        batch_size=batch_size,
        dtype=torch.float32,
    )
    ref_points_cam, ref_bev_mask = point_sampling_3d_to_2d(
        reference_points=reference_points_3d,
        pc_range=list(config.pc_range),
        lidar2img=lidar2img,
        img_metas=img_metas,
    )
    # Uploaded as the encoder uploads them.
    tt_points_cam = ttnn.from_torch(ref_points_cam, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_bev_mask = ttnn.from_torch(
        ref_bev_mask.to(torch.bfloat16), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    bev_reference_points = ttnn.from_torch(
        reference_points_3d[:, :, 0, :2].unsqueeze(2),
        device=device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
    )

    layer_kwargs = dict(
        embed_dims=embed_dims,
        num_heads=config.num_heads,
        num_levels=num_levels,
        num_points=config.num_points,
        num_cams=num_cams,
        feedforward_channels=FEEDFORWARD_CHANNELS,
    )

    ref_model = BEVFormerLayer(**layer_kwargs, tsa_num_points=config.tsa_num_points)
    ref_model.eval()
    # Without a previous BEV, upstream's encoder hands the self-attention each sample's grid twice.
    ref_2d = reference_bev_points(bev_h, bev_w, batch_size)
    ref_output = ref_model(
        bev_query,
        value=camera_features,
        bev_pos=bev_pos,
        ref_2d=torch.stack([ref_2d, ref_2d], 1).flatten(0, 1),
        bev_shape=bev_shape,
        spatial_shapes=spatial_shapes,
        reference_points_cam=ref_points_cam,
        bev_mask=ref_bev_mask,
        prev_bev=None,
    )

    tt_model = TTBEVFormerLayer(
        device=device,
        params=create_bevformer_layer_parameters(ref_model, device, dtype=ttnn.bfloat16),
        spatial_shapes=spatial_shapes,
        bev_shape=bev_shape,
        batch_first=True,
        temporal_num_points=config.tsa_num_points,
        **layer_kwargs,
    )

    tt_bev_query = ttnn.from_torch(bev_query, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    # The layer takes camera features batch-first; the encoder owns the permute.
    tt_value = ttnn.from_torch(
        camera_features.permute(2, 0, 1, 3), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    tt_bev_pos = ttnn.from_torch(bev_pos, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    rebatch_plan = build_rebatch_plan(tt_points_cam, tt_bev_mask, device)

    def op_fn():
        return tt_model(
            bev_query=tt_bev_query,
            value=tt_value,
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
