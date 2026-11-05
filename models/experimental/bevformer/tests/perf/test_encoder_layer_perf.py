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

No trace capture: the measured region dispatches the layer live, so read it as
end-to-end device time, not as a traced-replay figure. The rebatch plan, whose
count readback a capture region would reject, is built once outside it.
"""

import subprocess

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.experimental.bevformer.model_config import FEEDFORWARD_CHANNELS
from models.experimental.bevformer.tests.common import PRESETS, img_metas_for_dataset, sample_ego_shifts
from models.experimental.bevformer.reference.encoder import BEVFormerLayer
from models.experimental.bevformer.reference.point_sampling_3d_2d import (
    bev_reference_points as reference_bev_points,
    generate_reference_points,
    point_sampling_3d_to_2d,
)
from models.experimental.bevformer.tests.test_utils import check_with_pcc
from models.experimental.bevformer.tt import tt_ms_deformable_attention
from models.experimental.bevformer.tt.model_preprocessing import create_bevformer_layer_parameters
from models.experimental.bevformer.tt.tt_encoder import TTBEVFormerLayer
from models.experimental.bevformer.tt.tt_temporal_self_attention import shifted_reference_points
from models.experimental.bevformer.tt.tt_spatial_cross_attention import build_rebatch_plan

DEVICE_PERF_ITERS = 1


def _head_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


@torch.no_grad()
@pytest.mark.timeout(1200)
# Each upstream model at its own BEV grid; base also at 100x100.
@pytest.mark.parametrize(
    "config_name, bev_size",
    [
        pytest.param("nuscenes_base", (100, 100), id="bev_100x100-nuscenes_base"),
        pytest.param("nuscenes_base", (200, 200), id="bev_200x200-nuscenes_base"),
        pytest.param("nuscenes_small_upstream", (150, 150), id="bev_150x150-nuscenes_small_upstream"),
        pytest.param("nuscenes_tiny_upstream", (50, 50), id="bev_50x50-nuscenes_tiny_upstream"),
    ],
)
@pytest.mark.parametrize("batch_size", [1])
# The first frame has no previous BEV; every later one, the steady state, has one and an ego shift.
@pytest.mark.parametrize("frame", ["first", "previous_bev"])
@pytest.mark.parametrize("expected_pcc", [0.997])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32 * 1024}], indirect=True)
def test_bevformer_layer_perf(
    device,
    config_name,
    bev_size,
    batch_size,
    frame,
    expected_pcc,
    reset_seeds,
    ensure_gc,
    monkeypatch,
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
        layout=ttnn.ROW_MAJOR_LAYOUT,
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
    # Upstream's encoder hands the self-attention each sample's grid twice, the previous BEV's
    # shifted by the ego motion, and stacks the previous BEV with the encoder's input queries.
    ref_2d = reference_bev_points(bev_h, bev_w, batch_size)
    prev_bev = shift = None
    shifted_ref_2d = ref_2d
    if frame == "previous_bev":
        prev_bev = torch.stack([torch.randn_like(bev_query), bev_query], 1).flatten(0, 1)
        shift = sample_ego_shifts(batch_size)
        shifted_ref_2d = ref_2d + shift[:, None, None, :]
    ref_output = ref_model(
        bev_query,
        value=camera_features,
        bev_pos=bev_pos,
        ref_2d=torch.stack([shifted_ref_2d, ref_2d], 1).flatten(0, 1),
        bev_shape=bev_shape,
        spatial_shapes=spatial_shapes,
        reference_points_cam=ref_points_cam,
        bev_mask=ref_bev_mask,
        prev_bev=prev_bev,
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
    # The layer takes camera features batch-first, [B, num_cams, H*W, C].
    tt_value = ttnn.from_torch(
        camera_features.permute(2, 0, 1, 3), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    tt_bev_pos = ttnn.from_torch(bev_pos, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    rebatch_plan = build_rebatch_plan(tt_points_cam, tt_bev_mask, device)

    tt_prev_bev = None
    layer_reference_points = bev_reference_points
    if frame == "previous_bev":
        tt_prev_bev = ttnn.from_torch(prev_bev, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        # Built once per frame, as the encoder does.
        layer_reference_points = shifted_reference_points(
            bev_reference_points,
            ttnn.from_torch(
                shift.view(batch_size, 1, 1, 2), device=device, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT
            ),
        )

    def op_fn():
        return tt_model(
            bev_query=tt_bev_query,
            value=tt_value,
            bev_pos=tt_bev_pos,
            prev_bev=tt_prev_bev,
            reference_points_cam=tt_points_cam,
            bev_mask=tt_bev_mask,
            bev_reference_points=layer_reference_points,
            rebatch_plan=rebatch_plan,
        )

    # The layer's speed rests on both fused ops keeping their value in L1; a DRAM fallback only
    # logs a warning, once per shape and process, so catch it here with the record cleared.
    monkeypatch.setattr(tt_ms_deformable_attention, "_warned_dram_fallbacks", set())
    fallbacks = []
    sink = logger.add(
        lambda message: fallbacks.append(message),
        filter=lambda record: record["message"].startswith(tt_ms_deformable_attention.DRAM_FALLBACK_PREFIX),
    )
    # Doubles as the warmup: this call compiles the kernels and fills the program
    # cache, so the signposted iterations already run at steady state.
    tt_output = op_fn()
    logger.remove(sink)
    tt_output_torch = ttnn.to_torch(tt_output, dtype=torch.float32)
    passed, message = check_with_pcc(ref_output, tt_output_torch, expected_pcc)
    assert passed, f"PCC check failed: {message}"
    logger.info(f"PCC gate: {message}")
    assert not fallbacks, f"fused_msda fell back to DRAM: {fallbacks}"
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
