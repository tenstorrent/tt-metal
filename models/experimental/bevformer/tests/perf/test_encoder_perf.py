# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tracy harness for the BEVFormer encoder device path.

Same inputs as ``test_bevformer_encoder_forward``: PCC gate, warmup, then
signposted iterations so the report covers already-compiled, already-dispatched
programs.

Camera geometry comes from the dataset's fixed rig, not from random matrices.
``lidar2img`` decides ``bev_mask`` and therefore the spatial-cross-attention
rebatch length, which sizes every spatial-path tensor; drawing it from the RNG
would make the measured workload depend on the seed and on how many tensors were
allocated before it.

No trace capture. ``TTSpatialCrossAttention.forward`` calls ``ttnn.to_torch`` on
``bev_mask`` and on its rebatch indices, and the host result decides shapes for
the ops that follow; reads and writes both TT_FATAL inside a capture region.
Until that host round-trip is gone the signposted region carries host dispatch,
so read it as end-to-end device time, not as a traced-replay figure.
"""

import subprocess

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.experimental.bevformer.config.encoder_config import get_preset_config, img_metas_for_dataset
from models.experimental.bevformer.reference.encoder import BEVFormerEncoder
from models.experimental.bevformer.tests.test_utils import check_with_pcc
from models.experimental.bevformer.tt.model_preprocessing import create_bevformer_encoder_parameters
from models.experimental.bevformer.tt.tt_encoder import TTBEVFormerEncoder

PERF_WARMUP_ITERS = 1
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
@pytest.mark.parametrize("num_layers", [6])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("expected_pcc", [0.997])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32 * 1024}], indirect=True)
def test_bevformer_encoder_perf(
    device,
    config_name,
    bev_size,
    num_layers,
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

    bev_h, bev_w = bev_size
    num_queries = bev_h * bev_w
    embed_dims = model_config.embed_dims
    num_cams = dataset_config.num_cams
    num_levels = model_config.num_levels

    spatial_shapes = torch.tensor(dataset_config.spatial_shapes[:num_levels], dtype=torch.long)
    level_start_index = config.get_level_start_index()[:num_levels]

    bev_query = torch.randn(batch_size, num_queries, embed_dims, dtype=torch.float32)
    bev_pos = torch.randn(batch_size, num_queries, embed_dims, dtype=torch.float32)

    key_length = sum(h * w for h, w in spatial_shapes.tolist())
    camera_features = torch.randn(num_cams, key_length, batch_size, embed_dims, dtype=torch.float32)

    img_metas = img_metas_for_dataset(dataset_config, batch_size)

    encoder_kwargs = config.get_encoder_kwargs()
    encoder_kwargs.update({"num_layers": num_layers, "batch_first": True, "return_intermediate": False})

    ref_model = BEVFormerEncoder(**encoder_kwargs)
    ref_model.eval()
    ref_output = ref_model(
        bev_query=bev_query,
        key=camera_features,
        value=camera_features,
        bev_h=bev_h,
        bev_w=bev_w,
        bev_pos=bev_pos,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        prev_bev=None,
        img_metas=img_metas,
    )

    tt_model = TTBEVFormerEncoder(
        device=device,
        params=create_bevformer_encoder_parameters(torch_model=ref_model, device=device, dtype=ttnn.bfloat16),
        **encoder_kwargs,
    )

    # Uploaded once so the profiled iterations measure the encoder, not the transfer
    tt_bev_query = ttnn.from_torch(bev_query, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_bev_pos = ttnn.from_torch(bev_pos, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_camera_features = ttnn.from_torch(
        camera_features, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    tt_level_start_index = ttnn.from_torch(
        level_start_index, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )

    def op_fn():
        return tt_model(
            bev_query=tt_bev_query,
            key=tt_camera_features,
            value=tt_camera_features,
            bev_pos=tt_bev_pos,
            bev_h=bev_h,
            bev_w=bev_w,
            spatial_shapes=spatial_shapes,
            level_start_index=tt_level_start_index,
            prev_bev=None,
            img_metas=img_metas,
        )

    tt_output = op_fn()
    tt_output_torch = ttnn.to_torch(tt_output, dtype=torch.float32)
    passed, message = check_with_pcc(ref_output, tt_output_torch, expected_pcc)
    assert passed, f"PCC check failed: {message}"
    logger.info(f"PCC gate: {message}")
    ttnn.deallocate(tt_output)

    for _ in range(PERF_WARMUP_ITERS):
        out = op_fn()
        ttnn.synchronize_device(device)
        ttnn.deallocate(out)

    ttnn.synchronize_device(device)
    outputs = []
    signpost("start")
    for _ in range(DEVICE_PERF_ITERS):
        outputs.append(op_fn())
        ttnn.synchronize_device(device)
    signpost("stop")

    for out in outputs:
        ttnn.deallocate(out)
