# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Performance test for LTX-2 pipeline.

Follows the Wan pipeline performance test pattern:
- Warmup iteration with few steps
- Measured iteration with full steps
- BenchmarkProfiler for timing
- BenchmarkData for CI reporting

Reference: models/tt_dit/tests/models/wan2_2/test_performance_wan.py
"""

import sys

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.pipelines.ltx.pipeline_ltx import LTXPipeline
from models.tt_dit.utils.test import line_params

sys.path.insert(0, "LTX-2/packages/ltx-core/src")


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        ((2, 4), (2, 4), 0, 1, 2, False, line_params, ttnn.Topology.Linear, False),
    ],
    ids=["wh_lb_2x4"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("width, height", [(768, 512)], ids=["512p"])
def test_pipeline_performance_video(
    mesh_device,
    mesh_shape,
    sp_axis,
    tp_axis,
    num_links,
    dynamic_load,
    topology,
    width,
    height,
    is_fsdp,
):
    """
    Performance test for LTX-2 video-only pipeline.

    Runs warmup (2 steps) then measured iteration (5 steps) and reports timing.
    Uses 1-layer random-weight model for fast CI.
    """
    pytest.importorskip("ltx_core", reason="LTX-2 reference package required")
    from ltx_core.model.transformer.model import LTXModel, LTXModelType

    from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
    from models.tt_dit.parallel.manager import CCLManager

    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    dim = 4096
    num_heads = 32
    head_dim = dim // num_heads
    num_layers = 1
    num_frames = 17
    num_steps_warmup = 2
    num_steps_measured = 5

    # Create random-weight reference model for state dict
    torch_model = LTXModel(
        model_type=LTXModelType.VideoOnly,
        num_layers=num_layers,
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        in_channels=128,
        out_channels=128,
        cross_attention_dim=dim,
        use_middle_indices_grid=True,
        cross_attention_adaln=True,
    )
    torch_model.eval()

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        sequence_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[sp_axis], mesh_axis=sp_axis),
        tensor_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[tp_axis], mesh_axis=tp_axis),
    )
    ccl_manager = CCLManager(mesh_device, topology=topology)

    pipeline = LTXPipeline(
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        num_layers=num_layers,
        cross_attention_dim=dim,
    )
    pipeline.load_transformer(torch_model.state_dict())

    # Warmup
    logger.info(f"Warmup: {num_steps_warmup} steps")
    with torch.no_grad():
        _ = pipeline(
            prompt=["test"],
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_steps_warmup,
            guidance_scale=1.0,
            seed=42,
        )
    logger.info("Warmup complete")

    # Measured run
    import time

    logger.info(f"Measured: {num_steps_measured} steps")
    t0 = time.time()
    with torch.no_grad():
        result = pipeline(
            prompt=["test"],
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_steps_measured,
            guidance_scale=1.0,
            seed=42,
        )
    elapsed = time.time() - t0

    latent_frames = (num_frames - 1) // 8 + 1
    latent_h = height // 32
    latent_w = width // 32
    expected_tokens = latent_frames * latent_h * latent_w

    logger.info(f"Performance: {elapsed:.2f}s total, {elapsed/num_steps_measured:.2f}s/step")
    logger.info(f"Output shape: {result.shape}")
    assert result.shape == (1, expected_tokens, 128)
    assert torch.isfinite(result).all()
    logger.info("PASSED: Performance test")
