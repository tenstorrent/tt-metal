# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Stress and power tests: one Wan2.2-T2V-14B transformer block, run in a loop to hold the devices under a
model-like load. Weights and inputs are random, so no checkpoint download is needed. Outputs are not checked.

Not run in CI. Examples:
    pytest models/tt_dit/tests/models/wan2_2/test_stress_wan.py -k "test_wan_block_stress_duration and line_bh_4x8 and 1000s"
    pytest models/tt_dit/tests/models/wan2_2/test_stress_wan.py -k "test_wan_block_stress_iterations and bh_qb_2 and 720p and iter100_"
"""

import time

import pytest
import torch
from diffusers import WanTransformer3DModel as TorchWanTransformer3DModel
from loguru import logger
from tqdm import trange

import ttnn

from ....models.transformers.wan2_2.transformer_wan import WanTransformerBlock
from ....utils.mochi import get_rot_transformation_mat, stack_cos_sin
from ....utils.padding import pad_vision_seq_parallel
from ....utils.tensor import bf16_tensor, bf16_tensor_2dshard, from_torch
from ....utils.test import line_params, ring_params, skip_if_unsupported_num_links
from .test_transformer_wan import (
    CROSS_ATTN_NORM,
    DIM,
    EPS,
    FFN_DIM,
    HEAD_DIM,
    NUM_HEADS,
    PATCH_SIZE,
    _make_ccl_manager,
    _make_parallel_config,
)

MESH_PARAMS_ARGNAMES = ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp")


def _make_block_runner(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, B, T, H, W, prompt_seq_len):
    """Build one TT Wan block with random weights and random inputs. Returns a function that runs the block once."""
    skip_if_unsupported_num_links(mesh_device, num_links)

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    parallel_config = _make_parallel_config(mesh_device, sp_axis, tp_axis)
    ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)

    p_t, p_h, p_w = PATCH_SIZE
    spatial_seq_len = (T // p_t) * (H // p_h) * (W // p_w)

    torch.manual_seed(0)
    torch_model = TorchWanTransformer3DModel(
        num_attention_heads=NUM_HEADS,
        attention_head_dim=HEAD_DIM,
        ffn_dim=FFN_DIM,
        num_layers=1,
        cross_attn_norm=CROSS_ATTN_NORM,
        eps=EPS,
    ).blocks[0]
    torch_model.eval()

    tt_model = WanTransformerBlock(
        dim=DIM,
        ffn_dim=FFN_DIM,
        num_heads=NUM_HEADS,
        cross_attention_norm=CROSS_ATTN_NORM,
        eps=EPS,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())
    del torch_model

    spatial_input = torch.randn((B, spatial_seq_len, DIM), dtype=torch.float32)
    prompt_input = torch.randn((B, prompt_seq_len, DIM), dtype=torch.float32)
    temb_input = torch.randn((B, 6, DIM), dtype=torch.float32)
    rope_cos = torch.randn(B, spatial_seq_len, 1, HEAD_DIM // 2)
    rope_sin = torch.randn(B, spatial_seq_len, 1, HEAD_DIM // 2)
    torch_rope_cos, torch_rope_sin = stack_cos_sin(rope_cos, rope_sin)

    rope_cos_stack = torch_rope_cos.permute(0, 2, 1, 3)
    rope_sin_stack = torch_rope_sin.permute(0, 2, 1, 3)

    spatial_padded = pad_vision_seq_parallel(spatial_input.unsqueeze(0), num_devices=sp_factor)
    rope_cos_padded = pad_vision_seq_parallel(rope_cos_stack, num_devices=sp_factor)
    rope_sin_padded = pad_vision_seq_parallel(rope_sin_stack, num_devices=sp_factor)

    tt_spatial = bf16_tensor_2dshard(spatial_padded, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3})
    tt_prompt = bf16_tensor(prompt_input.unsqueeze(0), device=mesh_device)
    tt_temb = from_torch(temb_input.unsqueeze(0), device=mesh_device, dtype=ttnn.float32, mesh_axes=[..., tp_axis])
    tt_rope_cos = from_torch(rope_cos_padded, device=mesh_device, dtype=ttnn.float32, mesh_axes=[..., sp_axis, None])
    tt_rope_sin = from_torch(rope_sin_padded, device=mesh_device, dtype=ttnn.float32, mesh_axes=[..., sp_axis, None])
    tt_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)

    logger.info(
        f"Running TT model with spatial shape {tt_spatial.shape}, prompt shape {tt_prompt.shape}, rope_cos shape {tt_rope_cos.shape}, rope_sin shape {tt_rope_sin.shape}"
    )

    def run_block():
        return tt_model(
            spatial_1BND=tt_spatial,
            prompt_1BLP=tt_prompt,
            temb_1BTD=tt_temb,
            N=spatial_seq_len,
            rope_cos=tt_rope_cos,
            rope_sin=tt_rope_sin,
            trans_mat=tt_trans_mat,
        )

    return run_block


@pytest.mark.timeout(0)  # Disable pytest timeout
@pytest.mark.parametrize(
    MESH_PARAMS_ARGNAMES,
    [
        pytest.param((4, 8), 1, 0, 2, ring_params, ttnn.Topology.Ring, False, id="ring_bh_4x8sp1tp0"),
        pytest.param((4, 8), 1, 0, 2, line_params, ttnn.Topology.Linear, False, id="line_bh_4x8sp1tp0"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    ("B", "T", "H", "W", "prompt_seq_len"),
    [
        pytest.param(1, 21, 90, 160, 512, id="14b-720p"),
    ],
)
@pytest.mark.parametrize("test_duration_s", [100, 200, 500, 1000], ids=["100s", "200s", "500s", "1000s"])
def test_wan_block_stress_duration(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
    is_fsdp: bool,
    B: int,
    T: int,
    H: int,
    W: int,
    prompt_seq_len: int,
    test_duration_s: int,
) -> None:
    """Run the block for test_duration_s seconds, 40 calls (one model step) between device syncs."""
    run_block = _make_block_runner(
        mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, B, T, H, W, prompt_seq_len
    )
    run_block()
    logger.info("Warmup iteration complete")

    start = time.time()
    itr = 0
    while time.time() - start < test_duration_s:
        ttnn.synchronize_device(mesh_device)
        logger.info(f"Running iteration {itr}, time elapsed: {time.time() - start:.3f} seconds")
        for _ in range(40):
            run_block()
        itr += 1
    ttnn.synchronize_device(mesh_device)

    logger.info(f"Completed test after: {time.time() - start} seconds")


@pytest.mark.timeout(0)  # Disable pytest timeout
@pytest.mark.parametrize(
    MESH_PARAMS_ARGNAMES,
    [
        pytest.param((1, 2), 0, 1, 2, line_params, ttnn.Topology.Linear, True, id="p300"),
        pytest.param((2, 2), 0, 1, 2, line_params, ttnn.Topology.Linear, False, id="bh_qb_2"),
        pytest.param((4, 8), 1, 0, 2, line_params, ttnn.Topology.Linear, False, id="bh_glx"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    ("B", "T", "H", "W", "prompt_seq_len"),
    [
        pytest.param(1, 21, 60, 104, 118, id="14b-480p"),
        pytest.param(1, 21, 90, 160, 118, id="14b-720p"),
    ],
)
@pytest.mark.parametrize(
    "num_iterations",
    [0, 1, 10, 50, 100, 1000, 10000, 100000],
    ids=["iter0_", "iter1_", "iter10_", "iter50_", "iter100_", "iter1000_", "iter10000_", "iter100000_"],
)
def test_wan_block_stress_iterations(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
    is_fsdp: bool,
    B: int,
    T: int,
    H: int,
    W: int,
    prompt_seq_len: int,
    num_iterations: int,
) -> None:
    """Run the block num_iterations times 10 calls, syncing after each 10 so tqdm shows any slowdown.
    iter0_ runs only the warmup call, for a single-iteration performance measurement."""
    run_block = _make_block_runner(
        mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, B, T, H, W, prompt_seq_len
    )
    logger.info("Running warmup")
    run_block()
    ttnn.synchronize_device(mesh_device)

    inner_loop_num_iters = 10
    logger.info(f"Running {num_iterations} iterations")
    for _ in trange(num_iterations, desc="WAN iterations", unit="iter"):
        for _ in range(inner_loop_num_iters):
            run_block()
        ttnn.synchronize_device(mesh_device)
