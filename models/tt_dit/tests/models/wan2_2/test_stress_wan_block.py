# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Traced stress loop over one Wan2.2-T2V-14B transformer block with real block-0 weights.

Reports ms/iteration per window of `TT_DIT_STRESS_WINDOW_ITERS` (default 100) replays; run under
different `TT_METAL_TDP_LIMIT_WATTS` values to see sustained throughput at each power cap.
"""

import json
from pathlib import Path

import pytest
import torch
from huggingface_hub import snapshot_download
from loguru import logger
from safetensors import safe_open

import ttnn

from ....models.transformers.wan2_2.transformer_wan import WanTransformerBlock
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.mochi import get_rot_transformation_mat, stack_cos_sin
from ....utils.padding import pad_vision_seq_parallel
from ....utils.tensor import bf16_tensor, bf16_tensor_2dshard, from_torch
from ....utils.test import ring_params_req_exact_devices, skip_if_unsupported_num_links
from ....utils.trace_stress import run_traced_stress

MODEL_NAME = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
DIM = 5120
FFN_DIM = 13824
NUM_HEADS = 40
HEAD_DIM = DIM // NUM_HEADS
PATCH_SIZE = (1, 2, 2)
CROSS_ATTN_NORM = True
EPS = 1e-6
TRACE_REGION_SIZE = 200_000_000


def _block0_state_dict() -> dict[str, torch.Tensor]:
    """Block 0 of the high-noise transformer, read straight from the safetensors shards."""
    directory = Path(snapshot_download(MODEL_NAME, allow_patterns=["transformer/*"])) / "transformer"
    weight_map = json.loads((directory / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    prefix = "blocks.0."
    by_file: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        if key.startswith(prefix):
            by_file.setdefault(shard, []).append(key)
    state = {}
    for shard, keys in by_file.items():
        with safe_open(directory / shard, framework="pt") as handle:
            for key in keys:
                state[key[len(prefix) :]] = handle.get_tensor(key).to(torch.float32)
    return state


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param(
            (4, 8),
            1,
            0,
            2,
            {**ring_params_req_exact_devices, "trace_region_size": TRACE_REGION_SIZE},
            ttnn.Topology.Ring,
            False,
            id="4x8sp1tp0nl2_ring_is_fsdp0",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    ("B", "T", "H", "W", "prompt_seq_len"),
    [
        pytest.param(1, 21, 90, 160, 512, id="14b-720p"),
        pytest.param(1, 21, 60, 104, 512, id="14b-480p"),
    ],
)
def test_stress_wan_block(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    B: int,
    T: int,
    H: int,
    W: int,
    prompt_seq_len: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
    reset_seeds,
    request,
) -> None:
    skip_if_unsupported_num_links(mesh_device, num_links)

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=None,
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)

    p_t, p_h, p_w = PATCH_SIZE
    spatial_seq_len = (T // p_t) * (H // p_h) * (W // p_w)

    tt_block = WanTransformerBlock(
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
    tt_block.load_torch_state_dict(_block0_state_dict())

    spatial_input = torch.randn((B, spatial_seq_len, DIM), dtype=torch.float32)
    prompt_input = torch.randn((B, prompt_seq_len, DIM), dtype=torch.float32)
    temb_input = torch.randn((B, 6, DIM), dtype=torch.float32)
    rope_cos = torch.randn(B, spatial_seq_len, 1, HEAD_DIM // 2)
    rope_sin = torch.randn(B, spatial_seq_len, 1, HEAD_DIM // 2)
    torch_rope_cos, torch_rope_sin = stack_cos_sin(rope_cos, rope_sin)

    spatial_padded = pad_vision_seq_parallel(spatial_input.unsqueeze(0), num_devices=sp_factor)
    rope_cos_padded = pad_vision_seq_parallel(torch_rope_cos.permute(0, 2, 1, 3), num_devices=sp_factor)
    rope_sin_padded = pad_vision_seq_parallel(torch_rope_sin.permute(0, 2, 1, 3), num_devices=sp_factor)

    tt_spatial = bf16_tensor_2dshard(spatial_padded, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3})
    tt_prompt = bf16_tensor(prompt_input.unsqueeze(0), device=mesh_device)
    tt_temb = from_torch(temb_input.unsqueeze(0), device=mesh_device, dtype=ttnn.float32, mesh_axes=[..., tp_axis])
    tt_rope_cos = from_torch(rope_cos_padded, device=mesh_device, dtype=ttnn.float32, mesh_axes=[..., sp_axis, None])
    tt_rope_sin = from_torch(rope_sin_padded, device=mesh_device, dtype=ttnn.float32, mesh_axes=[..., sp_axis, None])
    tt_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)

    def forward() -> ttnn.Tensor:
        return tt_block(
            spatial_1BND=tt_spatial,
            prompt_1BLP=tt_prompt,
            temb_1BTD=tt_temb,
            N=spatial_seq_len,
            rope_cos=tt_rope_cos,
            rope_sin=tt_rope_sin,
            trans_mat=tt_trans_mat,
        )

    case = request.node.callspec.id.split("-")
    resolution = next(part for part in case if part.endswith("p"))
    logger.info(f"Wan block stress: spatial_seq_len={spatial_seq_len} ({spatial_seq_len // sp_factor} rows/device)")
    run_traced_stress(
        mesh_device,
        forward,
        name=f"wan_block_{resolution}",
        metadata={"spatial_seq_len": spatial_seq_len, "prompt_seq_len": prompt_seq_len},
    )
