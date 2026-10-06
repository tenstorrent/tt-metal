# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Traced stress loop over one MiniMax-H3 transformer block at production 768p context, real block-0 weights.

Needs no diffusers reference: weights come straight from the `MINIMAX_H3_MODEL_PATH` snapshot and the
rope tables from the pipeline's own `build_rope_tables`. Reports ms/iteration per window of
`TT_DIT_STRESS_WINDOW_ITERS` (default 100) replays; run under different `TT_METAL_TDP_LIMIT_WATTS`.
"""

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger
from safetensors import safe_open

import ttnn

from ....models.transformers.minimax_h3.attention_minimax_h3 import prepare_rope_tables
from ....models.transformers.minimax_h3.transformer_block_minimax_h3 import MiniMaxH3TransformerBlock
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....pipelines.minimax_h3.packing import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_FPS,
    MINIMAX_H3_MODALITY_NUM,
    audio_latent_num_frames,
    build_rope_tables,
    packed_sequence_length,
    padded_sequence_length,
    resolve_canvas_size,
    video_latent_num_frames,
)
from ....pipelines.minimax_h3.policy import align_num_frames
from ....utils.tensor import bf16_tensor_2dshard, from_torch
from ....utils.test import skip_if_unsupported_num_links
from ....utils.trace_stress import run_traced_stress
from .common import (
    _BH_ONLY,
    _ring_8k_trace,
    REAL_BLOCK_CONFIG,
    ROPE_FREQ_DIM,
    ROPE_THETA,
    TT_BLOCK_CONFIG,
    packed_layout,
    upload_rope,
)

MODEL_PATH_ENV = "MINIMAX_H3_MODEL_PATH"
HIDDEN_SIZE = REAL_BLOCK_CONFIG["hidden_size"]
ATTENTION_HEAD_DIM = REAL_BLOCK_CONFIG["attention_head_dim"]
TIME_EMBED_DIM = REAL_BLOCK_CONFIG["time_embed_dim"]
PATCH_SIZE = (1, 2, 2)
VAE_SPATIAL_DOWNSAMPLE = 16
PERF_ASPECT = (16, 9)


def _block0_state_dict() -> dict[str, torch.Tensor]:
    model_root = os.environ.get(MODEL_PATH_ENV)
    if not model_root:
        pytest.skip(f"set {MODEL_PATH_ENV} to a MiniMax-H3 diffusers snapshot to run this")
    directory = Path(model_root) / "transformer"
    weight_map = json.loads((directory / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    prefix = "transformer_blocks.0."
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


def _packed_sizes(duration_s: float, num_text_tokens: int) -> dict:
    """Token counts for `duration_s` seconds of 16:9 768P video (mirrors the block perf test)."""
    height, width = resolve_canvas_size(*PERF_ASPECT)
    grid_h = height // VAE_SPATIAL_DOWNSAMPLE // PATCH_SIZE[1]
    grid_w = width // VAE_SPATIAL_DOWNSAMPLE // PATCH_SIZE[2]
    num_frames = align_num_frames(int(duration_s * MINIMAX_H3_FPS))
    num_audio_latents = audio_latent_num_frames(num_frames)
    num_video = video_latent_num_frames(num_frames) * grid_h * grid_w
    return {
        "grid_h": grid_h,
        "grid_w": grid_w,
        "num_video": num_video,
        "num_audio": num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS,
        "num_text": num_text_tokens,
        "seq_len": packed_sequence_length(num_text_tokens, num_audio_latents, num_video),
    }


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param(
            (4, 8), 1, 0, 2, _ring_8k_trace, ttnn.Topology.Ring, False, id="4x8sp1tp0nl2_ring_is_fsdp0", marks=_BH_ONLY
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "duration_s",
    [
        pytest.param(15.0, id="15s_768p"),
        pytest.param(10.0, id="10s_768p"),
        pytest.param(5.0, id="5s_768p"),
    ],
)
def test_stress_minimax_h3_block(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    duration_s: float,
    is_fsdp: bool,
    topology: ttnn.Topology,
    reset_seeds,
) -> None:
    skip_if_unsupported_num_links(mesh_device, num_links)

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]
    sizes = _packed_sizes(duration_s, num_text_tokens=512)
    seq_len = sizes["seq_len"]
    padded_len = padded_sequence_length(seq_len, sp_factor)
    logger.info(f"H3 block stress {duration_s:g}s 768p: seq_len {seq_len} (padded {padded_len}, {padded_len // sp_factor} rows/device)")

    position_ids, tags, timestep_indices = packed_layout(
        sizes["num_text"], sizes["num_audio"], sizes["num_video"], (sizes["grid_h"], sizes["grid_w"]), padded_len
    )
    adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + tags.clamp(min=0)
    rope_cos, rope_sin = build_rope_tables(position_ids, rope_freq_dim=ROPE_FREQ_DIM, rope_theta=ROPE_THETA)
    rope_cos, rope_sin = prepare_rope_tables(rope_cos, rope_sin, ATTENTION_HEAD_DIM)

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=None,
    )
    tt_block = MiniMaxH3TransformerBlock(
        **TT_BLOCK_CONFIG,
        rotary_dim=2 * 3 * ROPE_FREQ_DIM,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
    )
    tt_block.load_torch_state_dict(_block0_state_dict())

    num_timesteps = 2
    tt_spatial = bf16_tensor_2dshard(
        torch.randn(1, 1, padded_len, HIDDEN_SIZE), device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3}
    )
    tt_temb = from_torch(torch.randn(1, 1, num_timesteps, TIME_EMBED_DIM), device=mesh_device, dtype=ttnn.float32)
    tt_adaln = from_torch(
        adaln_indices.to(torch.int32).reshape(1, 1, 1, padded_len),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.Layout.ROW_MAJOR,
        mesh_axes=[..., None, sp_axis],
    )
    tt_rope_cos, tt_rope_sin = upload_rope(rope_cos, rope_sin, mesh_device=mesh_device, sp_axis=sp_axis)
    tt_logical_n = from_torch(
        torch.tensor([seq_len], dtype=torch.int64).reshape(1, 1, 1, 1),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.Layout.ROW_MAJOR,
        mesh_axes=[..., None, None],
    )

    def forward() -> ttnn.Tensor:
        return tt_block(
            tt_spatial,
            tt_logical_n,
            temb=tt_temb,
            adaln_indices=tt_adaln,
            rope_cos=tt_rope_cos,
            rope_sin=tt_rope_sin,
        )

    run_traced_stress(
        mesh_device,
        forward,
        name=f"h3_block_{duration_s:g}s_768p",
        metadata={"seq_len": seq_len, "padded_len": padded_len},
    )
