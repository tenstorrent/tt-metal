# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device perf of one MiniMax-H3 transformer block at 768P; run under `scripts/run_safe_pytest.sh --profile`,
then `tt-perf-report <csv> --start-signpost start --end-signpost stop`. Profile one duration at a time with
`-k`: a multi-parameter profiled run yields a CSV containing only the first parameter's ops."""

from __future__ import annotations

import pytest
import torch
from diffusers.models.transformers.transformer_minimax_h3 import MINIMAX_H3_MODALITY_NUM, MiniMaxH3RotaryPosEmbed
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock as TorchMiniMaxH3Block
from loguru import logger
from tracy import signpost

import ttnn

from ....models.transformers.minimax_h3.attention_minimax_h3 import prepare_rope_tables
from ....models.transformers.minimax_h3.transformer_block_minimax_h3 import MiniMaxH3TransformerBlock
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....pipelines.minimax_h3.packing import (
    MINIMAX_H3_FPS,
    align_num_frames,
    audio_latent_num_frames,
    resolve_canvas_size,
    video_latent_num_frames,
)
from ....utils.tensor import bf16_tensor_2dshard, from_torch, local_device_to_torch
from ....utils.test import skip_if_unsupported_num_links
from .common import (
    GALAXY_RING,
    REAL_BLOCK_CONFIG,
    ROPE_FREQ_DIM,
    ROPE_THETA,
    TT_BLOCK_CONFIG,
    packed_layout,
    upload_rope,
)

HIDDEN_SIZE = REAL_BLOCK_CONFIG["hidden_size"]
HEAD_DIM = REAL_BLOCK_CONFIG["attention_head_dim"]
TIME_EMBED_DIM = REAL_BLOCK_CONFIG["time_embed_dim"]

PATCH_SIZE = (1, 2, 2)
VAE_SPATIAL_DOWNSAMPLE = 16  # prod(spatial_downsample_factors) from the video VAE config
NUM_TEXT_TOKENS = 512
ASPECT = (16, 9)


def _packed_sizes(duration_s: float) -> dict:
    """Token counts for `duration_s` seconds of 768P video, derived from the pipeline's own packing helpers."""
    height, width = resolve_canvas_size(*ASPECT)
    tokens_per_latent_frame = (height // VAE_SPATIAL_DOWNSAMPLE // PATCH_SIZE[1]) * (
        width // VAE_SPATIAL_DOWNSAMPLE // PATCH_SIZE[2]
    )
    num_frames = align_num_frames(int(duration_s * MINIMAX_H3_FPS))
    latent_frames = video_latent_num_frames(num_frames)
    num_audio = audio_latent_num_frames(num_frames)
    num_video = latent_frames * tokens_per_latent_frame
    return {
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "latent_frames": latent_frames,
        "grid_h": height // VAE_SPATIAL_DOWNSAMPLE // PATCH_SIZE[1],
        "grid_w": width // VAE_SPATIAL_DOWNSAMPLE // PATCH_SIZE[2],
        "num_video": num_video,
        "num_audio": num_audio,
        "num_text": NUM_TEXT_TOKENS,
        "seq_len": NUM_TEXT_TOKENS + num_audio + num_video,
    }


@GALAXY_RING
@pytest.mark.parametrize(
    "duration_s",
    [
        pytest.param(5.0, id="5s_768p"),
        pytest.param(10.0, id="10s_768p"),
        pytest.param(15.0, id="15s_768p"),
    ],
)
@pytest.mark.parametrize(
    "sp_simulate",
    [
        pytest.param(1, id="sp_sim1"),
        pytest.param(4, id="sp_sim4"),
    ],
)
def test_minimax_h3_transformer_block_perf(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    duration_s: float,
    sp_simulate: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
    reset_seeds,
) -> None:
    skip_if_unsupported_num_links(mesh_device, num_links)
    # Simulate a larger SP mesh (e.g. 4x32) on a smaller one (4x8) by shrinking the total sequence
    # so each device carries a shard the larger mesh would produce. `sp_simulate` is that SP ratio.
    SIM = sp_simulate

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    sizes = _packed_sizes(duration_s)
    seq_len = sizes["seq_len"]
    alignment = sp_factor * ttnn.TILE_SIZE * SIM
    padded_len = ((seq_len + alignment - 1) // alignment) * alignment
    padded_len = padded_len // SIM
    logger.info(
        f"{duration_s:g}s @ {sizes['height']}x{sizes['width']}: {sizes['num_frames']} frames -> "
        f"{sizes['latent_frames']} latent frames x {sizes['grid_h']}x{sizes['grid_w']} patches = "
        f"{sizes['num_video']} video + {sizes['num_audio']} audio + {sizes['num_text']} text "
        f"= seq_len {seq_len} (padded {padded_len}, {padded_len // sp_factor} rows/device)"
    )

    num_timesteps = 2
    # Simulate a 4x32 per-device shard on a 4x8 mesh: shrink the total sequence by SIM (32/8) so each
    # 4x8 device carries a 4x32-sized shard. num_video must stay a whole number of (grid_h, grid_w)
    # frames, so floor it to a frame boundary rather than dividing the raw token count.
    frame = sizes["grid_h"] * sizes["grid_w"]
    sim_num_video = (sizes["num_video"] // SIM // frame) * frame
    sim_seq_len = sizes["num_text"] // SIM + sizes["num_audio"] // SIM + sim_num_video
    position_ids, tags, timestep_indices = packed_layout(
        sizes["num_text"] // SIM,
        sizes["num_audio"] // SIM,
        sim_num_video,
        (sizes["grid_h"], sizes["grid_w"]),
        padded_len,
    )
    adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + tags.clamp(min=0)

    # Built only to source a correctly-keyed random state dict; its forward is never called.
    torch_block = TorchMiniMaxH3Block(**REAL_BLOCK_CONFIG).to(torch.float32)

    rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=ROPE_FREQ_DIM, rope_theta=ROPE_THETA)
    with torch.no_grad():
        rope_cos, rope_sin = rope(position_ids)
    rope_cos, rope_sin = prepare_rope_tables(rope_cos, rope_sin, HEAD_DIM)

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
    tt_block.load_torch_state_dict(torch_block.state_dict())
    del torch_block

    if SIM > 1:
        # The exp-ring gate keys on sequence_parallel.factor == 32, a proxy for "the per-device
        # shard is the 4x32 shape". SP simulation produces exactly that shard on a smaller mesh,
        # but the model only sees the real mesh's factor — force the flag so the simulated run
        # exercises what a real 4x32 would.
        tt_block.attn.use_exp_ring_sdpa = tt_block.attn.use_ring and sp_factor * SIM == 32
        if sp_factor * SIM == 32:
            assert tt_block.attn.use_exp_ring_sdpa, "simulated 4x32 run must exercise the exp ring SDPA path"
            assert tt_block.attn._exp_sdpa_program_config(padded_len // sp_factor) is not None, (
                "exp ring SDPA has no valid program config for the simulated shard "
                f"({padded_len // sp_factor} tokens/device); the run would silently measure the normal ring path"
            )

    tt_spatial = bf16_tensor_2dshard(
        torch.randn(1, 1, padded_len, HIDDEN_SIZE),
        device=mesh_device,
        shard_mapping={sp_axis: 2, tp_axis: 3},
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

    def run_block() -> ttnn.Tensor:
        out = tt_block(
            tt_spatial,
            N=sim_seq_len,  # simulated unpadded length: ring attention masks the pad tail via logical_n
            temb=tt_temb,
            adaln_indices=tt_adaln,
            rope_cos=tt_rope_cos,
            rope_sin=tt_rope_sin,
        )
        ttnn.synchronize_device(mesh_device)
        return out

    logger.info("iteration 1: compiling kernels and populating the program cache")
    run_block()

    logger.info("iteration 2: warm run (the profiled region)")
    signpost("start")
    tt_out = run_block()
    signpost("stop")

    assert tuple(tt_out.shape) == (
        1,
        1,
        padded_len // sp_factor,
        HIDDEN_SIZE // tp_factor,
    ), f"unexpected output shape {tuple(tt_out.shape)}"
    local = local_device_to_torch(tt_out).float()
    assert torch.isfinite(local).all(), "block output contains NaN or Inf"
    logger.info(f"output {tuple(tt_out.shape)}, local shard std={local.std().item():.4f}")
