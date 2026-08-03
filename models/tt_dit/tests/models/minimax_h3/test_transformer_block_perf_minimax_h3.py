# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device performance of one MiniMax-H3 transformer block at realistic 768P sequence lengths.

Run under the Tracy device profiler, which emits the per-op CSV this test exists to produce:

    scripts/run_safe_pytest.sh --profile \\
        models/tt_dit/tests/models/minimax_h3/test_transformer_block_perf_minimax_h3.py

The block runs twice per parameter: once to compile and populate the program cache, then once warm.
The warm iteration is bracketed by `signpost("start")` / `signpost("stop")`, the convention the rest of
tt_dit uses (see the LTX block test), so the report tool can isolate exactly it:

    tt-perf-report <csv> --start-signpost start --end-signpost stop

Both signposts matter. Without the closing one the analysed region runs to the end of the file and
folds the output readback into the measurement.

IMPORTANT: one profiled run yields one parameter's worth of ops. Running all three durations under a
single `--profile` invocation produced a CSV containing only the first (verified against the recorded
tensor shapes), so profile one duration at a time with `-k`:

    for d in 5s_768p 10s_768p 15s_768p; do
        scripts/run_safe_pytest.sh --profile <this file> -k $d
    done

No torch reference is run and no PCC is checked -- correctness lives in
`test_transformer_block_minimax_h3.py`. This test only asserts the output is the right shape and
finite, so that a broken run fails instead of quietly producing a CSV of nonsense.

NOTE: `--profile` makes the tracy wrapper mask pytest's exit code, so the run reports PASS as long as
profiling completed. Check the logged shapes and the CSV, not just the exit status.
"""

import pytest
import torch
from diffusers.models.transformers.transformer_minimax_h3 import MINIMAX_H3_MODALITY_NUM, MiniMaxH3RotaryPosEmbed
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock as TorchMiniMaxH3Block
from diffusers.modular_pipelines.minimax_h3.packing import (
    MINIMAX_H3_FPS,
    align_num_frames,
    audio_latent_num_frames,
    resolve_canvas_size,
    video_latent_num_frames,
)
from loguru import logger
from tracy import signpost

import ttnn

from ....models.transformers.minimax_h3.transformer_block_minimax_h3 import MiniMaxH3TransformerBlock
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.tensor import bf16_tensor_2dshard, from_torch
from ....utils.test import ring_params_req_exact_devices, skip_if_unsupported_num_links

# Real MiniMax-H3 block config.
HIDDEN_SIZE = 5376
NUM_HEADS = 56
HEAD_DIM = 128
FFN_DIM = 14336
TIME_EMBED_DIM = 2688
NORM_EPS = 1e-5
QK_NORM_EPS = 1e-5
ROPE_FREQ_DIM = 16
ROPE_THETA = 10000.0

PATCH_SIZE = (1, 2, 2)
# prod(spatial_downsample_factors) from the video VAE config: [2, 2, 2, 2, 1, 1].
VAE_SPATIAL_DOWNSAMPLE = 16
# Representative Qwen3-VL prompt length; the real value is prompt-dependent.
NUM_TEXT_TOKENS = 512
# 768P at 16:9. resolve_canvas_size caps the area at 768 * 1344, so this is the widest 768P canvas.
ASPECT = (16, 9)

TAG_VIDEO, TAG_TEXT, TAG_AUDIO = 0, 1, 2


def _packed_sizes(duration_s: float) -> dict:
    """Token counts for `duration_s` seconds of 768P video, from the pipeline's own helpers.

    Deliberately derived rather than hardcoded: frame alignment (`17n + 5`), the VAE's `5n + 2` latent
    frame count, the 40 Hz audio latent grid and the canvas area cap all come from `packing.py`, so
    these stay correct if the pipeline's constants change.
    """
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


def _packed_metadata(sizes: dict, padded_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`(position_ids, token_tags, timestep_indices)` for the padded packed sequence.

    Same layout as the correctness test: text, then audio, then video, with the first video frame at
    timestep 0 (clean conditioning) and everything else at timestep 1. The values do not affect
    device timing, but keeping them realistic avoids degenerate index patterns.
    """
    n_text, n_audio, n_video = sizes["num_text"], sizes["num_audio"], sizes["num_video"]
    frame = sizes["grid_h"] * sizes["grid_w"]

    def clock(n: int) -> torch.Tensor:
        return torch.stack([torch.arange(n), torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long)], -1)

    vt, vh, vw = torch.meshgrid(
        torch.arange(sizes["latent_frames"]),
        torch.arange(sizes["grid_h"]),
        torch.arange(sizes["grid_w"]),
        indexing="ij",
    )
    position_ids = torch.cat(
        [clock(n_text), clock(n_audio), torch.stack([vt.reshape(-1), vh.reshape(-1), vw.reshape(-1)], -1)]
    )
    tags = torch.cat(
        [
            torch.full((n_text,), TAG_TEXT, dtype=torch.long),
            torch.full((n_audio,), TAG_AUDIO, dtype=torch.long),
            torch.full((n_video,), TAG_VIDEO, dtype=torch.long),
        ]
    )
    timestep_indices = torch.cat(
        [
            torch.zeros(n_text, dtype=torch.long),
            torch.ones(n_audio, dtype=torch.long),
            torch.zeros(frame, dtype=torch.long),
            torch.ones(n_video - frame, dtype=torch.long),
        ]
    )

    pad = padded_len - position_ids.shape[0]
    if pad:
        position_ids = torch.cat([position_ids, torch.zeros((pad, 3), dtype=position_ids.dtype)])
        tags = torch.cat([tags, torch.zeros(pad, dtype=tags.dtype)])
        timestep_indices = torch.cat([timestep_indices, torch.zeros(pad, dtype=timestep_indices.dtype)])
    return position_ids, tags, timestep_indices


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param(
            (4, 8), 1, 0, 2, ring_params_req_exact_devices, ttnn.Topology.Ring, False, id="4x8sp1tp0nl2_ring_is_fsdp0"
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "duration_s",
    [
        pytest.param(5.0, id="5s_768p"),
        pytest.param(10.0, id="10s_768p"),
        pytest.param(15.0, id="15s_768p"),
    ],
)
def test_minimax_h3_transformer_block_perf(
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

    sizes = _packed_sizes(duration_s)
    seq_len = sizes["seq_len"]
    alignment = sp_factor * ttnn.TILE_SIZE
    padded_len = ((seq_len + alignment - 1) // alignment) * alignment
    logger.info(
        f"{duration_s:g}s @ {sizes['height']}x{sizes['width']}: {sizes['num_frames']} frames -> "
        f"{sizes['latent_frames']} latent frames x {sizes['grid_h']}x{sizes['grid_w']} patches = "
        f"{sizes['num_video']} video + {sizes['num_audio']} audio + {sizes['num_text']} text "
        f"= seq_len {seq_len} (padded {padded_len}, {padded_len // sp_factor} rows/device)"
    )

    num_timesteps = 2
    position_ids, tags, timestep_indices = _packed_metadata(sizes, padded_len)
    adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + tags.clamp(min=0)

    # The reference block is built only to source a correctly-keyed random state dict; its forward is
    # never called. Weight *values* do not affect device timing.
    torch_block = TorchMiniMaxH3Block(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        attention_head_dim=HEAD_DIM,
        ffn_dim=FFN_DIM,
        time_embed_dim=TIME_EMBED_DIM,
        norm_eps=NORM_EPS,
        qk_norm_eps=QK_NORM_EPS,
    ).to(torch.float32)

    rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=ROPE_FREQ_DIM, rope_theta=ROPE_THETA)
    with torch.no_grad():
        rope_cos, rope_sin = rope(position_ids)
    rotary_dim = rope_cos.shape[-1]

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=None,
    )

    tt_block = MiniMaxH3TransformerBlock(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        ffn_dim=FFN_DIM,
        time_embed_dim=TIME_EMBED_DIM,
        norm_eps=NORM_EPS,
        qk_norm_eps=QK_NORM_EPS,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
    )
    tt_block.load_torch_state_dict(torch_block.state_dict())
    del torch_block

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
    tt_rope_cos = from_torch(
        rope_cos.reshape(1, 1, padded_len, rotary_dim),
        device=mesh_device,
        dtype=ttnn.float32,
        mesh_axes=[..., sp_axis, None],
    )
    tt_rope_sin = from_torch(
        rope_sin.reshape(1, 1, padded_len, rotary_dim),
        device=mesh_device,
        dtype=ttnn.float32,
        mesh_axes=[..., sp_axis, None],
    )

    def run_block() -> ttnn.Tensor:
        out = tt_block(
            tt_spatial,
            # The true (unpadded) length: ring attention masks the pad tail via logical_n.
            N=seq_len,
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
    # Cheap guard that the profiled run actually computed something, without a reference.
    local = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).float()
    assert torch.isfinite(local).all(), "block output contains NaN or Inf"
    logger.info(f"output {tuple(tt_out.shape)}, local shard std={local.std().item():.4f}")
