# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from diffusers.models.transformers.transformer_minimax_h3 import MINIMAX_H3_MODALITY_NUM
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention as TorchMiniMaxH3Attention
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TokenRefiner as TorchMiniMaxH3TokenRefiner
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3Transformer3DModel as TorchMiniMaxH3Transformer,
)
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock as TorchMiniMaxH3Block
from loguru import logger
from safetensors import safe_open

import ttnn

from ....models.transformers.minimax_h3.adaln_cache_minimax_h3 import MiniMaxH3AdalnCache
from ....models.transformers.minimax_h3.attention_minimax_h3 import MiniMaxH3Attention, prepare_rope_tables
from ....models.transformers.minimax_h3.token_refiner_minimax_h3 import MiniMaxH3TokenRefiner
from ....models.transformers.minimax_h3.transformer_block_minimax_h3 import (
    MODALITY_NUM,
    NUM_MODULATION_PARAMS,
    MiniMaxH3TransformerBlock,
)
from ....models.transformers.minimax_h3.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....pipelines.minimax_h3 import adaln_precompute as ap
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor, bf16_tensor_2dshard, from_torch
from ....utils.test import ring_params_req_exact_devices, skip_if_unsupported_num_links
from .common import (
    GALAXY_4X8_RING,
    REAL_BLOCK_CONFIG,
    ROPE_FREQ_DIM,
    ROPE_THETA,
    TAG_AUDIO,
    TAG_TEXT,
    TAG_VIDEO,
    TT_BLOCK_CONFIG,
    packed_layout,
    randomize_norm_weights,
    upload_rope,
)

# MiniMax-H3 config, shared by the `transformer/` (t2va) and `transformer_ref/` partitions. Only
# `num_layers` is reduced: the torch reference of all 50 layers is far too slow on CPU, and 2 layers
# exercises every module before and after the block stack, which is what this test is for.
# The block-level dims come from `REAL_BLOCK_CONFIG` in `common.py`, shared with the perf test.
NUM_ATTENTION_HEADS = REAL_BLOCK_CONFIG["num_attention_heads"]
ATTENTION_HEAD_DIM = REAL_BLOCK_CONFIG["attention_head_dim"]
HIDDEN_SIZE = REAL_BLOCK_CONFIG["hidden_size"]
FFN_DIM = REAL_BLOCK_CONFIG["ffn_dim"]
TIME_EMBED_DIM = REAL_BLOCK_CONFIG["time_embed_dim"]
NORM_EPS = REAL_BLOCK_CONFIG["norm_eps"]
QK_NORM_EPS = REAL_BLOCK_CONFIG["qk_norm_eps"]
NUM_LAYERS = 2
NUM_REFINER_LAYERS = 2
IN_CHANNELS = 24
AUDIO_IN_CHANNELS = 32
PATCH_SIZE = (1, 2, 2)
TEXT_DIM = 5120
FREQ_DIM = 256
TIME_EMBED_HIDDEN_DIM = 5376
FINAL_NORM_EPS = 1e-5

VIDEO_PATCH_DIM = IN_CHANNELS * PATCH_SIZE[0] * PATCH_SIZE[1] * PATCH_SIZE[2]  # 96

# The kwargs the two transformer constructors (torch reference and TT) share. The reference
# additionally takes the rope config; on the TT side the rotary tables are caller-owned.
TRANSFORMER_CONFIG = dict(
    num_attention_heads=NUM_ATTENTION_HEADS,
    attention_head_dim=ATTENTION_HEAD_DIM,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_refiner_layers=NUM_REFINER_LAYERS,
    ffn_dim=FFN_DIM,
    in_channels=IN_CHANNELS,
    audio_in_channels=AUDIO_IN_CHANNELS,
    patch_size=PATCH_SIZE,
    text_dim=TEXT_DIM,
    freq_dim=FREQ_DIM,
    time_embed_hidden_dim=TIME_EMBED_HIDDEN_DIM,
    time_embed_dim=TIME_EMBED_DIM,
    norm_eps=NORM_EPS,
    qk_norm_eps=QK_NORM_EPS,
    final_norm_eps=FINAL_NORM_EPS,
)

# Env var pointing at a MiniMax-H3 diffusers snapshot, e.g.
#   MINIMAX_H3_MODEL_PATH=/data/cglagovich/MiniMax-H3-diffusers
# Optionally MINIMAX_H3_SUBFOLDER to pick the partition; defaults to the t2va `transformer`.
MODEL_PATH_ENV = "MINIMAX_H3_MODEL_PATH"
SUBFOLDER_ENV = "MINIMAX_H3_SUBFOLDER"


def _checkpoint_dir() -> Path:
    """The checkpoint partition the env vars point at, or a skip when they are not set up."""
    model_root = os.environ.get(MODEL_PATH_ENV)
    if not model_root:
        pytest.skip(f"set {MODEL_PATH_ENV} to a MiniMax-H3 diffusers snapshot to run this")
    directory = Path(model_root) / os.environ.get(SUBFOLDER_ENV, "transformer")
    if not directory.is_dir():
        pytest.skip(f"{directory} is not a directory")
    return directory


def _modality_metadata(
    num_text: int,
    num_audio: int,
    num_video: int,
    grid: tuple[int, int] = (8, 8),
    cond_spec: tuple[tuple[str, int], ...] = (),
):
    """Per-modality `(position_ids, token_tags, timestep_indices)` for one packed layout.

    Video rows get a (t, h, w) patch grid; text and audio rows advance the shared `t` clock with
    h = w = 0. Text and the first video frame are clean (timestep 0), the rest noisy (timestep 1), so
    the AdaLN table is addressed at four distinct `(timestep, modality)` rows including row 0.

    `grid` is the (h, w) patch grid of one latent frame. It is a parameter because the default 8x8
    gives 64 rows per frame, which is a multiple of TILE for any frame count and so can never
    exercise the unaligned assembly path. Production is 24x42 = 1008 rows/frame == 16 mod 32.

    `cond_spec` is the conditioning region between text and audio, as
    `((modality, rows, block_grid), ...)` in **packed order**:

    - `()` for `t2va`;
    - `(("video", 1008, (24, 42)),)` for one `fl2va` keyframe anchor, or
      `(("video", 2016, (24, 42)),)` for two -- one contiguous block covering both, which is how the
      pipeline passes them;
    - an interleaved list for `ref2va`, e.g. `(("audio", 414, None), ("video", 37296, (24, 42)))` for
      one video reference, whose soundtrack rows are packed immediately *before* its own video rows.

    A `"video"` block's rows are **video**-tagged -- they are video rows that happen to be pinned --
    and carry `block_grid`'s spatial grid, one frame per `gh * gw` rows, at successive anchor times.
    `block_grid` is **per block** and not the target's, because that is the defining property of a
    ref2va reference: it is prepared at its own resolution, so a 2048x2048 image reference is a
    64x64 patch grid against the target's 24x42. An `"audio"` block ignores `block_grid`, is
    **audio**-tagged, and advances the shared clock like any audio run.

    The timestep indices give conditioning rows their own levels rather than reusing 0 or 1, because
    production runs them at different noise than the generated rows: **2** for video conditioning
    (`max(t, 0.999)`) and **3** for audio conditioning (a literal `t = 1.0`). So a
    ref2va request with a soundtrack addresses four distinct timestep levels where t2va and fl2va
    address three, and the AdaLN table is exercised wider.
    """
    grid_h, grid_w = grid
    frame = grid_h * grid_w
    assert num_video % frame == 0, "num_video must fill whole (h, w) frames"
    grid_t = num_video // frame
    assert grid_t >= 2, "need at least one conditioning frame and one target frame"

    def clock(n):
        return torch.stack([torch.arange(n), torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long)], -1)

    vt, vh, vw = torch.meshgrid(torch.arange(grid_t), torch.arange(grid_h), torch.arange(grid_w), indexing="ij")

    meta = {
        "text": {
            "pos": clock(num_text),
            "tags": torch.full((num_text,), TAG_TEXT, dtype=torch.long),
            "ts": torch.zeros(num_text, dtype=torch.long),
        },
        "audio": {
            "pos": clock(num_audio),
            "tags": torch.full((num_audio,), TAG_AUDIO, dtype=torch.long),
            "ts": torch.ones(num_audio, dtype=torch.long),
        },
        "video": {
            "pos": torch.stack([vt.reshape(-1), vh.reshape(-1), vw.reshape(-1)], dim=-1),
            "tags": torch.full((num_video,), TAG_VIDEO, dtype=torch.long),
            "ts": torch.cat([torch.zeros(frame, dtype=torch.long), torch.ones(num_video - frame, dtype=torch.long)]),
        },
    }

    blocks = []
    for modality, rows, block_grid in cond_spec:
        if modality == "video":
            block_h, block_w = block_grid
            block_frame = block_h * block_w
            assert rows % block_frame == 0, "a video conditioning block must fill whole (h, w) frames"
            anchors = rows // block_frame
            bh, bw = torch.meshgrid(torch.arange(block_h), torch.arange(block_w), indexing="ij")
            anchor_t = torch.arange(anchors).repeat_interleave(block_frame)
            pos = torch.stack(
                [anchor_t, bh.reshape(-1).repeat(anchors), bw.reshape(-1).repeat(anchors)],
                dim=-1,
            )
            tag, level = TAG_VIDEO, 2
        elif modality == "audio":
            pos, tag, level = clock(rows), TAG_AUDIO, 3
        else:
            raise ValueError(f"a conditioning block is 'video' or 'audio', got {modality!r}")
        blocks.append(
            {
                "modality": modality,
                "rows": rows,
                "pos": pos,
                "tags": torch.full((rows,), tag, dtype=torch.long),
                "ts": torch.full((rows,), level, dtype=torch.long),
            }
        )
    meta["cond_blocks"] = blocks
    return meta


def _prepare_tt_inputs(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
    per_modality: dict,
    *,
    text_dim: int,
    video_patch_dim: int,
    audio_channels: int,
    head_dim: int,
    rope_freq_dim: int,
    rope_theta: float,
    B: int = 1,
) -> SimpleNamespace:
    """The transformer's shared prepare/upload pipeline: packed metadata, padding, rope tables,
    random host inputs and the full ten-tensor upload for `MiniMaxH3Transformer3DModel.forward`.

    Builds inputs only -- no model, no asserts -- so the 2-layer correctness test and the full-depth
    real-weights test share exactly one packing/upload path and keep their own verification. Each
    conditioning block's host tensor is stored on its `per_modality["cond_blocks"]` entry as
    `block["input"]`.

    Returns a namespace carrying the reference-side layout (`position_ids`, `tags`, `ts_idx`), the
    host input tensors, `padded_len`, the CCL/parallel plumbing, and `tt`: the kwargs dict for the
    TT model's forward.
    """
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]
    cond_blocks = per_modality["cond_blocks"]
    num_text = per_modality["text"]["tags"].shape[0]
    num_audio = per_modality["audio"]["tags"].shape[0]
    num_video = per_modality["video"]["tags"].shape[0]

    # ---- reference layout: [text | cond block 1 | ... | audio | video], contiguous ----
    # `packing.build_packed_sequence` / `build_ref2va_packed_sequence` order, and the conditioning
    # region's position between text and audio is what the model's concat has to agree with.
    segments = [per_modality["text"]] + cond_blocks + [per_modality["audio"], per_modality["video"]]
    position_ids = torch.cat([segment["pos"] for segment in segments])
    tags = torch.cat([segment["tags"] for segment in segments])
    ts_idx = torch.cat([segment["ts"] for segment in segments])
    seq_len = position_ids.shape[0]
    num_timesteps = int(ts_idx.max().item()) + 1

    # ---- TT layout: the same natural global order, zero-padded to a multiple of SP * TILE ----
    # The model assembles the packed sequence while replicated and only then fractures it with
    # mesh_partition, so the metadata needs no permutation -- only the padding tail.
    alignment = sp_factor * ttnn.TILE_SIZE
    padded_len = ((seq_len + alignment - 1) // alignment) * alignment
    pad_len = padded_len - seq_len
    logger.info(
        f"padded_len={padded_len} (pad_len={pad_len}), rows per SP device={padded_len // sp_factor}, "
        f"num_timesteps={num_timesteps}"
    )

    # Pad rows are excluded from attention by ring attention's logical_n, so their metadata is
    # arbitrary -- but the gather indices must still be in range, hence 0 rather than the
    # reference's -1 tag.
    def pad_rows(arr: torch.Tensor) -> torch.Tensor:
        if pad_len == 0:
            return arr
        return torch.cat([arr, torch.zeros((pad_len, *arr.shape[1:]), dtype=arr.dtype)], dim=0)

    rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=rope_freq_dim, rope_theta=rope_theta)
    with torch.no_grad():
        rope_cos, rope_sin = rope(pad_rows(position_ids))
    # The fused RoPE wants head_dim-wide tables in the interleaved layout, not the reference's
    # rotary_dim-wide half-split ones.
    rope_cos, rope_sin = prepare_rope_tables(rope_cos, rope_sin, head_dim)

    video_input = torch.randn((B, num_video, video_patch_dim), dtype=torch.float32)
    audio_input = torch.randn((B, num_audio, audio_channels), dtype=torch.float32)
    prompt_input = torch.randn((B, num_text, text_dim), dtype=torch.float32)
    # One tensor per conditioning block, at that block's own modality width: a video block is
    # `video_patch_dim` (96) wide and an audio block `audio_channels` (32), which is exactly why
    # they cannot be concatenated before projection and why the model takes a list.
    for block in cond_blocks:
        width = video_patch_dim if block["modality"] == "video" else audio_channels
        block["input"] = torch.randn((B, block["rows"], width), dtype=torch.float32)
    # Timesteps are consumed unscaled in [0, 1]; one entry per distinct noise level.
    timestep = torch.rand((num_timesteps,), dtype=torch.float32)

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=None,
    )

    # Per-row metadata covers the padded sequence and is sharded contiguously on SP -- the model
    # fractures the packed sequence the same way, with mesh_partition.
    tt_rope_cos, tt_rope_sin = upload_rope(rope_cos, rope_sin, mesh_device=mesh_device, sp_axis=sp_axis)

    def upload_row_metadata(arr: torch.Tensor) -> ttnn.Tensor:
        return from_torch(
            pad_rows(arr).to(torch.int32).reshape(1, 1, 1, padded_len),
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.Layout.ROW_MAJOR,
            mesh_axes=[..., None, sp_axis],
        )

    tt = dict(
        # The modality inputs are fully replicated: they are projected and concatenated into the
        # packed sequence before it is fractured, so every device needs all of them.
        video_1BVC=bf16_tensor(video_input.unsqueeze(0), device=mesh_device),
        audio_1BAC=bf16_tensor(audio_input.unsqueeze(0), device=mesh_device),
        prompt_1BLP=bf16_tensor(prompt_input.unsqueeze(0), device=mesh_device),
        # The typed conditioning region, in packed order.
        condition_blocks=[
            (bf16_tensor(block["input"].unsqueeze(0), device=mesh_device), block["modality"])
            for block in cond_blocks
        ]
        or None,
        # Raw timesteps: a handful of values, replicated, float32 so the sinusoid is computed in
        # fp32. Shaped [1, 1, T, 1] so it broadcasts against the [1, 1, 1, freq_dim/2] factor.
        timestep=from_torch(timestep.reshape(1, 1, num_timesteps, 1), device=mesh_device, dtype=ttnn.float32),
        adaln_indices=upload_row_metadata(ts_idx * MINIMAX_H3_MODALITY_NUM + tags.clamp(min=0)),
        timestep_indices=upload_row_metadata(ts_idx),
        rope_cos=tt_rope_cos,
        rope_sin=tt_rope_sin,
    )

    return SimpleNamespace(
        seq_len=seq_len,
        padded_len=padded_len,
        num_timesteps=num_timesteps,
        position_ids=position_ids,
        tags=tags,
        ts_idx=ts_idx,
        video_input=video_input,
        audio_input=audio_input,
        prompt_input=prompt_input,
        timestep=timestep,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        tt=tt,
    )


@GALAXY_4X8_RING
@pytest.mark.parametrize(
    ("num_text", "num_audio", "num_video", "grid", "cond_spec", "weights"),
    [
        # The two tile-aligned cases: every modality is a multiple of TILE (32), so the packed
        # sequence is assembled directly in TILE_LAYOUT. This is the cheap path.
        pytest.param(512, 256, 1280, (8, 8), (), "random", id="small_s2048"),  # 2048: aligned, no padding
        # 2112 is a multiple of TILE but not of SP * TILE, so this one exercises the tail padding.
        pytest.param(512, 256, 1344, (8, 8), (), "random", id="unaligned_s2112"),  # 2112 -> padded to 2304
        # The shape that ships: 1344x768 / 124 frames, 512-token prompt. 37 latent frames over a
        # 24x42 patch grid = 37296 video rows (== 16 mod 32) and 207 audio latents x 2 channels =
        # 414 audio rows (== 30 mod 32), so this is the ROW_MAJOR assembly path.
        #
        # The two cases above are tile-aligned by construction and cannot reach it; they are the
        # cheap regression net for the TILE path, while this is the case that gates what ships.
        pytest.param(512, 414, 37296, (24, 42), (), "random", id="prod_768p_5s"),  # 38222 -> padded to 38400
        # Real checkpoint values at reduced depth, on the production shape only -- the key-map and
        # trained-distribution check does not vary by shape, so one case carries it for the whole
        # list. Random weights cannot exercise the trained distribution, and they hide norm-weight
        # loading entirely unless `randomize_norm_weights` runs, `nn.RMSNorm` initialising to ones.
        # Skipped unless MINIMAX_H3_MODEL_PATH is set.
        pytest.param(512, 414, 37296, (24, 42), (), "checkpoint", id="prod_768p_5s_real_weights"),
        # ---- fl2va: a keyframe conditioning block between text and audio ----
        #
        # The shape that ships for fl2va: one "first" anchor adds rows_per_frame = 1008 condition rows,
        # so 39230 -> padded to 39424. Both the condition block (1008 == 16 mod 32) and the target video
        # (37296 == 16 mod 32) are unaligned, so this is the ROW_MAJOR path.
        pytest.param(512, 414, 37296, (24, 42), (("video", 1008, (24, 42)),), "random", id="prod_768p_5s_fl2va"),
        # Both fl2va anchors: 40238 -> padded to 40448. The condition block is 2016 rows (== 0 mod 32)
        # here while one anchor gives 1008 (== 16), so the two production fl2va cases cover both
        # residues of the condition stream without inventing a shape to do it.
        pytest.param(
            512, 414, 37296, (24, 42), (("video", 2016, (24, 42)),), "random", id="prod_768p_5s_fl2va_first_last"
        ),
        # ---- ref2va: a MODALITY-INTERLEAVED conditioning region ----
        #
        # One case carries all the ref2va ordering traps: an image-shaped block on its OWN 64x64 grid
        # (a 2048x2048 image reference is prepared at its own resolution -- 4096 rows, measured
        # against the reference packing -- NOT the target's 24x42), then a video block, then a
        # standalone audio block LAST rather than paired with a video. So a split that assumed
        # "audio always precedes video" is caught, as is one that used the target grid for every
        # block. The audio block also makes this the case with four distinct timestep levels, audio
        # conditioning being a literal t = 1.0, where t2va and fl2va address three.
        #
        # Production RESIDUES, not production lengths -- audio cond 414 (30 mod 32), video cond 1008
        # (16), target audio 414 (30), target video 3024 (16). What the interleaved region can get
        # wrong is which projection a block takes and which rows it lands on, both residue- and
        # order-sensitive rather than length-sensitive. The full ref2va lengths cost more than the
        # budget allows here because the torch reference's attention is O(n^2) on CPU; they are
        # covered at full depth by the real-weights test below.
        pytest.param(
            512,
            414,
            3024,
            (24, 42),
            (("video", 4096, (64, 64)), ("video", 1008, (24, 42)), ("audio", 414, None)),
            "random",
            id="ref2va_interleaved_audio_last",
        ),
    ],
)
def test_minimax_h3_transformer(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    num_text: int,
    num_audio: int,
    num_video: int,
    grid: tuple[int, int],
    cond_spec: tuple[tuple[str, int, tuple[int, int] | None], ...],
    weights: str,
    is_fsdp: bool,
    topology: ttnn.Topology,
    reset_seeds,
) -> None:
    # What this threshold does and does not police. Measured at real dims with 2 layers against the
    # torch reference (norm weights randomized), for mistakes this port could plausibly make:
    #   norm_out shift/scale swapped   0.9973 video / 0.9967 audio
    #   time_proj sin/cos flipped      0.9978 video / 0.9979 audio
    #   output rows selected wrongly   0.2076 video
    # 0.9995 catches all of those against a measured 0.999974.
    #
    # Known blind spot: with only 2 layers the residual stream carries the input embeddings almost
    # unchanged to the output, so anything that merely misassigns per-row metadata barely moves the
    # result. Feeding rows metadata belonging to other rows measured 0.999888 -- only 8.6e-5 below the
    # real measurement, far too thin to gate on. Assembling the packed sequence in natural global order
    # and fracturing it with mesh_partition keeps that error unreachable, the caller having no row
    # permutation to get wrong. A deeper stack would still be a more sensitive test
    # of the modulation path than this one; the block test covers that math directly instead.
    MIN_PCC = 0.9995

    skip_if_unsupported_num_links(mesh_device, num_links)

    num_cond = sum(rows for _, rows, _ in cond_spec)
    seq_len = num_text + num_cond + num_audio + num_video
    per_modality = _modality_metadata(num_text, num_audio, num_video, grid, cond_spec)
    cond_blocks = per_modality["cond_blocks"]

    # Walk the conditioning region once, in packed order, recording each block's row range. This is
    # the single walk everything else keys off -- the reference's per-modality index lists, the
    # per-block input tensors and the model's `condition_blocks` are all built from it, so there is no
    # second ordering to get wrong.
    cursor = num_text
    for block in cond_blocks:
        block["start"], block["stop"] = cursor, cursor + block["rows"]
        cursor = block["stop"]
    audio_start = cursor
    video_start = audio_start + num_audio

    text_indices = torch.arange(num_text)
    # The reference's index lists put the CONDITIONING rows of a modality first, then its target rows,
    # which is how its `index_copy` places a non-contiguous stream. Ours passes the conditioning rows
    # as typed blocks instead.
    cond_ranges = {
        modality: [torch.arange(b["start"], b["stop"]) for b in cond_blocks if b["modality"] == modality]
        for modality in ("video", "audio")
    }
    video_indices = torch.cat(cond_ranges["video"] + [torch.arange(video_start, seq_len)])
    audio_indices = torch.cat(cond_ranges["audio"] + [torch.arange(audio_start, video_start)])
    num_cond_video = sum(len(r) for r in cond_ranges["video"])
    num_cond_audio = sum(len(r) for r in cond_ranges["audio"])

    logger.info(
        f"seq_len={seq_len} (text={num_text} cond={num_cond} audio={num_audio} video={num_video}), "
        f"cond blocks={[(b['modality'], b['rows']) for b in cond_blocks]}, "
        f"cond video/audio rows={num_cond_video}/{num_cond_audio}, "
        f"layers={NUM_LAYERS} (reduced from 50)"
    )

    checkpoint_state = None
    if weights == "checkpoint":
        directory = _checkpoint_dir()
        start = time.time()
        checkpoint_state = _truncated_depth_state_dict(directory, NUM_LAYERS)
        logger.info(f"read {len(checkpoint_state)} tensors for {NUM_LAYERS} layers in {time.time() - start:.1f}s")

    torch_model = TorchMiniMaxH3Transformer(
        **TRANSFORMER_CONFIG,
        rope_freq_dim=ROPE_FREQ_DIM,
        rope_theta=ROPE_THETA,
    )
    if checkpoint_state is not None:
        # strict: the truncated dict must cover the truncated model exactly.
        torch_model.load_state_dict(checkpoint_state, strict=True)
        # fp32 reference throughout, so the comparison is against the reference maths rather than the
        # checkpoint's mixed bf16/fp32 storage.
        torch_model = torch_model.to(torch.float32)
    else:
        torch_model = torch_model.to(torch.float32)
        # Random weights leave every RMSNorm affine at ones, which makes norm weight loading
        # invisible to a PCC comparison. Real weights need no such help.
        randomize_norm_weights(torch_model)
    torch_model.eval()

    inputs = _prepare_tt_inputs(
        mesh_device,
        sp_axis,
        tp_axis,
        num_links,
        topology,
        per_modality,
        text_dim=TEXT_DIM,
        video_patch_dim=VIDEO_PATCH_DIM,
        audio_channels=AUDIO_IN_CHANNELS,
        head_dim=ATTENTION_HEAD_DIM,
        rope_freq_dim=ROPE_FREQ_DIM,
        rope_theta=ROPE_THETA,
    )

    logger.info("Running torch model")
    # The reference takes ONE stream per modality, covering that modality's index list: its
    # conditioning rows first, then its target rows. The port instead passes the conditioning rows as
    # typed blocks so it can place them in packed order without a scatter, so the two views of the
    # same rows are assembled here from the same walk.
    ref_video_input = torch.cat(
        [block["input"] for block in cond_blocks if block["modality"] == "video"] + [inputs.video_input], dim=1
    )
    ref_audio_input = torch.cat(
        [block["input"] for block in cond_blocks if block["modality"] == "audio"] + [inputs.audio_input], dim=1
    )
    with torch.no_grad():
        torch_out = torch_model(
            hidden_states=ref_video_input,
            audio_hidden_states=ref_audio_input,
            encoder_hidden_states=inputs.prompt_input,
            timestep=inputs.timestep,
            timestep_indices=inputs.ts_idx,
            token_tags=inputs.tags,
            position_ids=inputs.position_ids,
            video_indices=video_indices,
            audio_indices=audio_indices,
            text_indices=text_indices,
            return_dict=True,
        )
    # The port returns the TARGET rows only, for BOTH modalities -- see the note on `forward`'s return
    # value -- so drop the reference's leading conditioning rows of each before comparing. For ref2va
    # the audio stream has conditioning rows too, which t2va and fl2va do not.
    torch_video_out = torch_out.sample[:, num_cond_video:]
    torch_audio_out = torch_out.audio_sample[:, num_cond_audio:]
    logger.info(f"torch video {tuple(torch_video_out.shape)} audio {tuple(torch_audio_out.shape)}")

    tt_model = MiniMaxH3Transformer3DModel(
        **TRANSFORMER_CONFIG,
        mesh_device=mesh_device,
        ccl_manager=inputs.ccl_manager,
        parallel_config=inputs.parallel_config,
        is_fsdp=is_fsdp,
    )
    # Same weights into the TT model. For the checkpoint case this re-reads them from
    # `torch_model.state_dict()` rather than the raw dict, so the fp32 cast above applies to both
    # sides and any difference is the port, not the dtype.
    tt_model.load_torch_state_dict(torch_model.state_dict())

    logger.info("Running TT model")
    tt_video_out, tt_audio_out = tt_model(**inputs.tt)

    def compose_replicated(t: ttnn.Tensor) -> torch.Tensor:
        """Outputs are gathered back on SP inside the model, so they are fully replicated here.

        Composing both mesh axes onto leading dims keeps the replicas inspectable instead of
        collapsing them, so a device that computed something different is caught rather than hidden.
        """
        out = ttnn.to_torch(
            t,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=[0, 1], mesh_shape=tuple(mesh_device.shape)),
        )
        flat = out.reshape(-1, *out.shape[2:])
        for d in range(1, flat.shape[0]):
            torch.testing.assert_close(flat[0], flat[d], rtol=0, atol=0, msg=f"replica {d} diverged")
        return flat[:1]

    tt_video_out = compose_replicated(tt_video_out)
    tt_audio_out = compose_replicated(tt_audio_out)

    logger.info("Checking video output")
    assert_quality(torch_video_out, tt_video_out, pcc=MIN_PCC)
    logger.info("Checking audio output")
    assert_quality(torch_audio_out, tt_audio_out, pcc=MIN_PCC)


# ---------------------------------------------------------------------------
# Full-depth run with the real checkpoint
# ---------------------------------------------------------------------------

# Config keys the TT model does not take: the rotary embedding is computed by the caller.
_CALLER_OWNED_CONFIG_KEYS = ("rope_freq_dim", "rope_theta")


def _load_reference_state_dict(directory: Path, keep=None) -> dict[str, torch.Tensor]:
    """Read a sharded safetensors checkpoint into one state dict, shard by shard.

    `keep(key) -> bool` filters at read time rather than after, so a truncated-depth model does not
    pay to materialise 50 blocks' worth of weights to use two of them.
    """
    index_path = directory / "diffusion_pytorch_model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    by_file: dict[str, list[str]] = defaultdict(list)
    for key, shard in weight_map.items():
        if keep is None or keep(key):
            by_file[shard].append(key)

    state: dict[str, torch.Tensor] = {}
    for shard in sorted(by_file):
        with safe_open(directory / shard, framework="pt") as handle:
            for key in by_file[shard]:
                state[key] = handle.get_tensor(key)
    if keep is None:
        assert len(state) == len(weight_map), f"loaded {len(state)} of {len(weight_map)} tensors"
    return state


def _truncated_depth_state_dict(directory: Path, num_layers: int) -> dict[str, torch.Tensor]:
    """The real checkpoint restricted to the first `num_layers` transformer blocks.

    The block stack is the only depth-dependent part; every other module (input projections, timestep
    MLP, token refiner, norm_out, both output heads) is present exactly once and loads unchanged. So a
    2-layer model built from this dict is the real model's first two layers with real weights, and a
    strict load will still catch any key that does not map.
    """

    def keep(key: str) -> bool:
        if not key.startswith("transformer_blocks."):
            return True
        return int(key.split(".")[1]) < num_layers

    return _load_reference_state_dict(directory, keep=keep)


# 62 GB of safetensors, a full 50-layer upload and two forwards, at up to 111616 padded rows for the
# ref2va probe. `pytest.ini`'s 300 s default is a correctness-test budget and does not cover this.
@pytest.mark.timeout(5400)
@GALAXY_4X8_RING
@pytest.mark.parametrize(
    ("num_text", "num_audio", "num_video", "grid", "cond_spec"),
    [
        # The shape that ships: 1344x768 / 124 frames / 512-token prompt. 37 latent frames over a
        # 24x42 patch grid = 37296 video rows, 207 audio latents x 2 channels = 414 audio rows,
        # total 38222 padded to 38400 -- 4800 rows per device at SP=8. The shapes the 2-layer test
        # also runs (2048 and 2112 packed) are strictly smaller and are implied by this one
        # fitting: fit is monotone in padded length, and only this case asks the ROW_MAJOR
        # assembly question at full depth.
        pytest.param(512, 414, 37296, (24, 42), (), id="prod_768p_5s"),
        # ---- ref2va: does it fit? ----
        #
        # The ref2va shape probe, run at its ceiling. ref2va packed lengths were measured host-only
        # against the reference packing and run 1.2x-3.0x t2va's -- one 2048x2048 image reference
        # pads to 46080, one video reference with soundtrack to 81664 -- which is a residency
        # question the 2-layer correctness test cannot answer and the t2va shape above does not
        # reach. This runs the real 50 layers with the real checkpoint at the largest padded length
        # ref2va can ask for at this target resolution, so what it answers is exactly "does the
        # shape the e2e gate will ask for fit on the mesh"; fit being monotone in padded length,
        # the intermediate ref2va shapes above come with it.
        #
        # There is no reference here and no PCC; the shape/finiteness checks are the whole gate. The
        # verdict sets the e2e case list -- a case that does not fit becomes a documented gap with a
        # measured reason instead of a surprise at the end.
        #
        # Nine 2048x2048 image references: the documented ceiling on images, 111616 padded, 13952 rows
        # per device.
        pytest.param(
            36872,
            414,
            37296,
            (24, 42),
            tuple(("video", 4096, (64, 64)) for _ in range(9)),
            id="ref2va_9image_s111616",
        ),
    ],
)
def test_minimax_h3_transformer_real_weights(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    num_text: int,
    num_audio: int,
    num_video: int,
    grid: tuple[int, int],
    cond_spec: tuple[tuple[str, int, tuple[int, int] | None], ...],
    is_fsdp: bool,
    topology: ttnn.Topology,
    reset_seeds,
) -> None:
    """Run the full-depth model with the real checkpoint, on device only.

    No torch reference: 50 layers of a 33B-parameter model is impractical on CPU, so
    there is nothing to compute PCC against. What this covers instead, none of which the 2-layer
    correctness test can:

    * that all 638 real checkpoint keys map onto the TT module -- `load_torch_state_dict` is strict, so
      a single unmapped or misnamed key fails the test. With random weights the key *names* come from
      the reference class; here they come from the checkpoint itself.
    * that the full 50-layer stack fits in device memory at a realistic sequence length and runs.
    * that the output is numerically sane rather than NaN/inf or degenerate, which is the failure mode
      real weights expose and random weights often do not.

    Correctness of the arithmetic is covered by the 2-layer test and the per-module tests.
    """
    directory = _checkpoint_dir()

    skip_if_unsupported_num_links(mesh_device, num_links)

    sp_factor = tuple(mesh_device.shape)[sp_axis]

    config = {k: v for k, v in json.loads((directory / "config.json").read_text()).items() if not k.startswith("_")}
    rope_freq_dim = config["rope_freq_dim"]
    rope_theta = config["rope_theta"]
    model_kwargs = {k: v for k, v in config.items() if k not in _CALLER_OWNED_CONFIG_KEYS}
    model_kwargs["patch_size"] = tuple(model_kwargs["patch_size"])
    logger.info(f"loading {directory} (num_layers={model_kwargs['num_layers']})")

    num_cond = sum(rows for _, rows, _ in cond_spec)
    seq_len = num_text + num_cond + num_audio + num_video
    audio_channels = model_kwargs["audio_in_channels"]
    video_patch_dim = model_kwargs["in_channels"] * int(torch.tensor(model_kwargs["patch_size"]).prod())

    per_modality = _modality_metadata(num_text, num_audio, num_video, grid, cond_spec)
    cond_blocks = per_modality["cond_blocks"]

    inputs = _prepare_tt_inputs(
        mesh_device,
        sp_axis,
        tp_axis,
        num_links,
        topology,
        per_modality,
        text_dim=model_kwargs["text_dim"],
        video_patch_dim=video_patch_dim,
        audio_channels=audio_channels,
        head_dim=model_kwargs["attention_head_dim"],
        rope_freq_dim=rope_freq_dim,
        rope_theta=rope_theta,
    )

    tt_model = MiniMaxH3Transformer3DModel(
        **model_kwargs,
        mesh_device=mesh_device,
        ccl_manager=inputs.ccl_manager,
        parallel_config=inputs.parallel_config,
        is_fsdp=is_fsdp,
    )

    start = time.time()
    state_dict = _load_reference_state_dict(directory)
    logger.info(f"read {len(state_dict)} tensors from disk in {time.time() - start:.1f}s")
    start = time.time()
    # strict: any key the TT module does not claim, or any parameter left unfilled, fails here.
    tt_model.load_torch_state_dict(state_dict)
    del state_dict
    logger.info(f"loaded state dict onto the mesh in {time.time() - start:.1f}s")

    logger.info(
        f"running {model_kwargs['num_layers']} layers, seq_len={seq_len} (padded {inputs.padded_len}, "
        f"{inputs.padded_len // sp_factor} rows/device), cond blocks={[(b['modality'], b['rows']) for b in cond_blocks]}"
    )

    def forward():
        out = tt_model(**inputs.tt)
        ttnn.synchronize_device(mesh_device)
        return out

    # The first pass compiles every kernel in the stack, which dominates it and is largely independent
    # of sequence length, so it is not a throughput figure. A second warm pass is closer to a per-step
    # number -- but note that at bringup the warm times came out nearly equal (4.0s) at seq_len 2048
    # and (3.4s) at 21504, a 10x difference in work. That says the warm pass is bound by host dispatch
    # of the ~1000 ops in 50 unfused layers, not by device compute. Treat both as capacity checks, and
    # use `run_safe_pytest.sh --profile` for an actual performance measurement.
    start = time.time()
    forward()
    cold = time.time() - start
    start = time.time()
    tt_video_out, tt_audio_out = forward()
    warm = time.time() - start
    logger.info(f"forward pass: cold {cold:.2f}s (includes kernel compilation), warm {warm:.2f}s")

    def check(name: str, tensor: ttnn.Tensor, rows: int, channels: int) -> None:
        # Composing both mesh axes onto dims 0 and 1 consumes the tensor's two leading size-1 dims,
        # so one replica is [rows, channels] -- the batch axis is folded away, not lost.
        out = ttnn.to_torch(
            tensor,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=[0, 1], mesh_shape=tuple(mesh_device.shape)),
        )
        out = out.reshape(-1, *out.shape[2:])[0].float()
        assert out.shape == (rows, channels), f"{name}: got {tuple(out.shape)}, want {(rows, channels)}"
        assert torch.isfinite(out).all(), f"{name}: contains NaN or Inf"
        std, absmax = out.std().item(), out.abs().max().item()
        logger.info(f"{name}: shape={tuple(out.shape)} std={std:.4f} absmax={absmax:.4f}")
        # A velocity prediction from a trained model should vary across rows and stay in a sane range.
        # These bounds only catch gross breakage (a dead or exploding stack), not subtle error.
        assert std > 1e-3, f"{name}: near-constant output (std={std:.3g}), stack looks dead"
        assert absmax < 1e4, f"{name}: output magnitude {absmax:.3g} looks divergent"

    check("video", tt_video_out, num_video, video_patch_dim)
    check("audio", tt_audio_out, num_audio, audio_channels)


# ---------------------------------------------------------------------- the attention module, alone


# NOTE: MiniMax-H3's attention inner dim (56 * 128 = 7168) is *larger* than the residual stream
# (5376), unlike Wan. to_q/k/v are 5376 -> 7168 and to_out is 7168 -> 5376, all bias-free.


def _packed_position_ids(T: int, H: int, W: int) -> torch.Tensor:
    """`(seq_len, 3)` (t, h, w) rotary coordinates of a video patch grid, row-major over (t, h, w).

    The attention module does not distinguish modalities -- it is full self-attention over one packed
    sequence -- so a pure video grid is a sufficient stand-in for the real packed layout here.
    """
    p_t, p_h, p_w = PATCH_SIZE
    grid_t, grid_h, grid_w = T // p_t, H // p_h, W // p_w
    coords = torch.meshgrid(torch.arange(grid_t), torch.arange(grid_h), torch.arange(grid_w), indexing="ij")
    return torch.stack([c.reshape(-1) for c in coords], dim=-1)


@GALAXY_4X8_RING
@pytest.mark.parametrize(
    ("T", "H", "W"),
    [
        # Grid chosen so seq_len is divisible by sp_factor * TILE (8 * 32 = 256). A padless packed
        # sequence needs no attention mask, which is what the reference's fast path assumes. One
        # small shape: the module is length-agnostic, and the long-sequence question is answered by
        # the transformer tests above.
        pytest.param(4, 32, 32, id="small_s1024"),
    ],
)
def test_minimax_h3_attention(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    T: int,
    H: int,
    W: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
    reset_seeds,
) -> None:
    # Measured 0.9997 at bringup on this shape and on s21504 (T21 H64 W64);
    # 0.995 leaves margin without being a rubber stamp.
    MIN_PCC = 0.995

    skip_if_unsupported_num_links(mesh_device, num_links)

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]
    assert NUM_ATTENTION_HEADS % tp_factor == 0, f"{NUM_ATTENTION_HEADS} heads must divide across TP={tp_factor}"

    B = 1
    position_ids = _packed_position_ids(T, H, W)
    seq_len = position_ids.shape[0]
    assert seq_len % (sp_factor * ttnn.TILE_SIZE) == 0, (
        f"seq_len={seq_len} must be divisible by sp_factor * TILE ({sp_factor * ttnn.TILE_SIZE}) "
        "to keep the packed sequence padless"
    )
    logger.info(f"seq_len={seq_len} ({seq_len // sp_factor} per SP device), tp_factor={tp_factor}")

    # Reference attention with random weights -- the 66GB checkpoint is not needed to validate the op.
    torch_model = TorchMiniMaxH3Attention(
        hidden_size=HIDDEN_SIZE,
        heads=NUM_ATTENTION_HEADS,
        dim_head=ATTENTION_HEAD_DIM,
        qk_norm_eps=QK_NORM_EPS,
    ).to(torch.float32)
    # Without this the per-head norm_q/norm_k weights are all ones, so QK-norm weight loading is
    # untested; see `randomize_norm_weights`.
    randomize_norm_weights(torch_model)
    torch_model.eval()

    # Real RoPE rather than random cos/sin: MiniMax-H3 rotates only the leading 2 * 3 * rope_freq_dim
    # = 96 of the 128 head channels and passes the remaining 32 through unchanged. Note the rotate-half
    # split is over those 96 channels (pairing i with i+48), *not* over the full head_dim, so cos/sin
    # cannot simply be padded out to 128 and fed to a standard head_dim-wide rotary kernel.
    rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=ROPE_FREQ_DIM, rope_theta=ROPE_THETA)
    rope_cos, rope_sin = rope(position_ids)  # each (seq_len, 96)
    rotary_dim = rope_cos.shape[-1]
    # The reference consumes the raw 96-wide tables; the TT module's fused RoPE wants them permuted
    # into the interleaved layout and padded to head_dim with cos=1 / sin=0 on the pass-through.
    tt_rope_cos_t, tt_rope_sin_t = prepare_rope_tables(rope_cos, rope_sin, ATTENTION_HEAD_DIM)
    logger.info(
        f"rotary_dim={rotary_dim} of head_dim={ATTENTION_HEAD_DIM} ({ATTENTION_HEAD_DIM - rotary_dim} pass-through)"
    )

    spatial_input = torch.randn((B, seq_len, HIDDEN_SIZE), dtype=torch.float32)

    logger.info("Running torch model")
    with torch.no_grad():
        torch_out = torch_model(
            hidden_states=spatial_input,
            rotary_emb=(rope_cos, rope_sin),
            attention_mask=None,
        )

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=None,
    )

    tt_model = MiniMaxH3Attention(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_ATTENTION_HEADS,
        head_dim=ATTENTION_HEAD_DIM,
        rotary_dim=rotary_dim,
        qk_norm_eps=QK_NORM_EPS,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    # spatial: seq fractured on SP, hidden fractured on TP
    tt_spatial = bf16_tensor_2dshard(
        spatial_input.unsqueeze(0), device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3}
    )
    # cos/sin are shared by every head, so they are fractured on SP and replicated on TP
    tt_rope_cos, tt_rope_sin = upload_rope(tt_rope_cos_t, tt_rope_sin_t, mesh_device=mesh_device, sp_axis=sp_axis)
    logger.info(f"tt_spatial {tt_spatial.shape}, tt_rope_cos {tt_rope_cos.shape}")

    logger.info("Running TT model")
    tt_out = tt_model(
        tt_spatial,
        N=seq_len,
        rope_cos=tt_rope_cos,
        rope_sin=tt_rope_sin,
    )

    concat_dims = [None, None]
    concat_dims[sp_axis] = 2
    concat_dims[tp_axis] = 3
    tt_out = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=concat_dims, mesh_shape=tuple(mesh_device.shape)),
    )
    tt_out = tt_out[:, :, :seq_len, :]

    assert_quality(torch_out, tt_out, pcc=MIN_PCC)


# ---------------------------------------------------------------------- one transformer block


def _block_setup(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
    is_fsdp: bool,
    num_text: int,
    num_audio: int,
    num_video: int,
) -> SimpleNamespace:
    """One transformer-block fixture, shared by the block gate and the precomputed-AdaLN gate.

    Builds the packed-layout metadata, the torch reference block (norm weights randomized) and its
    output, the uploaded TT inputs, the shared TT constructor kwargs (`block_kwargs`), and a `run`
    closure that feeds a TT block and returns its composed output cropped to `seq_len`. Fixtures and
    inputs only -- callers construct the TT block(s) under test and do all verification themselves.
    """
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    position_ids, token_tags, timestep_indices = packed_layout(num_text, num_audio, num_video)
    seq_len = position_ids.shape[0]
    num_timesteps = int(timestep_indices.max().item()) + 1
    # Row -> AdaLN table row, exactly as the reference computes it.
    adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags.clamp(min=0)

    # Reference block with random weights -- the 66GB checkpoint is not needed to validate the block.
    torch_model = TorchMiniMaxH3Block(**REAL_BLOCK_CONFIG).to(torch.float32)
    # Without this every RMSNorm weight is ones and norm weight loading is untested; see
    # `randomize_norm_weights`.
    randomize_norm_weights(torch_model)
    torch_model.eval()

    rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=ROPE_FREQ_DIM, rope_theta=ROPE_THETA)
    rope_cos, rope_sin = rope(position_ids)  # each (seq_len, 96)
    rotary_dim = rope_cos.shape[-1]
    tt_rope_cos_t, tt_rope_sin_t = prepare_rope_tables(rope_cos, rope_sin, ATTENTION_HEAD_DIM)

    spatial_input = torch.randn((1, seq_len, HIDDEN_SIZE), dtype=torch.float32)
    # `temb` is the shared timestep embedding: one row per *distinct* timestep, not per batch item.
    temb_input = torch.randn((num_timesteps, TIME_EMBED_DIM), dtype=torch.float32)

    logger.info("Running torch model")
    with torch.no_grad():
        torch_out = torch_model(
            hidden_states=spatial_input,
            temb=temb_input,
            adaln_indices=adaln_indices,
            rotary_emb=(rope_cos, rope_sin),
            attention_mask=None,
        )

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=None,
    )
    block_kwargs = dict(
        **TT_BLOCK_CONFIG,
        rotary_dim=rotary_dim,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
    )

    # spatial: seq fractured on SP, hidden fractured on TP
    tt_spatial = bf16_tensor_2dshard(
        spatial_input.unsqueeze(0), device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3}
    )
    # temb is a handful of rows shared by every device: replicated. Kept float32 to match the
    # reference, where the SiLU runs at temb's own precision before the bfloat16 AdaLN projection.
    tt_temb = from_torch(
        temb_input.reshape(1, 1, num_timesteps, TIME_EMBED_DIM),
        device=mesh_device,
        dtype=ttnn.float32,
    )
    # One AdaLN table row index per local row of the packed sequence, so fractured on SP.
    tt_adaln_indices = from_torch(
        adaln_indices.to(torch.int32).reshape(1, 1, 1, seq_len),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.Layout.ROW_MAJOR,
        mesh_axes=[..., None, sp_axis],
    )
    # cos/sin are shared by every head: fractured on SP, replicated on TP.
    tt_rope_cos, tt_rope_sin = upload_rope(tt_rope_cos_t, tt_rope_sin_t, mesh_device=mesh_device, sp_axis=sp_axis)
    logger.info(f"tt_spatial {tt_spatial.shape}, tt_temb {tt_temb.shape}, tt_adaln_indices {tt_adaln_indices.shape}")

    concat_dims = [None, None]
    concat_dims[sp_axis] = 2
    concat_dims[tp_axis] = 3

    def run(block, **extra) -> torch.Tensor:
        """Run one TT block over the fixture's inputs; compose and crop the output to `seq_len`."""
        out = block(
            tt_spatial,
            N=seq_len,
            temb=extra.pop("temb", tt_temb),
            adaln_indices=tt_adaln_indices,
            rope_cos=tt_rope_cos,
            rope_sin=tt_rope_sin,
            **extra,
        )
        out = ttnn.to_torch(
            out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=concat_dims, mesh_shape=tuple(mesh_device.shape)),
        )
        return out[:, :, :seq_len, :]

    return SimpleNamespace(
        seq_len=seq_len,
        num_timesteps=num_timesteps,
        token_tags=token_tags,
        adaln_indices=adaln_indices,
        temb_input=temb_input,
        torch_model=torch_model,
        torch_out=torch_out,
        sp_factor=sp_factor,
        tp_factor=tp_factor,
        parallel_config=parallel_config,
        block_kwargs=block_kwargs,
        run=run,
    )


@GALAXY_4X8_RING
@pytest.mark.parametrize(
    ("num_text", "num_audio", "num_video"),
    [
        # seq_len must be divisible by sp_factor * TILE (8 * 32 = 256) so the packed sequence is
        # padless and needs no attention mask. One small shape: the AdaLN gather this test exists to
        # gate (see the measured bug bounds below) is per-row and length-independent, and the long
        # sequences run in the transformer tests above.
        pytest.param(512, 256, 1280, id="small_s2048"),
    ],
)
def test_minimax_h3_transformer_block(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    num_text: int,
    num_audio: int,
    num_video: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
    reset_seeds,
) -> None:
    # This threshold has to be tight, and not for numerical-precision reasons: the block output is
    # `residual + gate * branch`, so the residual stream dominates it and swamps errors in the
    # modulation path. Measured at real dims with the torch reference, every plausible per-row gather
    # bug still lands at PCC >= 0.9959 against the correct output (measured with the norm weights
    # randomized, see `randomize_norm_weights`):
    #   all rows -> table row 0    0.9959      row order reversed         0.9969
    #   off-by-one modality        0.9962      timestep ignored           0.9973
    #   tags/timesteps swapped     0.9967      modality ignored           0.9984
    # A loose threshold (0.99, say) would therefore pass a completely broken AdaLN gather. The real
    # implementation measured 0.999995 on this shape and on s21504, so 0.9995 sits clear of both
    # bounds.
    MIN_PCC = 0.9995

    skip_if_unsupported_num_links(mesh_device, num_links)

    fixture = _block_setup(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, num_text, num_audio, num_video)
    seq_len = fixture.seq_len
    assert seq_len == num_text + num_audio + num_video
    assert seq_len % (fixture.sp_factor * ttnn.TILE_SIZE) == 0, (
        f"seq_len={seq_len} must be divisible by sp_factor * TILE ({fixture.sp_factor * ttnn.TILE_SIZE}) "
        "to keep the packed sequence padless"
    )
    assert int(fixture.adaln_indices.max().item()) < fixture.num_timesteps * MINIMAX_H3_MODALITY_NUM
    logger.info(
        f"seq_len={seq_len} ({seq_len // fixture.sp_factor} per SP device), num_timesteps={fixture.num_timesteps}, "
        f"adaln table rows={fixture.num_timesteps * MINIMAX_H3_MODALITY_NUM}, "
        f"tags present={sorted(set(fixture.token_tags.tolist()))}"
    )

    tt_model = MiniMaxH3TransformerBlock(**fixture.block_kwargs)
    tt_model.load_torch_state_dict(fixture.torch_model.state_dict())

    logger.info("Running TT model")
    tt_out = fixture.run(tt_model)

    assert_quality(fixture.torch_out, tt_out, pcc=MIN_PCC)


# ---------------------------------------------------------------------- the token refiner


@GALAXY_4X8_RING
@pytest.mark.parametrize(
    "prompt_seq_len",
    [
        # 512 is the production prompt length; the refiner is length-agnostic beyond tile alignment.
        pytest.param(512, id="l512"),
    ],
)
def test_minimax_h3_token_refiner(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    prompt_seq_len: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
    reset_seeds,
) -> None:
    # Tight for the same reason as the block: the residual updates are unconditional, so the residual
    # stream dominates the output. Measured at real dims against the torch reference, with the norm
    # weights randomized (see `randomize_norm_weights`):
    #   norm weights never loaded  0.8870      swiglu up/gate swapped     0.9934
    #   final_norm skipped         0.8933      norm1/norm2 swapped        0.9861
    #   only 1 of 2 blocks         0.9909      qk-norm skipped            0.9992
    #   block order swapped        0.9992
    # The real implementation measures 0.999986, so 0.9995 clears the worst variant (0.9992).
    MIN_PCC = 0.9995

    skip_if_unsupported_num_links(mesh_device, num_links)

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    B = 1
    # The refiner runs over the text stream *before* it is scattered into the packed sequence, so it
    # sees a short standalone sequence. Unlike the block, it has no AdaLN, no RoPE and no mask: it is
    # a plain pre-norm transformer over the whole text stream.
    assert prompt_seq_len % ttnn.TILE_SIZE == 0

    torch_model = TorchMiniMaxH3TokenRefiner(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_ATTENTION_HEADS,
        attention_head_dim=ATTENTION_HEAD_DIM,
        ffn_dim=FFN_DIM,
        num_layers=NUM_REFINER_LAYERS,
        norm_eps=NORM_EPS,
        qk_norm_eps=QK_NORM_EPS,
        final_norm_eps=FINAL_NORM_EPS,
    ).to(torch.float32)
    # Without this the RMSNorm weights are all ones and norm weight loading is untested; see
    # `randomize_norm_weights`. Must happen before `state_dict()` is read below.
    randomize_norm_weights(torch_model)
    torch_model.eval()

    prompt_input = torch.randn((B, prompt_seq_len, HIDDEN_SIZE), dtype=torch.float32)

    logger.info(f"Running torch model, prompt_seq_len={prompt_seq_len}")
    with torch.no_grad():
        torch_out = torch_model(prompt_input)

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=None,
    )

    tt_model = MiniMaxH3TokenRefiner(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_ATTENTION_HEADS,
        head_dim=ATTENTION_HEAD_DIM,
        ffn_dim=FFN_DIM,
        num_layers=NUM_REFINER_LAYERS,
        norm_eps=NORM_EPS,
        qk_norm_eps=QK_NORM_EPS,
        final_norm_eps=FINAL_NORM_EPS,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    # The text stream is short and every SP device needs the whole of it (each device later scatters
    # text rows into its own slice of the packed sequence), so it is replicated on SP and fractured
    # on TP only.
    tt_prompt = bf16_tensor(prompt_input.unsqueeze(0), device=mesh_device, mesh_axis=tp_axis, shard_dim=3)
    logger.info(f"tt_prompt {tt_prompt.shape}")

    logger.info("Running TT model")
    tt_out = tt_model(tt_prompt)

    # Concat the SP axis onto dim 0 so the replicas can be inspected rather than silently averaged,
    # and the TP axis onto the hidden dim.
    concat_dims = [None, None]
    concat_dims[sp_axis] = 0
    concat_dims[tp_axis] = 3
    tt_out = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=concat_dims, mesh_shape=tuple(mesh_device.shape)),
    )
    assert tt_out.shape[0] == sp_factor

    # Every SP device must hold an identical copy: the refiner is replicated on that axis, so any
    # divergence means a device read something it should not have.
    for d in range(1, sp_factor):
        torch.testing.assert_close(tt_out[0], tt_out[d], rtol=0, atol=0, msg=f"SP replica {d} diverged from replica 0")

    tt_out = tt_out[:1]
    assert_quality(torch_out, tt_out, pcc=MIN_PCC)


# ---------------------------------------------------------------------- the precomputed-AdaLN block
#
# The precomputed-AdaLN block must match the same torch reference the projected path matches.
#
# `test_adaln_precompute_minimax_h3.py` gates the host table against a *recompute* of itself. That is
# necessary but not sufficient: a table can be internally consistent and still be wired into the block
# wrongly -- wrong row order, wrong parameter slice, a missing `1 +` on a scale, or a TP shard that
# takes the wrong hidden columns. Those bugs are invisible to a self-consistency check and invisible to
# the shapes.
#
# So this compares against **torch**, and against the projected path's own PCC, using one block and one
# step. The two TT paths are also compared to each other directly, which is the tightest of the three:
# they consume the same weights by construction, so anything but near-equality is a wiring error rather
# than precision.
#
# The threshold reasoning from the transformer-block section above carries over verbatim -- the
# residual stream dominates the block output, so every plausible gather bug still scores >= 0.9959 and
# only a tight bar catches it.


MIN_PCC = 0.9995
# The two TT paths read the same weights, so they should agree far more tightly than either agrees
# with torch. Anything looser than this is a wiring bug, not arithmetic.
MIN_PCC_PATHS_AGREE = 0.9999


class _SingleLayerTable:
    """One layer of the host table's surface, built directly from a block's own AdaLN weights.

    `MiniMaxH3AdalnCache` consumes the table structurally rather than importing its type (see the
    layering note there), which is what lets this stand in for `precompute_adaln_table` without the
    26 GB checkpoint. The projection itself is the real `adaln_precompute` code, so the numerical
    conventions under test -- fp32 SiLU before the bf16 cast, per-step `temb` -- are the shipping ones.
    """

    def __init__(self, temb: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, hidden_size: int) -> None:
        projected = ap.project_block_adaln(temb, weight, bias, hidden_size)
        self.block_params = projected.unsqueeze(0)  # [1 layer, rows * MODALITY_NUM, params, hidden]
        rows = temb.shape[0]
        self.final_shift = torch.zeros(rows, hidden_size, dtype=torch.bfloat16)
        self.final_scale = torch.zeros(rows, hidden_size, dtype=torch.bfloat16)
        self.step_offsets = torch.tensor([0, rows])
        self.num_layers = 1
        self.hidden_size = hidden_size
        self.num_steps = 1


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "is_fsdp", "topology"),
    [
        pytest.param((4, 8), 1, 0, 2, False, ttnn.Topology.Ring, id="4x8sp1tp0nl2_ring_is_fsdp0"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [ring_params_req_exact_devices], indirect=True)
@pytest.mark.parametrize(("num_text", "num_audio", "num_video"), [pytest.param(512, 256, 1280, id="small_s2048")])
def test_precomputed_adaln_matches_projected_path(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    num_text: int,
    num_audio: int,
    num_video: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
    reset_seeds,
) -> None:
    skip_if_unsupported_num_links(mesh_device, num_links)

    fixture = _block_setup(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, num_text, num_audio, num_video)

    # 1. The shipping projected path, as the reference gate runs it.
    projected_block = MiniMaxH3TransformerBlock(**fixture.block_kwargs)
    projected_block.load_torch_state_dict(fixture.torch_model.state_dict())
    projected_out = fixture.run(projected_block)

    # 2. The precomputed path: same weights, projected on host into a table instead.
    state = fixture.torch_model.state_dict()
    table = _SingleLayerTable(
        fixture.temb_input,
        state["adaln_proj.linear.weight"].bfloat16(),
        state["adaln_proj.linear.bias"].bfloat16(),
        HIDDEN_SIZE,
    )
    cache = MiniMaxH3AdalnCache(
        table,
        mesh_device=mesh_device,
        parallel_config=fixture.parallel_config,
        num_layers=1,
        hidden_size=HIDDEN_SIZE,
    )
    precomputed_block = MiniMaxH3TransformerBlock(**fixture.block_kwargs, precomputed_adaln=True)
    precomputed_block.load_torch_state_dict(fixture.torch_model.state_dict())
    tables = cache.block_tables(0)
    assert len(tables) == NUM_MODULATION_PARAMS
    assert tuple(tables[0].shape)[-2:] == (fixture.num_timesteps * MODALITY_NUM, HIDDEN_SIZE // fixture.tp_factor)
    precomputed_out = fixture.run(precomputed_block, temb=None, modulation_tables=tables)

    logger.info("projected path vs torch")
    assert_quality(fixture.torch_out, projected_out, pcc=MIN_PCC)
    logger.info("precomputed path vs torch")
    assert_quality(fixture.torch_out, precomputed_out, pcc=MIN_PCC)
    logger.info("precomputed vs projected -- same weights, so this is the wiring check")
    assert_quality(projected_out, precomputed_out, pcc=MIN_PCC_PATHS_AGREE)
