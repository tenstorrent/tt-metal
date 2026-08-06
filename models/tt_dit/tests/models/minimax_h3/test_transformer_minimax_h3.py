# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import os
import time
from collections import defaultdict
from pathlib import Path

import pytest
import torch
from diffusers.models.transformers.transformer_minimax_h3 import MINIMAX_H3_MODALITY_NUM, MiniMaxH3RotaryPosEmbed
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3Transformer3DModel as TorchMiniMaxH3Transformer,
)
from loguru import logger
from safetensors import safe_open

import ttnn

from ....models.transformers.minimax_h3.attention_minimax_h3 import prepare_rope_tables
from ....models.transformers.minimax_h3.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor, from_torch
from ....utils.test import ring_params_req_exact_devices, skip_if_unsupported_num_links
from .common import randomize_norm_weights

# MiniMax-H3 config, shared by the `transformer/` (t2va) and `transformer_ref/` partitions. Only
# `num_layers` is reduced: the torch reference of all 50 layers is far too slow on CPU, and 2 layers
# exercises every module before and after the block stack, which is what this test is for.
NUM_ATTENTION_HEADS = 56
ATTENTION_HEAD_DIM = 128
HIDDEN_SIZE = 5376
NUM_LAYERS = 2
NUM_REFINER_LAYERS = 2
FFN_DIM = 14336
IN_CHANNELS = 24
AUDIO_IN_CHANNELS = 32
PATCH_SIZE = (1, 2, 2)
TEXT_DIM = 5120
FREQ_DIM = 256
TIME_EMBED_HIDDEN_DIM = 5376
TIME_EMBED_DIM = 2688
ROPE_FREQ_DIM = 16
ROPE_THETA = 10000.0
NORM_EPS = 1e-5
QK_NORM_EPS = 1e-5
FINAL_NORM_EPS = 1e-5

VIDEO_PATCH_DIM = IN_CHANNELS * PATCH_SIZE[0] * PATCH_SIZE[1] * PATCH_SIZE[2]  # 96

TAG_VIDEO, TAG_TEXT, TAG_AUDIO = 0, 1, 2


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
    (`max(t, 0.999)`) and **3** for audio conditioning (a literal `t = 1.0`, campaign am. 115). So a
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
    "weights",
    [
        pytest.param("random", id="random_weights"),
        # Real checkpoint values at reduced depth. Random weights cannot exercise the trained
        # distribution, and they hid a live bug once already (nn.RMSNorm inits to ones, so norm weight
        # loading was invisible until `randomize_norm_weights` was added). Skipped unless
        # MINIMAX_H3_MODEL_PATH is set.
        pytest.param("checkpoint", id="real_weights"),
    ],
)
@pytest.mark.parametrize(
    ("num_text", "num_audio", "num_video", "grid", "cond_spec"),
    [
        # The three tile-aligned cases: every modality is a multiple of TILE (32), so the packed
        # sequence is assembled directly in TILE_LAYOUT. This is the cheap path and stays covered.
        pytest.param(512, 256, 1280, (8, 8), (), id="small_s2048"),  # 2048: already aligned, no padding
        # 2112 is a multiple of TILE but not of SP * TILE, so this one exercises the tail padding.
        pytest.param(512, 256, 1344, (8, 8), (), id="unaligned_s2112"),  # 2112 -> padded to 2304
        pytest.param(512, 256, 20736, (8, 8), (), id="s21504"),
        # The shape that ships: 1344x768 / 124 frames, 512-token prompt. 37 latent frames over a
        # 24x42 patch grid = 37296 video rows (== 16 mod 32) and 207 audio latents x 2 channels =
        # 414 audio rows (== 30 mod 32), so this is the ROW_MAJOR assembly path -- and before that
        # path existed, this case asserted rather than ran.
        #
        # The three cases above are tile-aligned by construction and cannot reach it. They are kept
        # as the cheap regression net for the TILE path, but this is the one that gates what ships.
        pytest.param(512, 414, 37296, (24, 42), (), id="prod_768p_5s"),  # 38222 -> padded to 38400
        # ---- fl2va: a keyframe conditioning block between text and audio ----
        #
        # The shape that ships for fl2va: one "first" anchor adds rows_per_frame = 1008 condition rows,
        # so 39230 -> padded to 39424. Both the condition block (1008 == 16 mod 32) and the target video
        # (37296 == 16 mod 32) are unaligned, so this is the ROW_MAJOR path.
        pytest.param(512, 414, 37296, (24, 42), (("video", 1008, (24, 42)),), id="prod_768p_5s_fl2va"),
        # Both fl2va anchors: 40238 -> padded to 40448. The condition block is 2016 rows (== 0 mod 32)
        # here while one anchor gives 1008 (== 16), so the two production fl2va cases cover both
        # residues of the condition stream without inventing a shape to do it.
        pytest.param(512, 414, 37296, (24, 42), (("video", 2016, (24, 42)),), id="prod_768p_5s_fl2va_first_last"),
        # ---- ref2va: a MODALITY-INTERLEAVED conditioning region ----
        #
        # The shape a single 2048x2048 image reference ships at, measured against the reference
        # packing (campaign am. 114): 4104 presentation tokens (4096 of them one vision block) and
        # 4096 condition rows, giving 45910 -> padded to 46080. All-video conditioning, so what this
        # adds over fl2va is scale (1.22x t2va's packed length) and a condition block whose 64x64
        # spatial grid is NOT the target's 24x42 -- a reference is prepared at its own resolution.
        # This one does fit the CPU reference inside the per-test budget; the two longer ref2va shapes
        # do not, which is why the interleaved cases below run at production residues instead.
        pytest.param(4104, 414, 37296, (24, 42), (("video", 4096, (64, 64)),), id="prod_ref2va_1image"),
        # A video reference WITH its soundtrack -- the case the whole typed-block design exists for.
        # Its audio rows are packed immediately BEFORE its video rows, so the conditioning region is
        # `[audio | video]` and the two projections must be applied per block. Also the first case with
        # FOUR distinct timestep levels, since audio conditioning runs at a literal t = 1.0.
        #
        # At PRODUCTION RESIDUES but not production length, deliberately. Every stream keeps the
        # residue mod TILE that decides the assembly path -- audio cond 414 (30), video cond 1008 (16),
        # target audio 414 (30), target video 3024 (16) -- while the sequence is 5372 rather than
        # 81488. The reason is the reference, not the device: this test compares against a torch model
        # whose full self-attention is O(n^2) on CPU, and at 81488 rows that alone exceeded the 300 s
        # per-test budget (measured, campaign am. 122). Length adds nothing here -- what the
        # interleaved region can get wrong is which projection a block takes and which rows it lands
        # on, and both are residue- and order-sensitive, not length-sensitive. The production LENGTHS
        # are covered at full depth by `test_minimax_h3_transformer_real_weights`, which has no CPU
        # reference to pay for.
        pytest.param(
            512,
            414,
            3024,
            (24, 42),
            (("audio", 414, None), ("video", 1008, (24, 42))),
            id="ref2va_interleaved_audio_video",
        ),
        # The ordering trap: an image-shaped block on its OWN 64x64 grid, then a video block, then a
        # standalone audio block LAST rather than paired with a video -- so a split that assumed
        # "audio always precedes video" is caught, as is one that used the target grid for every block.
        pytest.param(
            512,
            414,
            3024,
            (24, 42),
            (("video", 4096, (64, 64)), ("video", 1008, (24, 42)), ("audio", 414, None)),
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
    # and fracturing it with mesh_partition removes the way that error used to be reachable, since the
    # caller has no row permutation to get wrong. A deeper stack would still be a more sensitive test
    # of the modulation path than this one; the block test covers that math directly instead.
    MIN_PCC = 0.9995

    skip_if_unsupported_num_links(mesh_device, num_links)

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    B = 1
    num_cond = sum(rows for _, rows, _ in cond_spec)
    seq_len = num_text + num_cond + num_audio + num_video
    per_modality = _modality_metadata(num_text, num_audio, num_video, grid, cond_spec)
    cond_blocks = per_modality["cond_blocks"]

    # ---- reference layout: [text | cond block 1 | ... | audio | video], contiguous ----
    # `packing.build_packed_sequence` / `build_ref2va_packed_sequence` order, and the conditioning
    # region's position between text and audio is what the model's concat has to agree with. For
    # ref2va that region is modality-INTERLEAVED, so the row ranges below are collected per block
    # rather than as one slice.
    segments = [per_modality["text"]] + cond_blocks + [per_modality["audio"], per_modality["video"]]
    ref_position_ids = torch.cat([segment["pos"] for segment in segments])
    ref_tags = torch.cat([segment["tags"] for segment in segments])
    ref_ts_idx = torch.cat([segment["ts"] for segment in segments])

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
    num_timesteps = int(ref_ts_idx.max().item()) + 1

    logger.info(
        f"seq_len={seq_len} (text={num_text} cond={num_cond} audio={num_audio} video={num_video}), "
        f"cond blocks={[(b['modality'], b['rows']) for b in cond_blocks]}, "
        f"cond video/audio rows={num_cond_video}/{num_cond_audio}, "
        f"num_timesteps={num_timesteps}, layers={NUM_LAYERS} (reduced from 50)"
    )

    checkpoint_state = None
    if weights == "checkpoint":
        model_root = os.environ.get(MODEL_PATH_ENV)
        if not model_root:
            pytest.skip(f"set {MODEL_PATH_ENV} to a MiniMax-H3 diffusers snapshot to run this")
        directory = Path(model_root) / os.environ.get(SUBFOLDER_ENV, "transformer")
        if not directory.is_dir():
            pytest.skip(f"{directory} is not a directory")
        start = time.time()
        checkpoint_state = _truncated_depth_state_dict(directory, NUM_LAYERS)
        logger.info(f"read {len(checkpoint_state)} tensors for {NUM_LAYERS} layers in {time.time() - start:.1f}s")

    torch_model = TorchMiniMaxH3Transformer(
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
        rope_freq_dim=ROPE_FREQ_DIM,
        rope_theta=ROPE_THETA,
        norm_eps=NORM_EPS,
        qk_norm_eps=QK_NORM_EPS,
        final_norm_eps=FINAL_NORM_EPS,
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

    video_input = torch.randn((B, num_video, VIDEO_PATCH_DIM), dtype=torch.float32)
    audio_input = torch.randn((B, num_audio, AUDIO_IN_CHANNELS), dtype=torch.float32)
    prompt_input = torch.randn((B, num_text, TEXT_DIM), dtype=torch.float32)
    # One tensor per conditioning block, at that block's own modality width: a video block is
    # `VIDEO_PATCH_DIM` (96) wide and an audio block `AUDIO_IN_CHANNELS` (32), which is exactly why
    # they cannot be concatenated before projection and why the model takes a list.
    for block in cond_blocks:
        width = VIDEO_PATCH_DIM if block["modality"] == "video" else AUDIO_IN_CHANNELS
        block["input"] = torch.randn((B, block["rows"], width), dtype=torch.float32)
    # Timesteps are consumed unscaled in [0, 1]; one entry per distinct noise level.
    timestep = torch.rand((num_timesteps,), dtype=torch.float32)

    logger.info("Running torch model")
    # The reference takes ONE stream per modality, covering that modality's index list: its
    # conditioning rows first, then its target rows. The port instead passes the conditioning rows as
    # typed blocks so it can place them in packed order without a scatter, so the two views of the
    # same rows are assembled here from the same walk.
    ref_video_input = torch.cat(
        [block["input"] for block in cond_blocks if block["modality"] == "video"] + [video_input], dim=1
    )
    ref_audio_input = torch.cat(
        [block["input"] for block in cond_blocks if block["modality"] == "audio"] + [audio_input], dim=1
    )
    with torch.no_grad():
        torch_out = torch_model(
            hidden_states=ref_video_input,
            audio_hidden_states=ref_audio_input,
            encoder_hidden_states=prompt_input,
            timestep=timestep,
            timestep_indices=ref_ts_idx,
            token_tags=ref_tags,
            position_ids=ref_position_ids,
            video_indices=video_indices,
            audio_indices=audio_indices,
            text_indices=text_indices,
            return_dict=True,
        )
    # The port returns the TARGET rows only, for BOTH modalities -- see the note on `forward`'s return
    # value -- so drop the reference's leading conditioning rows of each before comparing. For ref2va
    # the audio stream has conditioning rows too, which t2va and fl2va never did.
    torch_video_out = torch_out.sample[:, num_cond_video:]
    torch_audio_out = torch_out.audio_sample[:, num_cond_audio:]
    logger.info(f"torch video {tuple(torch_video_out.shape)} audio {tuple(torch_audio_out.shape)}")

    # ---- TT layout: the same natural global order, zero-padded to a multiple of SP * TILE ----
    # The model assembles the packed sequence while replicated and only then fractures it with
    # mesh_partition, so the metadata needs no permutation -- only the padding tail.
    alignment = sp_factor * ttnn.TILE_SIZE
    padded_len = ((seq_len + alignment - 1) // alignment) * alignment
    pad_len = padded_len - seq_len
    logger.info(f"padded_len={padded_len} (pad_len={pad_len}), rows per SP device={padded_len // sp_factor}")

    def pad_rows(arr: torch.Tensor, value: int = 0) -> torch.Tensor:
        if pad_len == 0:
            return arr
        tail = torch.full((pad_len, *arr.shape[1:]), value, dtype=arr.dtype)
        return torch.cat([arr, tail], dim=0)

    # Pad rows are excluded from attention by ring attention's logical_n, so their metadata is
    # arbitrary -- but the gather indices must still be in range, hence 0 rather than the
    # reference's -1 tag.
    padded_position_ids = pad_rows(ref_position_ids)
    padded_ts_idx = pad_rows(ref_ts_idx)
    padded_adaln = pad_rows(ref_ts_idx * MINIMAX_H3_MODALITY_NUM + ref_tags.clamp(min=0))

    rope = torch_model.rope
    with torch.no_grad():
        rope_cos, rope_sin = rope(padded_position_ids)
    # The fused RoPE wants head_dim-wide tables in the interleaved layout, not the reference's
    # rotary_dim-wide half-split ones.
    rope_cos, rope_sin = prepare_rope_tables(rope_cos, rope_sin, ATTENTION_HEAD_DIM)
    rotary_dim = rope_cos.shape[-1]

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=None,
    )

    tt_model = MiniMaxH3Transformer3DModel(
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
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
    )
    # Same weights into the TT model. For the checkpoint case this deliberately re-reads them from
    # `torch_model.state_dict()` rather than the raw dict, so the fp32 cast above applies to both
    # sides and any difference is the port, not the dtype.
    tt_model.load_torch_state_dict(torch_model.state_dict())

    # The modality inputs are fully replicated: they are projected and concatenated into the packed
    # sequence before it is fractured, so every device needs all of them.
    tt_video = bf16_tensor(video_input.unsqueeze(0), device=mesh_device)
    tt_audio = bf16_tensor(audio_input.unsqueeze(0), device=mesh_device)
    tt_prompt = bf16_tensor(prompt_input.unsqueeze(0), device=mesh_device)
    # The typed conditioning region, in packed order. Built by the same walk as everything else.
    tt_cond = [
        (bf16_tensor(block["input"].unsqueeze(0), device=mesh_device), block["modality"]) for block in cond_blocks
    ] or None
    # Raw timesteps: a handful of values, replicated, float32 so the sinusoid is computed in fp32.
    # Shaped [1, 1, T, 1] so it broadcasts against the [1, 1, 1, freq_dim/2] frequency factor.
    tt_timestep = from_torch(timestep.reshape(1, 1, num_timesteps, 1), device=mesh_device, dtype=ttnn.float32)
    # Per-row metadata covers the padded sequence and is sharded contiguously on SP -- the model
    # fractures the packed sequence the same way, with mesh_partition.
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
    tt_adaln = from_torch(
        padded_adaln.to(torch.int32).reshape(1, 1, 1, padded_len),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.Layout.ROW_MAJOR,
        mesh_axes=[..., None, sp_axis],
    )
    tt_tsi = from_torch(
        padded_ts_idx.to(torch.int32).reshape(1, 1, 1, padded_len),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.Layout.ROW_MAJOR,
        mesh_axes=[..., None, sp_axis],
    )

    logger.info("Running TT model")
    tt_video_out, tt_audio_out = tt_model(
        video_1BVC=tt_video,
        audio_1BAC=tt_audio,
        prompt_1BLP=tt_prompt,
        condition_blocks=tt_cond,
        timestep=tt_timestep,
        adaln_indices=tt_adaln,
        timestep_indices=tt_tsi,
        rope_cos=tt_rope_cos,
        rope_sin=tt_rope_sin,
    )

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
# Env var pointing at a MiniMax-H3 diffusers snapshot, e.g.
#   MINIMAX_H3_MODEL_PATH=/data/cglagovich/MiniMax-H3-diffusers
# Optionally MINIMAX_H3_SUBFOLDER to pick the partition; defaults to the t2va `transformer`.
MODEL_PATH_ENV = "MINIMAX_H3_MODEL_PATH"
SUBFOLDER_ENV = "MINIMAX_H3_SUBFOLDER"

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
    ("num_text", "num_audio", "num_video", "grid", "cond_spec"),
    [
        pytest.param(512, 256, 1280, (8, 8), (), id="small_s2048"),
        pytest.param(512, 256, 20736, (8, 8), (), id="s21504"),
        # The shape that ships: 1344x768 / 124 frames / 512-token prompt. 37 latent frames over a
        # 24x42 patch grid = 37296 video rows, 207 audio latents x 2 channels = 414 audio rows,
        # total 38222 padded to 38400 -- 4800 rows per device at SP=8. Neither of the two cases
        # above reaches this: they are tile-aligned by construction and 1.8x smaller, so they ask
        # neither the ROW_MAJOR assembly question nor the residency one at full depth.
        pytest.param(512, 414, 37296, (24, 42), (), id="prod_768p_5s"),
        # ---- ref2va: does it fit? ----
        #
        # The campaign's shape probe. ref2va packed lengths were measured host-only against the
        # reference packing (am. 114) and run 1.2x-3.0x t2va's, which is a residency question the
        # 2-layer correctness test cannot answer and the t2va shape above does not reach. These three
        # run the real 50 layers with the real checkpoint at the real padded lengths, so what they
        # answer is exactly "does the shape the e2e gate will ask for fit on the mesh".
        #
        # There is no reference here and no PCC; the shape/finiteness checks are the whole gate. The
        # verdict sets the e2e case list -- a case that does not fit becomes a documented gap with a
        # measured reason instead of a surprise at the end.
        pytest.param(4104, 414, 37296, (24, 42), (("video", 4096, (64, 64)),), id="ref2va_1image_s46080"),
        pytest.param(
            6068,
            414,
            37296,
            (24, 42),
            (("audio", 414, None), ("video", 37296, (24, 42))),
            id="ref2va_1video_s81664",
        ),
        # Nine 2048x2048 image references: the documented ceiling on images, 111616 padded, 13952 rows
        # per device. The largest shape ref2va can ask for at this target resolution.
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

    Deliberately has no torch reference: 50 layers of a 33B-parameter model is impractical on CPU, so
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
    model_root = os.environ.get(MODEL_PATH_ENV)
    if not model_root:
        pytest.skip(f"set {MODEL_PATH_ENV} to a MiniMax-H3 diffusers snapshot to run this")
    directory = Path(model_root) / os.environ.get(SUBFOLDER_ENV, "transformer")
    if not directory.is_dir():
        pytest.skip(f"{directory} is not a directory")

    skip_if_unsupported_num_links(mesh_device, num_links)

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    config = {k: v for k, v in json.loads((directory / "config.json").read_text()).items() if not k.startswith("_")}
    rope_freq_dim = config["rope_freq_dim"]
    rope_theta = config["rope_theta"]
    model_kwargs = {k: v for k, v in config.items() if k not in _CALLER_OWNED_CONFIG_KEYS}
    model_kwargs["patch_size"] = tuple(model_kwargs["patch_size"])
    logger.info(f"loading {directory} (num_layers={model_kwargs['num_layers']})")

    B = 1
    num_cond = sum(rows for _, rows, _ in cond_spec)
    seq_len = num_text + num_cond + num_audio + num_video
    hidden_size = model_kwargs["hidden_size"]
    audio_channels = model_kwargs["audio_in_channels"]
    video_patch_dim = model_kwargs["in_channels"] * int(torch.tensor(model_kwargs["patch_size"]).prod())

    per_modality = _modality_metadata(num_text, num_audio, num_video, grid, cond_spec)
    cond_blocks = per_modality["cond_blocks"]
    segments = [per_modality["text"]] + cond_blocks + [per_modality["audio"], per_modality["video"]]
    position_ids = torch.cat([segment["pos"] for segment in segments])
    tags = torch.cat([segment["tags"] for segment in segments])
    ts_idx = torch.cat([segment["ts"] for segment in segments])
    num_timesteps = int(ts_idx.max().item()) + 1

    alignment = sp_factor * ttnn.TILE_SIZE
    padded_len = ((seq_len + alignment - 1) // alignment) * alignment
    pad_len = padded_len - seq_len

    def pad_rows(arr: torch.Tensor) -> torch.Tensor:
        if pad_len == 0:
            return arr
        return torch.cat([arr, torch.zeros((pad_len, *arr.shape[1:]), dtype=arr.dtype)], dim=0)

    rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=rope_freq_dim, rope_theta=rope_theta)
    with torch.no_grad():
        rope_cos, rope_sin = rope(pad_rows(position_ids))
    rope_cos, rope_sin = prepare_rope_tables(rope_cos, rope_sin, model_kwargs["attention_head_dim"])
    rotary_dim = rope_cos.shape[-1]

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=None,
    )

    tt_model = MiniMaxH3Transformer3DModel(
        **model_kwargs,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
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

    tt_video = bf16_tensor(torch.randn(1, B, num_video, video_patch_dim), device=mesh_device)
    tt_audio = bf16_tensor(torch.randn(1, B, num_audio, audio_channels), device=mesh_device)
    tt_prompt = bf16_tensor(torch.randn(1, B, num_text, model_kwargs["text_dim"]), device=mesh_device)
    tt_cond = [
        (
            bf16_tensor(
                torch.randn(
                    1,
                    B,
                    block["rows"],
                    video_patch_dim if block["modality"] == "video" else audio_channels,
                ),
                device=mesh_device,
            ),
            block["modality"],
        )
        for block in cond_blocks
    ] or None
    tt_timestep = from_torch(
        torch.rand(num_timesteps).reshape(1, 1, num_timesteps, 1), device=mesh_device, dtype=ttnn.float32
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
    tt_adaln = from_torch(
        pad_rows(ts_idx * MINIMAX_H3_MODALITY_NUM + tags.clamp(min=0)).to(torch.int32).reshape(1, 1, 1, padded_len),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.Layout.ROW_MAJOR,
        mesh_axes=[..., None, sp_axis],
    )
    tt_tsi = from_torch(
        pad_rows(ts_idx).to(torch.int32).reshape(1, 1, 1, padded_len),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.Layout.ROW_MAJOR,
        mesh_axes=[..., None, sp_axis],
    )

    logger.info(
        f"running {model_kwargs['num_layers']} layers, seq_len={seq_len} (padded {padded_len}, "
        f"{padded_len // sp_factor} rows/device), cond blocks={[(b['modality'], b['rows']) for b in cond_blocks]}"
    )

    def forward():
        out = tt_model(
            video_1BVC=tt_video,
            audio_1BAC=tt_audio,
            prompt_1BLP=tt_prompt,
            condition_blocks=tt_cond,
            timestep=tt_timestep,
            adaln_indices=tt_adaln,
            timestep_indices=tt_tsi,
            rope_cos=tt_rope_cos,
            rope_sin=tt_rope_sin,
        )
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
