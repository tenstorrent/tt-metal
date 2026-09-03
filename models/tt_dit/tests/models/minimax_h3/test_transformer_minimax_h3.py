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
from tracy import signpost

import ttnn

from ....models.transformers.minimax_h3.attention_minimax_h3 import MiniMaxH3Attention, prepare_rope_tables
from ....models.transformers.minimax_h3.token_refiner_minimax_h3 import MiniMaxH3TokenRefiner
from ....models.transformers.minimax_h3.transformer_block_minimax_h3 import MiniMaxH3TransformerBlock
from ....models.transformers.minimax_h3.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....pipelines.minimax_h3.packing import (
    MINIMAX_H3_FPS,
    align_num_frames,
    audio_latent_num_frames,
    resolve_canvas_size,
    video_latent_num_frames,
)
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor, bf16_tensor_2dshard, from_torch, local_device_to_torch
from ....utils.test import skip_if_unsupported_num_links
from .common import (
    GALAXY_RING,
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


def logical_length_tensor(mesh_device: ttnn.MeshDevice, value: int) -> ttnn.Tensor:
    """The `logical_n` device tensor the forwards take: [1, 1, 1, 1] uint32 ROW_MAJOR, replicated."""
    return from_torch(
        torch.tensor([value], dtype=torch.int64).reshape(1, 1, 1, 1),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.Layout.ROW_MAJOR,
        mesh_axes=[..., None, None],
    )


NUM_ATTENTION_HEADS = REAL_BLOCK_CONFIG["num_attention_heads"]
ATTENTION_HEAD_DIM = REAL_BLOCK_CONFIG["attention_head_dim"]
HIDDEN_SIZE = REAL_BLOCK_CONFIG["hidden_size"]
FFN_DIM = REAL_BLOCK_CONFIG["ffn_dim"]
TIME_EMBED_DIM = REAL_BLOCK_CONFIG["time_embed_dim"]
NORM_EPS = REAL_BLOCK_CONFIG["norm_eps"]
QK_NORM_EPS = REAL_BLOCK_CONFIG["qk_norm_eps"]
NUM_LAYERS = 2  # reduced from 50: the full-depth torch reference is far too slow on CPU
NUM_REFINER_LAYERS = 2
IN_CHANNELS = 24
AUDIO_IN_CHANNELS = 32
PATCH_SIZE = (1, 2, 2)
TEXT_DIM = 5120
FREQ_DIM = 256
TIME_EMBED_HIDDEN_DIM = 5376
FINAL_NORM_EPS = 1e-5

VIDEO_PATCH_DIM = IN_CHANNELS * PATCH_SIZE[0] * PATCH_SIZE[1] * PATCH_SIZE[2]  # 96

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

MODEL_PATH_ENV = "MINIMAX_H3_MODEL_PATH"


def _checkpoint_dir() -> Path:
    model_root = os.environ.get(MODEL_PATH_ENV)
    if not model_root:
        pytest.skip(f"set {MODEL_PATH_ENV} to a MiniMax-H3 diffusers snapshot to run this")
    directory = Path(model_root) / "transformer"
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
    """Per-modality packed-layout metadata; `cond_spec` is `((modality, rows, block_grid), ...)` in packed order, each block on its own grid, conditioning at timestep 2 (video) / 3 (audio)."""
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
    """Build packed metadata, rope tables, random host inputs and the TT forward kwargs -- inputs only, no model, no asserts."""
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]
    cond_blocks = per_modality["cond_blocks"]
    num_text = per_modality["text"]["tags"].shape[0]
    num_audio = per_modality["audio"]["tags"].shape[0]
    num_video = per_modality["video"]["tags"].shape[0]

    segments = [per_modality["text"]] + cond_blocks + [per_modality["audio"], per_modality["video"]]
    position_ids = torch.cat([segment["pos"] for segment in segments])
    tags = torch.cat([segment["tags"] for segment in segments])
    ts_idx = torch.cat([segment["ts"] for segment in segments])
    seq_len = position_ids.shape[0]
    num_timesteps = int(ts_idx.max().item()) + 1

    alignment = sp_factor * ttnn.TILE_SIZE
    padded_len = ((seq_len + alignment - 1) // alignment) * alignment
    pad_len = padded_len - seq_len
    logger.info(
        f"padded_len={padded_len} (pad_len={pad_len}), rows per SP device={padded_len // sp_factor}, "
        f"num_timesteps={num_timesteps}"
    )

    # pad rows are masked by ring attention's logical_n, but gather indices must stay in range: 0, not -1
    def pad_rows(arr: torch.Tensor) -> torch.Tensor:
        if pad_len == 0:
            return arr
        return torch.cat([arr, torch.zeros((pad_len, *arr.shape[1:]), dtype=arr.dtype)], dim=0)

    rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=rope_freq_dim, rope_theta=rope_theta)
    with torch.no_grad():
        rope_cos, rope_sin = rope(pad_rows(position_ids))
    rope_cos, rope_sin = prepare_rope_tables(rope_cos, rope_sin, head_dim)

    video_input = torch.randn((B, num_video, video_patch_dim), dtype=torch.float32)
    audio_input = torch.randn((B, num_audio, audio_channels), dtype=torch.float32)
    prompt_input = torch.randn((B, num_text, text_dim), dtype=torch.float32)
    for block in cond_blocks:
        width = video_patch_dim if block["modality"] == "video" else audio_channels
        block["input"] = torch.randn((B, block["rows"], width), dtype=torch.float32)
    timestep = torch.rand((num_timesteps,), dtype=torch.float32)

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=None,
    )

    tt_rope_cos, tt_rope_sin = upload_rope(rope_cos, rope_sin, mesh_device=mesh_device, sp_axis=sp_axis)

    def upload_row_metadata(arr: torch.Tensor) -> ttnn.Tensor:
        return from_torch(
            pad_rows(arr).to(torch.int32).reshape(1, 1, 1, padded_len),
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.Layout.ROW_MAJOR,
            mesh_axes=[..., None, sp_axis],
        )

    def upload_replicated_indices(arr: torch.Tensor) -> ttnn.Tensor:
        # The assembly and output-selection gathers run on SP-replicated tensors, so their index
        # tensors are replicated too -- `upload_row_metadata` without the shard.
        return from_torch(
            arr.to(torch.int32).reshape(1, 1, 1, -1),
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.Layout.ROW_MAJOR,
            mesh_axes=[..., None, None],
        )

    # The new forward takes fixed-capacity streams plus gather indices, exactly as the pipeline
    # builds them: every true length rides in index content, not in a shape. Here the caps are the
    # stream sizes rounded up to a tile -- the smallest that exercises the production gather path.
    tile = ttnn.TILE_SIZE

    def rup(n: int) -> int:
        return ((n + tile - 1) // tile) * tile

    def pad_stream(t: torch.Tensor, cap: int) -> torch.Tensor:  # [B, n, C] -> [B, cap, C]
        if t.shape[1] == cap:
            return t
        return torch.cat([t, torch.zeros(t.shape[0], cap - t.shape[1], t.shape[2], dtype=t.dtype)], dim=1)

    l_cap, v_cap, a_cap = rup(num_text), rup(num_video), rup(num_audio)
    video_cond = [b["input"] for b in cond_blocks if b["modality"] == "video"]
    audio_cond = [b["input"] for b in cond_blocks if b["modality"] == "audio"]
    cv_total = sum(b["rows"] for b in cond_blocks if b["modality"] == "video")
    ca_total = sum(b["rows"] for b in cond_blocks if b["modality"] == "audio")
    kv_cap = rup(cv_total) if cv_total else 0
    ka_cap = rup(ca_total) if ca_total else 0

    # Source-table row offsets, mirroring forward's concat order [text | cond video | cond audio |
    # audio | video]; a condition segment exists only when its stream is passed.
    cursor = l_cap
    off_cv, cursor = cursor, cursor + kv_cap
    off_ca, cursor = cursor, cursor + ka_cap
    off_audio, cursor = cursor, cursor + a_cap
    off_video = cursor

    # assembly_indices: packed order [text | cond blocks in order | audio | video | pad], each cond
    # block gathered from its modality arena with a per-modality cursor. Pad rows point at row 0.
    asm = torch.zeros(padded_len, dtype=torch.int64)
    asm[:num_text] = torch.arange(num_text)
    pos, cv_cur, ca_cur = num_text, 0, 0
    for block in cond_blocks:
        rows = block["rows"]
        if block["modality"] == "video":
            base, cv_cur = off_cv + cv_cur, cv_cur + rows
        else:
            base, ca_cur = off_ca + ca_cur, ca_cur + rows
        asm[pos : pos + rows] = torch.arange(base, base + rows)
        pos += rows
    asm[pos : pos + num_audio] = torch.arange(off_audio, off_audio + num_audio)
    pos += num_audio
    asm[pos : pos + num_video] = torch.arange(off_video, off_video + num_video)

    # Output selection: global packed row of each target row, entries past the true count pointing at
    # the modality's first target row.
    audio_start = num_text + cv_total + ca_total
    video_start = audio_start + num_audio
    v_out = torch.full((v_cap,), video_start, dtype=torch.int64)
    v_out[:num_video] = torch.arange(video_start, video_start + num_video)
    a_out = torch.full((a_cap,), audio_start, dtype=torch.int64)
    a_out[:num_audio] = torch.arange(audio_start, audio_start + num_audio)

    # Window boundaries fencing the true prompt tokens off from the arena's pad tail, exactly as
    # the pipeline builds them (`_prompt_windows`): SDPA's windowed mode synthesizes the mask on
    # device from the three boundaries.
    prompt_windows = None
    if l_cap != num_text:
        prompt_windows = from_torch(
            torch.tensor([0, num_text, l_cap], dtype=torch.int32),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.Layout.ROW_MAJOR,
            mesh_axes=[None],
        )

    def cond_arena(inputs: list[torch.Tensor], cap: int) -> ttnn.Tensor | None:
        if not inputs:
            return None
        return bf16_tensor(pad_stream(torch.cat(inputs, dim=1), cap).unsqueeze(0), device=mesh_device)

    # The step-invariant streams go through `prepare_static_sources` once, exactly as the pipeline
    # calls it; `tt` holds the per-step `forward` arguments.
    tt_static = dict(
        prompt_1BLP=bf16_tensor(pad_stream(prompt_input, l_cap).unsqueeze(0), device=mesh_device),
        prompt_windows=prompt_windows,
        condition_video_1BKC=cond_arena(video_cond, kv_cap),
        condition_audio_1BKC=cond_arena(audio_cond, ka_cap),
    )
    tt = dict(
        video_1BVC=bf16_tensor(pad_stream(video_input, v_cap).unsqueeze(0), device=mesh_device),
        audio_1BAC=bf16_tensor(pad_stream(audio_input, a_cap).unsqueeze(0), device=mesh_device),
        assembly_indices=upload_replicated_indices(asm),
        video_out_indices=upload_replicated_indices(v_out),
        audio_out_indices=upload_replicated_indices(a_out),
        timestep=from_torch(timestep.reshape(1, 1, num_timesteps, 1), device=mesh_device, dtype=ttnn.float32),
        adaln_indices=upload_row_metadata(ts_idx * MINIMAX_H3_MODALITY_NUM + tags.clamp(min=0)),
        timestep_indices=upload_row_metadata(ts_idx),
        rope_cos=tt_rope_cos,
        rope_sin=tt_rope_sin,
        logical_n=logical_length_tensor(mesh_device, seq_len),
        pad_to=padded_len,
    )

    return SimpleNamespace(
        seq_len=seq_len,
        padded_len=padded_len,
        num_timesteps=num_timesteps,
        num_video=num_video,
        num_audio=num_audio,
        position_ids=position_ids,
        tags=tags,
        ts_idx=ts_idx,
        video_input=video_input,
        audio_input=audio_input,
        prompt_input=prompt_input,
        timestep=timestep,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        tt_static=tt_static,
        tt=tt,
    )


@GALAXY_RING
@pytest.mark.parametrize(
    ("num_text", "num_audio", "num_video", "grid", "cond_spec", "weights"),
    [
        pytest.param(512, 256, 1280, (8, 8), (), "random", id="small_s2048"),
        pytest.param(
            512, 256, 1344, (8, 8), (), "random", id="unaligned_s2112"
        ),  # multiple of TILE, not SP*TILE: tail padding
        pytest.param(
            512, 414, 37296, (24, 42), (), "random", id="prod_768p_5s"
        ),  # 37296 == 16 mod 32: ROW_MAJOR assembly
        # skipped unless MINIMAX_H3_MODEL_PATH is set
        pytest.param(512, 414, 37296, (24, 42), (), "checkpoint", id="prod_768p_5s_real_weights"),
        pytest.param(512, 414, 37296, (24, 42), (("video", 1008, (24, 42)),), "random", id="prod_768p_5s_fl2va"),
        pytest.param(
            512, 414, 37296, (24, 42), (("video", 2016, (24, 42)),), "random", id="prod_768p_5s_fl2va_first_last"
        ),
        # production residues at reduced lengths; image ref on its OWN 64x64 grid, standalone audio block LAST
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
    MIN_PCC = 0.9995  # worst measured plausible bug 0.9967; real impl measured 0.999974

    skip_if_unsupported_num_links(mesh_device, num_links)

    num_cond = sum(rows for _, rows, _ in cond_spec)
    seq_len = num_text + num_cond + num_audio + num_video
    per_modality = _modality_metadata(num_text, num_audio, num_video, grid, cond_spec)
    cond_blocks = per_modality["cond_blocks"]

    cursor = num_text
    for block in cond_blocks:
        block["start"], block["stop"] = cursor, cursor + block["rows"]
        cursor = block["stop"]
    audio_start = cursor
    video_start = audio_start + num_audio

    text_indices = torch.arange(num_text)
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
        torch_model.load_state_dict(checkpoint_state, strict=True)
        torch_model = torch_model.to(torch.float32)
    else:
        torch_model = torch_model.to(torch.float32)
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
    # the port returns target rows only: drop the reference's leading conditioning rows before comparing
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
    tt_model.load_torch_state_dict(torch_model.state_dict())

    logger.info("Running TT model")
    tt_model.prepare_static_sources(**inputs.tt_static)
    tt_video_out, tt_audio_out = tt_model(**inputs.tt)

    def compose_replicated(t: ttnn.Tensor) -> torch.Tensor:
        """Compose both mesh axes and assert every replica is identical, so a diverged device is caught."""
        out = ttnn.to_torch(
            t,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=[0, 1], mesh_shape=tuple(mesh_device.shape)),
        )
        flat = out.reshape(-1, *out.shape[2:])
        for d in range(1, flat.shape[0]):
            torch.testing.assert_close(flat[0], flat[d], rtol=0, atol=0, msg=f"replica {d} diverged")
        return flat[:1]

    # The forward returns arena-capacity rows, true target rows leading; slice to the true counts.
    tt_video_out = compose_replicated(tt_video_out)[:, :num_video]
    tt_audio_out = compose_replicated(tt_audio_out)[:, :num_audio]

    logger.info("Checking video output")
    assert_quality(torch_video_out, tt_video_out, pcc=MIN_PCC)
    logger.info("Checking audio output")
    assert_quality(torch_audio_out, tt_audio_out, pcc=MIN_PCC)


# ---- full-depth run with the real checkpoint ----

_CALLER_OWNED_CONFIG_KEYS = ("rope_freq_dim", "rope_theta")


def _load_reference_state_dict(directory: Path, keep=None) -> dict[str, torch.Tensor]:
    """Read a sharded safetensors checkpoint; `keep(key)` filters at read time."""
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
    def keep(key: str) -> bool:
        if not key.startswith("transformer_blocks."):
            return True
        return int(key.split(".")[1]) < num_layers

    return _load_reference_state_dict(directory, keep=keep)


@pytest.mark.timeout(5400)  # 62 GB checkpoint + full-depth forwards exceed the 300 s default
@GALAXY_RING
@pytest.mark.parametrize(
    ("num_text", "num_audio", "num_video", "grid", "cond_spec"),
    [
        pytest.param(512, 414, 37296, (24, 42), (), id="prod_768p_5s"),
        # ref2va fit probe at its ceiling: nine 2048x2048 image refs, 111616 padded rows
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
    """Full-depth real-checkpoint run, device only: strict key mapping, memory fit, sane outputs -- no torch reference, no PCC."""
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
    tt_model.load_torch_state_dict(state_dict)
    del state_dict
    logger.info(f"loaded state dict onto the mesh in {time.time() - start:.1f}s")

    logger.info(
        f"running {model_kwargs['num_layers']} layers, seq_len={seq_len} (padded {inputs.padded_len}, "
        f"{inputs.padded_len // sp_factor} rows/device), cond blocks={[(b['modality'], b['rows']) for b in cond_blocks]}"
    )

    # Once per request, as the pipeline runs it; `forward` reads the stored prefix every call.
    tt_model.prepare_static_sources(**inputs.tt_static)

    def forward():
        out = tt_model(**inputs.tt)
        ttnn.synchronize_device(mesh_device)
        return out

    # cold/warm times are capacity checks only (warm pass is host-dispatch bound); use --profile for real perf
    start = time.time()
    forward()
    cold = time.time() - start
    start = time.time()
    tt_video_out, tt_audio_out = forward()
    warm = time.time() - start
    logger.info(f"forward pass: cold {cold:.2f}s (includes kernel compilation), warm {warm:.2f}s")

    def check(name: str, tensor: ttnn.Tensor, rows: int, channels: int) -> None:
        out = ttnn.to_torch(
            tensor,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=[0, 1], mesh_shape=tuple(mesh_device.shape)),
        )
        # The forward returns arena-capacity rows, true target rows leading; slice to the true count.
        out = out.reshape(-1, *out.shape[2:])[0].float()[:rows]
        assert out.shape == (rows, channels), f"{name}: got {tuple(out.shape)}, want {(rows, channels)}"
        assert torch.isfinite(out).all(), f"{name}: contains NaN or Inf"
        std, absmax = out.std().item(), out.abs().max().item()
        logger.info(f"{name}: shape={tuple(out.shape)} std={std:.4f} absmax={absmax:.4f}")
        assert std > 1e-3, f"{name}: near-constant output (std={std:.3g}), stack looks dead"
        assert absmax < 1e4, f"{name}: output magnitude {absmax:.3g} looks divergent"

    check("video", tt_video_out, num_video, video_patch_dim)
    check("audio", tt_audio_out, num_audio, audio_channels)


# ---- the attention module, alone ----


def _packed_position_ids(T: int, H: int, W: int) -> torch.Tensor:
    p_t, p_h, p_w = PATCH_SIZE
    grid_t, grid_h, grid_w = T // p_t, H // p_h, W // p_w
    coords = torch.meshgrid(torch.arange(grid_t), torch.arange(grid_h), torch.arange(grid_w), indexing="ij")
    return torch.stack([c.reshape(-1) for c in coords], dim=-1)


@GALAXY_RING
@pytest.mark.parametrize(
    ("T", "H", "W"),
    [
        pytest.param(4, 32, 32, id="small_s1024"),  # seq_len divisible by sp_factor * TILE: padless, no mask
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
    MIN_PCC = 0.995  # measured 0.9997 at bringup

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

    torch_model = TorchMiniMaxH3Attention(
        hidden_size=HIDDEN_SIZE,
        heads=NUM_ATTENTION_HEADS,
        dim_head=ATTENTION_HEAD_DIM,
        qk_norm_eps=QK_NORM_EPS,
    ).to(torch.float32)
    randomize_norm_weights(torch_model)
    torch_model.eval()

    # RoPE rotates only the leading 96 of 128 head channels, rotate-half pairing i with i+48
    rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=ROPE_FREQ_DIM, rope_theta=ROPE_THETA)
    rope_cos, rope_sin = rope(position_ids)  # each (seq_len, 96)
    rotary_dim = rope_cos.shape[-1]
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

    tt_spatial = bf16_tensor_2dshard(
        spatial_input.unsqueeze(0), device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3}
    )
    tt_rope_cos, tt_rope_sin = upload_rope(tt_rope_cos_t, tt_rope_sin_t, mesh_device=mesh_device, sp_axis=sp_axis)
    logger.info(f"tt_spatial {tt_spatial.shape}, tt_rope_cos {tt_rope_cos.shape}")

    logger.info("Running TT model")
    tt_out = tt_model(
        tt_spatial,
        logical_n=logical_length_tensor(mesh_device, seq_len),
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


# ---- one transformer block ----


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
    """Shared block fixture: packed metadata, torch reference + output, uploaded TT inputs, `block_kwargs`, and a `run` closure."""
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    position_ids, token_tags, timestep_indices = packed_layout(num_text, num_audio, num_video)
    seq_len = position_ids.shape[0]
    num_timesteps = int(timestep_indices.max().item()) + 1
    adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags.clamp(min=0)

    torch_model = TorchMiniMaxH3Block(**REAL_BLOCK_CONFIG).to(torch.float32)
    randomize_norm_weights(torch_model)
    torch_model.eval()

    rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=ROPE_FREQ_DIM, rope_theta=ROPE_THETA)
    rope_cos, rope_sin = rope(position_ids)
    rotary_dim = rope_cos.shape[-1]
    tt_rope_cos_t, tt_rope_sin_t = prepare_rope_tables(rope_cos, rope_sin, ATTENTION_HEAD_DIM)

    spatial_input = torch.randn((1, seq_len, HIDDEN_SIZE), dtype=torch.float32)
    # temb has one row per distinct timestep, not per batch item
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

    tt_spatial = bf16_tensor_2dshard(
        spatial_input.unsqueeze(0), device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3}
    )
    tt_temb = from_torch(
        temb_input.reshape(1, 1, num_timesteps, TIME_EMBED_DIM),
        device=mesh_device,
        dtype=ttnn.float32,  # fp32: the reference runs the SiLU at temb precision
    )
    tt_adaln_indices = from_torch(
        adaln_indices.to(torch.int32).reshape(1, 1, 1, seq_len),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.Layout.ROW_MAJOR,
        mesh_axes=[..., None, sp_axis],
    )
    tt_rope_cos, tt_rope_sin = upload_rope(tt_rope_cos_t, tt_rope_sin_t, mesh_device=mesh_device, sp_axis=sp_axis)
    logger.info(f"tt_spatial {tt_spatial.shape}, tt_temb {tt_temb.shape}, tt_adaln_indices {tt_adaln_indices.shape}")

    concat_dims = [None, None]
    concat_dims[sp_axis] = 2
    concat_dims[tp_axis] = 3

    def run(block, **extra) -> torch.Tensor:
        out = block(
            tt_spatial,
            logical_length_tensor(mesh_device, seq_len),
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


@GALAXY_RING
@pytest.mark.parametrize(
    ("num_text", "num_audio", "num_video"),
    [
        pytest.param(512, 256, 1280, id="small_s2048"),  # seq_len divisible by sp_factor * TILE: padless, no mask
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
    MIN_PCC = 0.9995  # every measured AdaLN gather bug scores >= 0.9959; real impl 0.999995

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


# ---- production-geometry block device-perf (Tracy signposts) ----
#
# Run under `scripts/run_safe_pytest.sh --profile`, then
# `tt-perf-report <csv> --start-signpost start --end-signpost stop`. Profile one duration at a
# time with `-k`: a multi-parameter profiled run yields a CSV containing only the first
# parameter's ops.

VAE_SPATIAL_DOWNSAMPLE = 16  # prod(spatial_downsample_factors) from the video VAE config
NUM_TEXT_TOKENS = 512
PERF_ASPECT = (16, 9)


def _packed_sizes(duration_s: float) -> dict:
    """Token counts for `duration_s` seconds of 768P video, derived from the pipeline's own packing helpers."""
    height, width = resolve_canvas_size(*PERF_ASPECT)
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
    tt_rope_cos, tt_rope_sin = upload_rope(rope_cos, rope_sin, mesh_device=mesh_device, sp_axis=sp_axis)

    def run_block() -> ttnn.Tensor:
        out = tt_block(
            tt_spatial,
            logical_length_tensor(mesh_device, sim_seq_len),  # simulated unpadded length
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


# ---- the token refiner ----


@GALAXY_RING
@pytest.mark.parametrize(
    "prompt_seq_len",
    [
        pytest.param(512, id="l512"),  # production prompt length
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
    MIN_PCC = 0.9995  # worst measured wiring bug 0.9992; real impl 0.999986

    skip_if_unsupported_num_links(mesh_device, num_links)

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    B = 1
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

    tt_prompt = bf16_tensor(prompt_input.unsqueeze(0), device=mesh_device, mesh_axis=tp_axis, shard_dim=3)
    logger.info(f"tt_prompt {tt_prompt.shape}")

    logger.info("Running TT model")
    tt_out = tt_model(tt_prompt)

    concat_dims = [None, None]
    concat_dims[sp_axis] = 0
    concat_dims[tp_axis] = 3
    tt_out = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=concat_dims, mesh_shape=tuple(mesh_device.shape)),
    )
    assert tt_out.shape[0] == sp_factor

    for d in range(1, sp_factor):
        torch.testing.assert_close(tt_out[0], tt_out[d], rtol=0, atol=0, msg=f"SP replica {d} diverged from replica 0")

    tt_out = tt_out[:1]
    assert_quality(torch_out, tt_out, pcc=MIN_PCC)
