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


def _modality_metadata(num_text: int, num_audio: int, num_video: int):
    """Per-modality `(position_ids, token_tags, timestep_indices)` for one packed layout.

    Video rows get a (t, h, w) patch grid; text and audio rows advance the shared `t` clock with
    h = w = 0. Text and the first video frame are clean (timestep 0), the rest noisy (timestep 1), so
    the AdaLN table is addressed at four distinct `(timestep, modality)` rows including row 0.
    """
    grid_h = grid_w = 8
    frame = grid_h * grid_w
    assert num_video % frame == 0, "num_video must fill whole (h, w) frames"
    grid_t = num_video // frame
    assert grid_t >= 2, "need at least one conditioning frame and one target frame"

    def clock(n):
        return torch.stack([torch.arange(n), torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long)], -1)

    vt, vh, vw = torch.meshgrid(torch.arange(grid_t), torch.arange(grid_h), torch.arange(grid_w), indexing="ij")

    return {
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
    ("num_text", "num_audio", "num_video"),
    [
        # Each modality's row count only needs to be a multiple of TILE (32); the packed sequence is
        # assembled globally and its tail padded to a multiple of SP * TILE (256).
        pytest.param(512, 256, 1280, id="small_s2048"),  # 2048: already aligned, no padding
        # 2112 is a multiple of TILE but not of SP * TILE, so this one exercises the padding path.
        pytest.param(512, 256, 1344, id="unaligned_s2112"),  # 2112 -> padded to 2304
        pytest.param(512, 256, 20736, id="s21504"),
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
    seq_len = num_text + num_audio + num_video
    per_modality = _modality_metadata(num_text, num_audio, num_video)

    # ---- reference layout: packed as [text | audio | video], contiguous ----
    ref_position_ids = torch.cat([per_modality[m]["pos"] for m in ("text", "audio", "video")])
    ref_tags = torch.cat([per_modality[m]["tags"] for m in ("text", "audio", "video")])
    ref_ts_idx = torch.cat([per_modality[m]["ts"] for m in ("text", "audio", "video")])
    text_indices = torch.arange(num_text)
    audio_indices = torch.arange(num_text, num_text + num_audio)
    video_indices = torch.arange(num_text + num_audio, seq_len)
    num_timesteps = int(ref_ts_idx.max().item()) + 1

    logger.info(
        f"seq_len={seq_len} (text={num_text} audio={num_audio} video={num_video}), "
        f"num_timesteps={num_timesteps}, layers={NUM_LAYERS} (reduced from 50)"
    )

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
    ).to(torch.float32)
    randomize_norm_weights(torch_model)
    torch_model.eval()

    video_input = torch.randn((B, num_video, VIDEO_PATCH_DIM), dtype=torch.float32)
    audio_input = torch.randn((B, num_audio, AUDIO_IN_CHANNELS), dtype=torch.float32)
    prompt_input = torch.randn((B, num_text, TEXT_DIM), dtype=torch.float32)
    # Timesteps are consumed unscaled in [0, 1]; one entry per distinct noise level.
    timestep = torch.rand((num_timesteps,), dtype=torch.float32)

    logger.info("Running torch model")
    with torch.no_grad():
        torch_out = torch_model(
            hidden_states=video_input,
            audio_hidden_states=audio_input,
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
    torch_video_out, torch_audio_out = torch_out.sample, torch_out.audio_sample
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
    tt_model.load_torch_state_dict(torch_model.state_dict())

    # The modality inputs are fully replicated: they are projected and concatenated into the packed
    # sequence before it is fractured, so every device needs all of them.
    tt_video = bf16_tensor(video_input.unsqueeze(0), device=mesh_device)
    tt_audio = bf16_tensor(audio_input.unsqueeze(0), device=mesh_device)
    tt_prompt = bf16_tensor(prompt_input.unsqueeze(0), device=mesh_device)
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


def _load_reference_state_dict(directory: Path) -> dict[str, torch.Tensor]:
    """Read a sharded safetensors checkpoint into one state dict, shard by shard."""
    index_path = directory / "diffusion_pytorch_model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    by_file: dict[str, list[str]] = defaultdict(list)
    for key, shard in weight_map.items():
        by_file[shard].append(key)

    state: dict[str, torch.Tensor] = {}
    for shard in sorted(by_file):
        with safe_open(directory / shard, framework="pt") as handle:
            for key in by_file[shard]:
                state[key] = handle.get_tensor(key)
    assert len(state) == len(weight_map), f"loaded {len(state)} of {len(weight_map)} tensors"
    return state


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
    ("num_text", "num_audio", "num_video"),
    [
        pytest.param(512, 256, 1280, id="small_s2048"),
        pytest.param(512, 256, 20736, id="s21504"),
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
    seq_len = num_text + num_audio + num_video
    hidden_size = model_kwargs["hidden_size"]
    audio_channels = model_kwargs["audio_in_channels"]
    video_patch_dim = model_kwargs["in_channels"] * int(torch.tensor(model_kwargs["patch_size"]).prod())

    per_modality = _modality_metadata(num_text, num_audio, num_video)
    position_ids = torch.cat([per_modality[m]["pos"] for m in ("text", "audio", "video")])
    tags = torch.cat([per_modality[m]["tags"] for m in ("text", "audio", "video")])
    ts_idx = torch.cat([per_modality[m]["ts"] for m in ("text", "audio", "video")])
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

    logger.info(f"running {model_kwargs['num_layers']} layers, seq_len={seq_len} (padded {padded_len})")

    def forward():
        out = tt_model(
            video_1BVC=tt_video,
            audio_1BAC=tt_audio,
            prompt_1BLP=tt_prompt,
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
