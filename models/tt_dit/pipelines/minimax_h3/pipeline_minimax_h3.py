# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 `t2va` and `fl2va`: a prompt, optionally a first and/or last keyframe, in; a video and
its synchronized soundtrack out.

Structured as the reference `MiniMaxH3Blocks` sequence so the two can be read side by side ---
setup, text encode, keyframe VAE encode, layout, latents, timesteps, denoise, decode. The reference
is guidance-distilled: one forward per step, no unconditional branch, no guider.

The three tasks the reference's `_workflow_map` names are all reachable from one `__call__`:
`t2va` (prompt only), `fl2va` (`image=`) and `fl2va_last_frame` (`last_image=`), plus both keyframes
together. A keyframe enters at two independent places --- the conditioner, as a `<Picture i>:` label
and a vision block whose rows are *video*-tagged, and the video VAE, as one noise-augmented anchor
frame of conditioning rows pinned at `t = 0.999` for every denoising step.

What runs where
---------------
The packed sequence holds all three modalities at once and is denoised by one 50-layer stack on
the mesh (TP=4 on axis 0, SP on axis 1 -- 8 on a Galaxy, 32 on a quad; see `_PRESETS_BH`).
Everything that decides *which* row gets which treatment is host-side and
already gated bit-exact against the reference --- the layout, the fp64 rotary grid, the per-row
timestep plan, both schedulers. That split is deliberate: those values are checkpoint contracts
where a reassociation is a silent desync between audio and video, and they cost nothing on host.

The scheduler steps run on host too, which is a bringup choice rather than a design one: the two
velocity read-backs are ~7 MB against a ~0.9 s step, and a host loop is debuggable. Moving them to
device is a perf question and out of scope here.

Residency
---------
The three big components do not fit at once: the DiT is ~16.6 GB/device at TP=4 (its adaLN
projections are resident --- this transformer computes modulation on device) and the video VAE
replicates ~9.8 GB of fp32 weights per device for its data-parallel fan-out. They are
registered as ``Module`` coresident exclusions: loading one stage evicts the others. The text
encoder is the cheap one --- FSDP over the non-TP axis puts it at ~1.6 GB/device --- but its 50 GB
disk read is not, which is why it is kept co-resident by default: every request pays the encode
itself (~2.8 s), never the reload. `coresident=False` evicts each stage and disables tracing.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import tqdm
from loguru import logger
from PIL import Image, ImageOps

import ttnn

from ...encoders.qwen3vl.loader_minimax_h3 import (
    MINIMAX_H3_TEXT_ENCODER_LAYER,
    build_minimax_h3_text_encoder,
    build_minimax_h3_vision_tower,
    load_minimax_h3_text_encoder_weights,
)
from ...encoders.qwen3vl.model_qwen3vl import create_rope_tensors, mrope_position_ids, vision_token_runs
from ...encoders.qwen3vl.vision_qwen3vl import pad_patches_for_sp, vision_cu_seqlens
from ...layers.audio_ops import weights_variant
from ...models.audio_vae.minimax_h3.convert_minimax_h3_audio import convert_minimax_h3_audio_state_dict
from ...models.audio_vae.minimax_h3.decoder_minimax_h3_audio import MiniMaxH3AudioDecoder
from ...models.audio_vae.minimax_h3.encoder_minimax_h3_audio import MiniMaxH3AudioEncoder
from ...models.transformers.minimax_h3.attention_minimax_h3 import prepare_rope_tables
from ...models.transformers.minimax_h3.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from ...models.vae.minimax_h3.vae_minimax_h3 import MiniMaxH3Vae, MiniMaxH3VaeConfig
from ...parallel.config import DiTParallelConfig, EncoderParallelConfig, ParallelFactor, VAEParallelConfig
from ...parallel.manager import CCLManager
from ...utils import cache
from ...utils.conv3d import conv3d_blocking_hash
from ...utils.tensor import bf16_tensor, from_torch, local_device_to_torch
from ...utils.tracing import StateTensor
from ..events import DenoiseStep, PipelineEventCallback, event_section, null_callback
from .conditioning import MINIMAX_H3_PIXEL_MEAN as _MINIMAX_H3_PIXEL_MEAN
from .conditioning import MINIMAX_H3_PIXEL_STD as _MINIMAX_H3_PIXEL_STD
from .conditioning import encode_keyframes, keyframe_condition_noise
from .packing import (
    MINIMAX_H3_ADALN_ROLES,
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_AUDIO_LATENTS_PER_SECOND,
    MINIMAX_H3_FPS,
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MINIMAX_H3_MAX_DURATION,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
    MiniMaxH3PackedSequence,
    adaln_indices,
    align_num_frames,
    audio_latent_num_frames,
    build_packed_sequence,
    build_rope_tables,
    build_slot_routing,
    patchify_video_latents,
    prepare_keyframe_image,
    resolve_canvas_size,
    slot_levels,
    unpack_audio_tokens,
    unpatchify_video_tokens,
    video_latent_num_frames,
)
from .packing_ref2va import (
    MINIMAX_H3_MAX_REFERENCE_IMAGES,
    MiniMaxH3PreparedReference,
    MiniMaxH3Reference,
    build_ref2va_packed_sequence,
    build_ref2va_presentation,
    sample_reference_video_frames,
)
from .policy import (
    MINIMAX_H3_DEFAULT_ASPECT_RATIO,
    MINIMAX_H3_DURATIONS_S,
    MINIMAX_H3_MAX_TEXT_TOKENS,
    MINIMAX_H3_SERVED_REFERENCE_RESIZE_MODE,
    served_canvases,
    served_envelope,
    served_reference_image_sizes,
    served_reference_video_canvases,
)
from .references import encode_references, prepare_references, reference_condition_shapes, split_condition_blocks
from .scheduler import MiniMaxH3Scheduler

# ImageNet statistics; the video VAE emits normalized RGB and the pipeline reverts it. Imported from
# `conditioning` rather than restated: the keyframe path normalizes *into* the VAE with these and the
# decode path reverts *out* of it with them, so two copies would make a drift a silent asymmetric
# bug -- encode with one, decode with the other -- rather than an obvious one.
MINIMAX_H3_PIXEL_MEAN = _MINIMAX_H3_PIXEL_MEAN
MINIMAX_H3_PIXEL_STD = _MINIMAX_H3_PIXEL_STD

# The timestep a ref2va reference soundtrack's rows run at: a literal 1.0, every step. They are
# clean -- posterior mean, no fp16 round trip, no noise augmentation -- unlike the visual
# conditioning rows, which sit at max(t, 0.999). See `references.py`.
MINIMAX_H3_AUDIO_CONDITION_TIMESTEP = 1.0

# Read from the two scheduler_config.json files, which hold nothing else.
VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0


_AUDIO_T_FACTOR_ENV = "MINIMAX_H3_AUDIO_T_FACTOR"
_DEFAULT_AUDIO_T_FACTOR = 8
_AUDIO_PACK_BANDS = {5: 2, 6: 4}


def _requested_audio_t_factor(audio_t_factor: int | None, default: int = _DEFAULT_AUDIO_T_FACTOR) -> tuple[int, bool]:
    """Explicit kwarg wins; else MINIMAX_H3_AUDIO_T_FACTOR; else `default`. Returns (factor, from_env)."""
    if audio_t_factor is not None:
        return audio_t_factor, False
    raw = os.environ.get(_AUDIO_T_FACTOR_ENV)
    if raw is None:
        return default, False
    try:
        return int(raw), True
    except ValueError:
        raise ValueError(f"{_AUDIO_T_FACTOR_ENV}={raw!r} must be an integer T-shard factor") from None


def _resolve_audio_t_shard(
    requested_factor: int, mesh_shape: tuple[int, ...], tp_axis: int, sp_axis: int
) -> tuple[int, int | None]:
    """Largest factor in (32, 8, 4) that is <= requested and matches a mesh axis (TP before SP), else
    (1, None) unsharded. Never shards higher than requested, so the default request of 8 caps the
    chain at 8; 32 (the quad's inter-host axis) is opt-in via audio_t_factor=32 and unvalidated."""
    for factor in (32, 8, 4):
        if factor <= requested_factor:
            axis = next((ax for ax in (tp_axis, sp_axis) if mesh_shape[ax] == factor), None)
            if axis is not None:
                return factor, axis
    return 1, None


# Cache namespace under TT_DIT_CACHE_DIR. `utils.cache` keys each entry on this plus the subfolder,
# the parallel config, the mesh shape, the dtype and the FSDP flag.
MODEL_NAME = "minimax-h3"

# Padded packed lengths served by resident denoise traces (one capture per rung), multiples of 1024
# for SP alignment. The top rung is the admission cap: a longer request raises.
MINIMAX_H3_BUCKET_LADDER = (22528, 31744, 44032, 61440, 86016, 120832)

# ref2va ladder; the top rung must admit everything the ref2va arena caps do (322336 rows, aligned).
MINIMAX_H3_REF2VA_BUCKET_LADDER = (32768, 61440, 86016, 118784, 176128, 245760, 322560)

# ref2va text-encoder pad targets below the prompt arena cap; the cap itself is always the top rung.
MINIMAX_H3_REF2VA_PRESENTATION_RUNGS = (1024, 4096, 8192, 16384, 32768)


def default_bucket_ladder(task: str) -> tuple[int, ...]:
    return MINIMAX_H3_REF2VA_BUCKET_LADDER if task == "ref2va" else MINIMAX_H3_BUCKET_LADDER


def validate_bucket_ladder(ladder: tuple[int, ...], alignment: int) -> None:
    """Raise unless `ladder` is non-empty, strictly ascending and rung-aligned to `alignment`."""
    if not ladder:
        raise ValueError("bucket_ladder must not be empty")
    if list(ladder) != sorted(set(ladder)):
        raise ValueError(f"bucket_ladder must be strictly ascending, got {ladder}")
    misaligned = tuple(rung for rung in ladder if rung % alignment)
    if misaligned:
        raise ValueError(f"bucket_ladder rungs {misaligned} are not multiples of sp_factor * TILE = {alignment}")


def select_bucket(seq_len: int, ladder: tuple[int, ...]) -> int:
    """The smallest rung >= `seq_len`. The top rung is the admission cap: beyond it raises."""
    for rung in ladder:
        if seq_len <= rung:
            return rung
    raise ValueError(
        f"packed sequence length {seq_len} exceeds the top trace bucket {ladder[-1]} "
        f"(ladder {ladder}); shorten the request or deploy with a taller ladder"
    )


class SeqLen(NamedTuple):
    padded: int
    logical: int


@dataclass(frozen=True)
class MiniMaxH3ArenaCaps:
    """Fixed per-deployment row capacities of the device arenas; a request exceeding one raises."""

    prompt: int = 5120
    video_rows: int = 111712
    audio_rows: int = 1216
    condition_video_rows: int = 2112
    condition_audio_rows: int = 1216

    @classmethod
    def for_task(cls, task: str) -> "MiniMaxH3ArenaCaps":
        """Defaults sized to the task's envelope; ref2va needs much larger prompt and conditioning caps."""
        if task == "ref2va":
            return cls(prompt=57344, condition_video_rows=149632, condition_audio_rows=2432)
        return cls()

    def validate(self) -> None:
        for cap in fields(self):
            value = getattr(self, cap.name)
            if value <= 0 or value % ttnn.TILE_SIZE:
                raise ValueError(f"arena cap {cap.name}={value} must be a positive multiple of {ttnn.TILE_SIZE}")


@dataclass
class _BucketState:
    """Per-rung device state whose shape depends on the padded length; bound once, never rebound."""

    rope_cos: StateTensor = field(default_factory=StateTensor)
    rope_sin: StateTensor = field(default_factory=StateTensor)
    adaln: StateTensor = field(default_factory=StateTensor)
    tsi: StateTensor = field(default_factory=StateTensor)
    assembly_idx: StateTensor = field(default_factory=StateTensor)
    warm: bool = False


# Per-mesh-shape defaults, following `pipelines/wan/pipeline_wan.py`'s `_PRESETS_BH`. An unlisted
# shape raises rather than defaulting, so it cannot silently ring-collective over a line fabric.
#
# TP stays on axis 0 at factor 4 and SP absorbs every extra device: TP does a per-layer collective and
# axis 0 is intra-host, while SP hides its KV all-gather inside ring attention and tolerates the
# inter-host hop. TP=4 also fits the shapes -- 56 // 4 = 14 heads, 5376 % (32 * 4) == 0 for the norms.
_PRESETS_BH: dict[tuple[int, ...], dict] = {
    # One Blackhole Galaxy: the working point MiniMaxH3.md documents.
    (4, 8): {"tp_axis": 0, "sp_axis": 1, "num_links": 2, "topology": ttnn.Topology.Ring, "coresident": True},
    # Quad Blackhole Galaxy, 4 MPI hosts x 32 chips. Same axes, links and topology; SP goes 8 -> 32,
    # which moves the SP alignment to 32 * TILE_SIZE = 1024 and re-keys every packed length.
    #
    # `trace_denoise` is quad-only, mirroring Wan's `traced = mesh_shape == (4, 32)`: at SP=32 a step
    # is dispatch-bound, so the trace is what makes the extra devices pay. 4x8 has enough work per
    # chip to not need it.
    (4, 32): {
        "tp_axis": 0,
        "sp_axis": 1,
        "num_links": 2,
        "topology": ttnn.Topology.Ring,
        # Every stage stays resident: evicting the transformer drops the per-request buffers it caches
        # (the padded-sequence zero rows), and rebuilding those inside a trace capture is a fatal write.
        "coresident": True,
        "trace_denoise": True,
        "audio_t_shard": True,
        "audio_t_factor": 32,
    },
}


def resolve_mesh_preset(mesh_shape: tuple[int, ...], *, required: bool = True) -> dict:
    """The measured defaults for this mesh shape, or `{}` when unlisted and `required` is False.

    An unlisted shape is only an error when something is left to the preset to fill in; a caller that
    passes every parallel setting explicitly is running an untuned shape deliberately.
    """
    shape = tuple(mesh_shape)
    preset = _PRESETS_BH.get(shape)
    if preset is None:
        if not required:
            return {}
        known = ", ".join(str(s) for s in _PRESETS_BH)
        msg = (
            f"no MiniMax-H3 preset for mesh shape {shape}; known shapes are {known}. Pass tp_axis, "
            "sp_axis, num_links and topology explicitly to run an untuned shape."
        )
        raise ValueError(msg)
    return preset


def draw_request_latents(
    generator: torch.Generator,
    *,
    condition_latent_shapes: tuple[tuple[int, int, int], ...],
    latent_channels: int,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    audio_latent_channels: int,
    patch_size: tuple[int, int, int],
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """`(condition_noise, video_rows, audio_rows)` off one generator, in the reference's draw order.

    The *order* is part of what the seed reproduces, and it is not the obvious one: the keyframe
    conditioning noise is drawn **first**, ahead of the video and then the audio latents. That is
    diffusers' order (`MiniMaxH3KeyframeVaeEncoderStep` runs before `MiniMaxH3PrepareLatentsStep`), not
    sglang's re-seed-per-condition, and getting it wrong changes every latent in the request rather
    than just the conditioning rows.

    A module-level function rather than inline in `__call__` so the order is testable without a mesh.
    With `condition_latent_shapes=()` it draws nothing extra and does not advance the generator, so
    `t2va` at a given seed is unaffected.
    """
    condition_noise = None
    if condition_latent_shapes:
        condition_noise = keyframe_condition_noise(
            condition_latent_shapes, latent_channels, patch_size, generator=generator
        )
    video_latents = torch.randn(
        (1, latent_channels, num_latent_frames, latent_height, latent_width),
        generator=generator,
        dtype=torch.float32,
    )
    video_rows = patchify_video_latents(video_latents, patch_size)
    audio_rows = torch.randn(
        (num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS, audio_latent_channels),
        generator=generator,
        dtype=torch.float32,
    )
    return condition_noise, video_rows, audio_rows


@dataclass
class MiniMaxH3Output:
    """One generation. `audio` is `(1, 2, samples)`; `video` is `(1, 3, F, H, W)` float for `"rgb_float"`
    or a planar `(F, H * 3 // 2, W)` uint8 numpy array for `"yuv420"`."""

    video: torch.Tensor
    audio: torch.Tensor
    sampling_rate: int
    num_frames: int
    fps: int = MINIMAX_H3_FPS
    video_format: str = "rgb_float"

    @property
    def video_seconds(self) -> float:
        return self.num_frames / self.fps

    @property
    def audio_seconds(self) -> float:
        return self.audio.shape[-1] / self.sampling_rate


def _is_host_rank() -> bool:
    return not ttnn.using_distributed_env() or int(ttnn.distributed_context_get_rank()) == 0


_TQDM_BAR_FORMAT = "\033[A\r\033[K{l_bar}{bar}{r_bar}\n"


def _tqdm_spacer() -> None:
    sys.stderr.write("\n")
    sys.stderr.flush()


class MiniMaxH3Pipeline:
    """`t2va` and `fl2va` on a Blackhole mesh. Build with `create_pipeline`."""

    def __init__(
        self,
        *,
        mesh_device: ttnn.MeshDevice,
        weights_dir: str | os.PathLike,
        tp_axis: int | None = None,
        sp_axis: int | None = None,
        num_links: int | None = None,
        topology: ttnn.Topology | None = None,
        coresident: bool | None = None,
        task: str = "t2va",
        audio_split_mode: str | None = None,
        audio_t_factor: int | None = None,
        audio_trace: bool | None = None,
        dit_fsdp: bool = False,
        trace_denoise: bool | None = None,
        bucket_denoise: bool | None = None,
        bucket_ladder: tuple[int, ...] | None = None,
        arena_caps: MiniMaxH3ArenaCaps | None = None,
        vae_output_type: str = "yuv420",
        adaln_slot_roles: tuple[str, ...] | None = None,
        warmup: bool = True,
    ) -> None:
        self.mesh_device = mesh_device
        self.weights_dir = Path(weights_dir)
        supplied = (tp_axis, sp_axis, num_links, topology)
        preset = resolve_mesh_preset(tuple(mesh_device.shape), required=any(v is None for v in supplied))
        tp_axis = preset["tp_axis"] if tp_axis is None else tp_axis
        sp_axis = preset["sp_axis"] if sp_axis is None else sp_axis
        num_links = preset["num_links"] if num_links is None else num_links
        topology = preset["topology"] if topology is None else topology
        coresident = preset.get("coresident", True) if coresident is None else coresident
        self.trace_denoise = preset.get("trace_denoise", False) if trace_denoise is None else trace_denoise
        env_audio_t_shard = os.environ.get("MINIMAX_H3_AUDIO_T_SHARD")
        self.audio_t_shard = (
            env_audio_t_shard == "1" if env_audio_t_shard is not None else preset.get("audio_t_shard", False)
        )
        self.coresident = coresident
        self.trace_denoise = self.trace_denoise and self.coresident
        self.bucket_denoise = self.trace_denoise or bool(bucket_denoise)
        self._log_generation = True
        self._buckets: dict[int, _BucketState] = {}
        self._force_bucket: int | None = None
        self._force_prompt_pad: int | None = None
        self._tt_video = StateTensor()
        self._tt_audio = StateTensor()
        self._tt_cond_video = StateTensor()
        self._tt_cond_audio = StateTensor()
        self._tt_video_out_idx = StateTensor()
        self._tt_audio_out_idx = StateTensor()
        self._tt_timestep = StateTensor()
        self._tt_logical_n = StateTensor()
        # One repository holds both partitions -- `transformer/` for t2va/fl2va and
        # `transformer_ref/` for ref2va -- with byte-identical `config.json`, so only the
        # weights differ. Fixed at construction because each is 62 GB and switching would
        # mean a full reload.
        if task not in ("t2va", "ref2va"):
            raise ValueError(f"task must be 't2va' (also serves fl2va) or 'ref2va', got {task!r}")
        self.task = task
        self.transformer_subfolder = "transformer_ref" if task == "ref2va" else "transformer"
        self.tp_axis, self.sp_axis = tp_axis, sp_axis
        shape = tuple(mesh_device.shape)
        if tp_axis == sp_axis:
            msg = f"tp_axis and sp_axis must differ, both are {tp_axis}"
            raise ValueError(msg)
        self.tp_factor, self.sp_factor = shape[tp_axis], shape[sp_axis]
        # The only residency control; see `_make_resident` for the measurements behind the default.
        self.coresident = coresident
        if audio_split_mode is None:
            audio_split_mode = "kernel"
        if audio_split_mode not in ("off", "weight", "full", "kernel"):
            raise ValueError(f"audio_split_mode must be 'off', 'weight', 'full' or 'kernel', got {audio_split_mode!r}")
        self.audio_split_mode = audio_split_mode
        self.audio_trace = True if audio_trace is None else bool(audio_trace)
        audio_t_factor, self._audio_t_factor_from_env = _requested_audio_t_factor(
            audio_t_factor, default=preset.get("audio_t_factor", _DEFAULT_AUDIO_T_FACTOR)
        )
        self.audio_t_factor, self._audio_t_axis = _resolve_audio_t_shard(
            audio_t_factor, shape, self.tp_axis, self.sp_axis
        )

        self.bucket_ladder = tuple(bucket_ladder if bucket_ladder is not None else default_bucket_ladder(task))
        validate_bucket_ladder(self.bucket_ladder, self.sp_factor * ttnn.TILE_SIZE)
        self.arena_caps = arena_caps or MiniMaxH3ArenaCaps.for_task(task)
        self.arena_caps.validate()
        self.presentation_ladder = tuple(
            rung for rung in MINIMAX_H3_REF2VA_PRESENTATION_RUNGS if rung < self.arena_caps.prompt
        ) + (self.arena_caps.prompt,)
        if self.bucket_denoise:
            caps = self.arena_caps
            admissible = caps.prompt + caps.condition_video_rows + caps.audio_rows + caps.video_rows
            if task == "ref2va":
                admissible += caps.condition_audio_rows
            if admissible > self.bucket_ladder[-1]:
                raise ValueError(
                    f"the arena caps admit a packed length up to {admissible}, beyond the top trace "
                    f"bucket {self.bucket_ladder[-1]}; raise the ladder or lower the caps"
                )
        if adaln_slot_roles is None:
            adaln_slot_roles = MINIMAX_H3_ADALN_ROLES if task == "ref2va" else ("video", "audio", "condition_video")
        unknown = tuple(role for role in adaln_slot_roles if role not in MINIMAX_H3_ADALN_ROLES)
        if unknown:
            raise ValueError(f"unknown AdaLN slot roles {unknown}; valid roles are {MINIMAX_H3_ADALN_ROLES}")
        self.adaln_slot_roles = tuple(adaln_slot_roles)

        self.ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
        self.audio_ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=ttnn.Topology.Linear)
        self.dit_parallel_config = DiTParallelConfig(
            tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=self.tp_factor),
            sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=self.sp_factor),
            cfg_parallel=None,
        )

        self.encoder_parallel_config = EncoderParallelConfig(
            tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=self.tp_factor),
            sequence_parallel=(
                ParallelFactor(mesh_axis=sp_axis, factor=self.sp_factor) if self.sp_factor > 1 else None
            ),
        )

        self.vae_parallel_config = VAEParallelConfig(tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=1))

        # Read from the partition that will actually be loaded, rather than assuming the
        # two configs stay byte-identical.
        self.transformer_config = self._read_config(self.transformer_subfolder)
        self.vae_config = MiniMaxH3VaeConfig.from_pretrained(self.weights_dir / "vae")
        self.audio_config = self._read_config("audio_vae")
        # The rotary tables are the caller's job, so these two are config for this class rather than
        # constructor arguments to the model.
        self.rope_freq_dim = self.transformer_config["rope_freq_dim"]
        self.rope_theta = self.transformer_config["rope_theta"]

        self._tokenizer = None
        self._host_text_encoder = None
        self._host_vae = None
        self._host_vae_encoder_loaded = False
        self._host_vae_decoder_loaded = False
        self._image_processor = None
        if vae_output_type not in ("float", "uint8", "yuv420"):
            raise ValueError(f"vae_output_type must be 'float', 'uint8' or 'yuv420', got {vae_output_type!r}")
        self.vae_output_type = vae_output_type
        self.vae_waves_per_device = 2
        self._video_processor = None
        self._vision_tower = None
        self._vision_config = None
        self._audio_decoder = None
        self._audio_encoder = None
        self.dit_fsdp = dit_fsdp
        self.last_seq_len: SeqLen | None = None

        self._host_log("building the Qwen3-VL text encoder")
        self.encoder_ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
        self._text_encoder, self._text_config = build_minimax_h3_text_encoder(
            self.weights_dir / "text_encoder",
            mesh_device=self.mesh_device,
            parallel_config=self.encoder_parallel_config,
            ccl_manager=self.encoder_ccl_manager,
            is_fsdp=True,
            load_weights=False,
        )
        self._transformer = self._build_transformer()
        yuv = self.vae_output_type == "yuv420"
        unit_pixels = self.vae_output_type in ("uint8", "yuv420")
        self._host_log("building the video VAE")
        self._vae = MiniMaxH3Vae(
            self.vae_config,
            task=self.task,
            mesh_device=self.mesh_device,
            weight_loader=self._cache_submodel,
            ccl_manager=self.encoder_ccl_manager,
            dtype=ttnn.bfloat16,
            device_stitch=yuv,
            pixel_denorm=(MINIMAX_H3_PIXEL_MEAN, MINIMAX_H3_PIXEL_STD) if unit_pixels else None,
            pixel_norm=(MINIMAX_H3_PIXEL_MEAN, MINIMAX_H3_PIXEL_STD),
            readback_uint8=self.vae_output_type == "uint8",
            waves_per_device=self.vae_waves_per_device,
        )
        self._vae.load_state(self._read_safetensors("vae"))

        if not self.coresident:
            self._text_encoder.register_coresident_exclusions(self._transformer, *self._vae.modules)
            self._transformer.register_coresident_exclusions(self._text_encoder, *self._vae.modules)
            for module in self._vae.modules:
                module.register_coresident_exclusions(self._text_encoder, self._transformer)

        if self.coresident:
            self._prepare_transformer()
        self._prepare_text_encoder()
        self._prepare_audio_decoder()

        if warmup:
            self._warmup_on_init()

    # ------------------------------------------------------------------ construction

    @classmethod
    def create_pipeline(
        cls,
        *,
        mesh_device: ttnn.MeshDevice,
        weights_dir: str | os.PathLike | None = None,
        tp_axis: int | None = None,
        sp_axis: int | None = None,
        num_links: int | None = None,
        topology: ttnn.Topology | None = None,
        task: str = "t2va",
        audio_split_mode: str | None = None,
        audio_t_factor: int | None = None,
        audio_trace: bool | None = None,
        dit_fsdp: bool = False,
        trace_denoise: bool | None = None,
        bucket_denoise: bool | None = None,
        bucket_ladder: tuple[int, ...] | None = None,
        arena_caps: MiniMaxH3ArenaCaps | None = None,
        vae_output_type: str = "yuv420",
        adaln_slot_roles: tuple[str, ...] | None = None,
        warmup: bool = True,
    ) -> "MiniMaxH3Pipeline":
        """`task="t2va"` serves both t2va and fl2va; `task="ref2va"` loads `transformer_ref/`.

        The parallel configuration defaults to this mesh shape's entry in `_PRESETS_BH`; pass any of
        `tp_axis`/`sp_axis`/`num_links`/`topology` to override it.

        `trace_denoise` defaults to the mesh preset; `bucket_ladder`, `arena_caps` and `adaln_slot_roles`
        default to the task's envelope.
        """
        weights_dir = weights_dir or os.environ.get("MINIMAX_H3_MODEL_PATH")
        if not weights_dir:
            raise ValueError(
                "MiniMax-H3 weights directory not set: pass weights_dir=... or set MINIMAX_H3_MODEL_PATH "
                "to a diffusers snapshot holding transformer/, text_encoder/, vae/ and audio_vae/."
            )
        return cls(
            mesh_device=mesh_device,
            weights_dir=weights_dir,
            tp_axis=tp_axis,
            sp_axis=sp_axis,
            num_links=num_links,
            topology=topology,
            task=task,
            audio_split_mode=audio_split_mode,
            audio_trace=audio_trace,
            audio_t_factor=audio_t_factor,
            dit_fsdp=dit_fsdp,
            trace_denoise=trace_denoise,
            bucket_denoise=bucket_denoise,
            bucket_ladder=bucket_ladder,
            arena_caps=arena_caps,
            vae_output_type=vae_output_type,
            adaln_slot_roles=adaln_slot_roles,
            warmup=warmup,
        )

    def _read_config(self, subfolder: str) -> dict:
        path = self.weights_dir / subfolder / "config.json"
        if not path.is_file():
            raise FileNotFoundError(f"no {subfolder}/config.json under {self.weights_dir}")
        return {k: v for k, v in json.loads(path.read_text()).items() if not k.startswith("_")}

    def _read_safetensors(self, subfolder: str) -> dict[str, torch.Tensor]:
        """A partition's weights, sharded or single-file. `transformer` and `vae` are sharded here."""
        from safetensors.torch import load_file

        directory = self.weights_dir / subfolder
        index = directory / "diffusion_pytorch_model.safetensors.index.json"
        state: dict[str, torch.Tensor] = {}
        if index.is_file():
            for shard in sorted(set(json.loads(index.read_text())["weight_map"].values())):
                state.update(load_file(str(directory / shard)))
        else:
            single = directory / "diffusion_pytorch_model.safetensors"
            if not single.is_file():
                raise FileNotFoundError(f"no safetensors (sharded or single) under {directory}")
            state.update(load_file(str(single)))
        return state

    # ------------------------------------------------------------------ residency

    def _host_log(self, message: str) -> None:
        """Construction / prepare logs: host rank only, including during warmup."""
        if _is_host_rank():
            logger.info(message)

    def _log(self, message: str) -> None:
        """Generation logs: host rank, measured call only. Warmup is silent."""
        if self._log_generation:
            self._host_log(message)

    @contextmanager
    def quiet(self):
        """Silence generation logs (per-step, packed length, VAE profile) for this call."""
        previous = self._log_generation
        self._log_generation = False
        try:
            yield
        finally:
            self._log_generation = previous

    # ------------------------------------------------------------------ text

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(str(self.weights_dir), subfolder="tokenizer")
        return self._tokenizer

    @property
    def image_processor(self):
        """The checkpoint's own image processor. It decides the patch grid, so nothing else may."""
        if self._image_processor is None:
            from transformers import AutoImageProcessor

            self._image_processor = AutoImageProcessor.from_pretrained(str(self.weights_dir), subfolder="text_encoder")
        return self._image_processor

    def _build_presentation(self, prompt: str, keyframes: Sequence[Image.Image]):
        """MiniMax-H3's token presentation, exactly as `encoders.py::encode_prompt` assembles it.

        Returns `(input_ids [1, L], token_tags [L], mm_token_type_ids [1, L], pixel_values, grid_thw)`.

        Per keyframe: a `"<Picture i>: "` label, then `<|vision_start|>`, then one `<|image_pad|>` per
        *merged* vision patch, then `<|vision_end|>`. Then the prompt, verbatim. No chat template and
        `add_special_tokens=False` throughout, so no BOS/EOS.

        Two different taggings come out of this and conflating them is a silent error:

        - `token_tags` is **H3's** per-row modality for the DiT's AdaLN, and the *whole vision block*
          including `<|vision_start|>`/`<|vision_end|>` is video-tagged.
        - `mm_token_type_ids` is **Qwen3-VL's**, feeding its own 3-D rotary grid, and marks only the
          `<|image_pad|>` run as image; the start/end sentinels count as text there.
        """
        tokenizer = self.tokenizer
        image_pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        vision_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end = tokenizer.convert_tokens_to_ids("<|vision_end|>")

        token_ids: list[int] = []
        token_tags: list[int] = []
        pixel_values = grid_thw = None
        if keyframes:
            processor = self.image_processor
            vision = processor(images=list(keyframes), return_tensors="pt")
            pixel_values, grid_thw = vision["pixel_values"], vision["image_grid_thw"]
            merge = processor.merge_size**2
            for index in range(len(keyframes)):
                num_image_tokens = int(grid_thw[index].prod()) // merge
                label = tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
                block = [vision_start] + [image_pad] * num_image_tokens + [vision_end]
                token_ids += label + block
                token_tags += [MINIMAX_H3_TEXT_TAG] * len(label) + [MINIMAX_H3_VIDEO_TAG] * len(block)

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if not prompt_ids:
            raise ValueError("prompt tokenized to zero tokens")
        if len(prompt_ids) > MINIMAX_H3_MAX_TEXT_TOKENS:
            raise ValueError(f"prompt is {len(prompt_ids)} tokens, over the {MINIMAX_H3_MAX_TEXT_TOKENS}-token budget")
        token_ids += prompt_ids
        token_tags += [MINIMAX_H3_TEXT_TAG] * len(prompt_ids)

        input_ids = torch.tensor([token_ids], dtype=torch.long)
        return (
            input_ids,
            torch.tensor(token_tags, dtype=torch.long),
            (input_ids == image_pad).long(),
            pixel_values,
            grid_thw,
        )

    @property
    def video_processor(self):
        """The checkpoint's own video processor, which decides a reference video's patch grid."""
        if self._video_processor is None:
            from transformers import AutoVideoProcessor

            self._video_processor = AutoVideoProcessor.from_pretrained(str(self.weights_dir), subfolder="text_encoder")
        return self._video_processor

    def _build_ref2va_presentation(self, prompt: str, references: Sequence):
        """H3's token presentation of a `ref2va` request, and the vision patches behind it.

        Returns `(input_ids [1, L], token_tags [L], mm_token_type_ids [1, L], pixel_values, grid_thw)`,
        with the vision inputs concatenated in **presentation order** -- which is the whole reason this
        is not two separate tower calls. `_scatter_rows` consumes the tower's merged rows *in run
        order*, so image and video patches batched separately (images first, then videos) would land in
        the wrong rows for any request whose video reference precedes an image. Concatenating both here,
        in reference order, makes the tower's output already correct and removes the reordering step
        that could disagree with the layout.

        Two taggings come out of this and conflating them is silent:

        - `token_tags` is **H3's** per-row modality for the DiT's AdaLN, and the *whole vision block*
          including its `<|vision_start|>` / `<|vision_end|>` sentinels is video-tagged.
        - `mm_token_type_ids` is **Qwen3-VL's**, feeding its own 3-D rotary grid, and marks only the pad
          runs -- `1` for `<|image_pad|>` and `2` for `<|video_pad|>`, with the sentinels counting as
          text.

        A video reference's `block_timestamps` are filled in here, because they come from the same 2 fps
        sampling that produces the frames the processor sees.
        """
        import numpy as _np

        tokenizer = self.tokenizer
        image_pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        video_pad = tokenizer.convert_tokens_to_ids("<|video_pad|>")

        def split(patches, grids):
            """Per-grid-entry patch rows, in the order the processor produced them."""
            counts = [int(grid.prod()) for grid in grids]
            assert sum(counts) == patches.shape[0], f"{sum(counts)} patches expected, processor gave {patches.shape[0]}"
            out, cursor = [], 0
            for count in counts:
                out.append(patches[cursor : cursor + count])
                cursor += count
            return out

        merge = self.image_processor.merge_size**2

        images = [reference.image for reference in references if reference.kind == "image"]
        image_patches, image_grids, image_token_counts = [], [], []
        if images:
            vision = self.image_processor(images=images, return_tensors="pt")
            image_grids = list(vision["image_grid_thw"])
            image_patches = split(vision["pixel_values"], image_grids)
            image_token_counts = [int(grid.prod()) // merge for grid in image_grids]

        videos = [reference for reference in references if reference.kind == "video"]
        video_patches, video_grids, video_block_token_counts = [], [], []
        if videos:
            sampled = [sample_reference_video_frames(reference.frames) for reference in videos]
            for reference, (_, block_timestamps) in zip(videos, sampled):
                reference.block_timestamps = block_timestamps
            vision = self.video_processor(
                videos=[_np.stack(frames) for frames, _ in sampled], do_sample_frames=False, return_tensors="pt"
            )
            video_grids = list(vision["video_grid_thw"])
            video_patches = split(vision["pixel_values_videos"], video_grids)
            # One vision block per merged frame pair, each labelled with a timestamp. A
            # processor that merged differently than the sampling predicted leaves the labels
            # and the blocks off by one.
            for reference, grid in zip(videos, video_grids):
                if int(grid[0]) != len(reference.block_timestamps):
                    raise ValueError(
                        f"the processor merged a reference video into {int(grid[0])} vision blocks but H3 "
                        f"labels {len(reference.block_timestamps)} of them"
                    )
            video_block_token_counts = [int(grid[1]) * int(grid[2]) // merge for grid in video_grids]

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if not prompt_ids:
            raise ValueError("prompt tokenized to zero tokens")
        if len(prompt_ids) > MINIMAX_H3_MAX_TEXT_TOKENS:
            raise ValueError(f"prompt is {len(prompt_ids)} tokens, over the {MINIMAX_H3_MAX_TEXT_TOKENS}-token budget")

        token_ids, token_tags = build_ref2va_presentation(
            tokenizer, prompt, references, image_token_counts, video_block_token_counts
        )
        if not token_ids:
            raise ValueError("the ref2va presentation tokenized to zero tokens")

        # Vision inputs in presentation order: the same walk the presentation used. `kinds`
        # comes along because a grid entry cannot say which modality it is -- a one-block video
        # has `t == 1` like an image -- and the rotary grid needs them separated by modality.
        ordered_patches, ordered_grids, kinds = [], [], []
        image_index = video_index = 0
        for reference in references:
            if reference.kind == "image":
                ordered_patches.append(image_patches[image_index])
                ordered_grids.append(image_grids[image_index])
                kinds.append("image")
                image_index += 1
            elif reference.kind == "video":
                ordered_patches.append(video_patches[video_index])
                ordered_grids.append(video_grids[video_index])
                kinds.append("video")
                video_index += 1

        input_ids = torch.tensor([token_ids], dtype=torch.long)
        type_ids = torch.zeros_like(input_ids)
        type_ids[input_ids == image_pad] = 1
        type_ids[input_ids == video_pad] = 2
        return (
            input_ids,
            torch.tensor(token_tags, dtype=torch.long),
            type_ids,
            torch.cat(ordered_patches) if ordered_patches else None,
            torch.stack(ordered_grids) if ordered_grids else None,
            kinds,
        )

    def _prepare_text_encoder(self):
        """Load the on-device Qwen3-VL conditioner. ``cache.load_model`` no-ops if already resident."""
        load_minimax_h3_text_encoder_weights(
            self._text_encoder,
            self.weights_dir / "text_encoder",
            parallel_config=self.encoder_parallel_config,
            mesh_device=self.mesh_device,
            is_fsdp=True,
        )
        return self._text_encoder

    def _prepare_vision_tower(self):
        if self._vision_tower is None:
            if self.sp_factor > 1:
                self._host_log(f"building the Qwen3-VL vision tower (tp{self.tp_factor}_sp{self.sp_factor})")
                self._vision_tower, self._vision_config = build_minimax_h3_vision_tower(
                    self.weights_dir / "text_encoder",
                    mesh_device=self.mesh_device,
                    parallel_config=EncoderParallelConfig(
                        tensor_parallel=ParallelFactor(mesh_axis=self.tp_axis, factor=self.tp_factor),
                        sequence_parallel=ParallelFactor(mesh_axis=self.sp_axis, factor=self.sp_factor),
                    ),
                    ccl_manager=self.encoder_ccl_manager,
                )
            else:
                self._host_log("building the Qwen3-VL vision tower (replicated)")
                self._vision_tower, self._vision_config = build_minimax_h3_vision_tower(
                    self.weights_dir / "text_encoder", mesh_device=self.mesh_device
                )
        return self._vision_tower

    def encode_prompt(
        self,
        prompt: str,
        *,
        keyframes: Sequence[Image.Image] = (),
        references: Sequence[MiniMaxH3PreparedReference] = (),
    ) -> tuple[ttnn.Tensor, torch.Tensor]:
        """Prompt to `(prompt_embeds [1, seq_len, 5120], text_token_tags [L])`, encoded on device.

        The presentation has no chat template and no special tokens. For `t2va` it is the verbatim
        prompt and every row is text-tagged. For `fl2va` each keyframe contributes
        `"<Picture i>: " + <|vision_start|> + N x <|image_pad|> + <|vision_end|>` ahead of the prompt,
        and the **whole vision block is video-tagged** -- that tag is what the DiT's AdaLN keys off, so
        text-tagging it would mis-modulate every one of those rows with no PCC signal anywhere.

        For `ref2va` the presentation is instead one label per reference, numbered per modality, with a
        vision block for an image and one *timestamped* vision block per merged frame pair for a video;
        an audio reference is a label alone, because a waveform never reaches the conditioner.
        `keyframes` and `references` are mutually exclusive.

        Every call runs the encoder; with the default co-residency the weights are already on
        device, so this costs the ~2.8 s forward, not the 50 GB reload.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if keyframes and references:
            raise ValueError("keyframes (fl2va) and references (ref2va) are different tasks; pass one or neither")

        if references:
            input_ids, tags, type_ids, pixel_values, grid_thw, vision_kinds = self._build_ref2va_presentation(
                prompt, references
            )
        else:
            input_ids, tags, type_ids, pixel_values, grid_thw = self._build_presentation(prompt, keyframes)
            # An fl2va keyframe is an image, and t2va has no vision entries at all.
            vision_kinds = ["image"] * (0 if grid_thw is None else len(grid_thw))
        # The vision path is taken by any request with an image or video reference.
        has_vision = grid_thw is not None
        seq_len = input_ids.shape[1]
        detail = ""
        if references:
            kinds = "+".join(reference.kind for reference in references)
            detail = f" ({int((type_ids > 0).sum())} of them vision, references {kinds})"
        elif keyframes:
            detail = f" ({int(type_ids.sum())} of them vision, {len(keyframes)} keyframe(s))"
        self._log(f"encoding {seq_len} presentation tokens on device" + detail)

        true_seq_len = seq_len
        sp_alignment = self.sp_factor * ttnn.TILE_SIZE
        if self.task == "ref2va" and self.sp_factor > 1:
            target = self._force_prompt_pad
            if target is None:
                target = select_bucket(seq_len, self.presentation_ladder)
            elif target not in self.presentation_ladder:
                raise ValueError(
                    f"forced prompt pad {target} is not in the presentation ladder {self.presentation_ladder}"
                )
            elif seq_len > target:
                raise ValueError(f"forced prompt pad {target} is smaller than the presentation {seq_len}")
            if seq_len < target:
                input_ids = torch.nn.functional.pad(input_ids, (0, target - seq_len))
                type_ids = torch.nn.functional.pad(type_ids, (0, target - seq_len))
                seq_len = target
        elif self.sp_factor > 1 and seq_len % sp_alignment:
            seq_len = ((seq_len + sp_alignment - 1) // sp_alignment) * sp_alignment
            input_ids = torch.nn.functional.pad(input_ids, (0, seq_len - true_seq_len))
            type_ids = torch.nn.functional.pad(type_ids, (0, seq_len - true_seq_len))

        encoder = self._prepare_text_encoder()

        # The vision tower, and the two ways its output enters the decoder. Run before the rope tables
        # so a tower failure surfaces before any decoder work.
        vision_kwargs = {}
        if has_vision:
            tower = self._prepare_vision_tower()
            vis_cos, vis_sin = tower.prepare_rope(grid_thw)
            p_patches, p_pos, (p_cos, p_sin), p_cu, logical = pad_patches_for_sp(
                pixel_values.float(),
                tower.prepare_pos_embeds(grid_thw),
                (vis_cos, vis_sin),
                vision_cu_seqlens(grid_thw),
                sp_factor=self.sp_factor,
            )
            path = "ring" if p_cu is None or len(p_cu) <= 2 else "windowed"
            # self._host_log(f"vision tower {path} attention, {p_patches.shape[0]} padded patches")
            sp_kw = {"mesh_axis": self.sp_axis, "shard_dim": 0} if self.sp_factor > 1 else {}
            merged, deepstack = tower.forward(
                bf16_tensor(p_patches, device=self.mesh_device, **sp_kw),
                pos_embeds=bf16_tensor(p_pos, device=self.mesh_device, **sp_kw),
                rope=(
                    bf16_tensor(p_cos, device=self.mesh_device, **sp_kw),
                    bf16_tensor(p_sin, device=self.mesh_device, **sp_kw),
                ),
                cu_seqlens=p_cu,
                logical_patches=logical,
            )
            # Both pad ids, in sequence order. `_scatter_rows` consumes the tower's rows in run
            # order and the patches were concatenated in presentation order, so the two match.
            pad_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in ("<|image_pad|>", "<|video_pad|>")]
            runs = vision_token_runs(input_ids, pad_ids)
            # One run per image and one per merged frame pair of a video, i.e. one per grid
            # entry once `t` is expanded. A mismatch scatters one reference's tokens into
            # another's rows.
            expected_runs = int(sum(int(grid[0]) for grid in grid_thw))
            assert (
                len(runs) == expected_runs
            ), f"expected {expected_runs} vision run(s) in the presentation, found {len(runs)}"
            covered = sum(length for _, length in runs)
            merged_rows = merged.shape[-2]
            assert covered == merged_rows, f"vision runs cover {covered} rows but the tower emitted {merged_rows}"
            # merged tokens REPLACE the `<|image_pad|>` row embeddings; deepstack features are ADDED to
            # those same rows after the first three decoder layers. Not interchangeable.
            vision_kwargs = {"vision_embeds": merged, "vision_runs": runs, "deepstack_embeds": deepstack}

        # With a vision run the three mRoPE axes diverge, so `mrope_interleaved` stops being a no-op
        # and the chunked section split is wrong. t2va keeps the default (shared `arange`) path, where
        # the two layouts are bit-identical -- measured.
        rope_scaling = self._text_config["rope_scaling"]
        position_ids = None
        if has_vision:
            if not rope_scaling.get("mrope_interleaved"):
                raise ValueError("this checkpoint does not declare mrope_interleaved; the vision rope path assumes it")

            # Qwen3-VL walks the sequence per modality run and pulls from the matching grid
            # iterator, so the two go in separately, each in the order its own runs appear.
            def grids_of(kind: str):
                selected = [grid for grid, entry in zip(grid_thw, vision_kinds) if entry == kind]
                return torch.stack(selected) if selected else None

            position_ids = mrope_position_ids(
                type_ids,
                image_grid_thw=grids_of("image"),
                video_grid_thw=grids_of("video"),
                spatial_merge_size=self._vision_config["spatial_merge_size"],
            )
        cos, sin = create_rope_tensors(
            1,
            seq_len,
            None,
            self._text_config["head_dim"],
            rope_scaling.get("rope_theta", self._text_config["rope_theta"]),
            rope_scaling["mrope_section"],
            position_ids=position_ids,
            interleaved=has_vision,
        )
        tt_ids = ttnn.from_torch(
            input_ids,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
        )
        # Causal, and a single un-padded presentation, so no mask is needed.
        taps = encoder.forward(
            tt_ids,
            attention_mask=None,
            pos_embeds=(bf16_tensor(cos, device=self.mesh_device), bf16_tensor(sin, device=self.mesh_device)),
            **vision_kwargs,
        )
        return taps[0], tags

    def _prepare_host_text_encoder(self):
        """The released Qwen3-VL conditioner on the host (CPU), loaded once, for `encode_prompt_host`."""
        if self._host_text_encoder is None:
            from transformers import Qwen3VLForConditionalGeneration

            self._host_log("building the Qwen3-VL conditioner on the HOST (reference encode)")
            hf = Qwen3VLForConditionalGeneration.from_pretrained(
                str(self.weights_dir / "text_encoder"), dtype=torch.bfloat16
            )
            self._host_text_encoder = hf.model.eval()
        return self._host_text_encoder

    def _split_host_vision_inputs(self, pixel_values, grid_thw, vision_kinds, dtype):
        """Split the presentation's concatenated vision patches into the conditioner's per-modality
        image/video inputs, reusing the device's pixels so only the encoder differs."""
        if grid_thw is None:
            return {}
        counts = [int(grid.prod()) for grid in grid_thw]
        assert (
            sum(counts) == pixel_values.shape[0]
        ), f"{sum(counts)} patches expected from the grids, presentation carries {pixel_values.shape[0]}"
        chunks, cursor = [], 0
        for count in counts:
            chunks.append(pixel_values[cursor : cursor + count])
            cursor += count

        def gather(kind: str):
            selected = [(chunks[i], grid_thw[i]) for i, entry in enumerate(vision_kinds) if entry == kind]
            if not selected:
                return None, None
            return torch.cat([p for p, _ in selected]), torch.stack([g for _, g in selected])

        vision_kwargs = {}
        image_pixels, image_grids = gather("image")
        if image_pixels is not None:
            vision_kwargs["pixel_values"] = image_pixels.to(dtype)
            vision_kwargs["image_grid_thw"] = image_grids
        video_pixels, video_grids = gather("video")
        if video_pixels is not None:
            vision_kwargs["pixel_values_videos"] = video_pixels.to(dtype)
            vision_kwargs["video_grid_thw"] = video_grids
        return vision_kwargs

    def encode_prompt_host(
        self,
        prompt: str,
        *,
        keyframes: Sequence[Image.Image] = (),
        references: Sequence[MiniMaxH3PreparedReference] = (),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Host/reference twin of `encode_prompt`, for isolating device-encode issues.

        Builds the identical presentation, then runs the released conditioner on the CPU (slow).
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if keyframes and references:
            raise ValueError("keyframes (fl2va) and references (ref2va) are different tasks; pass one or neither")

        if references:
            input_ids, tags, type_ids, pixel_values, grid_thw, vision_kinds = self._build_ref2va_presentation(
                prompt, references
            )
        else:
            input_ids, tags, type_ids, pixel_values, grid_thw = self._build_presentation(prompt, keyframes)
            vision_kinds = ["image"] * (0 if grid_thw is None else len(grid_thw))

        seq_len = input_ids.shape[1]
        self._log(f"encoding {seq_len} presentation tokens on HOST (reference conditioner)")

        encoder = self._prepare_host_text_encoder()
        vision_kwargs = self._split_host_vision_inputs(pixel_values, grid_thw, vision_kinds, encoder.dtype)

        with torch.no_grad():
            outputs = encoder(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                mm_token_type_ids=type_ids,
                use_cache=False,
                output_hidden_states=True,
                **vision_kwargs,
            )
        embeds = outputs.hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER].float()
        return embeds, tags

    # ------------------------------------------------------------------ denoiser

    def _dit_weight_mode(self) -> str:
        return "resident_adaln_fsdp" if self.dit_fsdp else "resident_adaln"

    def _build_transformer(self) -> MiniMaxH3Transformer3DModel:
        config = {k: v for k, v in self.transformer_config.items() if k not in ("rope_freq_dim", "rope_theta")}
        config["patch_size"] = tuple(config["patch_size"])
        weight_mode = self._dit_weight_mode()
        self._host_log(
            f"building the {config['num_layers']}-layer transformer from {self.transformer_subfolder}/, "
            f"TP={self.tp_factor}/SP={self.sp_factor} ({weight_mode})"
        )
        return MiniMaxH3Transformer3DModel(
            **config,
            mesh_device=self.mesh_device,
            ccl_manager=self.ccl_manager,
            parallel_config=self.dit_parallel_config,
            is_fsdp=self.dit_fsdp,
        )

    def _prepare_transformer(self) -> MiniMaxH3Transformer3DModel:
        cache.load_model(
            self._transformer,
            model_name=MODEL_NAME,
            subfolder=f"{self.transformer_subfolder}_{self._dit_weight_mode()}",
            parallel_config=self.dit_parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            mesh_device=self.mesh_device,
            get_torch_state_dict=lambda: self._read_safetensors(self.transformer_subfolder),
        )
        return self._transformer

    @property
    def patch_size(self) -> tuple[int, int, int]:
        return tuple(self.transformer_config["patch_size"])

    def _device_metadata(self, layout: MiniMaxH3PackedSequence, padded_len: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Rotary tables for the padded global sequence, sharded on SP the way the model fractures it.

        Pad rows are excluded from attention by ring attention's `logical_n`, so their rotary values
        are arbitrary --- but they must exist, hence the zero tail rather than a shorter table.
        """
        pad = padded_len - layout.sequence_length
        position_ids = layout.position_ids
        if pad:
            position_ids = torch.cat([position_ids, torch.zeros(pad, 3, dtype=position_ids.dtype)])
        cos, sin = build_rope_tables(position_ids, rope_freq_dim=self.rope_freq_dim, rope_theta=self.rope_theta)
        cos, sin = prepare_rope_tables(cos, sin, self.transformer_config["attention_head_dim"])
        rotary_dim = cos.shape[-1]

        def seq_sharded(t):
            # tt_dit's own from_torch, not ttnn's: `mesh_axes` is the wrapper's mesh-distribution
            # spec, and the row axis is fractured on SP exactly as the model fractures the sequence.
            return from_torch(
                t.reshape(1, 1, padded_len, rotary_dim),
                device=self.mesh_device,
                dtype=ttnn.float32,
                mesh_axes=[..., self.sp_axis, None],
            )

        return seq_sharded(cos), seq_sharded(sin)

    def _row_indices(self, values: torch.Tensor, padded_len: int) -> ttnn.Tensor:
        """An integer per-row index tensor, ROW_MAJOR and sharded on SP along the row axis."""
        pad = padded_len - values.shape[0]
        if pad:
            values = torch.cat([values, torch.zeros(pad, dtype=values.dtype)])
        return from_torch(
            values.to(torch.int32).reshape(1, 1, 1, padded_len),
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.Layout.ROW_MAJOR,
            mesh_axes=[..., None, self.sp_axis],
        )

    def _logical_length(self, value: int) -> ttnn.Tensor:
        """The logical packed length as a replicated [1, 1, 1, 1] uint32 tensor."""
        return from_torch(
            torch.tensor([value], dtype=torch.int64).reshape(1, 1, 1, 1),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.Layout.ROW_MAJOR,
            mesh_axes=[..., None, None],
        )

    def _replicated_indices(self, values: torch.Tensor) -> ttnn.Tensor:
        """An integer index tensor, ROW_MAJOR and replicated -- `_row_indices` without the shard."""
        return from_torch(
            values.to(torch.int32).reshape(1, 1, 1, -1),
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.Layout.ROW_MAJOR,
            mesh_axes=[..., None, None],
        )

    def _select_bucket(self, seq_len: int) -> int:
        """The rung this request pads to: `warmup`'s forced rung, else the smallest that fits."""
        if self._force_bucket is not None:
            rung = self._force_bucket
            if rung not in self.bucket_ladder:
                raise ValueError(f"forced bucket {rung} is not in the ladder {self.bucket_ladder}")
            if seq_len > rung:
                raise ValueError(f"forced bucket {rung} is smaller than the packed length {seq_len}")
            return rung
        return select_bucket(seq_len, self.bucket_ladder)

    @staticmethod
    def _pad_host_rows(rows: torch.Tensor, capacity: int) -> torch.Tensor:
        """`[n, C] -> [capacity, C]`, zero tail. The caps were checked before any upload."""
        if rows.shape[0] == capacity:
            return rows
        return torch.cat([rows, torch.zeros(capacity - rows.shape[0], rows.shape[-1], dtype=rows.dtype)])

    def _assembly_source_offsets(self, caps: MiniMaxH3ArenaCaps) -> dict[str, int]:
        """Row offset of each stream segment in the transformer's source table; must mirror `forward`."""
        offsets = {"text": 0, "condition_video": caps.prompt}
        cursor = caps.prompt + caps.condition_video_rows
        if self.task == "ref2va":
            offsets["condition_audio"] = cursor
            cursor += caps.condition_audio_rows
        offsets["audio"] = cursor
        offsets["video"] = cursor + caps.audio_rows
        return offsets

    def _assembly_indices(
        self,
        condition_spec: Sequence[tuple[str, int]],
        caps: MiniMaxH3ArenaCaps,
        l_len: int,
        a_len: int,
        v_len: int,
        rung: int,
    ) -> ttnn.Tensor:
        """Source-table row of each packed row: `[text | condition blocks | audio | video | pad]`.

        Pad rows point at source row 0 so the gathered content is finite.
        """
        src = self._assembly_source_offsets(caps)
        indices = torch.zeros(rung, dtype=torch.int32)
        indices[:l_len] = torch.arange(l_len)
        pos = l_len
        cursors = {"video": 0, "audio": 0}
        for modality, rows in condition_spec:
            base = src[f"condition_{modality}"] + cursors[modality]
            cursors[modality] += rows
            indices[pos : pos + rows] = torch.arange(base, base + rows)
            pos += rows
        indices[pos : pos + a_len] = torch.arange(src["audio"], src["audio"] + a_len)
        pos += a_len
        indices[pos : pos + v_len] = torch.arange(src["video"], src["video"] + v_len)
        return self._replicated_indices(indices)

    def _output_indices(self, start: int, count: int, capacity: int) -> ttnn.Tensor:
        """Padded-global-sequence row of each target row of one modality, at the arena capacity."""
        indices = torch.full((capacity,), start, dtype=torch.int32)
        indices[:count] = torch.arange(start, start + count)
        return self._replicated_indices(indices)

    def _prompt_windows(self, l_len: int, cap: int) -> ttnn.Tensor | None:
        """Window boundaries `[0, l_len, cap]` for the token refiner's windowed SDPA, so no real token
        attends to a pad row. None when the prompt fills the capacity exactly.
        """
        if l_len >= cap:
            return None
        return from_torch(
            torch.tensor([0, l_len, cap], dtype=torch.int32),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.Layout.ROW_MAJOR,
            mesh_axes=[None],
        )

    # ------------------------------------------------------------------ decode

    def _cache_submodel(self, module, subfolder: str, state: dict[str, torch.Tensor]) -> None:
        """Load one VAE sub-model through the cache, keyed by its shape and conv3d blocking.

        Each `(T, H, W)` sub-model holds its own shape-specialised conv3d weight layout, so the shape
        is part of the key; `conv3d_blocking_hash` covers the `C_in_block` and depth-to-space stride
        that `prepare_conv3d_weights` bakes into the cached bytes, exactly as `vae_wan2_1` does. The
        VAE is data-parallel with replicated weights, so its parallel config is the DP one.
        """
        blocking = conv3d_blocking_hash(module)
        cache.load_model(
            module,
            model_name=MODEL_NAME,
            subfolder=f"{subfolder}_{blocking}" if blocking else subfolder,
            parallel_config=self.vae_parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            mesh_device=self.mesh_device,
            dtype="fp32",  # the whole VAE checkpoint is fp32 and the port keeps it there
            get_torch_state_dict=lambda: state,
        )

    @property
    def vae(self) -> MiniMaxH3Vae:
        """The video VAE orchestrator, constructed at init; sub-models load on first use."""
        return self._vae

    def _prepare_audio_encoder(self) -> MiniMaxH3AudioEncoder:
        """The audio VAE's encoder half, for `ref2va` reference soundtracks.

        Only `ref2va` needs it -- `t2va` and `fl2va` never encode audio -- so it is built lazily and
        loaded through the same cache path as the decoder. The state filter is the encoder's four
        prefixes, which is what keeps the load *strict*: `convert_minimax_h3_audio_state_dict` returns
        both halves, and a non-strict load would hide a renamed key.
        """
        if self._audio_encoder is None:
            config = self.audio_config
            t_factor = tuple(self.mesh_device.shape)[1] if self.audio_t_shard else 1
            audio_parallel = ParallelFactor(factor=t_factor, mesh_axis=1) if t_factor > 1 else None
            self._host_log(f"building the audio encoder (ref2va reference soundtracks, t_factor={t_factor})")
            encoder = MiniMaxH3AudioEncoder(
                encoder_dim=config["encoder_dim"],
                encoder_rates=tuple(config["encoder_rates"]),
                latent_dim=config["latent_dim"],
                latent_channels=config["latent_channels"],
                num_attention_heads=config["num_attention_heads"],
                mesh_device=self.mesh_device,
                stereo_split_axis=0,
                parallel_config=audio_parallel,
                ccl_manager=self.audio_ccl_manager if audio_parallel is not None else None,
                split_mode="weight",
            )

            def read_state() -> dict[str, torch.Tensor]:
                converted = convert_minimax_h3_audio_state_dict(self._read_safetensors("audio_vae"))
                return {
                    k: v
                    for k, v in converted.items()
                    if k.startswith(("encoder.", "pre_block.", "mean_proj.", "logs_proj."))
                }

            cache.load_model(
                encoder,
                model_name=MODEL_NAME,
                # The audio precision levers change the module's parameter set, so they are part of
                # the cache key -- read off the module so the key cannot drift from what was built.
                subfolder="audio_encoder" + weights_variant(encoder.split_mode, encoder.max_c_in_block),
                parallel_config=self.vae_parallel_config,
                mesh_shape=tuple(self.mesh_device.shape),
                mesh_device=self.mesh_device,
                dtype="fp32",
                get_torch_state_dict=read_state,
            )
            self._audio_encoder = encoder
        return self._audio_encoder

    def _encode_keyframes(self, vae: MiniMaxH3Vae, keyframes: Sequence[Image.Image]) -> torch.Tensor:
        """Prepared keyframes to packed conditioning rows, via the device VAE encoder.

        `encode_keyframes` takes the encoder as an injected callable, so the device VAE plugs straight
        in. `encode_clip` is the keyframe entry point (T=1, taps=1).
        """
        return encode_keyframes(
            keyframes,
            vae.encode_clip,
            self.vae_config.latents_mean,
            self.vae_config.latents_std,
            self.patch_size,
            raw_pixels=True,
        )

    def _prepare_host_vae(self, *, want_encoder: bool = False, want_decoder: bool = False):
        """The released MiniMax-H3 video VAE on the host (CPU), built once and loaded per half on
        demand, for `_encode_keyframes_host` / `_decode_video_host`."""
        if self._host_vae is None:
            from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3

            self._host_log("building the video VAE on the HOST (reference encode/decode)")
            self._host_vae = AutoencoderKLMiniMaxH3(**self._read_config("vae")).eval()
        state = None
        if want_encoder and not self._host_vae_encoder_loaded:
            state = self._read_safetensors("vae")
            self._load_host_vae_half(state, ("encoder.", "quant_conv."))
            self._host_vae_encoder_loaded = True
        if want_decoder and not self._host_vae_decoder_loaded:
            state = state if state is not None else self._read_safetensors("vae")
            self._load_host_vae_half(state, ("decoder.", "post_quant_conv."))
            self._host_vae_decoder_loaded = True
        return self._host_vae

    def _load_host_vae_half(self, state: dict[str, torch.Tensor], prefixes: tuple[str, ...]) -> None:
        """Copy one half of the checkpoint into the host VAE; `unexpected` keys fail, `missing` ones are
        expected."""
        half = {k: v for k, v in state.items() if k.startswith(prefixes)}
        _, unexpected = self._host_vae.load_state_dict(half, strict=False)
        assert not unexpected, f"unexpected {prefixes[0]} keys loading the host VAE: {sorted(unexpected)[:5]}"

    def _encode_keyframes_host(self, vae: MiniMaxH3Vae, keyframes: Sequence[Image.Image]) -> torch.Tensor:
        """Host/reference twin of `_encode_keyframes`, for isolating device keyframe-encode issues.

        Same conditioning math, encoded on the CPU; `vae` is unused (signature parity).
        """
        reference = self._prepare_host_vae(want_encoder=True)
        self._log(f"encoding {len(keyframes)} keyframe(s) on HOST (reference VAE encoder)")
        with torch.no_grad():
            return encode_keyframes(
                keyframes,
                reference._encode_clip,
                self.vae_config.latents_mean,
                self.vae_config.latents_std,
                self.patch_size,
            )

    def _prepare_audio_decoder(self) -> MiniMaxH3AudioDecoder:
        if self._audio_decoder is None:
            config = self.audio_config
            _shard_desc = (
                f"factor {self.audio_t_factor} over mesh axis {self._audio_t_axis}"
                if self.audio_t_factor > 1
                else "unsharded (factor 1)"
            )
            _src = f" (from {_AUDIO_T_FACTOR_ENV})" if self._audio_t_factor_from_env else ""
            self._host_log(f"building the audio decoder ({_shard_desc}{_src})")
            audio_parallel_config = (
                ParallelFactor(factor=self.audio_t_factor, mesh_axis=self._audio_t_axis)
                if self.audio_t_factor > 1
                else None
            )
            audio_ccl = self.audio_ccl_manager if audio_parallel_config is not None else None
            batch_shard_axis = None
            if audio_parallel_config is not None and not ttnn.using_distributed_env():
                other = 1 - self._audio_t_axis
                if tuple(self.mesh_device.shape)[other] >= 2:
                    batch_shard_axis = other
                else:
                    logger.warning(f"audio batch shard skipped: mesh axis {other} has one device")
            decoder = MiniMaxH3AudioDecoder(
                latent_channels=config["latent_channels"],
                latent_dim=config["latent_dim"],
                decoder_dim=config["decoder_dim"],
                decoder_rates=tuple(config["decoder_rates"]),
                decoder_kernel_sizes=tuple(config["decoder_kernel_sizes"]),
                resblock_kernel_sizes=tuple(config["resblock_kernel_sizes"]),
                resblock_dilation_sizes=tuple(tuple(d) for d in config["resblock_dilation_sizes"]),
                mesh_device=self.mesh_device,
                parallel_config=audio_parallel_config,
                ccl_manager=audio_ccl,
                split_mode=self.audio_split_mode,
                pack_bands=_AUDIO_PACK_BANDS,
                act_mode="fused",
                batch_shard_axis=batch_shard_axis,
            )

            def read_state() -> dict[str, torch.Tensor]:
                """Only the decoder's half of the converted checkpoint.

                `convert_minimax_h3_audio_state_dict` returns both halves (`encoder.*`,
                `pre_block.*`, `mean_proj.*`, `logs_proj.*` belong to the encoder), which is why the
                existing tests load it with `strict=False`. Filtering to the two prefixes this module
                owns keeps the load *strict* -- so a renamed key still fails -- and lets this go
                through the same `cache.load_model` path as everything else.
                """
                converted = convert_minimax_h3_audio_state_dict(self._read_safetensors("audio_vae"))
                return {k: v for k, v in converted.items() if k.startswith(("dec_in_proj.", "decoder."))}

            cache.load_model(
                decoder,
                model_name=MODEL_NAME,
                # The audio precision levers change the module's parameter set, so they are part of
                # the cache key -- read off the module so the key cannot drift from what was built.
                subfolder="audio_decoder"
                + weights_variant(
                    decoder.split_mode,
                    decoder.max_c_in_block,
                    decoder.pack_bands,
                    act_mode=decoder.act_mode,
                    polyphase=decoder.polyphase_ups,
                ),
                parallel_config=self.vae_parallel_config,
                mesh_shape=tuple(self.mesh_device.shape),
                mesh_device=self.mesh_device,
                dtype="fp32",
                get_torch_state_dict=read_state,
            )
            self._audio_decoder = decoder
        return self._audio_decoder

    @property
    def audio_sampling_rate(self) -> int:
        rate = 1
        for r in self.audio_config["decoder_rates"]:
            rate *= r
        # 800 samples per latent at 40 latents/s == 32 kHz. The latents-per-second figure is shared
        # with `audio_latent_num_frames`, so the audio *length* and the audio *sample rate* cannot
        # drift apart into a silent desync.
        return rate * MINIMAX_H3_AUDIO_LATENTS_PER_SECOND

    def _denormalize(self, latents: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
        """Undo a per-channel normalization. The channel axis is 1 and the rest broadcast.

        `ndim` is derived rather than passed: a mismatched value broadcasts silently rather than
        raising.
        """
        shape = (1, -1) + (1,) * (latents.ndim - 2)
        return latents * torch.tensor(std).view(shape) + torch.tensor(mean).view(shape)

    # ------------------------------------------------------------------ the call

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        *,
        image: Image.Image | None = None,
        last_image: Image.Image | None = None,
        references: Sequence[MiniMaxH3Reference] | None = None,
        num_frames: int | None = 124,
        aspect_ratio: tuple[float, float] = (16, 9),
        height: int | None = None,
        width: int | None = None,
        reference_resize_mode: str = "match",
        num_inference_steps: int = 50,
        seed: int = 0,
        on_event: PipelineEventCallback | None = None,
    ) -> MiniMaxH3Output:
        """`image` and/or `last_image` select `fl2va`; `references` selects `ref2va`; neither `t2va`.

        `reference_resize_mode` applies to ref2va image references; only `match` is served.

        Note that `fl2va` at a given seed does **not** reproduce `t2va` at that seed, even with a
        keyframe that contributes nothing: the conditioning noise is the first draw off the request
        generator and shifts the video and audio streams behind it. That is the reference's draw
        order.
        """
        on_event = on_event if on_event is not None else null_callback

        if references is not None:
            if image is not None or last_image is not None:
                raise ValueError("references (ref2va) and image/last_image (fl2va) are different tasks")
            return self._call_ref2va(
                prompt,
                references=references,
                num_frames=num_frames,
                aspect_ratio=aspect_ratio,
                height=height,
                width=width,
                reference_resize_mode=reference_resize_mode,
                num_inference_steps=num_inference_steps,
                seed=seed,
                on_event=on_event,
            )
        if num_frames is None:
            raise ValueError("num_frames may only be left to the references, and only for ref2va")

        # 1. Setup: keyframes, canvas, frame alignment and the derived latent geometry.
        if (height is None) != (width is None):
            raise ValueError("pass both height and width, or neither")

        # EXIF-transpose and RGB before anything else. `prepare_keyframe_image` does neither, and both
        # matter: a phone photo carries its rotation in EXIF and would encode sideways, and a palette or
        # RGBA PNG would reach `normalize_keyframe_pixels`'s channel permute with the wrong channel
        # count. The reference's setup block does both here too.
        keyframe_anchors = tuple(anchor for anchor, k in (("first", image), ("last", last_image)) if k is not None)
        sources = [ImageOps.exif_transpose(k).convert("RGB") for k in (image, last_image) if k is not None]

        if height is None:
            # A keyframe's own dimensions decide the canvas; `aspect_ratio` only applies to t2va.
            height, width = resolve_canvas_size(*(sources[0].size if sources else aspect_ratio))
        ratio = self.vae_config.spatial_compression_ratio
        if height % 32 or width % 32:
            raise ValueError(f"canvas {height}x{width} must be a multiple of 32 on both axes")
        num_frames = align_num_frames(num_frames)
        latent_height, latent_width = height // ratio, width // ratio
        num_audio_latents = audio_latent_num_frames(num_frames)
        num_latent_frames = video_latent_num_frames(num_frames)

        # `stretch` keys on position in the list, not on the anchor name: the FIRST keyframe given is
        # the geometry anchor and is stretched to the canvas, and any later one is cover-cropped to
        # follow it. So a lone `last_image` is stretched. That is the reference's behaviour and it looks
        # like a bug until you see the `last`-only case pass.
        keyframes = [prepare_keyframe_image(k, height, width, stretch=(i == 0)) for i, k in enumerate(sources)]
        flavor = "t2va" if not keyframes else ("fl2va" if image is not None else "fl2va_last_frame")
        self._log(
            f"{flavor} {width}x{height}, {num_frames} frames ({num_frames / MINIMAX_H3_FPS:.2f} s), "
            f"{num_latent_frames} latent frames, {num_audio_latents} audio latents, "
            f"{num_inference_steps} steps, anchors={keyframe_anchors or '()'}"
        )

        # 2. Text (plus the vision block, for fl2va).
        with event_section(on_event, "encoder"):
            prompt_embeds, text_token_tags = self.encode_prompt(prompt, keyframes=keyframes)

        # Both schedules. Built here rather than after the layout because the keyframe step below needs
        # `scale_noise`, which takes its `t` at face value and works before `set_timesteps` -- but they
        # are set up fully so there is only one place that decides the schedule.
        scheduler = MiniMaxH3Scheduler(shift=VIDEO_SHIFT)
        audio_scheduler = MiniMaxH3Scheduler(shift=AUDIO_SHIFT)
        scheduler.set_timesteps(num_inference_steps)
        audio_scheduler.set_timesteps(num_inference_steps)

        # All noise for the request, off one generator, in the reference's draw order: conditioning
        # first, then video, then audio. The reference spreads these across two blocks -- the keyframe
        # VAE encoder draws the conditioning noise and `prepare_latents` draws the rest -- but the
        # observable contract is the order of the draws off one generator, and keeping them in one
        # function is what makes that order testable without a mesh.
        generator = torch.Generator().manual_seed(seed)
        condition_noise, video_rows, audio_rows = draw_request_latents(
            generator,
            condition_latent_shapes=((1, latent_height, latent_width),) * len(keyframes),
            latent_channels=self.vae_config.latent_channels,
            num_latent_frames=num_latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            num_audio_latents=num_audio_latents,
            audio_latent_channels=self.audio_config["latent_channels"],
            patch_size=self.patch_size,
        )

        # 3. Keyframe VAE encode, then noise-augment to t = 0.999. The reference's `vae_encoder` block,
        # in the reference's position: before `prepare_layout`, and before the DiT exists, so the
        # encoder gets its residency window uncontended.
        condition_rows = None
        if keyframes:
            # Every keyframe tile is exactly `tile_size` square: `split_tiles` returns `[tile_size] * n`
            # lengths unless one tile already covers the axis, and at 1344x768 neither does. So one
            # `(1, 256, 256)` encoder serves all 28 tiles, which is one wave on a 32-device mesh.
            with event_section(on_event, "vae_encode"):
                condition_rows = self._encode_keyframes(self._vae, keyframes)
                condition_rows = scheduler.scale_noise(condition_rows, MINIMAX_H3_KEYFRAME_NOISE_AUG, condition_noise)

        # 4. Layout. One conditioning block of `rows_per_frame` rows per anchor, between text and audio.
        layout = build_packed_sequence(
            text_token_tags,
            num_latent_frames,
            latent_height,
            latent_width,
            num_audio_latents,
            self.patch_size,
            keyframe_anchors,
        )

        # 5. Prepend the anchors, as the reference's `prepare_latents` does.
        if condition_rows is not None:
            assert condition_rows.shape[0] == layout.num_condition_video_rows, (
                f"keyframe encode produced {condition_rows.shape[0]} conditioning rows but the layout "
                f"expects {layout.num_condition_video_rows}"
            )
            video_rows = torch.cat([condition_rows, video_rows])

        # 6-7. Denoise and decode, shared with `ref2va`.
        return self._denoise_and_decode(
            layout=layout,
            prompt_embeds=prompt_embeds,
            video_rows=video_rows,
            audio_rows=audio_rows,
            scheduler=scheduler,
            audio_scheduler=audio_scheduler,
            num_inference_steps=num_inference_steps,
            num_latent_frames=num_latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            num_audio_latents=num_audio_latents,
            on_event=on_event,
        )

    @torch.no_grad()
    def _call_ref2va(
        self,
        prompt: str,
        *,
        references: Sequence[MiniMaxH3Reference],
        num_frames: int | None,
        aspect_ratio: tuple[float, float],
        height: int | None,
        width: int | None,
        reference_resize_mode: str,
        num_inference_steps: int,
        seed: int,
        on_event: PipelineEventCallback,
    ) -> MiniMaxH3Output:
        """`ref2va`: an ordered list of references in, a video and its soundtrack out.

        The order of operations is the reference's and is not interchangeable. A reference's latent
        geometry is only known **after** it is encoded, and the packed layout is built from that
        geometry, so the VAE encode has to run before the layout -- unlike `fl2va`, where a keyframe's
        geometry is the target's by construction. The conditioning noise is then drawn at those
        resolved shapes, and it is the *first* draw off the request generator, ahead of the video and
        audio noise.
        """
        if reference_resize_mode != MINIMAX_H3_SERVED_REFERENCE_RESIZE_MODE:
            raise ValueError(
                f"served ref2va resize mode is {MINIMAX_H3_SERVED_REFERENCE_RESIZE_MODE!r}, "
                f"got {reference_resize_mode!r}"
            )
        # 1. Setup. The canvas comes from the request, never from a reference: references do not bind
        # the generated geometry, which is the property that makes them cost extra rows rather than
        # change the output shape.
        if (height is None) != (width is None):
            raise ValueError("pass both height and width, or neither")
        if height is None:
            height, width = resolve_canvas_size(*aspect_ratio)
        ratio = self.vae_config.spatial_compression_ratio
        if height % 32 or width % 32:
            raise ValueError(f"canvas {height}x{width} must be a multiple of 32 on both axes")

        prepared, num_frames = prepare_references(
            references,
            num_frames,
            self.audio_sampling_rate,
            reference_resize_mode=reference_resize_mode,
            target_height=height,
            target_width=width,
        )
        latent_height, latent_width = height // ratio, width // ratio
        num_audio_latents = audio_latent_num_frames(num_frames)
        num_latent_frames = video_latent_num_frames(num_frames)
        kinds = "+".join(reference.kind + ("(+audio)" if reference.has_audio else "") for reference in prepared)
        self._log(
            f"ref2va {width}x{height}, {num_frames} frames ({num_frames / MINIMAX_H3_FPS:.2f} s), "
            f"{num_latent_frames} latent frames, {num_audio_latents} audio latents, "
            f"{num_inference_steps} steps, reference_resize_mode={reference_resize_mode}, "
            f"references=[{kinds}]"
        )

        # 2. Text, plus one vision block per image reference and one per merged frame pair of a video.
        with event_section(on_event, "encoder"):
            prompt_embeds, text_token_tags = self.encode_prompt(prompt, references=prepared)

        scheduler = MiniMaxH3Scheduler(shift=VIDEO_SHIFT)
        audio_scheduler = MiniMaxH3Scheduler(shift=AUDIO_SHIFT)
        scheduler.set_timesteps(num_inference_steps)
        audio_scheduler.set_timesteps(num_inference_steps)

        # 3. Reference VAE encode.
        has_visual = any(reference.kind != "audio" for reference in prepared)
        has_video = any(reference.kind == "video" for reference in prepared)
        has_audio = any(reference.has_audio for reference in prepared)
        vae = self._vae if has_visual else None
        audio_encoder = self._prepare_audio_encoder() if has_audio else None

        with event_section(on_event, "vae_encode"):
            condition_rows, audio_condition_rows = encode_references(
                prepared,
                encode_clip=(lambda pixels: vae.encode_clip(pixels)) if has_visual else None,
                encode_video=(lambda pixels: vae.encode(pixels)) if has_video else None,
                encode_audio=(lambda waveform: audio_encoder(waveform)[0]) if has_audio else None,
                latents_mean=self.vae_config.latents_mean,
                latents_std=self.vae_config.latents_std,
                audio_latents_mean=self.audio_config["latents_mean"],
                audio_latents_std=self.audio_config["latents_std"],
                patch_size=self.patch_size,
                audio_latent_channels=self.audio_config["latent_channels"],
                raw_pixels=True,
            )

        # 4. All the noise for the request, off one generator, in the reference's draw order:
        # conditioning first (one draw per VISUAL reference, at its own resolved shape), then video,
        # then audio. Drawn after the encode because only the encode knows those shapes.
        generator = torch.Generator().manual_seed(seed)
        condition_noise, video_rows, audio_rows = draw_request_latents(
            generator,
            condition_latent_shapes=reference_condition_shapes(prepared),
            latent_channels=self.vae_config.latent_channels,
            num_latent_frames=num_latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            num_audio_latents=num_audio_latents,
            audio_latent_channels=self.audio_config["latent_channels"],
            patch_size=self.patch_size,
        )

        # 5. Noise-augment the VISUAL condition rows to t = 0.999. The audio rows are left clean and
        # run at a literal t = 1.0 for every step -- see `references.py`.
        if condition_rows is not None:
            condition_rows = scheduler.scale_noise(condition_rows, MINIMAX_H3_KEYFRAME_NOISE_AUG, condition_noise)

        # 6. Layout, from the geometry the encode resolved.
        layout = build_ref2va_packed_sequence(
            text_token_tags,
            prepared,
            num_latent_frames,
            latent_height,
            latent_width,
            num_audio_latents,
            self.patch_size,
        )
        if condition_rows is not None:
            assert condition_rows.shape[0] == layout.num_condition_video_rows, (
                f"reference encode produced {condition_rows.shape[0]} video condition rows but the layout "
                f"expects {layout.num_condition_video_rows}"
            )
            video_rows = torch.cat([condition_rows, video_rows])
        if audio_condition_rows is not None:
            assert audio_condition_rows.shape[0] == layout.num_condition_audio_rows, (
                f"reference encode produced {audio_condition_rows.shape[0]} audio condition rows but the "
                f"layout expects {layout.num_condition_audio_rows}"
            )
            audio_rows = torch.cat([audio_condition_rows, audio_rows])

        # The typed conditioning region, in packed order. Derived from the same reference walk as the
        # layout, so the two cannot disagree about where a block starts.
        condition_spec = [
            (modality, block.shape[0])
            for block, modality in split_condition_blocks(prepared, condition_rows, audio_condition_rows)
        ]

        return self._denoise_and_decode(
            layout=layout,
            prompt_embeds=prompt_embeds,
            video_rows=video_rows,
            audio_rows=audio_rows,
            scheduler=scheduler,
            audio_scheduler=audio_scheduler,
            num_inference_steps=num_inference_steps,
            num_latent_frames=num_latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            num_audio_latents=num_audio_latents,
            on_event=on_event,
            condition_spec=condition_spec,
        )

    def _denoise_and_decode(
        self,
        *,
        layout: MiniMaxH3PackedSequence,
        prompt_embeds: ttnn.Tensor,
        video_rows: torch.Tensor,
        audio_rows: torch.Tensor,
        scheduler: MiniMaxH3Scheduler,
        audio_scheduler: MiniMaxH3Scheduler,
        num_inference_steps: int,
        num_latent_frames: int,
        latent_height: int,
        latent_width: int,
        num_audio_latents: int,
        on_event: PipelineEventCallback,
        condition_spec: Sequence[tuple[str, int]] | None = None,
    ) -> MiniMaxH3Output:
        """The half every task shares: denoise the packed sequence, then decode both modalities.

        `condition_spec` is the only thing the tasks differ by here, and only `ref2va` passes one.
        """
        transformer = self._prepare_transformer()
        with event_section(on_event, "denoising"):
            video_rows, audio_rows = self._denoise(
                transformer,
                layout,
                prompt_embeds,
                video_rows,
                audio_rows,
                scheduler,
                audio_scheduler,
                condition_spec=condition_spec,
                on_event=on_event,
            )

        with event_section(on_event, "vae"):
            video = self._decode_video(
                self._vae, video_rows, num_latent_frames, latent_height, latent_width, layout.num_condition_video_rows
            )

        with event_section(on_event, "audio"):
            audio = self._decode_audio(
                self._audio_decoder, audio_rows, num_audio_latents, layout.num_condition_audio_rows
            )

        yuv = self.vae_output_type == "yuv420"
        return MiniMaxH3Output(
            video=video,
            audio=audio,
            sampling_rate=self.audio_sampling_rate,
            num_frames=video.shape[0] if yuv else video.shape[2],
            video_format="yuv420" if yuv else "rgb_float",
        )

    @staticmethod
    def _warmup_image(size: int = 512) -> Image.Image:
        y, x = np.mgrid[0:size, 0:size].astype(np.uint8)
        return Image.fromarray(np.stack([x, y, x ^ y], axis=-1), "RGB")

    def _warmup_audio(self, seconds: float = 1.0) -> tuple[torch.Tensor, int]:
        rate = self.audio_sampling_rate
        return torch.zeros(MINIMAX_H3_AUDIO_CHANNELS, int(seconds * rate), dtype=torch.float32), rate

    def _warmup_video(self, num_frames: int, size: int = 256) -> np.ndarray:
        return np.zeros((num_frames, size, size, 3), dtype=np.uint8)

    def _warmup_on_init(self) -> None:
        """Compile and (when tracing) capture every module a served request touches, per task: the
        keyframe encoder for t2va/fl2va, and the image, video and audio encoders for ref2va."""
        height, width = resolve_canvas_size(16, 9)
        num_frames = align_num_frames(round(5 * MINIMAX_H3_FPS))
        if self.task != "ref2va":
            rung_requests = {
                max(self.bucket_ladder): dict(
                    image=self._warmup_image(),
                    last_image=self._warmup_image(),
                    num_frames=num_frames,
                    height=height,
                    width=width,
                )
            }
            self.warmup(
                image=self._warmup_image(),
                num_frames=num_frames,
                height=height,
                width=width,
                num_inference_steps=3,
                rung_requests=rung_requests,
            )
            return

        full_frames = align_num_frames(round(MINIMAX_H3_MAX_DURATION * MINIMAX_H3_FPS))
        mid_frames = align_num_frames(round(10 * MINIMAX_H3_FPS))
        waveform, sample_rate = self._warmup_audio(MINIMAX_H3_MAX_DURATION)
        ladder = sorted(self.bucket_ladder, reverse=True)
        video_audio = MiniMaxH3Reference(
            video=self._warmup_video(full_frames),
            fps=float(MINIMAX_H3_FPS),
            audio=waveform,
            sample_rate=sample_rate,
        )
        rung_requests = {
            ladder[0]: dict(
                references=[MiniMaxH3Reference(image=self._warmup_image()), video_audio],
                num_frames=full_frames,
                height=height,
                width=width,
            )
        }
        if len(ladder) > 1:
            images = [MiniMaxH3Reference(image=self._warmup_image()) for _ in range(MINIMAX_H3_MAX_REFERENCE_IMAGES)]
            rung_requests[ladder[1]] = dict(
                references=images,
                num_frames=num_frames,
                height=height,
                width=width,
            )
        if len(ladder) > 2:
            rung_requests[ladder[2]] = dict(
                references=[MiniMaxH3Reference(video=self._warmup_video(mid_frames), fps=float(MINIMAX_H3_FPS))],
                num_frames=mid_frames,
                height=height,
                width=width,
            )
        self.warmup(
            references=[MiniMaxH3Reference(image=self._warmup_image())],
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=3,
            rung_requests=rung_requests,
        )

    def warmup(
        self,
        *,
        prompt: str = "warmup",
        image: Image.Image | None = None,
        last_image: Image.Image | None = None,
        references: Sequence[MiniMaxH3Reference] | None = None,
        num_frames: int | None = 124,
        height: int | None = None,
        width: int | None = None,
        aspect_ratio: tuple[float, float] = (16, 9),
        num_inference_steps: int = 50,
        rung_requests: Mapping[int, dict] | None = None,
    ) -> None:
        """
        Buffer allocation and, when tracing, trace capture.
        """
        generation_kwargs = dict(
            image=image,
            last_image=last_image,
            references=references,
            num_frames=num_frames,
            height=height,
            width=width,
            aspect_ratio=aspect_ratio,
        )
        self._log_generation = False
        try:
            self(prompt, num_inference_steps=num_inference_steps, **generation_kwargs)
            if not self.bucket_denoise:
                return
            natural = self.last_seq_len.padded

            if self.task == "ref2va":
                self._warm_ref2va_prompt_encoder_envelope()
            else:
                self._warm_prompt_encoder_envelope()

            overrides = dict(rung_requests or {})
            fitted: dict[int, dict] = {}
            shrunk = generation_kwargs
            host = _is_host_rank()
            bind_rungs = sorted(self.bucket_ladder, reverse=True)
            if host:
                _tqdm_spacer()
            for rung in tqdm.tqdm(
                bind_rungs,
                desc=f"Initializing bucket buffers: {','.join(map(str, bind_rungs))}",
                disable=not host,
                file=sys.stderr,
                bar_format=_TQDM_BAR_FORMAT,
            ):
                bucket = self._buckets.get(rung)
                shrink = rung < natural and rung not in overrides
                request = overrides.get(rung, shrunk if shrink else generation_kwargs)
                if bucket is None or not bucket.warm:
                    request = self._run_forced_fit(rung, prompt, request, shrink=shrink)
                    if request is None:
                        continue
                    if shrink:
                        shrunk = request
                fitted[rung] = request
            if not self.trace_denoise:
                return
            capture_rungs = sorted(fitted, reverse=True)
            if host:
                _tqdm_spacer()
            for rung in tqdm.tqdm(
                capture_rungs,
                desc=f"Capturing bucket traces: {','.join(map(str, capture_rungs))}",
                disable=not host,
                file=sys.stderr,
                bar_format=_TQDM_BAR_FORMAT,
            ):
                if not self._rung_captured(rung):
                    shrink = rung < natural and rung not in overrides
                    self._run_forced_fit(rung, prompt, fitted[rung], shrink=shrink)
        finally:
            self._log_generation = True

    def _run_forced(self, rung: int, prompt: str, generation_kwargs: dict) -> None:
        """One short generation padded to `rung` regardless of its natural rung -- warmup's ladder walk."""
        self._force_bucket = rung
        try:
            self(prompt, num_inference_steps=2, **generation_kwargs)
        finally:
            self._force_bucket = None

    def _run_forced_fit(self, rung: int, prompt: str, generation_kwargs: dict, *, shrink: bool) -> dict | None:
        """`_run_forced`; with `shrink`, halve `num_frames` until the request fits `rung`. Returns the
        kwargs that ran, or None when even the shortest video does not fit.
        """
        kwargs = dict(generation_kwargs)
        while True:
            try:
                self._run_forced(rung, prompt, kwargs)
                return kwargs
            except ValueError as error:
                if not shrink or "smaller than the packed length" not in str(error):
                    raise
                frames = kwargs.get("num_frames") or 124
                if frames <= 5:
                    return None
                kwargs["num_frames"] = max(5, frames // 2)

    def _filler_prompt(self, num_tokens: int) -> str:
        """A prompt of exactly `num_tokens` tokens: only length keys the encoder's programs, so any
        single-token word repeated works."""
        return " village" * num_tokens

    def _warm_prompt_encoder_envelope(self) -> None:
        """Compile every prompt-encoding program a served t2va/fl2va request can reach, strictly
        before trace capture (a program first compiled under live traces can corrupt a replay).
        """
        if self.sp_factor <= 1:
            return

        caps = self.arena_caps
        alignment = self.sp_factor * ttnn.TILE_SIZE

        def align_up(value: int) -> int:
            return ((value + alignment - 1) // alignment) * alignment

        warm_image = Image.new("RGB", (64, 64), (127, 127, 127))
        before = self.mesh_device.num_program_cache_entries()

        for n_keyframes, canvas in served_envelope(self.task):
            if canvas is None:
                keyframes: list[Image.Image] = []
                vision_len = 0
            else:
                height, width = canvas
                keyframes = [
                    prepare_keyframe_image(warm_image, height, width, stretch=(i == 0)) for i in range(n_keyframes)
                ]
                probe = self._filler_prompt(1)
                probe_ids, *_ = self._build_presentation(probe, keyframes)
                probe_tokens = len(self.tokenizer(probe, add_special_tokens=False)["input_ids"])
                vision_len = probe_ids.shape[1] - probe_tokens

            budget = min(MINIMAX_H3_MAX_TEXT_TOKENS, caps.prompt - vision_len)
            buckets = range(align_up(vision_len + 1), align_up(vision_len + budget) + 1, alignment)
            for bucket in buckets:
                prompt = self._filler_prompt(min(bucket - vision_len, budget))
                landed = align_up(vision_len + len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"]))
                assert (
                    landed == bucket
                ), f"filler landed on {landed}, expected bucket {bucket} (vision_len {vision_len})"
                embeds, _ = self.encode_prompt(prompt, keyframes=keyframes)
                ttnn.deallocate(embeds)

        self._host_log(
            f"prompt encoder envelope warmed: +{self.mesh_device.num_program_cache_entries() - before} programs"
        )

    def _warm_ref2va_prompt_encoder_envelope(self) -> None:
        """Compile every prompt-encoding program a served ref2va request can reach, strictly before
        trace capture.
        """
        if self.sp_factor <= 1:
            return

        prompt = self._filler_prompt(1)
        before = self.mesh_device.num_program_cache_entries()

        def gray(size: tuple[int, int]) -> Image.Image:
            height, width = size
            return Image.new("RGB", (width, height), (127, 127, 127))

        def image_ref(size: tuple[int, int]) -> MiniMaxH3PreparedReference:
            return MiniMaxH3PreparedReference(kind="image", image=gray(size))

        def video_ref(num_frames: int, size: tuple[int, int]) -> MiniMaxH3PreparedReference:
            height, width = size
            return MiniMaxH3PreparedReference(
                kind="video", frames=np.full((num_frames, height, width, 3), 127, dtype=np.uint8)
            )

        units: list[tuple[str, list[MiniMaxH3PreparedReference], int | None]] = []

        def add(label: str, references: list[MiniMaxH3PreparedReference], *, pad_to: int | None = None) -> None:
            units.append((label, references, pad_to))

        pad_canvas = min(served_canvases(), key=lambda canvas: canvas[0] * canvas[1])
        pad_size = min(served_reference_image_sizes(*pad_canvas), key=lambda size: size[0] * size[1])
        for rung in self.presentation_ladder:
            add(f"presentation rung {rung}", [image_ref(pad_size)], pad_to=rung)

        for canvas, size in served_envelope(self.task):
            add(f"1 image at {size[1]}x{size[0]} (canvas {canvas[1]}x{canvas[0]})", [image_ref(size)])

        max_canvas = max(served_canvases(), key=lambda canvas: canvas[0] * canvas[1])
        add("2 images at max canvas", [image_ref(max_canvas) for _ in range(2)])
        add(
            "9 images at max canvas",
            [image_ref(max_canvas) for _ in range(MINIMAX_H3_MAX_REFERENCE_IMAGES)],
        )

        short_clip = align_num_frames(1)
        for size in served_reference_video_canvases():
            add(f"1 video at {size[1]}x{size[0]}", [video_ref(short_clip, size)])

        video_canvas = resolve_canvas_size(*MINIMAX_H3_DEFAULT_ASPECT_RATIO)
        for duration_s in (MINIMAX_H3_DURATIONS_S[0], MINIMAX_H3_DURATIONS_S[-1]):
            frames = align_num_frames(round(duration_s * MINIMAX_H3_FPS))
            add(f"1 video {duration_s}s at {video_canvas[1]}x{video_canvas[0]}", [video_ref(frames, video_canvas)])
        clip = align_num_frames(round(5 * MINIMAX_H3_FPS))
        add("3 videos totaling 15s", [video_ref(clip, video_canvas) for _ in range(3)])

        host = _is_host_rank()
        if host:
            _tqdm_spacer()
        for label, references, pad_to in tqdm.tqdm(
            units,
            desc="Warming ref2va prompt encoder",
            disable=not host,
            file=sys.stderr,
            bar_format=_TQDM_BAR_FORMAT,
        ):
            unit_before = self.mesh_device.num_program_cache_entries()
            self._force_prompt_pad = pad_to
            try:
                embeds, _ = self.encode_prompt(prompt, references=references)
            finally:
                self._force_prompt_pad = None
            if pad_to is not None:
                assert (
                    embeds.shape[1] == pad_to
                ), f"forced pad landed on {embeds.shape[1]}, expected presentation rung {pad_to}"
            ttnn.deallocate(embeds)
            # self._host_log(
            #     f"warmed prompt encoder for {label}: "
            #     f"+{self.mesh_device.num_program_cache_entries() - unit_before} programs"
            # )

        self._host_log(
            f"ref2va prompt encoder envelope warmed: "
            f"+{self.mesh_device.num_program_cache_entries() - before} programs"
        )

    def _rung_captured(self, rung: int) -> bool:
        transformer = self._transformer
        if transformer is None:
            return False
        run_blocks = type(transformer).run_blocks
        tracer = run_blocks._tracers_keyed.get(transformer, {}).get(rung)
        return tracer is not None and tracer.trace_captured

    def release_traces(self) -> None:
        decoder = self._audio_decoder
        if decoder is not None:
            decoder.release_trace()
        transformer = self._transformer
        if transformer is None:
            return
        transformer.release_traces()

    def _denoise(
        self,
        transformer: MiniMaxH3Transformer3DModel,
        layout: MiniMaxH3PackedSequence,
        prompt_embeds: ttnn.Tensor,
        video_rows: torch.Tensor,
        audio_rows: torch.Tensor,
        scheduler: MiniMaxH3Scheduler,
        audio_scheduler: MiniMaxH3Scheduler,
        condition_spec: Sequence[tuple[str, int]] | None = None,
        on_event: PipelineEventCallback = null_callback,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Denoise in place. `video_rows` is `[condition rows | target rows]`, cond first, as the
        reference's `latents` is; `num_condition_video_rows` is 0 for `t2va`. `audio_rows` is the same
        shape for `ref2va`, whose reference soundtracks contribute audio condition rows.

        `condition_spec` is the conditioning region as `[(modality, rows), ...]` in **packed order**,
        which is how the model wants it: `ref2va` interleaves the two modalities there, so the row
        counts alone cannot say where each block starts. Left out, it is derived as one `"video"` block
        covering the layout's video condition rows -- exactly `t2va` and `fl2va`.
        """
        num_cond = layout.num_condition_video_rows
        num_cond_audio = layout.num_condition_audio_rows
        if condition_spec is None:
            condition_spec = [("video", num_cond)] if num_cond else []
        spec_video = sum(rows for modality, rows in condition_spec if modality == "video")
        spec_audio = sum(rows for modality, rows in condition_spec if modality == "audio")
        if (spec_video, spec_audio) != (num_cond, num_cond_audio):
            raise ValueError(
                f"condition_spec covers {spec_video} video / {spec_audio} audio rows but the layout has "
                f"{num_cond} / {num_cond_audio}; the block list and the layout disagree"
            )
        # Kept to assert the invariant at the end of the loop. `fl2va` is the first task for which the
        # write mask matters, nothing re-imposes the anchors, and an overwritten anchor still denoises
        # into a plausible video that merely ignores the keyframe -- so no output metric would catch it.
        # `ref2va` needs the same guarantee on the audio rows, which it is the first task to have.
        t_preamble = time.time()
        anchor_rows = video_rows[:num_cond].clone() if num_cond else None
        anchor_audio_rows = audio_rows[:num_cond_audio].clone() if num_cond_audio else None

        caps = self.arena_caps
        l_len = layout.text_indices.shape[0]
        v_target = video_rows.shape[0] - num_cond
        a_target = audio_rows.shape[0] - num_cond_audio
        over = [
            f"{name} {count} > {cap}"
            for name, count, cap in (
                ("prompt tokens", l_len, caps.prompt),
                ("target video rows", v_target, caps.video_rows),
                ("target audio rows", a_target, caps.audio_rows),
                ("condition video rows", num_cond, caps.condition_video_rows),
                ("condition audio rows", num_cond_audio, caps.condition_audio_rows if self.task == "ref2va" else 0),
            )
            if count > cap
        ]
        if over:
            raise ValueError(f"request exceeds the arena caps: {', '.join(over)} (see MiniMaxH3ArenaCaps)")

        alignment = self.sp_factor * ttnn.TILE_SIZE
        if self.bucket_denoise:
            rung = self._select_bucket(layout.sequence_length)
        else:
            rung = ((layout.sequence_length + alignment - 1) // alignment) * alignment
        self.last_seq_len = SeqLen(padded=rung, logical=layout.sequence_length)
        self._log(
            f"packed sequence {layout.sequence_length} -> bucket {rung}, "
            f"{rung // self.sp_factor} rows/device, {num_cond} condition rows"
        )

        timesteps = scheduler.timesteps
        audio_timesteps = audio_scheduler.timesteps

        row_slot, slot_roles = build_slot_routing(layout, roles=self.adaln_slot_roles)

        state = self._buckets.setdefault(rung if self.bucket_denoise else 0, _BucketState())
        traced = self.trace_denoise and state.warm
        if self.trace_denoise and not state.warm:
            self.release_traces()

        t_rope = time.time()
        rope_cos, rope_sin = self._device_metadata(layout, rung)
        state.rope_cos.update(rope_cos, traced=traced)
        state.rope_sin.update(rope_sin, traced=traced)
        t_rope = time.time() - t_rope

        prompt_device = ttnn.reshape(prompt_embeds, (1, 1, prompt_embeds.shape[1], prompt_embeds.shape[2]))

        self._tt_cond_video.update(
            self._pad_host_rows(video_rows[:num_cond], caps.condition_video_rows).reshape(
                1, 1, caps.condition_video_rows, -1
            ),
            traced=traced,
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
        )
        if self.task == "ref2va":
            self._tt_cond_audio.update(
                self._pad_host_rows(audio_rows[:num_cond_audio], caps.condition_audio_rows).reshape(
                    1, 1, caps.condition_audio_rows, -1
                ),
                traced=traced,
                dtype=ttnn.bfloat16,
                device=self.mesh_device,
            )

        transformer.prepare_static_sources(
            prompt_1BLP=prompt_device,
            prompt_windows=self._prompt_windows(l_len, prompt_embeds.shape[1]),
            condition_video_1BKC=self._tt_cond_video.value,
            condition_audio_1BKC=self._tt_cond_audio.value if self.task == "ref2va" else None,
            prompt_cap=caps.prompt,
            traced=traced,
        )

        state.adaln.update(self._row_indices(adaln_indices(layout.token_tags, row_slot), rung), traced=traced)
        state.tsi.update(self._row_indices(row_slot, rung), traced=traced)
        state.assembly_idx.update(
            self._assembly_indices(condition_spec, caps, l_len, a_target, v_target, rung), traced=traced
        )
        self._tt_logical_n.update(self._logical_length(layout.sequence_length), traced=traced)
        audio_start = l_len + num_cond + num_cond_audio
        video_start = audio_start + a_target
        self._tt_video_out_idx.update(self._output_indices(video_start, v_target, caps.video_rows), traced=traced)
        self._tt_audio_out_idx.update(self._output_indices(audio_start, a_target, caps.audio_rows), traced=traced)

        self._tt_video.update(
            self._pad_host_rows(video_rows[num_cond:], caps.video_rows).reshape(1, 1, caps.video_rows, -1),
            traced=traced,
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
        )
        self._tt_audio.update(
            self._pad_host_rows(audio_rows[num_cond_audio:], caps.audio_rows).reshape(1, 1, caps.audio_rows, -1),
            traced=traced,
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
        )

        t_preamble = time.time() - t_preamble
        t_first = t_steady = 0.0
        if _is_host_rank():
            _tqdm_spacer()
        for i, t in enumerate(
            tqdm.tqdm(
                timesteps,
                desc="Denoising",
                disable=(not _is_host_rank()),
                file=sys.stderr,
                bar_format=_TQDM_BAR_FORMAT,
            )
        ):
            t_step = time.time()
            level_kwargs = {
                "video_timestep": float(t),
                "audio_timestep": float(audio_timesteps[i]),
            }
            if "condition_video" in slot_roles:
                level_kwargs["condition_video_timestep"] = max(float(t), MINIMAX_H3_KEYFRAME_NOISE_AUG)
            if "condition_audio" in slot_roles:
                level_kwargs["condition_audio_timestep"] = MINIMAX_H3_AUDIO_CONDITION_TIMESTEP
            levels = slot_levels(slot_roles, **level_kwargs)
            self._tt_timestep.update(
                levels.reshape(1, 1, -1, 1), traced=traced, dtype=ttnn.float32, device=self.mesh_device
            )

            video_velocity, audio_velocity = transformer(
                video_1BVC=self._tt_video.value,
                audio_1BAC=self._tt_audio.value,
                assembly_indices=state.assembly_idx.value,
                video_out_indices=self._tt_video_out_idx.value,
                audio_out_indices=self._tt_audio_out_idx.value,
                timestep=self._tt_timestep.value,
                adaln_indices=state.adaln.value,
                timestep_indices=state.tsi.value,
                rope_cos=state.rope_cos.value,
                rope_sin=state.rope_sin.value,
                logical_n=self._tt_logical_n.value,
                pad_to=rung,
                traced=traced,
            )

            ttnn.synchronize_device(self.mesh_device)
            if ttnn.using_distributed_env():
                ttnn.distributed_context_barrier()
            ttnn.multiply_(video_velocity, float(scheduler.step_coefficient(i)))
            ttnn.add_(self._tt_video.value, video_velocity)
            ttnn.multiply_(audio_velocity, float(audio_scheduler.step_coefficient(i)))
            ttnn.add_(self._tt_audio.value, audio_velocity)
            t_step = time.time() - t_step
            if i == 0:
                t_first = t_step
            else:
                t_steady += t_step
            on_event(DenoiseStep(step=i + 1, total=len(timesteps), sigma=float(t)))

        state.warm = True
        steady_steps = max(len(timesteps) - 1, 1)
        self._log(
            f"denoise breakdown: preamble {t_preamble:.1f}s (rope {t_rope:.1f}s) | "
            f"first step {t_first:.1f}s | steady {t_steady:.1f}s over {steady_steps} steps "
            f"({t_steady / steady_steps * 1000:.0f} ms/step)"
        )

        ttnn.synchronize_device(self.mesh_device)
        if ttnn.using_distributed_env():
            ttnn.distributed_context_barrier()
        video_rows[num_cond:] = (
            local_device_to_torch(self._tt_video.value)
            .reshape(-1, video_rows.shape[-1])[:v_target]
            .to(video_rows.dtype)
        )
        audio_rows[num_cond_audio:] = (
            local_device_to_torch(self._tt_audio.value)
            .reshape(-1, audio_rows.shape[-1])[:a_target]
            .to(audio_rows.dtype)
        )

        # RuntimeError, not AssertionError: these are real failures of the loop, not caller errors, and
        # they must not be strippable by `python -O`.
        for name, current, anchors in (
            ("video", video_rows[:num_cond], anchor_rows),
            ("audio", audio_rows[:num_cond_audio], anchor_audio_rows),
        ):
            if anchors is None or torch.equal(current, anchors):
                continue
            changed = int((current != anchors).any(dim=-1).sum())
            raise RuntimeError(
                f"{changed} of {anchors.shape[0]} {name} conditioning rows changed during denoising; the "
                "loop's write mask is wrong and the conditioning is not being honoured"
            )

        return video_rows, audio_rows

    def _decode_video(
        self,
        vae: MiniMaxH3Vae,
        rows: torch.Tensor,
        num_latent_frames: int,
        latent_height: int,
        latent_width: int,
        num_condition_video_rows: int,
    ) -> torch.Tensor:
        """Decode the *target* rows. `fl2va`'s leading condition rows are dropped, not decoded --- they
        are the keyframe, which the caller already has.
        """
        latents = unpatchify_video_tokens(
            rows[num_condition_video_rows:],
            num_latent_frames,
            latent_height,
            latent_width,
            self.vae_config.latent_channels,
            self.patch_size,
        )
        latents = self._denormalize(latents, self.vae_config.latents_mean, self.vae_config.latents_std)
        vae.log_profile = self._log_generation
        video = vae.decode(latents, output_type="yuv420" if self.vae_output_type == "yuv420" else "float")
        if self.vae_output_type == "yuv420":
            # De-normalized, clamped and colour-converted on device; nothing left to do on host.
            return video
        if self.vae_output_type == "uint8":
            # `float_to_uint8` already applied *both* halves of the mapping on device: `proj_out`'s
            # fold put pixels in [-1, 1], and the cast then took [-1, 1] -> [0, 255]. So the decode
            # returns 0..255 and the only step left is the scale. Treating it as [-1, 1] here (as
            # `add(1).mul(0.5).clamp(0,1)` does) de-normalizes twice and saturates every pixel at or
            # above 1/255 to white -- measured mean 0.994 against a correct 0.345.
            return video.float().div_(255.0)
        # The VAE emits ImageNet-normalized RGB.
        video = self._denormalize(video.float(), MINIMAX_H3_PIXEL_MEAN, MINIMAX_H3_PIXEL_STD).clamp(0, 1)
        return video

    def _decode_video_host(
        self,
        vae: MiniMaxH3Vae,
        rows: torch.Tensor,
        num_latent_frames: int,
        latent_height: int,
        latent_width: int,
        num_condition_video_rows: int,
    ) -> torch.Tensor:
        """Host/reference twin of `_decode_video`, for isolating device-decode issues.

        Decodes on the CPU and always returns `(1, 3, F, H, W)` float in `[0, 1]`; `vae` is unused.
        """

        latents = unpatchify_video_tokens(
            rows[num_condition_video_rows:],
            num_latent_frames,
            latent_height,
            latent_width,
            self.vae_config.latent_channels,
            self.patch_size,
        )
        latents = self._denormalize(latents, self.vae_config.latents_mean, self.vae_config.latents_std)
        reference = self._prepare_host_vae(want_decoder=True)
        self._log(f"decoding {num_latent_frames} latent frame(s) on HOST (reference VAE decoder)")
        with torch.no_grad():
            video = reference.decode(latents).sample
        return self._denormalize(video.float(), MINIMAX_H3_PIXEL_MEAN, MINIMAX_H3_PIXEL_STD).clamp(0, 1)

    def _decode_audio(
        self,
        audio_decoder: MiniMaxH3AudioDecoder,
        rows: torch.Tensor,
        num_audio_latents: int,
        num_condition_audio_rows: int = 0,
    ) -> torch.Tensor:
        """Decode the *target* audio rows. `ref2va`'s leading reference rows are dropped, not decoded.

        The drop is not optional bookkeeping: `unpack_audio_tokens` reshapes to
        `(2, num_audio_latents, C)` and would silently mis-split a longer tensor, folding half a
        reference soundtrack into the left channel. It asserts nothing, so this parameter is the only
        thing standing between a ref2va request and a scrambled soundtrack.
        """
        if num_condition_audio_rows:
            rows = rows[num_condition_audio_rows:]
        expected = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
        assert rows.shape[0] == expected, f"expected {expected} target audio rows to decode, got {rows.shape[0]}"
        latents = unpack_audio_tokens(rows, num_audio_latents)
        latents = self._denormalize(latents, self.audio_config["latents_mean"], self.audio_config["latents_std"])
        waveform = audio_decoder(latents, traced=self.audio_trace)
        # The audio VAE is mono and took the two stereo channels as two batch items.
        return waveform.float().permute(1, 0, 2)
