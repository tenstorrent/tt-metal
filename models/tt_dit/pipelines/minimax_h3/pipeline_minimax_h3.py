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
replicates ~9.8 GB of fp32 weights per device for its data-parallel fan-out. They are therefore
paged: each stage loads what it needs and releases the previous stage's weights first. The text
encoder is the cheap one --- FSDP over the non-TP axis puts it at ~1.6 GB/device --- but its 50 GB
disk read is not, which is why it is kept co-resident by default: every request pays the encode
itself (~2.8 s), never the reload.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger
from PIL import Image, ImageOps

import ttnn

from ...encoders.qwen3vl.loader_minimax_h3 import build_minimax_h3_text_encoder, build_minimax_h3_vision_tower
from ...encoders.qwen3vl.model_qwen3vl import create_rope_tensors, mrope_position_ids, vision_token_runs
from ...encoders.qwen3vl.vision_qwen3vl import vision_cu_seqlens
from ...layers.audio_ops import weights_variant
from ...models.audio_vae.minimax_h3.convert_minimax_h3_audio import convert_minimax_h3_audio_state_dict
from ...models.audio_vae.minimax_h3.decoder_minimax_h3_audio import MiniMaxH3AudioDecoder
from ...models.audio_vae.minimax_h3.encoder_minimax_h3_audio import MiniMaxH3AudioEncoder
from ...models.transformers.minimax_h3.adaln_cache_minimax_h3 import MiniMaxH3AdalnCache
from ...models.transformers.minimax_h3.attention_minimax_h3 import prepare_rope_tables
from ...models.transformers.minimax_h3.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from ...models.vae.minimax_h3.vae_minimax_h3 import DEFAULT_TILE_SIZE, MiniMaxH3Vae, MiniMaxH3VaeConfig
from ...parallel.config import DiTParallelConfig, EncoderParallelConfig, ParallelFactor, VAEParallelConfig
from ...parallel.manager import CCLManager
from ...utils import cache
from ...utils.conv3d import conv3d_blocking_hash
from ...utils.tensor import bf16_tensor, from_torch, local_device_to_torch
from ...utils.tracing import StateTensor
from ..events import DenoiseStep, PipelineEventCallback, SectionEnd, SectionStart, null_callback
from .adaln_precompute import precompute_adaln_table, request_step_timesteps
from .conditioning import MINIMAX_H3_PIXEL_MEAN as _MINIMAX_H3_PIXEL_MEAN
from .conditioning import MINIMAX_H3_PIXEL_STD as _MINIMAX_H3_PIXEL_STD
from .conditioning import encode_keyframes, keyframe_condition_noise
from .packing import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_AUDIO_LATENTS_PER_SECOND,
    MINIMAX_H3_FPS,
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
    MiniMaxH3PackedSequence,
    adaln_indices,
    align_num_frames,
    audio_latent_num_frames,
    build_packed_sequence,
    build_rope_tables,
    build_row_timesteps,
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
    MiniMaxH3PreparedReference,
    MiniMaxH3Reference,
    build_ref2va_packed_sequence,
    build_ref2va_presentation,
    sample_reference_video_frames,
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

# Cache namespace under TT_DIT_CACHE_DIR. `utils.cache` keys each entry on this plus the subfolder,
# the parallel config, the mesh shape, the dtype and the FSDP flag.
MODEL_NAME = "minimax-h3"

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
    },
}


def _audio_lever_flag(name):
    raw = os.environ.get(name)
    return None if raw is None else raw not in ("0", "false", "False", "")


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
    """One generation. `video` is `(1, 3, F, H, W)` in [0, 1]; `audio` is `(1, 2, samples)`."""

    video: torch.Tensor
    audio: torch.Tensor
    sampling_rate: int
    num_frames: int
    fps: int = MINIMAX_H3_FPS

    @property
    def video_seconds(self) -> float:
        return self.num_frames / self.fps

    @property
    def audio_seconds(self) -> float:
        return self.audio.shape[-1] / self.sampling_rate


def _is_host_rank() -> bool:
    return not ttnn.using_distributed_env() or int(ttnn.distributed_context_get_rank()) == 0


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
        precomputed_adaln: bool = False,
        dit_fsdp: bool = False,
        trace_denoise: bool | None = None,
    ) -> None:
        self.mesh_device = mesh_device
        self.weights_dir = Path(weights_dir)
        # Only consult the preset for what the caller left unset, so an untuned shape with every
        # parallel setting supplied runs rather than raising -- the escape hatch `create_pipeline`
        # documents. `coresident` is residency rather than parallelism and has a safe default, so it
        # does not hold the hatch shut for callers that cannot pass it.
        supplied = (tp_axis, sp_axis, num_links, topology)
        preset = resolve_mesh_preset(tuple(mesh_device.shape), required=any(v is None for v in supplied))
        tp_axis = preset["tp_axis"] if tp_axis is None else tp_axis
        sp_axis = preset["sp_axis"] if sp_axis is None else sp_axis
        num_links = preset["num_links"] if num_links is None else num_links
        topology = preset["topology"] if topology is None else topology
        coresident = preset.get("coresident", True) if coresident is None else coresident
        # Preset default (quad-only), overridable so the traced resident path is testable on a
        # single 4x8 Galaxy too rather than only where the preset turns it on.
        self.trace_denoise = preset.get("trace_denoise", False) if trace_denoise is None else trace_denoise
        # False during `warmup`: generation logs (stage times, per-step, VAE profile) are the
        # measured-call report, not the compile pass. Construction logs still go through `_host_log`.
        self._log_generation = True
        # Denoise generations completed. Tracing engages only after one untraced pass; see `_denoise`.
        # The request a live trace was captured at: shapes plus the AdaLN cache object. A capture is
        # only valid for that exact request, so both are compared before reusing it.
        self._trace_signature: tuple | None = None
        self._trace_adaln_cache = None
        # Persistent per-step trace I/O for the denoise loop, the LTX / Flux2 pattern: a ttnn trace
        # bakes its inputs' addresses, so the buffers that change every step live in one place and are
        # refreshed *in* (via `ttnn.copy` when traced) rather than reallocated. The static per-request
        # inputs -- rope, prompt, conditioning, and the resident routing indices -- are built once in
        # the preamble instead. `update(traced=False)` on the first, untraced pass allocates each
        # buffer; a shape change releases the trace and the next untraced pass rebinds them.
        self._tt_video = StateTensor()
        self._tt_audio = StateTensor()
        self._tt_timestep = StateTensor()
        # The per-request-constant trace inputs. They never change across steps, but they are read on
        # *every* traced step, so they must live in buffers whose address is stable across the capture
        # call and every later replay call -- otherwise the tracer copies a freshly allocated (post-
        # capture) tensor into the trace each step, and it is clobbered by `execute_trace` after the
        # first step (see `Tracer`'s "allocated after capture may be overwritten" contract). Updated
        # once in the preamble with `traced=traced`: the untraced warmup pass allocates the buffer
        # before capture, and traced passes `ttnn.copy` into that same buffer. `_tt_cond` is a list
        # because ref2va packs several conditioning blocks; it is sized on demand in the preamble.
        self._tt_rope_cos = StateTensor()
        self._tt_rope_sin = StateTensor()
        self._tt_prompt = StateTensor()
        self._tt_adaln = StateTensor()
        self._tt_tsi = StateTensor()
        self._tt_cond: list[StateTensor] = []
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

        self.ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
        self.dit_parallel_config = DiTParallelConfig(
            tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=self.tp_factor),
            sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=self.sp_factor),
            cfg_parallel=None,
        )
        self.encoder_parallel_config = EncoderParallelConfig(
            tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=self.tp_factor)
        )
        # Both VAEs are data-parallel over work units with *replicated* weights -- no tensor
        # parallelism at all -- so their cache key carries factor 1. Recording it as a config rather
        # than passing a literal keeps the key honest if that ever changes.
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

        # Nothing is built until the stage that needs it; see the class docstring on residency.
        self._tokenizer = None
        self._text_encoder = None
        self._text_config = None
        self._transformer = None
        self._vae = None
        self._encoder_state_loaded = False
        self._image_processor = None
        # `"yuv420"` builds the VAE for the device-stitched path: the canvas is blended, clamped and
        # colour-converted on device, and `_decode_video` returns planar `(T, H*3//2, W)` uint8 for
        # `export_video_audio_yuv` instead of a `(1, 3, T, H, W)` float tensor. Off by default because
        # it changes this method's return type, and every quality gate reads the float one.
        self.vae_output_type = "float"  # "float" | "uint8" | "yuv420"
        # Decode tiles per device per wave (batch dim). >1 cuts wave count at the cost of activation
        # memory; set before the first decode. 1 is the original one-tile-per-device schedule.
        self.vae_waves_per_device = 2
        self._video_processor = None
        self._vision_tower = None
        self._vision_config = None
        self._audio_decoder = None
        self._audio_encoder = None
        self._adaln_cache = None
        self._adaln_cache_steps: int | None = None
        # AdaLN strategy. The default (`precomputed_adaln=False`) keeps the `adaln_proj` weights
        # resident and projects `temb` on device every step. `precomputed_adaln=True` projects every
        # block's `adaln_proj` on host into a modulation table for the request's schedule, so the
        # 26 GB of projection weights (6.50 GB/device at TP=4) never reach the device.
        # `dit_fsdp=True` shards the DiT's SP-replicated weights over the SP axis (all-gathered per
        # use) to relieve memory pressure.
        self.precomputed_adaln = precomputed_adaln
        self.dit_fsdp = dit_fsdp
        self._resident: str | None = None
        # The last call's padded packed length. Exposed so a perf test can assert that `warmup` and the
        # measured call agree on it -- every program in the 50-block stack is keyed on this, so a
        # mismatch means the "warm" number was cold and nothing else would say so.
        self.last_padded_len: int | None = None

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
        precomputed_adaln: bool = False,
        dit_fsdp: bool = False,
        trace_denoise: bool | None = None,
    ) -> "MiniMaxH3Pipeline":
        """`task="t2va"` serves both t2va and fl2va; `task="ref2va"` loads `transformer_ref/`.

        The parallel configuration defaults to this mesh shape's entry in `_PRESETS_BH`; pass any of
        `tp_axis`/`sp_axis`/`num_links`/`topology` to override it.

        `precomputed_adaln=True` projects every block's `adaln_proj` on host into a modulation table
        so the 26 GB of projection weights never reach the device; the default (`False`) keeps those
        weights resident and projects `temb` on device each step. `dit_fsdp=True` shards the DiT over
        the SP axis. `trace_denoise` defaults to the mesh preset (on for the quad only); pass `True`
        to trace the denoise step on other shapes, e.g. to exercise the traced resident path on one
        4x8 Galaxy.
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
            precomputed_adaln=precomputed_adaln,
            dit_fsdp=dit_fsdp,
            trace_denoise=trace_denoise,
        )

    @staticmethod
    def _ranks_agree(local: bool) -> bool:
        """Whether *every* rank sees `local` as true. A collective; all ranks must call it.

        A per-rank `Path.is_file()` must not gate collective work: a shared cache can disagree between
        hosts, and a rank taking a cached early return skips collectives the others are still waiting
        in, which deadlocks rather than fails. Unanimity, as `utils/cache.py` does for the weight
        cache, so a partially populated cache costs a recompute instead of a hang.
        """
        if not ttnn.using_distributed_env():
            return local
        return all(ttnn.distributed_context_allgather_int(1 if local else 0))

    def _read_config(self, subfolder: str) -> dict:
        path = self.weights_dir / subfolder / "config.json"
        if not path.is_file():
            raise FileNotFoundError(f"no {subfolder}/config.json under {self.weights_dir}")
        return {k: v for k, v in json.loads(path.read_text()).items() if not k.startswith("_")}

    def _read_safetensors(self, subfolder: str) -> dict[str, torch.Tensor]:
        """A partition's weights, sharded or single-file. `transformer` and `vae` are sharded here.

        On the precomputed-AdaLN path the keys the table replaces are dropped per shard as it is read
        and never accumulated into the returned state. The shard still has to be read to get at the
        keys beside them -- avoiding that entirely would need per-key `safe_open` access -- but the
        26 GB never reaches the state dict, the device, or the weight cache. On the resident-AdaLN
        path (`precomputed_adaln=False`) those same keys are exactly what the on-device projections
        load, so they are kept.
        """
        from safetensors.torch import load_file

        # Either transformer partition: the keys the precomputed table replaces are the same in both,
        # so this must not test for the literal "transformer". Only dropped when the table actually
        # replaces them -- the resident path needs `adaln_proj` / `time_embedder` / `norm_out.linear`.
        drop = self.precomputed_adaln and subfolder.startswith("transformer")

        def keep(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            if not drop:
                return state
            return {
                k: v
                for k, v in state.items()
                if ".adaln_proj." not in k and not k.startswith("time_embedder.") and "norm_out.linear" not in k
            }

        directory = self.weights_dir / subfolder
        index = directory / "diffusion_pytorch_model.safetensors.index.json"
        state: dict[str, torch.Tensor] = {}
        if index.is_file():
            for shard in sorted(set(json.loads(index.read_text())["weight_map"].values())):
                state.update(keep(load_file(str(directory / shard))))
        else:
            single = directory / "diffusion_pytorch_model.safetensors"
            if not single.is_file():
                raise FileNotFoundError(f"no safetensors (sharded or single) under {directory}")
            state.update(keep(load_file(str(single))))
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

    def _make_resident(self, stage: str) -> None:
        """Release whatever the previous stage held before the next one allocates.

        `MiniMaxH3Vae` keeps its lazily-built per-shape encoders and decoders in a plain dict rather
        than as registered child `Module`s, so `deallocate_weights()` does not reach them; they are
        dropped explicitly. It still holds its host state dict, so rebuilding is a re-upload rather
        than a re-read from disk.

        **The text encoder is kept co-resident too.** Measured on a 4x8 Blackhole mesh with the
        precomputed AdaLN path: encoder, DiT and VAE fit together, and a prompt's Encoder row
        drops from **23.9 s to 2.8 s** because the 50 GB reload disappears. Every request runs the
        conditioner encode, so this is on the critical path of essentially every served request.
        `coresident=False` evicts each stage for a mesh where they do not fit.

        **The DiT and the video VAE are kept co-resident**, which is measured, not assumed: on a 4x8
        Blackhole mesh they fit together with no allocation failure. Eviction costs a per-shape decoder
        rebuild per generation plus a ~50 s DiT reload on the next one; co-resident measures the VAE
        decode row at **6.0 s against 17.6 s** and the fully-warm total at **69.1 s against 81.1 s**.
        `coresident=False` evicts each stage for a mesh where they do not fit.
        """
        if self._resident == stage:
            return
        # One `elif` chain, and the branches are mutually exclusive by construction: each tests
        # `self._resident`, which holds exactly one value. It reads like a fallthrough of independent
        # release actions and is not one.
        coresident = self.coresident
        if self._resident == "text" and self._text_encoder is not None and not coresident:
            self._host_log("releasing the text encoder")
            self._text_encoder.deallocate_weights()
            self._text_encoder = None
        elif self._resident == "dit" and self._transformer is not None and not coresident:
            self._host_log("releasing the transformer")
            self._transformer.deallocate_weights()
            self._transformer = None
        elif self._resident == "vae" and self._vae is not None and not coresident:
            self._host_log("releasing the video VAE")
            self._vae._encoders.clear()
            self._vae._decoders.clear()
        self._resident = stage

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

    def _prepare_vision_tower(self):
        if self._vision_tower is None:
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prompt to `(prompt_embeds [1, L, 5120], text_token_tags [L])`, encoded on device.

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

        self._make_resident("text")
        if self._text_encoder is None:
            self._text_encoder, self._text_config = build_minimax_h3_text_encoder(
                self.weights_dir / "text_encoder",
                mesh_device=self.mesh_device,
                parallel_config=self.encoder_parallel_config,
                ccl_manager=self.ccl_manager,
                is_fsdp=True,
            )
        config = self._text_config

        # The vision tower, and the two ways its output enters the decoder. Run before the rope tables
        # so a tower failure surfaces before any decoder work.
        vision_kwargs = {}
        if has_vision:
            tower = self._prepare_vision_tower()
            vis_cos, vis_sin = tower.prepare_rope(grid_thw)
            merged, deepstack = tower.forward(
                bf16_tensor(pixel_values.float(), device=self.mesh_device),
                pos_embeds=bf16_tensor(tower.prepare_pos_embeds(grid_thw), device=self.mesh_device),
                rope=(
                    bf16_tensor(vis_cos, device=self.mesh_device),
                    bf16_tensor(vis_sin, device=self.mesh_device),
                ),
                # One attention block per image and one per FRAME of a video, so `fl2va`'s
                # single image reduces to full attention. Block form rather than a dense mask:
                # an `s x s` mask is 17 GiB for a nine-image request. Requires the tower to be
                # replicated (sp_factor 1), per `vision_qwen3vl.py`.
                cu_seqlens=vision_cu_seqlens(grid_thw),
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
        rope_scaling = config["rope_scaling"]
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
            config["head_dim"],
            rope_scaling.get("rope_theta", config["rope_theta"]),
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
        taps = self._text_encoder.forward(
            tt_ids,
            attention_mask=None,
            pos_embeds=(bf16_tensor(cos, device=self.mesh_device), bf16_tensor(sin, device=self.mesh_device)),
            **vision_kwargs,
        )
        embeds = local_device_to_torch(taps[0]).float()

        return embeds, tags

    # ------------------------------------------------------------------ denoiser

    def _prepare_transformer(self) -> MiniMaxH3Transformer3DModel:
        self._make_resident("dit")
        if self._transformer is not None:
            return self._transformer

        config = {k: v for k, v in self.transformer_config.items() if k not in ("rope_freq_dim", "rope_theta")}
        config["patch_size"] = tuple(config["patch_size"])

        # Default (`precomputed_adaln=False`): the `adaln_proj` weights stay resident and `temb` is
        # projected on device each step. `precomputed_adaln=True` projects every `adaln_proj` on host
        # into a table for the exact schedule so the 26 GB of projection weights (6.50 GB/device at
        # TP=4) never reach the device. `dit_fsdp=True` additionally shards the DiT over SP.
        # See `models/transformers/minimax_h3/adaln_cache_minimax_h3.py`.
        adaln_mode = "precomputed_adaln" if self.precomputed_adaln else "resident_adaln"
        if self.dit_fsdp:
            adaln_mode += "_fsdp"
        self._host_log(
            f"building the {config['num_layers']}-layer transformer from {self.transformer_subfolder}/, "
            f"TP={self.tp_factor}/SP={self.sp_factor} ({adaln_mode})"
        )
        model = MiniMaxH3Transformer3DModel(
            **config,
            mesh_device=self.mesh_device,
            ccl_manager=self.ccl_manager,
            parallel_config=self.dit_parallel_config,
            precomputed_adaln=self.precomputed_adaln,
            is_fsdp=self.dit_fsdp,
            cache_padding=self.trace_denoise,
        )

        # Cache-aware: on a hit this reads pre-sharded device tensors instead of 62 GB of
        # safetensors plus every `_prepare_torch_state` fixup. With TT_DIT_CACHE_DIR unset it falls
        # through to the direct load and logs that it did. The load underneath is strict, and the key
        # count differs by mode (the resident path carries the `adaln_proj` tensors the precomputed
        # one drops), so a single unmapped key is a real bug.
        cache.load_model(
            model,
            model_name=MODEL_NAME,
            # `adaln_mode` keeps the four builds (precomputed / resident, each ± FSDP) in distinct
            # cache entries so a differently-sharded or differently-populated state can never read a
            # stale file, and the partition term keeps the two partitions -- which share a repository
            # and a config -- from hashing to one entry.
            subfolder=f"{self.transformer_subfolder}_{adaln_mode}",
            parallel_config=self.dit_parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            mesh_device=self.mesh_device,
            get_torch_state_dict=lambda: self._read_safetensors(self.transformer_subfolder),
        )
        self._transformer = model
        return model

    def _adaln_cache_path(self, num_inference_steps: int) -> Path:
        """Disk location for a built table.

        The key must cover **everything the rows depend on**, because a stale hit is silent: it
        modulates every block slightly wrong at every step, in the same direction, and nothing
        downstream can notice. That is the schedule (step count and both per-modality shifts), the
        conditioning floor, the model geometry, and the checkpoint itself.
        """
        cache_dir = Path(os.environ.get("TT_DIT_CACHE_DIR") or Path.home() / ".cache/tt-dit") / "minimax-h3-adaln"
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = "|".join(
            str(part)
            for part in (
                self.weights_dir.resolve(),
                # The partition: the two share a repository and a config, so without it
                # they hash to one file and a ref2va run modulates with t2va's weights.
                self.transformer_subfolder,
                num_inference_steps,
                VIDEO_SHIFT,
                AUDIO_SHIFT,
                MINIMAX_H3_KEYFRAME_NOISE_AUG,
                # ref2va carries a fourth level (the audio conditioning t = 1.0), so its table
                # has more rows per step than t2va's and the two must not share a file.
                self.task,
                self.transformer_config["num_layers"],
                self.transformer_config["hidden_size"],
                self.transformer_config["freq_dim"],
            )
        )
        return cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()[:32]}.adaln.pt"

    def _prepare_adaln_cache(self, num_inference_steps: int):
        """Host-build (or load) the modulation table for this schedule and upload it.

        A prepare, so it belongs outside every timed row: the build reads the checkpoint's AdaLN
        weights on host and is paid once per (checkpoint, schedule).
        """
        if self._adaln_cache is not None and self._adaln_cache_steps == num_inference_steps:
            return self._adaln_cache

        video = MiniMaxH3Scheduler(shift=VIDEO_SHIFT)
        audio = MiniMaxH3Scheduler(shift=AUDIO_SHIFT)
        video.set_timesteps(num_inference_steps)
        audio.set_timesteps(num_inference_steps)
        step_timesteps = request_step_timesteps(
            video.sigmas,
            audio.sigmas,
            MINIMAX_H3_KEYFRAME_NOISE_AUG,
            # A fourth level for ref2va only: reference soundtrack rows run at a literal
            # t = 1.0. t2va and fl2va have no audio conditioning rows and keep three.
            audio_condition_timestep=MINIMAX_H3_AUDIO_CONDITION_TIMESTEP if self.task == "ref2va" else None,
        )

        path = self._adaln_cache_path(num_inference_steps)
        # Unanimous even though both branches are host-only: a rank reading a half-written table
        # diverges numerically and silently.
        if self._ranks_agree(path.is_file()):
            self._host_log(f"AdaLN table from cache: {path}")
            table = torch.load(path, weights_only=False)
        else:
            self._host_log(
                f"building the AdaLN table on host for {len(step_timesteps)} forwards from "
                f"{self.transformer_subfolder}/ (reads the checkpoint)"
            )
            t0 = time.time()
            table = precompute_adaln_table(
                self.weights_dir / self.transformer_subfolder,
                step_timesteps,
                num_layers=self.transformer_config["num_layers"],
                hidden_size=self.transformer_config["hidden_size"],
                freq_dim=self.transformer_config["freq_dim"],
            )
            self._host_log(f"AdaLN table built in {time.time() - t0:.1f}s ({table.nbytes() / 1e9:.3f} GB); caching")
            is_distributed = ttnn.using_distributed_env()
            try:
                if not is_distributed or int(ttnn.distributed_context_get_rank()) == 0:
                    torch.save(table, path)
            except OSError as exc:
                # Warn rather than raise: the table is already in memory, so a failed write costs a
                # recompute next run and nothing else. Every rank then misses the cache together,
                # since `_ranks_agree` reads the same absent file, so the ranks stay consistent.
                if _is_host_rank():
                    logger.warning(f"could not cache the AdaLN table to {path}: {exc}")
            finally:
                # In `finally`, so a write error on rank 0 cannot leave its peers blocked here.
                if is_distributed:
                    ttnn.distributed_context_barrier()

        self._adaln_cache = MiniMaxH3AdalnCache(
            table,
            mesh_device=self.mesh_device,
            parallel_config=self.dit_parallel_config,
            num_layers=self.transformer_config["num_layers"],
            hidden_size=self.transformer_config["hidden_size"],
        )
        self._adaln_cache.assert_covers(len(step_timesteps))
        self._adaln_cache_steps = num_inference_steps
        self._adaln_table = table
        return self._adaln_cache

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
        """The video VAE, built and loaded on first access."""
        return self._prepare_vae()

    def _prepare_vae(
        self,
        *,
        decode_shape: tuple[int, int, int] | None = None,
        encode_shape: tuple[int, int, int] | None = None,
        encode_taps: int = 1,
    ) -> MiniMaxH3Vae:
        """Build the VAE and, given `decode_shape` / `encode_shape`, its per-shape sub-models too.

        These arguments are about measurement, not convenience. `MiniMaxH3Vae` builds a decoder **per
        distinct (T, H, W)** and uploads that decoder's ~4.6 GB of weights at construction. Without
        them that upload happens lazily inside `vae.decode()`, landing *inside* the timed VAE-decode
        row -- and weight upload is one-time construction cost the measurement contract does not
        count. `encode_shape` does the same for the keyframe encoder, smaller at 0.72 GB against the
        decoder's 9.7 but on the same principle.

        `load_encoder_state` is called whenever the encoder state is needed at all, and eagerly rather
        than lazily: `encode_clip` raises `RuntimeError("call load_encoder_state() before encoding")`,
        which is the right error but fires at encode time --- after the DiT has been built and inside a
        timed row. One `_read_safetensors("vae")` feeds both loaders; reading the 10.4 GB twice would be
        pure waste.
        """
        self._make_resident("vae")
        want_encoder = encode_shape is not None
        if self._vae is None:
            self._host_log("building the video VAE")
            # Used only for the wave readback, which keeps the decode off the MPI path; the VAE's
            # forward runs no collectives.
            yuv = self.vae_output_type == "yuv420"
            unit_pixels = self.vae_output_type in ("uint8", "yuv420")
            self._vae = MiniMaxH3Vae(
                self.vae_config,
                mesh_device=self.mesh_device,
                weight_loader=self._cache_submodel,
                ccl_manager=self.ccl_manager,
                device_stitch=yuv,
                # Folded into `proj_out`, so the decoder emits the `[-1, 1]` both the colour kernel
                # and the uint8 cast take, and `_decode_video` is left with at most a range shift.
                pixel_denorm=(MINIMAX_H3_PIXEL_MEAN, MINIMAX_H3_PIXEL_STD) if unit_pixels else None,
                readback_uint8=self.vae_output_type == "uint8",
                waves_per_device=self.vae_waves_per_device,
            )
            state = self._read_safetensors("vae")
            self._vae.load_decoder_state(state)
            if want_encoder:
                self._vae.load_encoder_state(state)
                self._encoder_state_loaded = True
        elif want_encoder and not self._encoder_state_loaded:
            self._vae.load_encoder_state(self._read_safetensors("vae"))
            self._encoder_state_loaded = True
        if decode_shape is not None and decode_shape not in self._vae._decoders:
            t0 = time.time()
            self._vae._decoder_for(*decode_shape)
            self._host_log(f"per-shape decoder {decode_shape} built in {time.time() - t0:.1f}s")
        if encode_shape is not None:
            # taps=1 for a single frame: the causal front-pad is zeros, so a 3-tap temporal conv
            # collapses to `weight[:, :, -1]` exactly, not approximately. A ref2va
            # video reference goes through `vae.encode` at taps=3 and needs its own per-shape
            # encoder, built here so the weight upload stays outside the timed encode row.
            key = (*encode_shape, encode_taps)
            if key not in self._vae._encoders:
                t0 = time.time()
                self._vae._encoder_for(*key)
                self._host_log(f"per-shape encoder {encode_shape} taps={encode_taps} built in {time.time() - t0:.1f}s")
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
            self._host_log("building the audio encoder (ref2va reference soundtracks)")
            encoder = MiniMaxH3AudioEncoder(
                encoder_dim=config["encoder_dim"],
                encoder_rates=tuple(config["encoder_rates"]),
                latent_dim=config["latent_dim"],
                latent_channels=config["latent_channels"],
                num_attention_heads=config["num_attention_heads"],
                mesh_device=self.mesh_device,
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
                subfolder="audio_encoder"
                + weights_variant(encoder.split_mode, encoder.tap_matmul, encoder.max_c_in_block),
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
        in. `temporal_taps` is not passed: `encode_clip` auto-selects 1 for `T == 1`, and
        leaving that as the single decision point means the keyframe path cannot disagree with the
        VAE's own view of what a keyframe is.
        """
        return encode_keyframes(
            keyframes,
            vae.encode_clip,
            self.vae_config.latents_mean,
            self.vae_config.latents_std,
            self.patch_size,
        )

    def decode_unit_shape(self) -> tuple[int, int, int]:
        """The `(T, H, W)` of one decoder work unit: one temporal chunk of one spatial tile.

        Takes no arguments: the unit is fixed by the VAE's tiling *independently of resolution and
        duration*, which is what lets one per-shape decoder serve the whole video and what the cache
        key records.

        `tile_size` is read off the VAE when one exists and falls back to the module default otherwise.
        The fallback is the branch t2va actually takes, because nothing has built the VAE at the point
        the caller evaluates this argument --- so it must be the real default, not a literal.
        """
        config = self.vae_config
        tile_size = self._vae.tile_size if self._vae is not None else DEFAULT_TILE_SIZE
        return (
            config.tokens_chunk_size + config.token_overlap,
            tile_size // config.spatial_compression_ratio,
            tile_size // config.spatial_compression_ratio,
        )

    def _prepare_audio_decoder(self) -> MiniMaxH3AudioDecoder:
        if self._audio_decoder is None:
            config = self.audio_config
            self._host_log("building the audio decoder")
            audio_levers = {
                k: v
                for k, v in (
                    ("split_mode", os.environ.get("MINIMAX_H3_PIPELINE_SPLIT_MODE")),
                    ("tap_matmul", _audio_lever_flag("MINIMAX_H3_PIPELINE_TAP_MATMUL")),
                    ("prefer_mac", _audio_lever_flag("MINIMAX_H3_PIPELINE_PREFER_MAC")),
                )
                if v is not None
            }
            decoder = MiniMaxH3AudioDecoder(
                latent_channels=config["latent_channels"],
                latent_dim=config["latent_dim"],
                decoder_dim=config["decoder_dim"],
                decoder_rates=tuple(config["decoder_rates"]),
                decoder_kernel_sizes=tuple(config["decoder_kernel_sizes"]),
                resblock_kernel_sizes=tuple(config["resblock_kernel_sizes"]),
                resblock_dilation_sizes=tuple(tuple(d) for d in config["resblock_dilation_sizes"]),
                mesh_device=self.mesh_device,
                ccl_manager=self.ccl_manager,
                **audio_levers,
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
                + weights_variant(decoder.split_mode, decoder.tap_matmul, decoder.max_c_in_block),
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
        num_inference_steps: int = 50,
        seed: int = 0,
        on_event: PipelineEventCallback | None = None,
    ) -> MiniMaxH3Output:
        """`image` and/or `last_image` select `fl2va`; `references` selects `ref2va`; neither `t2va`.

        Note that `fl2va` at a given seed does **not** reproduce `t2va` at that seed, even with a
        keyframe that contributes nothing: the conditioning noise is the first draw off the request
        generator and shifts the video and audio streams behind it. That is the reference's draw
        order.
        """
        # Stages fire `SectionStart` / `SectionEnd` for the caller to time, matching Wan.
        # Weight upload is one-time construction cost, so every `_prepare_*` happens outside a
        # section. The one exception: the first encoder section of a fresh pipeline loads the text
        # encoder inside the section; after that the encoder stays resident and the section measures
        # the encode alone.
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
        task = "t2va" if not keyframes else ("fl2va" if image is not None else "fl2va_last_frame")
        self._log(
            f"{task} {width}x{height}, {num_frames} frames ({num_frames / MINIMAX_H3_FPS:.2f} s), "
            f"{num_latent_frames} latent frames, {num_audio_latents} audio latents, "
            f"{num_inference_steps} steps, anchors={keyframe_anchors or '()'}"
        )

        # 2. Text (plus the vision block, for fl2va).
        on_event(SectionStart("encoder"))
        prompt_embeds, text_token_tags = self.encode_prompt(prompt, keyframes=keyframes)
        on_event(SectionEnd("encoder"))

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
            # The 0.72 GB encoder upload is a prepare and stays outside the timed row, same rule as the
            # DiT and the per-shape decoder.
            # Every keyframe tile is exactly `tile_size` square: `split_tiles` returns `[tile_size] * n`
            # lengths unless one tile already covers the axis, and at 1344x768 neither does. So one
            # `(1, 256, 256)` encoder serves all 28 tiles, which is one wave on a 32-device mesh.
            vae = self._prepare_vae()
            vae = self._prepare_vae(encode_shape=(1, vae.tile_size, vae.tile_size))
            on_event(SectionStart("vae_encode"))
            condition_rows = self._encode_keyframes(vae, keyframes)
            # `scheduler.scale_noise`, never a local copy: a reimplementation computing `1 - t` in
            # Python double instead of the sample dtype drifts 2.4e-7 (see `conditioning.py`), and a
            # test asserts no second implementation exists.
            condition_rows = scheduler.scale_noise(condition_rows, MINIMAX_H3_KEYFRAME_NOISE_AUG, condition_noise)
            on_event(SectionEnd("vae_encode"))

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

        prepared, num_frames = prepare_references(references, num_frames, self.audio_sampling_rate)
        latent_height, latent_width = height // ratio, width // ratio
        num_audio_latents = audio_latent_num_frames(num_frames)
        num_latent_frames = video_latent_num_frames(num_frames)
        kinds = "+".join(reference.kind + ("(+audio)" if reference.has_audio else "") for reference in prepared)
        self._log(
            f"ref2va {width}x{height}, {num_frames} frames ({num_frames / MINIMAX_H3_FPS:.2f} s), "
            f"{num_latent_frames} latent frames, {num_audio_latents} audio latents, "
            f"{num_inference_steps} steps, references=[{kinds}]"
        )

        # 2. Text, plus one vision block per image reference and one per merged frame pair of a video.
        on_event(SectionStart("encoder"))
        prompt_embeds, text_token_tags = self.encode_prompt(prompt, references=prepared)
        on_event(SectionEnd("encoder"))

        scheduler = MiniMaxH3Scheduler(shift=VIDEO_SHIFT)
        audio_scheduler = MiniMaxH3Scheduler(shift=AUDIO_SHIFT)
        scheduler.set_timesteps(num_inference_steps)
        audio_scheduler.set_timesteps(num_inference_steps)

        # 3. Reference VAE encode. Before the layout, because it is what resolves the geometry the
        # layout is built from -- and before the DiT exists, so the encoders get an uncontended
        # residency window. Both weight uploads are prepares and stay outside the timed row.
        has_visual = any(reference.kind != "audio" for reference in prepared)
        has_video = any(reference.kind == "video" for reference in prepared)
        has_audio = any(reference.has_audio for reference in prepared)
        vae = self._prepare_vae() if has_visual else None
        if has_visual:
            vae = self._prepare_vae(encode_shape=(1, vae.tile_size, vae.tile_size), encode_taps=1)
        if has_video:
            vae = self._prepare_vae(
                encode_shape=(self.vae_config.clip_length, vae.tile_size, vae.tile_size), encode_taps=3
            )
        audio_encoder = self._prepare_audio_encoder() if has_audio else None

        on_event(SectionStart("vae_encode"))
        condition_rows, audio_condition_rows = encode_references(
            prepared,
            encode_clip=(lambda pixels: vae.encode_clip(pixels)) if has_visual else None,
            encode_video=(lambda pixels: vae.encode(pixels)) if has_video else None,
            # The device encoder returns `(mean, logs)`; ref2va consumes the mean alone.
            encode_audio=(lambda waveform: audio_encoder(waveform)[0]) if has_audio else None,
            latents_mean=self.vae_config.latents_mean,
            latents_std=self.vae_config.latents_std,
            audio_latents_mean=self.audio_config["latents_mean"],
            audio_latents_std=self.audio_config["latents_std"],
            patch_size=self.patch_size,
            audio_latent_channels=self.audio_config["latent_channels"],
        )
        on_event(SectionEnd("vae_encode"))

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
        prompt_embeds: torch.Tensor,
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

        Extracted rather than duplicated because it is where the section events fire -- every
        `_prepare_*` outside a section, one section per stage -- and two copies of that would drift.
        `condition_spec` is the only thing the tasks differ by here, and only `ref2va` passes one.
        """
        # The weight load is a prepare, so it sits outside the section -- and so is the AdaLN table
        # build, which is paid once per (checkpoint, schedule).
        transformer = self._prepare_transformer()
        # None on the resident-AdaLN path: the blocks project `temb` themselves, so there is no table.
        adaln_cache = self._prepare_adaln_cache(num_inference_steps) if self.precomputed_adaln else None
        on_event(SectionStart("denoising"))
        video_rows, audio_rows = self._denoise(
            transformer,
            layout,
            prompt_embeds,
            video_rows,
            audio_rows,
            scheduler,
            audio_scheduler,
            adaln_cache,
            condition_spec=condition_spec,
            on_event=on_event,
        )
        on_event(SectionEnd("denoising"))

        # Same rule: `_prepare_*` before the section, never inside it -- and for the VAE that includes
        # the per-shape decoder, whose weight upload would otherwise be timed.
        vae = self._prepare_vae(decode_shape=self.decode_unit_shape())
        on_event(SectionStart("vae"))
        video = self._decode_video(
            vae, video_rows, num_latent_frames, latent_height, latent_width, layout.num_condition_video_rows
        )
        on_event(SectionEnd("vae"))

        audio_decoder = self._prepare_audio_decoder()
        on_event(SectionStart("audio"))
        audio = self._decode_audio(audio_decoder, audio_rows, num_audio_latents, layout.num_condition_audio_rows)
        on_event(SectionEnd("audio"))

        return MiniMaxH3Output(
            video=video,
            audio=audio,
            sampling_rate=self.audio_sampling_rate,
            num_frames=video.shape[2],
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
    ) -> None:
        """Compile and allocate everything a real call needs, so the next call measures compute only.

        The analogue of `LTXPipeline.warmup_buffers`. Runs one full generation at the target shape,
        including the text-encoder forward, so the encoder's own kernels compile here too. Every
        program, every per-shape conv3d blocking and every persistent buffer this working point
        touches is resident afterwards.

        "Fully warm" in every number this pipeline reports means *after* this.

        **Pass the real `prompt` and the real keyframes or references**, not the defaults. Every program
        in the 50-block stack is keyed on the *padded* packed length, so warming a different one warms
        nothing. `t2va` survives the one-token default only because 1 and 39 tokens both round up to
        37888; a keyframe's ~1010-row vision block does not. `ref2va` is further still: its padded
        lengths run 46080 to 111616 against t2va's 37888 and depend on the *number and resolution of
        the references*, so a warmup with different references warms nothing even at the same prompt.
        `last_padded_len` is exposed so a caller can assert the warm and measured lengths agree.
        """
        self._log_generation = False
        try:
            self(
                prompt,
                image=image,
                last_image=last_image,
                references=references,
                num_frames=num_frames,
                height=height,
                width=width,
                aspect_ratio=aspect_ratio,
                num_inference_steps=num_inference_steps,
            )
        finally:
            self._log_generation = True

    def release_traces(self) -> None:
        """Release the captured denoise trace, as `WanPipeline.release_traces` does.

        A trace holds device buffers for the whole request and nothing else drops them. A no-op when
        nothing was traced.
        """
        transformer = self._transformer
        if transformer is None:
            return
        tracer = MiniMaxH3Transformer3DModel.traced_step._tracers.get(transformer)
        if tracer is not None:
            tracer.release_trace()

    def _denoise(
        self,
        transformer: MiniMaxH3Transformer3DModel,
        layout: MiniMaxH3PackedSequence,
        prompt_embeds: torch.Tensor,
        video_rows: torch.Tensor,
        audio_rows: torch.Tensor,
        scheduler: MiniMaxH3Scheduler,
        audio_scheduler: MiniMaxH3Scheduler,
        adaln_cache,
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
        alignment = self.sp_factor * ttnn.TILE_SIZE
        padded_len = ((layout.sequence_length + alignment - 1) // alignment) * alignment
        self.last_padded_len = padded_len
        self._log(
            f"packed sequence {layout.sequence_length} -> {padded_len} padded, "
            f"{padded_len // self.sp_factor} rows/device, {num_cond} condition rows"
        )

        timesteps = scheduler.timesteps
        audio_timesteps = audio_scheduler.timesteps

        # Resident-AdaLN routing is constant for the whole request: a row's noise level is fixed by
        # its role. Only `slot_roles` -- which sizes the per-step timestep vector -- is needed before
        # the trace signature; the index tensors it implies are built once below, after `traced` is
        # known, so they land in the persistent buffers the trace bakes rather than a fresh per-call
        # allocation. The precomputed path keeps its own per-step absolute-row indexing in the loop.
        row_slot = slot_roles = None
        if adaln_cache is None:
            row_slot, slot_roles = build_slot_routing(layout)

        # Trace the per-step forward where the preset asks for it, as Wan does on the quad only. At
        # SP=32 a step is dominated by dispatching 50 blocks from host across four MPI ranks rather
        # than by the matmuls, and a trace replaces that dispatch with one replay.
        #
        # `traced_step` has one unkeyed `Tracer` per transformer, so a capture is valid only for the
        # request it was taken at: a different packed length fails the tracer's shape checks. Release
        # the trace whenever the signature changes -- which also lets the transformer reallocate its
        # padding buffer at the new shape without a live trace referencing the old one.
        #
        # The precomputed path additionally keys on the AdaLN cache object and step count: its tables
        # are captured by *address*, so a rebuilt cache would replay the old schedule's modulation.
        # The resident path projects modulation in-trace from the per-step `timestep`, so it keys on
        # shapes alone (with `num_slots`, which sets the timestep width) -- a new schedule at the same
        # resolution reuses the capture and never retraces.
        shape_signature = (tuple(video_rows.shape), tuple(audio_rows.shape), tuple(prompt_embeds.shape))
        if adaln_cache is None:
            signature = (*shape_signature, len(slot_roles))
            warm = signature == self._trace_signature
        else:
            signature = (*shape_signature, len(timesteps))
            warm = signature == self._trace_signature and adaln_cache is self._trace_adaln_cache
        if not warm:
            self.release_traces()
            self._trace_signature = None
            self._trace_adaln_cache = None

        # Not on the first generation at a signature: a capture can neither compile a program nor
        # allocate a buffer, and `CCLManager` fills its persistent buffers lazily. One untraced
        # generation does both, so the first pass at a shape runs untraced and later ones are traced.
        traced = self.trace_denoise and warm
        if traced:
            # `None` on the resident path; `traced_step` reads it off `self`.
            transformer.traced_adaln_cache = adaln_cache

        # Per-request-constant trace inputs -> persistent StateTensors, updated once now that the
        # trace decision is made. A ttnn trace bakes its inputs' addresses, so each must occupy one
        # stable buffer across the capture call and every replay; a per-call `from_torch` would be a
        # post-capture allocation the tracer's contract lets `execute_trace` clobber mid-replay --
        # exactly the corruption that surfaced once a pure-replay call followed the capture. `update`
        # allocates the buffer on the untraced warmup pass and `ttnn.copy`s into it when traced.
        t_rope = time.time()
        rope_cos, rope_sin = self._device_metadata(layout, padded_len)
        self._tt_rope_cos.update(rope_cos, traced=traced)
        self._tt_rope_sin.update(rope_sin, traced=traced)
        t_rope = time.time() - t_rope

        # [1, L, 5120] -> [1, 1, L, 5120], replicated: the model refines the text stream before the
        # packed sequence is fractured, so every device needs all of it.
        self._tt_prompt.update(
            prompt_embeds.reshape(1, 1, -1, prompt_embeds.shape[-1]),
            traced=traced,
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
        )

        # Conditioning blocks are invariant -- the loop writes only rows from `num_cond` /
        # `num_cond_audio` on, and the anchor check below raises if one moved. One resident buffer per
        # block, sized on demand; `t2va` has none, so `tt_cond` stays None.
        tt_cond = None
        if condition_spec:
            while len(self._tt_cond) < len(condition_spec):
                self._tt_cond.append(StateTensor())
            tt_cond = []
            video_cursor = audio_cursor = 0
            for block, (modality, rows) in enumerate(condition_spec):
                if modality == "audio":
                    chunk = audio_rows[audio_cursor : audio_cursor + rows]
                    audio_cursor += rows
                else:
                    chunk = video_rows[video_cursor : video_cursor + rows]
                    video_cursor += rows
                self._tt_cond[block].update(
                    chunk.unsqueeze(0).unsqueeze(0), traced=traced, dtype=ttnn.bfloat16, device=self.mesh_device
                )
                tt_cond.append((self._tt_cond[block].value, modality))

        # Resident-path index tensors are constant across the request, so uploaded once here. The
        # precomputed path refreshes the same two buffers per step in the loop instead.
        if adaln_cache is None:
            self._tt_adaln.update(
                self._row_indices(adaln_indices(layout.token_tags, row_slot), padded_len), traced=traced
            )
            self._tt_tsi.update(self._row_indices(row_slot, padded_len), traced=traced)

        # Target latents: uploaded once and advanced in place on device by the Euler step below, so
        # they stay resident across the loop -- no per-step re-upload, no host round-trip. Under
        # tracing `update` copies the fresh initial noise into the same buffer the capture read, which
        # both resets it for each replay and keeps the trace's input address valid.
        self._tt_video.update(
            video_rows[num_cond:].unsqueeze(0).unsqueeze(0),
            traced=traced,
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
        )
        self._tt_audio.update(
            audio_rows[num_cond_audio:].unsqueeze(0).unsqueeze(0),
            traced=traced,
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
        )

        t_preamble = time.time() - t_preamble
        t_first = t_steady = 0.0
        for i, t in enumerate(timesteps):
            t_step = time.time()
            # Every trace input is a persistent buffer hoisted above the loop; only the per-step
            # modulation input changes. The resident path refreshes just the per-slot noise levels;
            # the precomputed path refreshes the two index tensors into the same buffers.
            if adaln_cache is not None:
                # Precomputed path: the host-built table numbers levels per step in `step_timesteps`
                # order, so a row's absolute table row is found by matching its level *by value*.
                # `build_row_timesteps` numbers levels in its own order and the level count varies per
                # step (the conditioning floor collides with the video level early in the schedule and
                # separates later), so positional correspondence is not guaranteed. The per-row
                # indices are rebuilt each step into the resident buffers; `timestep` is unused here.
                unique, row_index = build_row_timesteps(
                    layout,
                    float(t),
                    float(audio_timesteps[i]),
                    max(float(t), MINIMAX_H3_KEYFRAME_NOISE_AUG),
                    MINIMAX_H3_AUDIO_CONDITION_TIMESTEP,
                )
                levels = self._adaln_table.step_timesteps(i)
                position = torch.tensor(
                    [int((levels == value).nonzero()[0, 0]) for value in unique], dtype=row_index.dtype
                )
                step_row_index = adaln_cache.step_offset(i) + position[row_index]
                self._tt_adaln.update(
                    self._row_indices(adaln_indices(layout.token_tags, step_row_index), padded_len), traced=traced
                )
                self._tt_tsi.update(self._row_indices(step_row_index, padded_len), traced=traced)
            else:
                # Resident-AdaLN path: the index tensors are constant, so only the per-slot noise
                # levels change per step. Each block projects `temb` for these levels on device (row
                # `slot * MODALITY_NUM + tag`), so a row indexes its slot directly -- no table offset,
                # nothing rebuilt from the host per step.
                levels = slot_levels(
                    slot_roles,
                    video_timestep=float(t),
                    audio_timestep=float(audio_timesteps[i]),
                    condition_video_timestep=max(float(t), MINIMAX_H3_KEYFRAME_NOISE_AUG),
                    condition_audio_timestep=MINIMAX_H3_AUDIO_CONDITION_TIMESTEP,
                )
                # Replicated, fp32 so the sinusoid is computed in fp32, and shaped [1, 1, num_slots, 1]
                # so it broadcasts against the frequency factor. `num_slots` is fixed for the request,
                # so the buffer's shape never changes and the trace can bake its address; only the
                # values are refreshed in place.
                self._tt_timestep.update(
                    levels.reshape(1, 1, -1, 1), traced=traced, dtype=ttnn.float32, device=self.mesh_device
                )

            # `None` timestep on the precomputed path -- its buffer is never populated.
            tt_timestep = self._tt_timestep.value if adaln_cache is None else None
            if traced:
                video_velocity, audio_velocity = transformer.traced_step(
                    video_1BVC=self._tt_video.value,
                    audio_1BAC=self._tt_audio.value,
                    prompt_1BLP=self._tt_prompt.value,
                    condition_blocks=tt_cond,
                    timestep=tt_timestep,
                    adaln_indices=self._tt_adaln.value,
                    timestep_indices=self._tt_tsi.value,
                    rope_cos=self._tt_rope_cos.value,
                    rope_sin=self._tt_rope_sin.value,
                    traced=True,
                )
            else:
                video_velocity, audio_velocity = transformer(
                    video_1BVC=self._tt_video.value,
                    audio_1BAC=self._tt_audio.value,
                    prompt_1BLP=self._tt_prompt.value,
                    condition_blocks=tt_cond,
                    timestep=tt_timestep,
                    adaln_indices=self._tt_adaln.value,
                    timestep_indices=self._tt_tsi.value,
                    rope_cos=self._tt_rope_cos.value,
                    rope_sin=self._tt_rope_sin.value,
                    adaln_cache=adaln_cache,
                )

            # On-device Euler, in place so the latents stay resident and the (traced) input buffers
            # are advanced directly: `next = sample + (sigma - sigma_next) * v`, the mirror of Flux2's
            # `multiply_`/`add_` step. `step_coefficient(i)` is the exact scalar `scheduler.step()`
            # applies -- see its derivation -- so this matches the host reference to the bf16 apply's
            # precision. Each stream steps its own schedule (shift 12.0 for video, 3.0 for audio), and
            # only the target rows are ever touched, so the keyframe anchors survive untouched.
            ttnn.multiply_(video_velocity, float(scheduler.step_coefficient(i)))
            ttnn.add_(self._tt_video.value, video_velocity)
            ttnn.multiply_(audio_velocity, float(audio_scheduler.step_coefficient(i)))
            ttnn.add_(self._tt_audio.value, audio_velocity)
            t_step = time.time() - t_step
            if i == 0:
                t_first = t_step
            else:
                t_steady += t_step
            if i % 10 == 0 or i == len(timesteps) - 1:
                self._log(f"  step {i + 1}/{len(timesteps)} t={float(t):.4f}")
            on_event(DenoiseStep(step=i + 1, total=len(timesteps), sigma=float(t)))

        # This request is now warm: a later call with the same signature may trace.
        self._trace_signature = signature
        self._trace_adaln_cache = adaln_cache
        steady_steps = max(len(timesteps) - 1, 1)
        self._log(
            f"denoise breakdown: preamble {t_preamble:.1f}s (rope {t_rope:.1f}s) | "
            f"first step {t_first:.1f}s | steady {t_steady:.1f}s over {steady_steps} steps "
            f"({t_steady / steady_steps * 1000:.0f} ms/step)"
        )

        # One read-back of the resident latents into the target region of the host rows. The
        # condition rows were never uploaded, so `[:num_cond]` stays pristine -- the return contract
        # (cond | target, cond first) and the decoders are unchanged, and the anchor check below is
        # then structural. The model holds only the target rows, so reshape to the row width.
        video_rows[num_cond:] = (
            local_device_to_torch(self._tt_video.value).reshape(-1, video_rows.shape[-1]).to(video_rows.dtype)
        )
        audio_rows[num_cond_audio:] = (
            local_device_to_torch(self._tt_audio.value).reshape(-1, audio_rows.shape[-1]).to(audio_rows.dtype)
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
        waveform = audio_decoder(latents)
        # The audio VAE is mono and took the two stereo channels as two batch items.
        return waveform.float().permute(1, 0, 2)
