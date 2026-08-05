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
the mesh (TP=4 x SP=8). Everything that decides *which* row gets which treatment is host-side and
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
disk read is not, which is why prompt embeddings are disk-cached.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image, ImageOps

import ttnn

from ...encoders.qwen3vl.loader_minimax_h3 import (
    MINIMAX_H3_TEXT_ENCODER_LAYER,
    build_minimax_h3_text_encoder,
    build_minimax_h3_vision_tower,
    load_minimax_h3_text_state_dict,
)
from ...encoders.qwen3vl.model_qwen3vl import create_rope_tensors, mrope_position_ids, vision_token_runs
from ...encoders.qwen3vl.vision_qwen3vl import vision_cu_seqlens
from ...models.audio_vae.minimax_h3.convert_minimax_h3_audio import convert_minimax_h3_audio_state_dict
from ...models.audio_vae.minimax_h3.decoder_minimax_h3_audio import MiniMaxH3AudioDecoder
from ...models.transformers.minimax_h3.attention_minimax_h3 import prepare_rope_tables
from ...models.transformers.minimax_h3.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from ...models.vae.minimax_h3.vae_minimax_h3 import DEFAULT_TILE_SIZE, MiniMaxH3Vae, MiniMaxH3VaeConfig
from ...parallel.config import DiTParallelConfig, EncoderParallelConfig, ParallelFactor, VAEParallelConfig
from ...parallel.manager import CCLManager
from ...utils import cache
from ...utils.conv3d import conv3d_blocking_hash
from ...utils.tensor import bf16_tensor, from_torch
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
    patchify_video_latents,
    prepare_keyframe_image,
    resolve_canvas_size,
    unpack_audio_tokens,
    unpatchify_video_tokens,
    video_latent_num_frames,
)
from .scheduler import MiniMaxH3Scheduler

# ImageNet statistics; the video VAE emits normalized RGB and the pipeline reverts it. Imported from
# `conditioning` rather than restated: the keyframe path normalizes *into* the VAE with these and the
# decode path reverts *out* of it with them, so two copies would make a drift a silent asymmetric
# bug -- encode with one, decode with the other -- rather than an obvious one.
MINIMAX_H3_PIXEL_MEAN = _MINIMAX_H3_PIXEL_MEAN
MINIMAX_H3_PIXEL_STD = _MINIMAX_H3_PIXEL_STD

# Read from the two scheduler_config.json files, which hold nothing else.
VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0

# Cache namespace under TT_DIT_CACHE_DIR. `utils.cache` keys each entry on this plus the subfolder,
# the parallel config, the mesh shape, the dtype and the FSDP flag.
MODEL_NAME = "minimax-h3"


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
    With `condition_latent_shapes=()` it draws nothing extra and advances the generator not at all, so
    `t2va` at a given seed is bit-identical to before this argument existed.
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
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def video_seconds(self) -> float:
        return self.num_frames / self.fps

    @property
    def audio_seconds(self) -> float:
        return self.audio.shape[-1] / self.sampling_rate


class MiniMaxH3Pipeline:
    """`t2va` and `fl2va` on a Blackhole mesh. Build with `create_pipeline`."""

    def __init__(
        self,
        *,
        mesh_device: ttnn.MeshDevice,
        weights_dir: str | os.PathLike,
        tp_axis: int = 0,
        sp_axis: int = 1,
        num_links: int = 2,
        topology: ttnn.Topology = ttnn.Topology.Ring,
        coresident: bool | None = None,
    ) -> None:
        self.mesh_device = mesh_device
        self.weights_dir = Path(weights_dir)
        self.tp_axis, self.sp_axis = tp_axis, sp_axis
        shape = tuple(mesh_device.shape)
        if tp_axis == sp_axis:
            msg = f"tp_axis and sp_axis must differ, both are {tp_axis}"
            raise ValueError(msg)
        self.tp_factor, self.sp_factor = shape[tp_axis], shape[sp_axis]
        # Resolved once here rather than read from the environment inside `_make_resident`, which runs
        # several times per generation. Also makes it settable by a test without touching os.environ.
        self.coresident = coresident if coresident is not None else os.environ.get("MINIMAX_H3_CORESIDENT", "1") == "1"

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

        self.transformer_config = self._read_config("transformer")
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
        self._vision_tower = None
        self._vision_config = None
        self._audio_decoder = None
        self._resident: str | None = None
        # The last call's (label, seconds) rows, as LTXPipeline exposes them, so a test can assert on
        # or report the breakdown without re-timing anything.
        self.last_timings: list[tuple[str, float]] = []
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
        tp_axis: int = 0,
        sp_axis: int = 1,
        num_links: int = 2,
        topology: ttnn.Topology = ttnn.Topology.Ring,
    ) -> "MiniMaxH3Pipeline":
        weights_dir = weights_dir or os.environ.get("MINIMAX_H3_DIFFUSERS_DIR")
        if not weights_dir:
            raise ValueError(
                "MiniMax-H3 weights directory not set: pass weights_dir=... or set MINIMAX_H3_DIFFUSERS_DIR "
                "to a diffusers snapshot holding transformer/, text_encoder/, vae/ and audio_vae/."
            )
        return cls(
            mesh_device=mesh_device,
            weights_dir=weights_dir,
            tp_axis=tp_axis,
            sp_axis=sp_axis,
            num_links=num_links,
            topology=topology,
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

    def _make_resident(self, stage: str) -> None:
        """Release whatever the previous stage held before the next one allocates.

        `MiniMaxH3Vae` keeps its lazily-built per-shape encoders and decoders in a plain dict rather
        than as registered child `Module`s, so `deallocate_weights()` does not reach them; they are
        dropped explicitly. It still holds its host state dict, so rebuilding is a re-upload rather
        than a re-read from disk.

        **The text encoder is always evicted** --- it is 50 GB on disk and runs once per prompt.

        **The DiT and the video VAE are kept co-resident**, which is measured, not assumed: on a 4x8
        Blackhole mesh they fit together with no allocation failure. Evicting between them was costing
        a per-shape decoder rebuild every generation and would cost a ~50 s DiT reload on the next
        one; keeping both took the VAE decode row from **17.6 s to 6.0 s** and the fully-warm total
        from **81.1 s to 69.1 s**. `MINIMAX_H3_CORESIDENT=0` restores the eviction for a mesh where
        they do not fit.
        """
        if self._resident == stage:
            return
        # One `elif` chain, and the branches are mutually exclusive by construction: each tests
        # `self._resident`, which holds exactly one value. It reads like a fallthrough of independent
        # release actions and is not one.
        coresident = self.coresident
        if self._resident == "text" and self._text_encoder is not None:
            logger.info("releasing the text encoder")
            self._text_encoder.deallocate_weights()
            self._text_encoder = None
        elif self._resident == "dit" and self._transformer is not None and not coresident:
            logger.info("releasing the transformer")
            self._transformer.deallocate_weights()
            self._transformer = None
        elif self._resident == "vae" and self._vae is not None and not coresident:
            logger.info("releasing the video VAE")
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

    def _prepare_vision_tower(self):
        if self._vision_tower is None:
            logger.info("building the Qwen3-VL vision tower (replicated)")
            self._vision_tower, self._vision_config = build_minimax_h3_vision_tower(
                self.weights_dir / "text_encoder", mesh_device=self.mesh_device
            )
        return self._vision_tower

    def _embed_cache_path(self, prompt: str, keyframes: Sequence[Image.Image] = ()) -> Path:
        """Disk cache path for one prompt presentation.

        The `t2va` key is byte-identical to what it was before keyframes existed, so an already
        populated cache stays valid and the two tasks' latency numbers stay comparable. An `fl2va` key
        folds in a digest of each **prepared** keyframe -- prepared, not source, because the same image
        on a different canvas yields a different vision-token grid and therefore different embeddings.
        """
        cache_dir = Path(os.environ.get("TT_DIT_CACHE_DIR") or Path.home() / ".cache/tt-dit") / "minimax-h3-embeddings"
        cache_dir.mkdir(parents=True, exist_ok=True)
        if keyframes:
            digests = [hashlib.md5(np.asarray(k.convert("RGB")).tobytes()).hexdigest() for k in keyframes]
            raw = "fl2va-device||" + prompt + "||" + "|".join(digests)
        else:
            raw = f"t2va-device||{prompt}"
        return cache_dir / f"{hashlib.md5(raw.encode()).hexdigest()}.device.pt"

    def encode_prompt(
        self, prompt: str, *, keyframes: Sequence[Image.Image] = (), use_cache: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prompt to `(prompt_embeds [1, L, 5120], text_token_tags [L])`, disk-cached.

        The presentation has no chat template and no special tokens. For `t2va` it is the verbatim
        prompt and every row is text-tagged. For `fl2va` each keyframe contributes
        `"<Picture i>: " + <|vision_start|> + N x <|image_pad|> + <|vision_end|>` ahead of the prompt,
        and the **whole vision block is video-tagged** -- that tag is what the DiT's AdaLN keys off, so
        text-tagging it would mis-modulate every one of those rows with no PCC signal anywhere.

        A cache hit skips a 50 GB weight read, which is the whole point of caching a tensor this small.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        cache_path = self._embed_cache_path(prompt, keyframes)
        if use_cache and cache_path.is_file():
            logger.info(f"prompt embeddings from cache: {cache_path}")
            embeds, tags = torch.load(cache_path, weights_only=False)
            return embeds, tags

        input_ids, tags, type_ids, pixel_values, grid_thw = self._build_presentation(prompt, keyframes)
        seq_len = input_ids.shape[1]
        logger.info(
            f"encoding {seq_len} presentation tokens on device"
            + ("" if not keyframes else f" ({int(type_ids.sum())} of them vision, {len(keyframes)} keyframe(s))")
        )

        self._make_resident("text")
        if self._text_encoder is None:
            # Built without weights, then loaded through the cache: a hit skips the 50 GB
            # safetensors read entirely, which is the whole cost of this stage.
            self._text_encoder, self._text_config = build_minimax_h3_text_encoder(
                self.weights_dir / "text_encoder",
                mesh_device=self.mesh_device,
                parallel_config=self.encoder_parallel_config,
                ccl_manager=self.ccl_manager,
                is_fsdp=True,
                load_weights=False,
            )
            cache.load_model(
                self._text_encoder,
                model_name=MODEL_NAME,
                subfolder="text_encoder",
                parallel_config=self.encoder_parallel_config,
                mesh_shape=tuple(self.mesh_device.shape),
                mesh_device=self.mesh_device,
                is_fsdp=True,
                get_torch_state_dict=lambda: load_minimax_h3_text_state_dict(
                    self.weights_dir / "text_encoder", num_layers=MINIMAX_H3_TEXT_ENCODER_LAYER
                ),
            )
        config = self._text_config

        # The vision tower, and the two ways its output enters the decoder. Run before the rope tables
        # so a tower failure surfaces before any decoder work.
        vision_kwargs = {}
        if keyframes:
            tower = self._prepare_vision_tower()
            vis_cos, vis_sin = tower.prepare_rope(grid_thw)
            merged, deepstack = tower.forward(
                bf16_tensor(pixel_values.float(), device=self.mesh_device),
                pos_embeds=bf16_tensor(tower.prepare_pos_embeds(grid_thw), device=self.mesh_device),
                rope=(
                    bf16_tensor(vis_cos, device=self.mesh_device),
                    bf16_tensor(vis_sin, device=self.mesh_device),
                ),
                # One block per image. `fl2va` never needs block-diagonal masking -- that is `ref2va`,
                # where a video contributes one block per frame.
                cu_seqlens=vision_cu_seqlens(grid_thw),
            )
            runs = vision_token_runs(input_ids, self.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
            expected_runs = len(keyframes)
            assert (
                len(runs) == expected_runs
            ), f"expected {expected_runs} vision run(s) in the presentation, found {runs}"
            # merged tokens REPLACE the `<|image_pad|>` row embeddings; deepstack features are ADDED to
            # those same rows after the first three decoder layers. Not interchangeable.
            vision_kwargs = {"vision_embeds": merged, "vision_runs": runs, "deepstack_embeds": deepstack}

        # With a vision run the three mRoPE axes diverge, so `mrope_interleaved` stops being a no-op
        # and the chunked section split is wrong. t2va keeps the default (shared `arange`) path, where
        # the two layouts are bit-identical -- measured, see amendment 74.
        rope_scaling = config["rope_scaling"]
        position_ids = None
        if keyframes:
            if not rope_scaling.get("mrope_interleaved"):
                raise ValueError("this checkpoint does not declare mrope_interleaved; the fl2va rope path assumes it")
            position_ids = mrope_position_ids(
                type_ids,
                image_grid_thw=grid_thw,
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
            interleaved=bool(keyframes),
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
        # Replicated across the mesh: read one replica rather than composing all 32 and discarding 31.
        embeds = ttnn.to_torch(ttnn.get_device_tensors(taps[0])[0]).float()

        if use_cache:
            torch.save((embeds, tags), cache_path)
            logger.info(f"cached prompt embeddings to {cache_path}")
        return embeds, tags

    # ------------------------------------------------------------------ denoiser

    def _prepare_transformer(self) -> MiniMaxH3Transformer3DModel:
        self._make_resident("dit")
        if self._transformer is not None:
            return self._transformer

        config = {k: v for k, v in self.transformer_config.items() if k not in ("rope_freq_dim", "rope_theta")}
        config["patch_size"] = tuple(config["patch_size"])

        logger.info(f"building the {config['num_layers']}-layer transformer, TP={self.tp_factor}/SP={self.sp_factor}")
        model = MiniMaxH3Transformer3DModel(
            **config,
            mesh_device=self.mesh_device,
            ccl_manager=self.ccl_manager,
            parallel_config=self.dit_parallel_config,
        )

        # Cache-aware: on a hit this reads pre-sharded device tensors instead of 62 GB of
        # safetensors plus every `_prepare_torch_state` fixup. The cache key already covers
        # parallel_config, mesh shape, dtype and FSDP, so a TP/SP change cannot read a stale cache.
        # With TT_DIT_CACHE_DIR unset it falls through to the direct load and logs that it did.
        # The load underneath is strict: 638 keys, and a single unmapped one is a real bug.
        cache.load_model(
            model,
            model_name=MODEL_NAME,
            subfolder="transformer",
            parallel_config=self.dit_parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            mesh_device=self.mesh_device,
            get_torch_state_dict=lambda: self._read_safetensors("transformer"),
        )
        self._transformer = model
        return model

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
    ) -> MiniMaxH3Vae:
        """Build the VAE and, given `decode_shape` / `encode_shape`, its per-shape sub-models too.

        These arguments are about measurement, not convenience. `MiniMaxH3Vae` builds a decoder **per
        distinct (T, H, W)** and uploads that decoder's ~4.6 GB of weights at construction --- which,
        before this argument existed, happened lazily inside `vae.decode()` and therefore landed
        *inside* the timed VAE-decode row. It is weight upload, i.e. one-time construction cost that
        the measurement contract does not count. Forcing it here moves it into the prepare, where the
        rest of the weight loading already is. `encode_shape` does the same for the keyframe encoder,
        which is far smaller (0.72 GB against the decoder's 9.7) but on the same principle.

        `load_encoder_state` is called whenever the encoder state is needed at all, and eagerly rather
        than lazily: `encode_clip` raises `RuntimeError("call load_encoder_state() before encoding")`,
        which is the right error but fires at encode time --- after the DiT has been built and inside a
        timed row. One `_read_safetensors("vae")` feeds both loaders; reading the 10.4 GB twice would be
        pure waste.
        """
        self._make_resident("vae")
        want_encoder = encode_shape is not None
        if self._vae is None:
            logger.info("building the video VAE")
            self._vae = MiniMaxH3Vae(self.vae_config, mesh_device=self.mesh_device, weight_loader=self._cache_submodel)
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
            logger.info(f"per-shape decoder {decode_shape} built in {time.time() - t0:.1f}s")
        if encode_shape is not None:
            # taps=1: a keyframe is one frame, and the causal front-pad is zeros, so a 3-tap temporal
            # conv collapses to `weight[:, :, -1]` exactly. Not an approximation -- see M8a.
            key = (*encode_shape, 1)
            if key not in self._vae._encoders:
                t0 = time.time()
                self._vae._encoder_for(*key)
                logger.info(f"per-shape keyframe encoder {encode_shape} built in {time.time() - t0:.1f}s")
        return self._vae

    def _encode_keyframes(self, vae: MiniMaxH3Vae, keyframes: Sequence[Image.Image]) -> torch.Tensor:
        """Prepared keyframes to packed conditioning rows, via the device VAE encoder.

        `encode_keyframes` takes the encoder as an injected callable, so the device VAE plugs straight
        in. `temporal_taps` is deliberately not passed: `encode_clip` auto-selects 1 for `T == 1`, and
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

        Takes no arguments, and that is the point --- the unit is fixed by the VAE's tiling
        *independently of resolution and duration*, which is what lets one per-shape decoder serve the
        whole video and what the cache key records. It previously accepted `num_latent_frames` and
        computed two values it then discarded; the return never depended on them.

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
            logger.info("building the audio decoder")
            decoder = MiniMaxH3AudioDecoder(
                latent_channels=config["latent_channels"],
                latent_dim=config["latent_dim"],
                decoder_dim=config["decoder_dim"],
                decoder_rates=tuple(config["decoder_rates"]),
                decoder_kernel_sizes=tuple(config["decoder_kernel_sizes"]),
                resblock_kernel_sizes=tuple(config["resblock_kernel_sizes"]),
                resblock_dilation_sizes=tuple(tuple(d) for d in config["resblock_dilation_sizes"]),
                mesh_device=self.mesh_device,
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
                subfolder="audio_decoder",
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

        `ndim` used to be a parameter; every call site passed `latents.ndim`, so it could only ever be
        wrong, and a mismatched value broadcasts silently rather than raising.
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
        num_frames: int = 124,
        aspect_ratio: tuple[float, float] = (16, 9),
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        seed: int = 0,
        use_prompt_cache: bool = True,
    ) -> MiniMaxH3Output:
        """`image` and/or `last_image` select `fl2va`; neither selects `t2va`.

        Note that `fl2va` at a given seed does **not** reproduce `t2va` at that seed, even with a
        keyframe that contributes nothing: the conditioning noise is the first draw off the request
        generator and shifts the video and audio streams behind it. That is the reference's order and
        it is correct, not a regression.
        """
        # (label, seconds) rows counted toward the total; **prepares and export excluded**, matching
        # `pipelines/ltx/pipeline_ltx_distilled.py`. Weight upload is one-time construction cost and
        # the measurement contract never counts it (`.claude/skills/README.md`), so every
        # `_prepare_*` happens outside a timed window. The one exception mirrors LTX exactly: a
        # prompt-cache *miss* loads the text encoder inside the Encoder row, which is why that row
        # carries a `(cache)` label when it was a hit.
        timings: list[tuple[str, float]] = []

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
        logger.info(
            f"{task} {width}x{height}, {num_frames} frames ({num_frames / MINIMAX_H3_FPS:.2f} s), "
            f"{num_latent_frames} latent frames, {num_audio_latents} audio latents, "
            f"{num_inference_steps} steps, anchors={keyframe_anchors or '()'}"
        )

        # 2. Text (plus the vision block, for fl2va).
        cached = use_prompt_cache and self._embed_cache_path(prompt, keyframes).is_file()
        t0 = time.time()
        prompt_embeds, text_token_tags = self.encode_prompt(prompt, keyframes=keyframes, use_cache=use_prompt_cache)
        t_encode = time.time() - t0
        timings.append(("Encoder (cache)" if cached else "Encoder", t_encode))
        logger.info(f"Encoding ({'cache' if cached else 'device'}): {t_encode:.1f}s")

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
            t0 = time.time()
            condition_rows = self._encode_keyframes(vae, keyframes)
            # `scheduler.scale_noise`, never a local copy: `conditioning.py` records that a local one
            # drifted 2.4e-7 by computing `1 - t` in Python double instead of the sample dtype, and
            # there is a test asserting no second implementation exists.
            condition_rows = scheduler.scale_noise(condition_rows, MINIMAX_H3_KEYFRAME_NOISE_AUG, condition_noise)
            t_keyframe = time.time() - t0
            timings.append(("Keyframe encode", t_keyframe))
            logger.info(f"Keyframe encode: {t_keyframe:.1f}s — {tuple(condition_rows.shape)}")

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

        # 6. Denoise. The weight load is a prepare, so it sits outside the timed row.
        transformer = self._prepare_transformer()
        t0 = time.time()
        video_rows, audio_rows = self._denoise(
            transformer, layout, prompt_embeds, video_rows, audio_rows, scheduler, audio_scheduler
        )
        t_denoise = time.time() - t0
        timings.append(("Denoise", t_denoise))
        logger.info(f"Denoise: {t_denoise:.1f}s — {num_inference_steps - 1} steps")

        # 7. Decode both modalities. Same rule: `_prepare_*` before `t0`, never inside it -- and for
        # the VAE that includes the per-shape decoder, whose weight upload would otherwise be timed.
        vae = self._prepare_vae(decode_shape=self.decode_unit_shape())
        t0 = time.time()
        video = self._decode_video(
            vae, video_rows, num_latent_frames, latent_height, latent_width, layout.num_condition_video_rows
        )
        t_vae_decode = time.time() - t0
        timings.append(("VAE decode", t_vae_decode))
        logger.info(f"VAE decode: {t_vae_decode:.1f}s — {tuple(video.shape)}")

        audio_decoder = self._prepare_audio_decoder()
        t0 = time.time()
        audio = self._decode_audio(audio_decoder, audio_rows, num_audio_latents)
        t_audio_decode = time.time() - t0
        timings.append(("Audio decode", t_audio_decode))
        logger.info(f"Audio decode: {t_audio_decode:.1f}s")

        self.last_timings = list(timings)
        total = sum(seconds for _, seconds in timings)
        logger.info(f"Total (compute): {total:.1f}s | frames={tuple(video.shape)}")

        return MiniMaxH3Output(
            video=video,
            audio=audio,
            sampling_rate=self.audio_sampling_rate,
            num_frames=video.shape[2],
            timings=dict(timings),
        )

    def warmup(
        self,
        *,
        prompt: str = "warmup",
        image: Image.Image | None = None,
        last_image: Image.Image | None = None,
        num_frames: int = 124,
        height: int | None = None,
        width: int | None = None,
        aspect_ratio: tuple[float, float] = (16, 9),
        num_inference_steps: int = 50,
    ) -> None:
        """Compile and allocate everything a real call needs, so the next call measures compute only.

        The analogue of `LTXPipeline.warmup_buffers`. Runs one full generation at the target shape
        with `use_prompt_cache=False`, which is what forces the text encoder's own kernels to compile
        rather than being skipped by a cache hit. Every program, every per-shape conv3d blocking and
        every persistent buffer this working point touches is resident afterwards.

        "Fully warm" in every number this pipeline reports means *after* this.

        **Pass the real `prompt` and the real keyframes**, not the defaults. Every program in the
        50-block stack is keyed on the *padded* packed length, so warming a different one warms nothing.
        `t2va` got away with the one-token default purely by luck -- 1 and 39 tokens both round up to
        37888 -- and a keyframe's ~1010-row vision block ends that. `last_padded_len` is exposed so a
        caller can assert the warm and measured lengths agree rather than trusting them to.
        """
        t0 = time.time()
        task = "t2va" if image is None and last_image is None else "fl2va"
        logger.info(f"warmup ({task}): {num_frames}f, {num_inference_steps} steps, prompt {len(prompt)} chars")
        self(
            prompt,
            image=image,
            last_image=last_image,
            num_frames=num_frames,
            height=height,
            width=width,
            aspect_ratio=aspect_ratio,
            num_inference_steps=num_inference_steps,
            use_prompt_cache=False,
        )
        logger.info(f"warmup ({task}) done in {time.time() - t0:.1f}s, padded_len={self.last_padded_len}")

    def _denoise(
        self,
        transformer: MiniMaxH3Transformer3DModel,
        layout: MiniMaxH3PackedSequence,
        prompt_embeds: torch.Tensor,
        video_rows: torch.Tensor,
        audio_rows: torch.Tensor,
        scheduler: MiniMaxH3Scheduler,
        audio_scheduler: MiniMaxH3Scheduler,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Denoise in place. `video_rows` is `[condition rows | target rows]`, cond first, as the
        reference's `latents` is; `num_condition_video_rows` is 0 for `t2va`.
        """
        num_cond = layout.num_condition_video_rows
        num_cond_audio = layout.num_condition_audio_rows
        # Kept to assert the invariant at the end of the loop. `fl2va` is the first task for which the
        # write mask matters, nothing re-imposes the anchors, and an overwritten anchor still denoises
        # into a plausible video that merely ignores the keyframe -- so no output metric would catch it.
        anchor_rows = video_rows[:num_cond].clone() if num_cond else None
        alignment = self.sp_factor * ttnn.TILE_SIZE
        padded_len = ((layout.sequence_length + alignment - 1) // alignment) * alignment
        self.last_padded_len = padded_len
        logger.info(
            f"packed sequence {layout.sequence_length} -> {padded_len} padded, "
            f"{padded_len // self.sp_factor} rows/device, {num_cond} condition rows"
        )

        # Constant for the whole loop: the rotary tables and the text stream.
        rope_cos, rope_sin = self._device_metadata(layout, padded_len)
        # [1, L, 5120] -> [1, 1, L, 5120], replicated: the model projects and refines the text stream
        # before the packed sequence is fractured, so every device needs all of it.
        tt_prompt = bf16_tensor(prompt_embeds.reshape(1, 1, -1, prompt_embeds.shape[-1]), device=self.mesh_device)

        timesteps = scheduler.timesteps
        audio_timesteps = audio_scheduler.timesteps
        for i, t in enumerate(timesteps):
            # Per-row noise levels, reduced to the (distinct timesteps, per-row index) pair the
            # model addresses its AdaLN table through.
            unique, row_index = build_row_timesteps(
                layout,
                float(t),
                float(audio_timesteps[i]),
                max(float(t), MINIMAX_H3_KEYFRAME_NOISE_AUG),
                1.0,
            )
            # The condition rows travel as their own argument, not prepended to the video stream: the
            # packed layout puts them between text and audio, so the model places them there itself.
            tt_cond = (
                None
                if not num_cond
                else bf16_tensor(video_rows[:num_cond].unsqueeze(0).unsqueeze(0), device=self.mesh_device)
            )
            tt_video = bf16_tensor(video_rows[num_cond:].unsqueeze(0).unsqueeze(0), device=self.mesh_device)
            tt_audio = bf16_tensor(audio_rows[num_cond_audio:].unsqueeze(0).unsqueeze(0), device=self.mesh_device)
            # Replicated, fp32 so the sinusoid is computed in fp32, and shaped [1, 1, T, 1] so it
            # broadcasts against the frequency factor.
            tt_timestep = from_torch(unique.reshape(1, 1, -1, 1), device=self.mesh_device, dtype=ttnn.float32)
            tt_adaln = self._row_indices(adaln_indices(layout.token_tags, row_index), padded_len)
            tt_tsi = self._row_indices(row_index, padded_len)

            video_velocity, audio_velocity = transformer(
                video_1BVC=tt_video,
                audio_1BAC=tt_audio,
                prompt_1BLP=tt_prompt,
                condition_1BKC=tt_cond,
                timestep=tt_timestep,
                adaln_indices=tt_adaln,
                timestep_indices=tt_tsi,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )

            # Replicated after the model's SP gather: read one replica. The model returns the *target*
            # rows only, so reshape to the row width rather than to `video_rows.shape`, which still
            # counts the condition rows.
            v = ttnn.to_torch(ttnn.get_device_tensors(video_velocity)[0]).reshape(-1, video_rows.shape[-1]).float()
            a = ttnn.to_torch(ttnn.get_device_tensors(audio_velocity)[0]).reshape(-1, audio_rows.shape[-1]).float()

            # tt_dit's scheduler returns the next sample directly; only the diffusers one wraps it.
            # Each stream steps its own schedule -- shift 12.0 for video, 3.0 for audio.
            #
            # Only the generated rows are ever written, which is what makes the keyframe anchors
            # survive: nothing re-imposes them, they are simply never touched. `t2va` has no condition
            # rows, so `[0:]` is the whole tensor and this is bit-identical to writing it outright.
            video_rows[num_cond:] = scheduler.step(v, t, video_rows[num_cond:])
            audio_rows[num_cond_audio:] = audio_scheduler.step(a, audio_timesteps[i], audio_rows[num_cond_audio:])
            if i % 10 == 0 or i == len(timesteps) - 1:
                logger.info(f"  step {i + 1}/{len(timesteps)} t={float(t):.4f}")

        if anchor_rows is not None and not torch.equal(video_rows[:num_cond], anchor_rows):
            # RuntimeError, not AssertionError: this is a real failure of the loop, not a caller error,
            # and it must not be strippable by `python -O`.
            changed = int((video_rows[:num_cond] != anchor_rows).any(dim=-1).sum())
            msg = (
                f"{changed} of {num_cond} keyframe anchor rows changed during denoising; the loop's "
                "write mask is wrong and the keyframe conditioning is not being honoured"
            )
            raise RuntimeError(msg)

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
        video = vae.decode(latents)
        # The VAE emits ImageNet-normalized RGB.
        video = self._denormalize(video.float(), MINIMAX_H3_PIXEL_MEAN, MINIMAX_H3_PIXEL_STD).clamp(0, 1)
        return video

    def _decode_audio(
        self, audio_decoder: MiniMaxH3AudioDecoder, rows: torch.Tensor, num_audio_latents: int
    ) -> torch.Tensor:
        latents = unpack_audio_tokens(rows, num_audio_latents)
        latents = self._denormalize(latents, self.audio_config["latents_mean"], self.audio_config["latents_std"])
        waveform = audio_decoder(latents)
        # The audio VAE is mono and took the two stereo channels as two batch items.
        return waveform.float().permute(1, 0, 2)
