# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Wan2.2-Animate-14B pipeline (character animation / character replacement) on TT hardware.

Proof of concept. Mirrors ``diffusers.WanAnimatePipeline`` (diffusers==0.38.0) segment by segment,
and reuses the tt_dit Wan 2.2 components unchanged:

  * ``TextEncoder`` (UMT5 on device), ``WanVAEDecoderAdapter`` and ``WanEncoder`` (VAE on device),
  * ``WanTransformer3DModel`` blocks, RoPE, embeddings and parallel configs (via
    ``WanAnimateTransformer3DModel``), ``WanPipelineConfig`` presets, the on-device UniPC solver.

New: the Animate transformer modules (``experimental/wan_animate``) and the segment loop below.
Host-side (CPU): CLIP ViT-H image encoder, the face motion encoder, image/mask/video preprocessing.

Deliberately not reused from ``WanPipeline``: its traced, DBCache-enabled, dual-expert denoising
loop. Animate has a single transformer and five extra per-segment inputs; the untraced loop here
keeps the POC small. Tracing it is follow-up work (see the design note).

Inputs are the outputs of the official preprocessing
(``wan/modules/animate/preprocess/preprocess_data.py``): pose video, face-crop video, reference
image and, for replacement, background and mask videos.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch
import tqdm
from diffusers.schedulers import UniPCMultistepScheduler
from loguru import logger
from PIL import Image

import ttnn

from models.tt_dit.models.vae.vae_wan2_1 import WanEncoder, WanVAEDecoderAdapter
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipelineConfig
from models.tt_dit.pipelines.wan.text_encoder import TextEncoder
from models.tt_dit.solvers import solver_for_scheduler
from models.tt_dit.utils import cache, tensor
from models.tt_dit.utils.conv3d import aligned_channels, conv3d_blocking_hash, conv_pad_height, conv_pad_width
from models.tt_dit.utils.tensor import bf16_tensor_2dshard, fast_device_to_host, float32_tensor, unflatten
from models.tt_dit.experimental.wan_animate import reference_math as rm
from models.tt_dit.experimental.wan_animate.transformer_wan_animate import WanAnimateTransformer3DModel, build_from_diffusers_config

DEFAULT_CHECKPOINT = "Wan-AI/Wan2.2-Animate-14B-Diffusers"
# Upstream default prompt (wan/configs/wan_animate_14B.py): "the person in the video is moving".
DEFAULT_PROMPT = "视频中的人在做动作"


def _resolve_file(checkpoint: str, relpath: str) -> str:
    if os.path.isdir(checkpoint):
        return os.path.join(checkpoint, relpath)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(checkpoint, relpath)


def load_transformer_state(checkpoint: str) -> dict[str, torch.Tensor]:
    """The Diffusers transformer state dict, read shard by shard (bf16 as stored; no torch model build)."""
    from safetensors.torch import load_file

    index = json.load(open(_resolve_file(checkpoint, "transformer/diffusion_pytorch_model.safetensors.index.json")))
    state: dict[str, torch.Tensor] = {}
    for shard in sorted(set(index["weight_map"].values())):
        state.update(load_file(_resolve_file(checkpoint, f"transformer/{shard}")))
    return state


def load_relighting_lora(path: str) -> dict[str, torch.Tensor]:
    """``relighting_lora/adapter_model.safetensors`` (PEFT) or the official ``relighting_lora.ckpt``."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)
    ckpt = torch.load(path, map_location="cpu")
    return ckpt.get("state_dict", ckpt)


def letterbox(image: Image.Image, width: int, height: int) -> Image.Image:
    """Official ``padding_resize`` / Diffusers ``resize_mode="fill"`` with black: keep aspect, pad."""
    scale = min(width / image.width, height / image.height)
    w, h = max(1, round(image.width * scale)), max(1, round(image.height * scale))
    canvas = Image.new("RGB", (width, height))
    canvas.paste(image.convert("RGB").resize((w, h), Image.BILINEAR), ((width - w) // 2, (height - h) // 2))
    return canvas


def frames_to_tensor(frames: list[Image.Image], width: int, height: int, *, gray: bool = False) -> torch.Tensor:
    """List of PIL frames -> [1, C, T, H, W]; RGB in [-1, 1], or grayscale in [0, 1] for masks."""
    arrs = []
    for f in frames:
        f = f.convert("L" if gray else "RGB")
        if f.size != (width, height):
            f = f.resize((width, height), Image.NEAREST if gray else Image.BILINEAR)
        arrs.append(np.asarray(f, dtype=np.float32))
    x = torch.from_numpy(np.stack(arrs))
    if gray:
        return (x / 255.0)[None, None]
    return (x / 127.5 - 1.0).permute(3, 0, 1, 2)[None]


@dataclass
class AnimateTimings:
    sections: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def time(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.sections[name] = self.sections.get(name, 0.0) + time.perf_counter() - start

    def report(self) -> str:
        return "\n".join(f"  {k:28s} {v:9.1f} s" for k, v in self.sections.items())


class WanAnimatePipeline:
    """Wan2.2-Animate-14B on a TT mesh. One output size per pipeline instance (set at construction)."""

    def __init__(
        self,
        *,
        mesh_device: ttnn.MeshDevice,
        width: int = 832,
        height: int = 480,
        segment_frame_length: int = 77,
        checkpoint_name: str = DEFAULT_CHECKPOINT,
        relighting_lora_path: str | None = None,
        cfg_enabled: bool = False,
        config_overrides: dict | None = None,
        vae_encoder_t_chunk: int | None = None,
    ) -> None:
        timings = AnimateTimings()
        self.timings = timings
        if (segment_frame_length - 1) % 4 != 0:
            msg = f"segment_frame_length must be 4n+1, got {segment_frame_length}"
            raise ValueError(msg)
        self.width, self.height = width, height
        self.segment_frame_length = segment_frame_length
        self.cfg_enabled = cfg_enabled
        self.mesh_device = mesh_device

        config = WanPipelineConfig.default(
            mesh_shape=mesh_device.shape,
            model_type="i2v",
            checkpoint_name=checkpoint_name,
            height=height,
            width=width,
            num_frames=segment_frame_length,
            cfg_enabled=cfg_enabled,
            cache_config=None,
            **(config_overrides or {}),
        )
        self.config = config
        self.checkpoint_name = config.checkpoint_name
        self.parallel_config = config.dit_parallel_config
        self.vae_parallel_config = config.vae_parallel_config
        self.is_fsdp = config.is_fsdp

        self.dit_ccl_manager = CCLManager(mesh_device=mesh_device, num_links=config.num_links, topology=config.topology)
        self.vae_ccl_manager = CCLManager(
            mesh_device=mesh_device, num_links=config.num_links, topology=ttnn.Topology.Linear
        )

        with timings.time("init/text_encoder"):
            self._text_encoder = TextEncoder(
                checkpoint_name=self.checkpoint_name,
                device=mesh_device,
                ccl_manager=self.vae_ccl_manager,
                encoder_parallel_config=config.encoder_parallel_config,
                dit_parallel_config=self.parallel_config,
                max_sequence_length=config.max_sequence_length,
            )
            self._text_encoder.prepare()

        with timings.time("init/transformer"):
            from diffusers.models.transformers.transformer_wan_animate import (
                WanAnimateTransformer3DModel as TorchAnimateTransformer,
            )

            torch_config = TorchAnimateTransformer.load_config(self.checkpoint_name, subfolder="transformer")
            self.transformer: WanAnimateTransformer3DModel = build_from_diffusers_config(
                _AttrDict(torch_config),
                mesh_device=mesh_device,
                ccl_manager=self.dit_ccl_manager,
                parallel_config=self.parallel_config,
                is_fsdp=self.is_fsdp,
            )
            self._relighting_lora_path = relighting_lora_path
            subfolder = "transformer_relight" if relighting_lora_path else "transformer"
            cache.load_model(
                self.transformer,
                get_torch_state_dict=self._transformer_state,
                model_name=os.path.basename(self.checkpoint_name.rstrip("/")),
                subfolder=subfolder,
                parallel_config=self.parallel_config,
                mesh_shape=tuple(mesh_device.shape),
                mesh_device=mesh_device,
                is_fsdp=self.is_fsdp,
            )

        with timings.time("init/vae"):
            self._vae = WanVAEDecoderAdapter(
                checkpoint_name=self.checkpoint_name,
                parallel_config=self.vae_parallel_config,
                ccl_manager=self.vae_ccl_manager,
                height=height,
                width=width,
                num_frames=segment_frame_length,
                vae_t_chunk_size=config.vae_t_chunk_size,
                vae_dtype=config.vae_dtype,
                sdpa_t_fracture_w_only=config.sdpa_t_fracture_w_only,
            )
            self._vae.reload_weights()
            # Forward chunk 4 is the reference encode schedule (1 frame, then groups of 4) and divides
            # 76 = segment_frame_length - 1, so no frame is dropped. The I2V table's 16 would drop 12.
            self._enc_t_chunk = vae_encoder_t_chunk or int(os.environ.get("WAN_ANIMATE_VAE_ENC_CHUNK", "4"))
            if (segment_frame_length - 1) % self._enc_t_chunk != 0:
                msg = f"VAE encoder chunk {self._enc_t_chunk} must divide segment_frame_length - 1"
                raise ValueError(msg)
            vc = self._vae.config
            self._vae_encoder = WanEncoder(
                base_dim=vc.base_dim,
                in_channels=vc.in_channels,
                z_dim=vc.z_dim,
                dim_mult=vc.dim_mult,
                num_res_blocks=vc.num_res_blocks,
                attn_scales=vc.attn_scales,
                temperal_downsample=vc.temperal_downsample,
                is_residual=vc.is_residual,
                mesh_device=mesh_device,
                ccl_manager=self.vae_ccl_manager,
                parallel_config=self.vae_parallel_config,
                # 0/0/None selects the fallback conv3d blocking table, valid for any frame count.
                height=0,
                width=0,
                encoder_t_chunk_size=None,
                latents_mean=vc.latents_mean,
                latents_std=vc.latents_std,
            )
            blocking = conv3d_blocking_hash(self._vae_encoder)
            cache.load_model(
                self._vae_encoder,
                model_name=os.path.basename(self.checkpoint_name.rstrip("/")),
                subfolder=f"vae_encoder_{blocking}_lnorm" if blocking else "vae_encoder_lnorm",
                parallel_config=self.vae_parallel_config,
                mesh_shape=tuple(mesh_device.shape),
                mesh_device=mesh_device,
                get_torch_state_dict=lambda: self._vae.torch_state_dict(),
            )

        with timings.time("init/clip_and_scheduler"):
            from transformers import CLIPImageProcessor, CLIPVisionModel

            self._clip = CLIPVisionModel.from_pretrained(
                self.checkpoint_name, subfolder="image_encoder", torch_dtype=torch.float32
            ).eval()
            self._clip_processor = CLIPImageProcessor.from_pretrained(self.checkpoint_name, subfolder="image_processor")
            self._solver = solver_for_scheduler(
                UniPCMultistepScheduler.from_pretrained(self.checkpoint_name, subfolder="scheduler")
            )

        logger.info("WanAnimatePipeline ready:\n" + timings.report())

    # ------------------------------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------------------------------
    def _transformer_state(self) -> dict[str, torch.Tensor]:
        state = load_transformer_state(self.checkpoint_name)
        if self._relighting_lora_path:
            merged = rm.merge_relighting_lora(state, load_relighting_lora(self._relighting_lora_path))
            logger.info(f"merged relighting LoRA into {merged} linears")
        return state

    # ------------------------------------------------------------------------------------------
    # Host <-> device helpers
    # ------------------------------------------------------------------------------------------
    @torch.no_grad()
    def vae_encode(self, video: torch.Tensor) -> torch.Tensor:
        """[1, 3, T, H, W] in [-1, 1] -> normalized latents [1, 16, (T-1)//4+1, H/8, W/8] (argmax mode)."""
        pc = self.vae_parallel_config
        x = video.permute(0, 2, 3, 4, 1).contiguous()  # BTHWC
        x, logical_h = conv_pad_height(x, pc.height_parallel.factor * 8)
        x, logical_w = conv_pad_width(x, pc.width_parallel.factor * 8)
        tt_x = bf16_tensor_2dshard(
            x,
            self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            shard_mapping={pc.height_parallel.mesh_axis: 2, pc.width_parallel.mesh_axis: 3},
        )
        c_pad = aligned_channels(3)
        if c_pad != 3:
            padded = ttnn.pad(tt_x, [(0, 0), (0, 0), (0, 0), (0, 0), (0, c_pad - 3)], value=0.0)
            ttnn.deallocate(tt_x)
            tt_x = padded
        with self.vae_ccl_manager.transient_ping_pong_buffers():
            enc, new_h, new_w = self._vae_encoder(
                tt_x, logical_h, logical_w=logical_w, encoder_t_chunk_size=self._enc_t_chunk
            )
            ttnn.deallocate(tt_x)
            dims = [None, None]
            dims[pc.height_parallel.mesh_axis] = 3
            dims[pc.width_parallel.mesh_axis] = 4
            out = fast_device_to_host(enc, self.mesh_device, dims, ccl_manager=self.vae_ccl_manager)
            ttnn.deallocate(enc)
        return out[:, :, :, :new_h, :new_w].float()

    @torch.no_grad()
    def vae_decode(self, latents: torch.Tensor) -> torch.Tensor:
        """[1, 16, T, h, w] normalized latents -> [1, 3, 4T-3, H, W] in [-1, 1]."""
        with self.vae_ccl_manager.transient_ping_pong_buffers():
            video = self._vae.decode(latents, output_type="pt")
        return video.float().clamp(-1, 1)

    @torch.no_grad()
    def encode_clip(self, image: Image.Image) -> torch.Tensor:
        inputs = self._clip_processor(images=image, return_tensors="pt")
        return self._clip(**inputs, output_hidden_states=True).hidden_states[-2]

    def _model_input(self, latents_tt: ttnn.Tensor, cond_tt: ttnn.Tensor) -> ttnn.Tensor:
        """Per-patch concat [noise 16 | y 20 + pose 17] in (pF pH pW C) order, as WanPipelineI2V.get_model_input."""
        if latents_tt.dtype == ttnn.float32:
            latents_tt = ttnn.typecast(latents_tt, ttnn.bfloat16)
        z = self._vae.config.z_dim
        groups = latents_tt.shape[-1] // z
        x = ttnn.concat([unflatten(latents_tt, -1, (groups, -1)), unflatten(cond_tt, -1, (groups, -1))], dim=-1)
        return ttnn.reshape(x, (*tuple(latents_tt.shape)[:-1], -1))

    # ------------------------------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------------------------------
    @torch.no_grad()
    def __call__(
        self,
        *,
        image: Image.Image,
        pose_video: list[Image.Image],
        face_video: list[Image.Image],
        background_video: list[Image.Image] | None = None,
        mask_video: list[Image.Image] | None = None,
        mode: str = "animate",
        prompt: str = DEFAULT_PROMPT,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        guidance_scale: float = 1.0,
        flow_shift: float | None = None,
        prev_segment_conditioning_frames: int = 1,
        seed: int = 0,
        max_frames: int | None = None,
    ) -> np.ndarray:
        """Returns uint8 frames [T, H, W, 3], T = len(pose_video) (or ``max_frames``)."""
        if mode not in ("animate", "replace"):
            msg = f"mode must be 'animate' or 'replace', got {mode!r}"
            raise ValueError(msg)
        if mode == "replace" and (background_video is None or mask_video is None):
            msg = "replace mode needs background_video and mask_video"
            raise ValueError(msg)
        if mode == "replace" and not self._relighting_lora_path:
            logger.warning("replace mode without the relighting LoRA: lighting will not adapt to the scene")
        do_cfg = guidance_scale > 1.0
        if do_cfg and not self.cfg_enabled:
            msg = "guidance_scale > 1 needs the pipeline built with cfg_enabled=True"
            raise ValueError(msg)

        t = self.timings = AnimateTimings()
        W, H, L, P = self.width, self.height, self.segment_frame_length, prev_segment_conditioning_frames
        h, w = H // 8, W // 8
        tr = self.transformer
        pt, ph, pw = tr.patch_size
        latent_t = (L - 1) // 4 + 1
        num_frames = len(pose_video) if max_frames is None else min(max_frames, len(pose_video))
        num_segments, pad = rm.num_segments_and_padding(num_frames, L, P)
        logger.info(f"{num_frames} frames -> {num_segments} segment(s) of {L} (+{pad} reflect-padded)")

        # ---- once per call: text, CLIP, reference latent, preprocessed videos ------------------
        with t.time("text_encoder"):
            prompt_embeds, negative_embeds = self._text_encoder.encode_cfg(
                [prompt], [negative_prompt], cfg_enabled=do_cfg
            )
            prompt_1BLP = tr.prepare_text_conditioning(prompt_embeds)
            negative_1BLP = tr.prepare_text_conditioning(negative_embeds) if do_cfg else None
        with t.time("clip_image_encoder(host)"):
            image_1BLP = tr.prepare_image_conditioning(self.encode_clip(image))
        with t.time("frames_to_tensors(host)"):
            ref_pixels = frames_to_tensor([letterbox(image, W, H)], W, H)  # [1, 3, 1, H, W]
            target = num_frames + pad
            pose = frames_to_tensor(rm.reflect_pad_frames(pose_video[:num_frames], target), W, H)
            s = tr.motion_encoder_size
            face = frames_to_tensor(rm.reflect_pad_frames(face_video[:num_frames], target), s, s)
            if mode == "replace":
                bg = frames_to_tensor(rm.reflect_pad_frames(background_video[:num_frames], target), W, H)
                msk = frames_to_tensor(rm.reflect_pad_frames(mask_video[:num_frames], target), W, H, gray=True)
        with t.time("vae_encode"):
            ref_latent = self.vae_encode(ref_pixels)
            y_ref = torch.cat([rm.i2v_mask(1, h, w, 1), ref_latent], dim=1)  # [1, 20, 1, h, w]

        generator = torch.Generator().manual_seed(seed)
        sp_axis = self.parallel_config.sequence_parallel.mesh_axis
        rope_args = None
        out_segments = []
        out_frames = None
        start = 0
        for seg in range(num_segments):
            end = start + L
            logger.info(f"segment {seg + 1}/{num_segments}: frames [{start}, {end})")
            # ---- per-segment conditioning -------------------------------------------------------
            with t.time("vae_encode"):
                cond = rm.segment_conditioning(
                    encode=self.vae_encode,
                    y_ref=y_ref,
                    pose_segment=pose[:, :, start:end],
                    prev_frames=out_frames,
                    start=start,
                    mode=mode,
                    prev_len=P,
                    background_segment=bg[:, :, start:end] if mode == "replace" else None,
                    mask_segment=msk[:, :, start:end] if mode == "replace" else None,
                )  # [1, 20 + 17, latent_t + 1, h, w]

            with t.time("face_motion(host+device)"):
                motion_1BKD = tr.prepare_face_conditioning(face[:, :, start:end])
                motion_uncond_1BKD = tr.prepare_face_conditioning(face[:, :, start:end] * 0 - 1) if do_cfg else None

            latents = torch.randn(1, 16, latent_t + 1, h, w, generator=generator)
            lat_host, n = tr.preprocess_spatial_input_host(latents)
            cond_host, _ = tr.preprocess_spatial_input_host(cond)
            padded_n = lat_host.shape[2]
            latents_tt = tensor.from_torch(
                lat_host, device=self.mesh_device, mesh_axes=[None, None, sp_axis, None], dtype=ttnn.float32
            )
            cond_tt = tensor.from_torch(
                cond_host, device=self.mesh_device, mesh_axes=[None, None, sp_axis, None], dtype=ttnn.bfloat16
            )
            if rope_args is None:
                rope_cos, rope_sin, trans_mat = tr.get_rope_features(latents)
                rope_args = dict(rope_cos_1HND=rope_cos, rope_sin_1HND=rope_sin, trans_mat=trans_mat)
            face_mask, image_mask = tr.get_masks(
                num_frames=(latent_t + 1) // pt, tokens_per_frame=(h // ph) * (w // pw), padded_n=padded_n
            )
            step_kwargs = dict(
                image_prompt_1BLP=image_1BLP,
                face_mask=face_mask,
                image_mask=image_mask,
                N=n,
                gather_output=False,
                **rope_args,
            )

            # ---- denoise ------------------------------------------------------------------------
            self._solver.set_schedule(num_inference_steps, shift=flow_shift)
            guidance_tt = (
                float32_tensor(torch.tensor(guidance_scale).reshape(1, 1, 1, 1), device=self.mesh_device)
                if do_cfg
                else None
            )
            with t.time("denoise"):
                for i, step_t in enumerate(tqdm.tqdm(self._solver.timesteps, desc=f"segment {seg + 1}")):
                    timestep = float32_tensor(torch.full((1, 1, 1, 1), float(step_t)), device=self.mesh_device)
                    model_in = self._model_input(latents_tt, cond_tt)
                    v = tr.animate_inner_step(
                        spatial_1BNI=model_in,
                        prompt_1BLP=prompt_1BLP,
                        motion_1BKD=motion_1BKD,
                        timestep=timestep,
                        **step_kwargs,
                    )
                    if do_cfg:
                        v_uncond = tr.animate_inner_step(
                            spatial_1BNI=model_in,
                            prompt_1BLP=negative_1BLP,
                            motion_1BKD=motion_uncond_1BKD,
                            timestep=timestep,
                            **step_kwargs,
                        )
                        v = ttnn.lerp(v_uncond, v, guidance_tt)
                    latents_tt = self._solver.step(step=i, latent=latents_tt, velocity_pred=v)
                ttnn.synchronize_device(self.mesh_device)

            gathered = self.dit_ccl_manager.all_gather_persistent_buffer(latents_tt, dim=2, mesh_axis=sp_axis)
            latents = tr.postprocess_spatial_output_host(tensor.local_device_to_torch(gathered), latent_t + 1, h, w, n)
            if not torch.isfinite(latents).all():
                msg = f"non-finite latents after segment {seg + 1}; see the runbook's NaN section"
                raise FloatingPointError(msg)

            # ---- decode ---------------------------------------------------------------------------
            with t.time("vae_decode"):
                out_frames = self.vae_decode(latents[:, :, 1:])  # drop the reference latent
            out_segments.append(out_frames if start == 0 else out_frames[:, :, P:])
            start += L - P

        video = torch.cat(out_segments, dim=2)[:, :, :num_frames]  # [1, 3, T, H, W]
        logger.info("WanAnimatePipeline timings:\n" + t.report())
        return ((video[0].permute(1, 2, 3, 0) + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).numpy()


class _AttrDict(dict):
    """A Diffusers config dict with attribute access (what ``build_from_diffusers_config`` reads)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e
