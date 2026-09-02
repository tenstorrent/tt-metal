# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan.py

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple
from urllib.parse import urlparse

import torch
import torchvision.transforms.functional as TF
from loguru import logger
from PIL import Image

import ttnn

from ...models.vae.vae_wan2_1 import WanEncoder
from ...utils import cache
from ...utils.conv3d import aligned_channels, conv3d_blocking_hash, conv_pad_height, conv_pad_width
from ...utils.tensor import bf16_tensor_2dshard, fast_device_to_host, unflatten
from .pipeline_wan import WanPipeline, WanPipelineConfig

if TYPE_CHECKING:
    from diffusers.schedulers import SchedulerMixin

_DEFAULT_I2V_CHECKPOINT = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

_CHECKPOINT_DIR_ENV = "WAN22_I2V_CHECKPOINT_DIR"

# encoder_t_chunk_size per mesh. None = full-T single pass, N = chunked with feat_cache;
# the two are numerically identical and full-T is faster. A mesh absent here uses full-T
# forward and the fallback conv3d blocking table.
_ENCODER_T_CHUNK_BY_MESH = {
    (4, 8): 16,
    (4, 32): 40,
}

# Truncated VAE encode: encode only the first I2V_ENCODE_FRAMES pixel frames and
# replicate the last latent to fill the remaining slots.
_I2V_ENCODE_FRAMES = int(os.environ.get("I2V_ENCODE_FRAMES", 81))


def _resolve_checkpoint(repo_id: str) -> str:
    """Prefer a local checkpoint directory over a hub repo id.

    All ranks share one HF cache on NFS; resolving by repo id makes each rank hit the hub
    and write cache metadata concurrently, and a rank that loses that race fails with a
    spurious missing-weights error. A directory path skips hub resolution.

    The directory must share the repo id's basename: the tt_dit cache namespace is
    os.path.basename(checkpoint_name), so a different basename orphans the compiled cache.
    Falls back to the repo id when unset, missing, or mismatched.
    """
    local = os.environ.get(_CHECKPOINT_DIR_ENV)
    if not local:
        return repo_id
    if not os.path.isdir(local):
        logger.warning(f"{_CHECKPOINT_DIR_ENV}={local!r} is not a directory; falling back to {repo_id}")
        return repo_id
    if os.path.basename(local.rstrip("/")) != os.path.basename(repo_id):
        logger.warning(
            f"{_CHECKPOINT_DIR_ENV}={local!r} basename does not match {repo_id!r}; "
            "using it would orphan the tt_dit cache. Falling back to the repo id."
        )
        return repo_id
    logger.info(f"Resolving {repo_id} from local checkpoint dir {local} (no hub lookup)")
    return local


class ImagePrompt(NamedTuple):
    image: Image.Image
    frame_pos: int


def _truncated_encode_frames(
    max_cond_pos: int,
    num_frames: int,
    *,
    encoder_t_chunk_size: int | None = None,
) -> int:
    """Pixel frames to encode, extended to cover the last conditioned frame.

    Rounded up to 4n+1 for temporal downsampling. When forward chunking is active, also
    aligned to 1 + k * encoder_t_chunk_size so WanEncoder.forward does not drop frames.
    """
    frames = max(_I2V_ENCODE_FRAMES, max_cond_pos + 1)
    step = encoder_t_chunk_size if encoder_t_chunk_size is not None else 4
    frames = (frames - 1 + step - 1) // step * step + 1
    return min(frames, num_frames)


def _load_pil_image(src) -> Image.Image:
    """Load a conditioning image from a PIL image, a local path, or an http(s) URL.

    Always converted to RGB: the encode path assumes 3 channels, and a greyscale or
    RGBA input would otherwise reach the channel pad with the wrong count.
    """
    if src is None:
        raise ValueError("No image provided")
    if isinstance(src, Image.Image):
        return src.convert("RGB")
    if not isinstance(src, (str, Path)):
        raise TypeError(f"Unsupported image input {type(src).__name__}; expected PIL.Image, str, or Path")
    src = str(src)
    try:
        if urlparse(src).scheme in ("http", "https"):
            import io

            import requests

            resp = requests.get(src, timeout=30)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        return Image.open(src).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to load image from {src!r}: {e}") from e


def _parse_image_prompts(image_prompt, num_frames: int) -> list[ImagePrompt]:
    """Resolve conditioning inputs to loaded RGB images at non-negative frame positions.

    Negative positions index from the end, so -1 is the last frame. The encode paths test
    ``i in cond_by_pos`` while walking forward from frame 0, so an unresolved negative
    position would silently drop the conditioning frame instead of failing, and would also
    skew the truncated encode's frame count.
    """
    if isinstance(image_prompt, ImagePrompt):
        image_prompt = [image_prompt]
    elif isinstance(image_prompt, (Image.Image, str, Path)):
        image_prompt = [ImagePrompt(image=image_prompt, frame_pos=0)]
    if not image_prompt:
        raise ValueError("I2V requires at least one conditioning image")

    resolved: list[ImagePrompt] = []
    seen: set[int] = set()
    for image, frame_pos in image_prompt:
        pos = frame_pos if frame_pos >= 0 else num_frames + frame_pos
        if not 0 <= pos < num_frames:
            raise ValueError(f"frame_pos {frame_pos} resolves to {pos}, outside [0, {num_frames})")
        if pos in seen:
            raise ValueError(f"Duplicate frame_pos {pos}")
        seen.add(pos)
        resolved.append(ImagePrompt(image=_load_pil_image(image), frame_pos=pos))
    return resolved


class WanPipelineI2V(WanPipeline):
    @classmethod
    def _config_overrides(cls) -> dict[str, object]:
        return {
            **super()._config_overrides(),
            "model_type": "i2v",
            "checkpoint_name": _resolve_checkpoint(_DEFAULT_I2V_CHECKPOINT),
        }

    def __init__(
        self,
        *,
        device: ttnn.MeshDevice,
        config: WanPipelineConfig,
        scheduler: SchedulerMixin | None = None,
        run_warmup: bool = True,
        lora_enabled: bool = False,
    ) -> None:
        # initialize without warmup; we warm up below with a sample image_prompt.
        super().__init__(device=device, config=config, scheduler=scheduler, run_warmup=False, lora_enabled=lora_enabled)

        enc_height, enc_width, enc_t_chunk = self._vae_encoder_build_dims()
        self.tt_vae_encoder = WanEncoder(
            base_dim=self._vae.config.base_dim,
            in_channels=self._vae.config.in_channels,
            z_dim=self._vae.config.z_dim,
            dim_mult=self._vae.config.dim_mult,
            num_res_blocks=self._vae.config.num_res_blocks,
            attn_scales=self._vae.config.attn_scales,
            temperal_downsample=self._vae.config.temperal_downsample,
            is_residual=self._vae.config.is_residual,
            mesh_device=self.mesh_device,
            ccl_manager=self.vae_ccl_manager,
            parallel_config=self.vae_parallel_config,
            height=enc_height,
            width=enc_width,
            encoder_t_chunk_size=enc_t_chunk,
            latents_mean=self._vae.config.latents_mean,
            latents_std=self._vae.config.latents_std,
        )

        # C_in_block decides how prepare_conv3d_weights reshapes the stored weights, so a
        # cache written for one blocking is unusable by another. The _lnorm suffix marks
        # weights with the latent normalization folded into quant_conv, which are likewise
        # unusable by a build that normalizes on the host.
        subfolder = "vae_encoder_lnorm"
        if enc_t_chunk is not None:
            blocking_hash = conv3d_blocking_hash(self.tt_vae_encoder)
            if blocking_hash:
                subfolder = f"vae_encoder_{blocking_hash}_lnorm"

        cache.load_model(
            self.tt_vae_encoder,
            model_name=os.path.basename(self.checkpoint_name),
            subfolder=subfolder,
            parallel_config=self.vae_parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            mesh_device=self.mesh_device,
            get_torch_state_dict=lambda: self._vae.torch_state_dict(),
        )

        if run_warmup:
            self._warmup()

    def _warmup(self) -> None:
        """Allocate buffers with a sample image_prompt sized to the target resolution.

        Runs a real ``__call__``, so it reaches ``prepare_latents``. A subclass whose
        ``prepare_latents`` needs state set in its own ``__init__`` must therefore pass
        ``run_warmup=False`` and call this once that state exists.
        """
        self(
            prompts=["warmup"],
            image_prompt=Image.new("RGB", (self._width, self._height)),
            num_inference_steps=2,
            guidance_scale=2 if self._cfg_enabled else 1,
            guidance_scale_2=2 if self._cfg_enabled else 1,
        )

    def _vae_encoder_build_dims(self) -> tuple[int, int, int | None]:
        """Dims that select the image encoder's conv3d blockings, applied at construction.

        The chunk size here is the frame count each conv3d call will actually see, which is
        independent of the ``encoder_t_chunk_size`` ``prepare_latents`` passes to
        ``WanEncoder.forward``: a forward pass chunked at 16 sees 16, an unchunked pass over
        the truncated encode sees all ``_I2V_ENCODE_FRAMES`` of it. Returning ``T=None``
        yields zero stage dims, i.e. the fallback blocking table.

        A late ``frame_pos`` can push the encode past ``_I2V_ENCODE_FRAMES``, which makes the
        unchunked meshes recompile; ``prepare_latents`` logs when that happens.
        """
        mesh_shape = tuple(self.mesh_device.shape)
        if mesh_shape not in _ENCODER_T_CHUNK_BY_MESH:
            logger.info(f"Image encoder: fallback conv3d blockings (no entry for mesh {mesh_shape})")
            return 0, 0, None
        chunk = min((_ENCODER_T_CHUNK_BY_MESH[mesh_shape] or _I2V_ENCODE_FRAMES), _I2V_ENCODE_FRAMES)
        logger.info(f"Image encoder: conv3d blockings for {self._width}x{self._height} T={chunk}")
        return self._height, self._width, chunk

    def get_model_input(self, latents, cond_latents):
        """
        Adapter function to enable I2V. For base T2V, just return the latents.
        """
        latents = super().get_model_input(latents, None)
        z_dim = self._vae.config.z_dim
        t_size = latents.shape[-1]
        model_input = ttnn.concat(
            [unflatten(latents, -1, (t_size // z_dim, -1)), unflatten(cond_latents, -1, (t_size // z_dim, -1))],
            dim=-1,
        )
        return ttnn.reshape(model_input, (*tuple(latents.shape)[:-1], -1))

    def _preprocess_frame(self, image: Image.Image, height: int, width: int, device) -> torch.Tensor:
        """Conditioning image to a (B,C,H,W) float32 tensor, nominally in [-1,1]."""
        img = TF.to_tensor(image).sub_(0.5).div_(0.5)
        img = torch.nn.functional.interpolate(img[None], size=(height, width), mode="bicubic", antialias=True)
        return img.to(device) if device is not None else img

    def prepare_latents(
        self,
        batch_size: int,
        image_prompt,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        """Noise latents plus the encoded image conditioning, as ``(latents, y)``.

        Materializing the whole conditioning video on the host costs 896 MB of float32 at
        81x720x1280, though every frame but the conditioned one is zero. So only the real
        frames are uploaded and written into a zeroed device buffer; the encode is truncated to
        ``_I2V_ENCODE_FRAMES`` and the tail latents are replicated on the host; and channel
        padding happens on device, so each frame crosses PCIe with 3 channels instead of 32.
        The encoder's conv3d blockings are picked to match (see ``_vae_encoder_build_dims``).
        """
        assert batch_size == 1, "Only batch size 1 is currently supported for I2V"

        image_prompt = _parse_image_prompts(image_prompt, num_frames)

        latents, _ = super().prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=dtype,
            device=device,
        )
        latent_shape = latents.shape
        h_lat, w_lat = latent_shape[-2], latent_shape[-1]

        in_channels = 3
        padded_channels = aligned_channels(in_channels)

        # ---- host: preprocess + pad ONLY the conditioned frames --------------
        # Channels stay at 3 through the transfer and are padded on device below.
        cond_by_pos: dict[int, ttnn.Tensor] = {}
        logical_h = None
        logical_w = None
        for image, frame_pos in image_prompt:
            img = self._preprocess_frame(image, height, width, device)
            # (B,C,H,W) -> (B,T=1,H,W,C)
            frame_BTHWC = img.unsqueeze(2).permute(0, 2, 3, 4, 1)
            frame_BTHWC, logical_h = conv_pad_height(
                frame_BTHWC, self.vae_parallel_config.height_parallel.factor * self.vae_scale_factor_spatial
            )
            frame_BTHWC, logical_w = conv_pad_width(
                frame_BTHWC, self.vae_parallel_config.width_parallel.factor * self.vae_scale_factor_spatial
            )
            tt_frame = bf16_tensor_2dshard(
                frame_BTHWC,
                self.mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                shard_mapping={
                    self.vae_parallel_config.height_parallel.mesh_axis: 2,
                    self.vae_parallel_config.width_parallel.mesh_axis: 3,
                },
            )
            if padded_channels != in_channels:
                tt_padded = ttnn.pad(
                    tt_frame,
                    [(0, 0), (0, 0), (0, 0), (0, 0), (0, padded_channels - in_channels)],
                    value=0.0,
                )
                ttnn.deallocate(tt_frame)
                tt_frame = tt_padded
            cond_by_pos[frame_pos] = tt_frame

        chunk = _ENCODER_T_CHUNK_BY_MESH.get(tuple(self.mesh_device.shape))
        encode_frames = _truncated_encode_frames(max(cond_by_pos), num_frames, encoder_t_chunk_size=chunk)

        # ---- device: one zeroed video, conditioned frames written in place ---
        # Assembling by concat instead would hold the pieces and the assembled result live at
        # once, so it cannot cost less than twice the video. slice_write takes rank 4 and
        # batch size is 1, so the buffer carries no batch axis and the write offset lands on T.
        _, _, h_pad, w_pad, c_pad = next(iter(cond_by_pos.values())).shape
        tt_video_THWC = ttnn.zeros(
            (encode_frames, h_pad, w_pad, c_pad),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
        )
        for pos, tt_frame in cond_by_pos.items():
            ttnn.experimental.slice_write(
                ttnn.squeeze(tt_frame, 0),
                tt_video_THWC,
                [pos, 0, 0, 0],
                [pos + 1, h_pad, w_pad, c_pad],
                [1, 1, 1, 1],
            )
            ttnn.deallocate(tt_frame)

        # Restore the batch axis for the encoder. Free: row major with an unchanged last
        # dimension reshapes to a view over the same buffer rather than a copy.
        tt_video_BTHWC = ttnn.unsqueeze(tt_video_THWC, 0)

        pc_before = self.mesh_device.num_program_cache_entries()
        encoded_BCTHW, new_logical_h, new_logical_w = self.tt_vae_encoder(
            tt_video_BTHWC, logical_h, logical_w=logical_w, encoder_t_chunk_size=chunk
        )
        pc_after = self.mesh_device.num_program_cache_entries()
        if pc_after != pc_before:
            # Steady state is zero: a non-zero delta after warmup means the encode shape
            # changed (e.g. a late frame_pos moved encode_frames) and programs recompiled.
            logger.info(
                f"VAE encode compiled {pc_after - pc_before} new program(s) "
                f"(encode_frames={encode_frames}, forward chunk={chunk})"
            )

        ttnn.deallocate(tt_video_BTHWC)

        # ---- device -> host --------------------------------------------------
        concat_dims = [None, None]
        concat_dims[self.vae_parallel_config.height_parallel.mesh_axis] = 3
        concat_dims[self.vae_parallel_config.width_parallel.mesh_axis] = 4
        encoded = fast_device_to_host(encoded_BCTHW, self.mesh_device, concat_dims, ccl_manager=self.vae_ccl_manager)
        ttnn.deallocate(encoded_BCTHW)
        ttnn.synchronize_device(self.mesh_device)

        # ---- host: replicate + mask ------------------------------------------
        # Same crop convention as prepare_latents; already normalized by latents_mean/std,
        # since the encoder's quant_conv absorbed them.
        encoded = encoded[:, :, :, :new_logical_h, :new_logical_w].to(dtype=dtype)

        # The truncated encode returns fewer latent frames than the transformer expects;
        # the dropped ones encode all-zero pixels, so they repeat the last computed latent.
        f_lat_full = latent_shape[2]
        n_lat = encoded.shape[2]
        if n_lat < f_lat_full:
            encoded = torch.cat([encoded, encoded[:, :, -1:, :, :].expand(-1, -1, f_lat_full - n_lat, -1, -1)], dim=2)
        elif n_lat > f_lat_full:
            encoded = encoded[:, :, :f_lat_full, :, :]

        msk = torch.zeros(batch_size, num_frames, h_lat, w_lat)
        for pos in cond_by_pos:
            msk[:, pos, :, :] = 1
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, h_lat, w_lat)
        msk = msk.transpose(1, 2)

        y = torch.cat([msk, encoded], dim=1)
        return latents, y
