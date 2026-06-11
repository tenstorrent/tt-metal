# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Wan2.2 lightx2v distilled I2V pipeline.

A thin subclass of :class:`WanPipelineI2V` that:

1. Uses ``Wan-AI/Wan2.2-I2V-A14B-Diffusers`` for tokenizer, text encoder, VAE,
   and scheduler — the lightx2v repo ships only the DiT.
2. Replaces both expert transformer state dicts with lightx2v's flat
   ``.safetensors`` files (4-step distill, high-noise + low-noise pair).
3. Defaults ``boundary_ratio=0.5`` (2 high-noise + 2 low-noise steps).

Test/caller passes ``num_inference_steps=4`` and ``guidance_scale=1.0`` for both
stages since CFG is baked into the distill.

A ``random_weights`` mode is provided for smoke-testing without HuggingFace
downloads. It (a) skips the two ~28 GB transformer subfolder downloads by
constructing config-only random ``TorchWanTransformer3DModel`` instances and
(b) feeds those random ``state_dict()``s into the TT model in place of the
lightx2v safetensors. Tokenizer, text encoder, VAE, and scheduler still come
from the base diffusers repo (~12 GB total).
"""
from __future__ import annotations

import contextlib
import os

import torch

import ttnn
from models.tt_dit.experimental.utils.lightx2v_loader import load_lightx2v_state_dict
from models.tt_dit.models.transformers.wan2_2.transformer_wan import TorchWanTransformer3DModel
from models.tt_dit.models.vae.vae_wan2_1 import WanEncoder
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipelineConfig
from models.tt_dit.pipelines.wan.pipeline_wan_i2v import WanPipelineI2V
from models.tt_dit.utils import cache
from models.tt_dit.utils.conv3d import (
    conv3d_blocking_hash,
    conv_pad_height,
    conv_pad_in_channels,
    conv_pad_width,
    set_force_t_out_block_1,
)
from models.tt_dit.utils.tensor import bf16_tensor_2dshard

# Hard-coded config for Wan2.2-I2V-A14B-Diffusers transformer subfolders. Used
# only in random_weights mode so we don't have to fetch transformer/config.json
# from HF. Both transformer and transformer_2 share this architecture; only
# weights differ between high-noise and low-noise experts.
_RANDOM_I2V_TRANSFORMER_CONFIG = dict(
    patch_size=(1, 2, 2),
    num_attention_heads=40,
    attention_head_dim=128,
    in_channels=36,  # I2V: 4 mask + 16 image latent + 16 noise latent
    out_channels=16,
    text_dim=4096,
    freq_dim=256,
    ffn_dim=13824,
    num_layers=40,
    cross_attn_norm=True,
    qk_norm="rms_norm_across_heads",
    eps=1e-6,
    rope_max_seq_len=1024,
    image_dim=None,
    added_kv_proj_dim=None,
    pos_embed_seq_len=None,
)


@contextlib.contextmanager
def _patch_torch_transformer_random(seed: int = 0):
    """Replace ``TorchWanTransformer3DModel.from_pretrained`` with a stub that
    instantiates from a hard-coded I2V config and never touches the network or
    disk. Restored on exit."""
    cls = TorchWanTransformer3DModel
    sentinel = object()
    saved = cls.__dict__.get("from_pretrained", sentinel)

    def _stub(_cls, *args, **kwargs):
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            return _cls(**_RANDOM_I2V_TRANSFORMER_CONFIG)

    cls.from_pretrained = classmethod(_stub)
    try:
        yield
    finally:
        if saved is sentinel:
            # `from_pretrained` was inherited; delete our override to restore inheritance.
            try:
                delattr(cls, "from_pretrained")
            except AttributeError:
                pass
        else:
            cls.from_pretrained = saved


class WanDistillPipelineI2V(WanPipelineI2V):
    LIGHTX2V_REPO = "lightx2v/Wan2.2-Distill-Models"
    HIGH_NOISE_FILE = "wan2.2_i2v_A14b_high_noise_lightx2v_4step.safetensors"
    LOW_NOISE_FILE = "wan2.2_i2v_A14b_low_noise_lightx2v_4step.safetensors"
    BASE_DIFFUSERS_REPO = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    CACHE_NAMESPACE = "Wan2.2-Distill-lightx2v-4step"
    RANDOM_CACHE_NAMESPACE = "Wan2.2-Distill-random"
    DISTILL_BOUNDARY_RATIO = 0.5

    # Truncated VAE image-encode: only encode the first ~33 pixel frames. For
    # I2V the conditioning image sits at frame 0 and all later frames are zeros,
    # which the causal Wan VAE encoder maps to a steady-state latent; encoding 33
    # frames (-> 9 latent frames) and replicating the last latent to fill the
    # rest is bit-equivalent to encoding all 81. Crucially, 33 frames also lands
    # on the SWEPT conv3d blocking entries in utils/conv3d.py (4x32/T=33, 4x8/T=16)
    # instead of the slow _DEFAULT_BLOCKINGS fallback the H/W=0 build falls into.
    DISTILL_ENCODE_FRAMES = 33
    # Forward chunk size per mesh shape; must match a swept (mesh, T) table.
    _ENCODER_T_CHUNK_BY_MESH = {(4, 8): 16}

    def __init__(
        self,
        *,
        device: ttnn.MeshDevice,
        config: WanPipelineConfig,
        lightx2v_local_dir: str | None = None,
        allow_download: bool | None = None,
        random_weights: bool | None = None,
    ) -> None:
        if allow_download is None:
            allow_download = os.environ.get("TT_DIT_ALLOW_HF_DOWNLOAD") == "1"
        if lightx2v_local_dir is None:
            lightx2v_local_dir = os.environ.get("LIGHTX2V_LOCAL_DIR")
        if random_weights is None:
            random_weights = os.environ.get("TT_DIT_RANDOM_WEIGHTS") == "1"

        self._lightx2v_local_dir = lightx2v_local_dir
        self._allow_download = allow_download
        self._random_weights = random_weights

        ctx = _patch_torch_transformer_random() if random_weights else contextlib.nullcontext()
        with ctx:
            super().__init__(device=device, config=config)

        # Replace the base VAE encoder (built with H/W=0 -> slow default conv3d
        # blockings, ~6s) with one built at the real resolution + truncated-T
        # chunk size, so it keys the swept blockings and runs a single short pass.
        #
        # NOTE: the swept blockings currently DEGRADE the 4-step distill output
        # (duplicate-subject artifact), while the truncation alone is quality-safe.
        # Gate the swept-blocking encoder behind an env flag (default OFF) until
        # the swept-vs-reference fidelity is sorted out; truncation (_encode_frames_for)
        # still applies either way and is the bulk of the safe win.
        if os.environ.get("WAN_DISTILL_FAST_VAE_ENCODER", "0") == "1":
            self._install_fast_vae_encoder()

    def _encode_frames_for(self, num_frames: int, max_cond_pos: int) -> int:
        # Encode only the first ~33 pixel frames (but always enough to cover the
        # furthest conditioned frame); the rest are zeros -> steady-state latent,
        # replicated downstream by the base prepare_latents.
        return max(min(self.DISTILL_ENCODE_FRAMES, num_frames), min(max_cond_pos + 1, num_frames))

    def _build_fast_vae_encoder(self, *, force_t_out_block_1: bool = False) -> WanEncoder:
        """Build a VAE encoder at the real resolution + truncated-T chunk size so
        it keys the swept conv3d blockings (instead of the slow H/W=0 default).

        ``force_t_out_block_1`` caps every encoder conv's ``T_out_block`` at 1,
        which keeps the fast C/H/W blocking but avoids the temporal-blocking
        artifact the 4-step distill is sensitive to. The blocking cache key
        (``conv3d_blocking_hash``) only depends on ``C_in_block``, which the cap
        doesn't change, so prepared weights are shared with the un-capped build.
        """
        mesh_shape = tuple(self.mesh_device.shape)
        chunk = self._ENCODER_T_CHUNK_BY_MESH.get(mesh_shape, self.DISTILL_ENCODE_FRAMES)

        set_force_t_out_block_1(force_t_out_block_1)
        try:
            enc = WanEncoder(
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
                height=self._height,
                width=self._width,
                encoder_t_chunk_size=chunk,
            )
        finally:
            set_force_t_out_block_1(False)

        # Cache prepared weights under a blocking-specific subfolder so we don't
        # reuse the slow-blocking weights the base loaded under "vae_encoder".
        blocking_key = conv3d_blocking_hash(enc)
        subfolder = f"vae_encoder_{blocking_key}" if blocking_key else "vae_encoder"
        cache.load_model(
            enc,
            model_name=os.path.basename(self.checkpoint_name),
            subfolder=subfolder,
            parallel_config=self.vae_parallel_config,
            mesh_shape=mesh_shape,
            get_torch_state_dict=lambda: self._vae.torch_state_dict(),
        )
        return enc

    def _encode_image_condition(self, image_prompt, *, enc_frames, height, width, dtype, device):
        """On-device conditioning assembly (gated by ``WAN_DISTILL_ONDEVICE_COND``).

        The base path builds the full ``enc_frames`` pixel video on the host
        (mostly zeros for I2V) and ships it to the device — at 720p/33 frames
        that host->device transfer dominates the image-encode stage (~2.9s).
        Here we instead transfer only the conditioned frame(s) (~5.5MB each) and
        build the zero frames directly on-device via binary doubling, mirroring
        the Prodia I2V pipeline. Falls back to the base host-build path when the
        flag is unset.
        """
        if os.environ.get("WAN_DISTILL_ONDEVICE_COND", "0") != "1":
            return super()._encode_image_condition(
                image_prompt, enc_frames=enc_frames, height=height, width=width, dtype=dtype, device=device
            )

        shard_mapping = {
            self.vae_parallel_config.height_parallel.mesh_axis: 2,
            self.vae_parallel_config.width_parallel.mesh_axis: 3,
        }
        h_factor = self.vae_parallel_config.height_parallel.factor * self.vae_scale_factor_spatial
        w_factor = self.vae_parallel_config.width_parallel.factor * self.vae_scale_factor_spatial

        # Per conditioned frame: preprocess on host -> pad -> shard to device.
        cond_by_pos: dict[int, ttnn.Tensor] = {}
        logical_h = logical_w = None
        for image, frame_pos in image_prompt:
            img = self.video_processor.preprocess(image, height=height, width=width).to(
                torch.device("cpu"), dtype=torch.float32
            )  # [B, C, H, W]
            frame_BTHWC = img.unsqueeze(2).permute(0, 2, 3, 4, 1)  # [B, 1, H, W, C]
            frame_BTHWC = conv_pad_in_channels(frame_BTHWC)
            frame_BTHWC, logical_h = conv_pad_height(frame_BTHWC, h_factor)
            frame_BTHWC, logical_w = conv_pad_width(frame_BTHWC, w_factor)
            cond_by_pos[frame_pos] = bf16_tensor_2dshard(
                frame_BTHWC,
                self.mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                shard_mapping=shard_mapping,
            )

        # Assemble the enc_frames video on-device: conditioned frames at their
        # positions, zero frames everywhere else (built by doubling a single
        # zero frame so we never allocate/transfer the full pixel video).
        tt_zero_1: ttnn.Tensor | None = None

        def _zero_run(n: int) -> ttnn.Tensor:
            nonlocal tt_zero_1
            if tt_zero_1 is None:
                tt_zero_1 = ttnn.zeros_like(next(iter(cond_by_pos.values())))
            z = tt_zero_1
            built = 1
            while built * 2 <= n:
                z = ttnn.concat([z, z], dim=1)
                built *= 2
            if built < n:
                z = ttnn.concat([z, z[:, : n - built, :, :, :]], dim=1)
            return z

        segments: list[ttnn.Tensor] = []
        zero_start: int | None = None
        for i in range(enc_frames):
            if i in cond_by_pos:
                if zero_start is not None:
                    segments.append(_zero_run(i - zero_start))
                    zero_start = None
                segments.append(cond_by_pos[i])
            elif zero_start is None:
                zero_start = i
        if zero_start is not None:
            segments.append(_zero_run(enc_frames - zero_start))

        tt_video_BTHWC = segments[0] if len(segments) == 1 else ttnn.concat(segments, dim=1)

        encoded = self._vae_encode_to_torch(tt_video_BTHWC, logical_h, logical_w, dtype)

        # Free device tensors. When concat produced a fresh buffer the inputs are
        # safe to drop; guard the single-segment case where tt_video aliases an input.
        if tt_video_BTHWC is not None and len(segments) > 1:
            ttnn.deallocate(tt_video_BTHWC)
        for tt in cond_by_pos.values():
            if not (len(segments) == 1 and tt is segments[0]):
                ttnn.deallocate(tt)
        if tt_zero_1 is not None and not (len(segments) == 1 and tt_zero_1 is segments[0]):
            ttnn.deallocate(tt_zero_1)
        return encoded

    def _install_fast_vae_encoder(self) -> None:
        mesh_shape = tuple(self.mesh_device.shape)
        # Drives the forward chunking (base prepare_latents reads this).
        self._encoder_t_chunk_size = self._ENCODER_T_CHUNK_BY_MESH.get(mesh_shape, self.DISTILL_ENCODE_FRAMES)
        force = os.environ.get("WAN_DISTILL_ENCODER_T_OUT_1", "0") == "1"
        self.tt_vae_encoder = self._build_fast_vae_encoder(force_t_out_block_1=force)

    def prepare_text_conditioning(self, tt_model, prompt_embeds, buffer, traced=False):
        # When CFG is baked in (guidance_scale=1.0), encode_prompt returns
        # negative_prompt_embeds=None. The base loop still calls this for the
        # negative buffer; forwarding None into the text embedder hits a
        # NoneType.padded_shape in Linear. combined_step already short-circuits
        # on do_classifier_free_guidance=False, so leaving the buffer as-is is
        # safe.
        if prompt_embeds is None:
            return buffer
        return super().prepare_text_conditioning(tt_model, prompt_embeds, buffer, traced)

    def _prepare_transformer(self, idx: int):
        state = self.transformer_states[idx]
        if self._random_weights:
            # Use the (random) state_dict the WanCheckpoint loaded under the
            # _patch_torch_transformer_random monkey-patch. Cache under a
            # separate namespace so a real-weights run doesn't reuse it.
            cache.load_model(
                state.model,
                model_name=self.RANDOM_CACHE_NAMESPACE,
                subfolder=state.checkpoint.subfolder,
                parallel_config=self.parallel_config,
                mesh_shape=tuple(self.mesh_device.shape),
                is_fsdp=self.is_fsdp,
                get_torch_state_dict=lambda s=state: s.checkpoint.state_dict(),
            )
            return

        filename = self.HIGH_NOISE_FILE if idx == 0 else self.LOW_NOISE_FILE
        cache.load_model(
            state.model,
            model_name=self.CACHE_NAMESPACE,
            subfolder=state.checkpoint.subfolder,
            parallel_config=self.parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            is_fsdp=self.is_fsdp,
            get_torch_state_dict=lambda f=filename: load_lightx2v_state_dict(
                self.LIGHTX2V_REPO,
                f,
                allow_download=self._allow_download,
                local_dir=self._lightx2v_local_dir,
            ),
        )

    @classmethod
    def create_pipeline(
        cls,
        *,
        mesh_device: ttnn.MeshDevice,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_links: int | None = None,
        dynamic_load: bool | None = None,
        topology: ttnn.Topology | None = None,
        is_fsdp: bool | None = None,
    ) -> WanDistillPipelineI2V:
        config = WanPipelineConfig.default(
            mesh_shape=mesh_device.shape,
            checkpoint_name=cls.BASE_DIFFUSERS_REPO,
            height=height,
            width=width,
            num_frames=num_frames,
            num_links=num_links,
            topology=topology,
            dynamic_load=dynamic_load,
            is_fsdp=is_fsdp,
            boundary_ratio=cls.DISTILL_BOUNDARY_RATIO,
            model_type="i2v",
            # CFG is baked into the distill (guidance_scale=1.0), so the uncond
            # forward is wasted work that lerp() discards. Disabling CFG makes
            # combined_step skip it, halving the per-step forwards (output is
            # identical since lerp(uncond, cond, 1.0) == cond).
            cfg_enabled=False,
        )
        return cls(device=mesh_device, config=config)
