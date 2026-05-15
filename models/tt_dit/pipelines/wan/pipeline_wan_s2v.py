# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan.py
# and from https://github.com/Wan-Video/Wan2.2/blob/main/wan/speech2video.py

"""WAN 2.2 Speech-to-Video pipeline.

The production checkpoint ``Wan-AI/Wan2.2-S2V-14B`` is **not** published with
a Diffusers wrapper — the safetensors at the root are keyed by the reference
repo's native module names, the T5 text encoder is a raw `.pth` file, the VAE
is a raw `.pth` file, and there is no `scheduler/`, `tokenizer/`, or
`transformer/` subfolder. The parent :class:`WanPipeline` therefore cannot
load this checkpoint directly. This pipeline:

  * **Reuses** the Diffusers-style ``Wan-AI/Wan2.2-T2V-A14B-Diffusers`` repo
    for the UMT5 tokenizer + text encoder and the VAE decoder/encoder
    (these components are weight-compatible across T2V / I2V / S2V).
  * **Loads** the S2V DiT transformer weights from the native S2V snapshot
    via :func:`wan_s2v_loader.load_s2v_state_dict` and the name translator at
    :func:`wan_s2v_weight_map.translate_s2v_state_dict`.
  * **Loads** the wav2vec2-large-xlsr-53 audio encoder from the bundled copy
    inside the S2V snapshot.
  * **Builds** ``UniPCMultistepScheduler`` from scratch with the WAN default
    ``flow_shift`` (5.0 for 720p, 3.0 for 480p).

The denoise loop and end-to-end inference are inherited from
:meth:`WanPipeline.__call__`; ``WanS2VTransformer3DModel.prepare_cond_emb`` is
invoked once per clip from :meth:`prepare_latents` to build the conditioning
caches that ``inner_step`` consumes.

Latent-space rationale for the conditioning constants
-----------------------------------------------------
S2V conditions on a reference image, motion latents, and a pose video
(``cond_states``). All three are produced in **normalized latent space** —
the VAE encoder's mean output ``mu`` minus ``latents_mean`` divided by
``latents_std`` (matches ``wan/modules/vae2_1.py:WanVAE_.encode``). The
"no motion" default is *not* literal latent zeros but
``vae_encode(zeros_in_pixel_space)`` after normalization (~−3.0 mean). Pose
conditioning, when absent, is literal latent zeros (matches the reference's
``COND[0] * 0`` shortcut for the no-pose case). See
:meth:`WanPipelineS2V._encode_normalized` for the VAE encode + normalize
helper used by both the reference latent and the motion-latent paths.

Not supported (production HF path only)
---------------------------------------
Any non-default S2V option raises :class:`NotImplementedError`:
``pose_video`` (only zero pose conditioning), multi-clip
(``num_repeat > 1``), ``enable_tts``, ``init_first_frame=True``, and
``adain_mode != "attn_norm"``. AdaIN modulation itself is gated off until
the upstream ttnn ``binary_ng`` broadcast issue is resolved.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.video_processor import VideoProcessor
from loguru import logger
from PIL import Image
from transformers import AutoTokenizer, UMT5EncoderModel, Wav2Vec2Model, Wav2Vec2Processor

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler  # noqa: F401  (kept for parity with base)

from ...encoders.umt5.model_umt5 import UMT5Config, UMT5Encoder
from ...encoders.wav2vec2 import Wav2Vec2Config, Wav2Vec2Encoder
from ...encoders.wav2vec2.audio_preprocess import (
    S2V_VIDEO_RATE,
    WAV2VEC2_HZ,
    get_audio_embed_bucket_fps,
    linear_interpolation,
    load_audio_to_input_values,
)
from ...models.transformers.wan2_2.transformer_wan_s2v import WanS2VTransformer3DModel
from ...models.vae.vae_wan2_1 import WanDecoder, WanEncoder
from ...parallel.config import DiTParallelConfig, EncoderParallelConfig, ParallelFactor, VaeHWParallelConfig
from ...parallel.manager import CCLManager
from ...solvers import UniPCSolver
from ...utils import cache
from ...utils.conv3d import conv3d_blocking_hash, conv_pad_height, conv_pad_in_channels
from ...utils.tensor import bf16_tensor_2dshard, fast_device_to_host
from .pipeline_wan import TransformerState, WanPipeline
from .wan_s2v_loader import find_s2v_snapshot, load_s2v_config, load_s2v_state_dict
from .wan_s2v_weight_map import translate_s2v_state_dict

# Production S2V repo (native naming, safetensors at root).
_DEFAULT_S2V_CHECKPOINT = "Wan-AI/Wan2.2-S2V-14B"
# Companion checkpoint used for tokenizer + text encoder + VAE (these are
# weight-compatible across the WAN 2.2 family).
_DEFAULT_AUX_CHECKPOINT = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"


@dataclass
class _S2VTransformerState:
    """Single-stage analog of :class:`pipeline_wan.TransformerState`."""

    model: WanS2VTransformer3DModel
    guidance_scale: float = 5.0
    prompt_buffer: object = field(default=None)
    negative_prompt_buffer: object = field(default=None)


class WanPipelineS2V(WanPipeline):
    """Speech-to-video sibling of ``WanPipeline``.

    Construction does not chain through ``WanPipeline.__init__`` because
    that path is hard-wired to Diffusers-style two-stage loading. We
    duplicate the relevant construction inline so we can:

      * skip the (non-existent) ``transformer/`` and ``transformer_2/``
        subfolders on the S2V repo,
      * load the S2V DiT from a flat safetensors stack via the native loader,
      * build a single transformer slot rather than the two-stage T2V/I2V split.

    ``call()`` and the warmup machinery are inherited from the parent. The
    parent's denoise loop is structured for two-stage T2V — running S2V
    inference end-to-end requires further wiring (ref/motion/cond/audio
    threading through ``inner_step``) tracked under #12.
    """

    def __init__(
        self,
        mesh_device,
        parallel_config: DiTParallelConfig,
        vae_parallel_config: VaeHWParallelConfig,
        encoder_parallel_config: EncoderParallelConfig,
        num_links: int,
        *,
        checkpoint_name: str = _DEFAULT_S2V_CHECKPOINT,
        aux_checkpoint_name: str = _DEFAULT_AUX_CHECKPOINT,
        scheduler: UniPCMultistepScheduler | None = None,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        is_fsdp: bool = False,
        dynamic_load: bool = False,
        vae_dtype: ttnn.DataType = ttnn.bfloat16,
        vae_t_chunk_size: int | None = 1,
        sdpa_t_fracture_w_only: bool = False,
        height: int = 0,
        width: int = 0,
        num_frames: int = 81,
        run_warmup: bool = False,
        audio_inject_layers: tuple[int, ...] | None = None,
    ):
        # Initialize the abstract DiffusionPipeline base directly (the parent
        # __init__ pulls in Diffusers loaders we can't satisfy here).
        # ``WanPipeline`` extends ``DiffusionPipeline`` + ``WanLoraLoaderMixin``;
        # we go through Python's MRO via ``object.__init__`` since neither
        # needs construction beyond what attribute assignment provides.
        from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # noqa: WPS433

        DiffusionPipeline.__init__(self)

        self.checkpoint_name = checkpoint_name
        self.aux_checkpoint_name = aux_checkpoint_name
        self.model_type = "s2v"
        self.vae_t_chunk_size = vae_t_chunk_size

        # ------------------------------------------------------------------
        # 1. Tokenizer + text encoder + VAE — load from the companion
        #    Diffusers-style WAN 2.2 T2V repo (the weights are
        #    interchangeable; the S2V repo only ships the raw `.pth` blobs).
        # ------------------------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(
            aux_checkpoint_name, subfolder="tokenizer", trust_remote_code=True
        )
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            aux_checkpoint_name, subfolder="text_encoder", trust_remote_code=True
        )
        self.vae = AutoencoderKLWan.from_pretrained(aux_checkpoint_name, subfolder="vae", trust_remote_code=True)

        # ------------------------------------------------------------------
        # 2. CCL managers.
        # ------------------------------------------------------------------
        self.dit_ccl_manager = CCLManager(
            mesh_device=mesh_device,
            num_links=num_links,
            topology=topology,
        )
        self.vae_ccl_manager = CCLManager(
            mesh_device=mesh_device,
            num_links=num_links,
            topology=ttnn.Topology.Linear,
        )
        self.encoder_ccl_manager = self.vae_ccl_manager

        self.is_fsdp = is_fsdp
        self.parallel_config = parallel_config
        self.vae_parallel_config = vae_parallel_config
        self.encoder_parallel_config = encoder_parallel_config
        self.mesh_device = mesh_device
        self.dynamic_load = dynamic_load

        # ------------------------------------------------------------------
        # 3. UMT5 text encoder on device.
        # ------------------------------------------------------------------
        umt5_config = UMT5Config(
            vocab_size=self.text_encoder.config.vocab_size,
            embed_dim=self.text_encoder.config.d_model,
            ff_dim=self.text_encoder.config.d_ff,
            kv_dim=self.text_encoder.config.d_kv,
            num_heads=self.text_encoder.config.num_heads,
            num_hidden_layers=self.text_encoder.config.num_layers,
            max_prompt_length=512,
            layer_norm_eps=self.text_encoder.config.layer_norm_epsilon,
            relative_attention_num_buckets=self.text_encoder.config.relative_attention_num_buckets,
            relative_attention_max_distance=self.text_encoder.config.relative_attention_max_distance,
        )
        self.tt_umt5_encoder = UMT5Encoder(
            config=umt5_config,
            mesh_device=self.mesh_device,
            ccl_manager=self.encoder_ccl_manager,
            parallel_config=self.encoder_parallel_config,
        )

        # ------------------------------------------------------------------
        # 4. S2V transformer (single stage) on device.
        # ------------------------------------------------------------------
        s2v_snapshot = find_s2v_snapshot(checkpoint_name)
        s2v_cfg = load_s2v_config(s2v_snapshot)
        if s2v_cfg.get("adain_mode", "attn_norm") != "attn_norm":
            raise NotImplementedError(
                f"adain_mode={s2v_cfg.get('adain_mode')!r} is not supported; production uses 'attn_norm'"
            )
        if s2v_cfg.get("enable_motioner", False):
            raise NotImplementedError(
                "enable_motioner=True (MotionerTransformers) is not supported; production uses FramePacker"
            )
        inject_layers = (
            tuple(audio_inject_layers) if audio_inject_layers is not None else tuple(s2v_cfg["audio_inject_layers"])
        )
        # wav2vec2-large-xlsr-53: num_hidden_layers=24 → num_audio_layers=25.
        num_audio_layers = 25

        self.transformer = WanS2VTransformer3DModel(
            patch_size=tuple(s2v_cfg.get("patch_size", (1, 2, 2))),
            num_heads=s2v_cfg["num_heads"],
            dim=s2v_cfg["dim"],
            in_channels=s2v_cfg.get("in_dim", 16),
            out_channels=s2v_cfg.get("out_dim", 16),
            text_dim=s2v_cfg.get("text_dim", 4096),
            freq_dim=s2v_cfg.get("freq_dim", 256),
            ffn_dim=s2v_cfg["ffn_dim"],
            num_layers=s2v_cfg["num_layers"],
            cross_attn_norm=s2v_cfg.get("cross_attn_norm", True),
            eps=s2v_cfg.get("eps", 1e-6),
            audio_dim=s2v_cfg["audio_dim"],
            num_audio_layers=num_audio_layers,
            num_audio_token=s2v_cfg.get("num_audio_token", 4),
            audio_inject_layers=inject_layers,
            enable_adain=s2v_cfg.get("enable_adain", True),
            enable_motioner=s2v_cfg.get("enable_motioner", False),
            enable_framepack=s2v_cfg.get("enable_framepack", True),
            motion_token_num=s2v_cfg.get("motion_token_num", 1024),
            mesh_device=self.mesh_device,
            ccl_manager=self.dit_ccl_manager,
            parallel_config=self.parallel_config,
            is_fsdp=self.is_fsdp,
            model_type="s2v",
        )

        # ------------------------------------------------------------------
        # 5. VAE decoder + reference-image encoder on device.
        # ------------------------------------------------------------------
        self.tt_vae = WanDecoder(
            base_dim=self.vae.config.base_dim,
            z_dim=self.vae.config.z_dim,
            dim_mult=self.vae.config.dim_mult,
            num_res_blocks=self.vae.config.num_res_blocks,
            attn_scales=self.vae.config.attn_scales,
            temperal_downsample=self.vae.config.temperal_downsample,
            out_channels=self.vae.config.out_channels,
            is_residual=self.vae.config.is_residual,
            mesh_device=self.mesh_device,
            ccl_manager=self.vae_ccl_manager,
            parallel_config=self.vae_parallel_config,
            dtype=vae_dtype,
            sdpa_t_fracture_w_only=sdpa_t_fracture_w_only,
            height=height,
            width=width,
            t_chunk_size=(num_frames - 1) // 4 + 1 if vae_t_chunk_size is None else vae_t_chunk_size,
            cached=vae_t_chunk_size is not None,
        )

        self.tt_vae_encoder = WanEncoder(
            base_dim=self.vae.config.base_dim,
            in_channels=self.vae.config.in_channels,
            z_dim=self.vae.config.z_dim,
            dim_mult=self.vae.config.dim_mult,
            num_res_blocks=self.vae.config.num_res_blocks,
            attn_scales=self.vae.config.attn_scales,
            temperal_downsample=self.vae.config.temperal_downsample,
            is_residual=self.vae.config.is_residual,
            mesh_device=self.mesh_device,
            ccl_manager=self.vae_ccl_manager,
            parallel_config=self.vae_parallel_config,
        )

        # ------------------------------------------------------------------
        # 6. Wav2Vec2 audio encoder (production large-xlsr-53 bundled in S2V).
        # ------------------------------------------------------------------
        bundled_wav2vec2 = s2v_snapshot / "wav2vec2-large-xlsr-53-english"
        if not bundled_wav2vec2.exists():
            msg = f"bundled wav2vec2 weights not found at {bundled_wav2vec2}"
            raise FileNotFoundError(msg)
        audio_model_id = str(bundled_wav2vec2)
        self.audio_processor = Wav2Vec2Processor.from_pretrained(audio_model_id)
        self.torch_audio_encoder = Wav2Vec2Model.from_pretrained(audio_model_id).eval()

        audio_parallel_config = EncoderParallelConfig(
            tensor_parallel=ParallelFactor(
                factor=encoder_parallel_config.tensor_parallel.factor,
                mesh_axis=encoder_parallel_config.tensor_parallel.mesh_axis,
            )
        )
        self.tt_audio_encoder = Wav2Vec2Encoder(
            config=Wav2Vec2Config.from_hf(self.torch_audio_encoder.config),
            mesh_device=self.mesh_device,
            ccl_manager=self.encoder_ccl_manager,
            parallel_config=audio_parallel_config,
        )

        # ------------------------------------------------------------------
        # 7. Scheduler — built manually since the S2V repo has no scheduler/.
        # ------------------------------------------------------------------
        # WAN T2V defaults: flow_shift=5.0 for 720p, 3.0 for 480p. We pick by
        # the requested height; the caller can override by passing scheduler=.
        if scheduler is None:
            flow_shift = 5.0 if height >= 720 else 3.0
            scheduler = UniPCMultistepScheduler(
                num_train_timesteps=1000,
                flow_shift=flow_shift,
                prediction_type="flow_prediction",
                use_flow_sigmas=True,
            )
        self._solver = UniPCSolver(scheduler=scheduler)

        # ------------------------------------------------------------------
        # 8. Build the transformer state list. The parent's call() expects a
        #    two-entry list keyed on boundary_ratio; with a single stage we
        #    populate both slots with the same model so the boundary check is
        #    a no-op. (boundary_ratio=None makes the loop always pick index 0.)
        # ------------------------------------------------------------------
        self.transformer_2 = self.transformer  # alias — single-stage S2V
        self.transformer_states = [
            TransformerState(self.transformer, "transformer", torch_model=None, guidance_scale=5.0),
            TransformerState(self.transformer_2, "transformer", torch_model=None, guidance_scale=5.0),
        ]
        # The denoise loop in the parent picks transformer index 0 whenever
        # ``t >= boundary_ratio * num_train_timesteps``. We want index 0 every
        # step, so set boundary_ratio just below 0 (any value ≤ 0 works since
        # all timesteps are ≥ 0). The CFG-only check at line 704 of
        # ``WanPipeline.__call__`` accepts a non-None boundary_ratio.
        self.register_to_config(boundary_ratio=0.0)
        self.register_to_config(expand_timesteps=False)

        # ------------------------------------------------------------------
        # 9. VAE bookkeeping.
        # ------------------------------------------------------------------
        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self._vae_latents_mean = torch.tensor(self.vae.config.latents_mean, dtype=self.vae.dtype).view(
            1, self.vae.config.z_dim, 1, 1, 1
        )
        self._vae_latents_std = torch.tensor(self.vae.config.latents_std, dtype=self.vae.dtype).view(
            1, self.vae.config.z_dim, 1, 1, 1
        )

        # persistent latent buffers (for traced calls)
        self.latent_buffer = None
        self.condition_buffer = None

        # ------------------------------------------------------------------
        # 10. Load weights: text encoder + VAE via the existing cache helpers;
        #     S2V transformer via the native loader + mapper; audio encoder
        #     via cache (its state dict comes from the HF model directly).
        # ------------------------------------------------------------------
        self._prepare_text_encoder_aux()
        self._prepare_vae_aux()
        self._load_s2v_transformer(s2v_snapshot)
        self._prepare_audio_encoder(audio_model_id)

        if dynamic_load and ttnn.device.is_blackhole():
            # 14B transformer + wav2vec2-large + VAE + UMT5 is tight on
            # BH 4×8. Mirror the parent's coresident-exclusion chain.
            self.transformer.register_coresident_exclusions(self.tt_vae)
            self.tt_vae.register_coresident_exclusions(self.transformer)

        if run_warmup:
            self.warmup_buffers(height=height, width=width)

    # ----------------------------------------------------------------------
    # create_pipeline factory.
    # ----------------------------------------------------------------------

    @staticmethod
    def create_pipeline(
        mesh_device,
        *,
        checkpoint_name: str = _DEFAULT_S2V_CHECKPOINT,
        aux_checkpoint_name: str = _DEFAULT_AUX_CHECKPOINT,
        scheduler=None,
        sp_axis=None,
        tp_axis=None,
        num_links=None,
        dynamic_load=None,
        topology=None,
        is_fsdp=None,
        sdpa_t_fracture_w_only=None,
        height: int = 0,
        width: int = 0,
        num_frames: int = 81,
        **_unused,
    ):
        # Pick a sane device-aware default. We mirror the BH 4×8 / WH 4×8
        # rows from ``WanPipeline.create_pipeline``; S2V doesn't target 4×32
        # in this revision.
        device_configs = {}
        if ttnn.device.is_blackhole():
            device_configs[(4, 8)] = {
                "sp_axis": 1,
                "tp_axis": 0,
                "num_links": 2,
                "dynamic_load": True,
                "topology": ttnn.Topology.Linear,
                "is_fsdp": False,
                "vae_t_chunk_size": 1,
            }
        else:
            device_configs[(4, 8)] = {
                "sp_axis": 1,
                "tp_axis": 0,
                "num_links": 4,
                "dynamic_load": False,
                "topology": ttnn.Topology.Ring,
                "is_fsdp": True,
                "vae_t_chunk_size": 1,
            }
        try:
            config = device_configs[tuple(mesh_device.shape)]
        except KeyError as err:
            msg = (
                f"WanPipelineS2V has no default config for mesh shape {tuple(mesh_device.shape)}; "
                "pass sp_axis/tp_axis/num_links/topology/is_fsdp explicitly."
            )
            raise NotImplementedError(msg) from err

        sp_axis = config["sp_axis"] if sp_axis is None else sp_axis
        tp_axis = config["tp_axis"] if tp_axis is None else tp_axis

        h_factor = tuple(mesh_device.shape)[tp_axis]
        w_factor = tuple(mesh_device.shape)[sp_axis]

        parallel_config = DiTParallelConfig(
            tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=h_factor),
            sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=w_factor),
            cfg_parallel=None,
        )
        vae_parallel_config = VaeHWParallelConfig(
            height_parallel=ParallelFactor(factor=h_factor, mesh_axis=tp_axis),
            width_parallel=ParallelFactor(factor=w_factor, mesh_axis=sp_axis),
        )
        encoder_parallel_config = EncoderParallelConfig(
            tensor_parallel=ParallelFactor(factor=h_factor, mesh_axis=tp_axis)
        )

        return WanPipelineS2V(
            mesh_device=mesh_device,
            parallel_config=parallel_config,
            vae_parallel_config=vae_parallel_config,
            encoder_parallel_config=encoder_parallel_config,
            num_links=num_links or config["num_links"],
            checkpoint_name=checkpoint_name,
            aux_checkpoint_name=aux_checkpoint_name,
            scheduler=scheduler,
            topology=topology or config["topology"],
            is_fsdp=is_fsdp if is_fsdp is not None else config["is_fsdp"],
            dynamic_load=dynamic_load if dynamic_load is not None else config["dynamic_load"],
            vae_t_chunk_size=config["vae_t_chunk_size"],
            sdpa_t_fracture_w_only=sdpa_t_fracture_w_only or False,
            height=height,
            width=width,
            num_frames=num_frames,
        )

    # ----------------------------------------------------------------------
    # Weight-load helpers.
    # ----------------------------------------------------------------------

    def _prepare_text_encoder_aux(self) -> None:
        cache.load_model(
            self.tt_umt5_encoder,
            model_name=os.path.basename(self.aux_checkpoint_name),
            subfolder="text_encoder",
            parallel_config=self.encoder_parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            get_torch_state_dict=lambda: self.text_encoder.state_dict(),
        )

    # Override the parent's per-step loaders. The S2V transformer is loaded
    # once up-front via the native mapper, so the per-step reloader is a
    # no-op. Same for the text encoder & VAE (loaded up front).
    def _prepare_text_encoder(self) -> None:
        return None

    def _prepare_transformer(self, idx: int) -> None:
        return None

    def _prepare_vae(self) -> None:
        return None

    def _prepare_vae_aux(self) -> None:
        blocking_key = conv3d_blocking_hash(self.tt_vae)
        subfolder = f"vae_{blocking_key}" if blocking_key else "vae"
        cache.load_model(
            self.tt_vae,
            model_name=os.path.basename(self.aux_checkpoint_name),
            subfolder=subfolder,
            parallel_config=self.vae_parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            get_torch_state_dict=lambda: self.vae.state_dict(),
        )
        cache.load_model(
            self.tt_vae_encoder,
            model_name=os.path.basename(self.aux_checkpoint_name),
            subfolder="vae_encoder",
            parallel_config=self.vae_parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            get_torch_state_dict=lambda: self.vae.state_dict(),
        )

    def _load_s2v_transformer(self, s2v_snapshot: Path) -> None:
        ref_sd = load_s2v_state_dict(s2v_snapshot)
        tt_sd = translate_s2v_state_dict(ref_sd)
        logger.info(f"Loading S2V transformer: {len(tt_sd)} parameters via native mapper.")
        self.transformer.load_torch_state_dict(tt_sd, strict=True)

    def _prepare_audio_encoder(self, audio_model_id: str) -> None:
        audio_parallel_config = EncoderParallelConfig(
            tensor_parallel=ParallelFactor(
                factor=self.encoder_parallel_config.tensor_parallel.factor,
                mesh_axis=self.encoder_parallel_config.tensor_parallel.mesh_axis,
            )
        )
        cache.load_model(
            self.tt_audio_encoder,
            model_name="wav2vec2-large-xlsr-53-english",
            subfolder="wav2vec2",
            parallel_config=audio_parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            get_torch_state_dict=lambda: self.torch_audio_encoder.state_dict(),
        )
        self.tt_audio_encoder.bind_cpu_modules(self.torch_audio_encoder)

    # ----------------------------------------------------------------------
    # Pre-denoise plumbing for S2V — invoked by __call__() once per clip.
    # ----------------------------------------------------------------------

    def _encode_normalized(self, pixels_BCTHW: torch.Tensor) -> torch.Tensor:
        """TT VAE encode + production ``(mu - latents_mean) / latents_std`` normalization.

        Pixel input shape ``[B, 3, T, H, W]``, output ``[B, z_dim=16, T_lat, H_lat, W_lat]``.
        Module docstring covers why normalized latents are used as the
        conditioning constants.
        """
        # Match the ref-image encode path: BCTHW → BTHWC, pad in-channels &
        # height to the VAE's 2D-sharded grid, upload 2D-sharded, run encoder,
        # gather back, take z_dim mean channels, then normalize.
        pixels_BTHWC = pixels_BCTHW.to(torch.float32).permute(0, 2, 3, 4, 1)
        pixels_BTHWC = conv_pad_in_channels(pixels_BTHWC)
        pixels_BTHWC, logical_h = conv_pad_height(
            pixels_BTHWC,
            self.vae_parallel_config.height_parallel.factor * self.vae_scale_factor_spatial,
        )
        pixels_dev = bf16_tensor_2dshard(
            pixels_BTHWC,
            self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            shard_mapping={
                self.vae_parallel_config.height_parallel.mesh_axis: 2,
                self.vae_parallel_config.width_parallel.mesh_axis: 3,
            },
        )
        # ``encoder_t_chunk_size=16`` processes 16 frames per iteration vs the
        # default 4 — ~3-4× fewer encoder calls for the 81-frame cond_states path.
        latent_BCTHW, new_logical_h = self.tt_vae_encoder(pixels_dev, logical_h, encoder_t_chunk_size=16)
        concat_dims = [None, None]
        concat_dims[self.vae_parallel_config.height_parallel.mesh_axis] = 3
        concat_dims[self.vae_parallel_config.width_parallel.mesh_axis] = 4
        latent_torch = fast_device_to_host(
            latent_BCTHW, self.mesh_device, concat_dims, ccl_manager=self.vae_ccl_manager
        )[:, :, :, :new_logical_h, :]
        # tt_vae_encoder already returns the mean (first z_dim channels) —
        # see ``WanEncoder.forward``'s output slice. Apply normalization.
        return ((latent_torch.float() - self._vae_latents_mean.float()) / self._vae_latents_std.float()).to(
            torch.float32
        )

    @torch.no_grad()
    def __call__(self, *args, audio_prompt: Optional[str] = None, **kwargs):
        """S2V entry point.

        ``audio_prompt`` is a first-class kwarg here. It is forwarded to
        :meth:`prepare_latents` via a private instance attribute because the
        parent ``WanPipeline.__call__`` predates the audio-prompt arg and
        does not propagate extra kwargs through to ``prepare_latents``.
        """
        if audio_prompt is None:
            raise ValueError("audio_prompt (path to a .wav/.mp3 file) is required for S2V")
        self._audio_prompt = audio_prompt
        try:
            return super().__call__(*args, **kwargs)
        finally:
            self._audio_prompt = None

    def prepare_latents(
        self,
        batch_size: int,
        image_prompt=None,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        latents: Optional[torch.Tensor] = None,
        audio_prompt: Optional[str] = None,
        pose_video: Optional[str] = None,
    ) -> tuple:
        if batch_size != 1:
            raise NotImplementedError(f"S2V only supports batch_size=1, got {batch_size}")
        if pose_video is not None:
            raise NotImplementedError("--pose_video conditioning is not implemented; only zero-pose is supported")
        if audio_prompt is None:
            audio_prompt = getattr(self, "_audio_prompt", None)
        if audio_prompt is None:
            raise ValueError("audio_prompt (path to a .wav/.mp3 file) is required for S2V")
        if image_prompt is None or not isinstance(image_prompt, Image.Image):
            raise ValueError("image_prompt (PIL.Image) is required for S2V (reference image)")

        # 1. Random noisy latents from the base path.
        latents, _ = WanPipeline.prepare_latents(
            self,
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=dtype,
            device=device,
            latents=latents,
        )

        # 2. Reference image → VAE encode + normalize.
        ref_tensor = self.video_processor.preprocess(image_prompt, height=height, width=width).to(
            device, dtype=torch.float32
        )
        ref_video = ref_tensor.unsqueeze(2)  # [B, C, T=1, H, W]
        ref_latent_torch = self._encode_normalized(ref_video)

        # 3. Audio: load wav, run wav2vec2, resample to video_rate, bucket to video FPS.
        # Matches the reference flow in `wan/speech2video.py:encode_audio`:
        # wav2vec2 outputs at 50 Hz → linear_interpolation → 30 Hz →
        # `get_audio_embed_bucket_fps` evenly samples `num_frames` features
        # for the clip at 16 fps. Skipping the resample or zero-padding short
        # clips (the prior behavior) misaligns the audio and the model
        # produces silent / incoherent output.
        input_values = load_audio_to_input_values(audio_prompt, self.audio_processor)
        all_hidden = self.tt_audio_encoder(input_values, output_hidden_states=True)
        # ``fast_device_to_host`` expects exactly two concat_dims for a 2D mesh.
        hidden_torch = torch.stack(
            [
                fast_device_to_host(h, self.mesh_device, [None, None], ccl_manager=self.encoder_ccl_manager).float()
                for h in all_hidden
            ],
            dim=1,
        )  # [B=1, num_layers, T_50Hz, audio_dim]
        bucketed_per_layer = []
        for layer_idx in range(hidden_torch.shape[1]):
            feat_30Hz = linear_interpolation(
                hidden_torch[0, layer_idx], input_fps=WAV2VEC2_HZ, output_fps=S2V_VIDEO_RATE
            )
            aligned, _ = get_audio_embed_bucket_fps(
                feat_30Hz, fps=16, batch_frames=num_frames, video_rate=S2V_VIDEO_RATE
            )
            bucketed_per_layer.append(aligned)
        audio_BLNF = torch.stack(bucketed_per_layer, dim=0).unsqueeze(0).permute(0, 1, 3, 2)
        # Snap the audio T to the spatial latent frame count so the per-frame
        # block-diagonal cross-attn mask has an integer hw_per_frame. Without
        # this, motion_frames=[17,5] with num_frames=81 produces 20 audio
        # frames vs 21 latent frames.
        num_latent_frames = latents.shape[2]
        self.transformer.prepare_audio_emb(audio_BLNF, target_num_frames=num_latent_frames)

        # 4. Motion latents — VAE-encoded zero pixels, NOT literal latent zeros.
        # (See module docstring for the latent-space rationale.)
        MOTION_FRAMES_PIXEL = 17  # reference config: s2v_14B motion_frames
        motion_pixels = torch.zeros(1, 3, MOTION_FRAMES_PIXEL, height, width, dtype=torch.float32)
        motion_latents_torch = self._encode_normalized(motion_pixels)

        # 5. Pose conditioning (cond_states): zero pose; matches the reference's
        # ``COND[0] * 0`` shortcut when ``pose_video`` is not provided.
        latent_shape = latents.shape
        cond_states_torch = torch.zeros(
            1,
            self.vae.config.z_dim,
            latent_shape[2],
            latent_shape[3],
            latent_shape[4],
            dtype=torch.float32,
        )

        # Build the on-device caches for pose / ref / motion / mask. After
        # this call the transformer's ``inner_step`` consumes ``_cached_*``
        # attributes.
        # NOTE: ``drop_first_motion=False`` is intentionally non-reference.
        # The reference sets this to True for the first clip, but an A/B run
        # on our pipeline showed the motion-dropped output is worse —
        # tracked separately while we hunt the upstream cause.
        self.transformer.prepare_cond_emb(
            noisy_latents_torch=latents,
            ref_latent_torch=ref_latent_torch,
            motion_latents_torch=motion_latents_torch,
            cond_states_torch=cond_states_torch,
            drop_first_motion=False,
        )
        return latents, None

    def get_model_input(self, latents, cond_latents):
        return super().get_model_input(latents, None)

    # ----------------------------------------------------------------------
    # Export.
    # ----------------------------------------------------------------------

    @staticmethod
    def export(frames, output_path: str, *, audio_path: Optional[str] = None, fps: int = 16) -> str:
        """Encode pipeline output to MP4, optionally muxing in ``audio_path``.

        When ``audio_path`` is provided the muxed file matches the
        reference's ``wan.utils.utils.merge_video_audio`` step.
        """
        from ...utils.video import export_to_video, export_to_video_with_audio

        if audio_path is not None:
            return export_to_video_with_audio(frames, output_path, audio_path=audio_path, fps=fps)
        return export_to_video(frames, output_path, fps=fps)
