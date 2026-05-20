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
    via :func:`wan_s2v_checkpoint.load_s2v_state_dict` and the name translator
    at :func:`wan_s2v_checkpoint.translate_s2v_state_dict`.
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
``pose_video`` (only zero pose conditioning), ``enable_tts``,
``init_first_frame=True``, and ``adain_mode != "attn_norm"``.
"""

from __future__ import annotations

import os
import tempfile
import wave
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from diffusers.models import AutoencoderKLWan
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.video_processor import VideoProcessor
from loguru import logger
from PIL import Image
from transformers import AutoTokenizer, UMT5EncoderModel, Wav2Vec2Model, Wav2Vec2Processor

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler  # noqa: F401  (kept for parity with base)

from ...encoders.umt5.model_umt5 import UMT5Config, UMT5Encoder
from ...encoders.wav2vec2.audio_preprocess import (
    S2V_AUDIO_SAMPLES_PER_CLIP,
    S2V_VIDEO_RATE,
    WAV2VEC2_HZ,
    WAV2VEC2_SAMPLE_RATE,
    get_audio_embed_bucket_fps,
    load_audio_to_input_values,
)
from ...encoders.wav2vec2.config_wav2vec2 import Wav2Vec2Config
from ...encoders.wav2vec2.model_wav2vec2 import Wav2Vec2Encoder
from ...models.transformers.wan2_2.s2v.transformer_wan_s2v import WanS2VTransformer3DModel
from ...models.vae.vae_wan2_1 import WanDecoder, WanEncoder
from ...parallel.config import DiTParallelConfig, EncoderParallelConfig, ParallelFactor, VaeHWParallelConfig
from ...parallel.manager import CCLManager
from ...solvers import UniPCSolver
from ...utils import cache, tensor
from ...utils.conv3d import conv3d_blocking_hash, conv_pad_height, conv_pad_in_channels
from ...utils.tensor import (
    bf16_tensor_2dshard,
    fast_device_to_host,
    float32_tensor,
    local_device_to_torch,
    typed_tensor_2dshard,
)
from ...utils.video import export_to_video, export_to_video_with_audio
from ...utils.wan_s2v_checkpoint import (
    find_s2v_snapshot,
    load_s2v_config,
    load_s2v_state_dict,
    translate_s2v_state_dict,
)
from .pipeline_wan import TransformerState, WanPipeline

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

    ``__init__`` does not chain through ``WanPipeline.__init__`` because that
    path is hard-wired to Diffusers-style two-stage loading. We inline the
    relevant bits so we can skip the missing ``transformer/`` subfolders on
    the S2V repo, load the DiT from a flat safetensors stack via the native
    loader, and build a single transformer slot.
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
        run_warmup: bool = True,
        audio_inject_layers: tuple[int, ...] | None = None,
    ):
        # Skip the parent __init__ — it's hard-wired to Diffusers loaders we
        # can't satisfy. DiffusionPipeline.__init__ does only attribute setup.
        DiffusionPipeline.__init__(self)

        self.checkpoint_name = checkpoint_name
        self.aux_checkpoint_name = aux_checkpoint_name
        self.model_type = "s2v"
        self.vae_t_chunk_size = vae_t_chunk_size

        # Per-call state populated by ``__call__`` / ``prepare_latents`` /
        # ``_postprocess_latents_for_vae`` and consumed downstream in the same
        # ``__call__``. Declared here so attribute access doesn't need
        # ``getattr(self, name, default)``.
        self._audio_prompt: Optional[str] = None
        self._s2v_profiler: Optional[BenchmarkProfiler] = None
        self._s2v_profiler_iteration: int = 0
        self._ref_latent_torch: Optional[torch.Tensor] = None
        self._s2v_prepended_latents: int = 0

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
        # Reference uses ``wan/utils/fm_solvers_unipc.py:FlowUniPCMultistepScheduler``
        # with ``sample_shift=3`` (wan_s2v_14B.py:57). Diffusers'
        # ``UniPCMultistepScheduler`` with ``use_flow_sigmas=True``,
        # ``prediction_type="flow_prediction"`` and ``flow_shift=3.0`` produces
        # byte-identical timesteps and sigmas at 5/40 steps (verified), so the
        # vendored scheduler isn't needed.
        if scheduler is None:
            scheduler = UniPCMultistepScheduler(
                num_train_timesteps=1000,
                use_flow_sigmas=True,
                prediction_type="flow_prediction",
                flow_shift=3.0,
                use_dynamic_shifting=False,
                solver_order=2,
                solver_type="bh2",
                predict_x0=True,
                lower_order_final=True,
            )
        self._solver = UniPCSolver(scheduler=scheduler)

        # ------------------------------------------------------------------
        # 8. Build the transformer state list. The parent's call() expects a
        #    two-entry list keyed on boundary_ratio; with a single stage we
        #    populate both slots with the same model so the boundary check is
        #    a no-op. (boundary_ratio=None makes the loop always pick index 0.)
        # ------------------------------------------------------------------
        self.transformer_2 = self.transformer  # alias — single-stage S2V
        # Reference s2v_14B sample_guide_scale=4.5 (wan_s2v_14B.py:59).
        self.transformer_states = [
            TransformerState(self.transformer, "transformer", torch_model=None, guidance_scale=4.5),
            TransformerState(self.transformer_2, "transformer", torch_model=None, guidance_scale=4.5),
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

        # Lazy multi-clip warmup tracking. ``__init__`` only warms the 1-clip
        # path (drop_first_motion=True). The first ``__call__`` that requests
        # num_clips>1 triggers a one-shot warmup of the clip-1+ shape
        # (drop_first_motion=False with motion latents). ``_warming_up`` guards
        # against re-entrancy when the lazy warmup recursively invokes self.
        self._multi_clip_warmed: bool = False
        self._warming_up: bool = False

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
                "dynamic_load": False,
                "topology": ttnn.Topology.Ring,
                "is_fsdp": False,
                # Full-T single pass on 4×8 (matches t2v decoder choice); sp_factor=8
                # halves per-device H vs 2×4, giving headroom for the full latent T=21.
                "vae_t_chunk_size": None,
            }
            # BH Loud Box (2x4, 8 chips): mirror the 4x8 config; sp_factor=4 / tp_factor=2.
            device_configs[(2, 4)] = {
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

    def warmup_buffers(
        self,
        height: int,
        width: int,
        image_prompt: Optional[Image.Image] = None,
        num_clips: int = 1,
    ) -> None:
        """Warm program caches for every S2V subsystem.

        The base ``WanPipeline.warmup_buffers`` calls ``run_single_prompt`` with
        only text + image inputs — that signature is invalid for S2V which
        requires ``audio_prompt``. This override synthesizes a black reference
        image and silent audio, then runs the full pipeline once with
        ``num_inference_steps=2``.

        Two distinct shape signatures exist:
        - Clip 0 (``drop_first_motion=True``, N_motion=0)
        - Clip 1+ (``drop_first_motion=False``, N_motion>0)

        The per-clip caches (frame_packer programs, ``_mask_constant_cache``,
        ``_mask_noisy_cache``, ``_adain_E_cache``) are keyed by
        ``(padded_noisy, padded_const, N_ref, N_motion)``. Default
        ``num_clips=1`` only warms the clip-0 signature — fast init for the
        common single-clip case. Multi-clip callers pay the second-shape
        warmup lazily via ``__call__``'s dispatcher.
        """
        sample_rate = WAV2VEC2_SAMPLE_RATE
        num_samples = num_clips * S2V_AUDIO_SAMPLES_PER_CLIP
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            warmup_audio_path = tmp.name
        try:
            with wave.open(warmup_audio_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(b"\x00\x00" * num_samples)

            warmup_image = image_prompt if image_prompt is not None else Image.new("RGB", (width, height))
            self._warming_up = True
            try:
                self(
                    prompt="warmup",
                    image_prompt=warmup_image,
                    audio_prompt=warmup_audio_path,
                    height=height,
                    width=width,
                    num_clips=num_clips,
                    num_inference_steps=2,
                    seed=0,
                    output_type="uint8",
                )
            finally:
                self._warming_up = False
            if num_clips >= 2:
                self._multi_clip_warmed = True
        finally:
            try:
                os.remove(warmup_audio_path)
            except OSError:
                pass

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
    # Reference-faithful frame-count constants (wan_s2v_14B + speech2video.py).
    # ----------------------------------------------------------------------

    # Per-clip pixel frame count (speech2video.py:404, ``infer_frames=80``).
    _INFER_FRAMES_PIXEL = 80
    # Motion-context pixel frames carried between clips
    # (config wan_s2v_14B.py:51, ``motion_frames=73``).
    _MOTION_FRAMES_PIXEL = 73
    # Latent frames the motion context occupies after VAE encode
    # (``(motion_frames + 3) // 4``, speech2video.py:491).
    _LAT_MOTION_FRAMES = 19
    # ``(infer_frames + 3 + motion_frames) // 4 - motion_lat_frames`` =
    # ``(80 + 3 + 73) // 4 - 19 = 20`` (speech2video.py:544-545).
    _LAT_TARGET_FRAMES = 20
    # Clip-0 pixel-transient trim. With ``drop_first_motion=True`` the VAE
    # decode is ``ref(1) + noisy(20) = 21 latents → 4*21-3 = 81 pixels``;
    # we keep the last ``INFER_FRAMES_PIXEL = 80`` and drop 3 more transient
    # frames at the front, matching ``speech2video.py:654-656``.
    _S2V_VAE_CLIP0_TRIM = 3

    # ----------------------------------------------------------------------
    # Pre-denoise plumbing for S2V — invoked by __call__() once per clip.
    # ----------------------------------------------------------------------

    def _encode_normalized(self, pixels_BCTHW: torch.Tensor, *, encoder_t_chunk_size: int = 16) -> torch.Tensor:
        """TT VAE encode + ``(mu - latents_mean) / latents_std`` normalization.

        ``pixels_BCTHW`` → ``[B, z_dim=16, T_lat, H_lat, W_lat]``.
        """
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
        latent_BCTHW, new_logical_h = self.tt_vae_encoder(
            pixels_dev, logical_h, encoder_t_chunk_size=encoder_t_chunk_size
        )
        concat_dims = [None, None]
        concat_dims[self.vae_parallel_config.height_parallel.mesh_axis] = 3
        concat_dims[self.vae_parallel_config.width_parallel.mesh_axis] = 4
        latent_torch = fast_device_to_host(
            latent_BCTHW, self.mesh_device, concat_dims, ccl_manager=self.vae_ccl_manager
        )[:, :, :, :new_logical_h, :]
        # tt_vae_encoder returns the mean (first z_dim channels) already.
        return ((latent_torch.float() - self._vae_latents_mean.float()) / self._vae_latents_std.float()).to(
            torch.float32
        )

    def _prepare_audio_full(self, audio_prompt: str) -> tuple[torch.Tensor, int]:
        """Run wav2vec2 + bucket the audio file to per-clip windows.

        Mirrors ``wan/speech2video.py:283-294``: wav2vec2 outputs at 50 Hz
        → linear_interpolation to 30 Hz → ``get_audio_embed_bucket_fps``
        evenly samples one feature per video frame at 16 fps, padding to
        ``num_repeat * INFER_FRAMES_PIXEL`` frames.

        Per-clip canonical chunking: ``load_audio_to_input_values`` snaps
        every audio file to an integer multiple of ``S2V_AUDIO_SAMPLES_PER_CLIP``
        (= 80 video frames worth of audio = 80 000 samples). We then invoke
        ``tt_audio_encoder`` once per 80 000-sample chunk so every wav2vec2
        forward sees the SAME input shape regardless of total audio length,
        and the warmup primes that canonical shape exactly. Reference
        semantics are preserved at the per-clip slice level (each clip
        downstream still gets exactly 80 audio frames via the unchanged
        bucketing logic on the concatenated features).

        Returns ``(audio_BLNF_full, num_repeat)`` where ``audio_BLNF_full``
        has shape ``[1, num_layers, audio_dim, num_repeat * 80]``.
        """
        import time

        _t0 = time.perf_counter()
        input_values = load_audio_to_input_values(audio_prompt, self.audio_processor)
        T_raw = input_values.shape[-1]
        assert T_raw % S2V_AUDIO_SAMPLES_PER_CLIP == 0, (
            f"load_audio_to_input_values must snap to a multiple of " f"{S2V_AUDIO_SAMPLES_PER_CLIP}; got T_raw={T_raw}"
        )
        n_chunks = T_raw // S2V_AUDIO_SAMPLES_PER_CLIP
        _t1 = time.perf_counter()

        # Per-chunk wav2vec2: every call has the canonical clip length so the
        # on-device program cache hits the same shape for every chunk. Each
        # chunk returns ``num_layers`` ttnn tensors; stack-on-device into
        # ``[num_layers, B, T_chunk, C]`` so the D2H below is one transfer
        # per chunk instead of ``num_layers`` (75 → 3 d2h calls on 16s audio).
        # Three smaller d2h calls overlap better than one big one (async-DMA
        # in flight), so we keep them separate and concat on host below.
        per_chunk_stacked_dev: list = []
        for c in range(n_chunks):
            start = c * S2V_AUDIO_SAMPLES_PER_CLIP
            end = start + S2V_AUDIO_SAMPLES_PER_CLIP
            chunk = input_values[..., start:end]
            chunk_hidden = self.tt_audio_encoder(chunk, output_hidden_states=True)
            per_chunk_stacked_dev.append(ttnn.stack(chunk_hidden, dim=0))
        ttnn.synchronize_device(self.mesh_device)
        _t2 = time.perf_counter()

        # One D2H per chunk, then concat on host along time. Permute to
        # ``[B, num_layers, T, C]`` to match downstream layout.
        per_chunk_host: list[torch.Tensor] = [
            fast_device_to_host(t, self.mesh_device, [None, None], ccl_manager=self.encoder_ccl_manager).float()
            for t in per_chunk_stacked_dev
        ]
        hidden_torch = torch.cat(per_chunk_host, dim=2).permute(1, 0, 2, 3).contiguous()
        num_layers = hidden_torch.shape[1]
        _t3 = time.perf_counter()

        # Vectorized interp + bucket across all layers in one pass. The
        # per-layer loop was ~25 separate F.interpolate calls + 25 gathers;
        # batching collapses both into single ops on the layer-batched
        # [num_layers, C, T] tensor.
        feat_LCT = hidden_torch[0].permute(0, 2, 1).contiguous()  # [L, C, T_50Hz]
        T_50Hz = feat_LCT.shape[-1]
        T_30Hz = int(T_50Hz / float(WAV2VEC2_HZ) * S2V_VIDEO_RATE)
        feat_LCT_30Hz = torch.nn.functional.interpolate(
            feat_LCT, size=T_30Hz, align_corners=True, mode="linear"
        )  # [L, C, T_30Hz]
        feat_LTC_30Hz = feat_LCT_30Hz.permute(0, 2, 1).contiguous()  # [L, T_30Hz, C]
        aligned_LBC, num_repeat = get_audio_embed_bucket_fps(
            feat_LTC_30Hz, fps=16, batch_frames=self._INFER_FRAMES_PIXEL, video_rate=S2V_VIDEO_RATE
        )
        audio_BLNF_full = aligned_LBC.unsqueeze(0).permute(0, 1, 3, 2)
        _t4 = time.perf_counter()
        logger.info(
            f"[wav2vec2 breakdown] n_chunks={n_chunks} load_audio={_t1 - _t0:.3f}s "
            f"tt_encoder_fwd={_t2 - _t1:.3f}s d2h+concat={_t3 - _t2:.3f}s "
            f"interp+bucket={_t4 - _t3:.3f}s ({num_layers} layers, batched) "
            f"total={_t4 - _t0:.3f}s"
        )
        return audio_BLNF_full, int(num_repeat)

    def _prepare_clip(
        self,
        *,
        clip_idx: int,
        audio_clip: torch.Tensor,
        ref_latent_torch: torch.Tensor,
        motion_latents_torch: torch.Tensor,
        drop_first_motion: bool,
        height: int,
        width: int,
        seed: int,
        _stage,
    ) -> torch.Tensor:
        """Set up the transformer caches for one clip and return its noise latents.

        Generates clip-specific noise (``seed + clip_idx``, reference 542),
        calls ``prepare_audio_emb`` on the audio slice, and ``prepare_cond_emb``
        with the motion latents threaded in from the previous clip (or the
        zero placeholder for clip 0 with ``drop_first_motion=True``).
        """
        latent_h = height // self.vae_scale_factor_spatial
        latent_w = width // self.vae_scale_factor_spatial

        torch.manual_seed(seed + clip_idx)
        latents = torch.randn(
            1, self.vae.config.z_dim, self._LAT_TARGET_FRAMES, latent_h, latent_w, dtype=torch.float32
        )

        with _stage(f"s2v_clip_{clip_idx}_prepare_audio_emb"):
            self.transformer.prepare_audio_emb(
                audio_clip,
                motion_frames=(self._MOTION_FRAMES_PIXEL, self._LAT_MOTION_FRAMES),
                target_num_frames=self._LAT_TARGET_FRAMES,
            )

        # ``cond_states_torch=None`` is the no-pose path the transformer's
        # ``prepare_cond_emb`` shape-caches WanPatchEmbed(zeros) for.
        with _stage(f"s2v_clip_{clip_idx}_prepare_cond_emb"):
            self.transformer.prepare_cond_emb(
                noisy_latents_torch=latents,
                ref_latent_torch=ref_latent_torch,
                motion_latents_torch=motion_latents_torch,
                cond_states_torch=None,
                drop_first_motion=drop_first_motion,
            )
        return latents

    def _denoise_clip(
        self,
        *,
        latents_torch: torch.Tensor,
        num_inference_steps: int,
        traced: bool,
    ) -> torch.Tensor:
        """Run the diffusion loop for one clip and return the post-denoise latents.

        Assumes text-conditioning buffers (``ts.prompt_buffer`` /
        ``negative_prompt_buffer``) are already populated — they're
        clip-invariant and built once in ``__call__``. Duplicated structure
        from ``WanPipeline.__call__`` lines 949-1055.
        """
        self._solver.set_schedule(num_inference_steps, device="cpu")
        timesteps = self._solver.timesteps

        sp_axis = self.transformer_states[0].model.parallel_config.sequence_parallel.mesh_axis
        ts = self.transformer_states[0]

        permuted_latent, patchified_seqlen = ts.model.preprocess_spatial_input_host(latents_torch)
        rope_cos_1HND, rope_sin_1HND, trans_mat = ts.model.get_rope_features(latents_torch)
        rope_args = {
            "rope_cos_1HND": rope_cos_1HND,
            "rope_sin_1HND": rope_sin_1HND,
            "trans_mat": trans_mat,
        }
        permuted_latent_tt = tensor.from_torch(
            permuted_latent,
            device=self.mesh_device,
            mesh_axes=[None, None, sp_axis, None],
            dtype=ts.model.output_dtype,
        )

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Refresh ``latent_buffer`` from the solver's evolving
                # ``permuted_latent_tt`` (initial noise on i=0, then the
                # previous step's denoised output). Matches the base
                # ``WanPipeline.__call__`` per-iteration update.
                if self.latent_buffer is None:
                    self.latent_buffer = permuted_latent_tt
                else:
                    ttnn.copy(permuted_latent_tt, self.latent_buffer)

                timestep = t.expand(latents_torch.shape[0])  # S2V uses expand_timesteps=False
                timestep = float32_tensor(
                    timestep.unsqueeze(1).unsqueeze(1).unsqueeze(1), device=(None if traced else self.mesh_device)
                )
                permuted_model_input = self.get_model_input(self.latent_buffer, self.condition_buffer)
                permuted_noise_pred_tt = ts.model.combined_step(
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    spatial_1BNI=permuted_model_input,
                    prompt_1BLP=ts.prompt_buffer,
                    negative_prompt_1BLP=ts.negative_prompt_buffer,
                    N=patchified_seqlen,
                    timestep=timestep,
                    **rope_args,
                    guidance_scale=ts.guidance_scale,
                    traced=traced,
                    gather_output=False,
                )
                permuted_latent_tt = self._solver.step(
                    step=i,
                    latent=self.latent_buffer,
                    velocity_pred=permuted_noise_pred_tt,
                )
                progress_bar.update()

        permuted_latent_tt = ts.model.ccl_manager.all_gather_persistent_buffer(
            permuted_latent_tt, dim=2, mesh_axis=sp_axis
        )
        permuted_latent = local_device_to_torch(permuted_latent_tt)
        latents_out = ts.model.postprocess_spatial_output_host(
            permuted_latent,
            F=latents_torch.shape[2],
            H=latents_torch.shape[3],
            W=latents_torch.shape[4],
            N=patchified_seqlen,
        )
        return latents_out

    def _decode_clip(
        self,
        *,
        latents_torch: torch.Tensor,
        prepend_latents_torch: torch.Tensor,
    ) -> torch.Tensor:
        """VAE-decode ``[prepend | noisy]`` and return ``[1, 3, T_decoded, H, W]`` (float).

        ``prepend_latents_torch`` is either ``ref_latents`` (clip 0 with
        ``drop_first_motion=True``) or the previous clip's motion latents.
        Both must already be in VAE input space (denormalized).
        """
        # Caller passes latents in normalized model space; denormalize to VAE
        # input space, prepend (also in VAE input space), and run VAE decode.
        latents = latents_torch.to(self.vae.dtype) * self._vae_latents_std + self._vae_latents_mean
        prepend = prepend_latents_torch.to(dtype=latents.dtype, device=latents.device)
        decode_latents = torch.cat([prepend, latents], dim=2)

        tt_latents_BTHWC, logical_h = self.tt_vae.prepare_input(decode_latents)
        tt_latents_BTHWC = typed_tensor_2dshard(
            tt_latents_BTHWC,
            self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            shard_mapping={
                self.vae_parallel_config.height_parallel.mesh_axis: 2,
                self.vae_parallel_config.width_parallel.mesh_axis: 3,
            },
            dtype=self.tt_vae.dtype,
        )
        self._prepare_vae()
        tt_video_BCTHW, new_logical_h = self.tt_vae(tt_latents_BTHWC, logical_h, t_chunk_size=self.vae_t_chunk_size)
        concat_dims = [None, None]
        concat_dims[self.vae_parallel_config.height_parallel.mesh_axis] = 3
        concat_dims[self.vae_parallel_config.width_parallel.mesh_axis] = 4
        video_BCTHW = fast_device_to_host(
            tt_video_BCTHW, self.mesh_device, concat_dims, ccl_manager=self.vae_ccl_manager
        )
        return video_BCTHW[:, :, :, :new_logical_h, :].float()  # crop to logical_h, keep float

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        image_prompt: Image.Image,
        audio_prompt: str,
        *,
        negative_prompt: str = (
            "画面模糊，最差质量，画面模糊，细节模糊不清，情绪激动剧烈，手快速抖动，字幕，丑陋的，残缺的，"
            "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
            "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        ),
        height: int = 480,
        width: int = 832,
        num_clips: Optional[int] = None,
        num_inference_steps: int = 40,
        guidance_scale: float = 4.5,
        guidance_scale_2: Optional[float] = None,
        seed: Optional[int] = None,
        output_type: str = "uint8",
        return_dict: bool = True,
        max_sequence_length: int = 512,
        # Match t2v at 4×8: traced=False. T2V only enables trace at 4×32
        # (test_performance_wan.py:191). ``ttnn.embedding(layout=TILE_LAYOUT)``
        # in the t5 forward writes during trace capture, which fails on 4×8
        # regardless of trace_region_size / fabric_router_config — a ttnn op-
        # level constraint, not a pipeline-config gap. The s2v denoise loop
        # without trace runs in 25.9 s / 5 steps; further gains come from
        # matmul tuning + the AdaIN-fused norm1 path, not trace.
        traced: bool = False,
        profiler: Optional[BenchmarkProfiler] = None,
        profiler_iteration: int = 0,
        # accepted for parity with WanPipeline.__call__ but ignored — S2V derives
        # the latent frame count from ``num_clips`` and the reference's fixed
        # ``infer_frames=80`` per clip.
        num_frames: Optional[int] = None,  # noqa: ARG002
    ):
        """S2V entry point. Multi-clip: each clip is 80 pixel frames; per-clip
        motion latents thread the previous clip's pixel tail through the VAE
        encoder. Matches ``wan/speech2video.py:540-670`` clip-for-clip.
        """
        if audio_prompt is None:
            raise ValueError("audio_prompt (path to a .wav/.mp3 file) is required for S2V")
        if image_prompt is None or not isinstance(image_prompt, Image.Image):
            raise ValueError("image_prompt (PIL.Image) is required for S2V (reference image)")

        self._s2v_profiler = profiler
        self._s2v_profiler_iteration = profiler_iteration
        self.transformer_states[0].guidance_scale = guidance_scale
        self.transformer_states[1].guidance_scale = guidance_scale if guidance_scale_2 is None else guidance_scale_2

        def _stage(name: str):
            return profiler(name, profiler_iteration) if profiler is not None else nullcontext()

        # 1. Text encode (once across all clips). Resulting prompt + negative
        # buffers also live across clips — they're clip-invariant.
        with _stage("encoder"):
            self._prepare_text_encoder()
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                num_videos_per_prompt=1,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                max_sequence_length=max_sequence_length,
                traced=traced,
            )
        ts = self.transformer_states[0]
        self._prepare_transformer(0)
        ts.prompt_buffer = self.prepare_text_conditioning(ts.model, prompt_embeds, ts.prompt_buffer, traced)
        ts.negative_prompt_buffer = self.prepare_text_conditioning(
            ts.model, negative_prompt_embeds, ts.negative_prompt_buffer, traced
        )

        with _stage("prepare_latents"):
            # 2. Reference image VAE encode (once).
            with _stage("s2v_vae_encode_ref"):
                ref_tensor = self.video_processor.preprocess(image_prompt, height=height, width=width).to(
                    "cpu", dtype=torch.float32
                )
                ref_video = ref_tensor.unsqueeze(2)  # [1, 3, 1, H, W]
                ref_latent_torch = self._encode_normalized(ref_video)
            # Pre-denormalized form for the VAE-decode prepend on clip 0
            # (the decode path expects VAE-input-space tensors).
            ref_latents_for_decode = (
                ref_latent_torch.to(self.vae.dtype) * self._vae_latents_std + self._vae_latents_mean
            )

            # 3. Audio: wav2vec2 + bucket the whole file.
            with _stage("s2v_wav2vec2"):
                audio_BLNF_full, num_repeat = self._prepare_audio_full(audio_prompt)
            # speech2video.py:488-489 — caller can request fewer clips than
            # the audio supports (e.g. early termination); never more.
            if num_clips is None:
                num_clips = num_repeat
            else:
                num_clips = min(num_clips, num_repeat)
            logger.info(
                f"S2V multi-clip: {num_clips} clips × {self._INFER_FRAMES_PIXEL} pixels = "
                f"{num_clips * self._INFER_FRAMES_PIXEL - self._S2V_VAE_CLIP0_TRIM} final frames"
            )

            # Lazy second-shape warmup. __init__ only warmed the 1-clip path
            # (drop_first_motion=True). On the first real call that needs
            # num_clips>1, run a one-shot 2-clip warmup to populate the
            # clip-1+ shape caches (drop_first_motion=False, N_motion>0).
            # Subsequent calls hit fully-warm caches. ``_warming_up`` guards
            # recursion when this dispatcher is reached from inside the
            # warmup invocation itself. The recursive call mutates several
            # self.* fields set at the top of __call__ — restore them after.
            if num_clips > 1 and not self._multi_clip_warmed and not self._warming_up:
                logger.info("Lazy multi-clip warmup: warming clip-1+ shape signature")
                self.warmup_buffers(height=height, width=width, num_clips=2)
                self._s2v_profiler = profiler
                self._s2v_profiler_iteration = profiler_iteration
                self.transformer_states[0].guidance_scale = guidance_scale
                self.transformer_states[1].guidance_scale = (
                    guidance_scale if guidance_scale_2 is None else guidance_scale_2
                )

            # 4. Initialize motion state. Clip 0 always has ``drop_first_motion=True``
            # which prepends ``ref_latents_for_decode`` instead of ``motion_latents``
            # in :meth:`_decode_clip`, so the initial motion latents are never read.
            # Use a zero placeholder of the right latent shape (skips a 8.6 s VAE
            # encode of 73 zero pixel frames).
            videos_last_frames = torch.zeros(1, 3, self._MOTION_FRAMES_PIXEL, height, width, dtype=torch.float32)
            latent_h = height // self.vae_scale_factor_spatial
            latent_w = width // self.vae_scale_factor_spatial
            motion_latents_torch = torch.zeros(
                1, self.vae.config.z_dim, self._LAT_MOTION_FRAMES, latent_h, latent_w, dtype=torch.float32
            )

            if seed is None:
                seed = int(torch.seed())

        # 5. Per-clip loop.
        clip_videos: list[torch.Tensor] = []
        for clip_idx in range(num_clips):
            drop_first_motion = clip_idx == 0
            with _stage(f"s2v_clip_{clip_idx}_total"):
                audio_clip = audio_BLNF_full[
                    ..., clip_idx * self._INFER_FRAMES_PIXEL : (clip_idx + 1) * self._INFER_FRAMES_PIXEL
                ]
                latents = self._prepare_clip(
                    clip_idx=clip_idx,
                    audio_clip=audio_clip,
                    ref_latent_torch=ref_latent_torch,
                    motion_latents_torch=motion_latents_torch,
                    drop_first_motion=drop_first_motion,
                    height=height,
                    width=width,
                    seed=seed,
                    _stage=_stage,
                )

                with _stage(f"s2v_clip_{clip_idx}_denoise"):
                    latents = self._denoise_clip(
                        latents_torch=latents,
                        num_inference_steps=num_inference_steps,
                        traced=traced,
                    )

                with _stage(f"s2v_clip_{clip_idx}_vae_decode"):
                    # Clip 0 with drop_first_motion prepends ref; everything
                    # else prepends the previous clip's motion latents.
                    if drop_first_motion:
                        prepend_for_decode = ref_latents_for_decode
                    else:
                        prepend_for_decode = (
                            motion_latents_torch.to(self.vae.dtype) * self._vae_latents_std + self._vae_latents_mean
                        )
                    image_BCTHW = self._decode_clip(
                        latents_torch=latents,
                        prepend_latents_torch=prepend_for_decode,
                    )

                # Trim per reference 654-656.
                image_BCTHW = image_BCTHW[:, :, -self._INFER_FRAMES_PIXEL :]
                if drop_first_motion:
                    image_BCTHW = image_BCTHW[:, :, self._S2V_VAE_CLIP0_TRIM :]

                # Slide videos_last_frames forward (reference 658-663) only if
                # there's a next clip that will consume it. On the final clip we
                # skip the motion-tail VAE encode entirely (~6.5 s saved).
                if clip_idx + 1 < num_clips:
                    overlap = min(self._MOTION_FRAMES_PIXEL, image_BCTHW.shape[2])
                    videos_last_frames = torch.cat(
                        [
                            videos_last_frames[:, :, overlap:],
                            image_BCTHW[:, :, -overlap:],
                        ],
                        dim=2,
                    )
                    with _stage(f"s2v_clip_{clip_idx}_vae_encode_motion"):
                        motion_latents_torch = self._encode_normalized(videos_last_frames)

                clip_videos.append(image_BCTHW)

        # 6. Concat clip outputs (reference 670).
        videos_BCTHW = torch.cat(clip_videos, dim=2)

        # 7. Final dtype / layout conversion.
        if output_type == "latent":
            video: object = videos_BCTHW
        elif output_type == "uint8":
            video_BTHWC = videos_BCTHW.permute(0, 2, 3, 4, 1)
            video = ((video_BTHWC.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).numpy()
        elif output_type == "np":
            video_BTHWC = videos_BCTHW.permute(0, 2, 3, 4, 1)
            video = ((video_BTHWC.clamp(-1, 1) + 1) * 0.5).numpy()
        else:
            video = self.video_processor.postprocess_video(videos_BCTHW, output_type=output_type)

        self._s2v_profiler = None
        if not return_dict:
            return (video,)
        return WanPipelineOutput(frames=video)

    def get_model_input(self, latents, cond_latents):
        return super().get_model_input(latents, None)

    @staticmethod
    def export(frames, output_path: str, *, audio_path: Optional[str] = None, fps: int = 16) -> str:
        """Encode pipeline output to MP4, optionally muxing in ``audio_path``."""
        if audio_path is not None:
            return export_to_video_with_audio(frames, output_path, audio_path=audio_path, fps=fps)
        return export_to_video(frames, output_path, fps=fps)
