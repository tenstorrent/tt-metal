# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan.py

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import tqdm
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.video_processor import VideoProcessor
from loguru import logger

import ttnn
from models.tt_dit.models.transformers.wan2_2.transformer_wan import WanCheckpoint, WanTransformer3DModel
from models.tt_dit.models.vae.vae_wan2_1 import WanVAEDecoderAdapter
from models.tt_dit.parallel.config import DiTParallelConfig, EncoderParallelConfig, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.events import DenoiseStep, PipelineEventCallback, SectionEnd, SectionStart, null_callback
from models.tt_dit.pipelines.pipeline_api import PipelineAPIMixin
from models.tt_dit.pipelines.wan.dbcache import ExpertCacheRunner, WanDBCacheConfig
from models.tt_dit.pipelines.wan.text_encoder import TextEncoder
from models.tt_dit.solvers import solver_for_scheduler
from models.tt_dit.utils import tensor
from models.tt_dit.utils.conv3d import conv3d_configs
from models.tt_dit.utils.dbcache import DBCacheConfig
from models.tt_dit.utils.tensor import float32_tensor
from models.tt_dit.utils.torch_threads import cap_torch_threads_to_half_the_cpus
from models.tt_dit.utils.tracing import Tracer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from diffusers.schedulers import SchedulerMixin

_UNSET = object()  # sentinel for "use config default" in WanPipelineConfig.default

_DEFAULT_CHECKPOINT = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

# The supported video sizes, (width, height): Wan 2.2's 480p and 720p sizes in 16:9 and 9:16.
_RESOLUTIONS_480P = ((832, 480), (480, 832))
_RESOLUTIONS_720P = ((1280, 720), (720, 1280))

# The spatial lengths the transformer runs at, in tokens per latent frame: those of the 480p and
# 720p sizes. Each size is padded to the smallest that fits it, so sizes with the same token count
# share traces and buffers.
_SPATIAL_BUCKETS_PER_FRAME = (1560, 3600)

_DEFAULT_RESOLUTION = (832, 480)

_PRESETS_WH: dict[tuple[int, ...], dict] = {
    (2, 4): {
        "sp_axis": 0,
        "tp_axis": 1,
        "num_links": 1,
        "dynamic_load": True,
        "topology": ttnn.Topology.Linear,
        "is_fsdp": True,
    },
    (4, 8): {
        "sp_axis": 1,
        "tp_axis": 0,
        "num_links": 4,
        "dynamic_load": False,
        "topology": ttnn.Topology.Ring,
        "is_fsdp": True,
    },
}

_PRESETS_BH: dict[tuple[int, ...], dict] = {
    (1, 4): {
        "sp_axis": 0,
        "tp_axis": 1,
        "num_links": 2,
        "dynamic_load": False,
        "topology": ttnn.Topology.Linear,
        "is_fsdp": True,
    },
    (2, 2): {
        "sp_axis": 0,
        "tp_axis": 1,
        "num_links": 2,
        "dynamic_load": False,
        "topology": ttnn.Topology.Linear,
        "is_fsdp": True,
        # 720p runs out of DRAM in the VAE decoder.
        "resolutions": _RESOLUTIONS_480P,
        "spatial_buckets_per_frame": _SPATIAL_BUCKETS_PER_FRAME,
    },
    (2, 4): {
        "sp_axis": 1,
        "tp_axis": 0,
        "num_links": 2,
        "dynamic_load": True,
        "topology": ttnn.Topology.Linear,
        "is_fsdp": False,
        "vae_t_chunk_size": 7,
    },
    (4, 8): {
        "sp_axis": 1,
        "tp_axis": 0,
        "num_links": 2,
        "dynamic_load": False,
        "topology": ttnn.Topology.Ring,
        "is_fsdp": False,
        "vae_t_chunk_size": None,  # full-T
        "resolutions": _RESOLUTIONS_720P + _RESOLUTIONS_480P,
        "spatial_buckets_per_frame": _SPATIAL_BUCKETS_PER_FRAME,
    },
    (4, 32): {
        "sp_axis": 1,
        "tp_axis": 0,
        "num_links": 2,
        "dynamic_load": False,
        "topology": ttnn.Topology.Ring,
        "is_fsdp": False,
        "vae_t_chunk_size": None,
        "sdpa_t_fracture_w_only": True,
    },
}


@dataclass(frozen=True, kw_only=True)
class WanPipelineConfig:
    topology: ttnn.Topology
    num_links: int

    dit_parallel_config: DiTParallelConfig
    encoder_parallel_config: EncoderParallelConfig
    vae_parallel_config: VaeHWParallelConfig

    boundary_ratio: float | None
    expand_timesteps: bool
    dynamic_load: bool
    is_fsdp: bool
    model_type: str
    vae_dtype: ttnn.DataType
    vae_t_chunk_size: int | None
    sdpa_t_fracture_w_only: bool

    # The supported sizes, (width, height). The first is the default.
    resolutions: tuple[tuple[int, int], ...]
    # The spatial lengths the transformer runs at, in tokens per latent frame. Each size is padded
    # to the smallest that fits it; a size larger than all of them runs unpadded.
    spatial_buckets_per_frame: tuple[int, ...]
    num_frames: int
    cfg_enabled: bool
    max_sequence_length: int

    checkpoint_name: str

    # DBCache (cache-dit style) step skipping, applied to every call unless the call passes its own
    # `cache_config` (``None`` disables caching for that call). `WanPipelineConfig.default` fills
    # this with `WanDBCacheConfig.default()` (the I2V recipe) for ``model_type="i2v"`` and leaves T2V
    # uncached; pass ``cache_config`` there explicitly to override either way.
    cache_config: WanDBCacheConfig | None = None

    @classmethod
    def default(
        cls,
        *,
        mesh_shape: ttnn.MeshShape,
        topology: ttnn.Topology | None = None,
        num_links: int | None = None,
        dit_parallel_config: DiTParallelConfig | None = None,
        encoder_parallel_config: EncoderParallelConfig | None = None,
        vae_parallel_config: VaeHWParallelConfig | None = None,
        boundary_ratio: float | None = None,
        expand_timesteps: bool = False,
        dynamic_load: bool | None = None,
        is_fsdp: bool | None = None,
        model_type: str = "t2v",
        vae_dtype: ttnn.DataType = ttnn.bfloat16,
        vae_t_chunk_size: object = _UNSET,
        sdpa_t_fracture_w_only: bool | None = None,
        height: int | None = None,
        width: int | None = None,
        resolutions: Sequence[tuple[int, int]] | None = None,
        spatial_buckets_per_frame: Sequence[int] | None = None,
        num_frames: int = 81,
        cfg_enabled: bool = True,
        max_sequence_length: int = 512,
        checkpoint_name: str = _DEFAULT_CHECKPOINT,
        cache_config: WanDBCacheConfig | DBCacheConfig | None = _UNSET,
    ) -> WanPipelineConfig:
        """The preset for ``mesh_shape``.

        ``height`` and ``width`` configure that single size, and ``resolutions`` a list of
        (width, height); otherwise the preset's sizes apply, by default 832x480. The spatial
        buckets default to the preset's when its sizes apply, and to none otherwise.
        """
        preset_dict = _PRESETS_BH if ttnn.device.is_blackhole() else _PRESETS_WH
        preset = preset_dict.get(tuple(mesh_shape), {})

        if (height is None) != (width is None):
            msg = "pass both height and width, or neither"
            raise ValueError(msg)
        if height is not None and resolutions is not None:
            msg = "pass height and width or resolutions, not both"
            raise ValueError(msg)
        if height is not None and width is not None:
            resolutions = ((width, height),)
        if resolutions is None:
            resolutions = preset.get("resolutions", (_DEFAULT_RESOLUTION,))
            if spatial_buckets_per_frame is None:
                spatial_buckets_per_frame = preset.get("spatial_buckets_per_frame", ())
        if spatial_buckets_per_frame is None:
            spatial_buckets_per_frame = ()
        if not resolutions:
            msg = "resolutions must not be empty"
            raise ValueError(msg)

        if cache_config is _UNSET:
            # The default recipe is validated on I2V only (T2V drifts past the A/B gate with it: PCC 0.90 at
            # 480p and 0.71 at 720p, see `WanDBCacheConfig.default`), and the residual-diff readback is
            # single-host only (`ExpertCacheRunner.read_sums`). T2V and multi-host meshes stay uncached
            # unless the caller passes a cache_config explicitly.
            i2v_single_host = model_type == "i2v" and not ttnn.using_distributed_env()
            cache_config = WanDBCacheConfig.default() if i2v_single_host else None

        if boundary_ratio is None:
            # The checkpoints' own, from their model_index.json.
            boundary_ratio = 0.9 if model_type == "i2v" else 0.875

        if dit_parallel_config is None or vae_parallel_config is None or encoder_parallel_config is None:
            sp_axis = preset["sp_axis"]
            tp_axis = preset["tp_axis"]
            h_factor = tuple(mesh_shape)[tp_axis]
            w_factor = tuple(mesh_shape)[sp_axis]
            if dit_parallel_config is None:
                dit_parallel_config = DiTParallelConfig.from_tuples(
                    cfg=(1, 0), sp=(w_factor, sp_axis), tp=(h_factor, tp_axis)
                )
            if vae_parallel_config is None:
                vae_parallel_config = VaeHWParallelConfig.from_tuples(
                    height=(h_factor, tp_axis), width=(w_factor, sp_axis)
                )
            if encoder_parallel_config is None:
                encoder_parallel_config = EncoderParallelConfig.from_tuple((h_factor, tp_axis))

        if vae_t_chunk_size is _UNSET:
            vae_t_chunk_size = preset.get("vae_t_chunk_size", 1)

        return cls(
            topology=topology if topology is not None else preset["topology"],
            num_links=num_links if num_links is not None else preset["num_links"],
            dit_parallel_config=dit_parallel_config,
            encoder_parallel_config=encoder_parallel_config,
            vae_parallel_config=vae_parallel_config,
            boundary_ratio=boundary_ratio,
            expand_timesteps=expand_timesteps,
            dynamic_load=dynamic_load if dynamic_load is not None else preset["dynamic_load"],
            is_fsdp=is_fsdp if is_fsdp is not None else preset["is_fsdp"],
            model_type=model_type,
            vae_dtype=vae_dtype,
            vae_t_chunk_size=vae_t_chunk_size,
            sdpa_t_fracture_w_only=(
                sdpa_t_fracture_w_only
                if sdpa_t_fracture_w_only is not None
                else preset.get("sdpa_t_fracture_w_only", False)
            ),
            resolutions=tuple((w, h) for w, h in resolutions),
            spatial_buckets_per_frame=tuple(spatial_buckets_per_frame),
            num_frames=num_frames,
            cfg_enabled=cfg_enabled,
            max_sequence_length=max_sequence_length,
            checkpoint_name=checkpoint_name,
            cache_config=WanDBCacheConfig.coerce(cache_config) if cache_config is not None else None,
        )


@dataclass
class TransformerState:
    model: WanTransformer3DModel
    checkpoint: WanCheckpoint
    guidance_scale: float
    prompt_buffer: object = field(default=None)
    negative_prompt_buffer: object = field(default=None)
    # By padded spatial sequence length.
    cache_runners: dict[int, ExpertCacheRunner] = field(default_factory=dict)


class WanPipeline(PipelineAPIMixin):
    r"""Pipeline for text-to-video generation using Wan.

    Args:
        mesh_device (`ttnn.MeshDevice`):
            The TT mesh device to run inference on.
        parallel_config (`DiTParallelConfig`):
            Parallelism configuration for the transformer.
        vae_parallel_config (`VaeHWParallelConfig`):
            Parallelism configuration for the VAE decoder.
        encoder_parallel_config (`EncoderParallelConfig`):
            Parallelism configuration for the text encoder.
        num_links (`int`):
            Number of links to use for CCL operations.
        checkpoint_name (`str`, *optional*, defaults to `"Wan-AI/Wan2.2-T2V-A14B-Diffusers"`):
            HuggingFace Hub repo ID to load model weights from.
        scheduler (`SchedulerMixin`, *optional*):
            Scheduler to use for denoising; it also selects the solver family. Defaults to
            `UniPCMultistepScheduler` loaded from the checkpoint.
        boundary_ratio (`float`, *optional*, defaults to `0.875` for T2V and `0.9` for I2V):
            Ratio of total timesteps used as the boundary for switching between the two transformers in two-stage
            denoising. `transformer` handles timesteps >= boundary_timestep and `transformer_2` handles timesteps <
            boundary_timestep. If `None`, only `transformer` is used for the entire denoising process.
        expand_timesteps (`bool`, *optional*, defaults to `False`):
            Whether to expand timesteps per-token for image-to-video (Wan2.2 TI2V) conditioning.
        dynamic_load (`bool`, *optional*, defaults to `False`):
            If `True`, model components are loaded/offloaded to device dynamically during inference.
        topology (`ttnn.Topology`, *optional*, defaults to `ttnn.Topology.Linear`):
            Fabric topology to use for CCL operations across devices.
        is_fsdp (`bool`, *optional*, defaults to `True`):
            Whether to use fully-sharded data parallelism for transformer weights.
        model_type (`str`, *optional*, defaults to `"t2v"`):
            Model variant identifier (e.g. `"t2v"` for text-to-video).
        vae_dtype (`ttnn.DataType`, *optional*, defaults to `ttnn.bfloat16`):
            Data type to use for VAE inference.
        vae_use_cache (`bool`, *optional*, defaults to `True`):
            Whether to cache VAE convolution programs across calls.
        sdpa_t_fracture_w_only (`bool`, *optional*, defaults to `False`):
            Whether to fracture SDPA only along the width dimension for temporal attention.
    """

    @classmethod
    def _config_overrides(cls) -> dict[str, object]:
        """`WanPipelineConfig.default` fields this variant pins, keyed by argument name.

        A variant overrides this instead of reimplementing `create_pipeline`, and layers on
        top via ``{**super()._config_overrides(), ...}``. A caller's `config_overrides` wins
        over anything returned here.
        """
        return {"model_type": "t2v", "checkpoint_name": _DEFAULT_CHECKPOINT}

    @classmethod
    def default_config(
        cls,
        *,
        mesh_device: ttnn.MeshDevice,
        checkpoint_name: str | None = None,
        height: int | None = None,
        width: int | None = None,
        resolutions: Sequence[tuple[int, int]] | None = None,
        num_frames: int = 81,
        cfg_enabled: bool = True,
        max_sequence_length: int = 512,
        config_overrides: dict[str, object] | None = None,
    ) -> WanPipelineConfig:
        """The config this variant runs by default, with `config_overrides` layered on top.

        `config_overrides` takes any `WanPipelineConfig.default` keyword. Pair this with a
        direct constructor call to build a variant whose `__init__` takes more than the
        arguments `create_pipeline` knows about.
        """
        overrides = {**cls._config_overrides(), **(config_overrides or {})}
        if checkpoint_name is not None:
            overrides["checkpoint_name"] = checkpoint_name
        return WanPipelineConfig.default(
            mesh_shape=mesh_device.shape,
            height=height,
            width=width,
            resolutions=resolutions,
            num_frames=num_frames,
            cfg_enabled=cfg_enabled,
            max_sequence_length=max_sequence_length,
            **overrides,
        )

    @classmethod
    def create_pipeline(
        cls,
        *,
        mesh_device: ttnn.MeshDevice,
        checkpoint_name: str | None = None,
        height: int | None = None,
        width: int | None = None,
        resolutions: Sequence[tuple[int, int]] | None = None,
        num_frames: int = 81,
        cfg_enabled: bool = True,
        max_sequence_length: int = 512,
        config_overrides: dict[str, object] | None = None,
        scheduler: SchedulerMixin | None = None,
        run_warmup: bool = True,
        lora_enabled: bool = False,
        pipeline_class: type[WanPipeline] | None = None,
    ) -> WanPipeline:
        """Build a pipeline from this variant's defaults plus the given overrides."""
        # Resolved off the class being constructed, not `cls`, so `pipeline_class` gets its
        # own overrides rather than those of whichever class the call was made on.
        pipeline_class_ = pipeline_class or cls
        config = pipeline_class_.default_config(
            mesh_device=mesh_device,
            checkpoint_name=checkpoint_name,
            height=height,
            width=width,
            resolutions=resolutions,
            num_frames=num_frames,
            cfg_enabled=cfg_enabled,
            max_sequence_length=max_sequence_length,
            config_overrides=config_overrides,
        )
        return pipeline_class_(
            device=mesh_device,
            config=config,
            scheduler=scheduler,
            run_warmup=run_warmup,
            lora_enabled=lora_enabled,
        )

    def __init__(
        self,
        *,
        device: ttnn.MeshDevice,
        config: WanPipelineConfig,
        scheduler: SchedulerMixin | None = None,
        run_warmup: bool = True,
        lora_enabled: bool = False,
    ) -> None:
        super().__init__()

        self.checkpoint_name = config.checkpoint_name
        self.model_type = config.model_type
        self._cfg_enabled = config.cfg_enabled
        self._resolutions = config.resolutions
        self._spatial_buckets_per_frame = sorted(set(config.spatial_buckets_per_frame))
        self._num_frames = config.num_frames
        for width, height in self._resolutions:
            if height % 16 != 0 or width % 16 != 0:
                msg = f"`height` and `width` have to be divisible by 16 but are {height} and {width}."
                raise ValueError(msg)

        self._checkpoint = WanCheckpoint(config.checkpoint_name, subfolder="transformer")
        self._checkpoint_2 = WanCheckpoint(config.checkpoint_name, subfolder="transformer_2")

        self.dit_ccl_manager = CCLManager(
            mesh_device=device,
            num_links=config.num_links,
            topology=config.topology,
        )
        self.vae_ccl_manager = CCLManager(
            mesh_device=device,
            num_links=config.num_links,
            topology=ttnn.Topology.Linear,  # NOTE: VAE always uses Linear topology. TODO: enable ring if given.
        )

        # See what options we have for topology. We should consider reusing CCL managers
        self.encoder_ccl_manager = self.vae_ccl_manager

        self.is_fsdp = config.is_fsdp
        self.parallel_config = config.dit_parallel_config
        self.vae_parallel_config = config.vae_parallel_config
        self.encoder_parallel_config = config.encoder_parallel_config
        self.mesh_device = device
        self.dynamic_load = config.dynamic_load

        self._text_encoder = TextEncoder(
            checkpoint_name=config.checkpoint_name,
            device=self.mesh_device,
            ccl_manager=self.encoder_ccl_manager,
            encoder_parallel_config=self.encoder_parallel_config,
            dit_parallel_config=self.parallel_config,
            max_sequence_length=config.max_sequence_length,
        )

        self.lora_enabled = lora_enabled

        self.transformer = self._checkpoint.build(
            ccl_manager=self.dit_ccl_manager,
            parallel_config=self.parallel_config,
            is_fsdp=self.is_fsdp,
            model_type=self.model_type,
            lora_enabled=lora_enabled,
        )

        self.transformer_2 = self._checkpoint_2.build(
            ccl_manager=self.dit_ccl_manager,
            parallel_config=self.parallel_config,
            is_fsdp=self.is_fsdp,
            model_type=self.model_type,
            lora_enabled=lora_enabled,
        )

        # The decoder picks its conv3d blockings by size, so each size gets one, but sizes with
        # the same blockings share it.
        self._vaes: dict[tuple[int, int], WanVAEDecoderAdapter] = {}
        vaes_by_blockings: dict[tuple[str, ...], WanVAEDecoderAdapter] = {}
        for width, height in self._resolutions:
            vae = WanVAEDecoderAdapter(
                checkpoint_name=config.checkpoint_name,
                parallel_config=self.vae_parallel_config,
                ccl_manager=self.vae_ccl_manager,
                height=height,
                width=width,
                num_frames=config.num_frames,
                vae_t_chunk_size=config.vae_t_chunk_size,
                vae_dtype=config.vae_dtype,
                sdpa_t_fracture_w_only=config.sdpa_t_fracture_w_only,
            )
            self._vaes[width, height] = vaes_by_blockings.setdefault(conv3d_configs(vae.decoder), vae)
        logger.info(f"{len(vaes_by_blockings)} VAE decoder(s) for {len(self._resolutions)} resolution(s)")
        self._vae = self._vaes[self._resolutions[0]]
        vae_decoders = [vae.decoder for vae in vaes_by_blockings.values()]

        self.transformer_states = [
            TransformerState(self.transformer, self._checkpoint, guidance_scale=4.0),
            TransformerState(self.transformer_2, self._checkpoint_2, guidance_scale=3.0),
        ]

        self._solver = solver_for_scheduler(
            scheduler
            # flow_shift 5.0 (cache-dit's / vLLM's 720p setting) instead of the official Wan 2.2
            # A14B T2V value of 12.0: it spreads the schedule toward low noise, which is what lets
            # DBCache skip 15-16 of 40 steps instead of 8 (see models/tt_dit/models/Wan2_2.md).
            # Per call: `pipeline(..., flow_shift=12.0)`.
            or UniPCMultistepScheduler.from_pretrained(self.checkpoint_name, subfolder="scheduler", flow_shift=5.0)
        )

        if self.dynamic_load:
            # setup models that cannot be loaded together with the corresponding model.
            # The module loading utility will take care of the necessary unloading.
            if ttnn.device.is_blackhole():
                self.transformer.register_coresident_exclusions(self.transformer_2)
                self.transformer_2.register_coresident_exclusions(self.transformer)
            else:
                # WH T3K has tighter DRAM — include VAE in the unload chain so
                # transformers and VAE never coexist in DRAM across pipeline runs.
                self.transformer.register_coresident_exclusions(self.transformer_2, *vae_decoders)
                self.transformer_2.register_coresident_exclusions(self.transformer, *vae_decoders)
                for decoder in vae_decoders:
                    decoder.register_coresident_exclusions(self.transformer, self.transformer_2)

        # Cache warmup: Load in reverse order of use to ensure the earliest required models stay loaded before call.
        self._prepare_transformer(1)
        self._prepare_transformer(0)
        self._prepare_text_encoder()
        for vae in vaes_by_blockings.values():
            vae.reload_weights()

        self._boundary_ratio = config.boundary_ratio
        self._expand_timesteps = config.expand_timesteps
        self._cache_config = config.cache_config
        self._last_cache_runners: list[ExpertCacheRunner] = []
        self.vae_scale_factor_temporal = self._vae.config.scale_factor_temporal
        self.vae_scale_factor_spatial = self._vae.config.scale_factor_spatial
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        # The spatial sequence each size is padded to. Sizes padded to the same length share
        # traces and the persistent buffers below.
        self._padded_sequence_lengths = {r: self._padded_sequence_length(*r) for r in self._resolutions}
        padded_lengths = sorted(set(self._padded_sequence_lengths.values()))
        sp_axis = self.parallel_config.sequence_parallel.mesh_axis

        # Persistent tensors, allocated before any trace is captured so that no captured trace
        # treats their memory as scratch. Each call writes the unpadded spatial length and the mask
        # of its rows; the latents, the image conditioning and the RoPE tables are copied into
        # per-length slots, since a tracer copies its inputs into the tensors it captured, which
        # must not be the RoPE tables cached for one particular size.
        self._sequence_length = tensor.from_torch(
            torch.tensor([0]), device=self.mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self._valid_rows = {
            n: tensor.from_torch(
                torch.zeros([1, 1, n, 1]), device=self.mesh_device, mesh_axes=[None, None, sp_axis, None]
            )
            for n in padded_lengths
        }
        self._latent_buffers: dict[int, ttnn.Tensor] = {}
        self._condition_buffers: dict[int, ttnn.Tensor] = {}
        self._rope_slots: dict[int, tuple[ttnn.Tensor, ttnn.Tensor]] = {}

        # One tracer per expert and padded length.
        self._step_tracers = {
            (idx, n): Tracer(ts.model.combined_step, device=self.mesh_device, prep_run=False, clone_prep_inputs=False)
            for idx, ts in enumerate(self.transformer_states)
            for n in padded_lengths
        }

        if run_warmup:
            self._warmup()

    @property
    def resolutions(self) -> tuple[tuple[int, int], ...]:
        """The (width, height) sizes a call may ask for; the first is the default."""
        return self._resolutions

    def _warmup(self) -> None:
        """Runs the pipeline once at each size, untraced.

        That compiles the programs of each size and allocates the persistent buffers of each padded
        length before the first trace capture.
        """
        for width, height in self._resolutions:
            logger.info(f"Pipeline allocation run at {width}x{height}...")
            self(
                prompts=["warmup"],
                num_inference_steps=2,
                guidance_scale=2 if self._cfg_enabled else 1,
                guidance_scale_2=2 if self._cfg_enabled else 1,
                width=width,
                height=height,
                **self._warmup_kwargs(width, height),
            )

    def _warmup_kwargs(self, width: int, height: int) -> dict[str, object]:
        """Extra arguments of the warmup run at a size; the I2V image."""
        del width, height
        return {}

    def _padded_sequence_length(self, width: int, height: int) -> int:
        """The spatial sequence length a size runs at: padded to its bucket, and to whole tiles on
        each sequence-parallel device.

        Without sequence parallelism, attention does not mask padding, so sizes are not bucketed.
        """
        pF, pH, pW = self.transformer.patch_size
        frames = ((self._num_frames - 1) // self.vae_scale_factor_temporal + 1) // pF
        tokens_per_frame = (height // self.vae_scale_factor_spatial // pH) * (
            width // self.vae_scale_factor_spatial // pW
        )

        sp_factor = self.parallel_config.sequence_parallel.factor
        if sp_factor > 1:
            tokens_per_frame = min(
                (b for b in self._spatial_buckets_per_frame if b >= tokens_per_frame), default=tokens_per_frame
            )

        divisor = ttnn.TILE_SIZE * sp_factor
        return -(-frames * tokens_per_frame // divisor) * divisor

    def _resolve_size(self, width: int | None, height: int | None, image_prompt: object) -> tuple[int, int]:
        """The size of a call: the given one, which must be configured, or by default the first."""
        del image_prompt
        if (width is None) != (height is None):
            msg = "pass both width and height, or neither"
            raise ValueError(msg)
        if width is None or height is None:
            return self._resolutions[0]
        if (width, height) not in self._padded_sequence_lengths:
            msg = f"unsupported size {width}x{height}, supported are {list(self._resolutions)}"
            raise ValueError(msg)
        return width, height

    def prepare_text_conditioning(self, tt_model, prompt_embeds, buffer, traced=False):
        prompt_1BLP = tt_model.prepare_text_conditioning(prompt_embeds)
        if buffer is None or not traced:
            buffer = prompt_1BLP
        else:
            ttnn.copy(prompt_1BLP, buffer)
        return buffer

    def _prepare_text_encoder(self) -> None:
        self._text_encoder.prepare()

    def _prepare_transformer(self, idx: int) -> None:
        state = self.transformer_states[idx]
        state.checkpoint.load(
            state.model,
            mesh_device=self.mesh_device,
            parallel_config=self.parallel_config,
            is_fsdp=self.is_fsdp,
        )

    def _step(
        self,
        *,
        step: int,
        t: float,
        transformer_idx: int,
        padded_sequence_length: int,
        permuted_latent_tt: ttnn.Tensor,
        mask: torch.Tensor,
        cond_latents: ttnn.Tensor | None,
        rope_args: dict,
        latents_batch_size: int,
        cfg_enabled: bool,
        traced: bool,
    ) -> ttnn.Tensor:
        ts = self.transformer_states[transformer_idx]
        timestep, guidance_scale_tt = self._timestep_and_guidance(
            t=t, ts=ts, mask=mask, latents_batch_size=latents_batch_size, traced=traced
        )
        permuted_model_input = self.get_model_input(permuted_latent_tt, cond_latents)

        # A tracer runs untraced only until it captures.
        step_fn = (
            functools.partial(self._step_tracers[transformer_idx, padded_sequence_length], traced=True)
            if traced
            else ts.model.combined_step
        )
        permuted_velocity_pred_tt = step_fn(
            do_classifier_free_guidance=cfg_enabled,
            spatial_1BNI=permuted_model_input,
            prompt_1BLP=ts.prompt_buffer,
            negative_prompt_1BLP=ts.negative_prompt_buffer,
            N=self._sequence_length,
            timestep=timestep,
            **rope_args,
            guidance_scale=guidance_scale_tt,
            gather_output=False,
        )

        return self._solver.step(
            step=step,
            latent=permuted_latent_tt,
            velocity_pred=permuted_velocity_pred_tt,
        )

    def _timestep_and_guidance(
        self, *, t: float, ts: TransformerState, mask: torch.Tensor, latents_batch_size: int, traced: bool
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Shared by `_step` and `_step_cached`: the 1D timestep tensor and the guidance scalar.

        Both are host tensors under tracing (the tracer copies them into its captured input slots)
        and device tensors otherwise.
        """
        if self._expand_timesteps:
            # seq_len: num_latent_frames * latent_height//2 * latent_width//2
            temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
            # batch_size, seq_len
            timestep = temp_ts.unsqueeze(0).expand(latents_batch_size, -1)
        else:
            timestep = torch.full((latents_batch_size,), t, dtype=torch.float32)

        assert timestep.ndim == 1, "Wan2.2-T2V/I2V requires a 1D timestep tensor"
        timestep = float32_tensor(
            timestep.unsqueeze(1).unsqueeze(1).unsqueeze(1), device=(None if traced else self.mesh_device)
        )

        # guidance_scale is passed as a 1-element device tensor (broadcast via ttnn.lerp's
        # tensor-weight overload) so it can be updated in place between traced executions,
        # mirroring how `timestep` above is threaded through the captured trace.
        guidance_scale_tt = float32_tensor(
            torch.tensor(ts.guidance_scale, dtype=torch.float32).reshape(1, 1, 1, 1),
            device=(None if traced else self.mesh_device),
        )
        return timestep, guidance_scale_tt

    def _step_cached(
        self,
        *,
        step: int,
        t: float,
        transformer_idx: int,
        padded_sequence_length: int,
        permuted_latent_tt: ttnn.Tensor,
        mask: torch.Tensor,
        cond_latents: ttnn.Tensor | None,
        rope_args: dict,
        latents_batch_size: int,
        cfg_enabled: bool,
        traced: bool,
    ) -> ttnn.Tensor:
        """DBCache variant of `_step`: head -> (body | cached residual) -> tail per CFG branch."""
        ts = self.transformer_states[transformer_idx]
        runner = ts.cache_runners[padded_sequence_length]

        timestep, guidance_scale_tt = self._timestep_and_guidance(
            t=t, ts=ts, mask=mask, latents_batch_size=latents_batch_size, traced=traced
        )
        permuted_model_input = self.get_model_input(permuted_latent_tt, cond_latents)

        prompts = [ts.prompt_buffer, ts.negative_prompt_buffer] if cfg_enabled else [ts.prompt_buffer]
        velocities = runner.run_step(
            spatial_1BNI=permuted_model_input,
            prompts_1BLP=prompts,
            N=self._sequence_length,
            timestep=timestep,
            traced=traced,
            gather_output=False,
            valid_11N1=self._valid_rows[padded_sequence_length],
            **rope_args,
        )

        if cfg_enabled:
            if guidance_scale_tt.device() is None:
                # The CFG lerp runs outside the traces here, so it needs a device tensor.
                guidance_scale_tt = guidance_scale_tt.to(self.mesh_device)
            cond, uncond = velocities
            permuted_velocity_pred_tt = ttnn.lerp(uncond, cond, guidance_scale_tt)
        else:
            (permuted_velocity_pred_tt,) = velocities

        return self._solver.step(
            step=step,
            latent=permuted_latent_tt,
            velocity_pred=permuted_velocity_pred_tt,
        )

    def _cache_runners(
        self, cache_config: WanDBCacheConfig | DBCacheConfig | None, padded_sequence_length: int
    ) -> list[ExpertCacheRunner] | None:
        """Get (or build) the per-expert cache runners of a padded length for ``cache_config`` and
        reset their contexts.

        Returns ``None`` when caching is disabled. Runners are kept on the `TransformerState`s, one
        per padded length, and all of an expert's are rebuilt (releasing their traces) when the
        block split (Fn/Bn) or the TaylorSeer order changes; other knobs are host-side and take
        effect immediately.
        """
        if cache_config is None:
            self._last_cache_runners = []
            return None
        cache_config = WanDBCacheConfig.coerce(cache_config)
        if ttnn.using_distributed_env() and (cache_config.high_noise.enabled or cache_config.low_noise.enabled):
            # Fail before the first denoising step rather than at the first residual-diff readback.
            msg = (
                "DBCache is not supported on multi-host meshes (the residual-diff readback needs a host-side "
                "all-reduce); build or call the pipeline with cache_config=None"
            )
            raise NotImplementedError(msg)

        runners = []
        for idx, ts in enumerate(self.transformer_states):
            expert_cfg = cache_config.for_expert(idx)
            if any(
                r.config.Fn_compute_blocks != expert_cfg.Fn_compute_blocks
                or r.config.Bn_compute_blocks != expert_cfg.Bn_compute_blocks
                or r.config.taylorseer_order != expert_cfg.taylorseer_order
                for r in ts.cache_runners.values()
            ):
                for r in ts.cache_runners.values():
                    r.release_traces()
                ts.cache_runners.clear()
            runner = ts.cache_runners.get(padded_sequence_length)
            if runner is None:
                runner = ExpertCacheRunner(
                    ts.model,
                    config=expert_cfg,
                    name=("high_noise", "low_noise")[idx],
                )
                ts.cache_runners[padded_sequence_length] = runner
            else:
                runner.config = expert_cfg
                runner.context.config = expert_cfg
            runner.reset()
            runners.append(runner)
        self._last_cache_runners = runners
        return runners

    def cache_summary(self) -> list[dict[str, object]]:
        """DBCache statistics (cached steps, residual diffs, optional profile) per expert for the most recent call."""
        out = []
        for runner in self._last_cache_runners:
            summary = runner.context.summary()
            summary["profile_ms"] = runner.profile_summary()
            out.append(summary)
        return out

    def get_model_input(self, latents: ttnn.Tensor, cond_latents: ttnn.Tensor | None) -> ttnn.Tensor:
        """Adapter function to enable I2V. For base T2V, just return the latents (cast to bf16)."""
        del cond_latents
        if latents.dtype == ttnn.float32:
            latents = ttnn.typecast(latents, ttnn.bfloat16)
        return latents

    def prepare_schedule(self, num_inference_steps: int, *, flow_shift: float | None = None) -> None:
        """Adopt the schedule for one denoising run.

        Override to derive the schedule from something other than the solver's own scheduler,
        e.g. an explicit sigma ladder stepped by a scheduler-less solver.

        Args:
            num_inference_steps: Number of denoising steps.
            flow_shift: Flow shift for this run only; the solver's construction-time value is
                used when omitted.
        """
        self._solver.set_schedule(num_inference_steps, shift=flow_shift)

    def prepare_latents(
        self,
        batch_size: int,
        image_prompt=None,  # unused in T2V
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        del image_prompt, dtype

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )

        latents = torch.randn(shape, dtype=torch.float32, device=torch.device(device))
        return latents, None

    DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"  # noqa: E501, RUF001

    @torch.no_grad()
    @cap_torch_threads_to_half_the_cpus()
    def __call__(
        self,
        *,
        prompts: Sequence[str],
        negative_prompts: Sequence[str] | None = None,
        image_prompt=None,
        num_inference_steps: int,
        guidance_scale: float = 4.0,
        guidance_scale_2: float | None = 3.0,
        flow_shift: float | None = None,
        boundary_ratio: float | None = None,
        num_videos_per_prompt: int | None = 1,
        seed: int = 0,
        output_type: str | None = "uint8",
        traced: bool = False,
        on_event: PipelineEventCallback | None = None,
        cache_config: WanDBCacheConfig | DBCacheConfig | None = _UNSET,
        width: int | None = None,
        height: int | None = None,
    ):
        """Generates a video from the prompt.

        ``width`` and ``height`` must be one of the configured resolutions, and default to the
        first.
        """
        on_event = on_event if on_event is not None else null_callback

        width, height = self._resolve_size(width, height, image_prompt)
        padded_sequence_length = self._padded_sequence_lengths[width, height]

        # DBCache: resolve the per-request config (None disables caching for this call) and
        # start fresh cache contexts for both experts.
        cache_runners = self._cache_runners(
            self._cache_config if cache_config is _UNSET else cache_config, padded_sequence_length
        )

        negative_prompts = (
            negative_prompts if negative_prompts is not None else [self.DEFAULT_NEGATIVE_PROMPT] * len(prompts)
        )
        num_frames = self._num_frames

        # Per-request boundary_ratio overrides the construction-time default. It only affects
        # host-side expert selection (no captured trace depends on it).
        effective_boundary_ratio = boundary_ratio if boundary_ratio is not None else self._boundary_ratio

        if guidance_scale > 1 and not self._cfg_enabled:
            msg = "guidance_scale > 1 requires CFG to be enabled"
            raise ValueError(msg)
        if guidance_scale_2 is not None and guidance_scale_2 > 1 and not self._cfg_enabled:
            msg = "guidance_scale_2 > 1 requires CFG to be enabled"
            raise ValueError(msg)

        if effective_boundary_ratio is None and guidance_scale_2 is not None:
            msg = (
                "`guidance_scale_2` is only supported when `boundary_ratio` is not None. "
                "Set it per-request (pass `boundary_ratio=...` to this call) or at construction time."
            )
            raise ValueError(msg)

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. "
                "Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        if effective_boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        self.transformer_states[0].guidance_scale = guidance_scale
        self.transformer_states[1].guidance_scale = guidance_scale_2

        device = "cpu"

        batch_size = len(prompts)

        # 3. Encode input prompt
        on_event(SectionStart("encoder"))
        self._prepare_text_encoder()
        prompt_embeds, negative_prompt_embeds = self._text_encoder.encode_cfg(
            prompts,
            negative_prompts,
            cfg_enabled=self._cfg_enabled,
            num_videos_per_prompt=num_videos_per_prompt,
            on_event=on_event,
        )
        on_event(SectionEnd("encoder"))

        # 4. Prepare schedule
        # flow_shift is host-side only (it reshapes the sigma schedule); no captured trace
        # depends on it. None restores the solver's construction-time shift, so a per-request
        # value never persists into a later request.
        self.prepare_schedule(num_inference_steps, flow_shift=flow_shift)
        timesteps = self._solver.timesteps

        # 5. Prepare latent variables
        torch.manual_seed(seed)

        on_event(SectionStart("prepare_latents"))
        latents, cond_latents = self.prepare_latents(
            batch_size=batch_size * num_videos_per_prompt,
            image_prompt=image_prompt,
            num_channels_latents=self._vae.config.z_dim,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            device=device,
        )
        on_event(SectionEnd("prepare_latents"))

        mask = torch.ones(latents.shape, dtype=torch.float32, device=device)

        # 6. Denoising loop
        if effective_boundary_ratio is not None:
            boundary_timestep = effective_boundary_ratio * self._solver.schedule.num_train_timesteps
        else:
            boundary_timestep = -1  # Always use transformer (no transformer_2)

        on_event(SectionStart("denoising"))

        permuted_latent_tt = None
        rope_args = None

        latent_frames, latent_height, latent_width = latents.shape[2], latents.shape[3], latents.shape[4]
        prepared_prompts = [False, False]

        with tqdm.tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                warmup_t2 = i == 1 and len(timesteps) == 2  # Ensure transformer_2 is also warmed up

                # 0=> wan2.1 or high-noise stage in wan2.2 (transformer) | 1=> low-noise stage in wan2.2 (transformer_2)
                transformer_idx = 0 if (t >= boundary_timestep) and not warmup_t2 else 1
                self._prepare_transformer(transformer_idx)
                ts = self.transformer_states[transformer_idx]
                if not prepared_prompts[transformer_idx]:
                    # Prepare the text conditioning in an optional persistent buffer depending on traced
                    ts.prompt_buffer = self.prepare_text_conditioning(ts.model, prompt_embeds, ts.prompt_buffer, traced)
                    ts.negative_prompt_buffer = self.prepare_text_conditioning(
                        ts.model, negative_prompt_embeds, ts.negative_prompt_buffer, traced
                    )
                    prepared_prompts[transformer_idx] = True

                if permuted_latent_tt is None:
                    # First iteration, preprocess spatial input and prepare rope features
                    permuted_latent, latents_sequence_length = ts.model.preprocess_spatial_input_host(
                        latents, padded_sequence_length
                    )

                    sp_axis = ts.model.parallel_config.sequence_parallel.mesh_axis

                    if cond_latents is not None:
                        cond_latents, _ = ts.model.preprocess_spatial_input_host(cond_latents, padded_sequence_length)
                        cond_latents_tt = tensor.from_torch(
                            cond_latents,
                            device=self.mesh_device,
                            mesh_axes=[None, None, sp_axis, None],
                            dtype=ttnn.bfloat16,
                        )
                        self._persist(self._condition_buffers, padded_sequence_length, cond_latents_tt)

                        # Conditioning now lives in the persistent buffer; drop the host copy
                        # so a later iteration cannot re-enter this branch and redo the work.
                        cond_latents = None

                    rope_cos_1HND, rope_sin_1HND, trans_mat = ts.model.get_rope_features(
                        latents, padded_sequence_length
                    )
                    rope_slots = self._rope_slots.get(padded_sequence_length)
                    if rope_slots is None:
                        rope_slots = (ttnn.clone(rope_cos_1HND), ttnn.clone(rope_sin_1HND))
                        self._rope_slots[padded_sequence_length] = rope_slots
                    else:
                        ttnn.copy(rope_cos_1HND, rope_slots[0])
                        ttnn.copy(rope_sin_1HND, rope_slots[1])
                    rope_args = {
                        "rope_cos_1HND": rope_slots[0],
                        "rope_sin_1HND": rope_slots[1],
                        "trans_mat": trans_mat,
                    }

                    ttnn.copy_host_to_device_tensor(
                        tensor.from_torch(
                            torch.tensor([latents_sequence_length]),
                            device=self.mesh_device,
                            dtype=ttnn.uint32,
                            layout=ttnn.ROW_MAJOR_LAYOUT,
                            on_host=True,
                        ),
                        self._sequence_length,
                    )
                    valid_rows = torch.zeros([1, 1, padded_sequence_length, 1])
                    valid_rows[:, :, :latents_sequence_length] = 1
                    ttnn.copy_host_to_device_tensor(
                        tensor.from_torch(
                            valid_rows, device=self.mesh_device, mesh_axes=[None, None, sp_axis, None], on_host=True
                        ),
                        self._valid_rows[padded_sequence_length],
                    )

                    permuted_latent_tt = tensor.from_torch(
                        permuted_latent,
                        device=self.mesh_device,
                        mesh_axes=[None, None, sp_axis, None],
                        dtype=ttnn.float32,
                    )

                latent_buffer = self._persist(self._latent_buffers, padded_sequence_length, permuted_latent_tt)

                step_fn = self._step_cached if cache_runners is not None else self._step
                permuted_latent_tt = step_fn(
                    step=i,
                    t=t,
                    transformer_idx=transformer_idx,
                    padded_sequence_length=padded_sequence_length,
                    permuted_latent_tt=latent_buffer,
                    mask=mask,
                    cond_latents=self._condition_buffers.get(padded_sequence_length),
                    rope_args=rope_args,
                    latents_batch_size=latents.shape[0],
                    cfg_enabled=self._cfg_enabled,
                    traced=traced,
                )

                progress_bar.update()
                on_event(DenoiseStep(step=i + 1, total=num_inference_steps, sigma=self._solver.sigmas[i]))

        self._current_timestep = None

        sp_axis = ts.model.parallel_config.sequence_parallel.mesh_axis
        permuted_latent_tt = ts.model.ccl_manager.all_gather_persistent_buffer(
            permuted_latent_tt, dim=2, mesh_axis=sp_axis
        )
        permuted_latent = tensor.local_device_to_torch(permuted_latent_tt)

        # Postprocess spatial output
        latents = ts.model.postprocess_spatial_output_host(
            permuted_latent, F=latent_frames, H=latent_height, W=latent_width, N=latents_sequence_length
        )

        on_event(SectionEnd("denoising"))

        if output_type == "latent":
            return latents

        include_last_latent = output_type == "pt_with_last_latent"
        if include_last_latent:
            output_type = "pt"
            last_latent_out = latents.detach().clone()

        video = self._decode_latents(latents, output_type=output_type, on_event=on_event, size=(width, height))

        if include_last_latent:
            return video, last_latent_out
        return video

    @staticmethod
    def _persist(buffers: dict[int, ttnn.Tensor], key: int, value: ttnn.Tensor) -> ttnn.Tensor:
        """Copies ``value`` into the persistent buffer ``key``; the first value becomes it."""
        buffer = buffers.get(key)
        if buffer is None:
            buffers[key] = value
            return value
        ttnn.copy(value, buffer)
        return buffer

    def _decode_latents(
        self,
        latents: torch.Tensor,
        *,
        output_type: str,
        on_event: PipelineEventCallback,
        size: tuple[int, int],
    ) -> torch.Tensor:
        on_event(SectionStart("vae"))
        # The decoder's CCL buffers are cached by shape, so each size would keep its own; the
        # untraced decode can release them instead.
        with self.vae_ccl_manager.transient_ping_pong_buffers():
            video_torch = self._vaes[size].decode(latents, output_type=output_type)

        if output_type == "uint8":
            video = video_torch.numpy()
        elif output_type == "np":
            video = video_torch.float().numpy()
        else:
            video = self.video_processor.postprocess_video(video_torch, output_type=output_type)

        on_event(SectionEnd("vae"))
        return video

    def synchronize_devices(self) -> None:
        ttnn.synchronize_device(self.mesh_device)

    def release_traces(self) -> None:
        for tracer in self._step_tracers.values():
            tracer.release_trace()
        for ts in self.transformer_states:
            for runner in ts.cache_runners.values():
                runner.release_traces()
