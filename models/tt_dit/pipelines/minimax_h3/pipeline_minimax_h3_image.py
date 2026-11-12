# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3-Image on a Blackhole Galaxy: text-to-image (t2i) and reference-image editing (edit).

H3-Image is the H3 video DiT run audio-free on a single frame, with a 6-step PDD sampler (one fused fp32 output head
per step, no CFG), the H3 Qwen3-VL-32B conditioner (layer-50 tap, same weights as H3) and a 2D ViT VAE decoder.
Host-side request handling -- canvas, reference sizing, the RNG draw order, anchors, the packed sequence and its RoPE
grid, the sigma grid and the uint8/LANCZOS output -- lives in `image_host_minimax_h3.py`; this file runs it on device.

Weights: a snapshot converted by `convert_h3_image.py` (diffusers layout: transformer/ [+ pdd_heads.safetensors],
vae/, tokenizer/, processor/, pdd_config.json) plus `text_encoder/` from MiniMax-H3 (symlinked in).

Device flow per request:
  Qwen3-VL (+ vision tower for refs) -> prompt rows; refs -> VAE encode_clip -> posterior/anchor rows (host)
  DiT: prepare_static_sources([text | refs]) once; 6 forwards with output_head=k; Euler on host in fp32
  VAE decode_clip (tiled ViT2D) -> ImageNet de-norm -> clamp -> uint8 (truncate) -> LANCZOS to the request size

On a ring fabric the DiT runs dynamic length: the packed length is a runtime argument and a request pads only to
SP * TILE rows, so once one request has run, no request size compiles a new program. Each fixed schedule step's adaLN
modulation tables are computed once and reused (`precompute_adaln`). Without a ring the pipeline falls back to the
`IMAGE_BUCKET_LADDER_4X8` rungs, warmed once each; its reference limit then shrinks to what the top rung holds.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
from loguru import logger
from PIL import Image

import ttnn

from ...encoders.qwen3vl.loader_minimax_h3 import build_minimax_h3_text_encoder, build_minimax_h3_vision_tower
from ...models.transformers.minimax_h3.adaln_tilerow import DEFAULT_MAX_MIXED_TILES, tilerow_remap
from ...models.transformers.minimax_h3.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from ...models.vae.minimax_h3.vae_minimax_h3 import MiniMaxH3Vae, MiniMaxH3VaeConfig
from ...parallel.config import DiTParallelConfig, EncoderParallelConfig, ParallelFactor, VAEParallelConfig
from ...parallel.manager import CCLManager
from ...utils import cache
from ...utils.conv3d import conv3d_blocking_hash
from ...utils.tensor import local_device_to_torch
from . import image_host_minimax_h3 as host
from .packing import padded_sequence_length
from .pipeline_minimax_h3 import MiniMaxH3ArenaCaps, MiniMaxH3Pipeline, resolve_mesh_preset, validate_bucket_ladder

MODEL_NAME_IMAGE = "minimax-h3-image"  # DiT + VAE cache namespace; the Qwen conditioner is H3's own ("minimax-h3")

# Packed lengths, multiples of SP * TILE = 256. 2K t2i ~4.4k, 2K edit (1024^2 ref) ~6.5k, 4K t2i ~16.7k,
# 4K edit ~20.8k (one 2048^2 ref), the 21:9 recommended size ~17k.
IMAGE_BUCKET_LADDER_4X8 = (4608, 5376, 6656, 8192, 10240, 12800, 16384, 17920, 20480, 25600, 32768)

# Target rows: 6272x2688 (21:9) is 16464, 4096^2 is 16384.
IMAGE_TARGET_ROWS = 17152
# Qwen presentation = the prompt (<= host.MAX_PROMPT_TOKENS) + every reference's vision tokens + this much chat
# template and "<Picture N>" / vision-delimiter tokens (~10 a reference).
IMAGE_PRESENTATION_OVERHEAD = 1024
# Presentation rungs (text + vision tokens) below the prompt arena cap: a short t2i prompt, a 1024^2 reference
# (1024 vision tokens) plus text, two to four references, then up to sixteen; the cap is always the top rung.
IMAGE_PRESENTATION_LADDER = (1024, 2048, 4096, 8192, 16384, 32768)
# Vision-tower pad rungs, in unmerged patches (one 1024^2 Qwen copy is 4096) below the patch cap, which tops it.
IMAGE_VISION_PATCH_LADDER = (4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152, 65536, 98304, 131072)


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def image_arena_caps(reference_area_limit: int = host.REFERENCE_AREA_LIMIT) -> MiniMaxH3ArenaCaps:
    """The arenas for any request whose references are within `reference_area_limit` effective pixels."""
    refs = host.reference_caps(reference_area_limit)
    return MiniMaxH3ArenaCaps(
        prompt=_round_up(host.MAX_PROMPT_TOKENS + refs.vision_tokens + IMAGE_PRESENTATION_OVERHEAD, 1024),
        video_rows=IMAGE_TARGET_ROWS,
        audio_rows=32,  # unused (audio-free); kept non-zero for MiniMaxH3ArenaCaps.validate
        condition_video_rows=_round_up(refs.rows, 1024),
        condition_audio_rows=32,
    )


IMAGE_ARENA_CAPS = image_arena_caps()


def _arena_rows(caps: MiniMaxH3ArenaCaps) -> int:
    """The longest packed sequence the arenas admit: a full prompt, full references and the largest target."""
    return caps.prompt + caps.condition_video_rows + caps.video_rows


def largest_reference_area_limit(max_packed: int) -> int:
    """The largest reference limit (in whole VAE rows of 1024 px) whose arenas fit `max_packed` packed rows: what a
    bucketed pipeline may admit, since nothing pads past its top rung. 4194304 (one 2048^2 reference) for 32768."""
    lo, hi = 0, host.REFERENCE_AREA_LIMIT // 1024
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _arena_rows(image_arena_caps(mid * 1024)) <= max_packed:
            lo = mid
        else:
            hi = mid - 1
    if lo == 0:
        raise ValueError(f"a {max_packed}-row ladder top cannot hold a request at the arenas' target and prompt caps")
    return lo * 1024


@dataclass
class ImageRequestTimings:
    encode_prompt: float = 0.0
    encode_references: float = 0.0
    denoise: float = 0.0
    denoise_steps: list[float] = field(default_factory=list)
    decode: float = 0.0
    # The host tail (tile stitch, un-normalize, uint8), when `defer_finish` left it to `ImageResult.finish`: set once
    # that runs, and not part of `decode` or `total` then.
    finish: float = 0.0
    total: float = 0.0
    # Program-cache entries added during each stage: nonzero on a warm pipeline means a JIT compile on the serving path.
    program_misses: dict[str, int] = field(default_factory=dict)


@dataclass
class ImageResult:
    image: Image.Image | None  # None with `defer_finish`: call `finish()`
    canvas: tuple[int, int]
    packed_length: int
    rung: int
    timings: ImageRequestTimings
    # With `defer_finish`: the host tail, returning the image. Host-only (no device work), so a server can run it on
    # another thread while the next request runs on the device.
    finish: Callable[[], Image.Image] | None = None


class MiniMaxH3ImagePipeline(MiniMaxH3Pipeline):
    """t2i + edit for MiniMax-H3-Image. Reuses the H3 pipeline's encoder/vision/metadata helpers; no audio anywhere."""

    def __init__(
        self,
        *,
        mesh_device: ttnn.MeshDevice,
        weights_dir: str | Path,
        bucket_ladder: tuple[int, ...] = IMAGE_BUCKET_LADDER_4X8,
        reference_area_limit: int | None = None,
        arena_caps: MiniMaxH3ArenaCaps | None = None,
        num_links: int | None = None,
        topology: ttnn.Topology | None = None,
        tp_axis: int | None = None,
        sp_axis: int | None = None,
        dit_fsdp: bool = False,
        load_dit: bool = True,
        dynamic_length: bool | None = None,
        max_sequence_length: int | None = None,
        precompute_adaln: bool = True,
        use_persistent_ccl_buffers: bool = True,
        device_latents: bool = True,
    ) -> None:
        """`dynamic_length` (default: on whenever the fabric is a ring): the DiT takes the packed length as a runtime
        word, a request pads only to SP * TILE rows, and every length within a length class reuses one program set --
        no bucket ladder and no JIT compile on the serving path once each class is warm. `max_sequence_length` sizes
        the fused norms' worker split and the ring attention's K/V gather pair. `dynamic_length=False` falls back to
        the bucket ladder (eager, one program set per rung).

        `reference_area_limit` is the per-request reference limit (`host.check_references`): the arenas, the
        presentation and vision-tower ladders and `max_sequence_length` (default: the arenas' sum) all derive from
        it, so every request it admits fits and is warmed. Its default is `host.REFERENCE_AREA_LIMIT` with dynamic
        length. Bucketed, nothing pads past the ladder's top rung, so the default is the largest limit whose arenas
        fit that rung (`largest_reference_area_limit`: 4.19 MP, one 2048^2 reference, for the default ladder), and a
        larger limit -- or `arena_caps` whose rows sum past the rung -- raises."""
        # Deliberately not super().__init__: that builds the audio VAE and the video presets. Only the attributes the
        # reused helpers read are set here.
        self.mesh_device = mesh_device
        self.weights_dir = Path(weights_dir)
        preset = resolve_mesh_preset(tuple(mesh_device.shape), required=False) or {}
        self.tp_axis = preset.get("tp_axis", 0) if tp_axis is None else tp_axis
        self.sp_axis = preset.get("sp_axis", 1) if sp_axis is None else sp_axis
        num_links = preset.get("num_links", 1) if num_links is None else num_links
        if topology is None:
            line = ttnn.get_fabric_config() == ttnn.FabricConfig.FABRIC_1D
            topology = ttnn.Topology.Linear if line else preset.get("topology", ttnn.Topology.Linear)
        shape = tuple(mesh_device.shape)
        self.tp_factor, self.sp_factor = shape[self.tp_axis], shape[self.sp_axis]
        self.task = "image"
        self.transformer_subfolder = "transformer"
        self.coresident = True
        self.trace_denoise = False
        ring = topology == ttnn.Topology.Ring
        if dynamic_length is None:
            dynamic_length = ring
        if dynamic_length and not ring:
            raise ValueError("dynamic_length needs a ring topology (its fused AGMM / MMRS pins exist for a ring)")
        bucket_ladder = tuple(bucket_ladder)
        if reference_area_limit is None:
            reference_area_limit = (
                host.REFERENCE_AREA_LIMIT if dynamic_length else largest_reference_area_limit(bucket_ladder[-1])
            )
        self.reference_area_limit = int(reference_area_limit)
        self.reference_caps = host.reference_caps(self.reference_area_limit)
        if arena_caps is None:
            arena_caps = image_arena_caps(self.reference_area_limit)
        if not dynamic_length and _arena_rows(arena_caps) > bucket_ladder[-1]:
            raise ValueError(
                f"the bucketed pipeline pads to at most {bucket_ladder[-1]} rows but its arenas admit "
                f"{_arena_rows(arena_caps)} (reference_area_limit {self.reference_area_limit}); use a reference limit "
                f"of at most {largest_reference_area_limit(bucket_ladder[-1])}, a longer ladder, or a ring fabric"
            )
        if max_sequence_length is None:
            max_sequence_length = _arena_rows(arena_caps)
        self.dynamic_length = dynamic_length
        self.bucket_denoise = not dynamic_length
        # Persistent CCL buffers: without them every all-gather-matmul host-syncs the device first (150 per DiT
        # step at 4K, ~0.45 ms of idle device each). The eager loops run inside `transient_ping_pong_buffers`, so
        # the per-length pairs they add are freed after each request and do not accumulate across lengths.
        self.use_persistent_ccl_buffers = use_persistent_ccl_buffers
        # Keep the latents and the Euler update on device (same bits as the host update); see _denoise_image.
        self.device_latents = device_latents
        self.dit_fsdp = dit_fsdp
        self.bucket_ladder = bucket_ladder
        validate_bucket_ladder(self.bucket_ladder, self.sp_factor * ttnn.TILE_SIZE)
        alignment = self.sp_factor * ttnn.TILE_SIZE
        self.max_sequence_length = -(-max_sequence_length // alignment) * alignment
        # Not `kv_gather_capacity`: on main that is a read-only property of the H3 pipeline.
        self._dit_kv_capacity = self.max_sequence_length if dynamic_length else self.bucket_ladder[-1]
        self.arena_caps = arena_caps
        self.arena_caps.validate()
        # The Qwen3-VL encoder compiles one program set per presentation length, so prompts pad up to a rung of this
        # ladder (warmed by `warmup`); it is weight-bound at these lengths, so the padding costs ~nothing.
        self.presentation_ladder = tuple(r for r in IMAGE_PRESENTATION_LADDER if r < self.arena_caps.prompt) + (
            self.arena_caps.prompt,
        )
        # The tower pads its patches to a warmed rung; the top one holds every admitted request's references.
        self.vision_patch_cap = _round_up(self.reference_caps.vision_patches, 4096)
        self.vision_patch_ladder = tuple(r for r in IMAGE_VISION_PATCH_LADDER if r < self.vision_patch_cap) + (
            self.vision_patch_cap,
        )
        self._force_bucket = self._force_prompt_pad = self._force_vision_pad = None
        self._log_generation = True
        # Each SP device gathers only its own rows, so forward skips `mesh_partition` -- the one op whose program was
        # still keyed on the padded length (one JIT compile per new length in the hot path).
        self._shard_assembly_indices = dynamic_length and self.sp_factor > 1
        self._tokenizer = self._image_processor = None
        self._vision_tower = self._vision_config = None
        self.last_program_cache_misses: dict[str, int] = {}

        self.ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
        self.encoder_ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
        self.dit_parallel_config = DiTParallelConfig(
            tensor_parallel=ParallelFactor(mesh_axis=self.tp_axis, factor=self.tp_factor),
            sequence_parallel=ParallelFactor(mesh_axis=self.sp_axis, factor=self.sp_factor),
            cfg_parallel=None,
        )
        self.encoder_parallel_config = EncoderParallelConfig(
            tensor_parallel=ParallelFactor(mesh_axis=self.tp_axis, factor=self.tp_factor),
            sequence_parallel=ParallelFactor(mesh_axis=self.sp_axis, factor=self.sp_factor)
            if self.sp_factor > 1
            else None,
        )
        self.vae_parallel_config = VAEParallelConfig(tensor_parallel=ParallelFactor(mesh_axis=self.tp_axis, factor=1))

        self.transformer_config = self._read_config("transformer")
        if self.transformer_config.get("audio_enabled", True):
            raise ValueError("transformer/config.json is not audio-free; convert with convert_h3_image.py")
        self.rope_freq_dim = self.transformer_config["rope_freq_dim"]
        self.rope_theta = self.transformer_config["rope_theta"]
        self.schedule = host.PddSchedule.from_pretrained(self.weights_dir)
        # The release runs the per-step output heads in fp32; H3_IMAGE_PDD_FP32=1 does the same on device.
        self.pdd_head_fp32 = os.environ.get("H3_IMAGE_PDD_FP32", "0") == "1"
        # Per-step adaLN tables computed once (the 6-step schedule is fixed) instead of every forward.
        self.precompute_adaln = precompute_adaln
        # Once here: `_prepare_request` calls `_prepare_transformer` on every request, and hashing the weight files
        # there would re-read the config and index each time.
        self._dit_cache_subfolder = (
            f"transformer_{self._weights_fingerprint()}_pdd{self.schedule.num_steps}"
            f"{'fp32' if self.pdd_head_fp32 else ''}_{self._dit_weight_mode()}"
        )
        self.vae_config = MiniMaxH3VaeConfig.from_pretrained(self.weights_dir / "vae")

        self._host_log("building the Qwen3-VL text encoder")
        self._text_encoder, self._text_config = build_minimax_h3_text_encoder(
            self.weights_dir / "text_encoder",
            mesh_device=mesh_device,
            parallel_config=self.encoder_parallel_config,
            ccl_manager=self.encoder_ccl_manager,
            is_fsdp=True,
            load_weights=False,
            kv_gather_capacity=padded_sequence_length(self.arena_caps.prompt, self.sp_factor),
            use_persistent_ccl_buffers=self.use_persistent_ccl_buffers,
        )
        self._transformer = self._build_transformer()
        # Prompts arrive padded to a warmed presentation rung, so the token refiner runs at that rung rather than
        # at the 32768-row prompt arena (0.25 s a request at the cap).
        self._transformer.refine_at_input_length = True
        self._vae = MiniMaxH3Vae(
            self.vae_config,
            task="image",
            mesh_device=mesh_device,
            weight_loader=self._cache_submodel,
            ccl_manager=self.encoder_ccl_manager,
            dtype=ttnn.bfloat16,
            pixel_norm=(host.PIXEL_MEAN, host.PIXEL_STD),
            waves_per_device=2,
        )
        self._vae.load_state(self._read_safetensors("vae"))
        if load_dit:
            self._prepare_transformer()
        # The Qwen conditioner (~64 GB) loads on the first encode_prompt (`_encode_prompt_device` prepares it), so a
        # DiT- or VAE-only test never pays for it.

    # ------------------------------------------------------------------ construction overrides

    def _build_transformer(self) -> MiniMaxH3Transformer3DModel:
        config = {k: v for k, v in self.transformer_config.items() if k not in ("rope_freq_dim", "rope_theta")}
        config["patch_size"] = tuple(config["patch_size"])
        self._host_log(
            f"building the {config['num_layers']}-layer audio-free H3-Image DiT, TP={self.tp_factor}/SP={self.sp_factor}"
        )
        return MiniMaxH3Transformer3DModel(
            **config,
            num_output_heads=self.schedule.num_steps,
            pdd_head_fp32=self.pdd_head_fp32,
            mesh_device=self.mesh_device,
            ccl_manager=self.ccl_manager,
            parallel_config=self.dit_parallel_config,
            is_fsdp=self.dit_fsdp,
            kv_gather_capacity=self._dit_kv_capacity,
            use_persistent_ccl_buffers=self.use_persistent_ccl_buffers,
            dynamic_length=self.dynamic_length,
        )

    def _select_bucket(self, seq_len: int) -> int:
        """Dynamic length: pad only to SP * TILE rows (any length within a length class reuses its programs).
        Bucketed: the smallest ladder rung (the parent's rule, including warmup's forced rung)."""
        if not self.dynamic_length:
            return super()._select_bucket(seq_len)
        padded = padded_sequence_length(seq_len, self.sp_factor)
        if padded > self.max_sequence_length:
            raise ValueError(f"packed length {seq_len} exceeds max_sequence_length {self.max_sequence_length}")
        return padded

    def _dit_state(self) -> dict[str, torch.Tensor]:
        from safetensors.torch import load_file

        state = self._read_safetensors("transformer")
        state.update(load_file(str(self.weights_dir / "transformer" / "pdd_heads.safetensors")))
        return state

    def _weights_fingerprint(self) -> str:
        """A short hash of the transformer config + weight index and of every weight file's (name, size, mtime_ns) --
        each shard and pdd_heads.safetensors -- so different checkpoints (a few-layer gate snapshot, a perf-tiled one,
        the release) and a re-conversion with unchanged shapes never share a cache entry; the cache key is otherwise
        only the parallel config and mesh. A copy that does not preserve mtimes rebuilds the cache once."""
        import hashlib

        tdir = self.weights_dir / "transformer"
        digest = hashlib.sha1()
        for name in ("config.json", "diffusion_pytorch_model.safetensors.index.json"):
            digest.update((tdir / name).read_bytes())
        for path in sorted(tdir.glob("*.safetensors")):
            stat = path.stat()
            digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
        return digest.hexdigest()[:12]

    def _prepare_transformer(self) -> MiniMaxH3Transformer3DModel:
        cache.load_model(
            self._transformer,
            model_name=MODEL_NAME_IMAGE,
            subfolder=self._dit_cache_subfolder,
            parallel_config=self.dit_parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            mesh_device=self.mesh_device,
            get_torch_state_dict=self._dit_state,
        )
        return self._transformer

    def _cache_submodel(self, module, subfolder: str, state: dict[str, torch.Tensor]) -> None:
        blocking = conv3d_blocking_hash(module)
        cache.load_model(
            module,
            model_name=MODEL_NAME_IMAGE,
            subfolder=f"{subfolder}_{blocking}" if blocking else subfolder,
            parallel_config=self.vae_parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            mesh_device=self.mesh_device,
            dtype="fp32",
            get_torch_state_dict=lambda: state,
        )

    def _prepare_vision_tower(self):
        if self._vision_tower is None:
            self._host_log(f"building the Qwen3-VL vision tower (tp{self.tp_factor}_sp{self.sp_factor})")
            self._vision_tower, self._vision_config = build_minimax_h3_vision_tower(
                self.weights_dir / "text_encoder",
                mesh_device=self.mesh_device,
                parallel_config=EncoderParallelConfig(
                    tensor_parallel=ParallelFactor(mesh_axis=self.tp_axis, factor=self.tp_factor),
                    sequence_parallel=ParallelFactor(mesh_axis=self.sp_axis, factor=self.sp_factor),
                ),
                ccl_manager=self.encoder_ccl_manager,
                kv_gather_capacity=padded_sequence_length(self.vision_patch_cap, self.sp_factor),
                use_persistent_ccl_buffers=self.use_persistent_ccl_buffers,
            )
        return self._vision_tower

    @property
    def image_processor(self):
        if self._image_processor is None:
            from transformers import AutoImageProcessor

            self._image_processor = AutoImageProcessor.from_pretrained(str(self.weights_dir), subfolder="processor")
        return self._image_processor

    def _build_presentation(self, prompt: str, keyframes: Sequence[Image.Image]):
        """H3-Image's presentation: per reference `"Picture i: "` (text-tagged; no angle brackets, unlike video
        ref2va), then `<|vision_start|> N x <|image_pad|> <|vision_end|>` (image-tagged), then the prompt verbatim,
        right-truncated at 4096 tokens. `keyframes` here are the Qwen-bounded reference images."""
        tokenizer = self.tokenizer
        image_pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        vision_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        ids: list[int] = []
        tags: list[int] = []
        pixel_values = grid_thw = None
        if keyframes:
            processor = self.image_processor
            vision = processor(images=list(keyframes), return_tensors="pt")
            pixel_values, grid_thw = vision["pixel_values"], vision["image_grid_thw"]
            merge = processor.merge_size**2
            for i in range(len(keyframes)):
                label = tokenizer(f"Picture {i + 1}: ", add_special_tokens=False)["input_ids"][: host.MAX_PROMPT_TOKENS]
                block = [vision_start] + [image_pad] * (int(grid_thw[i].prod()) // merge) + [vision_end]
                ids += label + block
                tags += [host.TEXT_TAG] * len(label) + [host.IMAGE_TAG] * len(block)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"][: host.MAX_PROMPT_TOKENS]
        if not prompt_ids:
            raise ValueError("prompt tokenized to zero tokens")
        ids += prompt_ids
        tags += [host.TEXT_TAG] * len(prompt_ids)
        if len(ids) > self.arena_caps.prompt:
            raise ValueError(f"presentation is {len(ids)} tokens, over the {self.arena_caps.prompt}-token prompt cap")
        input_ids = torch.tensor([ids], dtype=torch.long)
        return input_ids, torch.tensor(tags, dtype=torch.long), (input_ids == image_pad).long(), pixel_values, grid_thw

    # ------------------------------------------------------------------ request

    def __call__(self, prompt: str, **kwargs) -> ImageResult:
        """Generate one image. Host torch runs at half the CPU mask for the call: at torch's default (every logical
        CPU) its OpenMP pool fights tt-metal's busy-polling dispatch threads, which measured 130x slower on the
        decode's tile stitch (7.8 s vs 0.06 s at 2K) -- the same fix as the Wan/LTX pipelines (#60670)."""
        previous = torch.get_num_threads()
        torch.set_num_threads(self._torch_threads(deferred=bool(kwargs.get("defer_finish"))))
        if kwargs.get("defer_finish"):
            # The thread count is per process, and the deferred host tail may still be running on another thread when
            # this call returns (that is its point), so it is not restored: a deferred caller owns the process's torch.
            return self._generate(prompt, **kwargs)
        try:
            return self._generate(prompt, **kwargs)
        finally:
            torch.set_num_threads(previous)

    @staticmethod
    def _torch_threads(deferred: bool = False) -> int:
        """Half the CPU mask, or 8 when the host tail is deferred: then it overlaps the next request, and at half the
        CPUs it slowed that request's device path by up to ~0.1 s (dispatch threads starved). Measured on a 4x8
        Galaxy, 4 t2i + 2 edits at 4K back to back: 24 threads 3.2% faster than serial, 8 threads 6.6% (the tile
        stitch is memory-bound: 0.28 s at 8 threads vs 0.33 s at 24). H3_IMAGE_TORCH_THREADS overrides both, e.g.
        several replicas in one process should split the CPUs between them."""
        override = int(os.environ.get("H3_IMAGE_TORCH_THREADS", 0))
        if override:
            return override
        half = max(1, len(os.sched_getaffinity(0)) // 2)
        return min(8, half) if deferred else half

    def _generate(
        self,
        prompt: str,
        *,
        width: int,
        height: int,
        seed: int = 0,
        images: Sequence[Image.Image] | None = None,
        min_generation_area: int | None = host.DEFAULT_MIN_GENERATION_AREA,
        defer_finish: bool = False,
    ) -> ImageResult:
        """`defer_finish`: return once the device work is done (the decoded tiles are read back), leaving the host tail
        -- tile stitch, un-normalize, uint8 -- to `ImageResult.finish()` (same bits). A server runs it on another
        thread while the next request runs on the device; ~0.25 s of each 4K request is that tail."""
        timings = ImageRequestTimings()
        started = time.perf_counter()
        canvas = host.resolve_image_canvas(width, height, min_generation_area)
        cw, ch = canvas
        lh, lw = ch // host.VAE_RATIO, cw // host.VAE_RATIO
        # Before any device work: otherwise an oversized canvas is only rejected after the Qwen and VAE encodes.
        target_rows = (lh // 2) * (lw // 2)
        if target_rows > self.arena_caps.video_rows:
            raise ValueError(
                f"{width}x{height} resolves to a {cw}x{ch} canvas of {target_rows} latent rows, over the "
                f"{self.arena_caps.video_rows}-row target cap (4096x4096 is 16384)"
            )

        refs_rgb = [host.load_reference_image(image) for image in images or ()]
        if refs_rgb:  # over the limit is rejected (the served API answers 422 first), never shrunk
            host.check_references([im.size for im in refs_rgb], self.reference_area_limit)
        qwen_refs = [host.bound_short_edge(im, *host.QWEN_REFERENCE_SHORT_EDGE) for im in refs_rgb]
        vae_refs = [host.bound_short_edge(im, *host.VAE_REFERENCE_SHORT_EDGE) for im in refs_rgb]
        geometry = [host.reference_geometry(im) for im in vae_refs]
        noise = host.draw_request_noise(seed, canvas, geometry)

        programs = self.mesh_device.num_program_cache_entries

        def miss_since(start: int, stage: str) -> int:
            timings.program_misses[stage] = programs() - start
            return programs()

        count = programs()
        mark = time.perf_counter()
        prompt_embeds, text_tags = self.encode_prompt(prompt, keyframes=qwen_refs)
        timings.encode_prompt = time.perf_counter() - mark
        count = miss_since(count, "prompt")

        mark = time.perf_counter()
        anchors = []
        for im, eps_post, eps_anchor in zip(vae_refs, noise.posteriors, noise.anchors):
            moments = self._vae.encode_clip(host.vae_reference_pixels(im))
            clean, _ = host.reference_rows(moments, eps_post, self.vae_config.latents_mean, self.vae_config.latents_std)
            anchors.append(host.anchor_rows(clean, eps_anchor))
        ref_rows = torch.cat(anchors) if anchors else torch.zeros(0, 96)
        timings.encode_references = time.perf_counter() - mark
        count = miss_since(count, "refs")

        layout = host.build_image_packed_sequence(text_tags, geometry, (lh, lw))
        mark = time.perf_counter()
        target = self._denoise_image(prompt_embeds, layout, ref_rows, host.patchify(noise.target), timings)
        timings.denoise = time.perf_counter() - mark
        count = miss_since(count, "denoise")

        mark = time.perf_counter()
        latents = host.unpatchify(target, lh, lw)
        mean = torch.tensor(self.vae_config.latents_mean, dtype=torch.float32).view(1, -1, 1, 1, 1)
        std = torch.tensor(self.vae_config.latents_std, dtype=torch.float32).view(1, -1, 1, 1, 1)
        stitch = self._vae.decode_clip_deferred(latents * std + mean)
        miss_since(count, "decode")
        torch_threads = self._torch_threads(deferred=defer_finish)

        def finish() -> Image.Image:
            finish_started = time.perf_counter()
            if defer_finish:  # on the caller's thread, maybe after `__call__` returned
                torch.set_num_threads(torch_threads)
            decoded = stitch()[0, :, 0].float()  # ImageNet-normalized pixels
            pm = torch.tensor(host.PIXEL_MEAN).view(3, 1, 1)
            ps = torch.tensor(host.PIXEL_STD).view(3, 1, 1)
            # In place on the decoder's own output: the same ops in the same order as `(x * ps + pm).clamp(0, 1)`, so
            # the same bits, without three 4K-sized float temporaries.
            pixels01 = decoded.mul_(ps).add_(pm).clamp_(0, 1)
            image = host.finish_image(pixels01, (width, height))
            timings.finish = time.perf_counter() - finish_started
            return image

        result = ImageResult(
            image=None, canvas=canvas, packed_length=layout.sequence_length, rung=self.last_rung, timings=timings
        )
        if defer_finish:
            result.finish = finish
        else:
            result.image = finish()
            timings.finish = 0.0  # part of `decode` here
        timings.decode = time.perf_counter() - mark
        timings.total = time.perf_counter() - started
        return result

    # ------------------------------------------------------------------ denoise

    def _denoise_image(
        self,
        prompt_embeds: ttnn.Tensor,
        layout: host.ImagePackedSequence,
        ref_rows: torch.Tensor,
        target_rows: torch.Tensor,
        timings: ImageRequestTimings,
    ) -> torch.Tensor:
        programs = self.mesh_device.num_program_cache_entries
        before = programs()
        request = self._prepare_request(prompt_embeds, layout, ref_rows)
        timings.program_misses["denoise_prepare"] = programs() - before
        before = programs()
        x = target_rows.float().clone()  # fp32 on host, as the reference keeps it
        with self.ccl_manager.transient_ping_pong_buffers():
            if self.device_latents:
                # Latents stay on device (fp32, replicated) and the Euler update runs there, so the host can enqueue
                # step k+1 while step k still runs: no readback / host update / upload between steps. Measured
                # byte-identical to the host update at 2K and 4K, and ~0.1 s faster per request.
                cap = self.arena_caps.video_rows
                x_dev = ttnn.from_torch(
                    self._pad_host_rows(x, cap).reshape(1, 1, cap, -1),
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
                for k in range(self.schedule.num_steps):
                    step_started = time.perf_counter()
                    video = ttnn.typecast(x_dev, ttnn.bfloat16)
                    velocity = self._step(request, None, k, video_1BVC=video, readback=False)
                    update = ttnn.multiply(ttnn.typecast(velocity, ttnn.float32), self.schedule.dt(k))
                    x_dev = ttnn.add(x_dev, update)
                    for t in (video, velocity, update):
                        ttnn.deallocate(t)
                    timings.denoise_steps.append(time.perf_counter() - step_started)
                x = local_device_to_torch(x_dev).reshape(-1, x_dev.shape[-1])[: request["target_rows"]].float()
                ttnn.deallocate(x_dev)
            else:
                for k in range(self.schedule.num_steps):
                    step_started = time.perf_counter()
                    x += self.schedule.dt(k) * self._step(request, x, k)
                    timings.denoise_steps.append(time.perf_counter() - step_started)
        timings.program_misses["denoise_steps"] = programs() - before
        return x

    def _prepare_request(
        self, prompt_embeds: ttnn.Tensor, layout: host.ImagePackedSequence, ref_rows: torch.Tensor
    ) -> dict:
        """Per-request device state: source-table prefix (refined prompt + projected anchors) and the row tables."""
        caps = self.arena_caps
        transformer = self._prepare_transformer()
        L, R, T = layout.text_len, layout.num_ref_rows, layout.target_rows
        # A guard: `_generate` already rejects an oversized canvas and `check_references` oversized references.
        if L > caps.prompt or R > caps.condition_video_rows or T > caps.video_rows:
            raise ValueError(
                f"request exceeds the arena caps: text {L}/{caps.prompt}, refs {R}/{caps.condition_video_rows}, "
                f"target {T}/{caps.video_rows}"
            )
        rung = self._select_bucket(layout.sequence_length)
        self.last_rung = rung
        self._log(f"packed {layout.sequence_length} rows (text {L}, refs {R}, target {T}) -> rung {rung}")

        # Device metadata, all at the rung. Source table: [refined prompt @ prompt cap | refs @ cond cap | target].
        rope_cos, rope_sin = self._device_metadata(layout, rung)
        src_refs, src_target = caps.prompt, caps.prompt + caps.condition_video_rows
        assembly = torch.zeros(rung, dtype=torch.int32)
        assembly[:L] = torch.arange(L)
        assembly[L : L + R] = torch.arange(src_refs, src_refs + R)
        assembly[L + R : L + R + T] = torch.arange(src_target, src_target + T)
        adaln = layout.row_slot * 3 + layout.token_tags.clamp(min=0)
        if self._shard_assembly_indices:
            tt_assembly = self._row_indices(assembly, rung)
        else:
            tt_assembly = self._replicated_indices(assembly)
        tt_adaln = self._row_indices(adaln, rung)
        tt_tsi = self._row_indices(layout.row_slot, rung)
        tt_logical = self._logical_length(layout.sequence_length)
        tt_out = self._output_indices(layout.target_start, T, caps.video_rows)
        # The norms read their modulation through a per-tile-row map, as the H3 video pipeline does; without it the
        # blocks fall back to the per-token one-hot gather. A request with more run-boundary tiles than the map has
        # slots keeps the fallback. 2 timestep slots x 3 modalities = 6 modulation rows.
        tilerow_kwargs = {}
        padded_adaln = torch.cat([adaln, torch.zeros(rung - adaln.shape[0], dtype=adaln.dtype)])
        try:
            tile_map, expanded = tilerow_remap(
                padded_adaln, num_rows=2 * 3, sp_factor=self.sp_factor, max_mixed_tiles=DEFAULT_MAX_MIXED_TILES
            )
            tilerow_kwargs = {
                "adaln_tile_map": self._row_indices(tile_map, tile_map.shape[0]),
                "adaln_expanded_indices": self._row_indices(expanded, expanded.shape[0]),
            }
        except ValueError as err:
            self._log(f"adaLN tile-row map not used for this request ({err}); per-token gather instead")

        prompt_device = ttnn.reshape(prompt_embeds, (1, 1, prompt_embeds.shape[1], prompt_embeds.shape[2]))
        with self.ccl_manager.transient_ping_pong_buffers():
            transformer.prepare_static_sources(
                prompt_1BLP=prompt_device,
                prompt_len=L,
                condition_video_1BKC=self._upload_rows(ref_rows, caps.condition_video_rows),
                prompt_cap=caps.prompt,
            )
            ttnn.synchronize_device(self.mesh_device)
        ttnn.deallocate(prompt_device)
        return {
            "transformer": transformer,
            "rung": rung,
            "target_rows": T,
            "assembly": tt_assembly,
            "adaln": tt_adaln,
            "tsi": tt_tsi,
            "logical": tt_logical,
            "out": tt_out,
            "rope": (rope_cos, rope_sin),
            "tilerow": tilerow_kwargs,
        }

    def _upload_rows(self, rows: torch.Tensor, capacity: int) -> ttnn.Tensor:
        return ttnn.from_torch(
            self._pad_host_rows(rows.float(), capacity).reshape(1, 1, capacity, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _step(
        self,
        request: dict,
        x: torch.Tensor | None,
        k: int,
        *,
        video_1BVC: ttnn.Tensor | None = None,
        readback: bool = True,
    ) -> torch.Tensor | ttnn.Tensor:
        """One DiT forward at step k (timestep slots `[1 - sigma_k, 0.999]`, PDD head k): the target velocity, fp32."""
        timestep = torch.tensor([self.schedule.timestep(k), host.REFERENCE_TIMESTEP], dtype=torch.float32)
        tt_t = ttnn.from_torch(
            timestep.reshape(1, 1, 2, 1),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        transformer = request["transformer"]
        modulation_key = None
        if self.precompute_adaln:
            # The schedule is fixed, so step k always conditions on the same `[t_k, 0.999]`: its modulation tables
            # are computed on the first request and reused by every later one (keyed by the exact timestep pair;
            # `precompute_modulation` is a no-op for a key it already holds).
            modulation_key = (k, *timestep.tolist())
            transformer.precompute_modulation(modulation_key, tt_t)
        rope_cos, rope_sin = request["rope"]
        # Diagnostics, off by default. H3_IMAGE_STEP_TIMING=1 splits each step into host enqueue / device wait /
        # readback (enqueue close to the step time means the step is host-dispatch bound). H3_IMAGE_PROFILE_STEP=k
        # brackets step k with Tracy signposts for a `--profile` run, flushing the device profiler on both sides so
        # one step fits its buffers.
        timing = os.environ.get("H3_IMAGE_STEP_TIMING") == "1"
        profile = os.environ.get("H3_IMAGE_PROFILE_STEP") == str(k)
        if profile:
            from tracy import signpost

            ttnn.synchronize_device(self.mesh_device)
            ttnn.ReadDeviceProfiler(self.mesh_device)
            signpost("start")
        started = time.perf_counter()
        velocity, _ = transformer(
            video_1BVC=video_1BVC if video_1BVC is not None else self._upload_rows(x, self.arena_caps.video_rows),
            audio_1BAC=None,
            assembly_indices=request["assembly"],
            video_out_indices=request["out"],
            audio_out_indices=None,
            timestep=tt_t,
            adaln_indices=request["adaln"],
            timestep_indices=request["tsi"],
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            logical_n=request["logical"],
            pad_to=request["rung"],
            output_head=k,
            modulation_key=modulation_key,
            **request.get("tilerow", {}),
        )
        enqueued = time.perf_counter()
        if timing or profile:
            ttnn.synchronize_device(self.mesh_device)
        synced = time.perf_counter()
        if profile:
            signpost("stop")
            ttnn.ReadDeviceProfiler(self.mesh_device)
        if not readback:
            return velocity  # [1, 1, V_cap, C] on device; the caller owns it
        v = local_device_to_torch(velocity).reshape(-1, velocity.shape[-1])[: request["target_rows"]].float()
        ttnn.deallocate(velocity)
        if timing:
            done = time.perf_counter()
            logger.info(
                f"STEP_TIMING step {k} rows {request['rung']}: enqueue {1e3 * (enqueued - started):.1f} ms, device "
                f"wait {1e3 * (synced - enqueued):.1f} ms, readback {1e3 * (done - synced):.1f} ms"
            )
        return v

    # ------------------------------------------------------------------ warmup

    def warmup(self) -> float:
        """Compile every program a served request can reach, so none compiles on the serving path; returns seconds.

        The DiT and the VAE are length-agnostic once one request has run (dynamic length), but the text encoder keys
        programs on the presentation length: each rung of `presentation_ladder` is encoded once, text-only and (where
        a 1024^2 reference fits) with one, which is what an edit merges. Then a 4K t2i (the default size; the most
        VAE decode waves) and a 2K edit (the vision tower and the VAE encoder) run end to end. With dynamic length a
        4K edit with as many 1024^2 references as the limit admits (up to sixteen) warms the DiT's long length
        class; bucketed, `warm_rungs` warms each ladder rung instead (the bucketed limit admits no such request).
        """
        started = time.perf_counter()
        reference = Image.new("RGB", (1024, 1024), (128, 128, 128))
        for rung in self.presentation_ladder:
            for keyframes in ((), (reference,)):
                if keyframes and rung < 2048:  # 1024 vision tokens + the template do not fit a 1024 rung
                    continue
                self._force_prompt_pad = rung
                try:
                    embeds, _ = self.encode_prompt("warmup", keyframes=list(keyframes))
                finally:
                    self._force_prompt_pad = None
                if not keyframes:
                    # The DiT's static-source prefix pads the prompt rows from this rung to the arena cap: that pad
                    # is keyed on the rung, so run it here too (as `_prepare_request` does).
                    prompt_device = ttnn.reshape(embeds, (1, 1, embeds.shape[1], embeds.shape[2]))
                    with self.ccl_manager.transient_ping_pong_buffers():
                        self._prepare_transformer().prepare_static_sources(
                            prompt_1BLP=prompt_device,
                            prompt_len=1,
                            condition_video_1BKC=self._upload_rows(
                                torch.zeros(0, self.transformer_config["in_channels"] * 4),
                                self.arena_caps.condition_video_rows,
                            ),
                            prompt_cap=self.arena_caps.prompt,
                        )
                        ttnn.synchronize_device(self.mesh_device)
                    ttnn.deallocate(prompt_device)
                ttnn.deallocate(embeds)
        # The vision tower pads its patches to a rung of `vision_patch_ladder`: one encode per rung (one 1024^2
        # reference padded up to it), then every reachable (presentation rung, tower rung) pair of the vision merge.
        for vision_pad in self.vision_patch_ladder:
            self._force_vision_pad, self._force_prompt_pad = vision_pad, 2048
            try:
                embeds, _ = self.encode_prompt("warmup", keyframes=[reference])
            finally:
                self._force_vision_pad = self._force_prompt_pad = None
            ttnn.deallocate(embeds)
        self._warm_vision_merge(self.presentation_ladder, self.vision_patch_ladder)
        self("warmup", width=4096, height=4096, seed=0)
        num_refs = min(host.MAX_REFERENCES, self.reference_area_limit // (1024 * 1024))
        if num_refs:
            self("warmup", width=2048, height=2048, seed=0, images=[reference])
        if self.dynamic_length:
            # Sixteen references put the DiT past 6144 rows a device, its "long" length class (its own programs).
            if num_refs > 1:
                self("warmup", width=4096, height=4096, seed=0, images=[reference] * num_refs)
        else:
            self.warm_rungs()
        return time.perf_counter() - started

    def warm_rungs(self, rungs: Sequence[int] | None = None) -> dict[int, float]:
        """Run one tiny request forced to each rung so its DiT programs are compiled before serving (bucketed only:
        the dynamic-length DiT has no rungs, and `warmup` covers its length classes)."""
        if self.dynamic_length:
            raise RuntimeError("warm_rungs is for the bucketed (dynamic_length=False) pipeline; use warmup()")
        seconds = {}
        for rung in rungs or self.bucket_ladder:
            self._force_bucket = rung
            mark = time.perf_counter()
            try:
                self("warmup", width=1024, height=1024, seed=0, min_generation_area=None)
            finally:
                self._force_bucket = None
            seconds[rung] = time.perf_counter() - mark
        return seconds
