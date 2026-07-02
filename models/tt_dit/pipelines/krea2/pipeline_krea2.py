# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""End-to-end KREA-2 (Krea 2) tt_dit image-generation pipeline.

Mirrors the diffusers reference `Krea2Pipeline`
(diffusers_main/src/diffusers/pipelines/krea2/pipeline_krea2.py) and wires the already
ported + PCC-verified tt components:

  * DiT     : models/tt_dit/models/transformers/transformer_krea2.py (Krea2Checkpoint / Krea2Transformer)
  * encoder : models/tt_dit/pipelines/krea2/text_encoder.py (Qwen3-VL text tower)
  * VAE     : models/tt_dit/models/vae/vae_qwenimage.py (QwenImageVAEDecoderAdapter)

Turbo (`is_distilled=True`) defaults are baked in:
  * mu = 1.15 (fixed), num_inference_steps = 8, guidance_scale = 0.0 (single forward pass),
  * sigmas = linspace(1.0, 1/steps, steps),
  * per-step timestep = t / num_train_timesteps in [0, 1],
  * latents = latents + (sigma_next - sigma_curr) * noise_pred (Euler flow-matching step).

Structure/conventions follow models/tt_dit/pipelines/qwenimage/pipeline_qwenimage.py
(preset dicts keyed by mesh shape, dynamic model load/free, host<->device boundaries).

IMPORTANT: This module does not open a device on import. All device work happens inside
`Krea2Pipeline.create_pipeline` / `run`, which the main thread invokes on hardware.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import tqdm
from loguru import logger

import ttnn
from models.tt_dit.models.transformers.transformer_krea2 import Krea2Checkpoint
from models.tt_dit.models.vae.vae_qwenimage import QwenImageVAEDecoderAdapter
from models.tt_dit.parallel.config import (
    DiTParallelConfig,
    EncoderParallelConfig,
    ParallelFactor,
    VaeHWParallelConfig,
    VAEParallelConfig,
)
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.krea2.text_encoder import TextEncoder
from models.tt_dit.solvers import EulerSolver
from models.tt_dit.utils.tensor import bf16_tensor
from models.tt_dit.utils.tensor import to_torch as dit_to_torch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from PIL import Image

_DEFAULT_CHECKPOINT = "krea/Krea-2-Turbo"

# KREA-2 image-model constants (fixed by the checkpoint / reference).
NUM_TRAIN_TIMESTEPS = 1000
VAE_SCALE_FACTOR = 8  # 2 ** len(temperal_downsample=[F,T,T])
PATCH_SIZE = 2
PIXEL_TO_TOKEN = VAE_SCALE_FACTOR * PATCH_SIZE  # 16
NUM_CHANNELS_LATENTS = 16  # transformer.in_channels(64) // patch_size**2(4)
MAX_SEQUENCE_LENGTH = 512
TURBO_MU = 1.15
TURBO_STEPS = 8

# KREA-2 VAE latent statistics (vae/config.json).
VAE_Z_DIM = 16
VAE_LATENTS_MEAN = [
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
]
VAE_LATENTS_STD = [
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.916,
]

# Presets keyed by mesh shape. Turbo has no CFG, so cfg factor is 1 (single submesh),
# sequence parallel is factor-1, and tensor parallel is 4 on axis 1 (the transformer is
# built replicated at TP factor-1 today; see the RISKS note in the module docstring / demo).
_PRESETS_WH: dict[tuple[int, ...], dict] = {
    (2, 4): {
        "cfg": (1, 0),
        "sp": (1, 0),
        "tp": (4, 1),
        "encoder_tp": (4, 1),
        "vae_tp": (4, 1),
        "num_links": 1,
        "is_fsdp": True,
        "dynamic_load_encoder": True,
        "dynamic_load_vae": True,
    },
}

_PRESETS_BH: dict[tuple[int, ...], dict] = {
    (2, 4): {
        "cfg": (1, 0),
        "sp": (1, 0),
        "tp": (4, 1),
        "encoder_tp": (4, 1),
        "vae_tp": (4, 1),
        "num_links": 1,
        "is_fsdp": True,
        "dynamic_load_encoder": True,
        "dynamic_load_vae": True,
    },
}


# ======================================================================================
# Host helpers (numerically mirror the reference)
# ======================================================================================
def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 6400,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Reference `calculate_shift` (pipeline_krea2.py lines 61-71).

    Note: for Turbo (is_distilled) mu is fixed at 1.15; this is kept for the base path
    and for numeric parity checks.
    """
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def make_sigmas(num_inference_steps: int) -> np.ndarray:
    """Reference default sigma grid (pipeline_krea2.py line 613)."""
    return np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)


def flowmatch_sigmas_with_shift(sigmas: np.ndarray, mu: float) -> list[float]:
    """Apply the FlowMatchEulerDiscreteScheduler exponential time-shift and append the
    trailing 0.0, matching `scheduler.set_timesteps(sigmas=..., mu=...)` followed by
    `scheduler.sigmas`.

    time_shift_type == "exponential" (scheduler_config.json): sigma' = exp(mu) / (exp(mu)
    + (1/sigma - 1)). The scheduler appends a terminal sigma of 0.0 (length steps + 1).
    """
    sigmas = np.asarray(sigmas, dtype=np.float64)
    shifted = np.exp(mu) / (np.exp(mu) + (1.0 / sigmas - 1.0))
    return shifted.tolist() + [0.0]


def timesteps_from_sigmas(shifted_sigmas: list[float]) -> list[float]:
    """FlowMatch timesteps are sigmas * num_train_timesteps (drop the trailing 0.0)."""
    return [s * NUM_TRAIN_TIMESTEPS for s in shifted_sigmas[:-1]]


def pack_latents(
    latents: torch.Tensor, batch_size: int, num_channels_latents: int, height: int, width: int
) -> torch.Tensor:
    """Reference `_pack_latents` (pipeline_krea2.py lines 357-363). patch_size=2."""
    p = PATCH_SIZE
    latents = latents.view(batch_size, num_channels_latents, height // p, p, width // p, p)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // p) * (width // p), num_channels_latents * p * p)
    return latents


def unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Reference `_unpack_latents` (pipeline_krea2.py lines 365-379). Returns (B, C, 1, H, W)."""
    batch_size, _, channels = latents.shape
    p = PATCH_SIZE
    height = p * (int(height) // (VAE_SCALE_FACTOR * p))
    width = p * (int(width) // (VAE_SCALE_FACTOR * p))
    latents = latents.view(batch_size, height // p, width // p, channels // (p * p), p, p)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // (p * p), 1, height, width)
    return latents


def prepare_position_ids(text_seq_len: int, grid_height: int, grid_width: int) -> torch.Tensor:
    """Reference `prepare_position_ids` (pipeline_krea2.py lines 381-390).

    Text tokens at (0,0,0); image tokens at (0, h, w) over the latent grid.
    Returns (text_seq_len + grid_height * grid_width, 3) torch tensor.
    """
    text_ids = torch.zeros(text_seq_len, 3)
    image_ids = torch.zeros(grid_height, grid_width, 3)
    image_ids[..., 1] = torch.arange(grid_height)[:, None]
    image_ids[..., 2] = torch.arange(grid_width)[None, :]
    image_ids = image_ids.reshape(grid_height * grid_width, 3)
    return torch.cat([text_ids, image_ids], dim=0)


def prepare_latents(
    batch_size: int,
    num_channels_latents: int,
    height: int,
    width: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Reference `prepare_latents` (pipeline_krea2.py lines 392-419): randn in latent grid
    then pack. Returns packed (B, image_seq_len, in_channels)."""
    latent_height = height // VAE_SCALE_FACTOR
    latent_width = width // VAE_SCALE_FACTOR
    shape = (batch_size, num_channels_latents, latent_height, latent_width)
    latents = torch.randn(shape, generator=generator, dtype=torch.float32)
    return pack_latents(latents, batch_size, num_channels_latents, latent_height, latent_width)


# ======================================================================================
# DiT weight loading (no diffusers-main needed at runtime)
# ======================================================================================
def _local_snapshot(checkpoint_name: str) -> str:
    """Resolve a local HF snapshot dir for the checkpoint (offline).

    Honours HF_HOME / HF_HUB_CACHE if set, else falls back to /localdev/vsuresh/hf_cache.
    """
    cache_root = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", "/localdev/vsuresh/hf_cache"), "hub"
    )
    repo_dir = os.path.join(cache_root, "models--" + checkpoint_name.replace("/", "--"), "snapshots")
    if os.path.isdir(repo_dir):
        snaps = [os.path.join(repo_dir, d) for d in os.listdir(repo_dir)]
        snaps = [s for s in snaps if os.path.isdir(s)]
        if snaps:
            return snaps[0]
    # Assume checkpoint_name is already a local directory path.
    if os.path.isdir(checkpoint_name):
        return checkpoint_name
    msg = f"could not resolve a local snapshot for {checkpoint_name!r} (cache root {cache_root})"
    raise FileNotFoundError(msg)


def load_krea2_dit_checkpoint(checkpoint_name: str = _DEFAULT_CHECKPOINT) -> Krea2Checkpoint:
    """Build a `Krea2Checkpoint` by reading transformer/config.json and merging the 3
    safetensors shards via the .index.json weight_map. Does not require diffusers-main.
    """
    from safetensors.torch import load_file

    snapshot = _local_snapshot(checkpoint_name)
    tdir = os.path.join(snapshot, "transformer")

    with open(os.path.join(tdir, "config.json")) as f:
        config = json.load(f)
    # Drop diffusers bookkeeping keys the Krea2Transformer ctor does not accept.
    config = {k: v for k, v in config.items() if not k.startswith("_")}

    index_path = os.path.join(tdir, "diffusion_pytorch_model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    shard_files = sorted(set(weight_map.values()))
    state_dict: dict[str, torch.Tensor] = {}
    for shard in shard_files:
        logger.info("loading DiT shard {}", shard)
        state_dict.update(load_file(os.path.join(tdir, shard)))

    missing = [k for k in weight_map if k not in state_dict]
    if missing:
        msg = f"missing {len(missing)} keys after merging shards, e.g. {missing[:3]}"
        raise RuntimeError(msg)

    logger.info("merged {} DiT tensors from {} shards", len(state_dict), len(shard_files))
    return Krea2Checkpoint(config=config, state_dict=state_dict)


# ======================================================================================
# Pipeline config
# ======================================================================================
@dataclass(frozen=True, kw_only=True)
class Krea2PipelineConfig:
    topology: ttnn.Topology
    num_links: int

    dit_parallel_config: DiTParallelConfig
    encoder_parallel_config: EncoderParallelConfig
    vae_parallel_config: VAEParallelConfig

    use_torch_text_encoder: bool
    use_torch_vae_decoder: bool

    height: int
    width: int

    is_fsdp: bool
    dynamic_load_encoder: bool
    dynamic_load_vae: bool

    checkpoint_name: str

    @classmethod
    def default(
        cls,
        *,
        mesh_shape: ttnn.MeshShape,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        num_links: int | None = None,
        use_torch_text_encoder: bool = False,
        use_torch_vae_decoder: bool = False,
        height: int = 1024,
        width: int = 1024,
        is_fsdp: bool | None = None,
        dynamic_load_encoder: bool | None = None,
        dynamic_load_vae: bool | None = None,
        checkpoint_name: str = _DEFAULT_CHECKPOINT,
    ) -> Krea2PipelineConfig:
        preset_dict = _PRESETS_BH if ttnn.device.is_blackhole() else _PRESETS_WH
        key = tuple(mesh_shape)
        if key not in preset_dict:
            msg = f"no KREA-2 preset for mesh shape {key}; supported: {list(preset_dict)}"
            raise ValueError(msg)
        preset = preset_dict[key]

        return cls(
            topology=topology,
            num_links=num_links if num_links is not None else preset["num_links"],
            dit_parallel_config=DiTParallelConfig.from_tuples(cfg=preset["cfg"], sp=preset["sp"], tp=preset["tp"]),
            encoder_parallel_config=EncoderParallelConfig.from_tuple(preset["encoder_tp"]),
            vae_parallel_config=VAEParallelConfig.from_tuple(preset["vae_tp"]),
            use_torch_text_encoder=use_torch_text_encoder,
            use_torch_vae_decoder=use_torch_vae_decoder,
            height=height,
            width=width,
            is_fsdp=is_fsdp if is_fsdp is not None else preset["is_fsdp"],
            dynamic_load_encoder=(
                dynamic_load_encoder if dynamic_load_encoder is not None else preset["dynamic_load_encoder"]
            ),
            dynamic_load_vae=dynamic_load_vae if dynamic_load_vae is not None else preset["dynamic_load_vae"],
            checkpoint_name=checkpoint_name,
        )


# ======================================================================================
# Pipeline
# ======================================================================================
class Krea2Pipeline:
    """KREA-2 Turbo text-to-image pipeline (single forward pass, no CFG).

    Dynamic load plan (single submesh = full mesh, since Turbo has no CFG):
      1. build encoder (TP=4, FSDP over the other axis) -> encode prompt on device
      2. free encoder weights
      3. build the DiT (Krea2Transformer, replicated) -> run the denoising loop
      4. free the DiT weights
      5. build the VAE decoder -> decode -> image

    The transformer is built once and kept resident through the denoising loop; the
    encoder and VAE are (optionally) freed around it to fit memory.
    """

    def __init__(self, *, device: ttnn.MeshDevice, config: Krea2PipelineConfig) -> None:
        self._mesh_device = device
        self._config = config
        self._parallel_config = config.dit_parallel_config
        self._encoder_parallel_config = config.encoder_parallel_config
        self._vae_parallel_config = config.vae_parallel_config
        self._height = config.height
        self._width = config.width
        self._checkpoint_name = config.checkpoint_name

        logger.info("KREA-2 pipeline on mesh {}", tuple(device.shape))
        logger.info("DiT parallel config: {}", self._parallel_config)

        self._ccl_manager = CCLManager(device, num_links=config.num_links, topology=config.topology)

        # Lazily-built components (built inside run() to control resident memory).
        self._text_encoder: TextEncoder | None = None
        self._checkpoint: Krea2Checkpoint | None = None
        self._transformer = None
        self._vae: QwenImageVAEDecoderAdapter | None = None
        self._solver = EulerSolver()

    # ---- factory -----------------------------------------------------------------------
    @classmethod
    def create_pipeline(
        cls,
        *,
        mesh_device: ttnn.MeshDevice,
        width: int = 1024,
        height: int = 1024,
        checkpoint_name: str = _DEFAULT_CHECKPOINT,
        use_torch_text_encoder: bool = False,
        use_torch_vae_decoder: bool = False,
    ) -> Krea2Pipeline:
        config = Krea2PipelineConfig.default(
            mesh_shape=mesh_device.shape,
            width=width,
            height=height,
            checkpoint_name=checkpoint_name,
            use_torch_text_encoder=use_torch_text_encoder,
            use_torch_vae_decoder=use_torch_vae_decoder,
        )
        return cls(device=mesh_device, config=config)

    # ---- component builders ------------------------------------------------------------
    def _build_text_encoder(self) -> None:
        if self._text_encoder is not None:
            return
        logger.info("building KREA-2 text encoder (Qwen3-VL text tower)...")
        self._text_encoder = TextEncoder(
            checkpoint_name=self._checkpoint_name,
            device=self._mesh_device,
            ccl_manager=self._ccl_manager,
            parallel_config=self._encoder_parallel_config,
            use_torch=self._config.use_torch_text_encoder,
            is_fsdp=self._config.is_fsdp,
        )
        ttnn.synchronize_device(self._mesh_device)

    def _free_text_encoder(self) -> None:
        if self._text_encoder is None:
            return
        logger.info("freeing KREA-2 text encoder...")
        self._text_encoder.deallocate_encoder_weights()
        self._text_encoder = None
        ttnn.synchronize_device(self._mesh_device)

    def _build_transformer(self) -> None:
        if self._transformer is not None:
            return
        logger.info("loading KREA-2 DiT checkpoint (merging shards)...")
        self._checkpoint = load_krea2_dit_checkpoint(self._checkpoint_name)
        logger.info("building KREA-2 DiT on device...")
        self._transformer = self._checkpoint.build(
            mesh_device=self._mesh_device,
            ccl_manager=self._ccl_manager,
            parallel_config=self._parallel_config,
        )
        ttnn.synchronize_device(self._mesh_device)

    def _free_transformer(self) -> None:
        if self._transformer is None:
            return
        logger.info("freeing KREA-2 DiT...")
        self._transformer = None
        self._checkpoint = None
        ttnn.synchronize_device(self._mesh_device)

    def _build_vae(self) -> None:
        if self._vae is not None:
            return
        logger.info("building KREA-2 VAE decoder...")
        vae_hw = VaeHWParallelConfig(
            height_parallel=ParallelFactor(
                factor=self._mesh_device.shape[self._vae_parallel_config.tensor_parallel.mesh_axis],
                mesh_axis=self._vae_parallel_config.tensor_parallel.mesh_axis,
            ),
            # Do NOT shard width: on the full (2,4) mesh the non-TP axis has size 2, and
            # width-sharding the VAE across it produces a half-decoded (right-half-gray)
            # image. Qwen-Image avoids this by running the VAE on a (1,4) submesh (width
            # factor 1); we keep width unsharded here (replicated across the non-TP axis).
            width_parallel=ParallelFactor(
                factor=1,
                mesh_axis=1 - self._vae_parallel_config.tensor_parallel.mesh_axis,
            ),
        )
        self._vae = QwenImageVAEDecoderAdapter(
            checkpoint_name=self._checkpoint_name,
            parallel_config=vae_hw,
            ccl_manager=self._ccl_manager,
            use_torch=self._config.use_torch_vae_decoder,
        )
        # QwenImageVAEDecoderAdapter reads latents_mean/std from the KREA-2 vae/config.json
        # (subfolder="vae") itself, so no manual wiring is required here.
        if not self._config.use_torch_vae_decoder:
            self._vae.reload_weights()
        ttnn.synchronize_device(self._mesh_device)

    # ---- entry point -------------------------------------------------------------------
    @torch.no_grad()
    def run(
        self,
        prompt: str | Sequence[str],
        *,
        num_inference_steps: int = TURBO_STEPS,
        seed: int = 0,
        height: int | None = None,
        width: int | None = None,
        max_sequence_length: int = MAX_SEQUENCE_LENGTH,
        output_type: str = "pil",
    ) -> list[Image.Image] | torch.Tensor:
        """Generate one image for a single prompt (Turbo: single forward pass, no CFG)."""
        height = height if height is not None else self._height
        width = width if width is not None else self._width

        if isinstance(prompt, str):
            prompt = [prompt]
        assert len(prompt) == 1, "KREA-2 tt pipeline currently supports a single prompt"

        if height % PIXEL_TO_TOKEN != 0 or width % PIXEL_TO_TOKEN != 0:
            msg = f"height/width must be multiples of {PIXEL_TO_TOKEN}, got {height}x{width}"
            raise ValueError(msg)

        # --- 1. encode prompt (device tensors produced on host as torch) ---------------
        self._build_text_encoder()
        logger.info("encoding prompt...")
        prompt_embeds, prompt_embeds_mask = self._text_encoder.get_text_hidden_states(
            prompt, max_sequence_length=max_sequence_length
        )
        _dbg = os.environ.get("KREA2_DEBUG")
        if _dbg:
            logger.info(
                "DBG prompt_embeds shape={} mean={:.4f} std={:.4f} mask_sum={}",
                tuple(prompt_embeds.shape),
                prompt_embeds.float().mean().item(),
                prompt_embeds.float().std().item(),
                int(prompt_embeds_mask.sum().item()),
            )
        # Reference zeros embeddings at padded positions before consumption is NOT done for
        # KREA-2 (the transformer builds its own additive mask from the bool mask). We pass
        # the raw embeds + bool mask, matching the reference transformer call.
        self._free_text_encoder()

        # --- 2. prepare latents, position ids, schedule (host) -------------------------
        generator = torch.Generator().manual_seed(seed)
        latents_torch = prepare_latents(
            batch_size=1,
            num_channels_latents=NUM_CHANNELS_LATENTS,
            height=height,
            width=width,
            generator=generator,
        )  # (1, image_seq_len, 64)
        image_seq_len = latents_torch.shape[1]

        grid_height = height // PIXEL_TO_TOKEN
        grid_width = width // PIXEL_TO_TOKEN
        text_seq_len = prompt_embeds.shape[1]
        position_ids = prepare_position_ids(text_seq_len, grid_height, grid_width)

        # Turbo: fixed mu, guidance disabled.
        sigmas = make_sigmas(num_inference_steps)
        mu = TURBO_MU
        shifted_sigmas = flowmatch_sigmas_with_shift(sigmas, mu)
        timesteps = timesteps_from_sigmas(shifted_sigmas)
        self._solver.set_schedule(shifted_sigmas)

        # --- 3. build DiT and run the denoising loop -----------------------------------
        self._build_transformer()

        latents = bf16_tensor(latents_torch, device=self._mesh_device)
        encoder_hidden_states = bf16_tensor(prompt_embeds, device=self._mesh_device)

        logger.info("denoising ({} steps)...", len(timesteps))
        for i, t in enumerate(tqdm.tqdm(timesteps)):
            # Reference (pipeline_krea2.py line 644): timestep = t / num_train_timesteps in [0,1].
            timestep = torch.tensor([t / NUM_TRAIN_TIMESTEPS], dtype=torch.float32)

            noise_pred = self._transformer.forward(
                latents,
                encoder_hidden_states,
                timestep,
                position_ids,
                encoder_attention_mask=prompt_embeds_mask,
            )
            # Euler flow-matching step (matches scheduler.step / EulerSolver.step).
            if _dbg:
                _v = dit_to_torch(noise_pred).float()
                _l = dit_to_torch(latents).float()
                logger.info(
                    "DBG step {} t={:.4f} vel(mean={:.4f} std={:.4f}) lat(mean={:.4f} std={:.4f})",
                    i,
                    (t / NUM_TRAIN_TIMESTEPS),
                    _v.mean().item(),
                    _v.std().item(),
                    _l.mean().item(),
                    _l.std().item(),
                )
            latents = self._solver.step(step=i, latent=latents, velocity_pred=noise_pred)
            ttnn.synchronize_device(self._mesh_device)

        # Bring final latents back to host for unpack + VAE (reference unpacks on host).
        # `latents` is replicated across the mesh; use the tt_dit helper to strip the
        # redundant per-device copies (raw ttnn.to_torch needs a mesh composer otherwise).
        latents_out = dit_to_torch(latents).reshape(1, image_seq_len, NUM_CHANNELS_LATENTS * PATCH_SIZE * PATCH_SIZE)
        self._free_transformer()

        if output_type == "latent":
            return latents_out

        # --- 4. unpack + decode --------------------------------------------------------
        # Reference _unpack_latents -> (B, C, 1, H, W); the VAE adapter expects torch NHWC
        # (B, H, W, C) and unsqueezes the temporal dim itself.
        unpacked = unpack_latents(latents_out, height, width)  # (1, 16, 1, lat_h, lat_w)
        vae_in = unpacked[:, :, 0].permute(0, 2, 3, 1).contiguous()  # (1, lat_h, lat_w, 16)

        if _dbg:
            lat = unpacked[:, :, 0]  # (1,16,H,W)
            W = lat.shape[-1]
            lh, rh = lat[..., : W // 2], lat[..., W // 2 :]
            logger.info(
                "DBG final latents_out mean={:.4f} std={:.4f}",
                latents_out.float().mean().item(),
                latents_out.float().std().item(),
            )
            logger.info(
                "DBG unpacked left-half(mean={:.4f} std={:.4f}) right-half(mean={:.4f} std={:.4f})",
                lh.float().mean().item(),
                lh.float().std().item(),
                rh.float().mean().item(),
                rh.float().std().item(),
            )

        self._build_vae()
        logger.info("decoding image...")
        decoded = self._vae.decode(vae_in, traced=False)  # (B, C, H, W) in [-1, 1] range-ish
        if _dbg:
            dc = decoded.float()
            W = dc.shape[-1]
            logger.info(
                "DBG decoded shape={} left-half mean={:.4f} right-half mean={:.4f}",
                tuple(dc.shape),
                dc[..., : W // 2].mean().item(),
                dc[..., W // 2 :].mean().item(),
            )

        return self._postprocess(decoded, output_type=output_type)

    # allow `pipeline(prompt=...)` too
    __call__ = run

    @staticmethod
    def _postprocess(decoded: torch.Tensor, *, output_type: str) -> list[Image.Image] | torch.Tensor:
        from diffusers.image_processor import VaeImageProcessor

        processor = VaeImageProcessor(vae_scale_factor=PIXEL_TO_TOKEN)
        image = processor.postprocess(decoded, output_type=output_type)
        return image
