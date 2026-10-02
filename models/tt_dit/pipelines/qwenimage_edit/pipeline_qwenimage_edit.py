# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image-Edit pipeline on the shared tt_dit infra.

Architecture
------------
The transformer denoise -- which dominates compute -- runs on the Galaxy using the shared
``QwenImageTransformer`` at **TP=8 x SP=4** (all 32 Wormhole chips on a single image), the same
sequence-parallel layout Flux2 uses on WH Galaxy.

Host pre/post-processing (Qwen2.5-VL image+text encode and VAE encode/decode) is driven by the
reference ``diffusers.QwenImageEditPipeline``, because the tt_dit device encoder is text-only (no
vision tower) and the tt_dit VAE adapter is decode-only. Those device paths are a tracked follow-up;
the ``encoder_tp``/``vae_tp`` presets already exist in the base ``pipelines/qwenimage`` for when they
land. Until then this pipeline cleanly owns the device denoise and delegates the rest to the
reference implementation so results stay bit-faithful to Qwen-Image-Edit's host plumbing.

Edit-specific deltas handled by the reference host loop (vs. base text-to-image):
  * condition image is VAE-encoded and concatenated on the *token* dim: ``cat([latents, image_latents], dim=1)``
  * ``img_shapes`` has two entries (noise grid + condition grid) for RoPE
  * the VL encoder is conditioned on the input image
  * true-CFG with norm rescale
"""
from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

import diffusers as reference
import torch
from loguru import logger
from PIL import Image, ImageOps

import ttnn
from models.tt_dit.models.transformers.transformer_qwenimage import QwenImageTransformer
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.qwenimage_edit.vae_device import DeviceVAE
from models.tt_dit.utils import cache, tensor
from models.tt_dit.utils.padding import PaddingConfig
from models.tt_dit.utils.tracing import Tracer

if TYPE_CHECKING:
    from collections.abc import Sequence

_DEFAULT_CHECKPOINT = "Qwen/Qwen-Image-Edit"

# WH Galaxy preset: all 32 chips on one image, cfg replicated, TP across heads, SP across tokens.
#   axis 0 -> sequence parallel (4),  axis 1 -> tensor parallel (8)
_PRESETS_WH: dict[tuple[int, ...], dict] = {
    (4, 8): {"sp": (4, 0), "tp": (8, 1), "num_links": 4},
}


@dataclass(frozen=True, kw_only=True)
class QwenImageEditPipelineConfig:
    """Static configuration for :class:`QwenImageEditPipeline`."""

    topology: ttnn.Topology
    num_links: int
    dit_parallel_config: DiTParallelConfig
    checkpoint_name: str

    @classmethod
    def default(
        cls,
        *,
        mesh_shape: ttnn.MeshShape,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        num_links: int | None = None,
        checkpoint_name: str = _DEFAULT_CHECKPOINT,
    ) -> QwenImageEditPipelineConfig:
        preset = _PRESETS_WH.get(tuple(mesh_shape))
        if preset is None:
            msg = (
                f"No Qwen-Image-Edit preset for mesh shape {tuple(mesh_shape)}; "
                f"supported shapes: {list(_PRESETS_WH)}"
            )
            raise ValueError(msg)

        # No CFG parallelism: cond/uncond run sequentially on the full mesh so every forward uses
        # all 32 chips (TP=8 x SP=4). This is the literal "all chips on one image" layout.
        dit_parallel_config = DiTParallelConfig(
            cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
            sequence_parallel=ParallelFactor(factor=preset["sp"][0], mesh_axis=preset["sp"][1]),
            tensor_parallel=ParallelFactor(factor=preset["tp"][0], mesh_axis=preset["tp"][1]),
        )

        return cls(
            topology=topology,
            num_links=num_links if num_links is not None else preset["num_links"],
            dit_parallel_config=dit_parallel_config,
            checkpoint_name=checkpoint_name,
        )


@dataclass
class _StepInvariants:
    """Per-context tensors that don't change across denoise steps (RoPE + prompt + seq lengths).

    RoPE depends only on ``img_shapes``/``txt_seq`` and the prompt embeddings are fixed for a given
    conditioning context, so these are built once and *baked into the trace* -- only the latents and
    timestep move per step. ``sig`` guards against a resolution/prompt-length change across calls.
    """

    sig: tuple
    prompt: ttnn.Tensor
    s_cos: ttnn.Tensor
    s_sin: ttnn.Tensor
    p_cos: ttnn.Tensor
    p_sin: ttnn.Tensor
    combined_seq: int
    txt_seq: int


class _DeviceTransformer:
    """Drop-in replacement for ``QwenImageTransformer2DModel`` that runs on the Galaxy.

    Mirrors the subset of the HF transformer surface the reference edit pipeline touches: ``.config``,
    ``.pos_embed``, a no-op ``cache_context``, and a ``__call__`` with the HF signature. Each call
    shards the combined (noise + condition) token sequence across the SP axis, runs the shared TT
    transformer, and gathers the result back to host. RoPE + prompt embeddings are step-invariant,
    so they are built once per context (:class:`_StepInvariants`) and baked into the trace; only the
    latents and timestep are transferred per forward.
    """

    def __init__(
        self,
        *,
        tt_model: QwenImageTransformer,
        config: object,
        pos_embed: object,
        mesh_device: ttnn.MeshDevice,
        sp_axis: int,
        trace: bool = True,
        batch_cfg: bool = False,
    ) -> None:
        self.config = config
        self.pos_embed = pos_embed
        self._tt = tt_model
        self._device = mesh_device
        self._sp_axis = sp_axis
        self._sp_factor = tuple(mesh_device.shape)[sp_axis]
        self._trace = trace
        # CFG batching: run cond+uncond as a single batch-2 forward (same latents, [cond, uncond]
        # prompts zero-padded to a common length). Halves the number of device forwards. Mirrors the
        # base qwenimage single-mesh path (``ttnn.concat([latents, latents])`` + zeroed prompt pad).
        self._batch_cfg = batch_cfg
        # One trace per key ("cond"/"uncond" when not batching; "batch" when batching).
        self._tracers: dict[str, Tracer] = {}
        # Step-invariant RoPE/prompt tensors, cached per key (see _StepInvariants).
        self._invariants: dict[str, _StepInvariants] = {}
        self._ctx = "cond"
        # Batch-CFG interception state: the negative embeds are captured once (constant across
        # steps); the uncond half of each batched forward is cached and returned on the uncond call.
        self._uncond_eh: torch.Tensor | None = None
        self._pending_uncond_out: torch.Tensor | None = None
        self.forward_times: list[float] = []

    def reset_cfg_state(self) -> None:
        """Clear captured negative embeds / cached halves (call per new __call__)."""
        self._uncond_eh = None
        self._pending_uncond_out = None

    def cache_context(self, name: str):  # noqa: ANN202 - matches HF's context-manager API
        self._ctx = name
        return nullcontext()

    def release_traces(self) -> None:
        for tracer in self._tracers.values():
            tracer.release_trace()
        self._tracers.clear()
        self._invariants.clear()

    def _build_invariants(
        self,
        key: str,
        *,
        prompt: torch.Tensor,
        img_shapes: list,
        combined_seq: int,
        txt_seq: int,
        batch: int,
    ) -> _StepInvariants:
        """Build (or reuse) the step-invariant RoPE + prompt tensors for ``key``."""
        sig = (combined_seq, txt_seq, batch, repr(img_shapes))
        cached = self._invariants.get(key)
        if cached is not None and cached.sig == sig:
            return cached

        # A resolution / prompt-length change invalidates any trace captured under this key.
        stale = self._tracers.pop(key, None)
        if stale is not None:
            stale.release_trace()

        dev = self._device
        spatial_rope, prompt_rope = self.pos_embed.forward(img_shapes, [txt_seq] * batch, "cpu")
        inv = _StepInvariants(
            sig=sig,
            prompt=tensor.from_torch(prompt.to(torch.float32), device=dev),
            s_cos=tensor.from_torch(
                spatial_rope.real.repeat_interleave(2, dim=-1), device=dev, mesh_axes=[self._sp_axis, None]
            ),
            s_sin=tensor.from_torch(
                spatial_rope.imag.repeat_interleave(2, dim=-1), device=dev, mesh_axes=[self._sp_axis, None]
            ),
            p_cos=tensor.from_torch(prompt_rope.real.repeat_interleave(2, dim=-1), device=dev),
            p_sin=tensor.from_torch(prompt_rope.imag.repeat_interleave(2, dim=-1), device=dev),
            combined_seq=combined_seq,
            txt_seq=txt_seq,
        )
        self._invariants[key] = inv
        return inv

    def _run(
        self,
        key: str,
        *,
        hidden_states: torch.Tensor,
        prompt: torch.Tensor,
        img_shapes: list,
        timestep: torch.Tensor,
        allow_trace: bool,
    ) -> torch.Tensor:
        """Run one (possibly batched) transformer forward and gather the result to host."""
        batch, combined_seq, _ = hidden_states.shape
        txt_seq = prompt.shape[1]
        dev, sp = self._device, self._sp_axis

        inv = self._build_invariants(
            key, prompt=prompt, img_shapes=img_shapes, combined_seq=combined_seq, txt_seq=txt_seq, batch=batch
        )

        # Only the latents + timestep move per step; RoPE/prompt are the same cached device objects
        # (Tracer manages them as stable buffers and skips the copy when the address is unchanged).
        # The reference divides the timestep by 1000; the TT time-embedding wants raw scale.
        tt_spatial = tensor.from_torch(hidden_states.to(torch.float32), device=dev, mesh_axes=[None, sp, None])
        tt_timestep = tensor.from_torch(
            timestep.reshape(batch, 1).to(torch.float32) * 1000.0, dtype=ttnn.float32, device=dev
        )
        forward_kwargs = {
            "spatial": tt_spatial,
            "prompt": inv.prompt,
            "timestep": tt_timestep,
            "spatial_rope": (inv.s_cos, inv.s_sin),
            "prompt_rope": (inv.p_cos, inv.p_sin),
            "spatial_sequence_length": inv.combined_seq,
            "prompt_sequence_length": inv.txt_seq,
        }

        ttnn.synchronize_device(dev)
        t0 = time.time()
        if self._trace and allow_trace:
            tracer = self._tracers.get(key)
            if tracer is None:
                tracer = Tracer(self._tt.forward, device=dev, prep_run=True)
                self._tracers[key] = tracer
            out = tracer(**forward_kwargs, traced=True)
        else:
            out = self._tt.forward(**forward_kwargs)
        ttnn.synchronize_device(dev)
        self.forward_times.append(time.time() - t0)

        return tensor.to_torch(out, mesh_axes=[None, sp, None]).to(hidden_states.dtype)

    @staticmethod
    def _pad_tokens(x: torch.Tensor, length: int) -> torch.Tensor:
        """Zero-pad the token (dim-1) axis up to ``length`` (padding rows are already zero in HF)."""
        if x.shape[1] >= length:
            return x
        pad = x.new_zeros(x.shape[0], length - x.shape[1], x.shape[2])
        return torch.cat([x, pad], dim=1)

    def _wrap(self, torch_out: torch.Tensor, return_dict: bool):  # noqa: ANN201
        if return_dict:
            return reference.models.modeling_outputs.Transformer2DModelOutput(sample=torch_out)
        return (torch_out,)

    def __call__(
        self,
        *,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance: torch.Tensor | None,  # noqa: ARG002 - edit model is not guidance-distilled
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,  # noqa: ARG002 - shared transformer has no key mask
        img_shapes: list,
        attention_kwargs: dict | None = None,  # noqa: ARG002
        return_dict: bool = False,
    ):
        _batch, combined_seq, _ = hidden_states.shape
        if combined_seq % self._sp_factor != 0:
            msg = (
                f"Combined token sequence ({combined_seq}) is not divisible by the SP factor "
                f"({self._sp_factor}); use a square resolution so noise and condition token counts align."
            )
            raise ValueError(msg)

        # --- uncond call: capture negative embeds (constant); return the cached batched half. ---
        if self._ctx == "uncond":
            self._uncond_eh = encoder_hidden_states
            if self._pending_uncond_out is not None:
                out = self._pending_uncond_out
                self._pending_uncond_out = None
                return self._wrap(out, return_dict)
            # Step 0 (no batched half yet): run uncond on its own, eagerly (one-shot, don't trace).
            out = self._run(
                "uncond",
                hidden_states=hidden_states,
                prompt=encoder_hidden_states,
                img_shapes=img_shapes,
                timestep=timestep,
                allow_trace=not self._batch_cfg,
            )
            return self._wrap(out, return_dict)

        # --- cond call: batch cond+uncond into one forward once the negative embeds are known. ---
        if self._batch_cfg and self._uncond_eh is not None:
            length = max(encoder_hidden_states.shape[1], self._uncond_eh.shape[1])
            batched_prompt = torch.cat(
                [self._pad_tokens(encoder_hidden_states, length), self._pad_tokens(self._uncond_eh, length)], dim=0
            )
            batched_hs = torch.cat([hidden_states, hidden_states], dim=0)
            t = timestep.reshape(-1)[:1]
            batched_t = torch.cat([t, t], dim=0)
            out = self._run(
                "batch",
                hidden_states=batched_hs,
                prompt=batched_prompt,
                img_shapes=img_shapes * 2,
                timestep=batched_t,
                allow_trace=True,
            )
            self._pending_uncond_out = out[1:2]
            return self._wrap(out[0:1], return_dict)

        # Step 0 cond (or batching disabled): single forward.
        out = self._run(
            "cond",
            hidden_states=hidden_states,
            prompt=encoder_hidden_states,
            img_shapes=img_shapes,
            timestep=timestep,
            allow_trace=not self._batch_cfg,
        )
        return self._wrap(out, return_dict)


class QwenImageEditPipeline:
    """Qwen-Image-Edit with the denoise running on the WH Galaxy at TP=8 x SP=4.

    Device compute (the transformer) is owned by tt_dit; VL image+text encode and VAE encode/decode
    are delegated to the reference ``diffusers.QwenImageEditPipeline`` until device equivalents exist.
    """

    @classmethod
    def create_pipeline(
        cls,
        *,
        mesh_device: ttnn.MeshDevice,
        checkpoint_name: str = _DEFAULT_CHECKPOINT,
        trace: bool = True,
        batch_cfg: bool = False,  # measured regression at TP=8xSP=4: batch-2 forward ~3.1x batch-1
        device_vae: bool = True,
        device_vae_encode: bool = True,
    ) -> QwenImageEditPipeline:
        config = QwenImageEditPipelineConfig.default(
            mesh_shape=mesh_device.shape,
            checkpoint_name=checkpoint_name,
        )
        return cls(
            device=mesh_device,
            config=config,
            trace=trace,
            batch_cfg=batch_cfg,
            device_vae=device_vae,
            device_vae_encode=device_vae_encode,
        )

    def __init__(
        self,
        *,
        device: ttnn.MeshDevice,
        config: QwenImageEditPipelineConfig,
        trace: bool = True,
        batch_cfg: bool = False,
        device_vae: bool = True,
        device_vae_encode: bool = True,
    ) -> None:
        self._mesh_device = device
        self._config = config
        self._parallel_config = config.dit_parallel_config

        sp = self._parallel_config.sequence_parallel
        tp = self._parallel_config.tensor_parallel
        logger.info(f"Qwen-Image-Edit parallel config: {self._parallel_config}")
        logger.info(
            f"Mesh shape: {tuple(device.shape)} (SP={sp.factor}@axis{sp.mesh_axis}, TP={tp.factor}@axis{tp.mesh_axis})"
        )

        logger.info("loading reference QwenImageEditPipeline (host: VL encode, VAE, scheduler)...")
        self._hf = reference.QwenImageEditPipeline.from_pretrained(config.checkpoint_name, torch_dtype=torch.bfloat16)
        hf_transformer = self._hf.transformer
        cfg = hf_transformer.config

        self._ccl_manager = CCLManager(mesh_device=device, num_links=config.num_links, topology=config.topology)
        padding_config = (
            PaddingConfig.from_tensor_parallel_factor(cfg.num_attention_heads, cfg.attention_head_dim, tp.factor)
            if cfg.num_attention_heads % tp.factor != 0
            else None
        )

        logger.info("building TT transformer + loading edit weights...")
        tt_model = QwenImageTransformer(
            patch_size=cfg.patch_size,
            in_channels=cfg.in_channels,
            num_layers=cfg.num_layers,
            attention_head_dim=cfg.attention_head_dim,
            num_attention_heads=cfg.num_attention_heads,
            joint_attention_dim=cfg.joint_attention_dim,
            out_channels=cfg.out_channels,
            device=device,
            ccl_manager=self._ccl_manager,
            parallel_config=self._parallel_config,
            padding_config=padding_config,
        )
        cache.load_model(
            tt_model=tt_model,
            get_torch_state_dict=hf_transformer.state_dict,
            model_name=_model_name_for_cache(config.checkpoint_name),
            subfolder="transformer",
            parallel_config=self._parallel_config,
            mesh_shape=tuple(device.shape),
            mesh_device=device,
        )

        # Swap the device-backed transformer into the host pipeline.
        self._device_transformer = _DeviceTransformer(
            tt_model=tt_model,
            config=cfg,
            pos_embed=hf_transformer.pos_embed,
            mesh_device=device,
            sp_axis=sp.mesh_axis,
            trace=trace,
            batch_cfg=batch_cfg,
        )
        self._hf.transformer = self._device_transformer

        # Swap the device-backed VAE (encode + decode on the Galaxy) into the host pipeline.
        if device_vae:
            logger.info(f"building device VAE (encode={device_vae_encode}, decode=True)...")
            # Match the base qwenimage VAE convention: height on the TP axis, width on the SP axis.
            self._hf.vae = DeviceVAE(
                checkpoint_name=config.checkpoint_name,
                mesh_device=device,
                ccl_manager=self._ccl_manager,
                height_axis=tp.mesh_axis,
                width_axis=sp.mesh_axis,
                device_encode=device_vae_encode,
            )
        logger.info("Qwen-Image-Edit pipeline ready.")

    @property
    def last_forward_times(self) -> list[float]:
        """Per-transformer-forward device times (seconds) from the most recent ``__call__``."""
        return self._device_transformer.forward_times

    def __call__(
        self,
        *,
        image: Image.Image,
        prompt: str | Sequence[str],
        negative_prompt: str | Sequence[str] = " ",
        num_inference_steps: int = 20,
        true_cfg_scale: float = 4.0,
        side: int = 1024,
        letterbox: bool = True,
        seed: int = 0,
    ) -> list[Image.Image]:
        # The SP ring-attention kernel requires the combined (noise + condition) token sequence to
        # divide evenly across the SP axis and be tile-aligned. A square ``side`` canvas guarantees
        # this; letterboxing preserves the input aspect ratio (no distortion) by padding to square.
        if letterbox:
            pad_color = image.getpixel((2, 2)) if image.size[0] > 2 and image.size[1] > 2 else (0, 0, 0)
            image = ImageOps.pad(image.convert("RGB"), (side, side), color=pad_color)
        else:
            image = image.convert("RGB").resize((side, side))

        self._device_transformer.forward_times.clear()
        self._device_transformer.reset_cfg_state()
        generator = torch.Generator().manual_seed(seed)

        t_start = time.time()
        out = self._hf(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=side,
            width=side,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            generator=generator,
        )
        wall = time.time() - t_start

        fwd = self._device_transformer.forward_times
        warm = fwd[2:] if len(fwd) > 2 else fwd
        per_fwd_ms = (sum(warm) / len(warm) * 1000.0) if warm else float("nan")
        logger.info(
            f"[qwen-image-edit] {num_inference_steps} steps, {len(fwd)} forwards | "
            f"warm per-forward {per_fwd_ms:.1f} ms | denoise {sum(fwd):.1f} s | wall {wall:.1f} s"
        )
        return list(out.images)


def _model_name_for_cache(checkpoint_name: str) -> str:
    """Stable, filesystem-friendly model name for the weight cache."""
    return checkpoint_name.split("/")[-1].lower()
