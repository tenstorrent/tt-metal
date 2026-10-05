# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image-Edit pipeline on the shared tt_dit infra.

Architecture
------------
The transformer denoise -- which dominates compute -- runs on the Galaxy using the shared
``QwenImageTransformer``. Two layouts on the 32-chip (4, 8) mesh:

* CFG-parallel (default): two 4x4 submeshes, TP=4 x SP=4 each; the cond and uncond forwards of a
  step run concurrently, one per submesh (the base ``pipelines/qwenimage`` layout at (4, 8)).
* sequential: TP=8 x SP=4 on the full mesh; cond and uncond run back to back.

The Qwen2.5-VL image+text encode runs on host via the reference ``diffusers.QwenImageEditPipeline``
(the tt_dit device encoder is text-only, no vision tower); the VAE encode + decode run on device
(:mod:`.vae_device`). The denoise loop reproduces the reference loop's math step for step.

Edit-specific deltas vs. base text-to-image:
  * condition image is VAE-encoded and concatenated on the *token* dim: ``cat([latents, image_latents], dim=1)``
  * ``img_shapes`` has two entries (noise grid + condition grid) for RoPE
  * the VL encoder is conditioned on the input image
  * true-CFG with norm rescale
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import diffusers as reference
import numpy as np
import torch
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import calculate_dimensions, calculate_shift
from loguru import logger
from PIL import Image, ImageOps

import ttnn
from models.tt_dit.models.transformers.transformer_qwenimage import QwenImageTransformer
from models.tt_dit.parallel.config import DiTParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.cfg import create_submeshes
from models.tt_dit.pipelines.qwenimage_edit.vae_device import DeviceVAE
from models.tt_dit.utils import cache, tensor
from models.tt_dit.utils.padding import PaddingConfig
from models.tt_dit.utils.tracing import Tracer

_DEFAULT_CHECKPOINT = "Qwen/Qwen-Image-Edit"

# Prompt-length bucketing for the denoise trace. The trace is keyed on the prompt token
# length, which is ~1369 condition-image tokens plus the instruction, so nearly every new
# instruction changes it and re-captures the trace (~5 s). With a bucket, the prompt
# embeddings are zero-padded up to the next multiple of it (the base qwenimage pipeline
# zero-pads to a fixed 512 the same way), so instructions within a bucket share one trace.
# The padded tokens are attended to (the joint attention has no mask for a replicated
# prompt), which perturbs the output slightly; None disables padding.
_DEFAULT_PROMPT_BUCKET: int | None = None

# WH Galaxy preset: all 32 chips on one image, cfg replicated, TP across heads, SP across tokens.
#   axis 0 -> sequence parallel (4),  axis 1 -> tensor parallel (8)
_PRESETS_WH: dict[tuple[int, ...], dict] = {
    (4, 8): {"cfg": (1, 0), "sp": (4, 0), "tp": (8, 1), "num_links": 4},
}

# CFG-parallel preset: two 4x4 submeshes (split along axis 1), cond on one and uncond on the other,
# each TP=4 x SP=4. Same layout as the base Qwen-Image pipeline at (4, 8).
_PRESETS_WH_CFG_PARALLEL: dict[tuple[int, ...], dict] = {
    (4, 8): {"cfg": (2, 1), "sp": (4, 0), "tp": (4, 1), "num_links": 4},
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
        cfg_parallel: bool = False,
    ) -> QwenImageEditPipelineConfig:
        presets = _PRESETS_WH_CFG_PARALLEL if cfg_parallel else _PRESETS_WH
        preset = presets.get(tuple(mesh_shape))
        if preset is None:
            msg = f"No Qwen-Image-Edit preset for mesh shape {tuple(mesh_shape)}; " f"supported shapes: {list(presets)}"
            raise ValueError(msg)

        # cfg=1: cond/uncond run sequentially on the full mesh (TP=8 x SP=4).
        # cfg=2: cond/uncond run concurrently on two half-mesh submeshes (TP=4 x SP=4 each).
        dit_parallel_config = DiTParallelConfig.from_tuples(cfg=preset["cfg"], sp=preset["sp"], tp=preset["tp"])

        return cls(
            topology=topology,
            num_links=num_links if num_links is not None else preset["num_links"],
            dit_parallel_config=dit_parallel_config,
            checkpoint_name=checkpoint_name,
        )


class _VisionFeatureCache:
    """Memoize the Qwen2.5-VL vision tower across the prompt and negative-prompt encodes.

    The reference pipeline encodes the prompt and the negative prompt separately, each with the same
    condition image, so the vision tower (~45% of a host VL encode at 1024^2) runs twice on identical
    pixels. This wraps ``get_image_features`` on the VL model instance and returns the cached output
    when the pixels and grid match the previous call.
    """

    def __init__(self, vl_model: torch.nn.Module) -> None:
        self._fn = vl_model.get_image_features
        self._key: tuple[torch.Tensor, torch.Tensor] | None = None
        self._value = None
        vl_model.get_image_features = self

    def __call__(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor | None = None, **kwargs
    ):  # noqa: ANN204
        if (
            self._key is not None
            and image_grid_thw is not None
            and self._key[0].shape == pixel_values.shape
            and torch.equal(self._key[1], image_grid_thw)
            and torch.equal(self._key[0], pixel_values)
        ):
            return self._value
        self._value = self._fn(pixel_values, image_grid_thw, **kwargs)
        self._key = (pixel_values.clone(), image_grid_thw.clone() if image_grid_thw is not None else None)
        return self._value


class _DenoiseBranch:
    """One CFG branch (cond or uncond) of the traced transformer.

    ``launch`` writes the latents and this step's timestep modulation, then enqueues the traced
    forward without blocking, so branches on different submeshes execute concurrently; ``collect``
    reads the result back.

    All trace inputs (and the per-call modulation table they are refreshed from) live in persistent
    device buffers that are written in place and handed to the Tracer as-is, so it never copies them.
    A device tensor allocated after a trace is captured can be overwritten when that trace replays,
    so with two traces on one mesh (sequential CFG) every branch's buffers must be allocated before
    the first capture: callers run
    ``prepare`` on all branches before the first ``launch``, and on a signature change
    (``needs_reset``) ``reset`` all branches first.
    """

    def __init__(
        self,
        *,
        tt_model: QwenImageTransformer,
        device: ttnn.MeshDevice,
        sp_axis: int,
        tp_axis: int,
        trace: bool,
        use_2cq: bool = False,
    ) -> None:
        self._tt = tt_model
        # Two command queues: CQ1 carries the per-step latent write and the output read-back,
        # CQ0 the (traced) forward; events order them (write -> forward -> read). The mesh must
        # be opened with num_command_queues=2.
        self._use_2cq = use_2cq
        self._op_event = None  # CQ0: last forward done
        self._read_event = None  # CQ1: last read-back done
        self._device = device
        self._sp_axis = sp_axis
        self._tp_axis = tp_axis
        self._trace = trace
        self._tracer: Tracer | None = None
        self._sig: tuple | None = None
        self._inputs: dict | None = None
        self._mod_table: ttnn.Tensor | None = None
        self._mod_key: tuple | None = None
        self._out: ttnn.Tensor | None = None
        self._pending: tuple[list[ttnn.Tensor], int] | None = None

    @staticmethod
    def signature(*, prompt: torch.Tensor, img_shapes: list, combined_seq: int, num_steps: int) -> tuple:
        return (combined_seq, prompt.shape[1], repr(img_shapes), num_steps)

    def needs_reset(self, sig: tuple) -> bool:
        return self._sig is not None and sig != self._sig

    def reset(self) -> None:
        if self._tracer is not None:
            self._tracer.release_trace()
        self._tracer = None
        self._sig = None
        self._inputs = None
        self._mod_table = None
        self._mod_key = None

    def prepare(
        self,
        *,
        pos_embed: object,
        prompt: torch.Tensor,
        img_shapes: list,
        combined_seq: int,
        in_channels: int,
        timesteps: torch.Tensor,
    ) -> None:
        """Write the step-invariant inputs (prompt, RoPE, modulation table); allocate per-step buffers.

        The timestep-only modulation of every step (``compute_modulation``) is computed in one
        batched pass into a persistent device table; ``launch`` copies the step's row on device.
        """
        sig = self.signature(prompt=prompt, img_shapes=img_shapes, combined_seq=combined_seq, num_steps=len(timesteps))
        assert not self.needs_reset(sig), "reset() all branches before preparing a new signature"
        sp = self._sp_axis
        spatial_rope, prompt_rope = pos_embed.forward(img_shapes, [prompt.shape[1]], "cpu")
        host = {
            "prompt": self._host(prompt),
            "spatial_rope": (
                self._host(spatial_rope.real.repeat_interleave(2, dim=-1), mesh_axes=[sp, None]),
                self._host(spatial_rope.imag.repeat_interleave(2, dim=-1), mesh_axes=[sp, None]),
            ),
            "prompt_rope": (
                self._host(prompt_rope.real.repeat_interleave(2, dim=-1)),
                self._host(prompt_rope.imag.repeat_interleave(2, dim=-1)),
            ),
        }
        if self._inputs is None:
            self._sig = sig
            per_step = {
                "spatial": self.convert_hidden_states(torch.zeros(1, combined_seq, in_channels)),
            }
            width = self._tt.modulation_width
            self._mod_table = self._zeros([1, len(timesteps), width])
            self._mod_key = None
            self._inputs = {
                **_tree_to_device({**host, **per_step}, self._device),
                "modulation": self._zeros([1, 1, width]),
                "spatial_sequence_length": combined_seq,
                "prompt_sequence_length": prompt.shape[1],
            }
        else:
            _tree_write(host, self._inputs)

        key = tuple(timesteps.tolist())
        if key != self._mod_key:
            table = self._tt.compute_modulation(
                tensor.from_torch(timesteps.reshape(-1, 1), dtype=ttnn.float32, device=self._device)
            )
            ttnn.copy(table, self._mod_table)
            ttnn.deallocate(table)
            self._mod_key = key

    def _zeros(self, shape: list[int]) -> ttnn.Tensor:
        return ttnn.zeros(shape, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self._device)

    def _host(self, x: torch.Tensor, *, mesh_axes: list | None = None, dtype: ttnn.DataType = ttnn.bfloat16):
        return tensor.from_torch(
            x.to(torch.float32), device=self._device, mesh_axes=mesh_axes, dtype=dtype, on_host=True
        )

    def launch(self, *, hidden_states: torch.Tensor, step: int) -> None:
        """Write step ``step``'s latents + modulation row and enqueue the forward (non-blocking when traced)."""
        assert self._inputs is not None, "prepare() first"
        if self._use_2cq:
            # The previous forward must be done reading the latents before CQ1 overwrites them,
            # and CQ0 must not start this forward before the write lands.
            if self._op_event is not None:
                ttnn.wait_for_event(1, self._op_event)
            ttnn.copy_host_to_device_tensor(self.convert_hidden_states(hidden_states), self._inputs["spatial"], cq_id=1)
            ttnn.wait_for_event(0, ttnn.record_event(self._device, 1))
            # The previous read-back must be done before this forward overwrites the output.
            if self._read_event is not None:
                ttnn.wait_for_event(0, self._read_event)
        else:
            ttnn.copy_host_to_device_tensor(self.convert_hidden_states(hidden_states), self._inputs["spatial"])
        # Device-to-device; the sliced temporary is consumed before the trace replays.
        ttnn.copy(self._mod_table[:, step : step + 1, :], self._inputs["modulation"])

        if not self._trace:
            self._out = self._tt.forward(**self._inputs)
            if self._use_2cq:
                self._op_event = ttnn.record_event(self._device, 0)
            return
        if self._tracer is None:
            # The forward never writes its inputs in place, so the compile run can use them directly
            # instead of clones (the tile-padded modulation row alone is tens of MB).
            self._tracer = Tracer(self._tt.forward, device=self._device, prep_run=True, clone_prep_inputs=False)
        self._out = self._tracer(**self._inputs, traced=True, tracer_cq_id=0, tracer_blocking_execution=False)
        if self._use_2cq:
            self._op_event = ttnn.record_event(self._device, 0)

    def convert_hidden_states(self, hidden_states: torch.Tensor) -> ttnn.Tensor:
        return self._host(hidden_states, mesh_axes=[None, self._sp_axis, None])

    def start_collect(self, num_tokens: int) -> None:
        """Start a non-blocking read of the first ``num_tokens`` output tokens (finish with ``collect``).

        The output is sharded over SP and replicated over TP, and only the noise tokens (the leading
        part of the combined sequence) are needed, so only the TP-rank-0 shards covering them are
        read instead of gathering the whole mesh.
        """
        shard_len = self._out.shape[1]  # per-device (SP shard) length
        num_shards = -(-num_tokens // shard_len)
        by_sp = {}
        coords = self._out.tensor_topology().mesh_coords()
        if self._use_2cq:
            ttnn.wait_for_event(1, self._op_event)  # read on CQ1 only after the forward is done
        cq_id = 1 if self._use_2cq else None
        for coord, dev_tensor in zip(coords, ttnn.get_device_tensors(self._out), strict=True):
            sp_idx = coord[self._sp_axis]
            if sp_idx < num_shards and all(c == 0 for i, c in enumerate(coord) if i != self._sp_axis):
                by_sp[sp_idx] = dev_tensor.cpu(blocking=False, cq_id=cq_id)
        if self._use_2cq:
            self._read_event = ttnn.record_event(self._device, 1)
        self._pending = ([by_sp[i] for i in range(num_shards)], num_tokens)

    def collect(self) -> torch.Tensor:
        shards, num_tokens = self._pending
        self._pending = None
        if self._use_2cq:
            ttnn.event_synchronize(self._read_event)
        else:
            ttnn.synchronize_device(self._device)
        return torch.cat([ttnn.to_torch(t) for t in shards], dim=1)[:, :num_tokens]


def _tree_to_device(tree: dict, device: ttnn.MeshDevice) -> dict:
    return {k: tuple(t.to(device) for t in v) if isinstance(v, tuple) else v.to(device) for k, v in tree.items()}


def _tree_write(host: dict, dev: dict) -> None:
    for k, v in host.items():
        for h, d in zip(v, dev[k], strict=True) if isinstance(v, tuple) else ((v, dev[k]),):
            ttnn.copy_host_to_device_tensor(h, d)


class QwenImageEditPipeline:
    """Qwen-Image-Edit with the denoise running on the WH Galaxy.

    Two denoise layouts:

    * ``cfg_parallel=False``: TP=8 x SP=4 on all 32 chips; cond and uncond forwards run one after the
      other, driven by the reference ``diffusers.QwenImageEditPipeline`` loop.
    * ``cfg_parallel=True``: two 4x4 submeshes (TP=4 x SP=4 each); the cond and uncond forwards run
      concurrently, driven by this class's own denoise loop (same math as the reference loop).

    VL image+text encode stays on host; VAE encode/decode runs on device.
    """

    @classmethod
    def create_pipeline(
        cls,
        *,
        mesh_device: ttnn.MeshDevice,
        checkpoint_name: str = _DEFAULT_CHECKPOINT,
        trace: bool = True,
        cfg_parallel: bool = True,
        device_vae: bool = True,
        device_vae_encode: bool = True,
        prompt_bucket: int | None = _DEFAULT_PROMPT_BUCKET,
        use_2cq: bool = False,
    ) -> QwenImageEditPipeline:
        config = QwenImageEditPipelineConfig.default(
            mesh_shape=mesh_device.shape,
            checkpoint_name=checkpoint_name,
            cfg_parallel=cfg_parallel,
        )
        return cls(
            device=mesh_device,
            config=config,
            trace=trace,
            device_vae=device_vae,
            device_vae_encode=device_vae_encode,
            prompt_bucket=prompt_bucket,
            use_2cq=use_2cq,
        )

    def __init__(
        self,
        *,
        device: ttnn.MeshDevice,
        config: QwenImageEditPipelineConfig,
        trace: bool = True,
        device_vae: bool = True,
        device_vae_encode: bool = True,
        prompt_bucket: int | None = _DEFAULT_PROMPT_BUCKET,
        use_2cq: bool = False,
    ) -> None:
        self._mesh_device = device
        self.use_2cq = use_2cq
        self.prompt_bucket = prompt_bucket
        self._config = config
        self._parallel_config = config.dit_parallel_config
        self._cfg_parallel = self._parallel_config.cfg_parallel.factor == 2

        sp = self._parallel_config.sequence_parallel
        tp = self._parallel_config.tensor_parallel
        logger.info(f"Qwen-Image-Edit parallel config: {self._parallel_config}")

        # cfg=1 uses the full mesh; cfg=2 splits it into two submeshes (cond, uncond).
        self._submeshes = create_submeshes(device, self._parallel_config) if self._cfg_parallel else (device,)
        logger.info(
            f"Mesh shape: {tuple(device.shape)} -> {len(self._submeshes)} x {tuple(self._submeshes[0].shape)} "
            f"(SP={sp.factor}@axis{sp.mesh_axis}, TP={tp.factor}@axis{tp.mesh_axis})"
        )

        logger.info("loading reference QwenImageEditPipeline (host: VL encode, scheduler)...")
        self._hf = reference.QwenImageEditPipeline.from_pretrained(config.checkpoint_name, torch_dtype=torch.bfloat16)
        _VisionFeatureCache(self._hf.text_encoder.model)
        hf_transformer = self._hf.transformer
        cfg = hf_transformer.config
        self._transformer_config = cfg
        self._pos_embed = hf_transformer.pos_embed

        self._ccl_managers = [
            CCLManager(mesh_device=d, num_links=config.num_links, topology=config.topology) for d in self._submeshes
        ]
        padding_config = (
            PaddingConfig.from_tensor_parallel_factor(cfg.num_attention_heads, cfg.attention_head_dim, tp.factor)
            if cfg.num_attention_heads % tp.factor != 0
            else None
        )

        logger.info("building TT transformer(s) + loading edit weights...")
        tt_models = []
        for submesh, ccl_manager in zip(self._submeshes, self._ccl_managers, strict=True):
            tt_model = QwenImageTransformer(
                patch_size=cfg.patch_size,
                in_channels=cfg.in_channels,
                num_layers=cfg.num_layers,
                attention_head_dim=cfg.attention_head_dim,
                num_attention_heads=cfg.num_attention_heads,
                joint_attention_dim=cfg.joint_attention_dim,
                out_channels=cfg.out_channels,
                device=submesh,
                ccl_manager=ccl_manager,
                parallel_config=self._parallel_config,
                padding_config=padding_config,
            )
            cache.load_model(
                tt_model=tt_model,
                get_torch_state_dict=hf_transformer.state_dict,
                model_name=_model_name_for_cache(config.checkpoint_name),
                subfolder="transformer",
                parallel_config=self._parallel_config,
                mesh_shape=tuple(submesh.shape),
                mesh_device=submesh,
            )
            tt_models.append(tt_model)

        # Branch 0 = cond, branch 1 = uncond; one per submesh with CFG-parallel, else both share the
        # full-mesh model.
        models = tt_models if self._cfg_parallel else tt_models * 2
        devices = self._submeshes if self._cfg_parallel else (device, device)
        self._branches = [
            _DenoiseBranch(
                tt_model=m, device=d, sp_axis=sp.mesh_axis, tp_axis=tp.mesh_axis, trace=trace, use_2cq=use_2cq
            )
            for m, d in zip(models, devices, strict=True)
        ]
        self.forward_times: list[float] = []  # one entry per denoise step
        # The host transformer is only needed for its config, RoPE helper and state dict (loaded above).
        self._hf.transformer = None
        del hf_transformer

        # Swap the device-backed VAE (encode + decode) into the host pipeline. With CFG-parallel it
        # lives on the first submesh. Height on the TP axis, width on the SP axis (base qwenimage
        # convention).
        if device_vae:
            logger.info(f"building device VAE (encode={device_vae_encode}, decode=True)...")
            self._hf.vae = DeviceVAE(
                checkpoint_name=config.checkpoint_name,
                mesh_device=self._submeshes[0],
                # Own CCL manager: the traced transformer bakes in its manager's semaphores and
                # persistent buffers, which untraced VAE CCLs must not touch between replays.
                ccl_manager=CCLManager(
                    mesh_device=self._submeshes[0], num_links=config.num_links, topology=config.topology
                ),
                height_axis=tp.mesh_axis,
                width_axis=sp.mesh_axis,
                device_encode=device_vae_encode,
            )
        logger.info("Qwen-Image-Edit pipeline ready.")

    @property
    def last_step_times(self) -> list[float]:
        """Per-step transformer times (seconds, cond + uncond forwards) from the most recent ``__call__``."""
        return self.forward_times

    def __call__(
        self,
        *,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = " ",
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

        self.forward_times.clear()
        generator = torch.Generator().manual_seed(seed)

        t_start = time.time()
        images, timings = self._generate(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            side=side,
            generator=generator,
        )
        wall = time.time() - t_start

        steps = self.forward_times
        warm = steps[1:] if len(steps) > 1 else steps
        per_step_ms = (sum(warm) / len(warm) * 1000.0) if warm else float("nan")
        breakdown = " | ".join(f"{k} {v:.1f} s" for k, v in timings.items())
        logger.info(
            f"[qwen-image-edit] {num_inference_steps} steps (cfg_parallel={self._cfg_parallel}) | "
            f"warm per-step {per_step_ms:.1f} ms | denoise {sum(steps):.1f} s | wall {wall:.1f} s | {breakdown}"
        )
        return images

    @torch.no_grad()
    def _generate(
        self,
        *,
        image: Image.Image,
        prompt: str,
        negative_prompt: str,
        num_inference_steps: int,
        true_cfg_scale: float,
        side: int,
        generator: torch.Generator,
    ) -> tuple[list[Image.Image], dict[str, float]]:
        """Reference ``QwenImageEditPipeline.__call__`` with the transformer forwards on device.

        Mirrors diffusers' edit loop step for step (bf16 host latents, the same timestep rounding,
        true-CFG with norm rescale, FlowMatch Euler step). Both branches are enqueued before either
        result is read, so with CFG-parallel they run concurrently on their submeshes.
        """
        hf = self._hf
        timings: dict[str, float] = {}

        t = time.time()
        calc_w, calc_h, _ = calculate_dimensions(1024 * 1024, image.size[0] / image.size[1])
        multiple_of = hf.vae_scale_factor * 2
        height, width = side // multiple_of * multiple_of, side // multiple_of * multiple_of
        prompt_image = hf.image_processor.resize(image, calc_h, calc_w)
        image_tensor = hf.image_processor.preprocess(prompt_image, calc_h, calc_w).unsqueeze(2)

        prompt_embeds, _ = hf.encode_prompt(image=prompt_image, prompt=prompt, device="cpu")
        do_true_cfg = true_cfg_scale > 1 and negative_prompt is not None
        if do_true_cfg:
            negative_prompt_embeds, _ = hf.encode_prompt(image=prompt_image, prompt=negative_prompt, device="cpu")
        if self.prompt_bucket:
            prompt_embeds = _pad_to_bucket(prompt_embeds, self.prompt_bucket)
            if do_true_cfg:
                negative_prompt_embeds = _pad_to_bucket(negative_prompt_embeds, self.prompt_bucket)
        timings["vl_encode"] = time.time() - t

        t = time.time()
        num_channels_latents = self._transformer_config.in_channels // 4
        latents, image_latents = hf.prepare_latents(
            image_tensor, 1, num_channels_latents, height, width, prompt_embeds.dtype, "cpu", generator, None
        )
        timings["vae_encode"] = time.time() - t
        img_shapes = [
            [
                (1, height // hf.vae_scale_factor // 2, width // hf.vae_scale_factor // 2),
                (1, calc_h // hf.vae_scale_factor // 2, calc_w // hf.vae_scale_factor // 2),
            ]
        ]

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu = calculate_shift(
            latents.shape[1],
            hf.scheduler.config.get("base_image_seq_len", 256),
            hf.scheduler.config.get("max_image_seq_len", 4096),
            hf.scheduler.config.get("base_shift", 0.5),
            hf.scheduler.config.get("max_shift", 1.15),
        )
        hf.scheduler.set_timesteps(sigmas=sigmas, device="cpu", mu=mu)
        hf.scheduler.set_begin_index(0)
        timesteps = hf.scheduler.timesteps

        combined_seq = latents.shape[1] + image_latents.shape[1]
        sp_factor = self._parallel_config.sequence_parallel.factor
        if combined_seq % sp_factor != 0:
            msg = f"Combined token sequence ({combined_seq}) is not divisible by the SP factor ({sp_factor})"
            raise ValueError(msg)

        branches = self._branches if do_true_cfg else self._branches[:1]
        contexts = [prompt_embeds, negative_prompt_embeds] if do_true_cfg else [prompt_embeds]
        # All branches' trace inputs must be (re)allocated before any trace is captured, so a shape
        # change on either branch resets both (see _DenoiseBranch).
        # Same rounding as the reference loop: timestep cast to the latents dtype, then / 1000; the TT
        # time embedding takes the raw (x1000) scale.
        tt_timesteps = (timesteps.to(latents.dtype) / 1000).to(torch.float32) * 1000.0

        sigs = [
            _DenoiseBranch.signature(
                prompt=ctx, img_shapes=img_shapes, combined_seq=combined_seq, num_steps=len(tt_timesteps)
            )
            for ctx in contexts
        ]
        if any(b.needs_reset(sig) for b, sig in zip(branches, sigs, strict=False)):
            for branch in self._branches:
                branch.reset()
        for branch, ctx in zip(branches, contexts, strict=True):
            branch.prepare(
                pos_embed=self._pos_embed,
                prompt=ctx,
                img_shapes=img_shapes,
                combined_seq=combined_seq,
                in_channels=self._transformer_config.in_channels,
                timesteps=tt_timesteps,
            )

        t = time.time()
        n_lat = latents.shape[1]
        for i, step_t in enumerate(timesteps):
            latent_model_input = torch.cat([latents, image_latents], dim=1)

            t0 = time.time()
            for branch in branches:
                branch.launch(hidden_states=latent_model_input, step=i)
            for branch in branches:
                branch.start_collect(n_lat)
            preds = [branch.collect().to(latents.dtype) for branch in branches]
            self.forward_times.append(time.time() - t0)

            noise_pred = preds[0]
            if do_true_cfg:
                neg_noise_pred = preds[1]
                comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
                cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                noise_pred = comb_pred * (cond_norm / noise_norm)

            latents = hf.scheduler.step(noise_pred, step_t, latents, return_dict=False)[0]
        timings["denoise_loop"] = time.time() - t

        t = time.time()
        latents = hf._unpack_latents(latents, height, width, hf.vae_scale_factor)  # noqa: SLF001
        latents = latents.to(hf.vae.dtype)
        z_dim = hf.vae.config.z_dim
        latents_mean = torch.tensor(hf.vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(latents.dtype)
        latents_std = 1.0 / torch.tensor(hf.vae.config.latents_std).view(1, z_dim, 1, 1, 1).to(latents.dtype)
        latents = latents / latents_std + latents_mean
        decoded = hf.vae.decode(latents, return_dict=False)[0][:, :, 0]
        images = hf.image_processor.postprocess(decoded, output_type="pil")
        timings["vae_decode"] = time.time() - t
        return list(images), timings


def _pad_to_bucket(embeds: torch.Tensor, bucket: int) -> torch.Tensor:
    """Zero-pad [batch, seq, dim] prompt embeddings along seq up to the next multiple of ``bucket``."""
    pad = -embeds.shape[1] % bucket
    return torch.nn.functional.pad(embeds, (0, 0, 0, pad)) if pad else embeds


def _model_name_for_cache(checkpoint_name: str) -> str:
    """Stable, filesystem-friendly model name for the weight cache."""
    return checkpoint_name.split("/")[-1].lower()
