# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Wan2.2 I2V pipeline with LoRA adapters fused into the base weights.

Each expert (high/low noise) takes an ordered LoRA stack; stacks are fused
on CPU before TT conversion so inference has no LoRA-specific runtime cost.
See ``experimental/models/Wan2_2_LoRA.md`` for the adapter-key formats
detected by ``fuse_lora_state_dict`` and the supported namespaces.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import torch
import torchvision.transforms.functional as TF
from diffusers.schedulers import UniPCMultistepScheduler
from loguru import logger
from PIL import Image
from safetensors.torch import load_file

import ttnn
from models.tt_dit.experimental.utils.lightx2v_loader import wan_lightx2v_to_diffusers_key
from models.tt_dit.pipelines.events import PipelineEventCallback, SectionEnd, SectionStart, null_callback
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipeline, WanPipelineConfig
from models.tt_dit.pipelines.wan.pipeline_wan_i2v import ImagePrompt, WanPipelineI2V
from models.tt_dit.solvers import UniPCSolver, UniPCVariant
from models.tt_dit.utils import cache
from models.tt_dit.utils.conv3d import aligned_channels, conv_pad_height, conv_pad_in_channels, conv_pad_width
from models.tt_dit.utils.tensor import bf16_tensor_2dshard, fast_device_to_host


@dataclass(frozen=True)
class LoRASpec:
    """A single LoRA adapter file and its blend strength."""

    path: str
    scale: float = 1.0


LoRAArg = LoRASpec | str | Sequence[LoRASpec | str] | None


@dataclass(frozen=True)
class _FusionStats:
    applied_pairs: int = 0
    applied_direct: int = 0
    skipped_unknown: int = 0
    skipped_unmapped: int = 0
    skipped_shape_mismatch: int = 0

    @property
    def applied(self) -> int:
        return self.applied_pairs + self.applied_direct


def _normalize_lora_arg(arg: LoRAArg) -> list[LoRASpec]:
    """Coerce None / path / LoRASpec / sequence-of-either into LoRASpecs."""
    if arg is None:
        return []
    if isinstance(arg, LoRASpec):
        return [arg]
    if isinstance(arg, str):
        return [LoRASpec(arg)]
    if not isinstance(arg, Sequence):
        raise TypeError(f"Expected LoRASpec, str, sequence, or None; got {type(arg).__name__}")
    out: list[LoRASpec] = []
    for item in arg:
        if isinstance(item, LoRASpec):
            out.append(item)
        elif isinstance(item, str):
            out.append(LoRASpec(item))
        else:
            raise TypeError(f"Expected LoRASpec or str in LoRA list, got {type(item).__name__}")
    return out


_STRIP_PREFIXES = ("diffusion_model.", "transformer.", "unet.", "model.")

# PEFT may insert an adapter name before ``.weight``, e.g. ``lora_A.default.weight``.
_LOW_RANK_RE = re.compile(r"^(?P<base>.*)\.lora_(?P<slot>A|B|down|up)(?:\.[^.]+)?\.weight$")
_SLOT_MAP = {"A": "A", "down": "A", "B": "B", "up": "B"}


def _strip_known_prefixes(key: str) -> str:
    for prefix in _STRIP_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _kohya_to_lightx2v(key: str) -> str:
    """Convert kohya/A1111-style keys to lightx2v-style keys.

    ``lora_unet_blocks_0_cross_attn_k.lora_down.weight``
    → ``blocks.0.cross_attn.k.lora_down.weight``
    """
    if not key.startswith("lora_unet_"):
        return key
    parts = key.split(".", 1)
    module_path = parts[0][len("lora_unet_") :]
    suffix = f".{parts[1]}" if len(parts) > 1 else ""

    m = re.match(r"blocks_(\d+)_(cross_attn|self_attn)_([a-z]+)", module_path)
    if m:
        return f"blocks.{m.group(1)}.{m.group(2)}.{m.group(3)}{suffix}"

    m = re.match(r"blocks_(\d+)_(ffn)_(\d+)", module_path)
    if m:
        return f"blocks.{m.group(1)}.{m.group(2)}.{m.group(3)}{suffix}"

    logger.warning(f"Unrecognized kohya key structure: {key}")
    return key


def _diffusers_target(lightx2v_base_path: str, suffix: str) -> str:
    """Map ``<lightx2v_base>.weight``/``.bias`` to the diffusers parameter key."""
    weight_key = wan_lightx2v_to_diffusers_key(f"{lightx2v_base_path}.weight")
    if suffix == ".weight":
        return weight_key
    return weight_key[: -len(".weight")] + suffix


def _is_lora_key(raw_key: str) -> bool:
    key = _kohya_to_lightx2v(_strip_known_prefixes(raw_key))
    return bool(_LOW_RANK_RE.match(key)) or key.endswith((".diff", ".diff_b"))


def _has_lora_keys(state_dict: dict[str, torch.Tensor]) -> bool:
    """Return True iff the safetensors file contains at least one LoRA-style key."""
    return any(_is_lora_key(k) for k in state_dict)


def fuse_lora_state_dict(
    base_state_dict: dict[str, torch.Tensor],
    lora_state_dict: dict[str, torch.Tensor],
    *,
    scale: float = 1.0,
    return_stats: bool = False,
) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], _FusionStats]:
    """Return a fused state dict, optionally with stats describing what applied.

    The base dict is not mutated; entries that were touched are fresh tensors
    so the caller can chain fusions across a stack safely. See
    ``experimental/models/Wan2_2_LoRA.md`` for the supported adapter key
    formats. Raises ``KeyError`` when a low-rank pair is missing one half.
    """
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    direct_deltas: list[tuple[str, str, torch.Tensor]] = []  # (base_path, suffix, tensor)
    alphas: dict[str, float] = {}
    skipped_unknown: list[str] = []

    for raw_key, tensor in lora_state_dict.items():
        key = _kohya_to_lightx2v(_strip_known_prefixes(raw_key))
        m = _LOW_RANK_RE.match(key)
        if m:
            pairs.setdefault(m.group("base"), {})[_SLOT_MAP[m.group("slot")]] = tensor
            continue
        if key.endswith(".diff_b"):
            direct_deltas.append((key[: -len(".diff_b")], ".bias", tensor))
        elif key.endswith(".diff"):
            direct_deltas.append((key[: -len(".diff")], ".weight", tensor))
        elif key.endswith(".alpha"):
            alphas[key[: -len(".alpha")]] = tensor.item()
        else:
            skipped_unknown.append(raw_key)

    if skipped_unknown:
        logger.warning(
            f"LoRA fusion: {len(skipped_unknown)} unrecognized keys ignored. Examples: {skipped_unknown[:5]}"
        )

    fused = dict(base_state_dict)
    applied_pairs = 0
    skipped_unmapped: list[str] = []

    for base_path, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            raise KeyError(f"LoRA pair incomplete for '{base_path}': have {list(ab)}")
        diffusers_key = _diffusers_target(base_path, ".weight")
        if diffusers_key not in fused:
            skipped_unmapped.append(diffusers_key)
            continue
        base_weight = fused[diffusers_key]
        rank = ab["A"].shape[0]
        alpha = alphas.get(base_path, float(rank))
        effective_scale = scale * (alpha / rank)
        delta = effective_scale * (ab["B"].to(torch.float32) @ ab["A"].to(torch.float32))
        # ``+`` always allocates a new tensor; avoids the .to(fp32)+.add_()
        # alias trap when base_weight is already fp32 (would mutate the
        # caller's dict on a subsequent stacked-fusion pass).
        fused[diffusers_key] = (base_weight.float() + delta).to(base_weight.dtype)
        applied_pairs += 1

    applied_direct = 0
    skipped_shape_mismatch = 0
    for base_path, suffix, tensor in direct_deltas:
        diffusers_key = _diffusers_target(base_path, suffix)
        if diffusers_key not in fused:
            skipped_unmapped.append(diffusers_key)
            continue
        base = fused[diffusers_key]
        if base.shape != tensor.shape:
            logger.warning(
                f"LoRA direct delta shape mismatch for '{diffusers_key}': "
                f"base {tuple(base.shape)} vs delta {tuple(tensor.shape)}; skipping."
            )
            skipped_shape_mismatch += 1
            continue
        fused[diffusers_key] = (base.float() + scale * tensor.to(torch.float32)).to(base.dtype)
        applied_direct += 1

    if skipped_unmapped:
        logger.warning(
            f"LoRA fusion: {len(skipped_unmapped)} adapter targets not present in base; skipped. "
            f"Examples: {skipped_unmapped[:5]}"
        )

    logger.info(f"Fused {applied_pairs} low-rank pairs and {applied_direct} direct deltas (scale={scale})")
    stats = _FusionStats(
        applied_pairs=applied_pairs,
        applied_direct=applied_direct,
        skipped_unknown=len(skipped_unknown),
        skipped_unmapped=len(skipped_unmapped),
        skipped_shape_mismatch=skipped_shape_mismatch,
    )
    return (fused, stats) if return_stats else fused


def verify_fusion_changed_weights(
    base_sd: dict[str, torch.Tensor],
    fused_sd: dict[str, torch.Tensor],
    *,
    min_changed: int = 3,
    label: str,
) -> None:
    """Sanity check that at least ``min_changed`` weights actually differ post-fusion.

    Catches the canonical "LoRA silently failed to apply" failure mode where
    a key-mapping bug leaves every fused target absent from the base. Logs a
    sample of changed tensors with their L2 diff for visibility.

    Raises ``RuntimeError`` if no weights changed at all, or if fewer than
    ``min_changed`` weights changed (likely a partial load).
    """
    changed: list[tuple[str, float]] = []
    max_diff = 0.0
    for k, base in base_sd.items():
        fused = fused_sd.get(k)
        if fused is None or fused.shape != base.shape:
            continue
        if fused.data_ptr() == base.data_ptr():
            continue
        diff = (fused.to(torch.float32) - base.to(torch.float32)).norm().item()
        if diff > 0.0:
            changed.append((k, diff))
            max_diff = max(max_diff, diff)

    if max_diff == 0.0:
        raise RuntimeError(
            f"LoRA silently failed to apply -- weights are unchanged for '{label}'. "
            f"Verified {len(base_sd)} keys against fused dict; max L2 diff is 0.0."
        )

    sample = changed[:5]
    logger.info(
        f"LoRA fusion verified for '{label}': {len(changed)} tensors changed, "
        f"max L2 diff={max_diff:.4f}. Sample diffs:"
    )
    for k, d in sample:
        logger.info(f"  {k}: L2={d:.4f}")

    if len(changed) < min_changed:
        raise RuntimeError(
            f"LoRA fusion changed only {len(changed)} weights for '{label}' "
            f"(require >= {min_changed}). Likely indicates partial LoRA load."
        )


def _lora_stack_cache_namespace(specs_by_expert: dict[int, list[LoRASpec]]) -> str:
    """Hash ordered ``(resolved_path, scale)`` per expert so distinct stacks cache separately."""
    h = hashlib.sha1()
    for idx in sorted(specs_by_expert.keys()):
        h.update(f"expert_{idx}\x00".encode())
        for spec in specs_by_expert[idx]:
            h.update(str(Path(spec.path).resolve()).encode())
            h.update(b"\x00")
            h.update(f"{spec.scale:.6f}".encode())
            h.update(b"\x00")
    return f"Wan2.2-I2V-LoRA-{h.hexdigest()[:12]}"


# encoder_t_chunk_size per mesh. None = full-T single pass, N = chunked with feat_cache;
# the two are numerically identical and full-T is faster.
_ENCODER_T_CHUNK_BY_MESH = {
    (4, 8): 16,
    (4, 32): None,
}

# Truncated VAE encode: encode only the first 33 pixel frames (-> 9 latent frames) and
# replicate the last latent to fill the remaining slots. Every frame past the conditioned
# one is zero and the encoder is causal in T, so the dropped latents repeat the last
# computed one.
_I2V_ENCODE_FRAMES = 33

# Selects prepare_latents_ (truncated encode, device-side channel pad, swept blockings)
# over prepare_latents. Off until the truncation is validated against the full encode.
_TRUNCATED_ENCODE_ENV = "WAN22_I2V_TRUNCATED_ENCODE"

# Breaks the prepare_latents sections down per phase. Diagnostic only: it syncs at every
# phase boundary, which serializes host work that normally overlaps device work, so the
# reported total runs higher than an uninstrumented run.
_ENCODE_TIMERS_ENV = "WAN22_I2V_ENCODE_TIMERS"

# Frame count each conv3d sees, per mesh, used only to pick blockings at construction.
# 4x8 runs chunked at 16 so every conv3d call sees 16 frames; 4x32 runs the 33 frames in one
# full-T pass. Both hit the "720p image encoder" entries in conv3d._BLOCKINGS; a mesh absent
# here gets the fallback table. Only valid while _I2V_ENCODE_FRAMES truncation is on.
_ENCODER_BUILD_T_BY_MESH = {
    (4, 8): 16,
    (4, 32): _I2V_ENCODE_FRAMES,
}


def _truncated_encode_frames(max_cond_pos: int, num_frames: int) -> int:
    """Pixel frames to encode: the truncation point, extended to cover the last conditioned frame.

    Rounded up to the next 4n+1 that temporal downsampling expects, so the latent count is
    exact and the encoder keeps seeing the shapes its blockings were picked for.
    """
    frames = max(_I2V_ENCODE_FRAMES, max_cond_pos + 1)
    frames = ((frames - 2) // 4 + 1) * 4 + 1
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

    Negative positions index from the end, so -1 is the last frame. Both prepare_latents
    variants test ``i in cond_by_pos`` while walking forward from frame 0, so an
    unresolved negative position would silently drop the conditioning frame instead of
    failing, and would also skew the truncated encode's frame count.
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


def _resolve_checkpoint(repo_id: str) -> str:
    """Prefer a local checkpoint directory over a hub repo id.

    All ranks share one HF cache on NFS; resolving by repo id makes each rank hit the hub
    and write cache metadata concurrently, and a rank that loses that race fails with a
    spurious missing-weights error. A directory path skips hub resolution.

    WAN22_I2V_CHECKPOINT_DIR must share the repo id's basename: the tt_dit cache namespace
    is os.path.basename(checkpoint_name), so a different basename orphans the compiled
    cache. Falls back to the repo id when unset, missing, or mismatched.
    """
    local = os.environ.get("WAN22_I2V_CHECKPOINT_DIR")
    if not local:
        return repo_id
    if not os.path.isdir(local):
        logger.warning(f"WAN22_I2V_CHECKPOINT_DIR={local!r} is not a directory; falling back to {repo_id}")
        return repo_id
    if os.path.basename(local.rstrip("/")) != os.path.basename(repo_id):
        logger.warning(
            f"WAN22_I2V_CHECKPOINT_DIR={local!r} basename does not match {repo_id!r}; "
            "using it would orphan the tt_dit cache. Falling back to the repo id."
        )
        return repo_id
    logger.info(f"Resolving {repo_id} from local checkpoint dir {local} (no hub lookup)")
    return local


class WanPipelineI2VLora(WanPipelineI2V):
    """Wan2.2 I2V with LoRA stacks fused into the base PyTorch weights."""

    BASE_DIFFUSERS_REPO = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

    # Routes prepare_latents to prepare_latents_ (truncated encode, device-side channel pad,
    # resolution-aware blockings). The encoder's blockings key off the same flag so the two
    # stay consistent. Subclasses and tests can set the attribute directly.
    USE_TRUNCATED_ENCODE: bool = os.environ.get(_TRUNCATED_ENCODE_ENV, "0").lower() in ("1", "true")

    # Emits the per-phase prepare_latents sections. See _ENCODE_TIMERS_ENV for the caveat.
    ENCODE_TIMERS: bool = os.environ.get(_ENCODE_TIMERS_ENV, "0").lower() in ("1", "true")

    # Set for the duration of __call__ so prepare_latents can report its VAE encode.
    _on_event: PipelineEventCallback = null_callback

    # Phases reported under prepare_latents when ENCODE_TIMERS is on, in execution order.
    # The test's summary table reads these names; vae_encode is always emitted.
    ENCODE_PHASES = (
        "pl_noise",
        "pl_preprocess",
        "pl_upload",
        "pl_assemble",
        "vae_encode",
        "pl_readback",
        "pl_post",
    )

    @contextmanager
    def _encode_phase(self, name: str):
        """Time one prepare_latents phase into the pipeline's profiler.

        Device work is enqueued asynchronously, so without a sync a phase boundary marks
        where the host stopped issuing, not where the device finished, and the encode
        would absorb the upload and assembly time. Syncing makes the split honest at the
        cost of the overlap it otherwise hides, so this stays behind ENCODE_TIMERS and
        the default path emits no extra sections and adds no syncs.
        """
        if not self.ENCODE_TIMERS:
            yield
            return
        self._on_event(SectionStart(name))
        try:
            yield
        finally:
            ttnn.synchronize_device(self.mesh_device)
            self._on_event(SectionEnd(name))

    def __init__(
        self,
        *,
        device: ttnn.MeshDevice,
        config: WanPipelineConfig,
        lora_high: LoRAArg = None,
        lora_low: LoRAArg = None,
        flow_shift: float | None = None,
        run_warmup: bool = True,
    ) -> None:
        high_specs = _normalize_lora_arg(lora_high)
        low_specs = _normalize_lora_arg(lora_low)

        if not high_specs and not low_specs:
            raise ValueError(
                "WanPipelineI2VLora requires at least one LoRA. "
                "Pass lora_high and/or lora_low as a path, LoRASpec, or list."
            )

        for label, specs in [("lora_high", high_specs), ("lora_low", low_specs)]:
            for spec in specs:
                if not Path(spec.path).is_file():
                    raise FileNotFoundError(f"{label}: file does not exist: {spec.path}")

        self._lora_specs: dict[int, list[LoRASpec]] = {0: high_specs, 1: low_specs}
        self._cache_namespace = _lora_stack_cache_namespace(self._lora_specs)
        # Cleared after TT cache handoff (see _prepare_transformer) to free CPU memory.
        self._fused_state_dicts: dict[int, dict[str, torch.Tensor] | None] = {0: None, 1: None}

        # Warmup deferred: it runs a real __call__, and should see the schedule set below
        # rather than the base one it would otherwise compile against.
        super().__init__(device=device, config=config, run_warmup=False)

        # Distilled few-step LoRAs need a different flow shift than the base schedule,
        # which WanPipelineConfig pins at 12.0. Rebuilt here instead of plumbing it through
        # the shared config. __call__ re-derives sigmas per call so a post-init swap takes
        # effect, and this runs before the first traced call. Solver order/variant are
        # unchanged by flow_shift, so the compiled graph is identical.
        if flow_shift is not None:
            self._scheduler = UniPCMultistepScheduler.from_pretrained(
                self.checkpoint_name, subfolder="scheduler", flow_shift=flow_shift
            )
            self._solver = UniPCSolver(
                order=self._scheduler.config.solver_order,
                variant=UniPCVariant(self._scheduler.config.solver_type),
            )
            logger.info(f"WanPipelineI2VLora: scheduler flow_shift overridden to {flow_shift}")

        if run_warmup:
            self._warmup()

    def _preprocess_frame(self, image: Image.Image, height: int, width: int, device) -> torch.Tensor:
        """Conditioning image to a (B,C,H,W) float32 tensor, nominally in [-1,1]."""
        img = TF.to_tensor(image).sub_(0.5).div_(0.5)
        img = torch.nn.functional.interpolate(img[None], size=(height, width), mode="bicubic", antialias=True)
        return img.to(device) if device is not None else img

    def _vae_encoder_build_dims(self) -> tuple[int, int, int | None]:
        chunk = _ENCODER_BUILD_T_BY_MESH.get(tuple(self.mesh_device.shape)) if self.USE_TRUNCATED_ENCODE else None
        if chunk is None:
            logger.info(
                f"Image encoder: fallback conv3d blockings "
                f"(truncated encode {'on' if self.USE_TRUNCATED_ENCODE else f'off, set {_TRUNCATED_ENCODE_ENV}=1'})"
            )
            return super()._vae_encoder_build_dims()
        logger.info(
            f"Image encoder: truncated encode on, blockings for "
            f"{self._width}x{self._height} T={chunk} ({_I2V_ENCODE_FRAMES}-frame encode)"
        )
        return self._height, self._width, chunk

    def __call__(self, *args, on_event: PipelineEventCallback | None = None, **kwargs):
        # The base __call__ times prepare_latents as a whole but does not forward the
        # callback into it, so stash it here to let prepare_latents break out its
        # VAE encode. Same pattern WanPipelineSVI uses for its per-clip state.
        self._on_event = on_event if on_event is not None else null_callback
        try:
            return super().__call__(*args, on_event=on_event, **kwargs)
        finally:
            self._on_event = null_callback

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
        """I2V conditioning that uploads only the conditioned frames.

        The base implementation materializes the whole conditioning video on the host
        (896 MB float32 at 81x720x1280) and uploads all of it, though every frame but the
        conditioned one is zero. Here the real frames are padded and uploaded individually
        and the zero runs are built on device, then encoded in one pass.
        """
        if self.USE_TRUNCATED_ENCODE:
            return self.prepare_latents_(
                batch_size=batch_size,
                image_prompt=image_prompt,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=num_frames,
                dtype=dtype,
                device=device,
            )

        assert batch_size == 1, "Only batch size 1 is currently supported for I2V"

        image_prompt = _parse_image_prompts(image_prompt, num_frames)

        # Skip WanPipelineI2V.prepare_latents (the path being replaced) and take the
        # plain noise-latent allocation from the grandparent.
        latents, _ = WanPipeline.prepare_latents(
            self,
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

        # ---- host: preprocess + pad ONLY the conditioned frames --------------
        cond_by_pos: dict[int, ttnn.Tensor] = {}
        logical_h = None
        for image, frame_pos in image_prompt:
            img = self._preprocess_frame(image, height, width, device)
            # (B,C,H,W) -> (B,T=1,H,W,C)
            frame_BTHWC = img.unsqueeze(2).permute(0, 2, 3, 4, 1)
            frame_BTHWC = conv_pad_in_channels(frame_BTHWC)
            frame_BTHWC, logical_h = conv_pad_height(
                frame_BTHWC, self.vae_parallel_config.height_parallel.factor * self.vae_scale_factor_spatial
            )
            frame_BTHWC, _logical_w = conv_pad_width(
                frame_BTHWC, self.vae_parallel_config.width_parallel.factor * self.vae_scale_factor_spatial
            )
            cond_by_pos[frame_pos] = bf16_tensor_2dshard(
                frame_BTHWC,
                self.mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                shard_mapping={
                    self.vae_parallel_config.height_parallel.mesh_axis: 2,
                    self.vae_parallel_config.width_parallel.mesh_axis: 3,
                },
            )
            logical_w = _logical_w

        tt_zero_1 = None

        def _zero_run(n: int) -> ttnn.Tensor:
            nonlocal tt_zero_1
            if tt_zero_1 is None:
                tt_zero_1 = ttnn.zeros_like(next(iter(cond_by_pos.values())))
            z, built = tt_zero_1, 1
            while built * 2 <= n:
                z = ttnn.concat([z, z], dim=1)
                built *= 2
            if built < n:
                z = ttnn.concat([z, z[:, : n - built, :, :, :]], dim=1)
            return z

        segments: list[ttnn.Tensor] = []
        zero_start = None
        for i in range(num_frames):
            if i in cond_by_pos:
                if zero_start is not None:
                    segments.append(_zero_run(i - zero_start))
                    zero_start = None
                segments.append(cond_by_pos[i])
            elif zero_start is None:
                zero_start = i
        if zero_start is not None:
            segments.append(_zero_run(num_frames - zero_start))

        tt_video_BTHWC = segments[0] if len(segments) == 1 else ttnn.concat(segments, dim=1)

        chunk = _ENCODER_T_CHUNK_BY_MESH.get(tuple(self.mesh_device.shape))

        self._on_event(SectionStart("vae_encode"))
        encoded_BCTHW, new_logical_h, new_logical_w = self.tt_vae_encoder(
            tt_video_BTHWC, logical_h, logical_w=logical_w, encoder_t_chunk_size=chunk
        )
        # The encoder call only enqueues work; sync so the section covers the actual
        # device time rather than leaking it into the readback below.
        ttnn.synchronize_device(self.mesh_device)
        self._on_event(SectionEnd("vae_encode"))

        # tt_video_BTHWC may alias a conditioned frame or the zero tensor when there is
        # only one segment, so guard against a double free.
        owned = list(cond_by_pos.values()) + ([tt_zero_1] if tt_zero_1 is not None else [])
        if all(tt_video_BTHWC is not t for t in owned):
            ttnn.deallocate(tt_video_BTHWC)
        for tt in owned:
            ttnn.deallocate(tt)

        # ---- host: replicate + mask ------------------------------------------
        concat_dims = [None, None]
        concat_dims[self.vae_parallel_config.height_parallel.mesh_axis] = 3
        concat_dims[self.vae_parallel_config.width_parallel.mesh_axis] = 4
        encoded = fast_device_to_host(encoded_BCTHW, self.mesh_device, concat_dims, ccl_manager=self.vae_ccl_manager)
        ttnn.deallocate(encoded_BCTHW)
        ttnn.synchronize_device(self.mesh_device)
        # Same crop convention as WanPipelineI2V.prepare_latents.
        encoded = encoded[:, :, :, :new_logical_h, :new_logical_w].to(dtype=dtype)

        f_lat_full = latent_shape[2]
        if encoded.shape[2] != f_lat_full:
            encoded = encoded[:, :, :f_lat_full, :, :]

        msk = torch.zeros(batch_size, num_frames, h_lat, w_lat)
        for pos in cond_by_pos:
            msk[:, pos, :, :] = 1
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, h_lat, w_lat)
        msk = msk.transpose(1, 2)

        y = torch.cat([msk, encoded], dim=1)
        return latents, y

    def prepare_latents_(
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
        """prepare_latents with the three Prodia encode optimizations.

        On top of the upload-only-conditioned-frames scheme of ``prepare_latents``:
        the encode is truncated to ``_I2V_ENCODE_FRAMES`` and the tail latents are
        replicated on the host; channel padding moves to the device so each frame crosses
        PCIe with 3 channels instead of 32; and the encoder was built with real dims
        (see ``_vae_encoder_build_dims``) so its conv3d blockings come from the swept table.
        """
        assert batch_size == 1, "Only batch size 1 is currently supported for I2V"

        image_prompt = _parse_image_prompts(image_prompt, num_frames)

        with self._encode_phase("pl_noise"):
            # Skip WanPipelineI2V.prepare_latents (the path being replaced) and take the
            # plain noise-latent allocation from the grandparent.
            latents, _ = WanPipeline.prepare_latents(
                self,
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
        # Preprocess and upload are separate passes so each phase is entered once:
        # BenchmarkProfiler keys on name alone, so a phase re-entered per frame would
        # report only the last one.
        host_frames: list[tuple[int, torch.Tensor]] = []
        logical_h = None
        logical_w = None
        with self._encode_phase("pl_preprocess"):
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
                host_frames.append((frame_pos, frame_BTHWC))

        cond_by_pos: dict[int, ttnn.Tensor] = {}
        with self._encode_phase("pl_upload"):
            for frame_pos, frame_BTHWC in host_frames:
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

        encode_frames = _truncated_encode_frames(max(cond_by_pos), num_frames)

        if self.ENCODE_TIMERS:
            # bf16 on the wire, whatever the host dtype.
            xfer_mb = sum(t.numel() * 2 for _, t in host_frames) / 1e6
            logger.info(
                f"[encode] {len(host_frames)} frame(s) uploaded, {xfer_mb:.1f}MB at "
                f"{in_channels}ch (padded to {padded_channels} on device), "
                f"encode_frames={encode_frames}/{num_frames}"
            )
        host_frames.clear()

        tt_zero_1 = None

        def _zero_run(n: int) -> ttnn.Tensor:
            nonlocal tt_zero_1
            if tt_zero_1 is None:
                tt_zero_1 = ttnn.zeros_like(next(iter(cond_by_pos.values())))
            z, built = tt_zero_1, 1
            while built * 2 <= n:
                z = ttnn.concat([z, z], dim=1)
                built *= 2
            if built < n:
                z = ttnn.concat([z, z[:, : n - built, :, :, :]], dim=1)
            return z

        with self._encode_phase("pl_assemble"):
            segments: list[ttnn.Tensor] = []
            zero_start = None
            for i in range(encode_frames):
                if i in cond_by_pos:
                    if zero_start is not None:
                        segments.append(_zero_run(i - zero_start))
                        zero_start = None
                    segments.append(cond_by_pos[i])
                elif zero_start is None:
                    zero_start = i
            if zero_start is not None:
                segments.append(_zero_run(encode_frames - zero_start))

            tt_video_BTHWC = segments[0] if len(segments) == 1 else ttnn.concat(segments, dim=1)

        chunk = _ENCODER_T_CHUNK_BY_MESH.get(tuple(self.mesh_device.shape))

        pc_before = self.mesh_device.num_program_cache_entries()
        self._on_event(SectionStart("vae_encode"))
        encoded_BCTHW, new_logical_h, new_logical_w = self.tt_vae_encoder(
            tt_video_BTHWC, logical_h, logical_w=logical_w, encoder_t_chunk_size=chunk
        )
        # The encoder call only enqueues work; sync so the section covers the actual
        # device time rather than leaking it into the readback below.
        ttnn.synchronize_device(self.mesh_device)
        self._on_event(SectionEnd("vae_encode"))
        pc_after = self.mesh_device.num_program_cache_entries()
        if pc_after != pc_before:
            # Steady state is zero: a non-zero delta after warmup means the encode shape
            # changed (e.g. a late frame_pos moved encode_frames) and programs recompiled.
            logger.info(
                f"VAE encode compiled {pc_after - pc_before} new program(s) "
                f"(encode_frames={encode_frames}, forward chunk={chunk})"
            )

        # tt_video_BTHWC may alias a conditioned frame or the zero tensor when there is
        # only one segment, so guard against a double free.
        owned = list(cond_by_pos.values()) + ([tt_zero_1] if tt_zero_1 is not None else [])
        if all(tt_video_BTHWC is not t for t in owned):
            ttnn.deallocate(tt_video_BTHWC)
        for tt in owned:
            ttnn.deallocate(tt)

        # ---- device -> host --------------------------------------------------
        with self._encode_phase("pl_readback"):
            concat_dims = [None, None]
            concat_dims[self.vae_parallel_config.height_parallel.mesh_axis] = 3
            concat_dims[self.vae_parallel_config.width_parallel.mesh_axis] = 4
            encoded = fast_device_to_host(
                encoded_BCTHW, self.mesh_device, concat_dims, ccl_manager=self.vae_ccl_manager
            )
            ttnn.deallocate(encoded_BCTHW)
            ttnn.synchronize_device(self.mesh_device)

        # ---- host: replicate + mask ------------------------------------------
        with self._encode_phase("pl_post"):
            # Same crop convention as WanPipelineI2V.prepare_latents.
            encoded = encoded[:, :, :, :new_logical_h, :new_logical_w].to(dtype=dtype)

            # The truncated encode returns fewer latent frames than the transformer expects;
            # the dropped ones encode all-zero pixels, so they repeat the last computed latent.
            f_lat_full = latent_shape[2]
            n_lat = encoded.shape[2]
            if n_lat < f_lat_full:
                encoded = torch.cat(
                    [encoded, encoded[:, :, -1:, :, :].expand(-1, -1, f_lat_full - n_lat, -1, -1)], dim=2
                )
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

    def prepare_text_conditioning(self, tt_model, prompt_embeds, buffer, traced=False):
        # guidance_scale=1.0 → encoder returns None for negative embeds; combined_step
        # skips CFG so the untouched buffer is fine.
        if prompt_embeds is None:
            return buffer
        return super().prepare_text_conditioning(tt_model, prompt_embeds, buffer, traced)

    def _build_fused_state_dict(self, idx: int) -> dict[str, torch.Tensor] | None:
        specs = self._lora_specs[idx]
        state = self.transformer_states[idx]
        subfolder = state.checkpoint.subfolder
        if not specs:
            logger.info(f"No LoRA for expert idx={idx} ('{subfolder}') -- using base weights")
            return None

        base_sd = state.checkpoint.state_dict()
        fused_sd = base_sd
        for spec in specs:
            logger.info(f"Loading LoRA for '{subfolder}' from {spec.path} (scale={spec.scale})")
            lora_sd = load_file(str(spec.path))
            if not _has_lora_keys(lora_sd):
                raise RuntimeError(
                    f"No LoRA-style keys (lora_A/lora_B, lora_down/lora_up, diff/diff_b) found in {spec.path}"
                )
            previous_sd = fused_sd
            fused_sd, stats = fuse_lora_state_dict(previous_sd, lora_sd, scale=spec.scale, return_stats=True)
            label = f"{subfolder}: {Path(spec.path).name}"
            if stats.applied == 0:
                raise RuntimeError(
                    f"LoRA fusion applied no tensors for '{label}'. "
                    f"Skipped unmapped={stats.skipped_unmapped}, unknown={stats.skipped_unknown}, "
                    f"shape_mismatch={stats.skipped_shape_mismatch}."
                )
            verify_fusion_changed_weights(previous_sd, fused_sd, label=label)
        return fused_sd

    def _prepare_transformer(self, idx: int):
        state = self.transformer_states[idx]

        if not self._lora_specs[idx]:
            super()._prepare_transformer(idx)
            return

        def _get_state_dict(idx_=idx):
            cached = self._fused_state_dicts.get(idx_)
            if cached is not None:
                return cached
            sd = self._build_fused_state_dict(idx_)
            self._fused_state_dicts[idx_] = sd
            return sd

        cache.load_model(
            state.model,
            model_name=self._cache_namespace,
            subfolder=state.checkpoint.subfolder,
            parallel_config=self.parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            mesh_device=self.mesh_device,
            is_fsdp=self.is_fsdp,
            get_torch_state_dict=_get_state_dict,
        )
        self._fused_state_dicts[idx] = None

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
        boundary_ratio: float | None = 0.875,
        flow_shift: float | None = None,
        cfg_enabled: bool = True,
        lora_high: LoRAArg = None,
        lora_low: LoRAArg = None,
    ) -> WanPipelineI2VLora:
        """cfg_enabled=False halves denoising cost by dropping the unconditional pass.

        Only correct for adapters distilled to be guidance-free; with it off the caller
        must keep both guidance scales at 1.0, which __call__ enforces.
        """
        config = WanPipelineConfig.default(
            mesh_shape=mesh_device.shape,
            checkpoint_name=_resolve_checkpoint(cls.BASE_DIFFUSERS_REPO),
            height=height,
            width=width,
            num_frames=num_frames,
            num_links=num_links,
            topology=topology,
            dynamic_load=dynamic_load,
            is_fsdp=is_fsdp,
            boundary_ratio=boundary_ratio,
            cfg_enabled=cfg_enabled,
            model_type="i2v",
        )
        return cls(
            device=mesh_device,
            config=config,
            lora_high=lora_high,
            lora_low=lora_low,
            flow_shift=flow_shift,
        )
