"""Native-trunk Cosmos3 omni pipeline.

Supports four modes via one entry point:
  - I2V              : image given,  enable_sound=False
  - T2V              : image=None,   enable_sound=False
  - I2V+audio        : image given,  enable_sound=True
  - T2V+audio        : image=None,   enable_sound=True

Delegates the heavy build (TT VAE adapters, NativeLayerProxy install, host VAE
monkey-patches) to `build_cosmos3_i2v_native_pipeline`. The only structural
difference vs the I2V builder is the default HF repo and flow_shift preset.
"""

from __future__ import annotations

from typing import Any

import torch

import ttnn
from models.tt_dit.experimental.cosmos3_i2v.pipelines.pipeline_cosmos3_i2v_native import (
    build_cosmos3_i2v_native_pipeline,
)

HF_REPO_OMNI = "nvidia/Cosmos3-Super"

# Cosmos3 paper Table 21: shift=10 is the Audio-Visual omni preset. Same value
# also empirically improves 720p sharpness even without audio (see
# project_cosmos3_flow_shift_720p memory).
FLOW_SHIFT_OMNI = 10.0


def build_cosmos3_omni_native_pipeline(
    device: ttnn.MeshDevice,
    *,
    dtype: torch.dtype | None = None,
    hf_repo: str = HF_REPO_OMNI,
    enable_vae_tiling: bool = False,
    num_links: int | None = None,
    trunk_weight_dtype: ttnn.DataType = ttnn.bfloat16,
    use_tt_vae: bool = True,
    vae_encoder_t_chunk_size: int | None = None,
    vae_decoder_t_chunk_size: int | None = None,
    flow_shift: float = FLOW_SHIFT_OMNI,
) -> Any:
    # Distinct cache namespace: Super and I2V share architecture but differ in trunk
    # weights (sound_gen=True vs False). Reusing the I2V cache slot loads I2V-trained
    # decoder layers under a Super HF pipeline and produces pure noise.
    pipe = build_cosmos3_i2v_native_pipeline(
        device,
        dtype=dtype,
        hf_repo=hf_repo,
        enable_vae_tiling=enable_vae_tiling,
        num_links=num_links,
        trunk_weight_dtype=trunk_weight_dtype,
        use_tt_vae=use_tt_vae,
        vae_encoder_t_chunk_size=vae_encoder_t_chunk_size,
        vae_decoder_t_chunk_size=vae_decoder_t_chunk_size,
        flow_shift=flow_shift,
        cache_namespace="cosmos3-omni",
    )
    return pipe


def _release_native_trunks(pipe) -> None:
    layers = getattr(pipe.transformer, "layers", None)
    if not layers:
        return
    proxy = layers[0]
    trunks = []
    if hasattr(proxy, "_proxy_a") and hasattr(proxy, "_proxy_b"):
        trunks.append(proxy._proxy_a._native_trunk)
        trunks.append(proxy._proxy_b._native_trunk)
    elif hasattr(proxy, "_native_trunk"):
        trunks.append(proxy._native_trunk)
    for trunk in trunks:
        if hasattr(trunk, "deallocate_weights"):
            trunk.deallocate_weights()


def _make_release_callback(num_steps: int):
    def cb(pipe_ref, step, timestep, callback_kwargs):  # noqa: ARG001
        if step == num_steps - 1:
            _release_native_trunks(pipe_ref)
        return callback_kwargs

    return cb


def run_cosmos3_omni(
    pipe,
    *,
    prompt: str,
    image: torch.Tensor | None = None,
    enable_sound: bool = False,
    num_frames: int = 17,
    height: int = 704,
    width: int = 1280,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    fps: float = 24.0,
    generator: torch.Generator | None = None,
    output_type: str = "pt",
    free_trunk_before_vae: bool = True,
    **extra,
):
    if enable_sound:
        cfg = getattr(pipe.transformer, "config", None)
        if not getattr(cfg, "sound_gen", False):
            raise ValueError(
                "enable_sound=True requires a sound_gen=True checkpoint; "
                f"got sound_gen={getattr(cfg, 'sound_gen', None)} for "
                f"{getattr(cfg, '_name_or_path', '<unknown>')!r}. "
                f"Use hf_repo='{HF_REPO_OMNI}'."
            )
        if not hasattr(pipe, "sound_tokenizer") or pipe.sound_tokenizer is None:
            raise ValueError(
                "enable_sound=True but pipe.sound_tokenizer is missing. "
                "Checkpoint snapshot is incomplete; ensure the sound_tokenizer/ "
                "subfolder downloaded."
            )

    if free_trunk_before_vae and "callback_on_step_end" not in extra:
        extra["callback_on_step_end"] = _make_release_callback(num_inference_steps)
        extra.setdefault("callback_on_step_end_tensor_inputs", [])

    return pipe(
        prompt=prompt,
        image=image,
        num_frames=num_frames,
        height=height,
        width=width,
        fps=fps,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        enable_sound=enable_sound,
        generator=generator,
        output_type=output_type,
        **extra,
    )
