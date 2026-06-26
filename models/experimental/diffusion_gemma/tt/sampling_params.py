# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""vLLM-facing sampling-parameter seam for DiffusionGemma canvas sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MODEL_CAPABILITIES = {
    "supports_prefix_caching": False,
    "supports_async_decode": False,
    "supports_sample_on_device": True,
}


@dataclass(frozen=True)
class CanvasSamplingConfig:
    """Resolved per-step parameters consumed by the device canvas sampler."""

    temperature: float
    seed: int | None = None
    top_k: int | None = None
    top_p: float | None = None
    top_k_top_p_supported: bool = False


def _first_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return value[0]
    return value


def _get_param(params: Any, name: str, default: Any = None) -> Any:
    if params is None:
        return default
    if isinstance(params, dict):
        return params.get(name, default)
    return getattr(params, name, default)


def canvas_sampling_config_from_params(
    sampling_params: Any,
    *,
    default_temperature: float,
    default_seed: int | None = None,
) -> CanvasSamplingConfig:
    """Duck-type vLLM ``TTSamplingParams`` into DiffusionGemma canvas parameters.

    DiffusionGemma's released sampler is temperature + Gumbel-max over every
    canvas position. ``top_k``/``top_p`` are parsed and carried for the future
    vLLM bridge, but they intentionally do not alter sampling until the reference
    ships those filters.
    """
    raw_temperature = _first_value(_get_param(sampling_params, "temperature", default_temperature))
    temperature = default_temperature if raw_temperature is None else float(raw_temperature)
    if temperature <= 0:
        raise ValueError("DiffusionGemma canvas sampling requires temperature > 0")

    raw_seed = _first_value(_get_param(sampling_params, "seed", default_seed))
    seed = None if raw_seed is None else int(raw_seed)

    raw_top_k = _first_value(_get_param(sampling_params, "top_k", None))
    top_k = None if raw_top_k is None else int(raw_top_k)

    raw_top_p = _first_value(_get_param(sampling_params, "top_p", None))
    top_p = None if raw_top_p is None else float(raw_top_p)

    return CanvasSamplingConfig(
        temperature=temperature,
        seed=seed,
        top_k=top_k,
        top_p=top_p,
        top_k_top_p_supported=False,
    )


def canvas_sample_from_params(
    logits,
    sampling_params: Any,
    *,
    default_temperature: float,
    default_seed: int | None = None,
    gumbel_noise=None,
    use_vocab_chunked_noise: bool = False,
    use_vocab_permuted_noise: bool = False,
    vocab_chunk_size: int = 1,
):
    """Apply duck-typed TT sampling params to the device canvas sampler.

    This is the small vLLM seam for W4: callers pass the vLLM-owned
    ``TTSamplingParams`` object (or a dict with the same fields), and the helper
    maps temperature/seed onto the per-position DiffusionGemma canvas sampler.
    ``top_k``/``top_p`` remain parsed-only until the reference sampler ships
    those filters. The permuted-vocab RNG workaround is opt-in until it is
    validated at production vocabulary scale.
    """
    from models.experimental.diffusion_gemma.tt import sampling as TS

    config = canvas_sampling_config_from_params(
        sampling_params,
        default_temperature=default_temperature,
        default_seed=default_seed,
    )
    if gumbel_noise is None:
        if config.seed is None:
            raise ValueError("canvas_sample_from_params requires gumbel_noise or a sampling seed")
        if use_vocab_chunked_noise and use_vocab_permuted_noise:
            raise ValueError("choose at most one regenerated-noise workaround")
        if use_vocab_permuted_noise:
            gumbel_noise = TS.sample_gumbel_noise_with_permuted_vocab(
                logits.shape,
                device=logits.device(),
                seed=config.seed,
            )
        elif use_vocab_chunked_noise:
            gumbel_noise = TS.sample_gumbel_noise_by_vocab_chunks(
                logits.shape,
                device=logits.device(),
                seed=config.seed,
                vocab_chunk_size=vocab_chunk_size,
            )
        else:
            gumbel_noise = TS.sample_gumbel_noise(logits.shape, device=logits.device(), seed=config.seed)

    return TS.canvas_sample(logits, config.temperature, gumbel_noise)
