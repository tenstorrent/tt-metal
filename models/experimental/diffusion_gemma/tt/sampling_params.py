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
