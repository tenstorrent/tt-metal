# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch Chronos-1 helpers over the amazon-science submodule.

Chronos-2 golden modules live in ``reference.chronos2`` (verbatim vendor). This
file keeps the original Chronos tokenizer / pipeline re-exports for the demo.
"""

from __future__ import annotations

from models.experimental.chronos_forecast.common.chronos_src import ensure_chronos_on_path
from models.experimental.chronos_forecast.common.configs import ChronosModelConfig

ensure_chronos_on_path()

from chronos import (  # noqa: E402
    BaseChronosPipeline,
    ChronosBoltConfig,
    ChronosBoltPipeline,
    ChronosConfig,
    ChronosModel,
    ChronosPipeline,
    ChronosTokenizer,
    ForecastType,
    MeanScaleUniformBins,
)


def create_chronos_config(model_config: ChronosModelConfig | None = None) -> ChronosConfig:
    """Build an upstream ChronosConfig from the shared bring-up dataclass."""
    cfg = model_config or ChronosModelConfig()
    return ChronosConfig(
        tokenizer_class=cfg.tokenizer_class,
        tokenizer_kwargs=dict(cfg.tokenizer_kwargs),
        n_tokens=cfg.n_tokens,
        n_special_tokens=cfg.n_special_tokens,
        pad_token_id=cfg.pad_token_id,
        eos_token_id=cfg.eos_token_id,
        use_eos_token=cfg.use_eos_token,
        model_type=cfg.model_type,
        context_length=cfg.context_length,
        prediction_length=cfg.prediction_length,
        num_samples=cfg.num_samples,
        temperature=cfg.temperature,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
    )


__all__ = [
    "BaseChronosPipeline",
    "ChronosBoltConfig",
    "ChronosBoltPipeline",
    "ChronosConfig",
    "ChronosModel",
    "ChronosModelConfig",
    "ChronosPipeline",
    "ChronosTokenizer",
    "ForecastType",
    "MeanScaleUniformBins",
    "create_chronos_config",
]
