# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-M3 model variants: which repo config (``configs/<name>/config.json``) and default paths a run uses.

Both variants share the architecture and differ only in config values (RoPE base, attention layer pattern,
MSA flags) and checkpoint format. A run's variant is:

* detected from the checkpoint's own ``config.json`` when ``HF_MODEL`` points at one (``detect_variant``:
  the text config's ``architectures`` entry, else the model-defining fields), so a checkpoint is never
  paired with the wrong defaults;
* else ``M3_MODEL_VARIANT`` (tests / harnesses without a checkpoint, e.g. random weights);
* else ``MiniMax-M3``.

The model itself always reads the checkpoint's config when there is one; the repo config is what random-weight
tests and checkpoint-less tools build from, and the variant also picks the default weights / cache / golden dirs.

Pure Python (no ttnn / torch) so host tests and the prefill adapter can import it cheaply.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from loguru import logger

VARIANT_ENV = "M3_MODEL_VARIANT"
DEFAULT_VARIANT = "MiniMax-M3"
CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"

# Keys of the flattened repo config that describe the file or the HF runtime, not the model.
_METADATA_KEYS = frozenset(
    {
        "_comment",
        "architectures",
        "model_type",
        "torch_dtype",
        "transformers_version",
        "use_cache",
        "attention_dropout",
        "quantization_config",  # on the checkpoint's VL wrapper, not its text_config
    }
)


@dataclass(frozen=True)
class ModelVariant:
    name: str
    weights_dir: str  # default checkpoint dir on the exabox weka filesystem
    cache_root: str  # default tilized weight-cache root (holds tensor_cache_bfp8_MeshShape([sp, tp]) and golden/)
    precise_prefill: bool = False  # default of the prefill precision switches (tt/compute_configs.py)

    @property
    def config_path(self) -> Path:
        return CONFIGS_DIR / self.name / "config.json"

    @property
    def golden_dir(self) -> str:
        return self.cache_root + "/golden"

    @property
    def config(self) -> dict:
        """The flattened text config (a fresh copy; callers may mutate it)."""
        return json.loads(json.dumps(_read_json(str(self.config_path))))

    @property
    def architecture(self) -> str:
        return self.config["architectures"][0]


VARIANTS = {
    v.name: v
    for v in (
        ModelVariant(
            name="MiniMax-M3",
            weights_dir="/mnt/weka/model-weights/llm/minimax/MiniMax-M3",
            cache_root="/mnt/weka/model-cache/scratch/minimax/MiniMax-M3-cache/prefill",
        ),
        ModelVariant(
            name="MiniMax-M3.1",
            weights_dir="/mnt/weka/model-weights/llm/minimax/m3",
            cache_root="/mnt/weka/model-cache/scratch/minimax/m3-cache/prefill",
            precise_prefill=True,
        ),
    )
}


@lru_cache(maxsize=None)
def _read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def get_variant(name: str) -> ModelVariant:
    try:
        return VARIANTS[name]
    except KeyError:
        raise ValueError(f"unknown MiniMax-M3 model variant {name!r}; valid: {sorted(VARIANTS)}") from None


def text_config_dict(cfg: dict) -> dict:
    """The text-backbone part of a (possibly VL-wrapped) checkpoint ``config.json``."""
    return cfg.get("text_config", cfg)


def read_checkpoint_text_config(model_path) -> Optional[dict]:
    """The text config of the checkpoint at ``model_path``, or None when it has no ``config.json``."""
    path = Path(model_path) / "config.json"
    if not path.is_file():
        return None
    with open(path) as f:
        return text_config_dict(json.load(f))


def config_differences(a: dict, b: dict) -> list:
    """Model-defining keys whose values differ between two flattened text configs (metadata ignored)."""
    keys = (set(a) | set(b)) - _METADATA_KEYS
    return sorted(k for k in keys if a.get(k) != b.get(k))


def detect_variant(text_config: dict) -> ModelVariant:
    """The variant a checkpoint's text config belongs to.

    Matches the text config's ``architectures`` entry first (the two releases name their text model
    differently); a config without a known architecture matches the variant whose model-defining fields it
    equals exactly. An architecture match whose fields differ is still that variant (a truncated or patched
    checkpoint), with a warning: the model runs on the checkpoint's own values. Raises ValueError otherwise.
    """
    text_config = text_config_dict(text_config)
    archs = text_config.get("architectures") or []
    for variant in VARIANTS.values():
        if variant.architecture in archs:
            diffs = config_differences(variant.config, text_config)
            if diffs:
                logger.warning(
                    f"checkpoint config is a {variant.name} ({variant.architecture}) but differs from "
                    f"{variant.config_path} in {diffs}; the model uses the checkpoint's values"
                )
            return variant
    exact = [v for v in VARIANTS.values() if not config_differences(v.config, text_config)]
    if len(exact) == 1:
        return exact[0]
    raise ValueError(
        f"cannot tell which MiniMax-M3 variant this checkpoint config is (architectures={archs}); "
        f"closest differences: { {v.name: config_differences(v.config, text_config) for v in VARIANTS.values()} }"
    )


def checkpoint_dir_env() -> Optional[str]:
    """The checkpoint dir the environment names: ``$PREFILL_HF_MODEL`` (the prefill runner's) > ``$HF_MODEL``."""
    return os.environ.get("PREFILL_HF_MODEL") or os.environ.get("HF_MODEL")


def select_variant(hf_model: Optional[str] = None) -> ModelVariant:
    """The run's variant: detected from ``hf_model``'s config.json (default ``$HF_MODEL``) when there is one,
    else ``$M3_MODEL_VARIANT``, else ``MiniMax-M3``. An explicit ``M3_MODEL_VARIANT`` that contradicts the
    checkpoint is an error; it only decides when the checkpoint config is not recognized."""
    requested = os.environ.get(VARIANT_ENV) or None
    if requested is not None:
        requested = get_variant(requested)
    hf_model = hf_model if hf_model is not None else os.environ.get("HF_MODEL")
    text_config = read_checkpoint_text_config(hf_model) if hf_model else None
    if text_config is None:
        return requested or get_variant(DEFAULT_VARIANT)
    try:
        detected = detect_variant(text_config)
    except ValueError as e:
        if requested is None:
            raise ValueError(f"{e}. Set {VARIANT_ENV} to choose one.") from None
        logger.warning(f"{e}; using {VARIANT_ENV}={requested.name}")
        return requested
    if requested is not None and requested != detected:
        raise ValueError(f"{VARIANT_ENV}={requested.name} but the checkpoint {hf_model} is a {detected.name}")
    return detected


def load_config(variant: Optional[str] = None, **overrides) -> dict:
    """The repo config of ``variant`` (default: ``select_variant()``) as a dict, with ``overrides`` applied."""
    v = get_variant(variant) if variant else select_variant()
    cfg = v.config
    cfg.update(overrides)
    return cfg


def load_hf_config(variant: Optional[str] = None, **overrides) -> SimpleNamespace:
    """``load_config`` as an attribute namespace, the shape TT modules read (``hf_config.hidden_size`` ...)."""
    return SimpleNamespace(**load_config(variant, **overrides))


def with_layer_schedule(cfg: dict, moe_layer_freq, sparse_attention_freq) -> dict:
    """``cfg`` cut to ``len(moe_layer_freq)`` layers with an explicit MLP (1 = MoE) and attention (1 = MSA)
    schedule, for tests that build a few chosen layers instead of the variant's own pattern."""
    n = len(moe_layer_freq)
    assert len(sparse_attention_freq) == n, f"{len(sparse_attention_freq)} attention flags for {n} layers"
    sparse = dict(
        cfg["sparse_attention_config"],
        sparse_attention_freq=list(sparse_attention_freq),
        sparse_disable_index_value=[1] * n,
    )
    return dict(cfg, num_hidden_layers=n, moe_layer_freq=list(moe_layer_freq), sparse_attention_config=sparse)
