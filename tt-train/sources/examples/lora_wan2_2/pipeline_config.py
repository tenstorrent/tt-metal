# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Config for the Wan2.2 T2V-A14B LoRA pipeline. Values come from the YAML; see README.md."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = "${TT_METAL_HOME}/tt-train/configs/training_configs/wan2_2_t2v_a14b_lora.yaml"

_SECTIONS = (
    "model",
    "data",
    "lora",
    "optimizer",
    "training",
    "inference",
    "device",
    "logging",
)

SUBFOLDER = {"high": "transformer", "low": "transformer_2"}


@dataclass
class Config:
    MODEL_ID: str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

    DTYPE: str = "bfloat16"
    VAE_DTYPE: str = "bfloat16"
    GRADIENT_CHECKPOINTING: bool = True

    DATASET_ID: str = "showlab/OmniConsistency"
    STYLE: str = "LEGO"
    DATA_DIR: str = "data/lego"
    CACHE_DIR: str = "cache/wan22_14b_lego"

    TRAIN_H: int = 512
    TRAIN_W: int = 512
    # 4k+1. Stills are repeated into a static clip so the LoRA stylizes every
    # temporal position; a 1-frame cache leaves video inference unadapted.
    TRAIN_FRAMES: int = 13

    TRIGGER: str = "lg, "
    STRIP_STYLE_WORDS: bool = True
    TEXT_DROP_PROB: float = 0.10
    SUBSET_SIZE: int = 0
    MAX_SEQ: int = 512
    SEED: int = 42
    VAL_HOLDOUT: int = 4

    BOUNDARY_RATIO: float = 0.875
    TRAIN_EXPERTS: str = "both"

    LORA_RANK: int = 32
    LORA_ALPHA: int = 32
    LORA_TARGET_SET: str = "attn"
    # "gaussian" = N(0, 1/rank), matching PEFT. "kaiming" = ttml's own init,
    # ~4x smaller at rank 32, which trains a proportionally weaker adapter.
    LORA_A_INIT: str = "gaussian"
    LORA_PATH: str = "cache/wan22_14b_lego/wan22_14b_lego_lora.safetensors"

    LR: float = 1e-4
    WEIGHT_DECAY: float = 0.01
    BATCH: int = 1
    GRAD_ACCUM: int = 4
    MAX_STEPS: int = 3000
    GRAD_CLIP: float = 1.0
    TRAIN_FLOW_SHIFT: float = 3.0
    LOGNORM_MEAN: float = 0.0
    LOGNORM_STD: float = 1.0
    VAL_LOSS_EVERY: int = 200
    CKPT_EVERY: int = 500
    RESUME_STEP: int = 0

    INFER_H: int = 512
    INFER_W: int = 512
    INFER_FRAMES: int = 49
    INFER_FPS: int = 16
    INFER_STEPS: int = 40
    INFER_GUIDANCE: float = 7.0
    INFER_GUIDANCE_2: float = 5.0
    INFER_FLOW_SHIFT: float = 12.0
    INFER_OUTPUT: str = "cache/wan22_14b_lego/lego_video.mp4"
    VAL_PROMPT: str = "a cat sitting on a wooden table"
    NEG_PROMPT: str = ""
    INFER_NO_LORA: bool = False
    LORA_SCALE: float = 1.0
    INFER_HIGH_LORA: str = ""
    INFER_LOW_LORA: str = ""

    WANDB_PROJECT: str = "wan22-14b-lego-lora"
    WANDB_ENABLED: bool = True

    MESH_SHAPE: tuple = (1, 1)

    @classmethod
    def field_names(cls) -> set[str]:
        return {f.name for f in fields(cls)}

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "Config":
        resolved = Path(os.path.expandvars(str(path or DEFAULT_CONFIG_PATH)))
        if not resolved.is_file():
            raise FileNotFoundError(
                f"config not found: {resolved}\n"
                f"Pass -c/--config, or set TT_METAL_HOME so the default path resolves."
            )
        raw = yaml.safe_load(resolved.read_text()) or {}

        known = cls.field_names()
        values: dict = {}
        for section, entries in raw.items():
            if section not in _SECTIONS:
                raise ValueError(f"{resolved}: unknown section {section!r}; expected one of: {', '.join(_SECTIONS)}")
            if entries is None:
                continue
            if not isinstance(entries, dict):
                raise ValueError(f"{resolved}: section {section!r} must be a mapping, got {type(entries).__name__}")
            for key, value in entries.items():
                name = str(key).upper()
                if name not in known:
                    raise ValueError(f"{resolved}: unknown key {section}.{key!r} (no Config field {name})")
                if name in values:
                    raise ValueError(f"{resolved}: {key!r} set more than once (also in another section)")
                values[name] = os.path.expandvars(value) if isinstance(value, str) else value

        if "MESH_SHAPE" in values:
            values["MESH_SHAPE"] = tuple(values["MESH_SHAPE"])
        cfg = cls(**values)
        cfg.validate()
        return cfg

    def apply_overrides(self, overrides: list[str]) -> "Config":
        types = {f.name: f.type for f in fields(self)}
        for item in overrides or []:
            if "=" not in item:
                raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
            key, _, raw_value = item.partition("=")
            name = key.strip().upper()
            if name not in types:
                raise ValueError(f"--set: no Config field {name}")
            setattr(self, name, _coerce(name, raw_value.strip(), types[name]))
        self.validate()
        return self

    def validate(self) -> None:
        if self.TRAIN_EXPERTS not in ("low", "high", "both"):
            raise ValueError(f"TRAIN_EXPERTS must be low|high|both, got {self.TRAIN_EXPERTS!r}")
        if self.LORA_TARGET_SET not in ("attn", "attn+ffn"):
            raise ValueError(f"LORA_TARGET_SET must be attn|attn+ffn, got {self.LORA_TARGET_SET!r}")
        if self.LORA_A_INIT not in ("gaussian", "kaiming"):
            raise ValueError(f"LORA_A_INIT must be gaussian|kaiming, got {self.LORA_A_INIT!r}")
        if len(tuple(self.MESH_SHAPE)) != 2:
            raise ValueError(f"MESH_SHAPE must have two entries, got {self.MESH_SHAPE!r}")
        if (self.TRAIN_FRAMES - 1) % 4 != 0:
            raise ValueError(f"TRAIN_FRAMES must be 4k+1, got {self.TRAIN_FRAMES}")
        if self.INFER_FRAMES != 1 and (self.INFER_FRAMES - 1) % 4 != 0:
            raise ValueError(f"INFER_FRAMES must be 1 or 4k+1, got {self.INFER_FRAMES}")
        if self.BATCH < 1:
            raise ValueError(f"BATCH must be >= 1, got {self.BATCH}")

    def asdict(self) -> dict:
        d = asdict(self)
        d["MESH_SHAPE"] = list(self.MESH_SHAPE)
        return d

    def experts_to_load(self) -> list[str]:
        return {"low": ["low"], "high": ["high"], "both": ["high", "low"]}[self.TRAIN_EXPERTS]

    def expert_path(self, role: str) -> str:
        stem = Path(self.LORA_PATH)
        return str(stem.with_name(stem.stem + f"_{role}.safetensors"))


def _coerce(name: str, raw: str, declared_type) -> object:
    # `from __future__ import annotations` makes field types strings, so match on text.
    text = declared_type if isinstance(declared_type, str) else getattr(declared_type, "__name__", str(declared_type))
    if text.startswith("bool"):
        lowered = raw.lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
        raise ValueError(f"--set {name}: expected a boolean, got {raw!r}")
    if text.startswith("int"):
        return int(raw)
    if text.startswith("float"):
        return float(raw)
    if text.startswith("tuple"):
        parts = [p for p in raw.replace(",", " ").replace("[", " ").replace("]", " ").split() if p]
        return tuple(int(p) for p in parts)
    return raw
