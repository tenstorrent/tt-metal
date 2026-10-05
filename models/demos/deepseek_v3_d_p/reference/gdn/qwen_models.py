# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The four pinned Qwen GDN models and their Hugging Face ``config.json``.

``model_configs/<name>.json`` is the model's ``config.json`` at ``revision``, copied from the hub with a final newline
added (repository pre-commit); ``config_sha256`` is the upstream file's digest. All four models put their first GDN
(``linear_attention``) layer at index 0. Weight keys sit under ``model.language_model.`` when the config nests a
``text_config`` and under ``model.`` otherwise (Qwen3.8-2.4T is a text-only checkpoint).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models.demos.deepseek_v3_d_p.reference.gdn.config import GDNConfig

_CONFIG_DIR = Path(__file__).with_name("model_configs")


@dataclass(frozen=True)
class QwenGDNModel:
    repo: str
    revision: str
    config_sha256: str

    @property
    def local_name(self) -> str:
        """Directory name of local downloads: ``<org>--<name>``."""
        return self.repo.replace("/", "--")


QWEN_GDN_MODELS = {
    "qwen38_27b": QwenGDNModel(
        "Qwen/Qwen3.8-27B",
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab",
    ),
    "qwen36_35b": QwenGDNModel(
        "Qwen/Qwen3.6-35B-A3B",
        "995ad96eacd98c81ed38be0c5b274b04031597b0",
        "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99",
    ),
    "qwen38_2_4t": QwenGDNModel(
        "Qwen/Qwen3.8-2.4T-A95B",
        "207bd685a7e3696cfaff12ded7c6a7ea0f88c996",
        "4e3819548967e319ab435d044a3a331dbe3b078590ce822e9d74b79430533987",
    ),
    "qwen38_flash_next": QwenGDNModel(
        "Qwen/Qwen3.8-Flash-Next",
        "de4b8e4d43b917e7706784d8bb445c9af86a3540",
        "889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b",
    ),
}
QWEN_FIRST_GDN_LAYER = 0


def qwen_model_config(name: str) -> dict[str, Any]:
    """Return the pinned Hugging Face configuration mapping of ``name``."""
    if name not in QWEN_GDN_MODELS:
        raise ValueError(f"unknown Qwen GDN model {name!r}; expected one of {sorted(QWEN_GDN_MODELS)}")
    return json.loads((_CONFIG_DIR / f"{name}.json").read_text(encoding="utf-8"))


def qwen_gdn_config(name: str) -> GDNConfig:
    """Build the GDN layer configuration of ``name`` from its pinned config.json."""
    return GDNConfig.from_model_config(qwen_model_config(name))
