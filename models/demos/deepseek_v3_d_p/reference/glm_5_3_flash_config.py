# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash configuration, read from its Hugging Face config.json.

``glm_5_3_flash/config.json`` is ``zai-org/GLM-5.3-Flash`` ``config.json`` at
``GLM_5_3_FLASH_HF_REVISION``, copied byte for byte (sha256
``bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f``). The model type is
``glm5_next`` (text tower ``glm5_next_text``); KDA layers use a low-rank output gate and the
bounded decay gate, and ``linear_attn_config.kda_layers`` is 0-indexed (layer 0 is a KDA layer).
"""

import json
from pathlib import Path
from typing import Any

from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig

GLM_5_3_FLASH_HF_REVISION = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"
_CONFIG_PATH = Path(__file__).with_name("glm_5_3_flash") / "config.json"


def glm_5_3_flash_model_config() -> dict[str, Any]:
    """Return the pinned GLM-5.3-Flash Hugging Face configuration mapping."""
    return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))


def glm_5_3_flash_kda_config() -> KDAConfig:
    """Build the TT KDA configuration from the pinned GLM-5.3-Flash config.json."""
    return KDAConfig.from_model_config(glm_5_3_flash_model_config())
