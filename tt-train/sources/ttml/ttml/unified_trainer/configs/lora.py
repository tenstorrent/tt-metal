# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, ConfigDict, Field


class LoraConfig(BaseModel):
    """LoRA adapters, in PEFT's terms: HF module names, and the scaling ``alpha / r``.

    The model provider maps these onto the ttml model, and rejects targets the model can't train
    (see ``ttml.models.provider``).
    """

    model_config = ConfigDict(extra="forbid")

    r: int = Field(gt=0)
    alpha: int = Field(gt=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    # HF module names, e.g. ["q_proj", "k_proj", "v_proj", "o_proj"].
    target_modules: list[str] = Field(min_length=1)
    task_type: str = "CAUSAL_LM"
