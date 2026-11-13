# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from .provider import DatasetProvider
from .templates import SST2_LABELS, SST2_PROMPT_TEMPLATE, SST2_RESPONSE_TEMPLATE


class SST2Provider(DatasetProvider):
    """GLUE SST-2 sentiment classification, answered as ``{"label": "positive"}``."""

    def _load_raw_dataset(self) -> Any:
        from datasets import load_dataset

        return load_dataset("nyu-mll/glue", "sst2", split=self.split)

    def _prompt_and_response(self, example: dict) -> tuple[str, str]:
        prompt = SST2_PROMPT_TEMPLATE.substitute(input=example["sentence"])
        return prompt, SST2_RESPONSE_TEMPLATE.substitute(label=SST2_LABELS[example["label"]])
