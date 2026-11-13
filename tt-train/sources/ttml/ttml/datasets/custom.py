# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .provider import DatasetProvider
from .templates import custom_prompt_and_response, resolve_column_mapping

if TYPE_CHECKING:
    from ttml.unified_trainer.configs import SFTTrainerConfig


class CustomProvider(DatasetProvider):
    """A local JSON / JSONL / CSV / Parquet file per split, rendered with a template (``dataset.custom_options``)."""

    def __init__(self, config: SFTTrainerConfig, split: str, tokenizer: Any) -> None:
        custom = config.dataset.custom_options
        # `datasets` reads JSON Lines with its "json" builder.
        self.file_type = "json" if custom.file_type == "jsonl" else custom.file_type
        self.data_path = custom.train_dataset_path if split == "train" else custom.val_dataset_path
        self.template = custom.template
        self.column_mapping = custom.column_mapping
        super().__init__(config, split, tokenizer)

    @classmethod
    def has_split(cls, config: SFTTrainerConfig, split: str) -> bool:
        # Validation is optional: there is none without a file for it.
        return split == "train" or bool(config.dataset.custom_options.val_dataset_path)

    def _load_raw_dataset(self) -> Any:
        if not self.data_path:
            field = "train_dataset_path" if self.split == "train" else "val_dataset_path"
            raise ValueError(f"dataset.custom_options.{field} is required")
        from datasets import load_dataset

        raw_dataset = load_dataset(self.file_type, data_files={self.split: self.data_path}, split=self.split)
        self.column_mapping = resolve_column_mapping(self.template, self.column_mapping, set(raw_dataset.column_names))
        return raw_dataset

    def _prompt_and_response(self, example: dict) -> tuple[str, str]:
        return custom_prompt_and_response(example, self.template, self.column_mapping)
