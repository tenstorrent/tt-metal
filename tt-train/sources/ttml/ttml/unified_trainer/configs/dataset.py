# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CustomDatasetConfig(BaseModel):
    """A dataset read from local files, for ``dataset.id: custom``. Train and validation come from separate files."""

    model_config = ConfigDict(extra="forbid")

    # Anything ``datasets.load_dataset`` reads ("json", "csv", ...); "jsonl" is read as "json".
    file_type: str = "json"
    train_dataset_path: Optional[str] = None
    # No validation when unset.
    val_dataset_path: Optional[str] = None
    # How an example becomes a prompt and a response (see ``ttml.datasets.templates``).
    template: str = "alpaca"
    # Template field -> dataset column, for columns not named like the template's fields.
    column_mapping: Optional[dict[str, str]] = None


class DatasetConfig(BaseModel):
    """The data a run trains on: one of the implemented datasets, or local files."""

    model_config = ConfigDict(extra="forbid")

    # An implemented dataset, "sst2" or "alpaca", or "custom" (see ``ttml.datasets.provider``).
    id: str = Field(min_length=1)
    # The files of a "custom" dataset and how to read them; "custom" needs them.
    custom_options: Optional[CustomDatasetConfig] = None

    @model_validator(mode="after")
    def _check_custom_options(self) -> "DatasetConfig":
        # Ids are matched case-insensitively (see ``ttml.datasets.provider.get_provider``).
        if self.id.lower() == "custom" and self.custom_options is None:
            raise ValueError("custom_options is required when id is 'custom'")
        return self
