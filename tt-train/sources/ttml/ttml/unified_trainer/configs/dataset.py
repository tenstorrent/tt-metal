# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from pydantic import BaseModel, ConfigDict


class CustomDatasetConfig(BaseModel):
    """A dataset read from local files, for ``dataset_id: custom``. Train and validation come from separate files."""

    model_config = ConfigDict(extra="forbid")

    # Anything ``datasets.load_dataset`` reads ("json", "csv", ...); "jsonl" is read as "json".
    file_type: str = "json"
    train_dataset_path: Optional[str] = None
    # No validation when unset.
    val_dataset_path: Optional[str] = None
    # How an example becomes a prompt and a response (see ``ttml.unified_trainer.data.templates``).
    template: str = "alpaca"
    # Template field -> dataset column, for columns not named like the template's fields.
    column_mapping: Optional[dict[str, str]] = None
