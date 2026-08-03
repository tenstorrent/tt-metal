# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Real Monash tourism-monthly observations, the benchmark this checkpoint was trained on.

Read from the Hub's parquet conversion rather than through the ``datasets`` package: it drops a
heavyweight dependency and keeps working against current ``huggingface_hub`` releases, where
``datasets`` currently fails to import.
"""

from __future__ import annotations

import torch


def tourism_series(config, *, batch: int, split: str = "test") -> dict[str, torch.Tensor]:
    """Real Monash tourism-monthly observations -- the data this checkpoint was trained on.

    Read straight from the Hub's parquet conversion rather than through ``datasets``, which
    keeps the demo working against current ``huggingface_hub`` releases and drops a heavyweight
    dependency. Time features are reconstructed as the standard GluonTS monthly pair: the month
    of year scaled to [-0.5, 0.5], and a log-scaled age counter.
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "monash_tsf",
        f"tourism_monthly/{split}/0000.parquet",
        repo_type="dataset",
        revision="refs/convert/parquet",
    )
    frame = pd.read_parquet(path)

    past_length = int(config.context_length) + int(max(config.lags_sequence))
    horizon = int(config.prediction_length)
    needed = past_length + horizon

    usable = [row for _, row in frame.iterrows() if len(row["target"]) >= needed]
    if len(usable) < batch:
        raise ValueError(f"Only {len(usable)} tourism series reach {needed} steps; need {batch}.")

    values, time_features, static_categorical = [], [], []
    for row in usable[:batch]:
        target = torch.tensor(list(row["target"][-needed:]), dtype=torch.float32)
        values.append(target)

        # Absolute month index of each step, so month-of-year lines up with the calendar.
        offset = len(row["target"]) - needed
        start_month = pd.Timestamp(row["start"]).month - 1
        absolute = torch.arange(offset, offset + needed, dtype=torch.float32)
        month_of_year = ((start_month + absolute) % 12.0) / 11.0 - 0.5
        age = torch.log10(2.0 + absolute)
        time_features.append(torch.stack((month_of_year, age), dim=-1))

        static_categorical.append(int(row["feat_static_cat"][0]))

    values = torch.stack(values)
    time_features = torch.stack(time_features)

    return {
        "past_values": values[:, :past_length].contiguous(),
        "future_values": values[:, past_length:].contiguous(),
        "past_time_features": time_features[:, :past_length].contiguous(),
        "future_time_features": time_features[:, past_length:].contiguous(),
        "past_observed_mask": torch.ones(batch, past_length),
        "static_categorical_features": torch.tensor(static_categorical).reshape(batch, 1),
        "static_real_features": torch.zeros(batch, int(config.num_static_real_features)),
    }
