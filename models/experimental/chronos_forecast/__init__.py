# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Experimental Chronos forecasting model for TT-NN."""

from models.experimental.chronos_forecast.common.chronos_src import (
    CHRONOS_VERSION,
    DUMMY_MODEL_PATH,
    require_chronos,
)

__version__ = "0.1.0"

__all__ = [
    "CHRONOS_VERSION",
    "DUMMY_MODEL_PATH",
    "require_chronos",
]
