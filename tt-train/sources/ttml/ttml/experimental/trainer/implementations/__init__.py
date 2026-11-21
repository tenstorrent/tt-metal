# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from .base import Trainer
from .sft import SFTTrainer

# The trainers `python -m ttml.experimental.trainer --trainer <name>` runs, by name.
TRAINERS: dict[str, type[Trainer]] = {"sft": SFTTrainer}

__all__ = ["SFTTrainer", "TRAINERS", "Trainer"]
