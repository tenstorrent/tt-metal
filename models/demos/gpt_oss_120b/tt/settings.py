# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Every switch of gpt_oss_120b in one table. Environment variable = ``GPT_OSS_120B_<NAME>``. Nothing else in tt/ or
bringup/hooks.py reads the environment (the framework's settings lint fails it).

Each precision switch (math fidelity, fp32 accumulation) also names its full-precision value in MAX_PRECISION:
component and swap tests run with it; the shipped defaults are judged end to end by the ladder. Weights and the KV
cache are bfloat8_b from the start (the prefill spec's dataformats), not switches lowered later.
"""

from ._settings_base import Setting, Settings  # noqa: F401  (Setting: the entries agents add to TABLE)

TABLE = {}

MAX_PRECISION = {}

settings = Settings("GPT_OSS_120B_", TABLE, max_precision=MAX_PRECISION)
