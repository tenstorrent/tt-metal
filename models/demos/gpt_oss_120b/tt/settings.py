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

TABLE = {
    "NORM_FIDELITY": Setting("HiFi4", ("HiFi4", "HiFi3", "HiFi2", "LoFi"), "rule 7 default: rms_norm at HiFi4"),
    "NORM_FP32_ACC": Setting(True, (True, False), "rule 7 default: rms_norm with fp32 dest accumulation"),
}

MAX_PRECISION = {"NORM_FIDELITY": "HiFi4", "NORM_FP32_ACC": True}

settings = Settings("GPT_OSS_120B_", TABLE, max_precision=MAX_PRECISION)
