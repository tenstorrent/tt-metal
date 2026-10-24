# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Every switch of mimo_v26_flash_rl in one table. Environment variable = ``MIMO_V26_FLASH_RL_<NAME>``. Nothing else in tt/ or
bringup/hooks.py reads the environment (the framework's settings lint fails it).

Each precision switch (math fidelity, fp32 accumulation) also names its full-precision value in MAX_PRECISION:
component and swap tests run with it; the shipped defaults are judged end to end by the ladder. Weights and the KV
cache are bfloat8_b from the start (the prefill spec's dataformats), not switches lowered later.
"""

from ._settings_base import Setting, Settings  # noqa: F401  (Setting: the entries agents add to TABLE)

TABLE = {
    "NORM_FIDELITY": Setting(
        "HiFi4",
        ("HiFi4", "HiFi3", "HiFi2", "LoFi"),
        "rule 7 default: HiFi4 for the rms_norm multiplies (C.attn_dense.attn_norm)",
    ),
    "NORM_FP32_ACC": Setting(
        True, (True, False), "fp32 dest, as the reference's fp32 variance (C.attn_dense.attn_norm)"
    ),
}

MAX_PRECISION = {"NORM_FIDELITY": "HiFi4", "NORM_FP32_ACC": True}

settings = Settings("MIMO_V26_FLASH_RL_", TABLE, max_precision=MAX_PRECISION)
