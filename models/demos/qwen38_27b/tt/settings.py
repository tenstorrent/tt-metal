# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Every switch of qwen38_27b in one table. Environment variable = ``QWEN38_27B_<NAME>``. Nothing else in tt/ or
bringup/hooks.py reads the environment (the framework's settings lint fails it).

Each precision switch (math fidelity, fp32 accumulation) also names its full-precision value in MAX_PRECISION:
component and swap tests run with it; the shipped defaults are judged end to end by the ladder. Weights and the KV
cache are bfloat8_b from the start (the prefill spec's dataformats), not switches lowered later.
"""

from ._settings_base import Setting, Settings  # noqa: F401  (Setting: the entries agents add to TABLE)

TABLE = {
    "NORM_FIDELITY": Setting("HiFi4", ("HiFi4", "HiFi2", "LoFi"), "ENGINEER rule 7: HiFi4 by default"),
    "NORM_FP32_ACC": Setting(True, None, "ENGINEER rule 7: fp32 accumulation by default"),
    "NORM_WEIGHT_DTYPE": Setting(
        "float32",
        ("float32", "bfloat16"),
        "the (1 + w) gamma is pre-offset in fp32 at load; bf16 would round 1 + w to 8 mantissa bits (10 KB either way)",
    ),
}

MAX_PRECISION = {"NORM_FIDELITY": "HiFi4", "NORM_FP32_ACC": True, "NORM_WEIGHT_DTYPE": "float32"}

settings = Settings("QWEN38_27B_", TABLE, max_precision=MAX_PRECISION)
