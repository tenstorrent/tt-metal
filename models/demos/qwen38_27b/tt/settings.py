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
    "NORM_FIDELITY": Setting("HiFi4", ("HiFi4", "HiFi3", "HiFi2", "LoFi"), "rule 7 default: RMSNorm compute at HiFi4"),
    "NORM_FP32_ACC": Setting(True, (True, False), "rule 7 default: RMSNorm accumulates in fp32"),
    "NORM_WEIGHT_DTYPE": Setting(
        "float32",
        ("float32", "bfloat16"),
        "the folded 1 + w gamma kept in fp32: bf16 rounds 1 + w to 2^-8 per column (rms_norm takes no bfp8 gamma)",
    ),
}

MAX_PRECISION = {"NORM_FIDELITY": "HiFi4", "NORM_FP32_ACC": True, "NORM_WEIGHT_DTYPE": "float32"}

settings = Settings("QWEN38_27B_", TABLE, max_precision=MAX_PRECISION)
