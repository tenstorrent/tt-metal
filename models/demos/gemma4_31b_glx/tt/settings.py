# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Every switch of gemma4_31b_glx in one table. Environment variable = ``GEMMA4_31B_GLX_<NAME>``. Nothing else in tt/ or
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
        "ENGINEER rule 7 default; gemma4_d_p runs its norms at HiFi2 (a perf choice for later).",
    ),
    "NORM_FP32_ACC": Setting(True, None, "fp32 accumulation as in the reference's fp32 norm."),
    "NORM_SHARDED": Setting(
        True,
        None,
        "Block-sharded rms_norm (gemma4_d_p rms_norm.py geometry); False: interleaved, one core per tile row.",
    ),
    "COMPONENT_FAST_RUNTIME": Setting(
        False,
        None,
        "ttnn fast runtime for device_component modules (bring-up harness only). On this tt-metal the fast path "
        "skips ttnn's pre-operation hooks, so the framework's device-execution proof counts 0 ops; off = the same "
        "ops through ttnn's slow Python dispatch, which runs the hooks.",
    ),
}

MAX_PRECISION = {"NORM_FIDELITY": "HiFi4", "NORM_FP32_ACC": True}

settings = Settings("GEMMA4_31B_GLX_", TABLE, max_precision=MAX_PRECISION)
