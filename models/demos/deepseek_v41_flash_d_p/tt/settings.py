# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Every switch of deepseek_v41_flash_d_p in one table (core/model_settings.py). Environment variable =
``DS41_<NAME>``. Nothing else in tt/ or bringup/hooks.py reads the environment."""

from __future__ import annotations

from models.demos.common.bringup.core.model_settings import Setting, Settings

TABLE = {
    # ---- precision (owner rule 7: HiFi4 + fp32 accumulation for every matmul during bring-up)
    "MATMUL_FIDELITY": Setting(
        "HiFi4",
        ("HiFi4", "HiFi3", "HiFi2"),
        "owner rule: HiFi4 for every matmul (mHC projection, ...); lower only by an owner perf pick",
    ),
    "FP32_DEST_ACC": Setting(True, None, "owner rule: fp32 accumulation in every matmul / reduction"),
    "ATTN_IN_MM": Setting(
        "exact",
        ("exact", "fpu"),
        "the fp8 attention-input projection q(attn_norm) @ [wq_a | wkv] (tt/q_a.py:TtAttnInProj): exact = 32 masked "
        "copies in one batched matmul + SFPU fp32 adds (tt/exact_mm.py; an fp32 sum of exact products like the "
        "reference GEMM; ~8 ms vs 0.4 ms per layer at [1024, 2560] x [2560, 1792]); fpu = one ttnn.linear, whose "
        "in-tile partial sums round to ~TF32 (rel 1.8e-4: ~0.8 % fp8 flips in the window KV, DS41-16)",
    ),
    # ---- mHC (tt/mhc.py)
    "HC_TP_SUM": Setting(
        "gather_add",
        ("gather_add", "all_reduce"),
        "the [S/SP, 32] mHC partial-row exchange over axis 1: gather_add = all_gather + SFPU ttnn.add (exact fp32, "
        "K-30); all_reduce = one CCL whose FPU add reads fp32 at about TF32",
    ),
}

# Component and swap tests run at these values (framework rule); the shipped defaults above are judged end to end.
MAX_PRECISION = {
    "MATMUL_FIDELITY": "HiFi4",
    "FP32_DEST_ACC": True,
    "HC_TP_SUM": "gather_add",
    "ATTN_IN_MM": "exact",
}

settings = Settings("DS41_", TABLE, max_precision=MAX_PRECISION)
