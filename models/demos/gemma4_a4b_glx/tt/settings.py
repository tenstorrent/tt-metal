# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Every switch of gemma4_a4b_glx in one table. Environment variable = ``GEMMA4_A4B_GLX_<NAME>``. Nothing else in tt/ or
bringup/hooks.py reads the environment (the framework's settings lint fails it).

Each precision switch (math fidelity, fp32 accumulation) also names its full-precision value in MAX_PRECISION:
component and swap tests run with it; the shipped defaults are judged end to end by the ladder. Weights and the KV
cache are bfloat8_b from the start (the prefill spec's dataformats), not switches lowered later.
"""

from ._settings_base import Setting, Settings  # noqa: F401  (Setting: the entries agents add to TABLE)

TABLE = {
    "ATTN_MATMUL_FIDELITY": Setting(
        "HiFi4", ("HiFi4", "HiFi2", "LoFi"), "2026-10-13 C.sliding.attention: QKV/o_proj/rotary math fidelity"
    ),
    "ATTN_MATMUL_FP32_ACC": Setting(
        True, (True, False), "2026-10-13 C.sliding.attention: fp32 accumulation for attention matmuls"
    ),
    "SDPA_FIDELITY": Setting("HiFi4", ("HiFi4", "HiFi2", "LoFi"), "2026-10-13 C.sliding.attention: SDPA math fidelity"),
    "SDPA_FP32_ACC": Setting(True, (True, False), "2026-10-13 C.sliding.attention: SDPA fp32 accumulation"),
    "SDPA_Q_CHUNK": Setting(256, (32, 64, 128, 256, 512), "2026-10-13 C.sliding.attention: SDPA q chunk size"),
    "SDPA_K_CHUNK": Setting(128, (32, 64, 128, 256, 512), "2026-10-13 C.sliding.attention: SDPA k chunk size"),
    "GLOBAL_SDPA_Q_CHUNK": Setting(
        128, (32, 64, 128, 256), "2026-10-13 C.global.attention: chunked SDPA q chunk size (head dim 512)"
    ),
    "GLOBAL_SDPA_K_CHUNK": Setting(
        128, (64, 128, 256), "2026-10-13 C.global.attention: chunked SDPA k chunk size (head dim 512)"
    ),
    "MLP_MATMUL_FIDELITY": Setting(
        "HiFi4", ("HiFi4", "HiFi2", "LoFi"), "2026-10-13 C.sliding.mlp: gate/up/down math fidelity"
    ),
    "MLP_MATMUL_FP32_ACC": Setting(True, (True, False), "2026-10-13 C.sliding.mlp: fp32 accumulation for MLP matmuls"),
    "MLP_GELU_VARIANT": Setting(
        "Tanh", ("Tanh", "Accurate"), "2026-10-13 C.sliding.mlp: ttnn.gelu variant (reference is gelu tanh)"
    ),
    "ROUTER_FIDELITY": Setting(
        "HiFi4", ("HiFi4", "HiFi2", "LoFi"), "2026-10-13 C.sliding.router: rms_norm/linear/softmax math fidelity"
    ),
    "ROUTER_FP32_ACC": Setting(True, (True, False), "2026-10-13 C.sliding.router: fp32 accumulation for router ops"),
    "ROUTER_WEIGHT_DTYPE": Setting(
        "float32",
        ("float32", "bfloat16", "bfloat8_b"),
        "2026-10-13 C.sliding.router: router proj weight (scale folded in); bfp8 fails the selection checks",
    ),
    "EXPERTS_MATMUL_FIDELITY": Setting(
        "HiFi4", ("HiFi4", "HiFi2", "LoFi"), "2026-10-13 C.sliding.experts: expert gate/up/down/selector math fidelity"
    ),
    "EXPERTS_MATMUL_FP32_ACC": Setting(
        True, (True, False), "2026-10-13 C.sliding.experts: fp32 accumulation for expert matmuls"
    ),
    "EXPERTS_SPLIT_TOKENS": Setting(
        True,
        (True, False),
        "2026-10-13 C.sliding.experts: each mesh row runs the experts on its S/8 tokens, then all_gather over the rows",
    ),
}

MAX_PRECISION = {
    "ATTN_MATMUL_FIDELITY": "HiFi4",
    "ATTN_MATMUL_FP32_ACC": True,
    "SDPA_FIDELITY": "HiFi4",
    "SDPA_FP32_ACC": True,
    "MLP_MATMUL_FIDELITY": "HiFi4",
    "MLP_MATMUL_FP32_ACC": True,
    "ROUTER_FIDELITY": "HiFi4",
    "ROUTER_FP32_ACC": True,
    "ROUTER_WEIGHT_DTYPE": "float32",
    "EXPERTS_MATMUL_FIDELITY": "HiFi4",
    "EXPERTS_MATMUL_FP32_ACC": True,
}

settings = Settings("GEMMA4_A4B_GLX_", TABLE, max_precision=MAX_PRECISION)
