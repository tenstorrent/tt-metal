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
    "ATTN_FIDELITY": Setting(
        "HiFi4",
        ("HiFi4", "HiFi3", "HiFi2", "LoFi"),
        "rule 7 default: HiFi4 for the qkv and o_proj matmuls (C.attn_dense.attention)",
    ),
    "ATTN_FP32_ACC": Setting(True, (True, False), "fp32 dest for the qkv and o_proj matmuls (C.attn_dense.attention)"),
    "ATTN_OUT_FP32": Setting(
        True,
        (True, False),
        "o_proj partials in fp32 through the TP reduce_scatter, rounded to bf16 once after the sum "
        "(C.attn_dense.attention)",
    ),
    # Ring SDPA has no fp32-dest switch: with the prefix length in a device tensor (kv_actual_isl_tensor, the
    # served path) the op requires its streaming compute, i.e. fp32_dest_acc_en=False (TT_FATAL otherwise).
    "SDPA_FIDELITY": Setting(
        "HiFi4", ("HiFi4", "HiFi3", "HiFi2", "LoFi"), "full-attention ring SDPA fidelity (C.attn_dense.attention)"
    ),
    "SDPA_Q_CHUNK": Setting(128, None, "full-attention ring SDPA q chunk (640-row slab: 5 chunks x 16 heads)"),
    "SDPA_K_CHUNK": Setting(
        320,
        None,
        "full-attention ring SDPA k chunk (divides the 640-row slab and 7040 cache rows); layer-0 gate margin: 128 "
        "0.88, 160 0.87, 320 0.99, 640 0.94 (C.attn_dense.attention)",
    ),
    "SDPA_EXACT_EXP": Setting(
        True,
        (True, False),
        "full-attention ring SDPA softmax exp: True = accurate exp through the fork ttnn.bringup."
        "ring_joint_scaled_dot_product_attention(exact_exp=True); False = the source op's Schraudolph approximation "
        "(layer 0: output column rel up to 0.12) (C.attn_dense.attention)",
    ),
    "SDPA_SEGMENTED": Setting(
        False,
        (True, False),
        "ring SDPA segmented_accumulation (per-ring-iteration accumulators; needs one Q chunk per core). Layer-0 gate "
        "margin at k 320: on 0.989, off 1.003, so off (C.attn_dense.attention)",
    ),
    "SWA_SDPA_Q_CHUNK": Setting(
        128,
        (64, 128),
        "sliding-window ring SDPA q chunk (the op's chunked sliding path takes 64 or 128; k chunk is fixed at 128) "
        "(C.sliding_moe.attention)",
    ),
    "SWA_WEIGHTS_BF16_FIRST": Setting(
        True,
        (True, False),
        "sliding layers: qkv / o_proj weights rounded fp32 -> bf16 before bfloat8_b (C.sliding_moe.attention)",
    ),
    "SWA_OUT_FP32": Setting(
        True,
        (True, False),
        "sliding layers: the composed window softmax output stays fp32 through the head concat into o_proj "
        "(C.sliding_moe.attention)",
    ),
    "MLP_FIDELITY": Setting(
        "HiFi4",
        ("HiFi4", "HiFi3", "HiFi2", "LoFi"),
        "rule 7 default: HiFi4 for the dense MLP matmuls (C.attn_dense.mlp)",
    ),
    "MLP_FP32_ACC": Setting(True, (True, False), "fp32 dest for the dense MLP matmuls (C.attn_dense.mlp)"),
    "MLP_INTER_FP32": Setting(
        True, (True, False), "gate, up and silu(gate) * up in fp32 instead of bf16 (C.attn_dense.mlp)"
    ),
    "MLP_OUT_FP32": Setting(
        True,
        (True, False),
        "down_proj partials in fp32 through the TP reduce_scatter, rounded to bf16 once (C.attn_dense.mlp)",
    ),
    "RING_NUM_LINKS": Setting(2, (1, 2), "fabric links of the ring SDPA gather over SP (gemma4_d_p: 2 on Blackhole)"),
}

MAX_PRECISION = {
    "NORM_FIDELITY": "HiFi4",
    "NORM_FP32_ACC": True,
    "ATTN_FIDELITY": "HiFi4",
    "ATTN_FP32_ACC": True,
    "ATTN_OUT_FP32": True,
    "SDPA_FIDELITY": "HiFi4",
    "SDPA_EXACT_EXP": True,
    "SWA_OUT_FP32": True,
    "MLP_FIDELITY": "HiFi4",
    "MLP_FP32_ACC": True,
    "MLP_INTER_FP32": True,
    "MLP_OUT_FP32": True,
}

settings = Settings("MIMO_V26_FLASH_RL_", TABLE, max_precision=MAX_PRECISION)
