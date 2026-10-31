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
    "GDN_FIDELITY": Setting(
        "HiFi4", ("HiFi4", "HiFi3", "HiFi2", "LoFi"), "rule 7 default: Gated DeltaNet matmuls / norm at HiFi4"
    ),
    "GDN_FP32_ACC": Setting(True, (True, False), "rule 7 default: Gated DeltaNet matmuls / norm accumulate in fp32"),
    "GDN_CHUNK": Setting(
        64,
        (64, 32),
        "chunk_gated_delta_rule chunk size. 64: rank-4 path (q / k l2-normalised in fp32 by the model); half the "
        "inter-chunk state updates of 32, about half its error (layer-0 state + layer-3 inputs: o rel 0.028 vs 0.060). "
        "32: the rank-3 flat path with in-kernel l2 norm (faster)",
    ),
    "MLP_FIDELITY": Setting("HiFi4", ("HiFi4", "HiFi3", "HiFi2", "LoFi"), "rule 7 default: MLP matmuls at HiFi4"),
    "ATTN_FIDELITY": Setting(
        "HiFi4", ("HiFi4", "HiFi3", "HiFi2", "LoFi"), "rule 7 default: attention projections / q-k norms at HiFi4"
    ),
    "ATTN_FP32_ACC": Setting(True, (True, False), "rule 7 default: attention projections / norms accumulate in fp32"),
    "ATTN_QKV_HILO": Setting(
        True,
        (True, False),
        "q/k/v/gate projection as bfp8(W) + bfp8(W - bfp8(W)) (two bfp8 weights): with one bfp8 weight, layer 3's "
        "outlier input channel (|x| 92) shrinks the attention output row norms by 0.5 % on average (row ratio 0.980 "
        "vs golden, limit 0.982)",
    ),
    "SDPA_FIDELITY": Setting(
        "HiFi4",
        ("HiFi4", "HiFi3", "HiFi2", "LoFi"),
        "ring_joint SDPA fidelity; its fp32 accumulation stays off (kv_actual_isl runs only on the streaming kernel)",
    ),
    "SDPA_Q_CHUNK": Setting(128, (64, 128, 256, 320, 640), "ring_joint SDPA q chunk (divides the 640 local rows)"),
    "SDPA_K_CHUNK": Setting(128, (64, 128, 256, 512), "ring_joint SDPA k chunk (head_dim 256 caps it by L1)"),
    "SDPA_SEGMENTED": Setting(
        True,
        (True, False),
        "ring_joint segmented accumulation: per-ring-step accumulators instead of bf16 running sums over the whole "
        "prefix (needs one Q chunk per core)",
    ),
    "SDPA_NUM_LINKS": Setting(2, (1, 2), "fabric links of the ring SDPA K/V exchange (Blackhole: 2)"),
    "SDPA_TOPOLOGY": Setting("Linear", ("Linear", "Ring"), "ring SDPA CCL topology (minimax_m3 / DeepSeek: Linear)"),
    "MLP_FP32_ACC": Setting(True, (True, False), "rule 7 default: MLP matmuls accumulate in fp32"),
    "NUM_SLOTS": Setting(2, None, "users per prefill mesh (serving_answers slots = engine max_slots): KV / GDN state"),
    "KV_BLOCK_SIZE": Setting(
        64,
        (64, 32, 128),
        "the deployed kv_block_size: a follow-up turn restarts at align_down(prev_end, this); the GDN layers keep a "
        "state snapshot there",
    ),
}

MAX_PRECISION = {
    "NORM_FIDELITY": "HiFi4",
    "NORM_FP32_ACC": True,
    "NORM_WEIGHT_DTYPE": "float32",
    "GDN_FIDELITY": "HiFi4",
    "GDN_FP32_ACC": True,
    "GDN_CHUNK": 64,
    "MLP_FIDELITY": "HiFi4",
    "MLP_FP32_ACC": True,
    "ATTN_FIDELITY": "HiFi4",
    "ATTN_FP32_ACC": True,
    "ATTN_QKV_HILO": True,
    "SDPA_FIDELITY": "HiFi4",
    "SDPA_SEGMENTED": True,
}

settings = Settings("QWEN38_27B_", TABLE, max_precision=MAX_PRECISION)
