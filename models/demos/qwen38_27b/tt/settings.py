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
    "GDN_FIDELITY": Setting(
        "HiFi4",
        ("HiFi4", "HiFi2", "LoFi"),
        "ENGINEER rule 7: linear-attention projections and the fold combine at HiFi4",
    ),
    "GDN_FP32_ACC": Setting(True, None, "ENGINEER rule 7: fp32 accumulation by default"),
    "GDN_CHAIN_FIDELITY": Setting(
        "HiFi4",
        ("HiFi4", "HiFi2"),
        "chain_affine_transforms (A @ S + B over the SP rows); the op's own default is HiFi2, rule 7 asks HiFi4",
    ),
    "GDN_DELTA_CHUNK": Setting(
        64,
        (64, 32),
        "chunk_gated_delta_rule chunk. The op decays its fp32 state once per chunk with a (dl*I) @ S matmul whose "
        "operands the FPU reads with a shortened mantissa (about -0.1% per chunk on slow-decay heads). 64 (rank-4 "
        "path, q/k normalised in linear_attention.py) halves the decays: layer-0 post-chunk state vs golden rel 0.016 "
        "(worst head 0.029), gate margin 1.56; 32 (flat path, as qwen36) 0.029 / 0.057, margin 1.21 (C.linear.attention)",
    ),
    "GDN_CONV_CHANNEL_CHUNK": Setting(
        512, (512, 256, 128), "qkv_causal_conv1d_silu channel block (2560 per TP column = 5 blocks), as qwen36 measured"
    ),
    "MLP_FIDELITY": Setting("HiFi4", ("HiFi4", "HiFi2", "LoFi"), "ENGINEER rule 7: gate/up/down matmuls at HiFi4"),
    "MLP_FP32_ACC": Setting(True, None, "ENGINEER rule 7: fp32 accumulation by default"),
    "ATTN_FIDELITY": Setting(
        "HiFi4", ("HiFi4", "HiFi2", "LoFi"), "ENGINEER rule 7: full-attention projections, q/k norms and RoPE at HiFi4"
    ),
    "ATTN_FP32_ACC": Setting(True, None, "ENGINEER rule 7: fp32 accumulation by default"),
    "ATTN_SDPA_FIDELITY": Setting("HiFi4", ("HiFi4", "HiFi2", "LoFi"), "ring SDPA matmuls (QK^T, PV)"),
    "ATTN_SDPA_FP32_ACC": Setting(
        True,
        None,
        "ring SDPA fp32 DEST (the op's legacy compute path). False selects its streaming kernel (faster, bf16 DEST; "
        "the only path that takes kv_actual_isl, i.e. a chunk start that is not a multiple of the chunk)",
    ),
    "ATTN_SDPA_Q_CHUNK": Setting(
        64,
        (64, 128, 256),
        "ring SDPA q chunk (rows of the 640-row local Q slab). At fp32 DEST and head_dim 256, q128/k128 does not "
        "build: the kernels' text is 73.9 KB of a 70.6 KB kernel-config buffer (C.attn.attention); q64/k64 does",
    ),
    "ATTN_SDPA_K_CHUNK": Setting(64, (64, 128, 256), "ring SDPA k chunk; see ATTN_SDPA_Q_CHUNK"),
    "ATTN_RING_LINKS": Setting(1, (1, 2), "fabric links of the ring SDPA's KV all-gather over axis 0"),
    "HF_DIR": Setting(
        "/mnt/models/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        None,
        "the checkpoint the prefill runner's adapter loads (config and weights): the spec's hf dir",
    ),
    "KV_SLOTS": Setting(
        2, (1, 2), "request slots of the served state (serving_answers slots: 2); KV user-major, slot * n_attn + layer"
    ),
}

MAX_PRECISION = {
    "MLP_FIDELITY": "HiFi4",
    "MLP_FP32_ACC": True,
    "NORM_FIDELITY": "HiFi4",
    "NORM_FP32_ACC": True,
    "NORM_WEIGHT_DTYPE": "float32",
    "GDN_FIDELITY": "HiFi4",
    "GDN_FP32_ACC": True,
    "GDN_CHAIN_FIDELITY": "HiFi4",
    "GDN_DELTA_CHUNK": 64,
    "ATTN_FIDELITY": "HiFi4",
    "ATTN_FP32_ACC": True,
    "ATTN_SDPA_FIDELITY": "HiFi4",
    "ATTN_SDPA_FP32_ACC": True,
}

settings = Settings("QWEN38_27B_", TABLE, max_precision=MAX_PRECISION)
