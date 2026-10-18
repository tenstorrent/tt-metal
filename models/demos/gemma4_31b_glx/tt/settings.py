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
    "ATTN_PROJ_FIDELITY": Setting(
        "HiFi4",
        ("HiFi4", "HiFi3", "HiFi2", "LoFi"),
        "QKV and o_proj matmuls. gemma4_d_p runs them at LoFi from 512 rows per device (a perf choice for later).",
    ),
    "ATTN_PROJ_FP32_ACC": Setting(True, None, "fp32 accumulation for the QKV and o_proj matmuls."),
    "ATTN_SDPA_FIDELITY": Setting(
        "HiFi4",
        ("HiFi4", "HiFi3", "HiFi2", "LoFi"),
        "Ring SDPA (QK^T and PV) fidelity. gemma4_d_p uses HiFi2 with bf16 dest (a perf choice for later).",
    ),
    "ATTN_SDPA_FP32_ACC": Setting(
        False,
        (False,),
        "fp32 dest accumulation in the ring SDPA. The served path (kv_actual_isl) needs the op's streaming compute, "
        "which ring_joint_sdpa_program_factory.cpp only selects with fp32_dest_acc_en=false (TT_FATAL otherwise), so "
        "False is also the maximum precision. Softmax statistics are kept by the op as for gemma4_d_p.",
    ),
    "ATTN_HEAD_NORM": Setting(
        "fp32",
        ("fp32", "rms_norm"),
        "q/k/v per-head RMS norm. fp32: x * rsqrt(mean(x^2) + eps) * w as fp32 eltwise/reduce ops (matches the "
        "reference to bf16 input rounding, no norm bias); rms_norm: ttnn.rms_norm, measured -0.08% median row-norm "
        "bias on q at head_dim 256.",
    ),
    "ATTN_KV_FP32_WRITE": Setting(
        True,
        None,
        "K and V stay fp32 from the head norm / RoPE to the bfp8 cache write. The device's bf16 -> bfp8 conversion "
        "(typecast, or any op packing bfp8 from bf16) raises row norms by +0.28 % (+0.54 % through an op's packer); "
        "fp32 -> bfp8 typecast is unbiased. False: bf16 K/V, attention output median norm +0.5 % (fails the bias "
        "limit).",
    ),
    "ATTN_ROPE": Setting(
        "elementwise",
        ("elementwise", "llama"),
        "q/k RoPE. elementwise: x * cos + (x @ R) * sin in fp32 (bf16 tables; R the exact +-1 pair "
        "rotation). llama: rotary_embedding_llama (gemma4_d_p); at head_dim 256 it must run with bf16 dest and raises "
        "the q/k row norms by ~0.2%, which sharpens the softmax and pushes the attention output's median norm +0.5% "
        "(over the component bias limit).",
    ),
    "ATTN_SDPA_Q_CHUNK": Setting(128, (64, 128), "Ring SDPA q chunk (the chunked sliding path accepts 64 or 128)."),
    "ATTN_SDPA_K_CHUNK": Setting(128, (128,), "Ring SDPA k chunk (the chunked sliding path accepts only 128)."),
    "ATTN_SDPA_K_SPLITS": Setting(
        "auto",
        ("auto", "1", "3"),
        "Ring SDPA max_k_splits. auto: 3 when the cores hold three bands of every (head, Q chunk) unit (gemma4_d_p).",
    ),
    "ATTN_GLOBAL_Q_CHUNK": Setting(
        96,
        (32, 64, 96, 128),
        "Global ring_mla q chunk (head_dim 512, packed 640 K). gemma4_d_p: q96 / k256 at a 640-row slab; q160 overflows "
        "L1 beside k256.",
    ),
    "ATTN_GLOBAL_K_CHUNK": Setting(256, (128, 256), "Global ring_mla k chunk (gemma4_d_p: 256)."),
    "ATTN_GLOBAL_SEGMENTED": Setting(
        True,
        None,
        "Global ring_mla segmented accumulation when every (head, Q chunk) unit has its own core (gemma4_d_p).",
    ),
    "ATTN_GLOBAL_W_LAYOUT": Setting(
        "out_in",
        ("out_in", "in_out"),
        "Global QK and o_proj bfp8 storage. out_in: the checkpoint's [out, in], matmul transpose_b, bfp8 blocks along "
        "the input (as the framework's precision model). in_out: W^T, blocks along the output channels; a large o_proj "
        "channel flushes its 15 neighbours (layer 5 column 2294: rel 0.12 vs limit 0.038).",
    ),
    "ATTN_GLOBAL_SDPA": Setting(
        "composed",
        ("composed", "ring_mla"),
        "Global attention over the packed cache. composed: all_gather the slot's rows over SP, QK^T (fp32 out), causal "
        "mask on the current chunk, fp32 softmax, P V (fp32 out), all HiFi4 + fp32 acc. ring_mla: the fused ring op, "
        "whose streaming path (the only one with kv_actual_isl) runs bf16 DEST with bf16 O / sum / max state; at head "
        "dim 512 that puts layer 5's worst output column at 1.4x its component limit (composed: within it).",
    ),
    "ATTN_GLOBAL_FP32_REDUCE": Setting(
        True,
        None,
        "Global o_proj partials fp32 through the TP reduce_scatter, one bf16 rounding at the end; False: bf16 partials "
        "(rounded per chip, then summed in bf16).",
    ),
    "ATTN_GLOBAL_FP32_QK": Setting(
        True,
        None,
        "Global tied QK projection output in fp32 (into the fp32 head norms); False: bf16.",
    ),
    "ATTN_RING_NUM_LINKS": Setting(2, (1, 2), "Ethernet links per direction for the ring SDPA's CP exchange (BH: 2)."),
    "KV_SLOTS": Setting(2, None, "User slots in the served KV cache (serving_answers.yaml)."),
    "MLP_FIDELITY": Setting(
        "HiFi4",
        ("HiFi4", "HiFi3", "HiFi2", "LoFi"),
        "gate / up / down matmuls. gemma4_d_p runs them at LoFi with fp32 acc (a perf choice for later).",
    ),
    "MLP_FP32_ACC": Setting(True, None, "fp32 accumulation for the MLP matmuls."),
    "MLP_FP32_INTERMEDIATE": Setting(
        False,
        None,
        "gate (after gelu_tanh) and up outputs in fp32 before gate * up; False: bf16 as in gemma4_d_p.",
    ),
}

MAX_PRECISION = {
    "NORM_FIDELITY": "HiFi4",
    "NORM_FP32_ACC": True,
    "ATTN_PROJ_FIDELITY": "HiFi4",
    "ATTN_PROJ_FP32_ACC": True,
    "ATTN_SDPA_FIDELITY": "HiFi4",
    "ATTN_SDPA_FP32_ACC": False,  # the op's ceiling on the served path (see the setting)
    "ATTN_GLOBAL_SDPA": "composed",
    "ATTN_GLOBAL_FP32_REDUCE": True,
    "ATTN_GLOBAL_FP32_QK": True,
    "MLP_FIDELITY": "HiFi4",
    "MLP_FP32_ACC": True,
    "MLP_FP32_INTERMEDIATE": True,
}

settings = Settings("GEMMA4_31B_GLX_", TABLE, max_precision=MAX_PRECISION)
