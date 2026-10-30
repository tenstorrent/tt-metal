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
        False,
        (True, False),
        "full-attention ring SDPA softmax exp: True = accurate exp through the fork ttnn.bringup."
        "ring_joint_scaled_dot_product_attention(exact_exp=True); False = the source op's Schraudolph approximation "
        "(layer 0: output column rel up to 0.12) (C.attn_dense.attention). Off: the pinned tt-metal has no exact_exp "
        "variant; the e2e test passes with the stock op (run 20261030-132711-mimo-e2e-noexactexp)",
    ),
    "SDPA_SEGMENTED": Setting(
        True,
        (True, False),
        "ring SDPA segmented_accumulation (per-ring-iteration accumulators; needs one Q chunk per core). Off, the "
        "running output / sum drift over a long prefix: layer-0 attention at 51200 -> 56320 median row norm +2.0 % "
        "(ladder s56320 norm bias 0.0055-0.0073 > 0.005); on -0.17 % (L.s56320). Layer-0 gate margin at k 320: on "
        "1.14 with exact_exp (0.989 before it), off 1.003 (C.attn_dense.attention)",
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
    "FULL_WEIGHTS_BF16_FIRST": Setting(
        "all",
        ("none", "qkv", "o", "all"),
        "full layers: which dequantized weights (qkv, o_proj) are rounded to bf16 before the scale fold and bfloat8_b "
        "(C.attn_moe.attention)",
    ),
    "QKV_FP32": Setting(
        True,
        (True, False),
        "full layers: qkv projection out, head split and RoPE in fp32; k / v cast fp32 -> bfloat8_b for the cache, "
        "q to bf16 for the SDPA (bf16 -> bfloat8_b typecast biased +0.28 %: layer 5 column 774 0.078 vs limit 0.066) "
        "(C.attn_moe.attention)",
    ),
    "SWA_QKV_FP32": Setting(
        True,
        (True, False),
        "sliding layers: qkv projection out, head split, RoPE and the value scale in fp32; k / v cast fp32 -> "
        "bfloat8_b for the caches, the composed window softmax reads fp32 q / k / v (bf16: median row norm -1.3 %, "
        "ratio 0.970 in the layer-1 block) (S.sliding_moe.02)",
    ),
    "SWA_ROPE_FP32": Setting(
        True,
        (True, False),
        "sliding layers with SWA_QKV_FP32: RoPE with fp32 cos / sin (bf16 table + bf16 residual table, added), "
        "composed from fp32 eltwise ops instead of the fused rotary_embedding (S.sliding_moe.02)",
    ),
    "SWA_SPLIT_MM": Setting(
        True,
        (True, False),
        "sliding layers: the composed softmax's q k^T and p v matmuls with fp32 operands split into bf16 hi + lo "
        "(3 exact bf16 matmuls, fp32 dest) instead of one matmul on fp32 operands (S.sliding_moe.02)",
    ),
    "FULL_SCALES_AFTER_PROJ": Setting(
        True,
        (True, False),
        "full layers with QKV_FP32: the softmax scale and attention_value_scale multiply the fp32 q / v after the "
        "projection (fp32 tensor operands) instead of being folded into the bfloat8_b qkv weights "
        "(C.attn_moe.attention)",
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
    "ROUTER_FIDELITY": Setting(
        "HiFi4",
        ("HiFi4", "HiFi3", "HiFi2", "LoFi"),
        "rule 7 default: HiFi4 for the fp32-out router gate matmul (C.sliding_moe.router)",
    ),
    "ROUTER_FP32_ACC": Setting(
        True, (True, False), "fp32 dest for the router gate matmul: fp32 logits (C.sliding_moe.router)"
    ),
    "ROUTER_MASK": Setting(
        "scatter",
        ("scatter", "kth"),
        "router selection mask: scatter ones at topk's indices (exactly k per row) or choice >= the k-th value "
        "(selects k+1 on an exact tie) (C.attn_moe.router)",
    ),
    "EXPERTS_FIDELITY": Setting(
        "HiFi4",
        ("HiFi4", "HiFi3", "HiFi2", "LoFi"),
        "rule 7 default: HiFi4 for the routed experts' gate / up / down matmuls (C.sliding_moe.experts)",
    ),
    "EXPERTS_FP32_ACC": Setting(
        True, (True, False), "fp32 dest for the routed experts' matmuls (C.sliding_moe.experts)"
    ),
    "EXPERTS_INTER_FP32": Setting(
        True,
        (True, False),
        "routed experts: gate, up and silu(gate) * up in fp32 instead of bf16 (known issue: bf16 intermediates widen "
        "the per-token norm ratio) (C.sliding_moe.experts)",
    ),
    "EXPERTS_OUT_FP32": Setting(
        True,
        (True, False),
        "routed experts: the 4 dispatch groups' partials summed in fp32 by the TP reduce_scatter, rounded to bf16 "
        "once (C.sliding_moe.experts)",
    ),
    "EXPERTS_BUFFER_FACTOR": Setting(
        0,
        None,
        "dispatch buffer rows per chip = factor x the group's tokens (+ tile alignment); 0 = the worst case "
        "min(top_k, experts per chip): the dispatch kernel drops overflow rows silently (C.sliding_moe.experts)",
    ),
    "EXPERTS_NUM_LINKS": Setting(
        1, (1, 2), "fabric links of the experts' dispatch / combine over SP (C.sliding_moe.experts)"
    ),
    "EXPERTS_TOPOLOGY": Setting(
        "Linear",
        ("Linear", "Ring"),
        "fabric topology of the experts' dispatch / combine over SP (axis 0) (C.sliding_moe.experts)",
    ),
    "RING_NUM_LINKS": Setting(2, (1, 2), "fabric links of the ring SDPA gather over SP (gemma4_d_p: 2 on Blackhole)"),
    "SERVE_HF_DIR": Setting(
        "/mnt/models/huggingface/hub/models--XiaomiMiMo--MiMo-V2.6-Flash-RL/snapshots/"
        "dcd3fd83fb181622fbc606fbb1709901982a62c2",
        None,
        "checkpoint (config.json + safetensors) the prefill adapter serves (tt/runners/adapter.py, K.1)",
    ),
}

MAX_PRECISION = {
    "NORM_FIDELITY": "HiFi4",
    "NORM_FP32_ACC": True,
    "ATTN_FIDELITY": "HiFi4",
    "ATTN_FP32_ACC": True,
    "ATTN_OUT_FP32": True,
    "SDPA_FIDELITY": "HiFi4",
    "SDPA_EXACT_EXP": False,  # the pinned tt-metal has no ttnn.bringup fork
    "SWA_OUT_FP32": True,
    "QKV_FP32": True,
    "SWA_QKV_FP32": True,
    "SWA_SPLIT_MM": True,
    "SWA_ROPE_FP32": True,
    "MLP_FIDELITY": "HiFi4",
    "MLP_FP32_ACC": True,
    "MLP_INTER_FP32": True,
    "MLP_OUT_FP32": True,
    "ROUTER_FIDELITY": "HiFi4",
    "ROUTER_FP32_ACC": True,
    "EXPERTS_FIDELITY": "HiFi4",
    "EXPERTS_FP32_ACC": True,
    "EXPERTS_INTER_FP32": True,
    "EXPERTS_OUT_FP32": True,
}

settings = Settings("MIMO_V26_FLASH_RL_", TABLE, max_precision=MAX_PRECISION)
