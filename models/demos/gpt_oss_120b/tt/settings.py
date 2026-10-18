# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Every switch of gpt_oss_120b in one table. Environment variable = ``GPT_OSS_120B_<NAME>``. Nothing else in tt/ or
bringup/hooks.py reads the environment (the framework's settings lint fails it).

Each precision switch (math fidelity, fp32 accumulation) also names its full-precision value in MAX_PRECISION:
component and swap tests run with it; the shipped defaults are judged end to end by the ladder. Weights and the KV
cache are bfloat8_b from the start (the prefill spec's dataformats), not switches lowered later.
"""

from ._settings_base import Setting, Settings  # noqa: F401  (Setting: the entries agents add to TABLE)

FIDELITIES = ("HiFi4", "HiFi3", "HiFi2", "LoFi")

TABLE = {
    "NORM_FIDELITY": Setting("HiFi4", FIDELITIES, "rule 7 default: rms_norm at HiFi4"),
    "NORM_FP32_ACC": Setting(True, (True, False), "rule 7 default: rms_norm with fp32 dest accumulation"),
    "ATTN_MM_FIDELITY": Setting("HiFi4", FIDELITIES, "rule 7 default: QKV and o_proj linears at HiFi4"),
    "ATTN_MM_FP32_ACC": Setting(True, (True, False), "rule 7 default: QKV and o_proj with fp32 dest accumulation"),
    # fp32 QKV / heads / RoPE, then fp32 -> bfp8 into the cache: the device's bf16 -> bfp8 typecast biases norms +0.3 %
    "QKV_FP32": Setting(True, (True, False), "QKV, heads and RoPE in fp32 before the bfp8 cache cast"),
    # full layers: SDPA attends the chunk's own K / V in bf16 (a per-call bf16 copy of the slot's cache); the served
    # cache stays bfp8. False: SDPA reads the bfp8 cache (rel 0.067 vs 0.043 limit on L1: bfp8 on the chunk's own keys)
    "ATTN_CHUNK_KV_BF16": Setting(True, (True, False), "full-layer SDPA sees the chunk's own K / V in bf16"),
    # full layers: the model keeps a K residual cache lo = bfp8(k - hi) beside the served bfp8 K (hi, the contract's
    # cache), and SDPA's bf16 slot copy has prefix K = hi + lo. bfp8 prefix K alone: rel 0.017, worst row 0.068, ratio
    # [0.975, 1.030] vs the CPU step (S.attn_moe.02 CPU simulation; limits 0.02 / 0.05 / 0.02). ~1 MB per slot-layer
    "ATTN_FULL_K_LO": Setting(True, (True, False), "full layers keep a bfp8 K residual cache for SDPA's prefix K"),
    # sliding layers (ring SDPA takes only bfp8 K / V there): SDPA reads a per-call 128-wide slot copy, head dim
    # [hi | lo] with hi = bfp8(x), lo = bfp8(x - hi), and q duplicated, so q.k and P.v see x to ~2^-14. bfp8 on the
    # chunk's own K was rel 0.0246 vs the CPU step in S.sliding_moe.02 (limit 0.02); the split simulates at 0.003
    "ATTN_SLIDING_KV_SPLIT": Setting(True, (True, False), "sliding-layer SDPA sees the chunk's K / V as bfp8 hi + lo"),
    # bfp8 QKV weights are rel 0.012 of the sliding attention output against the CPU step (S.sliding_moe.02 device
    # probe: the worst row ratio 1.027 > 1.02, block out rel 0.0101 > 0.01); bf16 costs 3.5 MB more per chip per layer
    "ATTN_QKV_WEIGHT_DTYPE": Setting("bfloat16", ("bfloat16", "bfloat8_b"), "fused QKV weight dtype"),
    # rotary_embedding_indexed runs with a 16-bit dest (fp32_dest_acc_en TT_FATALs in program setup) and truncates:
    # q / k rel 0.0044, norms -0.2 % (S.sliding_moe.02 device probe). True: the op only fetches this chunk's cos / sin
    # rows (hi / lo bf16 tables, exact), the rotation is matmuls on bf16 hi / lo halves, the products fp32 eltwise
    "ROPE_FP32": Setting(True, (True, False), "q / k RoPE composed in fp32 instead of rotary_embedding_indexed"),
    "SDPA_FIDELITY": Setting("HiFi4", FIDELITIES, "rule 7 default: ring SDPA at HiFi4"),
    # ring_joint SDPA with attention_sink TT_FATALs with fp32 dest ("requires streaming compute; set
    # fp32_dest_acc_en=false", ring_joint_sdpa_device_operation.cpp): False is the most precise value the op allows.
    "SDPA_FP32_ACC": Setting(False, (False,), "ring SDPA with sinks runs only with fp32 dest off"),
    "SDPA_Q_CHUNK": Setting(128, (64, 128), "ring SDPA sliding supports Q blocks 64 / 128"),
    "SDPA_K_CHUNK": Setting(128, (128,), "ring SDPA sliding supports K blocks 128"),
    # full layers have no K block restriction: fewer, longer K blocks round the bf16 running max / sum less often
    "SDPA_FULL_K_CHUNK": Setting(512, (128, 256, 512), "ring SDPA K block on full-attention layers"),
    "O_PROJ_REDUCE_FP32": Setting(True, (True, False), "o_proj partials summed over TP in fp32, then bf16"),
    "ROUTER_FIDELITY": Setting("HiFi4", FIDELITIES, "rule 7 default: router linear and softmax at HiFi4"),
    "ROUTER_FP32_ACC": Setting(True, (True, False), "router logits in fp32 dest: top-4 ties are precision-bound"),
    # plan.md step 5: the router weight is bf16 (0.74 MB per chip), a small price for exact top-4 selection
    "ROUTER_WEIGHT_DTYPE": Setting("bfloat16", ("bfloat16", "bfloat8_b"), "router weight and bias dtype"),
    "CCL_NUM_LINKS": Setting(2, (1, 2), "fabric links per CCL on Blackhole (gpt_oss_d_p get_default_num_links)"),
    "EXPERTS_FIDELITY": Setting("HiFi4", FIDELITIES, "rule 7 default: expert gate / up / down linears at HiFi4"),
    "EXPERTS_FP32_ACC": Setting(True, (True, False), "rule 7 default: expert linears with fp32 dest accumulation"),
    # gate / up / SwiGLU hidden kept in fp32 before the down projection (bf16 widens the per-token norm ratio)
    "EXPERTS_MID_DTYPE": Setting("float32", ("float32", "bfloat16"), "dtype of the expert gate / up / hidden"),
    # dispatch buffer rows per chip = this x 8 x S/8; top-k (4) holds the worst case (no token is ever dropped)
    "EXPERTS_CAPACITY_FACTOR": Setting(4, (1, 2, 4), "dispatch buffer capacity factor (4 = no token dropped)"),
    "EXPERTS_REDUCE_FP32": Setting(True, (True, False), "the 4 dispatch groups' outputs summed over TP in fp32"),
    # P.1: the chunk is host-dispatch bound (~0.7 ms per ttnn call on the 32-chip mesh). True: gate and up are two
    # linears (bias fused, no slices) and the clamped SwiGLU is one multiply with SFPU activation chains on both
    # operands: 4 ops per local expert between extract and insert instead of 11. False: fused gate_up linear,
    # 2 slices and 7 eltwise ops
    "PERF_P1": Setting(True, (True, False), "experts: split gate / up linears, SwiGLU fused into one multiply"),
    # P.2: attention is host-dispatch bound too (ROPE_FP32's two rotations are 14 ttnn calls per layer). True:
    # rot(x W + b) = x (W R) + b R with R the meta rotation (a signed column permutation, exact in any dtype), so a second
    # QKV linear with the weight W R and one nlp_create_qkv_heads give rot(q), rot(k) in fp32, bit-equal to the columns
    # of q, k: 2 calls instead of 14, for 2.5 MB more bf16 weight per chip per layer. False: TtRope._rotate (hi / lo
    # bf16 halves times R)
    "PERF_P2": Setting(True, (True, False), "attention: RoPE rotation folded into a second QKV linear (W R)"),
    # serving_answers slots:the runner test interleaves two user slots; production ships 16 (1.1 GB per chip)
    "KV_SLOTS": Setting(2, None, "user slots the device model's KV state allocates (serving_answers slots)"),
}

MAX_PRECISION = {
    "NORM_FIDELITY": "HiFi4",
    "NORM_FP32_ACC": True,
    "ATTN_MM_FIDELITY": "HiFi4",
    "ATTN_MM_FP32_ACC": True,
    "QKV_FP32": True,
    "ATTN_CHUNK_KV_BF16": True,
    "ATTN_FULL_K_LO": True,
    "ATTN_SLIDING_KV_SPLIT": True,
    "ATTN_QKV_WEIGHT_DTYPE": "bfloat16",
    "ROPE_FP32": True,
    "SDPA_FIDELITY": "HiFi4",
    "SDPA_FP32_ACC": False,
    "O_PROJ_REDUCE_FP32": True,
    "ROUTER_FIDELITY": "HiFi4",
    "ROUTER_FP32_ACC": True,
    "ROUTER_WEIGHT_DTYPE": "bfloat16",
    "EXPERTS_FIDELITY": "HiFi4",
    "EXPERTS_FP32_ACC": True,
    "EXPERTS_MID_DTYPE": "float32",
    "EXPERTS_REDUCE_FP32": True,
}

settings = Settings("GPT_OSS_120B_", TABLE, max_precision=MAX_PRECISION)
