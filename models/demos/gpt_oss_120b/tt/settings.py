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
    "SDPA_FIDELITY": Setting("HiFi4", FIDELITIES, "rule 7 default: ring SDPA at HiFi4"),
    # ring_joint SDPA with attention_sink TT_FATALs with fp32 dest ("requires streaming compute; set
    # fp32_dest_acc_en=false", ring_joint_sdpa_device_operation.cpp): False is the most precise value the op allows.
    "SDPA_FP32_ACC": Setting(False, (False,), "ring SDPA with sinks runs only with fp32 dest off"),
    "SDPA_Q_CHUNK": Setting(128, (64, 128), "ring SDPA sliding supports Q blocks 64 / 128"),
    "SDPA_K_CHUNK": Setting(128, (128,), "ring SDPA sliding supports K blocks 128"),
    "O_PROJ_REDUCE_FP32": Setting(True, (True, False), "o_proj partials summed over TP in fp32, then bf16"),
    "CCL_NUM_LINKS": Setting(2, (1, 2), "fabric links per CCL on Blackhole (gpt_oss_d_p get_default_num_links)"),
}

MAX_PRECISION = {
    "NORM_FIDELITY": "HiFi4",
    "NORM_FP32_ACC": True,
    "ATTN_MM_FIDELITY": "HiFi4",
    "ATTN_MM_FP32_ACC": True,
    "QKV_FP32": True,
    "SDPA_FIDELITY": "HiFi4",
    "SDPA_FP32_ACC": False,
    "O_PROJ_REDUCE_FP32": True,
}

settings = Settings("GPT_OSS_120B_", TABLE, max_precision=MAX_PRECISION)
