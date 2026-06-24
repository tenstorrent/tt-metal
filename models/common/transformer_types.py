# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class Mode(Enum):
    DECODE = "decode"
    PREFILL = "prefill"


class TensorGroup(Enum):
    FF1_FF3 = "ff1_3"
    FF2 = "ff2"
    WQKV = "wqkv"
    WO = "wo"
    KV_CACHE = "kv_cache"
    ACTIVATION = "activation"


class PrecisionSetting(Enum):
    BFP4 = "bfp4"
    BFP8 = "bfp8"
    BF16 = "bf16"


class OpGroup(Enum):
    """
    LI_* are linear operator groups
    SDPA_* are scaled_dot_product_attention operator groups
    """

    LI_FF1_FF3 = "li_ff1_3"
    LI_FF2 = "li_ff2"
    LI_QKV_DECODE = "li_qkv_decode"
    LI_O_DECODE = "li_o_decode"
    SDPA_DECODE = "sdpa_decode"
    LI_QKV_PREFILL = "li_qkv_prefill"
    LI_O_PREFILL = "li_o_prefill"
    SDPA_PREFILL = "sdpa_prefill"
    ACCURACY = "accuracy"


class MathFidelitySetting(Enum):
    LOFI = "lofi"
    HIFI2 = "hifi2"
    HIFI2_NA = "hifi2na"
    HIFI2_FP16 = "hifi2fp16"
    HIFI2_NOL1ACC = "hifi2nol1acc"
    HIFI4 = "hifi4"
    HIFI4_FP32 = "hifi4fp32"
