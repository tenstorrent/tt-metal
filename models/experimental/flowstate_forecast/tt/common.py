# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Precision policy per role (skill §2, CONTRACT v1.1 §1).

accuracy (the gated target):
  embedding       fp32 tokens, fp32 weights, HiFi4, fp32 accumulation; bf16 out (the residual stream)
  B projection    bf16 u, fp32 Bbar weights (per s, the reference's complex64 expression cast once), HiFi4,
                  fp32 accumulation, fp32 out [Bu_re | Bu_im]
  LTI scan        route K, fp32 state / Bu / Lbar (B0, D0 amendment 1: every class)
  C projection    fp32 [x_re | x_im] in, fp32 [C_r | -C_i] weights, HiFi4, fp32 accumulation (cancellation, §7)
  gate, MLP       bf16 weights and activations, HiFi2, fp32 accumulation
  norms           HiFi4, fp32 accumulation
  decoder         folded with the Legendre basis in float64 on the host, fp32 weights, HiFi4
performance: the same, with bfp8 gate / MLP weights and LoFi there (measured per role in round 1+).
The round-0 settings are a starting point; every change is measured per role (skill §2).
"""

from types import SimpleNamespace

import ttnn


def _cfg(device, fidelity, fp32_acc=True):
    return ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=fp32_acc, packer_l1_acc=True
    )


def policy(device, precision="accuracy"):
    hifi4 = _cfg(device, ttnn.MathFidelity.HiFi4)
    hifi2 = _cfg(device, ttnn.MathFidelity.HiFi2)
    lofi = _cfg(device, ttnn.MathFidelity.LoFi)
    perf = precision == "performance"
    return SimpleNamespace(
        name=precision,
        embed=hifi4,
        b_proj=hifi4,
        c_proj=hifi4,
        dense=lofi if perf else hifi2,
        norm=hifi4,
        decoder=hifi4,
        b_weight_dtype=ttnn.float32,
        c_weight_dtype=ttnn.float32,
        c_input_dtype=ttnn.float32,
        dense_weight_dtype=ttnn.bfloat8_b if perf else ttnn.bfloat16,
        decoder_weight_dtype=ttnn.float32,
        mem=ttnn.DRAM_MEMORY_CONFIG,
    )
