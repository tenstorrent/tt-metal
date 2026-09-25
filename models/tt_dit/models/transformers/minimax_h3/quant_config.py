# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Opt-in precision knobs for the MiniMax-H3 transformer blocks, read from the environment.

MINIMAX_H3_MM_FIDELITY=LoFi|HiFi2|HiFi3|HiFi4  fidelity of the block matmuls (adaLN table, to_qkv, to_out, ff1, ff2)
MINIMAX_H3_MM_FP32_ACC=0|1                      fp32 destination accumulation for the same matmuls (default 1)
MINIMAX_H3_BF8_WEIGHTS=qkv,ff1[,out,ff2]        typecast the listed linears' weights to bfloat8_b after loading

Unset means the model's measured defaults (bf16 weights, HiFi2, fp32 acc). `out` and `ff2` feed fused
residual/gate kernels whose ternary inputs must match the weight format, so bf8 there may be rejected.
"""

from __future__ import annotations

import os

from loguru import logger

import ttnn


def _typecast_parameter(param, dtype) -> None:
    if param._data is None or param.dtype == dtype:
        return
    param._data = ttnn.typecast(param._data, dtype)
    param.dtype = dtype


def apply_env_quant_config(model) -> None:
    """`model` is the transformer or a single block (the block perf test)."""
    fidelity = os.environ.get("MINIMAX_H3_MM_FIDELITY")
    fp32_acc = os.environ.get("MINIMAX_H3_MM_FP32_ACC")
    bf8 = [name for name in os.environ.get("MINIMAX_H3_BF8_WEIGHTS", "").split(",") if name]
    if not (fidelity or fp32_acc or bf8):
        return

    compute_config = None
    if fidelity or fp32_acc:
        compute_config = ttnn.init_device_compute_kernel_config(
            model.mesh_device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, fidelity or "HiFi2"),
            math_approx_mode=True,
            fp32_dest_acc_en=fp32_acc != "0",
            packer_l1_acc=True,
        )
    logger.info(f"minimax-h3 block precision override: fidelity={fidelity} fp32_acc={fp32_acc} bf8_weights={bf8}")

    for block in getattr(model, "transformer_blocks", [model]):
        if compute_config is not None:
            block.mm_compute_kernel_config = compute_config
            block.attn.mm_compute_kernel_config = compute_config
        linears = {"qkv": block.attn.to_qkv, "out": block.attn.to_out, "ff1": block.ff.ff1, "ff2": block.ff.ff2}
        for name in bf8:
            linear = linears[name]
            _typecast_parameter(linear.weight, ttnn.bfloat8_b)
            if getattr(linear, "bias", None) is not None:
                _typecast_parameter(linear.bias, ttnn.bfloat8_b)
