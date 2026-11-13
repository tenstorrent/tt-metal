# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

OPS = ("rms_norm_residual",)
SEQ_LENS = (1024,)


def add_norm(residual, attn_out, norm_module, norm_config):
    norm = getattr(norm_module, "norm", None)
    if (
        norm is None
        or len(residual.shape) != 4
        or residual.shape[0] != 1
        or residual.shape[1] != 1
        or residual.shape[-2] not in SEQ_LENS
        or residual.shape[-1] != 4096
        or residual.dtype != ttnn.bfloat16
        or attn_out.dtype not in (ttnn.bfloat16, ttnn.bfloat8_b)
        or residual.memory_config() != ttnn.DRAM_MEMORY_CONFIG
        or attn_out.memory_config() != ttnn.DRAM_MEMORY_CONFIG
        or not ttnn.experimental.llama31_8b.rms_norm_residual_supported(residual, attn_out)
    ):
        return None
    out, h = ttnn.experimental.llama31_8b.rms_norm_residual(
        residual, attn_out, norm.weight, norm.eps, compute_kernel_config=norm.compute_kernel_config_hifi2
    )
    output_mem_config = norm_config.get("output_mem_config") if norm_config else None
    if output_mem_config is not None:
        out = ttnn.to_memory_config(out, output_mem_config)
    return out, h
