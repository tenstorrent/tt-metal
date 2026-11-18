# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Route the models' plain single-chip matmuls to the generated genify_matmul (#65150).

`TT_GENIFY_MATMUL=1` routes every integrated call site to genify_matmul; unset or `0` keeps the
stock TTNN op, so the same model code measures both. CCL-fused matmuls (#65143) are never routed.

genify_matmul(A, B, *, bias, activation, residual, gate, dtype, memory_config,
compute_kernel_config, chunks) computes epilogue(A @ B) in one program:
    activation="relu"    with bias: relu(A @ B + bias)
    activation="swiglu"  B's columns are (gate, up) 32-column tile pairs; output has N / 2 columns
    residual, gate       residual + gate * (A @ B)   (addcmul)
    A = [A_0, A_1, ...]  K segments read as concat(A_i, dim=-1), never materialized
    chunks=c             the output split along its last dim into c equal tensors (a list)
Placement follows the operands: an L1 height- or block-sharded A is read in place and the output
keeps A's sharding; a DRAM width-sharded B selects the DRAM-sharded decode path, whose output is
L1 width-sharded with a spec the op picks. See ttnn/ttnn/operations/genify_matmul/op_design.md.
"""

import os

import ttnn

ENV = "TT_GENIFY_MATMUL"


def enabled() -> bool:
    """True when the integrated call sites should run genify_matmul instead of the stock op."""
    return os.environ.get(ENV, "0").strip().lower() not in ("", "0", "false", "no", "off")


def compute_config(cfg):
    """The stock op's compute kernel config as the ttnn.ComputeConfigDescriptor genify_matmul takes.

    Keeps math fidelity, fp32 DEST accumulation and approx mode. packer_l1_acc has no counterpart:
    genify_matmul accumulates over K in DEST or its own partials buffer. None stays None (the op's
    default, HiFi4 with fp32 DEST accumulation).
    """
    if cfg is None or isinstance(cfg, ttnn.ComputeConfigDescriptor):
        return cfg
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=cfg.math_fidelity,
        math_approx_mode=cfg.math_approx_mode,
        fp32_dest_acc_en=cfg.fp32_dest_acc_en,
        dst_full_sync_en=getattr(cfg, "dst_full_sync_en", False),
    )


def matmul(
    input_tensor,
    weight,
    *,
    bias=None,
    activation=None,
    residual=None,
    gate=None,
    dtype=None,
    memory_config=None,
    compute_kernel_config=None,
    chunks=1,
):
    """genify_matmul with the stock op's argument types. `input_tensor` may be a list of K segments;
    chunks > 1 returns a list of `chunks` tensors."""
    from ttnn.operations.genify_matmul import genify_matmul

    return genify_matmul(
        input_tensor,
        weight,
        bias=bias,
        activation=activation,
        residual=residual,
        gate=gate,
        dtype=dtype,
        memory_config=memory_config,
        compute_kernel_config=compute_config(compute_kernel_config),
        **({"chunks": chunks} if chunks != 1 else {}),
    )
