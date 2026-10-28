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

import torch
import ttnn


def _cfg(device, fidelity, fp32_acc=True):
    return ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=fp32_acc, packer_l1_acc=True
    )


# R1-1 (sweep job 1156, paper): in0_block_w per role and precision; out_subblock 1x4, out_block_h <= 5
IN0_BLOCK_W = {
    "accuracy": {"b_proj": 2, "c_proj": 2, "gate": 4, "w1": 2, "w2": 2, "w3": 8},
    "performance": {"b_proj": 2, "c_proj": 2, "gate": 4, "w1": 4, "w2": 4, "w3": 8},
}


ACTIVATIONS = {"silu": ttnn.UnaryOpType.SILU, "sigmoid": ttnn.UnaryOpType.SIGMOID}


def swiglu_interleave(gate, up, tile=32):
    """[K, N] gate and up (matmul layout, or [1, N] biases) -> the tile-pair-interleaved [K, 2N] operand of
    minimal_matmul(fuse_swiglu=True): column tile 2p = gate tile p, 2p + 1 = up tile p."""
    K, N = gate.shape
    return torch.stack((gate.reshape(K, N // tile, tile), up.reshape(K, N // tile, tile)), 2).reshape(K, 2 * N)


# R2-1 (sweep job 1218, paper; accuracy: auto within 0.3 %): the fused SwiGLU's minimal_matmul blocks per precision; None = the op's auto config
SWIGLU_BLOCKS = {"accuracy": None, "performance": (8, 8, 8, 1, 4)}


def swiglu_config(device, precision):
    b = SWIGLU_BLOCKS[precision]
    if b is None:
        return None
    grid = device.compute_with_storage_grid_size()
    return ttnn.MinimalMatmulConfig(
        M_block_size=b[0], K_block_size=b[1], N_block_size=b[2], subblock_h=b[3], subblock_w=b[4],
        compute_with_storage_grid_size=(grid.x, grid.y),
    )


def mm_config(device, role, precision, M, K, N, act=None):
    """The 1D (in1-mcast) program config of a (M, K) x (K, N) matmul over the whole grid, rows split across cores.
    None (the auto config) when N is not a multiple of 4 tiles or the rows do not fill the grid."""
    grid = device.compute_with_storage_grid_size()
    cores = grid.x * grid.y
    Mt, Kt, Nt = -(-M // 32), K // 32, N // 32
    if Nt % 4 or Mt < cores:
        return None
    per_core_M = -(-Mt // cores)
    kw = IN0_BLOCK_W[precision][role]
    while Kt % kw:
        kw //= 2
    bh = next(b for b in (5, 4, 3, 2, 1) if per_core_M % b == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=kw,
        out_subblock_h=1,
        out_subblock_w=4,
        out_block_h=bh,
        out_block_w=Nt,
        per_core_M=per_core_M,
        per_core_N=Nt,
        fuse_batch=True,
        fused_activation=ttnn.UnaryWithParam(ACTIVATIONS[act]) if act else None,
        mcast_in0=False,
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
        program_configs=True,  # R1-1
        fused_swiglu=perf,  # R2-1 (accuracy: unfused, suite drift job 1238)
    )
