# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Text-tower projections: measured program configs (E5a `mm_sweep`), precision from the policy.

Weight dtype, compute fidelity and fp32 accumulation per weight group come from the precision
policy (`tt/precision.py`, `doc/datatype_sweep/`): text modules pass a `MatmulPolicy` as `policy=`.
Without one (vision tower callers) the call uses the E5a default: HiFi2, fp32 accumulation
(`EG2_MM_FP32`), bf16 output.

Program configs (doc/optimized_text/README.md, "Matmul search"):
    wide roles   (N >= 768 or K == 512: QKV, MLP gate/up, PLE, head) 2D multicast on the
                 full grid, in0_block_w 8 (2x faster than the default config at M = 512).
    deep roles   (K >= 1024 and N == 512: attention out, MLP down) at M >= 2048:
                 `ttnn.experimental.minimal_matmul` (fp32 accumulate); below that the 2D config.

`linear(x, w)` takes `x` [B, 1, S, K] (S a multiple of 32) and folds the batch into M, so one
program config covers every batch size. The program config is derived from the shapes only and
is cached, so the call adds no device op.
"""

from __future__ import annotations

import os
from functools import lru_cache

import ttnn
from models.demos.blackhole.embedding_gemma2.tt.config import compute_grid

# Default (no policy) fp32 dest accumulation for the 2D-multicast roles; minimal_matmul without a
# policy always accumulates in fp32. Text calls take both from their MatmulPolicy.
FP32_ACC = os.environ.get("EG2_MM_FP32", "1") == "1"


def compute_kernel_config_matmul(fp32_acc: bool = False):
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=True,
    )


_CKC = {False: None, True: None}


def _ckc(fp32_acc: bool):
    if _CKC[fp32_acc] is None:
        _CKC[fp32_acc] = compute_kernel_config_matmul(fp32_acc)
    return _CKC[fp32_acc]


def _activation(name: str | None):
    if name is None:
        return None
    if name == "gelu_tanh":
        return ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH)
    raise ValueError(f"unsupported fused activation {name}")


def _largest_divisor(n: int, limit: int) -> int:
    return max(d for d in range(1, limit + 1) if n % d == 0)


# Static circular-buffer budget per core for the 2D config (Blackhole L1 is 1.5 MB; SDPA/other
# program buffers and the allocator need the rest). Above it `linear` uses minimal_matmul.
_CB_BUDGET_BYTES = 900 * 1024
_TILE_BYTES = {"bf16": 2048, "bfp8": 1088}


def _cb_bytes(per_core_M: int, per_core_N: int, in0_block_w: int, fp32_acc: bool) -> int:
    in0 = 2 * per_core_M * in0_block_w * _TILE_BYTES["bf16"]
    in1 = 2 * in0_block_w * per_core_N * _TILE_BYTES["bfp8"]
    out = per_core_M * per_core_N * _TILE_BYTES["bf16"]
    interm = per_core_M * per_core_N * 4096 if fp32_acc else 0
    return in0 + in1 + out + interm


@lru_cache(maxsize=None)
def mcast2d_config(M: int, K: int, N: int, gx: int, gy: int, activation: str | None, fp32_acc: bool = False):
    """2D multicast config, or None when no in0_block_w keeps the CBs inside the L1 budget."""
    Mt, Kt, Nt = M // 32, K // 32, N // 32
    gy = min(gy, Mt)
    per_core_M = -(-Mt // gy)
    per_core_N = -(-Nt // gx)
    in0_block_w = _largest_divisor(Kt, 8)
    while _cb_bytes(per_core_M, per_core_N, in0_block_w, fp32_acc) > _CB_BUDGET_BYTES:
        if in0_block_w == 1:
            return None
        in0_block_w = _largest_divisor(Kt, in0_block_w - 1)
    max_sb = 4 if fp32_acc else 8
    sbw = _largest_divisor(per_core_N, max_sb)
    sbh = max(d for d in range(1, max_sb + 1) if per_core_M % d == 0 and d * sbw <= max_sb)
    act = _activation(activation)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=in0_block_w,
        out_subblock_h=sbh,
        out_subblock_w=sbw,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=act,
    )


# Where gelu_tanh runs for gelu(x @ w_gate) * other: as an SFPU pre-activation of the multiply's
# first input up to this many tokens (faster at 1x128 / 1x512 by 0.05-0.1 ms), in the matmul
# packer above it (faster at 1x2048 / 8x512 by 0.15-0.4 ms). E5a c5 A/B, burst.
GELU_IN_MUL_MAX_TOKENS = int(os.environ.get("EG2_GELU_IN_MUL_MAX_TOKENS", "1024"))


def gated_product(
    x: ttnn.Tensor,
    w_gate: ttnn.Tensor,
    other: ttnn.Tensor,
    *,
    memory_config,
    policy=None,
    dtype=ttnn.bfloat16,
) -> ttnn.Tensor:
    """gelu_tanh(x @ w_gate) * other, x [B, 1, S, K]. `policy` / `dtype` as in `linear`."""
    if x.shape[0] * x.shape[2] <= GELU_IN_MUL_MAX_TOKENS:
        gate = linear(x, w_gate, memory_config=memory_config, policy=policy, dtype=dtype)
        out = ttnn.multiply(
            gate,
            other,
            memory_config=memory_config,
            input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH)],
        )
    else:
        gate = linear(x, w_gate, activation="gelu_tanh", memory_config=memory_config, policy=policy, dtype=dtype)
        out = ttnn.multiply(gate, other, memory_config=memory_config)
    ttnn.deallocate(gate)
    return out


def linear(
    x: ttnn.Tensor,
    w: ttnn.Tensor,
    *,
    activation: str | None = None,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    policy=None,
    dtype=ttnn.bfloat16,
) -> ttnn.Tensor:
    """x [B, 1, S, K] @ w [1, 1, K, N] -> [B, 1, S, N] `dtype` TILE (optionally gelu_tanh fused).

    `policy` (a `precision.MatmulPolicy`) sets the compute fidelity and fp32 accumulation; the
    weight dtype is the one `w` was built with (`policy.dtype` at construction). None: HiFi2,
    fp32 accumulation `FP32_ACC` (2D config) / always (minimal_matmul)."""
    if policy is None:
        fp32_acc, ckc_2d, ckc_minimal = FP32_ACC, _ckc(FP32_ACC), _ckc(True)
    else:
        fp32_acc = policy.fp32_acc
        ckc_2d = ckc_minimal = policy.compute_kernel_config
    shape = x.shape
    B, S, K = shape[0], shape[-2], shape[-1]
    N = w.shape[-1]
    M = B * shape[1] * S
    x2 = ttnn.reshape(x, (1, 1, M, K)) if B * shape[1] != 1 else x
    grid = compute_grid(x.device())
    pc = None
    # minimal_matmul wins for narrow outputs at large M and for everything at M >= 8192 (E5a
    # mm_grid sweep); the 2D config wins elsewhere.
    # minimal_matmul's fused activation is slow (PLE gate + gelu at M=4096: 218 us vs 40 us without).
    if activation is not None or not ((N <= 768 and M >= 2048) or M >= 8192):
        pc = mcast2d_config(M, K, N, grid.x, grid.y, activation, fp32_acc)
    if pc is None:
        out = ttnn.experimental.minimal_matmul(
            x2,
            w,
            fused_activation=_activation(activation),
            compute_kernel_config=ckc_minimal,
            memory_config=memory_config,
            dtype=dtype,
        )
    else:
        out = ttnn.linear(
            x2,
            w,
            program_config=pc,
            compute_kernel_config=ckc_2d,
            memory_config=memory_config,
            dtype=dtype,
        )
    if x2 is not x:
        out = ttnn.reshape(out, (B, shape[1], S, N))
    return out
