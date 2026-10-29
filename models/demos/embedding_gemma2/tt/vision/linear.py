# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Vision-tower projections: the text helper `tt.linear` with a vision rule for narrow outputs.

`tt.linear.linear` routes narrow outputs (N <= 768) at M >= 2048 to `minimal_matmul`, which won
for the text shapes. For the vision shapes at M = 2560 (one image) the 2D multicast config is 2x
faster on the narrow roles (E5b `mm_sweep.py`, bf16 weights, HiFi2 + fp32 acc, L1 input):

    o_proj 768 -> 768     2D 13x10 in0_block_w 6: 24 us    minimal_matmul: 48 us (Tracy)
    down   3072 -> 768    2D 13x10 in0_block_w 8: 64 us    minimal_matmul: 108 us (Tracy)

At M = 10240 (4 images, 8 video frames) minimal_matmul wins every role (o 148 vs 222 us, down 334
vs 840 us) and the 2D config often exceeds L1, so those shapes keep the text rule.

Precision (E6b): weight dtype, compute fidelity, fp32 accumulation and output dtype per weight group
come from the vision section of the precision policy (`tt/precision.py`, doc/datatype_sweep/): the
modules pass a `MatmulPolicy` as `policy=`. Without one the call uses the E5b policy (HiFi2, fp32
accumulation `EG2_MM_FP32`, bf16 output).
"""

from __future__ import annotations

import os

import ttnn
from models.demos.embedding_gemma2.tt.linear import (
    FP32_ACC,
    MMSpec,
    _linear_spec,
    compute_kernel_config_matmul,
    linear,
    mcast2d_config,
)

# M (= batch x patches) up to which every vision projection uses the 2D multicast config
MCAST2D_MAX_M = 4096
# ... and inner dim up to this (down 3072 -> 768 with its DRAM input: 2D 213 us vs minimal 107 us, Tracy)
MCAST2D_MAX_K = 1024
_CKC = []

# Tuned configs per compute grid (E10.1, doc/prefill_towers/): VISION_TUNED[grid][(role, M)] -> MMSpec
# (tt/linear.py MMSpec / _linear_spec) replaces the rule below for that role and M on that grid only, at
# batch 1. Roles: "qkv", "o", "up", "gate", "down" (vision_linear `role=`). Grids without an entry (the
# P150a's 13 x 10) keep the rule unchanged. EG2_VISION_MM_TUNED=0 disables the table (same-session
# control); VISION_MM_OVERRIDES[(role, M)] (in-process, empty by default) takes precedence on every grid
# (the in-model config search sets it).
#   pool (288 x 2560 x 768, the pooler's one-hot matmul): 2D on 12 x 9 (was ttnn's pick, 6 cores, 279 us)
#   down (2560 x 3072 x 768): 2D, in0_block_w 12, with its input h in L1 (config.VISION_L1_WIDE_TUNED "h");
#       was minimal_matmul with h in DRAM (102 us)
# Search: tt-ops e10/search/vision_v{1,2,4}.jsonl; together -6.2 % burst at 1 x 2560 (doc/prefill_towers/).
VISION_TUNED: dict[tuple[int, int], dict[tuple[str, int], MMSpec]] = {
    (12, 10): {
        ("pool", 288): MMSpec("2d", grid=(12, 9)),
        ("down", 2560): MMSpec("2d", in0_block_w=12),
    }
}
VISION_TUNED_ON = os.environ.get("EG2_VISION_MM_TUNED", "1") == "1"
VISION_MM_OVERRIDES: dict[tuple[str, int], MMSpec] = {}


def vision_spec(role: str | None, M: int, grid, batch: int = 1) -> MMSpec | None:
    """Tuned / searched spec for (role, M) on `grid` (a CoreCoord), batch 1 only."""
    if role is None or batch != 1:
        return None
    spec = VISION_MM_OVERRIDES.get((role, M))
    if spec is None and VISION_TUNED_ON:
        spec = VISION_TUNED.get((grid.x, grid.y), {}).get((role, M))
    return spec


def _ckc():
    if not _CKC:
        _CKC.append(compute_kernel_config_matmul(FP32_ACC))
    return _CKC[0]


def vision_linear(
    x: ttnn.Tensor,
    w: ttnn.Tensor,
    *,
    activation: str | None = None,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    policy=None,
    role: str | None = None,
) -> ttnn.Tensor:
    """x [B, 1, S, K] @ w [1, 1, K, N] -> [B, 1, S, N] TILE (optionally gelu_tanh fused).

    `policy` (a `precision.MatmulPolicy`): compute fidelity, fp32 accumulation and output dtype
    (the weight dtype is the one `w` was built with). None: HiFi2, `FP32_ACC`, bf16 output.
    `role` names the projection for the tuned-config table (VISION_TUNED)."""
    shape = x.shape
    B, S, K = shape[0], shape[-2], shape[-1]
    N = w.shape[-1]
    M = B * shape[1] * S
    if policy is None:
        fp32_acc, ckc, dtype = FP32_ACC, _ckc(), ttnn.bfloat16
    else:
        fp32_acc, ckc, dtype = policy.fp32_acc, policy.compute_kernel_config, policy.out_ttnn_dtype
    grid = x.device().compute_with_storage_grid_size()
    spec = vision_spec(role, M, grid, batch=B * shape[1])
    if spec is not None:
        return _linear_spec(x, w, spec, activation, memory_config, fp32_acc, ckc, ckc, dtype, grid)
    if M > MCAST2D_MAX_M or K > MCAST2D_MAX_K:
        return linear(x, w, activation=activation, memory_config=memory_config, policy=policy, dtype=dtype)
    pc = mcast2d_config(M, K, N, grid.x, grid.y, activation, fp32_acc)
    if pc is None:
        return linear(x, w, activation=activation, memory_config=memory_config, policy=policy, dtype=dtype)
    x2 = ttnn.reshape(x, (1, 1, M, K)) if B * shape[1] != 1 else x
    out = ttnn.linear(x2, w, program_config=pc, compute_kernel_config=ckc, memory_config=memory_config, dtype=dtype)
    if x2 is not x:
        out = ttnn.reshape(out, (B, shape[1], S, N))
    return out
