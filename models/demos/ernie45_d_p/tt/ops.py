# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Small per-op TTNN building blocks: RMSNorm, interleaved RoPE tables, SwiGLU MLP (TP4)."""

from __future__ import annotations

import torch

import ttnn
from models.demos.ernie45_d_p.reference.ernie_ref import rope_cos_sin
from models.demos.ernie45_d_p.tt.common import COMPUTE_HIFI2, COMPUTE_HIFI4, cache_name, replicate, shard


class TtRMSNorm:
    def __init__(self, mesh, weight: torch.Tensor, eps: float, name: str):
        H = weight.shape[-1]
        self.eps = eps
        self.weight = replicate(
            mesh, weight.reshape(1, 1, H // 32, 32), layout=ttnn.ROW_MAJOR_LAYOUT, cache=cache_name("norm", name)
        )

    def __call__(self, x):
        return ttnn.rms_norm(x, weight=self.weight, epsilon=self.eps, compute_kernel_config=COMPUTE_HIFI4)


def rot_transformation_mat() -> torch.Tensor:
    """32x32 tile T with (x @ T) = rotate_interleaved(x): out[2k] = -x[2k+1], out[2k+1] = x[2k]."""
    t = torch.zeros(1, 1, 32, 32)
    t[..., torch.arange(0, 32, 2), torch.arange(1, 32, 2)] = 1.0
    t[..., torch.arange(1, 32, 2), torch.arange(0, 32, 2)] = -1.0
    return t


class TtRope:
    """Interleaved-pair RoPE (ERNIE / Meta layout) via rotary_embedding_llama.

    cos/sin for absolute positions [start, start+S) are built per chunk; the chunk offset lives
    entirely in these tables, so the op itself is position-agnostic.
    """

    def __init__(self, mesh, head_dim: int, theta: float):
        self.mesh, self.head_dim, self.theta = mesh, head_dim, theta
        self.trans_mat = replicate(mesh, rot_transformation_mat())
        self._tables = {}

    def tables(self, start: int, seq: int):
        key = (start, seq)
        if key not in self._tables:
            cos, sin = rope_cos_sin(torch.arange(start, start + seq), self.head_dim, self.theta)
            self._tables = {
                key: (replicate(self.mesh, cos[None, None]), replicate(self.mesh, sin[None, None]))
            }  # keep only the current chunk's tables resident
        return self._tables[key]

    def __call__(self, x, start: int):
        """x: [1, heads, S, D]."""
        cos, sin = self.tables(start, x.shape[-2])
        return ttnn.experimental.rotary_embedding_llama(
            x, cos, sin, self.trans_mat, is_decode_mode=False, compute_kernel_config=COMPUTE_HIFI4
        )


class TtSwiGLU:
    """down(silu(gate x) * up x); gate/up column-parallel, down row-parallel.

    Returns the per-chip PARTIAL sum [1,1,S,H]; caller does (or fuses) the all_reduce.
    """

    def __init__(self, mesh, w_gate, w_up, w_down, name: str, dtype=ttnn.bfloat16):
        # torch nn.Linear weights are [out, in]; ttnn.linear wants [in, out].
        self.w_gate = shard(
            mesh, w_gate.T.contiguous()[None, None], dim=-1, dtype=dtype, cache=cache_name(name, "gate")
        )
        self.w_up = shard(mesh, w_up.T.contiguous()[None, None], dim=-1, dtype=dtype, cache=cache_name(name, "up"))
        self.w_down = shard(
            mesh, w_down.T.contiguous()[None, None], dim=-2, dtype=dtype, cache=cache_name(name, "down")
        )

    def __call__(self, x):
        g = ttnn.linear(x, self.w_gate, compute_kernel_config=COMPUTE_HIFI2)
        u = ttnn.linear(x, self.w_up, compute_kernel_config=COMPUTE_HIFI2)
        h = ttnn.mul(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        out = ttnn.linear(h, self.w_down, compute_kernel_config=COMPUTE_HIFI2)
        ttnn.deallocate(h)
        return out


def all_reduce(x):
    return ttnn.all_reduce(x, cluster_axis=1)
