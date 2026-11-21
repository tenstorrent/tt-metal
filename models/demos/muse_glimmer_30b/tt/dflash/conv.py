# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""DFlash2 grouped dynamic causal convolution (kernel_size 2) on a block of rows inside a 32-row decode tile."""

import torch

import ttnn
from models.demos.muse_glimmer_30b.tt.common import as_tt, matmul_compute_config, replicate_mapper

_EXACT_CACHE = {}
_EXPAND_CACHE = {}
_SHIFT_CACHE = {}


def _exact_config(mesh_device):
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def _expand_matrix(mesh_device, hidden, group_size):
    """0/1 matrix [groups, hidden]: E[g, c] = 1 iff c // group_size == g."""
    key = (id(mesh_device), hidden, group_size)
    if key not in _EXPAND_CACHE:
        groups = hidden // group_size
        e = (torch.arange(hidden)[None, :] // group_size == torch.arange(groups)[:, None]).to(torch.bfloat16)
        _EXPAND_CACHE[key] = as_tt(e.view(1, 1, groups, hidden), mesh_device, dtype=ttnn.bfloat16)
    return _EXPAND_CACHE[key]


def block_shift_matrix(block_rows: int, rows: int = 32) -> torch.Tensor:
    """0/1 matrix [rows, rows] for the causal shift of several users' blocks in one decode tile.

    Row t of (S @ x) is the previous row of the same block (x[t-1]), or zero at the first row of a block. Rows
    [u*block_rows, (u+1)*block_rows) belong to user u, so a block never reads the last row of the user before it."""
    s = torch.zeros(rows, rows, dtype=torch.bfloat16)
    for t in range(1, rows):
        if t % block_rows:
            s[t, t - 1] = 1
    return s


def _shift_matrix(mesh_device, block_rows):
    """block_shift_matrix(block_rows) as a replicated [1, 1, 32, 32] device tensor, cached per block size."""
    key = (id(mesh_device), block_rows)
    if key not in _SHIFT_CACHE:
        s = block_shift_matrix(block_rows)
        _SHIFT_CACHE[key] = as_tt(s.view(1, 1, 32, 32), mesh_device, dtype=ttnn.bfloat16)
    return _SHIFT_CACHE[key]


class TtDynamicCausalConv:
    """DFlash2 grouped dynamic causal convolution on blocks of `block_rows` rows (one block per user) inside a 32-row decode tile.
    Weights are replicated on every device; inputs and outputs are replicated [1, 1, 32, H] bf16."""

    def __init__(
        self,
        mesh_device,
        base_kernel,
        kernel_projection,
        *,
        group_size=16,
        block_rows=16,
        dtype=ttnn.bfloat8_b,
        cache_file_name=None,
    ):
        self.mesh_device = mesh_device
        stages, kernel_size, hidden = base_kernel.shape
        assert stages == 2 and kernel_size == 2, f"only kernel_size 2 supported, got {tuple(base_kernel.shape)}"
        assert hidden % group_size == 0
        self.hidden = hidden
        self.groups = hidden // group_size
        assert kernel_projection.shape == (2 * kernel_size * self.groups, hidden)
        self.per_stage = kernel_size * self.groups

        self.proj_w = as_tt(
            kernel_projection.T.contiguous().view(1, 1, hidden, -1),
            mesh_device,
            dtype=dtype,
            mesh_mapper=replicate_mapper(mesh_device),
            cache_file_name=cache_file_name,
        )
        self.base = [
            [
                as_tt(base_kernel[s, o].reshape(1, 1, 1, hidden).to(torch.bfloat16), mesh_device, dtype=ttnn.bfloat16)
                for o in range(kernel_size)
            ]
            for s in range(2)
        ]
        self.expand = _expand_matrix(mesh_device, hidden, group_size)
        self.shift = _shift_matrix(mesh_device, block_rows)
        self.proj_cfg = matmul_compute_config(mesh_device)
        self.exact_cfg = _exact_config(mesh_device)

    def set_block_rows(self, block_rows):
        """Switch to blocks of `block_rows` rows; the shift matrices are cached and shared by all conv layers."""
        self.shift = _shift_matrix(self.mesh_device, block_rows)

    def _convolve(self, v, dyn, stage):
        if v.memory_config().is_sharded():
            v = ttnn.sharded_to_interleaved(v, ttnn.L1_MEMORY_CONFIG)
        mem = ttnn.L1_MEMORY_CONFIG
        coefs = []
        for o in range(2):
            c0 = stage * self.per_stage + o * self.groups
            d = ttnn.slice(dyn, [0, 0, 0, c0], [1, 1, 32, c0 + self.groups])
            c = ttnn.linear(d, self.expand, compute_kernel_config=self.exact_cfg, memory_config=mem)
            ttnn.deallocate(d)
            c2 = ttnn.add(c, self.base[stage][o], memory_config=mem)
            ttnn.deallocate(c)
            coefs.append(c2)
        shifted = ttnn.matmul(self.shift, v, compute_kernel_config=self.exact_cfg, memory_config=mem)
        t0 = ttnn.mul(coefs[0], v, memory_config=mem)
        t1 = ttnn.mul(coefs[1], shifted, memory_config=mem)
        out = ttnn.add(t0, t1, memory_config=mem)
        for t in (*coefs, shifted, t0, t1):
            ttnn.deallocate(t)
        return out

    def prepare(self, h):
        """h [1,1,32,H] -> (out [1,1,32,H], dyn [1,1,32,2*2*groups])."""
        if h.memory_config().is_sharded():
            h = ttnn.sharded_to_interleaved(h, ttnn.L1_MEMORY_CONFIG)
        dyn = ttnn.linear(h, self.proj_w, compute_kernel_config=self.proj_cfg, memory_config=ttnn.L1_MEMORY_CONFIG)
        return self._convolve(h, dyn, 0), dyn

    def finish(self, x, dyn):
        """x [1,1,32,H], dyn from prepare -> out [1,1,32,H]."""
        return self._convolve(x, dyn, 1)
