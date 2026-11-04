# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm (interleaved, with a width-sharded decode path) and weightless per-head RMSNorm."""

import torch

import ttnn
from models.demos.muse_glimmer_30b.tt.common import decode_memcfg, norm_compute_config, replicate_mapper

TILE = 32


def rms_norm_path_info(x_shape, grid):
    """Pure-python grid selection for the width-sharded decode path.

    grid: (grid_x, grid_y) or an object with .x/.y. Returns (gx, gy, block_w, subblock_w),
    or None if dim is not tile aligned or no multi-core grid divides dim // 32.
    """
    dim = int(x_shape[-1])
    if dim % TILE != 0:
        return None
    gxmax, gymax = (grid.x, grid.y) if hasattr(grid, "x") else (grid[0], grid[1])
    tiles = dim // TILE
    best = None  # (num_cores, gx, gy)
    for gy in range(1, gymax + 1):
        for gx in range(1, gxmax + 1):
            n = gx * gy
            if tiles % n == 0 and (best is None or n > best[0]):
                best = (n, gx, gy)
    if best is None or best[0] == 1:
        return None
    n, gx, gy = best
    block_w = tiles // n
    subblock_w = 4
    while subblock_w > 1 and block_w % subblock_w != 0:
        subblock_w -= 1
    return gx, gy, block_w, subblock_w


class RMSNorm:
    """y = rmsnorm(x) * w, or * (1 + w) if add_unit_offset; weight=None => weightless.

    Prefill (rows > 32) uses the interleaved path with a TILE gamma. Decode (1..32 rows, 4-D)
    uses a width-sharded path with a ROW_MAJOR gamma [1,1,dim//32,32]. It returns L1 width-sharded
    when ``memory_config`` is sharded, else interleaved in ``memory_config`` (default DRAM).
    """

    def __init__(
        self,
        mesh_device,
        weight: torch.Tensor | None,
        eps: float,
        add_unit_offset: bool = False,
        cache_file_name: str | None = None,
    ):
        self.mesh_device = mesh_device
        self.eps = eps
        self.compute_kernel_config = norm_compute_config(mesh_device)
        self.weight = None
        self.weight_rm = None
        self._dim = None if weight is None else int(weight.numel())
        if weight is not None:
            w = weight.float()
            if add_unit_offset:
                w = 1.0 + w
            # TILE gamma [1,1,1,H] enables the large-tensor RMSNorm path (bounds L1 for the fp32 intermediates at H=6656).
            w = w.to(torch.bfloat16).reshape(1, 1, 1, -1)
            self.weight = ttnn.as_tensor(
                w,
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=replicate_mapper(mesh_device),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache_file_name,
            )
        # ROW_MAJOR gamma [1,1,dim//32,32] for the sharded decode path (created lazily for weightless norms,
        # since dim is only known from the activation).
        self._rm_cache_name = None if cache_file_name is None else cache_file_name + "_rm"
        self._add_unit_offset = add_unit_offset
        if weight is not None and w.numel() % TILE == 0:
            self.weight_rm = self._make_rm_gamma(w.reshape(-1))

    def _make_rm_gamma(self, w_flat):
        dim = w_flat.numel()
        return ttnn.as_tensor(
            w_flat.to(torch.bfloat16).reshape(1, 1, dim // TILE, TILE),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=replicate_mapper(self.mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=self._rm_cache_name,
        )

    def _use_sharded(self, x):
        return len(x.shape) == 4 and 1 <= x.shape[-2] <= TILE

    def _forward_sharded(self, x, memory_config):
        rows, dim = x.shape[-2], x.shape[-1]
        if dim % TILE != 0 or (dim // TILE) < 2:
            return None
        if self.weight_rm is None and self.weight is None:
            # weightless norm: unit gamma so the sharded path can be used
            self.weight_rm = self._make_rm_gamma(torch.ones(dim))
        mem = decode_memcfg(dim, self.mesh_device)
        g = self.mesh_device.compute_with_storage_grid_size()
        own = not (x.is_sharded() and x.memory_config() == mem)
        x_sh = ttnn.to_memory_config(x, mem) if own else x
        block_w = mem.shard_spec.shape[1] // TILE
        subblock_w = next(s for s in (4, 3, 2, 1) if block_w % s == 0)
        pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[g.x, g.y], subblock_w=subblock_w, block_h=1, block_w=block_w, inplace=False
        )
        # Default kernel config (None) is the fast path for the sharded decode norm.
        y = ttnn.rms_norm(
            x_sh,
            epsilon=self.eps,
            weight=self.weight_rm,
            program_config=pc,
            compute_kernel_config=None,
            memory_config=mem,
        )
        if own:
            x_sh.deallocate(True)
        if memory_config is not None and memory_config.is_sharded():
            return y
        out = ttnn.sharded_to_interleaved(y, memory_config or ttnn.DRAM_MEMORY_CONFIG)
        y.deallocate(True)
        return out

    def __call__(self, x, memory_config=None):
        if self._use_sharded(x):
            out = self._forward_sharded(x, memory_config)
            if out is not None:
                return out
        kwargs = dict(epsilon=self.eps, compute_kernel_config=self.compute_kernel_config, memory_config=memory_config)
        if self.weight is not None:
            kwargs["weight"] = self.weight
        return ttnn.rms_norm(x, **kwargs)


def per_head_rms_norm(x, eps, mesh_device=None, fp32_acc=True):
    """Weightless RMSNorm over head_dim for [B, heads, S, hd] or [1, B, heads, hd] tensors."""
    orig_shape = x.shape
    head_dim = orig_shape[-1]
    n = 1
    for d in list(orig_shape)[:-1]:
        n *= d
    flat = ttnn.reshape(x, (1, 1, n, head_dim))
    cfg = None
    if fp32_acc:
        mesh_device = mesh_device if mesh_device is not None else x.device()
        cfg = norm_compute_config(mesh_device)
    out = ttnn.rms_norm(flat, epsilon=eps, compute_kernel_config=cfg)
    return ttnn.reshape(out, orig_shape)
