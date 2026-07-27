# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
LayerNorm (weight + bias) for the XTTS-v2 GPT, with two execution paths:

- **interleaved** (default): plain `ttnn.layer_norm`, works for any sequence length —
  used by the prefill core (`TTNNGPTCore`).
- **width-sharded** (`sharded=True`): a `LayerNormShardedMultiCoreProgramConfig` that shards
  the hidden dim across `shard_height` cores, for single-tile-height (batch=1 decode) inputs.
  Adapted from `models/tt_transformers/tt/multimodal/llama_layernorm.py::TtLayerNorm`.

The interleaved decode LayerNorm on `[1,1,1024]` runs effectively single-core; sharding the
reduction across 32 cores cut the traced decode step from ~12.4 to ~11.1 ms/token (~11%)
over the 62 LayerNorms per token, at equal PCC. The sharded weights are built lazily on the
first sharded call (before trace capture, so they are trace-safe), so a core that only ever
runs interleaved never allocates them.
"""

import ttnn

_TILE = 32


def _default_compute_config():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


class TTNNLayerNorm:
    def __init__(
        self, device, weight_t, bias_t, dim, eps=1e-5, dtype=ttnn.bfloat16, shard_height=_TILE, mesh_mapper=None
    ):
        self.device = device
        self.mesh_mapper = mesh_mapper  # replicate weights across a mesh; None on a single card
        self.dim = dim
        self.eps = eps
        self.dtype = dtype
        self.shard_height = shard_height
        self.compute_kernel_config = _default_compute_config()
        # keep host copies for the (lazy) sharded weight expansion
        self._w_torch = weight_t.reshape(-1).contiguous()
        self._b_torch = bias_t.reshape(-1).contiguous()
        self.weight = ttnn.from_torch(
            self._w_torch, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mesh_mapper
        )
        self.bias = ttnn.from_torch(
            self._b_torch, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mesh_mapper
        )
        self._sharded = None  # (weight, bias, mem_config, program_config), built on first sharded call

    def _build_sharded(self):
        sh = self.shard_height
        assert self.dim % sh == 0, f"dim {self.dim} must be a multiple of shard_height {sh}"
        width_per_core = self.dim // sh
        core_grid = ttnn.CoreGrid(x=8, y=sh // 8)
        mem_config = ttnn.create_sharded_memory_config(
            shape=(sh, width_per_core),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[core_grid.x, core_grid.y],
            subblock_w=width_per_core // _TILE,
            block_h=sh // _TILE,
            block_w=width_per_core // _TILE,
            inplace=False,
        )
        # the sharded LN wants weight/bias broadcast over the shard-height tile
        we = ttnn.from_torch(
            self._w_torch.view(1, 1, self.dim).expand(1, sh, self.dim).contiguous(),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=self.mesh_mapper,
        )
        be = ttnn.from_torch(
            self._b_torch.view(1, 1, self.dim).expand(1, sh, self.dim).contiguous(),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=self.mesh_mapper,
        )
        self._sharded = (we, be, mem_config, program_config)

    def __call__(self, x, sharded=False, compute_kernel_config=None):
        ckc = compute_kernel_config or self.compute_kernel_config
        if not sharded:
            return ttnn.layer_norm(x, weight=self.weight, bias=self.bias, epsilon=self.eps, compute_kernel_config=ckc)
        if self._sharded is None:
            self._build_sharded()
        we, be, mem_config, program_config = self._sharded
        xs = ttnn.interleaved_to_sharded(x, mem_config)
        y = ttnn.layer_norm(
            xs,
            weight=we,
            bias=be,
            epsilon=self.eps,
            program_config=program_config,
            memory_config=mem_config,
            compute_kernel_config=ckc,
        )
        return ttnn.sharded_to_interleaved(y)
