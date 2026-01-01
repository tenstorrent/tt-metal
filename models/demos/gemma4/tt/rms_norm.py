# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0


import os

from torch import nn

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.custom_ops.fused_norm_decode import FusedNormDecode
from models.demos.gemma4.tt.mesh_orientation import decode_flag
from models.demos.gemma4.utils.general_utils import get_cache_file_name

# Widest shard (column tiles per core) the norm runs on a consumer's grid: the 8-core wqkv grids (31B 21 tiles,
# 12B 15) are still cheaper than the S2I + I2S pair they remove (layer lab: -5..-14 us per layer over the pre-FF
# norm alone); GEMMA4_NORM_SHARDED_OUT_MAX_W overrides.
SHARDED_OUT_DEFAULT_MAX_W = 24


def sharded_layout_program_config(mem, max_block_w=None):
    """The width-sharded LayerNorm program config running on the grid of ``mem``, a consumer's L1 width-sharded
    one-row-tile activation layout (a DRAM-sharded decode matmul's ``act_memory_config``): the norm's output is
    left in that layout and the matmul reads it without its own reshard (GEMMA4_NORM_SHARDED_OUT=1 on the native
    norm path). None when the layout is not one rectangular grid from (0, 0) of whole column tiles, or when a
    shard is wider than ``max_block_w`` tiles (GEMMA4_NORM_SHARDED_OUT_MAX_W, default ``SHARDED_OUT_DEFAULT_MAX_W``):
    then the norm keeps its own grid and the matmul its reshard."""
    if not mem.is_sharded() or mem.memory_layout != ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        return None
    spec = mem.shard_spec
    if int(spec.shape[0]) != ttnn.TILE_SIZE or int(spec.shape[1]) % ttnn.TILE_SIZE != 0:
        return None
    ranges = list(spec.grid.ranges())
    if len(ranges) != 1 or (ranges[0].start.x, ranges[0].start.y) != (0, 0):
        return None
    gx, gy = ranges[0].end.x + 1, ranges[0].end.y + 1
    block_w = int(spec.shape[1]) // ttnn.TILE_SIZE
    cap = (
        int(os.environ.get("GEMMA4_NORM_SHARDED_OUT_MAX_W", str(SHARDED_OUT_DEFAULT_MAX_W)))
        if max_block_w is None
        else int(max_block_w)
    )
    if block_w > cap:
        return None
    subblock_w = 4
    while subblock_w > 1 and block_w % subblock_w != 0:
        subblock_w -= 1
    return ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[gx, gy], subblock_w=subblock_w, block_h=1, block_w=block_w, inplace=False
    )


class RMSNorm(nn.Module):
    def __init__(self, mesh_device, hf_config, state_dict, tensor_cache_path=None, mesh_config=None, with_scale=True):
        super().__init__()
        self.with_scale = with_scale

        if with_scale and state_dict and "weight" in state_dict:
            torch_weight = state_dict["weight"].reshape((1, 1, -1, ttnn.TILE_SIZE))
        else:
            torch_weight = None

        self.mesh_config = mesh_config or MeshConfig(mesh_device.shape, decode=ModeConfig(tp=mesh_device.shape[1]))
        self.is_distributed = False

        if with_scale:
            self.tt_weight = ttnn.as_tensor(
                torch_weight,
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                cache_file_name=get_cache_file_name(tensor_cache_path, "weight"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=(
                    self.mesh_config.shard_mapper(mesh_device, mesh_dims=(None, -2)) if self.is_distributed else None
                ),
            )
            # TILE gamma enables the large-tensor RMSNorm path, which bounds L1 usage
            # for FP32 intermediate buffers in the 5376-wide prefill norm.
            # Keep the row-major copy for the width-sharded decode path.
            flat_weight = ttnn.reshape(self.tt_weight, (1, 1, 1, hf_config.hidden_size))
            self.tt_weight_tile = ttnn.to_layout(flat_weight, ttnn.TILE_LAYOUT)
            # GEMMA4_NORM_FUSED=2: the [1, 1, 32, H] row-replicated gamma of custom_ops/fused_norm_decode (an
            # on-device repeat of the loaded row, so a warm weight cache never persists it as a placeholder)
            self.tt_weight_full = (
                FusedNormDecode.full_gamma(self.tt_weight_tile) if decode_flag("GEMMA4_NORM_FUSED") == "2" else None
            )
        else:
            self.tt_weight = None
            self.tt_weight_tile = None
            self.tt_weight_full = None

        self.eps = hf_config.rms_norm_eps
        self.mesh_device = mesh_device
        # Match the reference's FP32 prefill RMSNorm computation.
        self.prefill_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # Decode width-sharded fast path. The plain (interleaved) rms_norm runs
        # the RMS reduction over the full hidden width on few cores — ~76 us for
        # a single-token [1,1,32,hidden] norm on Gemma4-31B (hidden=5376). Width-
        # sharding the activation across a core grid parallelizes the reduction
        # (LayerNormShardedMultiCoreProgramConfig handles the cross-core gather),
        # cutting it to <10 us. Built lazily on first decode-shaped call so we
        # can read the activation's true (padded) hidden width, then cached.
        self._sharded_cfg = None  # (input_memcfg, program_config) or None if unavailable
        self._sharded_dim = None

    def _build_sharded_cfg(self, dim):
        """Pick the largest core grid whose core count divides dim/32 and build
        the width-sharded input memcfg + LayerNorm program config. Returns None
        if no usable grid divides the tile-width evenly (falls back to plain)."""
        if dim % ttnn.TILE_SIZE != 0:
            return None
        tiles = dim // ttnn.TILE_SIZE
        grid = self.mesh_device.compute_with_storage_grid_size()
        best = None  # (num_cores, gx, gy)
        for gy in range(1, grid.y + 1):
            for gx in range(1, grid.x + 1):
                n = gx * gy
                if tiles % n == 0 and (best is None or n > best[0]):
                    best = (n, gx, gy)
        if best is None or best[0] == 1:
            return None
        num_cores, gx, gy = best
        block_w = tiles // num_cores
        subblock_w = 4
        while subblock_w > 1 and block_w % subblock_w != 0:
            subblock_w -= 1
        input_memcfg = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, dim // num_cores),
            core_grid=ttnn.CoreGrid(x=gx, y=gy),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[gx, gy],
            subblock_w=subblock_w,
            block_h=1,
            block_w=block_w,
            inplace=False,
        )
        return (input_memcfg, program_config)

    def _forward_sharded(self, x):
        """Width-sharded decode RMSNorm: I2S -> sharded rms_norm -> S2I."""
        x_sh = ttnn.to_memory_config(x, self._sharded_cfg[0])
        out = ttnn.rms_norm(
            x_sh,
            weight=self.tt_weight,
            epsilon=self.eps,
            program_config=self._sharded_cfg[1],
        )
        x_sh.deallocate(True)
        out_interleaved = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        out.deallocate(True)
        return out_interleaved

    def _forward_sharded_out(self, x, mem, program_config):
        """Width-sharded decode RMSNorm on a consumer's L1 activation layout: I2S into it, the sharded rms_norm, and
        the output left in that layout (no S2I; the DRAM-sharded decode matmul reads it without its own reshard)."""
        x_sh = ttnn.to_memory_config(x, mem)
        out = ttnn.rms_norm(
            x_sh, weight=self.tt_weight, epsilon=self.eps, program_config=program_config, memory_config=mem
        )
        x_sh.deallocate(True)
        return out

    def forward(self, x, out_memory_config=None):
        """``out_memory_config``: a width-sharded decode-matmul activation layout for the decode norm's output: the
        fused decode norm (GEMMA4_NORM_FUSED=2) writes it; the native width-sharded LayerNorm runs on its grid when
        ``sharded_layout_program_config`` accepts it; the other paths ignore it (interleaved output)."""
        if self.is_distributed:
            activation_grid_bounding_box_size = x.memory_config().shard_spec.grid.bounding_box().grid_size()
            shard_height, shard_width = x.memory_config().shard_spec.shape
            program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=activation_grid_bounding_box_size,
                subblock_w=1,
                block_h=ttnn.core.divup(shard_height, ttnn.TILE_SIZE),
                block_w=ttnn.core.divup(shard_width, ttnn.TILE_SIZE),
                inplace=False,
            )

            tp_axis = int(getattr(self.mesh_config, "tp_axis", 1))
            tt_gathered_stats_memory_config = ttnn.create_sharded_memory_config(
                shape=[1, 1, 32, 32 * int(self.mesh_config.mesh_shape[tp_axis])],
                core_grid=ttnn.CoreGrid(y=1, x=1),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            )
            tt_stats = ttnn.rms_norm_pre_all_gather(x, program_config=program_config, dtype=ttnn.bfloat16)

            # Avoid deprecated all_gather kwargs (num_links/topology/mesh_device).
            tt_gathered_stats = ttnn.all_gather(
                tt_stats,
                dim=3,
                cluster_axis=tp_axis,
                memory_config=tt_gathered_stats_memory_config,
            )
            ttnn.deallocate(tt_stats)

            tt_output = ttnn.rms_norm_post_all_gather(
                x,
                tt_gathered_stats,
                program_config=program_config,
                epsilon=self.eps,
                weight=self.tt_weight,
                dtype=ttnn.bfloat16,
                stats=tt_gathered_stats,
            )
            ttnn.deallocate(tt_gathered_stats)
            return tt_output
        else:
            # Decode fast path: single-tile-height (32 rows) activation with a
            # learned weight and an interleaved layout → width-sharded rms_norm.
            # Prefill (height > 32) and the no-weight per-head norms keep the
            # plain path. Sharded config is dim-specific, so rebuild if the
            # activation width ever changes.
            if (
                len(x.shape) == 4
                and x.shape[0] == 1
                and x.shape[1] == 1
                and 1 <= x.shape[-2] <= ttnn.TILE_SIZE
                and not x.is_sharded()
                and x.shape[-1] % ttnn.TILE_SIZE == 0
                and decode_flag("GEMMA4_NORM_FUSED") == "2"
            ):
                # GEMMA4_NORM_FUSED=2: one program on 22 cores with FPU broadcasts (custom_ops/fused_norm_decode);
                # the layer-level forms (residual add, two inputs, three outputs) are taken by tt/layer.py.
                gamma = self.tt_weight_full if self.with_scale else None
                return FusedNormDecode.op(x, [gamma], self.eps, out_memory_configs=[out_memory_config])[0]
            if (
                self.with_scale
                and self.tt_weight is not None
                and len(x.shape) == 4
                and 1 <= x.shape[-2] <= ttnn.TILE_SIZE
                and not x.is_sharded()
            ):
                dim = x.shape[-1]
                if out_memory_config is not None and dim % ttnn.TILE_SIZE == 0:
                    # GEMMA4_NORM_SHARDED_OUT=1: the native kernel on the consumer matmul's activation grid, output
                    # left there (the base path's S2I + the matmul's I2S are not run)
                    pc = sharded_layout_program_config(out_memory_config)
                    if pc is not None:
                        return self._forward_sharded_out(x, out_memory_config, pc)
                if self._sharded_cfg is None or self._sharded_dim != dim:
                    self._sharded_dim = dim
                    self._sharded_cfg = self._build_sharded_cfg(dim)
                if self._sharded_cfg:
                    return self._forward_sharded(x)

            is_prefill = len(x.shape) == 4 and x.shape[-2] > ttnn.TILE_SIZE
            compute_kernel_config = self.prefill_compute_kernel_config if is_prefill else None
            weight = self.tt_weight_tile if is_prefill and self.tt_weight_tile is not None else self.tt_weight
            if self.with_scale:
                tt_output = ttnn.rms_norm(
                    x,
                    weight=weight,
                    epsilon=self.eps,
                    compute_kernel_config=compute_kernel_config,
                )
            else:
                tt_output = ttnn.rms_norm(
                    x,
                    epsilon=self.eps,
                    compute_kernel_config=compute_kernel_config,
                )
            return tt_output
