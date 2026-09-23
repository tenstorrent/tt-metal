# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from torch import nn

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.utils.general_utils import get_cache_file_name

_SHARDED_NORM_MAX_HEIGHT = 128
# In+out bf16 shards plus RMSNorm scratch must fit L1. 26B at height=1024 on
# an 8-core Wormhole grid overflows before scratch.
_SHARDED_NORM_MAX_PER_CORE_BYTES = 1024 * 1024


def activation_physical_height(shape) -> int:
    """Tile-padded N*C*H row count. Batched prefill is [B,1,S,H], so use B*S not S."""
    rows = 1
    for i in range(len(shape) - 1):
        rows *= int(shape[i])
    tile = ttnn.TILE_SIZE
    return ((rows + tile - 1) // tile) * tile


def width_shard_spec(mesh_device, dim, height):
    """``(input_memcfg, program_config)`` for width-sharded RMSNorm at ``(height, dim)``."""
    if height <= 0 or height > _SHARDED_NORM_MAX_HEIGHT:
        return None
    if dim % ttnn.TILE_SIZE != 0 or height % ttnn.TILE_SIZE != 0:
        return None
    tiles = dim // ttnn.TILE_SIZE
    grid = mesh_device.compute_with_storage_grid_size()
    best = None
    for gy in range(1, grid.y + 1):
        for gx in range(1, grid.x + 1):
            n = gx * gy
            if tiles % n != 0 or n <= 1:
                continue
            per_core_bytes = int(height) * (int(dim) // n) * 2 * 2
            if per_core_bytes > _SHARDED_NORM_MAX_PER_CORE_BYTES:
                continue
            if best is None or n > best[0]:
                best = (n, gx, gy)
    if best is None:
        return None
    num_cores, gx, gy = best
    block_w = tiles // num_cores
    subblock_w = 4
    while subblock_w > 1 and block_w % subblock_w != 0:
        subblock_w -= 1
    input_memcfg = ttnn.create_sharded_memory_config(
        shape=(height, dim // num_cores),
        core_grid=ttnn.CoreGrid(x=gx, y=gy),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[gx, gy],
        subblock_w=subblock_w,
        block_h=height // ttnn.TILE_SIZE,
        block_w=block_w,
        inplace=False,
    )
    return (input_memcfg, program_config)


def width_shard_input_memcfg(mesh_device, dim, height):
    spec = width_shard_spec(mesh_device, dim, height)
    return spec[0] if spec else None


def decode_width_shard_memcfg(mesh_device, dim):
    """Shared decode RMSNorm input memory config (one tile tall), or None."""
    return width_shard_input_memcfg(mesh_device, dim, ttnn.TILE_SIZE)


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
        else:
            self.tt_weight = None
            self.tt_weight_tile = None

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
        self._sharded_height = None

    def _build_sharded_cfg(self, dim, height=None):
        height = ttnn.TILE_SIZE if height is None else int(height)
        return width_shard_spec(self.mesh_device, dim, height)

    def _forward_sharded(self, x, already_sharded=False, keep_sharded=False, interleaved_memory_config=None):
        """Width-sharded decode RMSNorm with optional L1 handoffs."""
        x_sh = x if already_sharded else ttnn.to_memory_config(x, self._sharded_cfg[0])
        out = ttnn.rms_norm(
            x_sh,
            weight=self.tt_weight,
            epsilon=self.eps,
            program_config=self._sharded_cfg[1],
        )
        if not already_sharded:
            x_sh.deallocate(True)
        if keep_sharded:
            return out
        destination = interleaved_memory_config or ttnn.DRAM_MEMORY_CONFIG
        out_interleaved = ttnn.sharded_to_interleaved(out, destination)
        out.deallocate(True)
        return out_interleaved

    def forward(self, x, keep_sharded=False, interleaved_memory_config=None):
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

            tt_gathered_stats_memory_config = ttnn.create_sharded_memory_config(
                shape=[1, 1, 32, 32 * self.mesh_shape[1]],
                core_grid=ttnn.CoreGrid(y=1, x=1),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            )
            tt_stats = ttnn.rms_norm_pre_all_gather(x, program_config=program_config, dtype=ttnn.bfloat16)

            # Avoid deprecated all_gather kwargs (num_links/topology/mesh_device).
            tt_gathered_stats = ttnn.all_gather(
                tt_stats,
                dim=3,
                cluster_axis=1,
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
            # Width-sharded fast path for decode (one tile) and short
            # prefill. Auto-sharding every prefill ≤1024 hung T3K at ISL=128.
            padded_height = activation_physical_height(x.shape) if len(x.shape) == 4 else 0
            use_sharded_norm = padded_height == ttnn.TILE_SIZE or (
                keep_sharded and 1 <= padded_height <= _SHARDED_NORM_MAX_HEIGHT
            )
            if self.with_scale and self.tt_weight is not None and len(x.shape) == 4 and use_sharded_norm:
                dim = x.shape[-1]
                if self._sharded_cfg is None or self._sharded_dim != dim or self._sharded_height != padded_height:
                    self._sharded_dim = dim
                    self._sharded_height = padded_height
                    self._sharded_cfg = self._build_sharded_cfg(dim, padded_height)
                if self._sharded_cfg:
                    if x.is_sharded():
                        if x.memory_config() == self._sharded_cfg[0]:
                            return self._forward_sharded(
                                x,
                                already_sharded=True,
                                keep_sharded=keep_sharded,
                                interleaved_memory_config=interleaved_memory_config,
                            )
                    else:
                        return self._forward_sharded(
                            x,
                            keep_sharded=keep_sharded,
                            interleaved_memory_config=interleaved_memory_config,
                        )

            norm_input = x
            owns_norm_input = False
            if x.is_sharded():
                norm_input = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
                owns_norm_input = True
            output_kwargs = {}
            if interleaved_memory_config is not None:
                output_kwargs["memory_config"] = interleaved_memory_config
            is_prefill = len(x.shape) == 4 and x.shape[-2] > ttnn.TILE_SIZE
            compute_kernel_config = self.prefill_compute_kernel_config if is_prefill else None
            weight = self.tt_weight_tile if is_prefill and self.tt_weight_tile is not None else self.tt_weight
            if self.with_scale:
                tt_output = ttnn.rms_norm(
                    norm_input,
                    weight=weight,
                    epsilon=self.eps,
                    compute_kernel_config=compute_kernel_config,
                    **output_kwargs,
                )
            else:
                tt_output = ttnn.rms_norm(
                    norm_input,
                    epsilon=self.eps,
                    compute_kernel_config=compute_kernel_config,
                    **output_kwargs,
                )
            if owns_norm_input:
                norm_input.deallocate(True)
            return tt_output
