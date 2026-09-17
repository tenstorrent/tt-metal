# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os

from torch import nn

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.utils.general_utils import get_cache_file_name

# Row cutoff for the width-shard *search*. Height alone is not enough: 26B
# hidden=2816 on an 8-core WH grid needs 720,896 B/bank for the input *and*
# again for the output (1,441,792 B) against a 1,393,472 B bank — CI
# test_rms_norm / test_layer prefill_1024 OOMs. :func:`width_shard_spec`
# rejects layouts whose per-core I/O exceeds the bank. 31B hidden=5376
# still fits at 1024 (~197 KB/bank × 2). Longer prefill stays interleaved.
_SHARDED_NORM_MAX_HEIGHT = 1024
# Observed WH worker L1 bank after firmware (run 32690156816, 26B unit).
# BH banks are larger; override with GEMMA4_SHARDED_NORM_L1_BANK.
_DEFAULT_L1_BANK_BYTES = 1_393_472
_SHARDED_NORM_ELEM_BYTES = 2  # bf16 activations on this path
_TILE = 32
_FP32_BYTES = 4


def sharded_norm_scratch_bytes(height, dim, num_cores, dtype_bytes=_SHARDED_NORM_ELEM_BYTES) -> int:
    """Per-bank CB bytes the sharded RMSNorm kernel adds on top of live I/O."""
    del height
    if num_cores <= 0 or int(dim) % int(num_cores) != 0:
        return 1 << 62
    shard_w = int(dim) // int(num_cores)
    block_wt = max(1, shard_w // _TILE)
    gamma = _TILE * shard_w * int(dtype_bytes)
    stats = 2 * _TILE * _TILE * _FP32_BYTES
    col_mask = block_wt * _TILE * _TILE * int(dtype_bytes)
    return gamma + stats + col_mask


def sharded_norm_enabled() -> bool:
    return os.environ.get("GEMMA4_SHARDED_NORM", "1").lower() not in ("0", "false", "no")


def norm_keep_sharded_enabled() -> bool:
    return os.environ.get("GEMMA4_NORM_KEEP_SHARDED", "1").lower() not in ("0", "false", "no")


def maybe_interleave(tensor, memory_config=None):
    """DRAM-interleaved view of ``tensor``; no-op when already interleaved."""
    if tensor is None or not tensor.is_sharded():
        return tensor
    dest = memory_config or ttnn.DRAM_MEMORY_CONFIG
    out = ttnn.sharded_to_interleaved(tensor, dest)
    tensor.deallocate(True)
    return out


def align_to_memcfg(tensor, memcfg):
    """Reshard / I2S ``tensor`` onto ``memcfg``. Returns ``(aligned, owned)``."""
    if tensor is None or memcfg is None or not memcfg.is_sharded():
        return tensor, False
    if tensor.is_sharded() and tensor.memory_config() == memcfg:
        return tensor, False
    return ttnn.to_memory_config(tensor, memcfg), True


def sharded_norm_per_core_bytes(height, dim, num_cores, dtype_bytes=_SHARDED_NORM_ELEM_BYTES) -> int:
    """Bytes of one width-shard of a ``[height, dim]`` activation on ``num_cores``."""
    if num_cores <= 0 or dim % num_cores != 0:
        return 1 << 62
    return int(height) * (int(dim) // int(num_cores)) * int(dtype_bytes)


def sharded_norm_fits_l1(
    height,
    dim,
    num_cores,
    l1_bank_bytes=_DEFAULT_L1_BANK_BYTES,
    dtype_bytes=_SHARDED_NORM_ELEM_BYTES,
    scratch_bytes=None,
) -> bool:
    """True when input + output (+ scratch) fit in one L1 bank."""
    per = sharded_norm_per_core_bytes(height, dim, num_cores, dtype_bytes)
    scratch = (
        int(scratch_bytes)
        if scratch_bytes is not None
        else sharded_norm_scratch_bytes(height, dim, num_cores, dtype_bytes)
    )
    return (2 * per + scratch) <= int(l1_bank_bytes)


def _l1_bank_bytes(mesh_device) -> int:
    env = os.environ.get("GEMMA4_SHARDED_NORM_L1_BANK")
    if env:
        return int(env)
    try:
        dev = mesh_device.get_devices()[0] if hasattr(mesh_device, "get_devices") else mesh_device
        if hasattr(dev, "l1_size_per_core"):
            return int(dev.l1_size_per_core())
    except Exception:
        pass
    return _DEFAULT_L1_BANK_BYTES


def activation_physical_height(shape) -> int:
    """Tile-padded row count a width-sharded layout must use for ``shape``."""
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
    best = None  # (num_cores, gx, gy)
    for gy in range(1, grid.y + 1):
        for gx in range(1, grid.x + 1):
            n = gx * gy
            if tiles % n == 0 and (best is None or n > best[0]):
                best = (n, gx, gy)
    if best is None or best[0] == 1:
        return None
    num_cores, gx, gy = best
    if not sharded_norm_fits_l1(height, dim, num_cores, l1_bank_bytes=_l1_bank_bytes(mesh_device)):
        return None
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

    def forward(self, x):
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
            # Decode fast path: single-tile-height (32 rows) activation with a
            # learned weight and an interleaved layout → width-sharded rms_norm.
            # Prefill (height > 32) and the no-weight per-head norms keep the
            # plain path. Sharded config is dim-specific, so rebuild if the
            # activation width ever changes.
            if (
                self.with_scale
                and self.tt_weight is not None
                and len(x.shape) == 4
                and 1 <= x.shape[-2] <= ttnn.TILE_SIZE
                and not x.is_sharded()
            ):
                dim = x.shape[-1]
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
