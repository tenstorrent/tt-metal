# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from torch import nn

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.utils.general_utils import get_cache_file_name

# Row cutoff for the width-sharded fast path. The per-core input shard is
# (height, dim/num_cores) elements resident in L1 alongside other live
# buffers; at hidden=5376 / TP=8 a seq=4096 prefill norm needs ~786 KB/bank
# and TT_FATALs (Out of Memory) with only ~607 KB free. seq=1024 fits
# comfortably (~197 KB/bank). Longer prefill keeps the plain interleaved path.
_SHARDED_NORM_MAX_HEIGHT = 1024


def decode_width_shard_spec(mesh_device, dim):
    """``(input_memcfg, program_config, num_cores)`` for width-sharding a
    single-tile-tall ``[*, dim]`` decode activation, or ``None`` if no usable core
    grid divides its tile-width evenly.

    Pulled out of ``RMSNorm`` so that PRODUCERS of a decode activation can emit
    exactly this layout and skip the interleaved->sharded reshard the norm would
    otherwise do. ``ccl.ccl_allreduce`` uses it to have its all-gather write
    width-sharded L1 directly: measured 47.2 -> 43.5 us per all-reduce
    (-0.45 ms/decode step on 31B), bit-exact.

    Both sides MUST derive the layout from this one function -- a producer that
    hands the norm a shard spec differing in core count or block width silently
    costs a re-shard instead of saving one.
    """
    spec = width_shard_spec(mesh_device, dim, ttnn.TILE_SIZE)
    if spec is None:
        return None
    memcfg, program_config = spec
    tiles = dim // ttnn.TILE_SIZE
    grid = mesh_device.compute_with_storage_grid_size()
    num_cores = None
    for gy in range(1, grid.y + 1):
        for gx in range(1, grid.x + 1):
            n = gx * gy
            if tiles % n == 0 and (num_cores is None or n > num_cores):
                num_cores = n
    return (memcfg, program_config, num_cores)


def width_shard_spec(mesh_device, dim, height):
    """``(input_memcfg, program_config)`` for width-sharded RMSNorm at ``(height, dim)``.

    Shared by ``RMSNorm._build_sharded_cfg`` and ``ccl._norm_l1_gather_memcfg`` so
    an all-gather can write the exact layout the following norm expects.
    ``height`` must be tile-aligned and ``<= _SHARDED_NORM_MAX_HEIGHT``.
    """
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
    """Input memory config half of :func:`width_shard_spec`."""
    spec = width_shard_spec(mesh_device, dim, height)
    return spec[0] if spec else None


def decode_width_shard_memcfg(mesh_device, dim):
    """Just the input memory config from :func:`decode_width_shard_spec`."""
    spec = decode_width_shard_spec(mesh_device, dim)
    return spec[0] if spec else None


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
                mesh_mapper=self.mesh_config.shard_mapper(mesh_device, mesh_dims=(None, -2))
                if self.is_distributed
                else None,
            )
        else:
            self.tt_weight = None

        self.eps = hf_config.rms_norm_eps
        self.mesh_device = mesh_device

        # Width-sharded fast path (decode AND prefill). The plain (interleaved)
        # rms_norm runs the RMS reduction over the full hidden width on few
        # cores — e.g. ~90 us for a [1,1,128,hidden] prefill norm on Gemma4-31B
        # (hidden=5376) landing on only 4 of 64 cores, since the interleaved op
        # parallelizes over row-tiles, not the hidden width. Width-sharding the
        # activation across a core grid parallelizes the reduction
        # (LayerNormShardedMultiCoreProgramConfig handles the cross-core gather)
        # regardless of how many row-tiles (``block_h``) the activation has.
        # Built lazily on first call at a given (hidden width, row count) shape
        # so we can read the activation's true (padded) dims, then cached.
        self._sharded_cfg = None  # (input_memcfg, program_config) or None if unavailable
        self._sharded_dim = None
        self._sharded_height = None

    def _build_sharded_cfg(self, dim, height):
        """Width-sharded input memcfg + LayerNorm program config, or None.

        Delegates to :func:`width_shard_spec` so producers (``ccl.ccl_allreduce``)
        can emit byte-for-byte the same layout and skip the reshard entirely.
        """
        return width_shard_spec(self.mesh_device, dim, height)

    def _forward_sharded(self, x, already_sharded=False, return_sharded=False):
        """Width-sharded RMSNorm: [I2S ->] sharded rms_norm [-> S2I].

        ``already_sharded`` means the producer handed us this exact layout (see
        :func:`decode_width_shard_spec`), so the interleaved->sharded reshard is
        skipped. ``return_sharded`` means the CONSUMER wants that layout too, so
        the trailing sharded->interleaved is skipped as well.

        The norm itself is identical in all four combinations -- only the reshards
        around it change -- so every variant is bit-exact. Verified with
        ``torch.equal`` in ops_list/tools/sweeps/l1_stream.py (17.7 -> 10.8 us for
        the norm plus its reshards).

        Callers must only pass ``return_sharded=True`` when the consumer really
        does accept a width-sharded L1 tensor. A MATMUL does not: handing one a
        sharded in0 changes its blocking and measured maxabs=12.0 vs the
        interleaved result, so the norms feeding QKV / MLP keep their S2I."""
        if already_sharded:
            x_sh = x
        else:
            x_sh = ttnn.to_memory_config(x, self._sharded_cfg[0])
        out = ttnn.rms_norm(
            x_sh,
            weight=self.tt_weight,
            epsilon=self.eps,
            program_config=self._sharded_cfg[1],
        )
        if not already_sharded:
            x_sh.deallocate(True)
        if return_sharded:
            return out
        out_interleaved = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        out.deallocate(True)
        return out_interleaved

    def forward(self, x, return_sharded=False):
        """``return_sharded=True`` asks for the output left width-sharded in L1,
        which only the width-sharded fast path can honour; every other path
        ignores it and returns interleaved as before. See ``_forward_sharded``."""
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
            # Width-sharded fast path: a tile-aligned-height activation with a
            # learned weight. Covers decode (height <= 32) and short prefill
            # (height <= _SHARDED_NORM_MAX_HEIGHT). The no-weight per-head norms
            # keep the plain path. Sharded config is (width, height)-specific.
            padded_height = ((int(x.shape[-2]) + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
            if (
                self.with_scale
                and self.tt_weight is not None
                and len(x.shape) == 4
                and 1 <= padded_height <= _SHARDED_NORM_MAX_HEIGHT
            ):
                dim = x.shape[-1]
                if self._sharded_cfg is None or self._sharded_dim != dim or self._sharded_height != padded_height:
                    self._sharded_dim = dim
                    self._sharded_height = padded_height
                    self._sharded_cfg = self._build_sharded_cfg(dim, padded_height)
                if self._sharded_cfg:
                    # Decode only: a producer may already have written the exact
                    # layout we want (ccl_allreduce does). Prefill activations are
                    # too large for the L1-gather path and always take I2S here.
                    if x.is_sharded():
                        if padded_height <= ttnn.TILE_SIZE and x.memory_config() == self._sharded_cfg[0]:
                            return self._forward_sharded(x, already_sharded=True, return_sharded=return_sharded)
                    else:
                        return self._forward_sharded(x, return_sharded=return_sharded)

            if self.with_scale:
                tt_output = ttnn.rms_norm(
                    x,
                    weight=self.tt_weight,
                    epsilon=self.eps,
                )
            else:
                tt_output = ttnn.rms_norm(
                    x,
                    epsilon=self.eps,
                )
            return tt_output
