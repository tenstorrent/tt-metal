# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 decode sublayer norms on a core grid (``V41_DECODE_NORM=sharded``).

The block's ``attn_norm`` and ``ffn_norm`` are ``TtDistributedRmsNorm`` (the hidden split over the 4 TP chips:
a local
sum of squares, its all-gather, the normalisation with the gathered statistics). On the decode tile
(``[1, 1, 32, hidden/tp]`` = 40 tiles a chip) the interleaved ``rms_norm_pre_all_gather`` and
``rms_norm_post_all_gather`` run on **one core**: 18.6 and 37.1 us a site, 5.4 ms of the 75 ms one-row step
over
the 80 sites. :class:`DecodeRmsNorm` runs the same three steps on the tile width-sharded over
``cores`` cores with the sharded layernorm program config (``tt_transformers``' Galaxy decode norm: 10 cores
of 4
tiles for a 1,280-wide shard), the statistics gathered with the model's TP all-gather into an L1 shard, the
output
returned interleaved in DRAM as before. The weight, epsilon, axis and links are the wrapped norm's; nothing is
duplicated. The prefill keeps the wrapped norm (its rows are many).
"""

import os

import ttnn
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm

DECODE_NORM = os.environ.get("V41_DECODE_NORM", "0")
assert DECODE_NORM in ("0", "sharded"), DECODE_NORM
DECODE_NORM_CORES = int(os.environ.get("V41_DECODE_NORM_CORES", 10))
_TILE = 32
# a core count -> the compute grid (x, y) it is laid on
_GRIDS = {1: (1, 1), 2: (2, 1), 4: (4, 1), 5: (5, 1), 8: (8, 1), 10: (5, 2), 20: (10, 2), 40: (8, 5)}


class DecodeRmsNorm:
    """``norm`` (a ``TtDistributedRmsNorm``) applied to the decode tile on ``cores`` cores; ``tp_all_gather``:
    the
    model's ``V41Collectives.tp_all_gather`` (the statistics' gather)."""

    def __init__(self, norm: TtDistributedRmsNorm, width: int, tp_all_gather, cores: int = DECODE_NORM_CORES):
        assert cores in _GRIDS, f"V41_DECODE_NORM_CORES must be one of {sorted(_GRIDS)}, not {cores}"
        tiles = width // _TILE
        assert width % _TILE == 0 and tiles % cores == 0, (width, cores)
        self._norm, self.width, self.cores = norm, width, cores
        self._tp_all_gather = tp_all_gather
        gx, gy = _GRIDS[cores]
        block_w = tiles // cores
        subblock_w = max(s for s in (4, 3, 2, 1) if block_w % s == 0)
        self.in_memcfg = ttnn.create_sharded_memory_config(
            shape=(1, 1, _TILE, width),
            core_grid=ttnn.CoreGrid(y=gy, x=gx),
            strategy=ttnn.ShardStrategy.WIDTH,
        )
        self.progcfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            subblock_w=subblock_w,
            block_h=1,
            block_w=block_w,
            inplace=False,
        )
        tp = int(norm.mesh_device.shape[norm.cluster_axis])
        self.stats_memcfg = ttnn.create_sharded_memory_config(
            shape=(1, 1, _TILE, _TILE * tp),
            core_grid=ttnn.CoreGrid(y=1, x=1),
            strategy=ttnn.ShardStrategy.WIDTH,
        )

    def __call__(self, x):
        """``x`` bf16 TILE ``[1, 1, 32, width]`` interleaved -> the normalised tile, bf16 TILE, DRAM
        interleaved."""
        assert tuple(x.shape)[-2:] == (_TILE, self.width), tuple(x.shape)
        n = self._norm
        xs = ttnn.to_memory_config(x, self.in_memcfg)
        stats = ttnn.rms_norm_pre_all_gather(xs, dtype=ttnn.bfloat16, program_config=self.progcfg)
        gathered = self._tp_all_gather(stats, dim=3, memory_config=self.stats_memcfg)
        ttnn.deallocate(stats)
        out = ttnn.rms_norm_post_all_gather(
            xs, gathered, epsilon=n.epsilon, weight=n.weight, dtype=ttnn.bfloat16, program_config=self.progcfg
        )
        ttnn.deallocate(gathered)
        ttnn.deallocate(xs)
        res = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out)
        return res

    forward = __call__
