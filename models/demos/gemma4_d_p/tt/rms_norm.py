# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os

from loguru import logger
from torch import nn

import ttnn
from models.demos.gemma4_d_p.utils.general_utils import get_cache_file_name

TILE = 32
# DIAG (GEMMA4_NORM_SHARD): block-sharded RMSNorm. The default ttnn.rms_norm
# parallelises over ROWS only -- one core per 32 rows -- so the 5376-wide prefill norm
# runs on 8 of 120 cores at chunk 2048 and is width-bound. The sharded program config
# splits BOTH axes, but needs a block-sharded L1 input, so each call pays
# interleaved->sharded in and sharded->interleaved out.
_NORM_SHARD_MAX_BLOCK_TILES = 84  # bh*bw above this throws in dataflow_buffer.cpp
_norm_shard_cache = {}


def _norm_shard_cfg(rows: int, width: int):
    """Block-shard config for a (rows x width) norm, or None to keep the default path."""
    key = (rows, width)
    if key in _norm_shard_cache:
        return _norm_shard_cache[key]
    cfg = None
    mt, wt = rows // TILE, width // TILE
    if rows % TILE == 0 and width % TILE == 0:
        gx = 8
        if wt % gx == 0:
            bw = wt // gx
            bh = max(4, -(-mt // 8))
            if mt % bh == 0:
                gy = mt // bh
                sw = next(s for s in (3, 2, 1) if bw % s == 0)
                if gy <= 8 and bh * bw <= _NORM_SHARD_MAX_BLOCK_TILES:
                    memcfg = ttnn.create_sharded_memory_config(
                        shape=(bh * TILE, bw * TILE),
                        core_grid=ttnn.CoreGrid(x=gx, y=gy),
                        strategy=ttnn.ShardStrategy.BLOCK,
                        orientation=ttnn.ShardOrientation.ROW_MAJOR,
                        use_height_and_width_as_shard_shape=True,
                    )
                    prgcfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
                        compute_with_storage_grid_size=[gx, gy],
                        subblock_w=sw,
                        block_h=bh,
                        block_w=bw,
                        inplace=False,
                    )
                    cfg = (memcfg, prgcfg)
    _norm_shard_cache[key] = cfg
    return cfg


_SHARD_LOGGED = set()  # DIAG GEMMA4_NORM_SHARD one-shot witness per shape


class RMSNorm(nn.Module):
    def __init__(self, mesh_config, hf_config, state_dict, tensor_cache_path=None, with_scale=True):
        mesh_device = mesh_config.device
        super().__init__()
        self.with_scale = with_scale

        if with_scale and state_dict and "weight" in state_dict:
            torch_weight = state_dict["weight"].reshape((1, 1, -1, ttnn.TILE_SIZE))
        else:
            torch_weight = None

        self.mesh_config = mesh_config

        if with_scale:
            self.tt_weight = ttnn.as_tensor(
                torch_weight,
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                cache_file_name=get_cache_file_name(tensor_cache_path, "weight"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            # TILE gamma enables the large-tensor RMSNorm path, which bounds L1 usage
            # for the FP32 intermediate buffers in the 5376-wide prefill norm.
            flat_weight = ttnn.reshape(self.tt_weight, (1, 1, 1, hf_config.hidden_size))
            self.tt_weight = ttnn.to_layout(flat_weight, ttnn.TILE_LAYOUT)
        else:
            self.tt_weight = None

        self.eps = hf_config.rms_norm_eps
        self.mesh_device = mesh_device
        # Match the reference's FP32 prefill RMSNorm computation.
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def forward(self, x, memory_config=None):
        if os.environ.get("GEMMA4_NORM_SHARD", "0").lower() in ("1", "true", "yes"):  # DIAG
            cfg = _norm_shard_cfg(int(x.padded_shape[-2]), int(x.padded_shape[-1]))
            if cfg is not None:
                memcfg, prgcfg = cfg
                # DIAG witness: prove the flag actually reached the op. The other two
                # fixes log one; this one predates the practice, which made it the only
                # fix whose engagement could not be verified from a run log.
                _k = (int(x.padded_shape[-2]), int(x.padded_shape[-1]))
                if _k not in _SHARD_LOGGED:
                    _SHARD_LOGGED.add(_k)
                    logger.info(f"[DIAG] prefill norm: block-sharded rows={_k[0]} width={_k[1]}")
                xs = ttnn.to_memory_config(x, memcfg)
                out = ttnn.rms_norm(
                    xs,
                    weight=self.tt_weight,
                    epsilon=self.eps,
                    program_config=prgcfg,
                    memory_config=memcfg,
                    compute_kernel_config=self.compute_kernel_config,
                )
                xs.deallocate(True)
                # Honour the caller's requested placement -- mmanzoor's layer.py now asks
                # for L1 on the two norms that feed a residual add.
                interleaved = ttnn.sharded_to_interleaved(out, memory_config or ttnn.DRAM_MEMORY_CONFIG)
                out.deallocate(True)
                return interleaved
        return ttnn.rms_norm(
            x,
            weight=self.tt_weight,
            epsilon=self.eps,
            memory_config=memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )
