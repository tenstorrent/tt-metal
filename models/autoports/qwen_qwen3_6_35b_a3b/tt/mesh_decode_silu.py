# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TP4 decode SiLU with one worker for each of 32 (head, tile) pairs.

Inputs retain the eight-worker helper's contract: FP32 weighted carriers
[1,32,128] and BF16 Z [1,1,1024]. Output is BF16 [1,1,1024], physically
[1,32,1024], width-sharded over 32 cores with per-core shape [32,32]. The
optional debug output has the same shape/sharding and FP32 gate values.

The immutable compact LUT is loaded once onto every worker. Runtime allocation
and trace lifetime follow TP4LocalSiluMultiply; its two-page input buffers,
64-byte code-read alignment, lookup arithmetic and rounding remain unchanged.
Consumers own any conversion required by the narrower output shard layout.
"""

import types

from models.autoports.qwen_qwen3_6_35b_a3b.tt.fused_decoder import (
    LOCAL_SILU_COMPUTE_SOURCE,
    LOCAL_SILU_READER_SOURCE,
    SILU_COMPACT_ENTRIES,
    compact_silu_math_table,
    local_silu_math_table,
)
from models.autoports.qwen_qwen3_6_35b_a3b.tt.multichip_decoder import TP4LocalSiluMultiply


def _replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("TP4 SiLU source contract changed; update the tile-worker specialization")
    return source.replace(old, new, 1)


def _tile_call():
    reader = _replace_once(
        LOCAL_SILU_READER_SOURCE,
        " const uint32_t head=get_arg_val<uint32_t>(3);",
        " const uint32_t task=get_arg_val<uint32_t>(3),head=task/4,first_tile=task%4;",
    )
    reader = _replace_once(
        reader, " for(uint32_t tile=0;tile<4;++tile) {", " for(uint32_t tile=first_tile;tile<first_tile+1;++tile) {"
    )
    compute = _replace_once(
        LOCAL_SILU_COMPUTE_SOURCE,
        " for(uint32_t tile=0;tile<4;++tile) {",
        " for(uint32_t tile=0;tile<1;++tile) {",
    )
    # Reuse the existing descriptor/CB construction without modifying its
    # globals or the original class. Enumeration now supplies task=head*4+tile;
    # every worker produces exactly one tile in row-major core order.
    original = TP4LocalSiluMultiply.__call__
    namespace = dict(original.__globals__, LOCAL_SILU_READER_SOURCE=reader, LOCAL_SILU_COMPUTE_SOURCE=compute)
    function = types.FunctionType(
        original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__
    )
    function.__kwdefaults__ = original.__kwdefaults__
    return function


class TP4TileSilu(TP4LocalSiluMultiply):
    """32 workers using the baseline's exact SiLU/LUT/multiply arithmetic."""

    __call__ = _tile_call()

    def __init__(self, t, mesh, debug=False):
        self.t, self.mesh, self.debug = t, mesh, debug
        grid = mesh.compute_with_storage_grid_size()
        if grid.x * grid.y < 32:
            raise ValueError("TP4 tile SiLU requires at least 32 compute cores")
        self.coordinates = [(i % grid.x, i // grid.x) for i in range(32)]
        self.cores = t.num_cores_to_corerangeset(32, grid, row_wise=True)
        self.output_memory = t.MemoryConfig(
            t.TensorMemoryLayout.WIDTH_SHARDED,
            t.BufferType.L1,
            t.ShardSpec(self.cores, (32, 32), t.ShardOrientation.ROW_MAJOR),
        )
        table_memory = t.MemoryConfig(
            t.TensorMemoryLayout.HEIGHT_SHARDED,
            t.BufferType.L1,
            t.ShardSpec(self.cores, (1, SILU_COMPACT_ENTRIES), t.ShardOrientation.ROW_MAJOR),
        )
        self.table = t.from_torch(
            compact_silu_math_table(local_silu_math_table()).expand(32, SILU_COMPACT_ENTRIES).contiguous(),
            dtype=t.float32,
            layout=t.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=table_memory,
        )
        self.config = t.ComputeConfigDescriptor(
            math_fidelity=t.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=True
        )
        modes = [t.UnpackToDestMode.Default] * 64
        modes[0] = modes[1] = t.UnpackToDestMode.UnpackToDestFp32
        self.config.unpack_to_dest_mode = modes
