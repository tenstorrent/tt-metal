# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Carry a worker-disjoint residual grid through AR, norm, adds and layer boundaries.

The shared setup-owned AR receive buffers use the same new grid. Matmul inputs
still convert to their existing measured layouts inside decode_linear. There is
no restore to the origin8x4 residual grid after either reduction or decoder.
"""
import argparse
import sys

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt import multichip_decoder as production


def residual_spec(layout):
    if layout == "shifted8x4":
        return (0, 4, 7, 7), (8, 4)
    assert layout == "compact4x8"
    return (0, 0, 3, 7), (4, 8)


class ResidualGridCollectives:
    """One workspace per serial stack; concurrent stacks require separate instances."""
    layout = "shifted8x4"

    def __init__(self, mesh_device, ccl=None):
        self.device = mesh_device
        self.ccl = ccl or production.CCLManager(
            mesh_device, num_links=ttnn.get_num_links(mesh_device, 1), topology=ttnn.Topology.Linear
        )
        box, self.norm_extent = residual_spec(self.layout)
        x0, y0, x1, y1 = box
        device_grid = mesh_device.compute_with_storage_grid_size()
        assert device_grid.x > x1 and device_grid.y > y1
        assert not any(x0 <= x <= x1 and y0 <= y <= y1 for x, y in ((5, 3), (6, 3)))
        self.grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])

        def memory(width):
            return ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(self.grid, (32, width), ttnn.ShardOrientation.ROW_MAJOR),
            )

        self.ar_mem = memory(128)
        buffer_mem = memory(512)
        self.ar_buffers = [
            ttnn.empty(
                (1, 1, 32, 16384), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                device=mesh_device, memory_config=buffer_mem,
            )
            for _ in range(2)
        ]
        self.ar_semaphores = [ttnn.create_global_semaphore(mesh_device, self.ccl.ccl_cores, 0) for _ in range(2)]


class ARResidualGrid(production.MultichipDecoder):
    custom_workers = False

    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        assert kwargs.get("residual_layout", "replicated") == "replicated"
        assert kwargs.get("collective", "direct") == "direct"
        workspace = kwargs.get("collectives")
        if workspace is None:
            workspace = ResidualGridCollectives(kwargs["mesh_device"])
            kwargs["collectives"] = workspace
        assert isinstance(workspace, ResidualGridCollectives), "Use the shared alternate-grid workspace"
        layer = super().from_state_dict(*args, **kwargs)
        assert layer.native_local and layer.movement_optimized
        layer.norm_program = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=workspace.norm_extent,
            subblock_w=2, block_h=1, block_w=4, inplace=False,
        )
        print("AR_RESIDUAL_GRID", {
            "layout": workspace.layout, "custom_workers": cls.custom_workers,
            "links": 2, "residual_shard": [32, 128], "receive_shard": [32, 512],
            "buffer_count": 2, "restore_origin_residual": False,
        }, flush=True)
        return layer

    def decode_reduce(self, x):
        index = self.ar_index
        self.ar_index = 1 - index
        out = ttnn.experimental.all_reduce_async(
            x if self.movement_optimized and x.is_sharded() else ttnn.to_memory_config(x, self.ar_mem),
            self.ar_buffers[index], cluster_axis=1, mesh_device=self.device,
            multi_device_global_semaphore=self.ar_semaphores[index],
            memory_config=self.ar_mem, dtype=ttnn.bfloat16,
            topology=self.ccl.topology, num_links=2,
            use_noc1_only=False, use_optimal_ccl_for_llama=self.custom_workers,
        )
        assert out.memory_config() == self.ar_mem
        return out

    def decode_batch_packed(self, x, **kwargs):
        result = super().decode_batch_packed(x, **kwargs)
        assert result.memory_config() == self.ar_mem, "Carry alternate residual layout across decoder layers"
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--residual-grid", choices=("shifted8x4", "compact4x8"), default="shifted8x4")
    parser.add_argument("--custom-workers", action="store_true")
    options, remaining = parser.parse_known_args()
    ResidualGridCollectives.layout = options.residual_grid
    ARResidualGrid.custom_workers = options.custom_workers
    from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_packed_stack as stack
    stack.MultichipCollectives = ResidualGridCollectives
    stack.MultichipDecoder = ARResidualGrid
    sys.argv = [sys.argv[0], *remaining]
    stack.main()
