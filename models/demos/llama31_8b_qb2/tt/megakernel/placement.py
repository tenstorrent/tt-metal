# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A bijective relocation of the existing layer workers around DRAM readers."""
import ttnn


class ProjectionPlacement:
    def __init__(self, mesh, name, gu_workers, input_cores=()):
        self.relocation = {}
        if name == "row":
            return
        if gu_workers not in (8, 16):
            raise ValueError("DRAM-near placement supports eight or sixteen projection workers")
        original = [(x, y) for y in range(4 - gu_workers // 8, 4) for x in range(8)]
        optimal = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(mesh, ttnn.NOC.NOC_0)
        target = [(c.x, c.y) for c in optimal]
        if len(target) != 8 or len(set(target)) != 8:
            raise ValueError("Expected one distinct reader for each of eight DRAM banks")
        if gu_workers == 16:
            # The same directed-hop objective as the native DRAM factory.
            # Keep primaries first for the eight-worker O/down phases; GU maps
            # these worker ranks to bank-interleaved output column ranks.
            used, excluded = set(target), {(c.x, c.y) for c in input_cores}
            grid = mesh.compute_with_storage_grid_size()
            secondary = []
            for primary in optimal:
                candidates = [ttnn.CoreCoord(x, y) for x in range(grid.x) for y in range(grid.y)
                              if (x, y) not in used and (x, y) not in excluded]
                best = min(candidates, key=lambda c: ttnn._ttnn.multi_device.experimental.get_worker_noc_hop_distance(
                    mesh, ttnn.MeshCoordinate(0, 0), c, primary, ttnn.NOC.NOC_0))
                secondary.append((best.x, best.y)); used.add((best.x, best.y))
            target.extend(secondary)
        # Preserve bank order for projections. Move the displaced roles into
        # the vacated row3 slots; this is a permutation, not extra concurrency.
        self.relocation = dict(zip(original, target))
        displaced = [c for c in target if c not in original]
        vacant = [c for c in original if c not in target]
        self.relocation.update(zip(displaced, vacant))
        assert len(set(self.relocation.values())) == len(self.relocation)

    def map(self, cores, *, row_major=False):
        result = [ttnn.CoreCoord(*self.relocation.get((c.x, c.y), (c.x, c.y))) for c in cores]
        return sorted(result, key=lambda c: (c.y, c.x)) if row_major else result
