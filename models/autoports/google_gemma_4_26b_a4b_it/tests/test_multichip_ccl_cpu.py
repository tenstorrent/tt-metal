# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only regression for CCL worker/semaphore coverage on Blackhole."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.multichip_decoder import _MeshCCLManager


class MeshCCLSemaphoreTests(unittest.TestCase):
    def test_native_worker_geometry_has_semaphore_storage(self):
        # Native CCL chooses from the whole mesh grid, row-major. Four workers
        # plus one mux in each direction occupy ten cores, including x=8,9;
        # larger prefill RS uses eight workers per direction and eighteen cores.
        mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: ttnn.CoreCoord(11, 10))
        with patch("ttnn.create_global_semaphore", side_effect=lambda *_: object()) as create:
            manager = _MeshCCLManager(mesh, 1, ttnn.Topology.Linear)

        self.assertEqual(create.call_count, 12)
        self.assertEqual(manager.ccl_cores.num_cores(), 110)
        for call in create.call_args_list:
            allocated_mesh, cores, initial_value = call.args
            self.assertIs(allocated_mesh, mesh)
            self.assertEqual(initial_value, 0)
            for workers_per_direction in (1, 2, 4, 8):
                count = 2 * (1 + workers_per_direction)
                for index in range(count):
                    core = ttnn.CoreCoord(index % 11, index // 11)
                    self.assertTrue(cores.contains(core), f"No semaphore on native CCL worker {core}")


if __name__ == "__main__":
    unittest.main()
