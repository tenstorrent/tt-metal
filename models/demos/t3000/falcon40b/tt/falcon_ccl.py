# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn


class TT_CCL:
    def __init__(
        self,
        mesh_device,
    ):
        self.mesh_device = mesh_device
        self.worker_sub_device_id = ttnn.SubDeviceId(0)
        self.sub_device_crs = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(
                        self.mesh_device.compute_with_storage_grid_size().x - 1,
                        self.mesh_device.compute_with_storage_grid_size().y - 1,
                    ),
                )
            }
        )

        self.ag_semaphores_idx = 0
        self.ag_semaphore_handles = [[], []]

        self.rs_semaphores_idx = 0
        self.rs_semaphore_handles = [[], []]

        for i in range(2):
            for _ in range(2):
                self.ag_semaphore_handles[i].append(
                    ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0)
                )
            for _ in range(3):
                self.rs_semaphore_handles[i].append(
                    ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0)
                )

        worker_sub_device = ttnn.SubDevice([self.sub_device_crs])
        sub_device_manager = self.mesh_device.create_sub_device_manager([worker_sub_device], 0)
        self.mesh_device.load_sub_device_manager(sub_device_manager)
        self.mesh_device.set_sub_device_stall_group([self.worker_sub_device_id])

    def get_and_cycle_ag_semaphore_handles(self):
        current_idx = self.ag_semaphores_idx
        self.ag_semaphores_idx = (self.ag_semaphores_idx + 1) % 2
        return self.ag_semaphore_handles[current_idx]

    def get_and_cycle_rs_semaphore_handles(self):
        current_idx = self.rs_semaphores_idx
        self.rs_semaphores_idx = (self.rs_semaphores_idx + 1) % 2
        return self.rs_semaphore_handles[current_idx]

    def close(self):
        self.mesh_device.reset_sub_device_stall_group()
